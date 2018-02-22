# -*- coding: utf8 -*-

import sys
import os
import subprocess
from six.moves import shlex_quote
import tensorflow as tf
from mpi4py import MPI

from agents.agent import Agent
from agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous
from agents.ppo.ppo import cso_loss
from misc.utils import cluster_spec, FastSaver


class DPPO(Agent):
    """Distributed Proximal Policy Optimization agent."""

    RNN = False

    def __init__(self, env, monitor_path, **usercfg):
        super(DPPO, self).__init__()
        self.env = env
        self.env_name = env.spec.id
        self.monitor_path = monitor_path

        self.comm = MPI.COMM_SELF

        self.config.update(dict(
            n_workers=3,
            n_hidden=20,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=2.5e-4,
            n_iter=10000,
            n_epochs=4,
            n_local_steps=128,
            gradient_clip_value=0.5,
            entropy_coef=0.01,
            cso_epsilon=0.1,  # Clipped surrogate objective epsilon
            save_model=False
        ))
        self.config.update(usercfg)

        self.task_type = None  # To be filled in by subclasses

        worker_device = "/job:master/task:0/cpu:0"
        shared_device = tf.train.replica_device_setter(
            1, worker_device=worker_device)
        with tf.device(shared_device):
            with tf.variable_scope("new_network"):
                self.new_network = self.build_networks()
                if self.RNN:
                    self.initial_features = self.new_network.state_init
                else:
                    self.initial_features = None
                self.new_network_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            self._global_step = tf.get_variable(
                "global_step",
                [],
                tf.int32,
                initializer=tf.constant_initializer(0, dtype=tf.int32),
                trainable=False)
        self.action = self.new_network.action
        self.value = self.new_network.value
        self.states = self.new_network.states
        self.r = self.new_network.r
        self.adv = self.new_network.adv
        self.actions_taken = self.new_network.actions_taken

        with tf.device(worker_device):
            with tf.variable_scope("old_network"):
                self.old_network = self.build_networks()
                self.old_network_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

            self.set_old_to_new = tf.group(
                *[v1.assign(v2) for v1, v2 in zip(self.old_network_vars, self.new_network_vars)])

            # Reduces by taking the mean instead of summing
            self.actor_loss = -tf.reduce_mean(cso_loss(
                self.old_network, self.new_network, self.config["cso_epsilon"], self.adv))
            self.critic_loss = tf.reduce_mean(tf.square(self.value - self.r))
            self.loss = self.actor_loss + 0.5 * self.critic_loss - \
                self.config["entropy_coef"] * \
                tf.reduce_mean(self.new_network.entropy)

            grads = tf.gradients(self.loss, self.new_network_vars)

        self.n_steps = tf.shape(self.states)[0]
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = FastSaver()
        n_steps = tf.to_float(self.n_steps)
        summary_actor_loss = tf.summary.scalar(
            "model/Actor_loss", self.actor_loss / n_steps)
        summary_critic_loss = tf.summary.scalar(
            "model/Critic_loss", self.critic_loss / n_steps)
        summary_loss = tf.summary.scalar("model/Loss", self.loss / n_steps)
        summary_grad_norm = tf.summary.scalar(
            "model/grad_global_norm", tf.global_norm(grads))
        summary_var_norm = tf.summary.scalar(
            "model/var_global_norm", tf.global_norm(self.new_network_vars))
        self.model_summary_op = tf.summary.merge([
            summary_actor_loss,
            summary_critic_loss,
            summary_loss,
            summary_grad_norm,
            summary_var_norm
        ])
        self.writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "master"))

        # grads before clipping were passed to the summary, now clip and apply them
        grads, _ = tf.clip_by_global_norm(
            grads, self.config["gradient_clip_value"])
        self.optimizer = tf.train.AdamOptimizer(
            self.config["learning_rate"], name="optim")
        apply_grads = self.optimizer.apply_gradients(
            zip(grads, self.new_network_vars))

        inc_step = self._global_step.assign_add(self.n_steps)
        self.train_op = tf.group(apply_grads, inc_step)

        cluster = cluster_spec(self.config["n_workers"], 1, 1)
        self.server = tf.train.Server(
            cluster,
            job_name="master",
            task_index=0,
            config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=2
        ))

        init_op = tf.variables_initializer(self.new_network_vars)
        def init_fn(sess):
            sess.run(init_op)

        self.sv = tf.train.Supervisor(
            is_chief=False,
            logdir=monitor_path,
            summary_op=None,
            init_op=init_op,
            init_fn=init_fn,
            summary_writer=self.writer,
            global_step=self._global_step,
            save_model_secs=0 if not self.config["save_model"] else 300,
            save_summaries_secs=30
        )

        self.config_proto = tf.ConfigProto(
            device_filters=["/job:ps", "/job:master", "/job:worker"])

        return

    def build_networks(self):
        raise NotImplementedError

    def learn(self):
        """Run learning algorithm"""
        config = self.config
        current_folder = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)))
        cmd = [
            sys.executable,
            # TODO: move parameter server
            os.path.join(current_folder, "../actorcritic", "parameter_server.py"),
            self.config["n_workers"],
            "--n_masters", "1"]
        processed_cmd = " ".join(shlex_quote(str(x)) for x in cmd)
        ps_process = subprocess.Popen(processed_cmd, shell=True)
        self.comm.Spawn(
            sys.executable,
            args=[
                os.path.join(current_folder, "dppo_worker.py"),
                self.env_name,
                self.task_type,
                self.config["config_path"],
                "--seed", str(self.config["seed"]),
                "--monitor_path", self.monitor_path
            ],
            maxprocs=self.config["n_workers"]
        )
        n_updates = 0
        with self.sv.managed_session(self.server.target, config=self.config_proto) as sess, sess.as_default():
            print("Master in session")
            for _ in range(config["n_iter"]):
                # Collect trajectories until we get timesteps_per_batch total timesteps
                self.comm.bcast("collect", root=0)
                trajectories = self.comm.gather(None, root=0)[1:]
                tf.get_default_session().run(self.set_old_to_new)

                for states, actions, advs, returns, features in trajectories:
                    fetches = [self.model_summary_op, self.train_op]
                    feed_dict = {
                        self.states: states,
                        self.old_network.states: states,
                        self.actions_taken: actions,
                        self.old_network.actions_taken: actions,
                        self.adv: advs,
                        self.r: returns
                    }
                    if features != [] and features is not None:
                        feed_dict[self.old_network.rnn_state_in] = features
                        feed_dict[self.new_network.rnn_state_in] = features
                    summary, _ = tf.get_default_session().run(fetches, feed_dict)
                    self.writer.add_summary(summary, n_updates)
                    n_updates += 1
                    self.writer.flush()
        ps_process.terminate()

class DPPODiscrete(DPPO):

    def __init__(self, env, monitor_path, **usercfg):
        super(DPPODiscrete, self).__init__(env, monitor_path, **usercfg)
        self.task_type = "DPPOWorkerDiscrete"

    def build_networks(self):
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])


class DPPODiscreteCNN(DPPODiscrete):

    def __init__(self, env, monitor_path, **usercfg):
        super(DPPODiscreteCNN, self).__init__(env, monitor_path, **usercfg)
        self.task_type = "DPPOWorkerDiscreteCNN"

    def build_networks(self):
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])


class DPPOContinuous(DPPO):

    def __init__(self, env, monitor_path, **usercfg):
        super(DPPOContinuous, self).__init__(env, monitor_path, **usercfg)
        self.task_type = "DPPOWorkerContinuous"

    def build_networks(self):
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            self.config["n_hidden"])

    def get_env_action(self, action):
        return action
