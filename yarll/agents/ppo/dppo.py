# -*- coding: utf8 -*-

import sys
import os
import tensorflow as tf
from mpi4py import MPI
import numpy as np

from yarll.agents.agent import Agent
from yarll.agents.ppo.ppo import ppo_loss
from yarll.agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous
from yarll.misc.utils import FastSaver


class DPPO(Agent):
    """Distributed Proximal Policy Optimization agent."""

    RNN = False

    def __init__(self, env, monitor_path, **usercfg):
        super(DPPO, self).__init__()
        self.env = env
        self.env_name: str = env.spec.id
        self.monitor_path: str = monitor_path

        self.comm = MPI.COMM_SELF

        self.config.update(dict(
            n_workers=3,
            n_hidden_units=20,
            n_hidden_layers=2,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=2.5e-4,
            n_iter=10000,
            n_epochs=4,
            n_local_steps=128,
            gradient_clip_value=0.5,
            vf_coef=0.5,
            entropy_coef=0.01,
            cso_epsilon=0.1,  # Clipped surrogate objective epsilon
            learn_method="batches",
            batch_size=64,
            save_model=False
        ))
        self.config.update(usercfg)

        self.task_type = None  # To be filled in by subclasses

        self.n_updates: int = 0

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
        self.actions_taken = self.new_network.actions_taken
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.ret = tf.placeholder(tf.float32, [None], name="return")

        with tf.variable_scope("old_network"):
            self.old_network = self.build_networks()
            self.old_network_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        self.set_old_to_new = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.old_network_vars, self.new_network_vars)])

        # Reduces by taking the mean instead of summing
        self.actor_loss = -tf.reduce_mean(self.make_actor_loss(self.old_network, self.new_network, self.advantage))
        self.critic_loss = tf.reduce_mean(tf.square(self.value - self.ret))
        self.mean_entropy = tf.reduce_mean(self.new_network.entropy)
        self.loss = self.actor_loss + self.config["vf_coef"] * self.critic_loss + \
            self.config["entropy_coef"] * self.mean_entropy

        grads = tf.gradients(self.loss, self.new_network_vars)

        self.n_steps = tf.shape(self.states)[0]
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = FastSaver()
        summary_actor_loss = tf.summary.scalar(
            "model/Actor_loss", self.actor_loss)
        summary_critic_loss = tf.summary.scalar(
            "model/Critic_loss", self.critic_loss)
        summary_loss = tf.summary.scalar("model/Loss", self.loss)
        summary_entropy = tf.summary.scalar("model/Entropy", -self.mean_entropy)
        summary_grad_norm = tf.summary.scalar(
            "model/grad_global_norm", tf.global_norm(grads))
        summary_var_norm = tf.summary.scalar(
            "model/var_global_norm", tf.global_norm(self.new_network_vars))
        self.model_summary_op = tf.summary.merge([
            summary_actor_loss,
            summary_critic_loss,
            summary_loss,
            summary_entropy,
            summary_grad_norm,
            summary_var_norm
        ])
        self.writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "master"))

        # grads before clipping were passed to the summary, now clip and apply them
        if self.config["gradient_clip_value"] is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.config["gradient_clip_value"])

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(
                self.config["learning_rate"], name="optim")
            apply_grads = self.optimizer.apply_gradients(
                zip(grads, self.new_network_vars))

            inc_step = self._global_step.assign_add(self.n_steps)
            self.train_op = tf.group(apply_grads, inc_step)
        optimizer_variables = [var for var in tf.global_variables() if var.name.startswith("optimizer")]
        self.init_op = tf.variables_initializer(self.new_network_vars + optimizer_variables + [self._global_step])

    def make_actor_loss(self, old_network, new_network, advantage):
        return ppo_loss(old_network.action_log_prob, new_network.action_log_prob, self.config["cso_epsilon"], advantage)

    def build_networks(self):
        raise NotImplementedError

    def update_network(self, states, actions, advs, returns, features=None):
        fetches = [self.model_summary_op, self.train_op]
        feed_dict = {
            self.states: states,
            self.old_network.states: states,
            self.actions_taken: actions,
            self.old_network.actions_taken: actions,
            self.advantage: advs,
            self.ret: returns
        }
        if features != [] and features is not None:
            feed_dict[self.old_network.rnn_state_in] = features
            feed_dict[self.new_network.rnn_state_in] = features
        summary, _ = tf.get_default_session().run(fetches, feed_dict)
        self.writer.add_summary(summary, self.n_updates)
        self.n_updates += 1

    def learn_by_batches(self, trajectories):
        all_states, all_actions, all_advs, all_returns = [], [], [], []
        for states, actions, advs, returns, _ in trajectories:
            all_states.extend(states)
            all_actions.extend(actions)
            all_advs.extend(advs)
            all_returns.extend(returns)
        all_advs = np.array(all_advs)
        all_advs = (all_advs - all_advs.mean()) / all_advs.std()
        indices = np.arange(len(all_states))
        batch_size = int(self.config["batch_size"])
        for _ in range(int(self.config["n_epochs"])):
            np.random.shuffle(indices)
            for j in range(0, len(all_states), batch_size):
                batch_indices = indices[j:(j + batch_size)]
                batch_states = np.array(all_states)[batch_indices]
                batch_actions = np.array(all_actions)[batch_indices]
                batch_advs = np.array(all_advs)[batch_indices]
                batch_rs = np.array(all_returns)[batch_indices]
                self.update_network(batch_states, batch_actions, batch_advs, batch_rs)
            self.writer.flush()


    def learn_by_trajectories(self, trajectories):
        for _ in range(int(self.config["n_epochs"])):
            for states, actions, advs, returns, features in trajectories:
                self.update_network(states, actions, advs, returns, features)
            self.writer.flush()

    def learn(self):
        """Run learning algorithm"""
        config = self.config
        current_folder = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)))
        args = [
            os.path.join(current_folder, "dppo_worker.py"),
            self.env_name,
            self.task_type,
            self.config["config_path"],
            "--monitor_path", self.monitor_path
        ]
        seed = self.config["seed"]
        if seed is not None:
            args += ["--seed", str(seed)]
        comm = self.comm.Spawn(
            sys.executable,
            args=args,
            maxprocs=int(self.config["n_workers"])
        )
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess, sess.as_default():
            tf.get_default_session().run(self.init_op)
            for _ in range(config["n_iter"]):
                # Collect trajectories until we get timesteps_per_batch total timesteps
                for var in self.new_network_vars:
                    comm.Bcast(var.eval(), root=MPI.ROOT)
                trajectories = comm.gather(None, root=MPI.ROOT)
                tf.get_default_session().run(self.set_old_to_new)

                # Mix steps of all trajectories and learn by minibatches or not
                if self.config["learn_method"] == "batches":
                    self.learn_by_batches(trajectories)
                else:
                    self.learn_by_trajectories(trajectories)

class DPPODiscrete(DPPO):

    def __init__(self, env, monitor_path, **usercfg):
        super(DPPODiscrete, self).__init__(env, monitor_path, **usercfg)
        self.task_type = "DPPOWorkerDiscrete"

    def build_networks(self):
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))


class DPPODiscreteCNN(DPPODiscrete):

    def __init__(self, env, monitor_path, **usercfg):
        super(DPPODiscreteCNN, self).__init__(env, monitor_path, **usercfg)
        self.task_type = "DPPOWorkerDiscreteCNN"

    def build_networks(self):
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class DPPOContinuous(DPPO):

    def __init__(self, env, monitor_path, **usercfg):
        super(DPPOContinuous, self).__init__(env, monitor_path, **usercfg)
        self.task_type = "DPPOWorkerContinuous"

    def build_networks(self):
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def get_env_action(self, action):
        return action
