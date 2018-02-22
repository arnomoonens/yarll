#!/usr/bin/env python
# # -*- coding: utf8 -*-

import os
import sys
import argparse
import numpy as np
from mpi4py import MPI
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../.."
)))
from environment.registration import make  # pylint: disable=C0413
from misc.utils import load, json_to_dict, cluster_spec  # pylint: disable=C0413
from agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete, ActorCriticNetworkDiscreteCNN, ActorCriticNetworkDiscreteCNNRNN, ActorCriticDiscreteLoss, ActorCriticNetworkContinuous, ActorCriticContinuousLoss  # pylint: disable=C0413
from agents.env_runner import EnvRunner  # pylint: disable=C0413

class DPPOWorker(object):
    """Distributed Proximal Policy Optimization Worker."""

    def __init__(self, env_id, task_id, cluster, comm, monitor_path, config, seed=None):
        super(DPPOWorker, self).__init__()
        self.comm = comm
        self.config = config
        self.env = make(env_id)
        self.task_id = task_id
        if seed is not None:
            self.env.seed(seed)
        self.writer = tf.summary.FileWriter(os.path.join(
            monitor_path,
            "task{}".format(task_id)))

        # Only used (and overwritten) by agents that use an RNN
        self.initial_features = None

        # Build actor and critic networks
        worker_device = "/job:worker/task:{}/cpu:0".format(task_id)
        # Global network
        shared_device = tf.train.replica_device_setter(1, worker_device=worker_device)
        with tf.device(shared_device):
            with tf.variable_scope("new_network"):  # The workers only have 1 network
                self.global_network = self.build_networks()
                self.states = self.global_network.states
                self.action = self.global_network.action
                self.value = self.global_network.value
                self.actions_taken = self.global_network.actions_taken
                self.adv = self.global_network.adv
                self.r = self.global_network.r
                self.global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                self._global_step = tf.get_variable(
                    "global_step", [],
                    tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)

        self.env_runner = EnvRunner(
            self.env, self, {}, summary_writer=self.writer)

        self.server = tf.train.Server(
            cluster,
            job_name="worker",
            task_index=task_id,
            config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=2
        ))

        init_op = tf.global_variables_initializer()
        def init_fn(sess):
            sess.run(init_op)
        self.sv = tf.train.Supervisor(
            is_chief=(task_id == 0),  # The master (in dppo.py) is the chief
            logdir=monitor_path,
            init_op=init_op,
            init_fn=init_fn,
            summary_op=None,
            summary_writer=self.writer,
            global_step=self._global_step,
        )

        self.config_proto = tf.ConfigProto(
            device_filters=["/job:ps", "/job:master", "/job:worker/task:{}/cpu:0".format(task_id)])

    def build_networks(self):
        raise NotImplementedError

    def run(self):
        with self.sv.managed_session(self.server.target, config=self.config_proto) as sess, sess.as_default():
            print("DPPO worker {} in managed session".format(self.task_id))
            while not self.sv.should_stop():
                message = None
                while message != "collect":
                    message = self.comm.bcast(None, root=0)
                print("gonna collect")
                trajectory = self.env_runner.get_steps(
                    self.config["n_local_steps"], stop_at_trajectory_end=False)
                T = trajectory.steps
                value = 0 if trajectory.terminals[-1] else self.get_critic_value(
                    np.asarray(trajectory.states)[None, -1], trajectory.features[-1])
                vpred = np.asarray(trajectory.values + [value])
                gamma = self.config["gamma"]
                lambda_ = self.config["gae_lambda"]
                terminals = np.append(trajectory.terminals, 0)
                gaelam = advantages = np.empty(T, 'float32')
                lastgaelam = 0
                for t in reversed(range(T)):
                    nonterminal = 1 - terminals[t + 1]
                    delta = trajectory.rewards[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
                    gaelam[t] = lastgaelam = delta + gamma * lambda_ * nonterminal * lastgaelam
                returns = advantages + trajectory.values
                processed = trajectory.states, trajectory.actions, advantages, returns, trajectory.features[0]
                self.comm.gather(processed, root=0)

    @property
    def global_step(self):
        return self._global_step.eval()

    def get_critic_value(self, state, *rest):
        fetches = [self.global_network.value]
        feed_dict = {self.global_network.states: state}
        value = tf.get_default_session().run(fetches, feed_dict=feed_dict)[0].flatten()
        return value

    def choose_action(self, state, *rest):
        fetches = [self.global_network.action, self.global_network.value]
        feed_dict = {
            self.global_network.states: [state]
        }
        action, value = tf.get_default_session().run(fetches, feed_dict=feed_dict)
        return {"action": action, "value": value[0]}

    def get_env_action(self, action):
        return np.argmax(action)

    def new_trajectory(self):
        pass


class DPPOWorkerDiscrete(DPPOWorker):
    """DPPOWorker for a discrete action space."""

    def __init__(self, env_id, task_id, cluster, comm, monitor_path, config, seed=None):
        self.make_loss = ActorCriticDiscreteLoss
        super(DPPOWorkerDiscrete, self).__init__(
            env_id,
            task_id,
            cluster,
            comm,
            monitor_path,
            config,
            seed
        )

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])
        return ac_net


class DPPOWorkerDiscreteCNN(DPPOWorkerDiscrete):
    """DPPOWorker for a discrete action space."""

    def __init__(self, env_id, task_id, cluster, comm, monitor_path, config, seed=None):
        self.make_loss = ActorCriticDiscreteLoss
        super(DPPOWorkerDiscreteCNN, self).__init__(
            env_id,
            task_id,
            cluster,
            comm,
            monitor_path,
            config,
            seed
        )

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"],
            summary=False)
        return ac_net


class DPPOWorkerDiscreteCNNRNN(DPPOWorkerDiscreteCNN):
    """DPPOWorker for a discrete action space."""

    def __init__(self, env_id, task_id, cluster, comm, monitor_path, config, seed=None):
        self.make_loss = ActorCriticDiscreteLoss
        super(DPPOWorkerDiscreteCNNRNN, self).__init__(
            env_id,
            task_id,
            cluster,
            comm,
            monitor_path,
            config,
            seed
        )

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscreteCNNRNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"],
            summary=False)
        self.initial_features = ac_net.state_init
        return ac_net

    def choose_action(self, state, features):
        feed_dict = {
            self.states: [state]
        }

        feed_dict[self.global_network.rnn_state_in] = features
        action, rnn_state, value = tf.get_default_session().run(
            [self.action, self.global_network.rnn_state_out, self.value],
            feed_dict=feed_dict)
        return {"action": action, "value": value, "features": rnn_state}

    def get_critic_value(self, states, features):
        feed_dict = {
            self.states: states
        }
        feed_dict[self.global_network.rnn_state_in] = features
        return tf.get_default_session().run(self.value, feed_dict=feed_dict)[0]


class DPPOWorkerContinuous(DPPOWorker):
    """DPPOWorker for a continuous action space."""

    def __init__(self, env_id, task_id, cluster, comm, monitor_path, config, seed=None):
        self.make_loss = ActorCriticContinuousLoss
        super(DPPOWorkerContinuous, self).__init__(
            env_id,
            task_id,
            cluster,
            comm,
            monitor_path,
            config,
            seed
        )

    def build_networks(self):
        ac_net = ActorCriticNetworkContinuous(
            self.env.action_space,
            list(self.env.observation_space.shape),
            self.config["n_hidden"])
        return ac_net

    def get_env_action(self, action):
        return action


parser = argparse.ArgumentParser()

parser.add_argument("env_id", type=str,
                    help="Name of the environment on which to run")
parser.add_argument("cls", type=str, help="Which class to use for the task.")
parser.add_argument("config", type=str, help="Path to config file")
parser.add_argument("-seed", type=int, default=None,
                    help="Seed to use for the environment.")
parser.add_argument("--monitor_path", type=str,
                    help="Path where to save monitor files.")
parser.add_argument("--seed", type=int, default=None, help="Seed to use for environments.")


def main():
    comm = MPI.Comm.Get_parent()
    n_tasks = comm.Get_size()
    task_id = comm.Get_rank()
    args = parser.parse_args()
    spec = cluster_spec(n_tasks, 1, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()
    cls = load("agents.ppo.dppo_worker:" + args.cls)
    config = json_to_dict(args.config)

    task = cls(args.env_id, task_id, cluster, comm, args.monitor_path, config, args.seed)
    task.run()


if __name__ == '__main__':
    main()
