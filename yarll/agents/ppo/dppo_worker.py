#!/usr/bin/env python
# # -*- coding: utf8 -*-

import argparse
import os
from typing import Dict, Any
import numpy as np
from mpi4py import MPI
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from yarll.environment.registration import make
from yarll.misc.utils import load, json_to_dict
from yarll.agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete, ActorCriticNetworkDiscreteCNN, ActorCriticNetworkDiscreteCNNRNN, actor_critic_discrete_loss, ActorCriticNetworkContinuous, actor_critic_continuous_loss
from yarll.agents.env_runner import EnvRunner


class DPPOWorker(object):
    """Distributed Proximal Policy Optimization Worker."""

    def __init__(self, env_id: str, task_id: int, comm: MPI.Intercomm, monitor_path: str, config: Dict[str, Any], seed=None) -> None:
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            env_id: (str): write your description
            task_id: (str): write your description
            comm: (todo): write your description
            MPI: (int): write your description
            Intercomm: (todo): write your description
            monitor_path: (str): write your description
            config: (todo): write your description
            seed: (int): write your description
        """
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
        with tf.device('/cpu:0'):
            with tf.variable_scope("new_network"):  # The workers only have 1 network
                self.global_network = self.build_networks()
                self.states = self.global_network.states
                self.action = self.global_network.action
                self.value = self.global_network.value
                self.global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                self._global_step = tf.get_variable(
                    "global_step", [],
                    tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)

            self.env_runner = EnvRunner(
                self.env, self, {})

    def build_networks(self):
        """
        Builds a list of networks networks.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    def run(self):
        """
        Runs the model.

        Args:
            self: (todo): write your description
        """
        with tf.Session() as sess, sess.as_default():
            var_receivers = [np.zeros(var.shape.as_list(), dtype=var.dtype.as_numpy_dtype) for var in self.global_vars]
            while True:
                for var_receiver, tf_var in zip(var_receivers, self.global_vars):
                    self.comm.Bcast(var_receiver, root=0)
                    tf_var.load(var_receiver)
                experiences = self.env_runner.get_steps(
                    int(self.config["n_local_steps"]), stop_at_trajectory_end=False)
                T = experiences.steps
                value = 0 if experiences.terminals[-1] else self.get_critic_value(
                    np.asarray(experiences.states)[None, -1], experiences.features[-1])
                vpred = np.asarray(experiences.values + [value])
                gamma = self.config["gamma"]
                lambda_ = self.config["gae_lambda"]
                gaelam = advantages = np.empty(T, 'float32')
                last_gaelam = 0
                for t in reversed(range(T)):
                    nonterminal = 1 - experiences.terminals[t]
                    delta = experiences.rewards[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
                    gaelam[t] = last_gaelam = delta + gamma * lambda_ * nonterminal * last_gaelam
                returns = advantages + experiences.values
                processed = experiences.states, experiences.actions, advantages, returns, experiences.features[0]
                self.comm.gather(processed, root=0)

    @property
    def global_step(self):
        """
        Return the global step.

        Args:
            self: (todo): write your description
        """
        return self._global_step.eval()

    def get_critic_value(self, state, *rest):
        """
        Gets the value of the network.

        Args:
            self: (todo): write your description
            state: (str): write your description
            rest: (str): write your description
        """
        fetches = [self.global_network.value]
        feed_dict = {self.global_network.states: state}
        value = tf.get_default_session().run(fetches, feed_dict=feed_dict)[0].flatten()
        return value

    def choose_action(self, state, *rest):
        """
        Choose the selected action.

        Args:
            self: (todo): write your description
            state: (todo): write your description
            rest: (todo): write your description
        """
        fetches = [self.global_network.action, self.global_network.value]
        feed_dict = {
            self.global_network.states: [state]
        }
        action, value = tf.get_default_session().run(fetches, feed_dict=feed_dict)
        return {"action": action, "value": value[0]}

    def get_env_action(self, action):
        """
        Get the action action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        return np.argmax(action)

    def new_trajectory(self):
        """
        Return a new trajectory.

        Args:
            self: (todo): write your description
        """
        pass


class DPPOWorkerDiscrete(DPPOWorker):
    """DPPOWorker for a discrete action space."""

    def __init__(self, env_id, task_id, comm, monitor_path, config, seed=None):
        """
        Initialize the task.

        Args:
            self: (todo): write your description
            env_id: (str): write your description
            task_id: (str): write your description
            comm: (todo): write your description
            monitor_path: (str): write your description
            config: (todo): write your description
            seed: (int): write your description
        """
        self.make_loss = actor_critic_discrete_loss
        super(DPPOWorkerDiscrete, self).__init__(
            env_id,
            task_id,
            comm,
            monitor_path,
            config,
            seed
        )

    def build_networks(self):
        """
        Builds the network network

        Args:
            self: (todo): write your description
        """
        ac_net = ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))
        return ac_net


class DPPOWorkerDiscreteCNN(DPPOWorkerDiscrete):
    """DPPOWorker for a discrete action space."""

    def __init__(self, env_id, task_id, comm, monitor_path, config, seed=None):
        """
        Initialize the simulation.

        Args:
            self: (todo): write your description
            env_id: (str): write your description
            task_id: (str): write your description
            comm: (todo): write your description
            monitor_path: (str): write your description
            config: (todo): write your description
            seed: (int): write your description
        """
        self.make_loss = actor_critic_discrete_loss
        super(DPPOWorkerDiscreteCNN, self).__init__(
            env_id,
            task_id,
            comm,
            monitor_path,
            config,
            seed
        )

    def build_networks(self):
        """
        Builds a network from the network

        Args:
            self: (todo): write your description
        """
        ac_net = ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            summary=False)
        return ac_net


class DPPOWorkerDiscreteCNNRNN(DPPOWorkerDiscreteCNN):
    """DPPOWorker for a discrete action space."""

    def __init__(self, env_id, task_id, comm, monitor_path, config, seed=None):
        """
        Initialize the task.

        Args:
            self: (todo): write your description
            env_id: (str): write your description
            task_id: (str): write your description
            comm: (todo): write your description
            monitor_path: (str): write your description
            config: (todo): write your description
            seed: (int): write your description
        """
        self.make_loss = actor_critic_discrete_loss
        super(DPPOWorkerDiscreteCNNRNN, self).__init__(
            env_id,
            task_id,
            comm,
            monitor_path,
            config,
            seed
        )

    def build_networks(self):
        """
        Builds the network

        Args:
            self: (todo): write your description
        """
        ac_net = ActorCriticNetworkDiscreteCNNRNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            summary=False)
        self.initial_features = ac_net.state_init
        return ac_net

    def choose_action(self, state, features):
        """
        Chooses the action.

        Args:
            self: (todo): write your description
            state: (todo): write your description
            features: (todo): write your description
        """
        feed_dict = {
            self.states: [state]
        }

        feed_dict[self.global_network.rnn_state_in] = features
        action, rnn_state, value = tf.get_default_session().run(
            [self.action, self.global_network.rnn_state_out, self.value],
            feed_dict=feed_dict)
        return {"action": action, "value": value, "features": rnn_state}

    def get_critic_value(self, states, features):
        """
        Gets the network value for a tf. tnn. tnn.

        Args:
            self: (todo): write your description
            states: (str): write your description
            features: (str): write your description
        """
        feed_dict = {
            self.states: states
        }
        feed_dict[self.global_network.rnn_state_in] = features
        return tf.get_default_session().run(self.value, feed_dict=feed_dict)[0]


class DPPOWorkerContinuous(DPPOWorker):
    """DPPOWorker for a continuous action space."""

    def __init__(self, env_id, task_id, comm, monitor_path, config, seed=None):
        """
        Initialize the simulation.

        Args:
            self: (todo): write your description
            env_id: (str): write your description
            task_id: (str): write your description
            comm: (todo): write your description
            monitor_path: (str): write your description
            config: (todo): write your description
            seed: (int): write your description
        """
        self.make_loss = actor_critic_continuous_loss
        super(DPPOWorkerContinuous, self).__init__(
            env_id,
            task_id,
            comm,
            monitor_path,
            config,
            seed
        )

    def build_networks(self):
        """
        Builds a new networks

        Args:
            self: (todo): write your description
        """
        ac_net = ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))
        return ac_net

    def get_env_action(self, action):
        """
        Returns the action action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
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
    """
    Main function.

    Args:
    """
    comm = MPI.Comm.Get_parent()
    task_id = comm.Get_rank()
    args = parser.parse_args()
    cls = load("yarll.agents.ppo.dppo_worker:" + args.cls)
    config = json_to_dict(args.config)

    task = cls(args.env_id, task_id, comm, args.monitor_path, config, args.seed)
    task.run()


if __name__ == '__main__':
    main()
