# -*- coding: utf8 -*-

#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Other differences:
#  - Always choose the action with the highest probability
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

import os
from typing import Dict
import numpy as np

import tensorflow as tf
from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards, flatten, FastSaver
from yarll.misc.network_ops import conv2d, linear, normalized_columns_initializer
from yarll.misc.reporter import Reporter

class REINFORCE(Agent):
    """
    REINFORCE with baselines
    """
    def __init__(self, env, monitor_path: str, monitor: bool = False, video: bool = True, **usercfg) -> None:
        super(REINFORCE, self).__init__(**usercfg)
        self.env = env
        if monitor:
            self.env = wrappers.Monitor(self.env,
                                        monitor_path,
                                        force=True,
                                        video_callable=(None if video else False))
        self.monitor_path = monitor_path
        # Default configuration. Can be overwritten using keyword arguments.
        self.config.update(dict(
            batch_update="timesteps",
            timesteps_per_batch=1000,
            n_iter=100,
            gamma=0.99,  # Discount past rewards by a percentage
            learning_rate=0.05,
            entropy_coef=1e-3,
            n_hidden_layers=2,
            n_hidden_units=20,
            repeat_n_actions=1,
            save_model=False
        ))
        self.config.update(usercfg)

        self.states = tf.placeholder(
            tf.float32, [None] + list(self.env.observation_space.shape), name="states")  # Observation
        self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")  # Discrete action
        self.advantage = tf.placeholder(tf.float32, name="advantage")  # Advantage
        self.build_network()
        self.make_trainer()

        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = FastSaver()
        summary_loss = tf.summary.scalar("model/loss", self.summary_loss)
        summary_entropy = tf.summary.scalar("model/entropy", self.entropy)
        self.summary_op = tf.summary.merge([summary_loss, summary_entropy])

        self.init_op = tf.global_variables_initializer()
        # Launch the graph.
        self.session = tf.Session()
        self.writer = tf.summary.FileWriter(os.path.join(self.monitor_path, "task0"), self.session.graph)

        self.env_runner = EnvRunner(self.env, self, usercfg, summary_writer=self.writer)

    def _initialize(self) -> None:
        self.session.run(self.init_op)

    def build_network(self):
        raise NotImplementedError()

    def make_trainer(self):
        raise NotImplementedError()

    def choose_action(self, state, features) -> Dict[str, np.ndarray]:
        """Choose an action."""
        action = self.session.run([self.action], feed_dict={self.states: [state]})[0]
        return {"action": action}

    def learn(self):
        """Run learning algorithm"""
        self._initialize()
        reporter = Reporter()
        config = self.config
        total_n_trajectories = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.env_runner.get_trajectories()
            total_n_trajectories += len(trajectories)
            all_state = np.concatenate([trajectory.states for trajectory in trajectories])
            # Compute discounted sums of rewards
            rets = [discount_rewards(trajectory.rewards, config["gamma"]) for trajectory in trajectories]
            max_len = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(max_len - len(ret))]) for ret in rets]
            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action = np.concatenate([trajectory.actions for trajectory in trajectories])
            all_adv = np.concatenate(advs)
            # Do policy gradient update step
            episode_rewards = np.array([sum(trajectory.rewards) for trajectory in trajectories]) # episode total rewards
            episode_lengths = np.array([len(trajectory.rewards) for trajectory in trajectories]) # episode lengths
            # TODO: deal with RNN state
            summary, _ = self.session.run([self.summary_op, self.train], feed_dict={
                self.states: all_state,
                self.actions_taken: all_action,
                self.advantage: all_adv
            })
            self.writer.add_summary(summary, iteration)
            self.writer.flush()

            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
        if self.config["save_model"]:
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

class REINFORCEDiscrete(REINFORCE):
    def __init__(self, env, monitor_path: str, video: bool = True, **usercfg) -> None:
        super(REINFORCEDiscrete, self).__init__(env, monitor_path, video=video, **usercfg)

    def make_trainer(self):
        good_probabilities = tf.reduce_sum(tf.multiply(self.probs,
                                                       tf.one_hot(tf.cast(self.actions_taken, tf.int32),
                                                                  self.env_runner.nA)),
                                           reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * self.advantage
        loss = -tf.reduce_sum(eligibility)
        self.summary_loss = loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"])
        self.train = optimizer.minimize(loss)

    def build_network(self):

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=int(self.config["n_hidden_units"]),
            activation_fn=tf.tanh,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        self.probs = tf.contrib.layers.fully_connected(
            inputs=L1,
            num_outputs=self.env_runner.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

class REINFORCEDiscreteCNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        usercfg["n_hidden_units"] = 200
        super(REINFORCEDiscreteCNN, self).__init__(env, monitor_path, video=video, **usercfg)
        self.config.update(usercfg)

    def build_network(self):
        shape = list(self.env.observation_space.shape)

        x = self.states
        # Convolution layers
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        # Flatten
        shape = x.get_shape().as_list()
        reshape = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Fully connected layer 1
        self.L3 = tf.contrib.layers.fully_connected(
            inputs=reshape,
            num_outputs=int(self.config["n_hidden_units"]),
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(stddev=0.01),
            biases_initializer=tf.zeros_initializer())

        # Fully connected layer 2
        self.probs = tf.contrib.layers.fully_connected(
            inputs=self.L3,
            num_outputs=self.env_runner.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
            biases_initializer=tf.zeros_initializer())

        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

class REINFORCEDiscreteRNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(REINFORCEDiscreteRNN, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):

        n_states = tf.shape(self.states)[:1]
        states = tf.expand_dims(flatten(self.states), [0])

        enc_cell = tf.contrib.rnn.GRUCell(int(self.config["n_hidden_units"]))
        self.rnn_state_in = enc_cell.zero_state(1, tf.float32)
        L1, self.rnn_state_out = tf.nn.dynamic_rnn(cell=enc_cell,
                                                   inputs=states,
                                                   sequence_length=n_states,
                                                   initial_state=self.rnn_state_in,
                                                   dtype=tf.float32)
        self.probs = tf.contrib.layers.fully_connected(
            inputs=L1[0],
            num_outputs=self.env_runner.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
            biases_initializer=tf.zeros_initializer())
        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

    def choose_action(self, state, features):
        """Choose an action."""
        feed_dict = {
            self.states: [state],
            self.rnn_state_in: features
        }
        action, new_features = self.session.run([self.action, self.rnn_state_out], feed_dict=feed_dict)
        return {"action": action, "features": new_features}

class REINFORCEDiscreteCNNRNN(REINFORCEDiscreteRNN):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(REINFORCEDiscreteCNNRNN, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):

        shape = list(self.env.observation_space.shape)

        self.states = tf.placeholder(tf.float32, [None] + shape, name="states")
        self.N = tf.placeholder(tf.int32, name="N")

        x = self.states
        # Convolution layers
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        # Flatten
        shape = x.get_shape().as_list()
        reshape = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        reshape = tf.expand_dims(flatten(reshape), [0])
        self.enc_cell = tf.contrib.rnn.BasicLSTMCell(int(self.config["n_hidden_units"]))
        self.rnn_state_in = self.enc_cell.zero_state(1, tf.float32)
        self.L3, self.rnn_state_out = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                        inputs=reshape,
                                                        initial_state=self.rnn_state_in,
                                                        dtype=tf.float32)

        self.probs = tf.contrib.layers.fully_connected(
            inputs=self.L3[0],
            num_outputs=self.env_runner.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
            biases_initializer=tf.zeros_initializer())
        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

class REINFORCEContinuous(REINFORCE):
    def __init__(self, env, monitor_path, RNN=False, video=True, **usercfg):
        self.rnn = RNN
        super(REINFORCEContinuous, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        if self.rnn:
            self.build_network_rnn()
        else:
            self.build_network_normal()

    def make_trainer(self):
        loss = tf.reduce_mean(-self.action_log_prob * self.advantage) - self.config["entropy_coef"] * self.entropy
        self.summary_loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"])
        self.train = optimizer.minimize(loss)

    def build_network_normal(self):

        self.actions_taken = tf.placeholder(
            tf.float32,
            [None] + list(self.env.action_space.shape),
            name="actions_taken")

        x = self.states
        for i in range(int(self.config["n_hidden_layers"])):
            x = tf.tanh(linear(x, int(self.config["n_hidden_units"]), "L{}_mean".format(i + 1),
                               initializer=normalized_columns_initializer(1.0)))
        self.mean = linear(x, self.env.action_space.shape[0], "mean", initializer=normalized_columns_initializer(0.01))
        self.mean = tf.check_numerics(self.mean, "mean")

        self.log_std = tf.get_variable(
            name="logstd",
            shape=list(self.env.action_space.shape),
            initializer=tf.zeros_initializer()
        )
        std = tf.exp(self.log_std, name="std")
        std = tf.check_numerics(std, "std")

        self.action = self.mean + std * tf.random_normal(tf.shape(self.mean))
        self.action = tf.reshape(self.action, list(self.env.action_space.shape))

        neglogprob = 0.5 * tf.reduce_sum(tf.square((self.actions_taken - self.mean) / std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.actions_taken)[-1]) \
            + tf.reduce_sum(self.log_std, axis=-1)
        self.action_log_prob = -neglogprob
        self.entropy = tf.reduce_sum(self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def build_network_rnn(self):
        n_states = tf.shape(self.states)[:1]
        states = tf.expand_dims(flatten(self.states), [0])

        enc_cell = tf.contrib.rnn.GRUCell(int(self.config["n_hidden_units"]))
        L1, _ = tf.nn.dynamic_rnn(cell=enc_cell, inputs=states,
                                  sequence_length=n_states, dtype=tf.float32)

        L1 = L1[0]

        mu, sigma = mu_sigma_layer(L1, 1)

        self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.action = self.normal_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])
