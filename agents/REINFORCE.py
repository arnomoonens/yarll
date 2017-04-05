#!/usr/bin/env python
# -*- coding: utf8 -*-

#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Other differences:
#  - Always choose the action with the highest probability
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

import numpy as np
import os

import tensorflow as tf
from gym import wrappers
# import gym_ple

from agents.Agent import Agent
from misc.utils import discount_rewards, preprocess_image
from misc.Reporter import Reporter

class REINFORCE(Agent):
    """
    REINFORCE with baselines
    """
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(REINFORCE, self).__init__(env, **usercfg)
        self.env = wrappers.Monitor(self.env, monitor_path, force=True, video_callable=(None if video else False))
        self.monitor_path = monitor_path
        # Default configuration. Can be overwritten using keyword arguments.
        self.config.update(dict(
            batch_update="timesteps",
            timesteps_per_batch=10000,
            n_iter=100,
            gamma=0.99,  # Discount past rewards by a percentage
            decay=0.9,  # Decay of RMSProp optimizer
            epsilon=1e-9,  # Epsilon of RMSProp optimizer
            learning_rate=0.05,
            n_hidden_units=20,
            repeat_n_actions=1,
            save_model=False
        ))
        self.config.update(usercfg)
        self.build_network()
        init = tf.global_variables_initializer()
        # Launch the graph.
        self.session = tf.Session()
        self.session.run(init)
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()
        self.rewards = tf.placeholder("float", name="Rewards")
        self.episode_lengths = tf.placeholder("float", name="Episode_lengths")
        summary_loss = tf.summary.scalar("Loss", self.summary_loss)
        summary_rewards = tf.summary.scalar("Rewards", self.rewards)
        summary_episode_lengths = tf.summary.scalar("Episode_lengths", self.episode_lengths)
        self.summary_op = tf.summary.merge([summary_loss, summary_rewards, summary_episode_lengths])
        self.writer = tf.summary.FileWriter(os.path.join(self.monitor_path, "task0"), self.session.graph)

    def choose_action(self, state):
        """Choose an action."""
        action = self.session.run([self.action], feed_dict={self.states: [state]})[0]
        return action

    def learn(self):
        """Run learning algorithm"""
        reporter = Reporter()
        config = self.config
        total_n_trajectories = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.get_trajectories()
            total_n_trajectories += len(trajectories)
            all_state = np.concatenate([trajectory["state"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            rets = [discount_rewards(trajectory["reward"], config["gamma"]) for trajectory in trajectories]
            max_len = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(max_len - len(ret))]) for ret in rets]
            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_adv = np.concatenate(advs)
            # Do policy gradient update step
            episode_rewards = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # episode lengths
            summary, _ = self.session.run([self.summary_op, self.train], feed_dict={
                self.states: all_state,
                self.a_n: all_action,
                self.adv_n: all_adv,
                self.episode_lengths: np.mean(episode_lengths),
                self.rewards: np.mean(episode_rewards)
            })
            self.writer.add_summary(summary, iteration)
            self.writer.flush()

            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
        if self.config["save_model"]:
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

class REINFORCEDiscrete(REINFORCE):
    def __init__(self, env, monitor_path, rnn=False, video=True, **usercfg):
        self.nA = env.action_space.n
        self.rnn = rnn
        super(REINFORCEDiscrete, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        if self.rnn:
            self.build_network_rnn()
        else:
            self.build_network_normal()

    def build_network_normal(self):
        # Symbolic variables for observation, action, and advantage
        self.states = tf.placeholder(tf.float32, [None, self.nO], name="states")  # Observation
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.config["n_hidden_units"],
            activation_fn=tf.tanh,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        self.probs = tf.contrib.layers.fully_connected(
            inputs=L1,
            num_outputs=self.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

        good_probabilities = tf.reduce_sum(tf.multiply(self.probs, tf.one_hot(tf.cast(self.a_n, tf.int32), self.nA)), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * self.adv_n
        self.loss = -tf.reduce_sum(eligibility, name="loss")
        self.summary_loss = self.loss
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])
        self.train = optimizer.minimize(self.loss, name="train")

    def build_network_rnn(self):
        self.states = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape), name="states")  # Observation
        # self.n_states = tf.placeholder(tf.float32, shape=[None], name="n_states")  # Observation
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        n_states = tf.shape(self.states)[:1]

        states = tf.expand_dims(flatten(self.states), [0])

        enc_cell = tf.contrib.rnn.GRUCell(self.config["n_hidden_units"])
        L1, _ = tf.nn.dynamic_rnn(cell=enc_cell, inputs=states,
                                  sequence_length=n_states, dtype=tf.float32)
        self.probs = tf.contrib.layers.fully_connected(
            inputs=L1[0],
            num_outputs=self.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())
        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

        good_probabilities = tf.reduce_sum(tf.multiply(self.probs, tf.one_hot(tf.cast(self.a_n, tf.int32), self.nA)), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * self.adv_n
        eligibility = tf.Print(eligibility, [eligibility], first_n=5)
        loss = -tf.reduce_sum(eligibility)
        self.summary_loss = loss
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=0.9, epsilon=1e-9)
        self.train = optimizer.minimize(loss)

class REINFORCEContinuous(REINFORCE):
    def __init__(self, env, rnn, monitor_path, video=True, **usercfg):
        self.rnn = rnn
        super(REINFORCEContinuous, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        if self.rnn:
            self.build_network_rnn()
        else:
            self.build_network_normal()

    def build_network_normal(self):
        # Symbolic variables for observation, action, and advantage
        self.states = tf.placeholder(tf.float32, [None, self.nO], name="states")  # Observation
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Continuous action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.config["n_hidden_units"],
            activation_fn=tf.tanh,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        mu = tf.contrib.layers.fully_connected(
            inputs=L1,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())
        mu = tf.squeeze(mu, name="mu")

        sigma_L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.config["n_hidden_units"],
            activation_fn=tf.tanh,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())
        sigma = tf.contrib.layers.fully_connected(
            inputs=sigma_L1,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())
        sigma = tf.squeeze(sigma)
        sigma = tf.nn.softplus(sigma) + 1e-5

        self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.action = self.normal_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])
        loss = -self.normal_dist.log_prob(self.a_n) * self.adv_n
        # Add cross entropy cost to encourage exploration
        loss -= 1e-1 * self.normal_dist.entropy()
        loss = tf.clip_by_value(loss, -1e10, 1e10)
        self.summary_loss = tf.reduce_mean(loss)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=0.9, epsilon=1e-9)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"])
        self.train = optimizer.minimize(loss, global_step=tf.contrib.framework.get_global_step())

    def build_network_rnn(self):
        self.states = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape), name="states")  # Observation
        # self.n_states = tf.placeholder(tf.float32, shape=[None], name="n_states")  # Observation
        self.a_n = tf.placeholder(tf.float32, name="a_n")  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        n_states = tf.shape(self.states)[:1]

        states = tf.expand_dims(flatten(self.states), [0])

        enc_cell = tf.contrib.rnn.GRUCell(self.config["n_hidden_units"])
        L1, _ = tf.nn.dynamic_rnn(cell=enc_cell, inputs=states,
                                  sequence_length=n_states, dtype=tf.float32)

        L1 = L1[0]

        mu = tf.contrib.layers.fully_connected(
            inputs=L1,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())
        mu = tf.squeeze(mu, name="mu")

        sigma = tf.contrib.layers.fully_connected(
            inputs=L1,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())
        sigma = tf.squeeze(sigma)
        sigma = tf.nn.softplus(sigma) + 1e-5

        self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.action = self.normal_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])
        loss = -self.normal_dist.log_prob(self.a_n) * self.adv_n
        # Add cross entropy cost to encourage exploration
        loss -= 1e-1 * self.normal_dist.entropy()
        loss = tf.clip_by_value(loss, -1e10, 1e10)
        self.summary_loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"])
        self.train = optimizer.minimize(loss, global_step=tf.contrib.framework.get_global_step())

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

class REINFORCEDiscreteCNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        usercfg["n_hidden_units"] = 200
        super(REINFORCEDiscreteCNN, self).__init__(env, monitor_path, video=video, **usercfg)
        self.config.update(usercfg)

    def reset_env(self):
        return preprocess_image(self.env.reset())

    def step_env(self, action):
        state, reward, done, info = self.env.step(action)
        return preprocess_image(state), reward, done, info

    def build_network(self):
        image_size = 80
        image_depth = 1  # aka nr. of feature maps. Eg 3 for RGB images. 1 here because we use grayscale images

        self.states = tf.placeholder(tf.float32, [None, image_size, image_size, image_depth], name="states")
        self.a_n = tf.placeholder(tf.float32, name="a_n")
        self.N = tf.placeholder(tf.int32, name="N")
        self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage

        # Convolution layer 1
        depth = 32
        patch_size = 4
        self.w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, depth], stddev=0.01))
        self.b1 = tf.Variable(tf.zeros([depth]))
        self.L1 = tf.nn.relu(tf.nn.conv2d(self.states, self.w1, strides=[1, 2, 2, 1], padding="SAME") + self.b1)
        self.L1 = tf.nn.max_pool(self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Convolution layer 2
        self.w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.01))
        self.b2 = tf.Variable(tf.zeros([depth]))
        self.L2 = tf.nn.relu(tf.nn.conv2d(self.L1, self.w2, strides=[1, 2, 2, 1], padding="SAME") + self.b2)

        # Flatten
        shape = self.L2.get_shape().as_list()
        reshape = tf.reshape(self.L2, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Fully connected layer 1
        self.L3 = tf.contrib.layers.fully_connected(
            inputs=reshape,
            num_outputs=self.config["n_hidden_units"],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(stddev=0.01),
            biases_initializer=tf.zeros_initializer())

        # Fully connected layer 2
        self.probs = tf.contrib.layers.fully_connected(
            inputs=self.L3,
            num_outputs=self.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        good_probabilities = tf.reduce_sum(tf.multiply(self.probs, tf.one_hot(tf.cast(self.a_n, tf.int32), self.nA)), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * self.adv_n
        loss = -tf.reduce_sum(eligibility)
        self.summary_loss = loss
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=0.9, epsilon=1e-9)
        self.train = optimizer.minimize(loss)
