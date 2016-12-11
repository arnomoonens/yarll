#!/usr/bin/env python
# -*- coding: utf8 -*-

#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Other differences:
#  - Always choose the action with the highest probability
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

import numpy as np
import sys
import tensorflow as tf
import gym
from gym.spaces import Discrete, Box
from Learner import Learner
from ActionSelection import ProbabilisticCategoricalActionSelection
from utils import discount_rewards, preprocess_image
from Reporter import Reporter

class REINFORCELearner(Learner):
    """
    REINFORCE with baselines
    """

    def __init__(self, env, action_selection, **usercfg):
        super(REINFORCELearner, self).__init__(env, **usercfg)
        self.action_selection = action_selection
        # Default configuration. Can be overwritten using keyword arguments.
        self.config = dict(
            episode_max_length=env.spec.timestep_limit,
            batch_update="timesteps",
            timesteps_per_batch=10000,
            n_iter=100,
            gamma=1.0,
            learning_rate=0.05,
            n_hidden_units=20,
            repeat_n_actions=1
        )
        self.config.update(usercfg)

    def act(self, ob):
        """Choose an action."""
        pass

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
            self.sess.run([self.train], feed_dict={self.state: all_state, self.a_n: all_action, self.adv_n: all_adv})
            episode_rewards = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # episode lengths
            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
            # get_trajectory(self, env, config["episode_max_length"], render=True)

class REINFORCELearnerDiscrete(REINFORCELearner):

    def __init__(self, env, action_selection, **usercfg):
        super(REINFORCELearnerDiscrete, self).__init__(env, action_selection, **usercfg)
        self.nA = self.action_space.n
        self.build_network()

    def build_network(self):
        # Symbolic variables for observation, action, and advantage
        # These variables stack the results from many timesteps--the first dimension is the timestep
        self.state = tf.placeholder(tf.float32, name='state')  # Observation
        self.a_n = tf.placeholder(tf.float32, name='a_n')  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name='adv_n')  # Advantage

        W0 = tf.Variable(tf.random_normal([self.nO, self.config['n_hidden_units']]) / np.sqrt(self.nO), name='W0')
        b0 = tf.Variable(tf.zeros([self.config['n_hidden_units']]), name='b0')
        W1 = tf.Variable(1e-4 * tf.random_normal([self.config['n_hidden_units'], self.nA]), name='W1')
        b1 = tf.Variable(tf.zeros([self.nA]), name='b1')
        # Action probabilities
        L1 = tf.tanh(tf.matmul(self.state, W0) + b0[None, :])
        self.probs = tf.nn.softmax(tf.matmul(L1, W1) + b1[None, :], name='probs')

        good_probabilities = tf.reduce_sum(tf.mul(self.probs, tf.one_hot(tf.cast(self.a_n, tf.int32), self.nA)), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * self.adv_n
        eligibility = tf.Print(eligibility, [eligibility], first_n=5)
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['learning_rate'], decay=0.9, epsilon=1e-9)
        self.train = optimizer.minimize(loss)

        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, state):
        """Choose an action."""
        probs = self.sess.run([self.probs], feed_dict={self.state: [state]})[0][0]
        action = self.action_selection.select_action(probs)
        return action

class REINFORCELearnerDiscreteCNN(REINFORCELearnerDiscrete):
    def __init__(self, env, action_selection, **usercfg):
        super(REINFORCELearnerDiscreteCNN, self).__init__(env, action_selection, **usercfg)
        self.config['n_hidden_units'] = 200
        self.config.update(usercfg)

    def reset_env(self):
        return preprocess_image(self.env.reset())

    def step_env(self, action):
        state, reward, done, info = self.env.step(action)
        return preprocess_image(state), reward, done, info

    def build_network(self):
        image_size = 80
        image_depth = 1  # aka nr. of feature maps. Eg 3 for RGB images. 1 here because we use grayscale images

        self.state = tf.placeholder(tf.float32, [None, image_size, image_size, image_depth], name="state")
        self.a_n = tf.placeholder(tf.float32, name='a_n')
        self.N = tf.placeholder(tf.int32, name='N')
        self.adv_n = tf.placeholder(tf.float32, name='adv_n')  # Advantage

        # Convolution layer 1
        depth = 32
        patch_size = 4
        self.w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, depth], stddev=0.01))
        self.b1 = tf.Variable(tf.zeros([depth]))
        self.L1 = tf.nn.relu(tf.nn.conv2d(self.state, self.w1, strides=[1, 2, 2, 1], padding="SAME") + self.b1)
        self.L1 = tf.nn.max_pool(self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolution layer 2
        self.w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.01))
        self.b2 = tf.Variable(tf.zeros([depth]))
        self.L2 = tf.nn.relu(tf.nn.conv2d(self.L1, self.w2, strides=[1, 2, 2, 1], padding="SAME") + self.b2)

        # Flatten
        shape = self.L2.get_shape().as_list()
        reshape = tf.reshape(self.L2, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Fully connected layer 1
        self.w3 = tf.Variable(tf.truncated_normal([image_size // 8 * image_size // 8 * depth, self.config['n_hidden_units']], stddev=0.01))
        self.b3 = tf.Variable(tf.zeros([self.config['n_hidden_units']]))
        self.L3 = tf.nn.relu(tf.matmul(reshape, self.w3) + self.b3)

        # Fully connected layer 2
        self.w4 = tf.Variable(tf.truncated_normal([self.config['n_hidden_units'], self.nA]))
        self.b4 = tf.Variable(tf.zeros([self.nA]))
        self.probs = tf.nn.softmax(tf.matmul(self.L3, self.w4) + self.b4)

        good_probabilities = tf.reduce_sum(tf.mul(self.probs, tf.one_hot(tf.cast(self.a_n, tf.int32), self.nA)), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * self.adv_n
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['learning_rate'], decay=0.9, epsilon=1e-9)
        self.train = optimizer.minimize(loss)

        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)

def main():
    if(len(sys.argv) < 3):
        print("Please provide the name of an environment and a path to save monitor files")
        return
    env = gym.make(sys.argv[1])
    if isinstance(env.action_space, Discrete):
        action_selection = ProbabilisticCategoricalActionSelection()
        rank = len(env.observation_space.shape)  # Observation space rank
        if rank == 1:
            agent = REINFORCELearnerDiscrete(env, action_selection)
        else:
            agent = REINFORCELearnerDiscreteCNN(env, action_selection)
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    else:
        raise NotImplementedError
    try:
        env.monitor.start(sys.argv[2], force=True)
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
