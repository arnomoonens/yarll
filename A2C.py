#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import sys
import tensorflow as tf
import gym
from gym.spaces import Discrete, Box
import logging
from Learner import Learner
from ActionSelection import ProbabilisticCategoricalActionSelection
from utils import discount_rewards
from Reporter import Reporter

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

class A2C(Learner):
    """Advantage Actor Critic"""
    def __init__(self, env, action_selection, **usercfg):
        super(A2C, self).__init__(env, **usercfg)
        self.action_selection = action_selection

        self.config = dict(
            episode_max_length=100,
            timesteps_per_batch=1000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=200,
            gamma=0.99,
            actor_learning_rate=0.01,
            critic_learning_rate=0.05,
            actor_n_hidden=20,
            critic_n_hidden=20
        )
        self.config.update(usercfg)
        self.build_networks()
        return

    def build_networks(self):
        self.nA = self.action_space.n
        self.actor_input = tf.placeholder(tf.float32, name='actor_input')
        self.actions_taken = tf.placeholder(tf.float32, name='actions_taken')
        self.critic_feedback = tf.placeholder(tf.float32, name='critic_feedback')
        self.critic_rewards = tf.placeholder(tf.float32, name='critic_rewards')

        # Actor network
        W0 = tf.Variable(tf.random_normal([self.nO, self.config['actor_n_hidden']]), name='W0')
        b0 = tf.Variable(tf.zeros([self.config['actor_n_hidden']]), name='b0')
        L1 = tf.tanh(tf.matmul(self.actor_input, W0) + b0[None, :], name='L1')

        W1 = tf.Variable(tf.random_normal([self.config['actor_n_hidden'], self.nA]), name='W1')
        b1 = tf.Variable(tf.zeros([self.nA]), name='b1')
        self.prob_na = tf.nn.softmax(tf.matmul(L1, W1) + b1[None, :], name='prob_na')

        good_probabilities = tf.reduce_sum(tf.mul(self.prob_na, self.actions_taken), reduction_indices=[1])
        eligibility = tf.log(tf.select(tf.equal(good_probabilities, tf.fill(tf.shape(good_probabilities), 0.0)), tf.fill(tf.shape(good_probabilities), 1e-30), good_probabilities)) \
            * (self.critic_rewards - self.critic_feedback)
        loss = -tf.reduce_mean(eligibility)
        loss = tf.Print(loss, [loss], message='Actor loss=')
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['actor_learning_rate'], decay=0.9, epsilon=1e-9)
        self.actor_train = optimizer.minimize(loss)

        self.critic_state_in = tf.placeholder("float", [None, self.nO], name='critic_state_in')
        self.critic_target = tf.placeholder("float", name="critic_target")

        # Critic network
        critic_W0 = tf.Variable(tf.random_normal([self.nO, self.config['critic_n_hidden']]), name='W0')
        critic_b0 = tf.Variable(tf.zeros([self.config['actor_n_hidden']]), name='b0')
        critic_L1 = tf.tanh(tf.matmul(self.critic_state_in, critic_W0) + critic_b0[None, :], name='L1')

        critic_W1 = tf.Variable(tf.random_normal([self.config['actor_n_hidden'], 1]), name='W1')
        critic_b1 = tf.Variable(tf.zeros([1]), name='b1')
        self.critic_value = tf.matmul(critic_L1, critic_W1) + critic_b1[None, :]
        critic_loss = tf.reduce_mean(tf.square(self.critic_target - self.critic_value))
        critic_loss = tf.Print(critic_loss, [critic_loss], message='Critic loss=')
        critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['critic_learning_rate'], decay=0.9, epsilon=1e-9)
        self.critic_train = critic_optimizer.minimize(critic_loss)

        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, state):
        """Choose an action."""
        state = state.reshape(1, -1)
        prob = self.sess.run([self.prob_na], feed_dict={self.actor_input: state})[0][0]
        action = self.action_selection.select_action(prob)
        return action

    def get_critic_value(self, state):
        return self.sess.run([self.critic_value], feed_dict={self.critic_state_in: state})[0].flatten()

    def learn(self):
        """Run learning algorithm"""
        reporter = Reporter()
        config = self.config
        possible_actions = np.arange(self.nA)
        total_n_trajectories = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.get_trajectories(self.env)
            total_n_trajectories += len(trajectories)
            all_action = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_action = (possible_actions == all_action[:, None]).astype(np.float32)
            all_state = np.concatenate([trajectory["state"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            returns = np.concatenate([discount_rewards(trajectory["reward"], config["gamma"]) for trajectory in trajectories])
            qw_new = self.get_critic_value(all_state)

            self.sess.run([self.critic_train], feed_dict={self.critic_state_in: all_state, self.critic_target: returns.reshape(-1, 1)})  # Reshape or not?
            self.sess.run([self.actor_train], feed_dict={self.actor_input: all_state, self.actions_taken: all_action, self.critic_feedback: qw_new, self.critic_rewards: returns})

            episode_rewards = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # episode lengths
            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
            # get_trajectory(self, env, config["episode_max_length"], render=True)

class A2CContinuous(A2C):
    """Advantage Actor Critic for continuous action spaces."""
    def __init__(self, env, action_selection, **usercfg):
        super(A2CContinuous, self).__init__(env, action_selection, **usercfg)

    def build_networks(self):
        self.input_state = tf.placeholder(tf.float32, name='input_state')
        self.actions_taken = tf.placeholder(tf.float32, name='actions_taken')
        self.target = tf.placeholder(tf.float32, name='target')

        mu_W0 = tf.Variable(tf.random_normal([self.nO, self.config['actor_n_hidden']]), name='W0')
        mu_b0 = tf.Variable(tf.zeros([self.config['actor_n_hidden']]), name='b0')
        mu_L1 = tf.tanh(tf.matmul(self.input_state, mu_W0) + mu_b0[None, :], name='L1')

        mu_W1 = tf.Variable(tf.random_normal([self.config['actor_n_hidden'], 1]), name='W1')
        mu_b1 = tf.Variable(tf.zeros([1]), name='b1')
        self.mu = tf.matmul(mu_L1, mu_W1) + mu_b1[None, :]

        # Actor network
        sigma_W0 = tf.Variable(tf.random_normal([self.nO, self.config['actor_n_hidden']]), name='W0')
        sigma_b0 = tf.Variable(tf.zeros([self.config['actor_n_hidden']]), name='b0')
        sigma_L1 = tf.tanh(tf.matmul(self.input_state, sigma_W0) + sigma_b0[None, :], name='L1')

        sigma_W1 = tf.Variable(tf.random_normal([self.config['actor_n_hidden'], 1]), name='W1')
        sigma_b1 = tf.Variable(tf.zeros([1]), name='b1')
        self.sigma = tf.matmul(sigma_L1, sigma_W1) + sigma_b1[None, :]

        self.sigma = tf.squeeze(self.sigma)
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.normal_dist.sample_n(1)
        self.action = tf.clip_by_value(self.action, self.action_space.low[0], self.action_space.high[0])

        # Loss and train op
        self.loss = -self.normal_dist.log_prob(self.actions_taken) * self.target
        # Add cross entropy cost to encourage exploration
        self.loss -= 1e-1 * self.normal_dist.entropy()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config['actor_learning_rate'])
        self.actor_train = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

        self.critic_state_in = tf.placeholder("float", [None, self.nO], name='critic_state_in')
        self.critic_target = tf.placeholder("float", name="critic_target")

        # Critic network
        critic_W0 = tf.Variable(tf.zeros([self.nO, self.config['critic_n_hidden']]), name='W0')
        critic_b0 = tf.Variable(tf.zeros([self.config['actor_n_hidden']]), name='b0')
        critic_L1 = tf.tanh(tf.matmul(self.critic_state_in, critic_W0) + critic_b0[None, :], name='L1')

        critic_W1 = tf.Variable(tf.zeros([self.config['actor_n_hidden'], 1]), name='W1')
        critic_b1 = tf.Variable(tf.zeros([1]), name='b1')
        self.critic_value = tf.squeeze(tf.matmul(critic_L1, critic_W1) + critic_b1[None, :])

        critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_target, self.critic_value))
        # critic_loss = tf.Print(critic_loss, [critic_loss], message='Critic loss=')
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.config['critic_learning_rate'])
        self.critic_train = critic_optimizer.minimize(critic_loss, global_step=tf.contrib.framework.get_global_step())

        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, state):
        """Choose an action."""
        state = state.reshape(1, -1)
        return self.sess.run([self.action], feed_dict={self.input_state: state})[0][0]

    def learn(self):
        """Run learning algorithm"""
        reporter = Reporter()
        config = self.config
        total_n_trajectories = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.get_trajectories(self.env)
            total_n_trajectories += len(trajectories)
            all_action = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_state = np.concatenate([trajectory["state"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            returns = np.concatenate([discount_rewards(trajectory["reward"], config["gamma"]) for trajectory in trajectories])
            qw_new = self.get_critic_value(all_state)

            print(qw_new)
            self.sess.run([self.critic_train], feed_dict={self.critic_state_in: all_state, self.critic_target: returns.reshape(-1, 1)})
            target = np.mean((returns - qw_new) ** 2)
            self.sess.run([self.actor_train], feed_dict={self.input_state: all_state, self.actions_taken: all_action, self.target: target})

            episode_rewards = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # episode lengths
            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
            # get_trajectory(self, env, config["episode_max_length"], render=True)

def main():
    if(len(sys.argv) < 3):
        logging.error("Please provide the name of an environment and a path to save monitor files")
        return
    env = gym.make(sys.argv[1])
    if isinstance(env.action_space, Discrete):
        action_selection = ProbabilisticCategoricalActionSelection()
        agent = A2C(env, action_selection, episode_max_length=env.spec.timestep_limit)
    elif isinstance(env.action_space, Box):
        action_selection = ProbabilisticCategoricalActionSelection()
        agent = A2CContinuous(env, action_selection, episode_max_length=env.spec.timestep_limit)
    else:
        raise NotImplementedError
    try:
        env.monitor.start(sys.argv[2], force=True)
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
