#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import sys
import tensorflow as tf
import gym
import logging
from gym.spaces import Discrete, Box
from Learner import Learner
from ActionSelection.CategoricalActionSelection import ProbabilisticCategoricalActionSelection
from utils import discount_rewards
from Reporter import Reporter

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

class A2C(Learner):
    """Advantage Actor Critic"""
    def __init__(self, ob_space, action_space, action_selection, **usercfg):
        super(A2C, self).__init__(ob_space, action_space, **usercfg)
        self.action_selection = action_selection

        self.config = dict(
            episode_max_length=100,
            timesteps_per_batch=1000,
            n_iter=200,
            gamma=0.99,
            actor_learning_rate=0.01,
            critic_learning_rate=0.05,
            actor_n_hidden=20,
            critic_n_hidden=20
        )
        self.config.update(usercfg)

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
        self.critic_returns = tf.placeholder("float", name="critic_returns")

        # Critic network
        critic_W0 = tf.Variable(tf.random_normal([self.nO, self.config['critic_n_hidden']]), name='W0')
        critic_b0 = tf.Variable(tf.zeros([self.config['actor_n_hidden']]), name='b0')
        critic_L1 = tf.tanh(tf.matmul(self.critic_state_in, critic_W0) + critic_b0[None, :], name='L1')

        critic_W1 = tf.Variable(tf.random_normal([self.config['actor_n_hidden'], 1]), name='W1')
        critic_b1 = tf.Variable(tf.zeros([1]), name='b1')
        self.critic_value = tf.matmul(critic_L1, critic_W1) + critic_b1[None, :]
        critic_loss = tf.reduce_mean(tf.square(self.critic_value - self.critic_returns))
        critic_loss = tf.Print(critic_loss, [critic_loss], message='Critic loss=')
        critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['critic_learning_rate'], decay=0.9, epsilon=1e-9)
        self.critic_train = critic_optimizer.minimize(critic_loss)

        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, ob):
        """Choose an action."""
        ob = ob.reshape(1, -1)
        prob = self.sess.run([self.prob_na], feed_dict={self.actor_input: ob})[0][0]
        action = self.action_selection.select_action(prob)
        return action

    def get_critic_value(self, ob):
        return self.sess.run([self.critic_value], feed_dict={self.critic_state_in: ob})[0].flatten()

    def learn(self, env):
        """Run learning algorithm"""
        reporter = Reporter()
        config = self.config
        possible_actions = np.arange(self.nA)
        total_n_trajectories = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.get_trajectories(env)
            total_n_trajectories += len(trajectories)
            all_action = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_action = (possible_actions == all_action[:, None]).astype(np.float32)
            all_ob = np.concatenate([trajectory["ob"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            returns = np.concatenate([discount_rewards(trajectory["reward"], config["gamma"]) for trajectory in trajectories])
            qw_new = self.get_critic_value(all_ob)

            self.sess.run([self.critic_train], feed_dict={self.critic_state_in: all_ob, self.critic_returns: returns.reshape(-1, 1)})
            self.sess.run([self.actor_train], feed_dict={self.actor_input: all_ob, self.actions_taken: all_action, self.critic_feedback: qw_new, self.critic_rewards: returns})

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
        agent = A2C(env.observation_space, env.action_space, action_selection, episode_max_length=env.spec.timestep_limit)
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    else:
        raise NotImplementedError
    try:
        env.monitor.start(sys.argv[2], force=True)
        agent.learn(env)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
