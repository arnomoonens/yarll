#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
import numpy as np
import tensorflow as tf
import logging
import argparse

from gym import wrappers
from gym.spaces import Discrete, Box

from Learner import Learner
from utils import discount_rewards, save_config, ge_1
from Reporter import Reporter
from ActionSelection import ProbabilisticCategoricalActionSelection, ContinuousActionSelection
from Environment.registration import make_environment

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

class A2C(Learner):
    """Advantage Actor Critic"""
    def __init__(self, env, action_selection, monitor_dir, **usercfg):
        super(A2C, self).__init__(env, **usercfg)
        self.action_selection = action_selection
        self.monitor_dir = monitor_dir

        self.config.update(dict(
            timesteps_per_batch=10000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=400,
            gamma=0.99,
            actor_learning_rate=0.01,
            critic_learning_rate=0.05,
            actor_n_hidden=20,
            critic_n_hidden=20,
            repeat_n_actions=1
        ))
        self.config.update(usercfg)
        self.build_networks()
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
        summary_actor_loss = tf.summary.scalar("Actor_loss", self.summary_actor_loss)
        summary_critic_loss = tf.summary.scalar("Critic_loss", self.summary_critic_loss)
        summary_rewards = tf.summary.scalar("Rewards", self.rewards)
        summary_episode_lengths = tf.summary.scalar("Episode_lengths", self.episode_lengths)
        self.summary_op = tf.summary.merge([summary_actor_loss, summary_critic_loss, summary_rewards, summary_episode_lengths])
        self.writer = tf.summary.FileWriter(os.path.join(self.monitor_dir, "summaries"), self.session.graph)
        return

    def get_critic_value(self, state):
        return self.session.run([self.critic_value], feed_dict={self.states: state})[0].flatten()

    def choose_action(self, state):
        """Choose an action."""
        return self.session.run([self.action], feed_dict={self.states: [state]})[0]

    def learn(self):
        """Run learning algorithm"""
        reporter = Reporter()
        config = self.config
        possible_actions = np.arange(self.nA)
        total_n_trajectories = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.get_trajectories()
            total_n_trajectories += len(trajectories)
            all_action = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_action = (possible_actions == all_action[:, None]).astype(np.float32)
            all_state = np.concatenate([trajectory["state"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            returns = np.concatenate([discount_rewards(trajectory["reward"], config["gamma"]) for trajectory in trajectories])
            qw_new = self.get_critic_value(all_state)

            episode_rewards = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # episode lengths

            results = self.session.run([self.summary_op, self.critic_train, self.actor_train], feed_dict={
                self.states: all_state,
                self.critic_target: returns,
                self.states: all_state,
                self.actions_taken: all_action,
                self.critic_feedback: qw_new,
                self.critic_rewards: returns,
                self.rewards: np.mean(episode_rewards),
                self.episode_lengths: np.mean(episode_lengths)
            })
            self.writer.add_summary(results[0], iteration)
            self.writer.flush()

            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver.save(self.session, os.path.join(self.monitor_dir, "model"))

class A2CDiscrete(A2C):
    """A2C learner for a discrete action space"""
    def __init__(self, env, action_selection, monitor_dir, **usercfg):
        self.nA = env.action_space.n
        super(A2CDiscrete, self).__init__(env, action_selection, monitor_dir, **usercfg)

    def build_networks(self):
        self.states = tf.placeholder(tf.float32, [None, self.nO], name="states")
        self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
        self.critic_feedback = tf.placeholder(tf.float32, name="critic_feedback")
        self.critic_rewards = tf.placeholder(tf.float32, name="critic_rewards")

        # Actor network
        with tf.variable_scope("actor"):
            L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.config["actor_n_hidden"],
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            self.probs = tf.contrib.layers.fully_connected(
                inputs=L1,
                num_outputs=self.nA,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="probs")

            self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

            good_probabilities = tf.reduce_sum(tf.multiply(self.probs, self.actions_taken), reduction_indices=[1])
            eligibility = tf.log(tf.where(tf.equal(good_probabilities, tf.fill(tf.shape(good_probabilities), 0.0)), tf.fill(tf.shape(good_probabilities), 1e-30), good_probabilities)) \
                * (self.critic_rewards - self.critic_feedback)
            loss = tf.negative(tf.reduce_mean(eligibility), name="loss")
            self.summary_actor_loss = loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config["actor_learning_rate"])
            self.actor_train = self.optimizer.minimize(loss, global_step=tf.contrib.framework.get_global_step())

        with tf.variable_scope("critic"):
            self.critic_target = tf.placeholder("float", name="critic_target")
            # Critic network
            critic_L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.config["critic_n_hidden"],
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            self.critic_value = tf.contrib.layers.fully_connected(
                inputs=critic_L1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="value")

            critic_loss = tf.reduce_mean(tf.square(self.critic_target - self.critic_value), name="loss")
            self.summary_critic_loss = critic_loss
            critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.config["critic_learning_rate"])
            self.critic_train = critic_optimizer.minimize(critic_loss, global_step=tf.contrib.framework.get_global_step())

class A2CContinuous(A2C):
    """Advantage Actor Critic for continuous action spaces."""
    def __init__(self, env, action_selection, monitor_dir, **usercfg):
        super(A2CContinuous, self).__init__(env, action_selection, monitor_dir, **usercfg)

    def build_networks(self):
        self.states = tf.placeholder(tf.float32, [None, self.ob_space.shape[0]], name="states")
        self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
        self.critic_feedback = tf.placeholder(tf.float32, name="critic_feedback")
        self.critic_rewards = tf.placeholder(tf.float32, name="critic_rewards")

        # Actor network
        with tf.variable_scope("actor"):
            mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.states, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())
            mu = tf.squeeze(mu, name="mu")

            sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.states, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())
            sigma = tf.squeeze(sigma)
            sigma = tf.add(tf.nn.softplus(sigma), 1e-5, name="sigma")

            self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
            self.action = self.normal_dist.sample(1)
            self.action = tf.clip_by_value(self.action, self.action_space.low[0], self.action_space.high[0], name="action")

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(tf.squeeze(self.actions_taken)) * (self.critic_rewards - self.critic_feedback)
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()
            self.summary_actor_loss = tf.reduce_mean(self.loss)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config["actor_learning_rate"])
            self.actor_train = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

        with tf.variable_scope("critic"):
            self.critic_target = tf.placeholder("float", name="critic_target")

            # Critic network
            critic_L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.config["critic_n_hidden"],
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer())

            self.critic_value = tf.contrib.layers.fully_connected(
                inputs=critic_L1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer())

            critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_target, self.critic_value))
            self.summary_critic_loss = critic_loss
            critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.config["critic_learning_rate"])
            self.critic_train = critic_optimizer.minimize(critic_loss, global_step=tf.contrib.framework.get_global_step())

    def learn(self):
        """Run learning algorithm"""
        reporter = Reporter()
        config = self.config
        total_n_trajectories = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.get_trajectories()
            total_n_trajectories += len(trajectories)
            all_action = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_state = np.concatenate([trajectory["state"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            returns = np.concatenate([discount_rewards(trajectory["reward"], config["gamma"]) for trajectory in trajectories])
            qw_new = self.get_critic_value(all_state)

            episode_rewards = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # episode lengths

            results = self.session.run([self.summary_op, self.critic_train, self.actor_train], feed_dict={
                self.states: all_state,
                self.critic_target: returns,
                self.states: all_state,
                self.actions_taken: all_action,
                self.critic_feedback: qw_new,
                self.critic_rewards: returns,
                self.rewards: np.mean(episode_rewards),
                self.episode_lengths: np.mean(episode_lengths)
            })
            self.writer.add_summary(results[0], iteration)
            self.writer.flush()

            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)

        if self.config["save_model"]:
            self.saver.save(self.session, os.path.join(self.monitor_dir, "model"))

parser = argparse.ArgumentParser()
parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the experiment on.")
parser.add_argument("monitor_path", metavar="monitor_path", type=str, help="Path where Gym monitor files may be saved")
parser.add_argument("--no_video", dest="video", action="store_false", default=True, help="Don't render and show video.")
parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate used when optimizing weights.")
parser.add_argument("--iterations", default=100, type=ge_1, help="Number of iterations to run the algorithm.")
parser.add_argument("--save_model", action="store_true", default=False, help="Save resulting model.")

def main():
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)
    env = make_environment(args.environment)
    shared_args = {
        "monitor_dir": args.monitor_path,
        "n_iter": args.iterations,
        "save_model": args.save_model,
        "learning_rate": args.learning_rate
    }
    if isinstance(env.action_space, Discrete):
        action_selection = ProbabilisticCategoricalActionSelection()
        agent = A2CDiscrete(env, action_selection, **shared_args)
    elif isinstance(env.action_space, Box):
        action_selection = ContinuousActionSelection()
        agent = A2CContinuous(env, action_selection, **shared_args)
    else:
        raise NotImplementedError
    save_config(args.monitor_path, agent.config, [env.to_dict()])
    try:
        agent.env = wrappers.Monitor(agent.env, args.monitor_path, force=True, video_callable=(None if args.video else False))
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
