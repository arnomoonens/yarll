#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import numpy as np
import tensorflow as tf
import logging
import argparse

import gym
from gym.spaces import Discrete, Box

from Learner import Learner
from utils import discount_rewards
from ActionSelection import ProbabilisticCategoricalActionSelection

class KnowledgeTransferLearner(Learner):
    """Learner for variations of a task."""
    def __init__(self, env, action_selection, monitor_dir, **usercfg):
        super(KnowledgeTransferLearner, self).__init__(env, **usercfg)
        self.action_selection = action_selection
        self.monitor_dir = monitor_dir
        self.config = dict(
            episode_max_length=env.spec.timestep_limit,
            timesteps_per_batch=2000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=400,
            gamma=0.99,
            actor_learning_rate=0.01,
            critic_learning_rate=0.05,
            actor_n_hidden=20,
            critic_n_hidden=20,
            repeat_n_actions=1,
            n_task_variations=3,
            n_sparse_units=10
        )
        self.config.update(usercfg)
        self.build_networks()

    def build_networks(self):
        self.state = tf.placeholder(tf.float32, name='state')
        self.action_taken = tf.placeholder(tf.float32, name='action_taken')
        self.advantage = tf.placeholder(tf.float32, name='advantage')

        W0 = tf.Variable(tf.random_normal([self.nO, self.config['n_hidden_units']]) / np.sqrt(self.nO), name='W0')
        b0 = tf.Variable(tf.zeros([self.config['n_hidden_units']]), name='b0')
        # Action probabilities
        L1 = tf.tanh(tf.matmul(self.state, W0) + b0[None, :])

        knowledge_base = tf.Variable(tf.random.normal([self.config['n_hidden_units'], self.config['n_sparse_units']]))
        sparse_representations = [tf.Variable(tf.random.normal([self.config['n_sparse_units'], self.nA])) for _ in range(self.config['n_task_variations'])]

        variation_probs = [tf.nn.softmax(tf.matmul(L1, tf.matmul(knowledge_base, s))) for s in sparse_representations]

        self.losses = []
        self.trainers = []
        for probabilities in variation_probs:
            good_probabilities = tf.reduce_sum(tf.mul(probabilities, tf.one_hot(tf.cast(self.action_taken, tf.int32), self.nA)), reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * self.advantage
            # eligibility = tf.Print(eligibility, [eligibility], first_n=5)
            loss = -tf.reduce_sum(eligibility)
            self.losses.append(loss)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['learning_rate'], decay=0.9, epsilon=1e-9)
            self.trainers.append(optimizer.minimize(loss))

        init = tf.global_variables_initializer()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, state, task):
        """Choose an action."""
        probs = self.sess.run([self.variation_probs[task]], feed_dict={self.state: [state]})[0][0]
        action = self.action_selection.select_action(probs)
        return action

    # TODO: Need to implement:
    # - Collect trajectories for each task
    # - Compute the loss for each task C
    # - Compute the gradients
    # - Sum gradients
    # - Apply gradients
    # Back to start
    def learn(self):
        """Run learning algorithm"""
        # reporter = Reporter()
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
            result = self.sess.run([self.summary_op, self.train], feed_dict={
                                   self.state: all_state,
                                   self.a_n: all_action,
                                   self.adv_n: all_adv,
                                   self.episode_lengths: np.mean(episode_lengths),
                                   self.rewards: np.mean(episode_rewards)
                                   })
            self.writer.add_summary(result[0], iteration)
            self.writer.flush()

parser = argparse.ArgumentParser()
parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the experiment on.")
parser.add_argument("monitor_path", metavar="monitor_path", type=str, help="Path where Gym monitor files may be saved")

def main():
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    env = gym.make(args.environment)
    if isinstance(env.action_space, Discrete):
        action_selection = ProbabilisticCategoricalActionSelection()
        agent = KnowledgeTransferLearner(env, action_selection, args.monitor_path)
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    else:
        raise NotImplementedError
    try:
        env.monitor.start(args.monitor_path, force=True)
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
