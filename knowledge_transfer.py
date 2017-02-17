#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import numpy as np
import tensorflow as tf
import logging
import argparse

import gym
from gym import wrappers
from gym.spaces import Discrete, Box

from Learner import Learner
from utils import discount_rewards
from ActionSelection import ProbabilisticCategoricalActionSelection
from Reporter import Reporter
from gradient_ops import create_accumulative_gradients_op, add_accumulative_gradients_op, reset_accumulative_gradients_op

class TaskLearner(Learner):
    """Learner for a specific environment and with its own action selection."""
    def __init__(self, env, action_selection, probabilities_fetch, master, **usercfg):
        super(TaskLearner, self).__init__(env, **usercfg)
        self.action_selection = action_selection
        self.probabilities_fetch = probabilities_fetch
        self.master = master

    def choose_action(self, state):
        """Choose an action."""
        probs = self.master.session.run([self.probabilities_fetch], feed_dict={self.master.state: [state]})[0][0]
        action = self.action_selection.select_action(probs)
        return action

class KnowledgeTransferLearner(Learner):
    """Learner for variations of a task."""
    def __init__(self, envs, action_selection, monitor_dir, **usercfg):
        super(KnowledgeTransferLearner, self).__init__(envs[0], **usercfg)
        self.envs = envs
        self.action_selection = action_selection
        self.monitor_dir = monitor_dir
        self.nA = envs[0].action_space.n
        self.config = dict(
            episode_max_length=envs[0].spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'),
            timesteps_per_batch=2000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=400,
            gamma=0.99,
            learning_rate=0.005,
            n_hidden_units=20,
            repeat_n_actions=1,
            n_task_variations=3,
            n_sparse_units=10
        )
        self.config.update(usercfg)
        self.build_networks()
        self.task_learners = [TaskLearner(envs[i], action_selection, probs, self, **self.config) for i, probs in enumerate(self.variation_probs)]

    def build_networks(self):
        self.session = tf.Session()

        self.state = tf.placeholder(tf.float32, name='state')
        self.action_taken = tf.placeholder(tf.float32, name='action_taken')
        self.advantage = tf.placeholder(tf.float32, name='advantage')

        W0 = tf.Variable(tf.random_normal([self.nO, self.config['n_hidden_units']]) / np.sqrt(self.nO), name='W0')
        b0 = tf.Variable(tf.zeros([self.config['n_hidden_units']]), name='b0')
        # Action probabilities
        L1 = tf.tanh(tf.matmul(self.state, W0) + b0[None, :])

        knowledge_base = tf.Variable(tf.random_normal([self.config['n_hidden_units'], self.config['n_sparse_units']]))

        self.shared_vars = [W0, b0, knowledge_base]

        # Every task has its own sparse representation
        sparse_representations = [tf.Variable(tf.random_normal([self.config['n_sparse_units'], self.nA])) for _ in range(self.config['n_task_variations'])]

        self.variation_probs = [tf.nn.softmax(tf.matmul(L1, tf.matmul(knowledge_base, s))) for s in sparse_representations]
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['learning_rate'], decay=0.9, epsilon=1e-9)
        net_vars = self.shared_vars + sparse_representations
        self.accum_grads = create_accumulative_gradients_op(net_vars, 1)

        # self.writers = []
        self.losses = []
        for i, probabilities in enumerate(self.variation_probs):
            good_probabilities = tf.reduce_sum(tf.multiply(probabilities, tf.one_hot(tf.cast(self.action_taken, tf.int32), self.nA)), reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * self.advantage
            # eligibility = tf.Print(eligibility, [eligibility], first_n=5)
            loss = -tf.reduce_sum(eligibility)
            self.losses.append(loss)
            # writer = tf.summary.FileWriter(self.monitor_dir + '/task' + str(i), self.session.graph)

        # An add op for every task & its loss
        # add_accumulative_gradients_op(net_vars, accum_grads, loss, identifier)
        self.add_accum_grads = [add_accumulative_gradients_op(
            self.shared_vars + [sparse_representations[i]],
            self.accum_grads,
            loss,
            i)
            for i, loss in enumerate(self.losses)]

        self.apply_gradients = self.optimizer.apply_gradients(
            zip(self.accum_grads, net_vars))
        self.reset_accum_grads = reset_accumulative_gradients_op(net_vars, self.accum_grads, 1)

        init = tf.global_variables_initializer()

        # Launch the graph.
        self.session.run(init)

    def learn(self):
        """Run learning algorithm"""
        reporter = Reporter()
        config = self.config
        total_n_trajectories = np.zeros(len(self.envs))
        for iteration in range(config["n_iter"]):
            self.session.run([self.reset_accum_grads])
            for i, learner in enumerate(self.task_learners):
                # Collect trajectories until we get timesteps_per_batch total timesteps
                trajectories = learner.get_trajectories()
                total_n_trajectories[i] += len(trajectories)
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
                self.session.run([self.add_accum_grads[i]], feed_dict={
                    self.state: all_state,
                    self.action_taken: all_action,
                    self.advantage: all_adv
                })
                # summary = self.session.run([self.master.summary_op], feed_dict={
                #     self.reward: reward
                #     # self.master.episode_length: trajectory["steps"]
                # })

                # self.writer.add_summary(summary[0], iteration)
                # self.writer.flush()
                print("Task:", i)
                reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories[i])

            # Apply accumulated gradient after all the gradients of each task are summed
            self.session.run([self.apply_gradients])

                # self.writer.add_summary(result[0], iteration)
                # self.writer.flush()

parser = argparse.ArgumentParser()
parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the experiment on.")
parser.add_argument("monitor_path", metavar="monitor_path", type=str, help="Path where Gym monitor files may be saved")

def make_envs(env_name):
    """Make variations of the same game."""
    envs = []
    envs.append(gym.make(env_name))  # First one has the standard behaviour
    env = gym.make(env_name)
    env.length = 0.25  # 5 times longer
    env.masspole = 0.5  # 5 times heavier
    env.total_mass = (env.masspole + env.masscart)  # Recalculate
    env.polemass_length = (env.masspole * env.length)  # Recalculate
    envs.append(env)
    env = gym.make(env_name)
    env.length = 0.025  # 2 times shorter
    env.masspole = 0.05  # 2 times lighter
    env.total_mass = (env.masspole + env.masscart)  # Recalculate
    env.polemass_length = (env.masspole * env.length)  # Recalculate
    envs.append(env)
    return envs

def main():
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    if args.environment != "CartPole-v0":
        raise NotImplementedError("Only the environment \"CartPole-v0\" is supported right now.")
    envs = make_envs(args.environment)
    if isinstance(envs[0].action_space, Discrete):
        action_selection = ProbabilisticCategoricalActionSelection()
        agent = KnowledgeTransferLearner(envs, action_selection, args.monitor_path)
    else:
        raise NotImplementedError("Only environments with a discrete action space are supported right now.")
    try:
        # agent.env = wrappers.Monitor(agent.env, args.monitor_path, force=True)
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
