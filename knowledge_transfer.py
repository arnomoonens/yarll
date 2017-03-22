#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
import numpy as np
import tensorflow as tf
import logging
import argparse

from gym.spaces import Discrete

from Learner import Learner
from utils import discount_rewards, save_config
from Environment.CartPole import make_predef_CartPole_envs, make_random_CartPole_envs
from Reporter import Reporter
from gradient_ops import create_accumulative_gradients_op, add_accumulative_gradients_op, reset_accumulative_gradients_op

class TaskLearner(Learner):
    """Learner for a specific environment and with its own action selection."""
    def __init__(self, env, action, master, **usercfg):
        super(TaskLearner, self).__init__(env, **usercfg)
        self.action = action
        self.master = master

    def choose_action(self, state):
        """Choose an action."""
        return self.master.session.run([self.action], feed_dict={self.master.states: [state]})[0]

class KnowledgeTransferLearner(Learner):
    """Learner for variations of a task."""
    def __init__(self, envs, monitor_dir, **usercfg):
        super(KnowledgeTransferLearner, self).__init__(envs[0], **usercfg)
        self.envs = envs
        self.n_tasks = len(envs)
        self.monitor_dir = monitor_dir
        self.nA = envs[0].action_space.n
        self.config.update(dict(
            timesteps_per_batch=2000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=400,
            gamma=0.99,
            learning_rate=0.005,
            n_hidden_units=20,
            repeat_n_actions=1,
            n_sparse_units=10
        ))
        self.config.update(usercfg)
        self.build_networks()
        self.task_learners = [TaskLearner(envs[i], action, self, **self.config) for i, action in enumerate(self.action_tensors)]
        if self.config["save_model"]:
            for action_tensor in self.action_tensors:
                tf.add_to_collection("action", action_tensor)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()

    def build_networks(self):
        self.session = tf.Session()

        self.states = tf.placeholder(tf.float32, name="states")
        self.action_taken = tf.placeholder(tf.float32, name="action_taken")
        self.advantage = tf.placeholder(tf.float32, name="advantage")

        W0 = tf.Variable(tf.random_normal([self.nO, self.config["n_hidden_units"]]) / np.sqrt(self.nO), name='W0')
        b0 = tf.Variable(tf.zeros([self.config["n_hidden_units"]]), name='b0')
        # Action probabilities
        L1 = tf.tanh(tf.matmul(self.states, W0) + b0[None, :])

        knowledge_base = tf.Variable(tf.random_normal([self.config["n_hidden_units"], self.config["n_sparse_units"]]))

        self.shared_vars = [W0, b0, knowledge_base]

        # Every task has its own sparse representation
        sparse_representations = [tf.Variable(tf.random_normal([self.config["n_sparse_units"], self.nA])) for _ in range(self.n_tasks)]

        self.probs_tensors = [tf.nn.softmax(tf.matmul(L1, tf.matmul(knowledge_base, s))) for s in sparse_representations]

        self.action_tensors = [tf.squeeze(tf.multinomial(tf.log(probs), 1)) for probs in self.probs_tensors]

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=0.9, epsilon=1e-9)
        net_vars = self.shared_vars + sparse_representations
        self.accum_grads = create_accumulative_gradients_op(net_vars, 1)

        self.writers = []
        self.losses = []

        self.loss = tf.placeholder("float", name="loss")
        self.rewards = tf.placeholder("float", name="Rewards")
        self.episode_lengths = tf.placeholder("float", name="Episode_lengths")
        summary_loss = tf.summary.scalar("Loss", self.loss)
        summary_rewards = tf.summary.scalar("Rewards", self.rewards)
        summary_episode_lengths = tf.summary.scalar("Episode_lengths", self.episode_lengths)
        self.summary_op = tf.summary.merge([summary_loss, summary_rewards, summary_episode_lengths])

        for i, probabilities in enumerate(self.probs_tensors):
            good_probabilities = tf.reduce_sum(tf.multiply(probabilities, tf.one_hot(tf.cast(self.action_taken, tf.int32), self.nA)), reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * self.advantage
            # eligibility = tf.Print(eligibility, [eligibility], first_n=5)
            loss = -tf.reduce_sum(eligibility)
            self.losses.append(loss)
            writer = tf.summary.FileWriter(os.path.join(self.monitor_dir, "task" + str(i)), self.session.graph)
            self.writers.append(writer)

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
                results = self.session.run([self.losses[i], self.add_accum_grads[i]], feed_dict={
                    self.states: all_state,
                    self.action_taken: all_action,
                    self.advantage: all_adv
                })
                summary = self.session.run([self.summary_op], feed_dict={
                    self.loss: results[0],
                    self.rewards: np.mean(episode_rewards),
                    self.episode_lengths: np.mean(episode_lengths)
                })

                self.writers[i].add_summary(summary[0], iteration)
                self.writers[i].flush()
                print("Task:", i)
                reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories[i])

            # Apply accumulated gradient after all the gradients of each task are summed
            self.session.run([self.apply_gradients])

        if self.config["save_model"]:
            if not os.path.exists(self.monitor_dir):
                os.makedirs(self.monitor_dir)
            self.saver.save(self.session, os.path.join(self.monitor_dir, "model"))

parser = argparse.ArgumentParser()
parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the experiment on.")
parser.add_argument("monitor_path", metavar="monitor_path", type=str, help="Path where Gym monitor files may be saved")
parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate used when optimizing weights.")
parser.add_argument("--iterations", default=100, type=int, help="Number of iterations to run the algorithm.")
parser.add_argument("--save_model", action="store_true", default=False, help="Save resulting model.")
parser.add_argument("--random_envs", type=int, help="Number of environments with random parameters to generate.")

def main():
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)
    if args.environment != "CartPole-v0":
        raise NotImplementedError("Only the environment \"CartPole-v0\" is supported right now.")
    envs = make_random_CartPole_envs(args.random_envs) if args.random_envs else make_predef_CartPole_envs()
    if isinstance(envs[0].action_space, Discrete):
        agent = KnowledgeTransferLearner(
            envs, args.monitor_path,
            n_iter=args.iterations,
            save_model=args.save_model,
            learning_rate=args.learning_rate
        )
    else:
        raise NotImplementedError("Only environments with a discrete action space are supported right now.")
    save_config(args.monitor_path, agent.config, [env.to_dict() for env in envs])
    try:
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
