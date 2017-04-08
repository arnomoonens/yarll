#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf
import logging
from threading import Thread
import signal

from agents.Agent import Agent
from misc.utils import discount_rewards
from misc.Reporter import Reporter
from misc.gradient_ops import create_accumulative_gradients_op, add_accumulative_gradients_op, reset_accumulative_gradients_op
from agents.knowledge_transfer import TaskLearner

class AKTThread(Thread):
    """Asynchronous knowledge transfer learner thread. Used to learn using one specific variation of a task."""
    def __init__(self, master, env, task_id):
        super(AKTThread, self).__init__()
        self.master = master
        self.task_id = task_id
        self.add_accum_grad = None  # To be filled in later

        self.build_networks()
        self.states = self.master.states
        self.session = self.master.session
        self.task_learner = TaskLearner(env, self.action, self, **self.master.config)

        # Write the summary of each task in a different directory
        self.writer = tf.summary.FileWriter(os.path.join(self.master.monitor_path, "task" + str(self.task_id)), self.master.session.graph)

    def build_networks(self):
        with tf.variable_scope("task{}".format(self.task_id)):
            self.sparse_representation = tf.Variable(tf.random_normal([self.master.config["n_sparse_units"], self.master.nA]))
            self.probs = tf.nn.softmax(tf.matmul(self.master.L1, tf.matmul(self.master.knowledge_base, self.sparse_representation)))

            self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

            good_probabilities = tf.reduce_sum(tf.multiply(self.probs, tf.one_hot(tf.cast(self.master.action_taken, tf.int32), self.master.nA)), reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * self.master.advantage
            self.loss = -tf.reduce_sum(eligibility)

    def choose_action(self, state):
        """Choose an action."""
        return self.master.session.run([self.action], feed_dict={self.master.states: [state]})[0]

    def run(self):
        """Run the appropriate learning algorithm."""
        if self.master.learning_method == "REINFORCE":
            self.learn_REINFORCE()
        else:
            self.learn_Karpathy()

    def learn_REINFORCE(self):
        """Learn using updates like in the REINFORCE algorithm."""
        reporter = Reporter()
        config = self.master.config
        total_n_trajectories = 0
        iteration = 0
        while iteration < config["n_iter"] and not self.master.stop_requested:
            iteration += 1
            self.master.session.run([self.master.reset_accum_grads])
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.task_learner.get_trajectories()
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
            results = self.master.session.run([self.loss, self.add_accum_grad], feed_dict={
                self.master.states: all_state,
                self.master.action_taken: all_action,
                self.master.advantage: all_adv
            })
            print("Task:", self.task_id)
            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
            results = self.master.session.run([self.master.summary_op], feed_dict={
                self.master.loss: results[0],
                self.master.reward: np.mean(episode_rewards),
                self.master.episode_length: np.mean(episode_lengths)
            })
            self.writer.add_summary(results[0], iteration)
            self.writer.flush()
            self.master.session.run([self.master.apply_gradients])

    def learn_Karpathy(self):
        """Learn using updates like in the Karpathy algorithm."""
        config = self.master.config
        self.master.session.run([self.master.reset_accum_grads])

        iteration = 0
        while iteration < config['n_iter'] and not self.master.stop_requested:  # Keep executing episodes until the master requests a stop (e.g. using SIGINT)
            iteration += 1
            trajectory = self.task_learner.get_trajectory()
            reward = sum(trajectory["reward"])
            action_taken = trajectory["action"]

            discounted_episode_rewards = discount_rewards(trajectory["reward"], config["gamma"])
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            std = np.std(discounted_episode_rewards)
            std = std if std > 0 else 1
            discounted_episode_rewards /= std
            feedback = discounted_episode_rewards

            results = self.master.session.run([self.loss, self.add_accum_grad], feed_dict={
                self.master.states: trajectory["state"],
                self.master.action_taken: action_taken,
                self.master.advantage: feedback
            })
            results = self.master.session.run([self.master.summary_op], feed_dict={
                self.master.loss: results[0],
                self.master.reward: reward,
                self.master.episode_length: trajectory["steps"]
            })
            self.writer.add_summary(results[0], iteration)
            self.writer.flush()

            self.master.session.run([self.master.apply_gradients])
            self.master.session.run([self.master.reset_accum_grads])


class AsyncKnowledgeTransfer(Agent):
    """Asynchronous learner for variations of a task."""
    def __init__(self, envs, monitor_path, learning_method="REINFORCE", **usercfg):
        super(AsyncKnowledgeTransfer, self).__init__(envs[0], **usercfg)
        self.envs = envs
        self.learning_method = learning_method
        self.monitor_path = monitor_path
        self.nA = envs[0].action_space.n
        self.config.update(dict(
            timesteps_per_batch=10000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=200,
            gamma=0.99,  # Discount past rewards by a percentage
            decay=0.9,  # Decay of RMSProp optimizer
            epsilon=1e-9,  # Epsilon of RMSProp optimizer
            learning_rate=0.005,
            n_hidden_units=20,
            repeat_n_actions=1,
            n_task_variations=3,
            n_sparse_units=10,
        ))
        self.config.update(usercfg)

        self.stop_requested = False

        self.session = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True))

        self.build_networks()

        self.loss = tf.placeholder("float", name="loss")
        summary_loss = tf.summary.scalar("Loss", self.loss)
        self.reward = tf.placeholder("float", name="reward")
        summary_rewards = tf.summary.scalar("Reward", self.reward)
        self.episode_length = tf.placeholder("float", name="episode_length")
        summary_episode_lengths = tf.summary.scalar("Episode_length", self.episode_length)
        self.summary_op = tf.summary.merge([summary_loss, summary_rewards, summary_episode_lengths])

        self.jobs = [self.make_thread(env, i) for i, env in enumerate(self.envs)]

        net_vars = self.shared_vars + [job.sparse_representation for job in self.jobs]
        self.accum_grads = create_accumulative_gradients_op(net_vars, 1)
        for job in self.jobs:
            job.add_accum_grad = add_accumulative_gradients_op(
                self.shared_vars + [job.sparse_representation],
                self.accum_grads,
                job.loss,
                job.task_id)
        self.apply_gradients = self.optimizer.apply_gradients(zip(self.accum_grads, net_vars))
        self.reset_accum_grads = reset_accumulative_gradients_op(net_vars, self.accum_grads, 1)

        self.session.run(tf.global_variables_initializer())

        if self.config["save_model"]:
            for job in self.jobs:
                tf.add_to_collection("action", job.action)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()

    def build_networks(self):
        with tf.variable_scope("shared"):
            self.states = tf.placeholder(tf.float32, [None, self.nO], name="states")
            self.action_taken = tf.placeholder(tf.float32, name="action_taken")
            self.advantage = tf.placeholder(tf.float32, name="advantage")

            self.L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.config["n_hidden_units"],
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            self.knowledge_base = tf.Variable(tf.random_normal([self.config["n_hidden_units"], self.config["n_sparse_units"]]), name="knowledge_base")

            self.shared_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="shared")

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])

    def signal_handler(self, signal, frame):
        """When a (SIGINT) signal is received, request the threads (via the master) to stop after completing an iteration."""
        logging.info("SIGINT signal received: Requesting a stop...")
        self.stop_requested = True

    def learn(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

        if self.config["save_model"]:
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

    def make_thread(self, env, task_id):
        return AKTThread(self, env, task_id)
