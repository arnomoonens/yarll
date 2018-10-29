# -*- coding: utf8 -*-

import os
import logging
import signal
from threading import Thread
import numpy as np
import tensorflow as tf

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards, FastSaver
from yarll.misc.reporter import Reporter
from yarll.agents.knowledgetransfer import TaskPolicy

class AKTThread(Thread):
    """Asynchronous knowledge transfer learner thread. Used to learn using one specific variation of a task."""
    def __init__(self, master, env, task_id, n_iter, start_at_iter=0):
        super(AKTThread, self).__init__()
        self.master = master
        self.config = self.master.config
        self.task_id = task_id
        self.nA = env.action_space.n
        self.n_iter = n_iter
        self.start_at_iter = start_at_iter
        self.add_accum_grad = None  # To be filled in later

        self.build_networks()
        self.states = self.master.states
        self.session = self.master.session
        self.task_runner = EnvRunner(env, TaskPolicy(self.action, self), self.master.config)

        # Write the summary of each task in a different directory
        self.writer = tf.summary.FileWriter(os.path.join(self.master.monitor_path, "task" + str(self.task_id)), self.master.session.graph)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])

    def build_networks(self):
        with tf.variable_scope("task{}".format(self.task_id)):
            self.sparse_representation = tf.Variable(tf.truncated_normal([self.master.config["n_sparse_units"], self.nA], mean=0.0, stddev=0.02))
            self.probs = tf.nn.softmax(tf.matmul(self.master.L1, tf.matmul(self.master.knowledge_base, self.sparse_representation)))

            self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

            good_probabilities = tf.reduce_sum(tf.multiply(self.probs, tf.one_hot(tf.cast(self.master.action_taken, tf.int32), self.nA)), reduction_indices=[1])
            eligibility = tf.log(good_probabilities + 1e-10) * self.master.advantage
            self.loss = -tf.reduce_sum(eligibility)

    def run(self):
        """Run the appropriate learning algorithm."""
        if self.master.learning_method == "REINFORCE":
            self.learn_REINFORCE()
        else:
            self.learn_Karpathy()

    def learn_REINFORCE(self):
        """Learn using updates like in the REINFORCE algorithm."""
        reporter = Reporter()
        total_n_trajectories = 0
        iteration = self.start_at_iter
        while iteration < self.n_iter and not self.master.stop_requested:
            iteration += 1
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = self.task_runner.get_trajectories()
            total_n_trajectories += len(trajectories)
            all_state = np.concatenate([trajectory["state"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            rets = [discount_rewards(trajectory["reward"], self.config["gamma"]) for trajectory in trajectories]
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
            results = self.master.session.run([self.loss, self.apply_grad], feed_dict={
                self.master.states: all_state,
                self.master.action_taken: all_action,
                self.master.advantage: all_adv
            })
            print("Task:", self.task_id)
            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
            summary = self.master.session.run([self.master.summary_op], feed_dict={
                self.master.loss: results[0],
                self.master.reward: np.mean(episode_rewards),
                self.master.episode_length: np.mean(episode_lengths)
            })
            self.writer.add_summary(summary[0], iteration)
            self.writer.flush()

    def learn_Karpathy(self):
        """Learn using updates like in the Karpathy algorithm."""
        iteration = self.start_at_iter
        while iteration < self.n_iter and not self.master.stop_requested:  # Keep executing episodes until the master requests a stop (e.g. using SIGINT)
            iteration += 1
            trajectory = self.task_runner.get_trajectory()
            reward = sum(trajectory["reward"])
            action_taken = trajectory["action"]

            discounted_episode_rewards = discount_rewards(trajectory["reward"], self.config["gamma"])
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            std = np.std(discounted_episode_rewards)
            std = std if std > 0 else 1
            discounted_episode_rewards /= std
            feedback = discounted_episode_rewards

            results = self.master.session.run([self.loss, self.apply_grad], feed_dict={
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


class AsyncKnowledgeTransfer(Agent):
    """Asynchronous learner for variations of a task."""
    def __init__(self, envs, monitor_path, learning_method="REINFORCE", **usercfg):
        super(AsyncKnowledgeTransfer, self).__init__(**usercfg)
        self.envs = envs
        self.learning_method = learning_method
        self.monitor_path = monitor_path
        self.config.update(dict(
            timesteps_per_batch=10000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=200,
            switch_at_iter=None,  # None to deactivate, otherwhise an iteration at which to switch
            gamma=0.99,  # Discount past rewards by a percentage
            decay=0.9,  # Decay of RMSProp optimizer
            epsilon=1e-9,  # Epsilon of RMSProp optimizer
            learning_rate=0.005,
            n_hidden_units=10,
            repeat_n_actions=1,
            n_task_variations=3,
            n_sparse_units=10,
            feature_extraction=False
        ))
        self.config.update(usercfg)

        self.stop_requested = False

        self.session = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True))

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)
        self.build_networks()

        self.loss = tf.placeholder("float", name="loss")
        summary_loss = tf.summary.scalar("Loss", self.loss)
        self.reward = tf.placeholder("float", name="reward")
        summary_rewards = tf.summary.scalar("Reward", self.reward)
        self.episode_length = tf.placeholder("float", name="episode_length")
        summary_episode_lengths = tf.summary.scalar("Episode_length", self.episode_length)
        self.summary_op = tf.summary.merge([summary_loss, summary_rewards, summary_episode_lengths])

        self.jobs = []
        for i, env in enumerate(self.envs):
            self.jobs.append(
                self.make_thread(
                    env,
                    i,
                    self.config["switch_at_iter"] if self.config["switch_at_iter"] is not None and i != len(self.envs) - 1 else self.config["n_iter"],
                    start_at_iter=(0 if self.config["switch_at_iter"] is None or i != len(self.envs) - 1 else self.config["switch_at_iter"])))

        for i, job in enumerate(self.jobs):
            only_sparse = (self.config["switch_at_iter"] is not None and i == len(self.jobs) - 1)
            grads = tf.gradients(job.loss, (self.shared_vars if not(only_sparse) else []) + [job.sparse_representation])
            job.apply_grad = job.optimizer.apply_gradients(
                zip(
                    grads,
                    (self.shared_vars if not(only_sparse) else []) + [job.sparse_representation]
                ),
                global_step=self.global_step
            )

        self.session.run(tf.global_variables_initializer())

        if self.config["save_model"]:
            for job in self.jobs:
                tf.add_to_collection("action", job.action)
            tf.add_to_collection("states", self.states)
            self.saver = FastSaver()

    def build_networks(self):
        with tf.variable_scope("shared"):
            self.states = tf.placeholder(tf.float32, [None] + list(self.envs[0].observation_space.shape), name="states")
            self.action_taken = tf.placeholder(tf.float32, name="action_taken")
            self.advantage = tf.placeholder(tf.float32, name="advantage")

            if self.config["feature_extraction"]:
                self.L1 = tf.contrib.layers.fully_connected(
                    inputs=self.states,
                    num_outputs=self.config["n_hidden_units"],
                    activation_fn=tf.tanh,
                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                    biases_initializer=tf.zeros_initializer(),
                    scope="L1")
            else:
                self.L1 = self.states
            self.knowledge_base = tf.Variable(tf.truncated_normal([self.L1.get_shape()[-1].value, self.config["n_sparse_units"]], mean=0.0, stddev=0.02), name="knowledge_base")

            self.shared_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def signal_handler(self, signal, frame):
        """When a (SIGINT) signal is received, request the threads (via the master) to stop after completing an iteration."""
        logging.info("SIGINT signal received: Requesting a stop...")
        self.stop_requested = True

    def learn(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        if self.config["switch_at_iter"] is None:
            idx = None
        else:
            idx = -1
        for job in self.jobs[:idx]:
            job.start()
        for job in self.jobs[:idx]:
            job.join()
        try:
            self.jobs[idx].start()
            self.jobs[idx].join()
        except TypeError:  # idx is None
            pass

        if self.config["save_model"]:
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

    def make_thread(self, env, task_id, n_iter, start_at_iter=0):
        return AKTThread(self, env, task_id, n_iter, start_at_iter=start_at_iter)
