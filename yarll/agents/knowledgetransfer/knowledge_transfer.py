# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards, FastSaver
from yarll.misc.reporter import Reporter
from yarll.misc.network_ops import create_accumulative_gradients_op, add_accumulative_gradients_op, reset_accumulative_gradients_op

class TaskPolicy(object):
    """Policy for a specific class."""
    def __init__(self, action, master):
        super(TaskPolicy, self).__init__()
        self.action = action
        self.master = master

    def choose_action(self, state):
        """Choose an action."""
        return self.master.session.run([self.action], feed_dict={self.master.states: [state]})[0]

    def new_trajectory(self):
        pass

class KnowledgeTransfer(Agent):
    """Learner for variations of a task."""
    def __init__(self, envs, monitor_path, **usercfg):
        super(KnowledgeTransfer, self).__init__(**usercfg)
        self.envs = envs
        self.n_tasks = len(envs)
        self.monitor_path = monitor_path
        self.nA = envs[0].action_space.n
        self.config.update(dict(
            timesteps_per_batch=10000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=100,
            switch_at_iter=None,
            gamma=0.99,  # Discount past rewards by a percentage
            decay=0.9,  # Decay of RMSProp optimizer
            epsilon=1e-9,  # Epsilon of RMSProp optimizer
            learning_rate=0.005,
            n_hidden_units=10,
            repeat_n_actions=1,
            n_sparse_units=10,
            feature_extraction=False
        ))
        self.config.update(usercfg)

        self.build_networks()
        self.task_runners = [EnvRunner(envs[i], TaskPolicy(action, self), self.config) for i, action in enumerate(self.action_tensors)]
        if self.config["save_model"]:
            for action_tensor in self.action_tensors:
                tf.add_to_collection("action", action_tensor)
            tf.add_to_collection("states", self.states)
            self.saver = FastSaver()

    def build_networks(self):
        self.session = tf.Session()

        with tf.variable_scope("shared"):
            self.states = tf.placeholder(tf.float32, [None] + list(self.envs[0].observation_space.shape), name="states")
            self.action_taken = tf.placeholder(tf.float32, name="action_taken")
            self.advantage = tf.placeholder(tf.float32, name="advantage")

            L1 = None
            if self.config["feature_extraction"]:
                L1 = tf.contrib.layers.fully_connected(
                    inputs=self.states,
                    num_outputs=self.config["n_hidden_units"],
                    activation_fn=tf.tanh,
                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                    biases_initializer=tf.zeros_initializer(),
                    scope="L1")
            else:
                L1 = self.states

            knowledge_base = tf.Variable(tf.truncated_normal([L1.get_shape()[-1].value, self.config["n_sparse_units"]], mean=0.0, stddev=0.02), name="knowledge_base")

            self.shared_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="shared")

        # Every task has its own (sparse) representation
        sparse_representations = [
            tf.Variable(tf.truncated_normal([self.config["n_sparse_units"], self.nA], mean=0.0, stddev=0.02), name="sparse%d" % i)
            for i in range(self.n_tasks)
        ]

        self.probs_tensors = [tf.nn.softmax(tf.matmul(L1, tf.matmul(knowledge_base, s))) for s in sparse_representations]
        self.action_tensors = [tf.squeeze(tf.multinomial(tf.log(probs), 1)) for probs in self.probs_tensors]

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.config["learning_rate"],
            decay=self.config["decay"],
            epsilon=self.config["epsilon"]
        )
        net_vars = self.shared_vars + sparse_representations
        self.accum_grads = create_accumulative_gradients_op(net_vars, 0)

        self.loss = tf.placeholder("float", name="loss")
        summary_loss = tf.summary.scalar("Loss", self.loss)
        self.rewards = tf.placeholder("float", name="Rewards")
        summary_rewards = tf.summary.scalar("Reward", self.rewards)
        self.episode_lengths = tf.placeholder("float", name="Episode_lengths")
        summary_episode_lengths = tf.summary.scalar("Length", self.episode_lengths)
        self.summary_op = tf.summary.merge([summary_loss, summary_rewards, summary_episode_lengths])

        self.writers = []
        self.losses = []

        regularizer = tf.contrib.layers.l1_regularizer(.05)
        for i, probabilities in enumerate(self.probs_tensors):
            good_probabilities = tf.reduce_sum(tf.multiply(probabilities, tf.one_hot(tf.cast(self.action_taken, tf.int32), self.nA)),
                                               reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * self.advantage
            loss = -tf.reduce_sum(eligibility) + regularizer(sparse_representations[i])
            self.losses.append(loss)
            writer = tf.summary.FileWriter(os.path.join(self.monitor_path, "task" + str(i)), self.session.graph)
            self.writers.append(writer)

        # An add op for every task & its loss
        self.add_accum_grads = []
        for i, loss in enumerate(self.losses):
            # Use all variables if the switch tasks experiment is disactivated or it's not the last task
            all_vars = self.config["switch_at_iter"] is None or i != len(self.losses) - 1
            self.add_accum_grads.append(add_accumulative_gradients_op(
                (self.shared_vars if all_vars else []) + [sparse_representations[i]],
                ([self.accum_grads[0]] if all_vars else []) + [self.accum_grads[i + 1]],
                loss,
                i
            ))

        self.apply_gradients = self.optimizer.apply_gradients(
            zip(self.accum_grads, net_vars))
        self.reset_accum_grads = reset_accumulative_gradients_op(net_vars, self.accum_grads, 0)

        self.init_op = tf.global_variables_initializer()

    def _initialize(self):
        self.session.run(self.init_op)

    def learn(self):
        """Run learning algorithm"""
        self._initialize()
        reporter = Reporter()
        config = self.config
        total_n_trajectories = np.zeros(len(self.envs))
        for iteration in range(config["n_iter"]):
            self.session.run([self.reset_accum_grads])
            for i, task_runner in enumerate(self.task_runners):
                if self.config["switch_at_iter"] is not None:
                    if iteration >= self.config["switch_at_iter"] and i != (len(self.task_runners) - 1):
                        continue
                    elif iteration < self.config["switch_at_iter"] and i == len(self.task_runners) - 1:
                        continue
                # Collect trajectories until we get timesteps_per_batch total timesteps
                trajectories = task_runner.get_trajectories()
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
                results = self.session.run([self.losses[i], self.add_accum_grads[i], self.accum_grads], feed_dict={
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
            if not os.path.exists(self.monitor_path):
                os.makedirs(self.monitor_path)
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))
