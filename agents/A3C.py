#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf
import logging
from threading import Thread
import multiprocessing
import signal

from gym import wrappers

from Environment.registration import make_environment
from agents.Agent import Agent
from misc.utils import discount_rewards
from misc.gradient_ops import create_accumulative_gradients_op, add_accumulative_gradients_op, reset_accumulative_gradients_op, sync_gradients_op

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

# Based on:
# - Pseudo code from Asynchronous Methods for Deep Reinforcement Learning
# - Tensorflow code from https://github.com/yao62995/A3C/blob/master/A3C_atari.py

class ActorNetworkDiscrete(object):
    """Neural network for the Actor of an Actor-Critic algorithm using a discrete action space"""
    def __init__(self, state_shape, n_actions, n_hidden, scope, summary=True):
        super(ActorNetworkDiscrete, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.scope = scope

        with tf.variable_scope("{}_actor".format(scope)):
            self.states = tf.placeholder("float", [None, self.state_shape], name="states")
            self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
            self.critic_feedback = tf.placeholder(tf.float32, name="critic_feedback")
            self.critic_rewards = tf.placeholder(tf.float32, name="critic_rewards")

            L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.n_hidden,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            self.probs = tf.contrib.layers.fully_connected(
                inputs=L1,
                num_outputs=n_actions,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="probs")

            self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

            good_probabilities = tf.reduce_sum(tf.multiply(self.probs, self.actions_taken), reduction_indices=[1])
            # Replace probabilities that are zero with a small value and multiply by advantage:
            eligibility = tf.log(tf.where(tf.equal(good_probabilities, tf.fill(tf.shape(good_probabilities), 0.0)), tf.fill(tf.shape(good_probabilities), 1e-30), good_probabilities)) \
                * (self.critic_rewards - self.critic_feedback)
            self.loss = tf.negative(tf.reduce_mean(eligibility), name="loss")
            self.summary_loss = self.loss  # Loss to show as a summary
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class ActorNetworkContinuous(object):
    """Neural network for an Actor of an Actor-Critic algorithm using a continuous action space."""
    def __init__(self, action_space, state_shape, n_hidden, scope, summary=True):
        super(ActorNetworkContinuous, self).__init__()
        self.state_shape = state_shape
        self.n_hidden = n_hidden
        self.scope = scope

        with tf.variable_scope("{}_actor".format(scope)):
            self.states = tf.placeholder("float", [None, self.state_shape], name="states")
            self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
            self.critic_feedback = tf.placeholder(tf.float32, name="critic_feedback")  # Advantage
            self.critic_rewards = tf.placeholder(tf.float32, name="critic_rewards")

            L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.n_hidden,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="mu_L1")

            mu = tf.contrib.layers.fully_connected(
                inputs=L1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="mu")
            mu = tf.squeeze(mu, name="mu")

            sigma_L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.n_hidden,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="sigma_L1")

            sigma = tf.contrib.layers.fully_connected(
                inputs=sigma_L1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="sigma")
            sigma = tf.squeeze(sigma)
            sigma = tf.nn.softplus(sigma) + 1e-5

            self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
            self.action = self.normal_dist.sample(1)
            self.action = tf.clip_by_value(self.action, action_space.low[0], action_space.high[0], name="action")
            self.loss = -tf.reduce_mean(self.normal_dist.log_prob(self.actions_taken) * self.critic_feedback)
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()
            self.summary_loss = -tf.reduce_mean(self.loss)  # Loss to show as a summary
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class CriticNetwork(object):
    """Neural network for the Critic of an Actor-Critic algorithm"""
    def __init__(self, state_shape, n_hidden, scope, summary=True):
        super(CriticNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_hidden = n_hidden
        self.scope = scope

        with tf.variable_scope("{}_critic".format(scope)):
            self.states = tf.placeholder("float", [None, self.state_shape], name="states")
            self.target = tf.placeholder("float", name="critic_target")

            L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.n_hidden,
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            self.value = tf.contrib.layers.fully_connected(
                inputs=L1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="value")

            self.loss = tf.reduce_mean(tf.square(self.target - self.value))
            self.summary_loss = self.loss
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class A3CThread(Thread):
    """Single A3C learner thread."""
    def __init__(self, master, thread_id, clip_gradients=True):
        super(A3CThread, self).__init__(name=thread_id)
        self.thread_id = thread_id
        self.env = make_environment(master.env_name)
        self.master = master
        if thread_id == 0 and self.master.monitor:
            self.env = wrappers.Monitor(self.env, master.monitor_path, force=True, video_callable=(None if self.master.video else False))

        # Build actor and critic networks
        self.build_networks()

        # Write the summary of each thread in a different directory
        self.writer = tf.summary.FileWriter(os.path.join(self.master.monitor_path, "thread" + str(self.thread_id)), self.master.session.graph)

        self.actor_sync_net = sync_gradients_op(master.shared_actor_net, self.actor_net.vars, self.thread_id)
        self.actor_create_ag = create_accumulative_gradients_op(self.actor_net.vars, self.thread_id)
        self.actor_add_ag = add_accumulative_gradients_op(self.actor_net.vars, self.actor_create_ag, self.actor_net.loss, self.thread_id)
        self.actor_reset_ag = reset_accumulative_gradients_op(self.actor_net.vars, self.actor_create_ag, self.thread_id)

        self.critic_sync_net = sync_gradients_op(master.shared_critic_net, self.critic_net.vars, self.thread_id)
        self.critic_create_ag = create_accumulative_gradients_op(self.critic_net.vars, self.thread_id)
        self.critic_add_ag = add_accumulative_gradients_op(self.critic_net.vars, self.critic_create_ag, self.critic_net.loss, self.thread_id)
        self.critic_reset_ag = reset_accumulative_gradients_op(self.critic_net.vars, self.critic_create_ag, self.thread_id)

        if clip_gradients:
            # Clipped gradients
            gradient_clip_value = self.master.config["gradient_clip_value"]
            clip_actor_gradients = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in self.actor_create_ag]
            self.apply_actor_gradients = master.shared_actor_optimizer.apply_gradients(
                zip(clip_actor_gradients, master.shared_actor_net.vars), global_step=master.global_step)
            clip_critic_gradients = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in self.critic_create_ag]
            self.apply_critic_gradients = master.shared_critic_optimizer.apply_gradients(
                zip(clip_critic_gradients, master.shared_critic_net.vars), global_step=master.global_step)
        else:
            # Non-clipped gradients
            self.apply_actor_gradients = master.shared_actor_optimizer.apply_gradients(
                zip(self.actor_create_ag, master.shared_actor_net.vars), global_step=master.global_step)
            self.apply_critic_gradients = master.shared_critic_optimizer.apply_gradients(
                zip(self.critic_create_ag, master.shared_critic_net.vars), global_step=master.global_step)

    def transform_actions(self, actions):
        return actions

    def get_critic_value(self, states):
        return self.master.session.run([self.critic_net.value], feed_dict={self.critic_net.states: states})[0].flatten()

    def get_trajectory(self, episode_max_length, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        """
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        for i in range(episode_max_length):
            action = self.choose_action(state)  # Predict the next action (using a neural network) depending on the current state
            states.append(state.flatten())
            state, reward, done, _ = self.env.step(action)
            reward = np.clip(reward, -1, 1)  # Clip reward
            actions.append(action)
            rewards.append(reward)
            if done:
                break
            if render:
                self.env.render()
        return {
            "reward": np.array(rewards),
            "state": np.array(states),
            "action": np.array(actions),
            "done": done,  # Say if tajectory ended because a terminal state was reached
            "steps": i + 1
        }

    def choose_action(self, state):
        """Choose an action."""
        return self.master.session.run([self.actor_net.action], feed_dict={self.actor_net.states: [state]})[0]

    def run(self):
        # Assume global shared parameter vectors θ and θv and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'v
        sess = self.master.session
        t = 1  # thread step counter
        while self.master.T < self.master.config["T_max"] and not self.master.stop_requested:
            # Reset gradients: dθ = 0 and dθv = 0
            sess.run([self.actor_reset_ag, self.critic_reset_ag])
            # Synchronize thread-specific parameters θ' = θ and θ'v = θv
            sess.run([self.actor_sync_net, self.critic_sync_net])
            trajectory = self.get_trajectory(self.master.config["episode_max_length"])
            reward = sum(trajectory["reward"])
            trajectory["reward"][-1] = 0 if trajectory["done"] else self.get_critic_value(trajectory["state"][None, -1])[0]
            returns = discount_rewards(trajectory["reward"], self.master.config["gamma"])
            fetches = [self.actor_net.summary_loss, self.critic_net.summary_loss, self.actor_add_ag, self.critic_add_ag, self.master.global_step]  # What does the master global step thing do?
            ac_net = self.actor_net
            cr_net = self.critic_net
            qw_new = self.master.session.run([cr_net.value], feed_dict={cr_net.states: trajectory["state"]})[0].flatten()
            all_action = self.transform_actions(trajectory["action"])  # Transform actions back to the output shape of the actor network (e.g. one-hot for discrete action space)
            results = sess.run(fetches, feed_dict={
                ac_net.states: trajectory["state"],
                cr_net.states: trajectory["state"],
                ac_net.actions_taken: all_action,
                ac_net.critic_feedback: qw_new,
                ac_net.critic_rewards: returns,
                cr_net.target: returns.reshape(-1, 1)
            })
            summary = sess.run([self.master.summary_op], feed_dict={
                               self.master.actor_loss: results[0],
                               self.master.critic_loss: results[1],
                               self.master.reward: reward,
                               self.master.episode_length: trajectory["steps"]
                               })
            self.writer.add_summary(summary[0], t)
            self.writer.flush()
            sess.run([self.apply_actor_gradients, self.apply_critic_gradients])
            t += 1
            self.master.T += trajectory["steps"]

class A3CThreadDiscrete(A3CThread):
    """A3CThread for a discrete action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadDiscrete, self).__init__(master, thread_id)

    def build_networks(self):
        self.actor_net = ActorNetworkDiscrete(self.env.observation_space.shape[0], self.env.action_space.n, self.master.config["actor_n_hidden"], scope="local_actor_net")
        self.critic_net = CriticNetwork(self.env.observation_space.shape[0], self.master.config["critic_n_hidden"], scope="local_critic_net")

    def transform_actions(self, actions):
        possible_actions = np.arange(self.env.action_space.n)
        return (possible_actions == actions[:, None]).astype(np.float32)

class A3CThreadContinuous(A3CThread):
    """A3CThread for a continuous action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadContinuous, self).__init__(master, thread_id)

    def build_networks(self):
        self.actor_net = ActorNetworkContinuous(self.env.action_space, self.env.observation_space.shape[0], self.master.config["actor_n_hidden"], scope="local_actor_net")
        self.critic_net = CriticNetwork(self.env.observation_space.shape[0], self.master.config["critic_n_hidden"], scope="local_critic_net")

class A3C(Agent):
    """Asynchronous Advantage Actor Critic learner."""
    def __init__(self, env, monitor, monitor_path, video=True, **usercfg):
        super(A3C, self).__init__(env)
        self.env = env
        self.shared_counter = 0
        self.T = 0
        self.env_name = env.spec.id
        self.monitor = monitor
        self.monitor_path = monitor_path
        self.video = video

        self.config.update(dict(
            gamma=0.99,  # Discount past rewards by a percentage
            decay=0.9,  # Decay of RMSProp optimizer
            epsilon=1e-9,  # Epsilon of RMSProp optimizer
            actor_learning_rate=0.01,
            critic_learning_rate=0.05,
            actor_n_hidden=20,
            critic_n_hidden=20,
            gradient_clip_value=40,
            n_threads=multiprocessing.cpu_count(),  # Use as much threads as there are CPU threads on the current system
            T_max=5e5,
            repeat_n_actions=1,
            save_model=False
        ))
        self.config.update(usercfg)
        self.stop_requested = False

        self.build_networks()
        if self.config["save_model"]:
            tf.add_to_collection("action", self.shared_actor_net.action)
            tf.add_to_collection("states", self.shared_actor_net.states)
            self.saver = tf.train.Saver()

        self.shared_actor_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["actor_learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])
        self.shared_critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["critic_learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])

        self.global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

        self.session = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True))

        self.critic_loss = tf.placeholder("float", name="critic_loss")
        critic_loss_summary = tf.summary.scalar("Critic_loss", self.critic_loss)
        self.actor_loss = tf.placeholder("float", name="actor_loss")
        actor_loss_summary = tf.summary.scalar("Actor_loss", self.actor_loss)
        self.reward = tf.placeholder("float", name="reward")
        reward_summary = tf.summary.scalar("Reward", self.reward)
        self.episode_length = tf.placeholder("float", name="episode_length")
        episode_length_summary = tf.summary.scalar("Episode_length", self.episode_length)
        self.summary_op = tf.summary.merge([actor_loss_summary, critic_loss_summary, reward_summary, episode_length_summary])

        self.jobs = []
        for thread_id in range(self.config["n_threads"]):
            job = self.make_thread(thread_id)
            self.jobs.append(job)

        self.session.run(tf.global_variables_initializer())

        self.global_step_val = 0

    def signal_handler(self, signal, frame):
        """When a (SIGINT) signal is received, request the threads (via the master) to stop after completing an iteration."""
        logging.info("SIGINT signal received: Requesting a stop...")
        self.stop_requested = True

    def learn(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        self.train_step = 0
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
        if self.config["save_model"]:
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

class A3CDiscrete(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        super(A3CDiscrete, self).__init__(env, monitor, monitor_path, **usercfg)

    def build_networks(self):
        self.shared_actor_net = ActorNetworkDiscrete(self.env.observation_space.shape[0], self.env.action_space.n, self.config["actor_n_hidden"], scope="global_actor_net", summary=False)
        self.shared_critic_net = CriticNetwork(self.env.observation_space.shape[0], self.config["critic_n_hidden"], scope="global_critic_net", summary=False)

    def make_thread(self, thread_id):
        return A3CThreadDiscrete(self, thread_id)

class A3CContinuous(A3C):
    """A3C for a continuous action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        super(A3CContinuous, self).__init__(env, monitor, monitor_path, **usercfg)

    def build_networks(self):
        self.shared_actor_net = ActorNetworkContinuous(self.env.action_space, self.env.observation_space.shape[0], self.config["actor_n_hidden"], scope="global_actor_net", summary=False)
        self.shared_critic_net = CriticNetwork(self.env.observation_space.shape[0], self.config["critic_n_hidden"], scope="global_critic_net", summary=False)

    def make_thread(self, thread_id):
        return A3CThreadContinuous(self, thread_id)
