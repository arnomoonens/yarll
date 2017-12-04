# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf
import logging
import threading
import multiprocessing
import signal
import queue

from gym import wrappers

from environment.registration import make_environment
from agents.agent import Agent
from misc.utils import discount_rewards
from misc.network_ops import sync_networks_op, conv2d, mu_sigma_layer, flatten, normalized_columns_initializer, linear

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

# Based on:
# - Pseudo code from Asynchronous Methods for Deep Reinforcement Learning
# - Tensorflow code from https://github.com/yao62995/A3C/blob/master/A3C_atari.py

class ActorNetworkDiscrete(object):
    """Neural network for the Actor of an Actor-Critic algorithm using a discrete action space"""
    def __init__(self, state_shape, n_actions, n_hidden, summary=True):
        super(ActorNetworkDiscrete, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        with tf.variable_scope("actor"):
            self.states = tf.placeholder(tf.float32, [None] + state_shape, name="states")
            self.adv = tf.placeholder(tf.float32, name="advantage")
            self.actions_taken = tf.placeholder(tf.float32, [None, n_actions], name="actions_taken")

            L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.n_hidden,
                activation_fn=tf.tanh,
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            # Fully connected for Actor & Critic
            self.logits = linear(L1, n_actions, "actionlogits", normalized_columns_initializer(0.01))

            self.probs = tf.nn.softmax(self.logits)

            self.action = tf.squeeze(tf.multinomial(self.logits - tf.reduce_max(self.logits, [1], keep_dims=True), 1), [1], name="action")
            self.action = tf.one_hot(self.action, n_actions)[0, :]

            log_probs = tf.nn.log_softmax(self.logits)
            self.loss = - tf.reduce_sum(tf.reduce_sum(log_probs * self.actions_taken, [1]) * self.adv)

            self.summary_loss = self.loss  # Loss to show as a summary
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class ActorCriticNetworkDiscreteCNN(object):
    """docstring for ActorNetworkDiscreteCNNRNN"""
    def __init__(self, state_shape, n_actions, n_hidden, summary=True):
        super(ActorCriticNetworkDiscreteCNN, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.summary = summary

        self.states = tf.placeholder(tf.float32, [None] + state_shape, name="states")
        self.adv = tf.placeholder(tf.float32, name="advantage")
        self.actions_taken = tf.placeholder(tf.float32, [None, n_actions], name="actions_taken")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        x = self.states
        # Convolution layers
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        # Flatten
        shape = x.get_shape().as_list()
        reshape = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Fully connected for Actor & Critic
        self.logits = linear(reshape, n_actions, "actionlogits", normalized_columns_initializer(0.01))
        self.value = tf.reshape(linear(reshape, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.probs = tf.nn.softmax(self.logits)

        self.action = tf.squeeze(tf.multinomial(self.logits - tf.reduce_max(self.logits, [1], keep_dims=True), 1), [1], name="action")
        self.action = tf.one_hot(self.action, n_actions)[0, :]

        log_probs = tf.nn.log_softmax(self.logits)
        self.actor_loss = - tf.reduce_sum(tf.reduce_sum(log_probs * self.actions_taken, [1]) * self.adv)

        self.critic_loss = 0.5 * tf.reduce_sum(tf.square(self.value - self.r))

        entropy = - tf.reduce_sum(self.probs * log_probs)

        self.loss = self.actor_loss + 0.5 * self.critic_loss - entropy * 0.01
        self.summary_loss = self.loss

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class ActorCriticNetworkDiscreteCNNRNN(object):
    """docstring for ActorNetworkDiscreteCNNRNN"""
    def __init__(self, state_shape, n_actions, n_hidden, summary=True):
        super(ActorCriticNetworkDiscreteCNNRNN, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.summary = summary

        self.states = tf.placeholder(tf.float32, [None] + state_shape, name="states")
        self.adv = tf.placeholder(tf.float32, name="advantage")
        self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        x = self.states
        # Convolution layers
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        # Flatten
        reshape = tf.expand_dims(flatten(x), [0])

        lstm_size = 256
        self.enc_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        lstm_state_size = self.enc_cell.state_size
        c_init = np.zeros((1, lstm_state_size.c), np.float32)
        h_init = np.zeros((1, lstm_state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        self.rnn_state_in = self.enc_cell.zero_state(1, tf.float32)
        tf.add_to_collection("rnn_state_in_c", self.rnn_state_in.c)
        tf.add_to_collection("rnn_state_in_h", self.rnn_state_in.h)
        L3, self.rnn_state_out = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                   inputs=reshape,
                                                   initial_state=self.rnn_state_in,
                                                   dtype=tf.float32)
        tf.add_to_collection("rnn_state_out_c", self.rnn_state_out.c)
        tf.add_to_collection("rnn_state_out_h", self.rnn_state_out.h)
        L3 = tf.reshape(L3, [-1, lstm_size])

        # Fully connected for Actor

        self.logits = linear(L3, n_actions, "actionlogits", normalized_columns_initializer(0.01))
        self.value = tf.reshape(linear(L3, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.probs = tf.nn.softmax(self.logits)

        self.action = tf.squeeze(tf.multinomial(self.logits - tf.reduce_max(self.logits, [1], keep_dims=True), 1), [1], name="action")
        self.action = tf.one_hot(self.action, n_actions)[0, :]

        log_probs = tf.nn.log_softmax(self.logits)
        self.actor_loss = - tf.reduce_sum(tf.reduce_sum(log_probs * self.actions_taken, [1]) * self.adv)

        self.critic_loss = 0.5 * tf.reduce_sum(tf.square(self.value - self.r))

        self.entropy = - tf.reduce_sum(self.probs * log_probs)

        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.entropy * 0.01

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class ActorNetworkContinuous(object):
    """Neural network for an Actor of an Actor-Critic algorithm using a continuous action space."""
    def __init__(self, action_space, state_shape, n_hidden, summary=True):
        super(ActorNetworkContinuous, self).__init__()
        self.state_shape = state_shape
        self.n_hidden = n_hidden

        with tf.variable_scope("actor"):
            self.states = tf.placeholder("float", [None] + self.state_shape, name="states")
            self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
            self.adv = tf.placeholder(tf.float32, name="advantage")

            L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.n_hidden,
                activation_fn=tf.tanh,
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                biases_initializer=tf.zeros_initializer(),
                scope="mu_L1")

            mu, sigma = mu_sigma_layer(L1, 1)

            self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
            self.action = self.normal_dist.sample(1)
            self.action = tf.clip_by_value(self.action, action_space.low[0], action_space.high[0], name="action")
            self.loss = -tf.reduce_sum(self.normal_dist.log_prob(self.actions_taken) * self.adv)
            # Add cross entropy cost to encourage exploration
            self.loss -= 0.01 * tf.reduce_mean(self.normal_dist.entropy())
            self.summary_loss = -tf.reduce_mean(self.loss)  # Loss to show as a summary
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class CriticNetwork(object):
    """Neural network for the Critic of an Actor-Critic algorithm."""
    def __init__(self, state_shape, n_hidden, summary=True):
        super(CriticNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_hidden = n_hidden

        with tf.variable_scope("critic"):
            self.states = tf.placeholder("float", [None] + self.state_shape, name="states")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            L1 = tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.n_hidden,
                activation_fn=tf.tanh,
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            self.value = tf.reshape(linear(L1, 1, "value", normalized_columns_initializer(1.0)), [-1])

            self.loss = tf.reduce_sum(tf.square(self.value - self.r))
            self.summary_loss = self.loss
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


class Trajectory(object):
    """Experience gathered from an environment."""
    def __init__(self):
        super(Trajectory, self).__init__()
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.features = []
        self.terminal = False
        self.steps = 0

    def add(self, state, action, reward, value, features, terminal):
        """Add a single transition to the trajectory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.features.append(features)
        self.terminal = terminal
        self.steps += 1

    def extend(self, other):
        """
        Extend a trajectory with another one
        given that the current one hasn't ended yet.
        """
        assert not self.terminal, "Can't extend a terminal trajectory"
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.features.extend(other.features)
        self.terminal = other.terminal
        self.steps += other.steps

def env_runner(env, policy, n_steps, render=False, summary_writer=None):
    """
    Run agent-environment loop for maximally n_steps.
    Yield dictionary of results.
    """
    episode_steps = 0
    episode_reward = 0
    state = env.reset()
    features = policy.initial_features
    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    while True:
        trajectory = Trajectory()

        for _ in range(n_steps):
            fetched = policy.choose_action(state, features)  # Predict the next action (using a neural network) depending on the current state
            action = fetched[0]
            value = fetched[1]
            new_features = fetched[2:]
            new_state, reward, terminal, _ = env.step(policy.get_env_action(action))
            episode_steps += 1
            episode_reward += reward
            trajectory.add(state, action, reward, value, new_features, terminal)
            state = new_state
            features = new_features
            if terminal or episode_steps >= timestep_limit:
                policy.rnn_state = policy.initial_features
                if episode_steps >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    state = env.reset()
                if summary_writer is not None:
                    summary = tf.Summary()
                    summary.value.add(tag="global/Episode_length", simple_value=float(episode_steps))
                    summary.value.add(tag="global/Reward", simple_value=float(episode_reward))
                    summary_writer.add_summary(summary, policy.master.global_step.eval())
                    summary_writer.flush()
                episode_steps = 0
                episode_reward = 0
                break
            if render:
                env.render()
        yield trajectory

class RunnerThread(threading.Thread):
    """
    Thread that collects trajectories from the environment
    and puts them on a queue.
    """
    def __init__(self, env, policy, n_local_steps, render=False):
        super(RunnerThread, self).__init__()
        self.env = env
        self.policy = policy
        self.n_local_steps = n_local_steps
        self.render = render
        self.stop_requested = False

        self.queue = queue.Queue(5)

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        trajectory_provider = env_runner(self.env, self.policy, self.n_local_steps, self.render, self.summary_writer)
        while not self.stop_requested:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(trajectory_provider), timeout=600.0)

class A3CThread(threading.Thread):
    """Single A3C learner thread."""
    def __init__(self, master, thread_id, clip_gradients=True):
        super(A3CThread, self).__init__(name=thread_id)
        self.thread_id = thread_id
        self.clip_gradients = clip_gradients
        self.env = make_environment(master.env_name)
        self.master = master
        self.config = master.config
        if thread_id == 0 and self.master.monitor:
            self.env = wrappers.Monitor(self.env, master.monitor_path, force=True, video_callable=(None if self.master.video else False))

        # Only used (and overwritten) by agents that use an RNN
        self.initial_features = None

        # Build actor and critic networks
        with tf.variable_scope("t{}_net".format(self.thread_id)):
            self.action, self.value, self.actor_states, self.critic_states, self.actions_taken, self.losses, self.adv, self.r, self.n_steps = self.build_networks()
            self.sync_net = self.create_sync_net_op()
            inc_step = self.master.global_step.assign_add(self.n_steps)
            self.train_op = tf.group(self.make_trainer(), inc_step)
        # Write the summary of each thread in a different directory
        self.writer = tf.summary.FileWriter(os.path.join(self.master.monitor_path, "thread" + str(self.thread_id)), self.master.session.graph)

        self.runner = RunnerThread(self.env, self, 20, thread_id == 0 and self.master.video)

    def get_critic_value(self, states, *rest):
        return self.master.session.run(self.value, feed_dict={self.critic_states: states})[0]

    def get_env_action(self, action):
        return np.argmax(action)

    def choose_action(self, state, *rest):
        """Choose an action."""
        feed_dict = {
            self.actor_states: [state],
            self.critic_states: [state]
        }
        action, value = self.master.session.run([self.action, self.value], feed_dict=feed_dict)
        return action, value, []

    def pull_batch_from_queue(self):
        """
        Take a trajectory from the queue.
        Also immediately try to extend it if the episode
        wasn't over and more transitions are available
        """
        trajectory = self.runner.queue.get(timeout=600.0)
        while not trajectory.terminal:
            try:
                trajectory.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return trajectory

    def run(self):
        # Assume global shared parameter vectors θ and θv and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'v
        sess = self.master.session
        self.runner.start_runner(sess, self.writer)
        t = 1  # thread step counter
        while self.master.T < self.config["T_max"] and not self.master.stop_requested:
            # Synchronize thread-specific parameters θ' = θ and θ'v = θv
            sess.run(self.sync_net)
            trajectory = self.pull_batch_from_queue()
            v = 0 if trajectory.terminal else self.get_critic_value(np.asarray(trajectory.states)[None, -1], trajectory.features[-1][0])
            rewards_plus_v = np.asarray(trajectory.rewards + [v])
            vpred_t = np.asarray(trajectory.values + [v])
            delta_t = trajectory.rewards + self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
            batch_r = discount_rewards(rewards_plus_v, self.config["gamma"])[:-1]
            batch_adv = discount_rewards(delta_t, self.config["gamma"])
            fetches = self.losses + [self.train_op, self.master.global_step]
            states = np.asarray(trajectory.states)
            feed_dict = {
                self.actor_states: states,
                self.critic_states: states,
                self.actions_taken: np.asarray(trajectory.actions),
                self.adv: batch_adv,
                self.r: np.asarray(batch_r)
            }
            feature = trajectory.features[0][0]
            if feature != []:
                feed_dict[self.ac_net.rnn_state_in] = feature
            results = sess.run(fetches, feed_dict)
            n_states = states.shape[0]
            feed_dict = dict(zip(self.master.losses, map(lambda x: x / n_states, results)))
            summary = sess.run([self.master.summary_op], feed_dict)
            self.writer.add_summary(summary[0], results[-1])
            self.writer.flush()
            t += 1
            self.master.T += trajectory.steps
        self.runner.stop_requested = True

class A3CThreadDiscrete(A3CThread):
    """A3CThread for a discrete action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadDiscrete, self).__init__(master, thread_id)

    def build_networks(self):
        self.actor_net = actor_net = ActorNetworkDiscrete(list(self.env.observation_space.shape), self.env.action_space.n, self.config["actor_n_hidden"])
        self.critic_net = critic_net = CriticNetwork(list(self.env.observation_space.shape), self.config["critic_n_hidden"])

        actor_states = actor_net.states
        n_steps = tf.shape(actor_states)[0]
        adv = self.actor_net.adv
        r = self.critic_net.r
        self.loss_fetches = [self.actor_net.summary_loss, self.critic_net.summary_loss]
        return actor_net.action, critic_net.value, actor_net.states, critic_net.states, actor_net.actions_taken, [actor_net.loss, critic_net.loss], adv, r, n_steps

    def create_sync_net_op(self):
        actor_sync_net = sync_networks_op(self.master.shared_actor_net, self.actor_net.vars, self.thread_id)
        critic_sync_net = sync_networks_op(self.master.shared_critic_net, self.critic_net.vars, self.thread_id)
        return tf.group(actor_sync_net, critic_sync_net)

    def make_trainer(self):
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.config["actor_learning_rate"])
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.config["critic_learning_rate"])

        self.actor_sync_net = sync_networks_op(self.master.shared_actor_net, self.actor_net.vars, self.thread_id)
        actor_grads = tf.gradients(self.actor_net.loss, self.actor_net.vars)

        self.critic_sync_net = sync_networks_op(self.master.shared_critic_net, self.critic_net.vars, self.thread_id)
        critic_grads = tf.gradients(self.critic_net.loss, self.critic_net.vars)

        if self.clip_gradients:
            # Clipped gradients
            gradient_clip_value = self.config["gradient_clip_value"]
            processed_actor_grads = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in actor_grads]
            processed_critic_grads = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in critic_grads]
        else:
            # Non-clipped gradients: don't do anything
            processed_actor_grads = actor_grads
            processed_critic_grads = critic_grads

        # Apply gradients to the weights of the master network
        # Only increase global_step counter once per update of the 2 networks
        apply_actor_gradients = actor_optimizer.apply_gradients(
            zip(processed_actor_grads, self.master.shared_actor_net.vars))
        apply_critic_gradients = critic_optimizer.apply_gradients(
            zip(processed_critic_grads, self.master.shared_critic_net.vars))

        return tf.group(apply_actor_gradients, apply_critic_gradients)

class A3CThreadDiscreteCNN(A3CThreadDiscrete):
    """A3CThread for a discrete action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadDiscreteCNN, self).__init__(master, thread_id)

    def create_sync_net_op(self):
        return tf.group(*[v1.assign(v2) for v1, v2 in zip(self.ac_net.vars, self.master.shared_ac_net.vars)])

    def build_networks(self):
        self.ac_net = ac_net = ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["actor_n_hidden"],
            summary=False)
        action = ac_net.action
        value = ac_net.value
        actor_states = ac_net.states
        n_steps = tf.shape(actor_states)[0]
        critic_states = ac_net.states
        actions_taken = ac_net.actions_taken
        loss = ac_net.loss
        actor_loss = ac_net.actor_loss
        critic_loss = ac_net.critic_loss
        adv = ac_net.adv
        r = ac_net.r
        return action, value, actor_states, critic_states, actions_taken, [loss, actor_loss, critic_loss], adv, r, n_steps

    def make_trainer(self):
        optimizer = tf.train.AdamOptimizer(self.config["learning_rate"], name="optim")
        grads = tf.gradients(self.ac_net.loss, self.ac_net.vars)

        grads, _ = tf.clip_by_global_norm(grads, 40.0)

        # Apply gradients to the weights of the master network
        # Only increase global_step counter once per update of the 2 networks
        return optimizer.apply_gradients(
            zip(grads, self.master.shared_ac_net.vars))

class A3CThreadDiscreteCNNRNN(A3CThreadDiscreteCNN):
    """A3CThread for a discrete action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadDiscreteCNNRNN, self).__init__(master, thread_id)
        self.rnn_state = None

    def build_networks(self):
        self.ac_net = ac_net = ActorCriticNetworkDiscreteCNNRNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["actor_n_hidden"],
            summary=False)
        action = ac_net.action
        value = ac_net.value
        actor_states = ac_net.states
        n_steps = tf.shape(actor_states)[0]
        critic_states = ac_net.states
        actions_taken = ac_net.actions_taken
        loss = ac_net.loss
        actor_loss = ac_net.actor_loss
        critic_loss = ac_net.critic_loss
        adv = ac_net.adv
        r = ac_net.r
        self.initial_features = ac_net.state_init
        return action, value, actor_states, critic_states, actions_taken, [loss, actor_loss, critic_loss], adv, r, n_steps

    def choose_action(self, state, features):
        """Choose an action."""
        feed_dict = {
            self.actor_states: [state]
        }
        if self.rnn_state is not None:
            feed_dict[self.ac_net.rnn_state_in] = features
        action, rnn_state, value = self.master.session.run([self.ac_net.action, self.ac_net.rnn_state_out, self.value], feed_dict=feed_dict)
        return action, value, rnn_state

    def get_critic_value(self, states, features):
        feed_dict = {
            self.critic_states: states
        }
        if self.rnn_state is not None:
            feed_dict[self.ac_net.rnn_state_in] = features
        return self.master.session.run(self.value, feed_dict=feed_dict)[0]

class A3CThreadContinuous(A3CThread):
    """A3CThread for a continuous action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadContinuous, self).__init__(master, thread_id)

    def build_networks(self):
        self.actor_net = actor_net = ActorNetworkContinuous(self.env.action_space, list(self.env.observation_space.shape), self.config["actor_n_hidden"])
        self.critic_net = critic_net = CriticNetwork(list(self.env.observation_space.shape), self.config["critic_n_hidden"])
        self.loss_fetches = [actor_net.summary_loss, critic_net.summary_loss]
        actor_states = actor_net.states
        n_steps = tf.shape(actor_states)[0]
        adv = actor_net.adv
        r = critic_net.r
        return actor_net.action, critic_net.value, actor_net.states, critic_net.states, actor_net.actions_taken, [actor_net.loss, critic_net.loss], adv, r, n_steps

    def get_env_action(self, action):
        return action

    def make_trainer(self):
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.config["actor_learning_rate"])
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.config["critic_learning_rate"])

        self.actor_sync_net = sync_networks_op(self.master.shared_actor_net, self.actor_net.vars, self.thread_id)
        actor_grads = tf.gradients(self.actor_net.loss, self.actor_net.vars)

        self.critic_sync_net = sync_networks_op(self.master.shared_critic_net, self.critic_net.vars, self.thread_id)
        critic_grads = tf.gradients(self.critic_net.loss, self.critic_net.vars)

        if self.clip_gradients:
            # Clipped gradients
            gradient_clip_value = self.config["gradient_clip_value"]
            processed_actor_grads = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in actor_grads]
            processed_critic_grads = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in critic_grads]
        else:
            # Non-clipped gradients: don't do anything
            processed_actor_grads = actor_grads
            processed_critic_grads = critic_grads

        # Apply gradients to the weights of the master network
        # Only increase global_step counter once per update of the 2 networks
        apply_actor_gradients = actor_optimizer.apply_gradients(
            zip(processed_actor_grads, self.master.shared_actor_net.vars))
        apply_critic_gradients = critic_optimizer.apply_gradients(
            zip(processed_critic_grads, self.master.shared_critic_net.vars))

        return tf.group(apply_actor_gradients, apply_critic_gradients)

    def create_sync_net_op(self):
        actor_sync_net = sync_networks_op(self.master.shared_actor_net, self.actor_net.vars, self.thread_id)
        critic_sync_net = sync_networks_op(self.master.shared_critic_net, self.critic_net.vars, self.thread_id)
        return tf.group(actor_sync_net, critic_sync_net)

class A3C(Agent):
    """Asynchronous Advantage Actor Critic learner."""
    def __init__(self, env, monitor, monitor_path, video=True, **usercfg):
        super(A3C, self).__init__(**usercfg)
        self.env = env
        self.shared_counter = 0
        self.T = 0
        self.env_name = env.spec.id
        self.monitor = monitor
        self.monitor_path = monitor_path
        self.video = video

        self.config.update(dict(
            gamma=0.99,  # Discount past rewards by a percentage
            learning_rate=1e-4,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-4,
            actor_n_hidden=20,
            critic_n_hidden=20,
            gradient_clip_value=40,
            n_threads=multiprocessing.cpu_count(),  # Use as much threads as there are CPU threads on the current system
            T_max=8e5,
            episode_max_length=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"),
            repeat_n_actions=1,
            save_model=False
        ))
        self.config.update(usercfg)
        self.stop_requested = False

        self.session = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True))

        with tf.variable_scope("global"):
            self.build_networks()

            if self.config["save_model"]:
                tf.add_to_collection("action", self.action)
                tf.add_to_collection("states", self.states)
                self.saver = tf.train.Saver()

            self.losses, loss_summaries = self.create_summary_losses()
            self.reward = tf.placeholder("float", name="reward")
            tf.summary.scalar("Reward", self.reward)
            self.episode_length = tf.placeholder("float", name="episode_length")
            tf.summary.scalar("Episode_length", self.episode_length)
            self.summary_op = tf.summary.merge(loss_summaries)

        self.jobs = []
        for thread_id in range(self.config["n_threads"]):
            job = self.make_thread(thread_id)
            self.jobs.append(job)

        self.session.run(tf.global_variables_initializer())

    def make_thread(self, thread_id):
        return self.thread_type(self, thread_id)

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
            logging.info("Saving model")
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

    def create_summary_losses(self):
        self.actor_loss = tf.placeholder("float", name="actor_loss")
        actor_loss_summary = tf.summary.scalar("Actor_loss", self.actor_loss)
        self.critic_loss = tf.placeholder("float", name="critic_loss")
        critic_loss_summary = tf.summary.scalar("Critic_loss", self.critic_loss)
        return [self.actor_loss, self.critic_loss], [actor_loss_summary, critic_loss_summary]

class A3CDiscrete(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CThreadDiscrete
        super(A3CDiscrete, self).__init__(env, monitor, monitor_path, **usercfg)

    def build_networks(self):
        with tf.variable_scope("global"):
            self.shared_actor_net = ActorNetworkDiscrete(
                list(self.env.observation_space.shape),
                self.env.action_space.n,
                self.config["actor_n_hidden"],
                summary=False)
            self.states = self.shared_actor_net.states
            self.action = self.shared_actor_net.action
            self.shared_critic_net = CriticNetwork(list(self.env.observation_space.shape),
                                                   self.config["critic_n_hidden"],
                                                   summary=False)
            self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)

class A3CDiscreteCNN(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CThreadDiscreteCNN
        super(A3CDiscreteCNN, self).__init__(env, monitor, monitor_path, **usercfg)

    def build_networks(self):
        with tf.variable_scope("global"):
            self.shared_ac_net = ActorCriticNetworkDiscreteCNN(
                state_shape=list(self.env.observation_space.shape),
                n_actions=self.env.action_space.n,
                n_hidden=self.config["actor_n_hidden"],
                summary=False)
            self.states = self.shared_ac_net.states
            self.action = self.shared_ac_net.action
            self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)

    def create_summary_losses(self):
        self.actor_loss = tf.placeholder("float", name="actor_loss")
        actor_loss_summary = tf.summary.scalar("Actor_loss", self.actor_loss)
        self.critic_loss = tf.placeholder("float", name="critic_loss")
        critic_loss_summary = tf.summary.scalar("Critic_loss", self.critic_loss)
        self.loss = tf.placeholder("float", name="loss")
        loss_summary = tf.summary.scalar("Loss", self.loss)
        return [self.actor_loss, self.critic_loss, self.loss], [actor_loss_summary, critic_loss_summary, loss_summary]

class A3CDiscreteCNNRNN(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CThreadDiscreteCNNRNN
        super(A3CDiscreteCNNRNN, self).__init__(env, monitor, monitor_path, **usercfg)
        self.config["RNN"] = True

    def build_networks(self):
        with tf.variable_scope("global"):
            self.shared_ac_net = ActorCriticNetworkDiscreteCNNRNN(
                state_shape=list(self.env.observation_space.shape),
                n_actions=self.env.action_space.n,
                n_hidden=self.config["actor_n_hidden"],
                summary=False)
            self.states = self.shared_ac_net.states
            self.action = self.shared_ac_net.action
            self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)

    def create_summary_losses(self):
        self.actor_loss = tf.placeholder("float", name="actor_loss")
        actor_loss_summary = tf.summary.scalar("Actor_loss", self.actor_loss)
        self.critic_loss = tf.placeholder("float", name="critic_loss")
        critic_loss_summary = tf.summary.scalar("Critic_loss", self.critic_loss)
        self.loss = tf.placeholder("float", name="loss")
        loss_summary = tf.summary.scalar("loss", self.loss)
        return [self.actor_loss, self.critic_loss, self.loss], [actor_loss_summary, critic_loss_summary, loss_summary]

class A3CContinuous(A3C):
    """A3C for a continuous action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CThreadContinuous
        super(A3CContinuous, self).__init__(env, monitor, monitor_path, **usercfg)

    def build_networks(self):
        self.shared_actor_net = ActorNetworkContinuous(
            self.env.action_space,
            list(self.env.observation_space.shape),
            self.config["actor_n_hidden"],
            summary=False)
        self.states = self.shared_actor_net.states
        self.action = self.shared_actor_net.action
        self.shared_critic_net = CriticNetwork(list(self.env.observation_space.shape),
                                               self.config["critic_n_hidden"],
                                               summary=False)
        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)
