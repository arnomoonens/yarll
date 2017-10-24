# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf
import logging
from threading import Thread
import multiprocessing
import signal

from gym import wrappers

from environment.registration import make_environment
from agents.agent import Agent
from misc.utils import discount_rewards, preprocess_image
from misc.network_ops import sync_networks_op, conv2d, mu_sigma_layer, flatten, normalized_columns_initializer

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
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            self.probs = tf.contrib.layers.fully_connected(
                inputs=L1,
                num_outputs=n_actions,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                biases_initializer=tf.zeros_initializer(),
                scope="probs")

            self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

            self.loss = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.log(self.probs) * self.actions_taken, [1]) * (self.critic_rewards - self.critic_feedback)), name="loss")

            self.summary_loss = self.loss  # Loss to show as a summary
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class ActorCriticNetworkDiscreteCNN(object):
    """docstring for ActorNetworkDiscreteCNNRNN"""
    def __init__(self, state_shape, n_actions, n_hidden, scope, summary=True):
        super(ActorCriticNetworkDiscreteCNN, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.scope = scope
        self.summary = summary

        with tf.variable_scope("{}_actorcritic".format(scope)):
            image_size = 80
            image_depth = 1  # aka nr. of feature maps. Eg 3 for RGB images. 1 here because we use grayscale images

            self.states = tf.placeholder(tf.float32, [None, image_size, image_size, image_depth], name="states")
            self.N = tf.placeholder(tf.int32, name="N")
            self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage
            self.target = tf.placeholder("float", [None], name="critic_target")
            self.critic_feedback = tf.placeholder(tf.float32, name="critic_feedback")
            self.critic_rewards = tf.placeholder(tf.float32, name="critic_rewards")
            self.actions_taken = tf.placeholder(tf.float32, [None, n_actions], name="actions_taken")

            x = self.states
            # Convolution layers
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

            # Flatten
            shape = x.get_shape().as_list()
            reshape = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

            # Fully connected for Actor
            self.logits = tf.contrib.layers.fully_connected(
                inputs=reshape,
                num_outputs=self.n_actions,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=tf.zeros_initializer())

            self.probs = tf.nn.softmax(self.logits)

            self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

            # Fully connected for Critic
            self.value = tf.contrib.layers.fully_connected(
                inputs=reshape,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=tf.zeros_initializer())

            log_probs = tf.log(self.probs)
            td_diff = self.critic_rewards - self.critic_feedback
            self.actor_loss = - tf.reduce_sum(tf.reduce_sum(log_probs * self.actions_taken, [1]) * td_diff)

            self.critic_loss = 0.5 * tf.reduce_mean(tf.square(td_diff))

            entropy = - tf.reduce_sum(self.probs * tf.log(self.probs + 1e-8))

            self.loss = self.actor_loss + 0.5 * self.critic_loss - entropy * 0.01
            self.summary_loss = self.critic_loss

            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class ActorCriticNetworkDiscreteCNNRNN(object):
    """docstring for ActorNetworkDiscreteCNNRNN"""
    def __init__(self, state_shape, n_actions, n_hidden, scope, summary=True):
        super(ActorCriticNetworkDiscreteCNNRNN, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.scope = scope
        self.summary = summary

        with tf.variable_scope("{}_actorcritic".format(scope)):
            image_size = 80
            image_depth = 1  # aka nr. of feature maps. Eg 3 for RGB images. 1 here because we use grayscale images

            self.states = tf.placeholder(tf.float32, [None, image_size, image_size, image_depth], name="states")
            self.N = tf.placeholder(tf.int32, name="N")
            self.adv_n = tf.placeholder(tf.float32, name="adv_n")  # Advantage
            self.target = tf.placeholder("float", [None], name="critic_target")
            self.critic_feedback = tf.placeholder(tf.float32, name="critic_feedback")
            self.critic_rewards = tf.placeholder(tf.float32, name="critic_rewards")
            self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")

            x = self.states
            # Convolution layers
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

            # Flatten
            reshape = tf.expand_dims(flatten(x), [0])

            lstm_size = 256
            self.enc_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.rnn_state_in = self.enc_cell.zero_state(1, tf.float32)
            L3, self.rnn_state_out = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                       inputs=reshape,
                                                       initial_state=self.rnn_state_in,
                                                       dtype=tf.float32)
            L3 = tf.reshape(L3, [-1, lstm_size])
            # Fully connected for Actor
            self.logits = tf.contrib.layers.fully_connected(
                inputs=L3,
                num_outputs=self.n_actions,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=tf.zeros_initializer())

            self.probs = tf.nn.softmax(self.logits)

            self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

            # Fully connected for Critic
            self.value = tf.contrib.layers.fully_connected(
                inputs=L3,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=tf.zeros_initializer())

            log_probs = tf.log(self.probs)
            td_diff = self.critic_rewards - self.critic_feedback
            self.actor_loss = - tf.reduce_sum(tf.reduce_sum(log_probs * self.actions_taken, [1]) * td_diff)

            self.critic_loss = 0.5 * tf.reduce_mean(tf.square(td_diff))

            entropy = - tf.reduce_sum(self.probs * tf.log(self.probs + 1e-8))

            self.loss = self.actor_loss + 0.5 * self.critic_loss - entropy * 0.01
            self.summary_loss = self.critic_loss

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
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                biases_initializer=tf.zeros_initializer(),
                scope="mu_L1")

            mu, sigma = mu_sigma_layer(L1, 1)

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
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                biases_initializer=tf.zeros_initializer(),
                scope="L1")

            self.value = tf.contrib.layers.fully_connected(
                inputs=L1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
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
        self.clip_gradients = clip_gradients
        self.env = make_environment(master.env_name)
        self.master = master
        self.config = master.config
        self.state_preprocessor = None
        if thread_id == 0 and self.master.monitor:
            self.env = wrappers.Monitor(self.env, master.monitor_path, force=True, video_callable=(None if self.master.video else False))

        # Build actor and critic networks
        self.build_networks()

        # Write the summary of each thread in a different directory
        self.writer = tf.summary.FileWriter(os.path.join(self.master.monitor_path, "thread" + str(self.thread_id)), self.master.session.graph)

        self.sync_net = self.create_sync_net_op()
        self.train_op = self.make_trainer()

    def transform_actions(self, actions):
        return actions

    def get_critic_value(self, states):
        feed_dict = {
            self.ac_net.states: states
        }
        if self.rnn_state is not None:
            feed_dict[self.ac_net.rnn_state_in] = self.rnn_state
        value, self.rnn_state = self.master.session.run([self.ac_net.value, self.ac_net.rnn_state_out], feed_dict=feed_dict)
        return value

    def get_trajectory(self, episode_max_length, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory).
        Return dictionary of results
        """
        state = self.env.reset()
        state = state if self.state_preprocessor is None else self.state_preprocessor(state)
        self.rnn_state = None
        states = []
        actions = []
        rewards = []
        for i in range(episode_max_length):
            action = self.choose_action(state)  # Predict the next action (using a neural network) depending on the current state
            states.append(state)
            state, reward, done, _ = self.env.step(action)
            state = state if self.state_preprocessor is None else self.state_preprocessor(state)
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
        return self.master.session.run([self.action], feed_dict={self.actor_states: [state]})[0]

    def run(self):
        # Assume global shared parameter vectors θ and θv and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'v
        sess = self.master.session
        t = 1  # thread step counter
        while self.master.T < self.config["T_max"] and not self.master.stop_requested:
            # Synchronize thread-specific parameters θ' = θ and θ'v = θv
            sess.run([self.sync_net])
            trajectory = self.get_trajectory(self.config["episode_max_length"])
            reward = sum(trajectory["reward"])
            trajectory["reward"][-1] = 0 if trajectory["done"] else self.get_critic_value(trajectory["state"][None, -1])[0]
            returns = discount_rewards(trajectory["reward"], self.config["gamma"])
            fetches = self.loss_fetches + [self.train_op, self.master.global_step]

            qw_new = self.master.session.run([self.value], feed_dict={self.critic_states: trajectory["state"]})[0].flatten()
            all_action = self.transform_actions(trajectory["action"])  # Transform actions back to the output shape of the actor network (e.g. one-hot for discrete action space)
            results = sess.run(fetches, feed_dict={
                self.actor_states: trajectory["state"],
                self.critic_states: trajectory["state"],
                self.actions_taken: all_action,
                self.critic_feedback: qw_new,
                self.critic_rewards: returns,
                self.critic_target: returns
            })
            feed_dict = {
                self.master.reward: reward,
                self.master.episode_length: trajectory["steps"]
            }
            feed_dict.update(dict(zip(self.master.losses, results)))
            summary = sess.run([self.master.summary_op], feed_dict)

            self.writer.add_summary(summary[0], results[-1])
            self.writer.flush()
            t += 1
            self.master.T += trajectory["steps"]

class A3CThreadDiscrete(A3CThread):
    """A3CThread for a discrete action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadDiscrete, self).__init__(master, thread_id)

    def build_networks(self):
        self.actor_net = ActorNetworkDiscrete(self.env.observation_space.shape[0], self.env.action_space.n, self.config["actor_n_hidden"], scope="thread{}".format(self.thread_id))
        self.critic_net = CriticNetwork(self.env.observation_space.shape[0], self.config["critic_n_hidden"], scope="thread{}".format(self.thread_id))
        self.action = self.actor_net.action
        self.value = self.critic_net.value
        self.actor_states = self.actor_net.states
        self.critic_states = self.critic_net.states
        self.actions_taken = self.actor_net.actions_taken
        self.critic_feedback = self.actor_net.critic_feedback
        self.critic_rewards = self.actor_net.critic_rewards
        self.critic_target = self.critic_net.target
        self.loss_fetches = [self.actor_net.summary_loss, self.critic_net.summary_loss]

    def create_sync_net_op(self):
        actor_sync_net = sync_networks_op(self.master.shared_actor_net, self.actor_net.vars, self.thread_id)
        critic_sync_net = sync_networks_op(self.master.shared_critic_net, self.critic_net.vars, self.thread_id)
        return tf.group(actor_sync_net, critic_sync_net)

    def make_trainer(self):
        actor_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["actor_learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])
        critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["critic_learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])

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
            zip(processed_actor_grads, self.master.shared_actor_net.vars), global_step=self.master.global_step)
        apply_critic_gradients = critic_optimizer.apply_gradients(
            zip(processed_critic_grads, self.master.shared_critic_net.vars))

        return tf.group(apply_actor_gradients, apply_critic_gradients)

    def transform_actions(self, actions):
        possible_actions = np.arange(self.env.action_space.n)
        return (possible_actions == actions[:, None]).astype(np.float32)

class A3CThreadDiscreteCNN(A3CThreadDiscrete):
    """A3CThread for a discrete action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadDiscreteCNN, self).__init__(master, thread_id)
        self.state_preprocessor = preprocess_image

    def create_sync_net_op(self):
        return sync_networks_op(self.master.shared_ac_net, self.ac_net.vars, self.thread_id)

    def build_networks(self):
        self.ac_net = ActorCriticNetworkDiscreteCNN(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.config["actor_n_hidden"],
            scope="t{}_ac_net".format(self.thread_id),
            summary=False)
        self.loss_fetches = [self.ac_net.summary_loss]
        self.action = self.ac_net.action
        self.value = self.ac_net.value
        self.actor_states = self.ac_net.states
        self.critic_states = self.ac_net.states
        self.actions_taken = self.ac_net.actions_taken
        self.critic_feedback = self.ac_net.critic_feedback
        self.critic_rewards = self.ac_net.critic_rewards
        self.critic_target = self.ac_net.target

    def make_trainer(self):
        optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
        grads = tf.gradients(self.ac_net.loss, self.ac_net.vars)

        if False:
            # Clipped gradients
            gradient_clip_value = self.config["gradient_clip_value"]
            processed_grads = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in grads]
        else:
            # Non-clipped gradients: don't do anything
            processed_grads = grads

        # Apply gradients to the weights of the master network
        # Only increase global_step counter once per update of the 2 networks
        return optimizer.apply_gradients(
            zip(processed_grads, self.master.shared_ac_net.vars), global_step=self.master.global_step)

class A3CThreadDiscreteCNNRNN(A3CThreadDiscreteCNN):
    """A3CThread for a discrete action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadDiscreteCNNRNN, self).__init__(master, thread_id)
        self.state_preprocessor = preprocess_image

    def build_networks(self):
        self.ac_net = ActorCriticNetworkDiscreteCNNRNN(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.config["actor_n_hidden"],
            scope="t{}_ac_net".format(self.thread_id),
            summary=False)
        self.loss_fetches = [self.ac_net.summary_loss]
        self.action = self.ac_net.action
        self.value = self.ac_net.value
        self.actor_states = self.ac_net.states
        self.critic_states = self.ac_net.states
        self.actions_taken = self.ac_net.actions_taken
        self.critic_feedback = self.ac_net.critic_feedback
        self.critic_rewards = self.ac_net.critic_rewards
        self.critic_target = self.ac_net.target

    def choose_action(self, state):
        """Choose an action."""
        feed_dict = {
            self.ac_net.states: [state]
        }
        if self.rnn_state is not None:
            feed_dict[self.ac_net.rnn_state_in] = self.rnn_state
        action, self.rnn_state = self.master.session.run([self.ac_net.action, self.ac_net.rnn_state_out], feed_dict=feed_dict)
        return action

class A3CThreadContinuous(A3CThread):
    """A3CThread for a continuous action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadContinuous, self).__init__(master, thread_id)

    def build_networks(self):
        self.actor_net = ActorNetworkContinuous(self.env.action_space, self.env.observation_space.shape[0], self.config["actor_n_hidden"], scope="local_actor_net")
        self.critic_net = CriticNetwork(self.env.observation_space.shape[0], self.config["critic_n_hidden"], scope="local_critic_net")
        self.loss_fetches = [self.actor_net.summary_loss, self.critic_net.summary_loss]
        self.action = self.actor_net.action
        self.value = self.critic_net.value
        self.actor_states = self.actor_net.states
        self.critic_states = self.critic_net.states
        self.actions_taken = self.actor_net.actions_taken
        self.critic_feedback = self.actor_net.critic_feedback
        self.critic_rewards = self.actor_net.critic_rewards
        self.critic_target = self.critic_net.target

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
            decay=0.9,  # Decay of RMSProp optimizer
            epsilon=1e-9,  # Epsilon of RMSProp optimizer
            learning_rate=1e-4,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-4,
            actor_n_hidden=20,
            critic_n_hidden=20,
            gradient_clip_value=40,
            n_threads=min(16, multiprocessing.cpu_count()),  # Use as much threads as there are CPU threads on the current system
            T_max=8e5,
            episode_max_length=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"),
            repeat_n_actions=1,
            save_model=False
        ))
        self.config.update(usercfg)
        self.stop_requested = False

        self.build_networks()
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)

        self.session = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True))

        self.losses, loss_summaries = self.create_summary_losses()
        self.reward = tf.placeholder("float", name="reward")
        reward_summary = tf.summary.scalar("Reward", self.reward)
        self.episode_length = tf.placeholder("float", name="episode_length")
        episode_length_summary = tf.summary.scalar("Episode_length", self.episode_length)
        self.summary_op = tf.summary.merge(loss_summaries + [reward_summary, episode_length_summary])

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
        self.shared_actor_net = ActorNetworkDiscrete(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.config["actor_n_hidden"],
            scope="global_actor_net",
            summary=False)
        self.states = self.shared_actor_net.states
        self.action = self.shared_actor_net.action
        self.shared_critic_net = CriticNetwork(self.env.observation_space.shape[0],
                                               self.config["critic_n_hidden"],
                                               scope="global_critic_net",
                                               summary=False)

class A3CDiscreteCNN(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CThreadDiscreteCNN
        super(A3CDiscreteCNN, self).__init__(env, monitor, monitor_path, **usercfg)

    def build_networks(self):
        self.shared_ac_net = ActorCriticNetworkDiscreteCNN(
            state_shape=self.env.observation_space.shape[0],
            n_actions=self.env.action_space.n,
            n_hidden=self.config["actor_n_hidden"],
            scope="global_ac_net",
            summary=False)
        self.states = self.shared_ac_net.states
        self.action = self.shared_ac_net.action

    def create_summary_losses(self):
        self.loss = tf.placeholder("float", name="loss")
        loss_summary = tf.summary.scalar("loss", self.loss)
        return [self.loss], [loss_summary]

class A3CDiscreteCNNRNN(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CThreadDiscreteCNNRNN
        super(A3CDiscreteCNNRNN, self).__init__(env, monitor, monitor_path, **usercfg)
        self.config["RNN"] = True

    def build_networks(self):
        self.shared_ac_net = ActorCriticNetworkDiscreteCNNRNN(
            state_shape=self.env.observation_space.shape[0],
            n_actions=self.env.action_space.n,
            n_hidden=self.config["actor_n_hidden"],
            scope="global_ac_net",
            summary=False)
        self.states = self.shared_ac_net.states
        self.action = self.shared_ac_net.action

    def create_summary_losses(self):
        self.loss = tf.placeholder("float", name="loss")
        loss_summary = tf.summary.scalar("loss", self.loss)
        return [self.loss], [loss_summary]

class A3CContinuous(A3C):
    """A3C for a continuous action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CThreadContinuous
        super(A3CContinuous, self).__init__(env, monitor, monitor_path, **usercfg)

    def build_networks(self):
        self.shared_actor_net = ActorNetworkContinuous(
            self.env.action_space,
            self.env.observation_space.shape[0],
            self.config["actor_n_hidden"],
            scope="global_actor_net",
            summary=False)
        self.states = self.shared_actor_net.states
        self.action = self.shared_actor_net.action
        self.shared_critic_net = CriticNetwork(
            self.env.observation_space.shape[0],
            self.config["critic_n_hidden"],
            scope="global_critic_net",
            summary=False)
