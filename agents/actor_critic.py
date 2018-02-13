# -*- coding: utf8 -*-

"""
Functions and networks for actor-critic agents.
"""

import numpy as np
import tensorflow as tf

from misc.network_ops import normalized_columns_initializer, linear, conv2d, flatten, mu_sigma_layer

class ActorCriticNetworkDiscrete(object):
    """Neural network for the Actor of an Actor-Critic algorithm using a discrete action space"""
    def __init__(self, state_shape, n_actions, n_hidden):
        super(ActorCriticNetworkDiscrete, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        self.states = tf.placeholder(tf.float32, [None] + list(state_shape), name="states")
        self.adv = tf.placeholder(tf.float32, name="advantage")
        self.actions_taken = tf.placeholder(tf.float32, [None, n_actions], name="actions_taken")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.n_hidden,
            activation_fn=tf.tanh,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=tf.zeros_initializer(),
            scope="L1")

        # Fully connected for actor
        self.logits = linear(L1, n_actions, "actionlogits", normalized_columns_initializer(0.01))

        self.probs = tf.nn.softmax(self.logits)

        self.action = tf.squeeze(tf.multinomial(self.logits - tf.reduce_max(self.logits, [1], keep_dims=True), 1), [1], name="action")
        self.action = tf.one_hot(self.action, n_actions)[0, :]

        # Fully connected for critic
        self.value = tf.reshape(linear(L1, 1, "value", normalized_columns_initializer(1.0)), [-1])

class ActorCriticNetworkDiscreteCNN(object):
    """docstring for ActorCriticNetworkDiscreteCNNRNN"""
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

class ActorCriticNetworkDiscreteCNNRNN(object):
    """docstring for ActorCriticNetworkDiscreteCNNRNN"""
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

        # Fully connected for actor and critic
        self.logits = linear(L3, n_actions, "actionlogits", normalized_columns_initializer(0.01))
        self.value = tf.reshape(linear(L3, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.probs = tf.nn.softmax(self.logits)

        self.action = tf.squeeze(tf.multinomial(self.logits - tf.reduce_max(self.logits, [1], keep_dims=True), 1), [1], name="action")
        self.action = tf.one_hot(self.action, n_actions)[0, :]

def ActorCriticDiscreteLoss(network, vf_coef=0.5, entropy_coef=0.01, reducer="sum"):
    tf_reducer = tf.reduce_sum if reducer == "sum" else tf.reduce_mean
    log_probs = tf.nn.log_softmax(network.logits)
    actor_loss = - tf_reducer(tf.reduce_sum(log_probs * network.actions_taken, [1]) * network.adv)
    critic_loss = tf_reducer(tf.square(network.value - network.r))
    entropy = tf_reducer(network.probs * log_probs)
    loss = actor_loss + vf_coef * critic_loss - entropy_coef * entropy
    return actor_loss, critic_loss, loss

class ActorCriticNetworkContinuous(object):
    """Neural network for an Actor of an Actor-Critic algorithm using a continuous action space."""
    def __init__(self, state_shape, action_space, n_hidden):
        super(ActorCriticNetworkContinuous, self).__init__()
        self.state_shape = state_shape
        self.n_hidden = n_hidden

        self.states = tf.placeholder("float", [None] + list(state_shape), name="states")
        self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
        self.adv = tf.placeholder(tf.float32, name="advantage")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.n_hidden,
            activation_fn=tf.tanh,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=tf.zeros_initializer(),
            scope="mu_L1")

        mu, sigma = mu_sigma_layer(L1, 1)

        self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.action = self.normal_dist.sample(1)
        self.action = tf.clip_by_value(self.action, action_space.low[0], action_space.high[0], name="action")
        self.value = tf.reshape(linear(L1, 1, "value", normalized_columns_initializer(1.0)), [-1])

def ActorCriticContinuousLoss(network, entropy_coef=0.01, reducer="sum"):
    tf_reducer = tf.reduce_sum if reducer == "sum" else tf.reduce_mean
    actor_loss = - tf_reducer(network.normal_dist.log_prob(network.actions_taken) * network.adv)
    critic_loss = tf_reducer(tf.square(network.value - network.r))
    entropy = tf_reducer(network.normal_dist.entropy())
    loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
    return actor_loss, critic_loss, loss
