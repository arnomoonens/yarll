# -*- coding: utf8 -*-

"""
Functions and networks for actor-critic agents.
"""

from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, GRU
from tensorflow.keras.initializers import Orthogonal
import numpy as np

from yarll.misc.network_ops import CategoricalProbabilityDistribution, MultiCategoricalProbabilityDistribution, \
    NormalDistrLayer, normal_dist_log_prob, categorical_dist_entropy, bernoulli_dist_entropy

class ActorCriticNetwork(Model):

    def entropy(self, *args):
        raise NotImplementedError()

    def action_value(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

class ActorCriticNetworkLatent(ActorCriticNetwork):
    def __init__(self, n_latent: int, n_hidden_units: int, n_hidden_layers: int) -> None:
        super(ActorCriticNetworkLatent, self).__init__()

        self.logits = Sequential()
        for _ in range(n_hidden_layers):
            self.logits.add(Dense(n_hidden_units, activation="tanh", kernel_initializer=Orthogonal(gain=np.sqrt(2))))
        self.logits.add(Dense(n_latent, kernel_initializer=Orthogonal(gain=0.01)))

        self.value = Sequential()
        for _ in range(n_hidden_layers):
            self.value.add(Dense(n_hidden_units, activation="tanh", kernel_initializer=Orthogonal(gain=np.sqrt(2))))
        self.value.add(Dense(1, kernel_initializer=Orthogonal(gain=0.01)))

    def call(self, states: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(states, dtype=tf.float32)  # convert from Numpy array to Tensor
        return self.logits(x), self.value(x)

class ActorCriticNetworkDiscrete(ActorCriticNetworkLatent):
    def __init__(self, n_actions: int, n_hidden_units: int, n_hidden_layers: int) -> None:
        super(ActorCriticNetworkDiscrete, self).__init__(n_actions, n_hidden_units, n_hidden_layers)
        self.dist = CategoricalProbabilityDistribution()

    def action_value(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Source: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
        """
        logits, value = self.predict(states)
        action = self.dist(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def entropy(self, *args):
        logits, *_ = args
        return categorical_dist_entropy(logits)

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(actions, dtype=tf.int32), logits=logits)


class ActorCriticNetworkMultiDiscrete(ActorCriticNetworkLatent):
    def __init__(self, n_actions_per_dim: List[int], n_hidden_units: int, n_hidden_layers: int) -> None:
        self.n_actions_per_dim = tf.cast(n_actions_per_dim, tf.int32)
        super(ActorCriticNetworkMultiDiscrete, self).__init__(sum(n_actions_per_dim), n_hidden_units, n_hidden_layers)
        self.dist = MultiCategoricalProbabilityDistribution()

    def action_value(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Source: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
        """
        logits, value = self.predict(states)
        reshaped_logits = tf.split(logits, self.n_actions_per_dim, axis=-1)
        action = self.dist(reshaped_logits)
        return np.squeeze(action), np.squeeze(value, axis=-1)

    def entropy(self, *args):
        logits, *_ = args
        reshaped_logits = tf.split(logits, self.n_actions_per_dim, axis=-1)
        return tf.add_n([categorical_dist_entropy(l) for l in reshaped_logits])

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        map_result = tf.map_fn(lambda x: -tf.nn.sparse_softmax_cross_entropy_with_logits(x[0], x[1]),
                               (tf.transpose(tf.cast(actions, dtype=tf.int32)),
                                tf.transpose(tf.reshape(logits, (logits.shape[0], actions.shape[1], -1)), perm=(1, 0, 2))),
                               dtype=tf.float32)
        return tf.reduce_sum(map_result, axis=0)


class ActorCriticNetworkBernoulli(ActorCriticNetworkLatent):
    def __init__(self, n_actions: int, n_hidden_units: int, n_hidden_layers: int) -> None:
        super(ActorCriticNetworkBernoulli, self).__init__(n_actions, n_hidden_units, n_hidden_layers)

    def action_value(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Source: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/distributions.py#L457
        """
        logits, value = self.predict(states)
        probs = tf.sigmoid(logits)
        samples_from_uniform = tf.random.uniform(probs.shape)
        action = tf.cast(tf.less(samples_from_uniform, probs), tf.float32)

        return np.reshape(action.numpy(), (-1,)), np.squeeze(value, axis=-1)

    def entropy(self, *args):
        logits, *_ = args
        return bernoulli_dist_entropy(logits)

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(actions, tf.float32),
                                                                      logits=logits),
                              axis=-1)

class ActorCriticNetworkDiscreteCNN(ActorCriticNetwork):
    """docstring for ActorCriticNetworkDiscreteCNNRNN"""

    def __init__(self, n_actions: int, n_hidden: int) -> None:
        super(ActorCriticNetworkDiscreteCNN, self).__init__()

        self.shared_layers = Sequential()

        # Convolution layers
        for _ in range(4):
            self.shared_layers.add(Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="elu"))
        self.shared_layers.add(Flatten())

        self.shared_layers.add(Dense(n_hidden, activation="relu"))

        self.logits = Dense(n_actions)
        self.dist = CategoricalProbabilityDistribution()

        self.value = Dense(1)

    def call(self, states: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(states, dtype=tf.float32)  # convert from Numpy array to Tensor
        x = self.shared_layers(x)
        return self.logits(x), self.value(x)

    def action_value(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Source: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
        """
        logits, value = self.predict(states)
        action = self.dist(logits)

        return np.squeeze(action.numpy(), axis=-1), np.squeeze(value, axis=-1)

    def entropy(self, *args):
        logits, *_ = args
        return categorical_dist_entropy(logits)

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        map_result = tf.map_fn(lambda x: -tf.nn.sparse_softmax_cross_entropy_with_logits(x[0], x[1]),
                               (tf.transpose(tf.cast(actions, dtype=tf.int32)),
                                tf.transpose(tf.reshape(logits, (logits.shape[0], actions.shape[1], -1)), perm=(1, 0, 2))),
                               dtype=tf.float32)
        return tf.reduce_sum(map_result, axis=0)

class ActorCriticNetworkDiscreteCNNRNN(ActorCriticNetwork):
    """docstring for ActorCriticNetworkDiscreteCNNRNN"""

    def __init__(self, n_actions: int, rnn_size: int = 256) -> None:
        super(ActorCriticNetworkDiscreteCNNRNN, self).__init__()

        self.shared_layers = Sequential()

        # Convolution layers
        for _ in range(4):
            self.shared_layers.add(Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="elu"))
        self.shared_layers.add(Flatten())
        self.shared_layers.add(Lambda(lambda x: tf.expand_dims(x, [1])))

        self.rnn = GRU(rnn_size, return_state=True)
        self.initial_features = np.zeros((1, rnn_size))

        self.logits = Dense(n_actions)
        self.dist = CategoricalProbabilityDistribution()

        self.value = Dense(1)

    def call(self, state_hidden):
        states, hiddens = state_hidden
        x = tf.convert_to_tensor(states, dtype=tf.float32)  # convert from Numpy array to Tensor
        x = self.shared_layers(x)
        x, new_rnn_state = self.rnn(x, hiddens)
        return self.logits(x), self.value(x), new_rnn_state

    def action_value(self, states: np.ndarray, features=None):
        """
        Source: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
        """
        inp = states if features is None else [states, features]
        logits, value, *features = self.predict(inp)
        action = self.dist(logits)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1), None if not features else features[0]

    def entropy(self, *args):
        logits, *_ = args
        return categorical_dist_entropy(logits)

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        map_result = tf.map_fn(lambda x: -tf.nn.sparse_softmax_cross_entropy_with_logits(x[0], x[1]),
                               (tf.transpose(tf.cast(actions, dtype=tf.int32)),
                                tf.transpose(tf.reshape(logits, (logits.shape[0], actions.shape[1], -1)), perm=(1, 0, 2))),
                               dtype=tf.float32)
        return tf.reduce_sum(map_result, axis=0)

def actor_discrete_loss(actions, advantages, logits):
    """
    Adapted from: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
    """
    # sparse categorical CE loss obj that supports sample_weight arg on call()
    # from_logits argument ensures transformation into normalized probabilities
    weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # policy loss is defined by policy gradients, weighted by advantages
    # note: we only calculate the loss on the actions we've actually taken
    actions = tf.cast(actions, tf.int32)
    policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
    # entropy loss can be calculated via CE over itself
    # TODO: use this
    # entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
    # here signs are flipped because optimizer minimizes
    # return policy_loss - self.params['entropy']*entropy_loss
    return policy_loss

def critic_loss(returns, value):
    return tf.square(value - returns)


class ActorCriticNetworkContinuous(ActorCriticNetwork):
    """Neural network for an Actor of an Actor-Critic algorithm using a continuous action space."""

    def __init__(self, action_space_shape, n_hidden_units: int, n_hidden_layers: int = 1) -> None:
        super(ActorCriticNetworkContinuous, self).__init__()

        self.policy_hidden = Sequential()

        for _ in range(n_hidden_layers):
            self.policy_hidden.add(Dense(n_hidden_units, activation="tanh"))
        self.action_mean = NormalDistrLayer(action_space_shape[0])

        self.critic = Sequential()
        for _ in range(n_hidden_layers):
            self.critic.add(Dense(n_hidden_units, activation="tanh"))
        self.critic.add(Dense(1))

    def call(self, inp):
        x = self.policy_hidden(inp)
        action, mean = self.action_mean(x)
        return action, mean, self.critic(inp)

    def action_value(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        action, mean, value = self.predict(states)
        return np.squeeze(action, axis=0), np.squeeze(mean, axis=0), np.squeeze(value, axis=-1)

    def entropy(self, *args):
        return self.action_mean.entropy()

def actor_continuous_loss(actions_taken, mean, log_std, advantage):
    action_log_prob = normal_dist_log_prob(actions_taken, mean, log_std)
    loss = action_log_prob * advantage
    return loss
