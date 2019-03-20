# -*- coding: utf8 -*-

"""
Functions and networks for actor-critic agents.
"""

from typing import Sequence, Tuple
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, GRU, Reshape
import numpy as np

from yarll.misc.network_ops import ProbabilityDistribution

class ActorCriticNetwork(Model):
    pass

class ActorCriticNetworkDiscrete(ActorCriticNetwork):
    """
    Neural network for the Actor of an Actor-Critic algorithm using a discrete action space.
    """

    def __init__(self, n_actions: int, n_hidden_units: int, n_hidden_layers: int) -> None:
        super(ActorCriticNetworkDiscrete, self).__init__()

        self.logits = Sequential()
        for _ in range(n_hidden_layers):
            self.logits.add(Dense(n_hidden_units, activation="tanh"))
        self.logits.add(Dense(n_actions))

        self.dist = ProbabilityDistribution()

        self.value = Sequential()
        for _ in range(n_hidden_layers):
            self.value.add(Dense(n_hidden_units, activation="tanh"))
        self.value.add(Dense(1))

        # self.entropy = self.probs * self.log_probs

    def call(self, states: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(states, dtype=tf.float32)  # convert from Numpy array to Tensor
        return self.logits(x), self.value(x)

    def action_value(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Source: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
        """
        logits, value = self.predict(states)
        action = self.dist(logits)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


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
        self.dist = ProbabilityDistribution()

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

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

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
        self.dist = ProbabilityDistribution()

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



# def actor_critic_discrete_loss(logits,
#                                probs,
#                                value,
#                                actions_taken,
#                                advantage,
#                                ret,
#                                vf_coef: float = 0.5,
#                                entropy_coef: float = 0.01,
#                                reducer="sum"):
#     tf_reducer = tf.reduce_sum if reducer == "sum" else tf.reduce_mean
#     log_probs = tf.nn.log_softmax(logits)
#     actor_loss = - tf_reducer(tf.reduce_sum(log_probs * actions_taken, [1]) * advantage)
#     critic_loss = tf_reducer(tf.square(value - ret))
#     entropy = tf_reducer(probs * log_probs)
#     loss = actor_loss + vf_coef * critic_loss - entropy_coef * entropy
#     return actor_loss, critic_loss, loss

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
    return tf.keras.losses.mean_squared_error(returns, value)


class ActorCriticNetworkContinuous(ActorCriticNetwork):
    """Neural network for an Actor of an Actor-Critic algorithm using a continuous action space."""

    def __init__(self, state_shape: Sequence[int], action_space, n_hidden_units: int, n_hidden_layers: int = 1) -> None:
        super(ActorCriticNetworkContinuous, self).__init__()
        self.state_shape = state_shape

        x = self.states
        for i in range(n_hidden_layers):
            x = tf.tanh(linear(x, n_hidden_units, "L{}_mean".format(i + 1),
                               initializer=normalized_columns_initializer(1.0)))
        self.mean = linear(x, action_space.shape[0], "mean", initializer=normalized_columns_initializer(0.01))
        self.mean = tf.check_numerics(self.mean, "mean")

        self.log_std = tf.get_variable(
            name="logstd",
            shape=list(action_space.shape),
            initializer=tf.zeros_initializer()
        )
        self.std = tf.exp(self.log_std, name="std")
        self.std = tf.check_numerics(self.std, "std")

        self.action = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
        self.action = tf.reshape(self.action, list(action_space.shape))

        x = self.states
        for i in range(n_hidden_layers):
            x = tf.tanh(linear(x, n_hidden_units, "L{}_value".format(i + 1),
                               initializer=normalized_columns_initializer(1.0)))

        self.value = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])

        neglogprob = 0.5 * tf.reduce_sum(tf.square((self.actions_taken - self.mean) / self.std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.actions_taken)[-1]) \
            + tf.reduce_sum(self.log_std, axis=-1)
        self.action_log_prob = -neglogprob
        self.entropy = -tf.reduce_sum(self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=-1)


def actor_critic_continuous_loss(action_log_prob,
                                 entropy,
                                 value,
                                 advantage,
                                 ret,
                                 vf_coef: float = 0.5,
                                 entropy_coef: float = 0.01,
                                 reducer: str = "sum"):
    tf_reducer = tf.reduce_sum if reducer == "sum" else tf.reduce_mean
    actor_loss = - tf_reducer(action_log_prob * advantage)
    critic_loss = tf_reducer(tf.square(value - ret))
    loss = actor_loss + vf_coef * critic_loss - entropy_coef * entropy
    return actor_loss, critic_loss, loss
