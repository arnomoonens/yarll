# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
import numpy as np

# Keras layer that converts a (None, n_units) Tensor to a (None, 1, n_units) Tensor
# Where the 1 is the sequence length: 1 because only 1 step at a time
flatten_to_rnn = Lambda(lambda x: tf.expand_dims(x, [1]))

class CategoricalProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        """
        Evalzeorical ).

        Args:
            self: (todo): write your description
            logits: (todo): write your description
        """
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class MultiCategoricalProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        """
        Return the logits.

        Args:
            self: (todo): write your description
            logits: (todo): write your description
        """
        return tf.cast(tf.stack([tf.random.categorical(l, 1) for l in logits], axis=-1), tf.int32)

class NormalDistrLayer(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            n_outputs: (str): write your description
        """
        super(NormalDistrLayer, self).__init__()
        self.n_outputs = n_outputs
        self.mean = Dense(n_outputs)
        self.log_std = None # instantiated in build phase

    def build(self, _):
        """
        Build the graph.

        Args:
            self: (todo): write your description
            _: (todo): write your description
        """
        self.log_std = self.add_weight("log_std",
                                       shape=(self.n_outputs,),
                                       initializer=tf.initializers.zeros)

    def call(self, inp):
        """
        Return a new tf.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        mean = self.mean(inp)
        return mean + tf.exp(self.log_std) * tf.random.normal(tf.shape(mean), dtype=mean.dtype), mean

    def entropy(self):
        """
        Compute the entropy of - entropy.

        Args:
            self: (todo): write your description
        """
        return tf.reduce_sum(self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

def normal_dist_log_prob(actions_taken, mean, log_std):
    """
    Compute the log - normal distribution.

    Args:
        actions_taken: (str): write your description
        mean: (todo): write your description
        log_std: (todo): write your description
    """
    std = tf.exp(log_std)
    neglogprob = 0.5 * tf.reduce_sum(tf.square((actions_taken - mean) / std), axis=-1) \
        + 0.5 * tf.math.log(2.0 * np.pi) * tf.cast(tf.shape(actions_taken)[-1], tf.float32) \
        + tf.reduce_sum(log_std, axis=-1)
    return -neglogprob

def categorical_dist_entropy(logits):
    """
    Categorical entropy of the categorical distribution.

    Args:
        logits: (todo): write your description
    """
    # Source: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/distributions.py#L313
    a_0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    exp_a_0 = tf.exp(a_0)
    z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
    p_0 = exp_a_0 / z_0
    return tf.reduce_sum(p_0 * (tf.math.log(z_0) - a_0), axis=-1)

def bernoulli_dist_entropy(logits):
    """Entropy of Bernoulli distributions
    Source: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/distributions.py#L453
    Arguments:
        logits {tf.Tensor} -- Logits of the bernoulli distributions
    """
    probs = tf.sigmoid(logits)
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                 labels=probs), axis=-1)

def normalized_columns_initializer(std: float = 1.0):
    """
    Normalized normalizer.

    Args:
        std: (todo): write your description
    """
    def _initializer(shape, dtype=None, partition_info=None):
        """
        Initialize an initial tensor.

        Args:
            shape: (int): write your description
            dtype: (str): write your description
            partition_info: (todo): write your description
        """
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def fan_in_initializer(fan_in_size: int):
    """
    Returns an initial tensor.

    Args:
        fan_in_size: (int): write your description
    """
    def _initializer(shape, dtype=None, partition_info=None):
        """
        Returns a random initializer.

        Args:
            shape: (int): write your description
            dtype: (str): write your description
            partition_info: (todo): write your description
        """
        bound = 1 / np.sqrt(fan_in_size)
        out = np.random.uniform(-bound, bound, shape)
        return tf.constant(out.astype(np.float32))
    return _initializer


def linear_fan_in(x, output_size: int):
    """
    Initialize a tensor.

    Args:
        x: (array): write your description
        output_size: (int): write your description
    """
    input_size = x.shape[1].value
    w = tf.get_variable("w", [input_size, output_size], initializer=fan_in_initializer(input_size))
    b = tf.get_variable("b", [output_size], initializer=fan_in_initializer(input_size))
    x = tf.nn.xw_plus_b(x, w, b)
    return x, [w, b]

def create_accumulative_gradients_op(net_vars, identifier: int = 0):
    """Make an operation to create accumulative gradients"""
    accum_grads = []
    with tf.name_scope(name="create_accum_{}".format(identifier), values=net_vars):
        for var in net_vars:
            zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
            name = var.name.replace(":", "_") + "_accum_grad"
            accum_grad = tf.Variable(zero, name=name, trainable=False)
            accum_grads.append(accum_grad)
        return accum_grads

def add_accumulative_gradients_op(net_vars, accum_grads, loss, identifier: int = 0):
    """Make an operation to add a gradient to the total"""
    accum_grad_ops = []
    with tf.name_scope(name="grad_ops_{}".format(identifier), values=net_vars):
        grads = tf.gradients(loss, net_vars, gate_gradients=False,
                             aggregation_method=None,
                             colocate_gradients_with_ops=False)
    with tf.name_scope(name="accum_ops_{}".format(identifier), values=[]):
        for (grad, var, accum_grad) in zip(grads, net_vars, accum_grads):
            name = var.name.replace(":", "_") + "_accum_grad_ops"
            accum_ops = tf.assign_add(accum_grad, grad, name=name)
            accum_grad_ops.append(accum_ops)
        return tf.group(*accum_grad_ops, name="accum_group_{}".format(identifier))

def reset_accumulative_gradients_op(net_vars, accum_grads, identifier: int = 0):
    """Make an operation to reset the accumulation to zero."""
    reset_grad_ops = []
    with tf.name_scope(name="reset_grad_ops_{}".format(identifier), values=net_vars):
        for (var, accum_grad) in zip(net_vars, accum_grads):
            zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
            name = var.name.replace(":", "_") + "_reset_grad_ops"
            reset_ops = tf.assign(accum_grad, zero, name=name)
            reset_grad_ops.append(reset_ops)
        return tf.group(*reset_grad_ops, name="reset_accum_group_{}".format(identifier))

def create_sync_net_op(source_vars, target_vars):
    """
    Creates a copy of the net_net_net_net.

    Args:
        source_vars: (str): write your description
        target_vars: (todo): write your description
    """
    return tf.group(*[v1.assign(v2) for v1, v2 in zip(target_vars, source_vars)], name="sync_net")

def batch_norm_layer(x, training_phase, scope_bn: str, activation=None):
    """
    Batch normalization.

    Args:
        x: (todo): write your description
        training_phase: (todo): write your description
        scope_bn: (todo): write your description
        activation: (todo): write your description
    """
    return tf.cond(
        training_phase,
        lambda: tf.contrib.layers.batch_norm(
            x,
            activation_fn=activation,
            center=True,
            scale=True,
            updates_collections=None,
            is_training=True,
            reuse=None,
            scope=scope_bn,
            decay=0.9,
            epsilon=1e-5),
        lambda: tf.contrib.layers.batch_norm(
            x,
            activation_fn=activation,
            center=True,
            scale=True,
            updates_collections=None,
            is_training=False,
            reuse=True,
            scope=scope_bn,
            decay=0.9,
            epsilon=1e-5))


def kl_divergence(logits1, logits2):
    """
    R kl divergence divergence.

    Args:
        logits1: (tuple): write your description
        logits2: (tuple): write your description
    """
    a0 = logits1 - tf.reduce_max(logits1, axis=-1, keepdims=True)
    a1 = logits2 - tf.reduce_max(logits2, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)
