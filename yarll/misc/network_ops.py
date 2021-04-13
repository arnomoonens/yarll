# -*- coding: utf8 -*-

from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
import numpy as np

# Keras layer that converts a (None, n_units) Tensor to a (None, 1, n_units) Tensor
# Where the 1 is the sequence length: 1 because only 1 step at a time
flatten_to_rnn = Lambda(lambda x: tf.expand_dims(x, [1]))

class CategoricalProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class MultiCategoricalProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        return tf.cast(tf.stack([tf.random.categorical(l, 1) for l in logits], axis=-1), tf.int32)

class NormalDistrLayer(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super().__init__()
        self.n_outputs = n_outputs
        self.mean = Dense(n_outputs)
        self.log_std = None # instantiated in build phase

    def build(self, _):
        self.log_std = self.add_weight("log_std",
                                       shape=(self.n_outputs,),
                                       initializer=tf.initializers.zeros)

    def call(self, inp):
        mean = self.mean(inp)
        return mean + tf.exp(self.log_std) * tf.random.normal(tf.shape(mean), dtype=mean.dtype), mean

    def entropy(self):
        return tf.reduce_sum(self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

def normal_dist_log_prob(actions_taken, mean, log_std):
    std = tf.exp(log_std)
    neglogprob = 0.5 * tf.reduce_sum(tf.square((actions_taken - mean) / std), axis=-1) \
        + 0.5 * tf.math.log(2.0 * np.pi) * tf.cast(tf.shape(actions_taken)[-1], tf.float32) \
        + tf.reduce_sum(log_std, axis=-1)
    return -neglogprob

def categorical_dist_entropy(logits):
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
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def fan_in_initializer(fan_in_size: int):
    def _initializer(shape, dtype=None, partition_info=None):
        bound = 1 / np.sqrt(fan_in_size)
        out = np.random.uniform(-bound, bound, shape)
        return tf.constant(out.astype(np.float32))
    return _initializer


def linear_fan_in(x, output_size: int):
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
    return tf.group(*[v1.assign(v2) for v1, v2 in zip(target_vars, source_vars)], name="sync_net")

def kl_divergence(logits1, logits2):
    a0 = logits1 - tf.reduce_max(logits1, axis=-1, keepdims=True)
    a1 = logits2 - tf.reduce_max(logits2, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

# Custom Kaiming uniform initializer, the same as the one of the PyTorch linear layer.
CustomKaimingUniformKernelInitializer = partial(tf.keras.initializers.VarianceScaling,
                                                scale=(1 / 3),
                                                distribution="uniform")

class CustomKaimingUniformBiasInitializer(tf.keras.initializers.Initializer):
    def __init__(self, fan_in: int):
        self.fan_in = fan_in

    def __call__(self, shape, dtype=None):
        dtype = dtype if dtype is not None else tf.float32
        bias_bound = 1 / tf.sqrt(tf.cast(self.fan_in, dtype))
        return tf.random.uniform(shape, -bias_bound, bias_bound, dtype)

    def get_config(self):
        return dict(fan_in=self.fan_in)
