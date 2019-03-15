# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

class NormalDistrLayer(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(NormalDistrLayer, self).__init__()
        self.n_outputs = n_outputs
        self.mean = Dense(n_outputs)

    def build(self, input_shape):
        self.log_std = self.add_variable("log_std",
                                         shape=(self.n_outputs,),
                                         initializer=tf.initializers.zeros)

    def call(self, input):
        mean = self.mean(input)
        return mean + tf.exp(self.log_std) * tf.random.normal((self.n_outputs,)), mean

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

def batch_norm_layer(x, training_phase, scope_bn: str, activation=None):
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
    a0 = logits1 - tf.reduce_max(logits1, axis=-1, keepdims=True)
    a1 = logits2 - tf.reduce_max(logits2, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
