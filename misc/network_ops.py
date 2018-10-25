# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

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


def linear(x, size, name, initializer=None, bias_init=None):
    if bias_init is None:
        bias_init = tf.constant_initializer(0.0)
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=bias_init)
    return tf.matmul(x, w) + b

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    """
    2-dimensional convolutional layer.
    Source: https://github.com/openai/universe-starter-agent/blob/a3fdfba297c8c24d62d3c53978fb6fb26f80e76e/model.py
    """

    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def mu_sigma_layer(inputs, n_outputs: int):
    """
    Create a layer that makes a mu and sigma,
    e.g. to use in continuous action spaces.
    """

    mu = tf.contrib.layers.fully_connected(
        inputs=inputs,
        num_outputs=n_outputs,
        activation_fn=None,
        weights_initializer=normalized_columns_initializer(0.01),
        biases_initializer=tf.zeros_initializer(),
        scope="mu")
    mu = tf.squeeze(mu, name="mu")

    sigma = tf.contrib.layers.fully_connected(
        inputs=inputs,
        num_outputs=n_outputs,
        activation_fn=None,
        weights_initializer=normalized_columns_initializer(0.01),
        biases_initializer=tf.zeros_initializer(),
        scope="sigma")
    sigma = tf.squeeze(sigma)
    sigma = tf.nn.softplus(sigma) + 1e-5
    return mu, sigma

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
