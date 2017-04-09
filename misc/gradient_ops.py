#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf

def create_accumulative_gradients_op(net_vars, identifier=0):
    """Make an operation to create accumulative gradients"""
    accum_grads = []
    with tf.name_scope(name="create_accum_{}".format(identifier), values=net_vars):
        for var in net_vars:
            zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
            name = var.name.replace(":", "_") + "_accum_grad"
            accum_grad = tf.Variable(zero, name=name, trainable=False)
            accum_grads.append(accum_grad)
        return accum_grads

def add_accumulative_gradients_op(net_vars, accum_grads, loss, identifier=0):
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

def reset_accumulative_gradients_op(net_vars, accum_grads, identifier=0):
    """Make an operation to reset the accumulation to zero."""
    reset_grad_ops = []
    with tf.name_scope(name="reset_grad_ops_{}".format(identifier), values=net_vars):
        for (var, accum_grad) in zip(net_vars, accum_grads):
            zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
            name = var.name.replace(":", "_") + "_reset_grad_ops"
            reset_ops = tf.assign(accum_grad, zero, name=name)
            reset_grad_ops.append(reset_ops)
        return tf.group(*reset_grad_ops, name="reset_accum_group_{}".format(identifier))

def sync_gradients_op(source_net, local_net_vars, identifier=0):
    """Make an operation to sync the gradients"""
    sync_ops = []
    with tf.name_scope(name="sync_ops_{}".format(identifier), values=[]):
        for (target_var, source_var) in zip(local_net_vars, source_net.vars):
            ops = tf.assign(target_var, source_var)
            sync_ops.append(ops)
        return tf.group(*sync_ops, name="sync_group_{}".format(identifier))
