#!/usr/bin/env python
# -*- coding: utf8 -*-

import logging
import numpy as np

def discount_rewards(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x), 'float64')
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def print_iteration_stats(iteration, episode_rewards, episode_lengths):
    """Print statistics about rewards and episode lengths of the current iteration"""
    logging.info("-----------------")
    logging.info("Iteration: \t %i" % iteration)
    logging.info("NumTrajs: \t %i" % len(episode_rewards))
    logging.info("NumTimesteps: \t %i" % np.sum(episode_lengths))
    logging.info("MaxRew: \t %s" % episode_rewards.max())
    logging.info("MeanRew: \t %s +- %s" % (episode_rewards.mean(), episode_rewards.std() / np.sqrt(len(episode_rewards))))
    logging.info("MeanLen: \t %s +- %s" % (episode_lengths.mean(), episode_lengths.std() / np.sqrt(len(episode_lengths))))
    logging.info("-----------------")
    return
