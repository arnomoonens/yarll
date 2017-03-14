#!/usr/bin/env python
# -*- coding: utf8 -*-

# Make CartPole environments

import numpy as np
import gym

def make_CartPole_env(length=None, masspole=None):
    """Make a CartPole environment with possibly a different length and/or masspole."""
    env = gym.make("CartPole-v0")
    if masspole or length:
        if masspole:
            env.masspole = masspole
        if length:
            env.length = length
        env.total_mass = (env.masspole + env.masscart)  # Recalculate
        env.polemass_length = (env.masspole * env.length)  # Recalculate
    return env

def make_predef_CartPole_envs():
    """Make (3) predefined variations of the same game."""
    envs = []
    envs.append(make_CartPole_env())  # First one has the standard behaviour
    envs.append(make_CartPole_env(0.25, 0.5))  # 5 times longer, 5 times heavier
    envs.append(make_CartPole_env(0.025, 0.05))
    return envs

def make_random_CartPole_envs(n_envs):
    """Make n_envs random variations of the same game."""
    length_low = 0.01
    length_high = 5.0
    masspole_low = 0.01
    masspole_high = 1.0
    envs = []
    for i in range(n_envs):
        length = np.random.uniform(length_low, length_high)
        masspole = np.random.uniform(masspole_low, masspole_high)
        envs.append(make_CartPole_env(length, masspole))
    return envs
