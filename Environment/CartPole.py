#!/usr/bin/env python
# -*- coding: utf8 -*-

# Make CartPole environments

import numpy as np

from Environment.Environment import Environment

class CartPole(Environment):
    """Cartpole-v0 environment wrapper."""
    def __init__(self, length=None, masspole=None):
        super(CartPole, self).__init__("CartPole-v0")
        self.length = length
        self.masspole = masspole
        self.change_parameters(length, masspole)

    def change_parameters(self, length=None, masspole=None):
        """Change a CartPole environment using a different length and/or masspole."""
        if masspole or length:
            if masspole:
                self.env.masspole = masspole
            if length:
                self.env.length = length
            self.env.total_mass = (self.env.masspole + self.env.masscart)  # Recalculate
            self.env.polemass_length = (self.env.masspole * self.env.length)  # Recalculate

    def to_dict(self):
        """Extract most important parameters of the environment."""
        return {
            "name": self.name,
            "length": self.env.length,
            "masspole": self.env.masspole
        }

def make_predef_CartPole_envs():
    """Make (3) predefined variations of the same game."""
    envs = []
    envs.append(CartPole())  # First one has the standard behaviour
    envs.append(CartPole(0.25, 0.5))  # 5 times longer, 5 times heavier
    envs.append(CartPole(0.025, 0.05))
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
        envs.append(CartPole(length, masspole))
    return envs
