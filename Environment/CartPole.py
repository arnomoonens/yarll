#!/usr/bin/env python
# -*- coding: utf8 -*-

from Environment import Environment

class CartPole(Environment):
    """Cartpole-v0 environment wrapper."""

    changeable_parameters = [
        {
            "name": "length",
            "type": "range",
            "low": 0.01,
            "high": 5.0
        },
        {
            "name": "masspole",
            "type": "range",
            "low": 0.01,
            "high": 1.0
        },
    ]

    def __init__(self, length=None, masspole=None, **kwargs):
        super(CartPole, self).__init__("CartPole-v0", **kwargs)
        self.length = length
        self.masspole = masspole
        self.change_parameters(length, masspole)

    def change_parameters(self, length=None, masspole=None):
        """Change a CartPole environment using a different length and/or masspole."""
        if masspole or length:
            if masspole:
                self.env.env.masspole = masspole
            if length:
                self.env.env.length = length
            self.env.env.total_mass = (self.env.env.masspole + self.env.env.masscart)  # Recalculate
            self.env.env.polemass_length = (self.env.env.masspole * self.env.env.length)  # Recalculate
