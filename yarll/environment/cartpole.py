# -*- coding: utf8 -*-
import gym
from yarll.environment.wrappers import DescriptionWrapper

class CartPole(DescriptionWrapper):
    """Cartpole-v0 environment wrapper."""

    changeable_parameters: list = [
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
            "high": 5.0
        },
        {
            "name": "masscart",
            "type": "range",
            "low": 0.01,
            "high": 5.0
        },
    ]

    def __init__(self, length=None, masspole=None, masscart=None, ** kwargs):
        self.length = length
        self.masspole = masspole
        self.masscart = masscart
        super(CartPole, self).__init__(gym.make("OldCartPole-v0"), **kwargs)
        self.change_parameters(length=length, masspole=masspole, masscart=masscart)
        self.metadata["parameters"].update(self.changeable_parameters_values())

    def change_parameters(self, length=None, masspole=None, masscart=None):
        """Change a CartPole environment using a different length and/or masspole."""
        if length is not None:
            self.unwrapped.length = length
        if masspole is not None:
            self.unwrapped.masspole = masspole
        if masscart is not None:
            self.unwrapped.masscart = masscart
        self.unwrapped.total_mass = self.unwrapped.masspole + self.unwrapped.masscart  # Recalculate
        self.unwrapped.polemass_length = self.unwrapped.masspole * self.unwrapped.length  # Recalculate
