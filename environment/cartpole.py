# -*- coding: utf8 -*-

from environment import Environment

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
            "high": 5.0
        },
        {
            "name": "masscart",
            "type": "range",
            "low": 0.01,
            "high": 5.0
        },
    ]

    def __init__(self, length=None, masspole=None, masscart=None, **kwargs):
        super(CartPole, self).__init__("CartPole-v0", **kwargs)
        self.length = length
        self.masspole = masspole
        self.masscart = masscart
        self.change_parameters(length=length, masspole=masspole, masscart=masscart)

    def change_parameters(self, length=None, masspole=None, masscart=None):
        """Change a CartPole environment using a different length and/or masspole."""
        if length is not None:
            self.env.env.length = length
        if masspole is not None:
            self.env.env.masspole = masspole
        if masscart is not None:
            self.env.env.masscart = masscart
        self.env.env.total_mass = self.env.env.masspole + self.env.env.masscart  # Recalculate
        self.env.env.polemass_length = self.env.env.masspole * self.env.env.length  # Recalculate
