# -*- coding: utf8 -*-

import gym
from yarll.environment.wrappers import DescriptionWrapper

class Acrobot(DescriptionWrapper):
    """Acrobot-v1 environment wrapper."""

    changeable_parameters = [
        {
            "name": "link_length_1",
            "type": "range",
            "low": 0.2,
            "high": 2.0
        },
        {
            "name": "link_length_2",
            "type": "range",
            "low": 0.2,
            "high": 2.0
        },
        {
            "name": "link_mass_1",
            "type": "range",
            "low": 0.2,
            "high": 2.0
        },
        {
            "name": "link_mass_2",
            "type": "range",
            "low": 0.2,
            "high": 2.0
        },
    ]

    def __init__(self, link_length_1=None, link_length_2=None, link_mass_1=None, link_mass_2=None, **kwargs):
        super(Acrobot, self).__init__(gym.make("OldAcrobot-v1"), **kwargs)
        self.link_length_1 = link_length_1
        self.link_length_2 = link_length_2
        self.link_mass_1 = link_mass_1
        self.link_mass_2 = link_mass_2
        self.change_parameters(
            link_length_1=link_length_1,
            link_length_2=link_length_2,
            link_mass_1=link_mass_1,
            link_mass_2=link_mass_2)

    def change_parameters(self, link_length_1=None, link_length_2=None, link_mass_1=None, link_mass_2=None):
        """Change an Acrobot environment using a different length and/or masspole."""
        if link_length_1 is not None:
            self.unwrapped.LINK_LENGTH_1 = link_length_1
        if link_length_2 is not None:
            self.unwrapped.LINK_LENGTH_2 = link_length_2
        if link_mass_1 is not None:
            self.unwrapped.LINK_MASS_1 = link_mass_1
        if link_mass_2 is not None:
            self.unwrapped.LINK_MASS_2 = link_mass_2
