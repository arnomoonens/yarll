# -*- coding: utf8 -*-

from environment import Environment

class Acrobot(Environment):
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
        super(Acrobot, self).__init__("Acrobot-v1", **kwargs)
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
        """Change a Acrobot environment using a different length and/or masspole."""
        if link_length_1 is not None:
            self.env.env.LINK_LENGTH_1 = link_length_1
        if link_length_2 is not None:
            self.env.env.LINK_LENGTH_2 = link_length_2
        if link_mass_1 is not None:
            self.env.env.LINK_MASS_1 = link_mass_1
        if link_mass_2 is not None:
            self.env.env.LINK_MASS_2 = link_mass_2

    def to_dict(self):
        """
        Extract the name and other important aspects of the environment.
        By default, these include the changeable parameters.
        """
        d = {"name": self.name}
        for p in self.changeable_parameters:
            d[p["name"]] = self.env.env.__getattribute__(p["name"].upper())
        return d
