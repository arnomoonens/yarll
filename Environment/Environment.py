#!/usr/bin/env python
# -*- coding: utf8 -*-

from gym import make, Wrapper

class Environment(Wrapper):
    """Wrapper for a OpenAI Gym environment."""

    changeable_parameters = []

    def __init__(self, name, **args):
        super(Environment, self).__init__(make(name))
        self.name = name
        self.args = args

    def to_dict(self):
        """
        Extract the name and other important aspects of the environment.
        By default, these include the changeable parameters.
        """
        d = {"name": self.name}
        for p in self.changeable_parameters:
            d[p["name"]] = self.env.env.__getattribute__(p["name"])
        return d
