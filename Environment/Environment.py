#!/usr/bin/env python
# -*- coding: utf8 -*-

from gym import make, Wrapper

class Environment(Wrapper):
    """Wrapper for a OpenAI Gym environment."""
    def __init__(self, name):
        super(Environment, self).__init__(make(name))
        self.name = name

    def to_dict(self):
        return {"name": self.name}
