# -*- coding: utf8 -*-

from yarll.environment.wrappers import DescriptionWrapper

class Environment(DescriptionWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
