# -*- coding: utf8 -*-

from yarll.environment.wrappers import DescriptionWrapper

class Environment(DescriptionWrapper):
    def __init__(self, env, **kwargs):
        """
        Initialize environment.

        Args:
            self: (todo): write your description
            env: (todo): write your description
        """
        super(Environment, self).__init__(env)
