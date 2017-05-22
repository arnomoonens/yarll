# -*- coding: utf8 -*-

class Agent(object):
    """Reinforcement learning agent"""
    def __init__(self, **usercfg):
        super(Agent, self).__init__()
        self.config = usercfg

    def learn(self):
        """Learn in the current environment."""
        raise NotImplementedError()
