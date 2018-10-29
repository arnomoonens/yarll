# -*- coding: utf8 -*-

class Agent(object):
    """Reinforcement learning agent"""
    def __init__(self, **usercfg):
        super(Agent, self).__init__()
        self.config = usercfg
        # Only used (and overwritten) by agents that use an RNN
        self.initial_features = None

    def learn(self):
        """Learn in the current environment."""
        raise NotImplementedError()

    def get_env_action(self, action):
        return action

    def new_trajectory(self):
        """
        Notification by the environment runner that
        a new trajectory will be started
        """

        pass
