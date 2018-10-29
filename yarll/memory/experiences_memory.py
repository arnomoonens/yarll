# -*- coding: utf8 -*-

from collections import namedtuple

Experience = namedtuple("Experience", ["state", "action", "reward", "value", "features", "terminal"])

class ExperiencesMemory(object):
    """Experience gathered from an environment."""

    def __init__(self):
        super(ExperiencesMemory, self).__init__()
        self.experiences = []
        self.steps = 0

    def add(self, state, action, reward, value=None, features=None, terminal=False):
        """Add a single transition to the trajectory."""
        exp = Experience(state, action, reward, value, features, terminal)
        self.experiences.append(exp)
        self.steps += 1

    def extend(self, other):
        """
        Extend a trajectory with another one.
        """
        self.experiences.extend(other.experiences)
        self.steps += other.steps

    @property
    def states(self):
        return [exp.state for exp in self.experiences]

    @property
    def actions(self):
        return [exp.action for exp in self.experiences]

    @property
    def rewards(self):
        return [exp.reward for exp in self.experiences]

    @property
    def values(self):
        return [exp.value for exp in self.experiences]

    @property
    def features(self):
        return[exp.features for exp in self.experiences]

    @property
    def terminal(self):
        """Last experience is terminal."""
        return self.experiences[-1].terminal

    @property
    def terminals(self):
        return [exp.terminal for exp in self.experiences]
