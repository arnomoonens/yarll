# -*- coding: utf8 -*-

from collections import namedtuple

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "value", "features", "terminal"])

class ExperiencesMemory(object):
    """Experience gathered from an environment."""

    def __init__(self):
        """
        Initialize gradient.

        Args:
            self: (todo): write your description
        """
        super(ExperiencesMemory, self).__init__()
        self.experiences = []
        self.steps = 0

    def add(self, state, action, reward, value=None, features=None, terminal=False, next_state=None):
        """Add a single transition to the trajectory."""
        exp = Experience(state, action, reward, next_state, value, features, terminal)
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
        """
        List of all states.

        Args:
            self: (todo): write your description
        """
        return [exp.state for exp in self.experiences]

    @property
    def actions(self):
        """
        The list of actions associated actions.

        Args:
            self: (todo): write your description
        """
        return [exp.action for exp in self.experiences]

    @property
    def rewards(self):
        """
        Returns the reward reward.

        Args:
            self: (todo): write your description
        """
        return [exp.reward for exp in self.experiences]

    @property
    def values(self):
        """
        Return a list of all the values.

        Args:
            self: (todo): write your description
        """
        return [exp.value for exp in self.experiences]

    @property
    def features(self):
        """
        A list of experiment features.

        Args:
            self: (todo): write your description
        """
        return[exp.features for exp in self.experiences]

    @property
    def terminal(self):
        """Last experience is terminal."""
        return self.experiences[-1].terminal

    @property
    def terminals(self):
        """
        List of all terminal symbols.

        Args:
            self: (todo): write your description
        """
        return [exp.terminal for exp in self.experiences]

    @property
    def next_states(self):
        """
        Returns the next state.

        Args:
            self: (todo): write your description
        """
        return [exp.next_state for exp in self.experiences]

    def __getitem__(self, i):
        """
        Return the item at the given index.

        Args:
            self: (todo): write your description
            i: (todo): write your description
        """
        return self.experiences[i]
