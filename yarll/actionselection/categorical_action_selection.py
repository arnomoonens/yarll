# -*- coding: utf8 -*-

import numpy as np
from yarll.actionselection.action_selection import ActionSelection

class CategoricalActionSelection(ActionSelection):
    pass

class ProbabilisticCategoricalActionSelection(CategoricalActionSelection):
    """Sample from categorical distribution, specified by a vector of class probabilities"""

    @staticmethod
    def select_action(probabilities):
        """
        Select a single action.

        Args:
            probabilities: (todo): write your description
        """
        return np.random.choice(len(probabilities), p=probabilities)

class MaxCategoricalActionSelection(CategoricalActionSelection):
    """Choose the action with the highest probability."""

    @staticmethod
    def select_action(probabilities):
        """
        Return the action of the given probabilities.

        Args:
            probabilities: (todo): write your description
        """
        return np.argmax(probabilities)
