# -*- coding: utf8 -*-

import numpy as np
from actionselection.action_selection import ActionSelection

class CategoricalActionSelection(ActionSelection):
    pass

class ProbabilisticCategoricalActionSelection(CategoricalActionSelection):
    """Sample from categorical distribution, specified by a vector of class probabilities"""

    @staticmethod
    def select_action(self, probabilities):
        return np.random.choice(len(probabilities), p=probabilities)

class MaxCategoricalActionSelection(CategoricalActionSelection):
    """Choose the action with the highest probability."""

    @staticmethod
    def select_action(self, probabilities):
        return np.argmax(probabilities)
