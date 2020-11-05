# -*- coding: utf8 -*-

from yarll.actionselection.action_selection import ActionSelection
import numpy as np

class ContinuousActionSelection(ActionSelection):
    """Selection of an action in a continuous action space"""

    # def select_action(self, mu, sigma):
    #     return np.random.normal(mu, sigma)

    # Without sigma
    def select_action(self, mu):
        """
        Select a single action.

        Args:
            self: (todo): write your description
            mu: (todo): write your description
        """
        return np.random.normal(mu)
