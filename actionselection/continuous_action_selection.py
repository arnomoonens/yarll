# -*- coding: utf8 -*-

from actionselection.action_selection import ActionSelection
import numpy as np

class ContinuousActionSelection(ActionSelection):
    """Selection of an action in a continuous action space"""
    def __init__(self):
        super(ContinuousActionSelection, self).__init__()

    # def select_action(self, mu, sigma):
    #     return np.random.normal(mu, sigma)

    # Without sigma
    def select_action(self, mu):
        return np.random.normal(mu)
