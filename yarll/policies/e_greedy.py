# -*- coding: utf8 -*-

from yarll.policies.policy import Policy

import numpy as np

class EGreedy(Policy):
    """Take the best action with a probability and a random one otherwise."""
    def __init__(self, epsilon):
        super(EGreedy, self).__init__()
        self.epsilon = epsilon

    def select_action(self, values):
        idx = None
        if (np.random.rand() < self.epsilon):  # With a probability of epsilon...
            idx = np.random.randint(len(values))  # Choose a random action...
        else:
            idx = np.argmax(values)  # Else return the action with the highest value
        return idx, values[idx]  # Return the action and the associated value
