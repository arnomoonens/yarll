# -*- coding: utf8 -*-

import numpy as np
from yarll.policies.policy import Policy

class EGreedy(Policy):
    """Take the best action with a probability and a random one otherwise."""
    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def select_action(self, values):
        idx = None
        if (np.random.rand() < self.epsilon):  # With a probability of epsilon...
            idx = np.random.randint(len(values))  # Choose a random action...
        else:
            # Else return a random action out of those with the highest value
            idx = np.random.choice(np.where(values == np.max(values))[0])
        return idx, values[idx]  # Return the action and the associated value
