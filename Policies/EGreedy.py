from Policies.Policy import Policy

import numpy as np

class EGreedy(Policy):
    """Take the best action with a probability and a random one otherwise."""
    def __init__(self, epsilon):
        super(EGreedy, self).__init__()
        self.epsilon = epsilon

    def select_action(self, values):
        idx = None
        if (np.random.rand() < self.epsilon):
            idx = np.random.randint(len(values))
        else:
            idx = np.argmax(values)
        return idx, values[idx]
