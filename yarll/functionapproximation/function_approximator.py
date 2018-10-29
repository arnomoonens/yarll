# -*- coding: utf8 -*-

class FunctionApproximator(object):
    """Map states and actions using a function."""
    def __init__(self, n_actions: int) -> None:
        super(FunctionApproximator, self).__init__()
        self.n_actions = n_actions
        self.thetas = []
        self.features_shape = (0)

    def summed_thetas(self, state, action):
        raise NotImplementedError()

    def set_thetas(self, addition):
        self.thetas += addition
