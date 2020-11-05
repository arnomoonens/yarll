# -*- coding: utf8 -*-

class FunctionApproximator(object):
    """Map states and actions using a function."""
    def __init__(self, n_actions: int) -> None:
        """
        Initialize actions.

        Args:
            self: (todo): write your description
            n_actions: (todo): write your description
        """
        super(FunctionApproximator, self).__init__()
        self.n_actions = n_actions
        self.thetas = []
        self.features_shape = (0)

    def summed_thetas(self, state, action):
        """
        Return the sum of the given action.

        Args:
            self: (todo): write your description
            state: (todo): write your description
            action: (str): write your description
        """
        raise NotImplementedError()

    def set_thetas(self, addition):
        """
        Set the tas astas

        Args:
            self: (todo): write your description
            addition: (todo): write your description
        """
        self.thetas += addition
