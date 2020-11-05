# -*- coding: utf8 -*-

class Policy:
    """Decides which action to take."""

    def select_action(self, values):
        """
        Selects an action.

        Args:
            self: (todo): write your description
            values: (str): write your description
        """
        raise NotImplementedError()

    def __call__(self, values):
        """
        Perform the given action.

        Args:
            self: (todo): write your description
            values: (array): write your description
        """
        return self.select_action(values)
