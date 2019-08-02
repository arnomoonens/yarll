# -*- coding: utf8 -*-

class Policy:
    """Decides which action to take."""

    def select_action(self, values):
        raise NotImplementedError()

    def __call__(self, values):
        return self.select_action(values)
