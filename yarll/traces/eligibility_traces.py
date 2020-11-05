# -*- coding: utf8 -*-

import numpy as np

class EligibilityTraces(object):
    """Eligibility traces"""
    def __init__(self, features_shape, gamma: float, Lambda: float) -> None:
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            features_shape: (todo): write your description
            gamma: (float): write your description
            Lambda: (float): write your description
        """
        super(EligibilityTraces, self).__init__()
        self.features_shape = features_shape
        self.traces = np.zeros(self.features_shape)
        self.gamma: float = gamma
        self.Lambda: float = Lambda

    def replacing_traces(self, present_features):
        """
        Set the list of a list of traces.

        Args:
            self: (todo): write your description
            present_features: (str): write your description
        """
        self.traces = present_features  # replacing traces: set them to 1

    def decay(self):
        """Reduce the traces by taking a fraction of them, determined by gamma and lambda"""
        self.traces *= self.gamma * self.Lambda
