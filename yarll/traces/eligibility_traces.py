# -*- coding: utf8 -*-

import numpy as np

class EligibilityTraces(object):
    """Eligibility traces"""
    def __init__(self, features_shape, gamma: float, Lambda: float) -> None:
        super(EligibilityTraces, self).__init__()
        self.features_shape = features_shape
        self.traces = np.zeros(self.features_shape)
        self.gamma: float = gamma
        self.Lambda: float = Lambda

    def replacing_traces(self, present_features):
        self.traces = present_features  # replacing traces: set them to 1

    def decay(self):
        """Reduce the traces by taking a fraction of them, determined by gamma and lambda"""
        self.traces *= self.gamma * self.Lambda
