import numpy as np

class EligibilityTraces(object):
    """docstring for EligibilityTraces"""
    def __init__(self, features_shape, gamma, Lambda):
        super(EligibilityTraces, self).__init__()
        self.features_shape = features_shape
        self.traces = np.zeros(self.features_shape)
        self.gamma = gamma
        self.Lambda = Lambda

    def adapt_traces(self, present_features):
        self.traces = present_features  # replacing traces: set them to 1

    def decay(self):
        self.traces *= self.gamma * self.Lambda
