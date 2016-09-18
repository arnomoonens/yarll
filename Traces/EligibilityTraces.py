import numpy as np

class EligibilityTraces(object):
    """docstring for EligibilityTraces"""
    def __init__(self, features_shape, gamma, lambda_parameter):
        super(EligibilityTraces, self).__init__()
        self.features_shape = features_shape
        self.traces = np.zeros(self.features_shape)
        self.gamma = gamma
        self.lambda_parameter = lambda_parameter

    def adapt_traces(self, present_features):
        self.traces = present_features

    def decay(self):
        self.traces *= self.gamma * self.lambda_parameter
