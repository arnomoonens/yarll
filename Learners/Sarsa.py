class Sarsa(object):
    """Sarsa learner"""
    def __init__(self, gamma, alpha, lambda_value, policy, traces, function_approximation):
        super(Sarsa, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_value = lambda_value
        self.policy = policy
        self.traces = traces
        self.function_approximation = function_approximation

    def learn(self, reward, thetas):
        delta = reward - thetas
        # get Qa
        Qa = self.policy.select_action()
        delta += self.gamma * Qa
        thetas = thetas + self.alpha * delta * self.traces
        self.traces.update(self.gamma, self.lambda_value)
