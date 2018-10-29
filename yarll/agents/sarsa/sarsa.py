# -*- coding: utf8 -*-

import numpy as np

class Sarsa(object):
    """Sarsa learner for function approximation"""
    def __init__(self, gamma: float, alpha: float, policy, traces, function_approximation, actions, start_state, start_action) -> None:
        super(Sarsa, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.policy = policy
        self.traces = traces
        self.function_approximation = function_approximation
        self.actions = actions
        self.old_state = start_state
        self.old_action = start_action

    def step(self, state: np.ndarray, reward: float):
        """Do one step of updating traces, function approximation and selecting an action using a policy"""
        self.traces.replacing_traces(self.function_approximation.present_features(self.old_state, self.old_action))
        delta = reward - self.function_approximation.summed_thetas(self.old_state, self.old_action)
        Qs = [self.function_approximation.summed_thetas(state, action) for action in self.actions]
        action, Q = self.policy.select_action(Qs)
        delta += self.gamma * Q
        self.function_approximation.set_thetas(self.alpha * delta * self.traces.traces)
        self.traces.decay()
        self.old_state = state
        self.old_action = action
        return action

    def reset(self, policy, traces, function_approximation, start_state, start_action):
        """Reset the policy, traces, function approximation, start action and/or start state"""
        self.policy = policy
        self.traces = traces
        self.function_approximation = function_approximation
        self.old_state = start_state
        self.old_action = start_action
