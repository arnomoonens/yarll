#!/usr/bin/env python

import gym
from Policies.EGreedy import EGreedy
# from Learners.Sarsa import Sarsa
from Traces.EligibilityTraces import EligibilityTraces
from FunctionApproximation.TileCoding import TileCoding

env = gym.make('MountainCar-v0')

m = 10  # Number of tilings
n_x_tiles = 9
n_y_tiles = 9

parameters = {
    'lambda': 0.9,
    'epsilon': 0,  # fully greedy in this case
    'alpha': 0.05 * (0.1 / m),
    'gamma': 0.9  # TODO: HOW TO SET THIS PARAMETER?
}

O = env.observation_space
x_low, y_low = O.low
x_high, y_high = O.high

A = env.action_space

n_episodes = 100
policy = EGreedy(0.1)
function_approximation = TileCoding(x_low, x_high, y_low, y_high, m, n_x_tiles, n_y_tiles, env.action_space.n)
for i in range(n_episodes):
    print('Episode {}'.format(i))
    traces = EligibilityTraces(function_approximation.features_shape, parameters['gamma'], parameters['lambda'])
    state, action = env.reset(), 0
    state -= O.low
    done = False
    while not(done):
        old_state, old_action = state, action
        traces.adapt_traces(function_approximation.present_features(state, action))
        env.render()
        state, reward, done, info = env.step(action)
        if done:
            print("Reached the top!")
        delta = reward - function_approximation.summed_thetas(old_state, old_action)
        Qs = [function_approximation.summed_thetas(state, action) for action in range(A.n)]
        action, Q = policy.select_action(Qs)
        delta += parameters['gamma'] * Q
        function_approximation.set_thetas(parameters['alpha'] * delta * traces.traces)
        traces.decay()
