#!/usr/bin/env python

import gym
import sys
from Policies.EGreedy import EGreedy
# from Learners.Sarsa import Sarsa
from Traces.EligibilityTraces import EligibilityTraces
from FunctionApproximation.TileCoding import TileCoding

env = gym.make('MountainCar-v0')

m = 10  # Number of tilings
n_x_tiles = 9
n_y_tiles = 9

Lambda = 0.9
epsilon = 0  # fully greedy in this case
alpha = 0.05 * (0.1 / m)
gamma = 0.99

O = env.observation_space
x_low, y_low = O.low
x_high, y_high = O.high

A = env.action_space

def main(n_episodes, monitor_directory):
    policy = EGreedy(epsilon)
    function_approximation = TileCoding(x_low, x_high, y_low, y_high, m, n_x_tiles, n_y_tiles, env.action_space.n)
    env.monitor.start(monitor_directory)
    print("Going to run {} episodes".format(n_episodes))
    for i in range(n_episodes):
        print('Episode {}'.format(i))
        traces = EligibilityTraces(function_approximation.features_shape, gamma, Lambda)
        state, action = env.reset(), 0
        state -= O.low
        done = False
        iteration = 0
        while not(done):
            iteration += 1
            old_state, old_action = state, action
            traces.adapt_traces(function_approximation.present_features(old_state, old_action))
            # env.render()
            state, reward, done, info = env.step(action)
            if done:
                print("Done! Nr of iterations: {}".format(iteration))
            delta = reward - function_approximation.summed_thetas(old_state, old_action)
            Qs = [function_approximation.summed_thetas(state, action) for action in range(A.n)]
            action, Q = policy.select_action(Qs)
            delta += gamma * Q
            function_approximation.set_thetas(alpha * delta * traces.traces)
            traces.decay()

    env.monitor.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("""Please provide the following arguments:
            <number_of_episodes>
            <monitor_directory>""")
    else:
        main(int(sys.argv[1]), sys.argv[2])
