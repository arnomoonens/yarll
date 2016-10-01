#!/usr/bin/env python

import gym
import sys
from Policies.EGreedy import EGreedy
from Learners.Sarsa import Sarsa
from Traces.EligibilityTraces import EligibilityTraces
from FunctionApproximation.TileCoding import TileCoding

env = gym.make('MountainCar-v0')

m = 10  # Number of tilings
n_x_tiles = 9
n_y_tiles = 9

Lambda = 0.9
epsilon = 0  # fully greedy in this case
alpha = 0.05 * (0.5 / m)
gamma = 1

steps_per_episode = 200  # Maximum number of allowed steps per episode, as determined (for this environment) by the gym library

O = env.observation_space
x_low, y_low = O.low
x_high, y_high = O.high

A = env.action_space

# def draw_3d(tile_starts):
#     states = []
#     for i in range(n_x_tiles):
#         for j in range(n_y_tiles):
#             states.append([i, j])
#     states = np.array(states)

def main(n_episodes, monitor_directory):
    policy = EGreedy(epsilon)
    function_approximation = TileCoding(x_low, x_high, y_low, y_high, m, n_x_tiles, n_y_tiles, env.action_space.n)
    env.monitor.start(monitor_directory)
    print("Going to run {} episodes".format(n_episodes))
    for i in range(n_episodes):
        # print('Episode {}'.format(i))
        traces = EligibilityTraces(function_approximation.features_shape, gamma, Lambda)
        state, action = env.reset(), 0
        sarsa = Sarsa(gamma, alpha, policy, traces, function_approximation, range(A.n), state, action)
        done = False  # Done says if the goal is reached or the maximum number of allowed steps for the episode is reached (determined by the gym library itself)
        iteration = 0
        while not(done):
            iteration += 1
            # env.render()
            state, reward, done, info = env.step(action)
            if done and iteration < steps_per_episode:
                print("Episode {}: Less than {} steps were needed: {}".format(i, steps_per_episode, iteration))
            action = sarsa.step(state, reward)
    env.monitor.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("""Please provide the following arguments:
            <number_of_episodes>
            <monitor_directory>""")
    else:
        main(int(sys.argv[1]), sys.argv[2])
