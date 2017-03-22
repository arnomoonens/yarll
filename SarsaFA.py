#!/usr/bin/env python

import sys
import os
import argparse

from gym import wrappers

from Policies.EGreedy import EGreedy
from Learners.Sarsa import Sarsa
from Traces.EligibilityTraces import EligibilityTraces
from FunctionApproximation.TileCoding import TileCoding
from Environment import Environment

# def draw_3d(tile_starts):
#     states = []
#     for i in range(n_x_tiles):
#         for j in range(n_y_tiles):
#             states.append([i, j])
#     states = np.array(states)

class SarsaFALearner(object):
    """Learner using Sarsa and function approximation"""
    def __init__(self, env):
        super(SarsaFALearner, self).__init__()
        self.env = env
        m = 10  # Number of tilings
        self.config = dict(
            m=m,
            n_x_tiles=9,
            n_y_tiles=9,
            Lambda=0.9,
            epsilon=0,  # fully greedy in this case
            alpha=(0.05 * (0.5 / m)),
            gamma=1,
            steps_per_episode=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps")  # Maximum number of allowed steps per episode, as determined (for this environment) by the gym library
        )
        O = env.observation_space
        self.x_low, self.y_low = O.low
        self.x_high, self.y_high = O.high

        self.nA = env.action_space.n
        self.policy = EGreedy(self.config["epsilon"])
        self.function_approximation = TileCoding(self.x_low, self.x_high, self.y_low, self.y_high, m, self.config["n_x_tiles"], self.config["n_y_tiles"], self.nA)

    def learn(self, n_episodes):
        for i in range(n_episodes):
            # print('Episode {}'.format(i))
            traces = EligibilityTraces(self.function_approximation.features_shape, self.config["gamma"], self.config["Lambda"])
            state, action = self.env.reset(), 0
            sarsa = Sarsa(self.config["gamma"], self.config["alpha"], self.policy, traces, self.function_approximation, range(self.nA), state, action)
            done = False  # Done says if the goal is reached or the maximum number of allowed steps for the episode is reached (determined by the gym library itself)
            iteration = 0
            while not(done):
                iteration += 1
                # env.render()
                state, reward, done, info = self.env.step(action)
                if done and iteration < self.config["steps_per_episode"]:
                    print("Episode {}: Less than {} steps were needed: {}".format(i, self.config["steps_per_episode"], iteration))
                action = sarsa.step(state, reward)

parser = argparse.ArgumentParser()
# No environment argument for now: algorithm only works with MountainCar-v0 right now
# parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the experiment on.")
parser.add_argument("n_episodes", metavar="n_episodes", type=int, help="Number of episodes to run")
parser.add_argument("monitor_path", metavar="monitor_path", type=str, help="Path where Gym monitor files may be saved")

def main():
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)
    # env = Environment(args.environment)
    env = Environment("MountainCar-v0")
    # if isinstance(env.action_space, Discrete):
    #     agent = SarsaFALearner(env)
    # else:
    #     raise NotImplementedError("Only environments with a discrete action space are supported right now.")
    agent = SarsaFALearner(env)
    try:
        agent.env = wrappers.Monitor(agent.env, args.monitor_path, force=True)
        agent.learn(args.n_episodes)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
