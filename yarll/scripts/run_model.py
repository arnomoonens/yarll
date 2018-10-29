#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import argparse
import tensorflow as tf
from gym import wrappers

from yarll.environment.registration import make

class ModelRunner(object):
    """
    Run an already learned model.
    Currently only supports one variation of an environment.
    """
    def __init__(self, env, model_directory: str, save_directory: str, **usercfg) -> None:
        super(ModelRunner, self).__init__()
        self.env = env
        self.model_directory = model_directory
        self.save_directory = save_directory
        self.config = dict(
            episode_max_length=self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'),
            repeat_n_actions=1
        )
        self.config.update(usercfg)

        self.session = tf.Session()
        self.saver = tf.train.import_meta_graph(os.path.join(self.model_directory, "model.meta"))
        self.saver.restore(self.session, os.path.join(self.model_directory, "model"))

        self.action = tf.get_collection("action")[0]
        self.states = tf.get_collection("states")[0]

    def choose_action(self, state):
        """Choose an action."""
        return self.session.run([self.action], feed_dict={self.states: [state]})[0]

    def get_trajectory(self, render: bool = False):
        """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        """
        state = self.env.reset()
        for _ in range(self.config["episode_max_length"]):
            action = self.choose_action(state)
            for _ in range(self.config["repeat_n_actions"]):
                _, _, done, _ = self.env.step(action)
                if done:  # Don't continue if episode has already ended
                    break
            if done:
                break
            if render:
                self.env.render()
        return

    def run(self):
        for _ in range(self.config["n_iter"]):
            self.get_trajectory()


parser = argparse.ArgumentParser()
parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the model on.")
parser.add_argument("model_directory", type=str, help="Directory from where model files are loaded.")
parser.add_argument("save_directory", type=str, help="Directory where results of running the model are saved")
parser.add_argument("--iterations", default=100, type=int, help="Number of iterations to run the algorithm.")

def main():
    args = parser.parse_args()
    env = make(args.environment)
    runner = ModelRunner(env, args.model_directory, args.save_directory, n_iter=args.iterations)
    try:
        runner.env = wrappers.Monitor(runner.env, args.save_directory, video_callable=False, force=True)
        runner.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
