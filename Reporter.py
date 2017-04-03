# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
import logging

class Reporter(object):
    """Report iteration statistics using text and graphs."""
    def __init__(self):
        super(Reporter, self).__init__()
        logging.getLogger().setLevel("INFO")
        self.fig = None
        self.ax1 = None

    @staticmethod
    def print_iteration_stats(self, iteration, episode_rewards, episode_lengths, total_n_trajectories):
        """Print statistics about rewards and episode lengths of the current iteration"""
        logging.info("-----------------")
        logging.info("Iteration: \t\t %i" % iteration)
        logging.info("NumTrajs: \t\t %i" % len(episode_rewards))
        logging.info("NumTimesteps: \t %i" % np.sum(episode_lengths))
        logging.info("MaxRew: \t\t %s" % episode_rewards.max())
        logging.info("MeanRew: \t\t %s +- %s" % (episode_rewards.mean(), episode_rewards.std() / np.sqrt(len(episode_rewards))))
        logging.info("MeanLen: \t\t %s +- %s" % (episode_lengths.mean(), episode_lengths.std() / np.sqrt(len(episode_lengths))))
        logging.info("Total NumTrajs: \t %i" % total_n_trajectories)
        logging.info("-----------------")
        return

    def draw_rewards(self, mean_rewards):
        """Draw a plot with the mean reward for each batch of episodes."""
        if not self.fig:
            self.fig = plt.figure()
        if not self.ax1:
            self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax1.clear()
        self.ax1.plot(range(len(mean_rewards)), mean_rewards)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
