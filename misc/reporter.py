# -*- coding: utf8 -*-

import logging
import matplotlib
import numpy as np

gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg', 'agg']
for gui in gui_env:
    try:
        matplotlib.use(gui, warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except ImportError:
        continue

class Reporter(object):
    """Report iteration statistics using text and graphs."""
    def __init__(self):
        super(Reporter, self).__init__()
        logging.getLogger().setLevel("INFO")
        self.fig = None
        self.ax1 = None

    @staticmethod
    def print_iteration_stats(iteration: int, episode_rewards: np.ndarray, episode_lengths: np.ndarray, total_n_trajectories: int):
        """Print statistics about rewards and episode lengths of the current iteration."""
        logging.info("-----------------")
        logging.info("Iteration: \t\t {}".format(iteration))
        logging.info("NumTrajs: \t\t {}".format(len(episode_rewards)))
        logging.info("NumTimesteps: \t {}".format(np.sum(episode_lengths)))
        logging.info("MaxRew: \t\t {}".format(episode_rewards.max()))
        logging.info("MeanRew: \t\t {} +- {}".format(episode_rewards.mean(), episode_rewards.std() / np.sqrt(len(episode_rewards))))
        logging.info("MeanLen: \t\t {} +- {}".format(episode_lengths.mean(), episode_lengths.std() / np.sqrt(len(episode_lengths))))
        logging.info("Total NumTrajs: \t {}".format(total_n_trajectories))
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
