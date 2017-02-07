#!/usr/bin/env python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import argparse

# Source: http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy?rq=1
def moving_average(a, n):
    """Compute the moving average of an array a of numbers using a window length n"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main(stats_path, n, xmax):
    f = open(stats_path)
    contents = json.load(f)

    averaged_episode_rewards = moving_average(contents['episode_rewards'], n)
    fig = plt.figure()
    plt.plot(range(len(averaged_episode_rewards)), averaged_episode_rewards)
    plt.xlim(xmax=xmax)
    min_aer = min(averaged_episode_rewards)
    max_aer = max(averaged_episode_rewards)
    plt.ylim(ymin=min(0, min_aer), ymax=max(0, max_aer + 0.1 * max_aer))
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Total reward per episode")
    fig.canvas.set_window_title("Total reward per episode")

    fig = plt.figure()
    averaged_episode_lengths = moving_average(contents['episode_lengths'], n)
    plt.plot(range(len(averaged_episode_lengths)), averaged_episode_lengths)
    plt.xlim(xmax=xmax)
    max_ael = max(averaged_episode_lengths)
    plt.ylim(ymin=0, ymax=(max_ael + 0.1 * max_ael))
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Length per episode")
    fig.canvas.set_window_title("Length per episode")
    plt.show()

def ge_1(value):
    """Require the value for an argparse argument to be an integer >=1."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("%s must be an integer of at least 1." % value)
    return ivalue

parser = argparse.ArgumentParser()
parser.add_argument("stats_path", metavar="stats", type=str, help="Path to the stats JSON file.")
parser.add_argument("running_mean_length", metavar="rml", type=ge_1, help="Running mean length")
parser.add_argument("--xmax", type=ge_1, default=None, help="Maximum episode for which to show results.")

if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    main(args.stats_path, args.running_mean_length, args.xmax)
