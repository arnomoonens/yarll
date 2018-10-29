#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import re
import json
import argparse
import operator
import logging
import numpy as np
from tensorboard.backend.event_processing.plugin_event_multiplexer import EventMultiplexer
import matplotlib.pyplot as plt

from yarll.misc.utils import ge
from yarll.misc.exceptions import WrongArgumentsError

IMAGES_EXT = ".png"

logging.getLogger("tensorflow").setLevel(logging.WARNING)

def moving_average(a, n):
    """
    Compute the moving average of an array a of numbers using a window length n.
    Source: http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Source: https://github.com/tensorflow/tensorflow/blob/d413f62fba038d820b775d95ddd518fb6f43d3ed/tensorflow/tensorboard/components/vz_line_chart/vz-line-chart.ts#L432
def exponential_smoothing(a, weight):
    factor = (pow(1000, weight) - 1) / 999
    length = len(a)
    kernelRadius = np.floor(length * factor / 2)
    result = []
    for i, x in enumerate(a):
        actualKernelRadius = min(kernelRadius, i)
        start = int(i - actualKernelRadius)
        end = int(i + actualKernelRadius + 1)
        if end >= length:
            result.append(np.inf)
        elif x == np.inf:
            result.append(x)
        else:
            result.append(np.mean(list(filter(lambda x: x != np.inf, a[start:end]))))
    return result

def create_smoother(f, *args):
    return lambda x: f(x, *args)

def plot(x, y, x_label: str, scalar, xmax=None, ymin=None, ymax=None):
    x_label_upper = x_label[0].upper() + x_label[1:]
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlim(xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.xlabel(x_label_upper)
    plt.ylabel(scalar)
    plt.title("{} per {}".format(scalar, x_label))
    fig.canvas.set_window_title("{} per {}".format(scalar, x_label))

def plot_tasks(data, x_label: str, smoothing_function=None, xmin=None, xmax=None, max_reward=None, legend=True, save_directory=None, show_plots=True):
    x_label_upper = x_label[0].upper() + x_label[1:]
    for scalar, tasks in data.items():
        fig = plt.figure()
        for task, epochs_values in sorted(tasks.items(), key=operator.itemgetter(0)):
            mean = np.mean(epochs_values["values"], axis=0)
            if smoothing_function:
                mean = smoothing_function(mean)
            # percentiles = np.percentile(epochs_values["values"], [25, 75], axis=0)
            std = np.std(epochs_values["values"], axis=0)
            std = std[len(std) - len(mean):]
            # error_min, error_max = mean - std, mean + std
            plt.plot(epochs_values["epochs"], mean, label="Task " + str(task))
            # plt.fill_between(x, error_min, error_max, alpha=0.3)
            # min_y = min(min_y, min(error_min))
            # max_y = max(max_y, max(filter(lambda x: x != np.inf, error_max)))
        if legend:
            plt.legend()
        plt.xlim(xmin=xmin, xmax=xmax)
        # plt.ylim(ymin=min(0, min_y), ymax=max(0, max_y + 0.1 * max_y))
        if "reward" in scalar.lower() and max_reward is not None:
            ymax = max_reward * 1.1
        else:
            ymax = None
        plt.ylim(ymax=ymax)
        plt.xlabel(x_label_upper)
        plt.ylabel(scalar)
        plt.title("{} per {}".format(scalar, x_label))
        fig.canvas.set_window_title("{} per {}".format(scalar, x_label))
        if save_directory is not None:
            plt.savefig(os.path.join(save_directory, "{}_per_{}".format(scalar, x_label) + IMAGES_EXT))
    if show_plots:
        plt.show()

def plot_gym_monitor_stats(stats_path, xmax=None, smoothing_function=None, save_directory=None, show_plots=True):
    f = open(stats_path)
    contents = json.load(f)
    data = contents["episode_rewards"]
    if smoothing_function:
        data = smoothing_function(data)
    min_aer = min(data)
    max_aer = max(filter(lambda x: x != np.inf, data))
    ymin = min(0, min_aer)
    ymax = max(0, max_aer + 0.1 * max_aer)
    plot(range(len(data)), data, "episode", "total reward", xmax=xmax, ymin=ymin, ymax=ymax)
    if save_directory is not None:
        plt.savefig(os.path.join(save_directory, "totalreward_per_episode" + IMAGES_EXT))

    data = contents["episode_lengths"]
    if smoothing_function:
        data = smoothing_function(data)
    max_ael = max(filter(lambda x: x != np.inf, data))
    ymin = 0
    ymax = (max_ael + 0.1 * max_ael)
    plot(range(len(data)), data, "episode", "length", xmax=xmax, ymin=ymin, ymax=ymax)
    if save_directory is not None:
        plt.savefig(os.path.join(save_directory, "length_per_episode" + IMAGES_EXT))
    if show_plots:
        plt.show()

def tf_scalar_data(em):
    """Process scalar data of TensorFlow summaries."""
    runs = list(em.Runs().keys())
    scalars = em.Runs()[runs[0]]["tensors"]  # Assumes that the scalars of every run are the same
    pattern = re.compile("[a-zA-Z]+(\d+)$")
    data = {}
    for scalar in scalars:
        scalar_data = {}
        for run in runs:
            task = int(pattern.search(run).group(1))
            accum = em.GetAccumulator(run)
            if task in scalar_data:
                scalar_data[task]["values"].append([s.tensor_proto.float_val[0] for s in accum.Tensors(scalar)])
            else:
                values = []
                epochs = []
                for s in accum.Tensors(scalar):
                    values.append(s.tensor_proto.float_val[0])
                    epochs.append(s.step)
                scalar_data[task] = {
                    "epochs": epochs,
                    "values": [values]
                }
        data[scalar] = scalar_data
    return data

def plot_tf_scalar_summaries(summaries_dir,
                             xmin=None,
                             xmax=None,
                             smoothing_function=None,
                             max_reward=None,
                             x_label="episode",
                             legend=True,
                             save_directory=None,
                             show_plots=True):
    """Plot TensorFlow scalar summaries."""
    em = EventMultiplexer().AddRunsFromDirectory(summaries_dir).Reload()
    data = tf_scalar_data(em)
    plot_tasks(data,
               x_label,
               smoothing_function=smoothing_function,
               max_reward=max_reward,
               xmin=xmin,
               xmax=xmax,
               legend=legend,
               save_directory=save_directory,
               show_plots=show_plots)

def exp_smoothing_weight_test(value):
    """Require that the weight for exponential smoothing is a weight between 0.0 and 1.0"""
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError("{} must be a float between 0.0 and 1.0".format(value))
    return fvalue

parser = argparse.ArgumentParser()
parser.add_argument("stats_path", metavar="stats", type=str,
                    help="Path to the Tensorflow monitor stats.json file or summaries directory.")
parser.add_argument("--x_label", type=str, default="episode", choices=["episode", "epoch"],
                    help="Whether to use episode or epoch as x label.")
parser.add_argument("--no_legend", action="store_false", default=True, dest="legend",
                    help="Don't show a legend in the plots.")
parser.add_argument("--xmin", type=ge(0), default=None, help="minimum episode for which to show results.")
parser.add_argument("--xmax", type=ge(1), default=None, help="Maximum episode for which to show results.")
parser.add_argument("--max_reward", type=float, default=None, help="Maximum obtainable reward in the environment.")
parser.add_argument("--exp_smoothing", type=exp_smoothing_weight_test, default=None,
                    help="Use exponential smoothing with a weight 0<=w<=1.")
parser.add_argument("--moving_average", type=ge(1), default=None, help="Use a moving average with a window w>0")
parser.add_argument("--save", type=str, dest="save_dir", default=None, help="Save plots to the given directory.")
parser.add_argument("--show", action="store_true", default=False, help="Show plots.")
parser.add_argument(
    '-d', '--debug',
    help="Print debugging statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.WARNING
)

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.exp_smoothing is not None and args.moving_average is not None:
        raise WrongArgumentsError("Maximally 1 smoothing technique can be used.")
    smoothing_technique = None
    if args.exp_smoothing is not None:
        smoothing_technique = create_smoother(exponential_smoothing,
                                              args.exp_smoothing)  # args.exp_smoothing holds the weight
    elif args.moving_average is not None:
        smoothing_technique = create_smoother(moving_average,
                                              args.moving_average)  # args.moving_average holds the window
    if os.path.isfile(args.stats_path) and args.stats_path.endswith(".json"):
        plot_gym_monitor_stats(args.stats_path,
                               args.xmax,
                               smoothing_technique,
                               save_directory=args.save_dir,
                               show_plots=args.show)
    elif os.path.isdir(args.stats_path):
        shared_args = {
            "summaries_dir": args.stats_path,
            "xmin": args.xmin,
            "xmax": args.xmax,
            "smoothing_function": smoothing_technique,
            "max_reward": args.max_reward,
            "x_label": args.x_label,
            "legend": args.legend,
            "save_directory": args.save_dir,
            "show_plots": args.show
        }
        plot_tf_scalar_summaries(**shared_args)
    else:
        raise NotImplementedError("Only a Tensorflow monitor stats.json file or summaries directory is allowed.")
