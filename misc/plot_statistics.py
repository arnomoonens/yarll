#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import json
import argparse
import re
import operator
import logging
from tensorflow.python.summary.event_multiplexer import EventMultiplexer, GetLogdirSubdirectories

from misc.utils import ge
from misc.exceptions import WrongArgumentsError

import matplotlib
gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg', 'agg']
for gui in gui_env:
    try:
        matplotlib.use(gui, warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except ImportError:
        continue

IMAGES_EXT = ".png"

logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Source: http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, n):
    """Compute the moving average of an array a of numbers using a window length n."""
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

def plot(x, y, x_label, scalar, xmax=None, ymin=None, ymax=None):
    x_label_upper = x_label[0].upper() + x_label[1:]
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlim(xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.xlabel(x_label_upper)
    plt.ylabel(scalar)
    plt.title("{} per {}".format(scalar, x_label))
    fig.canvas.set_window_title("{} per {}".format(scalar, x_label))

def plot_tasks(data, x_label, smoothing_function=None, xmin=None, xmax=None, max_reward=None, legend=True, save_directory=None, show_plots=True):
    x_label_upper = x_label[0].upper() + x_label[1:]
    for scalar, tasks in data.items():
        fig = plt.figure()
        # min_y = np.inf
        # max_y = -np.inf
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
    scalars = em.Runs()[runs[0]]["scalars"]  # Assumes that the scalars of every run are the same

    pattern = re.compile("task(\d+)$")
    data = {}
    for scalar in scalars:
        scalar_data = {}
        for run in runs:
            task = int(pattern.search(run).group(1))
            if task in scalar_data:
                scalar_data[task]["values"].append([x.value for x in em.Scalars(run, scalar)])
            else:
                values = []
                epochs = []
                for s in em.Scalars(run, scalar):
                    values.append(s.value)
                    epochs.append(s.step)
                scalar_data[task] = {
                    "epochs": epochs,
                    "values": [values]
                }
        data[scalar] = scalar_data
    return data

def plot_tf_scalar_summaries(summaries_dir, xmin=None, xmax=None, smoothing_function=None, max_reward=None, x_label="episode", legend=True, save_directory=None, show_plots=True):
    """Plot TensorFlow scalar summaries."""
    em = EventMultiplexer().AddRunsFromDirectory(summaries_dir).Reload()
    data = tf_scalar_data(em)
    plot_tasks(data, x_label, smoothing_function=smoothing_function, max_reward=max_reward, xmin=xmin, xmax=xmax, legend=legend, save_directory=save_directory, show_plots=show_plots)

def plot_tf_scalar_summaries_subdirs(summaries_dir, xmin=None, xmax=None, smoothing_function=None, max_reward=None, x_label="episode", legend=True, save_directory=None, show_plots=True):
    """Process each subdirectory of summaries_dir separately before plotting TensorFlow scalar summaries."""
    _, subdirs, _ = next(os.walk(summaries_dir))

    data = {}
    for subdir in subdirs:
        if not subdir.startswith("exp"):
            continue
        em = EventMultiplexer().AddRunsFromDirectory(os.path.join(summaries_dir, subdir)).Reload()
        subdir_data = tf_scalar_data(em)
        for scalar, scalar_data in subdir_data.items():
            if scalar not in data:
                data[scalar] = scalar_data
            else:
                for task, epochs_values in scalar_data.items():
                    data[scalar][task]["values"].extend(epochs_values["values"])

    plot_tasks(data, x_label, smoothing_function=smoothing_function, max_reward=max_reward, xmin=xmin, xmax=xmax, legend=legend, save_directory=save_directory, show_plots=show_plots)

def plot_tf_scalar_summaries_splitted(summaries_dir, xmin=None, xmax=None, smoothing_function=None, max_reward=None, x_label="episode", legend=True, splitted_length=25, save_directory=None, show_plots=True):
    """Process sets of runs (of length splitted_length) separately before plotting TensorFlow scalar summaries."""
    _, rundirs, _ = next(os.walk(summaries_dir))
    data = {}
    for runs_group in range(int(np.ceil(len(rundirs) / splitted_length))):
        logging.info("Processing run group {}. Runs processed: {}/{}.".format(runs_group, runs_group * splitted_length, len(rundirs)))
        em = EventMultiplexer()
        for run_idx in range(runs_group * splitted_length, runs_group * splitted_length + splitted_length):
            run_path = os.path.join(summaries_dir, "run" + str(run_idx + 1))
            for subdir in GetLogdirSubdirectories(run_path):
                rpath = os.path.relpath(subdir, summaries_dir)
                em.AddRun(subdir, name=rpath)
        em.Reload()
        data_runs = tf_scalar_data(em)
        for scalar, scalar_data in data_runs.items():
            if scalar not in data:
                data[scalar] = scalar_data
            else:
                for task, epochs_values in scalar_data.items():
                    data[scalar][task]["values"].extend(epochs_values["values"])
    logging.info("Data processed, plotting...")
    plot_tasks(data, x_label, smoothing_function=smoothing_function, max_reward=max_reward, xmin=xmin, xmax=xmax, legend=legend, save_directory=save_directory, show_plots=show_plots)

def exp_smoothing_weight_test(value):
    """Require that the weight for exponential smoothing is a weight between 0 and 1"""
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError("{} must be a float between 0 and 1".format(value))
    return fvalue

parser = argparse.ArgumentParser()
parser.add_argument("stats_path", metavar="stats", type=str, help="Path to the Tensorflow monitor stats.json file or summaries directory.")
parser.add_argument("--x_label", type=str, default="episode", choices=["episode", "epoch"], help="Whether to use episode or epoch as x label.")
parser.add_argument("--no_legend", action="store_false", default=True, dest="legend", help="Don't show a legend in the plots.")
parser.add_argument("--xmin", type=ge(0), default=None, help="minimum episode for which to show results.")
parser.add_argument("--xmax", type=ge(1), default=None, help="Maximum episode for which to show results.")
parser.add_argument("--max_reward", type=float, default=None, help="Maximum obtainable reward in the environment.")
parser.add_argument("--exp_smoothing", type=exp_smoothing_weight_test, default=None, help="Use exponential smoothing with a weight 0<=w<=1.")
parser.add_argument("--moving_average", type=ge(1), default=None, help="Use a moving average with a window w>0")
parser.add_argument("--subdirs", action="store_true", default=False, help="Process each subdirectory separately (solves memory issues).")
parser.add_argument("--splitted", action="store_true", default=False, help="Process sets of runs separately (solves memory issues).")
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
        smoothing_technique = create_smoother(exponential_smoothing, args.exp_smoothing)  # args.exp_smoothing holds the weight
    elif args.moving_average is not None:
        smoothing_technique = create_smoother(moving_average, args.moving_average)  # args.moving_average holds the window
    if os.path.isfile(args.stats_path) and args.stats_path.endswith(".json"):
        plot_gym_monitor_stats(args.stats_path, args.xmax, smoothing_technique, save_directory=args.save_dir, show_plots=args.show)
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
        if args.subdirs:
            plot_tf_scalar_summaries_subdirs(**shared_args)
        elif args.splitted:
            plot_tf_scalar_summaries_splitted(splitted_length=25, **shared_args)
        else:
            plot_tf_scalar_summaries(**shared_args)
    else:
        raise NotImplementedError("Only a Tensorflow monitor stats.json file or summaries directory is allowed.")
