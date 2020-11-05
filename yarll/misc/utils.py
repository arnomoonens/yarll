# -*- coding: utf8 -*-

import itertools
import sys
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Callable, Dict, List, Sequence, Union
import pkg_resources
import tensorflow as tf
from scipy import signal
import numpy as np

import gym
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete

def discount_rewards(x: Sequence, gamma: float) -> np.ndarray:
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Source: http://stackoverflow.com/a/12201744/1735784
def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to a grayscale image.
    Uses the formula Y' = 0.299*R + 0.587*G + 0.114*B
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def _process_frame42(frame: np.ndarray) -> np.ndarray:
    """
    Process frames from frame

    Args:
        frame: (array): write your description
        np: (array): write your description
        ndarray: (array): write your description
    """
    import cv2
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Initialize space.

        Args:
            self: (todo): write your description
            env: (todo): write your description
        """
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the observation.

        Args:
            self: (todo): write your description
            observation: (str): write your description
            np: (todo): write your description
            ndarray: (array): write your description
        """
        return _process_frame42(observation)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess an image by converting it to grayscale and dividing its values by 256
    """
    img = img[35:195]  # crop
    img = img[::2, ::2]  # downsample by factor of 2
    return (rgb2gray(img) / 256.0)[:, :, None]

def execute_command(cmd: List[str]) -> str:
    """Execute a terminal command and return the stdout."""
    res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return res.decode()[:-1]  # decode to go from bytes to str, [:-1] to remove newline at end

def save_config(directory: Union[str, Path],
                config: Dict,
                agent_class: type,
                envs: list,
                repo_path: Union[str, Path] = Path(__file__).parent / "../../") -> None:
    """Save the configuration of an agent to a file."""
    filtered_config = {k: v for k, v in config.items() if not k.startswith("env")}
    filtered_config["envs"] = envs
    # Save git information if possible
    git_dir = Path(repo_path) / ".git"
    try:
        git = {
            "head": execute_command(["git", f"--git-dir={git_dir}", "rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": execute_command(["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"]),
            "message": execute_command(["git", f"--git-dir={git_dir}", "log", "-1", "--pretty=%B"])[:-1],
            "diff": execute_command(["git", f"--git-dir={git_dir}", "diff", "--no-prefix"])
        }
        filtered_config["git"] = git
    except subprocess.CalledProcessError:
        pass

    # save pip freeze output
    pipfreeze = execute_command([sys.executable, "-m", "pip", "freeze"])
    filtered_config["packages"] = pipfreeze.split("\n")
    # Save command used to run program
    filtered_config["program_command"] = " ".join(sys.argv)
    # Save agent class
    filtered_config["agent_class"] = str(agent_class)

    with open(Path(directory) / "config.json", "w") as outfile:
        json.dump(filtered_config, outfile, indent=4)

def json_to_dict(file: Union[str, Path]) -> dict:
    """Load a json file as a dictionary."""
    with open(file) as f:
        return json.load(f)

def ge(minimum: int) -> Callable[[Any], int]:
    """Require the value for an argparse argument to be an integer >= minimum."""
    def f(value):
        """
        Decorator toil. integer.

        Args:
            value: (todo): write your description
        """
        ivalue = int(value)
        if ivalue < minimum:
            raise argparse.ArgumentTypeError("{} must be an integer of at least 1.".format(value))
        return ivalue
    return f

def flatten(x):
    """
    Flatten a tensor.

    Args:
        x: (todo): write your description
    """
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def set_seed(seed: int):
    """
    Set the seed.

    Args:
        seed: (int): write your description
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load(name: str):
    """Load an object by string."""
    entry_point = pkg_resources.EntryPoint.parse("x={}".format(name))
    result = entry_point.resolve()
    return result

def cluster_spec(num_workers: int, num_ps: int, num_masters: int = 0) -> dict:
    """
    Generate a cluster specification (for distributed Tensorflow).
    """
    cluster = {}
    port = 12222

    all_ps = []
    host = "127.0.0.1"
    for _ in range(num_ps):
        all_ps.append("{}:{}".format(host, port))
        port += 1
    cluster["ps"] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append("{}:{}".format(host, port))
        port += 1
    cluster["worker"] = all_workers

    if num_masters > 0:
        all_masters = []
        for _ in range(num_masters):
            all_masters.append("{}:{}".format(host, port))
            port += 1
        cluster["master"] = all_masters
    return cluster

def soft_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable], tau: float) -> None:
    """Move each source variable by a factor of tau towards the corresponding target variable.

    Arguments:
        source_vars {Sequence[tf.Variable]} -- Source variables to copy from
        target_vars {Sequence[tf.Variable]} -- Variables to copy data to
        tau {float} -- How much to change to source var, between 0 and 1.
    """
    if len(source_vars) != len(target_vars):
        raise ValueError("source_vars and target_vars must have the same length.")
    for source, target in zip(source_vars, target_vars):
        target.assign((1.0 - tau) * target + tau * source)


def hard_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable]) -> None:
    """Copy source variables to target variables.

    Arguments:
        source_vars {Sequence[tf.Variable]} -- Source variables to copy from
        target_vars {Sequence[tf.Variable]} -- Variables to copy data to
    """
    soft_update(source_vars, target_vars, 1.0) # Tau of 1, so get everything from source and keep nothing from target

def flatten_list(l: List[List]):
    """
    Flatten a list into a list.

    Args:
        l: (todo): write your description
    """
    return list(itertools.chain.from_iterable(l))

spaces_mapping = {
    Discrete: "discrete",
    MultiDiscrete: "multidiscrete",
    Box: "continuous",
    MultiBinary: "multibinary"
}
