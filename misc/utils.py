#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse
from scipy import signal
import numpy as np
from os import path
import json

def discount_rewards(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Source: http://stackoverflow.com/a/12201744/1735784
def rgb2gray(rgb):
    """
    Convert an RGB image to a grayscale image.
    Uses the formula Y' = 0.299*R + 0.587*G + 0.114*B
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def preprocess_image(img):
    """
    Preprocess an image by converting it to grayscale and dividing its values by 256
    """
    img = img[35:195]  # crop
    img = img[::2, ::2]  # downsample by factor of 2
    return (rgb2gray(img) / 256.0)[:, :, None]

def save_config(directory, config, envs):
    """Save the configuration of an agent to a file."""
    config["envs"] = envs
    with open(path.join(directory, "config.json"), "w") as outfile:
        json.dump(config, outfile)

def json_to_dict(filename):
    """Load a json file as a dictionary."""
    with open(filename) as f:
        return json.load(f)

def ge_1(value):
    """Require the value for an argparse argument to be an integer >=1."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("{} must be an integer of at least 1.".format(value))
    return ivalue
