# -*- coding: utf8 -*-

import argparse
from scipy import signal
import numpy as np
from os import path
import json
import cv2
from gym.spaces.box import Box
from universe import vectorized
import tensorflow as tf

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

def _process_frame42(frame):
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

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]


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
    # Save git information if possible
    try:
        import pygit2
        repo = pygit2.Repository(path.abspath(path.join(path.dirname(path.realpath(__file__)), "..")))
        git = {
            "head": repo.head.shorthand,
            "commit": str(repo.head.target),
            "message": repo.head.get_object().message
        }
        config["git"] = git
    except ImportError:
        pass
    with open(path.join(directory, "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

def json_to_dict(filename):
    """Load a json file as a dictionary."""
    with open(filename) as f:
        return json.load(f)

def ge(minimum):
    """Require the value for an argparse argument to be an integer >= minimum."""
    def f(value):
        ivalue = int(value)
        if ivalue < minimum:
            raise argparse.ArgumentTypeError("{} must be an integer of at least 1.".format(value))
        return ivalue
    return f

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

class FastSaver(tf.train.Saver):
    """
    Disables write_meta_graph argument,
    which freezes entire process and is mostly useless.
    Source: https://github.com/openai/universe-starter-agent/blob/cbfa7901ff223adf89698f1de902811e4dabdca9/worker.py
    """
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)
