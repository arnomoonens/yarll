#!/usr/bin/env python
# -*- coding: utf8 -*-

import json
import argparse
import logging
from os.path import abspath, dirname
import gym

import yarll.environment
from yarll.misc.utils import ge

gym.envs.registration.logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("env_id", type=str, help="Name of environment for which to generate instances")
parser.add_argument("n", type=ge(1), help="Number of environments to generate")
parser.add_argument("destination", type=str, help="Where to write the environment descriptions (in json)")

def main():
    args = parser.parse_args()
    envs = yarll.environment.registration.make_random_environments(args.env_id, args.n)
    dicts = [env.metadata["parameters"] for env in envs]
    with open(args.destination, "w") as f:
        json.dump(dicts, f, indent=4)

if __name__ == '__main__':
    main()
