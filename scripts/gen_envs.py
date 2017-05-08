#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import json
import argparse
import logging
import gym
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))

import environment
from misc.utils import ge

gym.envs.registration.logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("env_name", type=str, help="Name of environment for which to generate instances")
parser.add_argument("n", type=ge(1), help="Number of environments to generate")
parser.add_argument("destination", type=str, help="Where to write the environment descriptions (in json)")

def main():
    args = parser.parse_args()
    envs = environment.make_random_environments(args.env_name, args.n)
    dicts = [env.to_dict() for env in envs]
    with open(args.destination, "w") as f:
        json.dump(dicts, f)

if __name__ == '__main__':
    main()
