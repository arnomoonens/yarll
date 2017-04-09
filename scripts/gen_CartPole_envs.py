#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
sys.path.append(".")

from Environment import make_random_environments
import json
import argparse
from misc.utils import ge_1

parser = argparse.ArgumentParser()
parser.add_argument("n", type=ge_1, help="Number of environments to generate")
parser.add_argument("destination", type=str, help="Where to write the environment descriptions (in json)")

def main():
    args = parser.parse_args()
    envs = make_random_environments("CartPole-v0", args.n)
    dicts = [env.to_dict() for env in envs]
    with open(args.destination, "w") as f:
        json.dump(dicts, f)

if __name__ == '__main__':
    main()
