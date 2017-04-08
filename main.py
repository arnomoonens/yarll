#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse
import os
from gym.spaces import Discrete

from Environment.registration import make_environment, make_environments
from agents.registration import make_agent
from misc.utils import json_to_dict, save_config

def run_experiment(spec, monitor_path=None):
    """Run an experiment using a specification dictionary."""
    args = spec["agent"]["args"]
    if monitor_path:
        args["monitor_path"] = monitor_path
    else:
        monitor_path = args["monitor_path"]
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
    envs_type = spec["environments"]["type"]
    if envs_type == "single":
        envs = [make_environment(spec["environments"]["source"])]
        args["env"] = envs[0]
    elif envs_type == "json":
        envs = make_environments(json_to_dict(spec["environments"]["source"]))
        args["envs"] = envs
    action_space_type = "discrete" if isinstance(envs[0].action_space, Discrete) else "continuous"
    agent = make_agent(spec["agent"]["name"], action_space_type, **args)
    save_config(monitor_path, agent.config, [env.to_dict() for env in envs])
    agent.learn()

parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str, help="JSON file with the experiment specification")
parser.add_argument("--monitor_path", metavar="monitor_path", default=None, type=str, help="Path where Gym monitor files may be saved")

def main():
    args = parser.parse_args()
    run_experiment(json_to_dict(args.experiment), args.monitor_path)

if __name__ == '__main__':
    main()
