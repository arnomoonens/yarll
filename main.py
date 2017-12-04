#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse
import datetime
import os
from gym.spaces import Discrete

from environment.registration import make_environment, make_environments
from agents.registration import make_agent
from misc.utils import json_to_dict, save_config

def run_experiment(spec, monitor_path=None, only_last=False, description=None):
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
    elif envs_type == "json":
        envs = make_environments(json_to_dict(spec["environments"]["source"]))
    args["envs"] = envs
    if len(envs) == 1 or only_last:
        args["env"] = envs[-1]
    action_space_type = "discrete" if isinstance(envs[0].action_space, Discrete) else "continuous"
    state_dimensions = "single" if len(envs[0].observation_space.shape) == 1 else "multi"
    agent = make_agent(spec["agent"]["name"], state_dimensions, action_space_type, **args)
    config = agent.config.copy()
    if description is not None:
        config["description"] = description
    config["start_time"] = datetime.datetime.now().astimezone().isoformat()
    save_config(monitor_path, config, [env.to_dict() for env in envs])
    agent.learn()

parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str, help="JSON file with the experiment specification.")
parser.add_argument("--description", type=str, help="Experiment description.")
parser.add_argument("--monitor_path", metavar="monitor_path", default=None, type=str, help="Path where Gym monitor files may be saved.")
parser.add_argument("--only_last", default=False, action="store_true", help="Only use the last environment in a list of provided environments.")

def main():
    args = parser.parse_args()
    run_experiment(json_to_dict(args.experiment), monitor_path=args.monitor_path, only_last=args.only_last, description=args.description)

if __name__ == '__main__':
    main()
