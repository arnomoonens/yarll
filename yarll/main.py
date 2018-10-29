#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse
from yarll.misc.utils import json_to_dict, save_config, set_seed

def run_experiment(spec, monitor_path=None, only_last=False, description=None, seed=None):
    """Run an experiment using a specification dictionary."""

    import os

    if seed is not None:
        set_seed(seed)

    import datetime
    import gym
    gym.logger.set_level(gym.logger.ERROR)
    from gym.spaces import Discrete
    from yarll.environment.registration import make, make_environments
    from yarll.agents.registration import make_agent

    args = spec["agent"]["args"]

    if monitor_path:
        args["monitor_path"] = monitor_path
    else:
        monitor_path = args["monitor_path"]
    args["config_path"] = os.path.join(monitor_path, "config.json")
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
    envs_type = spec["environments"]["type"]
    if envs_type == "single":
        envs = [make(spec["environments"]["source"])]
    elif envs_type == "json":
        envs = make_environments(json_to_dict(spec["environments"]["source"]))
    if seed is not None:
        for env in envs:
            env.seed(seed)
    args["seed"] = seed
    args["envs"] = envs
    if len(envs) == 1 or only_last:
        args["env"] = envs[-1]
    action_space_type = "discrete" if isinstance(envs[0].action_space, Discrete) else "continuous"
    state_dimensions = "single" if len(envs[0].observation_space.shape) == 1 else "multi"
    agent = make_agent(spec["agent"]["name"], state_dimensions, action_space_type, **args)
    config = agent.config.copy()
    if description is not None:
        config["description"] = description
    config["seed"] = str(seed)
    config["start_time"] = datetime.datetime.now().astimezone().isoformat()
    save_config(monitor_path, config, [env.metadata["parameters"] for env in envs])
    agent.learn()


parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str, help="JSON file with the experiment specification.")
parser.add_argument("--description", type=str, help="Experiment description.")
parser.add_argument("--monitor_path", metavar="monitor_path", default=None, type=str,
                    help="Path where Gym monitor files may be saved.")
parser.add_argument("--only_last", default=False, action="store_true",
                    help="Only use the last environment in a list of provided environments.")
parser.add_argument("--seed", default=None, type=int, help="Seed to use for the experiment.")

def main():
    args = parser.parse_args()
    run_experiment(
        json_to_dict(args.experiment),
        monitor_path=args.monitor_path,
        only_last=args.only_last,
        description=args.description,
        seed=args.seed
    )

if __name__ == '__main__':
    main()
