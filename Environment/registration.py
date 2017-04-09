#!/usr/bin/env python
# -*- coding: utf8 -*-

# Inspired by https://github.com/openai/gym/blob/master/gym/envs/registration.py
# When making an environment, we first look if we registered a version of it ourselves.
# Else, we make just make one using the Environment class.
import numpy as np

from Environment import Environment
from misc.Exceptions import ClassNotRegisteredError

environment_registry = {}

def register_environment(name, cls):
    """Register an enviroment of a name with a class to be instantiated."""
    environment_registry[name] = cls

def make_environment(name, **args):
    """Make an environment of a given name, possibly using extra arguments."""
    env = environment_registry.get(name, Environment)
    if name in environment_registry:
        return env(**args)
    else:
        return env(name, **args)

def make_environments(descriptions):
    """Make environments using a list of descriptions."""
    return [make_environment(**d) for d in descriptions]

def make_random_environments(env_name, n_envs):
    """Make n_envs random environments of the env_name class."""
    if env_name not in environment_registry:
        raise ClassNotRegisteredError("Class {} must be registered in order to be randomly instantiated.".format(env_name))
    cls = environment_registry.get(env_name)
    envs = []
    for _ in range(n_envs):
        params = {}
        for p in cls.changeable_parameters:
            params[p["name"]] = np.random.uniform(p["low"], p["high"])  # Assume for now only range parameters are used
        envs.append(cls(**params))
    return envs
