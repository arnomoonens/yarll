# -*- coding: utf8 -*-

# Inspired by https://github.com/openai/gym/blob/master/gym/envs/registration.py
# When making an environment, we first look if we registered a version of it ourselves.
# Else, we make just make one using the Environment class.
import numpy as np

import gym
from misc.utils import AtariRescale42x42
from universe.wrappers import Unvectorize, Vectorize

class EnvSpec(gym.envs.registration.EnvSpec):
    """
    Modified version of `gym.envs.registration.EnvSpec`
    to allow env initalization with different parameters.
    """

    def make(self, **kwargs):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self._entry_point is None:
            raise gym.error.Error('Attempting to make deprecated env {}. (HINT: is there a newer registered version of this env?)'.format(self.id))

        elif callable(self._entry_point):
            env = self._entry_point()
        else:
            cls = gym.envs.registration.load(self._entry_point)
            all_kwargs = self._kwargs
            all_kwargs.update(kwargs)
            env = cls(**all_kwargs)

        # Make the enviroment aware of which spec it came from.
        env.unwrapped._spec = self

        return env

def make(env_id, **kwargs):
    spec = gym.envs.registry.spec(env_id)
    env = spec.make(**kwargs)

    def to_dict():
        return {"id": env_id}
    if "to_dict" not in dir(env):
        env.to_dict = to_dict
    if "atari.atari_env" in env.unwrapped.__module__:
        to_dict = env.to_dict
        env = Vectorize(env)
        env = AtariRescale42x42(env)
        env = Unvectorize(env)
        env.to_dict = to_dict
    if (env.spec.timestep_limit is not None) and not spec.tags.get('vnc'):
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env,
                        max_episode_steps=env.spec.max_episode_steps,
                        max_episode_seconds=env.spec.max_episode_seconds)
        env.to_dict = env.env.to_dict
    return env

def make_environments(descriptions):
    """Make environments using a list of descriptions."""
    return [make(**d) for d in descriptions]

def make_random_environments(env_id, n_envs):
    """Make n_envs random environments of the env_name class."""
    spec = gym.envs.registry.spec(env_id)
    cls = gym.envs.registration.load(spec._entry_point)
    envs = []
    for _ in range(n_envs):
        args = {"id": env_id}
        for p in cls.changeable_parameters:
            if p["type"] == "range":
                args[p["name"]] = np.random.uniform(p["low"], p["high"])  # Assume for now only range parameters are used
        envs.append(make(**args))
    return envs
