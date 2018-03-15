# -*- coding: utf8 -*-

import numpy as np
import gym
from gym import Wrapper

class DescriptionWrapper(Wrapper):
    """
    Wrapper that mantains a description of the environment
    in its metadata.
    """

    changeable_parameters: list = []

    def __init__(self, env, **kwargs):
        super(DescriptionWrapper, self).__init__(env)
        self.args: dict = kwargs
        self.metadata["changeable_parameters"] = self.changeable_parameters
        self.metadata["parameters"] = {"name": self.__str__()}
        self.metadata["parameters"].update(self.changeable_parameters_values())

    def changeable_parameters_values(self) -> dict:
        params: dict = {}
        for p in self.changeable_parameters:
            params[p["name"]] = self.__getattribute__(p["name"])
        return params

class DiscreteObservationWrapper(gym.ObservationWrapper):
    """
    Converts envs with a discrete observation space (Discrete(N)) to one
    with a one-hot encoding (Box(N,)).
    """

    def __init__(self, env):
        super(DiscreteObservationWrapper, self).__init__(env)

        if not isinstance(self.env.observation_space, gym.spaces.Discrete):
            raise AssertionError(
                "The DiscreteObservationWrapper can only be applied to environments with a discrete observation space.")
        self.n: int = self.observation_space.n
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.env.observation_space.n,))

    def _observation(self, observation: int) -> np.ndarray:
        converted = np.zeros(self.n)
        converted[observation] = 1.0
        return converted
