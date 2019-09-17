# -*- coding: utf8 -*-

import gym
import numpy as np


class DescriptionWrapper(gym.Wrapper):
    """
    Wrapper that mantains a description of the environment
    in its metadata.
    """

    changeable_parameters: list = []

    def __init__(self, env, **kwargs):
        super(DescriptionWrapper, self).__init__(env)

        self.args: dict = kwargs
        self.metadata = self.metadata.copy()
        if "changeable_parameters" not in self.metadata:
            self.metadata["changeable_parameters"] = self.changeable_parameters
        self.metadata["parameters"] = {"env_id": self.spec.id}
        self.metadata["parameters"].update(self.changeable_parameters_values())

    def changeable_parameters_values(self) -> dict:
        params: dict = {}
        for p in self.metadata["changeable_parameters"]:
            params[p["name"]] = self.unwrapped.__getattribute__(p["name"])
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
                "This wrapper can only be applied to environments with a discrete observation space.")
        self.n: int = self.observation_space.n
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.env.observation_space.n,))

    def observation(self, observation: int) -> np.ndarray:
        converted = np.zeros(self.n)
        converted[observation] = 1.0
        return converted


class NormalizedObservationWrapper(gym.ObservationWrapper):
    """
    Normalizes observations such that the values are
    between 0.0 and 1.0.
    """

    def __init__(self, env):
        super(NormalizedObservationWrapper, self).__init__(env)
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise AssertionError(
                "This wrapper can only be applied to environments with a continuous observation space.")
        if np.inf in self.env.observation_space.low or np.inf in self.env.observation_space.high:
            raise AssertionError(
                "This wrapper cannot be used for observation spaces with an infinite lower/upper bound.")
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=np.zeros(self.env.observation_space.shape),
            high=np.ones(self.env.observation_space.shape)
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self.env.observation_space.low) / \
            (self.env.observation_space.high - self.env.observation_space.low)


class NormalizedRewardWrapper(gym.RewardWrapper):
    """
    Normalizes rewards such that the values are between 0.0 and 1.0.
    """

    def __init__(self, env, low, high):
        super(NormalizedRewardWrapper, self).__init__(env)
        self.low = low
        self.high = high
        self.reward_range = (0.0, 1.0)

    def reward(self, rew):
        return (rew - self.low) / (self.high - self.low)

class CenteredScaledActionWrapper(gym.ActionWrapper):
    """
    Centers and scales the actions such that they range between -1 and 1.
    """

    def __init__(self, env):
        if not isinstance(env.action_space, gym.spaces.Box):
            raise AssertionError("The action space must be a Box.")
        super(CenteredScaledActionWrapper, self).__init__(env)

        self._low = self.env.action_space.low
        self._high = self.env.action_space.high
        self._diff = self._high - self._low
        self.action_space = gym.spaces.Box(-1., 1., self.env.action_space.shape)

    def action(self, action):
        return self._low + self._diff / 2 + action * self._diff / 2

    def reverse_action(self, action):
        return (2 * action - self._high - self._low) / self._diff
