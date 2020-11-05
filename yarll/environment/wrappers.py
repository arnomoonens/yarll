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
        """
        Initialize the environment.

        Args:
            self: (todo): write your description
            env: (todo): write your description
        """
        super(DescriptionWrapper, self).__init__(env)

        self.args: dict = kwargs
        self.metadata = self.metadata.copy()
        if "changeable_parameters" not in self.metadata:
            self.metadata["changeable_parameters"] = self.changeable_parameters
        self.metadata["parameters"] = {"env_id": self.spec.id}
        self.metadata["parameters"].update(self.changeable_parameters_values())

    def changeable_parameters_values(self) -> dict:
        """
        Returns a dictionary of parameter parameters.

        Args:
            self: (todo): write your description
        """
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
        """
        Initialize the environment.

        Args:
            self: (todo): write your description
            env: (todo): write your description
        """
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
        """
        Convert an observation.

        Args:
            self: (todo): write your description
            observation: (str): write your description
        """
        converted = np.zeros(self.n)
        converted[observation] = 1.0
        return converted


class NormalizedObservationWrapper(gym.ObservationWrapper):
    """
    Normalizes observations such that the values are
    between 0.0 and 1.0.
    """

    def __init__(self, env):
        """
        Initialize observations.

        Args:
            self: (todo): write your description
            env: (todo): write your description
        """
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
        """
        Return the observation of an observation.

        Args:
            self: (todo): write your description
            observation: (str): write your description
            np: (todo): write your description
            ndarray: (array): write your description
        """
        return (observation - self.env.observation_space.low) / \
            (self.env.observation_space.high - self.env.observation_space.low)


class NormalizedRewardWrapper(gym.RewardWrapper):
    """
    Normalizes rewards such that the values are between 0.0 and 1.0.
    """

    def __init__(self, env, low=None, high=None):
        """
        Initialize the environment.

        Args:
            self: (todo): write your description
            env: (todo): write your description
            low: (todo): write your description
            high: (int): write your description
        """
        super(NormalizedRewardWrapper, self).__init__(env)
        self.low = low if low is not None else self.env.reward_range[0]
        self.high = high if high is not None else self.env.reward_range[1]
        self.reward_range = (0.0, 1.0)

    def reward(self, rew):
        """
        Compute the reward

        Args:
            self: (todo): write your description
            rew: (todo): write your description
        """
        return (rew - self.low) / (self.high - self.low)

class CenteredScaledActionWrapper(gym.ActionWrapper):
    """
    Centers and scales the actions such that they range between -1 and 1.
    """

    def __init__(self, env):
        """
        Initialize the environment.

        Args:
            self: (todo): write your description
            env: (todo): write your description
        """
        if not isinstance(env.action_space, gym.spaces.Box):
            raise AssertionError("The action space must be a Box.")
        super(CenteredScaledActionWrapper, self).__init__(env)

        self._low = self.env.action_space.low
        self._high = self.env.action_space.high
        self._diff = self._high - self._low
        self.action_space = gym.spaces.Box(-1., 1., self.env.action_space.shape)

    def action(self, action):
        """
        The action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        return self._low + self._diff / 2 + action * self._diff / 2

    def reverse_action(self, action):
        """
        Reverse action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        return (2 * action - self._high - self._low) / self._diff
