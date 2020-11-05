# -*- coding: utf8 -*-

import numpy as np

class OrnsteinUhlenbeckActionNoise(object):
    """
    Ornstein Uhlenbeck process action noise.
    """

    def __init__(self, n_actions: int, sigma: float, theta: float = .15, dt: float = 1e-2, x0=None) -> None:
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            n_actions: (todo): write your description
            sigma: (float): write your description
            theta: (todo): write your description
            dt: (float): write your description
            x0: (float): write your description
        """
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.theta: float = theta
        self.mu: np.ndarray = np.zeros(n_actions)
        self.sigma: float = sigma
        self.dt: float = dt
        self.x0: float = x0
        self.x_prev = None
        self.reset()

    def __call__(self) -> np.ndarray:
        """
        Return self ( x x.

        Args:
            self: (todo): write your description
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """
        Reset the state.

        Args:
            self: (todo): write your description
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self) -> str:
        """
        Return a human - readable representation of this parameter.

        Args:
            self: (todo): write your description
        """
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
