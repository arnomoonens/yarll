import numpy as np

from FunctionApproximation.FunctionApproximator import FunctionApproximator

class TileCoding(FunctionApproximator):
    """Map states to tiles"""
    def __init__(self, x_low, x_high, y_low, y_high, n_tilings, n_y_tiles, n_x_tiles, n_actions):
        super(TileCoding, self).__init__(n_actions)
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high
        self.n_x_tiles = n_x_tiles
        self.n_y_tiles = n_y_tiles
        self.n_tilings = n_tilings
        self.n_actions = n_actions

        if self.n_x_tiles % 1 != 0 or self.n_x_tiles <= 0:
            raise TypeError("Number of x tiles must be a positive natural number instead of {}".format(self.n_x_tiles))
        if self.n_y_tiles % 1 != 0 or self.n_y_tiles <= 0:
            raise TypeError("Number of y tiles must be a positive natural number instead of {}".format(self.n_y_tiles))

        self.tile_width = (self.x_high - self.x_low) / self.n_x_tiles
        self.tile_height = (self.y_high - self.y_low) / self.n_y_tiles

        self.tiling_width = self.tile_width * self.n_x_tiles
        self.tiling_height = self.tile_height * self.n_y_tiles

        self.tile_starts = []

        for _ in range(self.n_tilings):  # Each tiling starts at a random offset that is a fraction of the tile width and height
            self.tile_starts.append((self.x_low + np.random.rand() * self.tile_width, self.y_low + np.random.rand() * self.tile_height))

        self.features_shape = (self.n_tilings, self.n_y_tiles, self.n_x_tiles, self.n_actions)
        self.thetas = np.random.uniform(size=self.features_shape)  # Initialise randomly with values between 0 and 1

    def summed_thetas(self, state, action):
        """Theta values for features present for state and action."""
        summed = 0
        for i in range(self.n_tilings):
            shifted = state - self.tile_starts[i]  # Subtract the randomly chosen offsets
            x, y = shifted
            if (x >= 0 and x <= self.tiling_width) and (y >= 0 and y <= self.tiling_height):
                summed += self.thetas[i][int(y // self.tile_height)][int(x // self.tile_width)][action]
        return summed

    def present_features(self, state, action):
        """Features that are active for the given state and action."""
        result = np.zeros(self.thetas.shape)  # By default, all of them are inactve
        for i in range(self.n_tilings):
            shifted = state - self.tile_starts[i]
            x, y = shifted
            if (x >= 0 and x <= self.tiling_width) and (y >= 0 and y <= self.tiling_height):
                result[i][int(y // self.tile_height)][int(x // self.tile_width)][action] = 1  # Set the feature to active
        return result
