from typing import Union
import numpy as np

class Scaler:
    def fit(self, data):
        """
        Parameters ---------- data : a dictionary.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        pass

    def fit_single(self, data):
        """
        Fit a single dataset.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        pass

    def scale(self, x: np.ndarray) -> np.ndarray:
        """
        Scale x. nd.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            np: (todo): write your description
            ndarray: (array): write your description
        """
        raise NotImplementedError()

    def descale(self, x_scaled: np.ndarray) -> np.ndarray:
        """
        Descale x_scaled.

        Args:
            self: (todo): write your description
            x_scaled: (float): write your description
            np: (todo): write your description
            ndarray: (array): write your description
        """
        pass


class LowsHighsScaler(Scaler):
    """
    Scale using lows and highs such that the scaled outputs are between [-1, 1].
    """

    def __init__(self, lows: np.ndarray, highs: np.ndarray):
        """
        Initialize the texture states.

        Args:
            self: (todo): write your description
            lows: (todo): write your description
            np: (todo): write your description
            ndarray: (array): write your description
            highs: (int): write your description
            np: (todo): write your description
            ndarray: (array): write your description
        """
        assert all(np.isfinite(lows)) and all(np.isfinite(highs)), "Lows and highs must be finite numbers."
        self.lows = lows
        self.highs = highs

    def scale(self, x: np.ndarray) -> np.ndarray:
        """Scales values such that they're between [-1, 1]."""
        return np.nan_to_num(-1 + (x - self.lows) / (self.highs - self.lows) * 2)

    def descale(self, x_scaled: np.ndarray) -> np.ndarray:
        """Descale values from [-1, 1] to the original range."""
        return self.lows + (x_scaled + 1) / 2 * (self.highs - self.lows)

number_array = Union[int, float, np.ndarray]

class RunningMeanStdScaler(Scaler):
    """
    Calculates the running mean and standard deviation of values of shape `shape`.
    """

    def __init__(self, shape, epsilon=1e-2):
        """
        Initialize shape.

        Args:
            self: (todo): write your description
            shape: (int): write your description
            epsilon: (float): write your description
        """
        super(RunningMeanStdScaler, self).__init__()
        self.count = epsilon
        self._sum = np.zeros(shape, dtype="float64")
        self._sumsq = np.full(shape, epsilon, dtype="float64")

    def fit_single(self, data):
        """
        Update count, sum and sum squared using a new value `data`.
        """
        data = np.asarray(data, dtype="float64")
        self.count += 1
        self._sum += data
        self._sumsq += np.square(data)

    def fit(self, data):
        """
        Update count, sum and sum squared using multiple values `data`.
        """
        data = np.asarray(data, dtype="float64")
        self.count += np.shape(data)[0]
        self._sum += np.sum(data, axis=0)
        self._sumsq += np.square(data).sum(axis=0)

    @property
    def mean(self):
        """
        The mean of the distribution

        Args:
            self: (todo): write your description
        """
        return self._sum / self.count

    @property
    def std(self):
        """
        The standard deviation.

        Args:
            self: (todo): write your description
        """
        return np.sqrt(np.maximum((self._sumsq / self.count) - np.square(self.mean), 1e-2))

    def scale(self, x: number_array) -> Union[float, np.ndarray]:
        """
        Scale x to new array.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        if isinstance(x, np.ndarray):
            x = x.astype("float64")
        return np.clip((x - self.mean) / self.std, -5.0, 5.0)
