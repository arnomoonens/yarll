from typing import Dict, List, Tuple
import numpy as np
from yarll.memory.experiences_memory import Experience

class PreAllocMemory:
    def __init__(self, max_size: int, observation_shape: Tuple[int], action_shape: Tuple[int], states_dtype = np.float32):
        """
        Initialize internal state.

        Args:
            self: (todo): write your description
            max_size: (int): write your description
            observation_shape: (todo): write your description
            action_shape: (todo): write your description
            states_dtype: (todo): write your description
            np: (int): write your description
            float32: (todo): write your description
        """
        self.max_size = max_size
        self._pointer = 0 # where to start writing new data
        self.n_entries = 0 # Number of filled rows
        self.data = {
            "states0": np.empty((self.max_size, *observation_shape), dtype=states_dtype),
            "actions": np.empty((self.max_size, *action_shape), dtype=np.float32),
            "rewards": np.empty((self.max_size), dtype=np.float32),
            "states1": np.empty((self.max_size, *observation_shape), dtype=states_dtype),
            "terminals1": np.empty((self.max_size), dtype=np.bool)
        }

    def reallocate(self, new_max_size: int):
        """
        Reallocate all the samples.

        Args:
            self: (todo): write your description
            new_max_size: (int): write your description
        """
        if new_max_size == self.max_size:
            return
        ids = np.arange(self.n_entries) % new_max_size
        for k, v in self.data.items():
            new_arr = np.empty((new_max_size, *v.shape[1:]), dtype=v.dtype)
            new_arr[ids] = v
            self.data[k] = new_arr
        self._pointer = self._pointer % new_max_size
        self.max_size = new_max_size

    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Return a batch of samples.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
        """
        # Randomly sample batch_size examples
        ids = np.random.randint(0, self.n_entries, batch_size)
        return {k: v[ids] for k, v in self.get_all().items()}

    def get_all(self) -> Dict[str, np.ndarray]:
        """
        Returns : class as a dictionary.

        Args:
            self: (str): write your description
        """
        return {
            "states0": self.data["states0"][:self.n_entries],
            "actions": self.data["actions"][:self.n_entries],
            "rewards": self.data["rewards"][:self.n_entries],
            "states1": self.data["states1"][:self.n_entries],
            "terminals1": self.data["terminals1"][:self.n_entries].astype(np.float32)
        }

    def _update(self, n_samples: int):
        """
        Updates n_s of the number of - samples.

        Args:
            self: (todo): write your description
            n_samples: (int): write your description
        """
        self._pointer = (self._pointer + n_samples) % self.max_size
        self.n_entries = min(self.n_entries + n_samples, self.max_size)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        """
        Adds an action.

        Args:
            self: (todo): write your description
            state: (todo): write your description
            np: (int): write your description
            ndarray: (array): write your description
            action: (int): write your description
            np: (int): write your description
            ndarray: (array): write your description
            reward: (float): write your description
            new_state: (todo): write your description
            np: (int): write your description
            ndarray: (array): write your description
            terminal: (bool): write your description
        """
        self.data["states0"][self._pointer] = state
        self.data["actions"][self._pointer] = action
        self.data["rewards"][self._pointer] = reward
        self.data["states1"][self._pointer] = new_state
        self.data["terminals1"][self._pointer] = terminal
        self._update(1)

    def add_by_experiences(self, experiences: List[Experience]) -> None:
        """
        Add the given transition to the list.

        Args:
            self: (todo): write your description
            experiences: (todo): write your description
        """
        for experience in experiences:
            self.add(experience.state, experience.action, experience.reward,
                     experience.next_state, experience.terminal)

    def add_by_arrays(self,
                      states: np.ndarray,
                      actions: np.ndarray,
                      rewards: np.ndarray,
                      new_states: np.ndarray,
                      terminals: np.ndarray) -> None:
        """
        Add new samples.

        Args:
            self: (todo): write your description
            states: (todo): write your description
            np: (todo): write your description
            ndarray: (array): write your description
            actions: (todo): write your description
            np: (todo): write your description
            ndarray: (array): write your description
            rewards: (todo): write your description
            np: (todo): write your description
            ndarray: (array): write your description
            new_states: (int): write your description
            np: (todo): write your description
            ndarray: (array): write your description
            terminals: (bool): write your description
            np: (todo): write your description
            ndarray: (array): write your description
        """
        n_samples = states.shape[0]
        ids = np.arange(self._pointer, self._pointer + n_samples) % self.max_size
        self.data["states0"][ids] = states
        self.data["actions"][ids] = actions
        self.data["rewards"][ids] = rewards
        self.data["states1"][ids] = new_states
        self.data["terminals1"][ids] = terminals
        self._update(n_samples)

    def erase(self):
        """
        Erase the current pointer.

        Args:
            self: (todo): write your description
        """
        self._pointer = 0
        self.n_entries = 0
