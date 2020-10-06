from typing import Dict, List, Tuple
import numpy as np
from yarll.memory.experiences_memory import Experience

class PreAllocMemory:
    def __init__(self, max_size: int, observation_shape: Tuple[int], action_shape: Tuple[int], states_dtype = np.float32):
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
        # Randomly sample batch_size examples
        ids = np.random.randint(0, self.n_entries, batch_size)
        return {k: v[ids] for k, v in self.get_all().items()}

    def get_all(self) -> Dict[str, np.ndarray]:
        return {
            "states0": self.data["states0"][:self.n_entries],
            "actions": self.data["actions"][:self.n_entries],
            "rewards": self.data["rewards"][:self.n_entries],
            "states1": self.data["states1"][:self.n_entries],
            "terminals1": self.data["terminals1"][:self.n_entries].astype(np.float32)
        }

    def _update(self, n_samples: int):
        self._pointer = (self._pointer + n_samples) % self.max_size
        self.n_entries = min(self.n_entries + n_samples, self.max_size)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        self.data["states0"][self._pointer] = state
        self.data["actions"][self._pointer] = action
        self.data["rewards"][self._pointer] = reward
        self.data["states1"][self._pointer] = new_state
        self.data["terminals1"][self._pointer] = terminal
        self._update(1)

    def add_by_experiences(self, experiences: List[Experience]) -> None:
        for experience in experiences:
            self.add(experience.state, experience.action, experience.reward,
                     experience.next_state, experience.terminal)

    def add_by_arrays(self,
                      states: np.ndarray,
                      actions: np.ndarray,
                      rewards: np.ndarray,
                      new_states: np.ndarray,
                      terminals: np.ndarray) -> None:
        n_samples = states.shape[0]
        ids = np.arange(self._pointer, self._pointer + n_samples) % self.max_size
        self.data["states0"][ids] = states
        self.data["actions"][ids] = actions
        self.data["rewards"][ids] = rewards
        self.data["states1"][ids] = new_states
        self.data["terminals1"][ids] = terminals
        self._update(n_samples)

    def erase(self):
        self._pointer = 0
        self.n_entries = 0
