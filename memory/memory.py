# -*- coding: utf8 -*-

from collections import deque
import random
import numpy as np

class Memory(object):

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def get_batch(self, batch_size: int) -> dict:
        # Randomly sample batch_size examples
        experiences = random.sample(self.buffer, batch_size)
        return {
            "states0": np.asarray([exp[0] for exp in experiences]),
            "actions": np.asarray([exp[1] for exp in experiences]),
            "rewards": np.asarray([exp[2] for exp in experiences]),
            "states1": np.asarray([exp[3] for exp in experiences]),
            "terminals1": np.asarray([exp[4] for exp in experiences])
        }

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    @property
    def size(self) -> int:
        return self.buffer_size

    @property
    def n_entries(self) -> int:
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
