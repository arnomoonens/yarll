# -*- coding: utf8 -*-

from collections import deque
import random
from typing import Dict, List
import numpy as np

from yarll.memory.experiences_memory import Experience

class Memory:

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        # Randomly sample batch_size examples
        experiences = random.choices(self.buffer, k=batch_size)
        return {
            "states0": np.asarray([exp[0] for exp in experiences], np.float32),
            "actions": np.asarray([exp[1] for exp in experiences], np.float32),
            "rewards": np.asarray([exp[2] for exp in experiences], np.float32),
            "states1": np.asarray([exp[3] for exp in experiences], np.float32),
            "terminals1": np.asarray([exp[4] for exp in experiences], np.float32)
        }

    def get_all(self) -> Dict[str, np.ndarray]:
        return {
            "states0": np.asarray([exp[0] for exp in self.buffer], np.float32),
            "actions": np.asarray([exp[1] for exp in self.buffer], np.float32),
            "rewards": np.asarray([exp[2] for exp in self.buffer], np.float32),
            "states1": np.asarray([exp[3] for exp in self.buffer], np.float32),
            "terminals1": np.asarray([exp[4] for exp in self.buffer], np.float32)
        }

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool) -> None:
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def add_by_experiences(self, experiences: List[Experience]) -> None:
        for experience in experiences:
            self.add(experience.state, experience.action, experience.reward,
                     experience.next_state, experience.terminal)

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
