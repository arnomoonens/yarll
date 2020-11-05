# -*- coding: utf8 -*-

from collections import deque
import random
from typing import Dict, List, Optional, Union
import numpy as np

from yarll.memory.experiences_memory import Experience

class Memory:

    def __init__(self, buffer_size: Optional[int] = None) -> None:
        """
        Initialize the buffer.

        Args:
            self: (todo): write your description
            buffer_size: (int): write your description
        """
        self.buffer_size = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Returns a batch of data.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
        """
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
        """
        Return a dictionary of the buffer names.

        Args:
            self: (str): write your description
        """
        return {
            "states0": np.asarray([exp[0] for exp in self.buffer], np.float32),
            "actions": np.asarray([exp[1] for exp in self.buffer], np.float32),
            "rewards": np.asarray([exp[2] for exp in self.buffer], np.float32),
            "states1": np.asarray([exp[3] for exp in self.buffer], np.float32),
            "terminals1": np.asarray([exp[4] for exp in self.buffer], np.float32)
        }

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool) -> None:
        """
        Add an action to the buffer.

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
            done: (int): write your description
        """
        experience = (state, action, reward, new_state, done)
        if self.buffer_size is None or self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

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

    @property
    def size(self) -> Union[int, None]:
        """
        Returns the size of the buffer.

        Args:
            self: (todo): write your description
        """
        return self.buffer_size

    @property
    def n_entries(self) -> int:
        """
        The number of num entries.

        Args:
            self: (todo): write your description
        """
        return self.num_experiences

    def erase(self):
        """
        Erase the next buffer.

        Args:
            self: (todo): write your description
        """
        self.buffer = deque()
        self.num_experiences = 0
