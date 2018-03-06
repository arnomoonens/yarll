# -*- coding: utf8 -*-

# Based on https://github.com/openai/baselines/blob/882251878f04e1eb1ac9da965772eca9c1ab272a/baselines/ddpg/memory.py

import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen: int, shape: tuple, dtype='float32'):
        self.maxlen: int = maxlen
        self.start: int = 0
        self.length: int = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit: int, action_shape: tuple, observation_shape: tuple):
        self.limit: int = limit

        self.states0: RingBuffer = RingBuffer(limit, shape=observation_shape)
        self.actions: RingBuffer = RingBuffer(limit, shape=action_shape)
        self.rewards: RingBuffer = RingBuffer(limit, shape=(1,))
        self.terminals1: RingBuffer = RingBuffer(limit, shape=(1,))
        self.states1: RingBuffer = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size: int):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.n_entries - 2, size=batch_size)

        state0_batch = self.states0.get_batch(batch_idxs)
        state1_batch = self.states1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'states0': array_min2d(state0_batch),
            'states1': array_min2d(state1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, state0, action, reward, state1, terminal1, training=True):
        if not training:
            return

        self.states0.append(state0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states1.append(state1)
        self.terminals1.append(terminal1)

    @property
    def n_entries(self):
        return len(self.states0)
