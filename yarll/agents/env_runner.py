# -*- coding: utf8 -*-
from typing import Any, List, Optional
import tensorflow as tf
import numpy as np
from yarll.memory.experiences_memory import ExperiencesMemory, Experience
from yarll.misc.utils import RunningMeanStd, normalize


class EnvRunner(object):
    """Environment runner using a policy"""
    def __init__(self, env, policy, config: dict, normalize_states=False, state_preprocessor=None, summary_writer=None) -> None:
        super(EnvRunner, self).__init__()
        self.env = env
        self.policy = policy
        self.state: Optional[np.ndarray] = None
        self.features: Any = policy.initial_features
        self.config: dict = dict(
            batch_update="timesteps",
            episode_max_length=env.spec.max_episode_steps,
            timesteps_per_batch=10000,
            n_iter=100
        )
        self.episode_steps: int = 0
        self.episode_reward: float = 0.0
        self.n_episodes: int = 0
        self.config.update(config)
        self.state_preprocessor = state_preprocessor
        self.summary_writer = summary_writer

        # Normalize states before giving it as input to the network.
        # Mean and std are only updated at the end of `get_steps`.
        self.normalize_states = normalize_states
        if normalize_states:
            self.rms = RunningMeanStd(self.env.observation_space.shape)
        self.reset_env()

    def choose_action(self, state: np.ndarray):
        """Choose an action based on the current state in the environment."""
        return self.policy.choose_action(state, self.features)

    def normalize(self, state: np.ndarray) -> np.ndarray:
        return normalize(state, self.rms.mean, self.rms.std)

    def reset_env(self) -> None:
        """Reset the current environment and get the initial state"""
        self.state = self.env.reset()
        self.state = self.state if self.state_preprocessor is None else self.state_preprocessor(self.state)

    def step_env(self, action):
        """Execute an action in the current environment."""
        state, reward, done, info = self.env.step(self.policy.get_env_action(action))
        state = state if self.state_preprocessor is None else self.state_preprocessor(state)
        return state, reward, done, info

    def get_steps(self, n_steps: int, reset: bool = False, stop_at_trajectory_end: bool = True, render: bool = False) -> ExperiencesMemory:
        if reset:
            self.reset_env()
            self.policy.new_trajectory()
        memory = ExperiencesMemory()
        for _ in range(n_steps):
            input_state = self.normalize(self.state) if self.normalize_states else self.state
            results = self.choose_action(input_state)
            action = results["action"]
            value = results.get("value", None)
            new_features = results.get("features", None)
            new_state, rew, done, _ = self.step_env(action)
            memory.add(self.state, action, rew, value, terminal=done, features=self.features)
            self.state = new_state
            self.features = new_features
            self.episode_reward += rew
            self.episode_steps += 1
            if done or self.episode_steps >= self.config["episode_max_length"]:
                if self.summary_writer is not None:
                    summary = tf.Summary()
                    summary.value.add(tag="global/Episode_length", simple_value=float(self.episode_steps))
                    summary.value.add(tag="global/Reward", simple_value=float(self.episode_reward))
                    self.summary_writer.add_summary(summary, self.n_episodes)
                    self.summary_writer.flush()
                self.episode_reward = 0
                self.episode_steps = 0
                self.n_episodes += 1
                self.reset_env()
                self.features = self.policy.initial_features
                self.policy.new_trajectory()
                # Decide whether to stop when the episode (=trajectory) is done
                # or to keep collecting until n_steps
                if stop_at_trajectory_end:
                    break
            if render:
                self.env.render()

        if self.normalize_states:
            self.rms.add_values([exp.state for exp in memory.experiences])
            for i, exp in enumerate(memory.experiences):
                memory.experiences[i] = Experience(self.normalize(exp.state),
                                                   exp.action,
                                                   exp.reward,
                                                   exp.value,
                                                   exp.features,
                                                   exp.terminal)
        return memory

    def get_trajectory(self, stop_at_trajectory_end: bool = True, render: bool = False) -> ExperiencesMemory:
        return self.get_steps(self.config["episode_max_length"], stop_at_trajectory_end, render)

    def get_trajectories(self, stop_at_trajectory_end: bool = True, render: bool = False) -> List[ExperiencesMemory]:
        """Generate trajectories until a certain number of timesteps or trajectories."""
        use_timesteps = self.config["batch_update"] == "timesteps"
        trajectories = []
        timesteps_total = 0
        i = 0
        while(use_timesteps and timesteps_total < self.config["timesteps_per_batch"]) or\
            (not(use_timesteps) and i < self.config["trajectories_per_batch"]):
            i += 1
            trajectory = self.get_trajectory(stop_at_trajectory_end, render)
            trajectories.append(trajectory)
            timesteps_total += len(trajectory.rewards)
        return trajectories
