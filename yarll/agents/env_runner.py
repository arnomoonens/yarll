# -*- coding: utf8 -*-
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import tensorflow as tf
import numpy as np
from yarll.memory.experiences_memory import ExperiencesMemory, Experience
from yarll.misc.scalers import LowsHighsScaler, RunningMeanStdScaler
from yarll.misc.utils import memory_usage

class EnvRunner(object):
    """Environment runner using a policy"""
    def __init__(self,
                 env,
                 policy,
                 config: Dict[str, Any],
                 scale_states: bool = False,
                 state_preprocessor: Optional[Callable] = None,
                 summaries: bool = True, # Write tensorflow summaries
                 memory_usage: bool = True, # Include memory usage in summaries
                 episode_rewards_file: Optional[Union[Path, str]] = None, # write each episode reward to the given file
                 ) -> None:
        super().__init__()
        self.env = env
        self.policy = policy
        self.state: Optional[np.ndarray] = None
        self.features: Any = policy.initial_features
        self.config: dict = dict(
            batch_update="timesteps",
            episode_max_length=env.spec.max_episode_steps if env.spec is not None and env.spec.max_episode_steps is not None else np.inf,
            timesteps_per_batch=10000,
            n_iter=100
        )
        self.episode_steps: int = 0
        self.episode_reward: float = 0.0
        self.episodes_rewards: List[float] = []
        self.config.update(config)
        self.state_preprocessor = state_preprocessor
        self.summaries = summaries
        self.memory_usage = memory_usage
        self.episode_rewards_file = episode_rewards_file
        self.total_steps = 0
        self.total_episodes = 0

        # Normalize states before giving it as input to the network.
        # Mean and std are only updated at the end of `get_steps`.
        self.scale_states = scale_states
        if scale_states:
            # check obs space
            if all(np.isfinite(np.append(env.observation_space.low, env.observation_space.high))):
                self.state_scaler = LowsHighsScaler(env.observation_space.low, env.observation_space.high)
            else:
                print("Warning: No observation space bounds, scaling with RunningMeanStdScaler.")
                self.state_scaler = RunningMeanStdScaler(self.env.observation_space.shape)
        self.reset_env()

    def choose_action(self, state: np.ndarray):
        """Choose an action based on the current state in the environment."""
        return self.policy.choose_action(state, self.features)

    def scale_state(self, state: np.ndarray) -> np.ndarray:
        return self.state_scaler.scale(state)

    def reset_env(self) -> None:
        """Reset the current environment and get the initial state"""
        self.state = self.env.reset()
        self.state = self.state if self.state_preprocessor is None else self.state_preprocessor(self.state)

    def step_env(self, action):
        """
        Execute an action in the current environment.
        ! Not used in get_steps right now.
        """
        state, reward, done, info = self.env.step(self.policy.get_env_action(action))
        state = state if self.state_preprocessor is None else self.state_preprocessor(state)
        return state, reward, done, info

    def get_steps(self, n_steps: int, reset: bool = False, stop_at_trajectory_end: bool = True, render: bool = False) -> ExperiencesMemory:
        if reset:
            self.reset_env()
            self.policy.new_trajectory()
        memory = ExperiencesMemory()
        for _ in range(n_steps):
            input_state = np.asarray(self.state, dtype=np.float32)
            input_state = self.scale_state(input_state) if self.scale_states else input_state
            results = self.choose_action(input_state)
            action = results["action"]
            value = results.get("value", None)
            new_features = results.get("features", None)
            env_action = self.policy.get_env_action(action) # e.g., unnormalized action
            new_state, rew, done, _ = self.env.step(env_action)
            memory.add(self.state, action, rew, value, terminal=done, features=self.features, next_state=new_state)
            self.state = new_state
            self.features = new_features
            self.episode_reward += rew
            self.episode_steps += 1
            self.total_steps += 1
            if done or self.episode_steps >= self.config["episode_max_length"]:
                self.total_episodes += 1
                # summaries won't be written if there is no writer.as_default around it somewhere (e.g. in algorithm itself)
                if self.summaries:
                    tf.summary.scalar("env/episode_length", self.episode_steps, step=self.total_steps)
                    tf.summary.scalar("env/episode_reward", self.episode_reward, step=self.total_steps)
                    tf.summary.scalar("env/total_episodes", self.total_episodes, step=self.total_steps)
                    if self.memory_usage:
                        tf.summary.scalar("diagnostics/memory_usage_mb", memory_usage() / 1e6, step=self.total_steps)
                if self.episode_rewards_file is not None:
                    with open(self.episode_rewards_file, "a") as f:
                        f.write(f"{self.episode_reward}\n")
                self.episodes_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.episode_steps = 0
                self.reset_env()
                self.features = self.policy.initial_features
                self.policy.new_trajectory()
                # Decide whether to stop when the episode (=trajectory) is done
                # or to keep collecting until n_steps
                if stop_at_trajectory_end:
                    break
            if render:
                self.env.render()

        if self.scale_states:
            self.state_scaler.fit([exp.state for exp in memory.experiences])
            for i, exp in enumerate(memory.experiences):
                memory.experiences[i] = Experience(self.scale_state(exp.state),
                                                   exp.action,
                                                   exp.reward,
                                                   self.scale_state(exp.next_state),
                                                   exp.value,
                                                   exp.features,
                                                   exp.terminal)
        return memory

    def get_trajectory(self, stop_at_trajectory_end: bool = True, render: bool = False) -> ExperiencesMemory:
        return self.get_steps(self.config["episode_max_length"],
                              stop_at_trajectory_end=stop_at_trajectory_end,
                              render=render)

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
