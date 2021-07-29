# -*- coding: utf8 -*-
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from yarll.memory.experiences_memory import ExperiencesMemory
from yarll.misc.utils import memory_usage
from yarll.misc import summary_writer

class EnvRunner:
    """Environment runner using a policy"""

    def __init__(self,
                 env,
                 policy,
                 config: Dict[str, Any],
                 state_dtype = np.float32, # Datatype in which states are saved and given to policy
                 transition_preprocessor: Optional[Callable] = None,
                 summaries: bool = True, # Write summaries
                 summaries_every_episodes: Optional[int] = None, # If None, write one every episode
                 memory_usage_summary: bool = True,  # Include memory usage in summaries
                 episode_rewards_file: Optional[Union[Path, str]] = None,  # write each episode reward to the given file
                 ) -> None:
        super().__init__()
        self.env = env
        self.policy = policy
        self.state_dtype = np.dtype(state_dtype)
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
        self.transition_preprocessor = transition_preprocessor if transition_preprocessor is not None else \
            lambda state, action, reward, new_state, done, info: (state, action, reward, new_state, done)
        self.summaries = summaries
        self.summaries_every_episodes = summaries_every_episodes or 1 # If None, write one every episode, i.e. ep % 1 == 0
        self.memory_usage_summary = memory_usage_summary
        self.episode_rewards_file = episode_rewards_file
        self.total_steps = 0
        self.total_episodes = 0

        self.reset_env()

    def choose_action(self, state: np.ndarray):
        """Choose an action based on the current state in the environment."""
        return self.policy.choose_action(state, self.features)

    def reset_env(self) -> None:
        """Reset the current environment and get the initial state"""
        self.state = self.env.reset()
        self.state = np.asarray(self.state, dtype=self.state_dtype)

    def step_env(self, action):
        """
        Execute an action in the current environment.
        """
        state, reward, done, info = self.env.step(self.policy.get_env_action(action))
        self.state = np.asarray(self.state, dtype=self.state_dtype)
        return state, reward, done, info

    def get_steps(self,
                  n_steps: int,
                  reset: bool = False,
                  stop_at_trajectory_end: bool = True,
                  render: bool = False) -> ExperiencesMemory:
        if reset:
            self.reset_env()
            self.policy.new_trajectory()
        memory = ExperiencesMemory()
        for _ in range(n_steps):
            results = self.choose_action(self.state)
            action = results["action"]
            value = results.get("value", None)
            new_features = results.get("features", None)
            env_action = self.policy.get_env_action(action) # e.g., unnormalized action
            new_state, rew, done, info = self.env.step(env_action)
            new_state = np.asarray(new_state, dtype=self.state_dtype)
            proc_state, proc_action, proc_rew, proc_new_state, proc_done = self.transition_preprocessor(self.state,
                                                                                                        action,
                                                                                                        rew,
                                                                                                        new_state,
                                                                                                        done,
                                                                                                        info)
            memory.add(proc_state, proc_action, proc_rew, value, terminal=proc_done, features=self.features,
                       next_state=proc_new_state)
            self.state = new_state
            self.features = new_features
            self.episode_reward += rew
            self.episode_steps += 1
            self.total_steps += 1
            if done or self.episode_steps >= self.config["episode_max_length"]:
                self.total_episodes += 1
                # summaries won't be written if there is no writer.as_default around it somewhere (e.g. in algorithm itself)
                if self.summaries and (self.total_episodes % self.summaries_every_episodes) == 0:
                    summary_writer.add_scalar("env/episode_length", self.episode_steps, self.total_steps)
                    summary_writer.add_scalar("env/episode_reward", self.episode_reward, self.total_steps)
                    summary_writer.add_scalar("env/total_episodes", self.total_episodes, self.total_steps)
                    if self.memory_usage_summary:
                        summary_writer.add_scalar("diagnostics/memory_usage_mb",
                                                  memory_usage() / 1e6, self.total_steps)
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
