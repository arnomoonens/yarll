# -*- coding: utf8 -*-

import tensorflow as tf

class Trajectory(object):
    """Experience gathered from an environment."""
    def __init__(self):
        super(Trajectory, self).__init__()
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.features = []
        self.terminals = []
        self.steps = 0

    def add(self, state, action, reward, value=None, features=None, terminal=False):
        """Add a single transition to the trajectory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.features.append(features)
        self.terminals.append(terminal)
        self.steps += 1

    def extend(self, other):
        """
        Extend a trajectory with another one
        given that the current one hasn't ended yet.
        """
        if self.terminal:
            raise AssertionError("Can't extend a terminal trajectory.")
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.features.extend(other.features)
        self.terminal.extend(other.terminals)
        self.steps += other.steps

class EnvRunner(object):
    """Environment runner using a policy"""
    def __init__(self, env, policy, config, state_preprocessor=None, summary_writer=None):
        super(EnvRunner, self).__init__()
        self.env = env
        self.policy = policy
        self.features = policy.initial_features
        self.config = dict(
            batch_update="timesteps",
            episode_max_length=env.spec.max_episode_steps,
            timesteps_per_batch=10000,
            n_iter=100
        )
        self.episode_steps = 0
        self.episode_reward = 0
        self.n_episodes = 0
        self.config.update(config)
        self.state_preprocessor = state_preprocessor
        self.summary_writer = summary_writer
        self.reset_env()

    def choose_action(self, state):
        """Choose an action based on the current state in the environment."""
        return self.policy.choose_action(state, self.features)

    def reset_env(self):
        """Reset the current environment and get the initial state"""
        self.state = self.env.reset()
        self.state = self.state if self.state_preprocessor is None else self.state_preprocessor(self.state)

    def step_env(self, action):
        """Execute an action in the current environment."""
        state, reward, done, info = self.env.step(self.policy.get_env_action(action))
        state = state if self.state_preprocessor is None else self.state_preprocessor(state)
        return state, reward, done, info

    def get_steps(self, n_steps, reset=False, stop_at_trajectory_end=True, render=False):
        if reset:
            self.reset_env()
            self.policy.new_trajectory()
        traj = Trajectory()
        for i in range(n_steps):
            results = self.choose_action(self.state)
            action = results["action"]
            value = results.get("value", None)
            new_features = results.get("features", None)
            new_state, rew, done, _ = self.step_env(action)
            traj.add(self.state, action, rew, value, terminal=done, features=self.features)
            self.state = new_state
            self.features = new_features
            self.episode_reward += rew
            self.episode_steps += 1
            if done:
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
        return traj

    def get_trajectory(self, stop_at_trajectory_end=True, render=False):
        return self.get_steps(self.config["episode_max_length"], stop_at_trajectory_end, render)

    def get_trajectories(self, stop_at_trajectory_end=True, render=False):
        """Generate trajectories until a certain number of timesteps or trajectories."""
        use_timesteps = self.config["batch_update"] == "timesteps"
        trajectories = []
        timesteps_total = 0
        i = 0
        while (use_timesteps and timesteps_total < self.config["timesteps_per_batch"]) or (not(use_timesteps) and i < self.config["trajectories_per_batch"]):
            i += 1
            trajectory = self.get_trajectory(stop_at_trajectory_end, render)
            trajectories.append(trajectory)
            timesteps_total += len(trajectory.rewards)
        return trajectories
