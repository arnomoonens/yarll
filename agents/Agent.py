# -*- coding: utf8 -*-
import numpy as np

class Agent(object):
    """Reinforcement Agent"""
    def __init__(self, env, **usercfg):
        super(Agent, self).__init__()
        self.env = env
        self.ob_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.nO = self.ob_space.shape[0]
        self.config = dict(
            episode_max_length=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"),
            timesteps_per_batch=10000,
            n_iter=100)
        self.config.update(usercfg)

    def choose_action(self, state):
        """Return which action to take based on the given state"""
        pass

    def reset_env(self):
        """Reset the current environment and get the initial state"""
        return self.env.reset()

    def step_env(self, action):
        """Execute an action in the current environment."""
        return self.env.step(action)

    def get_trajectory(self, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        """
        state = self.reset_env()
        states = []
        actions = []
        rewards = []
        for i in range(self.config["episode_max_length"]):
            action = self.choose_action(state)
            states.append(state)
            for _ in range(self.config["repeat_n_actions"]):
                state, rew, done, _ = self.step_env(action)
                if done:  # Don't continue if episode has already ended
                    break
            actions.append(action)
            rewards.append(rew)
            if done:
                break
            if render:
                self.env.render()
        return {"reward": np.array(rewards),
                "state": np.array(states),
                "action": np.array(actions),
                "done": done,  # Tajectory ended because a terminal state was reached
                "steps": i + 1
                }

    def get_trajectories(self):
        """Generate trajectories until a certain number of timesteps or trajectories."""
        use_timesteps = self.config["batch_update"] == "timesteps"
        trajectories = []
        timesteps_total = 0
        i = 0
        while (use_timesteps and timesteps_total < self.config["timesteps_per_batch"]) or (not(use_timesteps) and i < self.config["trajectories_per_batch"]):
            i += 1
            trajectory = self.get_trajectory()
            trajectories.append(trajectory)
            timesteps_total += len(trajectory["reward"])
        return trajectories

    def learn(self):
        """Learn in the current environment."""
        pass
