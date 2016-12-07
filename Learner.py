# -*- coding: utf8 -*-
import numpy as np

class Learner(object):
    """Reinforcement Learner"""
    def __init__(self, env, **usercfg):
        super(Learner, self).__init__()
        self.env = env
        self.ob_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.nO = self.ob_space.shape[0]
        # self.nA = action_space.n
        self.config = dict(
            episode_max_length=100,
            timesteps_per_batch=10000,
            n_iter=100)
        self.config.update(usercfg)

    def act(self, state):
        pass

    def get_trajectory(self, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        """
        env = self.env
        state = env.reset()
        states = []
        actions = []
        rewards = []
        for _ in range(self.config['episode_max_length']):
            action = self.act(state)
            states.append(state.flatten())
            for _ in range(self.config['repeat_n_actions']):
                state, rew, done, _ = env.step(action)
                if done:  # Don't continue if episode has already ended
                    break
            actions.append(action)
            rewards.append(rew)
            if done:
                break
            if render:
                env.render()
        return {"reward": np.array(rewards),
                "state": np.array(states),
                "action": np.array(actions),
                "done": done  # Tajectory ended because a terminal state was reached
                }

    def get_trajectories(self):
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
        pass
