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

    def act(self, ob):
        pass

    def get_trajectory(self, env, episode_max_length, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        """
        ob = env.reset()
        obs = []
        actions = []
        rewards = []
        for _ in range(episode_max_length):
            action = self.act(ob)
            obs.append(ob.flatten())
            (ob, rew, done, _) = env.step(action)
            actions.append(action)
            rewards.append(rew)
            if done:
                break
            if render:
                env.render()
        return {"reward": np.array(rewards),
                "ob": np.array(obs),
                "action": np.array(actions)
                }

    def get_trajectories(self, env):
        trajectories = []
        timesteps_total = 0
        while timesteps_total < self.config["timesteps_per_batch"]:
            trajectory = self.get_trajectory(env, self.config["episode_max_length"])
            trajectories.append(trajectory)
            timesteps_total += len(trajectory["reward"])
        return trajectories

    def learn(self, env):
        pass
