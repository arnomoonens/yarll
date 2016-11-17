# -*- coding: utf8 -*-
import numpy as np

class Learner(object):
    """Reinforcement Learner"""
    def __init__(self, ob_space, action_space, **usercfg):
        super(Learner, self).__init__()
        self.ob_space = ob_space
        self.action_space = action_space
        self.nO = ob_space.shape[0]
        self.nA = action_space.n
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
            obs.append(ob)
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

    def learn(self, env):
        pass
