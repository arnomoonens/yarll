#!/usr/bin/env python
# -*- coding: utf8 -*-

#  Cross-Entropy Method
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html
import sys
import argparse

import numpy as np
import gym
from gym import wrappers
from gym.spaces import Discrete, Box

from Learner import Learner

# ================================================================
# Policies
# ================================================================

class DeterministicDiscreteActionLinearPolicy(object):
    """Deterministicially select an action from a discrete action space using a linear function."""
    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0: dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions: None].reshape(1, n_actions)

    def act(self, ob):
        """Select the action that got the highest value from the linear function."""
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a

class DeterministicContinuousActionLinearPolicy(object):
    """Deterministicially select an action from a continuous action space using a linear function."""
    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0:dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac:None]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a

class CEMLearner(Learner):
    """Cross-Entropy Method learner"""
    def __init__(self, env, **usercfg):
        super(CEMLearner, self).__init__(env, **usercfg)
        self.config = dict(
                                num_steps = 500,  # maximum length of episode
                                n_iter = 100,  # number of iterations of CEM
                                batch_size = 25,  # number of samples per batch
                                elite_frac = 0.2  # fraction of samples used as elite set
                            )
        if isinstance(env.action_space, Discrete):
            self.dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.n
        elif isinstance(env.action_space, Box):
            self.dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.shape[0]
        else:
            raise NotImplementedError
        # Initialize mean and standard deviation
        self.theta_mean = np.zeros(self.dim_theta)
        self.theta_std = np.ones(self.dim_theta)

    def make_policy(self, theta):
        if isinstance(self.env.action_space, Discrete):
            return DeterministicDiscreteActionLinearPolicy(theta, self.env.observation_space, self.env.action_space)
        elif isinstance(self.env.action_space, Box):
            return DeterministicContinuousActionLinearPolicy(theta, self.env.observation_space, self.env.action_space)
        else:
            raise NotImplementedError

    def noisy_evaluation(self, theta):
        policy = self.make_policy(theta)
        rew = self.do_episode(policy)
        return rew

    def do_episode(self, policy):
        total_rew = 0
        ob = self.env.reset()
        for t in range(self.config["num_steps"]):
            a = policy.act(ob)
            (ob, reward, done, _info) = self.env.step(a)
            total_rew += reward
            if done:
                break
        return total_rew

    def learn(self):
        for iteration in range(self.config["n_iter"]):
            # Sample parameter vectors
            thetas = [np.random.normal(self.theta_mean, self.theta_std, self.dim_theta) for _ in range(self.config["batch_size"])]
            rewards = [self.noisy_evaluation(theta) for theta in thetas]
            # Get elite parameters
            n_elite = int(self.config["batch_size"] * self.config["elite_frac"])
            elite_inds = np.argsort(rewards)[self.config["batch_size"] - n_elite:self.config["batch_size"]]
            elite_thetas = [thetas[i] for i in elite_inds]
            # Update theta_mean, theta_std
            theta_mean = np.mean(elite_thetas, axis=0)
            theta_std = np.std(elite_thetas, axis=0)
            print("iteration %i. mean f: %8.3g. max f: %8.3g" % (iteration, np.mean(rewards), np.max(rewards)))
            self.do_episode(self.make_policy(self.theta_mean))

parser = argparse.ArgumentParser()
parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the experiment on.")
parser.add_argument("monitor_path", metavar="monitor_path", type=str, help="Path where Gym monitor files may be saved")

def main():
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    env = gym.make(args.environment)
    if isinstance(env.action_space, Discrete):
        # action_selection = ProbabilisticCategoricalActionSelection()
        agent = CEMLearner(env, episode_max_length=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"))
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    else:
        raise NotImplementedError
    try:
        agent.env = wrappers.Monitor(agent.env, args.monitor_path, force=True)
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
