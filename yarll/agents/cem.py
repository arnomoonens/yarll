# -*- coding: utf8 -*-

#  Cross-Entropy Method
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html

from pathlib import Path
import numpy as np
from gym import wrappers
from gym.spaces import Discrete, Box, MultiBinary
import tensorflow as tf # for summaries

from yarll.agents.agent import Agent
from yarll.misc.exceptions import WrongShapeError

# ================================================================
# Policies
# ================================================================

class Policy(object):
    """
    Cross-Entropy Method policy.
    """

    def act(self, ob):
        """
        Decide which action to take given an observation.
        """
        raise NotImplementedError()

class DeterministicDiscreteActionLinearPolicy(Policy):
    """Deterministicially select an action from a discrete action space using a linear function."""
    def __init__(self, theta, ob_space, ac_space) -> None:
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        expected_shape = (dim_ob + 1) * n_actions
        if len(theta) != expected_shape:
            raise WrongShapeError("Expected a theta of length {} instead of {}".format(expected_shape, len(theta)))
        self.W = theta[0: dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions: None].reshape(1, n_actions)

    def act(self, ob):
        """Select the action that got the highest value from the linear function."""
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a


class DeterministicMultiBinaryActionLinearPolicy(Policy):
    """Deterministicially select an action from a discrete action space using a linear function."""

    def __init__(self, theta, ob_space, ac_space) -> None:
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        expected_shape = (dim_ob + 1) * n_actions
        if len(theta) != expected_shape:
            raise WrongShapeError("Expected a theta of length {} instead of {}".format(expected_shape, len(theta)))
        self.W = theta[0: dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions: None].reshape(1, n_actions)

    def act(self, ob):
        """Select the action that got the highest value from the linear function."""
        y = ob.dot(self.W) + self.b
        y = 1 / (1 + np.exp(-y))
        a = (y >= 0.5).astype(np.int32)
        return a

class DeterministicContinuousActionLinearPolicy(Policy):
    """Deterministicially select an action from a continuous action space using a linear function."""
    def __init__(self, theta, ob_space, ac_space) -> None:
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        expected_shape = (dim_ob + 1) * dim_ac
        if len(theta) != expected_shape:
            raise WrongShapeError("Expected a theta of length {} instead of {}".format(expected_shape, len(theta)))
        self.W = theta[0:dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac:None]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a

class CEM(Agent):
    """Cross-Entropy Method learner"""
    def __init__(self, env, monitor_path: str, video: bool = True, **usercfg) -> None:
        super(CEM, self).__init__(**usercfg)
        self.monitor_path = Path(monitor_path)
        self.env = wrappers.Monitor(env, monitor_path, force=True, video_callable=(None if video else False))
        self.config.update(dict(
            num_steps=env.spec.max_episode_steps,  # maximum length of episode
            n_iter=100,  # number of iterations of CEM
            batch_size=25,  # number of samples per batch
            elite_frac=0.2  # fraction of samples used as elite set
        ))
        self.config.update(usercfg)

        if isinstance(env.action_space, Discrete):
            self.dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.n
        elif isinstance(env.action_space, Box):
            self.dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.shape[0]
        elif isinstance(env.action_space, MultiBinary):
            self.dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.n
        else:
            raise NotImplementedError
        # Initialize mean and standard deviation
        self.theta_mean = np.zeros(self.dim_theta)
        self.theta_std = np.ones(self.dim_theta)

        self.total_steps = 0
        self.total_episodes = 0
        self.writer = tf.summary.create_file_writer(str(monitor_path)) # pylint: disable=no-member

    def make_policy(self, theta) -> Policy:
        if isinstance(self.env.action_space, Discrete):
            return DeterministicDiscreteActionLinearPolicy(theta, self.env.observation_space, self.env.action_space)
        elif isinstance(self.env.action_space, Box):
            return DeterministicContinuousActionLinearPolicy(theta, self.env.observation_space, self.env.action_space)
        elif isinstance(self.env.action_space, MultiBinary):
            return DeterministicMultiBinaryActionLinearPolicy(theta, self.env.observation_space, self.env.action_space)
        else:
            raise NotImplementedError

    def noisy_evaluation(self, theta) -> float:
        policy: Policy = self.make_policy(theta)
        rew = self.do_episode(policy)
        return rew

    def do_episode(self, policy: Policy, render=False):
        total_rew = 0
        ob = self.env.reset()
        done = False
        while not done:
            a = policy.act(ob)[0]
            ob, reward, done, _info = self.env.step(a)
            self.total_steps += 1
            if render:
                self.env.render()
            total_rew += reward
        self.total_episodes += 1
        tf.summary.scalar("env/episode_reward", total_rew, step=self.total_steps)
        tf.summary.scalar("env/total_episodes", self.total_episodes, step=self.total_steps)
        return total_rew

    def learn(self):
        with self.writer.as_default(): # pylint: disable=not-context-manager
            for iteration in range(self.config["n_iter"]):
                # Sample parameter vectors
                thetas = [np.random.normal(self.theta_mean, self.theta_std, self.dim_theta)
                          for _ in range(self.config["batch_size"])]
                rewards = [self.noisy_evaluation(theta) for theta in thetas]
                # Get elite parameters
                n_elite = int(self.config["batch_size"] * self.config["elite_frac"])
                elite_inds = np.argsort(rewards)[self.config["batch_size"] - n_elite:self.config["batch_size"]]
                elite_thetas = [thetas[i] for i in elite_inds]
                # Update theta_mean, theta_std
                self.theta_mean = np.mean(elite_thetas, axis=0)
                self.theta_std = np.std(elite_thetas, axis=0)
                print("iteration {:d}. mean f: {:>8.3g}. max f: {:>8.3g}".format(
                    iteration,
                    np.mean(rewards),
                    np.max(rewards)))
                self.do_episode(self.make_policy(self.theta_mean))
            theta_path = self.monitor_path / "best_theta"
            np.save(theta_path, elite_thetas[-1])
            self.do_episode(self.make_policy(self.theta_mean), render=True)
