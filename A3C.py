#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import numpy as np
import tensorflow as tf
import logging
import argparse
from threading import Thread
import multiprocessing

import gym
from gym.spaces import Discrete, Box

from Learner import Learner
from utils import discount_rewards
from ActionSelection import ProbabilisticCategoricalActionSelection, ContinuousActionSelection

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

# Based on:
# - Pseudo code from Asynchronous Methods for Deep Reinforcement Learning
# - Tensorflow code from https://github.com/yao62995/A3C/blob/master/A3C_atari.py

class ActorNetworkDiscrete(object):
    """Neural network for the Actor of an Actor-Critic algorithm using a discrete action space"""
    def __init__(self, state_shape, n_actions, n_hidden, scope, summary=True):
        super(ActorNetworkDiscrete, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.scope = scope

        with tf.variable_scope("%s_actor" % scope):
            self.state = tf.placeholder("float", [None, self.state_shape], name='state')
            self.actions_taken = tf.placeholder(tf.float32, name='actions_taken')
            self.critic_feedback = tf.placeholder(tf.float32, name='critic_feedback')
            self.critic_rewards = tf.placeholder(tf.float32, name='critic_rewards')

            W0 = tf.Variable(tf.random_normal([self.state_shape, self.n_hidden]), name='W0')
            b0 = tf.Variable(tf.zeros([self.n_hidden]), name='b0')
            L1 = tf.tanh(tf.matmul(self.state, W0) + b0[None, :], name='L1')

            W1 = tf.Variable(tf.random_normal([self.n_hidden, n_actions]), name='W1')
            b1 = tf.Variable(tf.zeros([n_actions]), name='b1')
            self.prob_na = tf.nn.softmax(tf.matmul(L1, W1) + b1[None, :], name='prob_na')
            good_probabilities = tf.reduce_sum(tf.mul(self.prob_na, self.actions_taken), reduction_indices=[1])
            # Replace probabilities that are zero with a small value and multiply by advantage:
            eligibility = tf.log(tf.select(tf.equal(good_probabilities, tf.fill(tf.shape(good_probabilities), 0.0)), tf.fill(tf.shape(good_probabilities), 1e-30), good_probabilities)) \
                * (self.critic_rewards - self.critic_feedback)
            self.loss = -tf.reduce_mean(eligibility)
            self.summary_loss = self.loss  # Loss to show as a summary
            self.vars = [W0, b0, W1, b1]

class ActorNetworkContinuous(object):
    """Neural network for an Actor of an Actor-Critic algorithm using a continuous action space."""
    def __init__(self, action_space, state_shape, n_hidden, scope, summary=True):
        super(ActorNetworkContinuous, self).__init__()
        self.state_shape = state_shape
        self.n_hidden = n_hidden
        self.scope = scope

        with tf.variable_scope("%s_actor" % scope):
            self.state = tf.placeholder("float", [None, self.state_shape], name='state')
            self.actions_taken = tf.placeholder(tf.float32, name='actions_taken')  # Not used (yet?)
            self.critic_feedback = tf.placeholder(tf.float32, name='critic_feedback')  # Advantage
            self.critic_rewards = tf.placeholder(tf.float32, name='critic_rewards')

            mu_W0 = tf.Variable(tf.random_normal([self.state_shape, self.n_hidden]) / np.sqrt(self.state_shape), name='mu_W0')
            mu_b0 = tf.Variable(tf.zeros([self.n_hidden]), name='mu_b0')
            mu_W1 = tf.Variable(1e-4 * tf.random_normal([self.n_hidden, 1]), name='mu_W1')
            mu_b1 = tf.Variable(tf.zeros([1]), name='mu_b1')
            # Action probabilities
            L1 = tf.tanh(tf.matmul(self.state, mu_W0) + mu_b0[None, :])
            mu = tf.matmul(L1, mu_W1) + mu_b1[None, :]
            mu = tf.squeeze(mu)

            sigma_W0 = tf.Variable(tf.random_normal([self.state_shape, self.n_hidden]) / np.sqrt(self.state_shape), name='sigma_W0')
            sigma_b0 = tf.Variable(tf.zeros([self.n_hidden]), name='sigma_b0')
            sigma_W1 = tf.Variable(1e-4 * tf.random_normal([self.n_hidden, 1]), name='sigma_W1')
            sigma_b1 = tf.Variable(tf.zeros([1]), name='sigma_b1')
            # Action probabilities
            sigma_L1 = tf.tanh(tf.matmul(self.state, sigma_W0) + sigma_b0[None, :])
            sigma = tf.matmul(sigma_L1, sigma_W1) + sigma_b1[None, :]
            sigma = tf.squeeze(sigma)
            sigma = tf.nn.softplus(sigma) + 1e-5

            self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
            self.action = self.normal_dist.sample_n(1)
            self.action = tf.clip_by_value(self.action, action_space.low[0], action_space.high[0])
            self.loss = -self.normal_dist.log_prob(self.actions_taken) * self.critic_feedback
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()
            self.summary_loss = -tf.reduce_mean(self.loss)  # Loss to show as a summary
            self.vars = [mu_W0, mu_b0, mu_W1, mu_b1, sigma_W0, sigma_b0, sigma_W1, sigma_b1]


class CriticNetwork(object):
    """Neural network for the Critic of an Actor-Critic algorithm"""
    def __init__(self, state_shape, n_hidden, scope, summary=True):
        super(CriticNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_hidden = n_hidden
        self.scope = scope

        with tf.variable_scope("%s_critic" % scope):
            self.state = tf.placeholder("float", [None, self.state_shape], name='state')
            self.target = tf.placeholder("float", name="critic_target")

            W0 = tf.Variable(tf.random_normal([self.state_shape, self.n_hidden]), name='W0')
            b0 = tf.Variable(tf.zeros([self.n_hidden]), name='b0')
            L1 = tf.tanh(tf.matmul(self.state, W0) + b0[None, :], name='L1')

            W1 = tf.Variable(tf.random_normal([self.n_hidden, 1]), name='W1')
            b1 = tf.Variable(tf.zeros([1]), name='b1')
            self.value = tf.matmul(L1, W1) + b1[None, :]
            self.loss = tf.reduce_mean(tf.square(self.target - self.value))
            self.summary_loss = self.loss
            self.vars = [W0, b0, W1, b1]

class A3CThread(Thread):
    """Single A3C learner thread."""
    def __init__(self, master, thread_id):
        super(A3CThread, self).__init__(name=thread_id)
        self.thread_id = thread_id
        self.env = gym.make(master.env_name)
        self.master = master
        if thread_id == 0 and self.master.monitor:
            self.env.monitor.start(self.master.monitor_dir, force=True)

        # Build actor and critic networks
        self.build_networks()

        # Write the summary of each thread in a different directory
        self.writer = tf.summary.FileWriter(self.master.monitor_dir + '/thread' + str(self.thread_id), self.master.session.graph)

        self.actor_sync_net = self.sync_gradients_op(master.shared_actor_net, self.actor_net.vars)
        self.actor_create_ag = self.create_accumulative_gradients_op(self.actor_net.vars)
        self.actor_add_ag = self.add_accumulative_gradients_op(self.actor_net.vars, self.actor_create_ag, self.actor_net.loss)
        self.actor_reset_ag = self.reset_accumulative_gradients_op(self.actor_net.vars, self.actor_create_ag)

        self.critic_sync_net = self.sync_gradients_op(master.shared_critic_net, self.critic_net.vars)
        self.critic_create_ag = self.create_accumulative_gradients_op(self.critic_net.vars)
        self.critic_add_ag = self.add_accumulative_gradients_op(self.critic_net.vars, self.critic_create_ag, self.critic_net.loss)
        self.critic_reset_ag = self.reset_accumulative_gradients_op(self.critic_net.vars, self.critic_create_ag)

        # Clipped gradients
        gradient_clip_value = self.master.config['gradient_clip_value']
        clip_actor_gradients = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in self.actor_create_ag]
        self.apply_actor_gradients = master.shared_actor_optimizer.apply_gradients(
            zip(clip_actor_gradients, master.shared_actor_net.vars), global_step=master.global_step)
        clip_critic_gradients = [tf.clip_by_value(grad, -gradient_clip_value, gradient_clip_value) for grad in self.critic_create_ag]
        self.apply_critic_gradients = master.shared_critic_optimizer.apply_gradients(
            zip(clip_critic_gradients, master.shared_critic_net.vars), global_step=master.global_step)

        # Non-clipped gradients
        # self.apply_actor_gradients = master.shared_actor_optimizer.apply_gradients(
        #     zip(self.actor_create_ag, master.shared_actor_net.vars), global_step=master.global_step)
        # self.apply_critic_gradients = master.shared_critic_optimizer.apply_gradients(
        #     zip(self.critic_create_ag, master.shared_critic_net.vars), global_step=master.global_step)

    def create_accumulative_gradients_op(self, net_vars):
        """Make an operation to create accumulative gradients"""
        accum_grads = []
        with tf.name_scope(name="create_accum_%d" % self.thread_id, values=net_vars):
            for var in net_vars:
                zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
                name = var.name.replace(":", "_") + "_accum_grad"
                accum_grad = tf.Variable(zero, name=name, trainable=False)
                accum_grads.append(accum_grad)
            return accum_grads

    def add_accumulative_gradients_op(self, net_vars, accum_grads, loss):
        """Make an operation to add a gradient to the total"""
        accum_grad_ops = []
        with tf.name_scope(name="grad_ops_%d" % self.thread_id, values=net_vars):
            grads = tf.gradients(loss, net_vars, gate_gradients=False,
                                 aggregation_method=None,
                                 colocate_gradients_with_ops=False)
        with tf.name_scope(name="accum_ops_%d" % self.thread_id, values=[]):
            for (grad, var, accum_grad) in zip(grads, net_vars, accum_grads):
                name = var.name.replace(":", "_") + "_accum_grad_ops"
                accum_ops = tf.assign_add(accum_grad, grad, name=name)
                accum_grad_ops.append(accum_ops)
            return tf.group(*accum_grad_ops, name="accum_group_%d" % self.thread_id)

    def reset_accumulative_gradients_op(self, net_vars, accum_grads):
        """Make an operation to reset the accumulation to zero"""
        reset_grad_ops = []
        with tf.name_scope(name="reset_grad_ops_%d" % self.thread_id, values=net_vars):
            for (var, accum_grad) in zip(net_vars, accum_grads):
                zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
                name = var.name.replace(":", "_") + "_reset_grad_ops"
                reset_ops = tf.assign(accum_grad, zero, name=name)
                reset_grad_ops.append(reset_ops)
            return tf.group(*reset_grad_ops, name="reset_accum_group_%d" % self.thread_id)

    def sync_gradients_op(self, source_net, local_net_vars):
        """Make an operation to sync the gradients"""
        sync_ops = []
        with tf.name_scope(name="sync_ops_%d" % self.thread_id, values=[]):
            for (target_var, source_var) in zip(local_net_vars, source_net.vars):
                ops = tf.assign(target_var, source_var)
                sync_ops.append(ops)
            return tf.group(*sync_ops, name="sync_group_%d" % self.thread_id)

    def get_critic_value(self, state):
        return self.master.session.run([self.critic_net.value], feed_dict={self.critic_net.state: state})[0].flatten()

    def get_trajectory(self, episode_max_length, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        """
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        for i in range(episode_max_length):
            action = self.choose_action(state)  # Predict the next action (using a neural network) depending on the current state
            states.append(state.flatten())
            state, reward, done, _ = self.env.step(action)
            reward = np.clip(reward, -1, 1)  # Clip reward
            actions.append(action)
            rewards.append(reward)
            if done:
                break
            if render:
                self.env.render()
        return {
            "reward": np.array(rewards),
            "state": np.array(states),
            "action": np.array(actions),
            "done": done,  # Say if tajectory ended because a terminal state was reached
            "steps": i + 1
        }

    def run(self):
        # Assume global shared parameter vectors θ and θv and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'v
        sess = self.master.session
        t = 1  # thread step counter
        while self.master.T < self.master.config['T_max']:
            # Reset gradients: dθ←0 and dθv←0
            # Synchronize thread-specific parameters θ' = θ and θ'v = θv
            sess.run([self.actor_reset_ag, self.critic_reset_ag])
            # sync variables
            sess.run([self.actor_sync_net, self.critic_sync_net])
            trajectory = self.get_trajectory(self.master.config['episode_max_length'])
            reward = sum(trajectory['reward'])
            trajectory['reward'][-1] = 0 if trajectory['done'] else self.get_critic_value(trajectory['state'][None, -1])[0]
            returns = discount_rewards(trajectory['reward'], self.master.config['gamma'])
            fetches = [self.actor_net.summary_loss, self.critic_net.summary_loss, self.actor_add_ag, self.critic_add_ag, self.master.global_step]  # What does the master global step thing do?
            ac_net = self.actor_net
            cr_net = self.critic_net
            qw_new = self.master.session.run([cr_net.value], feed_dict={cr_net.state: trajectory['state']})[0].flatten()
            all_action = self.transform_actions(trajectory['action'])  # Transform actions back to the output shape of the actor network (e.g. one-hot for discrete action space)
            results = sess.run(fetches, feed_dict={
                ac_net.state: trajectory["state"],
                cr_net.state: trajectory["state"],
                ac_net.actions_taken: all_action,
                ac_net.critic_feedback: qw_new,
                ac_net.critic_rewards: returns,
                cr_net.target: returns.reshape(-1, 1)
            })
            summary = sess.run([self.master.summary_op], feed_dict={
                               self.master.actor_loss: results[0],
                               self.master.critic_loss: results[1],
                               self.master.reward: reward,
                               self.master.episode_length: trajectory["steps"]
                               })
            self.writer.add_summary(summary[0], t)
            self.writer.flush()
            sess.run([self.apply_actor_gradients, self.apply_critic_gradients])
            t += 1
            self.master.T += trajectory['steps']

class A3CThreadDiscrete(A3CThread):
    """A3CThread for a discrete action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadDiscrete, self).__init__(master, thread_id)

    def build_networks(self):
        self.actor_net = ActorNetworkDiscrete(self.env.observation_space.shape[0], self.env.action_space.n, self.master.config['actor_n_hidden'], scope="local_actor_net")
        self.critic_net = CriticNetwork(self.env.observation_space.shape[0], self.master.config['critic_n_hidden'], scope="local_critic_net")

    def choose_action(self, state):
        """Choose an action."""
        prob = self.master.session.run([self.actor_net.prob_na], feed_dict={self.actor_net.state: [state]})[0][0]
        action = self.master.action_selection.select_action(prob)
        return action

    def transform_actions(self, actions):
        possible_actions = np.arange(self.env.action_space.n)
        return (possible_actions == actions[:, None]).astype(np.float32)

class A3CThreadContinuous(A3CThread):
    """A3CThread for a continuous action space."""
    def __init__(self, master, thread_id):
        super(A3CThreadContinuous, self).__init__(master, thread_id)

    def build_networks(self):
        self.actor_net = ActorNetworkContinuous(self.env.action_space, self.env.observation_space.shape[0], self.master.config['actor_n_hidden'], scope="local_actor_net")
        self.critic_net = CriticNetwork(self.env.observation_space.shape[0], self.master.config['critic_n_hidden'], scope="local_critic_net")

    def choose_action(self, state):
        """Choose an action."""
        action = self.master.session.run([self.actor_net.action], feed_dict={self.actor_net.state: [state]})[0]
        return action

    def transform_actions(self, actions):
        return actions  # Nothing has to be done in case of a continuous action space

class A3CLearner(Learner):
    """Asynchronous Advantage Actor Critic learner."""
    def __init__(self, env, action_selection, monitor, monitor_dir, **usercfg):
        super(A3CLearner, self).__init__(env)
        self.env = env
        self.shared_counter = 0
        self.T = 0
        self.action_selection = action_selection
        self.env_name = env.spec.id
        self.monitor = monitor
        self.monitor_dir = monitor_dir

        self.config = dict(
            episode_max_length=env.spec.timestep_limit,
            gamma=0.99,
            actor_learning_rate=0.01,
            critic_learning_rate=0.05,
            actor_n_hidden=20,
            critic_n_hidden=20,
            gradient_clip_value=40,
            n_threads=multiprocessing.cpu_count(),  # Use as much threads as there are CPU threads on the current system
            T_max=5e5,
            repeat_n_actions=1
        )
        self.config.update(usercfg)

        # self.shared_actor_net = ActorNetwork(env.observation_space.shape[0], env.action_space.n, self.config['actor_n_hidden'], scope="global_actor_net", summary=False)
        # self.shared_critic_net = CriticNetwork(env.observation_space.shape[0], self.config['critic_n_hidden'], scope="global_critic_net", summary=False)
        self.build_networks()

        self.shared_actor_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['actor_learning_rate'], decay=0.9, epsilon=1e-9)
        self.shared_critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['critic_learning_rate'], decay=0.9, epsilon=1e-9)

        self.global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

        self.session = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True))

        self.critic_loss = tf.placeholder("float", name="critic_loss")
        critic_loss_summary = tf.summary.scalar("Critic_loss", self.critic_loss)

        self.actor_loss = tf.placeholder("float", name="actor_loss")
        actor_loss_summary = tf.summary.scalar("Actor_loss", self.actor_loss)
        self.reward = tf.placeholder("float", name="reward")
        reward_summary = tf.summary.scalar("Reward", self.reward)
        self.episode_length = tf.placeholder("float", name="episode_length")
        episode_length_summary = tf.summary.scalar("Episode_length", self.episode_length)
        self.summary_op = tf.summary.merge([actor_loss_summary, critic_loss_summary, reward_summary, episode_length_summary])

        self.jobs = []
        for thread_id in range(self.config["n_threads"]):
            job = self.make_thread(thread_id)
            self.jobs.append(job)

        self.session.run(tf.global_variables_initializer())

        self.global_step_val = 0

    def learn(self):
        self.train_step = 0
        # signal.signal(signal.SIGINT, signal_handler)
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

class A3CLearnerDiscrete(A3CLearner):
    """A3CLearner for a discrete action space"""
    def __init__(self, env, action_selection, monitor, monitor_dir, **usercfg):
        super(A3CLearnerDiscrete, self).__init__(env, action_selection, monitor, monitor_dir, **usercfg)

    def build_networks(self):
        self.shared_actor_net = ActorNetworkDiscrete(self.env.observation_space.shape[0], self.env.action_space.n, self.config['actor_n_hidden'], scope="global_actor_net", summary=False)
        self.shared_critic_net = CriticNetwork(self.env.observation_space.shape[0], self.config['critic_n_hidden'], scope="global_critic_net", summary=False)

    def make_thread(self, thread_id):
        return A3CThreadDiscrete(self, thread_id)

class A3CLearnerContinuous(A3CLearner):
    """A3CLearner for a continuous action space"""
    def __init__(self, env, action_selection, monitor, monitor_dir, **usercfg):
        super(A3CLearnerContinuous, self).__init__(env, action_selection, monitor, monitor_dir, **usercfg)

    def build_networks(self):
        self.shared_actor_net = ActorNetworkContinuous(self.env.action_space, self.env.observation_space.shape[0], self.config['actor_n_hidden'], scope="global_actor_net", summary=False)
        self.shared_critic_net = CriticNetwork(self.env.observation_space.shape[0], self.config['critic_n_hidden'], scope="global_critic_net", summary=False)

    def make_thread(self, thread_id):
        return A3CThreadContinuous(self, thread_id)

parser = argparse.ArgumentParser()
parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the experiment on.")
parser.add_argument("--monitor", action="store_true", default=False, help="Track performance of a single thread using gym monitor.")
parser.add_argument("monitor_path", metavar="monitor_path", type=str, help="Path where Gym monitor files may be saved.")

def main():
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    env = gym.make(args.environment)
    if isinstance(env.action_space, Discrete):
        agent = A3CLearnerDiscrete(env, ProbabilisticCategoricalActionSelection(), args.monitor, args.monitor_path)
    elif isinstance(env.action_space, Box):
        agent = A3CLearnerContinuous(env, ContinuousActionSelection(), args.monitor, args.monitor_path)
    else:
        raise NotImplementedError
    try:
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
