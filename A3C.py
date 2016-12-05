#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import numpy as np
import tensorflow as tf
import logging
import argparse
from threading import Thread

import gym
from gym.spaces import Discrete, Box

from Learner import Learner
from utils import discount_rewards
from ActionSelection import ProbabilisticCategoricalActionSelection

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

# Based on:
# - Pseudo code from Asynchronous Methods for Deep Reinforcement Learning
# - Tensorflow code from https://github.com/yao62995/A3C/blob/master/A3C_atari.py

class ActorNetwork(object):
    """Neural network for the Actor of an Actor-Critic algorithm"""
    def __init__(self, state_shape, n_actions, n_hidden, scope, print_loss):
        super(ActorNetwork, self).__init__()
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
            eligibility = tf.log(tf.select(tf.equal(good_probabilities, tf.fill(tf.shape(good_probabilities), 0.0)), tf.fill(tf.shape(good_probabilities), 1e-30), good_probabilities)) \
                * (self.critic_rewards - self.critic_feedback)
            self.loss = -tf.reduce_mean(eligibility)
            if print_loss:
                self.loss = tf.Print(self.loss, [self.loss], message='Actor loss=')

            self.vars = [W0, b0, W1, b1]

class CriticNetwork(object):
    """Neural network for the Critic of an Actor-Critic algorithm"""
    def __init__(self, state_shape, n_hidden, scope, print_loss):
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
            if print_loss:
                self.loss = tf.Print(self.loss, [self.loss], message='Critic loss=')

            self.vars = [W0, b0, W1, b1]

class A3CThread(Thread):
    """Single A3C learner thread."""
    def __init__(self, master, thread_id):
        super(A3CThread, self).__init__(name=thread_id)
        self.thread_id = thread_id
        self.env = gym.make(master.env_name)
        self.master = master
        first_thread = thread_id == 0
        if first_thread:
            self.env.monitor.start(self.master.monitor_dir, force=True, video_callable=False)

        self.actor_net = ActorNetwork(self.env.observation_space.shape[0], self.env.action_space.n, self.master.config['actor_n_hidden'], scope="local_actor_net", print_loss=first_thread)
        self.critic_net = CriticNetwork(self.env.observation_space.shape[0], self.master.config['critic_n_hidden'], scope="local_critic_net", print_loss=first_thread)

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
                accum_grads.append(accum_grad.ref())
            return accum_grads

    def add_accumulative_gradients_op(self, net_vars, accum_grads, loss):
        """Make an operation to add a gradient to the total"""
        accum_grad_ops = []
        with tf.name_scope(name="grad_ops_%d" % self.thread_id, values=net_vars):
            var_refs = [v.ref() for v in net_vars]
            grads = tf.gradients(loss, var_refs, gate_gradients=False,
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

    def act(self, state):
        """Choose an action."""
        state = state.reshape(1, -1)
        prob = self.master.session.run([self.actor_net.prob_na], feed_dict={self.actor_net.state: state})[0][0]
        action = self.master.action_selection.select_action(prob)
        return action

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
            action = self.act(state)
            states.append(state.flatten())
            state, reward, done, _ = self.env.step(action)
            reward = np.clip(reward, -1, 1)  # Clip reward
            actions.append(action)
            rewards.append(reward)
            if done:
                break
            if render:
                self.env.render()
        return {"reward": np.array(rewards),
                "state": np.array(states),
                "action": np.array(actions),
                "done": done,  # Say if tajectory ended because a terminal state was reached
                "steps": i + 1
                }

    def run(self):
        # Assume global shared parameter vectors θ and θv and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'v
        sess = self.master.session
        possible_actions = np.arange(self.env.action_space.n)
        t = 1  # thread step counter
        while self.master.T < self.master.config['T_max']:
            if self.thread_id == 0:
                logging.info("Total steps: " + str(self.master.T))
            # Reset gradients: dθ←0 and dθv←0
            # Synchronize thread-specific parameters θ' = θ and θ'v = θv
            sess.run([self.actor_reset_ag, self.critic_reset_ag])
            # sync variables
            sess.run([self.actor_sync_net, self.critic_sync_net])
            t_start = t
            trajectory = self.get_trajectory(t_start + self.master.config['episode_max_length'])
            trajectory['reward'][-1] = 0 if trajectory['done'] else self.get_critic_value(trajectory['state'][None, -1])[0]
            returns = discount_rewards(trajectory['reward'], self.master.config['gamma'])
            fetches = [self.actor_add_ag, self.critic_add_ag, self.master.global_step]  # What does the master global step thing do?
            ac_net = self.actor_net
            cr_net = self.critic_net
            qw_new = self.master.session.run([cr_net.value], feed_dict={cr_net.state: trajectory['state']})[0].flatten()
            all_action = (possible_actions == trajectory['action'][:, None]).astype(np.float32)
            # print(returns.reshape(-1, 1))
            sess.run(fetches, feed_dict={
                     ac_net.state: trajectory["state"],
                     cr_net.state: trajectory["state"],
                     ac_net.actions_taken: all_action,
                     ac_net.critic_feedback: qw_new,
                     ac_net.critic_rewards: returns,
                     cr_net.target: returns.reshape(-1, 1),
                     })
            sess.run([self.apply_actor_gradients, self.apply_critic_gradients])
            self.master.T += trajectory['steps']

class A3CLearner(Learner):
    """Asynchronous Advantage Actor Critic learner."""
    def __init__(self, env, action_selection, monitor_dir, **usercfg):
        super(A3CLearner, self).__init__(env)
        self.env = env
        self.shared_counter = 0
        self.T = 0
        self.action_selection = action_selection
        self.env_name = env.spec.id
        self.monitor_dir = monitor_dir

        self.config = dict(
            episode_max_length=200,
            timesteps_per_batch=1000,
            trajectories_per_batch=10,
            batch_update="timesteps",
            n_iter=200,
            gamma=0.99,
            actor_learning_rate=0.01,
            critic_learning_rate=0.05,
            actor_n_hidden=20,
            critic_n_hidden=20,
            gradient_clip_value=40,
            n_threads=8,
            T_max=2e5
        )
        self.config.update(usercfg)

        self.shared_actor_net = ActorNetwork(env.observation_space.shape[0], env.action_space.n, self.config['actor_n_hidden'], scope="global_actor_net", print_loss=False)
        self.shared_critic_net = CriticNetwork(env.observation_space.shape[0], self.config['critic_n_hidden'], scope="global_critic_net", print_loss=False)

        self.shared_actor_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['actor_learning_rate'], decay=0.9, epsilon=1e-9)
        self.shared_critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['critic_learning_rate'], decay=0.9, epsilon=1e-9)

        self.global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

        self.jobs = []
        for thread_id in range(self.config["n_threads"]):
            job = A3CThread(self, thread_id)
            self.jobs.append(job)
        self.session = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True))
        self.session.run(tf.initialize_all_variables())

        self.global_step_val = 0

    def learn(self):
        self.train_step = 0
        # signal.signal(signal.SIGINT, signal_handler)
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

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
        agent = A3CLearner(env, ProbabilisticCategoricalActionSelection(), args.monitor_path, episode_max_length=env.spec.timestep_limit)
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    else:
        raise NotImplementedError
    try:
        agent.learn()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
