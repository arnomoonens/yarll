#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import numpy as np
import tensorflow as tf
import logging
import argparse

import gym
from gym.spaces import Discrete, Box
# import gym_ple

from Learner import Learner
from utils import discount_rewards, preprocess_image
from Reporter import Reporter
from ActionSelection import ProbabilisticCategoricalActionSelection

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

def choose_action(output, n_actions, temperature=1.0):
    # total = sum([np.exp(float(o) / temperature) for o in output])
    # probs = [np.exp(float(o) / temperature) / total for o in output]
    probs = output / np.sum(output)
    action = np.random.choice(n_actions, p=probs)
    return action, probs

class KPCNNLearner(Learner):
    """Karpathy policy gradient learner using a convolutional neural network"""
    def __init__(self, env, action_selection, **usercfg):
        super(KPCNNLearner, self).__init__(env, **usercfg)
        self.nA = env.action_space.n
        self.action_selection = action_selection
        # Default configuration. Can be overwritten using keyword arguments.
        self.config = dict(
            # episode_max_length=100,
            # timesteps_per_batch=10000,
            # n_iter=100,
            n_hidden_units=200,
            gamma=0.99,
            learning_rate=1e-3,
            batch_size=10,  # Amount of episodes after which to adapt gradients
            decay_rate=0.99,  # Used for RMSProp
            draw_frequency=50  # Draw a plot every 50 episodes
        )
        self.config.update(usercfg)
        self.build_network()

    def build_network(self):
        image_size = 80
        image_depth = 1  # aka nr. of feature maps. Eg 3 for RGB images. 1 here because we use grayscale images

        self.state = tf.placeholder(tf.float32, [None, image_size, image_size, image_depth], name="state")

        # Convolution layer 1
        depth = 32
        patch_size = 4
        self.w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, depth], stddev=0.01))
        self.b1 = tf.Variable(tf.zeros([depth]))
        self.L1 = tf.nn.relu(tf.nn.conv2d(self.state, self.w1, strides=[1, 2, 2, 1], padding="SAME") + self.b1)
        self.L1 = tf.nn.max_pool(self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolution layer 2
        self.w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.01))
        self.b2 = tf.Variable(tf.zeros([depth]))
        self.L2 = tf.nn.relu(tf.nn.conv2d(self.L1, self.w2, strides=[1, 2, 2, 1], padding="SAME") + self.b2)

        # Flatten
        shape = self.L2.get_shape().as_list()
        reshape = tf.reshape(self.L2, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Fully connected layer 1
        self.w3 = tf.Variable(tf.truncated_normal([image_size // 8 * image_size // 8 * depth, self.config['n_hidden_units']], stddev=0.01))
        self.b3 = tf.Variable(tf.zeros([self.config['n_hidden_units']]))
        self.L3 = tf.nn.relu(tf.matmul(reshape, self.w3) + self.b3)

        # Fully connected layer 2
        self.w4 = tf.Variable(tf.truncated_normal([self.config['n_hidden_units'], self.nA]))
        self.b4 = tf.Variable(tf.zeros([self.nA]))
        self.output = tf.nn.softmax(tf.matmul(self.L3, self.w4) + self.b4)

        self.vars = [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4
        ]

        self.action_taken = tf.placeholder(tf.float32, shape=[None, self.nA], name="action_taken")
        self.feedback = tf.placeholder(tf.float32, shape=[None, self.nA], name="feedback")
        loss = tf.reduce_mean(tf.squared_difference(self.action_taken, self.output) * self.feedback)

        self.create_accumulative_grads = self.create_accumulative_gradients_op(self.vars)
        self.accumulate_grads = self.add_accumulative_gradients_op(self.vars, self.create_accumulative_grads, loss)
        self.reset_accumulative_grads = self.reset_accumulative_gradients_op(self.vars, self.create_accumulative_grads)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['learning_rate'], decay=self.config['decay_rate'], epsilon=1e-9)
        self.apply_gradients = self.optimizer.apply_gradients(zip(self.create_accumulative_grads, self.vars))

        init = tf.global_variables_initializer()

        # Launch the graph.
        self.session = tf.Session()
        self.session.run(init)

    def create_accumulative_gradients_op(self, net_vars):
        """Make an operation to create accumulative gradients"""
        accum_grads = []
        for var in net_vars:
            zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
            name = var.name.replace(":", "_") + "_accum_grad"
            accum_grad = tf.Variable(zero, name=name, trainable=False)
            accum_grads.append(accum_grad)
        return accum_grads

    def add_accumulative_gradients_op(self, net_vars, accum_grads, loss):
        """Make an operation to add a gradient to the total"""
        accum_grad_ops = []
        grads = tf.gradients(loss, net_vars, gate_gradients=False,
                             aggregation_method=None,
                             colocate_gradients_with_ops=False)
        for (grad, var, accum_grad) in zip(grads, net_vars, accum_grads):
            name = var.name.replace(":", "_") + "_accum_grad_ops"
            accum_ops = tf.assign_add(accum_grad, grad, name=name)
            accum_grad_ops.append(accum_ops)
        return tf.group(*accum_grad_ops, name="accum_group")

    def reset_accumulative_gradients_op(self, net_vars, accum_grads):
        """Make an operation to reset the accumulation to zero"""
        reset_grad_ops = []
        for (var, accum_grad) in zip(net_vars, accum_grads):
            zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
            name = var.name.replace(":", "_") + "_reset_grad_ops"
            reset_ops = tf.assign(accum_grad, zero, name=name)
            reset_grad_ops.append(reset_ops)
        return tf.group(*reset_grad_ops, name="reset_accum_group")

    def act(self, state):
        nn_outputs = self.session.run([self.output], feed_dict={self.state: [state]})[0][0]
        action, probabilities = choose_action(nn_outputs, self.nA)
        return action, probabilities

    def get_trajectory(self, env, episode_max_length, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        Note that this function returns more than the get_trajectory in the Learner class.
        """
        state = preprocess_image(env.reset())
        prev_state = state
        states = []
        actions = []
        rewards = []
        episode_probabilities = []
        for _ in range(episode_max_length):
            delta = state - prev_state
            action, probabilities = self.act(delta)
            states.append(delta)
            prev_state = state
            state, rew, done, _ = env.step(action)
            state = preprocess_image(state)
            actions.append(action)
            rewards.append(rew)
            episode_probabilities.append(probabilities)
            if done:
                break
            if render:
                env.render()
        return {"reward": np.array(rewards),
                "state": np.array(states),
                "action": np.array(actions),
                "prob": np.array(episode_probabilities)
                }

    def learn(self, env):
        reporter = Reporter()

        self.session.run([self.reset_accumulative_grads])

        iteration = 0  # amount of batches processed
        episode_nr = 0
        episode_lengths = np.zeros(self.config['batch_size'])
        episode_rewards = np.zeros(self.config['batch_size'])
        mean_rewards = []
        while True:  # Keep executing episodes
            trajectory = self.get_trajectory(env, self.config["episode_max_length"])

            episode_rewards[episode_nr % self.config['batch_size']] = sum(trajectory['reward'])
            episode_lengths[episode_nr % self.config['batch_size']] = len(trajectory['reward'])
            episode_nr += 1
            action_taken = (np.arange(self.nA) == trajectory['action'][:, None]).astype(np.float32)  # one-hot encoding

            discounted_episode_rewards = discount_rewards(trajectory['reward'], self.config['gamma'])
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            std = np.std(discounted_episode_rewards)
            std = std if std > 0 else 1
            discounted_episode_rewards /= std
            feedback = np.reshape(np.repeat(discounted_episode_rewards, self.nA), (len(discounted_episode_rewards), self.nA))

            self.session.run([self.accumulate_grads], feed_dict={self.state: trajectory["state"], self.action_taken: action_taken, self.feedback: feedback})
            if episode_nr % self.config['batch_size'] == 0:  # batch is done
                iteration += 1
                self.session.run([self.apply_gradients])
                self.session.run([self.reset_accumulative_grads])
                reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, episode_nr)
                mean_rewards.append(episode_rewards.mean())
                if episode_nr % self.config['draw_frequency'] == 0:
                    reporter.draw_rewards(mean_rewards)

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
        agent = KPCNNLearner(env, ProbabilisticCategoricalActionSelection(), episode_max_length=env.spec.timestep_limit)
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    else:
        raise NotImplementedError
    try:
        env.monitor.start(args.monitor_path, force=True)
        agent.learn(env)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
