#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import os
import numpy as np
import tensorflow as tf
import logging
import argparse

from gym import wrappers
from gym.spaces import Discrete, Box
# import gym_ple

from Environment.registration import make_environment
from Learner import Learner
from utils import discount_rewards, preprocess_image, save_config
from Reporter import Reporter
from ActionSelection import ProbabilisticCategoricalActionSelection
from gradient_ops import create_accumulative_gradients_op, add_accumulative_gradients_op, reset_accumulative_gradients_op

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

def random_with_probability(output, n_actions, temperature=1.0):
    # total = sum([np.exp(float(o) / temperature) for o in output])
    # probs = [np.exp(float(o) / temperature) / total for o in output]
    probs = output / np.sum(output)
    action = np.random.choice(n_actions, p=probs)
    return action, probs

class KPCNNLearner(Learner):
    """Karpathy policy gradient learner using a convolutional neural network"""
    def __init__(self, env, action_selection, monitor_dir, **usercfg):
        super(KPCNNLearner, self).__init__(env, **usercfg)
        self.nA = env.action_space.n
        self.action_selection = action_selection
        self.monitor_dir = monitor_dir
        # Default configuration. Can be overwritten using keyword arguments.
        self.config.update(
            dict(
                # timesteps_per_batch=10000,
                # n_iter=100,
                n_hidden_units=200,
                learning_rate=1e-3,
                batch_size=10,  # Amount of episodes after which to adapt gradients
                gamma=0.99,  # Discount past rewards by a percentage
                decay=0.99,  # Decay of RMSProp optimizer
                epsilon=1e-9,  # Epsilon of RMSProp optimizer
                draw_frequency=50  # Draw a plot every 50 episodes
            )
        )
        self.config.update(usercfg)
        self.build_network()
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()

    def build_network(self):
        image_size = 80
        image_depth = 1  # aka nr. of feature maps. Eg 3 for RGB images. 1 here because we use grayscale images

        self.states = tf.placeholder(tf.float32, [None, image_size, image_size, image_depth], name="states")

        # Convolution layer 1
        depth = 32
        patch_size = 4
        self.w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, depth], stddev=0.01))
        self.b1 = tf.Variable(tf.zeros([depth]))
        self.L1 = tf.nn.relu(tf.nn.conv2d(self.states, self.w1, strides=[1, 2, 2, 1], padding="SAME") + self.b1)
        self.L1 = tf.nn.max_pool(self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Convolution layer 2
        self.w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.01))
        self.b2 = tf.Variable(tf.zeros([depth]))
        self.L2 = tf.nn.relu(tf.nn.conv2d(self.L1, self.w2, strides=[1, 2, 2, 1], padding="SAME") + self.b2)

        # Flatten
        shape = self.L2.get_shape().as_list()
        reshape = tf.reshape(self.L2, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Fully connected layer 1
        self.L3 = tf.contrib.layers.fully_connected(
            inputs=reshape,
            num_outputs=self.config["n_hidden_units"],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(stddev=0.01),
            biases_initializer=tf.zeros_initializer())

        # Fully connected layer 2
        self.probs = tf.contrib.layers.fully_connected(
            inputs=self.L3,
            num_outputs=self.nA,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.random_normal_initializer(),
            biases_initializer=tf.zeros_initializer())

        self.action = tf.squeeze(tf.multinomial(tf.log(self.probs), 1), name="action")

        self.vars = [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4
        ]

        self.action_taken = tf.placeholder(tf.float32, shape=[None, self.nA], name="action_taken")
        self.feedback = tf.placeholder(tf.float32, shape=[None, self.nA], name="feedback")
        loss = tf.reduce_mean(tf.squared_difference(self.action_taken, self.probs) * self.feedback)

        self.create_accumulative_grads = create_accumulative_gradients_op(self.vars)
        self.accumulate_grads = add_accumulative_gradients_op(self.vars, self.create_accumulative_grads, loss)
        self.reset_accumulative_grads = reset_accumulative_gradients_op(self.vars, self.create_accumulative_grads)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config["learning_rate"], decay=self.config["decay"], epsilon=self.config["epsilon"])

        self.apply_gradients = self.optimizer.apply_gradients(zip(self.create_accumulative_grads, self.vars))

        init = tf.global_variables_initializer()

        # Launch the graph.
        self.session = tf.Session()
        self.session.run(init)

    def choose_action(self, state):
        return self.session.run([self.action], feed_dict={self.states: [state]})[0]

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
        for _ in range(episode_max_length):
            delta = state - prev_state
            action = self.choose_action(delta)
            states.append(delta)
            prev_state = state
            state, rew, done, _ = env.step(action)
            state = preprocess_image(state)
            actions.append(action)
            rewards.append(rew)
            if done:
                break
            if render:
                env.render()
        return {
            "reward": np.array(rewards),
            "state": np.array(states),
            "action": np.array(actions),
        }

    def learn(self, env):
        reporter = Reporter()

        self.session.run([self.reset_accumulative_grads])

        iteration = 0  # amount of batches processed
        episode_nr = 0
        episode_lengths = np.zeros(self.config["batch_size"])
        episode_rewards = np.zeros(self.config["batch_size"])
        mean_rewards = []
        while True:  # Keep executing episodes
            trajectory = self.get_trajectory(env, self.config["episode_max_length"])

            episode_rewards[episode_nr % self.config["batch_size"]] = sum(trajectory["reward"])
            episode_lengths[episode_nr % self.config["batch_size"]] = len(trajectory["reward"])
            episode_nr += 1
            action_taken = (np.arange(self.nA) == trajectory["action"][:, None]).astype(np.float32)  # one-hot encoding

            discounted_episode_rewards = discount_rewards(trajectory["reward"], self.config["gamma"])
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            std = np.std(discounted_episode_rewards)
            std = std if std > 0 else 1
            discounted_episode_rewards /= std
            feedback = np.reshape(np.repeat(discounted_episode_rewards, self.nA), (len(discounted_episode_rewards), self.nA))

            self.session.run([self.accumulate_grads], feed_dict={self.states: trajectory["state"], self.action_taken: action_taken, self.feedback: feedback})
            if episode_nr % self.config["batch_size"] == 0:  # batch is done
                iteration += 1
                self.session.run([self.apply_gradients])
                self.session.run([self.reset_accumulative_grads])
                reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, episode_nr)
                mean_rewards.append(episode_rewards.mean())
                if episode_nr % self.config["draw_frequency"] == 0:
                    reporter.draw_rewards(mean_rewards)
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver.save(self.session, os.path.join(self.monitor_dir, "model"))

parser = argparse.ArgumentParser()
parser.add_argument("environment", metavar="env", type=str, help="Gym environment to execute the experiment on.")
parser.add_argument("monitor_path", metavar="monitor_path", type=str, help="Path where Gym monitor files may be saved")
parser.add_argument("--save_model", action="store_true", default=False, help="Save resulting model.")
parser.add_argument("--no_video", dest="video", action="store_false", default=True, help="Don't render and show video.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate used when optimizing weights.")

def main():
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)
    env = make_environment(args.environment)
    if isinstance(env.action_space, Discrete):
        agent = KPCNNLearner(
            env,
            ProbabilisticCategoricalActionSelection(),
            args.monitor_path,
            save_model=args.save_model,
            learning_rate=args.learning_rate
        )
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    else:
        raise NotImplementedError
    save_config(args.monitor_path, agent.config, [env.to_dict()])
    try:
        env = wrappers.Monitor(env, args.monitor_path, force=True, video_callable=(None if args.video else False))
        agent.learn(env)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
