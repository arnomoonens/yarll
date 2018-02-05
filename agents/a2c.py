# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf
import logging

from gym import wrappers

from agents.agent import Agent
from misc.utils import discount_rewards
from misc.network_ops import mu_sigma_layer, normalized_columns_initializer, linear
# from misc.reporter import Reporter
from agents.env_runner import EnvRunner

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

class ActorCriticNetworkDiscrete(object):
    """Neural network for the Actor of an Actor-Critic algorithm using a discrete action space"""
    def __init__(self, state_shape, n_actions, n_hidden):
        super(ActorCriticNetworkDiscrete, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        self.states = tf.placeholder(tf.float32, [None] + list(state_shape), name="states")
        self.adv = tf.placeholder(tf.float32, name="advantage")
        self.actions_taken = tf.placeholder(tf.float32, [None, n_actions], name="actions_taken")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.n_hidden,
            activation_fn=tf.tanh,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=tf.zeros_initializer(),
            scope="L1")

        # Fully connected for actor
        self.logits = linear(L1, n_actions, "actionlogits", normalized_columns_initializer(0.01))

        self.probs = tf.nn.softmax(self.logits)

        self.action = tf.squeeze(tf.multinomial(self.logits - tf.reduce_max(self.logits, [1], keep_dims=True), 1), [1], name="action")
        self.action = tf.one_hot(self.action, n_actions)[0, :]

        # Fully connected for critic
        self.value = tf.reshape(linear(L1, 1, "value", normalized_columns_initializer(1.0)), [-1])

        log_probs = tf.nn.log_softmax(self.logits)
        self.actor_loss = - tf.reduce_mean(tf.reduce_sum(log_probs * self.actions_taken, [1]) * self.adv)
        self.critic_loss = 0.5 * tf.reduce_mean(tf.square(self.value - self.r))
        self.entropy = - tf.reduce_mean(self.probs * log_probs)
        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.entropy * self.config["entropy_coef"]
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class ActorCriticNetworkContinuous(object):
    """Neural network for an Actor of an Actor-Critic algorithm using a continuous action space."""
    def __init__(self, state_shape, action_space, n_hidden):
        super(ActorCriticNetworkContinuous, self).__init__()
        self.state_shape = state_shape
        self.n_hidden = n_hidden

        self.states = tf.placeholder("float", [None] + list(state_shape), name="states")
        self.actions_taken = tf.placeholder(tf.float32, name="actions_taken")
        self.adv = tf.placeholder(tf.float32, name="advantage")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        L1 = tf.contrib.layers.fully_connected(
            inputs=self.states,
            num_outputs=self.n_hidden,
            activation_fn=tf.tanh,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=tf.zeros_initializer(),
            scope="mu_L1")

        mu, sigma = mu_sigma_layer(L1, 1)

        self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.action = self.normal_dist.sample(1)
        self.action = tf.clip_by_value(self.action, action_space.low[0], action_space.high[0], name="action")
        self.value = tf.reshape(linear(L1, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.actor_loss = - tf.reduce_mean(self.normal_dist.log_prob(self.actions_taken) * self.adv)
        self.critic_loss = tf.reduce_mean(tf.square(self.value - self.r))
        self.entropy = - tf.reduce_mean(self.normal_dist.entropy())
        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.entropy * self.config["entropy_coef"]

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class A2C(Agent):
    """Advantage Actor Critic"""
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(A2C, self).__init__(**usercfg)
        self.monitor_path = monitor_path

        self.env = wrappers.Monitor(
            env,
            monitor_path,
            force=True,
            video_callable=(None if video else False))
        self.env_runner = EnvRunner(self.env, self, usercfg)

        self.config.update(dict(
            n_iter=100,
            gamma=0.99,
            learning_rate=0.001,
            n_hidden=20,
            gradient_clip_value=0.5,
            n_local_steps=20,
            entropy_coef=0.01,
            save_model=False
        ))
        self.config.update(usercfg)
        self.build_networks()

        self.action = self.ac_net.action
        self.states = self.ac_net.states
        self.r = self.ac_net.r
        self.adv = self.ac_net.adv
        self.actions_taken = self.ac_net.actions_taken

        self._global_step = tf.get_variable(
            "global_step",
            [],
            tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

        self.optimizer = tf.train.AdamOptimizer(self.config["learning_rate"], name="optim")
        grads = tf.gradients(self.ac_net.loss, self.ac_net.vars)
        grads, _ = tf.clip_by_global_norm(grads, self.config["gradient_clip_value"])

        # Apply gradients to the weights of the master network
        apply_grads = self.optimizer.apply_gradients(zip(grads, self.ac_net.vars))

        self.n_steps = tf.shape(self.states)[0]
        inc_step = self._global_step.assign_add(self.n_steps)
        self.train_op = tf.group(apply_grads, inc_step)

        init = tf.global_variables_initializer()
        # Launch the graph.
        self.session = tf.Session()
        self.session.run(init)
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()
        n_steps = tf.to_float(self.n_steps)
        summary_actor_loss = tf.summary.scalar("Actor_loss", self.ac_net.actor_loss / n_steps)
        summary_critic_loss = tf.summary.scalar("Critic_loss", self.ac_net.critic_loss / n_steps)
        summary_loss = tf.summary.scalar("Loss", self.ac_net.loss / n_steps)
        self.loss_summary_op = tf.summary.merge(
            [summary_actor_loss, summary_critic_loss, summary_loss])
        self.writer = tf.summary.FileWriter(os.path.join(self.monitor_path, "summaries"), self.session.graph)
        self.env_runner.summary_writer = self.writer
        return

    @property
    def global_step(self):
        return self._global_step.eval(session=self.session)

    def get_critic_value(self, state):
        return self.session.run([self.ac_net.value], feed_dict={self.states: state})[0].flatten()

    def choose_action(self, state, *rest):
        """Choose an action."""
        feed_dict = {
            self.states: [state]
        }
        action, value = self.session.run([self.ac_net.action, self.ac_net.value], feed_dict=feed_dict)
        return action, value

    def get_env_action(self, action):
        return np.argmax(action)

    def learn(self):
        """Run learning algorithm"""
        config = self.config
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectory = self.env_runner.get_steps(self.config["n_local_steps"])
            v = 0 if trajectory.terminal else self.get_critic_value(np.asarray(trajectory.states)[None, -1])
            rewards_plus_v = np.asarray(trajectory.rewards + [v])
            vpred_t = np.asarray(trajectory.values + [v])
            delta_t = trajectory.rewards + self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
            batch_r = discount_rewards(rewards_plus_v, self.config["gamma"])[:-1]
            batch_adv = discount_rewards(delta_t, self.config["gamma"])
            fetches = [self.loss_summary_op, self.train_op, self._global_step]
            states = np.asarray(trajectory.states)
            feed_dict = {
                self.states: states,
                self.actions_taken: np.asarray(trajectory.actions),
                self.adv: batch_adv,
                self.r: np.asarray(batch_r)
            }
            summary, _, global_step = self.session.run(fetches, feed_dict)
            self.writer.add_summary(summary, global_step)
            self.writer.flush()

        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

class A2CDiscrete(A2C):
    def build_networks(self):
        self.ac_net = ActorCriticNetworkDiscrete(
            self.env.observation_space.shape,
            self.env.action_space.n,
            self.config["n_hidden"])

class A2CContinuous(A2C):
    def build_networks(self):
        self.ac_net = ActorCriticNetworkContinuous(
            self.env.observation_space.shape,
            self.env.action_space,
            self.config["n_hidden"])
