# -*- coding: utf8 -*-

import os
import tensorflow as tf
import numpy as np
from gym import wrappers

from agents.agent import Agent
from agents.actor_critic import ActorCriticNetworkDiscrete, ActorCriticNetworkDiscreteCNN, ActorCriticNetworkDiscreteCNNRNN, ActorCriticNetworkContinuous
from misc.utils import discount_rewards
from agents.env_runner import EnvRunner

def cso_loss(old_network, new_network, epsilon, advantage):
    # Probs of all actions
    old_log_probs = tf.nn.log_softmax(old_network.logits)
    new_log_probs = tf.nn.log_softmax(new_network.logits)
    # Prob of the action that was actually taken
    old_action_log_prob = tf.reduce_sum(old_log_probs * new_network.actions_taken, [1])
    new_action_log_prob = tf.reduce_sum(new_log_probs * new_network.actions_taken, [1])
    ratio = tf.exp(new_action_log_prob - old_action_log_prob)
    ratio_clipped = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
    # ratio = tf.Print(ratio, [ratio], summarize=32)
    return tf.minimum(ratio * advantage, ratio_clipped * advantage)

class PPO(Agent):
    """Proximal Policy Optimization agent."""
    def __init__(self, env, monitor_path, video=False, **usercfg):
        super(PPO, self).__init__(**usercfg)
        self.monitor_path = monitor_path
        self.env = wrappers.Monitor(
            env,
            monitor_path,
            force=True,
            video_callable=(None if video else False))
        self.env_runner = EnvRunner(self.env, self, usercfg)

        self.config.update(dict(
            n_hidden=20,
            gamma=0.99,
            learning_rate=0.001,
            n_local_steps=20,
            gradient_clip_value=0.5,
            entropy_coef=0.01,
            cso_epsilon=0.2  # Clipped surrogate objective epsilon
        ))

        with tf.variable_scope("old_network"):
            self.old_network = self.build_networks()
            self.old_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        with tf.variable_scope("new_network"):
            self.new_network = self.build_networks()
            self.new_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.action = self.new_network.action
        self.value = self.new_network.value
        self.states = self.new_network.states
        self.r = self.new_network.r
        self.adv = self.new_network.adv
        self.actions_taken = self.new_network.actions_taken

        self.set_old_to_new = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.old_network_vars, self.new_network_vars)])

        # Reduces by taking the mean instead of summing
        self.actor_loss = tf.reduce_mean(cso_loss(self.old_network, self.new_network, self.config["cso_epsilon"], self.adv))
        self.critic_loss = tf.reduce_mean(tf.square(self.value - self.r))
        log_probs = tf.nn.log_softmax(self.new_network.logits)
        entropy = tf.reduce_mean(self.new_network.probs * log_probs)
        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.config["entropy_coef"] * entropy

        self.optimizer = tf.train.AdamOptimizer(self.config["learning_rate"], name="optim")
        grads = tf.gradients(self.loss, self.new_network_vars)
        grads, _ = tf.clip_by_global_norm(grads, self.config["gradient_clip_value"])

        apply_grads = self.optimizer.apply_gradients(zip(grads, self.new_network_vars))

        self._global_step = tf.get_variable(
            "global_step",
            [],
            tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

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
        summary_actor_loss = tf.summary.scalar("model/Actor_loss", self.actor_loss / n_steps)
        summary_critic_loss = tf.summary.scalar("model/Critic_loss", self.critic_loss / n_steps)
        summary_loss = tf.summary.scalar("model/Loss", self.loss / n_steps)
        self.loss_summary_op = tf.summary.merge(
            [summary_actor_loss, summary_critic_loss, summary_loss])
        self.writer = tf.summary.FileWriter(os.path.join(self.monitor_path, "summaries"), self.session.graph)
        self.env_runner.summary_writer = self.writer
        return

    @property
    def global_step(self):
        return self._global_step.eval(session=self.session)

    def get_critic_value(self, state):
        return self.session.run([self.value], feed_dict={self.states: state})[0].flatten()

    def choose_action(self, state, *rest):
        """Choose an action."""
        feed_dict = {
            self.states: [state]
        }
        action, value = self.session.run([self.action, self.value], feed_dict=feed_dict)
        return action, value[0]

    def get_env_action(self, action):
        return np.argmax(action)

    def get_processed_trajectories(self):
        all_states = []
        all_actions = []
        all_advs = []
        all_rs = []
        for _ in range(10):
            trajectory = self.env_runner.get_steps(self.config["n_local_steps"])
            v = 0 if trajectory.terminal else self.get_critic_value(np.asarray(trajectory.states)[None, -1])
            rewards_plus_v = np.asarray(trajectory.rewards + [v])
            vpred_t = np.asarray(trajectory.values + [v])
            delta_t = trajectory.rewards + self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
            batch_r = discount_rewards(rewards_plus_v, self.config["gamma"])[:-1]
            batch_adv = discount_rewards(delta_t, self.config["gamma"])
            all_states.extend(trajectory.states)
            all_actions.extend(trajectory.actions)
            all_advs.extend(np.vstack(batch_adv).flatten().tolist())
            all_rs.extend(batch_r)
        return all_states, all_actions, all_advs, all_rs

    def learn(self):
        """Run learning algorithm"""
        config = self.config
        n_updates = 0
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            states, actions, advs, rs = self.get_processed_trajectories()
            self.session.run(self.set_old_to_new)

            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for i in range(0, len(states), 32):
                batch_states = np.array(states)[i:(i + 32)]
                batch_actions = np.array(actions)[i:(i + 32)]
                batch_advs = np.array(advs)[i:(i + 32)]
                batch_rs = np.array(rs)[i:(i + 32)]
                fetches = [self.loss_summary_op, self.train_op]
                feed_dict = {
                    self.states: batch_states,
                    self.old_network.states: batch_states,
                    self.actions_taken: batch_actions,
                    self.adv: batch_advs,
                    self.r: batch_rs
                }
                summary, _ = self.session.run(fetches, feed_dict)
                self.writer.add_summary(summary, n_updates)
                self.writer.flush()
                n_updates += 1

        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

class PPODiscrete(PPO):
    def build_networks(self):
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])

class PPODiscreteCNN(PPODiscrete):
    def build_networks(self):
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])

class PPODiscreteCNNRNN(PPODiscrete):
    def build_networks(self):
        return ActorCriticNetworkDiscreteCNNRNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])

class PPOContinuous(PPO):
    def build_networks(self):
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            self.config["n_hidden"])
