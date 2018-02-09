# -*- coding: utf8 -*-


import os
import tensorflow as tf
import numpy as np
from gym import wrappers
import threading
import queue

from agents.agent import Agent
from agents.actor_critic import ActorCriticNetworkDiscrete, ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous
from agents.ppo import cso_loss
from misc.utils import discount_rewards
from agents.env_runner import EnvRunner
from environment.registration import make

class DPPOWorker(threading.Thread):
    """Distributed Proximal Policy Optimization Worker."""
    def __init__(self, policy, env, queue, lock, should_update, should_collect, n_local_steps, min_trajectories, summary_writer=None):
        super(DPPOWorker, self).__init__()
        self.daemon = True
        self.policy = policy
        self.config = self.policy.config
        self.initial_features = policy.initial_features
        self.env = env
        self.queue = queue
        self.lock = lock
        self.should_update = should_update
        self.should_collect = should_collect
        self.n_local_steps = n_local_steps
        self.min_trajectories = min_trajectories
        self.env_runner = EnvRunner(self.env, self, {}, summary_writer=summary_writer)

    def run(self):
        while True:  # TODO: use coordinator.should_stop instead
            if not self.should_collect.is_set():
                self.should_collect.wait()
            trajectory = self.env_runner.get_steps(self.config["n_local_steps"], stop_at_trajectory_end=False)
            v = 0 if trajectory.terminal else self.get_critic_value(np.asarray(trajectory.states)[None, -1], trajectory.features[-1])
            rewards_plus_v = np.asarray(trajectory.rewards + [v])
            vpred_t = np.asarray(trajectory.values + [v])
            delta_t = trajectory.rewards + self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
            batch_r = discount_rewards(rewards_plus_v, self.config["gamma"])[:-1]
            batch_adv = discount_rewards(delta_t, self.config["gamma"])
            processed = trajectory.states, trajectory.actions, np.vstack(batch_adv).flatten().tolist(), batch_r, trajectory.features[0]
            self.queue.put(processed)
            self.policy.n_trajectories += 1
            if self.policy.n_trajectories >= self.min_trajectories:
                self.should_collect.clear()
                self.should_update.set()
                self.policy.n_trajectories = 0

    @property
    def global_step(self):
        return self.policy._global_step.eval(session=self.policy.session)

    def get_critic_value(self, state, *rest):
        with self.lock:
            value = self.policy.session.run([self.policy.value], feed_dict={self.policy.states: state})[0].flatten()
        return value

    def choose_action(self, state, *rest):
        with self.lock:
            action, value = self.policy.session.run([self.policy.action, self.policy.value], feed_dict={self.policy.states: [state]})
        return action, value[0], []

    def get_env_action(self, action):
        return np.argmax(action)

    def new_trajectory(self):
        pass

class DPPO(Agent):
    """Distributed Proximal Policy Optimization agent."""

    RNN = False

    def __init__(self, env, monitor_path, video=False, **usercfg):
        super(DPPO, self).__init__()
        self.env = env
        self.monitor_path = monitor_path
        self.env_name = env.spec.id

        self.config.update(dict(
            n_workers=4,
            n_hidden=20,
            gamma=0.99,
            learning_rate=0.001,
            n_iter=10000,
            n_local_steps=256,
            gradient_clip_value=0.5,
            entropy_coef=0.01,
            cso_epsilon=0.2,  # Clipped surrogate objective epsilon
            save_model=False
        ))

        with tf.variable_scope("old_network"):
            self.old_network = self.build_networks()
            self.old_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        with tf.variable_scope("new_network"):
            self.new_network = self.build_networks()
            if self.RNN:
                self.initial_features = self.new_network.state_init
            else:
                self.initial_features = None
            self.new_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.action = self.new_network.action
        self.value = self.new_network.value
        self.states = self.new_network.states
        self.r = self.new_network.r
        self.adv = self.new_network.adv
        self.actions_taken = self.new_network.actions_taken

        self.set_old_to_new = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.old_network_vars, self.new_network_vars)])

        # Reduces by taking the mean instead of summing
        self.actor_loss = -tf.reduce_mean(cso_loss(self.old_network, self.new_network, self.config["cso_epsilon"], self.adv))
        self.critic_loss = tf.reduce_mean(tf.square(self.value - self.r))
        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.config["entropy_coef"] * tf.reduce_mean(self.new_network.entropy)

        grads = tf.gradients(self.loss, self.new_network_vars)

        self._global_step = tf.get_variable(
            "global_step",
            [],
            tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

        self.n_steps = tf.shape(self.states)[0]
        self.session = tf.Session()
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = tf.train.Saver()
        n_steps = tf.to_float(self.n_steps)
        summary_actor_loss = tf.summary.scalar("model/Actor_loss", self.actor_loss / n_steps)
        summary_critic_loss = tf.summary.scalar("model/Critic_loss", self.critic_loss / n_steps)
        summary_loss = tf.summary.scalar("model/Loss", self.loss / n_steps)
        summary_grad_norm = tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        summary_var_norm = tf.summary.scalar("model/var_global_norm", tf.global_norm(self.new_network_vars))
        self.model_summary_op = tf.summary.merge(
            [summary_actor_loss, summary_critic_loss, summary_loss, summary_grad_norm, summary_var_norm])
        self.writer = tf.summary.FileWriter(os.path.join(self.monitor_path, "summaries"), self.session.graph)
        self.env_runner = EnvRunner(self.env, self, usercfg, summary_writer=self.writer)

        # grads before clipping were passed to the summary, now clip and apply them
        grads, _ = tf.clip_by_global_norm(grads, self.config["gradient_clip_value"])
        self.optimizer = tf.train.AdamOptimizer(self.config["learning_rate"], name="optim")
        apply_grads = self.optimizer.apply_gradients(zip(grads, self.new_network_vars))

        inc_step = self._global_step.assign_add(self.n_steps)
        self.train_op = tf.group(apply_grads, inc_step)

        init = tf.global_variables_initializer()
        self.session.run(init)

        self.lock = threading.Lock()
        self.should_update, self.should_collect = threading.Event(), threading.Event()
        self.queue = queue.Queue(self.config["n_workers"])
        self.n_trajectories = 0
        self.workers = []
        for i in range(self.config["n_workers"]):
            env = wrappers.Monitor(
                make(self.env_name),
                monitor_path,
                force=True,
                video_callable=(None if video else False))
            worker = DPPOWorker(
                self,
                env,
                self.queue,
                self.lock,
                self.should_update,
                self.should_collect,
                self.config["n_local_steps"],
                self.config["n_workers"],
                self.writer if i == 0 else None)
            self.workers.append(worker)
        return

    def learn(self):
        """Run learning algorithm"""
        config = self.config
        n_updates = 0

        self.should_collect.set()
        for worker in self.workers:
            worker.start()

        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            if not self.should_update.is_set():
                self.should_update.wait()
            print("Gonna update")
            trajectories = [self.queue.get() for _ in range(self.queue.qsize())]
            self.session.run(self.set_old_to_new)

            for states, actions, advs, rs, features in trajectories:
                fetches = [self.model_summary_op, self.train_op]
                feed_dict = {
                    self.states: states,
                    self.old_network.states: states,
                    self.actions_taken: actions,
                    self.old_network.actions_taken: actions,
                    self.adv: advs,
                    self.r: rs
                }
                if features != [] and features is not None:
                    feed_dict[self.old_network.rnn_state_in] = features
                    feed_dict[self.new_network.rnn_state_in] = features
                summary, _ = self.session.run(fetches, feed_dict)
                self.writer.add_summary(summary, n_updates)
                n_updates += 1
                self.writer.flush()
                self.should_update.clear()
                self.should_collect.set()

        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver.save(self.session, os.path.join(self.monitor_path, "model"))

class DPPODiscrete(DPPO):
    def build_networks(self):
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])

class DPPODiscreteCNN(DPPODiscrete):
    def build_networks(self):
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])

class DPPOContinuous(DPPO):
    def build_networks(self):
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            self.config["n_hidden"])

    def get_env_action(self, action):
        return action
