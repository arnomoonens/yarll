# -*- coding: utf8 -*-

import os
import tensorflow as tf
import numpy as np
from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.actorcritic.actor_critic import ActorCriticNetwork, ActorCriticNetworkDiscrete,\
ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import FastSaver

def ppo_loss(old_log, new_log, epsilon, advantage):
    ratio = tf.exp(new_log - old_log)
    ratio_clipped = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)
    return tf.minimum(ratio * advantage, ratio_clipped * advantage)

class PPO(Agent):
    """Proximal Policy Optimization agent."""
    RNN = False

    def __init__(self, env, monitor_path: str, monitor: bool = False, video: bool = False, **usercfg) -> None:
        super(PPO, self).__init__(**usercfg)
        self.monitor_path: str = monitor_path
        self.env = env
        if monitor:
            self.env = wrappers.Monitor(
                self.env,
                monitor_path,
                force=True,
                video_callable=(None if video else False))

        self.config.update(dict(
            n_hidden_units=20,
            n_hidden_layers=2,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=0.001,
            n_epochs=10,
            n_iter=10000,
            batch_size=64,  # Timesteps per training batch
            n_local_steps=256,
            normalize_states=False,
            gradient_clip_value=None,
            adam_epsilon=1e-5,
            vf_coef=0.5,
            entropy_coef=0.01,
            cso_epsilon=0.2,  # Clipped surrogate objective epsilon
            save_model=False
        ))
        self.config.update(usercfg)

        with tf.variable_scope("old_network"):
            self.old_network = self.build_networks()
            self.old_network_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        with tf.variable_scope("new_network"):
            self.new_network = self.build_networks()
            if self.RNN:
                self.initial_features = self.new_network.state_init
            else:
                self.initial_features = None
            self.new_network_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.action = self.new_network.action
        self.value = self.new_network.value
        self.states = self.new_network.states
        self.actions_taken = self.new_network.actions_taken
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.ret = tf.placeholder(tf.float32, [None], name="return")

        self.set_old_to_new = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.old_network_vars, self.new_network_vars)])

        self.actor_loss = -tf.reduce_mean(self.make_actor_loss(self.old_network, self.new_network, self.advantage))
        self.critic_loss = tf.reduce_mean(tf.square(self.value - self.ret))
        self.mean_entropy = tf.reduce_mean(self.new_network.entropy)
        self.loss = self.actor_loss + self.config["vf_coef"] * self.critic_loss + \
            self.config["entropy_coef"] * self.mean_entropy

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
            self.saver = FastSaver()

        summary_actor_loss = tf.summary.scalar("model/Actor_loss", self.actor_loss)
        summary_critic_loss = tf.summary.scalar("model/Critic_loss", self.critic_loss)
        summary_loss = tf.summary.scalar("model/Loss", self.loss)

        adv_mean, adv_std = tf.nn.moments(self.advantage, axes=[0])
        summary_adv_mean = tf.summary.scalar("model/advantage/mean", adv_mean)
        summary_adv_std = tf.summary.scalar("model/advantage/std", adv_std)

        # TODO: get from ppo_loss function
        # ratio_mean, ratio_std = tf.nn.moments(ratio, axes=[0])
        # summary_ratio_mean = tf.summary.scalar("model/ratio/mean", ratio_mean)
        # summary_ratio_std = tf.summary.scalar("model/ratio/std", ratio_std)

        summary_new_log_prob_mean = tf.summary.scalar(
            "model/new_log_prob/mean", tf.reduce_mean(self.new_network.action_log_prob))
        summary_old_log_prob_mean = tf.summary.scalar(
            "model/old_log_prob/mean", tf.reduce_mean(self.old_network.action_log_prob))

        summary_mean_mean = tf.summary.scalar("model/model/mean/mean", tf.reduce_mean(self.old_network.mean))
        summary_std_mean = tf.summary.scalar("model/model/std/mean", tf.reduce_mean(self.old_network.std))

        summary_ret = tf.summary.scalar("model/return/mean", tf.reduce_mean(self.ret))
        summary_entropy = tf.summary.scalar("model/entropy", -self.mean_entropy)
        summary_grad_norm = tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        summary_var_norm = tf.summary.scalar(
            "model/var_global_norm", tf.global_norm(self.new_network_vars))
        summaries = []
        for v in tf.trainable_variables():
            if "new_network" in v.name:
                summaries.append(tf.summary.histogram(v.name, v))
        summaries += [summary_actor_loss, summary_critic_loss,
                      summary_loss,
                      summary_mean_mean, summary_std_mean,
                      summary_adv_mean, summary_adv_std,
                      # summary_ratio_mean, summary_ratio_std,
                      summary_new_log_prob_mean, summary_old_log_prob_mean,
                      summary_ret, summary_entropy, summary_grad_norm, summary_var_norm]
        self.model_summary_op = tf.summary.merge(summaries)
        self.writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "summaries"), self.session.graph)
        self.env_runner = EnvRunner(self.env,
                                    self,
                                    usercfg,
                                    normalize_states=self.config["normalize_states"],
                                    summary_writer=self.writer)

        # grads before clipping were passed to the summary, now clip and apply them
        if self.config["gradient_clip_value"] is not None:
            grads, _ = tf.clip_by_global_norm(
                grads, self.config["gradient_clip_value"])
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config["learning_rate"],
            epsilon=self.config["adam_epsilon"],
            name="optim")
        apply_grads = self.optimizer.apply_gradients(
            zip(grads, self.new_network_vars))

        inc_step = self._global_step.assign_add(self.n_steps)
        self.train_op = tf.group(apply_grads, inc_step)

        self.init_op = tf.global_variables_initializer()
        return

    def _initialize(self):
        self.session.run(self.init_op)

    def make_actor_loss(self, old_network, new_network, advantage):
        return ppo_loss(old_network.action_log_prob, new_network.action_log_prob, self.config["cso_epsilon"], advantage)

    def build_networks(self):
        raise NotImplementedError

    @property
    def global_step(self):
        return self._global_step.eval(session=self.session)

    def get_critic_value(self, state, *rest):
        return self.session.run([self.value], feed_dict={self.states: state})[0].flatten()

    def choose_action(self, state, *rest):
        action, value = self.session.run(
            [self.action, self.value], feed_dict={self.states: [state]})
        return {"action": action, "value": value[0]}

    def get_env_action(self, action):
        return np.argmax(action)

    def get_processed_trajectories(self):
        experiences = self.env_runner.get_steps(
            int(self.config["n_local_steps"]), stop_at_trajectory_end=False)
        T = experiences.steps
        v = 0 if experiences.terminals[-1] else self.get_critic_value(
            np.asarray(experiences.states)[None, -1], experiences.features[-1])
        vpred = np.asarray(experiences.values + [v])
        gamma = self.config["gamma"]
        lambda_ = self.config["gae_lambda"]
        gaelam = advantages = np.empty(T, 'float32')
        last_gaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - experiences.terminals[t]
            delta = experiences.rewards[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = last_gaelam = delta + gamma * lambda_ * nonterminal * last_gaelam
        rs = advantages + experiences.values
        return experiences.states, experiences.actions, advantages, rs, experiences.features

    def learn(self):
        """Run learning algorithm"""
        self._initialize()
        config = self.config
        n_updates = 0
        for _ in range(int(config["n_iter"])):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            states, actions, advs, rs, _ = self.get_processed_trajectories()
            advs = np.array(advs)
            advs = (advs - advs.mean()) / advs.std()
            self.session.run(self.set_old_to_new)

            indices = np.arange(len(states))
            for _ in range(int(self.config["n_epochs"])):
                np.random.shuffle(indices)

                batch_size = int(self.config["batch_size"])
                for j in range(0, len(states), batch_size):
                    batch_indices = indices[j:(j + batch_size)]
                    batch_states = np.array(states)[batch_indices]
                    batch_actions = np.array(actions)[batch_indices]
                    batch_advs = np.array(advs)[batch_indices]
                    batch_rs = np.array(rs)[batch_indices]
                    losses = [self.actor_loss, self.critic_loss, self.loss]
                    fetches = losses + [self.model_summary_op, self.train_op]
                    feed_dict = {
                        self.states: batch_states,
                        self.old_network.states: batch_states,
                        self.actions_taken: batch_actions,
                        self.old_network.actions_taken: batch_actions,
                        self.advantage: batch_advs,
                        self.ret: batch_rs
                    }
                    results = self.session.run(fetches, feed_dict)
                    self.writer.add_summary(results[len(losses)], n_updates)
                    n_updates += 1
                self.writer.flush()

            if self.config["save_model"]:
                self.saver.save(self.session, os.path.join(self.monitor_path, "model"))


class PPODiscrete(PPO):
    def build_networks(self) -> ActorCriticNetwork:
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))


class PPODiscreteCNN(PPODiscrete):
    def build_networks(self) -> ActorCriticNetwork:
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class PPOContinuous(PPO):
    def build_networks(self) -> ActorCriticNetwork:
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def get_env_action(self, action):
        return action
