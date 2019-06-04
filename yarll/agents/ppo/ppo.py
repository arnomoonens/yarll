# -*- coding: utf8 -*-

import os
import tensorflow as tf
import numpy as np
from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.actorcritic.actor_critic import ActorCriticNetwork, ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous, critic_loss
from yarll.misc.network_ops import normal_dist_log_prob
from yarll.agents.env_runner import EnvRunner


def ppo_loss(old_logprob, new_logprob, epsilon, advantage):
    ratio = tf.exp(new_logprob - old_logprob)
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
            vf_coef=0.5,
            entropy_coef=0.01,
            cso_epsilon=0.2,  # Clipped surrogate objective epsilon
            summary_every_updates=1000,
            save_model=False
        ))
        self.config.update(usercfg)

        self.old_network = self.build_networks()

        self.new_network = self.build_networks()
        if self.RNN:
            self.initial_features = self.new_network.state_init
        else:
            self.initial_features = None

        # self.mean_entropy = tf.reduce_mean(self.new_network.entropy)
        # self.loss = self.actor_loss + self.config["vf_coef"] * self.critic_loss + \
        #     self.config["entropy_coef"] * self.mean_entropy

        # TODO: get from ppo_loss function
        # ratio_mean, ratio_std = tf.nn.moments(ratio, axes=[0])
        # summary_ratio_mean = tf.summary.scalar("model/ratio/mean", ratio_mean)
        # summary_ratio_std = tf.summary.scalar("model/ratio/std", ratio_std)

        # summary_var_norm = tf.summary.scalar(
        #     "model/var_global_norm", tf.global_norm(self.new_network_vars))
        # Weight summaries: not turned on right now because they take too much space
        # TODO: use config to make this optional
        # for v in tf.trainable_variables():
        #    if "new_network" in v.name:
        #        summaries.append(tf.summary.histogram(v.name, v))

        self.writer = tf.summary.create_file_writer(self.monitor_path)
        self.env_runner = EnvRunner(self.env,
                                    self,
                                    usercfg,
                                    normalize_states=self.config["normalize_states"],
                                    summary_writer=self.writer)

        optim_kwargs = {k: self.config[l]
                        for k, l in [("clipnorm", "gradient_clip_value")] if self.config[l] is not None}
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            **optim_kwargs)

    def _specific_summaries(self, n_updates: int) -> None:
        """Summaries that are specific to the variant of the algorithm."""
        return

    def _actor_loss(self, old_logprob, new_logprob, advantage):
        return ppo_loss(old_logprob, new_logprob, self.config["cso_epsilon"], advantage)

    def build_networks(self):
        raise NotImplementedError

    def choose_action(self, state, features) -> dict:
        action, value = self.new_network.action_value(state[None, :])
        return {"action": action, "value": value[0]}

    def get_processed_trajectories(self):
        trajectory = self.env_runner.get_steps(
            int(self.config["n_local_steps"]), stop_at_trajectory_end=False)
        features = trajectory.features
        features = np.concatenate(trajectory.features) if features[-1] is not None else np.array([None])
        T = trajectory.steps
        if trajectory.experiences[-1].terminal:
            v = 0
        else:
            inp = [np.asarray(trajectory.states)[None, -1]]
            if features[-1] is not None:
                inp.append(features[None, -1])
            v = self.new_network.action_value(*inp)[-2 if features[-1] is not None else -1]
        vpred = np.asarray(trajectory.values + [v])
        gamma = self.config["gamma"]
        lambda_ = self.config["gae_lambda"]
        gaelam = advantages = np.empty(T, 'float32')
        last_gaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - trajectory.terminals[t]
            delta = trajectory.rewards[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = last_gaelam = delta + gamma * lambda_ * nonterminal * last_gaelam
        rs = advantages + trajectory.values
        return trajectory.states, trajectory.actions, advantages, rs, trajectory.features

    def set_old_to_new(self):
        for old_var, new_var in zip(self.old_network.trainable_variables, self.new_network.trainable_variables):
            old_var.assign(new_var)

    def _critic_loss(self, returns, value):
        return critic_loss(returns, value)

    @tf.function
    def train(self, states, actions_taken, advantages, returns, features=None):
        states = tf.cast(states, dtype=tf.float32)
        actions_taken = tf.cast(actions_taken, dtype=tf.int32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        returns = tf.cast(returns, dtype=tf.float32)
        inp = states if features is None else [states, tf.cast(features, tf.float32)]
        with tf.GradientTape() as tape:
            new_res = self.new_network(inp)
            new_logits = new_res[0]
            values = tf.squeeze(new_res[1])
            old_res = self.old_network(inp)
            old_logits = old_res[0]
            new_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_taken, logits=new_logits)
            old_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_taken, logits=old_logits)
            mean_actor_loss = -tf.reduce_mean(self._actor_loss(old_log_prob, new_log_prob, advantages))
            mean_critic_loss = tf.reduce_mean(self._critic_loss(returns, values))
            loss = mean_actor_loss + self.config["vf_coef"] * mean_critic_loss
            gradients = tape.gradient(loss, self.new_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.new_network.trainable_weights))
        return mean_actor_loss, mean_critic_loss, loss, tf.linalg.global_norm(gradients), old_log_prob, new_log_prob, new_logits

    def learn(self):
        """Run learning algorithm"""
        config = self.config
        input_shape = (None, *self.env.observation_space.shape)
        self.old_network.build(input_shape)
        self.new_network.build(input_shape)
        n_updates = 0
        n_steps = 0
        with self.writer.as_default():
            for _ in range(int(config["n_iter"])):
                # Collect trajectories until we get timesteps_per_batch total timesteps
                states, actions, advs, rs, _ = self.get_processed_trajectories()
                n_steps += len(states)
                advs = np.array(advs)
                normalized_advs = (advs - advs.mean()) / advs.std()
                self.set_old_to_new()

                indices = np.arange(len(states))
                for _ in range(int(self.config["n_epochs"])):
                    np.random.shuffle(indices)
                    batch_size = int(self.config["batch_size"])
                    for j in range(0, len(states), batch_size):
                        batch_indices = indices[j:(j + batch_size)]
                        batch_states = np.array(states)[batch_indices]
                        batch_actions = np.array(actions)[batch_indices]
                        batch_advs = np.array(normalized_advs)[batch_indices]
                        batch_rs = np.array(rs)[batch_indices]
                        train_actor_loss, train_critic_loss, train_loss, \
                            grad_global_norm, new_log_prob, old_log_prob, new_network_output = self.train(batch_states,
                                                                                                          batch_actions,
                                                                                                          batch_advs,
                                                                                                          batch_rs)
                        if (n_updates % self.config["summary_every_updates"]) == 0:
                            tf.summary.scalar("model/Loss", train_loss, step=n_steps)
                            tf.summary.scalar("model/Actor_loss", train_actor_loss, step=n_steps)
                            tf.summary.scalar("model/Critic_loss", train_critic_loss, step=n_steps)
                            tf.summary.scalar("model/advantage/mean", np.mean(batch_advs), step=n_steps)
                            tf.summary.scalar("model/advantage/std", np.std(batch_advs), step=n_steps)
                            tf.summary.scalar("model/new_log_prob/mean", tf.reduce_mean(new_log_prob), n_steps)
                            tf.summary.scalar("model/old_log_prob/mean", tf.reduce_mean(old_log_prob), n_steps)
                            tf.summary.scalar("model/return/mean", np.mean(batch_rs), n_steps)
                            tf.summary.scalar("model/return/std", np.std(batch_rs), n_steps)
                            tf.summary.scalar("model/entropy",
                                              tf.reduce_mean(self.new_network.entropy(new_network_output)),
                                              n_steps)
                            tf.summary.scalar("model/grad_global_norm", grad_global_norm, n_steps)
                            self._specific_summaries(n_steps)
                        n_updates += 1

            if self.config["save_model"]:
                tf.saved_model.save(self.ac_net, os.path.join(self.monitor_path, "model"))


class PPODiscrete(PPO):
    def build_networks(self) -> ActorCriticNetwork:
        return ActorCriticNetworkDiscrete(
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))


class PPODiscreteCNN(PPODiscrete):
    def build_networks(self) -> ActorCriticNetwork:
        return ActorCriticNetworkDiscreteCNN(
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class PPOContinuous(PPO):
    def build_networks(self) -> ActorCriticNetwork:
        return ActorCriticNetworkContinuous(
            self.env.action_space.shape,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    @tf.function
    def train(self, states, actions_taken, advantages, returns, features=None):
        states = tf.cast(states, dtype=tf.float32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        returns = tf.cast(returns, dtype=tf.float32)
        inp = states if features is None else [states, tf.cast(features, tf.float32)]
        with tf.GradientTape() as tape:
            new_res = self.new_network(inp)
            new_mean = new_res[1]
            values = tf.squeeze(new_res[2])
            new_log_std = self.new_network.action_mean.log_std
            old_res = self.old_network(inp)
            old_mean = old_res[1]
            old_log_std = self.old_network.action_mean.log_std
            new_log_prob = normal_dist_log_prob(actions_taken, new_mean, new_log_std)
            old_log_prob = normal_dist_log_prob(actions_taken, old_mean, old_log_std)
            mean_actor_loss = -tf.reduce_mean(self._actor_loss(old_log_prob, new_log_prob, advantages))
            mean_critic_loss = tf.reduce_mean(self._critic_loss(returns, values))
            loss = mean_actor_loss + self.config["vf_coef"] * mean_critic_loss
            gradients = tape.gradient(loss, self.new_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.new_network.trainable_weights))

        return mean_actor_loss, mean_critic_loss, loss, tf.linalg.global_norm(gradients), old_log_prob, new_log_prob, new_mean

    def _specific_summaries(self, n_updates: int) -> None:
        tf.summary.scalar("model/std",
                          tf.reduce_mean(tf.exp(self.new_network.action_mean.log_std)),
                          n_updates)

    def choose_action(self, state, features) -> dict:
        action, _, value = self.new_network.action_value(state[None, :])
        return {"action": action, "value": value}

    def get_env_action(self, action):
        return action
