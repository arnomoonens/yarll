# -*- coding: utf8 -*-

import os
from typing import List
import tensorflow as tf
import numpy as np
from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.actorcritic.actor_critic import ActorCriticNetwork, ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous, critic_loss
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
            adam_epsilon=1e-5,
            vf_coef=0.5,
            entropy_coef=0.01,
            cso_epsilon=0.2,  # Clipped surrogate objective epsilon
            save_model=False
        ))
        self.config.update(usercfg)

        self.old_network = self.build_networks()

        self.new_network = self.build_networks()
        if self.RNN:
            self.initial_features = self.new_network.state_init
        else:
            self.initial_features = None

        # self.actor_loss = -tf.reduce_mean(self.make_actor_loss(self.old_network, self.new_network, self.advantage))
        # self.critic_loss = tf.reduce_mean(tf.square(self.value - self.ret))
        # self.mean_entropy = tf.reduce_mean(self.new_network.entropy)
        # self.loss = self.actor_loss + self.config["vf_coef"] * self.critic_loss + \
        #     self.config["entropy_coef"] * self.mean_entropy

        # summary_actor_loss = tf.summary.scalar("model/Actor_loss", self.actor_loss)
        # summary_critic_loss = tf.summary.scalar("model/Critic_loss", self.critic_loss)
        # summary_loss = tf.summary.scalar("model/Loss", self.loss)

        # adv_mean, adv_std = tf.nn.moments(self.advantage, axes=[0])
        # summary_adv_mean = tf.summary.scalar("model/advantage/mean", adv_mean)
        # summary_adv_std = tf.summary.scalar("model/advantage/std", adv_std)

        # TODO: get from ppo_loss function
        # ratio_mean, ratio_std = tf.nn.moments(ratio, axes=[0])
        # summary_ratio_mean = tf.summary.scalar("model/ratio/mean", ratio_mean)
        # summary_ratio_std = tf.summary.scalar("model/ratio/std", ratio_std)

        # summary_new_log_prob_mean = tf.summary.scalar(
        #     "model/new_log_prob/mean", tf.reduce_mean(self.new_network.action_log_prob))
        # summary_old_log_prob_mean = tf.summary.scalar(
        #     "model/old_log_prob/mean", tf.reduce_mean(self.old_network.action_log_prob))

        # summary_ret = tf.summary.scalar("model/return/mean", tf.reduce_mean(self.ret))
        # summary_entropy = tf.summary.scalar("model/entropy", -self.mean_entropy)
        # summary_grad_norm = tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        # summary_var_norm = tf.summary.scalar(
        #     "model/var_global_norm", tf.global_norm(self.new_network_vars))
        # summaries: List[tf.Tensor] = []
        # Weight summaries: not turned on right now because they take too much space
        # TODO: use config to make this optional
        # for v in tf.trainable_variables():
        #    if "new_network" in v.name:
        #        summaries.append(tf.summary.histogram(v.name, v))
        # summaries += self._specific_summaries()
        # summaries += [summary_actor_loss, summary_critic_loss,
        #               summary_loss,
        #               summary_adv_mean, summary_adv_std,
        #               # summary_ratio_mean, summary_ratio_std,
        #               summary_new_log_prob_mean, summary_old_log_prob_mean,
        #               summary_ret, summary_entropy, summary_grad_norm, summary_var_norm]
        # self.model_summary_op = tf.summary.merge(summaries)

        self.writer = tf.summary.create_file_writer(self.monitor_path)
        self.env_runner = EnvRunner(self.env,
                                    self,
                                    usercfg,
                                    normalize_states=self.config["normalize_states"],
                                    summary_writer=self.writer)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            epsilon=self.config["adam_epsilon"],
            clipnorm=self.config["gradient_clip_value"])

    def _specific_summaries(self) -> List[tf.Tensor]:
        """Summaries that are specific to the variant of the algorithm. None (empty list) for the base algorithm"""
        return []

    def _actor_loss(self, old_logits, new_logits, advantage):
        return ppo_loss(old_logits, new_logits, self.config["cso_epsilon"], advantage)

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
            v = self.new_network.action_value(*inp)[-2 if features[-1] is not None else -1][0]
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
        return self.config["vf_coef"] * critic_loss(returns, value)

    @tf.function
    def train(self, states, actions_taken, advantages, returns, features=None):
        states = tf.cast(states, dtype=tf.float32)
        actions_taken = tf.cast(actions_taken, dtype=tf.int32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        returns = tf.cast(returns, dtype=tf.float32)
        inp = states if features is None else [states, tf.cast(features, tf.float32)]
        actions_one_hot = tf.one_hot(actions_taken, self.env.action_space.n)
        with tf.GradientTape() as tape:
            new_res = self.new_network(inp)
            new_logits = new_res[0]
            values = new_res[1]
            old_res = self.old_network(inp)
            old_logits = old_res[0]
            new_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_taken, logits=new_logits)
            old_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_taken, logits=old_logits)
            mean_actor_loss = -tf.reduce_mean(self._actor_loss(old_log_prob, new_log_prob, advantages))
            mean_critic_loss = tf.reduce_mean(self._critic_loss(returns, values))
            loss = mean_actor_loss + self.config["vf_coef"] * mean_critic_loss
            gradients = tape.gradient(loss, self.new_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.new_network.trainable_weights))
        return mean_actor_loss, mean_critic_loss, loss

    def learn(self):
        """Run learning algorithm"""
        config = self.config
        n_updates = 0
        with self.writer.as_default():
            for _ in range(int(config["n_iter"])):
                # Collect trajectories until we get timesteps_per_batch total timesteps
                states, actions, advs, rs, _ = self.get_processed_trajectories()
                advs = np.array(advs)
                advs = (advs - advs.mean()) / advs.std()
                self.set_old_to_new()

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
                        train_actor_loss, train_critic_loss, train_loss = self.train(batch_states,
                                                                                     batch_actions,
                                                                                     batch_advs,
                                                                                     batch_rs)
                        tf.summary.scalar("model/loss", train_loss, step=n_updates)
                        tf.summary.scalar("model/actor_loss", train_actor_loss, step=n_updates)
                        tf.summary.scalar("model/critic_loss", train_critic_loss, step=n_updates)
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

    # def _specific_summaries(self) -> List[tf.Tensor]:
    #     summary_mean_mean = tf.summary.scalar("model/model/mean/mean", tf.reduce_mean(self.old_network.mean))
    #     summary_std_mean = tf.summary.scalar("model/model/std/mean", tf.reduce_mean(self.old_network.std))
    #     return [summary_mean_mean, summary_std_mean]

    def get_env_action(self, action):
        return action
