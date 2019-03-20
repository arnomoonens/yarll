# -*- coding: utf8 -*-

import os
import logging
from typing import Optional, Tuple
import tensorflow as tf
import numpy as np

from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkDiscreteCNNRNN, actor_discrete_loss,\
    critic_loss, ActorCriticNetworkContinuous, actor_critic_continuous_loss
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Don't use the scientific notation to print results
np.set_printoptions(suppress=True)


class A2C(Agent):
    """Advantage Actor Critic"""

    def __init__(self, env, monitor_path: str, video: bool = True, **usercfg) -> None:
        super(A2C, self).__init__(**usercfg)
        self.monitor_path = monitor_path

        self.env = wrappers.Monitor(
            env,
            monitor_path,
            force=True,
            video_callable=(None if video else False))

        self.config.update(dict(
            n_iter=100,
            gamma=0.99,
            learning_rate=0.001,
            n_hidden_units=20,
            n_hidden_layers=1,
            gradient_clip_value=0.5,
            n_local_steps=20,
            vf_coef=0.5,
            entropy_coef=0.01,
            loss_reducer="mean",
            save_model=False
        ))
        self.config.update(usercfg)
        # Only used (and overwritten) by agents that use an RNN
        self.initial_features = None
        self.ac_net: tf.keras.Model = self.build_networks()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"],
                                                  clipnorm=self.config["gradient_clip_value"])
        self.writer = tf.summary.create_file_writer(self.monitor_path)
        return

    def build_networks(self):
        return NotImplementedError("Abstract method")

    def _actor_loss(self, actions, advantages, logits):
        return NotImplementedError("Abstract method")

    def _critic_loss(self, returns, value):
        return NotImplementedError("Abstract method")

    def train(self, states, actions_taken, advantages, returns, features=None):
        return NotImplementedError("Abstract method")

    def choose_action(self, state, features) -> dict:
        action, value = self.ac_net.action_value(state[None,:])
        return {"action": action, "value": value[0]}

    def learn(self):
        """Run learning algorithm"""
        env_runner = EnvRunner(self.env, self, self.config)
        config = self.config
        with self.writer.as_default():
            for iteration in range(int(config["n_iter"])):
                # Collect trajectories until we get timesteps_per_batch total timesteps
                trajectory = env_runner.get_steps(int(self.config["n_local_steps"]))
                features = trajectory.features
                features = np.concatenate(trajectory.features) if features[-1] is not None else np.array([None])
                if trajectory.experiences[-1].terminal:
                    v = 0
                else:
                    inp = [np.asarray(trajectory.states)[None, -1]]
                    if features[-1] is not None:
                        inp.append(features[None, -1])
                    v = self.ac_net.action_value(*inp)[1][0]
                rewards_plus_v = np.asarray(trajectory.rewards + [v])
                vpred_t = np.asarray(trajectory.values + [v])
                delta_t = trajectory.rewards + \
                    self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
                batch_r = discount_rewards(
                    rewards_plus_v, self.config["gamma"])[:-1]
                batch_adv = discount_rewards(delta_t, self.config["gamma"])
                states = np.asarray(trajectory.states)
                iter_actor_loss, iter_critic_loss, iter_loss = self.train(states,
                                                                          np.asarray(trajectory.actions),
                                                                          batch_adv,
                                                                          batch_r,
                                                                          features=features if features[-1] is not None else None)
                tf.summary.scalar("model/loss", iter_loss, step=iteration)
                tf.summary.scalar("model/actor_loss", iter_actor_loss, step=iteration)
                tf.summary.scalar("model/critic_loss", iter_critic_loss, step=iteration)
            if self.config["save_model"]:
                tf.saved_model.save(self.ac_net, os.path.join(self.monitor_path, "model"))


class A2CDiscrete(A2C):

    def build_networks(self):
        return ActorCriticNetworkDiscrete(
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    @tf.function
    def train(self, states, actions_taken, advantages, returns, features=None):
        states = tf.cast(states, dtype=tf.float32)
        actions_taken = tf.cast(actions_taken, dtype=tf.int32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        returns = tf.cast(returns, dtype=tf.float32)
        inp = states if features is None else [states, tf.cast(features, tf.float32)]
        with tf.GradientTape() as tape:
            res = self.ac_net(inp)
            logits = res[0]
            values = res[1]
            mean_actor_loss = tf.reduce_mean(self._actor_loss(actions_taken, advantages, logits))
            mean_critic_loss = tf.reduce_mean(self._critic_loss(returns, values))
            loss = mean_actor_loss + self.config["vf_coef"] * mean_critic_loss
            gradients = tape.gradient(loss, self.ac_net.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.ac_net.trainable_weights))
        return mean_actor_loss, mean_critic_loss, loss

    def _actor_loss(self, actions, advantages, logits):
        return actor_discrete_loss(actions, advantages, logits)

    def _critic_loss(self, returns, value):
        return self.config["vf_coef"] * critic_loss(returns, value)


class A2CDiscreteCNN(A2CDiscrete):
    def build_networks(self):
        return ActorCriticNetworkDiscreteCNN(
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class A2CDiscreteCNNRNN(A2CDiscrete):
    def __init__(self, *args, **kwargs):
        super(A2CDiscreteCNNRNN, self).__init__(*args, **kwargs)
        self.initial_features = self.ac_net.initial_features

    def build_networks(self):
        return ActorCriticNetworkDiscreteCNNRNN(self.env.action_space.n)

    def choose_action(self, state, features) -> dict:
        """Choose an action."""
        action, value, rnn_state = self.ac_net.action_value(state[None, :], features)
        return {"action": action, "value": value[0], "features": rnn_state}

class A2CContinuous(A2C):
    def __init__(self, *args, **kwargs):
        super(A2CContinuous, self).__init__(*args, **kwargs)

    def build_networks(self):
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def make_loss(self):
        return actor_critic_continuous_loss(
            self.ac_net.action_log_prob,
            self.ac_net.entropy,
            self.ac_net.value,
            self.advantage,
            self.ret,
            self.config["vf_coef"],
            self.config["entropy_coef"],
            self.config["loss_reducer"]
        )

    def get_env_action(self, action):
        return action
