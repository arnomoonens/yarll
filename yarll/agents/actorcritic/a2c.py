# -*- coding: utf8 -*-

import os
import logging
from typing import Callable, Optional, Tuple
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

        loss_functions = self._loss_functions()

        self.ac_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"],
                                                               clipnorm=self.config["gradient_clip_value"]),
                            loss=loss_functions)
        self.writer = tf.summary.create_file_writer(self.monitor_path)
        return

    def build_networks(self):
        return NotImplementedError("Abstract method")

    def _loss_functions(self):
        return NotImplementedError("Abstract method")

    def train(self, states: np.ndarray, actions_taken: np.ndarray, advantages: np.ndarray, features: Optional[np.ndarray] = None) -> Tuple[float, float]:
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
                v = 0 if trajectory.terminals[-1] else self.ac_net.action_value(
                    np.asarray(trajectory.states)[None, -1])[1]
                rewards_plus_v = np.asarray(trajectory.rewards + [v])
                vpred_t = np.asarray(trajectory.values + [v])
                delta_t = trajectory.rewards + \
                    self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
                batch_r = discount_rewards(
                    rewards_plus_v, self.config["gamma"])[:-1]
                batch_adv = discount_rewards(delta_t, self.config["gamma"])
                states = np.asarray(trajectory.states)
                acts_and_advs = np.concatenate([np.asarray(trajectory.actions)[:, None], batch_adv[:, None]], axis=-1)
                # performs a full training step on the collected batch
                # note: no need to mess around with gradients, Keras API handles it
                # First output seems to be the sum of the losses
                _, train_actor_loss, train_critic_loss = self.ac_net.train_on_batch(states, [acts_and_advs, batch_r])
                tf.summary.scalar("model/actor_loss", train_actor_loss, step=iteration)
                tf.summary.scalar("model/critic_loss", train_critic_loss, step=iteration)
            if self.config["save_model"]:
                tf.saved_model.save(self.ac_net, os.path.join(self.monitor_path, "model"))


class A2CDiscrete(A2C):
    def __init__(self, *args, **kwargs):
        super(A2CDiscrete, self).__init__(*args, **kwargs)

    def build_networks(self):
        return ActorCriticNetworkDiscrete(
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def _loss_functions(self):
        return [actor_discrete_loss, critic_loss]


class A2CDiscreteCNN(A2CDiscrete):
    def build_networks(self):
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class A2CDiscreteCNNRNN(A2CDiscrete):
    def build_networks(self):
        return ActorCriticNetworkDiscreteCNNRNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))
        self.initial_features = self.ac_net.state_init

    def choose_action(self, state, features) -> dict:
        """Choose an action."""
        feed_dict = {
            self.ac_net.states: [state],
            self.ac_net.rnn_state_in: features
        }

        action, rnn_state, value = self.session.run(
            [self.ac_net.action, self.ac_net.rnn_state_out, self.ac_net.value],
            feed_dict=feed_dict)
        return {"action": action, "value": value[0], "features": rnn_state}

    def get_critic_value(self, states, features):
        feed_dict = {
            self.ac_net.states: states,
            self.ac_net.rnn_state_in: features
        }
        return self.session.run(self.ac_net.value, feed_dict=feed_dict)[0]


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
