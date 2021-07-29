import os
import logging
from typing import Optional, Tuple
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.tf2.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkDiscreteCNNRNN, actor_discrete_loss,\
    critic_loss, ActorCriticNetworkContinuous, actor_continuous_loss
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards
from yarll.misc import summary_writer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Don't use the scientific notation to print results
np.set_printoptions(suppress=True)


class A2C(Agent):
    """Advantage Actor Critic"""

    def __init__(self, env, monitor_path: str, video: bool = True, **usercfg) -> None:
        super().__init__(**usercfg)
        self.monitor_path = Path(monitor_path)

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

        self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.config["learning_rate"],
                                                      clipnorm=self.config["gradient_clip_value"])
        self.summary_writer = tf.summary.create_file_writer(str(self.monitor_path))
        summary_writer.set(self.summary_writer)

    def build_networks(self):
        return NotImplementedError("Abstract method")

    def _actor_loss(self, actions, advantages, logits):
        return NotImplementedError("Abstract method")

    def _critic_loss(self, returns, value):
        return self.config["vf_coef"] * critic_loss(returns, value)

    def train(self, states, actions_taken, advantages, returns, features=None):
        return NotImplementedError("Abstract method")

    def choose_action(self, state, features) -> dict:
        action, value = self.ac_net.action_value(state[None,:])
        return {"action": action, "value": value[0]}

    def learn(self):
        """Run learning algorithm"""
        env_runner = EnvRunner(self.env, self, self.config,
                               summaries_every_episodes=self.config.get("env_summaries_every_episodes", None),
                               transition_preprocessor=self.config.get("transition_preprocessor", None),
                               )
        config = self.config
        summary_writer.start()
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
                v = self.ac_net.action_value(*inp)[-2 if features[-1] is not None else -1][0]
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
            summary_writer.add_scalar("model/loss", iter_loss, step=iteration)
            summary_writer.add_scalar("model/actor_loss", iter_actor_loss, step=iteration)
            summary_writer.add_scalar("model/critic_loss", iter_critic_loss, step=iteration)
        summary_writer.stop()
        if self.config["save_model"]:
            tf.saved_model.save(self.ac_net, str(self.monitor_path / "model"))


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

class A2CDiscreteCNN(A2CDiscrete):
    def build_networks(self):
        return ActorCriticNetworkDiscreteCNN(
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class A2CDiscreteCNNRNN(A2CDiscrete):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_features = self.ac_net.initial_features

    def build_networks(self):
        return ActorCriticNetworkDiscreteCNNRNN(self.env.action_space.n)

    def choose_action(self, state, features) -> dict:
        """Choose an action."""
        action, value, rnn_state = self.ac_net.action_value(state[None, :], features)
        return {"action": action, "value": value[0], "features": rnn_state}

class A2CContinuous(A2C):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_networks(self):
        return ActorCriticNetworkContinuous(
            self.env.action_space.shape,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    @tf.function
    def train(self, states, actions_taken, advantages, returns, features=None):
        states = tf.cast(states, dtype=tf.float32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        returns = tf.cast(returns, dtype=tf.float32)
        inp = states if features is None else [states, tf.reshape(
            features, [features.shape[0], self.config["n_hidden_units"]])]
        with tf.GradientTape() as tape:
            res = self.ac_net(inp)
            mean = res[1]
            values = res[2]
            log_std = self.ac_net.action_mean.log_std
            mean_actor_loss = -tf.reduce_mean(self._actor_loss(actions_taken, mean, log_std, advantages))
            mean_critic_loss = tf.reduce_mean(self._critic_loss(returns, values))
            loss = mean_actor_loss + self.config["vf_coef"] * mean_critic_loss
            gradients = tape.gradient(loss, self.ac_net.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.ac_net.trainable_weights))
        return mean_actor_loss, mean_critic_loss, loss

    def choose_action(self, state, features) -> dict:
        action, _, value = self.ac_net.action_value(state[None, :])
        return {"action": action, "value": value[0]}

    def _actor_loss(self, actions_taken, mean, log_std, advantages):
        return actor_continuous_loss(actions_taken, mean, log_std, advantages)

    def get_env_action(self, action):
        return action
