# -*- coding: utf8 -*-

#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

import os
from typing import Dict
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GRU
from tensorflow.keras import Model
from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards, flatten
from yarll.misc.network_ops import NormalDistrLayer
from yarll.misc.reporter import Reporter


class REINFORCE(Agent):
    """
    REINFORCE with baselines
    """

    def __init__(self, env, monitor_path: str, monitor: bool = False, video: bool = True, **usercfg) -> None:
        super(REINFORCE, self).__init__(**usercfg)
        self.env = env
        if monitor:
            self.env = wrappers.Monitor(self.env,
                                        monitor_path,
                                        force=True,
                                        video_callable=(None if video else False))
        self.monitor_path = monitor_path
        # Default configuration. Can be overwritten using keyword arguments.
        self.config.update(dict(
            batch_update="timesteps",
            timesteps_per_batch=1000,
            n_iter=100,
            gamma=0.99,
            learning_rate=0.05,
            entropy_coef=1e-3,
            n_hidden_layers=2,
            n_hidden_units=20,
            repeat_n_actions=1,
            save_model=False
        ))
        self.config.update(usercfg)

        self.states = tf.keras.Input(self.env.observation_space.shape, name="states")
        self.network = self.build_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"])
        self.writer = tf.summary.create_file_writer(self.monitor_path)

    def build_network(self) -> tf.keras.Model:
        raise NotImplementedError()

    def choose_action(self, state, features) -> Dict[str, np.ndarray]:
        """Choose an action."""
        inp = tf.convert_to_tensor([state])
        probs = self.network(inp)
        action = tf.random.categorical(tf.math.log(probs), 1).numpy()[0, 0]
        return {"action": action}

    @tf.function
    def train(self, states, actions_taken, advantages, features=None):
        states = tf.cast(states, dtype=tf.float32)
        actions_taken = tf.cast(actions_taken, dtype=tf.int32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        inp = states if features is None else [states, tf.reshape(features, [features.shape[0], 32])]
        with tf.GradientTape() as tape:
            res = self.network(inp)
            probs = res if features is None else res[0]
            good_probs = tf.reduce_sum(tf.multiply(probs,
                                                   tf.one_hot(actions_taken, self.env.action_space.n)),
                                       axis=1)
            eligibility = tf.math.log(good_probs) * advantages
            loss = -tf.reduce_sum(eligibility)
            gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        return float(loss)

    def learn(self):
        """Run learning algorithm"""
        env_runner = EnvRunner(self.env, self, self.config)
        reporter = Reporter()
        config = self.config
        total_n_trajectories = 0
        with self.writer.as_default():
            for iteration in range(config["n_iter"]):
                # Collect trajectories until we get timesteps_per_batch total timesteps
                trajectories = env_runner.get_trajectories()
                total_n_trajectories += len(trajectories)
                all_state = np.concatenate([trajectory.states for trajectory in trajectories])
                # Compute discounted sums of rewards
                rets = [discount_rewards(trajectory.rewards, config["gamma"]) for trajectory in trajectories]
                max_len = max(len(ret) for ret in rets)
                padded_rets = [np.concatenate([ret, np.zeros(max_len - len(ret))]) for ret in rets]
                # Compute time-dependent baseline
                baseline = np.mean(padded_rets, axis=0)
                # Compute advantage function
                advs = [ret - baseline[:len(ret)] for ret in rets]
                all_action = np.concatenate([trajectory.actions for trajectory in trajectories])
                all_adv = np.concatenate(advs)
                # Do policy gradient update step
                episode_rewards = np.array([sum(trajectory.rewards)
                                            for trajectory in trajectories])  # episode total rewards
                episode_lengths = np.array([len(trajectory.rewards) for trajectory in trajectories])  # episode lengths
                features = np.concatenate([trajectory.features for trajectory in trajectories])
                loss = self.train(all_state,
                                  all_action,
                                  all_adv,
                                  features=features if self.initial_features is not None else None)
                tf.summary.scalar("model/loss", loss, step=iteration)

                reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
        if self.config["save_model"]:
            tf.saved_model.save(self.network, os.path.join(self.monitor_path, "model"))

class REINFORCEDiscrete(REINFORCE):
    def __init__(self, env, monitor_path: str, video: bool = True, **usercfg) -> None:
        super(REINFORCEDiscrete, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        x = self.states
        for _ in range(self.config["n_hidden_layers"]):
            x = Dense(self.config["n_hidden_units"], activation="tanh")(x)
        output = Dense(self.env.action_space.n, activation="softmax")(x)

        return Model(self.states, output, name="network")


class REINFORCEDiscreteCNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        usercfg["n_hidden_units"] = 200
        super(REINFORCEDiscreteCNN, self).__init__(env, monitor_path, video=video, **usercfg)
        self.config.update(usercfg)

    def build_network(self):
        shape = list(self.env.observation_space.shape)

        x = self.states

        # Convolution layers
        for _ in range(4):
            x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="elu")(x)

        # Flatten
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Fully connected layers
        x = Dense(self.config["n_hidden_units"], activation="relu")(x)
        outputs = Dense(self.env.action_space.n, activation="softmax")(x)

        return Model(self.states, outputs, name="network")


class REINFORCEDiscreteRNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        self.rnn_state_in = tf.keras.Input((32,))
        super(REINFORCEDiscreteRNN, self).__init__(env, monitor_path, video=video, **usercfg)
        self.initial_features = tf.zeros((1, 32))

    def build_network(self):
        states = tf.expand_dims(self.states, [1])

        rnn = GRU(32, return_state=True)
        x, new_state = rnn(states, self.rnn_state_in)
        probs = Dense(self.env.action_space.n, activation="softmax")(x)
        return Model([self.states, self.rnn_state_in], [probs, new_state])

    def choose_action(self, state, features):
        """Choose an action."""
        inp = tf.cast([state], tf.float32)
        features = tf.reshape(features, (1, 32))
        probs, new_state = self.network([inp, features])
        action = tf.random.categorical(tf.math.log(probs), 1).numpy()[0, 0]
        return {"action": action, "features": new_state}


class REINFORCEDiscreteCNNRNN(REINFORCEDiscreteRNN):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(REINFORCEDiscreteCNNRNN, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):

        shape = list(self.env.observation_space.shape)
        x = self.states

        # Convolution layers
        for _ in range(4):
            x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="elu")(x)

        # Flatten
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # -1 for the (unknown) batch size

        # Change shape for RNN
        states = tf.expand_dims(x, [1])

        rnn = GRU(32, return_state=True)
        x, new_state = rnn(states, self.rnn_state_in)
        probs = Dense(self.env.action_space.n, activation="softmax")(x)
        return Model([self.states, self.rnn_state_in], [probs, new_state])


class REINFORCEContinuous(REINFORCEDiscreteRNN):
    def __init__(self, env, monitor_path, RNN=False, video=True, **usercfg):
        self.rnn = RNN
        super(REINFORCEContinuous, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        return self.build_network_rnn() if self.rnn else self.build_network_normal()

    def choose_action(self, state, features):
        """Choose an action."""
        state = tf.cast([state], tf.float32)
        if self.rnn:
            features = tf.reshape(features, (1, 32))
            inp = [state, features]
        else:
            inp = state
        res = self.network(inp)
        return {"action": res[0][0], "features": res[2] if self.rnn else None}

    def build_network_normal(self):

        x = self.states
        for _ in range(int(self.config["n_hidden_layers"])):
            x = Dense(self.config["n_hidden_units"], activation="tanh")(x)
        action, mean = NormalDistrLayer(self.env.action_space.shape[0])(x)

        return Model(self.states, [action, mean])

    def build_network_rnn(self):

        # Change shape for RNN
        states = tf.expand_dims(self.states, [1])

        rnn = GRU(32, return_state=True)
        x, new_state = rnn(states, self.rnn_state_in)
        action, mean = NormalDistrLayer(self.env.action_space.shape[0])(x)
        return Model([self.states, self.rnn_state_in], [action, mean, new_state])

    @tf.function
    def train(self, states, actions_taken, advantages, features=None):
        states = tf.cast(states, dtype=tf.float32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        inp = states if features is None else [states, tf.reshape(features, [features.shape[0], 32])]
        with tf.GradientTape() as tape:
            res = self.network(inp)
            mean = res[1]
            log_std = self.network.layers[-1].log_std
            std = tf.exp(log_std)
            neglogprob = 0.5 * tf.reduce_sum(tf.square((actions_taken - mean) / std), axis=-1) \
                + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(actions_taken)[-1], tf.float32) \
                + tf.reduce_sum(log_std, axis=-1)
            action_log_prob = -neglogprob
            eligibility = action_log_prob * advantages
            loss = -tf.reduce_sum(eligibility)
            gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        return float(loss)
