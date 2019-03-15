# -*- coding: utf8 -*-

#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Other differences:
#  - Always choose the action with the highest probability
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
from yarll.misc.network_ops import conv2d, linear, normalized_columns_initializer
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
                # TODO: deal with RNN state
                features = np.concatenate([trajectory.features for trajectory in trajectories])
                loss = self.train(all_state, all_action, all_adv, features=features if self.initial_features is not None else None)
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


class REINFORCEContinuous(REINFORCE):
    def __init__(self, env, monitor_path, RNN=False, video=True, **usercfg):
        self.rnn = RNN
        super(REINFORCEContinuous, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        return self.build_network_rnn() if self.rnn else self.build_network_normal()

    def make_trainer(self):
        loss = tf.reduce_mean(-self.action_log_prob * self.advantage) - self.config["entropy_coef"] * self.entropy
        self.summary_loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"])
        self.train = optimizer.minimize(loss)

    def build_network_normal(self):

        self.actions_taken = tf.placeholder(
            tf.float32,
            [None] + list(self.env.action_space.shape),
            name="actions_taken")

        x = self.states
        for i in range(int(self.config["n_hidden_layers"])):
            x = tf.tanh(linear(x, int(self.config["n_hidden_units"]), "L{}_mean".format(i + 1),
                               initializer=normalized_columns_initializer(1.0)))
        self.mean = linear(x, self.env.action_space.shape[0], "mean", initializer=normalized_columns_initializer(0.01))
        self.mean = tf.check_numerics(self.mean, "mean")

        self.log_std = tf.get_variable(
            name="logstd",
            shape=list(self.env.action_space.shape),
            initializer=tf.zeros_initializer()
        )
        std = tf.exp(self.log_std, name="std")
        std = tf.check_numerics(std, "std")

        self.action = self.mean + std * tf.random_normal(tf.shape(self.mean))
        self.action = tf.reshape(self.action, list(self.env.action_space.shape))

        neglogprob = 0.5 * tf.reduce_sum(tf.square((self.actions_taken - self.mean) / std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.actions_taken)[-1]) \
            + tf.reduce_sum(self.log_std, axis=-1)
        self.action_log_prob = -neglogprob
        self.entropy = tf.reduce_sum(self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def build_network_rnn(self):
        n_states = tf.shape(self.states)[:1]
        states = tf.expand_dims(flatten(self.states), [0])

        enc_cell = tf.contrib.rnn.GRUCell(int(self.config["n_hidden_units"]))
        L1, _ = tf.nn.dynamic_rnn(cell=enc_cell, inputs=states,
                                  sequence_length=n_states, dtype=tf.float32)

        L1 = L1[0]

        mu, sigma = mu_sigma_layer(L1, 1)

        self.normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.action = self.normal_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])
