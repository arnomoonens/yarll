# -*- coding: utf8 -*-

#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

from pathlib import Path
from typing import Dict
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GRU
from tensorflow.keras import Model, Sequential
import tensorflow_addons as tfa
from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards
from yarll.misc.network_ops import NormalDistrLayer, flatten_to_rnn
from yarll.misc.reporter import Reporter


class REINFORCE(Agent):
    """
    REINFORCE with baselines
    """

    def __init__(self, env, monitor_path: str, monitor: bool = False, video: bool = True, **usercfg) -> None:
        """
        Initialize the video.

        Args:
            self: (todo): write your description
            env: (todo): write your description
            monitor_path: (str): write your description
            monitor: (todo): write your description
            video: (todo): write your description
            usercfg: (todo): write your description
        """
        super(REINFORCE, self).__init__(**usercfg)
        self.env = env
        if monitor:
            self.env = wrappers.Monitor(self.env,
                                        monitor_path,
                                        force=True,
                                        video_callable=(None if video else False))
        self.monitor_path = Path(monitor_path)
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
            save_model=False
        ))
        self.config.update(usercfg)

        self.network = self.build_network()
        self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.config["learning_rate"])
        self.writer = tf.summary.create_file_writer(str(self.monitor_path))

    def build_network(self) -> tf.keras.Model:
        """
        Build the keras.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def choose_action(self, state, features) -> Dict[str, np.ndarray]:
        """Choose an action."""
        raise NotImplementedError()

    @tf.function
    def train(self, states, actions_taken, advantages, features=None):
        """
        Train the model.

        Args:
            self: (todo): write your description
            states: (todo): write your description
            actions_taken: (str): write your description
            advantages: (todo): write your description
            features: (todo): write your description
        """
        states = tf.cast(states, dtype=tf.float32)
        actions_taken = tf.cast(actions_taken, dtype=tf.int32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        inp = states if features is None else [states, features]
        with tf.GradientTape() as tape:
            res = self.network(inp)
            if features is None:
                logits = res
            else:
                logits = res[0]
            log_probs = self.network.log_prob(actions_taken, logits)
            eligibility = log_probs * advantages
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
                if self.initial_features is not None:
                    features = np.concatenate([trajectory.features for trajectory in trajectories])
                loss = self.train(all_state,
                                  all_action,
                                  all_adv,
                                  features=tf.squeeze(features) if self.initial_features is not None else None)
                tf.summary.scalar("model/loss", loss, step=iteration)

                reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
        if self.config["save_model"]:
            tf.saved_model.save(self.network, self.monitor_path / "model")


class ActorDiscrete(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, n_actions, activation="tanh"):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            n_hidden_layers: (int): write your description
            n_hidden_units: (int): write your description
            n_actions: (todo): write your description
            activation: (str): write your description
        """
        super(ActorDiscrete, self).__init__()
        self.logits = Sequential()

        for _ in range(n_hidden_layers):
            self.logits.add(Dense(n_hidden_units, activation=activation))
        self.logits.add(Dense(n_actions))

    def call(self, inp):
        """
        Parameters ---------- inp.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        return self.logits(inp)

    def action(self, states):
        """
        Return the state action.

        Args:
            self: (todo): write your description
            states: (todo): write your description
        """
        logits = self.predict(states)
        probs = tf.nn.softmax(logits)
        return tf.random.categorical(tf.math.log(probs), 1)

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        """
        Evaluates the probability of actions.

        Args:
            self: (todo): write your description
            actions: (str): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
            logits: (todo): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
        """
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(actions, dtype=tf.int32), logits=logits)


class REINFORCEDiscrete(REINFORCE):
    def __init__(self, env, monitor_path: str, video: bool = True, **usercfg) -> None:
        """
        Initialize the environment.

        Args:
            self: (todo): write your description
            env: (todo): write your description
            monitor_path: (str): write your description
            video: (todo): write your description
            usercfg: (todo): write your description
        """
        super(REINFORCEDiscrete, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        """
        Return a superDisc network

        Args:
            self: (todo): write your description
        """
        return ActorDiscrete(self.config["n_hidden_layers"],
                             self.config["n_hidden_units"],
                             self.env.action_space.n)

    def choose_action(self, state, features) -> Dict[str, np.ndarray]:
        """Choose an action."""
        inp = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.network.action(inp).numpy()[0, 0]
        return {"action": action}

class ActorDiscreteCNN(Model):
    def __init__(self, n_actions, n_hidden_units, n_conv_layers=4, n_filters=32, kernel_size=3, strides=2, padding="same", activation="elu"):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            n_actions: (todo): write your description
            n_hidden_units: (int): write your description
            n_conv_layers: (int): write your description
            n_filters: (int): write your description
            kernel_size: (int): write your description
            strides: (int): write your description
            padding: (str): write your description
            activation: (str): write your description
        """
        super(ActorDiscreteCNN, self).__init__()
        self.conv_layers = Sequential()

        for _ in range(n_conv_layers):
            self.conv_layers.add(Conv2D(filters=n_filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=activation))
        self.conv_layers.add(Flatten())
        self.hidden = Dense(n_hidden_units)
        self.logits = Dense(n_actions)

    def call(self, state):
        """
        Compute the network.

        Args:
            self: (todo): write your description
            state: (todo): write your description
        """
        x = self.conv_layers(state)
        x = self.hidden(x)
        return self.logits(x)

    def action(self, states):
        """
        Return the state action.

        Args:
            self: (todo): write your description
            states: (todo): write your description
        """
        logits = self.predict(states)
        probs = tf.nn.softmax(logits)
        return tf.random.categorical(tf.math.log(probs), 1)

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        """
        Evaluates the probability of actions.

        Args:
            self: (todo): write your description
            actions: (str): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
            logits: (todo): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
        """
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(actions, dtype=tf.int32), logits=logits)

class REINFORCEDiscreteCNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        """
        Initialize video.

        Args:
            self: (todo): write your description
            env: (todo): write your description
            monitor_path: (str): write your description
            video: (todo): write your description
            usercfg: (todo): write your description
        """
        usercfg["n_hidden_units"] = 200
        super(REINFORCEDiscreteCNN, self).__init__(env, monitor_path, video=video, **usercfg)
        self.config.update(usercfg)

    def build_network(self):
        """
        Build an instance of superDiscCNN.

        Args:
            self: (todo): write your description
        """
        return ActorDiscreteCNN(self.env.action_space.n, self.config["n_hidden_units"])

class ActorDiscreteRNN(Model):
    def __init__(self, rnn_size, n_actions):
        """
        Initialize rnn.

        Args:
            self: (todo): write your description
            rnn_size: (int): write your description
            n_actions: (todo): write your description
        """
        super(ActorDiscreteRNN, self).__init__()
        self.expand = flatten_to_rnn
        self.rnn = GRU(rnn_size, return_state=True)
        self.logits = Dense(n_actions)

    def call(self, inp):
        """
        Return the hidden state.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        state, hidden = inp
        x = self.expand(state)
        x, new_hidden = self.rnn(x, hidden)
        return self.logits(x), new_hidden

    def action(self, inp):
        """
        Return the inverse of the model.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        logits, hidden = self.predict(inp)
        probs = tf.nn.softmax(logits)
        return tf.random.categorical(tf.math.log(probs), 1), hidden

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        """
        Evaluates the probability of actions.

        Args:
            self: (todo): write your description
            actions: (str): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
            logits: (todo): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
        """
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(actions, dtype=tf.int32), logits=logits)

class REINFORCEDiscreteRNN(REINFORCEDiscrete):
    def __init__(self, env, monitor_path, video=True, **usercfg):
        """
        Initialize the video.

        Args:
            self: (todo): write your description
            env: (todo): write your description
            monitor_path: (str): write your description
            video: (todo): write your description
            usercfg: (todo): write your description
        """
        super(REINFORCEDiscreteRNN, self).__init__(env, monitor_path, video=video, **usercfg)
        self.initial_features = tf.zeros((1, self.config["n_hidden_units"]))

    def build_network(self):
        """
        Create an instance of supernetwork.

        Args:
            self: (todo): write your description
        """
        return ActorDiscreteRNN(self.config["n_hidden_units"], self.env.action_space.n)

    def choose_action(self, state, features) -> Dict[str, np.ndarray]:
        """Choose an action."""
        inp = tf.convert_to_tensor([state], dtype=tf.float32)
        features = tf.reshape(features, (1, self.config["n_hidden_units"]))
        action, new_state = self.network.action([inp, features])
        return {"action": action.numpy()[0, 0], features: new_state}


class ActorDiscreteCNNRNN(Model):
    def __init__(self, rnn_size, n_actions):
        """
        Initialize rnn.

        Args:
            self: (todo): write your description
            rnn_size: (int): write your description
            n_actions: (todo): write your description
        """
        super(ActorDiscreteCNNRNN, self).__init__()
        self.conv_layers = Sequential()

        for _ in range(4):
            self.conv_layers.add(Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="elu"))
        self.conv_layers.add(Flatten())
        self.conv_layers.add(flatten_to_rnn)

        self.rnn = GRU(rnn_size, return_state=True)
        self.logits = Dense(n_actions)

    def call(self, inp):
        """
        Call the network.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        state, hidden = inp
        x = self.conv_layers(state)
        x, new_hidden = self.rnn(x, hidden)
        return self.logits(x), new_hidden

    def action(self, inp):
        """
        Return the inverse of the model.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        logits, hidden = self.predict(inp)
        probs = tf.nn.softmax(logits)
        return tf.random.categorical(tf.math.log(probs), 1), hidden

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        """
        Evaluates the probability of actions.

        Args:
            self: (todo): write your description
            actions: (str): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
            logits: (todo): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
        """
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(actions, dtype=tf.int32), logits=logits)

class REINFORCEDiscreteCNNRNN(REINFORCEDiscreteRNN):
    def build_network(self):
        """
        Return an instance of supernetwork.

        Args:
            self: (todo): write your description
        """
        return ActorDiscreteCNNRNN(self.config["n_hidden_units"], self.env.action_space.n)

class ActorBernoulli(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, n_actions, activation="tanh"):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            n_hidden_layers: (int): write your description
            n_hidden_units: (int): write your description
            n_actions: (todo): write your description
            activation: (str): write your description
        """
        super(ActorBernoulli, self).__init__()
        self.logits = Sequential()

        for _ in range(int(n_hidden_layers)):
            self.logits.add(Dense(n_hidden_units, activation=activation))
        self.logits.add(Dense(n_actions))

    def call(self, states):
        """
        See tf. _tensorflow graph.

        Args:
            self: (todo): write your description
            states: (todo): write your description
        """
        return self.logits(tf.convert_to_tensor(states, dtype=tf.float32))

    def action(self, inp):
        """
        Return the value of the model.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        logits = self.predict(inp)
        probs = tf.sigmoid(logits)
        samples_from_uniform = tf.random.uniform(probs.shape)
        return tf.cast(tf.less(samples_from_uniform, probs), tf.float32)

    def log_prob(self, actions: tf.Tensor, logits: tf.Tensor):
        """
        Computes the probability of all actions.

        Args:
            self: (todo): write your description
            actions: (str): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
            logits: (todo): write your description
            tf: (todo): write your description
            Tensor: (todo): write your description
        """
        return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(actions, tf.float32),
                                                                      logits=logits),
                              axis=-1)

class REINFORCEBernoulli(REINFORCE):

    def build_network(self):
        """
        Create a superBernoullioulli

        Args:
            self: (todo): write your description
        """
        return ActorBernoulli(self.config["n_hidden_units"], self.config["n_hidden_layers"], self.env.action_space.n)

    def choose_action(self, state, features):
        """Choose an action."""
        inp = tf.cast([state], tf.float32)
        action = self.network.action(inp)[0]
        return {"action": action}

class ActorContinuous(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, action_space_shape, activation="tanh"):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            n_hidden_layers: (int): write your description
            n_hidden_units: (int): write your description
            action_space_shape: (todo): write your description
            activation: (str): write your description
        """
        super(ActorContinuous, self).__init__()
        self.hidden = Sequential()

        for _ in range(int(n_hidden_layers)):
            self.hidden.add(Dense(n_hidden_units, activation=activation))
        self.action = NormalDistrLayer(action_space_shape[0])

    def call(self, inp):
        """
        Call the given action.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        x = self.hidden(inp)
        return self.action(x)


class ActorContinuousRNN(Model):
    def __init__(self, rnn_size, action_space_shape):
        """
        Initialize state space.

        Args:
            self: (todo): write your description
            rnn_size: (int): write your description
            action_space_shape: (todo): write your description
        """
        super(ActorContinuousRNN, self).__init__()
        # Change shape for RNN
        self.expand = flatten_to_rnn

        self.rnn = GRU(rnn_size, return_state=True)
        self.action = NormalDistrLayer(action_space_shape[0])

    def call(self, inp):
        """
        Return the state.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        state, hidden = inp
        x = self.expand(state)
        x, new_hidden = self.rnn(x, hidden)
        action, mean = self.action(x)
        return action, mean, new_hidden


class REINFORCEContinuous(REINFORCE):
    def __init__(self, env, monitor_path, rnn=False, video=True, **usercfg):
        """
        Initialize the video.

        Args:
            self: (todo): write your description
            env: (todo): write your description
            monitor_path: (str): write your description
            rnn: (int): write your description
            video: (todo): write your description
            usercfg: (todo): write your description
        """
        self.rnn = rnn
        self.initial_features = tf.zeros((1, self.config["n_hidden_units"])) if rnn else None
        super(REINFORCEContinuous, self).__init__(env, monitor_path, video=video, **usercfg)

    def build_network(self):
        """
        Builds the rnn network

        Args:
            self: (todo): write your description
        """
        return self.build_network_rnn() if self.rnn else self.build_network_normal()

    def choose_action(self, state, features):
        """Choose an action."""
        state = tf.cast([state], tf.float32)
        if self.rnn:
            features = tf.reshape(features, (1, self.config["n_hidden_units"]))
            inp = [state, features]
        else:
            inp = state
        res = self.network(inp)
        return {"action": res[0][0], "features": res[2] if self.rnn else None}

    def build_network_normal(self):
        """
        Return an instance of supernormalizers.

        Args:
            self: (todo): write your description
        """
        return ActorContinuous(self.config["n_hidden_layers"],
                               self.config["n_hidden_units"],
                               self.env.action_space.shape)

    def build_network_rnn(self):
        """
        Build an instance of rnnrnn.

        Args:
            self: (todo): write your description
        """
        return ActorContinuousRNN(self.config["n_hidden_units"], self.env.action_space.shape)

    @tf.function
    def train(self, states, actions_taken, advantages, features=None):
        """
        Train the model.

        Args:
            self: (todo): write your description
            states: (todo): write your description
            actions_taken: (str): write your description
            advantages: (todo): write your description
            features: (todo): write your description
        """
        states = tf.cast(states, dtype=tf.float32)
        advantages = tf.cast(advantages, dtype=tf.float32)
        inp = states if features is None else [states, tf.reshape(
            features, [features.shape[0], self.config["n_hidden_units"]])]
        with tf.GradientTape() as tape:
            res = self.network(inp)
            mean = res[1]
            log_std = self.network.layers[-1].log_std
            std = tf.exp(log_std)
            neglogprob = 0.5 * tf.reduce_sum(tf.square((actions_taken - mean) / std), axis=-1) \
                + 0.5 * tf.math.log(2.0 * np.pi) * std \
                + tf.reduce_sum(log_std, axis=-1)
            action_log_prob = -neglogprob
            eligibility = action_log_prob * advantages
            loss = -tf.reduce_sum(eligibility)
            gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        return loss
