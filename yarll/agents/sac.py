# -*- coding: utf8 -*-

"""
Soft Actor-Critic

Based on SAC implementation from https://github.com/rail-berkeley/softlearning
"""

from copy import deepcopy
import csv
from pathlib import Path
from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.memory.memory import Memory
from yarll.misc.utils import hard_update, soft_update


class SAC(Agent):
    def __init__(self, env, monitor_path: Union[Path, str], **usercfg) -> None:
        super(SAC, self).__init__(**usercfg)
        self.env = env
        self.monitor_path = Path(monitor_path)

        self.config.update(
            max_steps=100000,
            actor_learning_rate=3e-4,
            softq_learning_rate=3e-4,
            alpha_learning_rate=3e-4,
            n_softqs=2,
            reward_scale=1.0,
            n_hidden_layers=2,
            n_hidden_units=256,
            gamma=0.99,
            batch_size=256,
            tau=0.005,
            logprob_epsilon=1e-6,  # For numerical stability when computing tf.log
            n_train_steps=1,  # Number of parameter update steps per iteration
            replay_buffer_size=1e6,
            replay_start_size=256,  # Required number of replay buffer entries to start training
            hidden_layer_activation="relu",
            normalize_inputs=False,
            checkpoints=True,
            save_model=True
        )
        self.config.update(usercfg)

        self.state_shape: list = list(env.observation_space.shape)
        self.n_actions: int = env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        # TODO: check if author always does it like this or gives a fixed target entropy as parameter
        self.target_entropy = -np.prod(env.action_space.shape)

        # Make networks
        # action_output are the squashed actions and action_original those straight from the normal distribution
        self.actor_network = ActorNetwork(self.config["n_hidden_layers"],
                                          self.config["n_hidden_units"],
                                          self.n_actions,
                                          self.config["logprob_epsilon"],
                                          self.config["hidden_layer_activation"])
        self.softq_networks = [SoftQNetwork(self.config["n_hidden_layers"],
                                            self.config["n_hidden_units"],
                                            self.config["hidden_layer_activation"])
                               for _ in range(self.config["n_softqs"])]
        self.target_softq_networks = [deepcopy(net) for net in self.softq_networks]

        dummy_input_states = tf.random.uniform((1, *self.state_shape))
        dummy_input_actions = tf.random.uniform((1, *self.env.action_space.shape))
        for net, target_net in zip(self.softq_networks, self.target_softq_networks):
            net(dummy_input_states, dummy_input_actions)
            target_net(dummy_input_states, dummy_input_actions)
            hard_update(net.variables, target_net.variables)

        self._alpha = tf.Variable(tf.exp(0.0), name='alpha')

        # Make train ops
        self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.config["actor_learning_rate"])
        self.softq_optimizers = [tfa.optimizers.RectifiedAdam(learning_rate=self.config["softq_learning_rate"])
                                 for _ in self.softq_networks]
        self.alpha_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.config["alpha_learning_rate"])

        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))
        self.n_updates = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.writer = tf.summary.create_file_writer(str(self.monitor_path))

        self.env_runner = EnvRunner(self.env,
                                    self,
                                    usercfg,
                                    scale_states=self.config["normalize_inputs"])

        if self.config["checkpoints"]:
            checkpoint_directory = self.monitor_path / "checkpoints"
            self.ckpt = tf.train.Checkpoint(net=self.actor_network)
            self.cktp_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_directory, 10)
            self.checkpoint_every_episodes = 10

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        return self.actor_network(states)[0]

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return self.actor_network(state[None, :])[0].numpy()[0]

    @tf.function
    def train(self, state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch):
        # Calculate critic targets

        next_action_batch, next_logprob_batch = self.actor_network(state1_batch)
        next_qs_values = [net(state1_batch, next_action_batch) for net in self.target_softq_networks]
        next_q_values = tf.reduce_min(next_qs_values, axis=0)
        next_values = next_q_values - self._alpha * next_logprob_batch
        next_values = tf.expand_dims(1.0 - terminal1_batch, 1) * next_values
        softq_targets = self.config["reward_scale"] * tf.expand_dims(reward_batch, 1) + self.config["gamma"] * next_values
        softq_targets = tf.reshape(softq_targets, [self.config["batch_size"]])

        # Update critics
        softq_losses = []
        for net, optimizer in zip(self.softq_networks, self.softq_optimizers):
            with tf.GradientTape() as tape:
                softq = net(state0_batch, action_batch)
                softq_loss = 0.5 * tf.losses.MSE(y_true=softq_targets, y_pred=tf.squeeze(softq))
                softq_losses.append(tf.stop_gradient(softq_loss))
            softq_gradients = tape.gradient(softq_loss, net.trainable_weights)
            optimizer.apply_gradients(zip(softq_gradients, net.trainable_weights))

        # Update actor
        with tf.GradientTape() as tape:
            actions, action_logprob = self.actor_network(state0_batch)
            softqs_pred = [net(state0_batch, actions) for net in self.softq_networks]
            softq_pred = tf.reduce_min(softqs_pred, axis=0)
            actor_loss = self._alpha * action_logprob - softq_pred
        actor_gradients = tape.gradient(actor_loss, self.actor_network.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_weights))

        # Update alpha
        _, action_logprob = self.actor_network(state0_batch)
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * self._alpha * tf.stop_gradient(action_logprob + self.target_entropy)  # For batch
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
        alpha_gradients = tape.gradient(alpha_loss, [self._alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self._alpha]))

        softq_mean, softq_variance = tf.nn.moments(softq, axes=[0])
        return softq_mean[0], tf.sqrt(softq_variance[0]), softq_targets, tf.reduce_mean(softq_losses), tf.reduce_mean(actor_loss), alpha_loss, tf.reduce_mean(action_logprob)

    def learn(self):
        # Arrays to keep results from train function over different train steps in
        softq_means = np.empty((self.config["n_train_steps"],), np.float32)
        softq_stds = np.empty((self.config["n_train_steps"],), np.float32)
        softq_losses = np.empty((self.config["n_train_steps"],), np.float32)
        actor_losses = np.empty((self.config["n_train_steps"],), np.float32)
        alpha_losses = np.empty((self.config["n_train_steps"],), np.float32)
        action_logprob_means = np.empty((self.config["n_train_steps"],), np.float32)
        to_save = []
        total_episodes = 0
        with self.writer.as_default():
            for _ in range(self.config["max_steps"]):
                experience = self.env_runner.get_steps(1)[0]
                self.total_steps += 1
                self.replay_buffer.add(experience.state, experience.action, experience.reward,
                                       experience.next_state, experience.terminal)
                to_save.append(experience.state.tolist() + experience.action.tolist() +
                               [experience.reward] + experience.next_state.tolist())
                if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                    for i in range(self.config["n_train_steps"]):
                        sample = self.replay_buffer.get_batch(self.config["batch_size"])
                        softq_mean, softq_std, softq_targets, softq_loss, actor_loss, alpha_loss, action_logprob_mean = self.train(
                            sample["states0"],
                            np.resize(sample["actions"], [self.config["batch_size"],
                                                          self.n_actions]),  # for n_actions == 1
                            sample["rewards"],
                            sample["states1"],
                            sample["terminals1"])
                        softq_means[i] = softq_mean
                        softq_stds[i] = softq_std
                        softq_losses[i] = softq_loss
                        actor_losses[i] = actor_loss
                        alpha_losses[i] = alpha_loss
                        action_logprob_means[i] = action_logprob_mean
                    tf.summary.scalar("model/predicted_softq_mean", np.mean(softq_means), self.total_steps)
                    tf.summary.scalar("model/predicted_softq_std", np.mean(softq_stds), self.total_steps)
                    tf.summary.scalar("model/softq_targets", np.mean(softq_targets), self.total_steps)
                    tf.summary.scalar("model/softq_loss", np.mean(softq_losses), self.total_steps)
                    tf.summary.scalar("model/actor_loss", np.mean(actor_losses), self.total_steps)
                    tf.summary.scalar("model/alpha_loss", np.mean(alpha_losses), self.total_steps)
                    tf.summary.scalar("model/alpha", self._alpha, self.total_steps)
                    tf.summary.scalar("model/action_logprob_mean", np.mean(action_logprob_means), self.total_steps)
                    # Update the target networks
                    for net, target_net in zip(self.softq_networks, self.target_softq_networks):
                        soft_update(net.variables,
                                    target_net.variables,
                                    self.config["tau"])
                    self.n_updates += 1
                if experience.terminal:
                    total_episodes += 1
                    if self.config["checkpoints"] and (total_episodes % self.checkpoint_every_episodes) == 0:
                        self.cktp_manager.save()
        with open(self.monitor_path / "experiences.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(to_save)
        if self.config["save_model"]:
            tf.saved_model.save(self.actor_network, str(self.monitor_path / "model.h5"))

    def choose_action(self, state, features):
        return {"action": self.action(state)}

    def get_env_action(self, action):
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)


class ActorNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, n_actions, logprob_epsilon, hidden_layer_activation="relu"):
        super(ActorNetwork, self).__init__()
        self.logprob_epsilon = logprob_epsilon
        w_bound = 3e-3
        self.hidden = Sequential()
        for _ in range(n_hidden_layers):
            self.hidden.add(Dense(n_hidden_units, activation=hidden_layer_activation))

        self.mean = Dense(n_actions,
                          kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                          bias_initializer=tf.random_uniform_initializer(-w_bound, w_bound))
        self.log_std = Dense(n_actions,
                             kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                             bias_initializer=tf.random_uniform_initializer(-w_bound, w_bound))

    def call(self, inp):
        x = self.hidden(inp)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = normal_dist.sample() # gradient flows through this; using reparameterization trick
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_actions, 2) + self.logprob_epsilon)
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)
        return squashed_actions, logprob

    @tf.function
    def deterministic_actions(self, inp):
        x = self.hidden(inp)
        return tf.tanh(self.mean(x))


class SoftQNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, hidden_layer_activation="relu"):
        super(SoftQNetwork, self).__init__()
        self.softq = Sequential()
        for _ in range(n_hidden_layers):
            self.softq.add(Dense(n_hidden_units, activation=hidden_layer_activation))
        self.softq.add(Dense(1,
                             kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                             bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)))

    def call(self, states, actions):
        x = tf.concat([states, actions], 1)
        return self.softq(x)
