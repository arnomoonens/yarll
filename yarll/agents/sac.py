# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

from yarll.agents.agent import Agent
from yarll.memory.memory import Memory
from yarll.misc.utils import hard_update, soft_update

class SAC(Agent):
    def __init__(self, env, monitor_path: str, **usercfg) -> None:
        super(SAC, self).__init__(**usercfg)
        self.env = env
        self.monitor_path: str = monitor_path

        self.config.update(
            n_episodes=100000,
            n_timesteps=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"),
            actor_learning_rate=3e-4,
            softq_learning_rate=3e-4,
            value_learning_rate=3e-4,
            n_hidden_layers=2,
            gamma=0.99,
            batch_size=128,
            tau=0.01,
            n_actor_layers=2,
            logprob_epsilon=1e-6, # For numerical stability when computing tf.log
            n_hidden_units=128,
            n_train_steps=4, # Number of parameter update steps per iteration
            replay_buffer_size=1e6,
            replay_start_size=128,  # Required number of replay buffer entries to start training
            weights_summary=False # Add weight distributions graphs to Tensorboard (can take up a lot of space)
        )
        self.config.update(usercfg)

        self.state_shape: list = list(env.observation_space.shape)
        self.n_actions: int = env.action_space.shape[0]

        # Make networks
        # action_output are the squashed actions and action_original those straight from the normal distribution
        self.actor_network = ActorNetwork(self.config["n_hidden_layers"],
                                          self.config["n_hidden_units"],
                                          self.n_actions,
                                          self.config["logprob_epsilon"])
        self.softq_network = SoftQNetwork(self.config["n_hidden_layers"], self.config["n_hidden_units"])
        self.value_network = ValueNetwork(self.config["n_hidden_layers"], self.config["n_hidden_units"])
        self.target_value_network = ValueNetwork(self.config["n_hidden_layers"], self.config["n_hidden_units"])

        input_shape = (None, *self.state_shape)
        self.value_network.build(input_shape)
        self.target_value_network.build(input_shape)
        hard_update(self.value_network.variables, self.target_value_network.variables)

        # Make train ops
        self.softq_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["softq_learning_rate"])
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["value_learning_rate"])
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["actor_learning_rate"])

        # if self.config["weights_summary"]:
        #     summaries = []
        #     for v in self.actor_vars + self.softq_vars + self.value_vars:
        #         summaries.append(tf.summary.histogram(v.name, v))
        #     self.model_summary_op = tf.summary.merge(summaries)

        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))
        self.n_updates = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.writer = tf.summary.create_file_writer(self.monitor_path)

    def value(self, states: np.ndarray) -> np.ndarray:
        return self.value_network(states)

    def target_value(self, states: np.ndarray) -> np.ndarray:
        return self.target_value_network(states)

    def softq_value(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network(states, actions)

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        return self.actor_network(states)[0]

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return self.actor_network(state[None, :])[0][0]

    # @tf.function
    def train(self, model_summary=True):
        sample = self.replay_buffer.get_batch(self.config["batch_size"])
        # for n_actions == 1
        action_batch = np.resize(sample["actions"], [self.config["batch_size"], self.n_actions])

        # Calculate critic targets
        next_value_batch = self.target_value(sample["states1"])
        softq_targets = sample["rewards"] + (1 - sample["terminals1"]) * \
            self.config["gamma"] * tf.reshape(next_value_batch, [-1])
        softq_targets = tf.cast(tf.reshape(softq_targets, [self.config["batch_size"], 1]), tf.float32)

        states0 = sample["states0"]
        actions, action_logprob = self.actor_network(states0)
        new_softq = self.softq_network(states0, actions)

        with tf.GradientTape() as softq_tape:
            softq = self.softq_network(states0, action_batch)
            softq_loss = tf.reduce_mean(tf.square(softq - softq_targets))
        value_target = tf.stop_gradient(new_softq - action_logprob)
        with tf.GradientTape() as value_tape:
            values = self.value_network(states0)
            value_loss = tf.reduce_mean(tf.square(values - value_target))
        advantage = tf.stop_gradient(action_logprob - new_softq + values)
        with tf.GradientTape() as actor_tape:
            _, action_logprob = self.actor_network(states0)
            actor_loss = tf.reduce_mean(action_logprob * advantage)

        actor_gradients = actor_tape.gradient(actor_loss, self.actor_network.trainable_weights)
        softq_gradients = softq_tape.gradient(softq_loss, self.softq_network.trainable_weights)
        value_gradients = value_tape.gradient(value_loss, self.value_network.trainable_weights)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_weights))
        self.softq_optimizer.apply_gradients(zip(softq_gradients, self.softq_network.trainable_weights))
        self.value_optimizer.apply_gradients(zip(value_gradients, self.value_network.trainable_weights))

        if model_summary:
            tf.summary.scalar("model/predicted_softq_mean", np.mean(softq), self.total_steps)
            tf.summary.scalar("model/predicted_softq_std", np.std(softq), self.total_steps)
            tf.summary.scalar("model/softq_loss", softq_loss, self.total_steps)
            tf.summary.scalar("model/actor_loss", actor_loss, self.total_steps)
            tf.summary.scalar("model/value_loss", value_loss, self.total_steps)
            tf.summary.scalar("model/action_logprob_mean", np.mean(action_logprob), self.total_steps)

        # Update the target networks
        soft_update(self.value_network.variables, self.target_value_network.variables, self.config["tau"])
        self.n_updates += 1

    def learn(self):
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        with self.writer.as_default():
            for _ in range(self.config["n_episodes"]):
                state = self.env.reset().astype(np.float32)
                episode_reward = 0
                episode_length = 0
                for _ in range(self.config["n_timesteps"]):
                    action = self.action(state)
                    to_execute = action_low + (action + 1.0) * 0.5 * (action_high - action_low)
                    new_state, reward, done, _ = self.env.step(to_execute)
                    new_state = new_state.astype(np.float32)
                    episode_length += 1
                    self.total_steps += 1
                    episode_reward += reward
                    self.replay_buffer.add(state, action, reward, new_state, done)
                    if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                        for _ in range(self.config["n_train_steps"]):
                            self.train(model_summary=True)
                    state = new_state
                    if done:
                        self.total_episodes += 1
                        tf.summary.scalar("env/Episode_length", episode_length, self.total_steps)
                        tf.summary.scalar("env/Reward", episode_reward, self.total_steps)
                        tf.summary.scalar("env/N_episodes", self.total_episodes, step=self.total_steps)
                        break

class ActorNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, n_actions, logprob_epsilon):
        super(ActorNetwork, self).__init__()
        self.logprob_epsilon = logprob_epsilon
        w_bound = 3e-3
        self.hidden = Sequential()
        for _ in range(n_hidden_layers):
            self.hidden.add(Dense(n_hidden_units, activation="relu"))

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
        action = tf.stop_gradient(normal_dist.sample())
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - \
            tf.math.log(1.0 - tf.pow(squashed_actions, 2) + self.logprob_epsilon)
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)
        return action, logprob

class SoftQNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units):
        super(SoftQNetwork, self).__init__()
        self.softq = Sequential()
        for _ in range(n_hidden_layers):
            self.softq.add(Dense(n_hidden_units, activation="relu"))
        self.softq.add(Dense(1,
                             kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                             bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)))
    def call(self, states, actions):
        x = tf.concat([states, actions], 1)
        return self.softq(x)


class ValueNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units):
        super(ValueNetwork, self).__init__()
        self.value = Sequential()
        for _ in range(n_hidden_layers):
            self.value.add(Dense(n_hidden_units, activation="relu"))

        self.value.add(Dense(1,
                             kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                             bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)))

    def call(self, inp):
        return self.value(inp)
