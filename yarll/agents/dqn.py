"""
Deep learning variant of Q-learning,
with a deep neural network to approximate the Q-function and a replay buffer
and target network to improve stability.
"""

from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.memory.memory import Memory
from yarll.misc.utils import hard_update, soft_update
from yarll.policies.e_greedy import EGreedy

class DQN(Agent):
    """
    Deep Q-learning agent
    """

    def __init__(self, env, monitor_path: Union[str, Path], **usercfg):
        super().__init__(**usercfg)
        self.env = env
        self.monitor_path = Path(monitor_path)
        self.monitor_path.mkdir(parents=True, exist_ok=True)

        self.config.update(
            max_steps=100000,
            learning_rate=3e-4,
            n_hidden_layers=2,
            n_hidden_units=20,
            gamma=0.99,
            batch_size=32,
            tau=0.005,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=.999,
            n_train_steps=1,  # Number of parameter update steps per iteration
            replay_buffer_size=1e6,
            replay_start_size=32,  # Required number of replay buffer entries to start training
            hidden_layer_activation="relu",
            normalize_inputs=False,
            summaries=True,
            checkpoints=True,
            save_model=True,
            write_train_rewards=False
        )
        self.config.update(usercfg)
        self.n_actions = self.env.action_space.n
        self.policy = EGreedy(self.config["epsilon"])

        self.q_network = self.build_network()
        self.optimizer = Adam(learning_rate=self.config["learning_rate"])
        self.q_network.compile(loss='mse', optimizer=self.optimizer)

        self.target_q_network = self.build_network()
        # Initialize (target) q network by feeding dummy input
        dummy_input_states = tf.random.uniform((1, *env.observation_space.shape))
        self.q_network(dummy_input_states)
        self.target_q_network(dummy_input_states)
        hard_update(self.q_network.variables, self.target_q_network.variables)


        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))

        self.env_runner = EnvRunner(self.env,
                                    self,
                                    usercfg,
                                    scale_states=self.config["normalize_inputs"],
                                    summaries=self.config["summaries"],
                                    episode_rewards_file=(
                                        self.monitor_path / "train_rewards.txt" if self.config["write_train_rewards"] else None)
                                    )

        if self.config["checkpoints"]:
            checkpoint_directory = self.monitor_path / "checkpoints"
            self.ckpt = tf.train.Checkpoint(net=self.q_network)
            self.cktp_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_directory, 10)
            self.checkpoint_every_episodes = 10

        self.writer = tf.summary.create_file_writer(str(self.monitor_path))
        self.total_steps = 0
        self.n_updates = 0


    def build_network(self):
        network = Sequential()
        for _ in range(self.config["n_hidden_layers"]):
            network.add(Dense(self.config["n_hidden_units"],
                              activation=self.config["hidden_layer_activation"]))
        network.add(Dense(self.env.action_space.n, activation="linear"))
        return network

    def choose_action(self, state, features):
        q_values = self.q_network(state[None, :])[0]
        action, _ = self.policy(q_values)
        return {"action": action}

    @tf.function
    def train(self, state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch):
        next_q_values = self.target_q_network(state1_batch)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = reward_batch + (1. - terminal1_batch) * self.config["gamma"] * max_next_q_values
        with tf.GradientTape() as tape:
            predictions = self.q_network(state0_batch)
            actions_onehot = tf.one_hot(action_batch, self.n_actions)
            q_chosen = tf.reduce_sum(predictions * actions_onehot, axis=-1)
            loss = tf.reduce_mean(tf.square(q_chosen - target_q_values))
        gradients = tape.gradient(loss, self.q_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_weights))
        q_mean, q_variance = tf.nn.moments(q_chosen, axes=[0])
        return q_mean, tf.sqrt(q_variance), tf.reduce_mean(target_q_values), loss

    def learn(self):
        # Arrays to keep results from train function over different train steps in
        q_means = np.empty((self.config["n_train_steps"],), np.float32)
        q_stds = np.empty((self.config["n_train_steps"],), np.float32)
        target_qs = np.empty((self.config["n_train_steps"],), np.float32)
        losses = np.empty((self.config["n_train_steps"],), np.float32)
        total_episodes = 0
        with self.writer.as_default():
            for _ in range(self.config["max_steps"]):

                experience = self.env_runner.get_steps(1)[0]

                # Update epsilon
                self.policy.epsilon = max(self.config["epsilon_min"], self.policy.epsilon * self.config["epsilon_decay"])

                self.total_steps += 1
                self.replay_buffer.add(experience.state, experience.action, experience.reward,
                                       experience.next_state, experience.terminal)
                if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                    for i in range(self.config["n_train_steps"]):
                        sample = self.replay_buffer.get_batch(self.config["batch_size"])
                        q_mean, q_std, target_q, loss = self.train(
                            sample["states0"],
                            sample["actions"].astype(np.int),
                            sample["rewards"],
                            sample["states1"],
                            sample["terminals1"])
                        q_means[i] = q_mean
                        q_stds[i] = q_std
                        target_qs[i] = target_q
                        losses[i] = loss
                    tf.summary.scalar("model/predicted_q_mean", np.mean(q_means), self.total_steps)
                    tf.summary.scalar("model/predicted_q_std", np.mean(q_stds), self.total_steps)
                    tf.summary.scalar("model/target_q_mean", np.mean(target_qs), self.total_steps)
                    tf.summary.scalar("model/loss", np.mean(losses), self.total_steps)
                    # Update the target network
                    soft_update(self.q_network.variables,
                                self.target_q_network.variables,
                                self.config["tau"])
                    self.n_updates += 1
                if experience.terminal:
                    total_episodes += 1
                    if self.config["checkpoints"] and (total_episodes % self.checkpoint_every_episodes) == 0:
                        self.cktp_manager.save()
        if self.config["save_model"]:
            self.q_network.save_weights(str(self.monitor_path / "q_weights"))
