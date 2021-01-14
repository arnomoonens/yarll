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
import tensorflow_probability as tfp

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.distributions.bijectors import ConditionalScale, ConditionalShift
from yarll.memory.memory import Memory
from yarll.misc.utils import hard_update, soft_update

# TODO: put this in separate file
class DeterministicPolicy:
    def __init__(self, env, policy_fn):
        self.env = env
        self.policy_fn = policy_fn
        self.initial_features = None

        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def choose_action(self, state, features):
        res = self.policy_fn(state[None, :])[0].numpy()[0]
        return {"action": res}

    def get_env_action(self, action):
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def new_trajectory(self):
        pass

class SAC(Agent):
    def __init__(self, env, monitor_path: Union[Path, str], **usercfg) -> None:
        super().__init__(**usercfg)
        self.env = env
        self.monitor_path = Path(monitor_path)
        self.monitor_path.mkdir(parents=True, exist_ok=True)

        self.config.update(
            max_steps=100000,
            actor_learning_rate=1e-4,
            softq_learning_rate=1e-4,
            alpha_learning_rate=1e-4,
            n_softqs=2,
            reward_scale=1.0,
            n_hidden_layers=2,
            n_hidden_units=1024,
            gamma=0.99,
            batch_size=1024,
            tau=0.005,
            init_log_alpha=0.1,
            actor_update_frequency=1,
            critic_target_update_frequency=2,
            log_scale_bounds=(-5, 2),
            target_entropy=None,
            logprob_epsilon=1e-6,  # For numerical stability when computing tf.log
            n_train_steps=1,  # Number of parameter update steps per iteration
            replay_buffer_size=1e6,
            replay_start_size=1024,  # Required number of replay buffer entries to start training
            hidden_layer_activation="relu",
            normalize_inputs=False,
            summaries=True,
            checkpoints=True,
            checkpoint_every_episodes=10,
            checkpoints_max_to_keep=None,
            save_model=True,
            test_frequency=0,
            n_test_episodes=5,
            write_train_rewards=False
        )
        self.config.update(usercfg)

        self.state_shape: list = list(env.observation_space.shape)
        self.n_actions: int = env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        self.target_entropy = self.config["target_entropy"]
        if self.target_entropy is None:
            self.target_entropy = -np.prod(env.action_space.shape)

        # Make networks
        # action_output are the squashed actions and action_original those straight from the normal distribution
        self.actor_network = ActorNetwork(self.state_shape[0],
                                          self.config["n_hidden_layers"],
                                          self.config["n_hidden_units"],
                                          self.n_actions,
                                          self.config["logprob_epsilon"],
                                          self.config["hidden_layer_activation"],
                                          self.config["log_scale_bounds"])
        self.softq_networks = [SoftQNetwork(self.config["n_hidden_layers"],
                                            self.config["n_hidden_units"],
                                            self.config["hidden_layer_activation"])
                               for _ in range(self.config["n_softqs"])]
        self.target_softq_networks = [deepcopy(net) for net in self.softq_networks]

        dummy_input_states = tf.random.uniform((1, *self.state_shape))
        dummy_input_actions = tf.random.uniform((1, *self.env.action_space.shape))
        for net, target_net in zip(self.softq_networks, self.target_softq_networks):
            net((dummy_input_states, dummy_input_actions))
            target_net((dummy_input_states, dummy_input_actions))
            hard_update(net.variables, target_net.variables)

        self._log_alpha = tf.Variable(self.config["init_log_alpha"], name="log_alpha")
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp, name="alpha")


        # Make train ops
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.config["actor_learning_rate"])
        self.softq_optimizers = [tf.optimizers.Adam(learning_rate=self.config["softq_learning_rate"])
                                 for _ in self.softq_networks]
        self.alpha_optimizer = tf.optimizers.Adam(learning_rate=self.config["alpha_learning_rate"])

        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))
        self.n_updates = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.writer = tf.summary.create_file_writer(str(self.monitor_path)) if self.config["summaries"] else tf.summary.create_noop_writer()

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
            self.ckpt = tf.train.Checkpoint(net=self.actor_network)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                           checkpoint_directory,
                                                           max_to_keep=self.config["checkpoints_max_to_keep"])

        if self.config["test_frequency"] > 0 and self.config["n_test_episodes"] > 0:
            test_env = deepcopy(env)
            unw = test_env.unwrapped
            if hasattr(unw, "summaries"):
                unw.summaries = False
            deterministic_policy = DeterministicPolicy(test_env, self.actor_network.deterministic_actions)
            self.test_env_runner = EnvRunner(test_env,
                                             deterministic_policy,
                                             usercfg,
                                             scale_states=self.config["normalize_inputs"],
                                             summaries=False,
                                             episode_rewards_file=(
                                             self.monitor_path / "test_rewards.txt")
                                             )
            header = [""] # (epoch) id has no name in header
            header += [f"rew_{i}" for i in range(self.config["n_test_episodes"])]
            header += ["rew_mean", "rew_std"]
            self.test_results_file = self.monitor_path / "test_results.csv"
            with open(self.test_results_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)

            self.total_rewards = np.empty((self.config["n_test_episodes"],), dtype=np.float32)

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        return self.actor_network(tf.convert_to_tensor(states))[0]

    def deterministic_actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        return self.actor_network.deterministic_actions(tf.convert_to_tensor(states))

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return self.actor_network(state[None, :])[0].numpy()[0]

    @tf.function(experimental_relax_shapes=True)
    def train_critic(self, state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch):
        # Calculate critic targets
        next_action_batch, next_logprob_batch = self.actor_network(state1_batch)
        next_qs_values = [net((state1_batch, next_action_batch)) for net in self.target_softq_networks]
        next_q_values = tf.reduce_min(next_qs_values, axis=0)
        next_values = next_q_values - self._alpha * next_logprob_batch
        next_values = tf.expand_dims(1.0 - terminal1_batch, 1) * next_values
        softq_targets = self.config["reward_scale"] * tf.expand_dims(reward_batch, 1) + self.config["gamma"] * next_values
        softq_targets = tf.squeeze(softq_targets)

        # Update critics
        softq_losses = []
        for net, optimizer in zip(self.softq_networks, self.softq_optimizers):
            with tf.GradientTape() as tape:
                softq = net((state0_batch, action_batch))
                softq_loss = 0.5 * tf.losses.MSE(y_true=softq_targets, y_pred=tf.squeeze(softq))
                softq_losses.append(tf.stop_gradient(softq_loss))
            softq_gradients = tape.gradient(softq_loss, net.trainable_weights)
            optimizer.apply_gradients(zip(softq_gradients, net.trainable_weights))

        softq_mean, softq_variance = tf.nn.moments(softq, axes=[0])
        return softq_mean[0], tf.sqrt(softq_variance[0]), softq_targets, tf.reduce_mean(softq_losses)

    @tf.function(experimental_relax_shapes=True)
    def train_actor_alpha(self, state0_batch):
        # Update actor
        with tf.GradientTape() as tape:
            actions, action_logprob = self.actor_network(state0_batch)
            softqs_pred = [net((state0_batch, actions)) for net in self.softq_networks]
            min_softq_pred = tf.reduce_min(softqs_pred, axis=0)
            actor_loss = tf.reduce_mean(self._alpha * action_logprob - min_softq_pred)
        actor_gradients = tape.gradient(actor_loss, self.actor_network.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_weights))

        # Update alpha
        _, action_logprob = self.actor_network(state0_batch)
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * self._alpha * tf.stop_gradient(action_logprob + self.target_entropy)  # For batch
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
        alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self._log_alpha]))

        return tf.reduce_mean(actor_loss), alpha_loss, tf.reduce_mean(action_logprob)

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
            for step in range(self.config["max_steps"]):
                if self.config["test_frequency"] > 0 and (step % self.config["test_frequency"]) == 0 and self.config["n_test_episodes"] > 0:
                    for i in range(self.config["n_test_episodes"]):
                        test_trajectory = self.test_env_runner.get_trajectory(stop_at_trajectory_end=True)
                        self.total_rewards[i] = np.sum(test_trajectory.rewards)
                    test_rewards_mean = np.mean(self.total_rewards)
                    test_rewards_std = np.std(self.total_rewards)
                    to_write = [step] + self.total_rewards.tolist() + [test_rewards_mean, test_rewards_std]
                    with open(self.test_results_file, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow(to_write)
                experience = self.env_runner.get_steps(1)[0]
                self.total_steps += 1
                self.replay_buffer.add(experience.state, experience.action, experience.reward,
                                       experience.next_state, experience.terminal)
                to_save.append(experience.state.tolist() + experience.action.tolist() +
                               [experience.reward] + experience.next_state.tolist())
                if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                    for i in range(self.config["n_train_steps"]):
                        sample = self.replay_buffer.get_batch(self.config["batch_size"])
                        softq_mean, softq_std, softq_targets, softq_loss = self.train_critic(
                            sample["states0"],
                            np.resize(sample["actions"], [self.config["batch_size"],
                                                          self.n_actions]),  # for n_actions == 1
                            sample["rewards"],
                            sample["states1"],
                            sample["terminals1"])
                        if (step % self.config["actor_update_frequency"]) == 0:
                            actor_loss, alpha_loss, action_logprob_mean = self.train_actor_alpha(
                                sample["states0"])
                        softq_means[i] = softq_mean
                        softq_stds[i] = softq_std
                        softq_losses[i] = softq_loss
                        actor_losses[i] = actor_loss
                        alpha_losses[i] = alpha_loss
                        action_logprob_means[i] = action_logprob_mean
                        # Update the target networks
                        if (step % self.config["critic_target_update_frequency"]) == 0:
                            for net, target_net in zip(self.softq_networks, self.target_softq_networks):
                                soft_update(net.variables,
                                            target_net.variables,
                                            self.config["tau"])
                    tf.summary.scalar("model/predicted_softq_mean", np.mean(softq_means), self.total_steps)
                    tf.summary.scalar("model/predicted_softq_std", np.mean(softq_stds), self.total_steps)
                    tf.summary.scalar("model/softq_targets", np.mean(softq_targets), self.total_steps)
                    tf.summary.scalar("model/softq_loss", np.mean(softq_losses), self.total_steps)
                    tf.summary.scalar("model/actor_loss", np.mean(actor_losses), self.total_steps)
                    tf.summary.scalar("model/alpha_loss", np.mean(alpha_losses), self.total_steps)
                    tf.summary.scalar("model/alpha", self._log_alpha, self.total_steps)
                    tf.summary.scalar("model/action_logprob_mean", np.mean(action_logprob_means), self.total_steps)
                    self.n_updates += 1
                if experience.terminal:
                    if self.config["checkpoints"] and (total_episodes % self.config["checkpoint_every_episodes"]) == 0:
                        self.ckpt_manager.save(total_episodes)
                    total_episodes += 1
        with open(self.monitor_path / "experiences.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(to_save)
        if self.config["save_model"]:
            self.actor_network.save_weights(str(self.monitor_path / "actor_weights"))
            self.softq_networks[0].save_weights(str(self.monitor_path / "softq_weights"))

    def choose_action(self, state, features):
        return {"action": self.action(state)}

    def get_env_action(self, action):
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)


class ActorNetwork(Model):
    def __init__(self, input_dim, n_hidden_layers, n_hidden_units, n_actions, logprob_epsilon, hidden_layer_activation="relu", log_scale_bounds=(-5, 2)):
        super().__init__()
        self.logprob_epsilon = logprob_epsilon
        inp = tf.keras.Input((input_dim,))
        mean_log_scale_net = Sequential()
        for i in range(n_hidden_layers):
            mean_log_scale_net.add(Dense(n_hidden_units,
                                          activation=hidden_layer_activation,
                                          name=f"hidden_{i}"))

        mean_log_scale_net.add(Dense(n_actions * 2, name="mean_log_scale"))
        mean_log_scale_net = mean_log_scale_net(inp)
        mean, log_scale = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=-1)
        )(mean_log_scale_net)

        log_scale = tf.keras.layers.Lambda(
            lambda log_scale: tf.clip_by_value(log_scale, *log_scale_bounds)
        )(log_scale)

        self.mean_log_scale_model = tf.keras.Model(inputs=inp, outputs=(mean, log_scale))

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros((n_actions,)),
            scale_diag=tf.ones((n_actions,)))

        raw_action_distribution = tfp.bijectors.Chain((
            ConditionalShift(name='shift'),
            ConditionalScale(name='scale'),
        ))(base_distribution)

        self.base_distribution = base_distribution
        self.raw_action_distribution = raw_action_distribution
        self.squash_bijector = tfp.bijectors.Tanh()
        self.action_distribution = self.squash_bijector(self.raw_action_distribution)

    def call(self, inputs, training=None, mask=None):
        outputs = self.mean_log_scale_model(inputs)
        mean = outputs[0]
        log_scale = outputs[1]
        squashed_actions = self.action_distribution.sample(
            (tf.shape(inputs)[0],),
            bijector_kwargs={'shift': {'shift': mean},
                             'scale': {'scale': tf.exp(log_scale)}})
        log_prob = self.action_distribution.log_prob(
            squashed_actions,
            bijector_kwargs={'shift': {'shift': mean},
                             'scale': {'scale': tf.exp(log_scale)}})[..., tf.newaxis]
        return squashed_actions, log_prob

    @tf.function(experimental_relax_shapes=True)
    def deterministic_actions(self, inp):
        outputs = self.mean_log_scale_model(inp)
        return self.squash_bijector(outputs[0])


class SoftQNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, hidden_layer_activation="relu"):
        super().__init__()
        self.softq = Sequential()
        for _ in range(n_hidden_layers):
            self.softq.add(Dense(n_hidden_units, activation=hidden_layer_activation))
        self.softq.add(Dense(1))

    def call(self, inputs, training=None, mask=None):
        states, actions = inputs
        x = tf.concat([states, actions], 1)
        return self.softq(x)
