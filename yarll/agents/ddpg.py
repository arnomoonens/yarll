# -*- coding: utf8 -*-

from itertools import count
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from yarll.agents.agent import Agent
from yarll.memory.memory import Memory
from yarll.misc.noise import OrnsteinUhlenbeckActionNoise
from yarll.misc.network_ops import CustomKaimingUniformKernelInitializer, CustomKaimingUniformBiasInitializer
from yarll.misc.utils import hard_update, soft_update
from yarll.misc import summary_writer

class DDPG(Agent):
    def __init__(self, env, monitor_path: str, **usercfg) -> None:
        super().__init__(**usercfg)
        self.env = env
        self.monitor_path: str = monitor_path

        self.config.update(
            n_episodes=100000,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            ou_theta=0.15,
            ou_sigma=0.2,
            gamma=0.99,
            batch_size=64,
            tau=0.001,
            l2_loss_coef=1e-2,
            actor_layer_norm=True,
            critic_layer_norm=True,  # Batch norm for critic does not seem to work
            replay_buffer_size=1e6,
            replay_start_size=10000,  # Required number of replay buffer entries to start training
            summaries=True
        )
        self.config.update(usercfg)

        self.state_shape: list = list(env.observation_space.shape)
        self.n_actions: int = env.action_space.shape[0]
        self.states = tf.keras.Input(self.state_shape, name="states")
        self.actions_taken = tf.keras.Input((self.n_actions,), name="actions_taken")

        # Make actor and critic
        self.actor = self.build_actor_network()
        self.critic = self.build_critic_network()

        # Make target networks
        custom_objects = {"CustomKaimingUniformBiasInitializer": CustomKaimingUniformBiasInitializer}
        with tf.keras.utils.custom_object_scope(custom_objects):
            self.target_actor = tf.keras.models.clone_model(self.actor)
            self.target_critic = tf.keras.models.clone_model(self.critic)

        dummy_input_states = tf.random.uniform((1, *self.state_shape))
        dummy_input_actions = tf.random.uniform((1, self.n_actions))
        self.actor(dummy_input_states)
        hard_update(self.actor.variables, self.target_actor.variables)
        self.critic((dummy_input_states, dummy_input_actions))
        hard_update(self.critic.variables, self.target_critic.variables)

        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.config["actor_learning_rate"])
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.config["critic_learning_rate"])

        self.action_noise = OrnsteinUhlenbeckActionNoise(
            self.n_actions,
            self.config["ou_sigma"],
            self.config["ou_theta"]
        )

        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))

        self.n_updates = 0
        if self.config["summaries"]:
            self.summary_writer = tf.summary.create_file_writer(
                str(self.monitor_path))
            summary_writer.set(self.summary_writer)
        else:
            self.summary_writer = tf.summary.create_noop_writer()

    def build_actor_network(self):
        layer1_size = 400
        layer2_size = 300

        x = self.states
        if self.config["actor_layer_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)

        # Layer 1
        fan_in = self.state_shape[-1]
        x = Dense(layer1_size,
                  activation="relu",
                  kernel_initializer=CustomKaimingUniformKernelInitializer(),
                  bias_initializer=CustomKaimingUniformBiasInitializer(fan_in),
                 )(x)
        if self.config["actor_layer_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)

        # Layer 2
        fan_in = layer1_size
        x = Dense(layer2_size,
                  activation="relu",
                  kernel_initializer=CustomKaimingUniformKernelInitializer(),
                  bias_initializer=CustomKaimingUniformBiasInitializer(fan_in),
                  )(x)
        if self.config["actor_layer_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)

        # Output
        action_output = Dense(self.n_actions,
                              activation="tanh",
                              kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3),
                              bias_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3),
                             )(x)

        return tf.keras.Model(self.states, action_output, name="actor")

    def build_critic_network(self):
        layer1_size = 400
        layer2_size = 300

        x = self.states
        if self.config["critic_layer_norm"]:  # Defaults to False (= don't use it)
            x = tf.keras.layers.BatchNormalization()(x)

        fan_in = self.state_shape[-1]
        x = Dense(layer1_size,
                  activation="relu",
                  kernel_initializer=CustomKaimingUniformKernelInitializer(),
                  bias_initializer=CustomKaimingUniformBiasInitializer(fan_in),
                  kernel_regularizer=tf.keras.regularizers.L2(self.config["l2_loss_coef"]),
                  bias_regularizer=tf.keras.regularizers.L2(self.config["l2_loss_coef"])
                  )(x)
        if self.config["critic_layer_norm"]:  # Defaults to False (= don't use it)
            x = tf.keras.layers.BatchNormalization()(x)

        # Layer 2
        x = tf.concat([x, self.actions_taken], axis=-1)
        fan_in = layer1_size + self.n_actions
        x = Dense(layer2_size,
                  activation="relu",
                  kernel_initializer=CustomKaimingUniformKernelInitializer(),
                  bias_initializer=CustomKaimingUniformBiasInitializer(fan_in),
                  kernel_regularizer=tf.keras.regularizers.L2(self.config["l2_loss_coef"]),
                  bias_regularizer=tf.keras.regularizers.L2(self.config["l2_loss_coef"])
                  )(x)

       # Output
        output = Dense(1,
                       kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3),
                       bias_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3),
                       kernel_regularizer=tf.keras.regularizers.L2(self.config["l2_loss_coef"]),
                       bias_regularizer=tf.keras.regularizers.L2(self.config["l2_loss_coef"])
                      )(x)

        return tf.keras.Model([self.states, self.actions_taken], output, name="critic")

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return self.actor(np.expand_dims(state, 0), training=False)[0]

    def noise_action(self, state: np.ndarray):
        """Choose an action based on the actor and exploration noise."""
        action = self.action(state)
        return np.clip(action + self.action_noise(), -1., 1.)

    @tf.function
    def train_actor_critic(self, state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch):
        next_action_batch = self.target_actor(state1_batch, training=True)
        q_value_batch = self.target_critic((state1_batch, next_action_batch), training=True) # ! in TF1 version, training was set to False
        reward_batch = tf.expand_dims(reward_batch, 1)
        critic_targets = reward_batch + tf.expand_dims(1.0 - terminal1_batch, 1) * self.config["gamma"] * q_value_batch
        with tf.GradientTape() as tape:
            predicted_q = self.critic((state0_batch, action_batch), training=True)
            mse_losses = tf.losses.MSE(y_true=critic_targets, y_pred=predicted_q)
            mse_loss = tf.nn.compute_average_loss(mse_losses)
            l2_loss = sum(self.critic.losses)
            q_loss = mse_loss + l2_loss
        q_gradients = tape.gradient(q_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(q_gradients, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            predicted_action_batch = self.actor(state0_batch, training=True)
            qs = self.critic((state0_batch, predicted_action_batch), training=True)
            actor_loss = tf.nn.compute_average_loss(-qs)
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_weights))

        # Uncomment to check tensor shapes
        # tf.debugging.assert_shapes((
        #     (state0_batch, ("B", "nS")),
        #     (state1_batch, ("B", "nS")),
        #     (action_batch, ("B", "nA")),
        #     (next_action_batch, ("B", "nA")),
        #     (q_value_batch, ("B", 1)),
        #     (reward_batch, ("B", 1)),
        #     (critic_targets, ("B", 1)),
        #     (predicted_q, ("B", 1)),
        #     (mse_losses, ("B")),
        #     (mse_loss, (1,)),
        #     (l2_loss, (1,)),
        #     (q_loss, (1,)),
        #     (predicted_action_batch, ("B", "nA")),
        #     (qs, ("B", 1)),
        #     (actor_loss, (1,)),
        # ))

        return q_loss, predicted_q, mse_loss, l2_loss, actor_loss

    def train(self):
        sample = self.replay_buffer.get_batch(int(self.config["batch_size"]))

        # for n_actions = 1
        action_batch = np.resize(sample["actions"], [int(self.config["batch_size"]), self.n_actions])

        q_loss, predicted_q, q_mse_loss, q_l2_loss, actor_loss = self.train_actor_critic(
            sample["states0"],
            action_batch,
            sample["rewards"],
            sample["states1"],
            sample["terminals1"])

        summary_writer.add_scalar("model/critic_loss", float(q_loss), self.n_updates)
        summary_writer.add_scalar("model/critic_mse_loss", float(q_mse_loss), self.n_updates)
        summary_writer.add_scalar("model/critic_l2_loss", float(q_l2_loss), self.n_updates)
        summary_writer.add_scalar("model/actor_loss", float(actor_loss), self.n_updates)
        summary_writer.add_scalar("model/predicted_q_mean", np.mean(predicted_q), self.n_updates)
        summary_writer.add_scalar("model/predicted_q_std", np.std(predicted_q), self.n_updates)

        # Update the target networks
        soft_update(self.actor.variables, self.target_actor.variables, self.config["tau"])
        soft_update(self.critic.variables, self.target_critic.variables, self.config["tau"])
        self.n_updates += 1

    def learn(self):
        max_action = self.env.action_space.high
        summary_writer.start()
        for episode in range(int(self.config["n_episodes"])):
            state = self.env.reset()
            episode_reward = 0
            episode_actions = []
            for episode_length in count(start=1):
                action = self.noise_action(state)
                episode_actions.append(action)
                # ! Assumes action space between -max_action and max_action
                new_state, reward, done, _ = self.env.step(action * max_action)
                episode_reward += reward
                self.replay_buffer.add(state, action, reward, new_state, done)
                if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                    self.train()
                state = new_state
                if done:
                    self.action_noise.reset()
                    summary_writer.add_scalar("env/episode_length", float(episode_length), episode)
                    summary_writer.add_scalar("env/episode_reward", float(episode_reward), episode)
                    summary_writer.add_scalar("env/episode_mean_action", np.mean(episode_actions), episode)
                    break
        summary_writer.stop()
