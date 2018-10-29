# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf

from yarll.agents.agent import Agent
from yarll.memory.memory import Memory
from yarll.misc.noise import OrnsteinUhlenbeckActionNoise
from yarll.misc.network_ops import batch_norm_layer, fan_in_initializer, linear_fan_in

class DDPG(Agent):
    def __init__(self, env, monitor_path: str, **usercfg) -> None:
        super(DDPG, self).__init__(**usercfg)
        self.env = env
        self.monitor_path: str = monitor_path

        self.config.update(
            n_episodes=100000,
            n_timesteps=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"),
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            ou_theta=0.15,
            ou_sigma=0.2,
            gamma=0.99,
            batch_size=64,
            tau=0.001,
            l2_loss_coef=1e-2,
            n_actor_layers=2,
            n_hidden_units=64,
            actor_layer_norm=True,
            critic_layer_norm=False,  # Batch norm for critic does not seem to work
            replay_buffer_size=1e6,
            replay_start_size=10000  # Required number of replay buffer entries to start training
        )
        self.config.update(usercfg)

        self.state_shape: list = list(env.observation_space.shape)
        self.n_actions: int = env.action_space.shape[0]
        self.states = tf.placeholder(tf.float32, [None] + self.state_shape, name="states")
        self.actions_taken = tf.placeholder(tf.float32, [None, self.n_actions], name="actions_taken")
        self.critic_target = tf.placeholder(tf.float32, [None, 1], name="critic_target")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        with tf.variable_scope("actor"):
            self.action_output, self.actor_vars = self.build_actor_network()

        self.target_action_output, actor_target_update = self.build_target_actor_network(self.actor_vars)

        self.q_gradient_input = tf.placeholder("float", [None, self.n_actions], name="q_grad_input")
        self.actor_policy_gradients = tf.gradients(
            self.action_output, self.actor_vars, -self.q_gradient_input, name="actor_gradients")
        self.actor_train_op = tf.train.AdamOptimizer(
            self.config["actor_learning_rate"],
            name="actor_optimizer").apply_gradients(list(zip(self.actor_policy_gradients, self.actor_vars)))

        with tf.variable_scope("critic"):
            self.q_value_output, self.critic_vars = self.build_critic_network()

        self.target_q_value_output, critic_target_update = self.build_target_critic_network(self.critic_vars)

        l2_loss = tf.add_n([self.config["l2_loss_coef"] * tf.nn.l2_loss(var) for var in self.critic_vars])
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_target - self.q_value_output)) + l2_loss
        self.critic_train_op = tf.train.AdamOptimizer(
            self.config["critic_learning_rate"],
            name="critic_optimizer").minimize(self.critic_loss)
        self.action_gradients = tf.gradients(self.q_value_output, self.actions_taken, name="action_gradients")

        summaries = []
        for v in self.actor_vars + self.critic_vars:
            summaries.append(tf.summary.histogram(v.name, v))
        self.model_summary_op = tf.summary.merge(summaries)

        self.update_targets_op = tf.group(actor_target_update, critic_target_update, name="update_targets")


        self.action_noise = OrnsteinUhlenbeckActionNoise(
            self.n_actions,
            self.config["ou_sigma"],
            self.config["ou_theta"]
        )

        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))

        self.session = tf.Session()
        self.init_op = tf.global_variables_initializer()

        self.n_updates = 0

        self.summary_writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "summaries"), tf.get_default_graph())

    def _initalize(self):
        self.session.run(self.init_op)

    def build_actor_network(self):
        layer1_size = 400
        layer2_size = 300

        x = self.states
        if self.config["actor_layer_norm"]:
            x = batch_norm_layer(x, training_phase=self.is_training, scope_bn="batch_norm_0", activation=tf.identity)
        with tf.variable_scope("L1"):
            x, l1_vars = linear_fan_in(x, layer1_size)
            if self.config["actor_layer_norm"]:
                x = batch_norm_layer(x, training_phase=self.is_training, scope_bn="batch_norm_1", activation=tf.nn.relu)
        with tf.variable_scope("L2"):
            x, l2_vars = linear_fan_in(x, layer2_size)
            if self.config["actor_layer_norm"]:
                x = batch_norm_layer(x, training_phase=self.is_training, scope_bn="batch_norm_2", activation=tf.nn.relu)

        with tf.variable_scope("L3"):
            W3 = tf.Variable(tf.random_uniform([layer2_size, self.n_actions], -3e-3, 3e-3), name="w")
            b3 = tf.Variable(tf.random_uniform([self.n_actions], -3e-3, 3e-3), name="b")
            action_output = tf.tanh(tf.nn.xw_plus_b(x, W3, b3))
            l3_vars = [W3, b3]

        return action_output, l1_vars + l2_vars + l3_vars

    def build_target_actor_network(self, actor_vars: list):
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.config["tau"])
        target_update = ema.apply(actor_vars)
        target_net = [ema.average(v) for v in actor_vars]

        x = self.states
        if self.config["actor_layer_norm"]:
            x = batch_norm_layer(
                x, training_phase=self.is_training, scope_bn="target_batch_norm_0", activation=tf.identity)

        x = tf.nn.xw_plus_b(x, target_net[0], target_net[1])
        if self.config["actor_layer_norm"]:
            x = batch_norm_layer(
                x, training_phase=self.is_training, scope_bn="target_batch_norm_1", activation=tf.nn.relu)
        x = tf.nn.xw_plus_b(x, target_net[2], target_net[3])
        if self.config["actor_layer_norm"]:
            x = batch_norm_layer(
                x, training_phase=self.is_training, scope_bn="target_batch_norm_2", activation=tf.nn.relu)

        action_output = tf.tanh(tf.nn.xw_plus_b(x, target_net[4], target_net[5]))

        return action_output, target_update

    def build_critic_network(self):
        layer1_size = 400
        layer2_size = 300

        x = self.states
        with tf.variable_scope("L1"):
            if self.config["critic_layer_norm"]:  # Defaults to False (= don't use it)
                x = batch_norm_layer(x, training_phase=self.is_training,
                                     scope_bn="batch_norm_0", activation=tf.identity)
            x, l1_vars = linear_fan_in(x, layer1_size)
            x = tf.nn.relu(x)
        with tf.variable_scope("L2"):
            W2 = tf.get_variable(
                "w", [layer1_size, layer2_size], initializer=fan_in_initializer(layer1_size + self.n_actions))
            W2_action = tf.get_variable(
                "w_action", [self.n_actions, layer2_size], initializer=fan_in_initializer(layer1_size + self.n_actions))
            b2 = tf.get_variable(
                "b", [layer2_size], initializer=fan_in_initializer(layer1_size + self.n_actions))
            x = tf.nn.relu(tf.matmul(x, W2) + tf.matmul(self.actions_taken, W2_action) + b2)
        with tf.variable_scope("L3"):
            W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3), name="w")
            b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name="b")
            q_value_output = tf.nn.xw_plus_b(x, W3, b3, name="q_value")

        return q_value_output, l1_vars + [W2, W2_action, b2, W3, b3]

    def build_target_critic_network(self, critic_vars: list):

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.config["tau"])
        target_update = ema.apply(critic_vars)
        target_net = [ema.average(v) for v in critic_vars]

        x = self.states
        if self.config["critic_layer_norm"]:
            x = batch_norm_layer(x, training_phase=self.is_training, scope_bn="batch_norm_0", activation=tf.identity)
        x = tf.nn.relu(tf.nn.xw_plus_b(x, target_net[0], target_net[1]))
        x = tf.nn.relu(tf.matmul(x, target_net[2]) + tf.matmul(self.actions_taken, target_net[3]) + target_net[4])
        q_value_output = tf.nn.xw_plus_b(x, target_net[5], target_net[6])

        return q_value_output, target_update

    def actor_gradients(self, state_batch: np.ndarray, action_batch: np.ndarray):
        q, grads = tf.get_default_session().run([self.q_value_output, self.action_gradients], feed_dict={
            self.states: state_batch,
            self.actions_taken: action_batch,
            self.is_training: False
        })
        summary = tf.Summary()
        summary.value.add(tag="model/actor_loss", simple_value=float(-np.mean(q)))
        self.summary_writer.add_summary(summary, self.n_updates)
        return grads[0]

    def target_q(self, states: np.ndarray, actions: np.ndarray):
        return tf.get_default_session().run(self.target_q_value_output, feed_dict={
            self.states: states,
            self.actions_taken: actions,
            self.is_training: False
        })

    def q_value(self, states: np.ndarray, actions: np.ndarray):
        return tf.get_default_session().run(self.q_value_output, feed_dict={
            self.states: states,
            self.actions_taken: actions,
            self.is_training: False
            })

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        return tf.get_default_session().run(self.action_output, feed_dict={
            self.states: states,
            self.is_training: True
        })

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return tf.get_default_session().run(self.action_output, feed_dict={
            self.states: [state],
            self.is_training: False
        })[0]

    def target_actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states using the target actor network."""
        return tf.get_default_session().run(self.target_action_output, feed_dict={
            self.states: states,
            self.is_training: True
        })

    def train(self):
        sample = self.replay_buffer.get_batch(int(self.config["batch_size"]))

        # for n_actions = 1
        action_batch = np.resize(sample["actions"], [int(self.config["batch_size"]), self.n_actions])

        # Calculate critic targets
        next_action_batch = self.target_actions(sample["states1"])
        q_value_batch = self.target_q(sample["states1"], next_action_batch)
        critic_targets = sample["rewards"] + (1 - sample["terminals1"]) * \
            self.config["gamma"] * q_value_batch.squeeze()
        critic_targets = np.resize(critic_targets, [int(self.config["batch_size"]), 1]).astype(np.float32)
        # Update actor weights
        fetches = [self.q_value_output, self.critic_loss, self.critic_train_op]
        predicted_q, critic_loss, _ = tf.get_default_session().run(fetches, feed_dict={
            self.critic_target: critic_targets,
            self.states: sample["states0"],
            self.actions_taken: action_batch,
            self.is_training: True
        })

        summary = tf.Summary()
        summary.value.add(tag="model/critic_loss", simple_value=float(critic_loss))
        summary.value.add(tag="model/predicted_q_mean", simple_value=np.mean(predicted_q))
        summary.value.add(tag="model/predicted_q_std", simple_value=np.std(predicted_q))
        self.summary_writer.add_summary(summary, self.n_updates)

        # Update the actor using the sampled gradient:
        action_batch_for_gradients = self.actions(sample["states0"])
        q_gradient_batch = self.actor_gradients(sample["states0"], action_batch_for_gradients)

        tf.get_default_session().run(self.actor_train_op, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.states: sample["states0"],
            self.is_training: True
        })

        # Update the target networks
        tf.get_default_session().run([self.update_targets_op, self.model_summary_op])
        self.n_updates += 1

    def noise_action(self, state: np.ndarray):
        """Choose an action based on the actor and exploration noise."""
        action = self.action(state)
        return action + self.action_noise()

    def learn(self):
        max_action = self.env.action_space.high
        self._initalize()
        with self.session as sess, sess.as_default():
            for episode in range(int(self.config["n_episodes"])):
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                for _ in range(int(self.config["n_timesteps"])):
                    action = self.noise_action(state)
                    new_state, reward, done, _ = self.env.step(action * max_action)
                    episode_length += 1
                    episode_reward += reward
                    self.replay_buffer.add(state, action, reward, new_state, done)
                    if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                        self.train()
                    state = new_state
                    if done:
                        self.action_noise.reset()
                        summary = tf.Summary()
                        summary.value.add(tag="global/Episode_length",
                                          simple_value=float(episode_length))
                        summary.value.add(tag="global/Reward",
                                          simple_value=float(episode_reward))
                        self.summary_writer.add_summary(summary, episode)
                        self.summary_writer.flush()
                        break
