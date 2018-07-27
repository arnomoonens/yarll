# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf

from agents.agent import Agent
from memory.memory import Memory
from misc.network_ops import linear

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
            gamma=0.99,
            batch_size=64,
            tau=0.01,
            l2_loss_coef=1e-2,
            n_actor_layers=2,
            n_hidden_units=128,
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

        self.init_op = tf.global_variables_initializer()

        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))

        self.n_updates = 0

        self.summary_writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "summaries"), tf.get_default_graph())

    def build_actor_network(self):
        w_bound = 3e-3
        x = self.states
        with tf.variable_scope("actor"):
            x = linear(x, self.config["n_hidden_units"], "L1", tf.random_uniform_initializer(-w_bound, w_bound))
            x = tf.nn.relu(x)

            mean = linear(x, self.n_actions, "mean", tf.random_uniform_initializer(-w_bound, w_bound))
            log_std = linear(x, self.n_actions, "log_std", tf.random_uniform_initializer(-w_bound, w_bound))

            actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return mean, log_std, actor_vars

    def build_target_actor_network(self, actor_vars: list):
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.config["tau"])
        target_update = ema.apply(actor_vars)
        target_net = [ema.average(v) for v in actor_vars]

        x = self.states

        x = tf.nn.xw_plus_b(x, target_net[0], target_net[1])

        action_output = tf.tanh(tf.nn.xw_plus_b(x, target_net[4], target_net[5]))

        return action_output, target_update

    def build_softq_network(self):
        x = tf.concat([self.states, self.actions], 1)
        with tf.variable_scope("softq"):
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L1"))
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L2"))
            x = linear(x, 1, "softq", tf.random_uniform_initializer(-3e-3, 3e-3))

            softq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return x, softq_vars

    def build_value_network(self):
        x = self.states
        with tf.variable_scope("value"):
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L1"))
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L2"))
            x = linear(x, 1, "value", tf.random_uniform_initializer(-3e-3, 3e-3))

            value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return x, value_vars

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
        sample = self.replay_buffer.get_batch(self.config["batch_size"])

        # for n_actions = 1
        action_batch = np.resize(sample["actions"], [self.config["batch_size"], self.n_actions])

        # Calculate critic targets
        next_action_batch = self.target_actions(sample["states1"])
        q_value_batch = self.target_q(sample["states1"], next_action_batch)
        critic_targets = sample["rewards"] + (1 - sample["terminals1"]) * \
            self.config["gamma"] * q_value_batch.squeeze()
        critic_targets = np.resize(critic_targets, [self.config["batch_size"], 1]).astype(np.float32)
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
        with tf.Session() as sess, sess.as_default():
            sess.run(self.init_op)
            for episode in range(self.config["n_episodes"]):
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                for _ in range(self.config["n_timesteps"]):
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
