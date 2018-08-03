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
            n_train_steps=4, # Number of parameter update steps per iteration
            replay_buffer_size=1e6,
            replay_start_size=10000  # Required number of replay buffer entries to start training
        )
        self.config.update(usercfg)

        self.state_shape: list = list(env.observation_space.shape)
        self.n_actions: int = env.action_space.shape[0]
        self.states = tf.placeholder(tf.float32, [None] + self.state_shape, name="states")
        self.actions_taken = tf.placeholder(tf.float32, [None, self.n_actions], name="actions_taken")
        self.softq_target = tf.placeholder(tf.float32, [None, 1], name="softq_target")
        self.value_target = tf.placeholder(tf.float32, [None, 1], name="value_target")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Make networks
        self.action_output, self.action_logprob, self.actor_vars = self.build_actor_network()
        self.softq_output, self.softq_vars = self.build_softq_network()
        self.value_output, self.value_vars = self.build_value_network()
        self.target_value_output, self.value_target_update = self.build_target_value_network(self.value_vars)

        # Make losses
        # softq_target = r(s_t, a_t) - gamma * target_value_output(s_{t+1})
        self.softq_target = tf.Print(self.softq_target, [self.softq_target], message="softq=", first_n=5)
        self.softq_loss = tf.reduce_mean(self.softq_output * (self.softq_output - self.softq_target))
        value_target = self.softq_output + self.action_logprob
        self.value_loss = tf.reduce_mean(self.value_output * (self.value_output - value_target))
        self.actor_loss = tf.reduce_mean(self.action_logprob * \
        (self.action_logprob - self.softq_output + self.value_output))

        # Make train ops
        softq_train_op = tf.train.AdamOptimizer(
            self.config["softq_learning_rate"],
            name="softq_optimizer").minimize(self.softq_loss)
        value_train_op = tf.train.AdamOptimizer(
            self.config["value_learning_rate"],
            name="value_optimizer").minimize(self.value_loss)
        actor_train_op = tf.train.AdamOptimizer(
            self.config["actor_learning_rate"],
            name="actor_optimizer").minimize(self.actor_loss)

        self.train_op = tf.group(softq_train_op, value_train_op, actor_train_op, name="train_op")

        summaries = []
        for v in self.actor_vars + self.softq_vars + self.value_vars:
            summaries.append(tf.summary.histogram(v.name, v))
        self.model_summary_op = tf.summary.merge(summaries)

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

            normal_dist = tf.distributions.Normal(mean, tf.exp(log_std))
            actions = normal_dist.sample()

            logprob = normal_dist.log_prob(self.actions_taken)

            actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return actions, logprob, actor_vars

    def build_target_value_network(self, value_vars: list):
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.config["tau"])
        target_update = ema.apply(value_vars)
        target_net = [ema.average(v) for v in value_vars]

        x = self.states
        x = tf.nn.relu(tf.nn.xw_plus_b(x, target_net[0], target_net[1]))
        x = tf.nn.relu(tf.nn.xw_plus_b(x, target_net[2], target_net[3]))
        value = tf.nn.xw_plus_b(x, target_net[4], target_net[5])

        return value, target_update

    def build_softq_network(self):
        x = tf.concat([self.states, self.actions_taken], 1)
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

    def target_value(self, states: np.ndarray):
        return tf.get_default_session().run(self.target_value_output, feed_dict={
            self.states: states,
            self.is_training: False
        })

    def softq_value(self, states: np.ndarray, actions: np.ndarray):
        return tf.get_default_session().run(self.softq_output, feed_dict={
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

    def train(self):
        sample = self.replay_buffer.get_batch(self.config["batch_size"])

        # for n_actions = 1
        action_batch = np.resize(sample["actions"], [self.config["batch_size"], self.n_actions])

        # Calculate critic targets
        next_value_batch = self.target_value(sample["states1"])
        softq_targets = sample["rewards"] + (1 - sample["terminals1"]) * \
            self.config["gamma"] * next_value_batch.squeeze()
        softq_targets = np.resize(softq_targets, [self.config["batch_size"], 1]).astype(np.float32)
        # Update actor weights
        fetches = [self.softq_output, self.softq_loss, self.train_op]
        predicted_q, softq_loss, _ = tf.get_default_session().run(fetches, feed_dict={
            self.softq_target: softq_targets,
            self.states: sample["states0"],
            self.actions_taken: action_batch,
            self.is_training: True
        })

        summary = tf.Summary()
        summary.value.add(tag="model/softq_loss", simple_value=float(softq_loss))
        summary.value.add(tag="model/predicted_softq_mean", simple_value=np.mean(predicted_q))
        summary.value.add(tag="model/predicted_softq_std", simple_value=np.std(predicted_q))
        self.summary_writer.add_summary(summary, self.n_updates)

        # Update the target networks
        tf.get_default_session().run([self.value_target_update, self.model_summary_op])
        self.n_updates += 1

    def learn(self):
        max_action = self.env.action_space.high
        with tf.Session() as sess, sess.as_default():
            sess.run(self.init_op)
            for episode in range(self.config["n_episodes"]):
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                for _ in range(self.config["n_timesteps"]):
                    action = self.action(state)
                    new_state, reward, done, _ = self.env.step(action) # TODO: change this max_action thing
                    episode_length += 1
                    episode_reward += reward
                    self.replay_buffer.add(state, action, reward, new_state, done)
                    if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                        for _ in range(self.config["n_train_steps"]):
                            self.train()
                    state = new_state
                    if done:
                        summary = tf.Summary()
                        summary.value.add(tag="global/Episode_length",
                                          simple_value=float(episode_length))
                        summary.value.add(tag="global/Reward",
                                          simple_value=float(episode_reward))
                        self.summary_writer.add_summary(summary, episode)
                        self.summary_writer.flush()
                        break
