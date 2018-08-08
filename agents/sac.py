# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf

from agents.agent import Agent
from memory.memory import Memory
from misc.network_ops import linear, normalized_columns_initializer

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
            batch_size=128,
            tau=0.01,
            n_actor_layers=2,
            logprob_epsilon=1e-6, # For numerical stability when computing tf.log
            n_hidden_units=128,
            n_train_steps=4, # Number of parameter update steps per iteration
            replay_buffer_size=1e6,
            replay_start_size=128  # Required number of replay buffer entries to start training
        )
        self.config.update(usercfg)

        self.state_shape: list = list(env.observation_space.shape)
        self.n_actions: int = env.action_space.shape[0]
        self.states = tf.placeholder(tf.float32, [None] + self.state_shape, name="states")
        self.actions_taken = tf.placeholder(tf.float32, [None, self.n_actions], name="actions_taken")
        self.softq_target = tf.placeholder(tf.float32, [None, 1], name="softq_target")
        self.value_target = tf.placeholder(tf.float32, [None, 1], name="value_target")
        self.advantage = tf.placeholder(tf.float32, [None, 1], name="advantage")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Make networks
        # action_output are the squashed actions and action_original those straight from the normal distribution
        self.action_output, self.action_original, self.action_logprob, self.actor_vars = self.build_actor_network()
        self.softq_output, self.softq_vars = self.build_softq_network()
        self.value_output, self.value_vars = self.build_value_network()
        self.target_value_output, self.value_target_update = self.build_target_value_network(self.value_vars)

        # Make losses
        self.softq_loss = tf.reduce_mean(0.5 * tf.square(self.softq_output - self.softq_target), name="softq_loss")
        self.value_loss = tf.reduce_mean(0.5 * tf.square(self.value_output - self.value_target), name="value_loss")
        self.actor_loss = tf.reduce_mean(self.action_logprob * self.advantage, name="actor_loss")

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
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L1", normalized_columns_initializer()))
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L2", normalized_columns_initializer()))

            mean = linear(x,
                          self.n_actions,
                          "mean",
                          tf.random_uniform_initializer(-w_bound, w_bound),
                          tf.random_uniform_initializer(-w_bound, w_bound))
            log_std = linear(x,
                             self.n_actions,
                             "log_std",
                             tf.random_uniform_initializer(-w_bound, w_bound),
                             tf.random_uniform_initializer(-w_bound, w_bound))
            log_std_clipped = tf.clip_by_value(log_std, -20, 2, name="log_std_clipped") # TODO: is this in the paper?

            normal_dist = tf.distributions.Normal(mean, tf.exp(log_std_clipped), name="actions_normal_distr")
            actions = normal_dist.sample(name="actions")
            squashed_actions = tf.tanh(actions, name="squashed_actions") # Squash output between [-1, 1]

            logprob = normal_dist.log_prob(actions) - \
            tf.log(1.0 - tf.pow(self.actions_taken, 2) + self.config["logprob_epsilon"])

            actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return squashed_actions, actions, logprob, actor_vars

    def build_target_value_network(self, value_vars: list):
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.config["tau"])
        target_update = ema.apply(value_vars)
        target_net = [ema.average(v) for v in value_vars]

        x = self.states
        x = tf.nn.relu(tf.nn.xw_plus_b(x, target_net[0], target_net[1]))
        x = tf.nn.relu(tf.nn.xw_plus_b(x, target_net[2], target_net[3]))
        value = tf.nn.xw_plus_b(x, target_net[4], target_net[5], name="target_value")

        return value, target_update

    def build_softq_network(self):
        x = tf.concat([self.states, self.actions_taken], 1)
        with tf.variable_scope("softq"):
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L1", normalized_columns_initializer()))
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L2", normalized_columns_initializer()))
            x = linear(x, 1, "softq", tf.random_uniform_initializer(-3e-3, 3e-3), tf.random_uniform_initializer(-3e-3, 3e-3))

            softq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return x, softq_vars

    def build_value_network(self):
        x = self.states
        with tf.variable_scope("value"):
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L1", normalized_columns_initializer()))
            x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L2", normalized_columns_initializer()))
            x = linear(x, 1, "value", tf.random_uniform_initializer(-3e-3, 3e-3), tf.random_uniform_initializer(-3e-3, 3e-3))

            value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return x, value_vars

    def value(self, states: np.ndarray):
        return tf.get_default_session().run(self.value_output, feed_dict={
            self.states: states,
            self.is_training: False
        })

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
            self.is_training: False
        })

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return tf.get_default_session().run(self.action_output, feed_dict={
            self.states: [state],
            self.is_training: False
        })[0]

    def train(self, model_summary=True):
        sample = self.replay_buffer.get_batch(self.config["batch_size"])

        # for n_actions = 1
        action_batch = np.resize(sample["actions"], [self.config["batch_size"], self.n_actions])

        # Calculate critic targets
        next_value_batch = self.target_value(sample["states1"])
        softq_targets = sample["rewards"] + (1 - sample["terminals1"]) * \
            self.config["gamma"] * next_value_batch.squeeze()
        softq_targets = np.resize(softq_targets, [self.config["batch_size"], 1]).astype(np.float32)

        next_actions = self.actions(sample["states0"])
        next_q_value_batch = self.softq_value(sample["states0"], next_actions)
        value_batch = self.value(sample["states0"])

        log_prob, next_q_value_batch, value_batch = tf.get_default_session().run(
            [self.action_logprob, self.softq_output, self.value_output],
            feed_dict={
                self.states: sample["states0"],
                self.actions_taken: next_actions
            }
        )
        advantage_batch = log_prob - (next_q_value_batch - value_batch)

        value_targets = next_q_value_batch - log_prob
        # Update actor weights
        fetches = [self.train_op]
        if model_summary:
            fetches = [self.softq_output, self.softq_loss, self.actor_loss, self.value_loss] + fetches
        results = tf.get_default_session().run(fetches, feed_dict={
            self.softq_target: softq_targets,
            self.states: sample["states0"],
            self.actions_taken: action_batch,
            self.advantage: advantage_batch,
            self.value_target: value_targets,
            self.is_training: True
        })

        if model_summary:
            summary = tf.Summary()
            summary.value.add(tag="model/predicted_softq_mean", simple_value=np.mean(results[0]))
            summary.value.add(tag="model/predicted_softq_std", simple_value=np.std(results[0]))
            summary.value.add(tag="model/softq_loss", simple_value=float(results[1]))
            summary.value.add(tag="model/actor_loss", simple_value=float(results[2]))
            summary.value.add(tag="model/value_loss", simple_value=float(results[3]))
            self.summary_writer.add_summary(summary, self.n_updates)

        # Update the target networks
        fetches = [self.value_target_update]
        if model_summary:
            fetches = fetches + [self.model_summary_op]
        tf.get_default_session().run(fetches)
        self.n_updates += 1

    def learn(self):
        timestep_limit = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        with tf.Session() as sess, sess.as_default():
            sess.run(self.init_op)
            for episode in range(self.config["n_episodes"]):
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                for _ in range(self.config["n_timesteps"]):
                    action = self.action(state)
                    to_execute = 2 * (action - action_low) / (action_high - action_low) - 1
                    to_execute = np.clip(to_execute, action_low, action_high)
                    new_state, reward, done, _ = self.env.step(to_execute)
                    episode_length += 1
                    episode_reward += reward
                    self.replay_buffer.add(state, action, reward, new_state, done)
                    if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                        for _ in range(self.config["n_train_steps"]):
                            self.train(model_summary=(self.n_updates % 50) == 0)
                    state = new_state
                    if done or episode_length >= timestep_limit:
                        summary = tf.Summary()
                        summary.value.add(tag="global/Episode_length",
                                          simple_value=float(episode_length))
                        summary.value.add(tag="global/Reward",
                                          simple_value=float(episode_reward))
                        self.summary_writer.add_summary(summary, episode)
                        self.summary_writer.flush()
                        break
