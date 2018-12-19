# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf

from yarll.agents.agent import Agent
from yarll.memory.memory import Memory
from yarll.misc.network_ops import linear, normalized_columns_initializer

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

        # Placeholders
        self.states = tf.placeholder(tf.float32, [None] + self.state_shape, name="states")
        self.actions_taken = tf.placeholder(tf.float32, [self.config["batch_size"], self.n_actions],
                                            name="actions_taken")
        self.softq_target = tf.placeholder(tf.float32, [self.config["batch_size"], 1], name="softq_target")

        # Make networks
        # action_output are the squashed actions and action_original those straight from the normal distribution
        self.action_output, self.action_original, self.action_logprob, self.actor_vars = self.build_actor_network()
        self.softq_output, self.softq_vars = self.build_softq_network(self.actions_taken)
        self.new_softq_output, _ = self.build_softq_network(self.action_output, reuse_vars=True)
        self.value_output, self.value_vars = self.build_value_network()
        self.target_value_output, self.value_target_update = self.build_target_value_network(self.value_vars)

        # Make losses
        self.softq_loss = tf.reduce_mean(tf.square(self.softq_output - self.softq_target), name="softq_loss")
        value_target = tf.stop_gradient(self.new_softq_output - self.action_logprob)
        self.value_loss = tf.reduce_mean(tf.square(self.value_output - value_target), name="value_loss")
        advantage = tf.stop_gradient(self.action_logprob - self.new_softq_output + self.value_output)
        self.actor_loss = tf.reduce_mean(self.action_logprob * advantage, name="actor_loss")

        # Make train ops
        self.softq_train_op = tf.train.AdamOptimizer(
            learning_rate=self.config["softq_learning_rate"],
            name="softq_optimizer").minimize(self.softq_loss)
        self.value_train_op = tf.train.AdamOptimizer(
            learning_rate=self.config["value_learning_rate"],
            name="value_optimizer").minimize(self.value_loss)

        self.actor_train_op = tf.train.AdamOptimizer(
            learning_rate=self.config["actor_learning_rate"],
            name="actor_optimizer").minimize(self.actor_loss)

        if self.config["weights_summary"]:
            summaries = []
            for v in self.actor_vars + self.softq_vars + self.value_vars:
                summaries.append(tf.summary.histogram(v.name, v))
            self.model_summary_op = tf.summary.merge(summaries)

        self.init_op = tf.global_variables_initializer()

        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))

        self.n_updates = 0

        self.summary_writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "summaries"), tf.get_default_graph())

        self.session = tf.Session()

    def build_actor_network(self):
        w_bound = 3e-3
        x = self.states
        with tf.variable_scope("actor"):
            for i in range(self.config["n_hidden_layers"]):
                x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L{}".format(i + 1), normalized_columns_initializer()))

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
            log_std_clipped = tf.clip_by_value(log_std,
                                               -20, 2,
                                               name="log_std_clipped") # In autor's code but not in paper

            normal_dist = tf.distributions.Normal(mean, tf.exp(log_std_clipped), name="actions_normal_distr")
            actions = tf.stop_gradient(normal_dist.sample(name="actions"))
            squashed_actions = tf.tanh(actions, name="squashed_actions")  # Squash output between [-1, 1]

            logprob = normal_dist.log_prob(actions) - \
            tf.log(1.0 - tf.pow(squashed_actions, 2) + self.config["logprob_epsilon"])
            logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)

            actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return squashed_actions, actions, logprob, actor_vars

    def build_softq_network(self, actions, reuse_vars=False):
        x = tf.concat([self.states, actions], 1)
        with tf.variable_scope("softq") as scope:
            if reuse_vars:
                scope.reuse_variables()
            for i in range(self.config["n_hidden_layers"]):
                x = tf.nn.relu(linear(x, self.config["n_hidden_units"], "L{}".format(i + 1), normalized_columns_initializer()))
            x = linear(x,
                       1,
                       "output",
                       tf.random_uniform_initializer(-3e-3, 3e-3),
                       tf.random_uniform_initializer(-3e-3, 3e-3))

            softq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return x, softq_vars

    def build_value_network(self):
        x = self.states
        with tf.variable_scope("value"):
            for i in range(self.config["n_hidden_layers"]):
                x = tf.nn.relu(linear(x,
                                      self.config["n_hidden_units"],
                                      "L{}".format(i + 1),
                                      normalized_columns_initializer()))
            x = linear(x,
                       1,
                       "output",
                       tf.random_uniform_initializer(-3e-3, 3e-3),
                       tf.random_uniform_initializer(-3e-3, 3e-3))

            value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return x, value_vars

    def build_target_value_network(self, value_vars: list):
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.config["tau"])
        target_update = ema.apply(value_vars)
        target_net = [ema.average(v) for v in value_vars]

        x = self.states
        for i in range(self.config["n_hidden_layers"]):
            x = tf.nn.relu(tf.nn.xw_plus_b(x, target_net[i * 2], target_net[i * 2 + 1]))
        value = tf.nn.xw_plus_b(x,
                                target_net[2 * self.config["n_hidden_layers"]],
                                target_net[2 * self.config["n_hidden_layers"] + 1],
                                name="target_value")

        return value, target_update

    def value(self, states: np.ndarray):
        return tf.get_default_session().run(self.value_output, feed_dict={
            self.states: states
        })

    def target_value(self, states: np.ndarray):
        return tf.get_default_session().run(self.target_value_output, feed_dict={
            self.states: states
        })

    def softq_value(self, states: np.ndarray, actions: np.ndarray):
        return tf.get_default_session().run(self.softq_output, feed_dict={
            self.states: states,
            self.actions_taken: actions
            })

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        return tf.get_default_session().run(self.action_output, feed_dict={
            self.states: states
        })

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return self.actions([state])[0]

    def train(self, model_summary=True):
        sample = self.replay_buffer.get_batch(self.config["batch_size"])

        # for n_actions == 1
        action_batch = np.resize(sample["actions"], [self.config["batch_size"], self.n_actions])

        # Calculate critic targets
        next_value_batch = self.target_value(sample["states1"])
        softq_targets = sample["rewards"] + (1 - sample["terminals1"]) * \
            self.config["gamma"] * next_value_batch.squeeze()
        softq_targets = np.resize(softq_targets, [self.config["batch_size"], 1]).astype(np.float32)

        fetches = [self.action_logprob, self.actor_train_op, self.value_train_op]
        if model_summary:
            fetches += [self.actor_loss, self.value_loss]
        results = tf.get_default_session().run(
            fetches,
            feed_dict={self.states: sample["states0"]}
        )

        logprob = results[0]

        if model_summary:
            actor_loss = results[-2]
            value_loss = results[-1]

        fetches = [self.softq_train_op]
        if model_summary:
            fetches = [self.softq_output, self.softq_loss] + fetches
        results = tf.get_default_session().run(fetches, feed_dict={
            self.softq_target: softq_targets,
            self.states: sample["states0"],
            self.actions_taken: action_batch
        })

        if model_summary:
            summary = tf.Summary()
            summary.value.add(tag="model/predicted_softq_mean", simple_value=np.mean(results[0]))
            summary.value.add(tag="model/predicted_softq_std", simple_value=np.std(results[0]))
            summary.value.add(tag="model/softq_loss", simple_value=float(results[1]))
            summary.value.add(tag="model/actor_loss", simple_value=float(actor_loss))
            summary.value.add(tag="model/value_loss", simple_value=float(value_loss))
            summary.value.add(tag="model/action_logprob_mean", simple_value=np.mean(logprob))
            self.summary_writer.add_summary(summary, self.n_updates)

        # Update the target networks
        fetches = [self.value_target_update]
        if model_summary and self.config["weights_summary"]:
            fetches = [self.model_summary_op] + fetches
        res = tf.get_default_session().run(fetches)
        if model_summary and self.config["weights_summary"]:
            self.summary_writer.add_summary(res[0], self.n_updates)
        self.n_updates += 1

    def learn(self):
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        with self.session as sess, sess.as_default():
            sess.run(self.init_op)
            for episode in range(self.config["n_episodes"]):
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                for _ in range(self.config["n_timesteps"]):
                    action = self.action(state)
                    to_execute = action_low + (action + 1.0) * 0.5 * (action_high - action_low)
                    new_state, reward, done, _ = self.env.step(to_execute)
                    episode_length += 1
                    episode_reward += reward
                    self.replay_buffer.add(state, action, reward, new_state, done)
                    if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                        for _ in range(self.config["n_train_steps"]):
                            self.train(model_summary=True)
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
