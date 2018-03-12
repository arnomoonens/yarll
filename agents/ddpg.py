# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf

from agents.agent import Agent
from memory.memory import Memory
from misc.noise import OrnsteinUhlenbeckActionNoise
from misc.network_ops import fan_in_initializer, linear_fan_in
from misc.utils import RunningMeanStd, normalize

def batch_norm_layer(x, training_phase, scope_bn, activation=None):
    return tf.cond(
        training_phase,
        lambda: tf.contrib.layers.batch_norm(
            x,
            activation_fn=activation,
            center=True,
            scale=True,
            updates_collections=None,
            is_training=True,
            reuse=None,
            scope=scope_bn,
            decay=0.9,
            epsilon=1e-5),
        lambda: tf.contrib.layers.batch_norm(
            x,
            activation_fn=activation,
            center=True,
            scale=True,
            updates_collections=None,
            is_training=False,
            reuse=True,
            scope=scope_bn,
            decay=0.9,
            epsilon=1e-5))

class DDPG(Agent):
    def __init__(self, env, monitor_path: str, **usercfg):
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
            layer_norm=True,
            replay_buffer_size=1e6,
            replay_start_size=10000  # Required number of replay buffer entries to start training
        )
        self.config.update(usercfg)

        self.states_rms: RunningMeanStd = RunningMeanStd(env.observation_space.shape)

        self.n_actions: int = env.action_space.shape[0]
        self.states = tf.placeholder(
            tf.float32, [None] + list(env.observation_space.shape), name="states")
        self.actions_taken = tf.placeholder(tf.float32, [None, self.n_actions], name="actions_taken")
        self.critic_target = tf.placeholder(tf.float32, [None], name="critic_target")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Current networks
        actor, actor_vars = self.build_actor()
        self.actor_output = actor
        self.critic_with_placeholder, critic_vars = self.build_critic(self.actions_taken)

        # Target networks
        with tf.variable_scope("target"):
            self.target_actor, target_actor_update = self.build_target_actor(actor_vars)
            target_critic, target_critic_update = self.build_target_critic(self.actions_taken, critic_vars)
        self.target_network_value = target_critic

        self.target_update_op = tf.group(target_actor_update, target_critic_update)

        l2_loss = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.config["l2_loss_coef"]),
            weights_list=critic_vars
        )
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_with_placeholder - self.critic_target)) + l2_loss
        self.critic_loss = tf.Print(
            self.critic_loss,
            [self.critic_target, self.critic_with_placeholder, l2_loss, self.critic_loss],
            message="YINP, QOUT, WEIGHTDECAY, LOSS=",
            first_n=200
        )

        self.actor_loss = -tf.reduce_mean(self.critic_with_placeholder)

        actor_grads = tf.gradients(actor, actor_vars, -self.critic_with_placeholder, name="actor_grads")
        actor_optimizer = tf.train.AdamOptimizer(
            self.config["actor_learning_rate"],
            name="actor_optim")
        self.actor_train_op = actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

        # critic_grads = tf.gradients(self.critic_loss, critic_vars, name="critic_grads")
        # critic_optimizer = tf.train.AdamOptimizer(
        #     self.config["critic_learning_rate"],
        #     name="critic_optim")
        # self.critic_train_op = critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))
        self.critic_train_op = tf.train.AdamOptimizer(
            self.config["critic_learning_rate"],
            name="critic_optimizer"
        ).minimize(self.critic_loss, var_list=critic_vars, name="critic_train_op")

        self.init_op = tf.global_variables_initializer()

        self.action_noise = OrnsteinUhlenbeckActionNoise(
            self.n_actions,
            self.config["ou_sigma"],
            self.config["ou_theta"]
        )

        self.replay_buffer = Memory(int(self.config["replay_buffer_size"]))

        summaries = [
            tf.summary.scalar("model/actor_loss", self.actor_loss),
            tf.summary.scalar("model/critic_loss", self.critic_loss)
        ]
        # for v, g in zip(actor_vars + critic_vars, actor_grads + critic_grads):
        #     summaries.append(tf.summary.histogram(v.name, v))
        #     summaries.append(tf.summary.histogram(v.name + "_grad", g))
        self.model_summary_op = tf.summary.merge(summaries)

        self.summary_writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "summaries"), tf.get_default_graph())

    def build_actor(self):
        with tf.variable_scope("actor"):
            x = self.states
            trainable_vars = []
            hidden_units = [400, 300]
            for i in range(self.config["n_actor_layers"]):
                with tf.variable_scope("L{}".format(i + 1)):
                    x, layer_vars = linear_fan_in(x, hidden_units[i])
                    trainable_vars += layer_vars
                    if self.config["layer_norm"]:
                        x = batch_norm_layer(
                            x,
                            training_phase=self.is_training,
                            scope_bn="batch_norm_{}".format(i + 1)
                        )
                    x = tf.nn.relu(x)

            with tf.variable_scope("actions"):
                w = tf.get_variable(
                    "w",
                    [x.shape[1], 1],
                    initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
                b = tf.get_variable(
                    "b",
                    [self.n_actions],
                    initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
                trainable_vars += [w, b]
                x = tf.tanh(tf.nn.xw_plus_b(x, w, b), "action")
        return x, trainable_vars

    def build_target_actor(self, actor_vars):
        with tf.variable_scope("actor"):
            x = self.states
            ema = tf.train.ExponentialMovingAverage(decay=1-self.config["tau"])
            target_update = ema.apply(actor_vars)
            target_net = [ema.average(x) for x in actor_vars]
            for i in range(self.config["n_actor_layers"]):
                with tf.variable_scope("L{}".format(i + 1)):
                    w_idx = i * 2
                    b_idx = w_idx + 1
                    x = tf.nn.xw_plus_b(x, target_net[w_idx], target_net[b_idx])
                    if self.config["layer_norm"]:
                        x = batch_norm_layer(
                            x,
                            training_phase=self.is_training,
                            scope_bn="batch_norm_{}".format(i + 1)
                        )

            with tf.variable_scope("actions"):
                w_idx = self.config["n_actor_layers"] * 2
                b_idx = w_idx + 1
                x = tf.tanh(tf.nn.xw_plus_b(x, target_net[w_idx], target_net[b_idx]))
            return x, target_update

    def build_critic(self, actions_placeholder):
        with tf.variable_scope("critic"):
            x = self.states
            with tf.variable_scope("L1"):
                x, trainable_vars = linear_fan_in(x, 400)
                if self.config["layer_norm"]:
                    x = batch_norm_layer(
                        x,
                        training_phase=self.is_training,
                        scope_bn="batch_norm_1"
                    )
                x = tf.nn.relu(x)

            x = tf.concat([x, actions_placeholder], axis=-1)

            with tf.variable_scope("L2"):
                input_size = x.shape[1].value
                w = tf.get_variable("w", [input_size, 300],
                                    initializer=fan_in_initializer(input_size))
                b = tf.get_variable("b", [300],
                                    initializer=fan_in_initializer(input_size))
                x = tf.nn.relu(tf.nn.xw_plus_b(x, w, b))
                trainable_vars += [w, b]

            with tf.variable_scope("value"):
                w = tf.get_variable(
                    "w",
                    [x.shape[1], 1],
                    initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
                b = tf.get_variable(
                    "b",
                    [1],
                    initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
                trainable_vars += [w, b]
                x = tf.nn.xw_plus_b(x, w, b, "q_value_placeholder")
        return x, trainable_vars

    def build_target_critic(self, actions_placeholder, critic_vars):
        with tf.variable_scope("critic"):
            x = self.states
            ema = tf.train.ExponentialMovingAverage(decay=1-self.config["tau"])
            target_update = ema.apply(critic_vars)
            target_net = [ema.average(x) for x in critic_vars]
            with tf.variable_scope("L1"):
                x = tf.nn.xw_plus_b(x, target_net[0], target_net[1])
                if self.config["layer_norm"]:
                    x = batch_norm_layer(
                        x,
                        training_phase=self.is_training,
                        scope_bn="batch_norm_1"
                    )
                x = tf.nn.relu(x)

            x = tf.concat([x, actions_placeholder], axis=-1)

            with tf.variable_scope("L2"):
                x = tf.nn.relu(tf.nn.xw_plus_b(x, target_net[2], target_net[3]))

            with tf.variable_scope("value"):
                x = tf.nn.xw_plus_b(x, target_net[4], target_net[5])

            return x, target_update


    def choose_action(self, state):
        feed_dict = {
            self.states: [state],
            self.is_training: False
        }
        action = tf.get_default_session().run([self.actor_output], feed_dict=feed_dict)[0][0]
        action += self.action_noise()
        return action

    def actions(self, states):
        feed_dict = {
            self.states: states,
            self.is_training: True
        }
        return tf.get_default_session().run([self.actor_output], feed_dict=feed_dict)[0]

    def target_actions(self, states):
        feed_dict = {
            self.states: states,
            self.is_training: True
        }
        return tf.get_default_session().run([self.target_actor], feed_dict=feed_dict)[0]

    def learn(self):
        max_action = self.env.action_space.high
        with tf.Session() as sess, sess.as_default():
            sess.run(self.init_op)
            state = None
            n_episodes = 0
            episode_length = 0
            episode_reward = 0
            n_updates = 0
            for _ in range(self.config["n_episodes"]):
                for _ in range(self.config["n_timesteps"]):
                    if state is None:
                        state = self.env.reset()
                    self.states_rms.add_value(state)
                    if self.states_rms.count >= 2:
                        inp = normalize(state, self.states_rms.mean, self.states_rms.std)
                    else:
                        inp = state
                    action = self.choose_action(inp)
                    new_state, reward, done, _ = self.env.step(action * max_action)
                    episode_length += 1
                    episode_reward += reward
                    # Store transition
                    if self.states_rms.count >= 2:
                        inp_new = normalize(new_state, self.states_rms.mean, self.states_rms.std)
                    else:
                        inp_new = new_state
                    self.replay_buffer.add(inp, action, reward, inp_new, done)
                    if done:
                        state = self.env.reset()
                        self.action_noise.reset()
                        summary = tf.Summary()
                        summary.value.add(tag="global/Episode_length",
                                          simple_value=float(episode_length))
                        summary.value.add(tag="global/Reward",
                                          simple_value=float(episode_reward))
                        self.summary_writer.add_summary(summary, n_episodes)
                        n_episodes += 1
                        episode_length = 0
                        episode_reward = 0
                    else:
                        state = new_state
                    # Sample minibatch
                    if self.replay_buffer.n_entries >= self.config["replay_start_size"]:
                        sample = self.replay_buffer.get_batch(batch_size=self.config["batch_size"])
                        next_actions = self.target_actions(sample["states1"])
                        feed_dict = {
                            self.states: sample["states1"],
                            self.actions_taken: next_actions,
                            self.is_training: False
                        }
                        next_q_values = sess.run(self.target_network_value, feed_dict=feed_dict)
                        target_q = sample["rewards"] + (1.0 - sample["terminals1"]) * \
                        self.config["gamma"] * next_q_values.squeeze()
                        feed_dict = {
                            self.states: sample["states0"],
                            self.actions_taken: sample["actions"],
                            self.critic_target: np.squeeze(target_q),
                            self.is_training: True
                        }
                        sess.run([self.critic_train_op, self.critic_loss], feed_dict=feed_dict)
                        fetches = [
                            self.critic_with_placeholder,
                            self.actor_train_op,
                        ]
                        feed_dict = {
                            self.states: sample["states0"],
                            self.actions_taken: self.actions(sample["states0"]),
                            self.critic_target: np.squeeze(target_q),
                            self.is_training: True
                        }
                        predicted_q, _ = sess.run(fetches, feed_dict=feed_dict)
                        # self.summary_writer.add_summary(summary, n_updates)
                        summary = tf.Summary()
                        summary.value.add(tag="model/predicted_q_mean",
                                          simple_value=np.mean(predicted_q))
                        summary.value.add(tag="model/predicted_q_std",
                                          simple_value=np.std(predicted_q))
                        self.summary_writer.add_summary(summary, n_updates)
                        n_updates += 1
                        sess.run(self.target_update_op)

                    # Update target networks
                    self.summary_writer.flush()
