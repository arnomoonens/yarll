# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf

from agents.agent import Agent
from memory.memory import Memory
from misc.noise import OrnsteinUhlenbeckActionNoise
from misc.network_ops import linear, normalized_columns_initializer

def soft_target_update(source_network_vars, target_network_vars, tau):
    updates = []
    for source, target in zip(source_network_vars, target_network_vars):
        update = tf.assign(target, tau * source + (1.0 - tau) * target)
        updates.append(update)
    return tf.group(*updates)

def hard_target_update(source_network_vars, target_network_vars):
    return soft_target_update(source_network_vars, target_network_vars, 1.0)

class DDPG(Agent):
    def __init__(self, env, monitor_path: str, **usercfg):
        super(DDPG, self).__init__(**usercfg)
        self.env = env
        self.monitor_path: str = monitor_path

        self.config.update(
            n_episodes=500,
            n_timesteps=20,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-2,
            ou_theta=0.15,
            ou_sigma=0.2,
            epsilon=1,
            batch_size=128,
            discount=0.99,
            tau=0.001,
            n_actor_layers=2,
            n_hidden_units=64,
            layer_norm=True,
            replay_buffer_size=1e6
        )
        self.config.update(usercfg)

        self.n_actions: int = env.action_space.shape[0]
        self.states = tf.placeholder(
            tf.float32, [None] + list(env.observation_space.shape), name="states")
        self.actions_taken = tf.placeholder(
            tf.float32, [None, self.n_actions], name="actions_taken")
        self.critic_target = tf.placeholder(tf.float32, [None], name="critic_target")

        # Current networks
        actor, actor_vars = self.build_actor()
        self.actor_output = actor
        critic, critic_vars = self.build_critic()
        self.value = critic

        self.current_nets_init_op = tf.variables_initializer(actor_vars + critic_vars)

        # Target networks
        target_actor, target_actor_vars = self.build_actor()
        target_critic, target_critic_vars = self.build_critic()
        self.target_network_value = target_critic

        self.target_init_op = tf.group(
            hard_target_update(actor_vars, target_actor_vars),
            hard_target_update(critic_vars, target_critic_vars)
        )

        self.target_update_op = tf.group(
            soft_target_update(actor_vars, target_actor_vars, self.config["tau"]),
            soft_target_update(critic_vars, target_critic_vars, self.config["tau"])
        )

        self.critic_loss = tf.reduce_mean(tf.square(critic - self.critic_target))

        self.critic_loss =

        self.action_noise = OrnsteinUhlenbeckActionNoise(
            self.n_actions,
            self.config["ou_sigma"],
            self.config["ou_theta"]
        )

        self.replay_buffer = Memory(
            self.config["replay_buffer_size"],
            self.env.action_space.shape,
            self.env.observation_space.shape
            )

    def build_actor(self):
        with tf.variable_scope("actor"):
            x = self.states
            for i in range(self.config["n_actor_layers"]):
                x = linear(x, self.config["n_actor_hidden_units"], "L{}".format(i + 1),
                           initializer=normalized_columns_initializer(1.0))
                if self.config["layer_norm"]:
                    x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
            x = tf.tanh(linear(x, self.n_actions, "actions",
                               initializer=normalized_columns_initializer(0.01)))
            trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        return x, trainable_vars

    def build_critic(self):
        with tf.variable_scope("critic"):
            x = self.states
            x = linear(x, self.config["n_actor_hidden_units"], "L1",
                       initializer=normalized_columns_initializer(1.0))
            if self.config["layer_norm"]:
                x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, self.actions_taken], axis=-1)
            x = linear(x, self.config["n_actor_hidden_units"], "L2",
                       initializer=normalized_columns_initializer(1.0))
            if self.config["layer_norm"]:
                x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = linear(x, 1, "value", initializer=normalized_columns_initializer(0.01))
            trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        return x, trainable_vars

    def choose_action(self, state):
        feed_dict = {
            self.states: [state]
        }
        action = tf.get_default_session().run([self.actor_output], feed_dict=feed_dict)
        action += self.action_noise()
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action


    def learn(self):
        max_action = self.env.action_space.high
        with tf.Session() as sess, sess.as_default():
            sess.run([self.current_nets_init_op, self.target_init_op])
            state = None
            for episode in range(self.config["n_episodes"]):
                for timestep in range(self.config["n_timesteps"]):
                    if state is None:
                        state = self.env.reset()

                    action = self.choose_action(state)
                    new_state, reward, done, _ = self.env.step(action * max_action)
                    # Store transition
                    self.replay_buffer.append(state, action, reward, new_state, done)
                    if done:
                        state = self.env.reset()
                        self.action_noise.reset()
                    else:
                        state = new_state
                    # Sample minibatch
                    if self.replay_buffer.n_entries >= self.config["batch_size"]:
                        sample = self.replay_buffer.sample(batch_size=self.config["batch_size"])
                        feed_dict = {
                            self.states: sample["states1"]
                        }
                        next_q_values = sess.run(self.target_network_value, feed_dict=feed_dict)
                        target_q = sample["rewards"] + (1.0 - sample["terminals1"]) * \
                        self.config["gamma"] * next_q_values



                    # Update critic and actor
                    # Update target networks
                    sess.run(self.target_update_op)
