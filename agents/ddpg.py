# -*- coding: utf8 -*-

import tensorflow as tf

from agents.agent import Agent
from misc.noise import OrnsteinUhlenbeckActionNoise
from misc.network_ops import linear, normalized_columns_initializer

class DDPGAgent(Agent):
    def __init__(self, env, monitor_path: str, **usercfg):
        super(DDPGAgent, self).__init__(**usercfg)
        self.env = env
        self.monitor_path: str = monitor_path

        self.config.update(
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-2,
            ou_theta=0.15,
            ou_mu=0.0,
            ou_sigma=0.2,
            epsilon=1,
            batch_size=128,
            discount=0.99,
            tau=0.001,
            n_actor_layers=2,
            n_hidden_units=64,
            layer_norm=True
        )
        self.config.update(usercfg)

        self.n_actions = env.action_space.shape[0]
        self.states = tf.placeholder(
            tf.float32, [None] + list(env.observation_space.shape), name="states")
        self.actions_taken = tf.placeholder(
            tf.float32, [None, self.n_actions], name="actions_taken")

        self.actor = self.build_actor()
        self.critic = self.build_critic()

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
        return x

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
        return x

    def learn(self):
        pass
