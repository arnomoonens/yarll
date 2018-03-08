# -*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf

from agents.agent import Agent
from memory.memory import Memory
from misc.noise import OrnsteinUhlenbeckActionNoise
from misc.network_ops import linear, linear_fan_in
from misc.utils import RunningMeanStd, normalize

def soft_target_update(source_network_vars, target_network_vars, tau):
    updates = []
    for source, target in zip(source_network_vars, target_network_vars):
        update = tf.assign(target, tau * source + (1.0 - tau) * target)
        updates.append(update)
    return tf.group(*updates)

def hard_target_update(source_network_vars, target_network_vars):
    """
    Copy the values of the source network variables to the target network variables.
    The same as using `soft_target_update` with tau=0.
    However, this is not done this way as it would need already initialized values for
    the target network variables.
    """
    return tf.group(*[tf.assign(t, s) for s, t in zip(source_network_vars, target_network_vars)])

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
            epsilon=1,
            gamma=0.99,
            batch_size=64,
            tau=0.001,
            l2_loss_coef=1e-2,
            n_actor_layers=2,
            n_hidden_units=64,
            layer_norm=True,
            replay_buffer_size=1e6,
            replay_start_size=1000  # Required number of replay buffer entries to start training
        )
        self.config.update(usercfg)

        self.states_rms: RunningMeanStd = RunningMeanStd(env.observation_space.shape)

        self.n_actions: int = env.action_space.shape[0]
        self.states = tf.placeholder(
            tf.float32, [None] + list(env.observation_space.shape), name="states")
        self.actions_taken = tf.placeholder(tf.float32, [None, self.n_actions], name="actions_taken")
        self.critic_target = tf.placeholder(tf.float32, [None], name="critic_target")

        # Current networks
        actor, actor_vars = self.build_actor()
        self.actor_output = actor
        critic_with_placeholder, critic_vars = self.build_critic(self.actions_taken)
        critic_with_actor, _ = self.build_critic(actor, reuse_variables=True)

        # Target networks
        with tf.variable_scope("target"):
            target_actor, target_actor_vars = self.build_actor()
            target_critic, target_critic_vars = self.build_critic(target_actor)
        self.target_network_value = target_critic

        self.target_init_op = tf.group(
            hard_target_update(actor_vars, target_actor_vars),
            hard_target_update(critic_vars, target_critic_vars)
        )

        self.target_update_op = tf.group(
            soft_target_update(actor_vars, target_actor_vars, self.config["tau"]),
            soft_target_update(critic_vars, target_critic_vars, self.config["tau"])
        )

        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in critic_vars])
        self.critic_loss = tf.reduce_mean(tf.square(critic_with_placeholder - self.critic_target)) + l2_loss

        self.actor_loss = -tf.reduce_mean(critic_with_actor)

        actor_grads = tf.gradients(self.actor_loss, actor_vars)
        critic_grads = tf.gradients(self.critic_loss, critic_vars)

        with tf.variable_scope("optimizers"):
            actor_optimizer = tf.train.AdamOptimizer(
                self.config["actor_learning_rate"],
                name="actor_optim")

            self.actor_train_op = actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

            critic_optimizer = tf.train.AdamOptimizer(
                self.config["critic_learning_rate"],
                name="critic_optim")

            self.critic_train_op = critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        optimizer_variables = [
            var for var in tf.global_variables() if var.name.startswith("optimizers")]

        self.init_op = tf.variables_initializer(actor_vars + critic_vars + optimizer_variables)

        self.action_noise = OrnsteinUhlenbeckActionNoise(
            self.n_actions,
            self.config["ou_sigma"],
            self.config["ou_theta"]
        )

        self.replay_buffer = Memory(
            int(self.config["replay_buffer_size"]),
            self.env.action_space.shape,
            self.env.observation_space.shape
            )

        summaries = [
            tf.summary.scalar("model/actor_loss", self.actor_loss),
            tf.summary.scalar("model/critic_loss", self.critic_loss)
        ]
        for v in actor_vars + critic_vars:
            summaries.append(tf.summary.histogram(v.name, v))
        self.model_summary_op = tf.summary.merge(summaries)

        self.summary_writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "summaries"), tf.get_default_graph())

    def build_actor(self):
        with tf.variable_scope("actor"):
            x = self.states
            for i in range(self.config["n_actor_layers"]):
                with tf.variable_scope("L{}".format(i + 1)):
                    x = linear_fan_in(x, self.config["n_hidden_units"])
                    if self.config["layer_norm"]:
                        x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
                    x = tf.nn.relu(x)
            x = tf.tanh(linear(x, self.n_actions, "actions",
                               initializer=tf.random_uniform_initializer(-3e-3, 3e-3)))
            trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        return x, trainable_vars

    def build_critic(self, actions, reuse_variables=False):
        with tf.variable_scope("critic", reuse=reuse_variables):
            x = self.states
            with tf.variable_scope("L1"):
                x = linear_fan_in(x, self.config["n_hidden_units"])
                if self.config["layer_norm"]:
                    x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.concat([x, actions], axis=-1)
            with tf.variable_scope("L2"):
                x = linear_fan_in(x, self.config["n_hidden_units"])
                x = tf.nn.relu(x)

            x = linear(x, 1, "value", initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
            trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        return x, trainable_vars

    def choose_action(self, state):
        feed_dict = {
            self.states: [state]
        }
        action = tf.get_default_session().run([self.actor_output], feed_dict=feed_dict)[0][0]
        action += self.action_noise()
        return action


    def learn(self):
        max_action = self.env.action_space.high
        with tf.Session() as sess, sess.as_default():
            sess.run(self.init_op)
            sess.run(self.target_init_op)
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
                    self.replay_buffer.append(inp, action, reward, inp_new, done)
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
                        sample = self.replay_buffer.sample(batch_size=self.config["batch_size"])
                        feed_dict = {
                            self.states: sample["states1"]
                        }
                        next_q_values = sess.run(self.target_network_value, feed_dict=feed_dict)
                        target_q = sample["rewards"] + (1.0 - sample["terminals1"]) * \
                        self.config["gamma"] * next_q_values
                        fetches = [
                            self.model_summary_op,
                            self.actor_train_op,
                            self.critic_train_op
                        ]
                        feed_dict = {
                            self.states: sample["states0"],
                            self.actions_taken: sample["actions"],
                            self.critic_target: np.squeeze(target_q)
                        }
                        summary, _, _ = sess.run(fetches, feed_dict=feed_dict)
                        sess.run(self.target_update_op)
                        self.summary_writer.add_summary(summary, n_updates)
                        n_updates += 1

                    # Update target networks
                    self.summary_writer.flush()
