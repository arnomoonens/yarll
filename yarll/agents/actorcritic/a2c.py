# -*- coding: utf8 -*-

import os
import logging
import tensorflow as tf
import numpy as np

from gym import wrappers

from yarll.agents.agent import Agent
from yarll.agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
ActorCriticNetworkDiscreteCNN, ActorCriticNetworkDiscreteCNNRNN, actor_critic_discrete_loss,\
ActorCriticNetworkContinuous, actor_critic_continuous_loss
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards, FastSaver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Don't use the scientific notation to print results
np.set_printoptions(suppress=True)


class A2C(Agent):
    """Advantage Actor Critic"""

    def __init__(self, env, monitor_path: str, video: bool = True, **usercfg) -> None:
        super(A2C, self).__init__(**usercfg)
        self.monitor_path = monitor_path

        self.env = wrappers.Monitor(
            env,
            monitor_path,
            force=True,
            video_callable=(None if video else False))

        self.config.update(dict(
            n_iter=100,
            gamma=0.99,
            learning_rate=0.001,
            n_hidden_units=20,
            n_hidden_layers=1,
            gradient_clip_value=0.5,
            n_local_steps=20,
            vf_coef=0.5,
            entropy_coef=0.01,
            loss_reducer="mean",
            save_model=False
        ))
        self.config.update(usercfg)
        # Only used (and overwritten) by agents that use an RNN
        self.initial_features = None
        self.ac_net = None  # Overwritten by build_networks
        self.build_networks()

        self.action = self.ac_net.action
        self.states = self.ac_net.states
        self.actions_taken = self.ac_net.actions_taken
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.ret = tf.placeholder(tf.float32, [None], name="return")

        self.actor_loss, self.critic_loss, self.loss = self.make_loss()

        self.vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        self._global_step = tf.get_variable(
            "global_step",
            [],
            tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

        self.optimizer = tf.train.AdamOptimizer(
            self.config["learning_rate"], name="optim")
        grads = tf.gradients(self.loss, self.vars)
        grads, _ = tf.clip_by_global_norm(
            grads, self.config["gradient_clip_value"])

        # Apply gradients to the weights of the master network
        apply_grads = self.optimizer.apply_gradients(zip(grads, self.vars))

        self.n_steps = tf.shape(self.states)[0]
        inc_step = self._global_step.assign_add(self.n_steps)
        self.train_op = tf.group(apply_grads, inc_step)

        self.init_op = tf.global_variables_initializer()
        # Launch the graph.
        self.session = tf.Session()
        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver = FastSaver()
        n_steps = tf.to_float(self.n_steps)
        actor_loss_summary = tf.summary.scalar("model/actor_loss", tf.squeeze(self.actor_loss / n_steps))
        critic_loss_summary = tf.summary.scalar("model/critic_loss", tf.squeeze(self.critic_loss / n_steps))
        loss_summary = tf.summary.scalar("model/loss", tf.squeeze(self.loss / n_steps))
        self.loss_summary_op = tf.summary.merge(
            [actor_loss_summary, critic_loss_summary, loss_summary])
        self.writer = tf.summary.FileWriter(os.path.join(
            self.monitor_path, "summaries"), self.session.graph)
        self.env_runner = EnvRunner(self.env, self, usercfg, summary_writer=self.writer)
        return

    def _initialize(self):
        self.session.run(self.init_op)

    def build_networks(self):
        return NotImplementedError("Abstract method")

    def make_loss(self):
        return NotImplementedError("Abstract method")

    @property
    def global_step(self):
        return self._global_step.eval(session=self.session)

    def get_critic_value(self, state, features):
        return self.session.run([self.ac_net.value], feed_dict={self.states: state})[0].flatten()

    def choose_action(self, state, features) -> dict:
        action, value = self.session.run(
            [self.ac_net.action, self.ac_net.value], feed_dict={self.states: [state]})
        return {"action": action, "value": value[0]}

    def get_env_action(self, action) -> int:
        return np.argmax(action)

    def learn(self):
        """Run learning algorithm"""
        self._initialize()
        config = self.config
        for _ in range(int(config["n_iter"])):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectory = self.env_runner.get_steps(int(self.config["n_local_steps"]))
            v = 0 if trajectory.terminals[-1] else self.get_critic_value(
                np.asarray(trajectory.states)[None, -1], trajectory.features[-1])
            rewards_plus_v = np.asarray(trajectory.rewards + [v])
            vpred_t = np.asarray(trajectory.values + [v])
            delta_t = trajectory.rewards + \
                self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
            batch_r = discount_rewards(
                rewards_plus_v, self.config["gamma"])[:-1]
            batch_adv = discount_rewards(delta_t, self.config["gamma"])
            fetches = [self.loss_summary_op, self.train_op, self._global_step]
            states = np.asarray(trajectory.states)
            feed_dict = {
                self.states: states,
                self.actions_taken: np.asarray(trajectory.actions),
                self.advantage: batch_adv,
                self.ret: np.asarray(batch_r)
            }
            feature = trajectory.features[0]
            if feature != [] and feature is not None:
                feed_dict[self.ac_net.rnn_state_in] = feature
            summary, _, global_step = self.session.run(fetches, feed_dict)
            self.writer.add_summary(summary, global_step)
            self.writer.flush()

        if self.config["save_model"]:
            tf.add_to_collection("action", self.action)
            tf.add_to_collection("states", self.states)
            self.saver.save(self.session, os.path.join(
                self.monitor_path, "model"))


class A2CDiscrete(A2C):
    def __init__(self, *args, **kwargs):
        super(A2CDiscrete, self).__init__(*args, **kwargs)

    def build_networks(self):
        self.ac_net = ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def make_loss(self):
        return actor_critic_discrete_loss(
            self.ac_net.logits,
            self.ac_net.probs,
            self.ac_net.value,
            self.ac_net.actions_taken,
            self.advantage,
            self.ret,
            self.config["vf_coef"],
            self.config["entropy_coef"],
            self.config["loss_reducer"]
        )


class A2CDiscreteCNN(A2CDiscrete):
    def build_networks(self):
        self.ac_net = ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class A2CDiscreteCNNRNN(A2CDiscrete):
    def build_networks(self):
        self.ac_net = ActorCriticNetworkDiscreteCNNRNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))
        self.initial_features = self.ac_net.state_init

    def choose_action(self, state, features) -> dict:
        """Choose an action."""
        feed_dict = {
            self.ac_net.states: [state],
            self.ac_net.rnn_state_in: features
        }

        action, rnn_state, value = self.session.run(
            [self.ac_net.action, self.ac_net.rnn_state_out, self.ac_net.value],
            feed_dict=feed_dict)
        return {"action": action, "value": value[0], "features": rnn_state}

    def get_critic_value(self, states, features):
        feed_dict = {
            self.ac_net.states: states,
            self.ac_net.rnn_state_in: features
        }
        return self.session.run(self.ac_net.value, feed_dict=feed_dict)[0]


class A2CContinuous(A2C):
    def __init__(self, *args, **kwargs):
        super(A2CContinuous, self).__init__(*args, **kwargs)

    def build_networks(self):
        self.ac_net = ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def make_loss(self):
        return actor_critic_continuous_loss(
            self.ac_net.action_log_prob,
            self.ac_net.entropy,
            self.ac_net.value,
            self.advantage,
            self.ret,
            self.config["vf_coef"],
            self.config["entropy_coef"],
            self.config["loss_reducer"]
        )

    def get_env_action(self, action):
        return action
