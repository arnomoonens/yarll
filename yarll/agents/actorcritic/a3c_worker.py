#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import queue
import threading
import argparse
import time
from typing import Optional
import tensorflow as tf
import numpy as np
from gym import wrappers

from yarll.environment.registration import make
from yarll.misc.network_ops import create_sync_net_op
from yarll.misc.utils import discount_rewards, FastSaver, load, json_to_dict, cluster_spec
from yarll.agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete, ActorCriticNetworkDiscreteCNN, \
ActorCriticNetworkDiscreteCNNRNN, actor_critic_discrete_loss, ActorCriticNetworkContinuous, actor_critic_continuous_loss
from yarll.memory.experiences_memory import ExperiencesMemory

def env_runner(env, policy, n_steps: int, render=False, summary_writer=None):
    """
    Run agent-environment loop for maximally n_steps.
    Yields a dictionary of results.
    """
    episode_steps = 0
    episode_reward = 0
    n_episodes = 0
    state = env.reset()
    features = policy.initial_features
    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    while True:
        memory = ExperiencesMemory()

        for _ in range(n_steps):
            # Choose the next action (using a neural network) depending on the current state
            results = policy.choose_action(state, features)
            action = results["action"]
            value = results.get("value", None)
            new_features = results.get("features", None)
            # Execute the action in the environment
            new_state, reward, terminal, _ = env.step(policy.get_env_action(action))
            episode_steps += 1
            episode_reward += reward
            memory.add(state, action, reward, value, features, terminal)
            state = new_state
            features = new_features
            if terminal or episode_steps >= timestep_limit:
                n_episodes += 1
                features = policy.initial_features
                if episode_steps >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    state = env.reset()
                if summary_writer is not None:
                    summary = tf.Summary()
                    summary.value.add(tag="global/Episode_length", simple_value=float(episode_steps))
                    summary.value.add(tag="global/Reward", simple_value=float(episode_reward))
                    summary_writer.add_summary(summary, n_episodes)
                    summary_writer.flush()
                episode_steps = 0
                episode_reward = 0
                break
            if render:
                env.render()
        yield memory


class RunnerThread(threading.Thread):
    """
    Thread that collects trajectories from the environment
    and puts them on a queue.
    """

    def __init__(self, env, policy, n_local_steps: int, render=False) -> None:
        super(RunnerThread, self).__init__()
        self.env = env
        self.policy = policy
        self.n_local_steps = n_local_steps
        self.render = render
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.queue: queue.Queue = queue.Queue(maxsize=5)

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        trajectory_provider = env_runner(self.env, self.policy, self.n_local_steps, self.render, self.summary_writer)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.
            # This is an empirical observation.
            self.queue.put(next(trajectory_provider), timeout=600.0)


class A3CTask(object):
    """Single A3C learner thread."""

    def __init__(self,
                 env_id: str,
                 task_id: int,
                 cluster: tf.train.ClusterDef,
                 monitor_path: str,
                 config: dict,
                 clip_gradients: bool = True,
                 video: bool = False,
                 seed: Optional[int] = None) -> None:
        super(A3CTask, self).__init__()
        self.task_id = task_id
        self.config = config
        self.clip_gradients = clip_gradients
        self.env = make(env_id)
        self.env.seed(seed)
        if task_id == 0:
            self.env = wrappers.Monitor(
                self.env,
                monitor_path,
                force=True,
                video_callable=(None if video else False)
            )

        # Only used (and overwritten) by agents that use an RNN
        self.initial_features = None

        worker_device = "/job:worker/task:{}/cpu:0".format(task_id)
        # Global network
        shared_device = tf.train.replica_device_setter(
            ps_tasks=1,
            worker_device=worker_device,
            cluster=cluster)
        with tf.device(shared_device):
            with tf.variable_scope("global"):
                self.global_network = self.build_networks()
                self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                self._global_step = tf.get_variable(
                    "global_step",
                    [],
                    tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)

        # Local network
        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = self.build_networks()
                self.states = self.local_network.states
                self.actions_taken = self.local_network.actions_taken
                self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
                self.ret = tf.placeholder(tf.float32, [None], name="return")
                self.actor_loss, self.critic_loss, self.loss = self.make_loss()
                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                self.sync_net = create_sync_net_op(self.global_vars, self.local_vars)
                self.n_steps = tf.shape(self.local_network.states)[0]
                inc_step = self._global_step.assign_add(self.n_steps)

        device = shared_device if self.config["shared_optimizer"] else worker_device
        with tf.device(device):
            apply_optim_op = self.make_trainer()
            self.train_op = tf.group(apply_optim_op, inc_step)

            loss_summaries = self.create_summary_losses()
            self.reward = tf.placeholder("float", name="reward")
            tf.summary.scalar("Reward", self.reward)
            self.episode_length = tf.placeholder("float", name="episode_length")
            tf.summary.scalar("Episode_length", self.episode_length)
            self.summary_op = tf.summary.merge(loss_summaries)

        variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
        saver = FastSaver(variables_to_save)
        # Write the summary of each task in a different directory
        self.writer = tf.summary.FileWriter(os.path.join(monitor_path, "task{}".format(task_id)))

        self.runner = RunnerThread(self.env, self, int(self.config["n_local_steps"]), task_id == 0 and video)

        self.server = tf.train.Server(
            cluster,
            job_name="worker",
            task_index=task_id,
            config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2)
        )

        def init_fn(scaffold, sess):
            sess.run(init_all_op)

        self.report_uninit_op = tf.report_uninitialized_variables(variables_to_save)

        self.scaffold = tf.train.Scaffold(
            init_op=init_op,
            init_fn=init_fn,
            ready_for_local_init_op=self.report_uninit_op,
            saver=saver,
            ready_op=self.report_uninit_op
        )

        self.config_proto = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(task_id)])

        self.session = None

    def build_networks(self):
        raise NotImplementedError()

    def make_loss(self):
        raise NotImplementedError()

    def get_critic_value(self, states, features):
        return self.session.run(self.local_network.value, feed_dict={self.states: states})[0]

    def get_env_action(self, action):
        return np.argmax(action)

    def choose_action(self, state, features):
        """Choose an action."""
        feed_dict = {
            self.states: [state]
        }
        action, value = self.session.run([self.local_network.action, self.local_network.value],
                                         feed_dict=feed_dict)
        return {"action": action, "value": value}

    def make_trainer(self):
        optimizer = tf.train.AdamOptimizer(self.config["learning_rate"], name="optim")
        grads = tf.gradients(self.loss, self.local_vars)
        grads, _ = tf.clip_by_global_norm(grads, self.config["gradient_clip_value"])

        # Apply gradients to the weights of the master network
        return optimizer.apply_gradients(zip(grads, self.global_vars))

    def create_summary_losses(self):
        n_steps = tf.to_float(self.n_steps)
        actor_loss_summary = tf.summary.scalar("model/actor_loss", tf.squeeze(self.actor_loss / n_steps))
        critic_loss_summary = tf.summary.scalar("model/critic_loss", tf.squeeze(self.critic_loss / n_steps))
        loss_summary = tf.summary.scalar("model/loss", tf.squeeze(self.loss / n_steps))
        return [actor_loss_summary, critic_loss_summary, loss_summary]

    def pull_batch_from_queue(self):
        """
        Take a trajectory from the queue.
        Also immediately try to extend it if the episode
        wasn't over and more transitions are available.
        """
        trajectory = self.runner.queue.get(timeout=600.0)
        while not trajectory.terminal:
            try:
                trajectory.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return trajectory

    @property
    def global_step(self):
        return self._global_step.eval(session=self.session)

    def learn(self):
        # Assume global shared parameter vectors θ and θv and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'v
        if self.task_id != 0:
            time.sleep(5)
        with tf.train.MonitoredTrainingSession(
            master=self.server.target,
            is_chief=(self.task_id == 0),
            config=self.config_proto,
            save_summaries_secs=30,
            scaffold=self.scaffold
        ) as sess:
            self.session = sess
            sess.run(self.sync_net)
            self.runner.start_runner(sess, self.writer)
            while not sess.should_stop() and self.global_step < self.config["T_max"]:
                # Synchronize thread-specific parameters θ' = θ and θ'v = θv
                sess.run(self.sync_net)
                trajectory = self.pull_batch_from_queue()
                v = 0 if trajectory.terminal else self.get_critic_value(
                    np.asarray(trajectory.states)[None, -1], trajectory.features[-1])
                rewards_plus_v = np.asarray(trajectory.rewards + [v])
                vpred_t = np.asarray(trajectory.values + [v])
                delta_t = trajectory.rewards + self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
                batch_r = discount_rewards(rewards_plus_v, self.config["gamma"])[:-1]
                batch_adv = discount_rewards(delta_t, self.config["gamma"])
                fetches = [self.summary_op, self.train_op, self._global_step]
                states = np.asarray(trajectory.states)
                feed_dict = {
                    self.states: states,
                    self.actions_taken: np.asarray(trajectory.actions),
                    self.advantage: batch_adv,
                    self.ret: np.asarray(batch_r)
                }
                feature = trajectory.features[0]
                if feature != [] and feature is not None:
                    feed_dict[self.local_network.rnn_state_in] = feature
                summary, _, global_step = sess.run(fetches, feed_dict)
                self.writer.add_summary(summary, global_step)
                self.writer.flush()


class A3CTaskDiscrete(A3CTask):
    """A3CTask for a discrete action space."""

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))
        return ac_net

    def make_loss(self):
        return actor_critic_discrete_loss(
            self.local_network.logits,
            self.local_network.probs,
            self.local_network.value,
            self.local_network.actions_taken,
            self.advantage,
            self.ret,
            self.config["vf_coef"],
            self.config["entropy_coef"],
            self.config["loss_reducer"]
        )


class A3CTaskDiscreteCNN(A3CTaskDiscrete):
    """A3CTask for a discrete action space."""

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            summary=False)
        return ac_net


class A3CTaskDiscreteCNNRNN(A3CTaskDiscreteCNN):
    """A3CTask for a discrete action space."""

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscreteCNNRNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            summary=False)
        self.initial_features = ac_net.state_init
        return ac_net

    def choose_action(self, state, features):
        feed_dict = {
            self.local_network.states: [state],
            self.local_network.rnn_state_in: features
        }
        action, rnn_state, value = self.session.run(
            [self.local_network.action, self.local_network.rnn_state_out, self.local_network.value],
            feed_dict=feed_dict)
        return {"action": action, "value": value, "features": rnn_state}

    def get_critic_value(self, states, features):
        feed_dict = {
            self.local_network.states: states,
            self.local_network.rnn_state_in: features
        }
        return self.session.run(self.local_network.value, feed_dict=feed_dict)[0]


class A3CTaskContinuous(A3CTask):
    """A3CTask for a continuous action space."""

    def build_networks(self):
        ac_net = ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))
        return ac_net

    def make_loss(self):
        return actor_critic_continuous_loss(
            self.local_network.action_log_prob,
            self.local_network.entropy,
            self.local_network.value,
            self.advantage,
            self.ret,
            self.config["vf_coef"],
            self.config["entropy_coef"],
            self.config["loss_reducer"]
        )

    def get_env_action(self, action):
        return action


parser = argparse.ArgumentParser()

parser.add_argument("env_id", type=str, help="Name of the environment on which to run")
parser.add_argument("cls", type=str, help="Which class to use for the task.")
parser.add_argument("task_id", type=int, help="Task index.")
parser.add_argument("n_tasks", type=int, help="Total number of tasks in this experiment.")
parser.add_argument("config", type=str, help="Path to config file")
parser.add_argument("-seed", type=int, default=None, help="Seed to use for the environment.")
parser.add_argument("--monitor_path", type=str, help="Path where to save monitor files.")
parser.add_argument("--video", default=False, action="store_true", help="Generate video.")

def main():
    args = parser.parse_args()
    spec = cluster_spec(args.n_tasks, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()
    cls = load("agents.actorcritic.a3c_worker:" + args.cls)
    config = json_to_dict(args.config)
    task = cls(args.env_id, args.task_id, cluster, args.monitor_path, config, video=args.video)
    task.learn()

if __name__ == '__main__':
    main()
