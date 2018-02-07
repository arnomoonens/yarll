# -*- coding: utf8 -*-

"""
Asynchronous Advantage Actor Critic (A3C)
Based on:
- Pseudo code from Asynchronous Methods for Deep Reinforcement Learning
- Tensorflow code from https://github.com/yao62995/A3C/blob/master/A3C_atari.py and
  https://github.com/openai/universe-starter-agent/tree/f16f37d9d3bc8146cf68a75557e1ba89824b7e54
"""

import os
import numpy as np
import tensorflow as tf
import logging
import threading
import multiprocessing
import signal
import sys
import queue
import time

from gym import wrappers

from environment.registration import make
from agents.agent import Agent
from misc.utils import discount_rewards, FastSaver
from agents.actor_critic import ActorCriticNetworkDiscrete, ActorCriticNetworkDiscreteCNN, ActorCriticNetworkDiscreteCNNRNN, ActorCriticDiscreteLoss, ActorCriticNetworkContinuous, ActorCriticContinuousLoss
from agents.env_runner import Trajectory

logging.getLogger().setLevel("INFO")

def env_runner(env, policy, n_steps, render=False, summary_writer=None):
    """
    Run agent-environment loop for maximally n_steps.
    Yield dictionary of results.
    """
    episode_steps = 0
    episode_reward = 0
    state = env.reset()
    features = policy.initial_features
    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    while True:
        trajectory = Trajectory()

        for _ in range(n_steps):
            fetched = policy.choose_action(state, features)  # Predict the next action (using a neural network) depending on the current state
            action = fetched[0]
            value = fetched[1]
            new_features = fetched[2:]
            new_state, reward, terminal, _ = env.step(policy.get_env_action(action))
            episode_steps += 1
            episode_reward += reward
            trajectory.add(state, action, reward, value, features, terminal)
            state = new_state
            features = new_features
            if terminal or episode_steps >= timestep_limit:
                features = policy.initial_features
                if episode_steps >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    state = env.reset()
                if summary_writer is not None:
                    summary = tf.Summary()
                    summary.value.add(tag="global/Episode_length", simple_value=float(episode_steps))
                    summary.value.add(tag="global/Reward", simple_value=float(episode_reward))
                    summary_writer.add_summary(summary, policy.global_step.eval())
                    summary_writer.flush()
                episode_steps = 0
                episode_reward = 0
                break
            if render:
                env.render()
        yield trajectory

class RunnerThread(threading.Thread):
    """
    Thread that collects trajectories from the environment
    and puts them on a queue.
    """
    def __init__(self, env, policy, n_local_steps, render=False):
        threading.Thread.__init__(self)
        self.env = env
        self.policy = policy
        self.n_local_steps = n_local_steps
        self.render = render
        self.daemon = True
        self.queue = queue.Queue(maxsize=5)

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        trajectory_provider = env_runner(self.env, self.policy, self.n_local_steps, self.render, self.summary_writer)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.
            self.queue.put(next(trajectory_provider), timeout=600.0)

class A3CTask(threading.Thread):
    """Single A3C learner thread."""
    def __init__(self, master, task_id, cluster, clip_gradients=True):
        super(A3CTask, self).__init__(name=task_id)
        self.master = master
        self.config = master.config
        self.task_id = task_id
        self.clip_gradients = clip_gradients
        self.env = make(master.env_name)
        self.env.seed(self.config["seed"])
        if task_id == 0 and self.master.monitor:
            self.env = wrappers.Monitor(
                self.env,
                master.monitor_path,
                force=True,
                video_callable=(None if self.master.video else False)
            )

        # Only used (and overwritten) by agents that use an RNN
        self.initial_features = None

        # Build actor and critic networks
        self.graph = tf.Graph()
        with self.graph.as_default():
            worker_device = "/job:worker/task:{}/cpu:0".format(task_id)
            # Global network
            shared_device = tf.train.replica_device_setter(1, worker_device=worker_device)
            with tf.device(shared_device):
                with tf.variable_scope("global"):
                    self.global_network = self.build_networks()
                    self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                    self.global_step = tf.get_variable(
                        "global_step", [],
                        tf.int32,
                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                        trainable=False)

            # Local network
            with tf.device(worker_device):
                with tf.variable_scope("local"):
                    self.local_network = self.build_networks()
                    self.actor_loss, self.critic_loss, self.loss = self.make_loss(
                        self.local_network,
                        self.config["entropy_coef"],
                        self.config["loss_reducer"])
                    self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                    self.sync_net = self.create_sync_net_op()
                    self.n_steps = tf.shape(self.local_network.states)[0]
                    inc_step = self.global_step.assign_add(self.n_steps)

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
            self.writer = tf.summary.FileWriter(os.path.join(self.master.monitor_path, "task{}".format(self.task_id)))

            self.runner = RunnerThread(self.env, self, self.config["n_local_steps"], task_id == 0 and self.master.video)

            self.server = tf.train.Server(
                cluster,
                job_name="worker",
                task_index=task_id,
                config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))

            def init_fn(sess):
                sess.run(init_all_op)

            self.sv = tf.train.Supervisor(
                graph=self.graph,
                is_chief=(task_id == 0),
                logdir=self.master.monitor_path,
                summary_op=None,
                ready_op=tf.report_uninitialized_variables(variables_to_save),
                saver=saver,
                init_op=init_op,
                init_fn=init_fn,
                summary_writer=self.writer,
                global_step=self.global_step,
                save_model_secs=0 if not self.config["save_model"] else 300,
                save_summaries_secs=30
            )

            self.config_proto = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(task_id)])

    def get_critic_value(self, states, *rest):
        return tf.get_default_session().run(self.local_network.value, feed_dict={self.states: states})[0]

    def get_env_action(self, action):
        return np.argmax(action)

    def choose_action(self, state, *rest):
        """Choose an action."""
        feed_dict = {
            self.states: [state]
        }
        action, value = tf.get_default_session().run(
            [self.local_network.action, self.local_network.value],
            feed_dict=feed_dict)
        return action, value, []

    def create_sync_net_op(self):
        return tf.group(*[v1.assign(v2) for v1, v2 in zip(self.local_vars, self.global_vars)])

    def make_trainer(self):
        optimizer = tf.train.AdamOptimizer(self.config["learning_rate"], name="optim")
        grads = tf.gradients(self.loss, self.local_vars)
        grads, _ = tf.clip_by_global_norm(grads, self.config["gradient_clip_value"])

        # Apply gradients to the weights of the master network
        return optimizer.apply_gradients(
            zip(grads, self.global_vars))

    def create_summary_losses(self):
        n_steps = tf.to_float(self.n_steps)
        actor_loss_summary = tf.summary.scalar("Actor_loss", self.actor_loss / n_steps)
        critic_loss_summary = tf.summary.scalar("Critic_loss", self.critic_loss / n_steps)
        loss_summary = tf.summary.scalar("loss", self.loss / n_steps)
        return [actor_loss_summary, critic_loss_summary, loss_summary]

    def pull_batch_from_queue(self):
        """
        Take a trajectory from the queue.
        Also immediately try to extend it if the episode
        wasn't over and more transitions are available
        """
        trajectory = self.runner.queue.get(timeout=600.0)
        while not trajectory.terminal:
            try:
                trajectory.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return trajectory

    def run(self):
        # Assume global shared parameter vectors θ and θv and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'v
        with self.sv.managed_session(self.server.target, config=self.config_proto) as sess, sess.as_default():
            sess.run(self.sync_net)
            self.runner.start_runner(sess, self.writer)
            while not self.sv.should_stop() and self.master.T < self.config["T_max"]:
                # Synchronize thread-specific parameters θ' = θ and θ'v = θv
                sess.run(self.sync_net)
                trajectory = self.pull_batch_from_queue()
                v = 0 if trajectory.terminal else self.get_critic_value(np.asarray(trajectory.states)[None, -1], trajectory.features[-1][0])
                rewards_plus_v = np.asarray(trajectory.rewards + [v])
                vpred_t = np.asarray(trajectory.values + [v])
                delta_t = trajectory.rewards + self.config["gamma"] * vpred_t[1:] - vpred_t[:-1]
                batch_r = discount_rewards(rewards_plus_v, self.config["gamma"])[:-1]
                batch_adv = discount_rewards(delta_t, self.config["gamma"])
                fetches = [self.summary_op, self.train_op, self.global_step]
                states = np.asarray(trajectory.states)
                feed_dict = {
                    self.states: states,
                    self.actions_taken: np.asarray(trajectory.actions),
                    self.adv: batch_adv,
                    self.r: np.asarray(batch_r)
                }
                feature = trajectory.features[0][0]
                if feature != []:
                    feed_dict[self.local_network.rnn_state_in] = feature
                summary, _, global_step = sess.run(fetches, feed_dict)
                self.writer.add_summary(summary, global_step)
                self.writer.flush()
                self.master.T += trajectory.steps
        self.sv.stop()
        print("stopped")

class A3CTaskDiscrete(A3CTask):
    """A3CTask for a discrete action space."""
    def __init__(self, master, task_id, cluster):
        self.make_loss = ActorCriticDiscreteLoss
        super(A3CTaskDiscrete, self).__init__(master, task_id, cluster)

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"])

        self.states = ac_net.states
        self.actions_taken = ac_net.actions_taken
        self.adv = ac_net.adv
        self.r = ac_net.r
        return ac_net

class A3CTaskDiscreteCNN(A3CTaskDiscrete):
    """A3CTask for a discrete action space."""
    def __init__(self, master, task_id, cluster):
        self.make_loss = ActorCriticDiscreteLoss
        super(A3CTaskDiscreteCNN, self).__init__(master, task_id, cluster)

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"],
            summary=False)
        self.states = ac_net.states
        self.actions_taken = ac_net.actions_taken
        self.adv = ac_net.adv
        self.r = ac_net.r
        return ac_net

class A3CTaskDiscreteCNNRNN(A3CTaskDiscreteCNN):
    """A3CTask for a discrete action space."""
    def __init__(self, master, task_id, cluster):
        self.make_loss = ActorCriticDiscreteLoss
        super(A3CTaskDiscreteCNNRNN, self).__init__(master, task_id, cluster)

    def build_networks(self):
        ac_net = ActorCriticNetworkDiscreteCNNRNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden"],
            summary=False)
        self.states = ac_net.states
        self.actions_taken = ac_net.actions_taken
        self.adv = ac_net.adv
        self.r = ac_net.r
        self.initial_features = ac_net.state_init
        return ac_net

    def choose_action(self, state, features):
        feed_dict = {
            self.local_network.states: [state]
        }

        feed_dict[self.local_network.rnn_state_in] = features
        action, rnn_state, value = tf.get_default_session().run(
            [self.local_network.action, self.local_network.rnn_state_out, self.local_network.value],
            feed_dict=feed_dict)
        return action, value, rnn_state

    def get_critic_value(self, states, features):
        feed_dict = {
            self.local_network.states: states
        }
        feed_dict[self.local_network.rnn_state_in] = features
        return tf.get_default_session().run(self.local_network.value, feed_dict=feed_dict)[0]

class A3CTaskContinuous(A3CTask):
    """A3CTask for a continuous action space."""
    def __init__(self, master, task_id, cluster):
        self.make_loss = ActorCriticContinuousLoss
        super(A3CTaskContinuous, self).__init__(master, task_id, cluster)

    def build_networks(self):
        ac_net = ActorCriticNetworkContinuous(
            self.env.action_space,
            list(self.env.observation_space.shape),
            self.config["n_hidden"])
        self.states = ac_net.states
        self.actions_taken = ac_net.actions_taken
        self.adv = ac_net.adv
        self.r = ac_net.r
        return ac_net

    def get_env_action(self, action):
        return action

class PSProcess(multiprocessing.Process):
    """Parameter server"""
    def __init__(self, task_id, cluster):
        super(PSProcess, self).__init__()
        self.server = tf.train.Server(
            cluster,
            job_name="ps",
            task_index=task_id,
            config=tf.ConfigProto(device_filters=["/job:ps"]))

    def run(self):
        while True:
            time.sleep(1000)

def cluster_spec(num_workers, num_ps):
    """
    More tensorflow setup for data parallelism
    """
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

class A3C(Agent):
    """Asynchronous Advantage Actor Critic learner."""
    def __init__(self, env, monitor, monitor_path, video=True, **usercfg):
        super(A3C, self).__init__(**usercfg)
        self.env = env
        self.T = 0
        self.env_name = env.spec.id
        self.monitor = monitor
        self.monitor_path = monitor_path
        self.video = video

        self.config.update(dict(
            gamma=0.99,  # Discount past rewards by a percentage
            learning_rate=1e-4,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-4,
            n_hidden=20,
            critic_n_hidden=20,
            gradient_clip_value=50,
            n_tasks=multiprocessing.cpu_count(),  # Use as much tasks as there are cores on the current system
            T_max=8e5,
            shared_optimizer=False,
            episode_max_length=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"),
            repeat_n_actions=1,
            n_local_steps=20,
            entropy_coef=0.01,
            loss_reducer="sum",  # use tf.reduce_sum or tf.reduce_mean for the loss
            save_model=False
        ))
        self.config.update(usercfg)

        spec = cluster_spec(self.config["n_tasks"], 1)
        cluster = tf.train.ClusterSpec(spec).as_cluster_def()

        self.jobs = []
        for task_id in range(self.config["n_tasks"]):
            job = self.make_thread(task_id, cluster)
            self.jobs.append(job)

        self.ps = PSProcess(0, cluster)
        self.ps.daemon = True

    def make_thread(self, task_id, cluster):
        return self.thread_type(self, task_id, cluster)

    def signal_handler(self, signal, frame):
        logging.info("SIGINT signal received: Requesting a stop...")
        sys.exit(128 + signal)

    def learn(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGHUP, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.train_step = 0
        self.ps.start()
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
        self.ps.terminate()

class A3CDiscrete(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CTaskDiscrete
        super(A3CDiscrete, self).__init__(env, monitor, monitor_path, **usercfg)

class A3CDiscreteCNN(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CTaskDiscreteCNN
        super(A3CDiscreteCNN, self).__init__(env, monitor, monitor_path, **usercfg)

class A3CDiscreteCNNRNN(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CTaskDiscreteCNNRNN
        super(A3CDiscreteCNNRNN, self).__init__(env, monitor, monitor_path, **usercfg)
        self.config["RNN"] = True

class A3CContinuous(A3C):
    """A3C for a continuous action space"""
    def __init__(self, env, monitor, monitor_path, **usercfg):
        self.thread_type = A3CTaskContinuous
        super(A3CContinuous, self).__init__(env, monitor, monitor_path, **usercfg)
