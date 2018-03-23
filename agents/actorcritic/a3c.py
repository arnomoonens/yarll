# -*- coding: utf8 -*-

"""
Asynchronous Advantage Actor Critic (A3C)
Most of the work is done in `a3c_worker.py`.
Based on:
- Pseudo code from Asynchronous Methods for Deep Reinforcement Learning
- Tensorflow code from https://github.com/yao62995/A3C/blob/master/A3C_atari.py and
  https://github.com/openai/universe-starter-agent/tree/f16f37d9d3bc8146cf68a75557e1ba89824b7e54
"""

import logging
import multiprocessing
import subprocess
import signal
import sys
import os
from typing import Union
from six.moves import shlex_quote

from agents.agent import Agent

logging.getLogger().setLevel("INFO")

class A3C(Agent):
    """Asynchronous Advantage Actor Critic learner."""

    def __init__(self, env, monitor, monitor_path: str, video: bool = True, **usercfg) -> None:
        super(A3C, self).__init__(**usercfg)
        self.env = env
        self.env_name = env.spec.id
        self.monitor = monitor
        self.monitor_path = monitor_path
        self.video = video
        self.task_type: Union[None, str] = None # To be filled in by subclass

        self.config.update(dict(
            gamma=0.99,  # Discount past rewards by a percentage
            learning_rate=1e-4,
            n_hidden_units=20,
            n_hidden_layers=1,
            gradient_clip_value=50,
            n_tasks=multiprocessing.cpu_count(),  # Use as much tasks as there are cores on the current system
            T_max=8e5,
            shared_optimizer=False,
            episode_max_length=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"),
            n_local_steps=20,
            vf_coef=0.5,
            entropy_coef=0.01,
            loss_reducer="sum",  # use tf.reduce_sum or tf.reduce_mean for the loss
            save_model=False
        ))
        self.config.update(usercfg)

    def signal_handler(self, received_signal: int, frame):
        logging.info("SIGINT signal received: Requesting a stop...")
        sys.exit(128 + received_signal)

    def learn(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGHUP, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        current_folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        cmd = [
            sys.executable,
            os.path.join(current_folder, "parameter_server.py"),
            self.config["n_tasks"]]
        processed_cmd = " ".join(shlex_quote(str(x)) for x in cmd)
        ps_process = subprocess.Popen(processed_cmd, shell=True)
        worker_processes = []
        for task_id in range(self.config["n_tasks"]):
            cmd = [
                sys.executable,
                os.path.join(current_folder, "a3c_worker.py"),
                self.env_name,
                self.task_type,
                task_id,
                self.config["n_tasks"],
                self.config["config_path"],
                "--monitor_path", self.monitor_path
            ]
            processed_cmd = " ".join(shlex_quote(str(x)) for x in cmd)
            p = subprocess.Popen(processed_cmd, shell=True)
            worker_processes.append(p)
        for p in worker_processes:
            p.wait()
        ps_process.terminate()

class A3CDiscrete(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path: str, **usercfg) -> None:
        super(A3CDiscrete, self).__init__(env, monitor, monitor_path, **usercfg)
        self.task_type = "A3CTaskDiscrete"

class A3CDiscreteCNN(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path: str, **usercfg) -> None:
        super(A3CDiscreteCNN, self).__init__(env, monitor, monitor_path, **usercfg)
        self.task_type = "A3CTaskDiscreteCNN"

class A3CDiscreteCNNRNN(A3C):
    """A3C for a discrete action space"""
    def __init__(self, env, monitor, monitor_path: str, **usercfg) -> None:
        super(A3CDiscreteCNNRNN, self).__init__(env, monitor, monitor_path, **usercfg)
        self.task_type = "A3CTaskDiscreteCNNRNN"
        self.config["RNN"] = True

class A3CContinuous(A3C):
    """A3C for a continuous action space"""
    def __init__(self, env, monitor, monitor_path: str, **usercfg) -> None:
        super(A3CContinuous, self).__init__(env, monitor, monitor_path, **usercfg)
        self.task_type = "A3CTaskContinuous"
