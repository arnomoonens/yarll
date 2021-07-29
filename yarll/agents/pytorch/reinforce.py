#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

from pathlib import Path
from time import time
from typing import Dict

import numpy as np
from gym import wrappers
import torch
from torch import nn
from torch import distributions as pyd
from torch.utils.tensorboard import SummaryWriter

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import discount_rewards
from yarll.misc.reporter import Reporter
from yarll.misc import summary_writer

def to_numpy(x):
    return x.cpu().detach().numpy()

def mlp(input_dim, output_dim, n_hidden_units, n_hidden_layers, output_mod=None):
    if n_hidden_layers == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, n_hidden_units), nn.ReLU(inplace=True)]
        for _ in range(n_hidden_layers - 1):
            mods += [nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(n_hidden_units, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class REINFORCE(Agent):
    """
    REINFORCE with baselines
    """

    def __init__(self, env, monitor_path: str, monitor: bool = False, video: bool = True, **usercfg) -> None:
        super().__init__(**usercfg)
        self.env = env
        if monitor:
            self.env = wrappers.Monitor(self.env,
                                        monitor_path,
                                        force=True,
                                        video_callable=(None if video else False))
        self.monitor_path = Path(monitor_path)
        # Default configuration. Can be overwritten using keyword arguments.
        self.config.update(dict(
            batch_update="timesteps",
            timesteps_per_batch=200,
            n_iter=100,
            gamma=0.99,
            learning_rate=0.05,
            n_hidden_layers=2,
            n_hidden_units=20,
            log_std_init=0.0,
            save_model=False
        ))
        self.config.update(usercfg)

        self.network = self.build_network()
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.config["learning_rate"])
        self.summary_writer = SummaryWriter(self.monitor_path)
        summary_writer.set(self.summary_writer)

    def build_network(self):
        raise NotImplementedError()

    def choose_action(self, state, features) -> Dict[str, np.ndarray]:
        """Choose an action."""
        raise NotImplementedError()

    def train(self, states, actions_taken, advantages, features=None):
        """Train the policy network."""
        raise NotImplementedError()

    def learn(self):
        """Run learning algorithm"""
        env_runner = EnvRunner(self.env, self, self.config,
                               summaries_every_episodes=self.config.get("env_summaries_every_episodes", None),
                               transition_preprocessor=self.config.get("transition_preprocessor", None),
                               )
        reporter = Reporter()
        config = self.config
        total_n_trajectories = 0
        summary_writer.start()
        for iteration in range(config["n_iter"]):
            start_time = time()
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = env_runner.get_trajectories()
            total_n_trajectories += len(trajectories)
            all_state = np.concatenate([trajectory.states for trajectory in trajectories])
            # Compute discounted sums of rewards
            rets = [discount_rewards(trajectory.rewards, config["gamma"]) for trajectory in trajectories]
            max_len = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(max_len - len(ret))]) for ret in rets]
            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action = np.concatenate([trajectory.actions for trajectory in trajectories])
            all_adv = np.concatenate(advs)
            # Do policy gradient update step
            episode_rewards = np.array([sum(trajectory.rewards)
                                        for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory.rewards) for trajectory in trajectories])  # episode lengths
            loss, log_probs = self.train(all_state, all_action, all_adv)
            summary_writer.add_scalar("model/loss", loss, iteration)
            summary_writer.add_scalar("model/mean_advantage", np.mean(all_adv), iteration)
            summary_writer.add_scalar("model/mean_log_prob", np.mean(log_probs), iteration)
            summary_writer.add_scalar("model/mean_log_std",
                                      self.network.log_std.mean().cpu().detach().numpy(), iteration)
            summary_writer.add_scalar("model/mean_action", np.mean(all_action), iteration)
            summary_writer.add_scalar("diagnostics/iteration_duration_seconds", time() - start_time, iteration)

            reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, total_n_trajectories)
        summary_writer.stop()
        if self.config["save_model"]:
            torch.save(self.network.state_dict(), self.monitor_path / "actor_final.pt")

# TODO: Choose activation function
# TODO: Use SquashedNormal distribution
class ActorContinuous(nn.Module):
    def __init__(self, obs_dim, n_actions, n_hidden_layers, n_hidden_units, log_std_init: float = 0.0):
        super().__init__()
        self.trunk = mlp(obs_dim, n_actions, n_hidden_units, n_hidden_layers)
        self.log_std = nn.Parameter(torch.full((n_actions,), log_std_init))

    def forward(self, obs):
        mean = self.trunk(obs)

        std = torch.exp(self.log_std)
        return pyd.Normal(mean, std)

class REINFORCEContinuous(REINFORCE):
    def __init__(self, env, monitor_path, log_std_init: float = 0.0, video: bool = True, **usercfg):
        super().__init__(env, monitor_path, log_std_init=log_std_init, video=video, **usercfg)

        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def choose_action(self, state, features):
        """Choose an action."""
        dist = self.network(torch.FloatTensor(state[None, :]))
        action = dist.sample()[0]
        return {"action": to_numpy(action)}

    def get_env_action(self, action):
        """
        Converts an action from self.choose_action to an action to be given to the environment.
        """
        action = np.clip(action, -1., 1.)
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def build_network(self):
        return ActorContinuous(self.env.observation_space.shape[0],
                               self.env.action_space.shape[0],
                               self.config["n_hidden_layers"],
                               self.config["n_hidden_units"],
                               self.config["log_std_init"],
                               )

    def train(self, states, actions_taken, advantages, features=None):
        states = torch.as_tensor(states)
        actions_taken = torch.as_tensor(actions_taken)
        advantages = torch.as_tensor(advantages)
        dist = self.network(states)
        log_probs = dist.log_prob(actions_taken).sum(-1)
        losses = -log_probs * advantages
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return to_numpy(loss), to_numpy(log_probs)
