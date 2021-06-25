"""
Soft Actor-Critic

Based on SAC implementation from https://github.com/denisyarats/pytorch_sac
License:

MIT License

Copyright (c) 2019 Denis Yarats

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

and https://github.com/rail-berkeley/softlearning
Licence:
MIT License

Copyright (c) 2018 Softlearning authors and contributors

Softlearning uses a shared copyright model: each contributor holds copyright over
their contributions to Softlearning. The project versioning records all such
contribution and copyright details.

By contributing to the Softlearning repository through pull-request, comment,
or otherwise, the contributor releases their content to the license and
copyright terms herein.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from copy import deepcopy
import csv
from pathlib import Path
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.utils.tensorboard import SummaryWriter

from yarll.agents.agent import Agent
from yarll.agents.env_runner import EnvRunner
from yarll.memory.prealloc_memory import PreAllocMemory
from yarll.misc import summary_writer

# TODO: put this in separate file
class DeterministicPolicy:
    def __init__(self, env, policy_fn):
        self.env = env
        self.policy_fn = policy_fn
        self.initial_features = None

        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def choose_action(self, state, features):
        res = self.policy_fn(state[None, :])[0]
        return {"action": res}

    def get_env_action(self, action):
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def new_trajectory(self):
        pass

def hard_update(source_network, target_network):
    target_network.load_state_dict(source_network.state_dict())

def soft_update(source_network, target_network, tau):
    for source_param, target_param in zip(source_network.parameters(), target_network.parameters()):
        target_param.data.copy_(tau * source_param.data +
                                (1 - tau) * target_param.data)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def to_numpy(x):
    return x.cpu().detach().numpy()

class SAC(Agent):
    def __init__(self, env, monitor_path: Union[Path, str], **usercfg) -> None:
        super().__init__(**usercfg)
        self.env = env
        self.monitor_path = Path(monitor_path)
        self.monitor_path.mkdir(parents=True, exist_ok=True)

        self.config.update(
            max_steps=100000,
            actor_learning_rate=3e-4,
            softq_learning_rate=3e-4,
            alpha_learning_rate=1e-4,
            reward_scale=1.0,
            n_hidden_layers=2,
            n_hidden_units=256,
            gamma=0.99,
            batch_size=256,
            tau=0.005,
            init_log_alpha=0.1,
            actor_update_frequency=1,
            critic_target_update_frequency=2,
            target_entropy=None,
            logprob_epsilon=1e-6,  # For numerical stability when computing log
            n_train_steps=1,  # Number of parameter update steps per iteration
            replay_buffer_size=1e6,
            replay_start_size=256,  # Required number of replay buffer entries to start training
            gradient_clip_value=1.0,
            hidden_layer_activation="relu",
            device="cpu",
            normalize_inputs=False, # TODO: handle this
            summaries=True,
            checkpoints=True,
            checkpoint_every_episodes=10,
            checkpoints_max_to_keep=None,
            save_model=True,
            test_frequency=0,
            n_test_episodes=5,
            write_train_rewards=False
        )
        self.config.update(usercfg)

        self.state_shape: list = list(env.observation_space.shape)
        self.n_actions: int = env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.device = torch.device(self.config["device"])

        self.target_entropy = self.config["target_entropy"]
        if self.target_entropy is None:
            self.target_entropy = -np.prod(env.action_space.shape)

        # Make networks
        # action_output are the squashed actions and action_original those straight from the normal distribution
        input_dim = self.state_shape[0]
        self.actor_network = ActorNetwork(input_dim,
                                          self.n_actions,
                                          self.config["n_hidden_units"],
                                          self.config["n_hidden_layers"]).to(self.device)
        self.softq_networks = DoubleQCriticNetwork(input_dim,
                                                   self.n_actions,
                                                   self.config["n_hidden_units"],
                                                   self.config["n_hidden_layers"]).to(self.device)
        self.target_softq_networks = DoubleQCriticNetwork(input_dim,
                                                          self.n_actions,
                                                          self.config["n_hidden_units"],
                                                          self.config["n_hidden_layers"]).to(self.device)
        hard_update(self.softq_networks, self.target_softq_networks)

        self.log_alpha = torch.tensor(self.config["init_log_alpha"]).to(self.device)
        self.log_alpha.requires_grad = True

        # Make train ops
        # TODO: gradient_clip_value
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(),
                                                lr=self.config["actor_learning_rate"])
        # ! TF2 code has 1 optimizer per softq network
        self.softqs_optimizer = torch.optim.Adam(self.softq_networks.parameters(),
                                                 lr=self.config["softq_learning_rate"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                lr=self.config["alpha_learning_rate"])

        self.replay_buffer = PreAllocMemory(
            int(self.config["replay_buffer_size"]), self.state_shape, env.action_space.shape)
        self.n_updates = 0
        self.total_steps = 0
        self.total_episodes = 0
        if self.config["summaries"]:
            self.summary_writer = SummaryWriter(str(self.monitor_path))
            summary_writer.set(self.summary_writer)

        self.env_runner = EnvRunner(self.env,
                                    self,
                                    usercfg,
                                    transition_preprocessor=self.config.get("transition_preprocessor", None),
                                    summaries=self.config["summaries"],
                                    episode_rewards_file=(
                                        self.monitor_path / "train_rewards.txt" if self.config["write_train_rewards"] else None)
                                    )

        if self.config["checkpoints"]:
            self.checkpoint_directory = self.monitor_path / "checkpoints"
            self.checkpoint_directory.mkdir(exist_ok=True)

        if self.config["test_frequency"] > 0 and self.config["n_test_episodes"] > 0:
            test_env = deepcopy(env)
            unw = test_env.unwrapped
            if hasattr(unw, "summaries"):
                unw.summaries = False
            if hasattr(unw, "log_data"):
                unw.log_data = False
            deterministic_policy = DeterministicPolicy(test_env, self.deterministic_actions)
            self.test_env_runner = EnvRunner(test_env,
                                             deterministic_policy,
                                             usercfg,
                                             summary_writer=None,
                                             transition_preprocessor=self.config.get("transition_preprocessor", None),
                                             episode_rewards_file=(
                                                 self.monitor_path / "test_rewards.txt")
                                             )
            header = [""]  # (epoch) id has no name in header
            header += [f"rew_{i}" for i in range(self.config["n_test_episodes"])]
            header += ["rew_mean", "rew_std"]
            self.test_results_file = self.monitor_path / "test_results.csv"
            with open(self.test_results_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)

            self.total_rewards = np.empty((self.config["n_test_episodes"],), dtype=np.float32)

    def deterministic_actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        dist = self.actor_network(torch.FloatTensor(states).to(self.device))
        return to_numpy(dist.mean)

    def train(self, mode: bool = True) -> None:
        self.actor_network.train(mode)
        self.softq_networks.train(mode)

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        dist = self.actor_network(torch.FloatTensor(state[None, :]).to(self.device))
        sample = dist.sample()
        return to_numpy(sample)[0]

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train_critics(self, state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch):
        # Calculate critic targets
        next_action_batch_dist = self.actor_network(state1_batch)
        next_action_batch = next_action_batch_dist.rsample()
        next_logprob_batch = next_action_batch_dist.log_prob(next_action_batch).sum(-1, keepdim=True)
        next_qs_values = self.target_softq_networks(state1_batch, next_action_batch)
        next_q_values = torch.min(*next_qs_values)
        next_values = next_q_values - self.alpha.detach() * next_logprob_batch
        next_values = (1.0 - terminal1_batch) * next_values
        softq_targets = self.config["reward_scale"] * reward_batch + self.config["gamma"] * next_values
        softq_targets = softq_targets.detach()

        # Update critics
        softq1_values, softq2_values = self.softq_networks(state0_batch, action_batch)
        softqs_loss = F.mse_loss(softq1_values, softq_targets) + F.mse_loss(softq2_values, softq_targets)

        self.softqs_optimizer.zero_grad()
        softqs_loss.backward()
        self.softqs_optimizer.step()

        softq = torch.cat((softq1_values, softq2_values))
        softq_mean, softq_std = torch.std_mean(softq)
        return to_numpy(softq_mean), to_numpy(softq_std), to_numpy(softq_targets), to_numpy(softqs_loss)

    def train_actor(self, state0_batch):
        dist = self.actor_network(state0_batch)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        softqs_pred = self.softq_networks(state0_batch, action)
        min_softq_pred = torch.min(*softqs_pred)
        actor_losses = self.alpha.detach() * log_prob - min_softq_pred
        actor_loss = actor_losses.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return to_numpy(actor_loss), to_numpy(log_prob.mean())

    def train_alpha(self, state0_batch):
        dist = self.actor_network(state0_batch)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        alpha_losses = -1.0 * self.alpha * (log_prob + self.target_entropy).detach()
        alpha_loss = alpha_losses.mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return to_numpy(alpha_loss)

    def do_test_episodes(self, step) -> None:
        for i in range(self.config["n_test_episodes"]):
            test_trajectory = self.test_env_runner.get_trajectory(stop_at_trajectory_end=True)
            self.total_rewards[i] = np.sum(test_trajectory.rewards)
        test_rewards_mean = np.mean(self.total_rewards)
        test_rewards_std = np.std(self.total_rewards)
        to_write = [step] + self.total_rewards.tolist() + [test_rewards_mean, test_rewards_std]
        with open(self.test_results_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(to_write)

    def learn(self):
        self.train()
        # Arrays to keep results from train function over different train steps in
        softq_means = np.empty((self.config["n_train_steps"],), np.float32)
        softq_stds = np.empty((self.config["n_train_steps"],), np.float32)
        softq_losses = np.empty((self.config["n_train_steps"],), np.float32)
        actor_losses = np.empty((self.config["n_train_steps"],), np.float32)
        alpha_losses = np.empty((self.config["n_train_steps"],), np.float32)
        action_logprob_means = np.empty((self.config["n_train_steps"],), np.float32)
        total_episodes = 0
        episode_start_time = time()
        summary_writer.start()
        for step in range(self.config["max_steps"]):
            if self.config["test_frequency"] > 0 and (step % self.config["test_frequency"]) == 0 and self.config["n_test_episodes"] > 0:
                self.do_test_episodes(step)
            experience = self.env_runner.get_steps(1)[0]
            self.total_steps += 1
            self.replay_buffer.add(experience.state, experience.action, experience.reward,
                                   experience.next_state, experience.terminal)
            if self.replay_buffer.n_entries > self.config["replay_start_size"]:
                for i in range(self.config["n_train_steps"]):
                    sample = self.replay_buffer.get_batch(self.config["batch_size"])
                    states0 = torch.as_tensor(sample["states0"], device=self.device)
                    softq_mean, softq_std, softq_targets, softq_loss = self.train_critics(
                        states0,
                        torch.as_tensor(sample["actions"], device=self.device),
                        torch.as_tensor(sample["rewards"], device=self.device),
                        torch.as_tensor(sample["states1"], device=self.device),
                        torch.as_tensor(sample["terminals1"], device=self.device))
                    if (step % self.config["actor_update_frequency"]) == 0:
                        actor_loss, action_logprob_mean = self.train_actor(states0)
                        alpha_loss = self.train_alpha(states0)
                        actor_losses[i] = actor_loss
                        alpha_losses[i] = alpha_loss
                        action_logprob_means[i] = action_logprob_mean
                    else:
                        print("WARNING: ACTOR NOT UPDATED")
                    softq_means[i] = softq_mean
                    softq_stds[i] = softq_std
                    softq_losses[i] = softq_loss
                    # Update the target networks
                    if (step % self.config["critic_target_update_frequency"]) == 0:
                        soft_update(self.softq_networks,
                                    self.target_softq_networks,
                                    self.config["tau"])
                if self.config["summaries"]:
                    summary_writer.add_scalar("model/predicted_softq_mean", np.mean(softq_means), self.total_steps)
                    summary_writer.add_scalar("model/predicted_softq_std", np.mean(softq_stds), self.total_steps)
                    summary_writer.add_scalar("model/softq_targets", softq_targets.mean(), self.total_steps)
                    summary_writer.add_scalar("model/softq_loss", np.mean(softq_losses), self.total_steps)
                    if (step % self.config["actor_update_frequency"]) == 0:
                        summary_writer.add_scalar("model/actor_loss", np.mean(actor_losses), self.total_steps)
                        summary_writer.add_scalar("model/alpha_loss", np.mean(alpha_losses), self.total_steps)
                        summary_writer.add_scalar("model/alpha", to_numpy(self.alpha), self.total_steps)
                        summary_writer.add_scalar("model/action_logprob_mean",
                                                  np.mean(action_logprob_means), self.total_steps)
                self.n_updates += 1
            if experience.terminal:
                episode_end_time = time()
                summary_writer.add_scalar("diagnostics/episode_duration_seconds",
                                          episode_end_time - episode_start_time,
                                          self.total_steps)
                if self.config["checkpoints"] and (total_episodes % self.config["checkpoint_every_episodes"]) == 0:
                    torch.save(self.actor_network.state_dict(),
                               self.checkpoint_directory / f"actor_ep{total_episodes}.pt")
                total_episodes += 1
                episode_start_time = time()

        summary_writer.stop()
        if self.config["save_model"]:
            torch.save(self.actor_network.state_dict(), self.checkpoint_directory / "actor_final.pt")

    def choose_action(self, state, features):
        if self.total_steps < self.config["replay_start_size"]:
            action = np.random.uniform(-1.0, 1.0, self.env.action_space.shape)
        else:
            action = self.action(state)
        return {"action": action}

    def get_env_action(self, action):
        """
        Converts an action from self.choose_action to an action to be given to the environment.
        """
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [pyd.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def mlp(input_dim, output_dim, n_hidden_units, n_hidden_layers, output_mod=None):
    if n_hidden_layers == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, n_hidden_units), nn.ReLU(inplace=True)]
        for i in range(n_hidden_layers - 1):
            mods += [nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(n_hidden_units, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class ActorNetwork(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, n_actions, n_hidden_units, n_hidden_layers):
        super().__init__()

        self.trunk = mlp(obs_dim, 2 * n_actions, n_hidden_units, n_hidden_layers)
        self.apply(weight_init)

    def forward(self, obs):
        mu, scale = self.trunk(obs).chunk(2, dim=-1)
        scale = F.softplus(scale) + 1e-5
        dist = SquashedNormal(mu, scale)
        return dist


class DoubleQCriticNetwork(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, n_hidden_units, n_hidden_layers):
        super().__init__()

        self.Q1 = mlp(obs_dim + action_dim, 1, n_hidden_units, n_hidden_layers)
        self.Q2 = mlp(obs_dim + action_dim, 1, n_hidden_units, n_hidden_layers)
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2
