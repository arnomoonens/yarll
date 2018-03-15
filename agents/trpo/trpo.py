# -*- coding: utf8 -*-
import tensorflow as tf

from agents.ppo.ppo import PPO
from agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous
from misc.network_ops import kl_divergence

class TRPO(PPO):
    """Trust Region Policy Optimization algorithm."""

    def __init__(self, env, monitor_path: str, video=False, **usercfg) -> None:
        usercfg["kl_coef"] = 1 # beta
        super(TRPO, self).__init__(env, monitor_path, video=False, **usercfg)

    def make_actor_loss(self, old_network, new_network, advantage):
        adv_part = tf.exp(new_network.action_log_prob - old_network.action_log_prob) * advantage
        kl_part = self.config["kl_coef"] * kl_divergence(old_network.action_log_prob, new_network.action_log_prob)
        return adv_part - kl_part


class TRPODiscrete(TRPO):
    def build_networks(self) -> ActorCriticNetworkDiscrete:
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden_units"],
            self.config["n_hidden_layers"])


class TRPODiscreteCNN(TRPODiscrete):
    def build_networks(self) -> ActorCriticNetworkDiscreteCNN:
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden_units"])


class TRPOContinuous(TRPO):
    def build_networks(self) -> ActorCriticNetworkContinuous:
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            self.config["n_hidden_units"],
            self.config["n_hidden_layers"])

    def get_env_action(self, action):
        return action
