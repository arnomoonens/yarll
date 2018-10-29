# -*- coding: utf8 -*-
import tensorflow as tf

from yarll.agents.ppo.ppo import PPO
from yarll.agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous
from yarll.misc.network_ops import kl_divergence

def trpo_loss(old_log, new_log, beta, advantage):
    return tf.exp(new_log - old_log) * advantage - beta * kl_divergence(old_log, new_log)

class TRPO(PPO):
    """Trust Region Policy Optimization agent."""

    def __init__(self, env, monitor_path: str, video=False, **usercfg) -> None:
        usercfg["kl_coef"] = 1.0 # beta
        super(TRPO, self).__init__(env, monitor_path, video, **usercfg)

    def make_actor_loss(self, old_network, new_network, advantage):
        return trpo_loss(old_network.action_log_prob, new_network.action_log_prob, self.config["kl_coef"], advantage)


class TRPODiscrete(TRPO):
    def build_networks(self) -> ActorCriticNetworkDiscrete:
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))


class TRPODiscreteCNN(TRPODiscrete):
    def build_networks(self) -> ActorCriticNetworkDiscreteCNN:
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class TRPOContinuous(TRPO):
    def build_networks(self) -> ActorCriticNetworkContinuous:
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def get_env_action(self, action):
        return action
