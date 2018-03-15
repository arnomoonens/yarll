# -*- coding: utf8 -*-

from agents.ppo.dppo import DPPO
from agents.trpo.trpo import trpo_loss
from agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous

class DTRPO(DPPO):
    """Trust Region Policy Optimization agent."""

    def __init__(self, env, monitor_path: str, **usercfg) -> None:
        usercfg["kl_coef"] = 1.0  # beta
        super(DTRPO, self).__init__(env, monitor_path, **usercfg)

    def make_actor_loss(self, old_network, new_network, advantage):
        return trpo_loss(old_network.action_log_prob, new_network.action_log_prob, self.config["kl_coef"], advantage)


class DTRPODiscrete(DTRPO):
    def build_networks(self) -> ActorCriticNetworkDiscrete:
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden_units"],
            self.config["n_hidden_layers"])


class DTRPODiscreteCNN(DTRPODiscrete):
    def build_networks(self) -> ActorCriticNetworkDiscreteCNN:
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            self.config["n_hidden_units"])


class DTRPOContinuous(DTRPO):
    def build_networks(self) -> ActorCriticNetworkContinuous:
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            self.config["n_hidden_units"],
            self.config["n_hidden_layers"])

    def get_env_action(self, action):
        return action
