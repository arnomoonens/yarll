# -*- coding: utf8 -*-

from yarll.agents.ppo.dppo import DPPO
from yarll.agents.trpo.trpo import trpo_loss
from yarll.agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
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
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))


class DTRPODiscreteCNN(DTRPODiscrete):
    def build_networks(self) -> ActorCriticNetworkDiscreteCNN:
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class DTRPOContinuous(DTRPO):
    def build_networks(self) -> ActorCriticNetworkContinuous:
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def get_env_action(self, action):
        return action
