# -*- coding: utf8 -*-

from yarll.agents.ppo.dppo import DPPO
from yarll.agents.trpo.trpo import trpo_loss
from yarll.agents.actorcritic.actor_critic import ActorCriticNetworkDiscrete,\
    ActorCriticNetworkDiscreteCNN, ActorCriticNetworkContinuous

class DTRPO(DPPO):
    """Trust Region Policy Optimization agent."""

    def __init__(self, env, monitor_path: str, **usercfg) -> None:
        """
        Initialize the environment.

        Args:
            self: (todo): write your description
            env: (todo): write your description
            monitor_path: (str): write your description
            usercfg: (todo): write your description
        """
        usercfg["kl_coef"] = 1.0  # beta
        super(DTRPO, self).__init__(env, monitor_path, **usercfg)

    def make_actor_loss(self, old_network, new_network, advantage):
        """
        Make a new network loss.

        Args:
            self: (todo): write your description
            old_network: (todo): write your description
            new_network: (todo): write your description
            advantage: (todo): write your description
        """
        return trpo_loss(old_network.action_log_prob, new_network.action_log_prob, self.config["kl_coef"], advantage)


class DTRPODiscrete(DTRPO):
    def build_networks(self) -> ActorCriticNetworkDiscrete:
        """
        Constructs the network.

        Args:
            self: (todo): write your description
        """
        return ActorCriticNetworkDiscrete(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))


class DTRPODiscreteCNN(DTRPODiscrete):
    def build_networks(self) -> ActorCriticNetworkDiscreteCNN:
        """
        Construct a list of networks.

        Args:
            self: (todo): write your description
        """
        return ActorCriticNetworkDiscreteCNN(
            list(self.env.observation_space.shape),
            self.env.action_space.n,
            int(self.config["n_hidden_units"]))


class DTRPOContinuous(DTRPO):
    def build_networks(self) -> ActorCriticNetworkContinuous:
        """
        Return a network interface.

        Args:
            self: (todo): write your description
        """
        return ActorCriticNetworkContinuous(
            list(self.env.observation_space.shape),
            self.env.action_space,
            int(self.config["n_hidden_units"]),
            int(self.config["n_hidden_layers"]))

    def get_env_action(self, action):
        """
        Returns the action action.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        return action
