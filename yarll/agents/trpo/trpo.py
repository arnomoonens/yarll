# -*- coding: utf8 -*-
import tensorflow as tf

from yarll.agents.ppo.ppo import PPO, PPOContinuous, PPODiscrete, PPODiscreteCNN
from yarll.misc.network_ops import kl_divergence

def trpo_loss(old_log, new_log, beta, advantage):
    return tf.exp(new_log - old_log) * advantage - beta * kl_divergence(old_log, new_log)

class TRPO(PPO):
    """Trust Region Policy Optimization agent."""

    def __init__(self, env, monitor_path: str, video=False, **usercfg) -> None:
        usercfg["kl_coef"] = 1.0  # beta
        super(TRPO, self).__init__(env, monitor_path, video=video, **usercfg)

    def _actor_loss(self, old_logprob, new_logprob, advantage):
        return trpo_loss(old_logprob, new_logprob, self.config["kl_coef"], advantage)

class TRPODiscrete(TRPO, PPODiscrete):
    pass

class TRPODiscreteCNN(TRPO, PPODiscreteCNN):
    pass

class TRPOContinuous(TRPO, PPOContinuous):
    pass
