import gym
from environment.registration import EnvSpec

def register_env(name, entry_point, **kwargs):
    gym.envs.registry.env_specs["Old" + name] = gym.envs.registry.env_specs[name]
    gym.envs.registry.env_specs[name] = EnvSpec(name, entry_point=entry_point, **kwargs)
    return

register_env(
    "CartPole-v0",
    entry_point="environment.cartpole:CartPole",
    max_episode_steps=200,
    reward_threshold=195.0
)

register_env(
    "Acrobot-v1",
    entry_point="environment.acrobot:Acrobot",
    max_episode_steps=500
)
