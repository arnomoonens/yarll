import gym
from yarll.environment.registration import EnvSpec

def register_env(name, entry_point: str, tags=None, **kwargs):
    already_registered = name in gym.envs.registry.env_specs
    if already_registered:
        old_env_name = "Old" + name
        gym.envs.registry.env_specs[old_env_name] = gym.envs.registry.env_specs[name]
        new_tags = gym.envs.registry.env_specs[name].tags
    else:
        new_tags = {}
    if tags is not None:
        new_tags.update(tags)
    gym.envs.registry.env_specs[name] = EnvSpec(
        name,
        entry_point=entry_point,
        kwargs={"old_env_name": old_env_name} if already_registered else {},
        tags=new_tags,
        **kwargs)

register_env(
    "CartPole-v0",
    entry_point="yarll.environment.cartpole:CartPole",
    max_episode_steps=200
)

register_env(
    "Acrobot-v1",
    entry_point="yarll.environment.acrobot:Acrobot",
    max_episode_steps=500
)

register_env(
    "FrozenLake8x8-v0",
    entry_point="yarll.environment.environment:Environment",
    tags={
        "wrappers": ["environment.wrappers:DiscreteObservationWrapper"]
    }
)
