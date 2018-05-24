# -*- coding: utf8 -*-

from environment.wrappers import DescriptionWrapper

class Environment(DescriptionWrapper):
    def __init__(self, env, **kwargs):
        if (env.spec.timestep_limit is not None) and not env.spec.tags.get('vnc'):
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env,
                            max_episode_steps=env.spec.max_episode_steps,
                            max_episode_seconds=env.spec.max_episode_seconds)
        super(Environment, self).__init__(env)
