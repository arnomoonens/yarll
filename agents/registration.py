#!/usr/bin/env python
# -*- coding: utf8 -*-

from Agent import Agent
agent_registry = {}

def register_agent(name, cls):
    """Register an enviroment of a name with a class to be instantiated."""
    agent_registry[name] = cls

def make_agent(name, **args):
    """Make an agent of a given name, possibly using extra arguments."""
    env = agent_registry.get(name, Agent)
    if name in agent_registry:
        return env(**args)
    else:
        return env(name, **args)
