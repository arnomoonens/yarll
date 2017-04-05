#!/usr/bin/env python
# -*- coding: utf8 -*-

from misc.Exceptions import ClassNotRegisteredError

# Dictionary with names of algorithms as keys and a list of variants as values.
# Each variant has its action space type (and later on observation space type to support CNN/RNN agents)
agent_registry = {}

def register_agent(name, action_space, cls):
    """Register an enviroment of a name with a class to be instantiated."""
    if name not in agent_registry:
        agent_registry[name] = [{"action_space": action_space, "class": cls}]
    else:
        in_list = next((item for item in enumerate(agent_registry[name]) if item[1]["action_space"] == action_space), False)
        if in_list:
            agent_registry[name][in_list[0]]["class"] = cls
        else:
            agent_registry[name].append({"action_space": action_space, "class": cls})

def make_agent(name, action_space, **args):
    """Make an agent of a given name, possibly using extra arguments."""
    try:
        Agent = agent_registry[name]
        Agent = next((agent_type for agent_type in Agent if agent_type["action_space"] == action_space))
        Agent = Agent["class"]
    except (KeyError, StopIteration):
        raise ClassNotRegisteredError("The agent {} for action space {} is not registered.".format(name, action_space))
    return Agent(**args)
