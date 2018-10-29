# -*- coding: utf8 -*-

from yarll.misc.utils import load
from yarll.misc.exceptions import ClassNotRegisteredError

# Dictionary with names of algorithms as keys and a list of variants as values.
agent_registry: dict = {}

def register_agent(name: str, entry_point: str, state_dimensions: str, action_space: str, RNN: bool = False):
    """Register an enviroment of a name with a class to be instantiated."""
    if name not in agent_registry:
        agent_registry[name] = [{
            "state_dimensions": state_dimensions,
            "action_space": action_space,
            "RNN": RNN,
            "entry_point": entry_point}]
    else:
        in_list = next((item for item in enumerate(agent_registry[name]) if
                        item[1]["action_space"] == action_space and
                        item[1]["state_dimensions"] == state_dimensions and
                        item[1]["RNN"] == RNN), False)
        if in_list:
            agent_registry[name][in_list[0]]["entry_point"] = entry_point
        else:
            agent_registry[name].append({
                "state_dimensions": state_dimensions,
                "action_space": action_space,
                "RNN": RNN,
                "entry_point": entry_point})

def make_agent(name: str, state_dimensions: str, action_space: str, RNN: bool = False, **args):
    """Make an agent of a given name, possibly using extra arguments."""
    try:
        Agent = agent_registry[name]
        Agent = next((agent_type for agent_type in Agent if
                      agent_type["action_space"] == action_space and
                      agent_type["state_dimensions"] == state_dimensions and
                      agent_type["RNN"] == RNN))
        Agent = Agent["entry_point"]
        if not callable(Agent):
            Agent = load(Agent)
    except (KeyError, StopIteration):
        raise ClassNotRegisteredError(
            "The agent {} for state dimensionality {}, action space {} and RNN={} is not registered.".format(
                name,
                state_dimensions,
                action_space,
                RNN))
    return Agent(**args)
