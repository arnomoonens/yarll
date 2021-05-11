# -*- coding: utf8 -*-

from yarll.misc.utils import load
from yarll.misc.exceptions import ClassNotRegisteredError

# Dictionary with names of algorithms as keys and a list of variants as values.
agent_registry: dict = {}

def register_agent(name: str,
                   entry_point: str,
                   state_dimensions: str,
                   action_space: str,
                   rnn: bool = False,
                   backend: str = "tensorflow"):
    """Register an enviroment of a name with a class to be instantiated."""
    if name not in agent_registry:
        agent_registry[name] = [{
            "state_dimensions": state_dimensions,
            "action_space": action_space,
            "rnn": rnn,
            "backend": backend,
            "entry_point": entry_point}]
    else:
        in_list = next((item for item in enumerate(agent_registry[name]) if
                        item[1]["action_space"] == action_space and
                        item[1]["state_dimensions"] == state_dimensions and
                        item[1]["rnn"] == rnn and
                        item[1]["backend"] == backend), False)
        if in_list: # already registered before: update entry_point
            agent_registry[name][in_list[0]]["entry_point"] = entry_point
        else:
            agent_registry[name].append({
                "state_dimensions": state_dimensions,
                "action_space": action_space,
                "rnn": rnn,
                "backend": backend,
                "entry_point": entry_point})

def make_agent(name: str,
               state_dimensions: str,
               action_space: str,
               rnn: bool = False,
               backend: str = "tensorflow",
               **args):
    """Make an agent of a given name, possibly using extra arguments."""
    try:
        Agent = agent_registry[name]
        Agent = next((agent_type for agent_type in Agent if
                      agent_type["action_space"] == action_space and
                      agent_type["state_dimensions"] == state_dimensions and
                      agent_type["rnn"] == rnn and
                      agent_type["backend"] == backend))
        Agent = Agent["entry_point"]
        if not callable(Agent):
            Agent = load(Agent)
    except (KeyError, StopIteration) as not_found:
        raise ClassNotRegisteredError(
            f"The agent {name} for state dimensionality {state_dimensions}, action space {action_space}, rnn={rnn} and backend {backend} is not registered.") from not_found
    return Agent(rnn=rnn, **args)
