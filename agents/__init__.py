from agents.registration import register_agent, make_agent

# All agents
from agents.a2c import A2CDiscrete, A2CContinuous
from agents.a3c import A3CDiscrete, A3CContinuous
from agents.async_knowledge_transfer import AsyncKnowledgeTransfer
from agents.cem import CEM
from agents.karpathy import Karpathy
# from agents.karpathy_cnn import KPCNNLearner  # Not tested
from agents.knowledge_transfer import KnowledgeTransfer
from agents.reinforce import REINFORCEDiscrete, REINFORCEDiscreteCNN, REINFORCEContinuous  # REINFORCEDiscreteCNN not tested
from agents.sarsa_fa import SarsaFA

register_agent("A2C", state_dimensions="single", action_space="discrete", cls=A2CDiscrete)
register_agent("A2C", state_dimensions="single", action_space="continuous", cls=A2CContinuous)
register_agent("A3C", state_dimensions="single", action_space="discrete", cls=A3CDiscrete)
register_agent("A3C", state_dimensions="single", action_space="continuous", cls=A3CContinuous)
register_agent("AsyncKnowledgeTransfer", state_dimensions="single", action_space="discrete", cls=AsyncKnowledgeTransfer)
register_agent("CEM", state_dimensions="single", action_space="discrete", cls=CEM)
register_agent("CEM", state_dimensions="single", action_space="continuous", cls=CEM)
register_agent("Karpathy", state_dimensions="single", action_space="discrete", cls=Karpathy)
register_agent("KnowledgeTransfer", state_dimensions="single", action_space="discrete", cls=KnowledgeTransfer)
register_agent("REINFORCE", state_dimensions="single", action_space="discrete", cls=REINFORCEDiscrete)
register_agent("REINFORCE", state_dimensions="multi", action_space="discrete", cls=REINFORCEDiscreteCNN)
register_agent("REINFORCE", state_dimensions="single", action_space="continuous", cls=REINFORCEContinuous)
register_agent("SarsaFA", state_dimensions="single", action_space="discrete", cls=SarsaFA)
