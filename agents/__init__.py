from agents.registration import register_agent, make_agent

# All agents
from agents.a2c import A2CDiscrete, A2CContinuous
from agents.a3c import A3CDiscrete, A3CContinuous
from agents.async_knowledge_transfer import AsyncKnowledgeTransfer
from agents.cem import CEM
from agents.karpathy import Karpathy
# from agents.karpathy_cnn import KPCNNLearner  # Not tested
from agents.knowledge_transfer import KnowledgeTransfer
from agents.reinforce import REINFORCEDiscrete, REINFORCEContinuous  # REINFORCEDiscreteCNN not tested
from agents.sarsa_fa import SarsaFA

register_agent("A2C", "discrete", A2CDiscrete)
register_agent("A2C", "continuous", A2CContinuous)
register_agent("A3C", "discrete", A3CDiscrete)
register_agent("A3C", "continuous", A3CContinuous)
register_agent("AsyncKnowledgeTransfer", "discrete", AsyncKnowledgeTransfer)
register_agent("CEM", "discrete", CEM)
register_agent("CEM", "continuous", CEM)
register_agent("Karpathy", "discrete", Karpathy)
register_agent("KnowledgeTransfer", "discrete", KnowledgeTransfer)
register_agent("REINFORCE", "discrete", REINFORCEDiscrete)
register_agent("REINFORCE", "continuous", REINFORCEContinuous)
register_agent("SarsaFA", "discrete", SarsaFA)
