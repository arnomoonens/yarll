from agents.registration import register_agent, make_agent

# All agents
from agents.A2C import A2CDiscrete, A2CContinuous
from agents.A3C import A3CDiscrete, A3CContinuous
from agents.async_knowledge_transfer import AsyncKnowledgeTransfer
from agents.CEM import CEM
from agents.Karpathy import Karpathy
# from agents.Karpathy_CNN import KPCNNLearner  # Not tested
from agents.knowledge_transfer import KnowledgeTransfer
from agents.REINFORCE import REINFORCEDiscrete, REINFORCEContinuous  # REINFORCEDiscreteCNN not tested
from agents.SarsaFA import SarsaFA

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
