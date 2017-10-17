from agents.registration import register_agent, make_agent

# All agents
from agents.a2c import A2CDiscrete, A2CContinuous
from agents.a3c import A3CDiscrete, A3CContinuous
from agents.async_knowledge_transfer import AsyncKnowledgeTransfer
from agents.cem import CEM
from agents.karpathy import Karpathy
# from agents.karpathy_cnn import KPCNNLearner  # Not tested
from agents.knowledge_transfer import KnowledgeTransfer
from agents.reinforce import REINFORCEDiscrete, REINFORCEDiscreteCNN, REINFORCEContinuous, REINFORCEDiscreteRNN, REINFORCEDiscreteCNNRNN
from agents.sarsa_fa import SarsaFA

register_agent("A2C", A2CDiscrete, state_dimensions="single", action_space="discrete")
register_agent("A2C", A2CContinuous, state_dimensions="single", action_space="continuous")
register_agent("A3C", A3CDiscrete, state_dimensions="single", action_space="discrete")
register_agent("A3C", A3CContinuous, state_dimensions="single", action_space="continuous")
register_agent("AsyncKnowledgeTransfer", AsyncKnowledgeTransfer, state_dimensions="single", action_space="discrete")
register_agent("CEM", CEM, state_dimensions="single", action_space="discrete")
register_agent("CEM", CEM, state_dimensions="single", action_space="continuous")
register_agent("Karpathy", Karpathy, state_dimensions="single", action_space="discrete")
register_agent("KnowledgeTransfer", KnowledgeTransfer, state_dimensions="single", action_space="discrete")
register_agent("REINFORCE", REINFORCEDiscrete, state_dimensions="single", action_space="discrete")
register_agent("REINFORCE", REINFORCEDiscreteRNN, state_dimensions="single", action_space="discrete", RNN=True)
register_agent("REINFORCE", REINFORCEDiscreteCNN, state_dimensions="multi", action_space="discrete")
register_agent("REINFORCE", REINFORCEDiscreteCNNRNN, state_dimensions="multi", action_space="discrete", RNN=True)
register_agent("REINFORCE", REINFORCEContinuous, state_dimensions="single", action_space="continuous")
register_agent("SarsaFA", SarsaFA, state_dimensions="single", action_space="discrete")
