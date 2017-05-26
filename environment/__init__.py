from environment.environment import Environment
from environment.acrobot import Acrobot
from environment.cartpole import CartPole
from environment.registration import register_environment, make_environment, make_random_environments

register_environment("Acrobot-v1", Acrobot)
register_environment("CartPole-v0", CartPole)
