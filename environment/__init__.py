from environment.Environment import Environment
from environment.Acrobot import Acrobot
from environment.CartPole import CartPole
from environment.registration import register_environment, make_environment, make_random_environments

register_environment("Acrobot-v1", Acrobot)
register_environment("CartPole-v0", CartPole)
