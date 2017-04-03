from Environment.Environment import Environment
from Environment.CartPole import CartPole
from Environment.registration import register_environment, make_environment

register_environment("CartPole-v0", CartPole)