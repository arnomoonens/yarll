from pathlib import Path
import numpy as np
import tensorflow as tf

from yarll.agents.agent import Agent
from yarll.environment.environment import Environment
from yarll.policies.e_greedy import EGreedy

class QLearning(Agent):
    def __init__(self, env: Environment, monitor_path: str, **usercfg) -> None:
        super().__init__()
        self.env = env
        self.monitor_path = Path(monitor_path)

        self.config.update(
            n_episodes=1000,
            gamma=0.99,
            alpha=0.5,
            epsilon=0.1
        )
        self.config.update(usercfg)

        self.Q_values = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=np.float32)
        self.policy = EGreedy(self.config["epsilon"])

        self.writer = tf.summary.create_file_writer(str(self.monitor_path))

    def learn(self):
        env = self.env
        total_steps = 0
        with self.writer.as_default():
            for episode in range(self.config["n_episodes"]):
                done = False
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                while not done:
                    action, Q_value = self.policy(self.Q_values[state])
                    new_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    total_steps += 1
                    best_next_action = np.argmax(self.Q_values[new_state])
                    td_target = reward + self.config["gamma"] * self.Q_values[new_state, best_next_action]
                    td_delta = td_target - Q_value
                    self.Q_values[state, action] += self.config["alpha"] * td_delta

                    if done:
                        tf.summary.scalar("env/reward", episode_reward, total_steps)
                        tf.summary.scalar("env/N_episodes", episode + 1, total_steps)
                        tf.summary.scalar("env/episode_length", episode_length, total_steps)
                        break

                    state = new_state
