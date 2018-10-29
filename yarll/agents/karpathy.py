# -*- coding: utf8 -*-

import numpy as np
import logging

from gym import wrappers

from yarll.agents.agent import Agent
from yarll.misc.utils import discount_rewards
from yarll.misc.reporter import Reporter

logging.getLogger().setLevel("INFO")

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

# Adaption of Karpathy's Pong from Pixels article to apply it using a simple neural network on other environments

def scale_state(state, O):
    return state - O.low

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def random_with_probability(output, n_actions, temperature=1.0):
    # total = sum([np.exp(float(o) / temperature) for o in output])
    # probs = [np.exp(float(o) / temperature) / total for o in output]
    probs = output / np.sum(output)
    action = np.random.choice(n_actions, p=probs)
    return action, probs

class Karpathy(Agent):
    """Karpathy policy gradient agent"""
    def __init__(self, env, monitor_path, video=True, **usercfg):
        super(Karpathy, self).__init__(**usercfg)
        self.env = wrappers.Monitor(env, monitor_path, force=True, video_callable=(None if video else False))
        self.nA = self.env.action_space.n
        # Default configuration. Can be overwritten using keyword arguments.
        self.config.update(dict(
            # timesteps_per_batch=10000,
            # n_iter=100,
            episode_max_length=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps"),
            gamma=0.99,
            learning_rate=0.05,
            batch_size=10,  # Amount of episodes after which to adapt gradients
            decay_rate=0.99,  # Used for RMSProp
            n_hidden_units=20,
            draw_frequency=50,  # Draw a plot every 50 episodes
            repeat_n_actions=1
        ))
        self.config.update(usercfg)
        self.build_network()

    def build_network(self):
        self.w1 = np.random.randn(self.nO, self.config["n_hidden_units"]) / np.sqrt(self.config["n_hidden_units"])
        self.w2 = np.random.randn(self.config["n_hidden_units"], self.nA) / np.sqrt(self.nA)

    def choose_action(self, state):
        x1, nn_outputs = self.forward_step(state)
        action, probabilities = random_with_probability(nn_outputs, self.nA)
        return action, probabilities, x1

    def forward_step(self, state):
        x1 = np.dot(state, self.w1)
        x1[x1 < 0] = 0  # ReLU
        output = sigmoid(np.dot(x1, self.w2))
        return x1, output

    def backward_step(self, x0, x1, feedback):
        """Computes how much to change the weights from input->hidden layer and hidden->final layer"""
        change_w2 = np.dot(x1.T, feedback)  # 8x200 * 200x3 = 8x3
        dh = np.dot(feedback, self.w2.T)  # 200x3 * 3x8 = 200x8
        dh[x1 <= 0] = 0
        change_w1 = x0.T.dot(dh)  # 2x200 * 200x8 = 2x8
        return change_w1, change_w2

    def get_trajectory(self, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory)
        Return dictionary of results
        Note that this function returns more than the get_trajectory in the Learner class.
        """
        env = self.env
        state = env.reset()
        states = []
        actions = []
        rewards = []
        episode_probabilities = []
        x1s = []
        for _ in range(self.config["episode_max_length"]):
            action, probabilities, x1 = self.choose_action(state)
            x1s.append(x1)
            states.append(state)
            (state, rew, done, _) = env.step(action)
            actions.append(action)
            rewards.append(rew)
            episode_probabilities.append(probabilities)
            if done:
                break
            if render:
                env.render()
        return {"reward": np.array(rewards),
                "state": np.array(states),
                "action": np.array(actions),
                "prob": np.array(episode_probabilities),
                "x1": np.array(x1s)
                }

    def learn(self):
        reporter = Reporter()

        gradient1 = np.zeros_like(self.w1)
        gradient2 = np.zeros_like(self.w2)

        rmsprop1 = np.zeros_like(self.w1)
        rmsprop2 = np.zeros_like(self.w2)

        iteration = 0  # amount of batches processed
        episode_nr = 0
        episode_lengths = np.zeros(self.config["batch_size"])
        episode_rewards = np.zeros(self.config["batch_size"])
        mean_rewards = []
        while True:  # Keep executing episodes
            trajectory = self.get_trajectory(self.config["episode_max_length"])

            episode_rewards[episode_nr % self.config["batch_size"]] = sum(trajectory["reward"])
            episode_lengths[episode_nr % self.config["batch_size"]] = len(trajectory["reward"])
            episode_nr += 1
            action_taken = (np.arange(self.nA) == trajectory["action"][:, None]).astype(np.float32)  # one-hot encoding
            epdlogp = action_taken - trajectory["prob"]

            # episode_states = np.vstack(encountered_states)

            discounted_episode_rewards = discount_rewards(trajectory["reward"], self.config["gamma"])
            # print(discounted_episode_rewards)
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            discounted_episode_rewards /= np.std(discounted_episode_rewards)
            epdlogp *= np.reshape(np.repeat(discounted_episode_rewards, self.nA), (len(discounted_episode_rewards), self.nA))

            change_w1, change_w2 = self.backward_step(trajectory["state"], trajectory['x1'], epdlogp)

            gradient1 += change_w1
            gradient2 += change_w2

            if episode_nr % self.config["batch_size"] == 0:  # batch is done
                iteration += 1
                rmsprop1 = self.config["decay_rate"] * rmsprop1 + (1 - self.config["decay_rate"]) * gradient1**2
                rmsprop2 = self.config["decay_rate"] * rmsprop2 + (1 - self.config["decay_rate"]) * gradient2**2
                self.w1 += self.config["learning_rate"] * gradient1 / (np.sqrt(rmsprop1) + 1e-5)
                self.w2 += self.config["learning_rate"] * gradient2 / (np.sqrt(rmsprop2) + 1e-5)
                gradient1 = np.zeros_like(self.w1)
                gradient2 = np.zeros_like(self.w2)
                reporter.print_iteration_stats(iteration, episode_rewards, episode_lengths, episode_nr)
                mean_rewards.append(episode_rewards.mean())
                if episode_nr % self.config["draw_frequency"] == 0:
                    reporter.draw_rewards(mean_rewards)
