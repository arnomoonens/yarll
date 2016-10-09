import numpy as np
import gym
import sys

# Adaption of Karpathy's Pong from Pixels article to apply it using a very simple neural network on the MountainCar environment

env = gym.make('MountainCar-v0')
A = env.action_space
O = env.observation_space

x_low, y_low = O.low
x_high, y_high = O.high

gamma = 0.99
learning_rate = 0.1
batch_size = 10

def scale_state(state):
    x, y = state
    O_width = x_high - x_low
    O_height = y_high - y_low
    return [(x - x_low) / O_width, (y - y_low) / O_height]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def choose_action(output, temperature=1.0):
        # total = sum([np.exp(float(o) / temperature) for o in output])
        # probs = [np.exp(float(o) / temperature) / total for o in output]
        probs = output / sum(output)
        action = np.random.choice(len(output), p=probs)
        return action, probs[action]

def backward_step(state, feedback):
    """Return delta_w"""
    return np.dot(np.vstack(feedback), np.reshape(state, (1, O.shape[0])))

# J(theta)
def discount_rewards(rewards):
    discounted_r = np.zeros_like(rewards)
    summed = 0
    for i in reversed(range(len(rewards))):
        summed = summed * gamma + rewards[i]
        discounted_r[i] = summed
    return discounted_r

w = np.random.randn(A.n, O.shape[0])  # n_actions * n_state_features

n_episodes = 6000
env.monitor.start(sys.argv[1])

gradient = np.zeros_like(w)
episode_nr = 1
while True:  # Keep executing episodes
    episode_nr += 1
    state = env.reset()
    state = scale_state(state)
    done = False
    reward_sum = 0
    episode_rewards = []
    action_taken = []
    encountered_states = []
    while not done:  # Keep executing steps until done
        encountered_states.append(state)
        nn_outputs = sigmoid(np.dot(w, state))
        action, probability = choose_action(nn_outputs)
        state, reward, done, info = env.step(action)
        state = scale_state(state)
        reward_sum += reward
        episode_rewards.append(reward)

        y = np.zeros(A.n)
        y[action] = 1
        action_taken.append(y)

        if done:
            epdlogp = np.vstack(action_taken)
            episode_states = np.vstack(encountered_states)

            discounted_episode_rewards = discount_rewards(episode_rewards)
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            discounted_episode_rewards /= np.std(discounted_episode_rewards)
            epdlogp *= np.reshape(np.repeat(discounted_episode_rewards, A.n), (len(discounted_episode_rewards), A.n))

            episode_gradient = np.zeros_like(w)
            for i, state in enumerate(encountered_states):
                bw_step = backward_step(state, epdlogp[i])
                episode_gradient += bw_step
            gradient += episode_gradient

            if(len(encountered_states) < 200):
                print("Episode shorter than 200 episodes!")

            if episode_nr % batch_size == 0:  # batch is done
                w += learning_rate * gradient
                gradient = np.zeros_like(w)
