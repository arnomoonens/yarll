import numpy as np
import gym
import sys

# Adaption of Karpathy's Pong from Pixels article to apply it using a simple neural network on the MountainCar environment

env = gym.make('MountainCar-v0')
A = env.action_space
O = env.observation_space

gamma = 0.99
learning_rate = 1e-4
batch_size = 10

n_hidden_units = 4

def scale_state(state):
    return state - O.low

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward_step(state, w1, w2):
        x1 = np.dot(state, w1)
        x1[x1 < 0] = 0
        output = sigmoid(np.dot(x1, w2))
        return x1, output


def choose_action(output, temperature=1.0):
        # total = sum([np.exp(float(o) / temperature) for o in output])
        # probs = [np.exp(float(o) / temperature) / total for o in output]
        probs = output / np.sum(output)
        action = np.random.choice(A.n, p=probs)
        return action, probs

def backward_step(x0, x1, feedback):
    """Computes how much to change the weights from input->hidden layer and hidden->final layer"""
    change_w2 = np.dot(x1.T, feedback)  # 8x200 * 200x3 = 8x3
    dh = np.dot(feedback, w2.T)  # 200x3 * 3x8 = 200x8
    dh[x1 <= 0] = 0
    change_w1 = x0.T.dot(dh)  # 2x200 * 200x8 = 2x8
    return change_w1, change_w2

# J(theta)
def discount_rewards(rewards):
    discounted_r = np.zeros_like(rewards)
    summed = 0
    for i in reversed(range(len(rewards))):
        summed = summed * gamma + rewards[i]
        discounted_r[i] = summed
    return discounted_r

w1 = 2 * np.random.randn(O.shape[0], n_hidden_units) - 1  # n_actions * n_state_features
w2 = 2 * np.random.randn(n_hidden_units, A.n) - 1

n_episodes = 6000
env.monitor.start(sys.argv[1])

gradient1 = np.zeros_like(w1)
gradient2 = np.zeros_like(w2)

rmsprop1 = np.zeros_like(w1)
rmsprop2 = np.zeros_like(w2)

episode_nr = 0
while True:  # Keep executing episodes
    state = env.reset()
    state = scale_state(state)
    done = False
    reward_sum = 0
    episode_rewards = []
    action_taken = []
    encountered_states = []
    x1s = []
    while not done:  # Keep executing steps until done
        encountered_states.append(state)
        x1, nn_outputs = forward_step(state, w1, w2)
        action, probabilities = choose_action(nn_outputs)
        state, reward, done, info = env.step(action)
        state = scale_state(state)
        reward_sum += reward
        episode_rewards.append(reward)
        x1s.append(x1)

        y = np.zeros(A.n)
        y[action] = 1
        action_taken.append(y - probabilities)

        if done:
            episode_nr += 1
            epdlogp = np.vstack(action_taken)

            episode_states = np.vstack(encountered_states)

            discounted_episode_rewards = discount_rewards(episode_rewards)
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            discounted_episode_rewards /= np.std(discounted_episode_rewards)
            epdlogp *= np.reshape(np.repeat(discounted_episode_rewards, A.n), (len(discounted_episode_rewards), A.n))
            episode_gradient1 = np.zeros_like(w1)
            episode_gradient2 = np.zeros_like(w2)

            change_w1, change_w2 = backward_step(np.array(encountered_states), np.array(x1s), epdlogp)

            gradient1 += change_w1
            gradient2 += change_w2

            if(len(encountered_states) < 200):
                print("Episode shorter than 200 episodes!")

            if episode_nr % batch_size == 0:  # batch is done
                rmsprop1 = gamma * rmsprop1 + (1 - gamma) * gradient1**2
                rmsprop2 = gamma * rmsprop2 + (1 - gamma) * gradient2**2
                w1 += learning_rate * gradient1 / (np.sqrt(rmsprop1) + 1e-5)
                w2 += learning_rate * gradient2 / (np.sqrt(rmsprop2) + 1e-5)
                gradient1 = np.zeros_like(w1)
                gradient2 = np.zeros_like(w2)
