import numpy as np
import gym
import sys
import matplotlib.pyplot as plt


# Adaption of Karpathy's Pong from Pixels article to apply it using a simple neural network on the MountainCar environment

env = gym.make('CartPole-v0')
A = env.action_space
O = env.observation_space

gamma = 0.99
learning_rate = 0.05
batch_size = 10
decay_rate = 0.99

n_hidden_units = 20

draw_frequency = 50  # Draw a plot every 50 episodes

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

w1 = np.random.randn(O.shape[0], n_hidden_units) / np.sqrt(n_hidden_units)
w2 = np.random.randn(n_hidden_units, A.n) / np.sqrt(A.n)

n_episodes = 6000
env.monitor.start(sys.argv[1], force=True)

gradient1 = np.zeros_like(w1)
gradient2 = np.zeros_like(w2)

rmsprop1 = np.zeros_like(w1)
rmsprop2 = np.zeros_like(w2)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

episode_nr = 0
episode_lengths = np.zeros(batch_size)
episode_rewards = np.zeros(batch_size)
mean_rewards = []
while True:  # Keep executing episodes
    state = env.reset()
    # state = scale_state(state)
    done = False
    reward_sum = 0
    trajectory_rewards = []
    action_taken = []
    encountered_states = []
    x1s = []
    while not done:  # Keep executing steps until done
        encountered_states.append(state)
        x1, nn_outputs = forward_step(state, w1, w2)
        # print(nn_outputs)
        action, probabilities = choose_action(nn_outputs)
        # print(probabilities)
        state, reward, done, info = env.step(action)
        # state = scale_state(state)
        reward_sum += reward
        trajectory_rewards.append(reward)
        x1s.append(x1)

        y = np.zeros(A.n)
        y[action] = 1
        action_taken.append(y - probabilities)

        if done:
            episode_rewards[episode_nr % batch_size] = sum(trajectory_rewards)
            episode_lengths[episode_nr % batch_size] = len(trajectory_rewards)
            episode_nr += 1
            epdlogp = np.vstack(action_taken)

            episode_states = np.vstack(encountered_states)

            discounted_episode_rewards = discount_rewards(trajectory_rewards)
            # print(discounted_episode_rewards)
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            discounted_episode_rewards /= np.std(discounted_episode_rewards)
            epdlogp *= np.reshape(np.repeat(discounted_episode_rewards, A.n), (len(discounted_episode_rewards), A.n))
            episode_gradient1 = np.zeros_like(w1)
            episode_gradient2 = np.zeros_like(w2)

            change_w1, change_w2 = backward_step(np.array(encountered_states), np.array(x1s), epdlogp)

            gradient1 += change_w1
            gradient2 += change_w2

            # if(len(encountered_states) < 200):
            #     print("Episode shorter than 200 episodes!")

            if episode_nr % batch_size == 0:  # batch is done
                rmsprop1 = decay_rate * rmsprop1 + (1 - decay_rate) * gradient1**2
                rmsprop2 = decay_rate * rmsprop2 + (1 - decay_rate) * gradient2**2
                w1 += learning_rate * gradient1 / (np.sqrt(rmsprop1) + 1e-5)
                w2 += learning_rate * gradient2 / (np.sqrt(rmsprop2) + 1e-5)
                gradient1 = np.zeros_like(w1)
                gradient2 = np.zeros_like(w2)
                print("Mean episode lengths: \t %s +- %s" % (episode_lengths.mean(), episode_lengths.std() / np.sqrt(len(episode_lengths))))
                print("Max reward: \t %s" % episode_rewards.max())
                mean_episodes_reward = episode_rewards.mean()
                mean_rewards.append(mean_episodes_reward)
                print("Mean reward: \t %s +- %s" % (mean_episodes_reward, episode_rewards.std() / np.sqrt(len(episode_rewards))))
                if episode_nr % draw_frequency == 0:
                    ax1.clear()
                    ax1.plot(range(len(mean_rewards)), mean_rewards)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.show(block=False)
