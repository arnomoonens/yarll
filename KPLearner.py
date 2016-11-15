import numpy as np
import gym
import sys
import matplotlib.pyplot as plt

from utils import discount_rewards
from gym.spaces import Discrete, Box

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

def choose_action(output, temperature=1.0):
    # total = sum([np.exp(float(o) / temperature) for o in output])
    # probs = [np.exp(float(o) / temperature) / total for o in output]
    probs = output / np.sum(output)
    action = np.random.choice(A.n, p=probs)
    return action, probs

def get_trajectory(agent, env, episode_max_length, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    ob = env.reset()
    obs = []
    actions = []
    rewards = []
    episode_probabilities = []
    x1s = []
    for _ in range(episode_max_length):
        action, probabilities, x1 = agent.act(ob)
        x1s.append(x1)
        obs.append(ob)
        (ob, rew, done, _) = env.step(action)
        actions.append(action)
        rewards.append(rew)
        episode_probabilities.append(probabilities)
        if done:
            break
        if render:
            env.render()
    return {"reward": np.array(rewards),
            "ob": np.array(obs),
            "action": np.array(actions),
            "prob": np.array(episode_probabilities),
            "x1": np.array(x1s)
            }

class KPLearner(object):
    """Karpathy policy gradient learner"""
    def __init__(self, ob_space, action_space, **usercfg):
        super(KPLearner, self).__init__()
        self.nO = ob_space.shape[0]
        self.nA = action_space.n
        # Default configuration. Can be overwritten using keyword arguments.
        self.config = dict(
            # episode_max_length=100,
            # timesteps_per_batch=10000,
            # n_iter=100,
            gamma=0.99,
            stepsize=0.05,
            n_hidden_units=20)
        self.config.update(usercfg)

        self.w1 = np.random.randn(O.shape[0], n_hidden_units) / np.sqrt(n_hidden_units)
        self.w2 = np.random.randn(n_hidden_units, A.n) / np.sqrt(A.n)

        self.n_episodes = 6000

    def act(self, ob):
        x1, nn_outputs = self.forward_step(ob)
        action, probabilities = choose_action(nn_outputs)
        return action, probabilities, x1

    def forward_step(self, state):
        x1 = np.dot(state, self.w1)
        x1[x1 < 0] = 0
        output = sigmoid(np.dot(x1, self.w2))
        return x1, output

    def backward_step(self, x0, x1, feedback):
        """Computes how much to change the weights from input->hidden layer and hidden->final layer"""
        change_w2 = np.dot(x1.T, feedback)  # 8x200 * 200x3 = 8x3
        dh = np.dot(feedback, self.w2.T)  # 200x3 * 3x8 = 200x8
        dh[x1 <= 0] = 0
        change_w1 = x0.T.dot(dh)  # 2x200 * 200x8 = 2x8
        return change_w1, change_w2

    def learn(self, env):
        # env.monitor.start(sys.argv[1], force=True)

        gradient1 = np.zeros_like(self.w1)
        gradient2 = np.zeros_like(self.w2)

        rmsprop1 = np.zeros_like(self.w1)
        rmsprop2 = np.zeros_like(self.w2)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        episode_nr = 0
        episode_lengths = np.zeros(batch_size)
        episode_rewards = np.zeros(batch_size)
        mean_rewards = []
        while True:  # Keep executing episodes
            # state = scale_state(state)
            # while not done:  # Keep executing steps until done
            #     encountered_states.append(state)
            #     # x1, nn_outputs = forward_step(state, w1, w2)
            #     # action, probabilities = choose_action(nn_outputs)
            #     state, reward, done, info = env.step(action)
            #     # state = scale_state(state)
            #     reward_sum += reward
            #     trajectory_rewards.append(reward)
            #     x1s.append(x1)

            #     y = np.zeros(A.n)
            #     y[action] = 1
            #     action_taken.append(y - probabilities)
            trajectory = get_trajectory(self, env, self.config["episode_max_length"])

            episode_rewards[episode_nr % batch_size] = sum(trajectory['reward'])
            episode_lengths[episode_nr % batch_size] = len(trajectory['reward'])
            episode_nr += 1
            action_taken = (np.arange(A.n) == trajectory['action'][:, None]).astype(np.float32)
            epdlogp = action_taken - trajectory['prob']

            # episode_states = np.vstack(encountered_states)

            discounted_episode_rewards = discount_rewards(trajectory['reward'], self.config['gamma'])
            # print(discounted_episode_rewards)
            # standardize
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            discounted_episode_rewards /= np.std(discounted_episode_rewards)
            epdlogp *= np.reshape(np.repeat(discounted_episode_rewards, A.n), (len(discounted_episode_rewards), A.n))
            # episode_gradient1 = np.zeros_like(self.w1)
            # episode_gradient2 = np.zeros_like(self.w2)

            change_w1, change_w2 = self.backward_step(trajectory['ob'], trajectory['x1'], epdlogp)

            gradient1 += change_w1
            gradient2 += change_w2

            # if(len(encountered_states) < 200):
            #     print("Episode shorter than 200 episodes!")

            if episode_nr % batch_size == 0:  # batch is done
                rmsprop1 = decay_rate * rmsprop1 + (1 - decay_rate) * gradient1**2
                rmsprop2 = decay_rate * rmsprop2 + (1 - decay_rate) * gradient2**2
                self.w1 += learning_rate * gradient1 / (np.sqrt(rmsprop1) + 1e-5)
                self.w2 += learning_rate * gradient2 / (np.sqrt(rmsprop2) + 1e-5)
                gradient1 = np.zeros_like(self.w1)
                gradient2 = np.zeros_like(self.w2)
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

def main():
    if(len(sys.argv) < 3):
        print("Please provide the name of an environment and a path to save monitor files")
        return
    env = gym.make(sys.argv[1])
    print(env.action_space)
    if isinstance(env.action_space, Discrete):
        # action_selection = ProbabilisticCategoricalActionSelection()
        agent = KPLearner(env.observation_space, env.action_space, episode_max_length=env.spec.timestep_limit)
    elif isinstance(env.action_space, Box):
        raise NotImplementedError
    else:
        raise NotImplementedError
    try:
        env.monitor.start(sys.argv[2], force=True)
        agent.learn(env)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
