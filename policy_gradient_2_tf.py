import numpy as np
import sys
import tensorflow as tf
import gym
import math
from gym.spaces import Discrete, Box
from ActionSelection.CategoricalActionSelection import ProbabilisticCategoricalActionSelection

np.set_printoptions(suppress=True)  # Don't use the scientific notation to print results

env = gym.make('CartPole-v0')
A = env.action_space
O = env.observation_space

# episodes_per_batch = 10

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x), 'float64')
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

def get_trajectory(agent, env, episode_max_length, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    ob = env.reset()
    obs = []
    actions = []
    rewards = []
    for _ in range(episode_max_length):
        action = agent.act(ob)
        (ob, rew, done, _) = env.step(action)
        obs.append(ob)
        actions.append(action)
        rewards.append(rew)
        if done:
            break
        if render:
            env.render()
    return {"reward": np.array(rewards),
            "ob": np.array(obs),
            "action": np.array(actions)
            }


class QACLearner(object):
    """Q-value Actor-Critic"""
    def __init__(self, ob_space, action_space, action_selection, **usercfg):
        self.nO = ob_space.shape[0]
        self.nA = action_space.n
        self.action_selection = action_selection

        self.config = dict(
            episode_max_length=100,
            timesteps_per_batch=10000,
            n_iter=100,
            gamma=0.99,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-4,
            actor_n_hidden_units=4)
        self.config.update(usercfg)

        self.actor_input = tf.placeholder(tf.float32, name='actor_input')
        self.actions_taken = tf.placeholder(tf.float32, name='actions_taken')
        self.critic_feedback = tf.placeholder(tf.float32, name='critic_feedback')
        self.critic_input = tf.placeholder(tf.float32, name='critic_input')
        self.critic_target = tf.placeholder(tf.float32, name='critic_target')
        self.critic_rewards = tf.placeholder(tf.float32, name='critic_rewards')

        # Actor network
        W0 = tf.Variable(tf.random_normal([self.nO, self.config['actor_n_hidden_units']]), name='W0')
        b0 = tf.Variable(tf.zeros([self.config['actor_n_hidden_units']]), name='b0')
        L1 = tf.tanh(tf.matmul(self.actor_input, W0) + b0[None, :], name='L1')

        W1 = tf.Variable(tf.random_normal([self.config['actor_n_hidden_units'], self.nA]), name='W1')
        b1 = tf.Variable(tf.zeros([self.nA]), name='b1')
        self.prob_na = tf.nn.softmax(tf.matmul(L1, W1) + b1[None, :], name='prob_na')

        # actor_parameters = [W0, b0, W1, b1]

        good_probabilities = tf.reduce_sum(tf.mul(self.prob_na, tf.one_hot(tf.cast(self.actions_taken, tf.int32), A.n)), reduction_indices=[1])
        eligibility = -tf.log(good_probabilities + [1e-8]) * (self.critic_rewards - self.critic_feedback)  # critic_feedback should be returns - value
        # eligibility = tf.Print(eligibility, [eligibility], first_n=5)
        loss = tf.reduce_mean(eligibility)
        # parameters_gradients = tf.gradients(loss, actor_parameters, -critic_feedback)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['actor_learning_rate'], decay=0.9, epsilon=1e-9)
        self.actor_train = optimizer.minimize(loss)

        # critic_W0 = tf.Variable(tf.random_normal([self.nO, self.config['actor_n_hidden_units']]), name='critic_W0')
        # critic_b0 = tf.Variable(tf.zeros([self.config['actor_n_hidden_units']]), name='critic_b0')
        # critic_L1 = tf.tanh(tf.matmul(actor_input, W0) + b0[None, :], name='critic_L1')

        # critic_W1 = tf.Variable(tf.random_normal([self.config['actor_n_hidden_units'], n_output_units]), name='critic_W1')
        # critic_b1 = tf.Variable(tf.zeros([n_output_units]), name='critic_b1')
        # critic_output = tf.matmul(L1, W1) + b1[None, :]

        # critic_loss = tf.reduce_sum(tf.squared_difference(critic_output, critic_target))
        # critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=critic_learning_rate, decay=0.9, epsilon=1e-9)
        # critic_train = optimizer.minimize(critic_loss)

        N_HIDDEN_1 = 4
        N_HIDDEN_2 = 3
        self.critic_state_in = tf.placeholder("float", [None, self.nO], name='critic_state_in')
        self.critic_action_in = tf.placeholder("float", [None, A.n], name='critic_action_in')

        critic_W1 = tf.Variable(
            tf.random_uniform([self.nO, N_HIDDEN_1], -1 / math.sqrt(self.nO), 1 / math.sqrt(self.nO)))
        critic_B1 = tf.Variable(tf.random_uniform([N_HIDDEN_1], -1 / math.sqrt(self.nO), 1 / math.sqrt(self.nO)))
        critic_W2 = tf.Variable(tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + A.n),
                                1 / math.sqrt(N_HIDDEN_1 + A.n)))
        W2_acticritic_on = tf.Variable(
            tf.random_uniform([A.n, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + A.n),
                              1 / math.sqrt(N_HIDDEN_1 + A.n)))
        critic_B2 = tf.Variable(tf.random_uniform([N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + A.n),
                                1 / math.sqrt(N_HIDDEN_1 + A.n)))
        critic_W3 = tf.Variable(tf.random_uniform([N_HIDDEN_2, 1], -0.003, 0.003))
        critic_B3 = tf.Variable(tf.random_uniform([1], -0.003, 0.003))

        critic_H1 = tf.nn.softplus(tf.matmul(self.critic_state_in, critic_W1) + critic_B1)
        critic_H2 = tf.nn.tanh(tf.matmul(critic_H1, critic_W2) + tf.matmul(self.critic_action_in, W2_acticritic_on) + critic_B2)

        self.critic_q_model = tf.matmul(critic_H2, critic_W3) + critic_B3

        critic_loss = tf.reduce_sum(tf.squared_difference(self.critic_q_model, self.critic_target))
        critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['critic_learning_rate'], decay=0.9, epsilon=1e-9)
        self.critic_train = critic_optimizer.minimize(critic_loss)

        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, ob):
        """Choose an action."""
        ob = ob.reshape(1, -1)
        prob = self.sess.run([self.prob_na], feed_dict={self.actor_input: ob})[0][0]
        action = self.action_selection.select_action(prob)
        # action = categorical_sample_max(prob)
        return action

    def get_critic_value(self, ob, ac):
        ob = ob.reshape(1, -1)
        return self.sess.run([self.critic_q_model], feed_dict={self.critic_state_in: ob, self.critic_action_in: ac})[0][0]

    def learn(self, save_path):
        env.monitor.start(save_path, force=True)
        while(True):  # for each episode
            state = env.reset()
            action = A.sample()
            # t = 0
            done = False
            while not(done):
                # print(action)
                new_state, reward, done, info = env.step(action)
                new_action = self.act(new_state)
                # print(probabilities)
                # qw_old = qw_predict_fn([state])[0, 0]
                onehot_action = np.zeros((1, A.n))
                onehot_action[0, action] = 1
                qw_new = self.get_critic_value(state, onehot_action)[0]
                # theta, z = SARSALambdaLinFApp(state, action, reward, new_state, new_action, theta, z)
                # phi = actor.differential_log(state, action)
                # v = np.sum(actor.weights(new_state) * theta.T * psi(state))
                qw_target = reward + self.config['gamma'] * qw_new
                onehot_new_action = np.zeros((1, A.n))
                onehot_new_action[0, new_action] = 1
                self.sess.run([self.critic_train], feed_dict={self.critic_state_in: new_state.reshape(1, -1), self.critic_target: qw_target, self.critic_action_in: onehot_new_action})

                # train_fn([state], (onehot_new_action - probabilities) * error)
                self.sess.run([self.actor_train], feed_dict={self.actor_input: state.reshape(1, -1), self.actions_taken: [new_action], self.critic_feedback: qw_new, self.critic_rewards: [reward]})
                state = new_state
                action = new_action

    def learn2(self, env):
        """Run learning algorithm"""
        config = self.config
        for iteration in range(config["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajectories = []
            timesteps_total = 0
            while timesteps_total < config["timesteps_per_batch"]:
                trajectory = get_trajectory(self, env, config["episode_max_length"])
                trajectories.append(trajectory)
                timesteps_total += len(trajectory["reward"])
            all_ob = np.concatenate([trajectory["ob"] for trajectory in trajectories])
            # Compute discounted sums of rewards
            rets = [discount(trajectory["reward"], config["gamma"]) for trajectory in trajectories]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]
            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_adv = np.concatenate(advs)
            # Do policy gradient update step
            self.sess.run([self.train], feed_dict={self.ob_no: all_ob, self.a_n: all_action, self.adv_n: all_adv, self.N: len(all_ob)})
            episode_rewards = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # episode total rewards
            episode_lengths = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # episode lengths
            # Print stats
            print("-----------------")
            print("Iteration: \t %i" % iteration)
            print("NumTrajs: \t %i" % len(episode_rewards))
            print("NumTimesteps: \t %i" % np.sum(episode_lengths))
            print("MaxRew: \t %s" % episode_rewards.max())
            print("MeanRew: \t %s +- %s" % (episode_rewards.mean(), episode_rewards.std() / np.sqrt(len(episode_rewards))))
            print("MeanLen: \t %s +- %s" % (episode_lengths.mean(), episode_lengths.std() / np.sqrt(len(episode_lengths))))
            print("-----------------")
            get_trajectory(self, env, config["episode_max_length"], render=True)


def main():
    if(len(sys.argv) < 3):
        print("Please provide the name of an environment and a path to save monitor files")
        return
    env = gym.make(sys.argv[1])
    if isinstance(env.action_space, Discrete):
        action_selection = ProbabilisticCategoricalActionSelection()
        agent = QACLearner(env.observation_space, env.action_space, action_selection, episode_max_length=env.spec.timestep_limit)
    else:
        raise NotImplementedError
    try:
        agent.learn(sys.argv[2])
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
