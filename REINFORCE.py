#  Policy Gradient Implementation
#  Adapted for Tensorflow
#  Other differences:
#  - Always choose the action with the highest probability
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

import numpy as np
import sys
import tensorflow as tf
import gym
from gym.spaces import Discrete, Box
from ActionSelection.CategoricalActionSelection import ProbabilisticCategoricalActionSelection

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

class REINFORCEAgent(object):
    """
    REINFORCE with baselines
    """

    def __init__(self, ob_space, action_space, action_selection, **usercfg):
        self.nO = ob_space.shape[0]
        self.nA = action_space.n
        self.action_selection = action_selection
        # Default configuration. Can be overwritten using keyword arguments.
        self.config = dict(
            episode_max_length=100,
            timesteps_per_batch=10000,
            n_iter=100,
            gamma=1.0,
            stepsize=0.05,
            nhid=20)
        self.config.update(usercfg)

    def act(self, ob):
        """Choose an action."""
        pass

    def learn(self, env):
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
            # get_trajectory(self, env, config["episode_max_length"], render=True)

class REINFORCEAgentDiscrete(REINFORCEAgent):

    def __init__(self, ob_space, action_space, action_selection, **usercfg):
        super(REINFORCEAgentDiscrete, self).__init__(ob_space, action_space, action_selection, **usercfg)

        # Symbolic variables for observation, action, and advantage
        # These variables stack the results from many timesteps--the first dimension is the timestep
        self.ob_no = tf.placeholder(tf.float32, name='ob_no')  # Observation
        self.a_n = tf.placeholder(tf.float32, name='a_n')  # Discrete action
        self.adv_n = tf.placeholder(tf.float32, name='adv_n')  # Advantage

        W0 = tf.Variable(tf.random_normal([self.nO, self.config['nhid']]) / np.sqrt(self.nO), name='W0')
        b0 = tf.Variable(tf.zeros([self.config['nhid']]), name='b0')
        W1 = tf.Variable(1e-4 * tf.random_normal([self.config['nhid'], self.nA]), name='W1')
        b1 = tf.Variable(tf.zeros([self.nA]), name='b1')
        # Action probabilities
        L1 = tf.tanh(tf.matmul(self.ob_no, W0) + b0[None, :])
        self.prob_na = tf.nn.softmax(tf.matmul(L1, W1) + b1[None, :], name='prob_na')
        # N = self.ob_no.get_shape()[0].value
        self.N = tf.placeholder(tf.int32, name='N')
        # Loss function that we'll differentiate to get the policy gradient
        # Note that we've divided by the total number of timesteps
        # loss = T.log(prob_na[T.arange(N), self.a_n]).dot(self.adv_n) / N
        good_probabilities = tf.reduce_sum(tf.mul(self.prob_na, tf.one_hot(tf.cast(self.a_n, tf.int32), self.nA)), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * self.adv_n
        eligibility = tf.Print(eligibility, [eligibility], first_n=5)
        loss = -tf.reduce_sum(eligibility)
        # stepsize = tf.placeholder(tf.float32)
        # grads = T.grad(loss, params)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['stepsize'], decay=0.9, epsilon=1e-9)
        self.train = optimizer.minimize(loss)

        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, ob):
        """Choose an action."""
        ob = ob.reshape(1, -1)
        prob = self.sess.run([self.prob_na], feed_dict={self.ob_no: ob})[0][0]
        action = self.action_selection.select_action(prob)
        # action = categorical_sample_max(prob)
        return action

def main():
    if(len(sys.argv) < 3):
        print("Please provide the name of an environment and a path to save monitor files")
        return
    env = gym.make(sys.argv[1])
    if isinstance(env.action_space, Discrete):
        action_selection = ProbabilisticCategoricalActionSelection()
        agent = REINFORCEAgentDiscrete(env.observation_space, env.action_space, action_selection, episode_max_length=env.spec.timestep_limit)
    else:
        raise NotImplementedError
    try:
        env.monitor.start(sys.argv[2], force=True)
        agent.learn(env)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
