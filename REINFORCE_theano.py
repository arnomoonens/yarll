#  Policy Gradient Implementation
#  Original Theano version
#  Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab2.html

import numpy as np
import os
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float64"
import theano
import theano.tensor as T
import gym

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

def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()

def get_traj(agent, env, episode_max_length, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    ob = env.reset()
    obs = []
    acts = []
    rews = []
    for _ in range(episode_max_length):
        a = agent.act(ob)
        (ob, rew, done, _) = env.step(a)
        obs.append(ob)
        acts.append(a)
        rews.append(rew)
        if done:
            break
        if render:
            env.render()
    return {"reward": np.array(rews),
            "ob": np.array(obs),
            "action": np.array(acts)
            }

def sgd_updates(grads, params, stepsize):
    """
    Create list of parameter updates from stochastic gradient ascent
    """
    updates = []
    for (param, grad) in zip(params, grads):
        updates.append((param, param + stepsize * grad))
    return updates

def rmsprop_updates(grads, params, stepsize, rho=0.9, epsilon=1e-9):
    """
    Create a list of parameter updates from RMSProp
    """
    updates = []

    for param, grad in zip(params, grads):
        accum = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=param.dtype))
        accum_new = rho * accum + (1 - rho) * grad ** 2
        updates.append((accum, accum_new))
        updates.append((param, param + (stepsize * grad / T.sqrt(accum_new + epsilon))))
    return updates

class REINFORCEAgent(object):

    """
    REINFORCE with baselines
    Currently just works for discrete action space
    """

    def __init__(self, ob_space, action_space, **usercfg):
        """
        Initialize your agent's parameters
        """
        nO = ob_space.shape[0]
        nA = action_space.n
        # Here are all the algorithm parameters. You can modify them by passing in keyword args
        self.config = dict(episode_max_length=100, timesteps_per_batch=10000, n_iter=100, gamma=1.0, stepsize=0.05, nhid=20)
        self.config.update(usercfg)

        # Symbolic variables for observation, action, and advantage
        # These variables stack the results from many timesteps--the first dimension is the timestep
        ob_no = T.fmatrix()  # Observation
        a_n = T.ivector()  # Discrete action
        adv_n = T.fvector()  # Advantage

        def shared(arr):
            return theano.shared(arr.astype('float64'))
        # Create weights of neural network with one hidden layer
        W0 = shared(np.random.randn(nO, self.config['nhid']) / np.sqrt(nO))
        b0 = shared(np.zeros(self.config['nhid']))
        W1 = shared(1e-4 * np.random.randn(self.config['nhid'], nA))
        b1 = shared(np.zeros(nA))
        params = [W0, b0, W1, b1]
        # Action probabilities
        prob_na = T.nnet.softmax(T.tanh(ob_no.dot(W0) + b0[None, :]).dot(W1) + b1[None, :])
        N = ob_no.shape[0]
        # Loss function that we'll differentiate to get the policy gradient
        # Note that we've divided by the total number of timesteps
        loss = T.log(prob_na[T.arange(N), a_n]).dot(adv_n) / N
        stepsize = T.fscalar()
        grads = T.grad(loss, params)
        # Perform parameter updates.
        # I find that sgd doesn't work well
        # updates = sgd_updates(grads, params, stepsize)
        updates = rmsprop_updates(grads, params, stepsize)
        self.pg_update = theano.function([ob_no, a_n, adv_n, stepsize], [], updates=updates, allow_input_downcast=True)
        self.compute_prob = theano.function([ob_no], prob_na, allow_input_downcast=True)

    def act(self, ob):
        """
        Choose an action.
        """
        prob = self.compute_prob(ob.reshape(1, -1))
        action = categorical_sample(prob)
        return action

    def learn(self, env):
        """
        Run learning algorithm
        """
        cfg = self.config
        for iteration in range(cfg["n_iter"]):
            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajs = []
            timesteps_total = 0
            while timesteps_total < cfg["timesteps_per_batch"]:
                traj = get_traj(self, env, cfg["episode_max_length"])
                trajs.append(traj)
                timesteps_total += len(traj["reward"])
            all_ob = np.concatenate([traj["ob"] for traj in trajs])
            # Compute discounted sums of rewards
            rets = [discount(traj["reward"], cfg["gamma"]) for traj in trajs]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]
            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)
            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action = np.concatenate([traj["action"] for traj in trajs])
            all_adv = np.concatenate(advs)
            # Do policy gradient update step
            self.pg_update(all_ob, all_action, all_adv, cfg["stepsize"])
            eprews = np.array([traj["reward"].sum() for traj in trajs])  # episode total rewards
            eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths
            # Print stats
            print("-----------------")
            print("Iteration: \t %i" % iteration)
            print("NumTrajs: \t %i" % len(eprews))
            print("NumTimesteps: \t %i" % np.sum(eplens))
            print("MaxRew: \t %s" % eprews.max())
            print("MeanRew: \t %s +- %s" % (eprews.mean(), eprews.std() / np.sqrt(len(eprews))))
            print("MeanLen: \t %s +- %s" % (eplens.mean(), eplens.std() / np.sqrt(len(eplens))))
            print("-----------------")
            get_traj(self, env, cfg["episode_max_length"], render=True)


def main():
    env = gym.make("CartPole-v0")
    agent = REINFORCEAgent(env.observation_space, env.action_space, episode_max_length=env.spec.timestep_limit)
    agent.learn(env)

if __name__ == "__main__":
    main()
