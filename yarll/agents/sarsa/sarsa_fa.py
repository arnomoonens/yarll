# -*- coding: utf8 -*-

from gym import wrappers

from yarll.policies.e_greedy import EGreedy
from yarll.agents.sarsa import Sarsa
from yarll.traces.eligibility_traces import EligibilityTraces
from yarll.functionapproximation.tile_coding import TileCoding

# def draw_3d(tile_starts):
#     states = []
#     for i in range(n_x_tiles):
#         for j in range(n_y_tiles):
#             states.append([i, j])
#     states = np.array(states)

class SarsaFA(object):
    """Learner using Sarsa and function approximation"""
    def __init__(self, env, monitor_path: str, video: bool = True, **usercfg) -> None:
        super(SarsaFA, self).__init__()
        self.env = env
        self.env = wrappers.Monitor(self.env, monitor_path, force=True, video_callable=(None if video else False))
        m = usercfg.get("m", 10)  # Number of tilings
        self.config = dict(
            m=m,
            n_x_tiles=9,
            n_y_tiles=9,
            Lambda=0.9,
            epsilon=0,  # fully greedy in this case
            alpha=(0.05 * (0.5 / m)),
            gamma=1,
            n_iter=1000,
            steps_per_episode=env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps")
        )
        self.config.update(usercfg)
        O = env.observation_space
        self.x_low, self.y_low = O.low
        self.x_high, self.y_high = O.high

        self.nA = env.action_space.n
        self.policy = EGreedy(self.config["epsilon"])
        self.function_approximation = TileCoding(self.x_low, self.x_high,
                                                 self.y_low, self.y_high,
                                                 m,
                                                 int(self.config["n_x_tiles"]), int(self.config["n_y_tiles"]),
                                                 self.nA)

    def learn(self):
        for i in range(int(self.config["n_iter"])):
            traces = EligibilityTraces(self.function_approximation.features_shape,
                                       self.config["gamma"],
                                       self.config["Lambda"])
            state, action = self.env.reset(), 0
            sarsa = Sarsa(self.config["gamma"],
                          self.config["alpha"],
                          self.policy,
                          traces,
                          self.function_approximation,
                          range(self.nA), state, action)
            done = False
            iteration = 0
            while not done:
                iteration += 1
                state, reward, done, _ = self.env.step(action)
                if done and iteration < self.config["steps_per_episode"]:
                    print("Episode {}: Less than {} steps were needed: {}".format(i,
                                                                                  self.config["steps_per_episode"],
                                                                                  iteration))
                action = sarsa.step(state, reward)
