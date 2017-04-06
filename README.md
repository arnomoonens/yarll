# Deep Reinforcement Learning

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c329c8cdd744463dbda6a726e20f2383)](https://www.codacy.com/app/arnomoonens/DeepRL?utm_source=github.com&utm_medium=referral&utm_content=arnomoonens/DeepRL&utm_campaign=badger)

This code is part of my master thesis at the [VUB](http://www.vub.ac.be), Brussels.

## Status
Different algorithms have already been implemented:
- [Cross-Entropy Method](./agents/CEM.py)
- [Sarsa with with function approximation and eligibility traces](./agents/SarsaFA.py)
- [REINFORCE](./agents/REINFORCE.py) (convolutional neural network part has not been tested yet)
- [Karpathy's policy gradient algorithm](./agents/Karpathy.py) ([version using convolutional neural networks](./agents/Karpathy_CNN.py) has not been tested yet)
- [Advantage Actor Critic](./agents/A2C.py)
- [Asynchronous Advantage Actor Critic (A3C)](./agents/A3C.py)
- [(Sequential) knowledge transfer](./agents/knowledge_transfer.py)
- [Asynchronous knowledge transfer](./agents/async_knowledge_transfer.py)

### Sarsa + function apprixmation
The following parts are combined to learn to act in the [_Mountain Car_ environment](https://gym.openai.com/envs/MountainCar-v0):
- Sarsa
- Eligibility traces
- epsilon-greedy action selection policy
- Function approximation using tile coding

Example of a run after training with a total greedy action selection policy for 729 episodes of each 200 steps:
![Example run](./results/examplerun.gif)

Total reward per episode:
![Total reward per episode](./results/totalrewardperepisode.png)

Note that, after a few thousand episodes, the algorithm still isn't capable of consistently reaching the goal in less than 200 steps.

### REINFORCE
Adapted version of [this code](http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/pg-startercode.py) in order to work with [_Tensorflow_](https://www.tensorflow.org/).
Total reward per episode when applying this algorithm on the [_CartPole-v0_ environment](https://gym.openai.com/envs/CartPole-v0):
![Total reward per episode using REINFORCE](./results/reinforce-cartpole-v0-rewards.png)

### Karpathy Policy Gradient
Adapted version of the code of [this article](http://karpathy.github.io/2016/05/31/rl/) of Andrej Karpathy.
Total reward per episode when applying this algorithm on the [_CartPole-v0_ environment](https://gym.openai.com/envs/CartPole-v0):
![Total reward per episode using Karpathy](./results/karpathy-cartpole-v0-rewards.png)

How quickly the optimal reward is reached and kept heavily varies however because of randomness. Results of an earlier execution are also posted on the [OpenAI Gym](https://gym.openai.com/evaluations/eval_dyl7JQpTXGXY4lIe0pSA).

### Advantage Actor Critic
Total reward per episode when applying this algorithm on the [_CartPole-v0_ environment](https://gym.openai.com/envs/CartPole-v0):
![Total reward per episode using A2C](./results/a2c-cartpole-v0-rewards.png)

[OpenAI Gym page](https://gym.openai.com/evaluations/eval_8lGn053RQref7asqoiPPw)

### Asynchronous Advantage Actor Critic
Total reward per episode when applying this algorithm on the [_CartPole-v0_ environment](https://gym.openai.com/envs/CartPole-v0):
![Total reward per episode using A3C](./results/a3c-cartpole-v0-rewards.png)

This only shows the results of one of the A3C threads.
Results of another execution are also posted on the [OpenAI Gym](https://gym.openai.com/evaluations/eval_deHd1IsvTQeWAnEaSvvkg).
Results of an execution using the _Acrobot-v1_ environment can also be found [on OpenAI Gym](https://gym.openai.com/evaluations/eval_Ig1wrPzQlGipmBAhZ5Tw).
## How to run
First, install the requirements using [pip](https://pypi.python.org/pypi/pip):
```
pip install -r requirements.txt
```

Then you can run the Sarsa + Function approximation algorithm using:
```
python SarsaFA.py <episodes_to_run> <monitor_target_directory>
```

You can run the `CEM`, `REINFORCE`, `Karpathy`, `Karpathy_CNN`, `A2C` and `A3C` algorithm using:
```
python <algorithm_name>.py <environment_name> <monitor_target_directory>
```

You can plot the episode lengths and total reward per episode graphs using:
```
python plot_statistics.py <path_to_stats.json> <moving_average_window>
```
