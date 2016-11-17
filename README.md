# DeepRL
This code is part of my master thesis at the VUB, Brussels.

## Status
Currently, different algorithms have been implemented:
### Sarsa + function apprixmation
The following parts are combined to learn to act in the [Mountain Car environment](https://gym.openai.com/envs/MountainCar-v0):
- Sarsa
- Eligibility traces
- EGreedy action selection policy
- Function approximation using tile coding

Example of a run after training with a total greedy action selection policy for 729 episodes of each 200 steps:
![Example run](examplerun.gif)

Total reward per episode:
![Total reward per episode](./results/totalrewardperepisode.png)
Note that, after a few thousand episodes, the algorithm still isn't capable of consistently reaching the goal in less than 200 steps.

### REINFORCE
Adapted version of [this code](http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/pg-startercode.py) in order to work with [_Tensorflow_](https://www.tensorflow.org/).
Total reward per episode when applying this algorithm on the _CartPole-v0_ environment:
![Total reward per episode using REINFORCE](./results/reinforce-cartpole-v0-rewards.png)

### Karpathy Policy Gradient
Adapted version of the code of [this article](http://karpathy.github.io/2016/05/31/rl/) of Andrej Karpathy.
Total reward per episode when applying this algorithm on the _CartPole-v0_ environment:
![Total reward per episode using REINFORCE](./results/karpathy-cartpole-v0-rewards.png)
How quickly the optimal reward is reached and kept heavily varies however because of randomness. Results of an earlier execution are also posted on the [OpenAI Gym](https://gym.openai.com/evaluations/eval_dyl7JQpTXGXY4lIe0pSA).
## How to run
First, install the requirements using [pip](https://pypi.python.org/pypi/pip):
```
pip install -r requirements.txt
```
Then you can run the Sarsa + Function approximation using:
```
python MountainCar.py <episodes_to_run> <monitor_target_directory>
```
You can run the REINFORCE algorithm using:
```
python REINFORCE.py <environment_name> <monitor_target_directory>
```
You can run the Kartpathy policy gradient algorithm (applied to the _CartPole-v0_ environment) using:
```
python KarpathyLearner.py <monitor_target_directory>
```
