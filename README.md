# DeepRL
This code is part of my master thesis at the VUB, Brussels.

## Status
Currently, the following parts are combined to learn to act in the [Mountain Car environment](https://gym.openai.com/envs/MountainCar-v0):
- Sarsa
- Eligibility traces
- EGreedy action selection policy
- Function approximation using tile coding

Example of a run after training for 5000 episodes of each 200 steps:
![Example run](examplerun.gif)

## How to run
First, install the requirements using [pip](https://pypi.python.org/pypi/pip):
```
pip install -r requirements.txt
```
Then you can run the experiment using:
```
python MountainCar.py <episodes_to_run> <monitor_target_directory>
```
