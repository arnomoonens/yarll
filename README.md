# Deep Reinforcement Learning

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c329c8cdd744463dbda6a726e20f2383)](https://www.codacy.com/app/arnomoonens/DeepRL?utm_source=github.com&utm_medium=referral&utm_content=arnomoonens/DeepRL&utm_campaign=badger)

This code is part of [my master thesis](https://github.com/arnomoonens/Master-Thesis) at the [VUB](http://www.vub.ac.be), Brussels.

## Status

Different algorithms have currently been implemented:

- [Advantage Actor Critic](./agents/actorcritic/a2c.py)
- [Asynchronous Advantage Actor Critic (A3C)](./agents/actorcritic/a3c.py)
- [Deep Deterministic Policy Gradient (DDPG)](./agents/ddpg.py)
- [Proximal Policy Optimization (PPO)](./agents/ppo/ppo.py)
- [Distributed Policy Optimization (DPPO)](./agents/ppo/dppo.py)
- [Trust Region Policy Optimization (TRPO)](./agents/trpo/trpo.py)
- [Distributed Trust Region Policy Optimization (DTRPO)](./agents/trpo/dtrpo.py)
- [REINFORCE](./agents/reinforce.py) (convolutional neural network part has not been tested yet)
- [Cross-Entropy Method](./agents/cem.py)
- [Sarsa with with function approximation and eligibility traces](./agents/sarsa/sarsa_fa.py)
- [Karpathy's policy gradient algorithm](./agents/karpathy.py) ([version using convolutional neural networks](./agents/karpathy_cnn.py) has not been tested yet)
- [(Sequential) knowledge transfer](./agents/knowledgetransfer/knowledge_transfer.py)
- [Asynchronous knowledge transfer](./agents/knowledgetransfer/async_knowledge_transfer.py)

## Asynchronous Advantage Actor Critic (A3C)

The code for this algorithm can be found [here](./agents/actorcritic/a3c.py).
Example run after training using 16 threads for a total of 5 million timesteps on the _PongDeterministic-v4_ environment:

![Pong example run](./results/pong.gif)

## How to run

First, install the requirements using [pip](https://pypi.python.org/pypi/pip):

```Shell

pip install -r requirements.txt

```

### Algorithms/experiments

You can run algorithms by passing an experiment specification (in _json_ format) to `main.py`:

```Shell

python main.py <experiment_description>

```

[Example of an experiment specification](./experiment_spec_example.json)

### Statistics

Statistics can be plot using:

```Shell

python misc/plot_statistics.py <path_to_stats>

```

`<path_to_stats>` can be one of 2 things:

- A _json_ file generated using `gym.wrappers.Monitor`, in case it plots the episode lengths and total reward per episode.
- A directory containing _TensorFlow_ scalar summaries for different tasks, in which case all of the found scalars are plot.

Help about other arguments (e.g. for using smoothing) can be found by executing `python misc/plot_statistics.py -h`.

Alternatively, it is also possible to use [_Tensorboard_](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to show statistics in the browser by passing the directory with the scalar summaries as `--logdir` argument.
