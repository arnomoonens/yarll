import matplotlib.pyplot as plt
import numpy as np
import sys
import json

# Source: http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy?rq=1
def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main(stats_path, n):
    f = open(stats_path)
    contents = json.load(f)

    averaged_episode_rewards = moving_average(contents['episode_rewards'], n)
    fig = plt.figure()
    plt.plot(range(len(averaged_episode_rewards)), averaged_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Total reward per episode")
    fig.canvas.set_window_title("Total reward per episode")

    fig = plt.figure()
    averaged_episode_lengths = moving_average(contents['episode_lengths'], n)
    plt.plot(range(len(averaged_episode_lengths)), averaged_episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Length per episode")
    fig.canvas.set_window_title("Length per episode")
    plt.show()

if __name__ == '__main__':
    if(len(sys.argv) < 3):
        print("Please provide the path of the stats JSON file and a running mean length")
    else:
        main(sys.argv[1], int(sys.argv[2]))
