import json
import sys
import matplotlib.pyplot as plt

def main(stats_path):
    with open(stats_path) as json_file:
        data = json.load(json_file)
        plt.plot(data['episode_rewards'], 'ro')
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.title("Total reward per episode")
        plt.show()

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("Please provide the path of the stats JSON file")
    else:
        main(sys.argv[1])
