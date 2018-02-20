#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("envs_dir", type=str, help="Directory with environment specification files")

def main():
    args = parser.parse_args()

    _, _, env_spec_files = next(os.walk(args.envs_dir))
    values = None
    for env_spec_file in env_spec_files:
        with open(os.path.join(args.envs_dir, env_spec_file)) as f:
            envs = json.load(f)
            if values is None:
                values = [{} for _ in range(len(envs))]
            for i, env in enumerate(envs):
                for key, value in env.items():
                    if key == "name":
                        continue
                    values[i].setdefault(key, []).append(value)
    keys = values[0].keys()
    for key in keys:
        fig = plt.figure()
        plt.boxplot([env[key] for env in values])
        plt.title(key)
        fig.canvas.set_window_title(key)
    plt.show()

if __name__ == '__main__':
    main()
