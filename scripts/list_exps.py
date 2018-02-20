import os
import json
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Path to the directory.")


def main():
    args = parser.parse_args()
    dirs = sorted([d for d in os.listdir(args.directory) if os.path.isdir(os.path.join(args.directory, d))])
    header = ["RUN", "DESCR", "START"]
    data = []
    for d in dirs:
        with open(os.path.join(args.directory, d, "config.json")) as f:
            config = json.load(f)
        data.append([d, config["description"], config["start_time"]])
    df = pd.DataFrame(data, columns=header)
    df.set_index("RUN", inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.width", None, "display.max_colwidth", 100):
        print(df)

if __name__ == '__main__':
    main()
