import os
import json
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Path to the directory.")


def main():
    args = parser.parse_args()
    dirs = sorted([d for d in os.listdir(args.directory) if os.path.isdir(os.path.join(args.directory, d))], key=lambda x: int(x[3:]))
    header = ["RUN", "DESCR", "START", "BRANCH", "COMMITMSG"]
    data = []
    for d in dirs:
        with open(os.path.join(args.directory, d, "config.json")) as f:
            config = json.load(f)
        run_data = [
            d,
            config.get("description", ""),
            config.get("start_time", ""),
        ]
        run_data += [config["git"]["head"], config["git"]["message"]] if "git" in config else [""] * 2
        data.append(run_data)
    df = pd.DataFrame(data, columns=header)
    df.set_index("RUN", inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.width", None, "display.max_colwidth", 100):
        print(df)

if __name__ == '__main__':
    main()
