import argparse
import json
from os import path
from pathlib import Path

from rich.console import Console
from rich.table import Table

IGNORE_KEYS = set([
    "python_version",
    "program_command",
    "agent_class",
    "packages",
    "git",
    "start_time",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", type=Path, nargs='+')
    args = parser.parse_args()

    config_paths = args.config_paths
    common_path = Path(path.commonpath(config_paths))

    console = Console()
    table = Table(show_header=True)
    table.add_column("Key")
    all_keys = set()
    configs = []
    for cp in config_paths:
        table.add_column(str(cp.relative_to(common_path)))
        with open(cp, "r") as f:
            config = json.load(f)
            configs.append(config)
            all_keys = all_keys.union(set(config.keys()).difference(IGNORE_KEYS))

    for k in all_keys:
        show_key = False
        values = []
        for config in configs:
            val = config.get(k, None)
            if values and values[-1] != val:
                show_key = True
            values.append(val)
        if show_key:
            table.add_row(k, *[str(x) for x in values])

    console.print(table)

if __name__ == '__main__':
    main()
