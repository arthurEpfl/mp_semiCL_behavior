"""
Build abs_dict.pkl from pre-transformed per-fish CSV files.

Usage
-----
    uv run scripts/build_abs_dict.py \\
        --input  data/df_transformed/ \\
        --output abs_dict.pkl

The script expects files matching the pattern *_f{N}_transformed_abs.csv in --input.

The resulting abs_dict is a dict {"f0": df_f0, ..., "f14": df_f14} ready to be
passed to stack_dict_pipeline() in train.py. Dish geometry (center + radius) is
specified in the config file, not inferred here.
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/df_transformed"),
                   help="Directory containing *_f{N}_transformed_abs.csv files.")
    p.add_argument("--output", type=Path, default=Path("abs_dict.pkl"),
                   help="Output path for abs_dict.pkl.")
    return p.parse_args()


def main():
    args = parse_args()

    csvs = sorted(args.input.glob("*_transformed_abs.csv"))
    if not csvs:
        print(f"ERROR: No *_transformed_abs.csv files found in {args.input}")
        sys.exit(1)

    abs_dict = {}

    for path in csvs:
        m = re.search(r"_(f\d+)_transformed_abs", path.name)
        if not m:
            print(f"  Skipping {path.name}: cannot extract fish key.")
            continue

        fish_key = m.group(1)
        print(f"Loading {path.name} → {fish_key}")
        abs_dict[fish_key] = pd.read_csv(path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(abs_dict, f)
    print(f"\nSaved abs_dict ({len(abs_dict)} fish) → {args.output}")


if __name__ == "__main__":
    main()
