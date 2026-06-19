"""
Preprocess data and save df_for_training and df_base for inspection.

Usage
-----
    uv run scripts/preprocess.py --config configs/default.toml --output runs/run_001/

Outputs
-------
    df_for_training.pkl  — lagged feature matrix with targets (input to train.py)
    df_base.pkl          — pre-lag dataframe (input to sample.py)
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import load_config, prepare_data


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess zebrafish data.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    df_for_training, df_base = prepare_data(cfg)

    train_path = args.output / "df_for_training.pkl"
    base_path = args.output / "df_base.pkl"

    with open(train_path, "wb") as f:
        pickle.dump(df_for_training, f)
    print(f"Saved df_for_training → {train_path}  shape={df_for_training.shape}")

    with open(base_path, "wb") as f:
        pickle.dump(df_base, f)
    print(f"Saved df_base → {base_path}  shape={df_base.shape}")


if __name__ == "__main__":
    main()
