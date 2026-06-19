"""
Generate synthetic zebrafish trajectories from a trained checkpoint.

Usage
-----
    uv run scripts/sample.py \\
        --checkpoint runs/run_001/checkpoint.pt \\
        --base      runs/run_001/df_base.pkl \\
        --output    runs/run_001/samples/ \\
        --n_trials  5

Each output CSV has columns matching df_base (f_x, f_y, f_ori, f_speed, d_ori,
d{k}_x/y/stim, ...) and represents one synthetic trajectory segment.

The seed window (first max_lag_seconds of each segment) contains real fish data;
only the remainder is synthetic.
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_NN import load_checkpoint
from src.sampling import sample_trajectory


def parse_args():
    p = argparse.ArgumentParser(description="Sample synthetic zebrafish trajectories.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--base", required=True, type=Path, help="Path to df_base.pkl from train.py.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n_trials", type=int, default=5, help="Number of trial segments to sample.")
    p.add_argument(
        "--trial_length_s",
        type=float,
        default=60.0,
        help="Length of each trial segment in seconds (default: 60).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load checkpoint
    # ------------------------------------------------------------------ #
    print(f"Loading checkpoint from {args.checkpoint} ...")
    model, scalers, config, feature_cols = load_checkpoint(args.checkpoint)
    model.eval()

    data_cfg = config["data"]
    cols_to_lag = data_cfg["cols_to_lag"]
    horizon_frames = data_cfg.get("horizon_frames", 1)
    subsample = data_cfg.get("subsample", 1)
    max_lag_seconds = data_cfg.get("max_lag_seconds", 3)
    fps = data_cfg.get("fps", 30)

    # ------------------------------------------------------------------ #
    # 2. Load base dataframe
    # ------------------------------------------------------------------ #
    print(f"Loading df_base from {args.base} ...")
    with open(args.base, "rb") as f:
        df_base = pickle.load(f)

    df_base = df_base.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 3. Extract dish geometry (constant per fish; use first row)
    # ------------------------------------------------------------------ #
    dish_x = float(df_base["dish_x"].iloc[0]) if "dish_x" in df_base.columns else None
    dish_y = float(df_base["dish_y"].iloc[0]) if "dish_y" in df_base.columns else None
    dish_radius = float(df_base["dish_radius"].iloc[0]) if "dish_radius" in df_base.columns else None

    if dish_radius is not None:
        print(f"Dish geometry: center=({dish_x:.1f}, {dish_y:.1f}), radius={dish_radius:.1f} px")
    else:
        print("No dish geometry found in df_base — boundary clamping disabled.")

    # ------------------------------------------------------------------ #
    # 4. Chop df_base into trial segments and sample
    # ------------------------------------------------------------------ #
    trial_frames = int(args.trial_length_s * fps)
    n_available = len(df_base) // trial_frames
    n_trials = min(args.n_trials, n_available)

    if n_trials == 0:
        print(
            f"ERROR: df_base has {len(df_base)} frames but trial_length_s={args.trial_length_s} "
            f"requires at least {trial_frames} frames per trial."
        )
        sys.exit(1)

    print(f"Sampling {n_trials} trial(s) of {args.trial_length_s}s each ...")

    for i in range(n_trials):
        start = i * trial_frames
        segment = df_base.iloc[start : start + trial_frames].copy().reset_index(drop=True)

        window, predictions, params = sample_trajectory(
            segment,
            model,
            scalers,
            fps=fps,
            subsample=subsample,
            horizon_frames=horizon_frames,
            max_lag_seconds=max_lag_seconds,
            cols_to_lag=cols_to_lag,
            N_sampling_seconds=args.trial_length_s,
            dish_x=dish_x,
            dish_y=dish_y,
            dish_radius=dish_radius,
        )

        # Save synthetic trajectory
        out_path = args.output / f"trial_{i:03d}.csv"
        window.to_csv(out_path, index=False)

        # Save the matching real segment (post-seed, same frame window as synthetic)
        seed_len = int(max_lag_seconds * fps) + 1
        real_match = segment.iloc[seed_len : seed_len + len(window)].reset_index(drop=True)
        real_path = args.output / f"trial_{i:03d}_real.csv"
        real_match.to_csv(real_path, index=False)

        print(f"  Trial {i}: {len(window)} frames → {out_path} + {real_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
