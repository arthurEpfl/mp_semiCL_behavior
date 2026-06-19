"""
Compare synthetic trajectories produced by sample.py against real recordings.

Usage
-----
    uv run scripts/compare.py \\
        --samples    runs/run_001/samples/ \\
        --real       abs_dict.pkl \\
        --output     runs/run_001/comparison/ \\
        --dish_x     180 --dish_y 174 --dish_radius 183.5

Produces five PNG figures:
    speed_distribution.png       KDE of per-frame speed, real vs synthetic
    dori_distribution.png        KDE of per-frame turning angle (d_ori)
    position_heatmap.png         2D hexbin of fish position in dish coords
    dist_center_distribution.png KDE of normalised radial position
    autocorrelation.png          Autocorrelation of speed and d_ori up to 3 s
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import functions


def parse_args():
    p = argparse.ArgumentParser(description="Compare real and synthetic zebrafish trajectories.")
    p.add_argument("--samples", required=True, type=Path, help="Directory with trial_*.csv files.")
    p.add_argument("--real", required=True, type=Path, help="Path to abs_dict.pkl.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--fish_ids",
        nargs="+",
        type=int,
        default=None,
        help="Fish indices to include from real data (default: all).",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dish_x", type=float, default=180.0, help="Dish centre x (px).")
    p.add_argument("--dish_y", type=float, default=174.0, help="Dish centre y (px).")
    p.add_argument("--dish_radius", type=float, default=183.5, help="Dish radius (px).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_real(pkl_path: Path, fish_ids=None, dish_x=None, dish_y=None, dish_radius=None) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        abs_dict = pickle.load(f)
    if fish_ids is not None:
        abs_dict = {f"f{i}": abs_dict[f"f{i}"] for i in fish_ids if f"f{i}" in abs_dict}

    df = functions.stack_dict_pipeline(abs_dict, dish_x=dish_x, dish_y=dish_y, dish_radius=dish_radius)

    # Compute speed and d_ori for real data
    df = functions.compute_d_orientation(df)
    df = functions.compute_speed(df)
    return df


def load_synthetic(samples_dir: Path) -> pd.DataFrame:
    csvs = sorted(samples_dir.glob("trial_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No trial_*.csv files found in {samples_dir}")
    frames = [pd.read_csv(p) for p in csvs]
    return pd.concat(frames, ignore_index=True)


def autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation for lags 0..max_lag."""
    series = series - series.mean()
    var = np.dot(series, series)
    if var == 0:
        return np.zeros(max_lag + 1)
    result = np.array(
        [np.dot(series[: len(series) - k], series[k:]) / var for k in range(max_lag + 1)]
    )
    return result


# --------------------------------------------------------------------------- #
# Plot functions
# --------------------------------------------------------------------------- #

def plot_speed_distribution(real: pd.DataFrame, synth: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(real["f_speed"].dropna(), ax=ax, label="Real", fill=True, alpha=0.4)
    sns.kdeplot(synth["f_speed"].dropna(), ax=ax, label="Synthetic", fill=True, alpha=0.4)
    ax.set_xlabel("Speed (px/frame)")
    ax.set_ylabel("Density")
    ax.set_title("Speed distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_dori_distribution(real: pd.DataFrame, synth: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(real["d_ori"].dropna(), ax=ax, label="Real", fill=True, alpha=0.4)
    sns.kdeplot(synth["d_ori"].dropna(), ax=ax, label="Synthetic", fill=True, alpha=0.4)
    ax.set_xlabel("Δorientation (rad/frame)")
    ax.set_ylabel("Density")
    ax.set_title("Turning angle distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_position_heatmap(real: pd.DataFrame, synth: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, df, title in zip(axes, [real, synth], ["Real", "Synthetic"]):
        # Use dish-centered coordinates if available
        if "dish_x" in df.columns and "dish_y" in df.columns:
            x = df["f_x"] - df["dish_x"]
            y = df["f_y"] - df["dish_y"]
        else:
            x, y = df["f_x"], df["f_y"]
        h = ax.hexbin(x, y, gridsize=40, cmap="viridis", mincnt=1)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x (px, dish-centred)")
        ax.set_ylabel("y (px, dish-centred)")
        fig.colorbar(h, ax=ax, label="Count")

        # Draw dish boundary if radius is available
        if "dish_radius" in df.columns:
            r = df["dish_radius"].iloc[0]
            circle = plt.Circle((0, 0), r, fill=False, color="red", linewidth=1.5, linestyle="--")
            ax.add_patch(circle)
            ax.set_xlim(-r * 1.1, r * 1.1)
            ax.set_ylim(-r * 1.1, r * 1.1)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_dist_center(real: pd.DataFrame, synth: pd.DataFrame, out: Path):
    def get_dist(df):
        if "dish_x" in df.columns and "dish_radius" in df.columns:
            dx = df["f_x"] - df["dish_x"]
            dy = df["f_y"] - df["dish_y"]
            return (np.sqrt(dx ** 2 + dy ** 2) / df["dish_radius"]).dropna()
        return None

    real_dist = get_dist(real)
    synth_dist = get_dist(synth)

    if real_dist is None or synth_dist is None:
        print("Skipping dist_center plot (dish geometry not available).")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(real_dist, ax=ax, label="Real", fill=True, alpha=0.4)
    sns.kdeplot(synth_dist, ax=ax, label="Synthetic", fill=True, alpha=0.4)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="Dish edge")
    ax.set_xlabel("Distance from centre (normalised by radius)")
    ax.set_ylabel("Density")
    ax.set_title("Radial position distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_autocorrelation(real: pd.DataFrame, synth: pd.DataFrame, fps: int, out: Path):
    max_lag_s = 3.0
    max_lag = int(max_lag_s * fps)
    lags_s = np.arange(max_lag + 1) / fps

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(
        axes,
        ["f_speed", "d_ori"],
        ["Speed autocorrelation", "Turning angle autocorrelation"],
    ):
        real_vals = real[col].dropna().to_numpy()
        synth_vals = synth[col].dropna().to_numpy()

        if len(real_vals) > max_lag and len(synth_vals) > max_lag:
            ac_real = autocorr(real_vals, max_lag)
            ac_synth = autocorr(synth_vals, max_lag)
            ax.plot(lags_s, ac_real, label="Real")
            ax.plot(lags_s, ac_synth, label="Synthetic")
            ax.axhline(0, color="gray", linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")

        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading real data ...")
    real = load_real(
        args.real,
        fish_ids=args.fish_ids,
        dish_x=args.dish_x,
        dish_y=args.dish_y,
        dish_radius=args.dish_radius,
    )

    print(f"Loading synthetic data from {args.samples} ...")
    synth = load_synthetic(args.samples)

    # Ensure d_ori and f_speed columns exist in synthetic data
    if "d_ori" not in synth.columns and "f_ori" in synth.columns:
        synth = functions.compute_d_orientation(synth)
    if "f_speed" not in synth.columns and "f_x" in synth.columns:
        synth = functions.compute_speed(synth)

    print("Generating comparison figures ...")

    plot_speed_distribution(real, synth, args.output / "speed_distribution.png")
    print("  speed_distribution.png")

    plot_dori_distribution(real, synth, args.output / "dori_distribution.png")
    print("  dori_distribution.png")

    plot_position_heatmap(real, synth, args.output / "position_heatmap.png")
    print("  position_heatmap.png")

    plot_dist_center(real, synth, args.output / "dist_center_distribution.png")
    print("  dist_center_distribution.png")

    plot_autocorrelation(real, synth, fps=args.fps, out=args.output / "autocorrelation.png")
    print("  autocorrelation.png")

    print(f"\nAll figures saved to {args.output}")


if __name__ == "__main__":
    main()
