"""
Generate a 2×N comparison video: real fish trajectories (top row) vs
synthetic trajectories from sample.py (bottom row).

Each column shows the same stimulus segment: the real recording on top and the
model's sampled trajectory for that same stimulus sequence on the bottom.
Real and synthetic are guaranteed to share the same stimulus sequence because
sample.py writes trial_i_real.csv alongside trial_i.csv.

Usage
-----
    uv run scripts/visualize.py \\
        --samples   runs/run_001/samples/ \\
        --output    runs/run_001/comparison.mp4 \\
        --n         4

Output format is determined by the file extension:
    .mp4  requires ffmpeg
    .gif  requires Pillow (already a matplotlib dependency)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).parent.parent))


N_STIMULI = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, type=Path,
                   help="Directory with trial_*.csv and trial_*_real.csv files from sample.py.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output video path (.mp4 or .gif).")
    p.add_argument("--n", type=int, default=4,
                   help="Number of columns (trajectories per row). Default: 4.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dish_x", type=float, default=180.0)
    p.add_argument("--dish_y", type=float, default=174.0)
    p.add_argument("--dish_radius", type=float, default=183.5)
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (e.g. 4 = 4× faster). Default: 1.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_real_segments(samples_dir: Path, n: int) -> list[pd.DataFrame]:
    csvs = sorted(samples_dir.glob("trial_*_real.csv"))[:n]
    if not csvs:
        raise FileNotFoundError(
            f"No trial_*_real.csv files found in {samples_dir}. "
            "Re-run sample.py to regenerate (it now saves matching real segments)."
        )
    return [pd.read_csv(p).reset_index(drop=True) for p in csvs]


def load_synthetic_segments(samples_dir: Path, n: int) -> list[pd.DataFrame]:
    csvs = sorted(p for p in samples_dir.glob("trial_*.csv") if "_real" not in p.name)[:n]
    if not csvs:
        raise FileNotFoundError(f"No trial_*.csv files found in {samples_dir}")
    return [pd.read_csv(p).reset_index(drop=True) for p in csvs]


# --------------------------------------------------------------------------- #
# Per-panel setup helpers
# --------------------------------------------------------------------------- #

def _dish_limits(dish_x, dish_y, dish_radius, pad_frac=0.08):
    pad = dish_radius * pad_frac
    r = dish_radius + pad
    return dish_x - r, dish_x + r, dish_y - r, dish_y + r


def _init_panel(ax, traj_df, dish_x, dish_y, dish_radius, label, color):
    """Set up a single subplot and return the artists to animate."""
    xmin, xmax, ymin, ymax = _dish_limits(dish_x, dish_y, dish_radius)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    # Dish boundary circle
    circle = mpatches.Circle(
        (dish_x, dish_y), dish_radius,
        fill=False, edgecolor="gray", linewidth=1.2, linestyle="--", zorder=1,
    )
    ax.add_patch(circle)

    ax.set_title(label, fontsize=9, color=color, pad=3)

    arrow_len = dish_radius * 0.08
    row0 = traj_df.iloc[0]
    fish_quiv = ax.quiver(
        [row0["f_x"]], [row0["f_y"]],
        [np.cos(row0["f_ori"])], [np.sin(row0["f_ori"])],
        angles="xy", scale_units="xy", scale=1 / arrow_len,
        color=color, zorder=3,
    )
    stim_scatter = ax.scatter([], [], s=18, color="orange", zorder=2)
    time_text = ax.text(0.03, 0.97, "", transform=ax.transAxes,
                        va="top", fontsize=7, color="black")

    return fish_quiv, stim_scatter, time_text


def _update_panel(t, traj_df, fish_quiv, stim_scatter, time_text, fps):
    if t >= len(traj_df):
        return
    row = traj_df.iloc[t]
    fish_quiv.set_offsets([[row["f_x"], row["f_y"]]])
    fish_quiv.set_UVC(np.cos(row["f_ori"]), np.sin(row["f_ori"]))

    stim_x, stim_y = [], []
    for k in range(N_STIMULI):
        stim_col = f"d{k}_stim"
        if stim_col in traj_df.columns and int(row[stim_col]) == 1:
            stim_x.append(row[f"d{k}_x"])
            stim_y.append(row[f"d{k}_y"])
    if stim_x:
        stim_scatter.set_offsets(np.c_[stim_x, stim_y])
    else:
        stim_scatter.set_offsets(np.empty((0, 2)))

    time_text.set_text(f"t={t / fps:.1f}s")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    n = args.n

    print("Loading real segments ...")
    real_segs = load_real_segments(args.samples, n)
    print("Loading synthetic segments ...")
    synth_segs = load_synthetic_segments(args.samples, n)

    n = min(n, len(real_segs), len(synth_segs))
    if n == 0:
        print("ERROR: no segments available.")
        sys.exit(1)

    n_frames = min(min(len(s) for s in real_segs[:n]),
                   min(len(s) for s in synth_segs[:n]))

    print(f"Rendering {n} columns × 2 rows, {n_frames} frames each ...")

    # ------------------------------------------------------------------ #
    # Build figure
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(
        2, n,
        figsize=(3.5 * n, 7.5),
        gridspec_kw={"hspace": 0.15, "wspace": 0.05},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    # Row labels on the left
    for row_idx, label in enumerate(["Real", "Synthetic"]):
        axes[row_idx, 0].set_ylabel(label, fontsize=11, labelpad=6)

    # Initialise all panels
    artists = []  # list of (fish_quiv, stim_scatter, time_text, traj_df) per panel
    real_color = "#1f77b4"
    synth_color = "#d62728"

    for col in range(n):
        for row_idx, (segs, color, label_prefix) in enumerate([
            (real_segs,  real_color,  "Real"),
            (synth_segs, synth_color, "Synthetic"),
        ]):
            traj = segs[col]
            ax = axes[row_idx, col]
            label = f"{label_prefix} #{col + 1}"
            quiv, scatter, txt = _init_panel(
                ax, traj, args.dish_x, args.dish_y, args.dish_radius,
                label=label, color=color,
            )
            artists.append((quiv, scatter, txt, traj))

    # ------------------------------------------------------------------ #
    # Animation
    # ------------------------------------------------------------------ #
    def update(t):
        updated = []
        for quiv, scatter, txt, traj in artists:
            _update_panel(t, traj, quiv, scatter, txt, args.fps)
            updated.extend([quiv, scatter, txt])
        return updated

    interval_ms = 1000 / (args.fps * args.speed)
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval_ms, blit=False)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ext = args.output.suffix.lower()

    if ext == ".gif":
        writer = "pillow"
    else:
        writer = "ffmpeg"

    print(f"Saving → {args.output}  (writer={writer}) ...")
    anim.save(args.output, writer=writer, fps=args.fps * args.speed, dpi=120)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
