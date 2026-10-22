"""One-time: add PER-DOT absolute positions at each bout onset to the cleaned bouts CSV.

Adds ``d{k}_x, d{k}_y`` for the 8 dot slots (NaN where the slot is inactive), looked up from the
per-frame transformed CSVs by ``frame_onset`` -- the same source ``add_dot_geometry.py`` used for the
centroid ``dot_cx/dot_cy``, which is verified here to equal the mean over the active slots.
NON-DESTRUCTIVE: only ADDS columns (a backup copy is written first). Needed by the per-dot stimulus
code (``fit_MLP.stim_mode='shoal_pop'``, Model 26): the centroid code cannot represent two dot groups.

    python scripts/add_dot_positions.py
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from twin import paths
from twin.stim_program import STIM_NONE  # noqa: E402

K = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", type=Path, default=paths.BOUTS)
    ap.add_argument("--transformed", type=Path, default=paths.FRAMES)
    args = ap.parse_args()
    df = pd.read_csv(args.bouts)
    n0 = len(df)
    bak = args.bouts.with_suffix(".pre_dot_positions.csv")
    if not bak.exists():
        shutil.copy2(args.bouts, bak)
        print(f"backup -> {bak}")
    for k in range(K):
        df[f"d{k}_x"] = np.nan; df[f"d{k}_y"] = np.nan
    worst = 0.0
    for fid, g in df.groupby("fish_id"):
        cands = sorted(args.transformed.glob(f"*_f{int(fid)}_transformed_abs.csv"))
        if not cands:
            raise FileNotFoundError(f"no transformed CSV for fish {fid}")
        use = [f"d{k}_{v}" for k in range(K) for v in ("x", "y")] + [f"d{k}_stim" for k in range(K)]
        t = pd.read_csv(cands[0], usecols=use)
        fr = np.clip(g["frame_onset"].to_numpy(int), 0, len(t) - 1)
        for k in range(K):
            on = ~t[f"d{k}_stim"].astype(str).isin(list(STIM_NONE)).to_numpy() & np.isfinite(t[f"d{k}_x"].to_numpy(float))
            xk = np.where(on, t[f"d{k}_x"].to_numpy(float), np.nan)[fr]
            yk = np.where(on, t[f"d{k}_y"].to_numpy(float), np.nan)[fr]
            df.loc[g.index, f"d{k}_x"] = xk; df.loc[g.index, f"d{k}_y"] = yk
        X = df.loc[g.index, [f"d{k}_x" for k in range(K)]].to_numpy(float)
        Y = df.loc[g.index, [f"d{k}_y" for k in range(K)]].to_numpy(float)
        cx = np.nanmean(X, axis=1); cy = np.nanmean(Y, axis=1)
        ref_x = g["dot_cx"].to_numpy(float); ref_y = g["dot_cy"].to_numpy(float)
        both = np.isfinite(cx) & np.isfinite(ref_x)
        if both.any():
            worst = max(worst, float(np.nanmax(np.hypot(cx[both] - ref_x[both], cy[both] - ref_y[both]))))
        miss = np.isfinite(cx) != np.isfinite(ref_x)
        print(f"fish {fid}: {int(np.isfinite(cx).sum())} bouts with dots; centroid mismatch max {worst:.2e} px; "
              f"active-set disagreements {int(miss.sum())}", flush=True)
    assert len(df) == n0
    df.to_csv(args.bouts, index=False)
    print(f"added d0..d{K - 1}_x/y -> {args.bouts}  (worst centroid mismatch {worst:.2e} px)")


if __name__ == "__main__":
    main()
