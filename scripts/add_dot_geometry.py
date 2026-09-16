"""One-time: add per-bout ACTIVE-DOT geometry columns to the bouts CSV.

Adds ``dot_cx, dot_cy, dot_dist_px, dot_prox`` (fish-to-active-dot-centroid distance +
proximity weight at each bout onset), looked up from the per-frame transformed CSVs by
``frame_onset``. NON-DESTRUCTIVE: it only ADDS columns to the existing bouts CSV (no
re-segmentation), analogous to how ``dist_to_border`` was added. After running this,
``fit_MLP_border.build_border_records`` attaches the fields to each record, which the
approach/proximity metrics (and, later, range stimulus features) consume.

    python scripts/add_dot_geometry.py
    python scripts/add_dot_geometry.py --bouts data/bouts_GsizeMotionSingle.csv \
        --transformed data/data/df_transformed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import segment_bouts as SB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", type=Path, default=Path("data/bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--transformed", type=Path, default=Path("data/data/df_transformed"))
    args = ap.parse_args()

    df = pd.read_csv(args.bouts)
    n0 = len(df)
    out = SB.add_dist_to_dot(df, transformed_dir=args.transformed)
    assert len(out) == n0, "row count changed -- aborting (would corrupt the bouts CSV)"
    have = int(out["dot_dist_px"].notna().sum())
    out.to_csv(args.bouts, index=False)
    print(f"added dot geometry -> {args.bouts}")
    print(f"  {have}/{n0} bouts have an active dot at onset "
          f"(new cols: dot_cx, dot_cy, dot_dist_px, dot_prox)")


if __name__ == "__main__":
    main()