"""Measure the joint structure the current model IGNORES.

The bout model predicts only the turn; ``distance_px`` and ``dt_s`` are drawn from marginal
pools, independent of the turn and the stimulus. This asks whether that independence is wrong
in the REAL data -- i.e. how much a joint ``p(distance | turn, stimulus)`` model would buy:

  * corr(|dtheta|, distance)  -- the reorient-vs-cruise coupling (big turn -> small step?)
  * corr(distance, dt)        -- speed structure
  * corr(distance, dot_prox)  -- does the fish take bigger/directed bouts near the dots?
  * net approach on moving bouts -- does it actually close distance on the dots?

Read-only. Requires the dot-geometry columns (run scripts/add_dot_geometry.py first) for the
proximity/approach lines; the turn/distance/dt lines work without them.

    python scripts/diagnose_joint.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import evaluate as E
import fit_MLP_border as MB


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 10 else np.nan


def main():
    df = E.load_bouts(str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    records = MB.build_border_records(df)
    recs = list(records.values())
    have_dot = hasattr(recs[0], "dot_dist_px")
    print(f"{len(recs)} fish | dot geometry: {'present' if have_dot else 'MISSING (run add_dot_geometry.py)'}")

    rows = []
    for r in recs:
        adt, dist, dt = np.abs(r.dtheta), r.distance_px, r.dt_s
        row = dict(fish=r.fish_id, n=r.n,
                   c_turn_dist=_corr(adt, dist), c_dist_dt=_corr(dist, dt), c_turn_dt=_corr(adt, dt))
        if have_dot:
            dd = np.asarray(r.dot_dist_px, float); prox = np.asarray(r.dot_prox, float)
            mv = np.asarray(r.moving, bool)
            row["c_dist_prox"] = _corr(dist, prox)
            pair = mv[:-1] & mv[1:] & np.isfinite(dd[:-1]) & np.isfinite(dd[1:])
            row["approach_mv"] = float(np.mean(dd[1:][pair] - dd[:-1][pair])) if pair.any() else np.nan
        rows.append(row)
    print("\n=== per fish ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    adt = np.concatenate([np.abs(r.dtheta) for r in recs])
    dist = np.concatenate([r.distance_px for r in recs])
    dt = np.concatenate([r.dt_s for r in recs])
    print("\n=== pooled ===")
    print(f"corr(|dtheta|, distance) = {_corr(adt, dist):+.3f}   (reorient<->cruise coupling)")
    print(f"corr(distance,  dt)      = {_corr(dist, dt):+.3f}   (speed structure)")
    print(f"corr(|dtheta|,  dt)      = {_corr(adt, dt):+.3f}")
    if have_dot:
        prox = np.concatenate([np.asarray(r.dot_prox, float) for r in recs])
        print(f"corr(distance, dot_prox) = {_corr(dist, prox):+.3f}   (>0 => bigger bouts when NEAR dots)")
        appr = []
        for r in recs:
            dd = np.asarray(r.dot_dist_px, float); mv = np.asarray(r.moving, bool)
            pair = mv[:-1] & mv[1:] & np.isfinite(dd[:-1]) & np.isfinite(dd[1:])
            appr.append(dd[1:][pair] - dd[:-1][pair])
        appr = np.concatenate(appr)
        verdict = "APPROACHING" if np.nanmean(appr) < 0 else "receding"
        print(f"net approach on moving bouts = {np.nanmean(appr):+.2f} px/bout ({verdict}); "
              f"frac bouts approaching = {np.mean(appr < 0):.2f}")
    print("\nRead: large |corr(turn,distance)| => joint p(distance|turn) is worth it; "
          "corr(distance,prox)!=0 or net approach<0 => distance should be stimulus-driven.")


if __name__ == "__main__":
    main()