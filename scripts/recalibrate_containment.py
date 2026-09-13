"""Re-fit the generation-time containment bias of an EXISTING checkpoint -- no retraining.

Why this exists: the shipped 1-D containment (u0 = 0.75, coarse scale grid, <= 12 calibration
fish) over-corrects the bulk -- J10 rolls out at mean r/R 0.394 and J17-gr at 0.424 against a
real 0.449 -- while the r/R > 0.85 tail is still 1.5-2.6x real and both models leak mass past
r/R 0.95, where real fish NEVER go. The u0 = 0.65 sweep in the lineage doc lands occupancy
exactly on the real 0.0123 with the mean nearly flat ("a free improvement available all along,
never taken"). This driver takes it: a joint (scale x u0) sweep of the 1-D family AND the wide
graded family, on ALL 15 fish (a ~1% occupancy cannot be estimated on fewer -- the documented
6-fish failure picked a scale that measured 1.60x real), with a hard r/R > 0.95 tail guard.

The centre_bias lives in cfg_feat and is generation-only (invisible to the likelihood), so a
saved checkpoint can be recalibrated and re-saved without touching the net: Metric A is
unchanged by construction, only rollouts move. The recalibrated .pt is a LOCAL rollout/diagnostic
artefact -- ladder scoring still refits containment inside each fold on train fish only.

For J10-style checkpoints the cfg carries `indiv` (u_rad multiplies the bias), so the sweep
automatically rolls out with the widening active -- the re-fit-after-widening rule is honoured.

Usage:
    python scripts/recalibrate_containment.py --ckpt ladder_b_indiv150.pt --out ladder_b_indiv150_cal2q.pt
    python scripts/recalibrate_containment.py --ckpt ladder_re_swap_gr.pt --out ladder_re_swap_gr_cal2q.pt
    python scripts/recalibrate_containment.py --ckpt X.pt --out Y.pt --max-fish 2 --fast   # wiring check only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP_border as MB
import fit_MLP_mix as MX

_ROOT = Path(__file__).parent.parent
_DEFAULT_BOUTS = _ROOT / "clean_data" / "bouts_clean.csv"

# Sweep grids. Scale edges (1.5 / 3.0) bracket the shipped 2.25-2.50; u0 edges bracket the
# shipped 0.75 and the documented free improvement 0.65. TAIL_MAX enforces the hard physical
# fact P(r/R > 0.95)_real = 0.0000 -- a candidate that wins the two-moment objective while
# leaking there is manufacturing a feature real fish never produce.
SCALE_GRID = (1.5, 2.0, 2.25, 2.5, 3.0)
U0_GRID = (0.55, 0.60, 0.65, 0.70, 0.75)
TAIL_MAX = 0.002
# Acceptance bounds (vs the real targets measured on the same records)
OCC_TOL, MEAN_TOL = 0.005, 0.02


def _score_of(res, w_occ):
    return (w_occ * ((res["got_occ"] - res["target_occ"]) / res["sd_occ"]) ** 2
            + ((res["got_mean"] - res["target_mean"]) / res["sd_mean"]) ** 2)


def _report(tag, res, w_occ):
    print(f"  [{tag}] centre_bias = {{bulk {res['bulk']:.2f}, shell {res['shell']:.2f}, "
          f"u0 {res['u0']:.2f}{', graded' if res.get('graded') else ''}}}")
    print(f"        occ {res['got_occ']:.4f} (target {res['target_occ']:.4f})   "
          f"mean r/R {res['got_mean']:.4f} (target {res['target_mean']:.4f})   "
          f"p95 {res.get('got_p95', float('nan')):.4f}   score {_score_of(res, w_occ):.3f}")


def sweep_families(m, records, w_occ, gen_kw, seed=0, max_fish=15, fast=False):
    """Sweep the 1-D (scale x u0) and graded containment families on a LOADED model and return
    the better dict by the shared two-moment objective. Also used by add_rad_indiv.py so the
    post-widening re-fit uses the identical procedure."""
    scale_grid = (2.0, 2.5) if fast else SCALE_GRID
    u0_grid = (0.65, 0.75) if fast else U0_GRID
    print(f"\n[1/2] 1-D family: scale {scale_grid} x u0 {u0_grid} on {max_fish} fish ...")
    res1 = MX._search_containment(m._model, records, m._cfg_feat, grid=scale_grid,
                                  u0=u0_grid, max_fish=max_fish, w_occ=w_occ,
                                  tail_max=TAIL_MAX, seed=seed, **gen_kw)
    _report("1-D", res1, w_occ)
    if fast:
        return res1
    print(f"\n[2/2] graded family (bulk=0, approach-gated) on {max_fish} fish ...")
    # Wider grids than the fit-time defaults: the J10 recal pinned at u0=0.45 & shell=80 edges.
    res2 = MX._search_containment(m._model, records, m._cfg_feat, graded=True,
                                  u0_grid=(0.35, 0.45, 0.55, 0.60, 0.70),
                                  graded_shell_grid=(40.0, 60.0, 80.0, 100.0, 140.0, 200.0),
                                  max_fish=max_fish, w_occ=w_occ,
                                  tail_max=TAIL_MAX, seed=seed, **gen_kw)
    _report("graded", res2, w_occ)
    # Cross-family compare under the SAME soft tail penalty _search_containment uses internally.
    # (A lexicographic compliant-first rule was tried and misfired: on the indiv-widened J10 no
    # graded candidate could meet the guard, so a marginal 0.05%-of-bouts leak lost to a
    # 'compliant' config with mean r/R 0.352 -- far worse on the features that matter.)
    def _pscore(r):
        return (_score_of(r, w_occ)
                + 4.0 * w_occ * (max(0.0, r.get("got_p95", 0.0) - TAIL_MAX) / r["sd_occ"]) ** 2)
    return min((res1, res2), key=_pscore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bouts", default=str(_DEFAULT_BOUTS))
    ap.add_argument("--max-fish", type=int, default=15,
                    help="calibration fish (radial statistics are INVALID below 15; "
                         "use fewer only for wiring checks)")
    ap.add_argument("--fast", action="store_true", help="tiny grids -- wiring check only")
    ap.add_argument("--no-dt-snap", action="store_true",
                    help="do not enable the 1/30 s interval lattice in the saved cfg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.max_fish < 15:
        print(f"WARNING: --max-fish {args.max_fish} < 15 -- calibration will be noise; "
              "use only to check wiring")

    m = MX.load_mix(args.ckpt)
    # (load_joint/load_fishre restore the treatment flags from the blob/cfg themselves, so a
    # re-save here describes the checkpoint faithfully -- including vigor_drift_calib, which
    # cannot be inferred from cfg presence.)
    print(f"loaded {args.ckpt}  ({type(m).__name__}, stim_mode={m.stim_mode})")
    print(f"  old centre_bias: {m._cfg_feat.get('centre_bias')}")
    print(f"  cfg treatments : vigor={'vigor' in m._cfg_feat}  "
          f"kfb={'kin_feedback' in m._cfg_feat}  indiv={'indiv' in m._cfg_feat}  "
          f"fish_re={'fish_re' in m._cfg_feat}  innov={'kin_innov' in m._cfg_feat}")

    records = list(MB.build_border_records(pd.read_csv(args.bouts)).values())
    gen_kw = dict(closed_loop=m.closed_loop, guardrail=m.guardrail, eps=m.eps,
                  stim_loop=m.stim_loop)
    w_occ = float(getattr(m, "contain_w_occ", 2.6))
    best = sweep_families(m, records, w_occ, gen_kw, seed=args.seed,
                          max_fish=args.max_fish, fast=args.fast)

    print(f"\nwinner: {'graded' if best.get('graded') else '1-D'}")
    _report("winner", best, w_occ)
    occ_ok = abs(best["got_occ"] - best["target_occ"]) <= OCC_TOL
    mean_ok = abs(best["got_mean"] - best["target_mean"]) <= MEAN_TOL
    p95_ok = best.get("got_p95", 1.0) <= TAIL_MAX
    print(f"  bounds: occ within {OCC_TOL}: {'PASS' if occ_ok else 'FAIL'}   "
          f"mean within {MEAN_TOL}: {'PASS' if mean_ok else 'FAIL'}   "
          f"p95 <= {TAIL_MAX}: {'PASS' if p95_ok else 'FAIL'}")

    m._cfg_feat["centre_bias"] = {k: v for k, v in best.items()}
    m._cfg_feat["wall_mode"] = "reflect"          # containment always pairs with the bounce wall
    m.contain = True
    if not args.no_dt_snap:
        m._cfg_feat["dt_snap"] = True
        m.dt_snap = True
        print("  dt_snap=True written into cfg (emitted intervals now on the 1/30 s lattice)")
    # save_mix dispatches on class (joint -> save_joint, fishre -> save_fishre, kin/loop -> its
    # own blob). Hard-coding save_joint here silently corrupted kin checkpoints: every attribute
    # save_joint reads exists on MLPMixKinDensity, so it wrote cls='MLPMixJointDensity' over an
    # MDNMixKin state_dict and the failure only surfaced at the next load.
    MX.save_mix(m, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
