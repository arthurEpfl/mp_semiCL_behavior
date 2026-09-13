"""Retro-fit the SHALLOW-BORDER boundary onto an already-trained checkpoint. No retraining.

Containment is generation-time, so ``(bulk, d0, d1)`` can be fitted on a saved ``.pt`` in a few
minutes on a laptop -- which means the boundary can be measured, and the whole idea accepted or
rejected, before a single Colab hour is spent on it. Mirrors ``recalibrate_containment.py``, which
does the same job for the shell-gated bias.

    python scripts/recalibrate_boundary.py psire2_out/ladder_re_swap_psib.pt
    python scripts/recalibrate_boundary.py psij2_out/ladder_b_indiv_psib.pt --out J20_rej.pt

WHAT IT REPLACES. ``centre_bias`` has two terms: ``bulk`` (constant, sets the mean radius) and
``shell`` (a ramp inside u0, sets the tail). The shell term is the damaging one -- on the shipped
J23 calibration it reaches 39 deg at the wall, which is why the shell turn marginal reads ~100 deg
against a real 66. Rejection does the shell's job as a constraint on POSITION, so ``shell`` is
pinned to 0 and ``bulk`` is refitted with 0 first in its grid: if the free mean radius is already
right, the search returns bulk = 0 and there is no turn bias anywhere in the dish.

Measured on the archived J23 (2026-08-24, all 15 fish, no retrain): the search picks bulk = 0,
occupancy 0.0123 against a real 0.0121, P(r/R > 0.95) exactly 0, at 1.02 draws per bout with no
fallbacks -- where the shipped bias buys occupancy 0.0126 only by dragging mean r/R to 0.4276
against a real 0.4556.

READING THE RESULT. ``got_p95`` is no longer an informative acceptance test: with d0 >= 2 mm the
region r/R > 0.96 is unreachable by construction. Judge on ``got_occ`` / ``got_mean`` here, then
run ``diag_border_profile.py`` on the output for the radial SHAPE and the per-shell E|dtheta|,
which is the quantity the bias was distorting.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAIL_MAX = 0.002                 # kept for comparability with recalibrate_containment.py


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None, help="default: <ckpt stem>_rej.pt")
    ap.add_argument("--bouts", default=str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--max-fish", type=int, default=12)
    ap.add_argument("--w-occ", type=float, default=2.6)
    ap.add_argument("--max-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    args = ap.parse_args()

    recs = MB.build_border_records(E.load_bouts(args.bouts))
    allr = [recs[f] for f in sorted(recs)]
    m = MX.load_mix(args.ckpt)
    if not getattr(m, "_cfg_feat", None) or m._cfg_feat.get("centre_bias") is None:
        print("NOTE: this checkpoint has no fitted containment; fitting the boundary from scratch.")

    tgt, tmean, sd_occ, sd_mean = MX._radial_targets(allr)
    old = dict(m._cfg_feat.get("centre_bias") or {})
    print(f"\n{m.name}   ({len(allr)} fish)")
    print(f"  real targets      occ {tgt:.4f}   mean r/R {tmean:.4f}   "
          f"(between-fish sd {sd_occ:.4f} / {sd_mean:.4f})")
    if old:
        print(f"  shipped bias      bulk {old.get('bulk', 0):.1f}  shell {old.get('shell', 0):.1f}  "
              f"u0 {old.get('u0', 0):.2f}  ->  occ {old.get('got_occ', float('nan')):.4f}  "
              f"mean {old.get('got_mean', float('nan')):.4f}  p95 {old.get('got_p95', float('nan')):.5f}")

    t0 = time.time()
    cb, wr = MX._search_wall_reject(
        m._model, allr, m._cfg_feat, seed=args.seed, max_fish=args.max_fish,
        w_occ=args.w_occ, tail_max=TAIL_MAX, max_draws=args.max_draws,
        closed_loop=m.closed_loop, guardrail=m.guardrail, eps=m.eps, stim_loop=m.stim_loop)
    print(f"\n  fitted in {time.time() - t0:.0f} s")
    print(f"  boundary          bulk {cb['bulk']:.1f}  shell 0.0  d0 {wr['d0']:.1f} mm  "
          f"d1 {wr['d1']:.1f} mm")
    print(f"                    ->  occ {cb['got_occ']:.4f}  mean {cb['got_mean']:.4f}  "
          f"p95 {cb['got_p95']:.5f}")
    print(f"  mean error in between-fish sd:  "
          f"{'shipped %+.2f  ->  ' % ((old.get('got_mean', np.nan) - tmean) / sd_mean) if old else ''}"
          f"boundary {(cb['got_mean'] - tmean) / sd_mean:+.2f}")
    print(f"  cost              {wr['draws_per_bout']:.3f} draws/bout   "
          f"fallback {100 * wr['fallback_frac']:.3f}% of bouts")
    if cb["bulk"] == 0.0:
        print("  => bulk = 0: the search kept NO turn bias anywhere. The whole containment "
              "mechanism is now a position constraint.")

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0
    m._cfg_feat["wall_mode"] = "reflect"
    m._cfg_feat["centre_bias"] = cb
    m._cfg_feat["wall_reject"] = wr
    m.contain_mode = "reject"
    out = args.out or str(Path(args.ckpt).with_name(Path(args.ckpt).stem + "_rej.pt"))
    # save_mix DISPATCHES on the class (joint / fishre / mix). Hard-coding one serialiser here is
    # the exact bug recalibrate_containment.py once shipped: it wrote a fishre model through the
    # joint writer, producing a file that loaded without error and scored a different object.
    MX.save_mix(m, out)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
