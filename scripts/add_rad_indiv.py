"""Add generation-time RADIAL individuality to an existing fishre checkpoint -- no retraining.

The fishre likelihood latent z loads on forward logit / mu_l / mu_tau and has no spatial channel,
so every rollout shares one radial personality: measured between-fish SD of mean r/R 0.030 vs
real 0.075 (radial var_ratio ~0.80, under-dispersed). This driver calibrates
``fish_re['rad_sd']`` -- a per-trajectory multiplicative swing on the containment bias, drawn
from a SPAWNED stream in ``generate_mix_joint_loop`` so CRN pairing against the no-rad control
holds -- by method of moments (``MX._search_rad_indiv``), then RE-RUNS the containment sweep with
the widening active (widening drives wall contact; not re-fitting is how contain_vc lost), and
saves. Generation-only: Metric A is untouched by construction.

Usage:
    python scripts/add_rad_indiv.py --ckpt ladder_re_swap_gr_cal2q.pt --out ladder_re_swap_gr_urad.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP_border as MB
import fit_MLP_mix as MX
import fit_MLP_fishre as FR
import recalibrate_containment as RC

_ROOT = Path(__file__).parent.parent
_DEFAULT_BOUTS = _ROOT / "clean_data" / "bouts_clean.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bouts", default=str(_DEFAULT_BOUTS))
    ap.add_argument("--max-fish", type=int, default=15,
                    help="fish for the containment re-fit (15 = all; radial stats need 15)")
    ap.add_argument("--n-fish", type=int, default=10, help="fish for the rad_sd moment solve")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = MX.load_mix(args.ckpt)
    if not isinstance(m, FR.MLPMixJointREDensity):
        raise SystemExit(f"{args.ckpt} is {type(m).__name__}; rad_indiv is a fishre mechanism "
                         "(J10-style models already carry indiv['rad'])")
    if not m._cfg_feat.get("centre_bias"):
        raise SystemExit("checkpoint has no containment bias -- u_rad_re multiplies cb, so "
                         "rad_indiv is a no-op on a no-contain model (and fitting a fresh bias "
                         "here would silently convert a control checkpoint into a contained one)")
    print(f"loaded {args.ckpt}  (re_m={m.re_m}, contain bias {m._cfg_feat.get('centre_bias')})")

    records = list(MB.build_border_records(pd.read_csv(args.bouts)).values())
    gen_kw = dict(closed_loop=m.closed_loop, guardrail=m.guardrail, eps=m.eps,
                  stim_loop=m.stim_loop)

    tgt_rad = MX._indiv_targets(records)[1]
    print(f"\n[1/3] rad_sd moment solve (target between-fish SD of mean r/R = {tgt_rad:.4f}) ...")
    ri = MX._search_rad_indiv(m._model, records, m._cfg_feat, tgt_rad, seed=args.seed,
                              n_fish=args.n_fish, reps=args.reps, **gen_kw)
    print(f"  rad_sd = {ri['rad_sd']:.3f}   (base spread {ri['base_sd']:.4f} -> "
          f"probe {ri['probe_sd']:.4f}, target {ri['tgt']:.4f})")
    m._cfg_feat["fish_re"]["rad_sd"] = ri["rad_sd"]
    m.rad_indiv = True

    if ri["rad_sd"] > 0:
        print("\n[2/3] containment re-fit WITH the widening active ...")
        w_occ = float(getattr(m, "contain_w_occ", 2.6))
        best = RC.sweep_families(m, records, w_occ, gen_kw, seed=args.seed + 1,
                                 max_fish=args.max_fish)
        RC._report("winner", best, w_occ)
        m._cfg_feat["centre_bias"] = {k: v for k, v in best.items()}
        m._cfg_feat["wall_mode"] = "reflect"     # containment always pairs with the bounce wall
    else:
        print("\n[2/3] skipped -- rad_sd=0 means nothing widened, the existing bias stands")

    if ri["rad_sd"] > 0:
        print("\n[3/3] achieved radial spread at the final config ...")
        got = MX._search_rad_indiv(m._model, records, m._cfg_feat, tgt_rad, seed=args.seed + 7,
                                   n_fish=args.n_fish, reps=args.reps, probe=ri["rad_sd"],
                                   **gen_kw)
        # probe_sd of this pass IS the spread at the applied rad_sd (probe = applied value)
        print(f"  between-rollout SD of mean r/R at rad_sd={ri['rad_sd']:.3f}: "
              f"{got['probe_sd']:.4f} (target {tgt_rad:.4f})")
    else:
        print("\n[3/3] skipped -- moment solve returned rad_sd=0 (no widening applied)")

    FR.save_fishre(m, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
