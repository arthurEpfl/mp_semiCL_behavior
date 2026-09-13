"""Add a per-trajectory containment GATE offset (rad_gate) to an existing checkpoint.

Radial-individuality mechanism v2, for BOTH the joint (J10) and fishre (J17) families. The v1
multiplicative swing (indiv['rad'] / fish_re['rad_sd']) is measured inert on graded bulk=0
biases; shifting the GATE (each fish gets a personal home radius u0 + delta) moves per-fish mean
r/R directly -- the channel the classifier reads (real between-fish SD of mean r/R 0.075, gen
0.03-0.04, rad_frac_wall the top surviving feature because wall time has the right amount but
the wrong owner).

Procedure: paired two-point moment solve for the offset sd (MX._search_rad_gate, inertness
guard); if a real lever, apply and RE-RUN the containment sweep with the widening active (the
contain_vc lesson), then verify the achieved spread. Generation-only: Metric A untouched.

Usage:
    python scripts/add_rad_gate.py --ckpt ladder_b_indiv_psib_cal2q.pt --out ladder_b_indiv_psib_gate.pt
    python scripts/add_rad_gate.py --ckpt ladder_re_swap_gr_cal2q.pt  --out ladder_re_swap_gr_gate.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP_border as MB
import fit_MLP_mix as MX
import recalibrate_containment as RC

_ROOT = Path(__file__).parent.parent
_DEFAULT_BOUTS = _ROOT / "clean_data" / "bouts_clean.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bouts", default=str(_DEFAULT_BOUTS))
    ap.add_argument("--max-fish", type=int, default=15,
                    help="fish for the containment re-fit (radial stats need 15)")
    ap.add_argument("--n-fish", type=int, default=10, help="fish for the moment solve")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = MX.load_mix(args.ckpt)
    if not m._cfg_feat.get("centre_bias"):
        raise SystemExit("checkpoint has no containment bias -- rad_gate offsets the bias's "
                         "gate, so it is a no-op on a no-contain model")
    if not m._cfg_feat.get("joint"):
        raise SystemExit("rad_gate is implemented in the JOINT-family generator only "
                         "(generate_mix_joint_loop); this cfg is not joint")
    print(f"loaded {args.ckpt}  ({type(m).__name__}, "
          f"contain bias {m._cfg_feat.get('centre_bias')})")

    records = list(MB.build_border_records(pd.read_csv(args.bouts)).values())
    gen_kw = dict(closed_loop=m.closed_loop, guardrail=m.guardrail, eps=m.eps,
                  stim_loop=m.stim_loop)

    tgt_rad = MX._indiv_targets(records)[1]
    print(f"\n[1/3] gate-offset moment solve (target between-fish SD of mean r/R = {tgt_rad:.4f}) ...")
    ri = MX._search_rad_gate(m._model, records, m._cfg_feat, tgt_rad, seed=args.seed,
                             n_fish=args.n_fish, reps=args.reps, **gen_kw)
    print(f"  gate sd = {ri['sd']:.3f}   (base spread {ri['base_sd']:.4f} -> "
          f"probe {ri['probe_sd']:.4f}, target {ri['tgt']:.4f})")
    m._cfg_feat["rad_gate"] = {"sd": ri["sd"]}

    if ri["sd"] > 0:
        print("\n[2/3] containment re-fit WITH the widening active ...")
        w_occ = float(getattr(m, "contain_w_occ", 2.6))
        best = RC.sweep_families(m, records, w_occ, gen_kw, seed=args.seed + 1,
                                 max_fish=args.max_fish)
        RC._report("winner", best, w_occ)
        m._cfg_feat["centre_bias"] = {k: v for k, v in best.items()}
        m._cfg_feat["wall_mode"] = "reflect"

        print("\n[3/3] achieved radial spread at the final config ...")
        got = MX._search_rad_gate(m._model, records, m._cfg_feat, tgt_rad, seed=args.seed + 7,
                                  n_fish=args.n_fish, reps=args.reps, probe=ri["sd"], **gen_kw)
        # probe_sd of this pass IS the spread at the applied sd under the refit bias
        print(f"  between-rollout SD of mean r/R at gate sd={ri['sd']:.3f}: "
              f"{got['probe_sd']:.4f} (target {tgt_rad:.4f})")
    else:
        print("\n[2/3] skipped -- lever inert or solve returned 0; existing bias stands")

    MX.save_mix(m, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
