"""Compare two or more CHECKPOINTS on the SAME held-out fish -- a paired before/after.

WHY THIS EXISTS
---------------
``rollout_gan.py`` prints bits / window-acf / occupancy but no AUC, and the ladder's AUC comes
from refitting per fold, which a fine-tuned checkpoint cannot join. This closes that gap: point it
at the Stage-2a checkpoint and its GAN-fine-tuned descendant and it scores BOTH on the same
held-out fish, so "did the fine-tune help?" is answered on matched data.

READ IT AGAINST ITS OWN FLOOR, NOT AGAINST THE LADDER. This is ONE fold on ``--holdout`` fish;
the ladder averages 5 GroupKFold folds over all 15. The two AUCs are not comparable numbers. What
IS comparable is the difference between the models here, because they see identical fish, identical
windows and (via the same seed) paired rollouts.

    python scripts/eval_pair.py --models ladder_psib_gru.pt psib_gru_gan.pt --holdout 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP_border as MB              # noqa: E402
import fit_MLP_mix as MX                 # noqa: E402
import generative_metrics as GM          # noqa: E402


def windows_for(model, records, reps, seed, W=60):
    """{fish -> (real_feats, gen_feats)} per replicate, exactly as the ladder builds them."""
    out = [{} for _ in range(reps)]
    for i, r in enumerate(records):
        rad_r = GM._radial_of(r, r.x_px, r.y_px)
        dd_r = getattr(r, "dot_dist_px", None)
        has = getattr(r, "dot_cx", None) is not None and rad_r is not None
        real = GM.window_features_traj(r.dtheta, r.symbol, r.distance_px, r.dt_s,
                                       dd_r if has else None, W=W, radial=rad_r)
        gens = GM.sample_rollouts(model, r, np.random.default_rng([seed, int(i)]), R=reps)
        for rep, g in enumerate(gens):
            gx, gy = np.asarray(g["x_px"], float), np.asarray(g["y_px"], float)
            rad_g = GM._radial_of(r, gx, gy)
            dd_g = (np.hypot(gx - np.asarray(r.dot_cx, float),
                             gy - np.asarray(r.dot_cy, float)) if has else None)
            out[rep][int(i)] = (real, GM.window_features_traj(
                g["dtheta"], g["symbol"], g["distance_px"], g["dt_s"], dd_g, W=W, radial=rad_g))
    return out


def scorecard(model, records, reps, seed, W=60):
    wins = windows_for(model, records, reps, seed, W)
    auc, sd, _ = GM.c2st_auc_replicates(wins)
    disp = GM.dispersion_report(wins)
    bits, acf, occ, rad = [], {"dt": [], "dist": [], "rad": []}, [], []
    for i, r in enumerate(records):
        v = np.asarray(model.score(r)["logdens"], float)
        bits.append(-np.nanmean(v) / np.log(2.0))
        for g in GM.sample_rollouts(model, r, np.random.default_rng([seed + 7, int(i)]), R=reps):
            u = np.hypot(np.asarray(g["x_px"], float) - r.dish_cx,
                         np.asarray(g["y_px"], float) - r.dish_cy) / r.dish_R
            acf["dt"].append(GM.win_acf1(np.log(np.clip(g["dt_s"], 1e-6, None))))
            acf["dist"].append(GM.win_acf1(np.log(np.clip(g["distance_px"], 1e-6, None))))
            acf["rad"].append(GM.win_acf1(u))
            occ.append(float(np.mean(u > 0.85))); rad.append(float(np.mean(u)))
    return dict(auc=auc, auc_sd=sd, bits=float(np.mean(bits)),
                var_ratio=disp.get("var_ratio", {}), bfs=disp.get("between_fish_share", {}),
                acf={k: float(np.nanmean(v)) for k, v in acf.items()},
                occ=float(np.mean(occ)), rad=float(np.mean(rad)), wins=wins)


def real_targets(records, W=60):
    a = {"dt": [], "dist": [], "rad": []}; occ, rad = [], []
    for r in records:
        u = np.hypot(np.asarray(r.x_px, float) - r.dish_cx,
                     np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R
        a["dt"].append(GM.win_acf1(np.log(np.clip(r.dt_s, 1e-6, None))))
        a["dist"].append(GM.win_acf1(np.log(np.clip(r.distance_px, 1e-6, None))))
        a["rad"].append(GM.win_acf1(u))
        occ.append(float(np.mean(u > 0.85))); rad.append(float(np.mean(u)))
    return ({k: float(np.nanmean(v)) for k, v in a.items()}, float(np.mean(occ)), float(np.mean(rad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--bouts", default=str(Path(__file__).parent.parent / "data" /
                                           "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--holdout", type=int, default=3,
                    help="LAST n fish, matching rollout_gan.py's --holdout so the split is identical")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    MX._numpy2_unpickle_compat()
    recs = MB.build_border_records(pd.read_csv(a.bouts))
    rl = [recs[f] for f in sorted(recs)]
    ho = rl[-a.holdout:] if a.holdout else rl
    print(f"held-out fish: {len(ho)} of {len(rl)}   ({a.reps} rollout replicates each)\n")

    tgt_acf, tgt_occ, tgt_rad = real_targets(ho)
    cards = {}
    for path in a.models:
        m = MX.load_mix(path)
        cards[Path(path).stem] = scorecard(m, ho, a.reps, a.seed)
        print(f"  scored {Path(path).stem}", flush=True)

    floor = GM.c2st_negative_control(list(cards.values())[0]["wins"])
    print(f"\nreal-vs-real FLOOR on these fish: {floor:.4f}   <- read every AUC against THIS\n")
    names = list(cards)
    w = max(len(n) for n in names) + 2
    def row(lab, f):
        print(f"  {lab:<26}" + "".join(f"{f(cards[n]):>{w+6}}" for n in names))
    print(f"  {'':<26}" + "".join(f"{n:>{w+6}}" for n in names))
    row("c2st_traj_auc", lambda c: f"{c['auc']:.4f}")
    row("  (replicate sd)", lambda c: f"{c['auc_sd']:.4f}")
    row("Metric A bits (turn)", lambda c: f"{c['bits']:.4f}")
    for k, t in (("dt", tgt_acf["dt"]), ("dist", tgt_acf["dist"]), ("rad", tgt_acf["rad"])):
        row(f"wacf {k}  (real {t:+.3f})", lambda c, k=k: f"{c['acf'][k]:+.3f}")
    for g in ("turn", "distance", "dt", "dot-dist", "radial"):
        row(f"var_ratio {g}  (-> 1.0)", lambda c, g=g: f"{c['var_ratio'].get(g, float('nan')):.3f}")
    row("between_fish_share", lambda c: f"{c['bfs'].get('gen', float('nan')):.4f}")
    row(f"occ>0.85  (real {tgt_occ:.4f})", lambda c: f"{c['occ']:.4f}")
    row(f"mean r/R  (real {tgt_rad:.3f})", lambda c: f"{c['rad']:.3f}")
    print(f"\n  real between_fish_share on these fish: "
          f"{list(cards.values())[0]['bfs'].get('real', float('nan')):.4f}")
    print("\nWHAT THIS IS, AND IS NOT. One fold, so NOT comparable to the ladder's 5-fold AUC.\n"
          "And these fish are held out only from the GAN FINE-TUNE: the base checkpoint is the\n"
          "standalone ALL-fish fit, so it saw them, and the fine-tuned model inherits that. So this\n"
          "is a paired test of WHAT THE FINE-TUNE CHANGED, not a generalisation test. The\n"
          "difference between columns is the answer; both columns saw identical fish and seeds.")


if __name__ == "__main__":
    main()
