"""Where does a model's window variance live -- within a fish over time, or between fish?

WHY THIS EXISTS. ``var_ratio`` is POOLED variance, so by the law of total variance it sums two terms
that need OPPOSITE fixes. A model can match it by making each fish wander more over time instead of
by making different fish differ, and until now nothing reported could tell those apart. That is not
hypothetical: ``psib_gruz`` measured ``var_ratio distance = 3.13`` while its between-fish level
spread was too NARROW (0.115 vs a real 0.156), which forces W_gen/W_real > 3.13/(1-b) ~ 4.2 -- more
than all of the excess is within-fish, and adding a between-fish latent on top of it can only make
the pooled number worse.

READ IT WITH ``win_acf1``, WHICH IS PRINTED ALONGSIDE. Excess within-fish variance with an
across-window acf near the real +0.33 is genuine slow drift, correctly shaped. The SAME excess with
acf ~ 0 is white window-to-window jitter -- a process slower than a bout but faster than a window.
Neither number alone distinguishes them, and they call for opposite changes: shape the state's
timescale vs shrink its amplitude.

    python scripts/diag_variance_split.py ladder_psib_gruz.pt --fish 8 --reps 3
    python scripts/diag_variance_split.py a.pt b.pt          # paired, same fish and seeds
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
import generative_metrics as GM         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def windows_for(model, records, reps, seed, W=60):
    """{fish -> (real_feats, gen_feats)} per replicate, exactly as the ladder builds them."""
    out = [{} for _ in range(reps)]
    for i, r in enumerate(records):
        rad_r = GM._radial_of(r, r.x_px, r.y_px)
        has = getattr(r, "dot_cx", None) is not None and rad_r is not None
        real = GM.window_features_traj(r.dtheta, r.symbol, r.distance_px, r.dt_s,
                                       getattr(r, "dot_dist_px", None) if has else None,
                                       W=W, radial=rad_r)
        gens = GM.sample_rollouts(model, r, np.random.default_rng([seed, int(i)]), R=reps)
        for rep, g in enumerate(gens):
            gx, gy = np.asarray(g["x_px"], float), np.asarray(g["y_px"], float)
            dd_g = (np.hypot(gx - np.asarray(r.dot_cx, float),
                             gy - np.asarray(r.dot_cy, float)) if has else None)
            out[rep][int(i)] = (real, GM.window_features_traj(
                g["dtheta"], g["symbol"], g["distance_px"], g["dt_s"], dd_g,
                W=W, radial=GM._radial_of(r, gx, gy)))
    return out


def acf_table(model, records, reps, seed):
    """(gen, real) mean win_acf1 for dt / distance / radial -- the shape half of the read."""
    g_, r_ = {k: [] for k in ("dt", "dist", "rad")}, {k: [] for k in ("dt", "dist", "rad")}
    for i, r in enumerate(records):
        u = np.hypot(np.asarray(r.x_px, float) - r.dish_cx,
                     np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R
        r_["dt"].append(GM.win_acf1(np.log(np.clip(r.dt_s, 1e-6, None))))
        r_["dist"].append(GM.win_acf1(np.log(np.clip(r.distance_px, 1e-6, None))))
        r_["rad"].append(GM.win_acf1(u))
        for g in GM.sample_rollouts(model, r, np.random.default_rng([seed + 7, int(i)]), R=reps):
            gu = np.hypot(np.asarray(g["x_px"], float) - r.dish_cx,
                          np.asarray(g["y_px"], float) - r.dish_cy) / r.dish_R
            g_["dt"].append(GM.win_acf1(np.log(np.clip(g["dt_s"], 1e-6, None))))
            g_["dist"].append(GM.win_acf1(np.log(np.clip(g["distance_px"], 1e-6, None))))
            g_["rad"].append(GM.win_acf1(gu))
    return ({k: float(np.nanmean(v)) for k, v in g_.items()},
            {k: float(np.nanmean(v)) for k, v in r_.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--bouts", default=str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--fish", type=int, default=8,
                    help="fish to use. The between-fish term is estimated from THIS many units, so "
                         "its resolution is ~0.12 at 15 (measured on synthetic data); the within "
                         "term is precise at any size.")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    MX._numpy2_unpickle_compat()
    recs = MB.build_border_records(E.load_bouts(a.bouts))
    allr = [recs[f] for f in sorted(recs)][:a.fish]
    print(f"\n{len(allr)} fish, {a.reps} replicates\n")

    for path in a.models:
        m = MX.load_mix(path)
        t0 = time.time()
        wins = windows_for(m, allr, a.reps, a.seed)
        d = GM.dispersion_report(wins)
        gacf, racf = acf_table(m, allr, a.reps, a.seed)
        print(f"{Path(path).stem}   ({time.time() - t0:.0f} s)")
        print(f"  {'group':<10}{'var_ratio':>11}{'within':>9}{'between':>9}"
              f"   (all -> 1.00)")
        for g in ("turn", "distance", "dt", "dot-dist", "radial"):
            print(f"  {g:<10}{d['var_ratio'].get(g, float('nan')):>11.2f}"
                  f"{d['within_ratio'].get(g, float('nan')):>9.2f}"
                  f"{d['between_ratio'].get(g, float('nan')):>9.2f}")
        print(f"  between_fish_share  gen {d['between_fish_share']['gen']:.3f}"
              f"  real {d['between_fish_share']['real']:.3f}")
        print(f"  win_acf1   " + "  ".join(
            f"{k} {gacf[k]:+.3f} (real {racf[k]:+.3f})" for k in ("dt", "dist", "rad")))

        # STATE THE READ, rather than leaving it to be recomputed. The two numbers only mean
        # something together: amplitude without shape is jitter, shape without amplitude is inert.
        hot = [g for g in ("distance", "dt", "radial")
               if d["within_ratio"].get(g, 1.0) > 1.5]
        if hot:
            slow = abs(gacf["dist"]) > 0.15 or abs(gacf["dt"]) > 0.15
            print(f"\n  ** within-fish excess on: {', '.join(hot)}. "
                  + ("The across-window acf is ALSO up, so this is slow drift with too large an\n"
                     "     amplitude -- shrink the loading, keep the timescale."
                     if slow else
                     "The across-window acf is ~0, so this is WHITE window-to-window\n"
                     "     jitter: a process slower than a bout but faster than a window. More\n"
                     "     between-fish latent cannot fix it; the state's TIMESCALE is wrong. **"))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
