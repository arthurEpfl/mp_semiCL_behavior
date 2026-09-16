"""Retro-fit PER-CHANNEL latent gains onto an already-trained checkpoint. No retraining.

The loading gain is a generation-time hyperparameter, so `(g_turn, g_dist, g_dt)` can be refitted
on a saved `.pt` in minutes on a laptop -- which means the question below can be answered before a
single further GPU hour is spent. Same standing as `recalibrate_boundary.py`, which does the
equivalent job for the containment boundary.

    python scripts/recalibrate_gains.py ladder_psib_gruz.pt
    python scripts/recalibrate_gains.py ladder_psib_gruz.pt --out gruz_pc.pt --max-fish 12

THE QUESTION THIS ANSWERS. `psib_gruz` (GRU + fish_re latent) came out with between-fish travel
spread 3.13x too WIDE and forward-fraction spread ~4x too NARROW (gen 0.0129 vs real 0.0537). The
single global gain cannot express a correction whose two halves point in opposite directions, so it
drove to its grid FLOOR on 4 of 5 folds and matched neither. Three gains can. Whether they DO is
what this measures.

READING THE RESULT, in the order it is printed:

  latent OFF   -- all three loadings zeroed, so this is what the model produces on its own. For a
                  recurrent model this is the double-counting test: if the GRU alone already gives
                  close to the real between-fish spread, then z is redundant and the right move is
                  to drop the offending channel, not to tune it. A single global gain can never
                  show this -- it scales all three together and never tests zero.
  gain = 1     -- the fitted loadings as trained. The gap from here to `real` is the MAP shrinkage
                  the gain exists to correct, and it is per-channel.
  fitted       -- after coordinate descent. A channel PINNED at a grid edge did not converge: its
                  spread is not coming from z, and no gain will fix it.

  var_ratio / between_fish_share are REPORTED, never optimised. The objective is three per-fish
  LEVEL spreads, which is a proxy for them; if the levels land on target and var_ratio distance is
  still 3.13, the gain was never the mechanism and that is the finding.
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
import fit_MLP_fishre as FR             # noqa: E402
import generative_metrics as GM         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def window_report(model, records, seed, reps, W=60):
    """var_ratio + between_fish_share on the SAME features the ladder's C2ST reads.

    Reported next to the level statistics because the level spreads are only a PROXY for these:
    matching them is the objective, moving these is the point, and the two can come apart.
    """
    wins = [{} for _ in range(reps)]
    for i, r in enumerate(records):
        rad_r = GM._radial_of(r, r.x_px, r.y_px)
        has = getattr(r, "dot_cx", None) is not None and rad_r is not None
        real = GM.window_features_traj(r.dtheta, r.symbol, r.distance_px, r.dt_s,
                                       getattr(r, "dot_dist_px", None) if has else None,
                                       W=W, radial=rad_r)
        for rep, g in enumerate(GM.sample_rollouts(model, r, np.random.default_rng([seed, int(i)]),
                                                   R=reps)):
            gx, gy = np.asarray(g["x_px"], float), np.asarray(g["y_px"], float)
            dd_g = (np.hypot(gx - np.asarray(r.dot_cx, float),
                             gy - np.asarray(r.dot_cy, float)) if has else None)
            wins[rep][int(i)] = (real, GM.window_features_traj(
                g["dtheta"], g["symbol"], g["distance_px"], g["dt_s"], dd_g,
                W=W, radial=GM._radial_of(r, gx, gy)))
    d = GM.dispersion_report(wins)
    return d.get("var_ratio", {}), d.get("between_fish_share", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None, help="default: <ckpt stem>_pcgain.pt")
    ap.add_argument("--bouts", default=str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--max-fish", type=int, default=8,
                    help="fish in the SEARCH. Per-fish spread needs units: at 4 the objective is "
                         "flat within its own noise. 8 matches the in-fit search.")
    ap.add_argument("--draws", type=int, default=2, help="latent draws per fish, per grid point")
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--cap", type=int, default=None,
                    help="truncate search rollouts to N bouts. The estimand is a BETWEEN-fish "
                         "spread, so a few hundred bouts pins each level and the cost falls "
                         "linearly. Does NOT affect the window report, which uses full length.")
    ap.add_argument("--wide", action="store_true", help="the 8-point grid, extending down to 0.10")
    ap.add_argument("--windows", type=int, default=0,
                    help="replicates for the var_ratio / between_fish_share report (0 = skip). "
                         "This is the slow half -- full-length rollouts on every fish.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="refit from N different seeds and report whether the gains AGREE. The "
                         "objective is a spread estimated from max_fish x draws simulated fish, "
                         "and it is flat within its own noise at small settings -- measured, the "
                         "same seed gave turn 1.2 one run and 3.0 the next. Use 2+ before acting "
                         "on a result; costs N times the wall-clock.")
    ap.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    a = ap.parse_args()

    MX._numpy2_unpickle_compat()
    recs = MB.build_border_records(E.load_bouts(a.bouts))
    allr = [recs[f] for f in sorted(recs)]
    m = MX.load_mix(a.ckpt)
    net, cfg = m._model, m._cfg_feat
    if not hasattr(net, "z_turn"):
        raise SystemExit(f"{a.ckpt} has no z_* loadings -- not a fish_re model")
    if np.max(np.abs(np.asarray(getattr(m, "re_sd", [0.0]), float))) <= 0.0:
        raise SystemExit(f"{a.ckpt} has the latent PINNED at zero (re_zero=True); there is no "
                         "loading to gain. Refit with re_zero=False.")
    grid = FR.gain_grid_for("wide" if a.wide else "std")
    print(f"\n{m.name}   ({len(allr)} fish, {a.max_fish} in the search, "
          f"state={int(cfg.get('state', 0))})")
    print(f"  grid {grid}")
    print(f"  loading norms as trained: " + "  ".join(
        f"{k} {float(getattr(net, 'z_' + k).weight.norm()):.4f}" for k in ("turn", "dist", "dt")))

    t0 = time.time()
    runs = []
    for k in range(max(int(a.repeats), 1)):
        if a.repeats > 1:
            print(f"\n  --- seed {a.seed + k} ---")
        runs.append(FR.search_loading_gain_per_channel(
            net, allr, cfg, grid=grid, seed=a.seed + k, max_fish=a.max_fish, n_draws=a.draws,
            sweeps=a.sweeps, n_cap=a.cap, verbose=True,
            closed_loop=m.closed_loop, guardrail=m.guardrail, eps=m.eps, stim_loop=m.stim_loop))
    g, rep = runs[0]
    print(f"\n  fitted in {time.time() - t0:.0f} s")

    if len(runs) > 1:
        print(f"\n  SEED STABILITY ({len(runs)} refits) -- gains must agree, or the objective is "
              f"fitting noise")
        for k, (gk, _) in enumerate(runs):
            print(f"    seed {a.seed + k}: " + "  ".join(f"{c} {gk[c]:>4}" for c, _ in
                                                         FR.CHANNEL_LEVEL))
        unstable = [c for c, _ in FR.CHANNEL_LEVEL if len({r[0][c] for r in runs}) > 1]
        if unstable:
            print(f"    ** UNSTABLE on {', '.join(unstable)}: the same objective picked different\n"
                  f"       grid points across seeds, so those gains are NOT identified at these\n"
                  f"       settings. Do not fold them in -- raise --max-fish / --draws first. **")
        else:
            print("    all channels agree across seeds.")

    names = rep["level_names"]
    print(f"\n  between-fish SD, per level  ({rep['n_units']} simulated fish; ratio 1.00 = matched)")
    print(f"    {'':<20}{'real':>10}{'latent OFF':>13}{'gain = 1':>11}{'fitted':>10}{'ratio':>9}"
          f"{'noise':>9}")
    for li, nm in enumerate(names):
        r_, off, one, fit = (rep["target_sd"][li], rep["sd_latent_off"][li],
                             rep["sd_gain1"][li], rep["got_sd"][li])
        print(f"    {nm:<20}{r_:>10.4f}{off:>13.4f}{one:>11.4f}{fit:>10.4f}"
              f"{fit / max(r_, 1e-9):>9.2f}{rep['noise_floor'][li]:>9.4f}")
    print("\n  fitted gains: " + "  ".join(f"{k} {v}" for k, v in g.items()))
    if rep.get("saturated"):
        print("\n  reachable span over the WHOLE grid (others at their fitted values):")
        for k, li in FR.CHANNEL_LEVEL:
            lo, hi = rep["span"][k]
            print(f"    {k:>4}  {lo:.4f} .. {hi:.4f}   real {rep['target_sd'][li]:.4f}"
                  + ("   <- SATURATED" if k in rep["saturated"] else ""))
        print(f"  ** {', '.join(rep['saturated'])}: no gain in the grid reaches the target. This is"
              f"\n     NOT a calibration problem and a wider grid will not fix it -- that loading"
              f"\n     cannot carry the spread, and needs replacing rather than rescaling. **")
    elif rep["grid_edge"]:
        print(f"  MISSED TARGET on: {', '.join(rep['grid_edge'])}"
              + (f"  (at a grid edge: {', '.join(rep['at_grid_edge'])} -- widen it)"
                 if rep.get("at_grid_edge") else ""))

    # IS THERE ANY SIGNAL? A level whose measured sd is at or below its own within-trajectory noise
    # floor carries none, and the argmin over the grid is then picking noise -- which is how a
    # calibrator returns a confident number that changes on the next run. Measured on tiny_gruz at
    # --cap 600: forward fraction came out 0.0151 against a 0.0147 floor, and the turn gain landed
    # on 1.2 one run and 3.0 the next from the SAME seed. Caught here rather than downstream.
    print("\n  noise / measured spread, per level  (>0.7 = the gain is fitting noise)")
    print("    " + "   ".join(
        f"{nm.split()[-1]} {rep['noise_floor'][li] / max(rep['got_sd'][li], 1e-9):.2f}"
        for li, nm in enumerate(names)))
    dead = [nm for li, nm in enumerate(names)
            if rep["noise_floor"][li] >= 0.7 * rep["got_sd"][li]]
    if dead:
        print(f"\n  ** NO SIGNAL on: {', '.join(dead)}.  The measured spread is at or below the\n"
              f"     within-trajectory noise floor, so that channel's gain is fitting NOISE and is\n"
              f"     not to be trusted -- whatever value it printed. Raise --max-fish / --draws,\n"
              f"     and drop --cap, before reading it. **")
    if rep["n_cap"]:
        print(f"\n  --cap {rep['n_cap']}: the `noise` column is the WITHIN-trajectory SE of each "
              f"level; it adds in\n  quadrature to every sd above, so a short rollout INFLATES them "
              f"and shrinks the signal.")
    n_eff = rep["n_units"]
    print(f"  sampling SE of an sd from {n_eff} units is ~{100 / np.sqrt(2 * (n_eff - 1)):.0f}%, "
          f"so grid points closer than that are ties.")

    # THE DOUBLE-COUNTING READ. Corrected for the noise floor, because an uncorrected short-rollout
    # sd would read as "the model already has spread" -- the exact false positive this test must not
    # produce, since its conclusion is "delete a mechanism".
    off_ratio = []
    for li in range(len(names)):
        var = rep["sd_latent_off"][li] ** 2 - rep["noise_floor_off"][li] ** 2
        off_ratio.append(float(np.sqrt(max(var, 0.0)) / max(rep["target_sd"][li], 1e-9)))
    print("\n  latent OFF vs real, noise-corrected: " +
          "  ".join(f"{nm.split()[-1]} {r:.2f}x" for nm, r in zip(names, off_ratio)))
    if max(off_ratio) > 0.7:
        hot = [nm for nm, r in zip(names, off_ratio) if r > 0.7]
        print("    ^ the model already produces most of the real between-fish spread with the\n"
              f"      latent OFF, on: {', '.join(hot)}. That is DOUBLE-COUNTING -- z is adding\n"
              "      spread the state is already generating, and the fix is to drop that channel's\n"
              "      loading, not to shrink it.")
    else:
        print("    ^ little spread without the latent, so z and the state are NOT competing for\n"
              "      this variance; the residual mismatch is a calibration problem.")

    if a.windows:
        print(f"\n  window report ({a.windows} replicates, full-length rollouts) ...", flush=True)
        base = {n_: p.detach().clone() for n_, p in net.named_parameters() if n_.startswith("z_")}
        vr0, bf0 = window_report(m, allr, a.seed, a.windows)
        FR.apply_loading_gain(net, g, which="core")
        vr1, bf1 = window_report(m, allr, a.seed, a.windows)
        import torch
        with torch.no_grad():
            for n_, p in net.named_parameters():
                if n_ in base:
                    p.copy_(base[n_])
        print(f"    {'':<14}{'before':>10}{'after':>10}{'target':>10}")
        for k in ("turn", "distance", "dt", "dot-dist", "radial"):
            print(f"    var_ratio {k:<8}{vr0.get(k, float('nan')):>10.3f}"
                  f"{vr1.get(k, float('nan')):>10.3f}{1.0:>10.3f}")
        print(f"    {'between_fish':<14}{bf0.get('gen', float('nan')):>10.3f}"
              f"{bf1.get('gen', float('nan')):>10.3f}{bf0.get('real', float('nan')):>10.3f}")
        print("\n    NOT the objective. If the levels above landed on target and these did not,\n"
              "    the level spreads were the wrong proxy and the gain is not the mechanism.")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    out = a.out or str(Path(a.ckpt).with_name(Path(a.ckpt).stem + "_pcgain.pt"))
    FR.apply_loading_gain(net, g, which="core")
    m.re_gain, m.re_gain_report = g, rep
    MX.save_mix(m, out)
    print(f"\nwrote {out}   (gains FOLDED into the loadings, with per-channel re-centring)")
    print("Score it against the parent with:  python scripts/eval_pair.py --models "
          f"{a.ckpt} {out}")


if __name__ == "__main__":
    raise SystemExit(main())
