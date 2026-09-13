"""Structural check for the two new generation-time mechanisms, on the SHIPPED k8 checkpoint.

No retraining: both are generation-time, so they can be exercised by mutating ``_cfg_feat`` on a
loaded model. That makes this a minutes-long check of the two claims that justify the rung, and
of the two failure modes that killed their predecessors.

  indiv       per-trajectory latents on turning style and spatial preference. Between-fish share
              is 0.074 against a real 0.255 and has not moved in eleven ladders, because there is
              no mechanism -- vigor covers dt and distance only and _calibrate_vigor hard-sets the
              turn term to zero. CLAIM: raises the ACROSS-ROLLOUT spread of forward fraction
              (real sd 0.054, gen 0.005) and of mean r/R (real 0.075, gen 0.016-0.044) WITHOUT
              widening the pooled |dtheta| marginal -- which is exactly how the old vigor_turn
              failed (abs_dtheta_w1 6.67 -> 8.43, because it scaled |dtheta| itself).

  innov_comp  widen the kinematic innovation about the per-bout conditional mean. CLAIM: raises
              sd(log distance) toward real 0.450 (k8 sits at ~0.83 of target) while leaving the
              lag-1 autocorrelation alone -- acf(k) = phi^k is scale-free. That orthogonality is
              the whole point: the mem ladder showed the shrinkage is NECESSARY for the acf
              (without it dist acf@1 goes 0.191 -> 0.394 against a real 0.218 and becomes the top
              C2ST tell), so the spread has to be recovered some other way. Not kin_var_comp,
              which rescaled about the fixed training log-mean, kept the history unscaled, and
              lost twice.

    python scripts/smoke_indiv.py --ckpt mem_out/ladder_k8.pt

Structural only, in-sample. Local numbers in this project overstate held-out ones badly; the real
A/B is the paired ladder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP as M                     # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402


def profile(model, recs, fids, reps=2, seed=0):
    """Per-rollout summaries; the SPREAD across rollouts is the individuality being tested."""
    fw, rad, sdl, acf1, absmed, wall = [], [], [], [], [], []
    alld = []
    for rep in range(reps):
        rng = np.random.default_rng([seed, 99, rep])
        for f in fids:
            g = model.sample(recs[f], rng)
            a = np.abs(np.asarray(g["dtheta"], float))
            fw.append(float(np.mean(a < M.DTHETA0))); absmed.append(float(np.median(a)))
            alld.append(a)
            d = np.asarray(g["distance_px"], float)
            ld = np.log(np.clip(d, 1e-6, None))
            sdl.append(float(np.std(ld)))
            x = ld - ld.mean()
            acf1.append(float((x[:-1] * x[1:]).mean() / max(x.var(), 1e-12)))
            u = MX._radial_frac(recs[f], g.get("x_px"), g.get("y_px"))
            if u is not None:
                rad.append(float(np.mean(u))); wall.append(float(np.mean(u > 0.85)))
    return dict(fw_sd=float(np.std(fw)), rad_sd=float(np.std(rad)) if rad else np.nan,
                absmed_sd=float(np.std(absmed)), rad_mean=float(np.mean(rad)) if rad else np.nan,
                wall=float(np.mean(wall)) if wall else np.nan,
                sd_logd=float(np.mean(sdl)), dist_acf1=float(np.mean(acf1)),
                absdth=np.concatenate(alld))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mem_out/ladder_k8.pt")
    ap.add_argument("--bouts", default=str(Path(__file__).parent.parent / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--fish", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    from scipy.stats import wasserstein_distance
    df = pd.read_csv(args.bouts)
    recs = MB.build_border_records(df)
    fids = sorted(recs)[:args.fish]
    tgt_fw, tgt_rad = MX._indiv_targets([recs[f] for f in sorted(recs)])
    rl = np.concatenate([np.log(np.clip(np.asarray(recs[f].distance_px, float), 1e-6, None)) for f in sorted(recs)])
    tgt_sd = float(np.std(rl[np.isfinite(rl)]))
    real_absdth = np.abs(np.concatenate([np.asarray(recs[f].dtheta, float) for f in fids]))
    ra = np.concatenate([np.asarray(np.abs(recs[f].dtheta), float) for f in fids])
    real_absmed_sd = float(np.std([float(np.median(np.abs(recs[f].dtheta))) for f in sorted(recs)]))

    m = MX.load_mix(args.ckpt)
    base = dict(m._cfg_feat)
    print(f"checkpoint {args.ckpt}   |   real targets: fw_sd {tgt_fw:.4f}  rad_sd {tgt_rad:.4f}  "
          f"sd(log dist) {tgt_sd:.3f}  absmed_sd {real_absmed_sd:.2f}\n")

    arms = [("base (k8)", {}),
            ("+indiv", {"indiv": dict(fw=0.60, rad=0.45)}),
            ("+innov", {"kin_innov": dict(dist=1.25, dt=1.15)}),
            ("+both", {"indiv": dict(fw=0.60, rad=0.45), "kin_innov": dict(dist=1.25, dt=1.15)})]
    hdr = (f"{'arm':12s}{'fw_sd':>9s}{'rad_sd':>9s}{'absmed_sd':>11s}{'sd(log d)':>11s}"
           f"{'dist_acf1':>11s}{'absdth_w1':>11s}{'mean r/R':>10s}{'frac>.85':>10s}")
    print(hdr); print("-" * len(hdr))
    out = {}
    for name, patch in arms:
        cf = dict(base); cf.pop("indiv", None); cf.pop("kin_innov", None); cf.update(patch)
        m._cfg_feat = cf
        p = profile(m, recs, fids, reps=args.reps)
        w1 = float(wasserstein_distance(real_absdth, p["absdth"]))
        out[name] = p
        print(f"{name:12s}{p['fw_sd']:9.4f}{p['rad_sd']:9.4f}{p['absmed_sd']:11.2f}"
              f"{p['sd_logd']:11.3f}{p['dist_acf1']:11.3f}{w1:11.3f}{p['rad_mean']:10.3f}{p['wall']:10.4f}")
    m._cfg_feat = base
    print(f"{'REAL':12s}{tgt_fw:9.4f}{tgt_rad:9.4f}{real_absmed_sd:11.2f}{tgt_sd:11.3f}"
          f"{0.218:11.3f}{0.0:11.3f}{0.456:10.3f}{0.0121:10.4f}")

    b, i, v = out["base (k8)"], out["+indiv"], out["+innov"]
    print("\nACCEPTANCE")
    ok = True
    for lab, got, want in (("indiv raises fw_sd", i["fw_sd"] > 3 * b["fw_sd"], True),
                           ("indiv raises rad_sd", i["rad_sd"] > 1.5 * b["rad_sd"], True),
                           ("innov raises sd(log d)", v["sd_logd"] > b["sd_logd"] * 1.05, True),
                           ("innov leaves acf ~alone (<0.03)", abs(v["dist_acf1"] - b["dist_acf1"]) < 0.03, True),
                           ("indiv does not break containment", i["wall"] < 3 * b["wall"] + 0.005, True)):
        print(f"  {'OK ' if got == want else 'FAIL'} {lab}")
        ok &= (got == want)
    print("\nThe acf check is the load-bearing one: if innov_comp moves dist_acf1 it is not")
    print("orthogonal to kin_feedback, and it will lose exactly the way removing the shrinkage did.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
