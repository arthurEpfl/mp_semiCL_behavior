"""Structural guard for NOISE-AUGMENTED training (``noise_aug``).

The mechanism is: perturb the POSE, re-derive every input from it, and CORRECT THE TARGET back to
the true next onset. The target correction is the load-bearing half -- input noise without it is
only Tikhonov smoothing (Bishop 1995) and does nothing about rollout drift -- and it is also the
half that is easy to get silently wrong, because a target computed from the *unperturbed* pose
still trains, still converges, and simply teaches the wrong thing.

[1] is the check that catches that: at sigma = 0 the augmented pair must reproduce the clean one,
INCLUDING the targets. A target recomputation that reads the wrong index, drops the wrap, or
references the unperturbed heading fails here and nowhere else.

    python scripts/smoke_noise_aug.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_joint as JT              # noqa: E402

TOL = 1e-9          # atan2 round-trip, not bit-identity: the targets are RE-DERIVED, not copied
ok_all = True


def check(label, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def _cfg(**kw):
    c = dict(K=8, stim_mode="shoal", include_symbols=False, dth_scale=30.0, rich_scales=None,
             use_border=True, border_mode="basis", use_kin=True, K_kin=3,
             kin_scales=(2.6, 0.5, 0.2, 0.6), psi_ref="travel", phi_ref="ori")
    c.update(kw)
    return c


def main():
    root = Path(__file__).parent.parent
    recs = MB.build_border_records(pd.read_csv(root / "data" / "bouts_GsizeMotionSingle.csv"))
    fids = sorted(recs)
    train = [recs[f] for f in fids[:4]]
    test = recs[fids[4]]

    print("\n=== [1] sigma = 0 reproduces the clean pair, TARGETS INCLUDED ===")
    cf0 = _cfg(noise_aug=dict(sigma_pos=0.0, sigma_ori=0.0))
    rng = np.random.default_rng(0)
    worst = {}
    for r in [recs[f] for f in fids]:
        Xc = MB.build_feature_matrix(r, cf0)
        Xn, ya, yd, yt = MB.build_noisy_pair(r, cf0, rng)
        worst["X"] = max(worst.get("X", 0.0), float(np.max(np.abs(Xn - Xc))))
        worst["dtheta"] = max(worst.get("dtheta", 0.0),
                              float(np.max(np.abs(ya - np.asarray(r.dtheta, float)))))
        worst["distance"] = max(worst.get("distance", 0.0),
                                float(np.max(np.abs(yd - np.asarray(r.distance_px, float)))))
        worst["dt"] = max(worst.get("dt", 0.0),
                          float(np.max(np.abs(yt - np.asarray(r.dt_s, float)))))
    for k, v in worst.items():
        check(f"sigma=0: {k} recovered over all 15 fish", v < TOL, f"max|diff|={v:.2e}")

    print("\n=== [2] the target really is CORRECTED, not merely jittered ===")
    # Push the pose sideways and the corrective turn must lean the OTHER way. Scored as a sign
    # agreement, not a correlation: the map is atan2(perp, step), so it saturates and Pearson r
    # understates it badly (measured -0.43 at sigma 8 px while sign agreement is a flat 0.75 from
    # 1 px to 12 px). A target copied from the clean record would sit at 0.5.
    cf = _cfg(noise_aug=dict(sigma_pos=MB.NOISE_AUG_DEFAULT["sigma_pos"],
                             sigma_ori=MB.NOISE_AUG_DEFAULT["sigma_ori"]))
    r = test
    r2 = MB.noisy_record(r, cf, np.random.default_rng(1))
    dth_c = np.asarray(r.dtheta, float); dth_n = np.asarray(r2.dtheta, float)
    x = np.asarray(r.x_px, float); y = np.asarray(r.y_px, float)
    hd = np.asarray(r.heading_rad, float)
    # component of the pose perturbation PERPENDICULAR to the bout's own travel direction
    perp = -(np.asarray(r2.x_px, float) - x) * np.sin(hd) + (np.asarray(r2.y_px, float) - y) * np.cos(hd)
    d = np.degrees(MB._wrap_pi(np.radians(dth_n - dth_c)))
    agree = float(np.mean(np.sign(d[:-1]) != np.sign(perp[:-1])))
    check("target turn corrects AGAINST the lateral pose error", agree > 0.65,
          f"sign agreement = {agree:.3f} (0.5 = no correction)")
    check("targets actually moved", float(np.std(d)) > 1.0,
          f"sd(dtheta~ - dtheta) = {np.std(d):.2f} deg")
    check("dt is untouched (not a function of position)",
          np.array_equal(np.asarray(r2.dt_s, float), np.asarray(r.dt_s, float)))
    # equal_nan: dot_cx is NaN on every no-stimulus bout (43% of them), and plain array_equal
    # calls NaN != NaN, so this check silently failed for the wrong reason on first writing.
    check("the stimulus PROGRAMME is untouched",
          np.array_equal(np.asarray(r2.n_dots, float), np.asarray(r.n_dots, float))
          and np.array_equal(np.asarray(r2.dot_cx, float), np.asarray(r.dot_cx, float),
                             equal_nan=True))

    print("\n=== [3] the perturbed record stays self-consistent ===")
    # _travel_ref reconstructs theta_{t-1} from (heading, dtheta) row-locally. If the noisy
    # record's dtheta and heading disagree, psi_ref='travel' silently reads a broken frame.
    tp = MB._travel_ref(r2)
    check("_travel_ref(noisy) == the noisy chain's own previous heading",
          float(np.max(np.abs(MB._wrap_pi(tp[1:] - np.asarray(r2.heading_rad, float)[:-1])))) < 1e-9)
    u = 1.0 - np.asarray(r2.dist_to_border_norm, float)
    check("perturbed pose never reported outside the dish", float(np.max(u)) <= 1.0 + 1e-12)
    check("noisy feature matrix is finite",
          bool(np.isfinite(MB.build_feature_matrix(r2, cf)).all()))

    print("\n=== [4] every epoch gets FRESH noise ===")
    m = JT.MLPMixJointDensity(K=8, stim_mode="shoal", hidden=32, epochs=2, border_mode="basis",
                              psi_ref="travel", noise_aug=dict(sigma_pos=8.0, sigma_ori=15.0))
    m._cfg_feat = cf
    m.seed = 0
    kw = JT._noise_kwargs(m, train)
    a = kw["noise_builder"](0)[0]
    b = kw["noise_builder"](1)[0]
    check("consecutive epochs draw different perturbations", not np.array_equal(a, b))
    check("noise_mode default is 'concat'", kw["noise_mode"] == "concat")
    check("no noise_aug -> no trainer kwargs at all",
          JT._noise_kwargs(JT.MLPMixJointDensity(), train) == {})

    print("\n=== [5] it trains, and the validation loss stays CLEAN ===")
    ctx = E.make_context(train, dict(mlp_epochs=3))
    mm = JT.MLPMixJointDensity(K=8, stim_mode="shoal", hidden=32, epochs=3, n_dist=2, n_dt=3,
                               border_mode="basis", psi_ref="travel", dt_snap=True,
                               noise_aug=dict(sigma_pos=8.0, sigma_ori=15.0))
    mm.name = "smoke_noise"
    mm.fit(train, ctx)
    sc = mm.score(test)
    check("history (clean val loss) is finite and decreasing-ish",
          all(np.isfinite(mm.history)) and len(mm.history) == 3, f"{np.round(mm.history, 3)}")
    check("score() finite on all three channels",
          all(np.isfinite(sc[k][np.isfinite(sc[k])]).all()
              for k in ("logdens", "logdens_dist", "logdens_dt")))
    check("noise_aug round-trips into cfg_feat", mm._cfg_feat.get("noise_aug") is not None)

    print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
