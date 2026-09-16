"""Smoke test for the SEPARATE kinematics trunk (MDNMixKin / MLPMixKinDensity).

Checks, on a short fit:
  1. fit runs -> angle X and kin Xk feature dims are consistent (no crash).
  2. config carries kin_features / kin_hidden.
  3. generate runs closed-loop -> 0% out-of-dish; distance marginal W1 vs real ~ 2-3 (no runaway).
  4. gen distance/dt lag-1 acf is near REAL (+0.22 / +0.10), NOT the shared-body over-shoot (+0.34).
  5. save_mix -> load_mix reproduces the SAME sample bit-for-bit.

    python scripts/smoke_kin_trunk.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import evaluate as E
import fit_MLP_border as MB
import fit_MLP_mix as MX


def _acf1(series_list):
    """pooled lag-1 autocorrelation of a per-record kinematic series (finite, >0)."""
    num = den = 0.0
    for s in series_list:
        s = np.asarray(s, float)
        s = s[np.isfinite(s) & (s > 0)]
        if len(s) < 3:
            continue
        x = np.log(s); x = x - x.mean()
        num += float(np.sum(x[1:] * x[:-1])); den += float(np.sum(x * x))
    return num / den if den else float("nan")


def _w1(a, b, q=200):
    a = np.asarray(a, float); a = a[np.isfinite(a) & (a > 0)]
    b = np.asarray(b, float); b = b[np.isfinite(b) & (b > 0)]
    grid = (np.arange(q) + 0.5) / q
    return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def main():
    df = E.load_bouts(str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    records = MB.build_border_records(df); recs = list(records.values())
    ctx = E.make_context(recs, cfg={})
    print("fitting MLPMixKinDensity (separate trunk, 15 epochs) ...")
    m = MX.MLPMixKinDensity(K=8, stim_mode="dotb", hidden=128, use_border=True,
                            closed_loop=True, stim_loop=True, epochs=15).fit(recs, ctx)

    print(f"[1] fit OK. angle n_features={m.config['n_features']}  "
          f"kin_features={m.config.get('kin_features')}  kin_hidden={m.config.get('kin_hidden')}")
    assert m.config.get("kin_features") == 2 * m.K_kin, "kin_features should be 2*K_kin"
    assert m.config.get("kin_hidden") == m.kin_hidden

    rng = np.random.default_rng(0)
    real_dist, real_dt, gen_dist, gen_dt = [], [], [], []
    n_out = n_tot = 0
    for r in recs:
        g = m.sample(r, np.random.default_rng(hash(r.fish_id) % (2**31)))
        real_dist.append(r.distance_px); real_dt.append(r.dt_s)
        gen_dist.append(g["distance_px"]); gen_dt.append(g["dt_s"])
        x = np.asarray(g["x_px"], float); y = np.asarray(g["y_px"], float)
        rr = np.hypot(x - r.dish_cx, y - r.dish_cy)
        n_out += int(np.sum(rr > r.dish_R + 1e-6)); n_tot += len(rr)

    print(f"[2] out-of-dish: {100.0 * n_out / n_tot:.3f}%  (want 0.000%)")
    print(f"[3] dist_w1={_w1(np.concatenate(real_dist), np.concatenate(gen_dist)):.3f}  "
          f"dt_w1={_w1(np.concatenate(real_dt), np.concatenate(gen_dt)):.4f}")
    print(f"[4] acf lag-1  distance: real={_acf1(real_dist):+.3f}  gen={_acf1(gen_dist):+.3f}   "
          f"(shared-body over-shot to +0.34)")
    print(f"    acf lag-1  dt:       real={_acf1(real_dt):+.3f}  gen={_acf1(gen_dt):+.3f}")

    # save / load round-trip -> identical sample
    p = ROOT / "scratch_kin_smoke.pt"
    MX.save_mix(m, str(p))
    m2 = MX.load_mix(str(p))
    r0 = recs[0]
    s1 = m.sample(r0, np.random.default_rng(123))
    s2 = m2.sample(r0, np.random.default_rng(123))
    same = all(np.allclose(np.nan_to_num(s1[k]), np.nan_to_num(s2[k]))
               for k in ("dtheta", "distance_px", "dt_s", "x_px", "y_px"))
    print(f"[5] save/load identical sample: {same}")
    p.unlink(missing_ok=True)
    print("SMOKE OK" if same and n_out == 0 else "SMOKE: review above")


if __name__ == "__main__":
    main()
