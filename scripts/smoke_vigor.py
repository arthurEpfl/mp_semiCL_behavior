"""Verify the swim-vigor latent closes the measured gaps (vs the plain kin model), on held-out fish.

Fits vigor=True, vigor_turn=True (all three channels) and checks real-vs-gen for:
  [B] between-fish LEVEL std of mean(log dt/dist)   -- was gen 0.016/0.010 vs real 0.174/0.156
  [C2] within-fish window-median drift lag-1 acf     -- was gen +0.06 vs real +0.38
  [A] within-window dt_std                           -- was gen 0.70x real
  TURN [B] between-traj std of log(mean|dtheta|)     -- turn-magnitude individuality (new lever)
  TURN absdth_median per-window spread               -- the c2st feature we target
  marginal dist_w1 / dt_w1 / abs_dtheta_w1 (must NOT blow up), 0% out-of-dish, save/load identical.

    python scripts/smoke_vigor.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import evaluate as E
import fit_MLP_border as MB
import fit_MLP_mix as MX

W = 60


def _fin(x):
    x = np.asarray(x, float); return x[np.isfinite(x) & (x > 0)]


def _win(x, fn):
    x = np.asarray(x, float)
    return [fn(x[a:a + W]) for a in range(0, max(1, len(x) - W + 1), W) if len(x[a:a + W]) == W]


def _winmed_c(x):
    lx = np.log(np.clip(np.asarray(x, float), 1e-6, None))
    m = np.array(_win(lx, np.nanmedian))
    return (m - m.mean()) if len(m) >= 3 else np.zeros(0)


def _acf1(series):
    num = den = 0.0
    for s in series:
        s = np.asarray(s, float); s = s[np.isfinite(s)]
        if len(s) < 3:
            continue
        x = s - s.mean(); num += float((x[1:] * x[:-1]).sum()); den += float((x * x).sum())
    return num / den if den else float("nan")


def _w1(a, b, q=200):
    a, b = _fin(a), _fin(b); grid = (np.arange(q) + 0.5) / q
    return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def _logmeanabs(dth):
    a = np.abs(np.asarray(dth, float)); a = a[np.isfinite(a)]
    return float(np.log(max(a.mean(), 1e-2))) if len(a) else np.nan


def main():
    df = E.load_bouts(str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    recs = list(MB.build_border_records(df).values())
    fids = [r.fish_id for r in recs]
    test = set(fids[::4]); train = [r for r in recs if r.fish_id not in test]
    testr = [r for r in recs if r.fish_id in test]
    ctx = E.make_context(train, cfg={})
    print(f"{len(train)} train / {len(testr)} test fish; fitting kin_vigor_turn (40 epochs) ...")
    m = MX.MLPMixKinDensity(K=8, stim_mode="dotb", hidden=128, use_border=True,
                            closed_loop=True, stim_loop=True, epochs=40, vigor=True, vigor_turn=True).fit(train, ctx)
    print("  calibrated vigor:", {k: [round(v, 3) for v in m._cfg_feat['vigor'][k]]
                                   for k in ('dt', 'dist', 'turn')})

    Rlvl_dt, Glvl_dt, Rlvl_di, Glvl_di = [], [], [], []
    Rlvl_tn, Glvl_tn = [], []                             # log(mean|dtheta|) per trajectory (turn level)
    Rwm_dt, Gwm_dt = [], []
    Rdtstd, Gdtstd = [], []
    Rabsmed, Gabsmed = [], []                             # per-window median|dtheta| (absdth_median feature)
    Rd, Gd, Rt, Gt, Ra, Ga = [], [], [], [], [], []
    n_out = n_tot = 0
    for r in testr:
        for rep in range(3):                              # 3 draws/fish -> more gen trajectories for the std
            g = m.sample(r, np.random.default_rng(1000 * rep + hash(r.fish_id) % 997))
            Glvl_dt.append(np.log(_fin(g["dt_s"])).mean()); Glvl_di.append(np.log(_fin(g["distance_px"])).mean())
            Glvl_tn.append(_logmeanabs(g["dtheta"]))
            Gwm_dt.append(_winmed_c(g["dt_s"])); Gdtstd += _win(g["dt_s"], np.nanstd)
            Gabsmed += _win(np.abs(g["dtheta"]), np.nanmedian)
            Gd.append(g["distance_px"]); Gt.append(g["dt_s"]); Ga.append(np.abs(g["dtheta"]))
            x = np.asarray(g["x_px"], float); y = np.asarray(g["y_px"], float)
            rr = np.hypot(x - r.dish_cx, y - r.dish_cy); n_out += int((rr > r.dish_R + 1e-6).sum()); n_tot += len(rr)
        Rlvl_dt.append(np.log(_fin(r.dt_s)).mean()); Rlvl_di.append(np.log(_fin(r.distance_px)).mean())
        Rlvl_tn.append(_logmeanabs(r.dtheta))
        Rwm_dt.append(_winmed_c(r.dt_s)); Rdtstd += _win(r.dt_s, np.nanstd)
        Rabsmed += _win(np.abs(r.dtheta), np.nanmedian)
        Rd.append(r.distance_px); Rt.append(r.dt_s); Ra.append(np.abs(r.dtheta))

    print("\n[B] between-traj LEVEL std of mean(log .):  (real -> gen; was gen 0.016/0.010)")
    print(f"    dt   real={np.std(Rlvl_dt):.3f}  gen={np.std(Glvl_dt):.3f}")
    print(f"    dist real={np.std(Rlvl_di):.3f}  gen={np.std(Glvl_di):.3f}")
    print(f"[C2] window-median dt drift acf:  real={_acf1(Rwm_dt):+.3f}  gen={_acf1(Gwm_dt):+.3f}")
    print(f"[A] within-window dt_std:  real={np.mean(_fin(Rdtstd)):.3f}  gen={np.mean(_fin(Gdtstd)):.3f}")
    print("\nTURN [B] between-traj std of log(mean|dtheta|):  (was gen ~0)")
    print(f"    real={np.nanstd(Rlvl_tn):.3f}  gen={np.nanstd(Glvl_tn):.3f}")
    print(f"TURN absdth_median per-window spread (std across windows):  "
          f"real={np.nanstd(_fin(Rabsmed)):.3f}  gen={np.nanstd(_fin(Gabsmed)):.3f}")
    print(f"\nmarginal  dist_w1={_w1(np.concatenate(Rd), np.concatenate(Gd)):.3f}  "
          f"dt_w1={_w1(np.concatenate(Rt), np.concatenate(Gt)):.4f}  "
          f"absdth_w1={_w1(np.concatenate(Ra), np.concatenate(Ga)):.3f}")
    print(f"out-of-dish: {100.0 * n_out / n_tot:.3f}%")

    p = ROOT / "scratch_vigor.pt"; MX.save_mix(m, str(p)); m2 = MX.load_mix(str(p))
    r0 = testr[0]
    s1 = m.sample(r0, np.random.default_rng(7)); s2 = m2.sample(r0, np.random.default_rng(7))
    same = all(np.allclose(np.nan_to_num(s1[k]), np.nan_to_num(s2[k])) for k in ("dtheta", "distance_px", "dt_s"))
    p.unlink(missing_ok=True)
    print(f"save/load identical: {same}")
    print("VIGOR SMOKE OK" if same and n_out == 0 else "VIGOR SMOKE: review above")


if __name__ == "__main__":
    main()
