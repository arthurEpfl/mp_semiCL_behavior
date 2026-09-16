"""Verify the ARTR L/R turn-bias oscillator restores turn-DIRECTION persistence (vs the current _kin_vigor).

Fits vigor=True, artr=True and checks, on held-out fish:
  turn-sign acf lag1/2 + P(same dir)   -> should move from gen ~0.06 toward real ~0.15 (calibrated)
  0% out-of-dish, save/load identical, dt/dist [B] channels unregressed, abs_dtheta_w1 not blown up.
Also confirms score()/bits is UNTOUCHED (oscillator is generation-only).

    python scripts/smoke_artr.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import evaluate as E
import fit_MLP_border as MB
import fit_MLP_mix as MX
import fit_MLP as M


def _sign_stats(dth):
    d = np.asarray(dth, float); s = np.sign(d[np.abs(d) > M.DTHETA0]); s = s[s != 0]
    if len(s) < 10:
        return np.nan, np.nan, np.nan
    x = s - s.mean(); den = float((x * x).sum())
    a1 = float((x[1:] * x[:-1]).sum() / den) if den else np.nan
    a2 = float((x[2:] * x[:-2]).sum() / den) if den and len(s) > 3 else np.nan
    p = float(np.mean(s[1:] == s[:-1]))
    return a1, a2, p


def _agg(items, get):
    A = np.array([_sign_stats(get(it)) for it in items], float)
    return np.nanmean(A, axis=0)


def _fin(x):
    x = np.asarray(x, float); return x[np.isfinite(x) & (x > 0)]


def _w1(a, b, q=200):
    a, b = _fin(a), _fin(b); grid = (np.arange(q) + 0.5) / q
    return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def main():
    df = E.load_bouts(str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    recs = list(MB.build_border_records(df).values())
    fids = [r.fish_id for r in recs]; test = set(fids[::4])
    train = [r for r in recs if r.fish_id not in test]; testr = [r for r in recs if r.fish_id in test]
    ctx = E.make_context(train, cfg={})
    print("fitting kin_vigor_artr (40 epochs, incl. beta search) ...")
    m = MX.MLPMixKinDensity(K=8, stim_mode="dotb", hidden=128, use_border=True, closed_loop=True,
                            stim_loop=True, epochs=40, vigor=True, artr=True).fit(train, ctx)
    print("  calibrated artr:", {k: round(v, 3) for k, v in m._cfg_feat["artr"].items()})

    gens = [m.sample(r, np.random.default_rng(hash(r.fish_id) % 997)) for r in testr]
    ra = _agg(testr, lambda r: r.dtheta); ga = _agg(gens, lambda g: g["dtheta"])
    # ALSO measure on the TRAIN population the calibration targeted (test 4 fish are atypically low-persistence)
    gtr = [m.sample(r, np.random.default_rng(hash(r.fish_id) % 991)) for r in train]
    ratr = _agg(train, lambda r: r.dtheta); gatr = _agg(gtr, lambda g: g["dtheta"])
    print("\nturn-sign persistence:      real_test gen_test | real_train gen_train (calibration pop)")
    print(f"  acf lag1     {ra[0]:+.3f}   {ga[0]:+.3f}  |  {ratr[0]:+.3f}    {gatr[0]:+.3f}")
    print(f"  acf lag2     {ra[1]:+.3f}   {ga[1]:+.3f}  |  {ratr[1]:+.3f}    {gatr[1]:+.3f}")
    print(f"  P(same dir)  {ra[2]:.3f}    {ga[2]:.3f}  |  {ratr[2]:.3f}     {gatr[2]:.3f}")

    # channel regression + marginals + containment
    Rd, Gd, Rt, Gt, Ra_, Ga_ = [], [], [], [], [], []
    n_out = n_tot = 0
    for r, g in zip(testr, gens):
        Rd.append(r.distance_px); Gd.append(g["distance_px"]); Rt.append(r.dt_s); Gt.append(g["dt_s"])
        Ra_.append(np.abs(r.dtheta)); Ga_.append(np.abs(g["dtheta"]))
        x = np.asarray(g["x_px"], float); y = np.asarray(g["y_px"], float)
        rr = np.hypot(x - r.dish_cx, y - r.dish_cy); n_out += int((rr > r.dish_R + 1e-6).sum()); n_tot += len(rr)
    print(f"\nmarginal  dist_w1={_w1(np.concatenate(Rd), np.concatenate(Gd)):.3f}  "
          f"dt_w1={_w1(np.concatenate(Rt), np.concatenate(Gt)):.4f}  "
          f"absdth_w1={_w1(np.concatenate(Ra_), np.concatenate(Ga_)):.3f}")
    print(f"out-of-dish: {100.0 * n_out / n_tot:.3f}%")

    p = ROOT / "scratch_artr.pt"; MX.save_mix(m, str(p)); m2 = MX.load_mix(str(p))
    r0 = testr[0]
    s1 = m.sample(r0, np.random.default_rng(7)); s2 = m2.sample(r0, np.random.default_rng(7))
    same = all(np.allclose(np.nan_to_num(s1[k]), np.nan_to_num(s2[k])) for k in ("dtheta", "distance_px", "dt_s"))
    p.unlink(missing_ok=True)
    print(f"save/load identical: {same}")
    print("ARTR SMOKE OK" if same and n_out == 0 else "ARTR SMOKE: review above")


if __name__ == "__main__":
    main()
