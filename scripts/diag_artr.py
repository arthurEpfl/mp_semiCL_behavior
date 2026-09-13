"""Measure the turn-DIRECTION persistence (ARTR oscillator signature) real vs the current model.

The c2st turn tell has a big persistence/transition part (acf_lag*, T_LR/T_LL). Real zebrafish have a slow
left/right turning bias (ARTR): L/R turn direction is autocorrelated over several bouts. This measures, on
turning bouts (symbol L/R, skipping forward F), for real and generated (current _kin_vigor model):
  - acf_lag1/2/3 of the turn sign (+1 R, -1 L)
  - persistence P(next turn same direction as current)
  - mean run-length of same-direction turn streaks  -> the oscillator TIMESCALE (sets the latent's phi)
  - the L/R symbol transition persistence (T_LL+T_RR vs T_LR+T_RL)
so we know the target acf, the gap, and the switch timescale before designing the latent.

    python scripts/diag_artr.py
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


def _turnsign(dth):
    """+1 / -1 on turning bouts (|dtheta| > DTHETA0), dropping forward bouts."""
    d = np.asarray(dth, float)
    turning = np.abs(d) > M.DTHETA0
    s = np.sign(d[turning])
    return s[s != 0]


def _acf(s, lags=(1, 2, 3)):
    s = np.asarray(s, float)
    if len(s) < max(lags) + 2:
        return [np.nan] * len(lags)
    x = s - s.mean(); den = float((x * x).sum())
    return [float((x[k:] * x[:-k]).sum() / den) if den else np.nan for k in lags]


def _runlen(s):
    if len(s) < 2:
        return np.nan
    runs, c = [], 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            c += 1
        else:
            runs.append(c); c = 1
    runs.append(c)
    return float(np.mean(runs))


def _persist(s):
    if len(s) < 2:
        return np.nan
    return float(np.mean(s[1:] == s[:-1]))


def _agg(recs, get):
    A = {"acf": [], "persist": [], "runlen": []}
    for r in recs:
        s = _turnsign(get(r))
        if len(s) < 10:
            continue
        A["acf"].append(_acf(s)); A["persist"].append(_persist(s)); A["runlen"].append(_runlen(s))
    acf = np.nanmean(np.array(A["acf"]), axis=0)
    return acf, float(np.nanmean(A["persist"])), float(np.nanmean(A["runlen"]))


def main():
    df = E.load_bouts(str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    recs = list(MB.build_border_records(df).values())
    fids = [r.fish_id for r in recs]; test = set(fids[::4])
    train = [r for r in recs if r.fish_id not in test]; testr = [r for r in recs if r.fish_id in test]
    ctx = E.make_context(train, cfg={})
    print("fitting current best (_kin_vigor, 40ep) ...")
    m = MX.MLPMixKinDensity(K=8, stim_mode="dotb", hidden=128, use_border=True,
                            closed_loop=True, stim_loop=True, epochs=40, vigor=True).fit(train, ctx)

    gens = []
    for r in testr:
        gens.append(m.sample(r, np.random.default_rng(hash(r.fish_id) % 997)))

    racf, rp, rrl = _agg(testr, lambda r: r.dtheta)
    gacf, gp, grl = _agg(range(len(gens)), lambda i: gens[i]["dtheta"])

    print("\nturn-DIRECTION persistence (turning bouts only):   real     gen")
    print(f"  acf sign lag1        {racf[0]:+.3f}   {gacf[0]:+.3f}")
    print(f"  acf sign lag2        {racf[1]:+.3f}   {gacf[1]:+.3f}")
    print(f"  acf sign lag3        {racf[2]:+.3f}   {gacf[2]:+.3f}")
    print(f"  P(next same dir)     {rp:.3f}    {gp:.3f}")
    print(f"  mean run-length      {rrl:.2f}     {grl:.2f}   (-> switch prob ~ 1/runlen; latent phi)")


if __name__ == "__main__":
    main()
