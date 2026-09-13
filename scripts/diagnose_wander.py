"""Diagnose WHY the honest simulator wanders further from the dots than real fish.

The honest `dotb_hloop` fish is FURTHER from the dots on average (mean ~110 vs 88 px) and spends
more time far -- a proximity-REGULATION gap that per-bout feature tweaks didn't touch. This decomposes
the excess fish-to-dot distance into candidate causes so we target the right fix:
  1. unregulated EXPLORATION -- is the excess distance on non-engaged (non-moving-dot) bouts?
  2. STEP-SIZE regulation  -- do real fish shorten their bouts when far (and gen, drawing from a fixed
                              pool, doesn't)? -> the joint step-size model would help.
  3. FOLLOWING too weak/gated -- does gen make LONGER away-excursions (following can't pull it back)?

Read-only: fits the honest model, rolls out, prints the decomposition + a heuristic verdict.

    python scripts/diagnose_wander.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import evaluate as E
import fit_MLP_border as MB
import fit_MLP_mix as MX

NEAR, FAR = 40.0, 120.0


def _cat(lst):
    lst = [np.asarray(a, float) for a in lst if np.size(a)]
    return np.concatenate(lst) if lst else np.array([np.nan])


def _mean(lst):
    return float(np.nanmean(_cat(lst)))


def _streaks(mask):
    """lengths of consecutive-True runs (here: consecutive receding bouts)."""
    out, c = [], 0
    for v in mask:
        if v:
            c += 1
        elif c:
            out.append(c); c = 0
    if c:
        out.append(c)
    return out


def main():
    df = E.load_bouts(str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    records = MB.build_border_records(df); recs = list(records.values())
    ctx = E.make_context(recs, cfg={})
    print("fitting honest dotb_hloop (40 epochs) ...")
    m = MX.MLPMixLoopDensity(K=8, stim_mode="dotb", hidden=128, use_border=True,
                             closed_loop=True, stim_loop=True, epochs=40).fit(recs, ctx)
    rng = np.random.default_rng(0)

    Rmv, Rno, Gmv, Gno = [], [], [], []                 # dist-to-dot by engagement
    Rsn, Rsf, Gsn, Gsf = [], [], [], []                 # step size near / far
    Rstreak, Gstreak = [], []                            # away-excursion run lengths
    Rlunge, Glunge = [], []                              # moving-bout closing rate

    for r in recs:
        dd = np.asarray(r.dot_dist_px, float); ok = np.isfinite(dd)
        if ok.sum() < 5:
            continue
        dcx = np.asarray(r.dot_cx, float); dcy = np.asarray(r.dot_cy, float)
        mv = np.asarray(r.moving, bool)
        rstep = np.asarray(r.distance_px, float)
        Rmv.append(dd[ok & mv]); Rno.append(dd[ok & ~mv])
        Rsn.append(rstep[ok & (dd < NEAR)]); Rsf.append(rstep[ok & (dd > FAR)])
        Rstreak += _streaks(np.diff(dd[ok]) > 0)         # dist increasing = receding
        pm = (ok & mv)[:-1] & (ok & mv)[1:]
        Rlunge.append((dd[:-1] - dd[1:])[pm])            # + = got closer
        for _ in range(4):
            g = m.sample(r, rng)
            gdd = np.hypot(g["x_px"] - dcx, g["y_px"] - dcy)
            gstep = np.asarray(g["distance_px"], float)
            Gmv.append(gdd[ok & mv]); Gno.append(gdd[ok & ~mv])
            Gsn.append(gstep[ok & (gdd < NEAR)]); Gsf.append(gstep[ok & (gdd > FAR)])
            Gstreak += _streaks(np.diff(gdd[ok]) > 0)
            Glunge.append((gdd[:-1] - gdd[1:])[pm])

    print("\n=== 1) mean dist-to-dot by engagement (px) ===")
    print(f"{'':11s}   REAL    GEN   excess(gen-real)")
    exc = {}
    for k, (RL, GL) in [("moving-dot", (Rmv, Gmv)), ("non-moving", (Rno, Gno))]:
        rr, gg = _mean(RL), _mean(GL); exc[k] = gg - rr
        print(f"{k:11s} {rr:6.1f} {gg:6.1f}   {gg - rr:+6.1f}")

    print("\n=== 2) step size (distance_px) when NEAR vs FAR (px) ===")
    rn, rf, gn, gf = _mean(Rsn), _mean(Rsf), _mean(Gsn), _mean(Gsf)
    print(f"{'':10s}  REAL    GEN")
    print(f"near(<40) {rn:6.1f} {gn:6.1f}")
    print(f"far(>120) {rf:6.1f} {gf:6.1f}")
    print(f"  real far/near step ratio = {rf / max(rn,1e-9):.2f}   "
          f"gen far/near = {gf / max(gn,1e-9):.2f}   (<0.9 => shortens steps when far)")

    print("\n=== 3) away-excursion streak length (consecutive receding bouts) ===")
    print(f"mean streak  REAL {np.mean(Rstreak):.2f}   GEN {np.mean(Gstreak):.2f}   "
          f"90th pct  REAL {np.quantile(Rstreak,.9):.0f}  GEN {np.quantile(Gstreak,.9):.0f}")

    print("\n=== 4) moving-bout lunge (px/bout, + = closing) ===")
    print(f"REAL {np.nanmean(_cat(Rlunge)):+.3f}   GEN {np.nanmean(_cat(Glunge)):+.3f}")

    print("\n=== verdict ===")
    ruled_out = []
    if not (exc["non-moving"] > max(exc["moving-dot"], 0) * 1.3):
        ruled_out.append("exploration (excess is NOT concentrated on non-engaged bouts)")
    if not (rf / max(rn, 1e-9) < 0.9 and gf / max(gn, 1e-9) > 0.95):
        ruled_out.append("step-size (real doesn't shorten bouts when far either -> joint step model won't help)")
    if not (np.mean(Gstreak) > np.mean(Rstreak) * 1.2):
        ruled_out.append("longer excursions (gen away-streaks are not longer than real)")
    print("RULED OUT:", "; ".join(ruled_out) if ruled_out else "(none)")
    # the positive finding: does gen hold station further even when ENGAGED?
    if exc["moving-dot"] > 8:
        print(f"FINDING: EQUILIBRIUM / SET-POINT -- on engaging (moving-dot) bouts gen holds station "
              f"{exc['moving-dot']:+.0f} px FURTHER than real ({_mean(Gmv):.0f} vs {_mean(Rmv):.0f} px) "
              f"while lunging inward ({np.nanmean(_cat(Glunge)):+.2f}/bout). So its attraction is directionally "
              f"right (following_gap_h matches) but too WEAK to hold the tight close band -> the fix is "
              f"following STRENGTH / tracking-tightness (turn magnitude toward the dot), not steps/exploration.")


if __name__ == "__main__":
    main()
