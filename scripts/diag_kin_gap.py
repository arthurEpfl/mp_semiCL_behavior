"""WHY are dt_std / dist_std / dt_median the top traj-C2ST tells even with the AR kinematics model?

Hypothesis: the kin model pools ALL fish and its distance/dt heads read ONLY generated kin history
(no fish signal), so EVERY generated trajectory has the ~same distance/dt distribution -> the per-window
dt_median / dt_std spread is too NARROW vs real (real fish are heterogeneous, ICC 0.7-0.9). The classifier
then separates real from gen on those moments. This script measures the gap and its decomposition:

  A. within-window dispersion:  distribution of per-window dt_std / dist_std, real vs gen.
  B. between-fish heterogeneity: std ACROSS fish of each fish's mean(log dt) / mean(log dist), real vs gen
     (the smoking gun: gen ~ 0 if the model has no individuality).
  C. within-fish slow drift:    lag-1 acf of the window-level dt_median series, real vs gen.
  D. turn diffuse-check:        turn-only c2st_importances (is any single turn feature a real tell?).

Fit is a SMOKE fit (structure, not performance) -- absolute numbers won't match Colab, but the DIRECTION
of each gap (gen vs real) is what we read. python scripts/diag_kin_gap.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import evaluate as E
import fit_MLP_border as MB
import fit_MLP_mix as MX
import generative_metrics as GM

W = 60


def _win(x, fn):
    x = np.asarray(x, float)
    return [fn(x[a:a + W]) for a in range(0, max(1, len(x) - W + 1), W) if len(x[a:a + W]) == W]


def _fin(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def main():
    df = E.load_bouts(str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    records = MB.build_border_records(df)
    recs = list(records.values())
    fids = [r.fish_id for r in recs]
    test = set(fids[::4])                                   # ~1/4 held-out fish
    train = [r for r in recs if r.fish_id not in test]
    testr = [r for r in recs if r.fish_id in test]
    print(f"{len(train)} train fish, {len(testr)} test fish")
    ctx = E.make_context(train, cfg={})
    print("fitting kin model (40 epochs, smoke) ...")
    m = MX.MLPMixKinDensity(K=8, stim_mode="dotb", hidden=128, use_border=True,
                            closed_loop=True, stim_loop=True, epochs=40).fit(train, ctx)

    # roll out held-out fish
    R_dtstd, G_dtstd, R_dstd, G_dstd, R_dtmed, G_dtmed = [], [], [], [], [], []
    R_fmean_dt, G_fmean_dt, R_fmean_di, G_fmean_di = [], [], [], []   # per-fish mean(log .)
    R_win_dtmed_series, G_win_dtmed_series = [], []
    turn_wins = {}
    for r in testr:
        g = m.sample(r, np.random.default_rng(hash(r.fish_id) % (2**31)))
        rd, rt = _fin(r.distance_px), _fin(r.dt_s)
        gd, gt = _fin(g["distance_px"]), _fin(g["dt_s"])
        R_dtstd += _win(r.dt_s, np.nanstd); G_dtstd += _win(g["dt_s"], np.nanstd)
        R_dstd += _win(r.distance_px, np.nanstd); G_dstd += _win(g["distance_px"], np.nanstd)
        rm = _win(r.dt_s, np.nanmedian); gm = _win(g["dt_s"], np.nanmedian)
        R_dtmed += rm; G_dtmed += gm
        R_win_dtmed_series.append(rm); G_win_dtmed_series.append(gm)
        R_fmean_dt.append(np.nanmean(np.log(rt[rt > 0]))); G_fmean_dt.append(np.nanmean(np.log(gt[gt > 0])))
        R_fmean_di.append(np.nanmean(np.log(rd[rd > 0]))); G_fmean_di.append(np.nanmean(np.log(gd[gd > 0])))
        # turn windows for the diffuse-check
        rf = GM.window_features(r.dtheta, r.symbol)
        gf = GM.window_features(g["dtheta"], g["symbol"])
        if len(rf) and len(gf):
            turn_wins[r.fish_id] = (rf, gf)

    def line(tag, R, G):
        R, G = _fin(R), _fin(G)
        print(f"  {tag:16s} real={np.mean(R):8.3f}  gen={np.mean(G):8.3f}   "
              f"gen/real={np.mean(G)/max(np.mean(R),1e-9):5.2f}")

    print("\n[A] within-window dispersion (mean of per-window statistic):")
    line("dt_std", R_dtstd, G_dtstd)
    line("dist_std", R_dstd, G_dstd)

    print("\n[B] BETWEEN-FISH heterogeneity (std ACROSS fish of each fish's mean log-value):")
    print(f"  mean(log dt)   : real std={np.nanstd(R_fmean_dt):.3f}  gen std={np.nanstd(G_fmean_dt):.3f}   "
          f"gen/real={np.nanstd(G_fmean_dt)/max(np.nanstd(R_fmean_dt),1e-9):.2f}")
    print(f"  mean(log dist) : real std={np.nanstd(R_fmean_di):.3f}  gen std={np.nanstd(G_fmean_di):.3f}   "
          f"gen/real={np.nanstd(G_fmean_di)/max(np.nanstd(R_fmean_di),1e-9):.2f}")

    print("\n[C] across-WINDOW spread of dt_median (heterogeneity the classifier sees on dt_median):")
    line("std(dt_median)", [np.nanstd(_fin(R_dtmed))], [np.nanstd(_fin(G_dtmed))])

    def acf1_series(series_list):
        num = den = 0.0
        for s in series_list:
            s = _fin(s)
            if len(s) < 3:
                continue
            x = s - s.mean(); num += float(np.sum(x[1:] * x[:-1])); den += float(np.sum(x * x))
        return num / den if den else float("nan")
    print("\n[C2] within-fish slow drift (lag-1 acf of window-level dt_median):")
    print(f"  real={acf1_series(R_win_dtmed_series):+.3f}  gen={acf1_series(G_win_dtmed_series):+.3f}")

    print("\n[D] turn-only c2st_importances (any single turn feature a real tell?):")
    imp = GM.c2st_importances(turn_wins, GM.C2ST_FEATURE_NAMES).sort_values(ascending=False)
    print(imp.head(6).to_string())
    print(f"  turn c2st AUC (held-out) = {GM.c2st_auc(turn_wins)[0]:.3f}")


if __name__ == "__main__":
    main()
