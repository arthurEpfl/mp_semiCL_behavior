"""How should the repulsion turn term combine SEVERAL close dots? A held-out likelihood test on our own recordings
(2026-10-18).

The turn term (report Model 43, fit_MLP_joint.repulsion_shifts) was calibrated on single-dot bouts; in groups it sums
over dots. Real fish in our experiment do pass close to individual dots inside 2-8-dot groups, so the combination rule
can be scored on real multi-dot bouts. Every fish is scored by the fold model that never saw it (F + turn, 43a / 43b,
network + its own profile term, teacher-forced), with only the turn term changed:
    none     s = 0                                       (F without the term)
    sum      s = -a sum_j sin(phi_j) sig_j               (the term as built)
    nearest  s = -a sin(phi_j*) sig_j*,  j* = the nearest dot
    average  s = -a sum_j sin(phi_j) sig_j / max(1, sum_j sig_j)   (two dots on one side push like one; opposite cancel)
sig_j = sigma((r_c - d_j) / w). Log density of the real turn under each, on moving multi-dot bouts with >= 2 dots in
reach (where the rules differ); bits per bout relative to 'sum', bootstrap over fish.

    python repulsion_combine.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "Model"))
import fit_MLP_border as MB  # noqa: E402
import fit_MLP_joint as JT  # noqa: E402
import fit_MLP_mix as MX  # noqa: E402
import model_figures as MF  # noqa: E402
import report_models as RM  # noqa: E402

RULES = ("none", "sum", "nearest", "average")
LN2 = np.log(2.0)


def base_angle(model, rec):
    """Teacher-forced turn-head parameters of the network (+ its profile term), WITHOUT the repulsion term."""
    cf = model._cfg_feat
    X = MB.build_feature_matrix(rec, cf)
    az, lc, lt = JT._cond_cols(rec.dtheta, rec.distance_px, cf["dth_scale"], cf["kin_scales"], y_dt=rec.dt_s)
    with torch.no_grad():
        ang, _, _ = model._model.joint_forward(torch.tensor(X), torch.tensor(az), torch.tensor(lc), torch.tensor(lt))
    return ang


def shifts(rec, rp):
    """Per-bout turn shift s under each rule, the number of dots in reach (sig > 0.5) and the nearest dot (mm)."""
    dx = np.asarray(rec.dot_xs, float); dy = np.asarray(rec.dot_ys, float)
    x = np.asarray(rec.x_px, float)[:, None]; y = np.asarray(rec.y_px, float)[:, None]
    h = np.asarray(MB._travel_ref(rec), float)[:, None]
    on = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(x) & np.isfinite(h)
    d = np.where(on, np.hypot(dx - x, dy - y) / JT.REP_PX_PER_MM, np.inf)
    phi = np.angle(np.exp(1j * (np.arctan2(dy - y, dx - x) - h)))
    sig = np.where(on, 1.0 / (1.0 + np.exp(-np.clip((rp["r_c"] - d) / rp.get("w", 1.0), -50, 50))), 0.0)
    c = np.where(on, np.sin(phi) * sig, 0.0)
    a = float(rp["a"])
    j = np.argmin(d, axis=1)
    s = {"none": np.zeros(len(d)), "sum": -a * c.sum(1),
         "nearest": -a * c[np.arange(len(d)), j],
         "average": -a * c.sum(1) / np.maximum(1.0, sig.sum(1))}
    return s, (sig > 0.5).sum(1), d.min(1)


def score(key):
    u = RM._unit(key)
    with contextlib.redirect_stdout(io.StringIO()):
        models, held = MF.fold_models(u["src"], u["arm"])
    recs = MF.records()
    rows = []
    for f, ck in sorted(held.items()):
        m, rec = models[ck], recs[f]
        if getattr(rec, "dot_xs", None) is None:
            raise RuntimeError(f"fish {f}: record carries no per-dot positions")
        rp = m._cfg_feat["repulsion"]
        ang = base_angle(m, rec)
        S, n_in, dmin = shifts(rec, rp)
        logw = ang[0]
        dth = torch.tensor(np.asarray(rec.dtheta, np.float64), dtype=logw.dtype)
        ll = {}
        for r in RULES:
            s = torch.tensor(S[r], dtype=logw.dtype)
            z = torch.zeros_like(s)
            lw = torch.log_softmax(logw + torch.stack([z, -s, -s, s, s], dim=1), dim=1)
            ll[r] = MX.mix_logdens(dth, (lw,) + tuple(ang[1:])).numpy()
        nd = np.asarray(rec.n_dots, float); mv = np.asarray(rec.moving, float)
        rows.append(pd.DataFrame(dict(fish=f, n_dots=nd, moving=mv, n_in=n_in, dmin=dmin,
                                      **{f"ll_{r}": ll[r] for r in RULES})))
    return pd.concat(rows, ignore_index=True)


def table(D, sel, n_boot=2000, seed=0):
    g = D[sel]
    per = g.groupby("fish")[[f"ll_{r}" for r in RULES]].sum()
    n = g.groupby("fish").size()
    out = {"bouts": int(len(g)), "fish": int(g.fish.nunique())}
    rng = np.random.default_rng(seed)
    fish = per.index.to_numpy()
    for r in RULES:
        diff = (per[f"ll_{r}"] - per["ll_sum"]) / LN2                       # bits, fish totals
        est = diff.sum() / n.sum()
        bs = []
        for _ in range(n_boot):
            k = rng.choice(fish, len(fish))
            bs.append(diff.loc[k].sum() / n.loc[k].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[r] = f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    return out


def main():
    res = []
    for key, lab in (("43a", "seed 0"), ("43b", "seed 101")):
        D = score(key)
        mv = (D.moving > 0) & (D.n_dots >= 2)
        for name, sel in (("multi-dot, >= 2 dots in reach", mv & (D.n_in >= 2)),
                          ("multi-dot, >= 3 dots in reach", mv & (D.n_in >= 3)),
                          ("multi-dot, exactly 1 dot in reach", mv & (D.n_in == 1)),
                          ("single dot in reach (1-dot groups)", (D.moving > 0) & (D.n_dots == 1) & (D.n_in == 1))):
            res.append(dict(seed=lab, bouts_set=name, **table(D, sel)))
    T = pd.DataFrame(res)
    out = HERE.parent / "Model" / "repulsion_scenes" / "combination_rule.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print("held-out log density of the real turn, bits per bout RELATIVE TO 'sum' (> 0 = better than sum), "
              "95 % bootstrap over fish")
        print(T.to_string(index=False))
    return T


if __name__ == "__main__":
    main()
