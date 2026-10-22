"""Likelihood fit of short-range terms on top of F + turn (report Model 43), on the Larsch screen data.

UNPUBLISHED DATA (Larsch lab): code only; caches in Model/twin_test/ (git-ignored).

1. RECORDS. The screen's sighted siblings (fit half of the clutches, twin_test/clutch_split.json), segmented with the
   training data's own `segment_bouts.segment_fish`, become model records: positions in model px (mm x 3.60) around the
   training dish centre, the 35 mm dish, the single dot as centroid AND as the only individual dot, N = 1, moving;
   active-only like training (bouts followed by > 10 s of rest dropped, a new run after each); block clock held late.
2. BASE. The model's teacher-forced per-bout predictions at the population fish (z = 0, no drift): the turn mixture, the
   step and interval log-normal mixtures (the interval truncated at 10 s), conditioned on the real turn / step. F + turn's
   repulsion term is included (side input), so a new term is fitted ON TOP of it.
3. TERMS (each a few numbers, gated by g(d) = sigma((r - d) / w) on the dot's distance d, mm):
     size   the away-side LARGE-angle component's logit + v * g(d)   (turning away harder; the mixture's larger-mean
            Gamma on the side away from the dot)
     step   the step's log-means + u * g(d)                           (u < 0: shorter steps near the dot)
     pause  the interval's log-means + p * g(d)                       (p > 0: longer pauses near the dot)
   fitted by maximising the exact log-likelihood of the near-dot bouts (d < 20 mm) of the fit half, by gradient.

    python repulsion_lik.py records KEY          # KEY = 43a (seed 0) / 43b (seed 101): build + cache the base
    python repulsion_lik.py fit KEY TERM ...     # TERM in size / step / pause
"""
from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from twin import twin_test as TT  # noqa: E402

NEAR_MM = 20.0
CX, CY = 180.0, 174.0               # the training dish centre (model px); the dish radius is 35 mm here
MAX_FISH = 400                      # fit-half fish used (a random subset, fixed seed): plenty for a few numbers


def _segment_plate(args):
    """Real side as twin_test._real_plate, but returning per-fish bout tables with the dot (model px) instead of sums."""
    plate, rows = args
    import pyarrow.parquet as pq
    import loader
    from twin import segment_bouts as SB
    z = zipfile.ZipFile(TT.ZIP)
    cols = ["frame"] + [f"{k}_{int(r['well']):02d}" for r in rows for k in ("x", "y", "heading", "flag")]
    tab = pq.read_table(io.BytesIO(z.read(f"schizscreen_twin_v1/trajectories/plate_{plate:03d}.parquet")), columns=cols,
                        filters=[("frame", ">=", TT.WIN[0]), ("frame", "<", TT.WIN[1])]).to_pandas()
    st = TT.stimulus()
    frames = tab.frame.to_numpy()
    epi = st.episode.to_numpy().astype(str)
    kind_all = st.stim_kind.to_numpy().astype(str)
    out = {}
    for r in rows:
        w = int(r["well"])
        x, y, o = loader.to_mm(tab[f"x_{w:02d}"].to_numpy(float), tab[f"y_{w:02d}"].to_numpy(float),
                               tab[f"heading_{w:02d}"].to_numpy(float) / loader.HEADING_SCALE,
                               r["centre_x_px"], r["centre_y_px"], r["px_per_mm"])
        fl = tab[f"flag_{w:02d}"].fillna(1).to_numpy().astype(np.uint8)
        bad = ((fl & loader.DEFAULT_MASK) > 0) | ~np.isfinite(x)
        df = pd.DataFrame({"frame": frames, "stimulus": epi[frames],
                           "f_x": np.where(bad, np.nan, CX + x * TT.PX), "f_y": np.where(bad, np.nan, CY + y * TT.PX),
                           "f_ori": np.where(bad, np.nan, o)})
        with contextlib.redirect_stdout(io.StringIO()):
            b = pd.DataFrame(SB.segment_fish(df, "f", int(r["fish_id"]), 1.0))
        if b.empty:
            continue
        dxa, dya = TT.dot_mm(st, r["centre_x_px"], r["centre_y_px"], r["px_per_mm"])
        f = b.frame_onset.to_numpy(int)
        b["dot_cx"] = CX + dxa[f] * TT.PX
        b["dot_cy"] = CY + dya[f] * TT.PX
        b["kind"] = kind_all[f]
        out[int(r["fish_id"])] = b
    return out




# --------------------------------------------------------------------------- #
# Likelihood of the near-dot bouts with a candidate term
# --------------------------------------------------------------------------- #
def _stack(rows):
    cat = lambda k: np.concatenate([r[k] for r in rows])
    cat_l = lambda k, j: np.concatenate([r[k][j] for r in rows])
    return dict(d=cat("d"), phi=cat("phi"), dth=cat("dth"), dist=cat("dist"), dt=cat("dt"),
                ang=[cat_l("ang", j) for j in range(len(rows[0]["ang"]))],
                dpar=[cat_l("dpar", j) for j in range(3)], tpar=[cat_l("tpar", j) for j in range(3)])


def loglik(D, term, theta, dt_hi):
    """Per-bout log-likelihood (turn, step, interval) with ``term`` at parameters ``theta`` (torch, differentiable)."""
    import torch
    from twin import fit_MLP_mix as MX
    T = lambda a: torch.as_tensor(a, dtype=torch.float64)
    d, phi = T(D["d"]), T(D["phi"])
    logw, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = [T(a) for a in D["ang"]]
    dlw, dmu, dsg = [T(a) for a in D["dpar"]]
    tlw, tmu, tsg = [T(a) for a in D["tpar"]]
    g = None
    off = 0.0
    if term is not None:
        # a LOCAL gate (reach 3-14 mm, width 0.3-2 mm) + a global offset over every cached bout (0-20 mm): the offset
        # absorbs fish-level differences (the screen's older fish bout ~18 % faster everywhere), so ``amp`` is the
        # near-dot correction relative to the 15-20 mm bouts
        amp, off = theta[0], theta[3]
        r = 3.0 + 11.0 * torch.sigmoid(theta[1]); w = 0.3 + 1.7 * torch.sigmoid(theta[2])
        g = torch.sigmoid((r - d) / w)
    if term == "size":
        # the larger-mean Gamma on the side AWAY from the dot (dot at phi > 0 -> away = dtheta < 0 = components 1, 2)
        away_left = phi > 0
        mL1, mL2, mR1, mR2 = aL1 * tL1, aL2 * tL2, aR1 * tR1, aR2 * tR2
        big_L = torch.where(mL2 > mL1, 2, 1); big_R = torch.where(mR2 > mR1, 4, 3)
        comp = torch.where(away_left, big_L, big_R)
        shift = torch.zeros_like(logw).scatter(1, comp.reshape(-1, 1), (off + amp * g).reshape(-1, 1))
        logw = torch.log_softmax(logw + shift, dim=1)
    if term == "step":
        dmu = dmu + (off + amp * g).reshape(-1, 1)
    if term == "pause":
        tmu = tmu + (off + amp * g).reshape(-1, 1)
    la = MX.mix_logdens(T(D["dth"]), (logw, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2))
    ld = MX._lnk_logpdf(torch.clamp(T(D["dist"]), min=1e-6), dlw, dmu, dsg)
    lt = MX._dt_logpdf(torch.clamp(T(D["dt"]), min=1e-6), tlw, tmu, tsg, hi=dt_hi)
    return la, ld, lt


def fit(key, term, steps=400, lr=0.05):
    import torch
    blob = pickle.load(open(TT.OUT / f"replik_{key}.pkl", "rb"))
    D = _stack(blob["rows"])
    hi = blob["meta"]["dt_trunc"]
    ok = np.isfinite(D["dt"]) & np.isfinite(D["dist"]) & (D["dist"] > 0)
    D = {k: (v[ok] if not isinstance(v, list) else [a[ok] for a in v]) for k, v in D.items()}
    with torch.no_grad():
        base = [x.sum().item() for x in loglik(D, None, None, hi)]
    theta = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        la, ld, lt = loglik(D, term, theta, hi)
        loss = -(la.sum() + ld.sum() + lt.sum()) / len(D["d"])
        loss.backward()
        opt.step()
    with torch.no_grad():
        la, ld, lt = loglik(D, term, theta, hi)
    new = [la.sum().item(), ld.sum().item(), lt.sum().item()]
    n = len(D["d"])
    w = float(0.3 + 1.7 * torch.sigmoid(theta[2]).item()); rr = float(3.0 + 11.0 * torch.sigmoid(theta[1]).item())
    with torch.no_grad():                                   # the offset alone: what the fish-level difference is worth
        th0 = theta.clone(); th0[0] = 0.0
        la0, ld0, lt0 = loglik(D, term, th0, hi)
    off_only = [la0.sum().item(), ld0.sum().item(), lt0.sum().item()]
    res = dict(key=key, term=term, n_bouts=n, amp=float(theta[0]), r=rr, w=w, offset=float(theta[3]),
               gain_bits_near_term=(sum(new) - sum(off_only)) / n / np.log(2),
               gain_bits_per_bout={c: (new[i] - base[i]) / n / np.log(2) for i, c in enumerate(("turn", "step", "interval"))},
               gain_bits_total_per_bout=(sum(new) - sum(base)) / n / np.log(2))
    print(json.dumps(res, indent=1), flush=True)
    return res




# --------------------------------------------------------------------------- #
# Diagnosis: which component of the backing off do the models miss?
# --------------------------------------------------------------------------- #
COMPONENTS = (("mean |turn| (deg)", "sabs", "n"), ("share of turns > 90 deg", "n90", "n"),
              ("mean |turn| of turns away (deg)", "sabs_away", "naway"), ("mean log step", "slst", "ng"),
              ("mean log interval (<= 10 s)", "sldt", "ndt"), ("share of pauses > 10 s", "nlong", "nall"),
              ("P(turn toward)", "ntow", "nturn"), ("gain (mm/bout)", "gain", "ng"))




