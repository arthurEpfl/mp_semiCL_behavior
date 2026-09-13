"""Gate A/B: is a slow latent STATE carrying anything the kinematic history does not?

THE DECISION THIS INFORMS. The model has no learned state. Everything slower than its lag
window is supplied by generation-time latents (`vigor`, `artr`) that are moment-matched after
training and invisible to any likelihood. Replacing them with a sticky discrete state trained
through the exact forward algorithm is the expensive option; adding lags is the cheap one. The
two are observationally equivalent on the autocorrelation alone (a plain AR(32) reproduces the
real acf through lag 30), so acf cannot decide it. This can.

GATE A -- does a level-only state layer find slow structure at all, on the FROZEN emission?
  Fit a K-state chain over the RESIDUALS of the trained kin trunk, where a state may shift each
  channel's LEVEL but may not change its shape (variance shared across states). That constraint
  is the whole design: with a free per-state emission the chain spends its states on the
  MARGINAL instead -- it learns a "long pause" mode and a "short bout" mode and dwell collapses
  to a couple of bouts, which looks like a latent and is really a mixture model.
  A per-fish nuisance intercept is subtracted first, because between-fish level spread is larger
  than within-fish drift and without it the chain collapses the other way, onto individuality
  (one state per fish, dwell in the thousands).

GATE B -- and this is the one that decides the model class. Refit the SAME state layer on a
  K_kin=8 trunk. If eight lags already absorb the slow structure, the fitted loadings shrink and
  the latent is redundant; if the loadings survive, the state is carrying something no practical
  history length provides. Neither the cheap nor the expensive proposal specified this test.

    python scripts/smoke_state_layer.py                 # K_kin 3 vs 8, K=4 states
    python scripts/smoke_state_layer.py --states 2 3 4 6

Dwell is reported BOTH free and clamped. Read the free one: `--self-bounds` forces dwell into
[20, 200] by construction, so a "dwell in range" acceptance under the clamp is circular.
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
from smoke_kin_memory import _scales, train_kin        # noqa: E402


def trunk_residuals(net, recs, K_kin, scales, chan=("dist", "dt")):
    """Per-bout residual of log-distance / log-dt about the frozen trunk's PREDICTED MEAN.

    Teacher-forced on the real history, so this is the part of each channel the trunk cannot
    explain from ``K_kin`` lags -- exactly what a state layer would have to carry.
    """
    import torch
    cf = dict(K=K_kin, kin_scales=scales, K_kin=K_kin)
    out = {}
    for f, r in recs.items():
        Xk = torch.tensor(MB._kin_history_block(r, cf), dtype=torch.float32)
        with torch.no_grad():
            bk = net.kin_body(Xk)
            do = net.dist_head(bk)
            k = net.n_dist
            w = torch.softmax(do[:, :k], 1); mu = do[:, k:2 * k]
            pred_d = (w * mu).sum(1).numpy()                       # E[log distance]
            lin = torch.nan_to_num(torch.log(torch.clamp(
                torch.tensor(np.asarray(r.distance_px, float), dtype=torch.float32), min=1e-6)))
            to = net.dt_head(torch.cat([bk, lin.reshape(-1, 1)], 1))
            kt = net.n_dt
            wt = torch.softmax(to[:, :kt], 1); mut = to[:, kt:2 * kt]
            pred_t = (wt * mut).sum(1).numpy()                     # E[log dt | log distance]
        ld = np.log(np.clip(np.asarray(r.distance_px, float), 1e-6, None))
        lt = np.log(np.clip(np.asarray(r.dt_s, float), 1e-6, None))
        res = np.c_[ld - pred_d, lt - pred_t]
        ok = np.isfinite(res).all(1)
        res = res[ok]
        res -= res.mean(0)                                          # per-fish nuisance intercept
        out[f] = res
    return out


def fit_state_layer(seqs, K=4, iters=40, self_bounds=None, seed=0):
    """Baum-Welch for a K-state chain with per-state channel LEVELS and a SHARED variance.

    Shared variance is what makes it a level-only latent rather than a mixture over shapes.
    ``self_bounds`` optionally clamps the self-transition each M-step.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack(seqs); D = X.shape[1]
    sd = X.std(0) + 1e-9
    Z = [(s / sd) for s in seqs]
    mu = rng.normal(0, 0.5, (K, D))
    var = np.ones(D)
    A = np.full((K, K), 1.0 / K); pi = np.full(K, 1.0 / K)
    ll_hist = []
    for _ in range(iters):
        Nk = np.zeros(K); Sx = np.zeros((K, D)); Sxx = 0.0; Axi = np.zeros((K, K))
        Pi = np.zeros(K); LL = 0.0; Ntot = 0
        for s in Z:
            lb = -0.5 * (((s[:, None, :] - mu[None]) ** 2 / var[None]).sum(-1)
                         + np.log(2 * np.pi * var).sum())
            B = np.exp(lb - lb.max(1, keepdims=True)); B = np.maximum(B, 1e-300)
            T = len(s); al = np.zeros((T, K)); c = np.zeros(T)
            al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
            for t in range(1, T):
                al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
            be = np.zeros((T, K)); be[-1] = 1.0
            for t in range(T - 2, -1, -1):
                be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
            g = al * be; g /= g.sum(1, keepdims=True)
            LL += float(np.log(c).sum() + (lb.max(1)).sum()); Ntot += T
            Pi += g[0]; Nk += g.sum(0); Sx += g.T @ s
            Sxx += float(np.einsum("tk,tkd->", g, (s[:, None, :] - mu[None]) ** 2))
            for t in range(T - 1):
                Axi += np.outer(al[t], B[t + 1] * be[t + 1]) * A / c[t + 1]
        mu = Sx / np.maximum(Nk[:, None], 1e-9)
        var = np.full(D, max(Sxx / max(Nk.sum() * D, 1e-9), 1e-4))
        A = Axi / np.maximum(Axi.sum(1, keepdims=True), 1e-12)
        if self_bounds is not None:
            lo, hi = self_bounds
            for k in range(K):
                p = np.clip(A[k, k], lo, hi)
                off = A[k].sum() - A[k, k]
                A[k] = A[k] * ((1 - p) / off if off > 1e-12 else 0.0); A[k, k] = p
        pi = Pi / max(Pi.sum(), 1e-12)
        ll_hist.append(LL / max(Ntot, 1))
    occ = Nk / max(Nk.sum(), 1e-9)
    dwell = 1.0 / np.maximum(1.0 - np.diag(A), 1e-6)
    # tau2 = correlation time of the chain from its 2nd eigenvalue: the timescale the latent
    # actually carries, independent of how the states are labelled.
    ev = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    lam2 = float(ev[1]) if len(ev) > 1 else 0.0
    tau2 = float(-1.0 / np.log(max(lam2, 1e-9))) if lam2 < 1 else np.inf
    load_sd = np.sqrt(np.maximum((occ[:, None] * (mu - (occ[:, None] * mu).sum(0)) ** 2).sum(0), 0))
    return dict(occ=occ, dwell=dwell, tau2=tau2, mu=mu, load_sd=load_sd,
                ll=ll_hist[-1], A=A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=str(Path(__file__).parent.parent / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--kkin", type=int, nargs="+", default=[3, 8])
    ap.add_argument("--states", type=int, nargs="+", default=[4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--on", choices=("residual", "observed", "both"), default="both",
                    help="fit the state layer on trunk residuals or on the observed series")
    args = ap.parse_args()

    recs = MB.build_border_records(pd.read_csv(args.bouts))
    scales, clip = _scales(list(recs.values()))
    print("real slow AR(1) tau: 55 bouts (log dt), 83 (log distance); real across-window "
          "corr(log dt, log dist) = -0.386\n")
    hdr = (f"{'fit on':10s}{'trunk':8s}{'K':>3s}{'clamp':>7s}{'tau2':>8s}"
           f"{'dwell (bouts)':>26s}{'load sd dist':>14s}{'load sd dt':>12s}{'corr':>8s}")
    print(hdr); print("-" * len(hdr))
    # OBSERVED = the raw per-fish-demeaned log series. This is the reference: the slow band is
    # measurably THERE (window-median log-dt acf +0.34, bout-rate acf +0.51 at 1 min). If a
    # level-only chain cannot find it here, the fitting procedure is at fault, not the model.
    if args.on in ("observed", "both"):
        obs = []
        for f in sorted(recs):
            r = recs[f]
            z = np.c_[np.log(np.clip(np.asarray(r.distance_px, float), 1e-6, None)),
                      np.log(np.clip(np.asarray(r.dt_s, float), 1e-6, None))]
            z = z[np.isfinite(z).all(1)]
            obs.append(z - z.mean(0))
        for K in args.states:
            for clamp in (None, (0.95, 0.995)):
                r = fit_state_layer(obs, K=K, iters=args.iters, self_bounds=clamp, seed=args.seed)
                o = np.argsort(r["mu"][:, 1])[::-1]
                dw = " ".join(f"{r['dwell'][k]:.0f}" for k in o)
                w = r["occ"]
                c = float(np.cov(r["mu"][:, 0], r["mu"][:, 1], aweights=w)[0, 1]
                          / max(np.sqrt(np.cov(r["mu"][:, 0], aweights=w)
                                        * np.cov(r["mu"][:, 1], aweights=w)), 1e-9))
                print(f"{'observed':10s}{'-':8s}{K:3d}{('yes' if clamp else 'free'):>7s}"
                      f"{r['tau2']:8.1f}{dw:>26s}{r['load_sd'][0]:14.3f}{r['load_sd'][1]:12.3f}{c:8.2f}")
    if args.on == "observed":
        return
    for K_kin in args.kkin:
        net, _ = train_kin(list(recs.values()), K_kin, scales, clip,
                           epochs=args.epochs, seed=args.seed)
        res = trunk_residuals(net, recs, K_kin, scales)
        seqs = [res[f] for f in sorted(res)]
        for K in args.states:
            for clamp in (None, (0.95, 0.995)):
                r = fit_state_layer(seqs, K=K, iters=args.iters,
                                    self_bounds=clamp, seed=args.seed)
                o = np.argsort(r["mu"][:, 1])[::-1]
                dw = " ".join(f"{r['dwell'][k]:.0f}" for k in o)
                # sign structure: real dt and distance move OPPOSITE ways across windows
                w = r["occ"]
                c = float(np.cov(r["mu"][:, 0], r["mu"][:, 1], aweights=w)[0, 1]
                          / max(np.sqrt(np.cov(r["mu"][:, 0], aweights=w)
                                        * np.cov(r["mu"][:, 1], aweights=w)), 1e-9))
                print(f"{'residual':10s}{'K_kin='+str(K_kin):8s}{K:3d}{('yes' if clamp else 'free'):>7s}"
                      f"{r['tau2']:8.1f}{dw:>26s}{r['load_sd'][0]:14.3f}{r['load_sd'][1]:12.3f}{c:8.2f}")
    print("""
GATE A (does a level-only state find slow structure on the frozen emission?)
  * tau2 >= 30 bouts on the FREE fit (the clamped one is circular -- self_bounds forces it)
  * occupancy-weighted loading sd > 0.15 z-units on BOTH channels
  * loadings not all one sign; corr(dist, dt) across states should be NEGATIVE, matching the
    real across-window corr(log dt, log dist) = -0.386 that no current mechanism can express
GATE B (is the latent redundant given more lags?)
  * compare K_kin=3 vs K_kin=8 at the same K: if the loadings collapse, lags absorb it and the
    architecture step is not justified; if they survive, the state carries something lags do not""")


if __name__ == "__main__":
    main()
