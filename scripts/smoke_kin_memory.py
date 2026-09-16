"""Does a longer kinematic memory help, and does the free-running loop survive it?

THE QUESTION. The kinematics trunk sees ``K_kin=3`` lags of z-scored log-distance and log-dt
and nothing else. Generated log-dt autocorrelation matches real at lags 1-3 (.228/.204/.180 vs
.234/.191/.167) and dies from lag 5 (.086/.038/.022 vs .140/.111/.084). That is the AR(3)
signature: with three lags you cannot produce a slow tail without inflating the head. It is
also why ``kin_feedback`` exists -- it shrinks the history to kill a lag-1 OVERSHOOT that is
itself an artefact of having three lags -- and it paid for that with the distance dispersion
(dist_std 5.84 vs real 8.39), which is the model's dominant tell.

WHY A STANDALONE PROBE. The full ladder rung costs ~1h50m of Colab. This trains the kin HALF
of ``MDNMixKin`` on its own -- the same ``kin_body``/``dist_head``/``dt_head`` modules, the
same mixture log-normal likelihood, the same numpy rollout cell -- in ~25 s per arm on CPU,
and answers the only two questions that gate the rung.

THE GATE IS STABILITY, NOT SCORE. Long histories free-run unstable, and validation loss does
NOT see it -- K=32 has the best held-out NLL of any arm here and two of its five seeds wander
with acf 0.63 at lag 10 against a real 0.111. Nothing in the teacher-forced objective penalises
it and ``fit()`` has no detector, so a bad fold would train cleanly, poison the containment and
ARTR calibrations that roll out after it, and surface only as an unexplained AUC. Measured over
5 seeds (heavy-tailed heads, n_dist=2/n_dt=3, realistic q99.9 clip):

    K=3   stable 3/3   dt acf@10 +0.035   sd(log dist) 0.454     (real +0.111 / 0.450)
    K=8   stable 3/3   dt acf@10 +0.127   sd(log dist) 0.451     <- best match, no pathology
    K=16  stable 4/5   dt acf@10 +0.168   one hard COLLAPSE
    K=32  stable 3/5   dt acf@10 +0.320   two near-unit-root seeds; 3x acf overshoot

So K=8 is the ceiling. Note a 3-seed run passed K=16 cleanly -- use >=5 seeds before trusting a
new K, because the failure is intermittent rather than systematic.

    python scripts/smoke_kin_memory.py                 # default arms
    python scripts/smoke_kin_memory.py --K 3 8 16 --seeds 4

Reads DIRECTION only. Local in-sample numbers in this project have overstated held-out ones by
close to an order of magnitude (dist_w1 0.32 local vs 2.79 on Colab); the AUC column here is
for RANKING arms, not for predicting the ladder result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP as M                     # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402
import generative_metrics as GM         # noqa: E402

ACF_LAGS = (1, 3, 5, 10, 20, 30, 50)
# real targets, all 15 fish, data/bouts_GsizeMotionSingle.csv
REAL_ACF_DT = (0.234, 0.167, 0.140, 0.111, 0.084, 0.074, 0.042)
REAL_ACF_DIST = (0.218, 0.112, 0.074, 0.051, 0.049, 0.043, 0.025)


def _acf(x, lags=ACF_LAGS):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x - x.mean()
    v = x.var()
    return np.array([float((x[:-L] * x[L:]).mean() / v) if v > 0 and L < len(x) else np.nan
                     for L in lags])


def _scales(recs):
    ad = np.concatenate([r.distance_px for r in recs]); ad = ad[np.isfinite(ad) & (ad > 0)]
    at = np.concatenate([r.dt_s for r in recs]); at = at[np.isfinite(at) & (at > 0)]
    return ((float(np.log(ad).mean()), float(max(np.log(ad).std(), 1e-3)),
             float(np.log(at).mean()), float(max(np.log(at).std(), 1e-3))),
            (float(np.quantile(ad, 0.999)), float(np.quantile(at, 0.999))))


def train_kin(recs, K_kin, scales, clip, n_dist=2, n_dt=3, epochs=40, seed=0,
              kin_hidden=32, kin_wd=1e-4, lr=1e-3, batch=256):
    """Train ONLY the kin trunk + heads of a real ``MDNMixKin`` (angle side left untouched)."""
    M._seed_everything(seed)
    cf = dict(K=K_kin, kin_scales=scales, K_kin=K_kin, kin_clip=clip)   # _kin_history_block falls back to 'K'
    Xk = np.concatenate([MB._kin_history_block(r, cf) for r in recs], 0)
    yd = np.concatenate([np.asarray(r.distance_px, float) for r in recs])
    yt = np.concatenate([np.asarray(r.dt_s, float) for r in recs])
    net = MX.MDNMixKin(1, 4, 2 * K_kin, kin_hidden, n_dist=n_dist, n_dt=n_dt)
    kin_p = (list(net.kin_body.parameters()) + list(net.dist_head.parameters())
             + list(net.dt_head.parameters()))
    opt = torch.optim.Adam(kin_p, lr=lr, weight_decay=kin_wd)
    Xk_t = torch.tensor(Xk, dtype=torch.float32)
    yd_t = torch.tensor(yd, dtype=torch.float32); yt_t = torch.tensor(yt, dtype=torch.float32)
    n = len(Xk_t); nval = max(int(0.15 * n), 1)
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g)
    tr, va = perm[nval:], perm[:nval]

    def nll(ix):
        bk = net.kin_body(Xk_t[ix])
        do = net.dist_head(bk)
        k = net.n_dist
        dpar = (torch.log_softmax(do[:, :k], 1), do[:, k:2 * k],
                torch.nn.functional.softplus(do[:, 2 * k:3 * k]) + M.EPS)
        yd_b = yd_t[ix]; yt_b = yt_t[ix]
        lin = torch.nan_to_num(torch.log(torch.clamp(yd_b, min=1e-6)))
        to = net.dt_head(torch.cat([bk, lin.reshape(-1, 1)], 1))
        kt = net.n_dt
        tpar = (torch.log_softmax(to[:, :kt], 1), to[:, kt:2 * kt],
                torch.nn.functional.softplus(to[:, 2 * kt:3 * kt]) + M.EPS)
        md = torch.isfinite(yd_b) & (yd_b > 0); mt = torch.isfinite(yt_b) & (yt_b > 0)
        ld = MX._lnk_logpdf(torch.clamp(yd_b, min=1e-6), *dpar)
        lt = MX._lnk_logpdf(torch.clamp(yt_b, min=1e-6), *tpar)
        z = torch.zeros((), device=Xk_t.device)
        return -((ld[md].mean() if md.any() else z) + (lt[mt].mean() if mt.any() else z))

    best, best_state, bad = np.inf, None, 0
    for _ep in range(epochs):
        net.train()
        for i in range(0, len(tr), batch):
            opt.zero_grad(); loss = nll(tr[i:i + batch]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nll(va))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, float(best)


def free_run(net, K_kin, scales, clip, n=40000, seed=0, kin_feedback=None):
    """Free-run the kin loop in numpy exactly as ``generate_mix_kin_loop`` does."""
    rng = np.random.default_rng(seed)
    _, kin_fn, dt_params = MX._numpy_predictor_kin(net)
    dm, ds, tm, ts = scales
    dist_hi, dt_hi = clip
    kfb = kin_feedback or {}
    fd, ft = float(kfb.get("dist", 1.0)), float(kfb.get("dt", 1.0))
    ld_h = np.zeros(n); lt_h = np.zeros(n); di = np.zeros(n); dtv = np.zeros(n)
    row = np.zeros(2 * K_kin, dtype=np.float32)
    for t in range(n):
        row[:] = 0.0
        for k in range(1, K_kin + 1):
            if t - k >= 0:
                row[k - 1] = fd * (ld_h[t - k] - dm) / ds
                row[K_kin + k - 1] = ft * (lt_h[t - k] - tm) / ts
        bkin, dpar = kin_fn(row)
        w, mu, sig = dpar
        j = rng.choice(len(w), p=w)
        d = float(np.clip(np.exp(rng.normal(mu[j], sig[j])), 1e-6, dist_hi))
        tw, tmu, tsig = dt_params(bkin, np.log(d))
        jt = rng.choice(len(tw), p=tw)
        tv = float(np.clip(np.exp(rng.normal(tmu[jt], tsig[jt])), 1e-6, dt_hi))
        di[t] = d; dtv[t] = tv
        ld_h[t] = np.log(d); lt_h[t] = np.log(tv)
    return di, dtv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=str(Path(__file__).parent.parent / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--K", type=int, nargs="+", default=[3, 8])
    ap.add_argument("--kin-hidden", type=int, nargs="+", default=[32],
                    help="sweep kin-trunk WIDTH. Capacity is the one thing bits_per_bout cannot "
                         "adjudicate for these heads, because Metric A is turn-only.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-free", type=int, default=40000)
    ap.add_argument("--hold", type=int, default=3, help="held-out fish")
    args = ap.parse_args()

    recs = MB.build_border_records(pd.read_csv(args.bouts))
    fids = sorted(recs)
    test_ids, train_ids = fids[::5][:args.hold], [f for f in fids if f not in fids[::5][:args.hold]]
    train = [recs[f] for f in train_ids]
    scales, clip = _scales(train)
    dm, ds, tm, ts = scales
    real_d = np.concatenate([np.abs(recs[f].distance_px) for f in fids])
    real_d = real_d[np.isfinite(real_d) & (real_d > 0)]
    real_sd_logd = float(np.std(np.log(real_d)))
    print(f"{len(train_ids)} train fish, {len(test_ids)} held out; real sd(log distance) = {real_sd_logd:.3f}")
    print(f"train log-scales: dist mu {dm:.3f} sd {ds:.3f} | dt mu {tm:.3f} sd {ts:.3f}\n")

    hdr = (f"{'arm':16s}{'seed':>5s}{'val NLL':>10s}{'stable':>8s}{'m(log d)':>10s}"
           f"{'sd(log d)':>11s}{'m(log dt)':>11s}" + "".join(f"{('dt@'+str(L)):>8s}" for L in ACF_LAGS))
    print(hdr); print("-" * len(hdr))
    rows = []
    for K in args.K:
      for kh in args.kin_hidden:
        for kfb_name, kfb in (("kfb=1", None), ("kfb=.52/.82", {"dist": 0.52, "dt": 0.82})):
            for s in range(args.seeds):
                net, v = train_kin(train, K, scales, clip, epochs=args.epochs, seed=s, kin_hidden=kh)
                d, tv = free_run(net, K, scales, clip, n=args.n_free, seed=100 + s, kin_feedback=kfb)
                ml, sl, mt = float(np.mean(np.log(d))), float(np.std(np.log(d))), float(np.mean(np.log(tv)))
                # DIVERGENCE TEST -- TWO distinct failure modes, both measured, and the second
                # is the one that will bite silently:
                #   (1) COLLAPSE: every bout pinned to the 1e-6 floor, mean log dt ~ -12.6 and
                #       acf ~0.998 at every lag. Unmistakable. (K=16 seed 4.)
                #   (2) NEAR-UNIT-ROOT DRIFT: levels stay plausible but the loop wanders, giving
                #       acf 0.67/0.63/0.57 at lags 1/10/50 against a real 0.234/0.111/0.042.
                #       (K=32 seeds 3-4.) A mean/level check does NOT catch this -- an earlier
                #       version of this test passed all of them.
                a = _acf(np.log(tv))
                stable = bool(abs(ml - dm) < 0.5 and abs(mt - tm) < 0.5
                              and np.nanmax(a[3:]) < 0.9            # mode 1
                              and a[-1] < 0.25)                     # mode 2 (real acf@50 = 0.042)
                arm = f"K={K} h{kh} {kfb_name}"
                print(f"{arm:16s}{s:5d}{v:10.4f}{('OK' if stable else 'DIVERGED'):>8s}"
                      f"{ml:10.3f}{sl:11.3f}{mt:11.3f}" + "".join(f"{x:8.3f}" for x in a))
                rows.append(dict(K=K, kh=kh, kfb=kfb_name, seed=s, stable=stable, sd_logd=sl, nll=v,
                                 **{f"dt_acf{L}": x for L, x in zip(ACF_LAGS, a)}))
    print(f"{'REAL':16s}{'':5s}{'':10s}{'':8s}{dm:10.3f}{real_sd_logd:11.3f}{tm:11.3f}"
          + "".join(f"{x:8.3f}" for x in REAL_ACF_DT))

    df = pd.DataFrame(rows)
    print("\nsummary (mean over stable seeds):")
    for (K, kh, kfb), g in df.groupby(["K", "kh", "kfb"]):
        ok = g[g.stable]
        n_ok = len(ok)
        if not n_ok:
            print(f"  K={K} h{kh} {kfb:12s} 0/{len(g)} stable  <- REJECT")
            continue
        print(f"  K={K} h{kh} {kfb:12s} {n_ok}/{len(g)} stable  NLL {ok.nll.mean():.4f}  sd(log d) {ok.sd_logd.mean():.3f} "
              f"(real {real_sd_logd:.3f})  dt acf@10 {ok.dt_acf10.mean():+.3f} (real {REAL_ACF_DT[3]:+.3f}) "
              f" acf@20 {ok.dt_acf20.mean():+.3f} (real {REAL_ACF_DT[4]:+.3f})")
    print("\nACCEPTANCE for adding a K_kin rung to the ladder:")
    print("  * stable in EVERY seed (any divergence => do not ship that K)")
    print(f"  * sd(log distance) within 5% of real ({real_sd_logd:.3f})")
    print("  * dt acf at lags 10-20 materially above the K=3 arm and no overshoot at lag 1")


if __name__ == "__main__":
    main()
