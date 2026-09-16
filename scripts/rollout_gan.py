"""Rollout-aware training: a discriminator judges whole windows, the gradient flows into the roll-out.

THE METHOD, IN ONE PARAGRAPH
----------------------------
Hold B trajectories running. Every step, advance them W bouts through the DIFFERENTIABLE torch
rollout (``rollout_torch``), hand the generated window to a small 1-D CNN discriminator alongside a
window of real bouts, train the discriminator to tell them apart and the model to fool it, and
anchor the whole thing with the ordinary teacher-forced likelihood so the emission cannot collapse.
Then ``detach()`` the state and continue -- so the trajectories are genuinely long-run while the
gradient stays bounded.

WHY THIS AND NOT THE PREVIOUS ATTEMPT
-------------------------------------
The first rollout-aware build (energy distance + REINFORCE, now deleted) returned a null, and the
post-mortem found three causes. Each is addressed by construction here, and each fix is backed by a
measurement rather than an argument:

  no capacity      3771 MEMORYLESS head params were being tuned against across-window
                   autocorrelation -- real +0.33, model ~0.00, 6 sigma. A quantity outside the
                   model class cannot be optimised into it.
                   -> a GRU state carried through the rollout (fit_MLP_fishre.state_chain)
  variance         one scalar reward for 60 bouts x 3 channels; between-arm differences came out
                   SMALLER than within-arm noise.
                   -> PATHWISE gradients: every emission is reparameterised, so the critic's
                      gradient reaches the emission parameters directly
  dilution         energy distance over 39 standardised features weights all directions equally,
                   but the defect lives in ~4 of them.
                   -> a LEARNED critic, which is what the c2st metric itself is

WHAT THE DISCRIMINATOR SEES
---------------------------
The RAW window, not hand-made features: ``[dtheta/scale, log dist (z), log dt (z), r/R]`` over W
bouts. No feature engineering, so no column-order hazard and no chance of leaving out the direction
that mattered -- it learns which one that is.

WHAT IS DELIBERATELY GONE
-------------------------
The calibrator stack. No rejection sampling, no kin_feedback, no containment search. The critic
sees r/R, so containment is LEARNED. That is the whole point: one objective instead of six that
each fix a moment the others cannot see.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                     # noqa: E402
import fit_MLP_border as MB              # noqa: E402
import fit_MLP_mix as MX                 # noqa: E402
import fit_MLP_joint as JT               # noqa: E402
import fit_MLP_fishre as FR              # noqa: E402
import rollout_torch as RT               # noqa: E402

N_CH = 4                                 # dtheta, log dist, log dt, r/R


# =========================================================================== #
#  Critic
# =========================================================================== #
class WindowCritic(nn.Module):
    """Small 1-D CNN over a (N_CH, W) behaviour window.

    Spectral norm throughout, which is the cheapest thing that keeps a GAN on a low-dimensional
    signal stable -- this is a 4x60 window, not an image, so nothing heavier is warranted. Kernel 5
    with dilation gives it a ~30-bout receptive field, enough to see run-lengths and the slow drift
    the whole exercise is about, without letting it memorise a whole window.
    """

    def __init__(self, ch=N_CH, width=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv1d(ch, width, 5, padding=2)), nn.LeakyReLU(0.2),
            sn(nn.Conv1d(width, width, 5, padding=4, dilation=2)), nn.LeakyReLU(0.2),
            sn(nn.Conv1d(width, width, 5, padding=8, dilation=4)), nn.LeakyReLU(0.2),
        )
        # SN stays on the output too. A previous version removed it on the theory that an all-SN
        # stack is 1-Lipschitz end to end and therefore caps the logit near +-0.1 -- that theory is
        # FALSE and was measured false: trained on two clearly different window populations, the
        # all-SN critic reaches D(real) 0.989 / D(fake) 0.005 with logits up to 7.9. SN bounds
        # sensitivity to input perturbations, not output magnitude, and 60 timesteps plus max
        # pooling leave ample range. So a critic pinned at D ~ 0.5 is NOT architecturally muzzled;
        # it is under-trained or facing a genuinely hard discrimination.
        self.out = sn(nn.Linear(2 * width, 1))

    def forward(self, w):
        z = self.net(w)
        # mean AND max pooling: the mean carries the level/dispersion statistics, the max carries
        # "does this window contain something real fish never do" (a wall pile-up, a runaway step).
        return self.out(torch.cat([z.mean(-1), z.amax(-1)], dim=1)).squeeze(1)


def channels_from(dth, dist, dtv, rad, cfg):
    """(B, N_CH, W) critic input, on the scales the net itself uses."""
    dm, ds, tm, ts = cfg["kin_scales"]
    return torch.stack([dth / float(cfg["dth_scale"]),
                        (torch.log(torch.clamp(dist, min=1e-6)) - dm) / ds,
                        (torch.log(torch.clamp(dtv, min=1e-6)) - tm) / ts,
                        rad], dim=1)


class RealWindows:
    """Random real windows, on exactly the channels the critic reads from the generated side."""

    def __init__(self, records, cfg, W, device):
        self.W, self.dev = int(W), device
        chans = []
        for r in records:
            rad = np.hypot(np.asarray(r.x_px, float) - r.dish_cx,
                           np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R
            # channels_from stacks on dim=1, which is (B, N_CH, W) for the batched generated side
            # but (n, N_CH) for these 1-D real arrays -- transpose to the (N_CH, n) the slicer wants
            c = channels_from(*[torch.tensor(np.asarray(a, np.float32)) for a in
                                (r.dtheta, r.distance_px, r.dt_s, rad)], cfg).transpose(0, 1)
            chans.append(torch.nan_to_num(c).to(device))       # (N_CH, n)
        self.chans = chans

    def sample(self, n, rng=None):
        """``n`` random real windows.

        Indices are drawn with NUMPY, deliberately. Drawing them with a CUDA generator means an
        ``int()`` per index, i.e. a device->host sync per window -- 2n of them every training step,
        which on GPU costs more than the rollout it is being compared against. The windows
        themselves are already resident on the device, so only the slice bounds need to be host-side.
        """
        rng = rng or np.random
        fi = rng.integers(len(self.chans), size=int(n))
        out = []
        for i in fi:
            c = self.chans[int(i)]
            o = int(rng.integers(c.shape[1] - self.W))
            out.append(c[:, o:o + self.W])
        return torch.stack(out)


# =========================================================================== #
#  Training
# =========================================================================== #
def _anchor_arrays(records, cfg, device):
    """Teacher-forced tensors for the MLE anchor -- the same object Stage 2a trained on."""
    X = np.concatenate([MB.build_feature_matrix(r, cfg) for r in records], 0)
    ya = np.concatenate([np.asarray(r.dtheta, float) for r in records])
    yd = np.concatenate([np.asarray(r.distance_px, float) for r in records])
    yt = np.concatenate([np.asarray(r.dt_s, float) for r in records])
    az, lc = JT._cond_cols(ya, yd, cfg["dth_scale"], cfg["kin_scales"])
    u = FR.state_gru_input(ya, yd, yt, cfg["dth_scale"], cfg["kin_scales"])
    lens = [int(r.n) for r in records]
    t = lambda a, d=np.float32: torch.tensor(np.asarray(a, d), device=device)
    return dict(X=t(X), az=t(az), lc=t(lc), u=t(u), ya=t(ya), yd=t(yd), yt=t(yt), lens=lens)


def anchor_nll(net, A, B, T, rng, device):
    """Teacher-forced NLL on B contiguous chunks of T bouts -- exact, no marginalisation."""
    starts, off = [], 0
    bounds = []
    for L in A["lens"]:
        bounds.append((off, off + L)); off += L
    for _ in range(B):
        lo, hi = bounds[int(rng.integers(len(bounds)))]
        s = int(rng.integers(lo, max(hi - T, lo + 1)))
        starts.append(np.arange(s, min(s + T, hi - 1)))
    T0 = min(len(s) for s in starts)
    ix = torch.tensor(np.stack([s[:T0] for s in starts]), dtype=torch.long, device=device)
    la, ld, lt, _ = net.state_chain(A["X"][ix], A["az"][ix], A["lc"][ix], A["u"][ix],
                                    A["ya"][ix], A["yd"][ix], A["yt"][ix])
    yd_, yt_ = A["yd"][ix], A["yt"][ix]
    md = torch.isfinite(yd_) & (yd_ > 0)
    mt = torch.isfinite(yt_) & (yt_ > 0)
    z = torch.zeros((), device=device)
    return -(la.mean() + (ld[md].mean() if md.any() else z) + (lt[mt].mean() if mt.any() else z))


def train(model, records, *, steps=3000, batch=64, W=60, lr_g=1e-5, lr_d=1e-3, anchor=1.0,
          gan_w=1.0, tau0=1.0, tau1=0.4, anchor_chunk=64, seed=0, device=None, log_every=50,
          verbose=True, n_critic=2):
    """Alternate {advance W bouts -> critic -> one guarded step -> detach}. Returns the model."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg = model._model, dict(model._cfg_feat)
    if not int(cfg.get("state", 0)):
        raise TypeError("rollout_gan needs the recurrent model (state>0): with a memoryless net "
                        "there is no parameter that can move a long-horizon statistic, which is "
                        "exactly why the previous rollout-aware attempt returned a null.")
    # the calibrator stack is OFF here by construction -- see the module docstring
    for k in ("wall_reject", "kin_feedback", "kin_innov", "vigor", "indiv", "dt_snap"):
        cfg.pop(k, None)
    cfg["wall_mode"] = "reflect"
    net.to(dev).train()

    critic = WindowCritic().to(dev)
    opt_g = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr_d, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()

    static = RT.build_static(records, cfg, dev)
    real = RealWindows(records, cfg, W, dev)
    A = _anchor_arrays(records, cfg, dev)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev.type).manual_seed(seed + 1)
    idx = torch.tensor(rng.integers(0, len(records), batch), dtype=torch.long, device=dev)
    st = RT.RolloutState(static, idx, cfg, dev, hidden=int(net.state_hidden),
                         z=RT.draw_latent(cfg, batch, dev, gen))
    stats, hist = {}, []
    t0 = time.time()

    for step in range(steps):
        tau = tau0 + (tau1 - tau0) * min(1.0, step / max(steps * 0.6, 1))
        # ---- advance the SAME trajectories W bouts (this is the TBPTT window) -------------- #
        o = RT.rollout(net, static, st, cfg, W, tau=tau, reparam=True, generator=gen)
        fake = channels_from(o["dtheta"], o["dist"], o["dt"], o["radial"], cfg)
        real_w = real.sample(batch, rng)

        # ---- critic ------------------------------------------------------------------------ #
        # n_critic > 1: the critic sees the SAME generated window again but fresh real windows,
        # so it gets several cheap updates per expensive rollout. Standard, and it matters here
        # because the rollout is the costly half -- the critic must not be the thing lagging.
        fake_d = fake.detach()
        for _c in range(int(n_critic)):
            d_real = critic(real.sample(batch, rng))
            d_fake = critic(fake_d)
            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

        # ---- generator: non-saturating GAN + MLE anchor ------------------------------------ #
        d_g = critic(fake)
        loss_gan = bce(d_g, torch.ones_like(d_g))
        nll = anchor_nll(net, A, max(4, batch // 8), anchor_chunk, rng, dev)
        loss_g = gan_w * loss_gan + anchor * nll
        opt_g.zero_grad(set_to_none=True)
        JT.guarded_step(opt_g, loss_g, [p for p in net.parameters() if p.requires_grad],
                        stats=stats)

        st.detach()                       # keep the trajectory, cut the graph
        row = dict(step=step, d_real=float(torch.sigmoid(d_real.detach()).mean()),
                   d_fake=float(torch.sigmoid(d_fake.detach()).mean()), loss_d=float(loss_d),
                   loss_gan=float(loss_gan), anchor=float(nll), tau=tau,
                   rad=float(o["radial"].mean()), occ=float((o["radial"] > 0.85).float().mean()))
        hist.append(row)
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:5d}  D(real) {row['d_real']:.3f}  D(fake) {row['d_fake']:.3f}  "
                  f"gan {row['loss_gan']:.4f}  anchor {row['anchor']:.4f}  tau {tau:.2f}  "
                  f"r/R {row['rad']:.3f}  occ {row['occ']:.4f}  [{time.time() - t0:6.1f}s]",
                  flush=True)
        # a critic that has won outright means the generator gradient has gone flat; say so
        if verbose and step and step % (log_every * 10) == 0 and row["d_fake"] < 0.02:
            print("    note: D(fake) < 0.02 -- the critic is saturating. Lower lr_d or raise "
                  "gan_w; a saturated critic gives the generator almost no gradient.")
    JT._warn_unstable(stats, [h["anchor"] for h in hist])
    net.eval()
    # BACK TO CPU. `score` and `sample_batch` build their tensors on the CPU with no device
    # argument, so leaving the net on CUDA makes the very next call -- the AFTER report -- die on a
    # device mismatch, AFTER the whole run has been paid for and BEFORE anything is saved. It never
    # reproduces locally because there is no CUDA there, which is exactly what makes it worth
    # pinning here rather than trusting the happy path.
    net.to(torch.device("cpu"))
    model.history_gan = hist
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a Stage-2a (state>0) checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bouts", default=str(Path(__file__).parent.parent / "data" /
                                           "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--lr-g", type=float, default=1e-5)
    ap.add_argument("--lr-d", type=float, default=1e-3)
    ap.add_argument("--n-critic", type=int, default=2)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--gan-w", type=float, default=1.0)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    MX._numpy2_unpickle_compat()
    recs = MB.build_border_records(pd.read_csv(a.bouts))
    rl = [recs[f] for f in sorted(recs)]
    ho = []
    if a.holdout:
        rl, ho = rl[:-a.holdout], rl[-a.holdout:]
        print(f"training on {len(rl)} fish, holding out {len(ho)}")
    model = MX.load_mix(a.model)
    guard = ho or rl
    print(f"BEFORE  {report(model, guard)}")
    train(model, rl, steps=a.steps, batch=a.batch, W=a.window, lr_g=a.lr_g, lr_d=a.lr_d,
          anchor=a.anchor, gan_w=a.gan_w, seed=a.seed, n_critic=a.n_critic)
    # SAVE BEFORE REPORTING. The report rolls out and scores, i.e. it is the most failure-prone
    # thing here, and it runs after thousands of training steps have already been paid for. Saving
    # first means a reporting bug costs a diagnostic, not the run.
    MX.save_mix(model, a.out)
    print("wrote", a.out)
    print(f"AFTER   {report(model, guard)}")


def report(model, records, seed=0):
    """The whole scorecard on one line -- never the AUC alone, and never Metric A alone."""
    import generative_metrics as GM
    bits, acf, occ, rad = [], {"dt": [], "dist": [], "rad": []}, [], []
    for i, r in enumerate(records):
        v = np.asarray(model.score(r)["logdens"], float)
        bits.append(-np.nanmean(v) / np.log(2.0))
        g = model.sample(r, np.random.default_rng([seed, int(i)]))
        u = np.hypot(g["x_px"] - r.dish_cx, g["y_px"] - r.dish_cy) / r.dish_R
        acf["dt"].append(GM.win_acf1(np.log(np.clip(g["dt_s"], 1e-6, None))))
        acf["dist"].append(GM.win_acf1(np.log(np.clip(g["distance_px"], 1e-6, None))))
        acf["rad"].append(GM.win_acf1(u))
        occ.append(float(np.mean(u > 0.85))); rad.append(float(np.mean(u)))
    return (f"bits {np.mean(bits):.4f}  wacf dt {np.nanmean(acf['dt']):+.3f} "
            f"dist {np.nanmean(acf['dist']):+.3f} rad {np.nanmean(acf['rad']):+.3f}  "
            f"r/R {np.mean(rad):.3f}  occ {np.mean(occ):.4f}")


if __name__ == "__main__":
    main()
