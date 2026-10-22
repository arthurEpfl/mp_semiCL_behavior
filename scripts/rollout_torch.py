"""Batched, DIFFERENTIABLE free-run of the joint model, in torch.

WHY THIS EXISTS
---------------
Every rollout in this repo has been numpy: one trajectory at a time, ~1.5 s each, and no
gradients. That forced the first rollout-aware attempt into REINFORCE, whose variance was larger
than the effect it was trying to measure (between-arm differences came out SMALLER than
within-arm noise). It also made a training step cost ~17 s, so nothing could converge.

Here the whole rollout is torch and batched: B trajectories advance together, so a W-bout window
is W small matmuls on a batch, i.e. milliseconds on a GPU rather than seconds per trajectory. And
because every emission is REPARAMETERISED, a loss computed on the generated window differentiates
straight back into the emission parameters -- a pathwise gradient instead of a score-function one.
That is the single change that makes rollout-aware training tractable.

THE ONE THING THAT MUST BE TRUE
-------------------------------
This must be the SAME object the numpy generator produces, or we train one model and ship another
-- the most repeated trap in this project (kin lineage vs joint, z=0 vs z~prior, the a_z
conditioner, Tauchen vs the exact AR(1)). It is checked distributionally, not bit-for-bit, by
``verify_against_numpy`` below: same weights, same cfg, marginals + across-window acf + radial
occupancy must agree within replicate noise.

CONTAINMENT (was deliberately not ported; now is, under reparam=False)
----------------------------------------------------------------------
Originally neither ``centre_bias`` nor the shallow-border REJECTION sampler was ported, on the
reasoning that rollout-aware training should LEARN containment: the discriminator sees r/R directly.
That is right for the GAN and wrong for every other caller. With ``contain=True`` the calibrators
fit (bulk, shell) or (d0, d1) against a rollout that then ignored them entirely -- calibrating one
object and generating from another, on the channel that carries ~half the C2ST permutation
importance. Both are implemented now:

  * ``centre_bias``  always active, and differentiable (it is a shift on the emitted turn).
  * ``wall_reject``  a FIXED-COUNT MASKED redraw rather than the numpy while-loop, so it batches:
    each pass redraws only the rows still rejected. It carries no pathwise derivative -- a redraw is
    a discrete accept/reject -- so it RAISES under ``reparam=True`` rather than silently biasing a
    gradient. That is also the regime the GAN drops it in by design, so nothing is lost there.

A reflecting guardrail stays regardless, because that is physics (the fish cannot leave the dish),
not calibration. ``verify_against_numpy`` still compares with ``wall_reject`` removed on BOTH sides,
so it tests the feature/emission/integration port rather than the redraw loop.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP as M                      # noqa: E402
import fit_MLP_border as MB              # noqa: E402
import fit_MLP_mix as MX                 # noqa: E402

_LOG2PI = float(math.log(2.0 * math.pi))


# =========================================================================== #
#  Feature blocks, ported to batched torch
# =========================================================================== #
def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _hypot(a, b, eps=1e-12):
    """``hypot`` with a finite gradient at the origin.

    ``torch.hypot(0, 0)`` and ``sqrt(0)`` both have an INFINITE derivative, and the resulting NaN
    propagates through a masked-out branch regardless of the mask -- 0 * NaN is NaN in the backward
    pass, so zeroing a channel in the forward does not protect the gradient. Measured: every single
    generator step was skipped by the NaN guard until this was fixed, so the model silently never
    trained while the loss curve looked perfectly healthy.
    """
    return torch.sqrt(a * a + b * b + eps)


def border_channels_t(dnorm, psi, mode="basis", lc=None):
    """(B, 3) raw or (B, 9) basis border channels -- the torch twin of ``MB.border_channels``.

    Same softmax-normalised Gaussian basis on log wall-distance, so the profile flat-extrapolates
    outside the knots exactly as the numpy version does.
    """
    s, c = torch.sin(psi), torch.cos(psi)
    if mode != "basis":
        return torch.stack([dnorm, s, c], dim=1)
    ld = torch.log(torch.clamp(dnorm, min=MB.BORDER_MIN))
    z = -0.5 * ((ld.unsqueeze(1) - lc.unsqueeze(0)) / MB.BORDER_W) ** 2
    G = torch.softmax(z, dim=1)                      # == exp(z - max) / sum, i.e. MB's normalisation
    return torch.cat([dnorm.unsqueeze(1), G * s.unsqueeze(1), G * c.unsqueeze(1)], dim=1)


def shoal_channels_t(phi, r_px, n_dots, moving, use_n=True, lc=None, act=None):
    """(B, 18) / (B, 13) shoal channels -- the torch twin of ``M.shoal_channels``.

    ``act`` gates every channel to exactly 0 off-stimulus, as in the numpy version, so a bout with
    no active dot contributes nothing rather than a spurious bearing.
    """
    if act is None:
        act = (torch.isfinite(phi) & torch.isfinite(r_px) & (r_px > 0.0)).to(phi.dtype)
    phi0 = torch.nan_to_num(phi)
    s, c = torch.sin(phi0) * act, torch.cos(phi0) * act
    r_safe = torch.where(act > 0, torch.clamp(torch.nan_to_num(r_px, nan=M.SHOAL_R_REF),
                                              min=M.SHOAL_R_MIN),
                         torch.full_like(r_px, M.SHOAL_R_REF))
    lr = torch.log(r_safe)
    N = torch.nan_to_num(n_dots)
    mv = torch.nan_to_num(moving)
    nz = (torch.log2(torch.clamp(N, min=1.0)) / math.log2(M.SHOAL_N_MAX)) * act
    rz = torch.clamp((lr - math.log(M.SHOAL_R_REF)) / M.SHOAL_R_OCT, -1.0, 1.0) * act
    z = -0.5 * ((lr.unsqueeze(1) - lc.unsqueeze(0)) / M.SHOAL_RBF_W) ** 2
    G = torch.softmax(z, dim=1) * act.unsqueeze(1)
    cols = [act.unsqueeze(1), mv.unsqueeze(1)]
    if use_n:
        cols.append(nz.unsqueeze(1))
    cols += [rz.unsqueeze(1), G * s.unsqueeze(1), G * c.unsqueeze(1)]
    if use_n:
        cols.append(G * (nz * s).unsqueeze(1))
    cols += [(mv * s).unsqueeze(1), (mv * c).unsqueeze(1)]
    return torch.cat(cols, dim=1)


def advance_t(x, y, h, d, cx, cy, R, eps=1e-3, n_bounce=3):
    """Specular reflection off the dish boundary, batched and differentiable.

    The numpy ``MB.advance`` loops until the step is used up; here the loop is UNROLLED a fixed
    number of times with a mask, because a data-dependent while-loop cannot be batched. Three
    bounces is far beyond what a real step needs (a step is ~15 px against a ~178 px radius, so a
    second bounce is already vanishingly rare) -- any residual is absorbed by the final clamp.
    """
    # HOST SYNCS ARE FREE ON CPU AND RUINOUS ON GPU. The `bool(...)` early exits below force a
    # device->host synchronisation, up to 3 per timestep. On CPU that costs nothing and skips real
    # work. On CUDA it stalls the launch pipeline every bout -- for the GAN (60 timesteps x 6000
    # steps) that is over a million syncs, which would dominate the whole run. So the early exits
    # are taken only on CPU; on GPU the loop always runs its fixed iterations, which is correct
    # because every iteration is a masked no-op for rows that have already finished (d = 0).
    sync_ok = x.device.type == "cpu"
    for _ in range(n_bounce):
        nx, ny = x + d * torch.cos(h), y + d * torch.sin(h)
        r = _hypot(nx - cx, ny - cy)
        inside = r <= (R - eps)
        if sync_ok and bool(inside.all()):
            x, y = nx, ny
            break
        # travel to the boundary along the heading: solve |p + t*u - c| = R for t > 0
        px, py = x - cx, y - cy
        ux, uy = torch.cos(h), torch.sin(h)
        b = px * ux + py * uy
        cq = px * px + py * py - (R - eps) ** 2
        disc = torch.clamp(b * b - cq, min=0.0)
        t_hit = torch.clamp(-b + torch.sqrt(disc + 1e-12), min=0.0)
        t_hit = torch.minimum(t_hit, d)
        hx, hy = x + t_hit * ux, y + t_hit * uy
        # mirror the heading about the inward normal at the hit point
        nrm = _hypot(hx - cx, hy - cy).clamp(min=1e-9)
        n_x, n_y = (hx - cx) / nrm, (hy - cy) / nrm
        dot = ux * n_x + uy * n_y
        rx, ry = ux - 2.0 * dot * n_x, uy - 2.0 * dot * n_y
        h_ref = torch.atan2(ry, rx)
        x = torch.where(inside, nx, hx)
        y = torch.where(inside, ny, hy)
        h = torch.where(inside, h, h_ref)
        d = torch.where(inside, torch.zeros_like(d), d - t_hit)
        if sync_ok and bool((d <= 0).all()):
            break
    # hard floor: nothing may end outside the dish, whatever the unrolled bounces left
    r = _hypot(x - cx, y - cy)
    over = r > (R - eps)
    if sync_ok:
        if bool(over.any()):                       # CPU: skip the work when nothing is outside
            sc = torch.where(over, (R - eps) / r.clamp(min=1e-9), torch.ones_like(r))
            x = cx + (x - cx) * sc
            y = cy + (y - cy) * sc
    else:                                          # GPU: branchless, no sync
        sc = torch.where(over, (R - eps) / r.clamp(min=1e-9), torch.ones_like(r))
        x = cx + (x - cx) * sc
        y = cy + (y - cy) * sc
    return x, y, h


# =========================================================================== #
#  Static per-record tensors
# =========================================================================== #
def build_static(records, cfg_feat, device):
    """Everything the rollout reads that does NOT depend on the generated pose.

    The stimulus PROGRAMME (dot centroid, moving, N) is indexed by bout and teacher-forced, exactly
    as the numpy generator does -- the fish's BEARING to it is recomputed from the generated pose,
    which is what makes the loop closed.
    """
    n = max(int(r.n) for r in records)
    B = len(records)
    z = lambda: np.zeros((B, n), dtype=np.float32)
    dot_cx, dot_cy, mv, nd = np.full((B, n), np.nan, np.float32), np.full((B, n), np.nan, np.float32), z(), z()
    valid = np.zeros((B, n), np.float32)
    cx = np.zeros(B, np.float32); cy = np.zeros(B, np.float32); R = np.zeros(B, np.float32)
    x0 = np.zeros(B, np.float32); y0 = np.zeros(B, np.float32); h0 = np.zeros(B, np.float32)
    for i, r in enumerate(records):
        k = int(r.n)
        valid[i, :k] = 1.0
        dcx = np.asarray(getattr(r, "dot_cx", np.full(k, np.nan)), np.float32)
        dcy = np.asarray(getattr(r, "dot_cy", np.full(k, np.nan)), np.float32)
        dot_cx[i, :k] = dcx; dot_cy[i, :k] = dcy
        mv[i, :k] = np.asarray(r.moving, np.float32)
        nd[i, :k] = np.asarray(getattr(r, "n_dots", np.zeros(k)), np.float32)
        cx[i], cy[i], R[i] = r.dish_cx, r.dish_cy, r.dish_R
        x0[i], y0[i] = r.x_px[0], r.y_px[0]
        h0[i] = (float(MB._travel_ref(r)[0]) if cfg_feat.get("psi_ref") == "travel"
                 else float(r.f_ori_rad[0]))
    t = lambda a: torch.tensor(a, device=device)
    # PER-RECORD length. Records differ in length, and the stimulus programme must wrap on the
    # record's OWN length: wrapping on the padded width would march short fish through rows of
    # NaN dot centroid, i.e. silently roll them out as if the stimulus had been switched off.
    nlen = torch.tensor(np.array([int(r.n) for r in records], np.int64), device=device)
    return dict(dot_cx=t(dot_cx), dot_cy=t(dot_cy), moving=t(mv), n_dots=t(nd), valid=t(valid),
                cx=t(cx), cy=t(cy), R=t(R), x0=t(x0), y0=t(y0), h0=t(h0),
                nlen=nlen, n=n, B=B)


class RolloutState:
    """The carried state of B ongoing trajectories. ``detach()`` is what makes TBPTT work."""

    def __init__(self, static, idx, cfg_feat, device, hidden=0, z=None, h0=None):
        K = int(cfg_feat["K"]); Kk = int(cfg_feat.get("K_kin", 3))
        dm, _ds, tm, _ts = cfg_feat["kin_scales"]
        b = len(idx)
        self.idx = idx
        self.x = static["x0"][idx].clone()
        self.y = static["y0"][idx].clone()
        self.h = static["h0"][idx].clone()
        self.dth_hist = torch.zeros((b, K), device=device)          # lag 1..K, newest first
        # SEEDED AT THE LOG-MEAN, not at zero. The kin block feeds (log d - dm)/ds, so a zeros
        # init would present (0 - dm)/ds ~ -4.5 for the first K_kin bouts -- a large spurious
        # input, where the numpy generator feeds exactly 0 (it skips lags before t=0).
        self.ld_hist = torch.full((b, Kk), float(dm), device=device)
        self.lt_hist = torch.full((b, Kk), float(tm), device=device)
        self.t = torch.zeros(b, dtype=torch.long, device=device)
        self.gru = torch.zeros((b, hidden), device=device) if hidden else None
        # PER-TRAJECTORY SEED. Drawn from the SAME N(0, I) prior the embedding was fitted under, so
        # a rollout is an honest "new fish" rather than a replay of a training fish's seed. Held at
        # zero when the model has no seed (h0_dim = 0), which reproduces the gru1 behaviour exactly.
        if h0 is not None and self.gru is not None:
            self.gru = h0
        # SLOW AR(1) DRIFT STATE, one scalar per trajectory. Seeded from its STATIONARY prior
        # N(0, 1), matching the numpy generator (fit_MLP_joint.py:541) -- seeding at 0 would start
        # every trajectory at the population mean and take ~tau bouts to forget it, which on a
        # 60-bout window is a visible transient in exactly the statistic this state exists to move.
        self.s = None
        if cfg_feat.get("drift"):
            # generator=None here is safe ONLY because `rollout` forks and seeds the
            # global RNG from the caller's generator; without that this draw would be
            # the one unreproducible thing in an otherwise seeded rollout.
            self.s = torch.randn((b, 1), device=device)
        # PER-TRAJECTORY LATENT, drawn once, exactly as the numpy generator does (one draw per
        # rollout from the fitted prior). Holding it at 0 would score the population fish while the
        # numpy side ran z ~ N(0, sd) -- the same z=0-vs-z~prior mismatch that made the previous
        # rollout_finetune.py silently wrong.
        self.z = z

    def detach(self):
        """Cut the graph but KEEP the values -- the trajectory continues, the gradient does not.

        This is the whole TBPTT mechanism, and it does double duty: it bounds the backprop depth
        AND it is what lets a trajectory run for thousands of bouts across many optimiser steps, so
        the model is judged on genuinely long-run behaviour rather than fresh short starts.
        """
        for k in ("x", "y", "h", "dth_hist", "ld_hist", "lt_hist"):
            setattr(self, k, getattr(self, k).detach())
        if self.gru is not None:
            self.gru = self.gru.detach()
        if self.s is not None:
            # The drift state carries no parameter gradient of its own -- its innovations are draws
            # and its phi is PINNED, not fitted -- but it is a graph node once it has been multiplied
            # into a head, so it detaches with everything else. `z` does not, because it is a
            # constant draw that never enters the recursion.
            self.s = self.s.detach()
        return self


# =========================================================================== #
#  Reparameterised emission
# =========================================================================== #
def _gumbel_hard(logits, tau, generator=None):
    """Straight-through Gumbel-softmax one-hot.

    Hard forward (so the sampled trajectory is a real draw from the mixture, not a blend of
    components) and soft backward (so the gradient reaches the mixture weights). This is the only
    biased piece of the estimator; ``tau`` is annealed and the marginals are watched.
    """
    u = torch.rand(logits.shape, device=logits.device, generator=generator).clamp_(1e-9, 1 - 1e-9)
    g = -torch.log(-torch.log(u))
    soft = torch.softmax((logits + g) / tau, dim=-1)
    idx = soft.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
    return hard + soft - soft.detach()


def sample_angle(ang, dth_scale, tau, generator=None, reparam=True):
    """One reparameterised draw from the 5-component turn mixture (Normal + 2 Gammas per side).

    ``Gamma.rsample`` carries implicit reparameterisation gradients w.r.t. both concentration and
    rate, so the gradient reaches the shape parameters and not only the mixture weights. NOTE the
    numpy generator uses ``rng.gamma(a, theta)`` where theta is a SCALE, so the torch rate is
    1/theta -- getting that backwards would silently change the step-size law.
    """
    logw, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = ang
    if reparam:
        oh = _gumbel_hard(logw, tau, generator)
    else:
        oh = torch.nn.functional.one_hot(torch.distributions.Categorical(logits=logw).sample(),
                                         num_classes=5).to(logw.dtype)
    eps = torch.randn(sF.shape, device=sF.device, generator=generator)
    vF = sF * eps

    def _g(a, th):
        d = torch.distributions.Gamma(a.clamp(min=1e-4), 1.0 / th.clamp(min=1e-6))
        return d.rsample() if reparam else d.sample()

    v = torch.stack([vF, -_g(aL1, tL1), -_g(aL2, tL2), _g(aR1, tR1), _g(aR2, tR2)], dim=1)
    return torch.clamp((oh * v).sum(1), -180.0, 180.0)


def sample_lognorm_mix(par, hi, tau, generator=None, reparam=True, innov=1.0, trunc=None):
    """One reparameterised draw from a log-normal mixture (the distance / interval heads).

    ``trunc`` (seconds / px, None = off): the mixture TRUNCATED at that bound (MX._dt_logpdf) --
    component weights conditioned on the bound, then an inverse-CDF draw of the truncated normal
    in log space, so no draw ever exceeds it and nothing is clipped.

    ``innov`` is ``kin_innov``'s compensation factor: the draw is widened about THIS bout's
    conditional mean IN LOG SPACE, ``ld -> m + c (ld - m)`` with ``m = sum_j w_j mu_j``, exactly as
    the numpy generator does it (fit_MLP_joint.py:756-761). Applying it here rather than to the
    returned value matters: the clamp must come ONCE and AFTER the rescale, or a draw that lands on
    the cap gets rescaled away from it and the two generators disagree on the ~1% of bouts that sit
    on a clip. ``innov=1.0`` leaves the expression untouched, so every arm without the mechanism is
    bit-identical.

    Why this compensation exists at all: ``kin_feedback`` damps the free-running loop to fix its
    autocorrelation, and damping an AR damps its spread too. For a linear AR, scaling the innovation
    multiplies the stationary variance while leaving acf(k) = phi^k alone -- so lambda sets the acf
    and c the spread, and the two are only correct together.
    """
    logw, mu, sig = par
    if trunc is not None:
        # truncated: weights x Phi_k(bound), then eps = Phi^-1(u * Phi_k) per component
        log_F = torch.special.log_ndtr((float(np.log(trunc)) - mu) / sig)
        logw = logw + log_F
        u = torch.rand(mu.shape, device=mu.device, generator=generator).clamp_(1e-9, 1 - 1e-9)
        eps = torch.special.ndtri((u * torch.exp(log_F)).clamp(1e-12, 1 - 1e-12))
        hi = float(trunc)              # the bound is the cap; no clip below it
    else:
        eps = torch.randn(mu.shape, device=mu.device, generator=generator)
    oh = (_gumbel_hard(logw, tau, generator) if reparam else
          torch.nn.functional.one_hot(torch.distributions.Categorical(logits=logw).sample(),
                                      num_classes=logw.shape[1]).to(logw.dtype))
    ld = (oh * (mu + sig * eps)).sum(1)
    if innov != 1.0:
        m = (torch.softmax(logw, dim=1) * mu).sum(1)
        ld = m + float(innov) * (ld - m)
    return torch.clamp(torch.exp(ld), min=1e-3, max=float(hi))


# =========================================================================== #
#  The rollout
# =========================================================================== #
def _heads(net, x, hstate):
    """Trunk output the three heads read. With a GRU state present it is [body(x), h]."""
    b = net.body(x)
    return torch.cat([b, hstate], dim=1) if hstate is not None else b


def draw_latent(cfg_feat, n, device, generator=None):
    """One per-trajectory latent draw from the fitted prior, matching the numpy generator."""
    re = cfg_feat.get("fish_re")
    if not re:
        return None
    sd = torch.tensor(np.asarray(re["sd"], np.float32).reshape(1, -1), device=device)
    return torch.randn((n, sd.shape[1]), device=device, generator=generator) * sd


def rollout(net, static, st, cfg_feat, T, *, tau=1.0, reparam=True, generator=None):
    """Advance B trajectories T bouts, returning the generated window (and updating ``st``).

    SEED DISCIPLINE, and why this wrapper exists. Several draws inside CANNOT take a generator:
    ``torch.distributions.Categorical.sample`` (the mixture component, in all three heads) and
    ``Gamma.sample``/``rsample`` (the four turn components) read the GLOBAL torch RNG regardless of
    what is passed. So every non-reparameterised rollout -- which is every metric and every
    calibrator rollout -- was unreproducible from its seed, and worse, NOT CRN-paired across models
    or grid points. The calibrators read paired differences, so that noise went straight into every
    fitted constant. Measured: re-running the same cfg with the same seed gave a different
    trajectory.

    Fixed by forking the global RNG and seeding it FROM ``generator``, so the rollout is a pure
    function of the seed and the caller's own global stream is left exactly as it was found.
    """
    if generator is None:
        return _rollout_impl(net, static, st, cfg_feat, T, tau=tau, reparam=reparam,
                             generator=None)
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item())
    dev_t = st.x.device
    with torch.random.fork_rng(devices=([dev_t.index] if dev_t.type == "cuda" else []),
                               enabled=True):
        torch.manual_seed(seed)
        return _rollout_impl(net, static, st, cfg_feat, T, tau=tau, reparam=reparam,
                             generator=generator)


def _rollout_impl(net, static, st, cfg_feat, T, *, tau=1.0, reparam=True, generator=None):
    """The rollout proper. Call ``rollout``, not this -- see its seed-discipline note.

    Returns dict of (B, T) tensors: dtheta, dist, dt, x, y, radial -- all differentiable when
    ``reparam`` is True. The CHAIN is preserved exactly as the numpy generator has it: turn first,
    then distance conditioned on |dtheta|, then interval conditioned on both.
    """
    dev = st.x.device
    K = int(cfg_feat["K"]); Kk = int(cfg_feat.get("K_kin", 3))
    dth_scale = float(cfg_feat["dth_scale"])
    dm, ds, tm, ts = cfg_feat["kin_scales"]
    dist_hi, dt_hi = cfg_feat.get("kin_clip", (400.0, 600.0))
    mode = cfg_feat.get("border_mode", "raw")
    use_border = bool(cfg_feat.get("use_border"))
    stim_mode = cfg_feat["stim_mode"]
    if stim_mode.endswith(M.RZLO_SUFFIX):           # shoal_channels_t clips rz at -1; use the numpy / batch engines
        raise NotImplementedError(f"stim_mode {stim_mode!r}: the lower-clip variants are not in the torch engine")
    use_n = stim_mode != "shoalr"
    b_lc =torch.tensor(MB._BORDER_LC, dtype=torch.float32, device=dev)
    s_lc = torch.tensor(M._SHOALF_LC if stim_mode == "shoalf" else M._SHOAL_LC,
                        dtype=torch.float32, device=dev)
    kfb = cfg_feat.get("kin_feedback", 1.0)
    kfb_d, kfb_t = ((float(kfb["dist"]), float(kfb["dt"])) if isinstance(kfb, dict)
                    else (float(kfb), float(kfb)))
    # KIN_INNOV: widen the innovation about THIS bout's conditional mean. Absent from this file until
    # now, which meant a kin_innov-calibrated checkpoint rolled out WITHOUT its compensation -- the
    # calibrator restored a spread that the torch rollout then silently dropped, and the model was
    # scored on trajectories narrower than the ones it was calibrated to produce. lambda sets the
    # acf and c the spread (see fit_MLP_joint.py:623-627); the two are only correct together.
    _ki = cfg_feat.get("kin_innov")
    ci_d = float(_ki.get("dist", 1.0)) if _ki else 1.0
    ci_t = float(_ki.get("dt", 1.0)) if _ki else 1.0
    # DRIFT: the slow AR(1) shared across all three heads. Loadings and the re-centring constants are
    # read EXACTLY as the numpy generator reads them (fit_MLP_joint.py:534-541) -- including the
    # g_dist/g_dt gains folded into the loading and the dist_c/dt_c that cancel the marginal
    # inflation those gains would otherwise cause.
    # CONTAINMENT. Neither mechanism was ported when this file was written, on the reasoning that
    # rollout-aware training should LEARN containment. That is right for the GAN and wrong for every
    # other caller: with `contain=True` the calibrators fit (bulk, shell) or (d0, d1) against a
    # rollout that then ignored them, so the model was calibrated to one object and generated from
    # another -- and radial carries ~half the C2ST permutation importance. Both are here now.
    _cb = cfg_feat.get("centre_bias", 0.0)
    cb_bulk = cb_shell = 0.0
    cb_u0 = 0.75
    if isinstance(_cb, dict):
        cb_bulk = float(_cb.get("bulk", 0.0)); cb_shell = float(_cb.get("shell", 0.0))
        cb_u0 = float(_cb.get("u0", 0.75))
    else:
        cb_bulk = float(_cb)
    # SHALLOW-BORDER REJECTION, as a fixed-count MASKED redraw rather than the numpy while-loop:
    # each pass redraws only the rows still rejected, so it batches. Correct only where gradients
    # are not needed -- a redraw is a discrete accept/reject and carries no pathwise derivative --
    # so it is REFUSED under reparam, which is also the regime the GAN deliberately drops it in.
    _wr = cfg_feat.get("wall_reject")
    rd0 = rd1 = None
    rmax = 0
    if _wr:
        if reparam:
            raise ValueError("wall_reject with reparam=True: rejection is a discrete accept/reject "
                             "with no pathwise gradient. Drop it for adversarial training (which is "
                             "what rollout_gan does) or roll out with reparam=False.")
        rd0 = float(_wr.get("d0", MB.WALL_REJECT_DEFAULT["d0"]))
        rd1 = float(_wr.get("d1", MB.WALL_REJECT_DEFAULT["d1"]))
        rmax = max(int(_wr.get("max_draws", MB.WALL_REJECT_DEFAULT["max_draws"])), 1)
    _dr = cfg_feat.get("drift")
    d_turn = d_dist = d_dt = d_dist_c = d_dt_c = d_phi = 0.0
    if _dr:
        d_turn = float(_dr.get("turn", 0.0))
        d_dist = float(_dr.get("dist", 0.0)) * float(_dr.get("g_dist", 1.0))
        d_dt = float(_dr.get("dt", 0.0)) * float(_dr.get("g_dt", 1.0))
        d_dist_c = float(_dr.get("dist_c", 0.0)); d_dt_c = float(_dr.get("dt_c", 0.0))
        d_phi = float(_dr["phi"])
        # d_bord is NOT read: measured refuted (fit_MLP_fishre.py:563-584) -- the acf gain it buys
        # costs occupancy 2.3-20x real. If it is ever revived it belongs on the border GAIN, not as
        # an additive turn bias.
    z = st.z
    cx, cy, R = static["cx"][st.idx], static["cy"][st.idx], static["R"][st.idx]
    out = {k: [] for k in ("dtheta", "dist", "dt", "x", "y", "radial")}

    rows = st.idx
    for _ in range(T):
        tt = torch.minimum(st.t, static["nlen"][rows] - 1)
        # ---- border block, from the GENERATED pose -------------------------------------- #
        r_now = _hypot(st.x - cx, st.y - cy)
        psi = wrap_pi(torch.atan2(cy - st.y, cx - st.x) - st.h)
        blk_b = (border_channels_t(torch.clamp((R - r_now) / R, min=0.0), psi, mode, b_lc)
                 if use_border else torch.zeros((len(rows), 0), device=dev))
        # ---- stimulus block, bearing recomputed from the GENERATED pose ------------------ #
        dcx_r = static["dot_cx"][rows, tt]; dcy_r = static["dot_cy"][rows, tt]
        # SANITISE BEFORE THE TRIG, not after. Off-stimulus bouts carry a NaN dot centroid; masking
        # the resulting channels to zero fixes the forward but leaves NaN in the BACKWARD (0 * NaN
        # = NaN), which silently zeroed every generator update. The placeholder is offset by 1 px
        # so the range is never exactly 0 either.
        act = torch.isfinite(dcx_r) & torch.isfinite(dcy_r)
        dcx = torch.where(act, dcx_r, st.x.detach() + 1.0)
        dcy = torch.where(act, dcy_r, st.y.detach() + 1.0)
        phi = wrap_pi(torch.atan2(dcy - st.y, dcx - st.x) - st.h)
        r_dot = _hypot(dcx - st.x, dcy - st.y)
        blk_s = shoal_channels_t(phi, r_dot, static["n_dots"][rows, tt],
                                 static["moving"][rows, tt], use_n=use_n, lc=s_lc,
                                 act=act.to(phi.dtype))
        # ---- history blocks -------------------------------------------------------------- #
        blk_h = st.dth_hist / dth_scale
        blk_k = torch.cat([kfb_d * (st.ld_hist - dm) / ds, kfb_t * (st.lt_hist - tm) / ts], dim=1)
        # block order MUST match MB.build_feature_matrix: history, stim, stimhist, border, kin
        x_row = torch.cat([blk_h, blk_s, blk_b, blk_k], dim=1)

        # ---- chained emission ------------------------------------------------------------ #
        trunk = _heads(net, x_row, st.gru)
        head_out = net.head(trunk)
        if z is not None and hasattr(net, "z_turn"):
            head_out = torch.cat([head_out[:, :1] + net.z_turn(net._z_core(z)), head_out[:, 1:]], 1)
        if d_turn and st.s is not None:
            # The SAME shift `_params_at` applies at score time: a translation of the first mixture
            # logit. Doing it here rather than on the softmaxed weights keeps the two paths one
            # expression, which is what stops the trained and rolled-out objects diverging.
            head_out = torch.cat([head_out[:, :1] + d_turn * st.s, head_out[:, 1:]], 1)
        ang = net._angle_params(head_out)
        dth = sample_angle(ang, dth_scale, tau, generator, reparam)
        if cb_bulk or cb_shell:
            # shell-gated inward turn bias, exactly MX/JT's form: bulk everywhere plus a ramp that
            # only fires past u0. `psi_c` is the bearing to the dish CENTRE from the generated pose.
            _u = _hypot(st.x - cx, st.y - cy) / R
            cb = cb_bulk + cb_shell * torch.clamp((_u - cb_u0) / max(1.0 - cb_u0, 1e-6), min=0.0)
            psi_c = wrap_pi(torch.atan2(cy - st.y, cx - st.x) - st.h)
            dth = torch.clamp(dth + cb * torch.sin(psi_c), -180.0, 180.0)

        a_z = torch.abs(dth) / dth_scale
        k_d = net.n_dist
        do = net.dist_head(torch.cat([trunk, a_z.unsqueeze(1)], dim=1))
        dmu = do[:, k_d:2 * k_d]
        if z is not None and hasattr(net, "z_dist"):
            dmu = dmu + net.z_dist(net._z_core(z))
        if d_dist and st.s is not None:
            dmu = dmu + (d_dist * st.s - d_dist_c)
        dpar = (torch.log_softmax(do[:, :k_d], 1), dmu,
                torch.nn.functional.softplus(do[:, 2 * k_d:3 * k_d]) + M.EPS)
        # innov re-centres about THIS bout's conditional mean, which includes every shift applied
        # to `dmu` above -- widening about the fixed TRAINING log-mean instead is `kin_var_comp`,
        # which lost twice.
        dist = sample_lognorm_mix(dpar, dist_hi, tau, generator, reparam, innov=ci_d)

        ld_c = (torch.log(torch.clamp(dist, min=1e-6)) - dm) / ds
        k_t = net.n_dt
        to = net.dt_head(torch.cat([trunk, a_z.unsqueeze(1), ld_c.unsqueeze(1)], dim=1))
        tmu = to[:, k_t:2 * k_t]
        if z is not None and hasattr(net, "z_dt"):
            tmu = tmu + net.z_dt(net._z_core(z))
        if d_dt and st.s is not None:
            tmu = tmu + (d_dt * st.s - d_dt_c)
        tpar = (torch.log_softmax(to[:, :k_t], 1), tmu,
                torch.nn.functional.softplus(to[:, 2 * k_t:3 * k_t]) + M.EPS)
        dtv = sample_lognorm_mix(tpar, dt_hi, tau, generator, reparam, innov=ci_t)

        if rd0 is not None:
            # REDRAW THE WHOLE TRIPLE, never just the step: the heads are chained, so keeping the
            # turn and resampling the distance would sample from a different law. Judged on where
            # the bout LANDS, and only the still-rejected rows are redrawn.
            for _k in range(rmax):
                _nx = st.x + dist * torch.cos(st.h + dth * (math.pi / 180.0))
                _ny = st.y + dist * torch.sin(st.h + dth * (math.pi / 180.0))
                _u = _hypot(_nx - cx, _ny - cy) / R
                _d_mm = 0.5 * MB.DISH_MM * (1.0 - _u)
                _acc = torch.clamp((_d_mm - rd0) / max(rd1 - rd0, 1e-9), 0.0, 1.0)
                _bad = torch.rand(_acc.shape, device=dev, generator=generator) >= _acc
                if not bool(_bad.any()):
                    break
                # one fresh draw of the whole chain, spliced in only on the rejected rows
                _dth2 = sample_angle(ang, dth_scale, tau, generator, reparam)
                if cb_bulk or cb_shell:
                    _dth2 = torch.clamp(_dth2 + cb * torch.sin(psi_c), -180.0, 180.0)
                _az2 = torch.abs(_dth2) / dth_scale
                _do2 = net.dist_head(torch.cat([trunk, _az2.unsqueeze(1)], dim=1))
                _dmu2 = _do2[:, k_d:2 * k_d]
                if z is not None and hasattr(net, "z_dist"):
                    _dmu2 = _dmu2 + net.z_dist(net._z_core(z))
                if d_dist and st.s is not None:
                    _dmu2 = _dmu2 + (d_dist * st.s - d_dist_c)
                _dist2 = sample_lognorm_mix(
                    (torch.log_softmax(_do2[:, :k_d], 1), _dmu2,
                     torch.nn.functional.softplus(_do2[:, 2 * k_d:3 * k_d]) + M.EPS),
                    dist_hi, tau, generator, reparam, innov=ci_d)
                _ldc2 = (torch.log(torch.clamp(_dist2, min=1e-6)) - dm) / ds
                _to2 = net.dt_head(torch.cat([trunk, _az2.unsqueeze(1), _ldc2.unsqueeze(1)], dim=1))
                _tmu2 = _to2[:, k_t:2 * k_t]
                if z is not None and hasattr(net, "z_dt"):
                    _tmu2 = _tmu2 + net.z_dt(net._z_core(z))
                if d_dt and st.s is not None:
                    _tmu2 = _tmu2 + (d_dt * st.s - d_dt_c)
                _dtv2 = sample_lognorm_mix(
                    (torch.log_softmax(_to2[:, :k_t], 1), _tmu2,
                     torch.nn.functional.softplus(_to2[:, 2 * k_t:3 * k_t]) + M.EPS),
                    dt_hi, tau, generator, reparam, innov=ci_t)
                dth = torch.where(_bad, _dth2, dth)
                dist = torch.where(_bad, _dist2, dist)
                dtv = torch.where(_bad, _dtv2, dtv)
                ld_c = (torch.log(torch.clamp(dist, min=1e-6)) - dm) / ds

        # ---- integrate ------------------------------------------------------------------- #
        st.h = st.h + dth * (math.pi / 180.0)
        st.x, st.y, st.h = advance_t(st.x, st.y, st.h, dist, cx, cy, R,
                                     eps=float(cfg_feat.get("eps", 1e-3)))
        if st.gru is not None:
            st.gru = net.state_cell(
                torch.cat([x_row, (dth / dth_scale).unsqueeze(1),
                           ld_c.unsqueeze(1),
                           ((torch.log(torch.clamp(dtv, min=1e-6)) - tm) / ts).unsqueeze(1)], 1),
                st.gru)
        if st.s is not None:
            # ONCE PER BOUT, and never per redraw -- the numpy generator's explicit contract
            # (fit_MLP_joint.py:854-857). The sqrt(1 - phi^2) keeps the stationary variance at 1, so
            # the loading alone sets the amplitude and the timescale alone sets the shape; without it
            # the two would be entangled and neither could be calibrated against its own moment.
            st.s = (d_phi * st.s
                    + math.sqrt(max(1.0 - d_phi * d_phi, 0.0))
                    * torch.randn(st.s.shape, device=dev, generator=generator))
        st.dth_hist = torch.cat([dth.unsqueeze(1), st.dth_hist[:, :K - 1]], dim=1)
        st.ld_hist = torch.cat([torch.log(torch.clamp(dist, min=1e-6)).unsqueeze(1),
                                st.ld_hist[:, :Kk - 1]], dim=1)
        st.lt_hist = torch.cat([torch.log(torch.clamp(dtv, min=1e-6)).unsqueeze(1),
                                st.lt_hist[:, :Kk - 1]], dim=1)
        # wrap on the record's OWN length, so a long rollout re-reads that fish's real programme
        st.t = (st.t + 1) % static["nlen"][rows]

        out["dtheta"].append(dth); out["dist"].append(dist); out["dt"].append(dtv)
        out["x"].append(st.x); out["y"].append(st.y)
        out["radial"].append(_hypot(st.x - cx, st.y - cy) / R)
    return {k: torch.stack(v, dim=1) for k, v in out.items()}


# =========================================================================== #
#  THE load-bearing check
# =========================================================================== #
def verify_against_numpy(model, records, n_bouts=4000, reps=3, seed=0, verbose=True):
    """Does the torch rollout produce the SAME OBJECT as the numpy generator?

    Distributional, not bit-for-bit: the two consume different RNG streams, so only the laws can be
    compared. Marginals, the across-window autocorrelation table and radial occupancy must agree
    within replicate spread. Compared with ``wall_reject`` removed on BOTH sides, because the
    rejection loop is deliberately not ported (see the module docstring).
    """
    import generative_metrics as GM
    net, cfg = model._model, dict(model._cfg_feat)
    cfg.pop("wall_reject", None)
    cfg["wall_mode"] = "reflect"
    # PIN THE LATENT AT ZERO on both sides. z is drawn once per trajectory and shifts all three
    # head levels, so with it live the replicate-to-replicate spread of absdth / dist / dt swamps
    # the porting bug this check exists to find -- it left a uniform ~6% shift sitting at 1.6-2.6
    # sigma, which is neither clearly a bug nor clearly noise. At sd = 0 both sides are the
    # population fish and the only variance left is emission sampling, which averages out over
    # thousands of bouts. (The latent path itself is exercised by `draw_latent`, and z=0 vs
    # z~prior is compared separately below.)
    if cfg.get("fish_re"):
        cfg["fish_re"] = dict(cfg["fish_re"], sd=np.zeros_like(np.asarray(cfg["fish_re"]["sd"],
                                                                          float)))
    dev = torch.device("cpu")
    net.to(dev).eval()

    def _stats(dth, dist, dtv, rad):
        return dict(absdth=float(np.mean(np.abs(dth))), absdth_sd=float(np.std(np.abs(dth))),
                    dist=float(np.median(dist)), dt=float(np.median(dtv)),
                    rad_mean=float(np.mean(rad)), occ=float(np.mean(rad > 0.85)),
                    acf_dt=float(GM.win_acf1(np.log(np.clip(dtv, 1e-6, None)))),
                    acf_dist=float(GM.win_acf1(np.log(np.clip(dist, 1e-6, None)))),
                    acf_rad=float(GM.win_acf1(rad)))

    npy = []
    for rep in range(reps):
        for i, r in enumerate(records):
            g = MX._gen_for(cfg)(net, r, cfg, np.random.default_rng([seed + rep, int(i)]),
                                 closed_loop=True, guardrail=True, stim_loop=True,
                                 eps=model.eps)
            rad = np.hypot(g["x_px"] - r.dish_cx, g["y_px"] - r.dish_cy) / r.dish_R
            npy.append(_stats(g["dtheta"], g["distance_px"], g["dt_s"], rad))

    static = build_static(records, cfg, dev)
    tor = []
    with torch.no_grad():
        for rep in range(reps):
            gen = torch.Generator(device="cpu").manual_seed(1000 + rep)
            st = RolloutState(static, torch.arange(len(records), device=dev), cfg, dev,
                              hidden=int(getattr(net, "state_hidden", 0)),
                              z=draw_latent(cfg, len(records), dev, gen))
            o = rollout(net, static, st, cfg, n_bouts, tau=1.0, reparam=False, generator=gen)
            for i in range(len(records)):
                tor.append(_stats(o["dtheta"][i].numpy(), o["dist"][i].numpy(),
                                  o["dt"][i].numpy(), o["radial"][i].numpy()))

    # PAIRED BY FISH. Both sides roll out the SAME records, and the between-fish spread of these
    # statistics is far larger than any porting bug worth catching -- unpaired, the SE is dominated
    # by which fish happened to be in the set and the test has almost no power (it waved through a
    # uniform ~6% shift on four statistics at once). Pairing cancels that, exactly as the
    # common-random-numbers pairing does everywhere else in this repo.
    nf, keys = len(records), list(npy[0])
    ok = True
    if verbose:
        print(f"{'stat':<12}{'numpy':>12}{'torch':>12}{'diff':>10}{' pairSE':>9}{'  t':>7}   verdict")
    for k in keys:
        A = np.array([d[k] for d in npy]).reshape(reps, nf).mean(0)      # per-fish, numpy
        B = np.array([d[k] for d in tor]).reshape(reps, nf).mean(0)      # per-fish, torch
        dif = B - A
        se = float(dif.std(ddof=1) / np.sqrt(nf)) + 1e-12
        tstat = float(dif.mean() / se)
        # tolerate a genuinely small effect even if precisely resolved: what matters is that the
        # laws agree to within a couple of percent, not that 50 rollouts cannot tell them apart.
        good = abs(tstat) <= 3.0 or abs(dif.mean()) <= 0.02 * abs(A.mean())
        ok &= good
        if verbose:
            print(f"{k:<12}{A.mean():>12.4f}{B.mean():>12.4f}{dif.mean():>10.4f}{se:>9.4f}"
                  f"{tstat:>7.1f}   {'ok' if good else 'MISMATCH'}")
    if verbose:
        print("\nTORCH ROLLOUT MATCHES THE NUMPY GENERATOR" if ok else
              "\nMISMATCH -- do NOT train on this; it is a different object")
    return ok


def main():
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bouts", default=str(Path(__file__).parent.parent / "data" /
                                           "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--fish", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    MX._numpy2_unpickle_compat()
    recs = MB.build_border_records(pd.read_csv(a.bouts))
    rl = [recs[f] for f in sorted(recs)[:a.fish]]
    m = MX.load_mix(a.model)
    sys.exit(0 if verify_against_numpy(m, rl, n_bouts=a.n, reps=a.reps) else 1)


if __name__ == "__main__":
    main()
