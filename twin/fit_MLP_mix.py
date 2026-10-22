"""The turn head's emission and the shared pieces of the model family.

The turn of a bout is a 5-component mixture:

    F = Normal(0, sigmaF)
    L = wL1 . -Gamma(aL1,tL1) + wL2 . -Gamma(aL2,tL2)      (short + long left turns)
    R = wR1 . +Gamma(aR1,tR1) + wR2 . +Gamma(aR2,tR2)      (short + long right turns)

Also here: the closed-loop stimulus row (``_dot_stim_row``), checkpoint saving / loading (the joint models), the
step and interval log-densities, and the drift's generation-gain search.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from twin import evaluate as E
from twin import fit_MLP as M
from twin import fit_MLP_border as MB

_LOG2PI = float(math.log(2.0 * math.pi))


# =========================================================================== #
#  5-component MDN head (feedforward body + 2 Gammas per side)
# =========================================================================== #
class MDNMix(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 14)     # 5 weight logits + sigmaF + 4 (alpha,theta) pairs

    @staticmethod
    def _angle_params(out):
        logw = F.log_softmax(out[:, :5], dim=1)
        sp = F.softplus
        return (logw,
                sp(out[:, 5]) + M.EPS,                               # sigmaF
                sp(out[:, 6]) + M.ALPHA_MIN, sp(out[:, 7]) + M.EPS,  # aL1, tL1
                sp(out[:, 8]) + M.ALPHA_MIN, sp(out[:, 9]) + M.EPS,  # aL2, tL2
                sp(out[:, 10]) + M.ALPHA_MIN, sp(out[:, 11]) + M.EPS,  # aR1, tR1
                sp(out[:, 12]) + M.ALPHA_MIN, sp(out[:, 13]) + M.EPS)  # aR2, tR2

    def forward(self, x):
        return self._angle_params(self.head(self.body(x)))


#  Scale floor for the mixture components. NOT cosmetic -- see mix_logdens.
#  Far below anything physical (the smallest fitted sigmaF on the shipped rej checkpoint is 10.6
#  DEGREES, and the smallest Gamma scale is 5.1) and far above the value at which the gradient
#  overflows (1e-9). That separation is the whole point: on any sane model this never binds, so it
#  cannot change a fitted result, while making the degenerate corner unreachable.
SCALE_FLOOR = 1e-3


def mix_logdens(dtheta, params):
    """Per-bout log predictive density of the 5-component mixture (nats over degrees).

    THE SCALE FLOOR IS LOAD-BEARING, and it fixes a bug that silently discarded most of training.
    An unused mixture component is unconstrained in scale -- once its weight goes to zero the
    likelihood no longer cares how narrow it is -- so sigma drifts down toward ``M.EPS`` (1e-12).
    At sigma <= 1e-9 the term ``dtheta**2 / (2 sigma**2)`` has a gradient that OVERFLOWS to inf,
    while the same component's logsumexp weight underflows to exactly 0. The backward pass then
    computes 0 * inf = NaN.

    The loss stays perfectly FINITE throughout, so nothing looks wrong: only the gradient is
    poisoned, and ``JT.guarded_step`` silently drops the update. Measured on the recurrent fit:
    **264 of 351 steps (75%) discarded, 0 non-finite losses** -- i.e. the model was training on a
    quarter of its budget while reporting a healthy curve. The failure rate rises with batch x
    chunk simply because more bouts per loss means more chances to hit a collapsed component.
    """
    logw, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = params
    sF = sF.clamp(min=SCALE_FLOOR)
    tL1 = tL1.clamp(min=SCALE_FLOOR); tL2 = tL2.clamp(min=SCALE_FLOOR)
    tR1 = tR1.clamp(min=SCALE_FLOOR); tR2 = tR2.clamp(min=SCALE_FLOOR)
    logNF = -0.5 * (_LOG2PI + 2.0 * torch.log(sF)) - dtheta ** 2 / (2.0 * sF ** 2)
    gL1 = M._gamma_logpdf_torch(-dtheta, aL1, tL1)
    gL2 = M._gamma_logpdf_torch(-dtheta, aL2, tL2)
    gR1 = M._gamma_logpdf_torch(+dtheta, aR1, tR1)
    gR2 = M._gamma_logpdf_torch(+dtheta, aR2, tR2)
    comp = torch.stack([logw[:, 0] + logNF, logw[:, 1] + gL1, logw[:, 2] + gL2,
                        logw[:, 3] + gR1, logw[:, 4] + gR2], dim=1)
    return torch.logsumexp(comp, dim=1)




def symbol_logq_mix(w, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2, dtheta0=None, eps=1e-300):
    """(n,3) predictive log-PMF over {F,L,R}: integrate each component over +/-dtheta0
    (generalises fit_MLP.symbol_logq_full to 2 Gammas per side)."""
    from scipy.special import erf, gammainc
    d0 = M.DTHETA0 if dtheta0 is None else dtheta0
    sqrt2 = math.sqrt(2.0)
    Phi_hi = 0.5 * (1.0 + erf((d0 / sF) / sqrt2))
    Phi_lo = 0.5 * (1.0 + erf((-d0 / sF) / sqrt2))
    nF, nL, nR = Phi_hi - Phi_lo, Phi_lo, 1.0 - Phi_hi
    GL1 = gammainc(aL1, d0 / tL1); GL2 = gammainc(aL2, d0 / tL2)
    GR1 = gammainc(aR1, d0 / tR1); GR2 = gammainc(aR2, d0 / tR2)
    qF = w[:, 0] * nF + w[:, 1] * GL1 + w[:, 2] * GL2 + w[:, 3] * GR1 + w[:, 4] * GR2
    qL = w[:, 0] * nL + w[:, 1] * (1.0 - GL1) + w[:, 2] * (1.0 - GL2)
    qR = w[:, 0] * nR + w[:, 3] * (1.0 - GR1) + w[:, 4] * (1.0 - GR2)
    q = np.maximum(np.stack([qF, qL, qR], axis=1), eps)
    return np.log(q / q.sum(axis=1, keepdims=True))




def _dot_stim_row(stim_mode, dot_cx_t, dot_cy_t, x, y, h, moving_t, n_dots_t=0.0, dish=None):
    """The stimulus row of one bout from the CURRENT generated pose (x, y, h) and the dot centroid shown
    (dot_cx_t, dot_cy_t): the fish reacts to its OWN bearing and range to the dots, through the same
    ``fit_MLP.shoal_channels`` as at train time (so the layout cannot drift); count and moving are block constants.
    ``dish`` is unused (kept in the signature)."""
    if stim_mode not in M.SHOAL_MODES:
        raise ValueError(f"unknown stim_mode {stim_mode!r}")
    kw = M.shoal_kw(stim_mode)
    if not np.isfinite(dot_cx_t):
        return M.shoal_channels([np.nan], [np.nan], [n_dots_t], [moving_t], **kw)[0]
    phi = MB._wrap_pi(np.arctan2(dot_cy_t - y, dot_cx_t - x) - h)
    r_px = float(np.hypot(dot_cx_t - x, dot_cy_t - y))
    return M.shoal_channels([phi], [r_px], [n_dots_t], [moving_t], **kw)[0]


# The codes whose row can be rebuilt from a generated pose (every code of the thesis).
ONLINE_STIM_MODES = M.SHOAL_MODES




# =========================================================================== #
#  Evaluation-pipeline adapter
# =========================================================================== #
class MLPMixDensity(E.GenerativeModel):
    """MDN-MLP with the 5-component (2-Gamma/side) emission; full mode only. Inputs
    (history + stim + optional border) come from fit_MLP_border.build_feature_matrix."""
    has_density = True
    has_symbol = True
    can_sample = True

    def __init__(self, K=8, stim_mode="simple", include_symbols=False, hidden=128,
                 epochs=60, lr=1e-3, patience=8, batch_size=256, seed=0, use_border=True,
                 border_mode='raw', wall_mode='clip',
                 use_adapt=False, adapt_lambda=0.6):
        self.mode = "full"
        self.name = "mlp_border_mix" if use_border else "mlp_mix"
        self.K = K
        self.stim_mode = stim_mode
        self.include_symbols = include_symbols
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.use_border = use_border
        self.border_mode = border_mode      # 'raw' (3 ch) or 'basis' (9 ch, distance-tuned)
        self.wall_mode = wall_mode          # 'clip' (legacy) or 'reflect' (billiard bounce)
        self.use_adapt = use_adapt          # sensory-adaptation covariate (leaky-integrated proximity)
        self.adapt_lambda = adapt_lambda


class MLPMixLoopDensity(MLPMixDensity):
    """Mix-emission MLP that free-runs CLOSED-LOOP with a hard containment guardrail.

    ``fit`` / ``score`` are inherited from ``MLPMixDensity`` UNCHANGED, so Metric A
    (bits/bout) is identical to ``mlp_border_mix``; only ``sample`` differs:

      closed_loop=True  -> honest self-contained simulator (the 'real' C2ST), 0% out-of-dish.
      closed_loop=False -> guardrail-only: wall covariate teacher-forced, so the turn
                           distribution (and thus C2ST) matches ``mlp_border_mix`` while the
                           rendered path is still contained.
    """

    def __init__(self, *args, closed_loop=True, guardrail=True, eps=1e-3, stim_loop=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"{self.name}_{'loop' if closed_loop else 'guard'}"
        self.closed_loop = closed_loop
        self.guardrail = guardrail
        self.eps = eps
        self.stim_loop = stim_loop     # honest: recompute the dot bearing from the generated pose


# =========================================================================== #
#  Checkpoint save / load  (train in Colab -> download -> render/score locally)
# =========================================================================== #
def save_mix(model, path):
    """Persist a FITTED ``MLPMixDensity`` / ``MLPMixLoopDensity`` so it can be reloaded and
    used (``.sample`` / ``.score``) WITHOUT retraining. Stores the MDN weights + the feature
    config + the dt/distance resample pools + the wrapper flags -- everything the numpy
    sampler and the density scorer need. Records are NOT stored (rebuild them locally with
    ``fit_MLP_border.build_border_records``)."""
    if not hasattr(model, "_model"):
        raise RuntimeError("model is not fitted; call .fit(...) before save_mix")
    # Different head layout -> own serialiser. isinstance, NOT a class-name string: a SUBCLASS of
    # the joint model (e.g. the per-fish random-effect variant) used to fall straight through this
    # test to the generic branch below, which writes a checkpoint with none of its extra parameters
    # -- and that checkpoint then LOADS WITHOUT ERROR as a plain joint model. Silent and plausible.
    from twin import fit_MLP_joint as _J
    if isinstance(model, _J.MLPMixJointDensity):
        try:
            from twin import fit_MLP_fishre as _FR
        except ImportError:
            _FR = None
        if _FR is not None and isinstance(model, _FR.MLPMixJointREDensity):
            return _FR.save_fishre(model, path)
        return _J.save_joint(model, path)
    raise TypeError(f"{type(model).__name__}: only the joint models are saved")


def _numpy2_unpickle_compat():
    """Let a numpy-1.x runtime unpickle arrays saved under numpy-2.x. numpy 2.0 renamed
    ``numpy.core`` -> ``numpy._core``, so a .pt saved in Colab (numpy 2) references
    ``numpy._core.*`` which a local numpy 1 lacks. Alias the missing modules to their
    ``numpy.core`` equivalents (no-op on numpy 2). Enables train-in-Colab -> load-locally."""
    import importlib
    import sys as _sys
    try:
        importlib.import_module("numpy._core")           # present on numpy >= 2
        return
    except ModuleNotFoundError:
        pass
    for name in ("numpy._core", "numpy._core.multiarray", "numpy._core.numeric",
                 "numpy._core.umath", "numpy._core._multiarray_umath"):
        try:
            _sys.modules[name] = importlib.import_module(name.replace("numpy._core", "numpy.core"))
        except Exception:
            pass


def load_mix(path, map_location="cpu", closed_loop=None, guardrail=None):
    """Reload a model saved by ``save_mix``, ready to ``.sample`` / ``.score`` (no fit).
    ``closed_loop`` / ``guardrail`` override the saved sampling flags if given (handy to
    reload a trained model as either the honest loop or the guardrail-only variant)."""
    import torch
    _numpy2_unpickle_compat()                            # Colab (numpy 2) -> local (numpy 1) compat
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                                   # older torch: no weights_only kwarg
        blob = torch.load(path, map_location=map_location)
    _cls = blob.get("cls")
    if _cls in ("MLPMixJointDensity", "MLPMixJointREDensity"):
        # joint checkpoints have chained heads (dist_head/dt_head hang off the SHARED body), and
        # the random-effect subclass adds an embedding plus four loading matrices on top, so the
        # MDNMix rebuild below cannot take either state_dict. Delegate, mirroring save_mix.
        # Doing this here means every downstream consumer accepts either checkpoint with no change of its own.
        # NOTE both branches must be kept in step with save_mix: a class that can be SAVED by a
        # specialised serialiser and not LOADED by one fails only at the call site, far away.
        if _cls == "MLPMixJointREDensity":
            from twin import fit_MLP_fishre as _J
            m = _J.load_fishre(path, map_location=map_location)
        else:
            from twin import fit_MLP_joint as _J
            m = _J.load_joint(path, map_location=map_location)
        if closed_loop is not None:
            m.closed_loop = bool(closed_loop)
        if guardrail is not None:
            m.guardrail = bool(guardrail)
        return m
    raise ValueError(f"{path}: checkpoint class {_cls!r} is not a joint model")




def _lnk_logpdf(x, logw, mu, sigma):                     # K-component log-normal; logw/mu/sigma (n,K)
    """Mixture log-normal for the kinematic channels.

    Why more than one component. The real upper tails are far heavier than a single log-normal
    can carry: dt q99/q99.9/max = 18.7/103.6/426.5 s against a 1.23 s median, and distance
    44/85/218 px against a 15 px median. A 1-comp distance head and a 2-comp dt head reproduce
    the BULK correctly (gen dt_median 1.292 vs real 1.281) but truncate the tail -- gen dt q99.9
    is 24.6 s vs real 103.6, and only 0.055% of generated bouts exceed 30 s vs 0.63% of real.
    That collapses every tail-sensitive window feature: across-window variance ratios of
    dt_std 0.04, dt_mean 0.08, dt_q90 0.12 (while dt_median, a bulk statistic, sits at 0.46).
    The clip is NOT the cause -- it binds on 0.001% of generated bouts.
    An extra component gives the head a slow/rare mode to place the long pauses in."""
    lx = torch.log(x).reshape(-1, 1)
    comp = logw + (-lx - torch.log(sigma) - 0.5 * _LOG2PI - (lx - mu) ** 2 / (2.0 * sigma ** 2))
    return torch.logsumexp(comp, dim=1)


def _dt_logpdf(x, logw, mu, sigma, hi=None):
    """The INTERVAL head's log-density: ``_lnk_logpdf`` when ``hi`` is None, else the same mixture
    TRUNCATED at ``hi`` seconds -- renormalised on (0, hi] by ``- log sum_k w_k Phi((log hi - mu_k)
    / sigma_k)``.

    WHY. The active-only data are cut at 10 s by construction (no bout on either side of a > 10 s
    gap), but the head was fitted as an untruncated log-normal mixture and its draws CLIPPED at the
    train q99.9. Measured on Model 8's control: q99 7.2 s vs 6.7 real, > 8 s 0.71% vs 0.44%, and
    0.43% of generated intervals piled at the clip; quantile-mapping the marginal onto the real one
    removes about half of the per-window interval-spread excess. With the cut in the likelihood the
    6-10 s density is fitted to the data's own frequency and the sampler draws inside the bound
    (`generate_mix_joint_loop`, `rollout_torch.sample_lognorm_mix`), so there is nothing to clip.
    Exact: the density integrates to 1 on (0, hi]. ``hi=None`` is bit-identical to before."""
    base = _lnk_logpdf(x, logw, mu, sigma)
    if hi is None:
        return base
    zc = (float(np.log(hi)) - mu) / sigma                                       # (n, K)
    return base - torch.logsumexp(logw + torch.special.log_ndtr(zc), dim=1)








def _gen_for(cfg_feat, records=None):
    """The free-run generator a calibrator rolls the model out with: the joint model's closed loop."""
    from twin import fit_MLP_joint as _J
    return _J.generate_mix_joint_loop


# =========================================================================== #
#  Drift: fit the spatial loading against the ACROSS-WINDOW autocorrelation
# =========================================================================== #
WIN_W = 60                       # must match GM.window_features_traj's W
# Must BRACKET the solution in both directions -- a gain pinned at a grid edge is a documented trap
# in this repo (the fishre core gain sat at 0.4, the edge, for two arms before it was noticed).
# Measured on rej with loadings turn 0.076 / dist 0.464 / dt 0.077: dt overshot to +0.437 and
# distance to +0.529 against targets +0.332 / +0.337, i.e. the answer is BELOW 1, and an old grid
# starting at 0.5 could not have expressed it. Hence the low end.
# 5.0 added 2026-09-15 for the TWO-STATE design: the interval chain's likelihood-fitted amplitude
# can sit well under what the window acf needs (a 4-epoch smoke wanted x3 at the old top), and a
# value pinned at the edge is not a fitted value. One extra point per sweep.
DRIFT_GAIN_GRID = (0.15, 0.3, 0.6, 1.0, 1.8, 3.0, 5.0)


def window_acf1(series, W=WIN_W):
    """Lag-1 autocorrelation of the WINDOW-MEAN series, computed within one trajectory.

    This is the statistic the whole drift mechanism exists to move, and it is deliberately a
    WITHIN-fish quantity: demeaning removes any static per-trajectory latent, so `fish_re`,
    `rad_indiv`, `rad_gate` and `re_bgain` all score exactly zero on it by construction. That is
    why six mechanisms in a row could not touch it.
    """
    ch = np.asarray(series, float)
    nw = len(ch) // int(W)
    if nw < 12:
        return np.nan
    w = ch[:nw * int(W)].reshape(nw, int(W))
    w = w[np.isfinite(w).all(1)]
    if len(w) < 12:
        return np.nan
    x = w.mean(1); x = x - x.mean()
    d = float((x * x).sum())
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else np.nan


def _radial_series(rec, x, y):
    return np.hypot(np.asarray(x, float) - rec.dish_cx,
                    np.asarray(y, float) - rec.dish_cy) / rec.dish_R


def _pav(y):
    """Pool-adjacent-violators: the non-decreasing sequence closest (least squares) to ``y``."""
    blocks = []                                          # [sum, count]
    for v in y:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for s, c in blocks:
        out += [s / c] * c
    return np.array(out)


def _pick_gain(grid, acf, target, plateau_tol=0.03):
    """The gain for ONE channel from its (gain, acf) curve.

    WHY NOT THE RAW ARGMIN. Measured on the slow2 checkpoints: below gain ~0.8 the step window-acf
    is steep and noisy between seeds (0.02-0.28 at the same gain), so the nearest grid point to
    the target on one noisy sample was 0.6 on two folds where the same weights, re-measured, gave
    0.05 -- while 1.0 gave a stable 0.37. And the interval chain of the two-state design SATURATES
    (a tau-35 state decorrelates within two 60-bout windows), so the argmin ran to the top of the
    grid on every fold and inflated the amplitude x5 for an acf it could not buy.

    So: (1) the curve is made monotone (PAV) -- the acf can only rise with the gain, a dip is
    noise; (2) the gain is INTERPOLATED in log-gain to hit the target exactly, so it is no longer
    tied to the grid; (3) if the smoothed curve never reaches the target, the search stops at the
    plateau (the smallest gain within ``plateau_tol`` of the smoothed maximum) and says so, rather
    than buying nothing at the grid edge. Returns (gain, status, smoothed curve)."""
    g = np.asarray(grid, float)
    s = _pav(acf)
    if target <= s[0]:
        return float(g[0]), "below", s
    p = int(np.argmax(s >= s[-1] - plateau_tol))         # where the plateau starts
    j = int(np.argmax(s >= target)) if target <= s[-1] else len(s)   # first point at/above target
    if j > p:
        # The target is unreachable, or reached only on the flat tail (a 0.01 acf gain for a x2
        # amplitude): stop where the curve stops rising.
        return float(g[p]), "saturated", s
    i = j - 1                                            # s[i] < target <= s[j], j >= 1
    if s[j] <= s[i]:
        return float(g[j]), "ok", s
    f = (target - s[i]) / (s[j] - s[i])
    return float(np.exp(np.log(g[i]) + f * (np.log(g[j]) - np.log(g[i])))), "ok", s


def _drift_acfs(model, recs, cf, seed, reps, gen_kw):
    """Mean generated across-window acf1 for (dt, distance, radial) over ``reps`` replicates."""
    o = {"dt": [], "dist": [], "rad": []}
    _gen = _gen_for(cf, recs)
    for r in recs:
        for rep in range(reps):
            g = _gen(model, r, cf,
                     np.random.default_rng([seed + rep, int(r.fish_id)]), **gen_kw)
            o["dt"].append(window_acf1(np.log(np.clip(g["dt_s"], 1e-6, None))))
            o["dist"].append(window_acf1(np.log(np.clip(g["distance_px"], 1e-6, None))))
            o["rad"].append(window_acf1(_radial_series(r, g["x_px"], g["y_px"])))
    return {k: float(np.nanmean(v)) for k, v in o.items()}


def _search_drift_gain(model, records, cfg_feat, seed=0, max_fish=8, reps=2,
                       gain_grid=None, **gen_kw):
    """Fit the two KINEMATIC gains against the measured across-window acf of dt and distance.

    WHY A SEARCH AND NOT A GRADIENT. The likelihood-fitted loadings optimise one-step predictive
    density, which does not identify an across-window autocorrelation -- the same mis-identification
    that put dt_latent's tau at 107-152 against a much shorter real timescale. So the amplitudes are
    fitted by likelihood and then put on target against a ROLLOUT statistic.

    RADIAL IS REPORTED, NOT OPTIMISED. The spatial channel (`bord`) was built and refuted; see
    DRIFT_DEFAULT for the grid that shows it either fails to move the acf or destroys occupancy.

    WHY THIS OBJECTIVE AND NOT THE PREVIOUS ONE. The energy-distance policy gradient in
    ``rollout_aware.py`` diluted a 4-of-39-feature defect across a 39-dimensional omnibus and then
    estimated it with a score-function gradient; its between-arm differences came out SMALLER than
    its own within-arm noise. Here TWO scalars are fitted against THREE statistics the model misses
    by >6 sigma each (measured, 15 fish x 3 replicates: dt +0.332 vs -0.001, distance +0.337 vs
    -0.002, radial +0.377 vs +0.043). Same well-posed shape as ``_search_wall_reject``.

    ``reps`` matters and is not padding: the radial acf on a single replicate of 6 fish swung
    -0.035 to +0.117 across seeds, which is larger than the effect being fitted. Replicating is
    what makes this a measurement rather than noise-fitting.
    """
    # resolved at CALL time (not bound as a default) so a caller can swap the grid
    gain_grid = tuple(DRIFT_GAIN_GRID if gain_grid is None else gain_grid)
    tg = {"dt": [], "dist": [], "rad": []}
    for r in records:
        tg["dt"].append(window_acf1(np.log(np.clip(r.dt_s, 1e-6, None))))
        tg["dist"].append(window_acf1(np.log(np.clip(r.distance_px, 1e-6, None))))
        tg["rad"].append(window_acf1(_radial_series(r, r.x_px, r.y_px)))
    tgt = {k: float(np.nanmean(v)) for k, v in tg.items()}
    recs = list(records)[:max_fish]
    base = dict(cfg_feat.get("drift") or {})
    if not base:
        raise ValueError("_search_drift_gain needs cfg_feat['drift'] (publish it after the fit)")
    d_dist, d_dt = float(base.get("dist", 0.0)), float(base.get("dt", 0.0))
    # TWO-STATE design (drift with dt_off + dt_latent): the interval chain is dt_latent, so the dt
    # sweep scales ITS amplitude (lam, with the same log-normal re-centring) and the slow chain's
    # dt loading -- exactly 0 -- gets no gain.
    dtl = dict(cfg_feat.get("dt_latent") or {})
    dt_via_latent = bool(dtl) and abs(d_dt) < 1e-12
    s_lam = float(dtl.get("lam", 0.0)) if dt_via_latent else 0.0

    def _cfg(gd, gt):
        # Re-centring so a scaled loading cannot inflate the kinematic marginals -- see the
        # generator's drift block. exp((g^2-1) d^2 / 2) is the exact inflation for a unit-variance
        # zero-mean state, so subtracting its log from the location cancels it.
        cf = dict(cfg_feat)
        if dt_via_latent:
            cf["drift"] = dict(base, g_dist=float(gd), g_dt=1.0,
                               dist_c=0.5 * (gd * gd - 1.0) * d_dist * d_dist, dt_c=0.0)
            cf["dt_latent"] = dict(dtl, g_lam=float(gt),
                                   lam_c=0.5 * (gt * gt - 1.0) * s_lam * s_lam)
        else:
            cf["drift"] = dict(base, g_dist=float(gd), g_dt=float(gt),
                               dist_c=0.5 * (gd * gd - 1.0) * d_dist * d_dist,
                               dt_c=0.5 * (gt * gt - 1.0) * d_dt * d_dt)
        return cf

    # TWO INDEPENDENT 1-D searches, not a product grid: each kinematic channel's across-window acf
    # responds to its OWN loading, and a single shared gain provably cannot serve both (measured --
    # the gain that put dt on target left distance at +0.118 against +0.253). The pick on each
    # curve is `_pick_gain` (monotone-smoothed, interpolated, plateau-stopped), not the raw argmin.
    curve_d = [(float(g), _drift_acfs(model, recs, _cfg(g, 1.0), seed, reps, gen_kw))
               for g in gain_grid]
    gd, st_d, sm_d = _pick_gain(gain_grid, [c["dist"] for _, c in curve_d], tgt["dist"])
    curve_t = [(float(g), _drift_acfs(model, recs, _cfg(gd, g), seed + 50, reps, gen_kw))
               for g in gain_grid]
    gt, st_t, sm_t = _pick_gain(gain_grid, [c["dt"] for _, c in curve_t], tgt["dt"])
    # ONE re-measurement at the chosen pair, on its own stream -- what the model will actually do
    got = _drift_acfs(model, recs, _cfg(gd, gt), seed + 100, reps, gen_kw)
    rep = dict(target=tgt, got=got, g_dist=gd, g_dt=gt,
               curve_dist=curve_d, curve_dt=curve_t, dt_via_latent=dt_via_latent,
               pick=dict(dist=st_d, dt=st_t),
               smoothed_dist=[float(v) for v in sm_d], smoothed_dt=[float(v) for v in sm_t],
               reps=int(reps), grid=list(map(float, gain_grid)))
    for k, st_, sm in (("dist", st_d, sm_d), ("dt", st_t, sm_t)):
        if st_ == "saturated":
            # The timescale, not the grid, is the limit: amplitude cannot buy more lag-1 window acf
            # once the state decorrelates within ~2 windows. Stopping at the plateau keeps the
            # marginal intact; the honest reading is "this tau reaches X, the real value is Y".
            print(f"    NOTE: {k} window-acf SATURATES at {sm[-1]:+.3f} (real {tgt[k]:+.3f}) -- gain "
                  f"stopped at the plateau ({gd if k == 'dist' else gt:.2f}); a longer tau is "
                  f"needed for the rest, not a larger amplitude")
        elif st_ == "below":
            print(f"    NOTE: {k} window-acf is above target even at the smallest gain "
                  f"({sm[0]:+.3f} vs {tgt[k]:+.3f}); gain floored at {gain_grid[0]}")
    miss = [k for k in ("dt", "dist") if abs(got[k] - tgt[k]) > 0.5 * abs(tgt[k])
            and rep["pick"][k] == "ok"]
    if miss:
        # Loud, because "the search mis-picked" and "the mechanism does not work" look identical
        # in the results table, and they need opposite responses.
        print(f"    WARNING: drift re-measured off target on {miss} "
              f"(got { {k: round(got[k], 3) for k in miss} } vs real "
              f"{ {k: round(tgt[k], 3) for k in miss} }) although the smoothed curve crossed it; "
              f"the curve is noisy at this budget -- raise reps before reading this arm.")
    # radial is REPORTED, never optimised -- see DRIFT_DEFAULT for the measurements that refuted
    # the spatial channel. Printing it keeps the open gap visible next to the two that closed.
    return gd, gt, rep






