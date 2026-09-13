"""Rollout-aware fine-tuning: optimise the FREE-RUNNING window distribution, not a moment.

WHY THIS EXISTS
---------------
Six mechanisms now share one signature: each hit the moment it targeted and paid for it
somewhere the objective could not see. ``rad1`` is the cleanest case -- its spatial-gain search
looked at exactly one number, ``sd(mean r/R)``, hit it (radial importance 0.27 -> 0.10,
individuality 32% -> 44%), and cheerfully accepted 313% distance individuality and
``dist_w1`` 3.31 -> 6.25 as the price. c2st_traj 0.6821 -> 0.7883.

That is not a tuning failure. Every calibrator in this project optimises a MOMENT while the
metric measures a DISTRIBUTION, and a per-channel calibrator structurally cannot see the trade
it is making. One-step MLE has the matching blind spot on the other axis: it fits
``p(y_t | x_t)`` on real histories and never once looks at where the model's own 4700-bout
trajectories end up.

The objective here is the thing the metric actually measures:

    L(theta) = NLL_onestep(theta; real bouts)
             + beta * D( {window features: REAL} , {window features: ROLLOUT(theta)} )

with ``D`` the ENERGY DISTANCE on standardised window features,

    D^2(X, Y) = 2 E||x - y|| - E||x - x'|| - E||y - y'||

Energy distance rather than a learned discriminator: no inner-loop minimax, no bandwidth to
tune, it is a proper metric (zero iff the distributions match), and it is a DIFFERENT functional
from the RandomForest AUC -- so the C2ST survives as an external check instead of becoming the
training loss.

THE LOOP is an alternation, and that is the whole point. Each step rolls out fresh trajectories
with the CURRENT weights, scores them, takes one optimiser step, and throws them away: after the
step the policy has changed, so those rollouts are off-policy and reusing them would give a wrong
gradient within a few steps.

WHY WINDOWS
-----------
The rollout is NOT chopped -- the closed loop runs continuously for the whole record, so exposure
to long-horizon drift is unaffected. Only the FEATURE READ-OUT is windowed, for three reasons:

  1. Estimability. Whole-trajectory features give ONE point per fish: ~15 real and ~15 generated
     points in 39-d, and an energy distance between two 15-point clouds in 39 dimensions is noise.
     W = 60 gives ~71 windows per fish.
  2. It must match the metric. ``GM.window_features_traj`` uses W=60 with ``step = step or W``,
     i.e. NON-OVERLAPPING. Do not be tempted to overlap for more samples: correlated windows bias
     the within-set term ``E||x - x'||`` downward, making BOTH clouds look artificially tight, and
     it does not cancel between real and generated because their autocorrelation structure differs
     -- which is one of the things being fitted.
  3. The target only exists at window level. ``GM.dispersion_report``'s ``between_fish_share`` is
     var(per-fish MEANS of window features) / total window variance -- real 0.255, gen 0.078. With
     one point per fish the denominator does not exist.

WHY D NEEDS TWO TERMS
---------------------
A pooled window cloud's spread SUMS within-fish temporal variability and between-fish
variability. A model can match the pooled spread by making each fish wander more in r/R over time
instead of by making different fish differ -- the exact wrong answer to the radial deficit, and
invisible to a pooled energy distance. Since windows are labelled by fish, decompose:

    D_total = D(pooled window clouds) + gamma * D(per-fish mean vectors)

The second cloud is compared UNPAIRED even though the same fish ids are on both sides, and that
is deliberate: ``z`` is drawn from the prior each rollout, so generated trajectory i is a NEW
fish, not an attempt to reproduce real fish i. Pairing them would score the model on an identity
it was never given.

BUT ``gamma_bf`` DEFAULTS TO 0, and that is a measurement, not a retreat from the argument. With
15 fish the between-fish clouds hold 10 and 5 points in 39 dimensions, and the U-statistic
estimate of D^2 there came out NEGATIVE (-0.066 in-sample, -0.312 held-out) -- impossible for a
proper metric, so at that sample size the term is estimating noise. Weighted into the reward it
enters as a large per-fish CONSTANT spread over all ~71 of that fish's windows, which is the worst
possible shape for a spurious signal. Two further reasons it is not needed:
  * the POOLED cloud already contains between-fish variation, since windows from different fish
    are different points -- that was always the load-bearing half of the argument;
  * re-measured, rej's ``between_fish_share`` is 0.2151 against a real 0.2551, a 0.04 gap. The
    large deficit is pooled UNDER-dispersion (var_ratio dt 0.442, radial 0.563, distance 0.626).
So ``between_fish_share`` is DIAGNOSED via ``report_dispersion`` rather than optimised. Raise
``gamma_bf`` only with enough fish for the estimator to be positive.

CREDIT ASSIGNMENT
-----------------
``D`` is set-level, so the episode is the window, not the bout. The per-point witness

    h(x) = (2/m) sum_j ||x - y_j||  -  (2/n) sum_k ||x - x_k||

satisfies D^2 = mean_i h(x_i) + const(X) - const(Y), so ``-h`` is the reward whose expectation the
policy gradient descends. Each bout in a window inherits its window's advantage (episodic
REINFORCE, batch-mean baseline).

WHAT IS APPROXIMATE, STATED UP FRONT
------------------------------------
  * The reward is window-level, so all 60 bouts in a window share the credit.
  * REINFORCE is high-variance; the batch baseline helps but does not remove it. This is why the
    MLE anchor and the ``pg_weight=0`` control are load-bearing rather than optional.
  * With ``wall_reject`` on, the emitted triple is drawn from ``p(u) a(u) / Z`` rather than
    ``p(u)``, so ``grad log p`` is missing a ``-grad log Z`` term. MEASURED on the ``rej``
    checkpoint: the acceptance ramp is active on 0.94% of bouts (40 of 4272, 0 fallbacks), so the
    correction is confined to under 1% of the score function. ``finetune`` reports the realised
    redraw fraction every step so this stays an observed number rather than an assumption.

WHY NOT rollout_finetune.py
---------------------------
That file predates this model family and is incompatible in five separate ways (kin-lineage
generator, a ``dth_base`` key the joint generator did not return, policy evaluated at z=0 while
the rollout used z ~ prior, a shared body it would train while claiming to touch only turning,
and a single-channel 20-bin radial histogram with no population term). What is PORTED from it:
the MLE anchor, the density-ratio reward form, the ``pg_weight=0`` control, and the discipline of
rolling out through the existing verified generator instead of re-implementing feature assembly.

Usage
-----
    python scripts/rollout_aware.py --model ladder_psib_rej.pt --out rej_ra.pt
    python scripts/rollout_aware.py --model ladder_psib_rej.pt --out rej_ctl.pt --pg-weight 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP as M                     # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402
import fit_MLP_joint as JT              # noqa: E402
import fit_MLP_fishre as FR             # noqa: E402
import generative_metrics as GM         # noqa: E402

_ROOT = Path(__file__).parent.parent
_DEFAULT_BOUTS = _ROOT / "data" / "bouts_GsizeMotionSingle.csv"

W_DEFAULT = 60          # MUST equal GM.window_features_traj's W -- see module docstring
EPS = 1e-8


# =========================================================================== #
#  Window features -- ONE definition, shared with the metric
# =========================================================================== #
def real_window_feats(rec, W=W_DEFAULT):
    """Real-side window features for one record, built exactly as ``cross_validate_generative``
    builds them (same dot-distance source, same radial, same W)."""
    rad = GM._radial_of(rec, rec.x_px, rec.y_px)
    dd = getattr(rec, "dot_dist_px", None)
    has_pos = (getattr(rec, "dot_cx", None) is not None) and rad is not None
    return GM.window_features_traj(rec.dtheta, rec.symbol, rec.distance_px, rec.dt_s,
                                   dd if has_pos else None, W=W,
                                   radial=(rad if rad is not None else None))


def gen_window_feats(rec, g, W=W_DEFAULT):
    """Generated-side window features, mirroring the metric's construction bout for bout."""
    gx = np.asarray(g["x_px"], float); gy = np.asarray(g["y_px"], float)
    rad = GM._radial_of(rec, gx, gy)
    has_pos = getattr(rec, "dot_cx", None) is not None and rad is not None
    dd = (np.hypot(gx - np.asarray(rec.dot_cx, float),
                   gy - np.asarray(rec.dot_cy, float)) if has_pos else None)
    return GM.window_features_traj(g["dtheta"], g["symbol"], g["distance_px"], g["dt_s"],
                                   dd, W=W, radial=rad)


# =========================================================================== #
#  Energy distance and its per-point witness
# =========================================================================== #
def _pdist(A, B):
    """Euclidean distances (nA, nB). float64 throughout: the witness is a difference of two
    O(10) sums and float32 loses ~3 digits of the signal we are trying to read."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    d2 = (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * (A @ B.T))
    return np.sqrt(np.maximum(d2, 0.0))


def energy_witness(Xg, Xr, rr_mean=None):
    """(h (n_g,), D2 float) for the energy distance between generated ``Xg`` and real ``Xr``.

        h(x) = (2/m) sum_j ||x - y_j||  -  (2/n) sum_k ||x - x_k||
        D^2  = mean_i h(x_i) + mean_{i,k}||x_i - x_k|| - mean_{j,l}||y_j - y_l||

    ``rr_mean`` is the real-real term; it is constant in theta, so pass it in precomputed. It has
    no effect on the reward at all -- only on the reported D^2.
    """
    Xg = np.asarray(Xg, np.float64); Xr = np.asarray(Xr, np.float64)
    n, m = len(Xg), len(Xr)
    if n == 0 or m == 0:
        return np.zeros(n), float("nan")
    gr = _pdist(Xg, Xr).mean(1)                      # (n,) mean distance to the real cloud
    gg = _pdist(Xg, Xg)
    gg_row = gg.sum(1) / max(n - 1, 1)               # exclude the self-distance (which is 0)
    h = 2.0 * gr - 2.0 * gg_row
    if rr_mean is None:
        rr = _pdist(Xr, Xr)
        rr_mean = rr.sum() / max(m * (m - 1), 1)
    D2 = float(h.mean() + gg.sum() / max(n * (n - 1), 1) - rr_mean)
    return h, D2


def real_real_mean(Xr):
    """The constant real-real term of D^2 (precompute once; it never moves)."""
    Xr = np.asarray(Xr, np.float64)
    m = len(Xr)
    return float(_pdist(Xr, Xr).sum() / max(m * (m - 1), 1))


# =========================================================================== #
#  Log-probability of the actions the rollout actually took
# =========================================================================== #
def clip_count(g, cfg_feat):
    """How many emitted bouts sat on an emission clip.

    The generator clips the turn at +/-180 deg and the two kinematic channels at the REAL q99.9
    (``kin_clip``), so the law the rollout actually samples is the mixture with its tails folded
    onto atoms -- and ``mix_logdens`` / ``_lnk_logpdf`` are not that law. Clipped bouts are also
    the ones with the largest ``|grad log p|``, so a small count can still carry real weight.
    MEASURED: 1.03% of bouts on the shipped rej checkpoint (and the score-function identity still
    holds there at ratio 1.05), against 9.94% on a deliberately under-trained 4-epoch model. This
    is reported every step so it stays an observed number rather than an assumption.
    """
    dist_hi, dt_hi = cfg_feat["kin_clip"]
    return int(np.sum((np.abs(g["dth_base"]) >= 180.0 - 1e-9)
                      | (g["dist_base"] >= dist_hi - 1e-9) | (g["dist_base"] <= 1e-3 + 1e-12)
                      | (g["dt_base"] >= dt_hi - 1e-9) | (g["dt_base"] <= 1e-3 + 1e-12)))


def policy_params(net, g, cfg_feat, *, device=None, z_override=None):
    """((angle, dist, dt) params, (a_turn, a_dist, a_dt) actions) for the policy that ACTED.

    THE LOAD-BEARING FUNCTION. Three things have to line up with the generator, or the policy
    gradient is a valid-looking gradient for a different model:

      * the ACTIONS are ``dth_base`` / ``dist_base`` / ``dt_base``, not the sample dict's
        ``dtheta`` / ``distance_px`` / ``dt_s``. Those are post-containment-bias, post-vigor and
        post-``dt_snap`` respectively -- measured on the rej checkpoint, ``dt_s`` differs from the
        drawn value by exactly one camera frame (0.0167 s).
      * the LATENT is ``z_re``, the per-trajectory draw that acted. Evaluating at z=0 (which
        rollout_finetune.py did) scores the population fish, not the one that moved.
      * the CONDITIONERS are derived exactly as ``_emit`` derives them, and the two sources are
        NOT the same value. ``a_z`` uses the POST-bias turn -- the generator computes
        ``a_z = abs(dth_t) / dth_scale`` AFTER the containment shift, so the chained heads saw the
        shifted turn -- while ``ld_c`` uses the BASE distance. Getting this backwards is invisible
        on any model with ``bulk = 0`` (including rej, where the two coincide exactly) and breaks
        the score-function identity on every model that has a containment bias. It was a real bug
        here, caught by smoke_rollout_aware.py section [2], which runs at bulk=4.0 for this reason.
    """
    dev = device or torch.device("cpu")
    X = torch.tensor(np.asarray(g["X"], np.float32), device=dev)
    ya = torch.tensor(np.asarray(g["dth_base"], np.float32), device=dev)
    yd = torch.tensor(np.asarray(g["dist_base"], np.float32), device=dev)
    yt = torch.tensor(np.asarray(g["dt_base"], np.float32), device=dev)
    az, lc = JT._cond_cols(g["dtheta"], g["dist_base"],
                           cfg_feat["dth_scale"], cfg_feat["kin_scales"])
    az = torch.tensor(az, device=dev); lc = torch.tensor(lc, device=dev)
    m = int(getattr(net, "m", 0))
    zv = np.asarray(g["z_re"] if z_override is None else z_override, np.float32).reshape(1, -1)
    z = (torch.tensor(zv, device=dev).expand(X.shape[0], -1) if zv.size == m
         else torch.zeros((X.shape[0], m), device=dev))

    ang, dpar, tpar = net.joint_forward_re(X, az, lc, z)
    # The timing latent shifts the interval mixture's LOCATION per bout, exactly as it does in the
    # generator (``tmu[jt] + s_lam * s_state``) and in dt_level_logliks.
    s = np.asarray(g.get("s_state", np.zeros(X.shape[0])), np.float32)
    if getattr(net, "dt_latent", None) and np.any(s != 0.0):
        st = torch.tensor(s, device=dev).reshape(-1, 1)
        tpar = (tpar[0], tpar[1] + net.s_lam * st, tpar[2])
    return (ang, dpar, tpar), (ya, yd, yt)


def logp_actions(net, g, cfg_feat, *, device=None, z_override=None):
    """(la, ld, lt, mask): per-bout log-densities of the emitted triple, WITH grad."""
    (ang, dpar, tpar), (ya, yd, yt) = policy_params(net, g, cfg_feat, device=device,
                                                    z_override=z_override)
    la = MX.mix_logdens(ya, ang)
    ld = MX._lnk_logpdf(torch.clamp(yd, min=1e-6), *dpar)
    lt = MX._lnk_logpdf(torch.clamp(yt, min=1e-6), *tpar)
    mask = torch.isfinite(la) & torch.isfinite(ld) & torch.isfinite(lt)
    return la, ld, lt, mask


# =========================================================================== #
#  The MLE anchor
# =========================================================================== #
def _anchor_arrays(records, cfg_feat):
    """Real features/targets/conditioners under the SAME builder used at fit time."""
    X = np.concatenate([MB.build_feature_matrix(r, cfg_feat) for r in records], 0)
    ya = np.concatenate([np.asarray(r.dtheta, float) for r in records])
    yd = np.concatenate([np.asarray(r.distance_px, float) for r in records])
    yt = np.concatenate([np.asarray(r.dt_s, float) for r in records])
    az, lc = JT._cond_cols(ya, yd, cfg_feat["dth_scale"], cfg_feat["kin_scales"])
    return (np.asarray(X, np.float32), az, lc,
            np.asarray(ya, np.float32), np.asarray(yd, np.float32), np.asarray(yt, np.float32))


def anchor_nll(net, arrays, ix, nodes, log_pi, *, device=None):
    """Prior-marginal per-bout NLL on real bouts ``ix``, differentiable.

    WHICH marginal, and why. The fishre trainer anchors on the FITTED per-fish embedding and
    Metric A reports the sequentially FILTERED marginal; neither is usable here. The embedding is
    indexed by position in the original training-record list, which a checkpoint does not carry,
    so reconstructing it invites a silent off-by-one that would anchor each fish to another fish's
    individuality. The filtered marginal is a numpy cumsum inside ``@torch.no_grad``.

    So the anchor is the PRIOR marginal, ``logsumexp_q(log_pi_q + L_q(y))`` per bout, over the
    model's own GHQ grid. It is differentiable, it needs no fish identity, and it is the "new
    fish" likelihood -- which is the object the generator samples and the C2ST evaluates. It is
    NOT Metric A; Metric A is reported separately, before and after, as the hard regression guard.

    KNOWN CONSEQUENCE, and it confounds the ``pg_weight=0`` control. This is NOT the objective the
    model was fitted under (that used the per-fish embedding), so the control is not "more of the
    same training" -- it optimises a subtly different thing, one that pulls the heads toward the
    population mean. Measured over 100 control steps on 10 fish: in-sample D2 rose monotonically
    0.250 -> 0.414 while ``between_fish_share`` fell 0.1024 -> 0.0939 and ``var_ratio`` turn
    tightened 0.851 -> 0.795. That degradation is consistent with EITHER exposure bias (the
    one-step likelihood and the rollout distribution being different objectives) OR simple
    homogenisation by this anchor, and this design cannot separate them. Anyone reading the
    control as evidence for exposure bias must first rebuild the anchor on the fitted embeddings.
    """
    dev = device or torch.device("cpu")
    X, az, lc, ya, yd, yt = arrays
    Xt = torch.tensor(X[ix], device=dev)
    azb = torch.tensor(az[ix], device=dev); lcb = torch.tensor(lc[ix], device=dev)
    yab = torch.tensor(ya[ix], device=dev)
    ydb = torch.tensor(yd[ix], device=dev); ytb = torch.tensor(yt[ix], device=dev)
    n, Q = len(ix), nodes.shape[0]
    per_node = []
    for q in range(Q):
        zq = torch.tensor(nodes[q], device=dev).reshape(1, -1).expand(n, -1)
        la, ld, lt, md, mt = FR._chan_logliks(net, Xt, azb, lcb, yab, ydb, ytb, zq)
        zt = torch.zeros_like(ld)
        per_node.append(la + torch.where(md, ld, zt) + torch.where(mt, lt, zt))
    L = torch.stack(per_node, dim=1)                                     # (n, Q)
    lp = torch.tensor(np.asarray(log_pi, np.float32), device=dev).reshape(1, -1)
    return -(torch.logsumexp(L + lp, dim=1)).mean()


# =========================================================================== #
#  Trainable scope
# =========================================================================== #
def trainable(net, train_body=False):
    """Heads + latent loadings; the shared body is frozen by default.

    The body is shared by all three heads, so training it under a SINGLE-channel objective would
    silently move distance and dt -- the defect in rollout_finetune.py. Under the JOINT divergence
    that objection dissolves (it sees every channel), so ``train_body=True`` is a supported option
    rather than a bug; it is simply not the default.
    """
    groups = ["head", "dist_head", "dt_head", "z_turn", "z_dist", "z_dt", "z_bord"]
    if train_body:
        groups.append("body")
    ps, names = [], []
    for g in groups:
        mod = getattr(net, g, None)
        if mod is None:
            continue
        for p in mod.parameters():
            ps.append(p); names.append(g)
    return ps, sorted(set(names))


# =========================================================================== #
#  The fine-tune
# =========================================================================== #
def eval_D2(model, records, mu, sd, Rz, rr_pool, Rz_fish, rr_bf, *, seed=12345, W=W_DEFAULT):
    """(D2_pool, D2_bf) over ALL records at a FIXED seed -- the readable learning curve.

    The per-step D2 printed by ``finetune`` is measured on whichever 6 fish that step drew, so it
    moves with the batch as much as with the model and cannot be read as progress. This uses every
    record and a fixed rollout seed, so successive calls differ only by the weights.
    """
    Gz, owner = [], []
    for i, r in enumerate(records):
        g = model.sample(r, np.random.default_rng([seed, int(i)]))
        wf = gen_window_feats(r, g, W)
        Gz.append((wf - mu) / sd if len(wf) else np.zeros((0, Rz.shape[1])))
        owner.append(np.full(len(Gz[-1]), i))
    Gz = np.vstack(Gz); owner = np.concatenate(owner)
    _, D2p = energy_witness(Gz, Rz, rr_pool)
    Gf = np.stack([Gz[owner == i].mean(0) for i in range(len(records)) if (owner == i).any()])
    _, D2b = energy_witness(Gf, Rz_fish, rr_bf)
    return D2p, D2b


def finetune(model, records, *, steps=150, batch=6, lr=1e-5, beta=1.0, gamma_bf=0.0,
             anchor=1.0, pg_weight=1.0, anchor_batch=2048, W=W_DEFAULT, seed=0,
             train_body=False, verbose=True, log_every=10, eval_every=0, holdout=()):
    """Alternate {roll out -> score the window cloud -> one guarded step}. Returns the model.

    ``holdout`` -- records the fine-tune never touches. THE ONLY HONEST READ-OUT, and the reason
    it exists: the anchor trains on every bout of ``records``, INCLUDING the contiguous per-fish
    blocks that the original fit held out for early stopping. So a falling in-sample D2 is partly
    the model finally being allowed to fit those blocks, and cannot be read as the rollout
    distribution genuinely improving. Measured: rej stopped at epoch 41/50 on a real validation
    plateau (9.3897), yet in-sample D2 keeps dropping under pure MLE continuation.
    """
    net, cfg = model._model, model._cfg_feat
    if not hasattr(net, "joint_forward_re"):
        # Fail loudly rather than silently scoring a different chain. This objective needs the
        # latent-aware forward pass; on a plain MDNMixJoint there is no z to condition on and the
        # rollout's per-trajectory individuality would be unscoreable.
        raise TypeError(f"rollout_aware needs the fishre family (MDNMixJointRE); got "
                        f"{type(net).__name__}, which has no joint_forward_re.")
    dev = torch.device("cpu")
    net.to(dev)

    # ---- fixed real side -------------------------------------------------------------- #
    real_w = {i: real_window_feats(r, W) for i, r in enumerate(records)}
    R = np.vstack([real_w[i] for i in sorted(real_w)])
    # Standardise against REAL and FREEZE it. Re-estimating per batch would make the objective
    # move under the model, so a step could lower D by changing the yardstick.
    mu, sd = R.mean(0), R.std(0) + EPS
    Rz = (R - mu) / sd
    rr_pool = real_real_mean(Rz)
    Rz_fish = np.stack([((real_w[i] - mu) / sd).mean(0) for i in sorted(real_w)])
    rr_bf = real_real_mean(Rz_fish)
    if verbose:
        print(f"  real: {len(records)} fish, {len(R)} windows of {R.shape[1]} features "
              f"(W={W}, non-overlapping)")

    arrays = _anchor_arrays(records, cfg)
    nodes, log_pi = FR._ghq(int(net.m), model.re_sd, k=int(getattr(model, "re_ghq_k", 7)))
    if verbose:
        print(f"  anchor: prior marginal over {nodes.shape[0]} GHQ nodes, "
              f"{anchor_batch} bouts/step of {len(arrays[0])}")

    for p in net.parameters():
        p.requires_grad_(False)
    train_p, gnames = trainable(net, train_body)
    for p in train_p:
        p.requires_grad_(True)
    opt = torch.optim.Adam(train_p, lr=lr)
    if verbose:
        print(f"  training: {', '.join(gnames)}  "
              f"({sum(p.numel() for p in train_p)} params, body {'ON' if train_body else 'frozen'})")

    gen_kw = dict(closed_loop=model.closed_loop, guardrail=model.guardrail, eps=model.eps,
                  stim_loop=model.stim_loop, keep_features=True)
    gen_fn = MX._gen_for(cfg)
    rng = np.random.default_rng(seed)
    stats, hist, evals = {}, [], []
    t0 = time.time()

    ho = list(holdout)
    if ho:
        ho_w = {i: real_window_feats(r, W) for i, r in enumerate(ho)}
        Hz = np.vstack([ho_w[i] for i in sorted(ho_w)])
        # Standardise the held-out side with ITS OWN real statistics, so its D2 is on the same
        # footing as the in-sample one rather than inheriting the training fish's scale.
        hmu, hsd = Hz.mean(0), Hz.std(0) + EPS
        Hz = (Hz - hmu) / hsd
        h_rr = real_real_mean(Hz)
        Hz_fish = np.stack([((ho_w[i] - hmu) / hsd).mean(0) for i in sorted(ho_w)])
        h_rr_bf = real_real_mean(Hz_fish)

    def _fixed_eval(step):
        d = eval_D2(model, records, mu, sd, Rz, rr_pool, Rz_fish, rr_bf, W=W)
        row = dict(step=step, D2_pool=d[0], D2_bf=d[1])
        msg = (f"    [eval @ {step:3d}]  in-sample ({len(records)} fish):  "
               f"D2_pool {d[0]:8.4f}  D2_bf {d[1]:8.4f}")
        if ho:
            e = eval_D2(model, ho, hmu, hsd, Hz, h_rr, Hz_fish, h_rr_bf, W=W)
            row.update(D2_pool_ho=e[0], D2_bf_ho=e[1])
            msg += f"   |  HELD-OUT ({len(ho)}):  D2_pool {e[0]:8.4f}  D2_bf {e[1]:8.4f}"
        evals.append(row)
        if verbose:
            print(msg, flush=True)

    if eval_every:
        _fixed_eval(-1)
    for step in range(steps):
        picks = rng.choice(len(records), size=min(batch, len(records)), replace=False)
        rolls, n_bouts, n_redraw, n_clip = [], 0, 0, 0
        for i in picks:
            r = records[int(i)]
            g = gen_fn(net, r, cfg, np.random.default_rng([seed, step, int(i)]), **gen_kw)
            rolls.append((int(i), r, g))
            n_bouts += len(g["dtheta"]); n_redraw += int(g.get("n_redraw_bouts", 0))
            n_clip += int(clip_count(g, cfg))

        # ---- window cloud and rewards ------------------------------------------------- #
        Gz, owner, n_win = [], [], []
        for i, r, g in rolls:
            wf = gen_window_feats(r, g, W)
            wz = (wf - mu) / sd if len(wf) else np.zeros((0, R.shape[1]))
            Gz.append(wz); owner.append(np.full(len(wz), i)); n_win.append(len(wz))
        Gz = np.vstack(Gz) if len(Gz) else np.zeros((0, R.shape[1]))
        owner = np.concatenate(owner) if len(owner) else np.zeros(0, int)
        h_pool, D2_pool = energy_witness(Gz, Rz, rr_pool)

        # Between-fish term: the per-fish MEAN window vector. UNPAIRED against the real per-fish
        # means, because z is a fresh prior draw -- generated trajectory i is a new fish, not a
        # reconstruction of real fish i.
        Gz_fish = np.stack([Gz[owner == i].mean(0) if (owner == i).any()
                            else np.zeros(R.shape[1]) for i, _, _ in rolls])
        h_bf, D2_bf = energy_witness(Gz_fish, Rz_fish, rr_bf)

        # per-window reward = -(pooled witness) - gamma * (its fish's between-fish witness)
        #
        # NOTE ``beta`` IS A NO-OP AT THE DEFAULT gamma_bf=0, and that is worth stating rather than
        # leaving as a trap: the standardisation below divides out any common scalar, so only the
        # RATIO beta:gamma_bf changes the advantage. The magnitude knob is ``pg_weight``, which
        # multiplies the loss term after standardisation -- tune that (watch |g_pg|/|g_anchor|),
        # not beta.
        bf_by_owner = {i: float(h_bf[k]) for k, (i, _, _) in enumerate(rolls)}
        rew = -(beta * h_pool + gamma_bf * np.array([bf_by_owner[i] for i in owner]))
        adv_w = (rew - rew.mean()) / (rew.std() + 1e-8)

        # ---- policy gradient ---------------------------------------------------------- #
        off = np.cumsum([0] + n_win)
        pg_terms, nb = [], 0
        for k, (i, r, g) in enumerate(rolls):
            nw = n_win[k]
            if nw == 0:
                continue
            la, ld, lt, msk = logp_actions(net, g, cfg, device=dev)
            logp = la + ld + lt
            # bout -> window map. The tail bouts of an incomplete final window get NO reward and
            # are dropped rather than silently credited to the last window -- window_features_traj
            # discards them too, so crediting them would push on bouts the objective never saw.
            nb_used = nw * W
            advt = torch.tensor(np.repeat(adv_w[off[k]:off[k + 1]], W).astype(np.float32),
                                device=dev)
            m_used = msk[:nb_used]
            pg_terms.append(-(advt[m_used] * logp[:nb_used][m_used]).sum())
            nb += int(m_used.sum())
        pg = (torch.stack(pg_terms).sum() / max(nb, 1)) if pg_terms else torch.zeros((), device=dev)

        # ---- anchor + step ------------------------------------------------------------ #
        aix = rng.choice(len(arrays[0]), size=min(anchor_batch, len(arrays[0])), replace=False)
        nll = anchor_nll(net, arrays, aix, nodes, log_pi, device=dev)

        # GRADIENT BALANCE, measured at the logging steps only (two extra backward passes).
        # Without it, "the objective did nothing" and "the objective was outvoted by the anchor"
        # look identical from the outside -- and they need opposite fixes. A ratio far below 1
        # means beta is too small (or anchor too large) and the run is just extra MLE training,
        # which is precisely what the pg_weight=0 control is there to be confused with.
        gbal = float("nan")
        if verbose and (step % log_every == 0 or step == steps - 1):
            def _gn(term):
                opt.zero_grad(set_to_none=True)
                term.backward(retain_graph=True)
                n = float(torch.sqrt(sum((p.grad ** 2).sum() for p in train_p
                                         if p.grad is not None)))
                opt.zero_grad(set_to_none=True)
                return n
            if pg_weight:
                gbal = _gn(pg_weight * pg) / max(_gn(anchor * nll), 1e-12)

        loss = pg_weight * pg + anchor * nll
        JT.guarded_step(opt, loss, train_p, stats=stats)

        rec_row = dict(step=step, D2_pool=D2_pool, D2_bf=D2_bf, anchor_nll=float(nll),
                       pg=float(pg), redraw_frac=n_redraw / max(n_bouts, 1),
                       clip_frac=n_clip / max(n_bouts, 1), n_win=int(len(Gz)), grad_bal=gbal)
        hist.append(rec_row)
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:3d}  D2_pool {D2_pool:8.4f}  D2_bf {D2_bf:8.4f}  "
                  f"anchor {float(nll):8.4f}  pg {float(pg):+9.4f}  "
                  f"|g_pg|/|g_anchor| {gbal:7.3f}  "
                  f"redraw {rec_row['redraw_frac']:.4f}  clip {rec_row['clip_frac']:.4f}"
                  f"  [{time.time() - t0:5.1f}s]", flush=True)

        if eval_every and (step + 1) % eval_every == 0:
            _fixed_eval(step)

    JT._warn_unstable(stats, [h["anchor_nll"] for h in hist])
    for p in net.parameters():
        p.requires_grad_(True)
    model.history_ra = hist
    model.evals_ra = evals
    return model


# =========================================================================== #
#  Reporting
# =========================================================================== #
def report_dispersion(model, records, seed=0, W=W_DEFAULT):
    """``var_ratio`` + ``between_fish_share`` for the current weights.

    Read these TOGETHER at every checkpoint. ``between_fish_share`` rising (real 0.255, gen 0.078)
    while ``var_ratio`` stays near 1 is the intended route; the pooled spread widening while
    ``between_fish_share`` stalls means the model bought the match with within-fish wander, which
    is the failure the two-term objective exists to catch.
    """
    d = {}
    for i, r in enumerate(records):
        g = model.sample(r, np.random.default_rng([seed, int(i), 0]))
        d[i] = (real_window_feats(r, W), gen_window_feats(r, g, W))
    return GM.dispersion_report(d)


def metric_a(model, records):
    """Mean per-bout bits on the three channels (the hard regression guard)."""
    out = {}
    for key in ("logdens", "logdens_dist", "logdens_dt"):
        vals = []
        for r in records:
            v = np.asarray(model.score(r)[key], float)
            vals.append(v[np.isfinite(v)])
        v = np.concatenate(vals)
        out[key] = float(-np.mean(v) / np.log(2.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bouts", default=str(_DEFAULT_BOUTS))
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="weight on the POOLED term. Only its ratio to --gamma-bf matters (the "
                         "advantage is standardised, which divides out any common scalar), so at "
                         "the default gamma_bf=0 this does nothing. Use --pg-weight for magnitude")
    ap.add_argument("--gamma-bf", type=float, default=0.0,
                    help="weight on the between-fish term. 0 by default: at 15 fish its D^2 "
                         "estimate goes NEGATIVE, i.e. it is noise (see finetune's docstring)")
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--pg-weight", type=float, default=1.0,
                    help="0 = the CONTROL: anchor only, i.e. plain extra MLE training")
    ap.add_argument("--train-body", action="store_true")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="fixed-seed D2 over ALL fish, so the curve is readable; 0 = off")
    ap.add_argument("--max-fish", type=int, default=0, help="0 = all")
    ap.add_argument("--holdout", type=int, default=0,
                    help="reserve the LAST n fish; the fine-tune never sees them and their D2 is "
                         "reported separately. The only honest read-out (see finetune's docstring)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    MX._numpy2_unpickle_compat()
    recs = MB.build_border_records(pd.read_csv(args.bouts))
    rl = [recs[f] for f in sorted(recs)]
    if args.max_fish:
        rl = rl[:args.max_fish]
    ho = []
    if args.holdout:
        rl, ho = rl[:-args.holdout], rl[-args.holdout:]
        print(f"fine-tuning on {len(rl)} fish; holding out {len(ho)}")
    model = MX.load_mix(args.model)

    guard = ho or rl          # report Metric A where it is honest to read it
    print(f"BEFORE  metric A (bits/bout): {metric_a(model, guard)}")
    print(f"BEFORE  dispersion: {report_dispersion(model, guard)}")
    finetune(model, rl, steps=args.steps, batch=args.batch, lr=args.lr, beta=args.beta,
             gamma_bf=args.gamma_bf, anchor=args.anchor, pg_weight=args.pg_weight,
             train_body=args.train_body, seed=args.seed, eval_every=args.eval_every,
             holdout=ho)
    print(f"AFTER   metric A (bits/bout): {metric_a(model, guard)}")
    print(f"AFTER   dispersion: {report_dispersion(model, guard)}")
    MX.save_mix(model, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
