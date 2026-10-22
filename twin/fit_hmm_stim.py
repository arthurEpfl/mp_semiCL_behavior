"""
Stimulus-driven (input-driven) 3-state HMM for zebrafish bouts, in the style of
Dommanget-Kott et al. 2026 but EXTENDED so the fish reacts to the visual stimulus
(the moving dots of the semi-closed-loop assay -- see Perrino thesis 2026).

WHY / WHAT
----------
`scripts/fit_hmm.py` fit a spontaneous 3-state HMM (Forward / two turn states) to
the discretized bouts, but it ignored the stimulus. Here the reorientation of
each bout is biased toward the dots: the model is an INPUT-DRIVEN HMM (a.k.a.
IO-HMM / GLM-HMM) whose transition matrix and forward emission depend, at every
bout, on the EGOCENTRIC BEARING to the active dots

    phi_n = stim_angle_rad   (angle of the active-dot centroid minus the fish's
                              heading; + = dots on the fish's LEFT)

recomputed each bout. We turn phi into a bounded bias  b_n = sin(phi_n)  (present
& moving stimulus only, else 0). States are kept SIGN-NEUTRAL to avoid any
left/right label confusion:

    state 0 = Forward : Normal(dtheta ; mu_F , sigma),  mu_F = g * b_n   (deg)
    state 1 = turn-   : Gamma(-dtheta ; alpha, theta)  on dtheta < 0
    state 2 = turn+   : Gamma(+dtheta ; alpha, theta)  on dtheta > 0

Because dtheta and phi share the same arctan2 world frame, turning TOWARD the
dots always means sign(dtheta) = sign(phi); so a stimulus on the left (b>0) must
favour turn+ (dtheta>0) and vice-versa. The input-driven transition is

    logit p(j | i, b_n) = logT0[i,j] + kappa * c_j * b_n ,  c = (0, -1, +1)

with kappa >= 0 the learned "following gain". When b_n = 0 (no moving stimulus)
this reduces exactly to the spontaneous symmetric HMM.

OPTIMISER (how Baum-Welch is adapted for the stimulus parameters)
-----------------------------------------------------------------
Same EM skeleton as fit_hmm.py; only the transition/emission are now functions of
b_n, so the forward-backward uses a TIME-VARYING transition T^(n) = T(b_n):
  * E-step : scaled forward-backward with per-bout T^(n) and emission b(dtheta;b_n).
  * M-step : - turn Gamma (alpha, theta)  -> same weighted MLE as fit_hmm (b-free).
             - forward mean g & sigma      -> weighted least squares through origin
                                              (mu_F = g*b) + weighted variance.
             - transition T0 and kappa     -> NO closed form: maximise the expected
                                              complete-data log-likelihood (a small
                                              multinomial-logit / GLM-HMM M-step)
                                              over (T0-params, kappa) with L-BFGS,
                                              using the accumulated xi responsibilities
                                              and their inputs b_n.
Training: up to 300 EM iters, stop when the log-likelihood gain < 1e-6, a few
random restarts, keep the best.

Everything reuses the emission / Gamma-MLE building blocks of fit_hmm.py.

OUTPUTS
-------
* an .npz model (pi0, T0-params, kappa, g, sigma, alpha, theta),
* per-frame trajectory CSVs in the df_base schema (f_x/f_y/f_ori + d{k}_stim/x/y,
  pixels) that scripts/visualize.py renders straight into a comparison.mp4.

CLI
---
    uv run scripts/fit_hmm_stim.py fit  --bouts data/bouts_GsizeMotionSingle.csv \
                                        --output data/hmm_stim_GsizeMotionSingle.npz
    uv run scripts/fit_hmm_stim.py video --model  data/hmm_stim_GsizeMotionSingle.npz \
                                         --absdict notebooks/abs_dict.pkl \
                                         --roi data/data/ROIdef.csv \
                                         --output runs/hmm_stim/samples --n 4
    uv run scripts/visualize.py --samples runs/hmm_stim/samples \
                                --output runs/hmm_stim/comparison.mp4 --n 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from twin import fit_hmm as H                                   # emission / Gamma-MLE helpers
from twin.segment_bouts import (parse_stimulus, wrap_pi, STIM_NONE, fish_prefix, FPS,
                           PX_PER_MM)

STATE_NAMES = ["forward", "turn-", "turn+"]           # 0=F, 1=dtheta<0, 2=dtheta>0
TURN_SIGN = np.array([0.0, -1.0, 1.0])                # c_j in the transition logit
EPS = H.EPS
# PIXEL-NATIVE: bout distances/positions are pixels. Rich stimulus features that
# need a length scale use a SINGLE global nominal mm-per-px (matches how the dots
# were placed, MM_TO_PX=3.67), not a per-fish scale.
S_F = 1.0 / PX_PER_MM                                 # = 0.27248 mm/px (global)

# Dish drawn by scripts/visualize.py (its argparse defaults == configs/default.toml).
# The synthetic fish are CLAMPED to this same circle so they stay inside the drawn
# dish -- exactly like the NN sampler (src/sampling.py). abs_dict fish coordinates
# already live in this frame (all real fish stay within ~178 px of the centre).
DISH_X, DISH_Y, DISH_RADIUS = 180.0, 174.0, 183.5


# =========================================================================== #
#  Sequence loading (bouts + per-bout stimulus bearing)
# =========================================================================== #
def bias_from_phi(phi, present):
    """Bounded following signal b = sin(phi) where a moving stimulus is present,
    else 0. sin gives sign (turn toward the dots) and vanishes straight ahead."""
    b = np.where(present, np.sin(phi), 0.0)
    return np.nan_to_num(b, nan=0.0)


def load_sequences_with_stim(df, min_len=10, require_moving=True,
                             group_cols=("fish_id", "segment_id"),
                             order_col="bout_index"):
    """Per-trajectory (dtheta_deg, bias b) sequences.

    A bout drives following (b != 0) only where the stimulus bearing is defined
    and (if require_moving) the dots are moving; static / absent stimulus -> b=0,
    so those bouts constrain only the spontaneous part of the model.
    """
    seqs = []
    cols = list(group_cols)
    for _, g in df.sort_values(cols + [order_col]).groupby(cols, sort=False):
        dth = g["dtheta_deg"].to_numpy(float)
        phi = g["stim_angle_rad"].to_numpy(float)
        present = np.isfinite(phi)
        if require_moving and "moving" in g.columns:
            present = present & g["moving"].to_numpy(bool)
        b = bias_from_phi(phi, present)
        good = np.isfinite(dth)
        dth, b = dth[good], b[good]
        if len(dth) >= min_len:
            seqs.append((dth, b))
    return seqs


# =========================================================================== #
#  Input-driven transitions + emissions
# =========================================================================== #
def _L0_from_params(u):
    """Symmetric baseline transition LOGITS (3x3) from 4 free params.
    u = (u_F->turn, u_turn->F, u_turn->same, u_turn->flip)."""
    u1, u2, u3, u4 = u
    return np.array([[0.0, u1, u1],
                     [u2, u3, u4],
                     [u2, u4, u3]])


def trans_matrix(L0, kappa, b):
    """Row-stochastic input-driven transition for a single scalar bias b."""
    logits = L0 + kappa * b * TURN_SIGN[None, :]          # tilt turn columns
    logits = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits)


def init_dist(pi0, kappa, b):
    """Input-tilted initial distribution."""
    logits = np.log(pi0 + EPS) + kappa * b * TURN_SIGN
    logits -= logsumexp(logits)
    return np.exp(logits)


def emission_probs(dtheta, b, sigma, alpha, theta, g):
    """(n,3) emission PROBABILITIES. Forward mean shifts with the stimulus."""
    n = len(dtheta)
    B = np.empty((n, 3))
    mu = g * b                                            # forward mean (deg)
    B[:, 0] = np.exp(-0.5 * ((dtheta - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    B[:, 1] = np.exp(H._gamma_logpdf(-dtheta, alpha, theta))   # turn- (dtheta<0)
    B[:, 2] = np.exp(H._gamma_logpdf(+dtheta, alpha, theta))   # turn+ (dtheta>0)
    return B


# =========================================================================== #
#  Scaled forward-backward with time-varying transitions
# =========================================================================== #
def _io_forward_backward(pi_vec, T_seq, B):
    """Scaled (Rabiner) forward-backward, transition into bout t = T_seq[t].

    Returns loglik, gamma (n,3), xi (n-1,3,3), gamma0 (3,). xi[t] is the
    responsibility of the transition bout t -> t+1 (driven by input b_{t+1}).
    """
    n = B.shape[0]
    alpha = np.empty((n, 3)); c = np.empty(n)
    a = pi_vec * B[0]
    s = a.sum(); s = s if s > 0 else 1e-300
    c[0] = s; alpha[0] = a / s
    for t in range(1, n):
        a = (alpha[t - 1] @ T_seq[t]) * B[t]
        s = a.sum(); s = s if s > 0 else 1e-300
        c[t] = s; alpha[t] = a / s
    loglik = float(np.sum(np.log(c)))

    beta = np.empty((n, 3)); beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (T_seq[t + 1] @ (B[t + 1] * beta[t + 1])) / c[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    if n > 1:                                            # vectorised xi (no t-loop)
        M = alpha[:-1, :, None] * T_seq[1:] * (B[1:] * beta[1:])[:, None, :]
        xi = M / np.maximum(M.sum(axis=(1, 2), keepdims=True), 1e-300)
    else:
        xi = np.zeros((0, 3, 3))
    return loglik, gamma, xi, gamma[0]


# =========================================================================== #
#  M-step for the transition GLM (T0 params + kappa)
# =========================================================================== #
def _transition_negLL(params, Xi_from, Xi_full, Bvals):
    """Negative expected complete-data LL of the transition GLM.

    Xi_full : (N,3,3) summed transition responsibilities. Bvals : (N,) inputs.
    Xi_from : (N,3)   row marginals (from-state responsibilities).
    """
    u, kappa = params[:4], params[4]
    L0 = _L0_from_params(u)
    # per-sample logits (N,3,3): L0 + kappa*b*c on columns
    logits = L0[None] + (kappa * Bvals)[:, None, None] * TURN_SIGN[None, None, :]
    logZ = logsumexp(logits, axis=2, keepdims=True)      # (N,3,1)
    logP = logits - logZ                                 # (N,3,3)
    ll = np.sum(Xi_full * logP)
    return -ll


def _fit_transition(Xi_full, Bvals, x0, maxiter=40):
    Xi_from = Xi_full.sum(axis=2)
    res = minimize(_transition_negLL, x0, args=(Xi_from, Xi_full, Bvals),
                   method="L-BFGS-B", options=dict(maxiter=maxiter))
    return res.x


# input-bias bins: the transition GLM M-step sums xi into ~40 bins of b in [-1,1]
# so it optimises over (n_bins,3,3) instead of the full (N_transitions,3,3).
_B_EDGES = np.linspace(-1.0, 1.0, 42)
_B_CENTERS = 0.5 * (_B_EDGES[:-1] + _B_EDGES[1:])


# =========================================================================== #
#  Full Baum-Welch for the input-driven HMM
# =========================================================================== #
def _init_params(seqs, rng):
    alld = np.concatenate([s[0] for s in seqs])
    fwd = alld[np.abs(alld) < H.DTHETA0_DEG]
    turn = np.abs(alld[np.abs(alld) >= H.DTHETA0_DEG])
    sigma = float(max(np.std(fwd) if len(fwd) > 5 else 8.0, 1.0))
    if len(turn) > 5:
        m, v = turn.mean(), turn.var()
        alpha = float(max(m * m / max(v, EPS), H.ALPHA_MIN))
        theta = float(max(v / max(m, EPS), 1.0))
    else:
        alpha, theta = 2.0, 20.0
    # baseline transition logits: sticky-ish, random jitter
    u = np.array([rng.uniform(-0.5, 0.5),     # F->turn vs F->F
                  rng.uniform(-1.0, 0.0),     # turn->F
                  rng.uniform(0.0, 1.0),      # turn->same
                  rng.uniform(-0.5, 0.5)])    # turn->flip
    kappa = float(rng.uniform(0.5, 1.5))
    g = 0.0
    pi0 = np.array([0.2, 0.4, 0.4])
    return dict(pi0=pi0, u=u, kappa=kappa, sigma=sigma, alpha=alpha,
                theta=theta, g=g)


def _build_T_seq(b, L0, kappa):
    """Per-bout transition matrices for a whole sequence (T_seq[t] uses b[t])."""
    logits = L0[None] + (kappa * b)[:, None, None] * TURN_SIGN[None, None, :]
    logits = logits - logsumexp(logits, axis=2, keepdims=True)
    return np.exp(logits)


def fit_io_hmm(seqs, n_restarts=4, max_iter=300, tol=1e-6, seed=0, verbose=True):
    """Fit the stimulus-driven HMM by input-driven Baum-Welch (see module docstring)."""
    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + r)
        P = _init_params(seqs, rng)
        prev_ll = -np.inf
        history = []
        for it in range(max_iter):
            L0 = _L0_from_params(P["u"])
            total_ll = 0.0
            gammas, gamma0 = [], np.zeros(3)
            Xi_binned = np.zeros((len(_B_CENTERS), 3, 3))   # transitions by input bin
            # ---- E-step ----
            for dth, b in seqs:
                T_seq = _build_T_seq(b, L0, P["kappa"])
                pi_vec = init_dist(P["pi0"], P["kappa"], b[0])
                B = emission_probs(dth, b, P["sigma"], P["alpha"], P["theta"], P["g"])
                ll, gamma, xi, g0 = _io_forward_backward(pi_vec, T_seq, B)
                total_ll += ll
                gammas.append((gamma, dth, b))
                gamma0 += g0
                if len(xi):                             # bin xi by driving input b_{t+1}
                    idx = np.clip(np.digitize(b[1:], _B_EDGES) - 1, 0, len(_B_CENTERS) - 1)
                    np.add.at(Xi_binned, idx, xi)
            history.append(total_ll)

            # ---- M-step ----
            # (1) transition GLM: T0 params + kappa (optimised over input bins)
            x0 = np.concatenate([P["u"], [P["kappa"]]])
            xopt = _fit_transition(Xi_binned, _B_CENTERS, x0)
            P["u"], P["kappa"] = xopt[:4], float(xopt[4])

            # (2) forward mean g + sigma (weighted LS through origin)
            num_g = den_g = num_s = den_s = 0.0
            # (3) turn Gamma sufficient stats
            Wt = Sx = Slog = 0.0
            for gamma, dth, b in gammas:
                gF, gL, gR = gamma[:, 0], gamma[:, 1], gamma[:, 2]
                num_g += np.sum(gF * b * dth); den_g += np.sum(gF * b * b)
                neg, pos = dth < 0, dth > 0
                Wt += gL[neg].sum() + gR[pos].sum()
                Sx += np.sum(gL[neg] * (-dth[neg])) + np.sum(gR[pos] * dth[pos])
                Slog += (np.sum(gL[neg] * np.log(-dth[neg] + EPS))
                         + np.sum(gR[pos] * np.log(dth[pos] + EPS)))
            P["g"] = float(num_g / den_g) if den_g > EPS else 0.0
            for gamma, dth, b in gammas:
                gF = gamma[:, 0]
                num_s += np.sum(gF * (dth - P["g"] * b) ** 2); den_s += gF.sum()
            P["sigma"] = float(max(np.sqrt(num_s / max(den_s, EPS)), 0.5))
            P["alpha"], P["theta"] = H._weighted_gamma_mle(Wt, Sx, Slog)

            # (4) initial distribution (symmetric baseline from gamma0)
            pF = gamma0[0]; pT = (gamma0[1] + gamma0[2]) / 2
            pi = np.array([pF, pT, pT]); P["pi0"] = pi / pi.sum()

            if total_ll - prev_ll < tol and it > 0:
                break
            prev_ll = total_ll

        if verbose:
            print(f"  restart {r}: loglik={total_ll:.1f} iters={it+1} "
                  f"kappa={P['kappa']:.3f} g={P['g']:.2f} sigma={P['sigma']:.2f} "
                  f"alpha={P['alpha']:.2f} theta={P['theta']:.2f}")
        if best is None or total_ll > best["loglik"]:
            best = dict(**P, loglik=total_ll, history=history, n_iter=it + 1)
    return best


# --------------------------------------------------------------------------- #
def T0_matrix(model):
    """Baseline (b=0) transition matrix of a fitted model."""
    return trans_matrix(_L0_from_params(model["u"]), model["kappa"], 0.0)


def save_model(path, m):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pi0=m["pi0"], u=m["u"], kappa=m["kappa"], g=m["g"],
             sigma=m["sigma"], alpha=m["alpha"], theta=m["theta"],
             loglik=m.get("loglik", np.nan))


def load_model(path):
    d = np.load(path)
    return dict(pi0=d["pi0"], u=d["u"], kappa=float(d["kappa"]), g=float(d["g"]),
                sigma=float(d["sigma"]), alpha=float(d["alpha"]),
                theta=float(d["theta"]), loglik=float(d["loglik"]))


# =========================================================================== #
#  Bout-level sampling given a bias schedule (for notebook checks)
# =========================================================================== #
def sample_state(prev, b, model, rng):
    T = trans_matrix(_L0_from_params(model["u"]), model["kappa"], b)
    return int(rng.choice(3, p=T[prev]))


def sample_dtheta(state, b, model, rng):
    if state == 0:
        return rng.normal(model["g"] * b, model["sigma"])
    mag = rng.gamma(model["alpha"], model["theta"])
    return -mag if state == 1 else +mag


# =========================================================================== #
#  Per-frame trajectory roll-out for the comparison video
# =========================================================================== #
def _active_dot_centroid(row, n_stimuli=8):
    """Absolute (x,y) centroid of active dots at a frame, or None."""
    xs, ys = [], []
    for k in range(n_stimuli):
        if str(row.get(f"d{k}_stim")) not in STIM_NONE and np.isfinite(row.get(f"d{k}_x", np.nan)):
            xs.append(row[f"d{k}_x"]); ys.append(row[f"d{k}_y"])
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _stim01_frame(row, n_stimuli=8):
    """Convert a real frame's d{k}_stim string labels to 0/1 (+ keep x/y)."""
    out = {}
    for k in range(n_stimuli):
        active = str(row.get(f"d{k}_stim")) not in STIM_NONE and np.isfinite(row.get(f"d{k}_x", np.nan))
        out[f"d{k}_stim"] = int(active)
        out[f"d{k}_x"] = float(row.get(f"d{k}_x", np.nan))
        out[f"d{k}_y"] = float(row.get(f"d{k}_y", np.nan))
    return out


def rollout_segment(seg, prefix, model, px_per_mm, dt_pool_s, dist_pool_mm,
                    rng, fps=FPS, seed_seconds=3.0, bout_frames=3,
                    dish_center=None, dish_radius=None, n_stimuli=8):
    """Roll the input-driven HMM out over one real per-frame segment.

    Seeds from the real fish for `seed_seconds`, then generates bout-by-bout:
    the fish HOLDS still during the inter-bout interval and executes the sampled
    reorientation + step over `bout_frames` frames at each bout onset. The dots
    (and thus the stimulus bearing that drives the model) come from the real
    segment, so the synthetic fish reacts to exactly the stimuli the real one saw.

    Returns (window_df, real_df): both per-frame, df_base schema, dots as 0/1.
    """
    import pandas as pd
    seg = seg.reset_index(drop=True)
    n = len(seg)
    seed = int(seed_seconds * fps) + 1

    fx = seg[f"{prefix}_x"].to_numpy(float)
    fy = seg[f"{prefix}_y"].to_numpy(float)
    fo = seg[f"{prefix}_ori"].to_numpy(float)

    # synthetic pose arrays: seed = real fish
    sx = fx.copy(); sy = fy.copy(); so = fo.copy()
    x, y, ori = float(fx[seed - 1]), float(fy[seed - 1]), float(fo[seed - 1])

    if dish_center is None:
        dish_center = (DISH_X, DISH_Y)
    if dish_radius is None:
        dish_radius = DISH_RADIUS
    cx, cy = dish_center

    t = seed
    while t < n:
        # ---- decide the next bout at frame t ----
        row = seg.iloc[t]
        cen = _active_dot_centroid(row, n_stimuli)
        if cen is not None:
            phi = wrap_pi(np.arctan2(cen[1] - y, cen[0] - x) - ori)
            moving = parse_stimulus(row.get("stimulus")).get("moving", False)
            b = float(np.sin(phi)) if moving else 0.0
        else:
            b = 0.0
        state = sample_state(getattr(rollout_segment, "_prev", 0), b, model, rng)
        rollout_segment._prev = state
        dtheta = np.radians(sample_dtheta(state, b, model, rng))
        dist_px = float(rng.choice(dist_pool_mm))   # pool is already px (px_per_mm unused)
        dt_frames = max(1, int(round(float(rng.choice(dt_pool_s)) * fps)))

        # ---- execute: hold, then move over bout_frames ----
        new_ori = (ori + dtheta)
        nx = x + dist_px * np.cos(new_ori)
        ny = y + dist_px * np.sin(new_ori)
        if dish_radius is not None:
            dr = np.hypot(nx - cx, ny - cy)
            if dr > dish_radius:
                f = dish_radius / dr
                nx, ny = cx + (nx - cx) * f, cy + (ny - cy) * f
        move = min(bout_frames, dt_frames)
        for j in range(dt_frames):
            if t >= n:
                break
            if j < move:                       # linear ramp of the bout step
                f = (j + 1) / move
                sx[t] = x + (nx - x) * f
                sy[t] = y + (ny - y) * f
                so[t] = ori + wrap_pi(new_ori - ori) * f
            else:                              # wait at the new pose
                sx[t], sy[t], so[t] = nx, ny, new_ori
            t += 1
        x, y, ori = nx, ny, (new_ori % (2 * np.pi))
    rollout_segment._prev = 0

    # ---- assemble df_base-schema frames (post-seed only, like sample.py) ----
    def assemble(px, py, po):
        rows = []
        for i in range(seed, n):
            base = dict(f_x=float(px[i]), f_y=float(py[i]),
                        f_ori=float(po[i] % (2 * np.pi)))
            base.update(_stim01_frame(seg.iloc[i], n_stimuli))
            rows.append(base)
        return pd.DataFrame(rows)

    window = assemble(sx, sy, so)              # synthetic
    real = assemble(fx, fy, fo)                # matching real
    return window, real


# =========================================================================== #
#  RICH / MULTI-COVARIATE STIMULUS MODEL
# ---------------------------------------------------------------------------
#  Instead of the single scalar b = sin(phi) (which collapses the whole dot
#  group to its centroid bearing), the stimulus at each bout is described by a
#  small feature vector that keeps group size, spatial spread, distance and
#  speed. Per active dot k (at the bout onset), in the fish frame:
#      r_k   = distance (mm),  phi_k = egocentric bearing,
#      g(r)  = 1/(1 + r/R0)                      (retinal-size / proximity weight,
#                                                 present even for static dots),
#      vhat_k = dot speed / VREF                 (motion, 0 for static dots).
#  Aggregated (NOT centroided) into two channels:
#      presence:  L_prox = sum_k g_k sin phi_k ,  S_prox = sum_k g_k
#      motion  :  L_mot  = sum_k g_k vhat_k sin phi_k ,  S_mot = sum_k g_k vhat_k
#  (F_prox = sum_k g_k cos phi_k is a frontal/approach term, kept for optional
#  step-size scaling in generation.) Static dots keep full PRESENCE weight, so a
#  cluster of close non-moving dots is NOT zeroed out -- whether it matters is
#  decided by the fitted presence gain, not assumed.
#
#  Transition GLM (respecting left/right symmetry), with sign pushes
#  c=(0,-1,+1) (odd, which way) and e=(0,1,1) (even, whether to turn):
#      steer_n = beta_odd . u_odd_n   (u_odd = [L_prox, L_mot])
#      prop_n  = beta_even . u_even_n (u_even = [S_prox, S_mot])
#      logit p(j|i) = logT0[i,j] + steer_n c_j + prop_n e_j
#  so beta_odd steers toward near/fast/coherent groups and beta_even raises the
#  turn propensity for salient groups. The forward emission drifts by g*steer_n.
#  The simple model is the special case u_odd=[sin phi], u_even=[].
# =========================================================================== #
E_EVEN = np.array([0.0, 1.0, 1.0])           # even push (turn propensity)
R0_MM = 20.0                                 # proximity softening (mm)
VREF = 3.5                                   # speed normaliser (mm/s)


def _covariates(fx, fy, fori, dx, dy, act, dxp, dyp, actp, s_f):
    """Stimulus feature vector [L_prox, L_mot, S_prox, S_mot, F_prox] at one frame.

    fx,fy,fori: fish pose (px, px, rad). dx,dy,act: per-dot (len 8) positions (px)
    and active mask this frame; dxp,dyp,actp: previous frame (for the speed). s_f
    is the global nominal mm/px (distances -> nominal mm; angles are scale-free).
    """
    if not act.any():
        return np.zeros(5)
    ex, ey = dx - fx, dy - fy
    r_mm = s_f * np.hypot(ex, ey)
    phi = wrap_pi(np.arctan2(ey, ex) - fori)
    gk = 1.0 / (1.0 + r_mm / R0_MM)
    vv = FPS * s_f * np.hypot(dx - dxp, dy - dyp)      # per-dot speed (mm/s)
    vhat = np.where(act & actp, vv, 0.0) / VREF        # 0 if not tracked last frame
    s, c = np.sin(phi), np.cos(phi)
    m = act
    Lp = np.sum(gk[m] * s[m]);  Fp = np.sum(gk[m] * c[m]);  Sp = np.sum(gk[m])
    Lm = np.sum(gk[m] * vhat[m] * s[m]);  Sm = np.sum(gk[m] * vhat[m])
    return np.array([Lp, Lm, Sp, Sm, Fp])


def _fish_arrays(fish_df, n_stimuli=8):
    """Extract per-frame numpy arrays (fast lookups) from one fish's frame table."""
    prefix = fish_prefix(fish_df)
    FX = fish_df[f"{prefix}_x"].to_numpy(float)
    FY = fish_df[f"{prefix}_y"].to_numpy(float)
    FORI = fish_df[f"{prefix}_ori"].to_numpy(float)
    DX = np.stack([fish_df[f"d{k}_x"].to_numpy(float) for k in range(n_stimuli)], axis=1)
    DY = np.stack([fish_df[f"d{k}_y"].to_numpy(float) for k in range(n_stimuli)], axis=1)
    ACT = np.stack([
        (~fish_df[f"d{k}_stim"].astype(str).isin(STIM_NONE).to_numpy())
        & np.isfinite(DX[:, k]) for k in range(n_stimuli)], axis=1)
    return FX, FY, FORI, DX, DY, ACT


def build_rich_sequences(bouts, abs_dict, roi, min_len=10, scales=None):
    """Per-trajectory (dtheta, u_odd, u_even, u_front) sequences with rich stimulus
    features, plus the per-feature standardisation scales.

    Features come from the per-frame dot positions in `abs_dict` (indexed by the
    bout's `frame_onset`), so nothing about the group is collapsed to a centroid.

    `scales`: if given (from a training fold), the features are standardised with
    those RMS scales instead of scales computed here -- this keeps cross-validation
    leak-free (the test fold never contributes to its own standardisation).
    """
    order = bouts.sort_values(["fish_id", "segment_id", "bout_index"])
    cache = {}                                          # fish -> (arrays, frame->row)
    dth_g, U_g = [], []
    for (fid, sid), grp in order.groupby(["fish_id", "segment_id"], sort=False):
        key = f"f{fid}"
        if key not in abs_dict:
            continue
        if fid not in cache:
            fdf = abs_dict[key]
            arrs = _fish_arrays(fdf)
            fmap = {int(f): i for i, f in enumerate(fdf["frame"].to_numpy())}
            cache[fid] = (arrs, fmap)
        (FX, FY, FORI, DX, DY, ACT), fmap = cache[fid]
        s_f = S_F                                        # global nominal mm/px (roi unused)
        dth, U = [], []
        for fo, dtheta in zip(grp["frame_onset"].to_numpy(int),
                              grp["dtheta_deg"].to_numpy(float)):
            i = fmap.get(int(fo))
            if i is None:
                continue
            ip = i - 1 if i > 0 else i
            U.append(_covariates(FX[i], FY[i], FORI[i], DX[i], DY[i], ACT[i],
                                 DX[ip], DY[ip], ACT[ip], s_f))
            dth.append(dtheta)
        if len(dth) >= min_len:
            dth_g.append(np.array(dth)); U_g.append(np.array(U))
    if scales is None:
        allU = np.concatenate(U_g, axis=0)
        scales = np.sqrt(np.mean(allU ** 2, axis=0))    # scale-only (keeps 0 -> 0)
        scales[scales < 1e-9] = 1.0
    seqs = []
    for dth, U in zip(dth_g, U_g):
        Us = U / scales
        seqs.append((dth, Us[:, 0:2], Us[:, 2:4], Us[:, 4]))
    return seqs, scales


# ---- generic (steer, prop) transition / emission / sampling ---------------- #
def _trans_scalars(L0, steer, prop):
    logits = L0 + steer * TURN_SIGN[None, :] + prop * E_EVEN[None, :]
    logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits)


def _build_T_seq_g(steer, prop, L0):
    logits = (L0[None] + steer[:, None, None] * TURN_SIGN[None, None, :]
              + prop[:, None, None] * E_EVEN[None, None, :])
    logits -= logsumexp(logits, axis=2, keepdims=True)
    return np.exp(logits)


def _emission_probs_g(dtheta, steer, sigma, alpha, theta, g):
    n = len(dtheta); B = np.empty((n, 3)); mu = g * steer
    B[:, 0] = np.exp(-0.5 * ((dtheta - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    B[:, 1] = np.exp(H._gamma_logpdf(-dtheta, alpha, theta))
    B[:, 2] = np.exp(H._gamma_logpdf(+dtheta, alpha, theta))
    return B


def _transition_obj_grad(params, Xi, Uodd, Ueven, n_odd):
    """Negative expected transition log-likelihood AND its analytic gradient
    (softmax-GLM). Returning the gradient lets L-BFGS avoid the ~17x numerical
    gradient evaluations, which is the difference between seconds and minutes."""
    u = params[:4]; bo = params[4:4 + n_odd]; be = params[4 + n_odd:]
    L0 = _L0_from_params(u)
    steer = Uodd @ bo
    prop = Ueven @ be if Ueven.shape[1] else np.zeros(len(Uodd))
    eta = (L0[None] + steer[:, None, None] * TURN_SIGN[None, None, :]
           + prop[:, None, None] * E_EVEN[None, None, :])          # (N,3,3)
    logP = eta - logsumexp(eta, axis=2, keepdims=True)
    nll = -np.sum(Xi * logP)
    P = np.exp(logP)
    mrow = Xi.sum(axis=2, keepdims=True)                            # (N,3,1)
    delta = Xi - mrow * P                                           # d(logL)/d(eta)
    D = delta.sum(axis=0)                                           # (3,3)
    g_u = -np.array([D[0, 1] + D[0, 2], D[1, 0] + D[2, 0],
                     D[1, 1] + D[2, 2], D[1, 2] + D[2, 1]])
    C = (delta * TURN_SIGN[None, None, :]).sum(axis=(1, 2))         # (N,) push toward c
    g_bo = -(Uodd * C[:, None]).sum(axis=0)
    if Ueven.shape[1]:
        Ec = (delta * E_EVEN[None, None, :]).sum(axis=(1, 2))
        g_be = -(Ueven * Ec[:, None]).sum(axis=0)
    else:
        g_be = np.zeros(0)
    return nll, np.concatenate([g_u, g_bo, g_be])


def fit_io_hmm_rich(seqs, scales, n_restarts=3, max_iter=200, tol=1e-6, seed=0,
                    verbose=True):
    """Fit the rich (multi-covariate) input-driven HMM by input-driven Baum-Welch."""
    n_odd = seqs[0][1].shape[1]; n_even = seqs[0][2].shape[1]
    alld = np.concatenate([s[0] for s in seqs])
    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + r)
        fwd = alld[np.abs(alld) < H.DTHETA0_DEG]
        turn = np.abs(alld[np.abs(alld) >= H.DTHETA0_DEG])
        sigma = float(max(np.std(fwd) if len(fwd) > 5 else 8.0, 1.0))
        m_, v_ = (turn.mean(), turn.var()) if len(turn) > 5 else (40.0, 400.0)
        alpha = float(max(m_ * m_ / max(v_, EPS), H.ALPHA_MIN)); theta = float(max(v_ / max(m_, EPS), 1.0))
        u = np.array([rng.uniform(-.5, .5), rng.uniform(-1, 0),
                      rng.uniform(0, 1), rng.uniform(-.5, .5)])
        beta_odd = rng.uniform(-0.2, 0.2, n_odd); beta_even = np.zeros(n_even)
        g = 0.0; pi0 = np.array([0.2, 0.4, 0.4])
        prev_ll = -np.inf; history = []
        for it in range(max_iter):
            L0 = _L0_from_params(u)
            total_ll = 0.0; gammas = []; gamma0 = np.zeros(3)
            Xis, Uodds, Uevens = [], [], []
            for dth, Uo, Ue, _ in seqs:
                steer = Uo @ beta_odd
                prop = Ue @ beta_even if n_even else np.zeros(len(dth))
                Tseq = _build_T_seq_g(steer, prop, L0)
                B = _emission_probs_g(dth, steer, sigma, alpha, theta, g)
                ll, gamma, xi, g0 = _io_forward_backward(pi0, Tseq, B)
                total_ll += ll; gammas.append((gamma, dth, Uo)); gamma0 += g0
                if len(xi):
                    Xis.append(xi); Uodds.append(Uo[1:]); Uevens.append(Ue[1:])
            history.append(total_ll)
            # transition GLM
            Xi = np.concatenate(Xis, 0); Uo = np.concatenate(Uodds, 0)
            Ue = np.concatenate(Uevens, 0) if n_even else np.zeros((len(Uo), 0))
            x0 = np.concatenate([u, beta_odd, beta_even])
            res = minimize(_transition_obj_grad, x0, args=(Xi, Uo, Ue, n_odd),
                           jac=True, method="L-BFGS-B", options=dict(maxiter=25))
            u = res.x[:4]; beta_odd = res.x[4:4 + n_odd]; beta_even = res.x[4 + n_odd:]
            # emission M-step (recompute steer with new beta)
            num_g = den_g = num_s = den_s = 0.0; Wt = Sx = Slog = 0.0
            for gamma, dth, Uo_s in gammas:
                steer = Uo_s @ beta_odd; gF = gamma[:, 0]
                num_g += np.sum(gF * steer * dth); den_g += np.sum(gF * steer ** 2)
                neg, pos = dth < 0, dth > 0
                Wt += gamma[:, 1][neg].sum() + gamma[:, 2][pos].sum()
                Sx += np.sum(gamma[:, 1][neg] * (-dth[neg])) + np.sum(gamma[:, 2][pos] * dth[pos])
                Slog += (np.sum(gamma[:, 1][neg] * np.log(-dth[neg] + EPS))
                         + np.sum(gamma[:, 2][pos] * np.log(dth[pos] + EPS)))
            g = float(num_g / den_g) if den_g > EPS else 0.0
            for gamma, dth, Uo_s in gammas:
                steer = Uo_s @ beta_odd; gF = gamma[:, 0]
                num_s += np.sum(gF * (dth - g * steer) ** 2); den_s += gF.sum()
            sigma = float(max(np.sqrt(num_s / max(den_s, EPS)), 0.5))
            alpha, theta = H._weighted_gamma_mle(Wt, Sx, Slog)
            pF = gamma0[0]; pT = (gamma0[1] + gamma0[2]) / 2
            pi0 = np.array([pF, pT, pT]); pi0 = pi0 / pi0.sum()
            if total_ll - prev_ll < tol and it > 0:
                break
            prev_ll = total_ll
        if verbose:
            print(f"  restart {r}: loglik={total_ll:.1f} iters={it+1} "
                  f"beta_odd={np.round(beta_odd,3)} beta_even={np.round(beta_even,3)} "
                  f"g={g:.2f} sigma={sigma:.2f} alpha={alpha:.2f} theta={theta:.2f}")
        if best is None or total_ll > best["loglik"]:
            best = dict(mode="rich", pi0=pi0, u=u, beta_odd=beta_odd,
                        beta_even=beta_even, g=g, sigma=sigma, alpha=alpha,
                        theta=theta, scales=scales, loglik=total_ll,
                        history=history, n_iter=it + 1)
    return best


def save_model_rich(path, m):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mode="rich", pi0=m["pi0"], u=m["u"], beta_odd=m["beta_odd"],
             beta_even=m["beta_even"], g=m["g"], sigma=m["sigma"],
             alpha=m["alpha"], theta=m["theta"], scales=m["scales"],
             loglik=m.get("loglik", np.nan))


def load_model_any(path):
    """Load either a simple or a rich model (detected from the file)."""
    d = np.load(path, allow_pickle=True)
    if "beta_odd" in d:
        return dict(mode="rich", pi0=d["pi0"], u=d["u"], beta_odd=d["beta_odd"],
                    beta_even=d["beta_even"], g=float(d["g"]), sigma=float(d["sigma"]),
                    alpha=float(d["alpha"]), theta=float(d["theta"]),
                    scales=d["scales"], loglik=float(d["loglik"]))
    return {**load_model(path), "mode": "simple"}


def T0_matrix_g(m):
    """Baseline (no-stimulus) transition matrix, simple or rich model."""
    return _trans_scalars(_L0_from_params(m["u"]), 0.0, 0.0)


# ---- rich rollout for the comparison video --------------------------------- #
def rollout_segment_rich(seg, prefix, model, px_per_mm, dt_pool_s, dist_pool_mm,
                         rng, fps=FPS, seed_seconds=3.0, bout_frames=3,
                         dish_center=None, dish_radius=None, n_stimuli=8):
    """Like rollout_segment but drives the fish with the rich stimulus features
    (recomputed each bout from the synthetic pose + the real dots + dot speed)."""
    import pandas as pd
    seg = seg.reset_index(drop=True); n = len(seg)
    seed = int(seed_seconds * fps) + 1
    s_f = S_F                                   # global nominal mm/px (matches the fit)
    scales = model["scales"]; bo = model["beta_odd"]; be = model["beta_even"]
    L0 = _L0_from_params(model["u"])

    FXr, FYr, FORIr = (seg[f"{prefix}_x"].to_numpy(float),
                       seg[f"{prefix}_y"].to_numpy(float),
                       seg[f"{prefix}_ori"].to_numpy(float))
    DX = np.stack([seg[f"d{k}_x"].to_numpy(float) for k in range(n_stimuli)], axis=1)
    DY = np.stack([seg[f"d{k}_y"].to_numpy(float) for k in range(n_stimuli)], axis=1)
    ACT = np.stack([
        (~seg[f"d{k}_stim"].astype(str).isin(STIM_NONE).to_numpy()) & np.isfinite(DX[:, k])
        for k in range(n_stimuli)], axis=1)

    sx, sy, so = FXr.copy(), FYr.copy(), FORIr.copy()
    x, y, ori = float(FXr[seed - 1]), float(FYr[seed - 1]), float(FORIr[seed - 1])
    if dish_center is None:
        dish_center = (DISH_X, DISH_Y)
    if dish_radius is None:
        dish_radius = DISH_RADIUS
    cx, cy = dish_center
    state = 0
    t = seed
    while t < n:
        ip = t - 1 if t > 0 else t
        U = _covariates(x, y, ori, DX[t], DY[t], ACT[t], DX[ip], DY[ip], ACT[ip], s_f)
        Us = U / scales
        steer = float(Us[0:2] @ bo)
        prop = float(Us[2:4] @ be) if len(be) else 0.0
        T = _trans_scalars(L0, steer, prop)
        state = int(rng.choice(3, p=T[state]))
        if state == 0:
            dth = rng.normal(model["g"] * steer, model["sigma"])
        else:
            mag = rng.gamma(model["alpha"], model["theta"])
            dth = -mag if state == 1 else mag
        dtheta = np.radians(dth)
        dist_px = float(rng.choice(dist_pool_mm))   # pool is already px (px_per_mm unused)
        dt_frames = max(1, int(round(float(rng.choice(dt_pool_s)) * fps)))
        new_ori = ori + dtheta
        nx = x + dist_px * np.cos(new_ori); ny = y + dist_px * np.sin(new_ori)
        if dish_radius is not None:
            dr = np.hypot(nx - cx, ny - cy)
            if dr > dish_radius:
                f = dish_radius / dr
                nx, ny = cx + (nx - cx) * f, cy + (ny - cy) * f
        move = min(bout_frames, dt_frames)
        for j in range(dt_frames):
            if t >= n:
                break
            if j < move:
                f = (j + 1) / move
                sx[t] = x + (nx - x) * f; sy[t] = y + (ny - y) * f
                so[t] = ori + wrap_pi(new_ori - ori) * f
            else:
                sx[t], sy[t], so[t] = nx, ny, new_ori
            t += 1
        x, y, ori = nx, ny, new_ori % (2 * np.pi)

    def assemble(px, py, po):
        rows = []
        for i in range(seed, n):
            base = dict(f_x=float(px[i]), f_y=float(py[i]), f_ori=float(po[i] % (2 * np.pi)))
            base.update(_stim01_frame(seg.iloc[i], n_stimuli))
            rows.append(base)
        return pd.DataFrame(rows)
    return assemble(sx, sy, so), assemble(FXr, FYr, FORIr)


def generate_trials_rich(model, abs_dict, roi, bouts, out_dir, n=4,
                         trial_length_s=60.0, seed_seconds=3.0, seed=0, verbose=True):
    """Rich-model counterpart of generate_trials (writes trial_*.csv for visualize.py)."""
    import pandas as pd
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dt_pool = bouts["dt_s"].dropna().to_numpy(); dist_pool = bouts["distance_px"].dropna().to_numpy()
    rng = np.random.default_rng(seed); trial_frames = int(trial_length_s * FPS)
    made = 0
    for key in sorted(abs_dict, key=lambda k: int(k[1:])):
        if made >= n:
            break
        fid = int(key[1:]); fdf = abs_dict[key].reset_index(drop=True)
        prefix = fish_prefix(fdf)
        px_per_mm = PX_PER_MM             # nominal (unused for scaling; dist pool is px)
        lab = fdf["stimulus"].astype(str)
        sides = lab.map(lambda s: parse_stimulus(s).get("side"))
        mov = lab.map(lambda s: parse_stimulus(s).get("moving"))
        is_target = sides.isin(["left", "right"]) & mov.astype(bool)
        starts = np.where(is_target.to_numpy() & ~is_target.shift(fill_value=False).to_numpy())[0]
        # clamp to the SAME dish the video draws (matches the NN sampler).
        # Pipeline is pixel-native: the bout-distance pool is already in pixels.
        cx, cy, dish_r = DISH_X, DISH_Y, DISH_RADIUS
        for s0 in starts:
            beg = max(0, s0 - int(2 * FPS)); end = beg + trial_frames
            if end > len(fdf):
                continue
            seg = fdf.iloc[beg:end]
            if not np.isfinite(seg[f"{prefix}_x"].to_numpy(float)).all():
                continue
            window, real = rollout_segment_rich(
                seg, prefix, model, px_per_mm, dt_pool, dist_pool, rng,
                seed_seconds=seed_seconds, dish_center=(cx, cy), dish_radius=dish_r)
            window.to_csv(out_dir / f"trial_{made:03d}.csv", index=False)
            real.to_csv(out_dir / f"trial_{made:03d}_real.csv", index=False)
            if verbose:
                print(f"  fish {fid} frames {beg}:{end} -> trial_{made:03d} "
                      f"({len(window)} frames, {px_per_mm:.2f} px/mm)")
            made += 1
            break
    return made


# =========================================================================== #
#  CLI
# =========================================================================== #
def _cli_fit(args):
    import pandas as pd
    if args.simple:
        if args.output is None:
            args.output = Path("data/hmm_stim_GsizeMotionSingle.npz")
        return _cli_fit_simple(args)
    if args.output is None:
        args.output = Path("data/hmm_stim_rich_GsizeMotionSingle.npz")
    df = pd.read_csv(args.bouts)
    import pickle
    with open(args.absdict, "rb") as f:
        abs_dict = pickle.load(f)
    roi = pd.read_csv(args.roi, header=None,
                      names=["xoff", "yoff", "diameter", "x", "y", "radius"])
    seqs, scales = build_rich_sequences(df, abs_dict, roi, min_len=args.min_len)
    n_bouts = sum(len(s[0]) for s in seqs)
    print(f"loaded {len(seqs)} trajectories ({n_bouts} bouts); feature RMS scales "
          f"[Lprox,Lmot,Sprox,Smot,Fprox]={np.round(scales,3)}")
    m = fit_io_hmm_rich(seqs, scales, n_restarts=args.restarts,
                        max_iter=args.max_iter, seed=args.seed)
    print("\nbest RICH input-driven HMM:")
    print(f"  loglik            : {m['loglik']:.1f}  ({m['n_iter']} iters)")
    print(f"  beta_odd  [steer] : Lprox={m['beta_odd'][0]:+.3f}  Lmot={m['beta_odd'][1]:+.3f}")
    print(f"  beta_even [propen]: Sprox={m['beta_even'][0]:+.3f}  Smot={m['beta_even'][1]:+.3f}")
    print(f"  g (fwd drift)     : {m['g']:+.3f}")
    print(f"  sigma (fwd)       : {m['sigma']:.2f} deg")
    print(f"  turn Gamma        : alpha={m['alpha']:.2f} theta={m['theta']:.2f} deg")
    print(f"  T0 (rows F/turn-/turn+):\n{np.round(T0_matrix_g(m),3)}")
    print("  interpretation: beta>0 => stronger following. Lprox uses ALL active "
          "dots (static incl.); Lmot/Smot add the moving-dot bonus.")
    save_model_rich(args.output, m)
    print("saved model ->", args.output)


def _cli_fit_simple(args):
    import pandas as pd
    df = pd.read_csv(args.bouts)
    seqs = load_sequences_with_stim(df, min_len=args.min_len,
                                    require_moving=not args.include_static)
    n_bouts = sum(len(s[0]) for s in seqs)
    n_drive = sum(int(np.sum(s[1] != 0)) for s in seqs)
    print(f"loaded {len(seqs)} trajectories ({n_bouts} bouts, {n_drive} "
          f"stimulus-driven) min_len={args.min_len}")
    m = fit_io_hmm(seqs, n_restarts=args.restarts, max_iter=args.max_iter, seed=args.seed)
    print("\nbest input-driven HMM:")
    print(f"  loglik        : {m['loglik']:.1f}  ({m['n_iter']} iters)")
    print(f"  kappa (follow): {m['kappa']:.3f}   [>0 => turns toward the dots]")
    print(f"  g (fwd drift) : {m['g']:.3f} deg per unit sin(phi)")
    print(f"  sigma (fwd)   : {m['sigma']:.2f} deg")
    print(f"  turn Gamma    : alpha={m['alpha']:.2f} theta={m['theta']:.2f} deg "
          f"(mean turn {m['alpha']*m['theta']:.1f})")
    print(f"  T0 (b=0, rows F/turn-/turn+):\n{np.round(T0_matrix(m),3)}")
    save_model(args.output, m)
    print("saved model ->", args.output)


def generate_trials(model, abs_dict, roi, bouts, out_dir, n=4,
                    trial_length_s=60.0, seed_seconds=3.0, seed=0, verbose=True):
    """Write `n` df_base-schema trial CSVs (+ *_real) for scripts/visualize.py.

    Picks lateralized moving-dot blocks (one trial per fish) and rolls the
    input-driven HMM out over each. Returns the number of trials written.
    Callable from a notebook -- does not touch any original repo file.
    """
    import pandas as pd
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dt_pool = bouts["dt_s"].dropna().to_numpy()
    dist_pool = bouts["distance_px"].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    trial_frames = int(trial_length_s * FPS)

    made = 0
    for key in sorted(abs_dict, key=lambda k: int(k[1:])):
        if made >= n:
            break
        fid = int(key[1:])
        fdf = abs_dict[key].reset_index(drop=True)
        prefix = fish_prefix(fdf)
        px_per_mm = PX_PER_MM             # nominal (unused for scaling; dist pool is px)
        lab = fdf["stimulus"].astype(str)
        sides = lab.map(lambda s: parse_stimulus(s).get("side"))
        mov = lab.map(lambda s: parse_stimulus(s).get("moving"))
        is_target = sides.isin(["left", "right"]) & mov.astype(bool)
        starts = np.where(is_target.to_numpy() & ~is_target.shift(fill_value=False).to_numpy())[0]
        # clamp to the SAME dish the video draws (matches the NN sampler)
        cx, cy, dish_r = DISH_X, DISH_Y, DISH_RADIUS
        for s0 in starts:
            beg = max(0, s0 - int(2 * FPS))
            end = beg + trial_frames
            if end > len(fdf):
                continue
            seg = fdf.iloc[beg:end]
            if not np.isfinite(seg[f"{prefix}_x"].to_numpy(float)).all():
                continue
            window, real = rollout_segment(
                seg, prefix, model, px_per_mm, dt_pool, dist_pool, rng,
                seed_seconds=seed_seconds, dish_center=(cx, cy), dish_radius=dish_r)
            window.to_csv(out_dir / f"trial_{made:03d}.csv", index=False)
            real.to_csv(out_dir / f"trial_{made:03d}_real.csv", index=False)
            if verbose:
                print(f"  fish {fid} frames {beg}:{end} -> trial_{made:03d} "
                      f"({len(window)} frames, {px_per_mm:.2f} px/mm)")
            made += 1
            break                              # one trial per fish for variety
    return made


def _cli_video(args):
    import pickle
    import pandas as pd
    model = load_model_any(args.model)                 # auto-detects simple/rich
    with open(args.absdict, "rb") as f:
        abs_dict = pickle.load(f)
    roi = pd.read_csv(args.roi, header=None,
                      names=["xoff", "yoff", "diameter", "x", "y", "radius"])
    bouts = pd.read_csv(args.bouts)
    gen = generate_trials_rich if model.get("mode") == "rich" else generate_trials
    made = gen(model, abs_dict, roi, bouts, args.output, n=args.n,
               trial_length_s=args.trial_length_s,
               seed_seconds=args.seed_seconds, seed=args.seed)
    print(f"wrote {made} trial(s) to {args.output}  (model: {model.get('mode','simple')})")
    print(f"now:  uv run scripts/visualize.py --samples {args.output} "
          f"--output {args.output.parent}/comparison.mp4 --n {made}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="fit the input-driven HMM (rich by default)")
    pf.add_argument("--bouts", type=Path, default=Path("data/bouts_GsizeMotionSingle.csv"))
    pf.add_argument("--output", type=Path, default=None,
                    help="Model output (default: hmm_stim_rich_*.npz for rich, "
                         "hmm_stim_*.npz for --simple).")
    pf.add_argument("--absdict", type=Path, default=Path("notebooks/abs_dict.pkl"),
                    help="Per-frame dots (needed for the rich stimulus features).")
    pf.add_argument("--roi", type=Path, default=Path("data/data/ROIdef.csv"))
    pf.add_argument("--min-len", type=int, default=10)
    pf.add_argument("--restarts", type=int, default=3)
    pf.add_argument("--max-iter", type=int, default=200)
    pf.add_argument("--simple", action="store_true",
                    help="Old scalar sin(phi) model (centroid bearing only); "
                         "reads only the bouts CSV.")
    pf.add_argument("--include-static", action="store_true",
                    help="[simple only] let static-dot bouts also drive following.")
    pf.add_argument("--seed", type=int, default=0)
    pf.set_defaults(func=_cli_fit)

    pv = sub.add_parser("video", help="write df_base trial CSVs for visualize.py")
    pv.add_argument("--model", type=Path, default=Path("data/hmm_stim_GsizeMotionSingle.npz"))
    pv.add_argument("--absdict", type=Path, default=Path("notebooks/abs_dict.pkl"))
    pv.add_argument("--roi", type=Path, default=Path("data/data/ROIdef.csv"))
    pv.add_argument("--bouts", type=Path, default=Path("data/bouts_GsizeMotionSingle.csv"))
    pv.add_argument("--output", type=Path, default=Path("runs/hmm_stim/samples"))
    pv.add_argument("--n", type=int, default=4)
    pv.add_argument("--trial-length-s", type=float, default=60.0)
    pv.add_argument("--seed-seconds", type=float, default=3.0)
    pv.add_argument("--seed", type=int, default=0)
    pv.set_defaults(func=_cli_video)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
