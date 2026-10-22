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






# input-bias bins: the transition GLM M-step sums xi into ~40 bins of b in [-1,1]
# so it optimises over (n_bins,3,3) instead of the full (N_transitions,3,3).
_B_EDGES = np.linspace(-1.0, 1.0, 42)
_B_CENTERS = 0.5 * (_B_EDGES[:-1] + _B_EDGES[1:])






def _stim01_frame(row, n_stimuli=8):
    """Convert a real frame's d{k}_stim string labels to 0/1 (+ keep x/y)."""
    out = {}
    for k in range(n_stimuli):
        active = str(row.get(f"d{k}_stim")) not in STIM_NONE and np.isfinite(row.get(f"d{k}_x", np.nan))
        out[f"d{k}_stim"] = int(active)
        out[f"d{k}_x"] = float(row.get(f"d{k}_x", np.nan))
        out[f"d{k}_y"] = float(row.get(f"d{k}_y", np.nan))
    return out




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




