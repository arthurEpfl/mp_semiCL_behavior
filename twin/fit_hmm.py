"""
Fit a 3-state Markov Chain and a 3-state Hidden Markov Model to the segmented
bouts (data/bouts_GsizeMotionSingle.csv), in the same style as Dommanget-Kott
et al. 2026 (PLoS Comput Biol; "Linking brain and behavior states ... using
hidden Markov models"), and generate synthetic swim trajectories from the HMM.

WHAT THE PAPER DOES (reproduced here, in Python)
------------------------------------------------
* Markov Chain (MC): threshold the reorientation angle dtheta at +/- dtheta0
  (=10 deg) into Forward / Left / Right, then estimate the 3x3 transition matrix
  by counting transitions (Eq 2). This mirrors utils/MarkovChains.py of the paper
  repo (threshold_classifier + bout_transitions + steady_state).

* Hidden Markov Model (HMM), model type `ZebrafishHMM_G3_Sym`:
    - 3 hidden states: 0=Forward (F), 1=Left (L), 2=Right (R).
    - Emissions E(dtheta | state):
        F : Normal(dtheta ; 0, sigma)                       (centered forward)
        L : Gamma(-dtheta ; alpha, theta)  on dtheta < 0    (turn magnitude)
        R : Gamma(+dtheta ; alpha, theta)  on dtheta > 0
      (single shared Gamma (alpha, theta) for both turns; alpha >= 1, theta > 0.)
      This is exactly the emission set stored in the paper's fitted models
      (`sforw` scalar + `turn` = [alpha, theta]); dtheta is in DEGREES.
    - Left-right SYMMETRY is enforced on the transition matrix and initial
      distribution: P(F->L)=P(F->R), P(L->L)=P(R->R), P(L->R)=P(R->L),
      P(L->F)=P(R->F), and one shared turn emission.
    - OPTIMISATION = Baum-Welch (Expectation-Maximisation). The paper's "custom
      Julia implementation" is the generic Baum-Welch of HiddenMarkovModels.jl
      (Dalle 2024) specialised to this emission type; there is no other optimizer.
      We implement Baum-Welch directly (log-space forward-backward + a symmetric,
      tied M-step with a weighted Gamma MLE for the turn emission).
      Training: up to 500 EM iterations, stop when the log-likelihood gain < 1e-6
      (paper's Sect 4.4); transition matrix initialised at random, respecting
      left-right symmetry; a few random restarts, keep the best log-likelihood.

* Generation (paper Sect 2.7.2): sample a state sequence from the HMM, sample a
  reorientation angle from each state's emission, resample the inter-bout interval
  dt and traveled distance d from the EMPIRICAL pools (ignoring bout-type
  dependence, as the paper does), then reconstruct x,y by cumulative sums (Eq 5).

INPUT  : data/bouts_GsizeMotionSingle.csv (from scripts/segment_bouts.py).
OUTPUT : an .npz HMM model + a few generated trajectories (.csv), plus a printed
         diagnostics summary.

Usage
-----
    uv run scripts/fit_hmm.py \
        --bouts   data/bouts_GsizeMotionSingle.csv \
        --output  data/hmm_GsizeMotionSingle.npz \
        --restarts 5 --min-len 10
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import digamma, gammaln, polygamma

# --------------------------- constants / conventions ------------------------ #
STATE_NAMES = ["forward", "left", "right"]   # 0=F, 1=L (dtheta<0), 2=R (dtheta>0)
DTHETA0_DEG = 10.0        # MC threshold (paper: +/- 10 deg)
LOG_FLOOR = -700.0        # log of ~1e-304, used where an emission has no support
ALPHA_MIN = 1.0           # paper constraint on the Gamma shape (alpha >= 1)
EPS = 1e-12




# =========================================================================== #
#  Markov Chain (thresholded) -- mirrors the paper's utils/MarkovChains.py
# =========================================================================== #
def threshold_classifier(seqs, dtheta0=DTHETA0_DEG):
    """Classify each dtheta into 0=Forward, 1=Left (dtheta<-dtheta0),
    2=Right (dtheta>+dtheta0). (Paper repo convention: left = negative.)"""
    out = []
    for s in seqs:
        b = np.zeros_like(s, dtype=np.int8)
        b[s < -dtheta0] = 1
        b[s > +dtheta0] = 2
        out.append(b)
    return out




def bout_transitions(bout_seqs, n_state=3):
    """Row-stochastic transition matrix by counting transitions (paper Eq 2)."""
    P = np.zeros((n_state, n_state))
    for b in bout_seqs:
        for i in range(len(b) - 1):
            P[b[i], b[i + 1]] += 1
    return P / P.sum(axis=1, keepdims=True)


def steady_state(T):
    """Stationary distribution of a row-stochastic transition matrix T."""
    n = T.shape[0]
    q = np.c_[T - np.eye(n), np.ones(n)]
    return np.linalg.solve(q @ q.T, np.ones(n))




def reconstruct_xy(angle_deg, dist_px, theta0=0.0):
    """Reconstruct a 2-D path from per-bout reorientation angles and distances (px)
    (paper Eq 5): theta_n = theta0 + cumsum(dtheta); x,y = cumsum(d cos/sin)."""
    theta = theta0 + np.cumsum(np.radians(angle_deg))
    x = np.cumsum(dist_px * np.cos(theta))
    y = np.cumsum(dist_px * np.sin(theta))
    return x, y, theta




