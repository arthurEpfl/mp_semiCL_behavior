"""
Mixture Density Network (MDN / MLP) for zebrafish bout reorientation angles.

WHAT
----
A feed-forward MLP conditions on an autoregressive history window of previous
reorientation angles (dtheta) plus a stimulus feature, and outputs a 3-component
mixture over the SAME emission family as scripts/fit_hmm.py:

    F (forward) = Normal(0, sigma)
    L (left)    = Gamma(-dtheta; alpha_L, theta_L)   on dtheta < 0
    R (right)   = Gamma(+dtheta; alpha_R, theta_R)   on dtheta > 0
    p(dtheta) = wF*N + wL*Gamma(-dtheta) + wR*Gamma(+dtheta),   sum(w) = 1

Two variants (one `MDN` class, `mode`-parameterised):
  * 'basic' : MLP -> 3 mixture-weight logits; sigma/alpha/theta are GLOBAL
              learnable scalars, gamma SYMMETRIC (shared L/R). Homoscedastic.
  * 'full'  : MLP -> 8 per-bout outputs = 3 weight-logits + [sigma_F, alpha_L,
              theta_L, alpha_R, theta_R] (F mean fixed at 0; gammas ASYMMETRIC).
              Heteroscedastic.

This replaces the HMM's latent-state prior over {F,L,R} with a direct feedforward
function of (recent dtheta history + stimulus). It exposes the same per-bout
one-step predictive density `log p(dtheta_t | features_t)` as the HMM, so it plugs
straight into the evaluation pipeline (evaluate.cross_validate_likelihood) and is
ranked alongside `spontaneous` / `simple_stim` / `marginal_density` in Metric A.

dtheta is in DEGREES and the density is over degrees (matching fit_hmm) -- no
radian conversion anywhere.

INPUT  : data/bouts_GsizeMotionSingle.csv (via evaluate.build_records).
OUTPUT : a torch checkpoint (.pt) = {state_dict, config}; optional generated CSV.

Usage
-----
    uv run scripts/fit_MLP.py --bouts data/bouts_GsizeMotionSingle.csv \
        --output data/mlp_basic.pt --mode basic --K 5 --stim-mode simple \
        --hidden 64 --epochs 60 --lr 1e-3
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from twin import fit_hmm as H          # emission constants + numpy reference emissions
from twin import evaluate as E         # GenerativeModel, build_records, make_context, symbol_logmass

# ----------------------------- constants ------------------------------------ #
DTHETA0 = H.DTHETA0_DEG       # 10.0 deg forward/turn threshold
LOG_FLOOR = H.LOG_FLOOR       # -700.0
ALPHA_MIN = H.ALPHA_MIN       # 1.0  (Gamma shape floor, paper constraint)
EPS = H.EPS                   # 1e-12
LOG2PI = math.log(2 * math.pi)

# ------------------- 'shoal' stimulus block constants ----------------------- #
# Non-monotone RANGE tuning x group SIZE x MOTION. Real fish steer toward the shoal
# most strongly at 55-90 px (centroid distance), weakly beyond ~120 px, and REVERSE
# below ~25 px; a monotone proximity weight (the old 'dot' block's 1/(1+r/20))
# cannot represent a peak or a sign change, which is why 'dot'/'dotb' both give a
# flat range profile in free-run.
SHOAL_RBF_PX = (25.0, 50.0, 90.0, 160.0)     # basis knots, px (centroid distance)
SHOAL_RBF_W = 0.40                            # basis width, natural-log units
SHOAL_R_REF = 70.0                            # rz pivot, px
SHOAL_R_OCT = float(np.log(4.0))              # rz scale: +-1 == +-2 octaves
SHOAL_R_MIN = 2.0                             # px floor (r reaches 0.64 px in the data)
SHOAL_N_MAX = 8.0                             # largest group size
# LOWER-CLIP CHECK (Model 42): the "_rzlo" modes are their base mode with rz clipped below at the 2 px floor instead of
# -1 (17.5 px), i.e. rz in [log4(2 / 70), 1] = [-2.56, 1]; same columns, same everything else. 4.5 % of dot bouts
# have the centroid closer than 17.5 px. Only the clip moves: the bumps and the aligned-and-far input are unchanged.
RZLO_SUFFIX = "_rzlo"
RZ_LO_FLOOR = float(np.log(SHOAL_R_MIN / SHOAL_R_REF) / SHOAL_R_OCT)    # -2.565
SHOAL_COLS = ("act", "moving", "nz", "rz",
              "gs1", "gs2", "gs3", "gs4",
              "gc1", "gc2", "gc3", "gc4",
              "gns1", "gns2", "gns3", "gns4",
              "mv_sin", "mv_cos")                                  # 18
SHOALR_COLS = ("act", "moving", "rz",
               "gs1", "gs2", "gs3", "gs4",
               "gc1", "gc2", "gc3", "gc4",
               "mv_sin", "mv_cos")                                 # 13 (range-only ablation)
# FAR-FIELD variant. With knots at (25,50,90,160) the softmax basis saturates beyond the outermost
# knot -- ghat_4 -> 1 as r grows -- so the model cannot distinguish 160 px from 400 px and whatever
# pull it learned at the last knot persists indefinitely. Measured consequence: the generated
# restoring force is -3.4 px/bout beyond 120 px where real fish are at -0.53, i.e. it keeps hauling
# itself toward the shoal from any distance. Real fish-to-shoal range reaches 280 px at q99, so an
# extra knot there gives the head the resolution to LEARN the far-field back-off instead of having
# a taper imposed on it.
SHOALF_RBF_PX = (25.0, 50.0, 90.0, 160.0, 280.0)
SHOALF_COLS = ("act", "moving", "nz", "rz",
               "gs1", "gs2", "gs3", "gs4", "gs5",
               "gc1", "gc2", "gc3", "gc4", "gc5",
               "gns1", "gns2", "gns3", "gns4", "gns5",
               "mv_sin", "mv_cos")                                 # 21
_SHOAL_LC = np.log(np.asarray(SHOAL_RBF_PX, float))
_SHOALF_LC = np.log(np.asarray(SHOALF_RBF_PX, float))
# GROUP-SIZE STEP (Model 24). MEASURED (teacher-forced, moving dots at 40-120 px, 15 held-out fish): the real
# turn toward the dots is a STEP in group size -- 10.7 deg for one dot, 16.5 / 13.5 / 19.5 / 18.3 for 2 / 4 /
# 6 / 8 -- while the fitted law is a RAMP (11.6 / 13.7 / 14.5 / 16.3 / 17.6), because N enters only as
# nz = log2(N)/3 in `nz` and `gns` and a 0.002-bit/bout signal is learned as the simplest (linear)
# dependence. At N = 2 the ramp gives a third of the attraction the real fish already give in full (13.7 vs
# 16.5 deg), which integrated over a block is the 2.7 mm by which the 2-dot plateau sits too far out.
# `n2` = 1[N >= 2] (and `gn2` = basis x n2 x sin) lets the net put the jump at two dots:
#   'shoal_n2'   ramp + step (23)      'shoal_step' step in place of the ramp (18)
#   'shoal_lean' step, and the two redundant flags `act` (= the basis weights' sum) and `moving` out (16)
SHOALN2_COLS = ("act", "moving", "nz", "n2", "rz",
                "gs1", "gs2", "gs3", "gs4", "gc1", "gc2", "gc3", "gc4",
                "gns1", "gns2", "gns3", "gns4", "gn21", "gn22", "gn23", "gn24",
                "mv_sin", "mv_cos")                                # 23
SHOALS_COLS = ("act", "moving", "n2", "rz",
               "gs1", "gs2", "gs3", "gs4", "gc1", "gc2", "gc3", "gc4",
               "gn21", "gn22", "gn23", "gn24",
               "mv_sin", "mv_cos")                                 # 18
SHOALL_COLS = ("n2", "rz",
               "gs1", "gs2", "gs3", "gs4", "gc1", "gc2", "gc3", "gc4",
               "gn21", "gn22", "gn23", "gn24",
               "mv_sin", "mv_cos")                                 # 16
# THE REPORT SEQUENCE (Model/Report_models.ipynb): the centroid code built up from its simplest form. Both reuse
# ``shoal_channels`` with a SINGLE-KNOT range basis (every g_j(r) = 1, so g * sin = sin), so the record builder, the
# generator's closed-loop row and the batched engine stay one code path; the full code is 'shoal' (S4, 18).
#   'shoal_simple'  presence, motion, log count, log range, bearing sin / cos            (6)  -- no products
#   'shoal_int'     + the products: nz * sin, moving * sin, moving * cos                 (9)  -- S4 without the bumps
#   'shoal_rx'      + range x bearing, LINEAR in log range and split by condition        (14) -- the bumps' job, lean:
#                   moving * rz * sin / cos, static * rz * sin / cos (static = dots shown and not moving), and
#                   nz * rz * sin (count x range, the bump code's `gns`); one sign flip per condition at most
SHOALS_SIMPLE_COLS = ("act", "moving", "nz", "rz", "sin", "cos")
SHOALS_INT_COLS = ("act", "moving", "nz", "rz", "sin", "cos", "nz_sin", "mv_sin", "mv_cos")
SHOALS_RX_COLS = SHOALS_INT_COLS + ("mv_rz_sin", "mv_rz_cos", "st_rz_sin", "st_rz_cos", "nz_rz_sin")
# THE ABLATIONS of the report's ablation model D (report Model 22), each 'shoal_int' minus one kind of information:
#   'shoal_int_nn'  no dot count: without log count and count x sin (7)
#   'shoal_int_nm'  no moving indicator: without the moving flag and moving x sin / cos (6)
#   'shoal_null'    no stimulus columns at all (0) -- the online path still runs, so programme / record providers work
SHOALS_INT_NN_COLS = ("act", "moving", "rz", "sin", "cos", "mv_sin", "mv_cos")
SHOALS_INT_NM_COLS = ("act", "nz", "rz", "sin", "cos", "nz_sin")
SHOAL_NULL_COLS = ()
# the same two ablations of the S4x code (report Model 29, ablations of 26e = 18a without the static block):
#   'shoal_rx_nn'  no dot count: without log count, count x sin and count x range x sin (11)
#   'shoal_rx_nm'  no moving indicator: without the moving flag, moving x sin / cos and the moving AND static range x
#                  bearing products (static = shown and not moving, itself a moving-indicator distinction) (7)
SHOALS_RX_NN_COLS = ("act", "moving", "rz", "sin", "cos", "mv_sin", "mv_cos", "mv_rz_sin", "mv_rz_cos", "st_rz_sin",
                     "st_rz_cos")
SHOALS_RX_NM_COLS = ("act", "nz", "rz", "sin", "cos", "nz_sin", "nz_rz_sin")
# two more ablations of 26e's S4x code (the ablation study, Final_report_ablations.ipynb):
#   'shoal_rx_nr'  range removed: without log range and the range x bearing products (8) -- S4i without rz; the
#                  aligned-and-far flag (range x cos) goes with it (stim_align off in the recipe)
#   'shoal_cond'   moving vs static only: the presence and moving flags, no bearing, no range, no count (2) -- the fish
#                  knows that dots are shown and whether they move, not where they are or how many
SHOALS_RX_NR_COLS = ("act", "moving", "nz", "sin", "cos", "nz_sin", "mv_sin", "mv_cos")
SHOAL_COND_COLS = ("act", "moving")
# 'shoal_cnc' = S4 + count x range x COS: the four bumps x log count x cos(bearing), the mirror of `gns` (which is built
# on sin, so it is ~0 exactly when the dots are AHEAD -- where real fish straighten with the count and the fit does
# not). 22 columns: SHOAL_COLS with gnc1-4 right after gns1-4.
SHOALC_COLS = ("act", "moving", "nz", "rz", "gs1", "gs2", "gs3", "gs4", "gc1", "gc2", "gc3", "gc4",
               "gns1", "gns2", "gns3", "gns4", "gnc1", "gnc2", "gnc3", "gnc4", "mv_sin", "mv_cos")
SHOAL_MODES = ("shoal", "shoalr", "shoalf", "shoal_n2", "shoal_step", "shoal_lean", "shoal_simple", "shoal_int",
               "shoal_rx", "shoal_cnc", "shoal_int_nn", "shoal_int_nm", "shoal_null", "shoal_rx_nn", "shoal_rx_nm",
               "shoal_rx_nr", "shoal_cond", "shoal" + RZLO_SUFFIX, "shoal_rx" + RZLO_SUFFIX)


# ------------------- 'shoal_pop': the PER-DOT population code (Model 26) ------------------- #
# The shoal block above is built from ONE object, the active-dot centroid: its bearing terms are the
# first Fourier harmonic of the fish's stimulus panorama and can hold exactly one peak, so two dot
# groups (one left, one right) collapse into a single point between them before the net sees anything.
# Here every dot deposits its own bump at its own bearing and range on a coarse map -- 6 bearing centres
# (60 deg apart, von Mises bumps at 0.5 one centre away) x the 4 range knots -- the bumps are SUMMED
# over the active dots and log-compressed, so a compact shoal is one peak whose height grows sub-
# linearly with its size (local count) and two groups are two peaks. Group size also enters globally,
# as the ramp `nz` and the step `n2` (Model 24); `rz` is the log range to the NEAREST dot; `moving` is
# the fraction of active dots that move (1 / 0 in every real block); `mv_sin/mv_cos` the bearing to the
# centroid of the MOVING dots; a second map over the STATIC dots only replaces the static block.
# For a single group this carries the same information as the shoal + static blocks up to the
# compression, so the single-group profile and AUC are the check; for two groups it is the only
# encoding that lets the fitted net, rather than the input, decide between the peaks.
POP_ANG_DEG = (0.0, 60.0, 120.0, 180.0, -120.0, -60.0)      # bearing centres, travel frame, + = left
POP_KAPPA = float(np.log(2.0) / (1.0 - np.cos(np.radians(60.0))))   # bump = 0.5 one centre away
POP_MODE = "shoal_pop"
POP_COLS = (("act", "moving", "nz", "n2", "rz")
            + tuple(f"p{a}_{r}" for a in range(len(POP_ANG_DEG)) for r in range(len(SHOAL_RBF_PX)))
            + ("mv_sin", "mv_cos")
            + tuple(f"s{a}_{r}" for a in range(len(POP_ANG_DEG)) for r in range(len(SHOAL_RBF_PX))))   # 55
# 'shoal_popd' (per-DIRECTION summaries): the three global summaries `moving`, `nz`, `n2` -- the only inputs
# that total both groups of a two-group programme, and so the only ones that leave the training distribution
# there -- become six columns each, computed over the dots whose NEAREST bearing centre is that direction:
# `mv{a}` = fraction of those dots that move, `nz{a}` = log2(count)/3, `n2{a}` = 1[count >= 2]. On every real block
# a compact group sits at one centre, so these read as the global ones there and split cleanly by side otherwise.
POPD_MODE = "shoal_popd"
POPD_COLS = (("act", "rz")
             + tuple(f"p{a}_{r}" for a in range(len(POP_ANG_DEG)) for r in range(len(SHOAL_RBF_PX)))
             + ("mv_sin", "mv_cos")
             + tuple(f"s{a}_{r}" for a in range(len(POP_ANG_DEG)) for r in range(len(SHOAL_RBF_PX)))
             + tuple(f"mv{a}" for a in range(len(POP_ANG_DEG)))
             + tuple(f"nz{a}" for a in range(len(POP_ANG_DEG)))
             + tuple(f"n2{a}" for a in range(len(POP_ANG_DEG))))                                       # 70
# 'shoal_popn' (Model 30, NORMALISED map): `popd`'s bump cells are log1p(count at that bearing and range), so
# 54 of its 70 columns grow with the dot count and the "where" and the "how many" are MULTIPLIED together
# inside every cell -- which a linear layer cannot separate. MEASURED consequence: the arm's block-level
# attraction (plateau below the stimulus-blind baseline of ~23.3 mm) is almost exactly linear in
# log2(N) -- -1.3 / 2.4 / 4.0 / 5.9 / 6.7 mm at N = 1/2/4/6/8, fitted by -1.3 + 8 nz to R^2 ~ 0.99 --
# while the real law is a STEP: 1.2 mm for one dot and 6.2 / 5.1 / 6.4 / 6.9 flat from two upward.
# To saturate, `popd` has to cancel a magnitude baked into all 48 bump cells using its six n2 columns.
# Here each map is divided by its own total, so it is a DISTRIBUTION (2 and 8 dots in the same arrangement
# give the identical map) and the magnitude becomes one explicit channel per map. No information is lost --
# raw = normalised x magnitude -- but a count-independent response is now the DEFAULT, and count-dependence
# costs one weight on `nz` / `n2` instead of a cancellation across 48 columns.
POPN_MODE = "shoal_popn"
POPN_COLS = (("act", "rz")
             + tuple(f"p{a}_{r}" for a in range(len(POP_ANG_DEG)) for r in range(len(SHOAL_RBF_PX)))
             + ("pmag", "mv_sin", "mv_cos")
             + tuple(f"s{a}_{r}" for a in range(len(POP_ANG_DEG)) for r in range(len(SHOAL_RBF_PX)))
             + ("smag",)
             + tuple(f"mv{a}" for a in range(len(POP_ANG_DEG)))
             + tuple(f"nz{a}" for a in range(len(POP_ANG_DEG)))
             + tuple(f"n2{a}" for a in range(len(POP_ANG_DEG))))                                       # 72
# 'shoal_popl' (Model 31, LOCAL): every channel describes ONE dot group, so a group's columns do not change
# when another group appears elsewhere -- the property a multi-group rollout needs and popn lacks.
# MEASURED on popn with two groups of 4 moving dots at +60 / -60 deg: group A's own map cells are HALVED
# by adding group B (global normalisation), `pmag` is a global total, and mv_sin/mv_cos -- the bearing to
# the centroid of ALL moving dots -- reads 0 deg, i.e. points the fish at the empty space between them.
#   p / s maps   each angular group's bump map divided by THAT GROUP's own total, summed over groups.
#                With one group on screen (>= 95 % of real bouts) this is IDENTICAL to popn's map.
#   pm / sm      raw bump mass per bearing centre, log1p -- how many dots at that bearing (local up to
#                the soft bump tails, as in popd).
#   ms / mc      per moving group, [sin, cos] of the bearing to its centroid, spread over the bearing
#                slots by bump weights normalised to sum to 1. Summed over the six slots this is EXACTLY
#                popn's mv_sin / mv_cos when there is one moving group -- which matters: zeroing
#                mv_sin/mv_cos in the fitted popn removes 73 % of its turn toward the dots (popd 39 %).
#   mv / nz      per direction, as in popd / popn (hard assignment of each group to its nearest centre).
#   dropped      mv_sin, mv_cos, pmag, smag (global); n2 (per direction n2 = 1 exactly when nz >= 1/3).
# 'shoal_popld' adds `dr` = per moving group, the mean velocity of its dots ACROSS the fish's line of sight
# (px/s / DRIFT_SCALE_PXS, + = the bearing increasing, the sign convention of phi), soft-assigned like
# ms/mc. MEASURED, real fish, moving blocks, 40-160 px, dots within 30 deg of the heading: they turn toward
# the dots 8.0 deg when the dots are drifting FURTHER off the midline against 3.9 deg when drifting back
# (+4.1 +/- 2.9 deg, 15 fish), while the fitted popd shows no difference at all (-0.02): no input told it
# which way the dots move. Needs the dots' positions at the previous bout (``dot_tangential``).
POPL_MODE = "shoal_popl"
POPLD_MODE = "shoal_popld"
DRIFT_SCALE_PXS = 10.0     # median per-dot speed on moving blocks (10.7 px/s measured)
DRIFT_MAX_DT_S = 10.0      # no drift across a > 10 s gap (quiescence); stimulus blocks are 50 s apart
_A_ = len(POP_ANG_DEG)
POPL_COLS = (("act", "rz")
             + tuple(f"p{a}_{r}" for a in range(_A_) for r in range(len(SHOAL_RBF_PX)))
             + tuple(f"pm{a}" for a in range(_A_))
             + tuple(f"ms{a}" for a in range(_A_)) + tuple(f"mc{a}" for a in range(_A_))
             + tuple(f"s{a}_{r}" for a in range(_A_) for r in range(len(SHOAL_RBF_PX)))
             + tuple(f"sm{a}" for a in range(_A_))
             + tuple(f"mv{a}" for a in range(_A_)) + tuple(f"nz{a}" for a in range(_A_)))   # 86
POPLD_COLS = POPL_COLS + tuple(f"dr{a}" for a in range(_A_))                                # 92
# '_lab' (Model 34): the same columns as popd / popl, but a GROUP is a stimulus group -- dots sharing a label --
# instead of an angular cluster. MEASURED: the 45-deg bearing-gap rule split 26 % of real 2-dot bouts (39 % of
# 4-8-dot bouts) into separate 1-dot "groups", because the two dots are ~11 mm apart and the fish is ~15 mm from
# them; a split 2-dot group reads nz = 0 = one dot. Every real block is ONE group (dots are never on both sides of
# the fish), so the record path labels every dot 0; a provider may attach ``last_groups`` (one id per dot slot)
# for multi-group programmes. A group's count and moving fraction go to the bearing slot of its circular-mean
# bearing, whatever its angular extent.
POPD_LAB_MODE = "shoal_popd_lab"
POPL_LAB_MODE = "shoal_popl_lab"
POP_MODES = (POP_MODE, POPD_MODE, POPN_MODE, POPL_MODE, POPLD_MODE, POPD_LAB_MODE, POPL_LAB_MODE)


def pop_cols(stim_mode):
    """Canonical column names of a population mode -- the single source of truth."""
    return {POP_MODE: POP_COLS, POPD_MODE: POPD_COLS, POPN_MODE: POPN_COLS,
            POPL_MODE: POPL_COLS, POPLD_MODE: POPLD_COLS,
            POPD_LAB_MODE: POPD_COLS, POPL_LAB_MODE: POPL_COLS}[stim_mode]


def pop_kw(stim_mode):
    """The ``pop_channels`` keyword set of a population mode -- ONE place, read by the record builder
    (``_stim_block``) and the closed-loop row (``fit_MLP_joint``'s generator) alike."""
    if stim_mode == POPL_LAB_MODE:
        return dict(local=True, drift=False, by_label=True)
    if stim_mode == POPD_LAB_MODE:
        return dict(per_dir=True, norm=False, by_label=True)
    if stim_mode in (POPL_MODE, POPLD_MODE):
        return dict(local=True, drift=stim_mode == POPLD_MODE)
    return dict(per_dir=stim_mode in (POPD_MODE, POPN_MODE), norm=stim_mode == POPN_MODE)


def _label_clusters(on_i, grp_i=None):
    """Index arrays of the dot GROUPS among the active dots of ONE bout, by label (``grp_i``, one id per dot
    slot; None = every active dot in one group)."""
    idx = np.flatnonzero(np.asarray(on_i, bool))
    if idx.size == 0:
        return []
    if grp_i is None:
        return [idx]
    g = np.asarray(grp_i)[idx]
    return [idx[g == u] for u in np.unique(g)]


def _clusters(phi_i, on_i, by_label=False, grp_i=None):
    return _label_clusters(on_i, grp_i) if by_label else _angular_clusters(phi_i, on_i)


def dot_tangential(xs_prev, ys_prev, xs, ys, x, y, dt):
    """(n, K) each dot's velocity ACROSS the fish's line of sight, px/s: + = the dot's bearing is
    increasing (the handedness of ``dot_geometry``'s phi, verified identical on every real bout), NaN
    where undefined (a dot missing at either bout, dt <= 0 or > DRIFT_MAX_DT_S). ``xs_prev``/``ys_prev``
    are the SAME dot slots at the previous bout, ``dt`` the time between the two samples in the clock
    the positions were read on. Shared by the record builder and the generator. Scalar-safe in n."""
    xs = np.atleast_2d(np.asarray(xs, float)); ys = np.atleast_2d(np.asarray(ys, float))
    xp = np.atleast_2d(np.asarray(xs_prev, float)); yp = np.atleast_2d(np.asarray(ys_prev, float))
    x = np.atleast_1d(np.asarray(x, float))[:, None]; y = np.atleast_1d(np.asarray(y, float))[:, None]
    dt = np.atleast_1d(np.asarray(dt, float))[:, None]
    ux, uy = xs - x, ys - y
    L = np.hypot(ux, uy)
    with np.errstate(invalid="ignore", divide="ignore"):
        vt = (ux * (ys - yp) - uy * (xs - xp)) / (L * dt)
    ok = np.isfinite(vt) & (dt > 0) & (dt <= DRIFT_MAX_DT_S) & (L > 0)
    return np.where(ok, vt, np.nan)


def _dot_patch_1(phi_i, r_i):
    """(K, A, R) per-dot bump x range weight for ONE bout -- ``pop_bumps``'s patch before the sum over dots
    (same formulas; the caller selects the dots)."""
    phi = np.nan_to_num(np.asarray(phi_i, float)); r = np.asarray(r_i, float)
    ang = np.radians(np.asarray(POP_ANG_DEG, float))
    B = np.exp(POP_KAPPA * (np.cos(phi[:, None] - ang[None, :]) - 1.0))                          # (K, A)
    lr = np.log(np.clip(np.nan_to_num(r, nan=SHOAL_R_REF), SHOAL_R_MIN, None))
    z = -0.5 * ((lr[:, None] - _SHOAL_LC[None, :]) / SHOAL_RBF_W) ** 2                           # (K, R)
    z = z - z.max(axis=1, keepdims=True)
    G = np.exp(z); G = G / G.sum(axis=1, keepdims=True)
    return B[:, :, None] * G[:, None, :]


def _slot_weights(psi):
    """(A,) bump weights of a bearing over the six centres, normalised to sum to 1."""
    w = np.exp(POP_KAPPA * (np.cos(psi - np.radians(np.asarray(POP_ANG_DEG, float))) - 1.0))
    return w / w.sum()


def _popl_bout(phi_i, r_i, on_i, mvon_i, ston_i, vt_i, drift, by_label=False, grp_i=None):
    """ONE bout's local channels (everything in ``POPL_COLS`` / ``POPLD_COLS`` except act, rz, mv, nz)."""
    A, R = _A_, len(SHOAL_RBF_PX)
    P = np.zeros((A, R)); S = np.zeros((A, R)); pm = np.zeros(A); sm = np.zeros(A)
    ms = np.zeros(A); mc = np.zeros(A); dr = np.zeros(A)
    if on_i.any():
        patch = _dot_patch_1(phi_i, r_i)
        pm = np.log1p(patch[on_i].sum(axis=(0, 2)))
        if ston_i.any():
            sm = np.log1p(patch[ston_i].sum(axis=(0, 2)))
        for g in _clusters(phi_i, on_i, by_label, grp_i):
            pg = patch[g].sum(0); P += pg / pg.sum()
        for g in _clusters(phi_i, ston_i, by_label, grp_i):
            pg = patch[g].sum(0); S += pg / pg.sum()
        for g in _clusters(phi_i, mvon_i, by_label, grp_i):
            psi = np.arctan2((r_i[g] * np.sin(phi_i[g])).mean(), (r_i[g] * np.cos(phi_i[g])).mean())
            w = _slot_weights(psi)
            ms += w * np.sin(psi); mc += w * np.cos(psi)
            if drift and vt_i is not None:
                v = vt_i[g]; v = v[np.isfinite(v)]
                if v.size:
                    dr += w * (v.mean() / DRIFT_SCALE_PXS)
    return P.ravel(), pm, ms, mc, S.ravel(), sm, dr


def _angular_labels(phi, mask):
    """(n, K) int cluster label of every dot flagged in ``mask`` (-1 elsewhere): ``_angular_clusters`` for all rows
    at once. Dots sorted by bearing, a new cluster after every circular gap > POP_GAP_DEG; when the wrap-around gap
    (last dot -> first) is NOT a cut, the dots after the last cut belong to the first cluster, as the loop's rotation
    makes them."""
    n, K = phi.shape
    lab = np.full((n, K), -1, int)
    if K == 0:
        return lab
    ph = np.where(mask, _wrap(np.nan_to_num(phi)), np.inf)
    order = np.argsort(ph, axis=1, kind="stable")
    s = np.take_along_axis(ph, order, 1)
    m = np.take_along_axis(mask, order, 1)
    cnt = m.sum(1)
    first = s[:, 0]
    nxt = np.concatenate([s[:, 1:], np.full((n, 1), np.inf)], 1)
    last_ix = np.maximum(cnt - 1, 0)
    wrapnext = first + 2 * np.pi
    j = np.arange(K)[None, :]
    nxt = np.where(j == last_ix[:, None], wrapnext[:, None], nxt)
    gap = np.where(m, nxt - s, 0.0)
    cut = m & (gap > np.radians(POP_GAP_DEG))
    wrap_cut = np.take_along_axis(cut, last_ix[:, None], 1)[:, 0]
    inner = cut & (j < last_ix[:, None])                        # cuts after dot j, excluding the wrap gap
    c = np.concatenate([np.zeros((n, 1), int), np.cumsum(inner, 1)[:, :-1]], 1)   # cuts strictly before j
    n_inner = inner.sum(1)
    # no wrap cut: the tail after the last inner cut joins cluster 0 (the loop rotates it to the front)
    tail = (~wrap_cut)[:, None] & (c == n_inner[:, None]) & (n_inner[:, None] > 0)
    c = np.where(tail, 0, c)
    c = np.where(m, c, -1)
    np.put_along_axis(lab, order, c, 1)
    return lab


def _label_groups_iter(lab):
    """Yield (L, (n, K) mask) for every cluster label present."""
    for L in range(int(lab.max()) + 1 if lab.size else 0):
        g = lab == L
        if g.any():
            yield L, g


def _slots_from_labels(phi, lab, mv):
    """Vectorised per-direction assignment from cluster labels: each cluster's count / moving count to the centre
    nearest its circular-mean bearing (``_nearest_centre``). Returns (cnt_a, mvc_a)."""
    n, A = phi.shape[0], len(POP_ANG_DEG)
    ph = np.nan_to_num(phi); ang = np.radians(np.asarray(POP_ANG_DEG, float))
    cnt_a = np.zeros((n, A)); mvc_a = np.zeros((n, A))
    for L, g in _label_groups_iter(lab):
        k = g.sum(1); rows = np.flatnonzero(k > 0)
        kk = k[rows]
        mm = np.arctan2((np.sin(ph) * g).sum(1)[rows] / kk, (np.cos(ph) * g).sum(1)[rows] / kk)
        a = np.argmin(np.abs(_wrap(mm[:, None] - ang[None, :])), axis=1)
        np.add.at(cnt_a, (rows, a), kk)
        np.add.at(mvc_a, (rows, a), (g & mv).sum(1)[rows])
    return cnt_a, mvc_a


def _popl_from_labels(phi, r, on, mvon, ston, act, lab_on, lab_st, lab_mv):
    """Vectorised ``_popl_bout`` (no drift) from cluster labels of the active / static / moving dots."""
    n, K = phi.shape
    A, R = _A_, len(SHOAL_RBF_PX)
    ph = np.nan_to_num(phi); r0 = np.nan_to_num(r)
    ang = np.radians(np.asarray(POP_ANG_DEG, float))
    Bm = np.exp(POP_KAPPA * (np.cos(ph[:, :, None] - ang[None, None, :]) - 1.0))
    lr = np.log(np.clip(r0, SHOAL_R_MIN, None))
    z = -0.5 * ((lr[:, :, None] - _SHOAL_LC[None, None, :]) / SHOAL_RBF_W) ** 2
    z = z - z.max(axis=2, keepdims=True)
    G = np.exp(z); G = G / G.sum(axis=2, keepdims=True)
    patch = Bm[:, :, :, None] * G[:, :, None, :]

    def _norm_sum(lab):
        out = np.zeros((n, A, R))
        for L, g in _label_groups_iter(lab):
            tot = (patch * g[:, :, None, None]).sum(1)
            s = tot.sum((1, 2))
            out += np.where(s[:, None, None] > 0, tot / np.where(s > 0, s, 1.0)[:, None, None], 0.0)
        return out.reshape(n, A * R)

    P = _norm_sum(lab_on); S = _norm_sum(lab_st)
    pm = np.log1p((patch * on[:, :, None, None]).sum((1, 3)))
    sm = np.log1p((patch * ston[:, :, None, None]).sum((1, 3)))
    ms = np.zeros((n, A)); mc = np.zeros((n, A))
    for L, g in _label_groups_iter(lab_mv):
        k = g.sum(1); rows = np.flatnonzero(k > 0)
        psi = np.arctan2((r0 * np.sin(ph) * g).sum(1)[rows] / k[rows], (r0 * np.cos(ph) * g).sum(1)[rows] / k[rows])
        w = np.exp(POP_KAPPA * (np.cos(psi[:, None] - ang[None, :]) - 1.0)); w = w / w.sum(1, keepdims=True)
        ms[rows] += w * np.sin(psi)[:, None]; mc[rows] += w * np.cos(psi)[:, None]
    z0 = ~act
    P[z0] = 0.0; pm[z0] = 0.0; S[z0] = 0.0; sm[z0] = 0.0
    return P, pm, ms, mc, S, sm


def _one_group_slots(phi, on, mvon, cnt, act):
    """Vectorised per-direction assignment when every active dot of a row is ONE group: the slot nearest the
    circular mean of the active dots' bearings (``_nearest_centre``), the group's count and moving count there.
    Returns (slot (n,), cnt_a (n, A), mvc_a (n, A))."""
    n, A = phi.shape[0], len(POP_ANG_DEG)
    ph = np.nan_to_num(phi)
    c = np.maximum(cnt, 1)
    m = np.arctan2((np.sin(ph) * on).sum(1) / c, (np.cos(ph) * on).sum(1) / c)
    slot = np.argmin(np.abs(_wrap(m[:, None] - np.radians(np.asarray(POP_ANG_DEG, float))[None, :])), axis=1)
    cnt_a = np.zeros((n, A)); mvc_a = np.zeros((n, A))
    rows = np.flatnonzero(act)
    cnt_a[rows, slot[rows]] = cnt[rows]
    mvc_a[rows, slot[rows]] = mvon.sum(1)[rows]
    return slot, cnt_a, mvc_a


def _popl_one_group(phi, r, on, mvon, ston, act):
    """Vectorised ``_popl_bout`` (no drift) when every active dot of a row is one group: (P, pm, ms, mc, S, sm)."""
    n, K = phi.shape
    A, R = _A_, len(SHOAL_RBF_PX)
    ph = np.nan_to_num(phi); r0 = np.nan_to_num(r)
    ang = np.radians(np.asarray(POP_ANG_DEG, float))
    Bm = np.exp(POP_KAPPA * (np.cos(ph[:, :, None] - ang[None, None, :]) - 1.0))                  # (n, K, A)
    lr = np.log(np.clip(r0, SHOAL_R_MIN, None))
    z = -0.5 * ((lr[:, :, None] - _SHOAL_LC[None, None, :]) / SHOAL_RBF_W) ** 2                     # (n, K, R)
    z = z - z.max(axis=2, keepdims=True)
    G = np.exp(z); G = G / G.sum(axis=2, keepdims=True)
    patch = Bm[:, :, :, None] * G[:, :, None, :]                                                  # (n, K, A, R)

    def _map(mask):
        tot = (patch * mask[:, :, None, None]).sum(1)                                             # (n, A, R)
        s = tot.sum((1, 2))
        norm = np.where(s[:, None, None] > 0, tot / np.where(s > 0, s, 1.0)[:, None, None], 0.0)
        return norm.reshape(n, A * R), np.log1p(tot.sum(2))

    P, pm = _map(on); S, sm = _map(ston)
    anymv = mvon.any(1)
    cm = np.maximum(mvon.sum(1), 1)
    psi = np.arctan2((r0 * np.sin(ph) * mvon).sum(1) / cm, (r0 * np.cos(ph) * mvon).sum(1) / cm)
    w = np.exp(POP_KAPPA * (np.cos(psi[:, None] - ang[None, :]) - 1.0)); w = w / w.sum(1, keepdims=True)
    ms = np.where(anymv[:, None], w * np.sin(psi)[:, None], 0.0)
    mc = np.where(anymv[:, None], w * np.cos(psi)[:, None], 0.0)
    z0 = ~act
    P[z0] = 0.0; pm[z0] = 0.0; S[z0] = 0.0; sm[z0] = 0.0
    return P, pm, ms, mc, S, sm


def _wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


POP_GAP_DEG = 45.0        # a larger angular gap between bearing-sorted dots separates two groups


def _angular_clusters(phi_i, on_i):
    """Index arrays of the clusters of active dots in ONE bout: dots sorted by bearing, split where the
    circular gap to the next dot exceeds ``POP_GAP_DEG`` (the wrap-around gap included)."""
    idx = np.flatnonzero(np.asarray(on_i, bool))
    if idx.size == 0:
        return []
    ph = _wrap(np.asarray(phi_i, float)[idx])
    order = idx[np.argsort(ph)]; ph = np.sort(ph)
    gaps = np.diff(np.r_[ph, ph[0] + 2 * np.pi])                 # gap after each dot, wrap-around last
    cuts = np.flatnonzero(gaps > np.radians(POP_GAP_DEG))
    if cuts.size == 0:
        return [order]
    # rotate so the sequence starts right after a cut, then split at the remaining cuts
    start = (cuts[-1] + 1) % len(order)
    order = np.roll(order, -start); gaps = np.roll(gaps, -start)
    cuts = np.flatnonzero(gaps[:-1] > np.radians(POP_GAP_DEG)) + 1
    return np.split(order, cuts)


def _nearest_centre(phis):
    """Index of the bearing centre nearest the circular mean of ``phis``."""
    m = np.arctan2(np.sin(phis).mean(), np.cos(phis).mean())
    return int(np.argmin(np.abs(_wrap(m - np.radians(np.asarray(POP_ANG_DEG, float))))))


def pop_bumps(phi_k, r_k, on_k, raw=False):
    """(n, A*R) log(1 + sum over the dots flagged in ``on_k`` of bearing bump x range weight).
    ``raw=True`` returns the sum itself, uncompressed -- what ``POPN_COLS`` normalises."""
    phi = np.nan_to_num(np.asarray(phi_k, float)); r = np.asarray(r_k, float); on = np.asarray(on_k, bool)
    n = phi.shape[0]
    ang = np.radians(np.asarray(POP_ANG_DEG, float))
    B = np.exp(POP_KAPPA * (np.cos(phi[:, :, None] - ang[None, None, :]) - 1.0))          # (n, K, A)
    lr = np.log(np.clip(np.where(on, np.nan_to_num(r, nan=SHOAL_R_REF), SHOAL_R_REF), SHOAL_R_MIN, None))
    z = -0.5 * ((lr[:, :, None] - _SHOAL_LC[None, None, :]) / SHOAL_RBF_W) ** 2         # (n, K, R)
    z = z - z.max(axis=2, keepdims=True)
    G = np.exp(z); G = G / G.sum(axis=2, keepdims=True)
    # sum over dots of bump x range weight, without materialising the (n, K, A, R) patch
    s = np.einsum("nka,nkr->nar", B * on[:, :, None].astype(float), G, optimize=True).reshape(n, -1)
    if raw:
        return s                                       # un-compressed mass, for the normalised layout
    return np.log1p(s)


def pop_channels(phi_k, r_k, on_k, mv_k, per_dir=False, norm=False, local=False, drift=False, vt_k=None,
                 by_label=False, grp_k=None):
    """(n, 55) per-dot population channels (``POP_COLS``) from per-dot bearings ``phi_k`` (rad, travel
    frame), ranges ``r_k`` (px), activity ``on_k`` and per-dot moving flags ``mv_k``, all (n, K).
    ``per_dir=True``: the (n, 70) ``POPD_COLS`` layout instead (per-direction summaries).
    ``norm=True`` (needs ``per_dir``): the (n, 72) ``POPN_COLS`` layout -- each bump map divided by its
    own total, so it carries only WHERE the dots are, with HOW MANY in its own `pmag` / `smag` channel.
    ``local=True``: the (n, 86) ``POPL_COLS`` layout, or (n, 92) ``POPLD_COLS`` with ``drift`` and the
    per-dot tangential velocities ``vt_k`` (n, K) from ``dot_tangential`` -- see the POPL comment.
    Scalar-safe in n (the generator passes (1, K)); K may be any width, including 0."""
    phi = np.atleast_2d(np.asarray(phi_k, float)); r = np.atleast_2d(np.asarray(r_k, float))
    on = np.atleast_2d(np.asarray(on_k, bool)) & np.isfinite(phi) & np.isfinite(r) & (r > 0.0)
    mv = np.atleast_2d(np.nan_to_num(np.asarray(mv_k, float))) > 0.0
    n = phi.shape[0]
    cnt = on.sum(axis=1)
    act = cnt > 0
    mvon = on & mv; ston = on & ~mv
    moving = np.where(act, mvon.sum(axis=1) / np.maximum(cnt, 1), 0.0)
    nz = np.where(act, np.clip(np.log2(np.maximum(cnt, 1)) / np.log2(SHOAL_N_MAX), 0.0, 1.0), 0.0)
    n2 = np.where(act, (cnt >= 2).astype(float), 0.0)
    r_near = np.where(on, r, np.inf).min(axis=1) if phi.shape[1] else np.full(n, np.inf)
    lr = np.log(np.clip(np.where(act, r_near, SHOAL_R_REF), SHOAL_R_MIN, None))
    rz = np.where(act, np.clip((lr - np.log(SHOAL_R_REF)) / SHOAL_R_OCT, -1.0, 1.0), 0.0)
    nb = len(POP_ANG_DEG) * len(SHOAL_RBF_PX)
    grp = None if grp_k is None else np.atleast_2d(np.asarray(grp_k))
    gi = (lambda i: None) if grp is None else (lambda i: grp[i])  # noqa: E731
    # ONE-GROUP FAST PATH ('_lab' modes with no labels = every active dot is one group, which is every real
    # bout): the per-row cluster loops below, vectorised. Same formulas; verified equal to the loop path.
    # This is what makes the batched fine-tune rollout (scripts/rollout_batch.py) affordable.
    one = bool(by_label and grp is None and not drift)
    if one:
        slot, cnt_a, mvc_a = _one_group_slots(phi, on, mvon, cnt, act)
    if local and one:
        P1, pm1, ms1, mc1, S1, sm1 = _popl_one_group(phi, r, on, mvon, ston, act)
        mv1 = mvc_a / np.maximum(cnt_a, 1.0)
        nz1 = np.where(cnt_a > 0, np.clip(np.log2(np.maximum(cnt_a, 1.0)) / np.log2(SHOAL_N_MAX), 0.0, 1.0), 0.0)
        out = np.concatenate([act.astype(float)[:, None], rz[:, None], P1, pm1, ms1, mc1, S1, sm1, mv1, nz1],
                             axis=1).astype(float)
        assert out.shape[1] == len(POPL_COLS), (out.shape, len(POPL_COLS))
        return out
    ang_vec = (not by_label) and grp is None                  # ANGULAR clusters, vectorised (verified vs the loop)
    if ang_vec:
        lab_on = _angular_labels(phi, on)
        cnt_a, mvc_a = _slots_from_labels(phi, lab_on, mv)
    if local and ang_vec and not drift:
        P1, pm1, ms1, mc1, S1, sm1 = _popl_from_labels(phi, r, on, mvon, ston, act, lab_on,
                                                        _angular_labels(phi, ston), _angular_labels(phi, mvon))
        mv1 = mvc_a / np.maximum(cnt_a, 1.0)
        nz1 = np.where(cnt_a > 0, np.clip(np.log2(np.maximum(cnt_a, 1.0)) / np.log2(SHOAL_N_MAX), 0.0, 1.0), 0.0)
        out = np.concatenate([act.astype(float)[:, None], rz[:, None], P1, pm1, ms1, mc1, S1, sm1, mv1, nz1],
                             axis=1).astype(float)
        assert out.shape[1] == len(POPL_COLS), (out.shape, len(POPL_COLS))
        return out
    if local:
        A = _A_
        vt = None if (not drift or vt_k is None) else np.atleast_2d(np.asarray(vt_k, float))
        blocks = {k: np.zeros((n, w)) for k, w in (("P", nb), ("pm", A), ("ms", A), ("mc", A), ("S", nb),
                                                    ("sm", A), ("mv", A), ("nz", A), ("dr", A))}
        for i in range(n):
            if not on[i].any():
                continue
            P_, pm_, ms_, mc_, S_, sm_, dr_ = _popl_bout(phi[i], np.nan_to_num(r[i]), on[i], mvon[i], ston[i],
                                                         None if vt is None else vt[i], drift, by_label, gi(i))
            blocks["P"][i], blocks["pm"][i], blocks["ms"][i], blocks["mc"][i] = P_, pm_, ms_, mc_
            blocks["S"][i], blocks["sm"][i], blocks["dr"][i] = S_, sm_, dr_
            # per direction, exactly as popd / popn: each group to its nearest centre
            cnt = np.zeros(A); mvc = np.zeros(A)
            for members in _clusters(phi[i], on[i], by_label, gi(i)):
                a = _nearest_centre(phi[i][members])
                cnt[a] += len(members); mvc[a] += float(mv[i][members].sum())
            blocks["mv"][i] = mvc / np.maximum(cnt, 1.0)
            blocks["nz"][i] = np.where(cnt > 0, np.clip(np.log2(np.maximum(cnt, 1.0)) / np.log2(SHOAL_N_MAX),
                                                        0.0, 1.0), 0.0)
        cols = [act.astype(float)[:, None], rz[:, None], blocks["P"], blocks["pm"], blocks["ms"], blocks["mc"],
                blocks["S"], blocks["sm"], blocks["mv"], blocks["nz"]] + ([blocks["dr"]] if drift else [])
        out = np.concatenate(cols, axis=1).astype(float)
        want = POPLD_COLS if drift else POPL_COLS
        assert out.shape[1] == len(want), (out.shape, len(want))
        return out
    P_all = pop_bumps(phi, r, on, raw=norm) if phi.shape[1] else np.zeros((n, nb))
    P_st = pop_bumps(phi, r, ston, raw=norm) if phi.shape[1] else np.zeros_like(P_all)
    if norm:
        # WHERE / HOW MANY split: the map becomes a distribution (sums to 1 whenever any dot is on) and
        # its mass moves into one channel. Exactly invertible: raw = P * expm1(mag).
        def _split(P):
            tot = P.sum(axis=1)
            return np.where(tot[:, None] > 0.0, P / np.maximum(tot, 1e-12)[:, None], 0.0), np.log1p(tot)
        P_all, p_mag = _split(P_all)
        P_st, s_mag = _split(P_st)
    # bearing to the centroid of the MOVING dots (positions relative to the fish: r (cos, sin))
    nm = mvon.sum(axis=1)
    cxm = (np.where(mvon, np.nan_to_num(r) * np.cos(np.nan_to_num(phi)), 0.0)).sum(axis=1) / np.maximum(nm, 1)
    cym = (np.where(mvon, np.nan_to_num(r) * np.sin(np.nan_to_num(phi)), 0.0)).sum(axis=1) / np.maximum(nm, 1)
    pm = np.arctan2(cym, cxm)
    mv_sin = np.where(nm > 0, np.sin(pm), 0.0); mv_cos = np.where(nm > 0, np.cos(pm), 0.0)
    if per_dir:
        # GROUP-WISE assignment: the active dots are split into angular clusters (a gap of more than
        # POP_GAP_DEG between bearing-sorted neighbours starts a new cluster), and each cluster's count and
        # moving count go to the centre nearest its circular-mean bearing. A compact shoal is ONE cluster
        # whatever its angular spread, so on every real block exactly one direction is populated and reads
        # as the global summaries (a per-dot nearest-centre rule split compact groups on a third of the
        # bouts); two separated groups are two clusters.
        A = len(POP_ANG_DEG)
        if not one and not ang_vec:
            cnt_a = np.zeros((n, A)); mvc_a = np.zeros((n, A))
            for i in range(n):
                for members in _clusters(phi[i], on[i], by_label, gi(i)):
                    a = _nearest_centre(phi[i][members])
                    cnt_a[i, a] += len(members); mvc_a[i, a] += float(mv[i][members].sum())
        mv_a = mvc_a / np.maximum(cnt_a, 1.0)
        nz_a = np.where(cnt_a > 0, np.clip(np.log2(np.maximum(cnt_a, 1.0)) / np.log2(SHOAL_N_MAX), 0.0, 1.0), 0.0)
        n2_a = (cnt_a >= 2).astype(float)
        head = [act.astype(float)[:, None], rz[:, None], P_all]
        if norm:
            head += [p_mag[:, None]]
        head += [mv_sin[:, None], mv_cos[:, None], P_st]
        if norm:
            head += [s_mag[:, None]]
        out = np.concatenate(head + [mv_a, nz_a, n2_a], axis=1).astype(float)
        want = POPN_COLS if norm else POPD_COLS
        assert out.shape[1] == len(want), (out.shape, len(want))
        return out
    out = np.concatenate([act.astype(float)[:, None], moving[:, None], nz[:, None], n2[:, None], rz[:, None],
                          P_all, mv_sin[:, None], mv_cos[:, None], P_st], axis=1).astype(float)
    assert out.shape[1] == len(POP_COLS), (out.shape, len(POP_COLS))
    return out


def dot_geometry(xs, ys, x, y, ref_heading):
    """Per-dot (bearing in the frame of ``ref_heading``, range) from absolute dot positions ``xs``,
    ``ys`` (n, K) and the fish's position (n,) -- the one geometry both the record block and the
    generator's row use."""
    xs = np.atleast_2d(np.asarray(xs, float)); ys = np.atleast_2d(np.asarray(ys, float))
    x = np.atleast_1d(np.asarray(x, float))[:, None]; y = np.atleast_1d(np.asarray(y, float))[:, None]
    ref = np.atleast_1d(np.asarray(ref_heading, float))[:, None]
    phi = _wrap(np.arctan2(ys - y, xs - x) - ref)
    return phi, np.hypot(xs - x, ys - y)


def pop_row(xs, ys, on, mv_k, x, y, h, per_dir=False, norm=False, local=False, drift=False, vt_k=None,
            by_label=False, grp_k=None):
    """ONE bout's population row from the CURRENT generated pose ``(x, y, h)`` and the dots' absolute
    positions -- the closed-loop counterpart of ``_stim_block``'s 'shoal_pop' branch (same functions)."""
    xs = np.asarray(xs, float).reshape(1, -1); ys = np.asarray(ys, float).reshape(1, -1)
    phi, r = dot_geometry(xs, ys, [x], [y], [h])
    return pop_channels(phi, r, np.asarray(on, bool).reshape(1, -1), np.asarray(mv_k, float).reshape(1, -1),
                        per_dir=per_dir, norm=norm, local=local, drift=drift,
                        vt_k=None if vt_k is None else np.asarray(vt_k, float).reshape(1, -1),
                        by_label=by_label, grp_k=None if grp_k is None else np.asarray(grp_k).reshape(1, -1))[0]


def shoal_cols(stim_mode):
    """Canonical column names for the shoal modes -- the single source of truth."""
    if stim_mode.endswith(RZLO_SUFFIX):                  # the lower-clip variants keep their base mode's columns
        return shoal_cols(stim_mode[:-len(RZLO_SUFFIX)])
    if stim_mode in POP_MODES:
        return pop_cols(stim_mode)
    return {"shoal": SHOAL_COLS, "shoalf": SHOALF_COLS, "shoal_n2": SHOALN2_COLS,
            "shoal_step": SHOALS_COLS, "shoal_lean": SHOALL_COLS, "shoal_simple": SHOALS_SIMPLE_COLS,
            "shoal_int": SHOALS_INT_COLS, "shoal_rx": SHOALS_RX_COLS,
            "shoal_cnc": SHOALC_COLS, "shoal_int_nn": SHOALS_INT_NN_COLS, "shoal_int_nm": SHOALS_INT_NM_COLS,
            "shoal_null": SHOAL_NULL_COLS, "shoal_rx_nn": SHOALS_RX_NN_COLS,
            "shoal_rx_nm": SHOALS_RX_NM_COLS, "shoal_rx_nr": SHOALS_RX_NR_COLS,
            "shoal_cond": SHOAL_COND_COLS}.get(stim_mode, SHOALR_COLS)


def shoal_kw(stim_mode):
    """The ``shoal_channels`` keyword set of a shoal mode -- ONE place, read by the record builder
    (``_stim_block``) and the closed-loop row (``fit_MLP_mix._dot_stim_row``) alike."""
    if stim_mode.endswith(RZLO_SUFFIX):                  # base mode's keywords + rz clipped at the 2 px floor
        return dict(shoal_kw(stim_mode[:-len(RZLO_SUFFIX)]), rz_lo=RZ_LO_FLOOR)
    if stim_mode in ("shoal_simple", "shoal_int", "shoal_rx"):
        return dict(use_n=True, far=False, n_mode="log", flags=True, single_knot=True,
                    products=(stim_mode != "shoal_simple"), range_products=(stim_mode == "shoal_rx"))
    if stim_mode == "shoal_cnc":
        return dict(use_n=True, far=False, n_mode="log", flags=True, count_cos=True)
    if stim_mode == "shoal_int_nn":
        return dict(use_n=False, far=False, n_mode="log", flags=True, single_knot=True, products=True)
    if stim_mode == "shoal_int_nm":
        return dict(use_n=True, far=False, n_mode="log", flags=True, single_knot=True, products=True, moving_cols=False)
    if stim_mode == "shoal_null":
        return dict(null=True)
    if stim_mode == "shoal_rx_nn":
        return dict(use_n=False, far=False, n_mode="log", flags=True, single_knot=True, products=True,
                    range_products=True)
    if stim_mode == "shoal_rx_nm":
        return dict(use_n=True, far=False, n_mode="log", flags=True, single_knot=True, products=True,
                    range_products=True, moving_cols=False)
    if stim_mode == "shoal_rx_nr":
        return dict(use_n=True, far=False, n_mode="log", flags=True, single_knot=True, products=True, use_range=False)
    if stim_mode == "shoal_cond":
        return dict(cond_only=True)
    return dict(use_n=(stim_mode != "shoalr"), far=(stim_mode == "shoalf"),
                n_mode={"shoal_n2": "both", "shoal_step": "step", "shoal_lean": "step"}.get(stim_mode, "log"),
                flags=(stim_mode != "shoal_lean"))


def shoal_channels(phi, r_px, n_dots, moving, use_n=True, far=False, n_mode="log", flags=True,
                   single_knot=False, products=True, range_products=False, count_cos=False, moving_cols=True,
                   null=False, use_range=True, cond_only=False, rz_lo=-1.0):
    """(n, 18) -- or (n, 13) when ``use_n`` is False -- 'shoal' stimulus channels.

    ``n_mode`` = how group size enters: 'log' (nz = log2 N / 3, the archived ramp), 'step'
    (n2 = 1[N >= 2]) or 'both'; ``flags=False`` drops the `act` / `moving` columns (see
    ``SHOALN2_COLS`` / ``SHOALS_COLS`` / ``SHOALL_COLS``). Column order is ``SHOAL_COLS`` / ``SHOALR_COLS``. Vectorised AND scalar-safe, so
    ``_stim_block`` (record arrays, train/score time) and ``fit_MLP_mix._dot_stim_row``
    (ONE row rebuilt from the GENERATED pose during closed-loop rollout) share this
    single code path -- the by-hand column-order hazard cannot reappear for this mode.

    phi     egocentric bearing to the ACTIVE-DOT CENTROID (rad, + = fish's left);
            NaN when no dot is active.
    r_px    fish-to-centroid distance (px); NaN when no dot is active.
    n_dots  N = n_dots_left + n_dots_right, a block constant in {1,2,4,6,8}; 0 off-block.
    moving  1.0 on moving-stimulus bouts (block constant).

    Every channel is exactly 0.0 on no-stimulus bouts and bounded O(1) by construction,
    so no leak-free scale fitting is needed (``rich_scales`` stays unused).
    """
    phi = np.atleast_1d(np.asarray(phi, float))
    if null:                                         # 'shoal_null': the no-stimulus ablation
        return np.zeros((phi.shape[0], 0), dtype=float)
    r = np.atleast_1d(np.asarray(r_px, float))
    N = np.atleast_1d(np.nan_to_num(np.asarray(n_dots, float)))
    mv = np.atleast_1d(np.nan_to_num(np.asarray(moving, float)))
    act = np.isfinite(phi) & np.isfinite(r) & (r > 0.0)
    if cond_only:                                    # 'shoal_cond': presence and moving flags only
        return np.concatenate([act.astype(float)[:, None], np.where(act, mv, 0.0)[:, None]], axis=1).astype(float)
    s = np.where(act, np.sin(np.nan_to_num(phi)), 0.0)
    c = np.where(act, np.cos(np.nan_to_num(phi)), 0.0)
    # log-range, floored; off-dot rows are parked at the pivot then zeroed by `act`
    lr = np.log(np.where(act, np.clip(r, SHOAL_R_MIN, None), SHOAL_R_REF))
    nz = np.where(act, np.log2(np.clip(N, 1.0, None)) / np.log2(SHOAL_N_MAX), 0.0)
    rz = np.where(act, np.clip((lr - np.log(SHOAL_R_REF)) / SHOAL_R_OCT, rz_lo, 1.0), 0.0)
    # softmax-normalised basis: sum_j ghat_j == 1, so the gain profile FLAT-extrapolates
    # outside the knots (raw Gaussians would force gain -> 0 at both ends, wrong at the
    # near end where a sustained NEGATIVE gain is required) and sum_j g*sin == sin exactly,
    # which is why raw sin/cos are redundant and omitted.
    if single_knot:                                  # the report sequence: ONE bump, 1 wherever dots are shown
        G = act[:, None].astype(float)
    else:
        _lc = _SHOALF_LC if far else _SHOAL_LC
        z = -0.5 * ((lr[:, None] - _lc[None, :]) / SHOAL_RBF_W) ** 2             # (n, 4) or (n, 5)
        z = z - z.max(axis=1, keepdims=True)
        G = np.exp(z)
        G = G / G.sum(axis=1, keepdims=True)
        G = G * act[:, None].astype(float)
    n2 = np.where(act, (N >= 2.0).astype(float), 0.0)
    cols = ([act.astype(float)[:, None]] + ([mv[:, None]] if moving_cols else [])) if flags else []
    if use_n and n_mode in ("log", "both"):
        cols.append(nz[:, None])
    if use_n and n_mode in ("step", "both"):
        cols.append(n2[:, None])
    cols += ([rz[:, None]] if use_range else []) + [G * s[:, None], G * c[:, None]]
    if products and use_n and n_mode in ("log", "both"):
        cols.append(G * (nz * s)[:, None])
    if count_cos:                                    # 'shoal_cnc': the cos mirror of gns (count x range x cos)
        if single_knot or not (products and use_n and n_mode == "log"):
            raise ValueError("count_cos belongs to the bump code 'shoal_cnc'")
        cols.append(G * (nz * c)[:, None])
    if products and use_n and n_mode in ("step", "both"):
        cols.append(G * (n2 * s)[:, None])
    if products and moving_cols:
        cols += [(mv * s)[:, None], (mv * c)[:, None]]
    if range_products:                               # 'shoal_rx': range x bearing per condition, linear in rz
        if not (single_knot and products and n_mode == "log" and use_range):
            raise ValueError("range_products belongs to the single-knot report code ('shoal_rx')")
        st = act.astype(float) * (1.0 - mv)          # static dots: shown and not moving
        if moving_cols:                              # (the moving-indicator ablation drops both condition pairs)
            cols += [(mv * rz * s)[:, None], (mv * rz * c)[:, None], (st * rz * s)[:, None], (st * rz * c)[:, None]]
        if use_n:                                    # (the count ablation drops count x range x sin)
            cols.append((nz * rz * s)[:, None])
    out = np.concatenate(cols, axis=1).astype(float)
    if not use_range:
        if not (single_knot and products and use_n and moving_cols and not range_products):
            raise ValueError("use_range=False belongs to the range ablation 'shoal_rx_nr'")
        _exp = SHOALS_RX_NR_COLS
    elif single_knot and range_products and not use_n:
        _exp = SHOALS_RX_NN_COLS
    elif single_knot and range_products and not moving_cols:
        _exp = SHOALS_RX_NM_COLS
    elif single_knot and not use_n:
        _exp = SHOALS_INT_NN_COLS
    elif single_knot and not moving_cols:
        _exp = SHOALS_INT_NM_COLS
    elif single_knot:
        _exp = SHOALS_RX_COLS if range_products else (SHOALS_INT_COLS if products else SHOALS_SIMPLE_COLS)
    elif not use_n:
        _exp = SHOALR_COLS
    elif far:
        _exp = SHOALF_COLS
    elif count_cos:
        _exp = SHOALC_COLS
    else:
        _exp = {("log", True): SHOAL_COLS, ("both", True): SHOALN2_COLS, ("step", True): SHOALS_COLS,
                ("step", False): SHOALL_COLS}[(n_mode, bool(flags))]
    assert out.shape[1] == len(_exp), (out.shape, len(_exp))
    return out


def static_channels(phi, r_px, n_dots, moving):
    """(n, 8) STATIC-dot bearing channels (Model 17): the shoal range basis x sin(bearing) and
    x cos(bearing), non-zero only on bouts with STATIC dots (``moving == 0``, dots active).

    MEASURED (Model 16, teacher-forced on static-dot bouts): real fish turn AWAY from static dots
    once within ~60 px (toward-turn -8.8 deg with the dots ahead, |dtheta| 53) and are neutral
    beyond, while every arm keeps a weak attraction (+2 deg, 42) -- the shoal block's range x
    bearing products are shared between conditions and only ``mv_sin`` / ``mv_cos`` are
    condition-specific, so a static law could only be the moving law minus those two terms, never
    a sign flip. Same basis, same conventions and the same scalar-safe path as ``shoal_channels``,
    so the generator's closed-loop row (``fit_MLP_joint``) and the record block agree by
    construction."""
    phi = np.atleast_1d(np.asarray(phi, float))
    r = np.atleast_1d(np.asarray(r_px, float))
    N = np.atleast_1d(np.nan_to_num(np.asarray(n_dots, float)))
    mv = np.atleast_1d(np.nan_to_num(np.asarray(moving, float)))
    st = np.isfinite(phi) & np.isfinite(r) & (r > 0.0) & (N > 0.0) & (mv <= 0.0)
    s = np.where(st, np.sin(np.nan_to_num(phi)), 0.0)
    c = np.where(st, np.cos(np.nan_to_num(phi)), 0.0)
    lr = np.log(np.where(st, np.clip(r, SHOAL_R_MIN, None), SHOAL_R_REF))
    z = -0.5 * ((lr[:, None] - _SHOAL_LC[None, :]) / SHOAL_RBF_W) ** 2
    z = z - z.max(axis=1, keepdims=True)
    G = np.exp(z)
    G = G / G.sum(axis=1, keepdims=True) * st[:, None].astype(float)
    return np.concatenate([G * s[:, None], G * c[:, None]], axis=1).astype(float)


# --------------------------- device / seeding ------------------------------- #
def _get_device():
    """CPU/CUDA/MPS pick (mirrors src/model_NN._get_device)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed, deterministic=True):
    """Seed torch + legacy numpy, and (by default) make CUDA fits REPRODUCIBLE.

    Verified: on CPU two identical-config fits are already bitwise identical, so the code has no
    RNG leak. But the same config fitted twice on Colab's A100 gave different weights -- held-out
    bits_per_bout 7.7388 vs 7.7365, which involves no sampling at all and so is a pure function of
    the weights. That fit-to-fit spread is ~0.003 on c2st_traj, comparable to the ~0.005 rollout
    SE the paired contrast reports, and the paired-on-replicate contrast treats the fitted model as
    FIXED so it cannot see it. Any effect below ~0.01 AUC is therefore unresolvable without this.

    cuBLAS matmul reductions use atomics unless a fixed workspace is configured, hence the env var
    (must be set before the first CUDA context; harmless on CPU).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not deterministic:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False          # benchmark autotunes per run -> different algorithms
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _inv_softplus(y):
    """raw such that softplus(raw) == y (so params start at data moments)."""
    return float(np.log(np.expm1(max(y, 1e-4))))


# =========================================================================== #
#  Feature builder -- the SINGLE source of truth (reused by fit/score/sample)
# =========================================================================== #
def _history_block(dtheta, K, dth_scale, run_pos=None):
    """(n, K) AR block: column k-1 = dtheta[t-k], zero-padded for t < k.

    ``run_pos`` (position within a quiescence-free run) additionally zeroes slot k-1 wherever
    ``run_pos < k``, so a bout early in a run never carries a turn from BEFORE its gap. Without
    it the padding happens once per record -- one per fish -- and ~11% of rows carry at least
    one lag reaching across minutes of stillness. ``None`` reproduces the old behaviour exactly.
    """
    n = len(dtheta)
    Hh = np.zeros((n, K))
    for k in range(1, K + 1):
        Hh[k:, k - 1] = dtheta[:-k]
    if run_pos is not None:
        rp = np.asarray(run_pos)
        for k in range(1, K + 1):
            Hh[rp < k, k - 1] = 0.0
    return Hh / dth_scale


def _symbol_block(symbol, K):
    """(n, 3K) one-hot of the previous K symbols (zeros for missing history)."""
    n = len(symbol)
    S = np.zeros((n, 3 * K))
    for k in range(1, K + 1):
        for t in range(k, n):
            S[t, 3 * (k - 1) + int(symbol[t - k])] = 1.0
    return S


def _stim_block(rec, stim_mode, rich_scales, phi_ref="ori"):
    """(n, s_dim) stimulus features from FishRecord fields only (teacher-forceable).

    ``phi_ref='travel'`` reads the dot bearing referenced to the PREVIOUS travel heading
    (``rec.stim_angle_travel``, exact absolute-bearing recovery in ``build_border_records``)
    instead of the raw body orientation -- matching the frame the closed-loop generator's
    per-bout recompute (``MX._dot_stim_row``) actually uses. Default = old behaviour."""
    n = rec.n
    if phi_ref == "travel" and stim_mode not in ("dot", "dotb", "s1", "s2") + POP_MODES + SHOAL_MODES:
        raise ValueError(f"phi_ref='travel' is not supported for stim_mode {stim_mode!r}")
    if stim_mode in POP_MODES:
        # PER-DOT population code from the record's per-dot positions (add_dot_positions.py) and the
        # fish's own pose; bearings in the frame the closed-loop row uses: the previous travel heading
        # (phi_ref='travel', theta_{t-1} = heading_rad[t] - dtheta[t], as stim_angle_travel) or f_ori.
        xs = getattr(rec, "dot_xs", None); ys = getattr(rec, "dot_ys", None)
        if xs is None or ys is None:
            raise RuntimeError("stim_mode='shoal_pop' needs rec.dot_xs / rec.dot_ys (run "
                               "scripts/add_dot_positions.py, then fit_MLP_border.build_border_records)")
        if phi_ref == "travel":
            ref = _wrap(np.asarray(rec.heading_rad, float) - np.radians(np.asarray(rec.dtheta, float)))
        else:
            ref = np.asarray(rec.f_ori_rad, float)
        phi_k, r_k = dot_geometry(xs, ys, rec.x_px, rec.y_px, ref)
        on = np.isfinite(np.asarray(xs, float)) & np.isfinite(np.asarray(ys, float))
        mv_k = np.repeat(np.nan_to_num(np.asarray(rec.moving, float))[:, None], on.shape[1], axis=1)
        kw = pop_kw(stim_mode)
        if kw.get("by_label"):
            kw = dict(kw, grp_k=getattr(rec, "dot_group", None))   # None = one group per bout (every real block)
        if kw.get("drift"):
            # each dot's velocity across the line of sight since the PREVIOUS bout of the record, on the
            # record's own frame clock -- the generator does the same on whichever clock its dots come from
            from twin.fit_MLP_border import _FPS
            X = np.asarray(xs, float); Y = np.asarray(ys, float)
            Xp = np.vstack([np.full((1, X.shape[1]), np.nan), X[:-1]])
            Yp = np.vstack([np.full((1, Y.shape[1]), np.nan), Y[:-1]])
            fo = np.asarray(rec.frame_onset, float)
            dt = np.r_[0.0, np.diff(fo)] / _FPS
            kw = dict(kw, vt_k=dot_tangential(Xp, Yp, X, Y, rec.x_px, rec.y_px, dt))
        return pop_channels(phi_k, r_k, on, mv_k, **kw)   # (n, 55 / 70 / 72 / 86 / 92)
    _phi = None
    if stim_mode in ("dot", "dotb", "s1", "s2") + SHOAL_MODES:
        if phi_ref == "travel":
            if getattr(rec, "stim_angle_travel", None) is None:
                raise RuntimeError("phi_ref='travel' needs rec.stim_angle_travel "
                                   "(bouts CSV via fit_MLP_border.build_border_records).")
            _phi = np.asarray(rec.stim_angle_travel, float)
        else:
            _phi = np.asarray(rec.stim_angle, float)
    if stim_mode == "none":
        return np.zeros((n, 0))
    if stim_mode == "simple":
        return rec.b.reshape(n, 1).astype(float)      # sin(phi) on moving bouts, else 0
    if stim_mode == "rich":
        if rec.u_raw is None:
            raise RuntimeError("stim_mode='rich' needs rec.u_raw "
                               "(build_records with abs_dict & roi).")
        scales = rich_scales if rich_scales is not None else 1.0
        return (np.asarray(rec.u_raw, float) / scales).astype(float)   # (n,5) standardised
    if stim_mode == "rich_csv":
        if rec.stim_csv is None:
            raise RuntimeError("stim_mode='rich_csv' needs rec.stim_csv "
                               "(build_records reads the CSV stimulus columns).")
        return np.asarray(rec.stim_csv, float)   # (n,7) already normalised & bounded, from CSV
    if stim_mode == "dot":
        # proximity-weighted egocentric bearing to the active-dot centroid, present on ALL
        # active-dot bouts (moving AND static -- unlike `simple`, which is sin(phi) on moving
        # bouts only). Built from the Phase-1 dot geometry (add_dot_geometry.py); NaN->0 off-dot.
        if getattr(rec, "dot_prox", None) is None:
            raise RuntimeError("stim_mode='dot' needs rec.dot_prox "
                               "(run scripts/add_dot_geometry.py, then build_border_records).")
        phi = _phi
        g = np.nan_to_num(np.asarray(rec.dot_prox, float))            # proximity weight, 0 off-dot
        s = np.nan_to_num(np.sin(phi)); c = np.nan_to_num(np.cos(phi))  # 0 where no active dot
        mv = np.asarray(rec.moving, float)
        return np.stack([s, c, g * s, g * c, g, mv], axis=1).astype(float)   # (n, 6)
    if stim_mode in ("s1", "s2"):
        # The notebook's S1 / S2. Deliberately NOT `dot`: same width, different content.
        #   [act, sin phi, cos phi, g, moving, nz]        (S1, 6)
        #   + outside                                     (S2, 7)
        # vs dot's [s, c, g*s, g*c, g, mv] -- S1 spends its budget on the ACTIVE gate and
        # GROUP SIZE and has no range x bearing interaction, so the two are a real contrast
        # rather than a relabelling. g is the monotone proximity weight 1/(1 + r_mm/20).
        phi = _phi
        act = np.isfinite(phi)
        s_ = np.where(act, np.sin(np.nan_to_num(phi)), 0.0)
        c_ = np.where(act, np.cos(np.nan_to_num(phi)), 0.0)
        if getattr(rec, "dot_prox", None) is None:
            raise RuntimeError(f"stim_mode={stim_mode!r} needs rec.dot_prox "
                               "(run scripts/add_dot_geometry.py, then build_border_records).")
        g_ = np.where(act, np.nan_to_num(np.asarray(rec.dot_prox, float)), 0.0)
        N_ = np.nan_to_num(np.asarray(getattr(rec, "n_dots", np.zeros(n)), float))
        nz_ = np.where(act, np.log2(np.clip(N_, 1.0, None)) / np.log2(SHOAL_N_MAX), 0.0)
        mv_ = np.nan_to_num(np.asarray(rec.moving, float))
        cols = [act.astype(float), s_, c_, g_, mv_, nz_]
        if stim_mode == "s2":
            # outside = the dot CENTROID lies beyond the dish wall. 11% of real stimulus
            # bouts, and at MATCHED range the shoal is followed ~0.18 less often there, so it
            # carries information a range channel cannot supply.
            if getattr(rec, "dot_cx", None) is None or getattr(rec, "dish_R", None) is None:
                raise RuntimeError("stim_mode='s2' needs rec.dot_cx/dot_cy and the dish "
                                   "(build_border_records supplies both).")
            r_dot = np.hypot(np.asarray(rec.dot_cx, float) - rec.dish_cx,
                             np.asarray(rec.dot_cy, float) - rec.dish_cy) / float(rec.dish_R)
            cols.append(np.where(act, (np.nan_to_num(r_dot) > 1.0).astype(float), 0.0))
        return np.stack(cols, axis=1).astype(float)
    if stim_mode == "dotb":
        # bearing-only ablation of 'dot': [sin φ, cos φ, moving] present on ALL active-dot bouts
        # (moving AND static) -- unlike `simple` (sin φ on moving bouts only). Drops the proximity
        # channels to isolate whether they were what worsened abs_dtheta_w1. Needs only rec.stim_angle
        # + rec.moving (no Phase-1 dot geometry required).
        phi = _phi
        s = np.nan_to_num(np.sin(phi)); c = np.nan_to_num(np.cos(phi))
        mv = np.asarray(rec.moving, float)
        return np.stack([s, c, mv], axis=1).astype(float)   # (n, 3)
    if stim_mode in SHOAL_MODES:
        # Non-monotone RANGE tuning (softmax basis on log r) x group SIZE x MOTION, all
        # from bouts-CSV columns only. 'shoalr' is the range-only ablation (drops N).
        # Unlike 'dot', the range channels are signed-free -- the net may learn a NEGATIVE
        # near-field gain, which the measured close-range turn-away requires.
        if getattr(rec, "dot_dist_px", None) is None or getattr(rec, "n_dots", None) is None:
            raise RuntimeError(f"stim_mode={stim_mode!r} needs rec.dot_dist_px and rec.n_dots "
                               "(run scripts/add_dot_geometry.py, then "
                               "fit_MLP_border.build_border_records).")
        return shoal_channels(_phi, rec.dot_dist_px, rec.n_dots,
                              np.asarray(rec.moving, float),
                              **shoal_kw(stim_mode))               # (n, 18/13/21/23/16)
    raise ValueError(f"unknown stim_mode {stim_mode!r}")


def build_feature_matrix(rec, cfg_feat):
    """(n, D) input matrix for one record from its OWN dtheta history + stimulus.

    cfg_feat = dict(K, stim_mode, include_symbols, dth_scale, rich_scales).
    """
    blocks = [_history_block(rec.dtheta, cfg_feat["K"], cfg_feat["dth_scale"])]
    if cfg_feat["include_symbols"]:
        blocks.append(_symbol_block(rec.symbol, cfg_feat["K"]))
    blocks.append(_stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                              phi_ref=cfg_feat.get("phi_ref", "ori")))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_feature_sequences(records, cfg_feat):
    """Stack (X (N,D), y (N,)) over records. y = dtheta target (deg)."""
    Xs, ys = [], []
    for r in records:
        Xs.append(build_feature_matrix(r, cfg_feat))
        ys.append(r.dtheta.astype(np.float32))
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


# =========================================================================== #
#  torch MDN module (basic: 3 outputs + global scalars; full: 8 per-bout)
# =========================================================================== #
class MDN(nn.Module):
    def __init__(self, n_features, hidden, mode="basic", init_emis=None):
        super().__init__()
        self.mode = mode
        self.body = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        if mode == "basic":
            self.head = nn.Linear(hidden, 3)                 # weight logits only
            s0, a0, t0 = init_emis or (30.0, 1.5, 25.0)
            self.raw_sigma = nn.Parameter(torch.tensor(_inv_softplus(s0)))
            self.raw_alpha = nn.Parameter(torch.tensor(_inv_softplus(max(a0 - ALPHA_MIN, 1e-3))))
            self.raw_theta = nn.Parameter(torch.tensor(_inv_softplus(t0)))
        elif mode == "full":
            self.head = nn.Linear(hidden, 8)                 # 3 logits + sigmaF,aL,tL,aR,tR
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def forward(self, x):
        h = self.body(x)
        if self.mode == "basic":
            logw = F.log_softmax(self.head(h), dim=1)        # (n,3)
            sig = F.softplus(self.raw_sigma) + EPS           # scalar tensors (0-dim)
            aL = F.softplus(self.raw_alpha) + ALPHA_MIN
            tL = F.softplus(self.raw_theta) + EPS
            return logw, sig, aL, tL, aL, tL                 # symmetric: L == R
        out = self.head(h)
        logw = F.log_softmax(out[:, :3], dim=1)
        sp = F.softplus
        return (logw,
                sp(out[:, 3]) + EPS,                         # sigma_F  (n,)
                sp(out[:, 4]) + ALPHA_MIN, sp(out[:, 5]) + EPS,   # alpha_L, theta_L
                sp(out[:, 6]) + ALPHA_MIN, sp(out[:, 7]) + EPS)   # alpha_R, theta_R


# --------------------------- mixture log-density ---------------------------- #
def _gamma_logpdf_torch(x, alpha, theta):
    """Differentiable log Gamma(x; shape=alpha, scale=theta); LOG_FLOOR where x<=0.

    alpha/theta may be scalar or (n,) tensors and broadcast against x (n,).
    """
    pos = x > 0
    x_safe = torch.where(pos, x, torch.ones_like(x))         # avoid log(0)/neg -> NaN grad
    logp = ((alpha - 1.0) * torch.log(x_safe) - x_safe / theta
            - alpha * torch.log(theta) - torch.lgamma(alpha))
    return torch.where(pos, logp, torch.full_like(x, LOG_FLOOR))


def mixture_logdens(dtheta, params):
    """Per-bout log predictive density (nats over DEGREES). params = MDN.forward output."""
    logw, sigmaF, aL, tL, aR, tR = params
    logNF = -0.5 * (LOG2PI + 2.0 * torch.log(sigmaF)) - dtheta ** 2 / (2.0 * sigmaF ** 2)
    logGL = _gamma_logpdf_torch(-dtheta, aL, tL)             # Left: support dtheta < 0
    logGR = _gamma_logpdf_torch(+dtheta, aR, tR)             # Right: support dtheta > 0
    comp = torch.stack([logw[:, 0] + logNF,
                        logw[:, 1] + logGL,
                        logw[:, 2] + logGR], dim=1)           # (n,3)
    return torch.logsumexp(comp, dim=1)                      # (n,)


def mdn_nll(model, X, y):
    """Mean negative log-likelihood over a batch (the training objective)."""
    return -mixture_logdens(y, model(X)).mean()


# ------------------------------ training loop ------------------------------- #
def fit_mlp(X, y, mode, hidden, *, epochs=60, lr=1e-3, batch_size=256, patience=8,
            val_frac=0.15, seed=0, init_emis=None, device=None, verbose=False):
    """Train an MDN by NLL with a held-out validation split + early stopping.

    Returns (model_on_cpu, history dict with 'train'/'val' loss lists).
    """
    device = device or _get_device()
    _seed_everything(seed)
    N = len(X)
    idx = np.random.permutation(N)
    n_val = max(1, int(val_frac * N))
    vi, ti = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(X[ti], device=device); yt = torch.tensor(y[ti], device=device)
    Xv = torch.tensor(X[vi], device=device); yv = torch.tensor(y[vi], device=device)

    model = MDN(X.shape[1], hidden, mode=mode, init_emis=init_emis).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val, best_state, no_imp = float("inf"), None, 0
    hist = {"train": [], "val": []}
    nb = int(np.ceil(len(ti) / batch_size))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ti), device=device)
        tot = 0.0
        for b in range(nb):
            bi = perm[b * batch_size:(b + 1) * batch_size]
            loss = mdn_nll(model, Xt[bi], yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        model.eval()
        with torch.no_grad():
            vloss = float(mdn_nll(model, Xv, yv).item())
        hist["train"].append(tot / max(1, nb)); hist["val"].append(vloss)
        if verbose:
            print(f"  epoch {ep:3d}  train {hist['train'][-1]:.4f}  val {vloss:.4f}")
        if vloss < best_val - 1e-4:
            best_val, no_imp = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to("cpu"), hist


# =========================================================================== #
#  Scoring helpers (log predictive density + 3-symbol PMF)
# =========================================================================== #
@torch.no_grad()
def predict_params(model, X):
    """Numpy per-bout mixture params (w (n,3), sigmaF, aL, tL, aR, tR each (n,))."""
    model.eval()
    n = X.shape[0]
    logw, sigmaF, aL, tL, aR, tR = model(torch.tensor(X))

    def col(z):
        return np.full(n, float(z)) if z.dim() == 0 else z.numpy().astype(float)

    return (np.exp(logw.numpy()), col(sigmaF), col(aL), col(tL), col(aR), col(tR))


def symbol_logq_full(w, sigmaF, aL, tL, aR, tR, dtheta0=DTHETA0, eps=1e-300):
    """(n,3) predictive log-PMF over {F,L,R} for a PER-BOUT-parameter mixture.

    Integrates each component over the +/-dtheta0 bins:
      Forward Normal(0,sigmaF): masses nL/nF/nR via the erf CDF.
      Left  Gamma(alpha_L,theta_L) on dtheta<0: GcL into F (0<mag<dtheta0), 1-GcL into L.
      Right Gamma(alpha_R,theta_R) on dtheta>0: GcR into F, 1-GcR into R.
    (This is the asymmetric per-bout generalisation of evaluate.symbol_logmass.)
    """
    from scipy.special import erf, gammainc
    sqrt2 = math.sqrt(2.0)
    Phi_hi = 0.5 * (1.0 + erf((dtheta0 / sigmaF) / sqrt2))
    Phi_lo = 0.5 * (1.0 + erf((-dtheta0 / sigmaF) / sqrt2))
    nF = Phi_hi - Phi_lo
    nL = Phi_lo
    nR = 1.0 - Phi_hi
    GcL = gammainc(aL, dtheta0 / tL)        # Left  magnitude CDF at dtheta0 (per bout)
    GcR = gammainc(aR, dtheta0 / tR)        # Right magnitude CDF at dtheta0 (per bout)
    qF = w[:, 0] * nF + w[:, 1] * GcL + w[:, 2] * GcR
    qL = w[:, 0] * nL + w[:, 1] * (1.0 - GcL)
    qR = w[:, 0] * nR + w[:, 2] * (1.0 - GcR)
    q = np.maximum(np.stack([qF, qL, qR], axis=1), eps)
    return np.log(q / q.sum(axis=1, keepdims=True))


# =========================================================================== #
#  Autoregressive generation
# =========================================================================== #
def _numpy_predictor(model):
    """Return fn(x_vec) -> (w (3,), sigmaF, aL, tL, aR, tR) doing the MDN forward in
    NUMPY. The autoregressive sampler calls the net once per bout; a numpy forward on
    this tiny net avoids torch per-call overhead (~50x faster) and is numerically
    identical to MDN.forward."""
    sd = model.state_dict()
    W1 = sd["body.0.weight"].numpy(); b1 = sd["body.0.bias"].numpy()
    W2 = sd["head.weight"].numpy();   b2 = sd["head.bias"].numpy()

    def softplus(z):
        return np.logaddexp(0.0, z)

    if model.mode == "basic":
        sig = float(softplus(sd["raw_sigma"].numpy())) + EPS
        aL = float(softplus(sd["raw_alpha"].numpy())) + ALPHA_MIN
        tL = float(softplus(sd["raw_theta"].numpy())) + EPS

        def fwd(x):
            h = np.maximum(W1 @ x + b1, 0.0)
            o = W2 @ h + b2
            w = np.exp(o - o.max()); w /= w.sum()
            return w, sig, aL, tL, aL, tL
    else:
        def fwd(x):
            h = np.maximum(W1 @ x + b1, 0.0)
            o = W2 @ h + b2
            lw = o[:3]; w = np.exp(lw - lw.max()); w /= w.sum()
            return (w, float(softplus(o[3])) + EPS,
                    float(softplus(o[4])) + ALPHA_MIN, float(softplus(o[5])) + EPS,
                    float(softplus(o[6])) + ALPHA_MIN, float(softplus(o[7])) + EPS)
    return fwd


def generate_mlp(model, rec, cfg_feat, dt_pool, dist_pool, rng):
    """Free-run rollout: feed generated dtheta history back; stimulus teacher-forced
    from `rec`. Returns dict(dtheta, symbol, dt_s, distance_px) of length rec.n."""
    n = rec.n
    K = cfg_feat["K"]
    dth_scale = cfg_feat["dth_scale"]
    stim = _stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                       phi_ref=cfg_feat.get("phi_ref", "ori"))  # teacher-forced
    fwd = _numpy_predictor(model)
    dth = np.zeros(n)
    for t in range(n):
        hist = np.zeros(K)
        for k in range(1, K + 1):
            if t - k >= 0:
                hist[k - 1] = dth[t - k]
        row = [hist / dth_scale]
        if cfg_feat["include_symbols"]:
            sb = np.zeros(3 * K)
            for k in range(1, K + 1):
                if t - k >= 0:
                    d = dth[t - k]
                    s = 0 if abs(d) < DTHETA0 else (1 if d < 0 else 2)
                    sb[3 * (k - 1) + s] = 1.0
            row.append(sb)
        row.append(stim[t])
        x = np.concatenate(row).astype(np.float32)
        w, sF, aL, tL, aR, tR = fwd(x)
        c = rng.choice(3, p=w)
        if c == 0:
            dth[t] = rng.normal(0.0, sF)
        elif c == 1:
            dth[t] = -rng.gamma(aL, tL)
        else:
            dth[t] = +rng.gamma(aR, tR)
        dth[t] = float(np.clip(dth[t], -180.0, 180.0))   # bound the fed-back angle (a reorientation is wrap-limited; stops rollout blow-up)
    return dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                dt_s=rng.choice(dt_pool, size=n), distance_px=rng.choice(dist_pool, size=n))


# =========================================================================== #
#  Save / load
# =========================================================================== #
def save_mlp(path, model, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_mlp(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = MDN(cfg["n_features"], cfg["hidden"], mode=cfg["mode"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


# =========================================================================== #
#  Evaluation-pipeline adapter (evaluate.GenerativeModel)
# =========================================================================== #
class MLPDensity(E.GenerativeModel):
    """Mixture-density MLP as a drop-in generative model for Metric A / B."""
    has_density = True
    has_symbol = True
    can_sample = True

    def __init__(self, mode="basic", K=5, stim_mode="simple", include_symbols=False,
                 hidden=64, epochs=60, lr=1e-3, patience=8, batch_size=256, seed=0):
        self.mode = mode
        self.name = f"mlp_{mode}"
        self.K = K
        self.stim_mode = stim_mode
        self.include_symbols = include_symbols
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.requires_absdict = (stim_mode == "rich")

    def fit(self, train_records, ctx):
        c = ctx.cfg or {}
        self.K = int(c.get("mlp_K", self.K))
        self.stim_mode = c.get("mlp_stim_mode", self.stim_mode)
        self.include_symbols = bool(c.get("mlp_include_symbols", self.include_symbols))
        self.hidden = int(c.get("mlp_hidden", self.hidden))
        self.epochs = int(c.get("mlp_epochs", self.epochs))
        self.lr = float(c.get("mlp_lr", self.lr))
        self.requires_absdict = (self.stim_mode == "rich")
        self._dt_pool, self._dist_pool = ctx.dt_pool, ctx.dist_pool
        self._rich_scales = ctx.rich_scales

        # train-fold-only scale + emission init (leak-free)
        alld = np.concatenate([r.dtheta for r in train_records])
        self._dth_scale = float(max(np.std(alld), 1.0))
        fwd = alld[np.abs(alld) < DTHETA0]
        turn = np.abs(alld[np.abs(alld) >= DTHETA0])
        s0 = float(max(np.std(fwd) if len(fwd) > 5 else 8.0, 0.5))
        m_, v_ = (turn.mean(), turn.var()) if len(turn) > 5 else (40.0, 400.0)
        a0 = float(max(m_ * m_ / max(v_, 1e-12), ALPHA_MIN))
        t0 = float(max(v_ / max(m_, 1e-12), 1.0))

        self._cfg_feat = dict(K=self.K, stim_mode=self.stim_mode,
                              include_symbols=self.include_symbols,
                              dth_scale=self._dth_scale, rich_scales=self._rich_scales)
        X, y = build_feature_sequences(train_records, self._cfg_feat)
        self._model, self.history = fit_mlp(
            X, y, self.mode, self.hidden, epochs=self.epochs, lr=self.lr,
            patience=self.patience, batch_size=self.batch_size, seed=self.seed,
            init_emis=(s0, a0, t0))
        self.config = dict(mode=self.mode, hidden=self.hidden, n_features=X.shape[1],
                           K=self.K, stim_mode=self.stim_mode,
                           include_symbols=self.include_symbols, dth_scale=self._dth_scale,
                           rich_scales=(None if self._rich_scales is None
                                        else np.asarray(self._rich_scales).tolist()))
        return self

    def score(self, rec):
        X = build_feature_matrix(rec, self._cfg_feat)
        w, sigmaF, aL, tL, aR, tR = predict_params(self._model, X)
        with torch.no_grad():
            logdens = mixture_logdens(
                torch.tensor(rec.dtheta.astype(np.float32)),
                self._model(torch.tensor(X))).numpy()
        if self.mode == "basic":
            slq = E.symbol_logmass(w, sigmaF[0], aL[0], tL[0], mu=0.0)
        else:
            slq = symbol_logq_full(w, sigmaF, aL, tL, aR, tR)
        return dict(logdens=logdens, symbol_logq=slq)

    def sample(self, rec, rng):
        return generate_mlp(self._model, rec, self._cfg_feat,
                            self._dt_pool, self._dist_pool, rng)


# =========================================================================== #
#  CLI
# =========================================================================== #
def main():
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bouts", type=Path, default=Path("data/bouts_GsizeMotionSingle.csv"))
    p.add_argument("--output", type=Path, default=Path("data/mlp_basic.pt"))
    p.add_argument("--mode", choices=["basic", "full"], default="basic")
    p.add_argument("--K", type=int, default=5, help="AR history window (bouts)")
    p.add_argument("--stim-mode", choices=["none", "simple", "rich", "rich_csv", "dot", "dotb"], default="simple")
    p.add_argument("--include-symbols", action="store_true")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-len", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-output", type=Path, default=None)
    p.add_argument("--n-gen", type=int, default=2000)
    args = p.parse_args()

    df = pd.read_csv(args.bouts)
    records = E.build_records(df, min_len=args.min_len)   # rich mode needs abs_dict/roi
    recs = list(records.values())
    print(f"loaded {len(recs)} fish ({sum(r.n for r in recs)} bouts, min_len={args.min_len})")

    ctx = E.make_context(recs, cfg=dict(mlp_epochs=args.epochs, mlp_hidden=args.hidden,
                                        mlp_K=args.K, mlp_stim_mode=args.stim_mode,
                                        mlp_lr=args.lr,
                                        mlp_include_symbols=args.include_symbols))
    m = MLPDensity(mode=args.mode, K=args.K, stim_mode=args.stim_mode,
                   include_symbols=args.include_symbols, hidden=args.hidden,
                   epochs=args.epochs, lr=args.lr, seed=args.seed)
    m.fit(recs, ctx)
    print(f"trained {m.name}: features D={m.config['n_features']}, "
          f"final val NLL={m.history['val'][-1]:.4f} ({len(m.history['val'])} epochs)")

    # in-sample mean bits/bout as a quick sanity readout
    ld = np.concatenate([m.score(r)["logdens"] for r in recs])
    print(f"in-sample: {np.mean(ld):.4f} nats/bout | {-np.mean(ld) / np.log(2):.4f} bits/bout")

    save_mlp(args.output, m._model, m.config)
    print("saved ->", args.output)

    if args.gen_output is not None:
        rng = np.random.default_rng(args.seed)
        gen = m.sample(recs[0], rng)
        pd.DataFrame({"bout_angle": gen["dtheta"][:args.n_gen]}).to_csv(args.gen_output, index=False)
        print(f"generated {min(args.n_gen, len(gen['dtheta']))} bouts -> {args.gen_output}")


if __name__ == "__main__":
    main()
