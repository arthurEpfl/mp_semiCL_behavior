"""The stimulus codes and feature helpers of the bout models.

A bout's stimulus enters the network as a short code of the dot group's CENTROID (``shoal_channels``), its bearing
from the fish's previous travel heading and its range, with the group's count and whether it moves; static dots also
get their own range x bearing block (``static_channels``). The codes of the thesis:

    shoal_simple  presence, motion, log count, log range, bearing sin / cos                   (6)   M1-M3
    shoal_int     + count x sin, moving x sin, moving x cos                                    (9)   M4, M5
    shoal_rx      + range x bearing per condition, linear in log range                         (14)  M6
    shoal         the range bumps: bearing terms on four Gaussian bumps in log range           (18)  D3, D4, F
    shoal_null / shoal_cond / shoal_rx_nn / shoal_rx_nm / shoal_rx_nr                           the stimulus ablations
    <code>_rzlo   the same code with the log range clipped at the 2 px floor instead of 17.5 px

Every channel is 0 on bouts without dots and O(1) by construction. ``_stim_block`` builds a record's block (training,
scoring) and ``emission._dot_stim_row`` one row from a generated pose (closed-loop rollouts), through the same
``shoal_channels``.
"""

import os

import numpy as np
import torch

# ----------------------------- constants ------------------------------------ #
DTHETA0 = 10.0                # deg: the forward / turn threshold of the bout symbols
LOG_FLOOR = -700.0            # log of ~1e-304, where an emission has no support
ALPHA_MIN = 1.0               # the Gamma shape floor (alpha >= 1)
EPS = 1e-12

# ------------------- 'shoal' stimulus block constants ----------------------- #
# Non-monotone RANGE tuning x group SIZE x MOTION. Real fish steer toward the shoal
# most strongly at 55-90 px (centroid distance), weakly beyond ~120 px, and REVERSE
# below ~25 px; a monotone proximity weight (the old 'dot' block's 1/(1+r/20))
# cannot represent a peak or a sign change.
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
# THE MAIN LINE: the centroid code built up from its simplest form. Both reuse
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
# 'shoal_null'   no stimulus columns at all (0) -- the online path still runs, so programme / record providers work
SHOAL_NULL_COLS = ()
# THE STIMULUS ABLATIONS of M6's S4x code:
#   'shoal_rx_nn'  no dot count: without log count, count x sin and count x range x sin (11)
#   'shoal_rx_nm'  no moving indicator: without the moving flag, moving x sin / cos and the moving AND static range x
#                  bearing products (static = shown and not moving, itself a moving-indicator distinction) (7)
SHOALS_RX_NN_COLS = ("act", "moving", "rz", "sin", "cos", "mv_sin", "mv_cos", "mv_rz_sin", "mv_rz_cos", "st_rz_sin",
                     "st_rz_cos")
SHOALS_RX_NM_COLS = ("act", "nz", "rz", "sin", "cos", "nz_sin", "nz_rz_sin")
#   'shoal_rx_nr'  range removed: without log range and the range x bearing products (8) -- S4i without rz; the
#                  aligned-and-far flag (range x cos) goes with it (stim_align off in the recipe)
#   'shoal_cond'   moving vs static only: the presence and moving flags, no bearing, no range, no count (2) -- the fish
#                  knows that dots are shown and whether they move, not where they are or how many
SHOALS_RX_NR_COLS = ("act", "moving", "nz", "sin", "cos", "nz_sin", "mv_sin", "mv_cos")
SHOAL_COND_COLS = ("act", "moving")
SHOAL_MODES = ("shoal", "shoal_simple", "shoal_int", "shoal_rx", "shoal_null", "shoal_rx_nn", "shoal_rx_nm",
               "shoal_rx_nr", "shoal_cond", "shoal" + RZLO_SUFFIX, "shoal_rx" + RZLO_SUFFIX)
_SHOAL_LC = np.log(np.asarray(SHOAL_RBF_PX, float))


def shoal_cols(stim_mode):
    """Canonical column names of a code -- the single source of truth."""
    if stim_mode.endswith(RZLO_SUFFIX):                  # the lower-clip variants keep their base mode's columns
        return shoal_cols(stim_mode[:-len(RZLO_SUFFIX)])
    return {"shoal": SHOAL_COLS, "shoal_simple": SHOALS_SIMPLE_COLS, "shoal_int": SHOALS_INT_COLS,
            "shoal_rx": SHOALS_RX_COLS, "shoal_null": SHOAL_NULL_COLS, "shoal_rx_nn": SHOALS_RX_NN_COLS,
            "shoal_rx_nm": SHOALS_RX_NM_COLS, "shoal_rx_nr": SHOALS_RX_NR_COLS,
            "shoal_cond": SHOAL_COND_COLS}[stim_mode]


def shoal_kw(stim_mode):
    """The ``shoal_channels`` keyword set of a code -- ONE place, read by the record builder (``_stim_block``) and the
    closed-loop row (``emission._dot_stim_row``) alike."""
    if stim_mode.endswith(RZLO_SUFFIX):                  # base mode's keywords + rz clipped at the 2 px floor
        return dict(shoal_kw(stim_mode[:-len(RZLO_SUFFIX)]), rz_lo=RZ_LO_FLOOR)
    if stim_mode in ("shoal_simple", "shoal_int", "shoal_rx"):
        return dict(use_n=True, single_knot=True, products=(stim_mode != "shoal_simple"),
                    range_products=(stim_mode == "shoal_rx"))
    if stim_mode == "shoal_null":
        return dict(null=True)
    if stim_mode == "shoal_rx_nn":
        return dict(use_n=False, single_knot=True, products=True, range_products=True)
    if stim_mode == "shoal_rx_nm":
        return dict(use_n=True, single_knot=True, products=True, range_products=True, moving_cols=False)
    if stim_mode == "shoal_rx_nr":
        return dict(use_n=True, single_knot=True, products=True, use_range=False)
    if stim_mode == "shoal_cond":
        return dict(cond_only=True)
    if stim_mode == "shoal":
        return dict(use_n=True)
    raise ValueError(f"unknown stim_mode {stim_mode!r}")


def shoal_channels(phi, r_px, n_dots, moving, use_n=True, single_knot=False, products=True, range_products=False,
                   moving_cols=True, null=False, use_range=True, cond_only=False, rz_lo=-1.0):
    """The stimulus channels of a code (``shoal_kw``), column order ``shoal_cols``. Vectorised AND scalar-safe, so
    ``_stim_block`` (record arrays, train / score time) and ``emission._dot_stim_row`` (ONE row rebuilt from the
    GENERATED pose during a closed-loop rollout) share this single code path.

    phi     egocentric bearing to the ACTIVE-DOT CENTROID (rad, + = fish's left); NaN when no dot is active.
    r_px    fish-to-centroid distance (px); NaN when no dot is active.
    n_dots  N = n_dots_left + n_dots_right, a block constant in {1,2,4,6,8}; 0 off-block.
    moving  1.0 on moving-stimulus bouts (block constant).
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
    if single_knot:                                  # the main line: ONE bump, 1 wherever dots are shown
        G = act[:, None].astype(float)
    else:
        z = -0.5 * ((lr[:, None] - _SHOAL_LC[None, :]) / SHOAL_RBF_W) ** 2       # (n, 4)
        z = z - z.max(axis=1, keepdims=True)
        G = np.exp(z)
        G = G / G.sum(axis=1, keepdims=True)
        G = G * act[:, None].astype(float)
    cols = [act.astype(float)[:, None]] + ([mv[:, None]] if moving_cols else [])
    if use_n:
        cols.append(nz[:, None])
    cols += ([rz[:, None]] if use_range else []) + [G * s[:, None], G * c[:, None]]
    if products and use_n:
        cols.append(G * (nz * s)[:, None])
    if products and moving_cols:
        cols += [(mv * s)[:, None], (mv * c)[:, None]]
    if range_products:                               # 'shoal_rx': range x bearing per condition, linear in rz
        if not (single_knot and products and use_range):
            raise ValueError("range_products belongs to the single-knot code ('shoal_rx')")
        st = act.astype(float) * (1.0 - mv)          # static dots: shown and not moving
        if moving_cols:                              # (the moving-indicator ablation drops both condition pairs)
            cols += [(mv * rz * s)[:, None], (mv * rz * c)[:, None], (st * rz * s)[:, None], (st * rz * c)[:, None]]
        if use_n:                                    # (the count ablation drops count x range x sin)
            cols.append((nz * rz * s)[:, None])
    out = np.concatenate(cols, axis=1).astype(float)
    if not use_range:
        _exp = SHOALS_RX_NR_COLS
    elif single_knot and range_products and not use_n:
        _exp = SHOALS_RX_NN_COLS
    elif single_knot and range_products and not moving_cols:
        _exp = SHOALS_RX_NM_COLS
    elif single_knot:
        _exp = SHOALS_RX_COLS if range_products else (SHOALS_INT_COLS if products else SHOALS_SIMPLE_COLS)
    else:
        _exp = SHOAL_COLS
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
    so the generator's closed-loop row (``joint``) and the record block agree by
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


# --------------------------- seeding --------------------------------------- #
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


# --------------------------- feature blocks -------------------------------- #
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


def _stim_block(rec, stim_mode, rich_scales=None, phi_ref="ori"):
    """(n, s_dim) stimulus features of a record (teacher-forceable). ``phi_ref='travel'`` reads the dot bearing
    referenced to the PREVIOUS travel heading (``rec.stim_angle_travel``), the frame the closed-loop generator's
    per-bout recompute (``MX._dot_stim_row``) uses; ``rich_scales`` is unused (kept in the configs)."""
    if stim_mode not in SHOAL_MODES:
        raise ValueError(f"unknown stim_mode {stim_mode!r}")
    if phi_ref == "travel":
        if getattr(rec, "stim_angle_travel", None) is None:
            raise RuntimeError("phi_ref='travel' needs rec.stim_angle_travel "
                               "(bouts CSV via features.build_border_records).")
        _phi = np.asarray(rec.stim_angle_travel, float)
    else:
        _phi = np.asarray(rec.stim_angle, float)
    if getattr(rec, "dot_dist_px", None) is None or getattr(rec, "n_dots", None) is None:
        raise RuntimeError(f"stim_mode={stim_mode!r} needs rec.dot_dist_px and rec.n_dots "
                           "(features.build_border_records).")
    return shoal_channels(_phi, rec.dot_dist_px, rec.n_dots, np.asarray(rec.moving, float), **shoal_kw(stim_mode))


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

