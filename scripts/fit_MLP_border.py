"""Border-aware MLP density model (per-bout) -- a drop-in variant of ``fit_MLP``.

Mirrors ``scripts/fit_MLP.py`` (same ``basic``/``full`` MDN, same ``stim_mode='simple'``
stimulus scalar) but adds a physical-dish **border covariate** to every input row:

    [dist_to_border_norm, sin(psi), cos(psi)]

where ``psi`` is the egocentric bearing from the fish to the physical dish centre
(``psi = wrap(atan2(cy - y, cx - x) - f_ori)``) -- exactly analogous to the stimulus
bearing ``b = sin(phi)``. Distance alone can't say *which way* to turn; the sin/cos of
the inward bearing supply the direction.

Everything numerical (MDN, training loop, scoring/gamma maths, numpy predictor) is
reused from ``fit_MLP``; only the feature-assembly path and the record builder are new,
so ``fit_MLP.py`` and ``evaluate.py`` are untouched. Use ``MLPBorderDensity`` next to
``fit_MLP.MLPDensity`` in the same evaluate harness to A/B the wall covariate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import evaluate as E
import fit_MLP as M
from segment_bouts import PX_PER_MM, R0_MM, load_physical_dish

# raw physical ROIdef (true 100 mm dish geometry) -> per-fish (cx, cy, radius)
_DEFAULT_ROI = (Path(__file__).resolve().parents[1]
                / "data" / "data" / "Original_data" / "ROIdef2025-02-20T12_09_16.csv")

DISH_MM = 100.0               # physical dish DIAMETER (Larsch rig); r/R -> mm via 50*(1 - r/R)


def _wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def wall_mm(u):
    """Distance from the dish wall in mm, given u = r/R.

    Uses each record's OWN radius implicitly: the dish is 100 mm across for every fish, so
    ``50 * (1 - u)`` is exact per fish with no px/mm constant and no per-dish ROI-detection
    noise entering the conversion. Commensurate with the metric's ``r/R > 0.85`` shell
    (= 7.5 mm) and with the measured real occupancy (bout-level max r/R 0.990 = 0.5 mm;
    P(r/R > 0.95) = 2.8e-5; nothing at all between 1 and 2 mm of the wall)."""
    return 0.5 * DISH_MM * (1.0 - np.asarray(u, float))


def snap_dt(t, fps=30.0):
    """Quantise a generated inter-bout interval to the camera's frame lattice (multiples of 1/fps).

    Real dt is ``diff(frame_onset)/FPS`` -- exact lattice points -- while the emitted log-normal is
    continuous, and a random forest separates real from generated windows on the lattice alone
    (dt_median: isolated AUC 0.95, monotone-rank AUC 0.46 -- a measurement artefact, not
    behaviour). Passing the sample through the same measurement operator as the data removes it.
    ``max(.,1)`` guards the 1e-3-clipped emission from rounding to zero, which would break every
    ``finite & > 0`` mask downstream."""
    return max(round(float(t) * fps), 1) / fps


# =========================================================================== #
#  Records: baseline FishRecords + attached pose / border / dish
# =========================================================================== #
def build_border_records(df, roi_path=_DEFAULT_ROI, min_len=10):
    """Build the canonical per-fish records (via ``evaluate.build_records``) and attach
    the per-bout pose + border fields the wall covariate needs:
    ``x_px, y_px, f_ori_rad, dist_to_border_norm`` (arrays) and the physical dish
    ``dish_cx, dish_cy, dish_R`` (scalars, from ``load_physical_dish``).

    Alignment is guaranteed by re-grouping ``df`` with the SAME logic as
    ``evaluate.build_records`` (sort by (fish_id, segment_id, bout_index), group by
    (fish_id, segment_id), keep ``good = isfinite(dtheta_deg)``, drop groups shorter
    than ``min_len``). ``FishRecord`` is a plain dataclass, so the extra attributes
    attach cleanly; baseline models never read them.
    """
    records = E.build_records(df, min_len=min_len)      # keyed by fish_id
    dish = load_physical_dish(roi_path)                 # index = fish_id -> cx, cy, radius
    group_cols = ["fish_id", "segment_id"]
    for keys, g in df.sort_values(group_cols + ["bout_index"]).groupby(group_cols, sort=False):
        fid = int(keys[0] if isinstance(keys, tuple) else keys)
        g = g.sort_values("bout_index")
        good = np.isfinite(g["dtheta_deg"].to_numpy(float))
        if good.sum() < min_len:
            continue
        rec = records.get(fid)
        if rec is None:                                 # was skipped by build_records
            continue
        rec.x_px = g["x_px"].to_numpy(float)[good]
        rec.y_px = g["y_px"].to_numpy(float)[good]
        rec.f_ori_rad = g["f_ori_rad"].to_numpy(float)[good]
        if "heading_rad" in g.columns:                            # travel direction of bout t (onset_t -> onset_{t+1})
            rec.heading_rad = g["heading_rad"].to_numpy(float)[good]
            # Dot bearing re-referenced to the PREVIOUS travel heading (for phi_ref='travel').
            # The absolute bearing recovers exactly as stim_angle + f_ori (segment_bouts measures
            # stim_angle relative to f_ori at the same onset), so no information is lost; NaN
            # (no active dot) propagates. theta_{t-1} = heading_rad[t] - dtheta[t], row-local.
            _prev = _wrap_pi(rec.heading_rad - np.radians(g["dtheta_deg"].to_numpy(float)[good]))
            rec.stim_angle_travel = _wrap_pi(g["stim_angle_rad"].to_numpy(float)[good]
                                             + rec.f_ori_rad - _prev)
        rec.dist_to_border_norm = g["dist_to_border_norm"].to_numpy(float)[good]
        rec.frame_onset = g["frame_onset"].to_numpy(int)[good]     # per-bout frame index (for the video bridge)
        if "dot_dist_px" in g.columns:                            # per-bout dot geometry (add_dot_geometry.py)
            rec.dot_dist_px = g["dot_dist_px"].to_numpy(float)[good]
            rec.dot_cx = g["dot_cx"].to_numpy(float)[good]
            rec.dot_cy = g["dot_cy"].to_numpy(float)[good]
            rec.dot_prox = g["dot_prox"].to_numpy(float)[good]
        if "n_dots_left" in g.columns:                            # group size N (stim_mode='shoal')
            rec.n_dots = (np.nan_to_num(g["n_dots_left"].to_numpy(float))
                          + np.nan_to_num(g["n_dots_right"].to_numpy(float)))[good]
        rec.dish_cx = float(dish["cx"].loc[fid])
        rec.dish_cy = float(dish["cy"].loc[fid])
        rec.dish_R = float(dish["radius"].loc[fid])
    return records


# =========================================================================== #
#  Feature builder -- fit_MLP's blocks + the new border block (same chokepoint)
# =========================================================================== #
# ---- border basis: distance-tuned wall avoidance (border_mode='basis') ---- #
# Measured: real fish turn toward the dish centre on 71-73% of bouts inside the wall shell
# (dist_to_border_norm 0.05-0.20), falling to 57% mid-dish. The raw 3-channel block gives the
# net one linear knob, and it learns a nearly FLAT 0.61->0.56 -- a 10-12 point deficit exactly
# in the shell. Consequences: it heads outward on 64% of at-wall bouts, is 2.4x more likely
# than a real fish to still be at the wall 5 bouts later, and occupies r/R>0.85 on 7.8% of
# bouts vs 1.2% real (r/R>0.95 on 2.8% where real fish NEVER go). Same defect shape as the
# shoal range tuning, so the same remedy: a softmax basis over the covariate, crossed with the
# bearing, letting the response be sharply distance-tuned instead of monotone.
BORDER_KNOTS = (0.03, 0.10, 0.25, 0.55)      # dist_to_border_norm (0 = wall, 1 = centre)
BORDER_W = 0.55                               # basis width in natural-log units
BORDER_MIN = 0.005                            # floor before log (dnorm reaches 0 exactly)
BORDER_COLS_RAW = ("dnorm", "sin_psi", "cos_psi")
BORDER_COLS_BASIS = ("dnorm", "bs1", "bs2", "bs3", "bs4", "bc1", "bc2", "bc3", "bc4")
_BORDER_LC = np.log(np.asarray(BORDER_KNOTS, float))


def border_cols(cfg_feat):
    return (BORDER_COLS_BASIS if (cfg_feat or {}).get("border_mode") == "basis"
            else BORDER_COLS_RAW)


def border_channels(dnorm, psi, mode="raw"):
    """(n, 3) raw or (n, 9) basis border channels. Vectorised AND scalar-safe so the
    train-time block and the closed-loop per-bout recompute share ONE code path -- the
    layout is otherwise replicated by hand in three places with nothing tying them together.

    dnorm  (R - r)/R in [0, 1]: 0 at the wall, 1 at the dish centre.
    psi    egocentric bearing to the dish centre (rad).
    """
    dn = np.atleast_1d(np.asarray(dnorm, float))
    ps = np.atleast_1d(np.asarray(psi, float))
    s = np.nan_to_num(np.sin(ps)); c = np.nan_to_num(np.cos(ps))
    dn = np.nan_to_num(dn)
    if mode != "basis":
        return np.stack([dn, s, c], axis=1).astype(np.float32)
    ld = np.log(np.clip(dn, BORDER_MIN, None))
    z = -0.5 * ((ld[:, None] - _BORDER_LC[None, :]) / BORDER_W) ** 2
    z = z - z.max(axis=1, keepdims=True)
    G = np.exp(z); G = G / G.sum(axis=1, keepdims=True)      # sum_j = 1 -> flat-extrapolates
    out = np.concatenate([dn[:, None], G * s[:, None], G * c[:, None]], axis=1)
    assert out.shape[1] == len(BORDER_COLS_BASIS), out.shape
    return out.astype(np.float32)


def _travel_ref(rec):
    """(n,) PREVIOUS travel heading theta_{t-1} (rad) -- the angular reference the closed-loop
    rollout actually presents when it computes bearings from its accumulated heading ``h``.

    Exact per-row identity from the segmentation (segment_bouts.py): ``dtheta_t = wrap(theta_t -
    theta_{t-1})`` and ``heading_rad[t] = theta_t``, so ``theta_{t-1} = heading_rad[t] - dtheta_t``.
    Row-local, so it is immune to segment boundaries and dropped bouts. This matters because the
    tracked body orientation ``f_ori_rad`` differs from the travel heading by 40.6 deg median
    (10.3% of bouts by >90 deg -- uncorrected body-axis flips): training a bearing on f_ori and
    querying it at rollout with the clean travel heading is errors-in-variables attenuation of
    exactly the response the border block exists to carry.

    Known residual: on the few percent of rollout bouts that BOUNCE (wall_mode='reflect'),
    the internal post-bounce heading is not the chord of the realised displacement, so the
    train/rollout identity is approximate on exactly those bouts. Inherent to any guardrail --
    real fish never bounce, so no training frame exists for it; the fix still removes the 40-deg
    noise from the ~97% of bouts that do not touch the wall."""
    if getattr(rec, "heading_rad", None) is None:
        raise RuntimeError("psi_ref='travel' needs rec.heading_rad "
                           "(bouts CSV column 'heading_rad' via build_border_records).")
    hd = np.asarray(rec.heading_rad, float)
    return _wrap_pi(hd - np.radians(np.asarray(rec.dtheta, float)))


def _border_block(rec, cfg_feat):
    """(n, 3) raw ``[dist_to_border_norm, sin(psi), cos(psi)]`` or, with
    ``cfg_feat['border_mode']='basis'``, (n, 9) distance-tuned wall channels.
    From FishRecord fields only (teacher-forceable). psi = bearing to the dish centre.

    ``cfg_feat['psi_ref']='travel'`` references psi to the PREVIOUS travel heading (see
    ``_travel_ref``) instead of the raw body orientation -- matching what the closed-loop
    generator feeds the net. Key absent (every archived checkpoint) = f_ori, bit-identical."""
    x = np.asarray(rec.x_px, float)
    y = np.asarray(rec.y_px, float)
    ori = (_travel_ref(rec) if (cfg_feat or {}).get("psi_ref") == "travel"
           else np.asarray(rec.f_ori_rad, float))
    psi = _wrap_pi(np.arctan2(rec.dish_cy - y, rec.dish_cx - x) - ori)
    dnorm = np.asarray(rec.dist_to_border_norm, float)
    return border_channels(dnorm, psi, (cfg_feat or {}).get("border_mode", "raw"))


def _adapt_block(rec, cfg_feat):
    """(n, 2) sensory-adaptation block ``[prox, prox - a]`` where ``prox`` = fish-to-dot proximity
    (``rec.dot_prox``, Phase-1) and ``a`` is its causal leaky integral
    ``a[t] = λ a[t-1] + (1-λ) prox[t]``. Feeds the model the CHANGE in proximity relative to a
    habituating baseline, so it can learn to DISENGAGE from a sustained stimulus (sensory adaptation;
    cf. bacterial-chemotaxis perfect adaptation -- the response tracks the temporal derivative, not
    the absolute level, so the fish doesn't get stuck orbiting a held dot). λ = ``cfg_feat['adapt_lambda']``."""
    if getattr(rec, "dot_prox", None) is None:
        raise RuntimeError("stim_mode adapt block needs rec.dot_prox "
                           "(run scripts/add_dot_geometry.py, then build_border_records).")
    prox = np.nan_to_num(np.asarray(rec.dot_prox, float))
    lam = float(cfg_feat.get("adapt_lambda", 0.6))
    n = len(prox)
    a = np.empty(n)
    if n:
        a[0] = prox[0]
        for t in range(1, n):
            a[t] = lam * a[t - 1] + (1.0 - lam) * prox[t]
    return np.stack([prox, prox - a], axis=1).astype(np.float32)


# =========================================================================== #
#  Stimulus BLOCK structure -- one definition, shared by the features and the metrics
# =========================================================================== #
_FPS = 30.0
STIMHIST_CYCLE_S = 110.0       # one 60 s stimulus block + one 50 s pause (the rig's programme)
STIMHIST_AGE_S = 330.0         # three cycles: the scale over which an older block is 'recent'


def block_onsets(rec, fps=_FPS):
    """``(onset_frames (n,), block_id (n,), frames (n,))`` -- the FRAME at which the bout's
    current block began, a monotone block counter, and the bout's own onset frame.

    A block boundary is any change in the stimulus condition (N, moving). THIS IS THE ONE
    DEFINITION: ``generative_metrics._time_in_block`` (which powers ``dot_dist_bias_late``)
    and ``_stimhist_block`` (which powers the K_stim features) both call it, so a feature and
    the metric that judges it can never drift apart.

    Returns FRAMES, not seconds, so the caller divides exactly once. Dividing here and again in
    the caller moves ``_time_in_block`` by ~1e-12 s, which is nothing physically but is compared
    against the hard ``LATE_BLOCK_S = 40.0`` threshold -- and a bout sitting exactly on it would
    change side, silently perturbing an archived metric."""
    n = rec.n
    fr = np.asarray(getattr(rec, "frame_onset", np.arange(n) * fps), float)
    nd = np.asarray(getattr(rec, "n_dots", np.zeros(n)), float)
    mv = np.asarray(rec.moving, float)
    cond = nd * 10.0 + mv
    start = np.empty(n); bid = np.zeros(n, dtype=np.int64)
    cur = fr[0] if n else 0.0
    k = 0
    for t in range(n):
        if t > 0 and cond[t] != cond[t - 1]:
            cur = fr[t]; k += 1
        start[t] = cur; bid[t] = k
    return start, bid, fr


def stimhist_cols(K_stim):
    """Column names for ``_stimhist_block``: 1 + 4*(K_stim-1)."""
    cols = ["tib"]
    for j in range(1, int(K_stim)):
        cols += [f"act_m{j}", f"mv_m{j}", f"nz_m{j}", f"age_m{j}"]
    return tuple(cols)


def _stimhist_block(rec, cfg_feat):
    """(n, 1 + 4*(K_stim-1)) STIMULUS MEMORY: where we are in the current block/pause cycle,
    plus the CONDITION of the previous ``K_stim - 1`` stimulus onsets.

    The model has never had any of this. ``M._stim_block`` is a pure per-bout snapshot and every
    shoal channel is exactly 0 during a pause, so the net cannot know whether a block just ended,
    how long it has been running, or what it saw last -- while the METRICS have known since
    ``LATE_BLOCK_S`` was added that block phase matters (``dot_dist_bias`` is ~2/3 a block-onset
    placement artefact).

    An "onset" here is the start of a block with dots present (pauses are not onsets, so K_stim=3
    really means three stimulus presentations, not one presentation and two pauses). Slot 0 is the
    most recent onset at or before this bout -- which during a pause is the block that just ended,
    so the memory survives the gap. Its CONDITION is already carried by the shoal block's
    act/moving/nz, so only its AGE is added here (``tib``, on the 110 s cycle scale); the older
    slots carry the full condition.

    Teacher-forceable: every channel is a function of the record's own block programme, so the
    closed-loop generators index it by bout exactly as they already index ``M._stim_block``."""
    n = rec.n
    Ks = int(cfg_feat.get("K_stim", 1))
    if Ks <= 1 or n == 0:
        return np.zeros((n, 0), dtype=np.float32)
    start_fr, bid, fr = block_onsets(rec)
    start_s = start_fr / _FPS
    t_on = fr / _FPS
    nd = np.nan_to_num(np.asarray(getattr(rec, "n_dots", np.zeros(n)), float))
    mv = np.nan_to_num(np.asarray(rec.moving, float))
    # one row per block: (onset_s, is_stim, moving, nz)
    first = np.concatenate([[True], bid[1:] != bid[:-1]])
    b_on = start_s[first]
    b_act = (nd[first] > 0).astype(float)
    b_mv = mv[first]
    b_nz = np.log2(np.clip(nd[first], 1.0, None)) / np.log2(M.SHOAL_N_MAX)
    keep = b_act > 0                                   # STIMULUS onsets only
    s_on, s_mv, s_nz = b_on[keep], b_mv[keep], b_nz[keep]
    # index of the most recent stimulus onset at or before each bout (-1 before the first one)
    k = np.searchsorted(s_on, t_on, side="right") - 1
    out = np.zeros((n, 1 + 4 * (Ks - 1)), dtype=np.float32)
    has0 = k >= 0
    out[has0, 0] = np.clip((t_on[has0] - s_on[k[has0]]) / STIMHIST_CYCLE_S, 0.0, 1.0)
    out[~has0, 0] = 1.0                                # before the first block: 'long ago'
    for j in range(1, Ks):
        c = 1 + 4 * (j - 1)
        idx = k - j
        ok = idx >= 0
        out[ok, c + 0] = 1.0
        out[ok, c + 1] = s_mv[idx[ok]]
        out[ok, c + 2] = s_nz[idx[ok]]
        out[ok, c + 3] = np.clip((t_on[ok] - s_on[idx[ok]]) / STIMHIST_AGE_S, 0.0, 1.0)
        out[~ok, c + 3] = 1.0
    assert out.shape[1] == len(stimhist_cols(Ks)), out.shape
    return out


def build_feature_matrix(rec, cfg_feat):
    """(n, D) input matrix: fit_MLP's history (+symbols) + stim (+ the K_stim stimulus-memory
    block), then the border block when ``cfg_feat['use_border']`` and the adaptation block when
    ``cfg_feat['use_adapt']``.
    Column order MUST match the generators (``generate_mlp*`` / ``generate_mix*``)."""
    blocks = [M._history_block(rec.dtheta, cfg_feat["K"], cfg_feat["dth_scale"])]
    if cfg_feat["include_symbols"]:
        blocks.append(M._symbol_block(rec.symbol, cfg_feat["K"]))
    blocks.append(M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                                phi_ref=cfg_feat.get("phi_ref", "ori")))
    if int(cfg_feat.get("K_stim", 1)) > 1:
        blocks.append(_stimhist_block(rec, cfg_feat))
    if cfg_feat.get("use_border"):
        blocks.append(_border_block(rec, cfg_feat))
    if cfg_feat.get("use_adapt"):
        blocks.append(_adapt_block(rec, cfg_feat))
    if cfg_feat.get("use_kin"):
        blocks.append(_kin_history_block(rec, cfg_feat))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _kin_history_block(rec, cfg_feat):
    """(n, 2K) AUTOREGRESSIVE kinematic history: prev-K z-scored ``log(distance_px)`` and ``log(dt_s)``.
    The real bout-kinematics autocorrelation (lag-1 ~ +0.22 distance / +0.10 dt) lives HERE -- without a
    kinematic history a distance/dt head just relearns the marginal (distance ⊥ turn/stimulus) and samples
    i.i.d. Uses a SHORT lag ``cfg_feat['K_kin']`` (default 3): the real acf is ~lag-1, so few lags suffice and
    over-long history drives AR over-persistence/runaway. Scales fixed on the train fold (leak-free)."""
    Kk = int(cfg_feat.get("K_kin", cfg_feat["K"]))
    dm, ds, tm, ts = cfg_feat["kin_scales"]          # (dist log-mean, dist log-std, dt log-mean, dt log-std)
    dist = np.asarray(rec.distance_px, float); dtv = np.asarray(rec.dt_s, float)
    ldist = np.nan_to_num((np.log(np.clip(dist, 1e-6, None)) - dm) / ds)   # last-bout NaN -> 0 (= mean)
    ldt = np.nan_to_num((np.log(np.clip(dtv, 1e-6, None)) - tm) / ts)
    n = len(dist)
    H = np.zeros((n, 2 * Kk), dtype=np.float32)
    for k in range(1, Kk + 1):
        H[k:, k - 1] = ldist[:-k]                     # cols 0..Kk-1   = prev log-distance (lags 1..Kk)
        H[k:, Kk + k - 1] = ldt[:-k]                  # cols Kk..2Kk-1 = prev log-dt
    return H


def build_feature_sequences(records, cfg_feat):
    """Stack (X (N,D), y (N,)) over records. y = dtheta target (deg)."""
    Xs, ys = [], []
    for r in records:
        Xs.append(build_feature_matrix(r, cfg_feat))
        ys.append(r.dtheta.astype(np.float32))
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


# =========================================================================== #
#  NOISE AUGMENTATION -- perturb the POSE, correct the target back to the truth
# =========================================================================== #
_NOISE_STIM_OK = ("none", "dot", "dotb", "shoal", "shoalr", "shoalf")

# Sized by measurement, not by analogy. The knob trades COVERAGE of drifted states against
# DISTORTION of the conditional the model is meant to learn, and the two pull in opposite
# directions. Measured over 6 fish (real turn sd 56.96 deg, median step 15.03 px):
#
#   sigma_pos   sd(dtheta~ - dtheta)   / real turn sd   sd(dnorm shift)   sign agreement
#      1 px           6.9 deg               0.12            0.006             0.750
#      2 px          14.2 deg               0.25            0.011             0.750
#      4 px          31.0 deg               0.54            0.022             0.751
#      8 px          59.6 deg               1.05            0.044             0.737
#     12 px          76.3 deg               1.34            0.066             0.713
#
# At 8 px the corrective signal is LARGER than the behaviour it is supposed to decorate, so the
# net would spend most of its capacity learning to undo a jump no rollout produces in one bout.
# 4 px (1.1 mm, 0.023 R) keeps the correction at ~half the real turn sd while still straddling
# the innermost border-basis knot (dnorm 0.03 = 5.3 px), which is the channel the drift actually
# corrupts. sigma_ori 15 deg is modest next to the 40.6 deg median f_ori-vs-travel discrepancy
# that already exists in the data. THIS IS THE OBVIOUS THING TO SWEEP if the arm is promising.
NOISE_AUG_DEFAULT = {"sigma_pos": 4.0, "sigma_ori": 15.0, "mode": "concat", "frac": 0.5}


def noisy_record(rec, cfg_feat, rng):
    """A shallow copy of ``rec`` whose POSE is perturbed and whose every downstream field --
    inputs AND targets -- is re-derived from that perturbed pose.

    One-step MLE is teacher-forced on the REAL pose; the free-running rollout feeds the net its
    own drifted pose, which it has never been trained on (measured: mean r/R off by 0.03-0.05 R
    before any containment patch). This is the standard remedy -- corrupt the input state and
    ADJUST THE TARGET so the model learns to correct back to the true next state. It is the
    training-noise scheme of Sanchez-Gonzalez et al. 2020 (GNS) and Pfaff et al. 2021
    (MeshGraphNets); the same idea is DART (Laskey et al. 2017) in imitation learning and
    Data-as-Demonstrator (Venkatraman et al. 2015) for time series. Bishop 1995 is why the target
    correction is the load-bearing half: input noise WITHOUT it is only Tikhonov smoothing and
    does nothing about drift.

    PERTURB THE POSE, NOT THE CHANNELS. Every input the closed loop recomputes for itself is a
    function of the pose -- the border distance and bearing, the shoal bearing and range -- and so
    is every target. Drawing one (eps_x, eps_y, eps_ori) per bout and re-deriving all of them
    therefore (a) keeps the channels mutually CONSISTENT (a jittered border distance that
    disagreed with its own bearing is a state the rollout can never be in), and (b) gives the
    history blocks correlated, random-walk-like noise for free, since ``dtheta`` and
    ``distance_px`` are themselves re-derived and then windowed by ``_history_block`` /
    ``_kin_history_block``. GNS makes exactly this point about i.i.d. vs accumulated noise.

    Targets, corrected back to the TRUE next onset (this is the whole mechanism):
        theta~[t]    = atan2(y[t+1] - y~[t], x[t+1] - x~[t])
        dtheta~[t]   = wrap(theta~[t] - theta~[t-1])
        distance~[t] = hypot(x[t+1] - x~[t], y[t+1] - y~[t])
    ``dt`` is untouched (it is not a function of position). The true next onset is exact for
    EVERY bout including the last, because ``(distance_px[t], heading_rad[t])`` reconstruct it
    (verified against the CSV to 7e-15 px).

    The stimulus PROGRAMME is real and is never perturbed -- only the fish's own pose relative to
    it. ``dot_cx``/``dot_cy``/``n_dots``/``moving``/``frame_onset`` all pass through unchanged.
    """
    na = cfg_feat.get("noise_aug") or {}
    s_pos = float(na.get("sigma_pos", 0.0))
    s_ori = float(na.get("sigma_ori", 0.0))
    if cfg_feat.get("stim_mode") not in _NOISE_STIM_OK:
        raise ValueError(f"noise_aug does not know how to re-derive stim_mode "
                         f"{cfg_feat.get('stim_mode')!r} from a perturbed pose "
                         f"(supported: {_NOISE_STIM_OK})")
    import copy as _copy
    r2 = _copy.copy(rec)
    n = rec.n
    x = np.asarray(rec.x_px, float); y = np.asarray(rec.y_px, float)
    hd = np.asarray(rec.heading_rad, float)
    dth = np.asarray(rec.dtheta, float)
    d0 = np.asarray(rec.distance_px, float)
    # TRUE next onset for every bout, last one included (exact from distance + heading).
    xn = np.empty(n); yn = np.empty(n)
    if n > 1:
        xn[:-1] = x[1:]; yn[:-1] = y[1:]
    xn[-1] = x[-1] + d0[-1] * np.cos(hd[-1]); yn[-1] = y[-1] + d0[-1] * np.sin(hd[-1])

    ez = np.zeros(n)
    ex = rng.normal(0.0, s_pos, n) if s_pos > 0 else ez
    ey = rng.normal(0.0, s_pos, n) if s_pos > 0 else ez
    eo = rng.normal(0.0, np.radians(s_ori), n) if s_ori > 0 else ez
    xt = x + ex; yt = y + ey

    th2 = np.arctan2(yn - yt, xn - xt)
    d2 = np.hypot(xn - xt, yn - yt)
    th_prev = np.empty(n)
    # Bout 0 has no predecessor to perturb, so its angular reference stays the real one -- the
    # exact row-local identity _travel_ref uses. Everything after references the perturbed chain,
    # which keeps wrap(th2 - radians(dth2)) == th_prev, i.e. _travel_ref(r2) is self-consistent.
    th_prev[0] = _wrap_pi(hd[0] - np.radians(dth[0]))
    if n > 1:
        th_prev[1:] = th2[:-1]
    dth2 = np.degrees(_wrap_pi(th2 - th_prev))

    r2.dtheta = dth2.astype(float)
    r2.distance_px = d2.astype(float)
    r2.symbol = E.symbols_from_dtheta(dth2)
    r2.x_px = xt; r2.y_px = yt
    r2.heading_rad = th2
    r2.f_ori_rad = _wrap_pi(np.asarray(rec.f_ori_rad, float) + eo)
    # The fish cannot be outside its own dish, so clamp rather than emit a negative distance the
    # border basis has no support for.
    u2 = np.hypot(xt - rec.dish_cx, yt - rec.dish_cy) / rec.dish_R
    r2.dist_to_border_norm = np.clip(1.0 - u2, 0.0, 1.0)
    if getattr(rec, "dot_cx", None) is not None:
        ddx = np.asarray(rec.dot_cx, float) - xt
        ddy = np.asarray(rec.dot_cy, float) - yt
        bear = np.arctan2(ddy, ddx)                       # absolute bearing to the dot centroid
        r2.dot_dist_px = np.hypot(ddx, ddy)
        r2.dot_prox = 1.0 / (1.0 + (r2.dot_dist_px / PX_PER_MM) / R0_MM)
        r2.stim_angle = _wrap_pi(bear - r2.f_ori_rad)
        r2.stim_angle_travel = _wrap_pi(bear - th_prev)
    return r2


def build_noisy_pair(rec, cfg_feat, rng):
    """``(X~, dtheta~, distance~, dt)`` for one record -- the augmented training pair."""
    r2 = noisy_record(rec, cfg_feat, rng)
    return (build_feature_matrix(r2, cfg_feat), np.asarray(r2.dtheta, float),
            np.asarray(r2.distance_px, float), np.asarray(r2.dt_s, float))


def build_noisy_sequences(records, cfg_feat, rng):
    """Stack ``build_noisy_pair`` over records, in the SAME row order as
    ``[build_feature_matrix(r) for r in records]`` -- so a train/val index split computed on the
    clean matrix selects the corresponding noisy rows."""
    Xs, ya, yd, yt = [], [], [], []
    for r in records:
        a, b, c, d = build_noisy_pair(r, cfg_feat, rng)
        Xs.append(a); ya.append(b); yd.append(c); yt.append(d)
    return (np.concatenate(Xs, 0), np.concatenate(ya), np.concatenate(yd), np.concatenate(yt))


# =========================================================================== #
#  Autoregressive generation (copy of fit_MLP.generate_mlp + teacher-forced border)
# =========================================================================== #
def generate_mlp(model, rec, cfg_feat, dt_pool, dist_pool, rng):
    """Free-run rollout: generated dtheta history fed back; stimulus AND border
    covariate teacher-forced from ``rec`` (like fit_MLP). The border row is appended
    in the SAME column order as ``build_feature_matrix``."""
    n = rec.n
    K = cfg_feat["K"]
    dth_scale = cfg_feat["dth_scale"]
    stim = M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                         phi_ref=cfg_feat.get("phi_ref", "ori"))   # teacher-forced
    border = _border_block(rec, cfg_feat) if cfg_feat.get("use_border") else np.zeros((n, 0))
    fwd = M._numpy_predictor(model)
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
                    s = 0 if abs(d) < M.DTHETA0 else (1 if d < 0 else 2)
                    sb[3 * (k - 1) + s] = 1.0
            row.append(sb)
        row.append(stim[t])
        row.append(border[t])
        x = np.concatenate(row).astype(np.float32)
        w, sF, aL, tL, aR, tR = fwd(x)
        c = rng.choice(3, p=w)
        if c == 0:
            dth[t] = rng.normal(0.0, sF)
        elif c == 1:
            dth[t] = -rng.gamma(aL, tL)
        else:
            dth[t] = +rng.gamma(aR, tR)
        dth[t] = float(np.clip(dth[t], -180.0, 180.0))   # bound the fed-back angle (wrap-safe; stops rollout blow-up)
    return dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                dt_s=rng.choice(dt_pool, size=n), distance_px=rng.choice(dist_pool, size=n))


# =========================================================================== #
#  Evaluation-pipeline adapter (subclass of fit_MLP.MLPDensity)
# =========================================================================== #
class MLPBorderDensity(M.MLPDensity):
    """Border-aware MDN MLP. Identical to ``fit_MLP.MLPDensity`` except the feature
    matrix carries the 3-column wall covariate (``use_border=True``), threaded through
    train / score / sample so it is used consistently everywhere."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"mlp_border_{self.mode}"

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

        # train-fold-only scale + emission init (leak-free); border cols are raw (bounded)
        alld = np.concatenate([r.dtheta for r in train_records])
        self._dth_scale = float(max(np.std(alld), 1.0))
        fwd = alld[np.abs(alld) < M.DTHETA0]
        turn = np.abs(alld[np.abs(alld) >= M.DTHETA0])
        s0 = float(max(np.std(fwd) if len(fwd) > 5 else 8.0, 0.5))
        m_, v_ = (turn.mean(), turn.var()) if len(turn) > 5 else (40.0, 400.0)
        a0 = float(max(m_ * m_ / max(v_, 1e-12), M.ALPHA_MIN))
        t0 = float(max(v_ / max(m_, 1e-12), 1.0))

        self._cfg_feat = dict(K=self.K, stim_mode=self.stim_mode,
                              include_symbols=self.include_symbols,
                              dth_scale=self._dth_scale, rich_scales=self._rich_scales,
                              use_border=True)
        X, y = build_feature_sequences(train_records, self._cfg_feat)
        self._model, self.history = M.fit_mlp(
            X, y, self.mode, self.hidden, epochs=self.epochs, lr=self.lr,
            patience=self.patience, batch_size=self.batch_size, seed=self.seed,
            init_emis=(s0, a0, t0))
        self.config = dict(mode=self.mode, hidden=self.hidden, n_features=X.shape[1],
                           K=self.K, stim_mode=self.stim_mode, use_border=True,
                           include_symbols=self.include_symbols, dth_scale=self._dth_scale,
                           rich_scales=(None if self._rich_scales is None
                                        else np.asarray(self._rich_scales).tolist()))
        return self

    def score(self, rec):
        X = build_feature_matrix(rec, self._cfg_feat)
        w, sigmaF, aL, tL, aR, tR = M.predict_params(self._model, X)
        with torch.no_grad():
            logdens = M.mixture_logdens(
                torch.tensor(rec.dtheta.astype(np.float32)),
                self._model(torch.tensor(X))).numpy()
        if self.mode == "basic":
            slq = E.symbol_logmass(w, sigmaF[0], aL[0], tL[0], mu=0.0)
        else:
            slq = M.symbol_logq_full(w, sigmaF, aL, tL, aR, tR)
        return dict(logdens=logdens, symbol_logq=slq)

    def sample(self, rec, rng):
        return generate_mlp(self._model, rec, self._cfg_feat,
                            self._dt_pool, self._dist_pool, rng)


# =========================================================================== #
#  Closed-loop generation (position feedback + hard guardrail) + scheduled sampling
# =========================================================================== #
def _ray_to_wall(x, y, h, cx, cy, R):
    """Distance from (x,y) along heading h to the circular wall (radius R about
    (cx,cy)). The fish is inside (|d| < R) so the positive root is real and > 0."""
    dx, dy = x - cx, y - cy
    ux, uy = np.cos(h), np.sin(h)
    ddotu = dx * ux + dy * uy
    disc = ddotu * ddotu - (dx * dx + dy * dy - R * R)
    return -ddotu + np.sqrt(max(disc, 0.0))


def advance(x, y, h, d, cx, cy, R, eps=1e-3, mode="clip", max_bounce=4, stats=None):
    """Move the fish distance ``d`` along heading ``h``, keeping it inside the dish.
    Returns the new ``(x, y, h)`` -- note the HEADING can change under 'reflect'.

    ``stats``: optional dict, incremented in place with ``n_hit`` (steps the guardrail had to
    touch at all) and ``n_bounce`` (total reflections). Pass one -- these are otherwise
    INVISIBLE. The emitted ``dtheta`` is computed BEFORE the guardrail runs, so a bounce never
    reaches the 20 turn features, and the 4 radial features that would see it are exactly what
    ``_search_containment`` recalibrates back onto target afterwards. An uninstrumented
    quantity that a downstream calibrator can absorb is the precise shape of the kin_feedback
    trap, so count it.

    mode='clip' (legacy): truncate the step at the wall. This is why generated fish pile up
      on the boundary. A clipped step sets the position to r = R - eps, i.e. it PLACES the
      fish exactly on the wall, and leaves the heading pointing OUTWARD so the next bout
      starts blocked too. Measured on the shoal_wall checkpoint: 2.5% of steps are clipped,
      59% of those land at r/R > 0.95, and that alone accounts for the 2.3% of generated
      bouts beyond r/R 0.95 -- a region real fish NEVER occupy (real max r/R = 0.990).
      28% of all at-wall bouts directly follow a clip.

    mode='reflect': specular reflection off the boundary, like a billiard in a disc. Travel
      to the wall, mirror the heading about the inward normal, continue with the remaining
      distance (up to ``max_bounce`` times). This preserves the step-length marginal (unlike
      clipping, which truncates it) AND leaves the fish heading INWARD, so the next bout is
      not immediately blocked. It is the least-distorting repair for a step the emission
      already committed to.
    """
    if mode != "reflect":
        L = min(d, max(_ray_to_wall(x, y, h, cx, cy, R) - eps, 0.0))
        if stats is not None and L < d:
            stats["n_hit"] = stats.get("n_hit", 0) + 1
        return x + L * np.cos(h), y + L * np.sin(h), h
    remaining = float(d)
    for _b in range(max_bounce):
        a = _ray_to_wall(x, y, h, cx, cy, R)
        if remaining <= a - eps:
            return x + remaining * np.cos(h), y + remaining * np.sin(h), h
        if stats is not None:
            stats["n_bounce"] = stats.get("n_bounce", 0) + 1
            if _b == 0:
                stats["n_hit"] = stats.get("n_hit", 0) + 1
        step = max(a - eps, 0.0)
        x += step * np.cos(h); y += step * np.sin(h)
        remaining -= step
        r = np.hypot(x - cx, y - cy)
        if r < 1e-9:                                  # degenerate; nothing to reflect about
            return x, y, h
        nx, ny = (x - cx) / r, (y - cy) / r           # outward unit normal
        dot = np.cos(h) * nx + np.sin(h) * ny
        if dot <= 0:                                  # already heading inward
            return x + remaining * np.cos(h), y + remaining * np.sin(h), h
        h = float(np.arctan2(np.sin(h) - 2 * dot * ny, np.cos(h) - 2 * dot * nx))
    return x, y, h                                    # bounce budget exhausted: stop short


WALL_REJECT_DEFAULT = {"d0": 2.0, "d1": 10.0, "max_draws": 20}


def accept_prob(u, d0, d1):
    """P(accept a bout that LANDS at r/R = u) under the shallow-border ramp. Scalar-safe.

    THE PHYSICAL CLAIM. A Larsch-rig dish is shallow at the rim, so there is an annulus a fish
    simply cannot enter, and the correct model of it is a constraint on POSITION -- not a bias on
    turning. That distinction is the whole point: ``centre_bias`` steers by adding
    ``cb * sin(psi_c)`` to dtheta, so buying the tail costs the shell turn marginal (measured
    E|dtheta| ~100 deg against a real 66) and drags the mean radius below target. Rejection
    truncates the emission's own conditional instead, so inside the shell the turn distribution
    keeps the model's own shape, conditioned on feasibility.

    THE RAMP, measured on all 15 fish against the physical ROI (100 mm dish). Real occupancy by
    distance from the wall, pooled over 4.7 M frames: 0.021% inside 1 mm, then EXACTLY NOTHING
    between 1 and 2 mm, 0.005% in 2-3 mm, 0.06% in 4-5 mm, 0.8% in 6-8 mm, 3.2% inside 10 mm,
    27% inside 20 mm. The empty 1-2 mm bin above occupied 0-1 mm is a discontinuity a continuous
    trajectory cannot produce, so the sub-millimetre mass is tracking glitch (it is almost all one
    fish). Hence a hard floor at ~2 mm and full acceptance by ~10 mm -- NOT the 20 mm a first
    guess suggests, which would reject a quarter of real behaviour.

    ``d0``/``d1`` in mm from the wall; below ``d0`` acceptance is 0, above ``d1`` it is 1."""
    d = wall_mm(u)
    if d1 <= d0:
        return float(d >= d0)
    return float(np.clip((d - d0) / (d1 - d0), 0.0, 1.0))


def generate_mlp_loop(model, rec, cfg_feat, dt_pool, dist_pool, rng, guardrail=True, eps=1e-3):
    """CLOSED-LOOP free-run: track the absolute pose and recompute the wall covariate
    from the GENERATED position each bout (the tracked movement heading stands in for
    the f_ori reference used at train time). An optional hard guardrail caps each step
    at the ray-to-wall length so a step can NEVER cross the dish. Stimulus is
    teacher-forced from ``rec``; step length is pool-resampled (no speed head yet).
    Returns the usual dict PLUS the tracked ``x_px, y_px`` (extra keys are ignored by
    the generative metrics but let us measure containment)."""
    n = rec.n
    K = cfg_feat["K"]
    dth_scale = cfg_feat["dth_scale"]
    stim = M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                         phi_ref=cfg_feat.get("phi_ref", "ori"))   # teacher-forced
    fwd = M._numpy_predictor(model)
    cx, cy, R = rec.dish_cx, rec.dish_cy, rec.dish_R
    x, y, h = float(rec.x_px[0]), float(rec.y_px[0]), float(rec.f_ori_rad[0])
    dth = np.zeros(n); dist = np.zeros(n); xs = np.empty(n); ys = np.empty(n)
    for t in range(n):
        xs[t], ys[t] = x, y
        # wall covariate from the CURRENT pose (h ~ f_ori reference)
        r = np.hypot(x - cx, y - cy)
        psi = _wrap_pi(np.arctan2(cy - y, cx - x) - h)
        border_t = border_channels((R - r) / R, psi, cfg_feat.get('border_mode', 'raw'))[0]
        # generated dtheta history fed back
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
                    s = 0 if abs(d) < M.DTHETA0 else (1 if d < 0 else 2)
                    sb[3 * (k - 1) + s] = 1.0
            row.append(sb)
        row.append(stim[t])
        row.append(border_t)   # SAME column order as build_feature_matrix
        xrow = np.concatenate(row).astype(np.float32)
        w, sF, aL, tL, aR, tR = fwd(xrow)
        c = rng.choice(3, p=w)
        if c == 0:
            dth[t] = rng.normal(0.0, sF)
        elif c == 1:
            dth[t] = -rng.gamma(aL, tL)
        else:
            dth[t] = +rng.gamma(aR, tR)
        dth[t] = float(np.clip(dth[t], -180.0, 180.0))   # bound the fed-back angle (wrap-safe; stops blow-up)
        # integrate pose: turn first, then step along the NEW heading (as reconstruct_xy does)
        h = h + np.radians(dth[t])
        L = float(rng.choice(dist_pool))
        if guardrail:
            L = min(L, max(_ray_to_wall(x, y, h, cx, cy, R) - eps, 0.0))
        dist[t] = L
        x = x + L * np.cos(h)
        y = y + L * np.sin(h)
    return dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                dt_s=rng.choice(dt_pool, size=n), distance_px=dist, x_px=xs, y_px=ys)


class MLPBorderLoopDensity(MLPBorderDensity):
    """Border MLP that (a) free-runs CLOSED-LOOP with a hard containment guardrail, and
    (b) can train with SCHEDULED SAMPLING (self-generated dtheta history). Training and
    scoring are inherited from MLPBorderDensity UNCHANGED when ``scheduled=False`` -- so
    its Metric A is then identical to ``mlp_border_full``; only ``sample()`` differs."""

    def __init__(self, *args, scheduled=False, sched_min_p=0.5, sched_decay_epochs=40,
                 guardrail=True, eps=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"mlp_border_{self.mode}_loop"
        self.scheduled = scheduled
        self.sched_min_p = sched_min_p
        self.sched_decay_epochs = sched_decay_epochs
        self.guardrail = guardrail
        self.eps = eps

    def fit(self, train_records, ctx):
        if not self.scheduled:
            return super().fit(train_records, ctx)     # identical to MLPBorderDensity
        return self._fit_scheduled(train_records, ctx)

    def sample(self, rec, rng):
        return generate_mlp_loop(self._model, rec, self._cfg_feat,
                                 self._dt_pool, self._dist_pool, rng,
                                 guardrail=self.guardrail, eps=self.eps)

    def _mixed_matrix(self, rec, gen_dtheta, p, rng):
        """Feature matrix whose dtheta history is a REAL-w.p.-p / GENERATED-else sequence
        (coherent: build a fed sequence, then window it); stim + border are teacher-forced;
        the target stays real."""
        if self._cfg_feat["include_symbols"]:
            raise NotImplementedError("scheduled sampling with include_symbols is unsupported")
        n = rec.n
        K = self._cfg_feat["K"]
        fed = np.where(rng.random(n) < p, rec.dtheta.astype(float), np.asarray(gen_dtheta, float))
        Hh = np.zeros((n, K), dtype=np.float32)
        for k in range(1, K + 1):
            Hh[k:, k - 1] = fed[:-k]
        Hh /= self._cfg_feat["dth_scale"]
        blocks = [Hh,
                  M._stim_block(rec, self._cfg_feat["stim_mode"], self._cfg_feat["rich_scales"],
                                phi_ref=self._cfg_feat.get("phi_ref", "ori")),
                  _border_block(rec, self._cfg_feat)]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def _fit_scheduled(self, train_records, ctx):
        """Scheduled sampling (2-pass/epoch, CPU): each epoch generate a rollout per fish
        with the CURRENT model, mix generated dtheta into the history with prob (1-p),
        p decaying 1 -> sched_min_p; one-step NLL against the REAL target; val split is
        teacher-forced. Reuses M.MDN / M.mdn_nll."""
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

        alld = np.concatenate([r.dtheta for r in train_records])
        self._dth_scale = float(max(np.std(alld), 1.0))
        fwd_ = alld[np.abs(alld) < M.DTHETA0]
        turn = np.abs(alld[np.abs(alld) >= M.DTHETA0])
        s0 = float(max(np.std(fwd_) if len(fwd_) > 5 else 8.0, 0.5))
        m_, v_ = (turn.mean(), turn.var()) if len(turn) > 5 else (40.0, 400.0)
        a0 = float(max(m_ * m_ / max(v_, 1e-12), M.ALPHA_MIN))
        t0 = float(max(v_ / max(m_, 1e-12), 1.0))

        self._cfg_feat = dict(K=self.K, stim_mode=self.stim_mode,
                              include_symbols=self.include_symbols,
                              dth_scale=self._dth_scale, rich_scales=self._rich_scales,
                              use_border=True)

        M._seed_everything(self.seed)
        rng = np.random.default_rng(self.seed)
        Xreal, yreal = build_feature_sequences(train_records, self._cfg_feat)
        N = len(Xreal); idx = rng.permutation(N); nval = max(1, int(0.15 * N))
        vi, ti = idx[:nval], idx[nval:]
        Xv, yv = torch.tensor(Xreal[vi]), torch.tensor(yreal[vi])

        model = M.MDN(Xreal.shape[1], self.hidden, mode=self.mode, init_emis=(s0, a0, t0))
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        best_val, best_state, no_imp = float("inf"), None, 0
        hist = {"train": [], "val": []}
        bs = self.batch_size
        for ep in range(self.epochs):
            p = 1.0 if self.sched_decay_epochs <= 0 else max(
                self.sched_min_p, 1.0 - (1.0 - self.sched_min_p) * ep / self.sched_decay_epochs)
            if p >= 0.999:
                Xtr, ytr = Xreal[ti], yreal[ti]
            else:
                model.eval()
                Xs = [self._mixed_matrix(
                          r, generate_mlp(model, r, self._cfg_feat,
                                          self._dt_pool, self._dist_pool, rng)["dtheta"], p, rng)
                      for r in train_records]
                Xall = np.concatenate(Xs, 0)
                Xtr, ytr = Xall[ti], yreal[ti]
            Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
            model.train()
            perm = torch.randperm(len(Xt))
            nb = int(np.ceil(len(Xt) / bs)); tot = 0.0
            for b in range(nb):
                bi = perm[b * bs:(b + 1) * bs]
                loss = M.mdn_nll(model, Xt[bi], yt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.item())
            model.eval()
            with torch.no_grad():
                vloss = float(M.mdn_nll(model, Xv, yv).item())
            hist["train"].append(tot / max(1, nb)); hist["val"].append(vloss)
            if vloss < best_val - 1e-4:
                best_val, no_imp = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model, self.history = model, hist
        self.config = dict(mode=self.mode, hidden=self.hidden, n_features=Xreal.shape[1],
                           K=self.K, stim_mode=self.stim_mode, use_border=True, scheduled=True,
                           sched_min_p=self.sched_min_p, sched_decay_epochs=self.sched_decay_epochs,
                           include_symbols=self.include_symbols, dth_scale=self._dth_scale,
                           rich_scales=(None if self._rich_scales is None
                                        else np.asarray(self._rich_scales).tolist()))
        return self


# =========================================================================== #
#  Smoke test:  uv run scripts/fit_MLP_border.py
# =========================================================================== #
if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    df = E.load_bouts(str(root / "data" / "bouts_GsizeMotionSingle.csv"))
    records = build_border_records(df)
    recs = list(records.values())
    ctx = E.make_context(recs, cfg={})
    m = MLPBorderDensity(mode="basic", K=5, stim_mode="simple", hidden=32, epochs=3)
    m.fit(recs, ctx)
    r0 = recs[0]
    X = build_feature_matrix(r0, m._cfg_feat)
    print(f"records: {len(recs)} | feature dim D = {X.shape[1]} (expect 9) | name = {m.name}")
    sc = m.score(r0)
    print(f"score logdens finite: {np.isfinite(sc['logdens']).all()}  "
          f"mean nats/bout = {np.nanmean(sc['logdens']):.3f}")
    g = m.sample(r0, np.random.default_rng(0))
    print(f"sample dtheta finite: {np.isfinite(g['dtheta']).all()}  n = {len(g['dtheta'])}")
