"""The model's input row and the dish.

``build_border_records`` turns the bout table into per-fish records with the physical dish geometry;
``build_feature_matrix`` builds a record's input matrix, block by block (each when its ``cfg_feat`` key is on):

    turn history | stimulus code (stimulus.py) | wall: distance to the wall and bearing to the dish centre |
    step and interval history | lag-valid flag | block clock | run clock | aligned-and-far input |
    static-dot block | dot-jump phase | dot direction of motion

The generator (``joint.generate_mix_joint_loop``) rebuilds the same row from a generated pose with the per-row
helpers here (``border_channels``, ``align_feature``, ``jump_channels``, ``dmove_rows``, ...). Also: the per-bout
regime weights of the x3 turn weight (``regime_weights``), the stimulus blocks' onsets (``block_onsets``), and the
dish geometry of a rollout (``advance``: the step clipped at the wall; ``accept_prob``: the shallow-rim rejection).
"""
from __future__ import annotations


import numpy as np

from twin import evaluate as E
from twin import paths
from twin import stimulus as M
from twin.segment_bouts import load_physical_dish

# raw physical ROIdef (true 100 mm dish geometry) -> per-fish (cx, cy, radius)
_DEFAULT_ROI = paths.ROI_PHYS

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


# =========================================================================== #
#  Records: baseline FishRecords + attached pose / border / dish
# =========================================================================== #
def build_border_records(df, roi_path=_DEFAULT_ROI, min_len=10, active_only=True):
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
    if active_only:
        if "active" not in df.columns:
            raise ValueError("active_only=True needs the `active` column "
                             "(clean_data/bouts_clean.csv, not data/bouts_GsizeMotionSingle.csv)")
        # Drop the gap-adjacent rows BEFORE building records, so rec.dtheta and every attached
        # array describe the same population the model is trained on. On a gap row `distance`
        # and `dtheta` are measured ACROSS the pause (implied speed 0.82 vs 11.9 px/s), and dt
        # runs to 426 s while an active-trained model stops near 10 s -- left in, dt_w1 and the
        # C2ST dt features measure the masking decision rather than the model.
        df = df[df["active"].to_numpy(bool)].copy()
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
        if "run_id" in g.columns:                                 # clean_data/bouts_clean.csv
            # Per-bout run id: a maximal stretch with no >10 s quiescence gap. The metrics use
            # it to avoid pairing bouts across a gap once the real side is masked to `active`
            # (a rollout has no gaps, so splicing would bias only the real reference).
            rec.run_id = g["run_id"].to_numpy()[good]
        if "active" in g.columns:
            rec.active = g["active"].to_numpy(bool)[good]
        if "run_pos" in g.columns:
            # Position within the run. The history blocks pad slot k-1 wherever run_pos < k,
            # so a bout early in a run never carries a lag from before its quiescence gap.
            rp = g["run_pos"].to_numpy()[good]
            if active_only and "run_id" in g.columns:
                # RE-COUNTED among the bouts that are IN the record. The table's run_pos = 0 is the
                # gap_before bout, which `active` drops (1,284 of 1,298 runs), so the record's
                # first bout of a run carried run_pos = 1: its lag-1 slot was then NOT padded and
                # read the array's previous element -- the bout before the pause, across the gap
                # (one slot at one bout per run, 1.4 % of bouts) -- and lag_valid / the run clock
                # sat one bout off the generator's t = 0. Counting from the record's own run start
                # makes all three exactly what the generator produces.
                rid = g["run_id"].to_numpy()[good]
                rp = np.zeros(len(rid), dtype=int)
                for t in range(1, len(rid)):
                    rp[t] = rp[t - 1] + 1 if rid[t] == rid[t - 1] else 0
            rec.run_pos = rp
        rec.frame_onset = g["frame_onset"].to_numpy(int)[good]     # per-bout frame index (for the video bridge)
        if "dot_dist_px" in g.columns:                            # per-bout dot geometry (add_dot_geometry.py)
            rec.dot_dist_px = g["dot_dist_px"].to_numpy(float)[good]
            rec.dot_cx = g["dot_cx"].to_numpy(float)[good]
            rec.dot_cy = g["dot_cy"].to_numpy(float)[good]
            rec.dot_prox = g["dot_prox"].to_numpy(float)[good]
        if "d0_x" in g.columns:                                   # per-dot positions (add_dot_positions.py)
            rec.dot_xs = np.stack([g[f"d{k}_x"].to_numpy(float)[good] for k in range(8)], axis=1)
            rec.dot_ys = np.stack([g[f"d{k}_y"].to_numpy(float)[good] for k in range(8)], axis=1)
        if "n_dots_left" in g.columns:                            # group size N (stim_mode='shoal')
            rec.n_dots = (np.nan_to_num(g["n_dots_left"].to_numpy(float))
                          + np.nan_to_num(g["n_dots_right"].to_numpy(float)))[good]
        rec.dish_cx = float(dish["cx"].loc[fid])
        rec.dish_cy = float(dish["cy"].loc[fid])
        rec.dish_R = float(dish["radius"].loc[fid])
    return records


# =========================================================================== #
#  Feature builder -- stimulus's blocks + the new border block (same chokepoint)
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
BORDER_COLS_BASIS = ("dnorm", "bs1", "bs2", "bs3", "bs4", "bc1", "bc2", "bc3", "bc4")
_BORDER_LC = np.log(np.asarray(BORDER_KNOTS, float))




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


def reanchor_track(dcx, dcy, real_pose, sim_pose):
    """Move a replayed dot track rigidly from the REAL fish's onset pose to the SIMULATED fish's: the rig places the
    dots relative to the fish at block onset, so a rollout that reaches the onset somewhere else must see them relative
    to itself. ``dcx``/``dcy`` (..., T) positions in px (NaN stays NaN); each pose is (x, y, heading rad), scalars or
    arrays broadcasting against the leading axes. Distances and bearings to the fish at onset are preserved exactly;
    with sim_pose == real_pose it is the identity (up to rounding). Used by both rollout engines (the profile
    fine-tune's free-warm-up starts, ``_reanchor_at``)."""
    xr, yr, hr = (np.asarray(v, float)[..., None] for v in real_pose)
    xs, ys, hs = (np.asarray(v, float)[..., None] for v in sim_pose)
    a = hs - hr
    ca, sa = np.cos(a), np.sin(a)
    ux = np.asarray(dcx, float) - xr; uy = np.asarray(dcy, float) - yr
    return xs + ca * ux - sa * uy, ys + sa * ux + ca * uy


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


# =========================================================================== #
#  Stimulus BLOCK structure -- one definition, shared by the features and the metrics
# =========================================================================== #
_FPS = 30.0


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


# SHORT-TERM STIMULUS LAGS (Model 38, `stim_lags` = K): where the dot group's centroid was at the previous K bouts,
# seen from where the fish is NOW (current position and reference heading) -- [sin, cos] of that bearing and the
# centroid-range code (`rz`'s formula) per lag, 3K columns. The difference between "now" and "the past, seen from now"
# is the dots' own motion relative to the fish (sweeping, approaching, receding), separate from the fish's own turns
# (already in the turn lags). A lag is 0 when its bout had no dots, sat in a different stimulus block, or lay before
# a > 10 s pause (a different RUN): the memory never reaches across a block boundary or a quiescent gap -- a
# free-running rollout is one run, so there it resets at block boundaries only. The existing `K_stim` block is a
# different thing (the PROGRAMME's previous blocks; null on the likelihood, and incompatible with the evaluation's
# re-anchored stimulus). APPENDED LAST, after every other block, so every archived layout is unchanged.
def stim_segments(nd, mv, run_start=None):
    """(n,) segment id: +1 at every change of the stimulus condition (count, moving) and at every run start."""
    cond = np.nan_to_num(np.asarray(nd, float)) * 10.0 + np.nan_to_num(np.asarray(mv, float))
    new = np.r_[True, cond[1:] != cond[:-1]] if len(cond) else np.zeros(0, bool)
    if run_start is not None:
        new = new | np.asarray(run_start, bool)
    return np.cumsum(new) - 1


# DOT-JUMP PHASE (Model 41, `stim_jump`): moving dots do not glide; each jumps about 2.1 mm every 18 frames (0.6 s) and
# is still in between, and real fish lock their bout starts to these jumps (onsets suppressed 0.07-0.13 s after a jump,
# enriched 0.33-0.47 s after it; thesis Ch. 3). The jump timeline is identical for every fish (the semi-closed loop only
# moves a block's track rigidly), so the phase is a pure function of the PROGRAMME FRAME, like the block clock: frames
# since the last jump of any shown dot (a block's first frame counts as a jump), as [sin, cos] of 2 pi age / 18, gated
# to moving-dot bouts (0 on pauses and static dots). With one dot it is the exact 18-frame clock; with several dots the
# jumps interleave and the phase is short and irregular. Computed once from the first fish's per-frame dot positions
# and cached under clean_data/. APPENDED after the stimulus lags, so every archived layout is unchanged.
JUMP_PERIOD_FR = 18
JUMP_STEP_PX = 1.0                         # a dot 'jumps' on a frame where it moves more than this (jumps are ~7.7 px)
JUMP_AGE_CAP = 255
_JUMP_TABLE = {}


def jump_age_table(rebuild=False):
    """(n_frames,) uint8: frames since the last dot jump (or block onset) at each programme frame, capped at 255."""
    if "t" in _JUMP_TABLE and not rebuild:
        return _JUMP_TABLE["t"]
    cache = paths.CACHE / "dot_jump_age.npy"
    if cache.exists() and not rebuild:
        _JUMP_TABLE["t"] = np.load(cache)
        return _JUMP_TABLE["t"]
    import pandas as pd
    src = sorted(paths.FRAMES.glob("*_f0_transformed_abs.csv"))
    if not src:
        raise FileNotFoundError(f"jump_age_table needs {paths.FRAMES}/*_f0_transformed_abs.csv")
    d = pd.read_csv(src[0], usecols=["frame", "stimulus"] + [f"d{k}_{s}" for k in range(8) for s in ("x", "y")])
    X = d[[f"d{k}_x" for k in range(8)]].to_numpy(float)
    Y = d[[f"d{k}_y" for k in range(8)]].to_numpy(float)
    step = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0))
    jump = np.r_[False, (np.nan_to_num(step, nan=0.0) > JUMP_STEP_PX).any(axis=1)]
    lab = d["stimulus"].astype(str).to_numpy()
    onset = np.r_[True, lab[1:] != lab[:-1]]                 # a block's first frame counts as a jump
    event = jump | onset
    fr = d["frame"].to_numpy(int)
    age = np.full(int(fr.max()) + 1, JUMP_AGE_CAP, dtype=np.int64)
    last = None
    for i, f in enumerate(fr):
        if event[i]:
            last = f
        if last is not None:
            age[f] = min(f - last, JUMP_AGE_CAP)
    tab = age.astype(np.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, tab)
    _JUMP_TABLE["t"] = tab
    return tab


def jump_age_at(frames):
    """Frames since the last jump at programme frame(s) ``frames`` (rounded; beyond the table -> the cap)."""
    tab = jump_age_table()
    f = np.rint(np.atleast_1d(np.asarray(frames, float))).astype(np.int64)
    ok = (f >= 0) & (f < len(tab))
    return np.where(ok, tab[np.clip(f, 0, len(tab) - 1)], JUMP_AGE_CAP).astype(float)


def jump_channels(age_fr, moving, n_dots):
    """(n, 2) [sin, cos] of 2 pi age / JUMP_PERIOD_FR, 0 unless the bout is shown moving dots. Scalar-safe: the
    record builder, the numpy generator (n = 1) and the batched engine all call this one function."""
    age = np.atleast_1d(np.asarray(age_fr, float))
    g = ((np.nan_to_num(np.atleast_1d(np.asarray(moving, float))) > 0)
         & (np.nan_to_num(np.atleast_1d(np.asarray(n_dots, float))) > 0))
    ph = 2.0 * np.pi * np.minimum(age, JUMP_PERIOD_FR - 1) / JUMP_PERIOD_FR
    return np.stack([np.where(g, np.sin(ph), 0.0), np.where(g, np.cos(ph), 0.0)], axis=1).astype(np.float32)


def _jump_block(rec, cfg_feat):
    """(n, 2) ``jump_channels`` on a record, read at each bout's onset frame."""
    n = rec.n
    if not cfg_feat.get("stim_jump") or n == 0:
        return np.zeros((n, 0), dtype=np.float32)
    return jump_channels(jump_age_at(np.asarray(rec.frame_onset, float)), np.asarray(rec.moving, float),
                         np.asarray(getattr(rec, "n_dots", np.zeros(n)), float))


# DOT MOTION DIRECTION (Model 51, `stim_dmove`, 2026-10-20): where the dot group's centroid MOVED between the previous
# bout's onset and this one, as [sin, cos] of that displacement's direction in the bout's reference frame (the
# ``phi_ref`` heading the bearing uses). Known at the bout's onset (no look-ahead); 0 on the first bout of a stimulus
# segment or run (no previous bout under the same condition), on no-dot and static bouts, and when the centroid moved
# less than DMOVE_MIN_PX (no direction to speak of). The input the screen asks for: real fish move along the dot's
# motion, a model that sees only its position cannot. APPENDED LAST, after every other block.
DMOVE_MIN_PX = 1.8                          # 0.5 mm at 3.60 px/mm


def dmove_rows(dx_prev, dy_prev, dx_now, dy_now, h, ok):
    """(n, 2) the dot-motion block from the centroid at the previous and the current bout, the reference heading ``h``
    and a validity mask ``ok`` (same segment, moving dots now). Scalar-safe: the record builder and the numpy
    generator (n = 1) call this one function."""
    dxp, dyp = np.atleast_1d(np.asarray(dx_prev, float)), np.atleast_1d(np.asarray(dy_prev, float))
    dxn, dyn = np.atleast_1d(np.asarray(dx_now, float)), np.atleast_1d(np.asarray(dy_now, float))
    h = np.atleast_1d(np.asarray(h, float)); ok = np.atleast_1d(np.asarray(ok, bool))
    vx, vy = dxn - dxp, dyn - dyp
    good = ok & np.isfinite(vx) & np.isfinite(vy) & (np.hypot(np.nan_to_num(vx), np.nan_to_num(vy)) > DMOVE_MIN_PX)
    psi = _wrap_pi(np.arctan2(np.nan_to_num(vy), np.nan_to_num(vx)) - np.nan_to_num(h))
    return np.stack([np.where(good, np.sin(psi), 0.0), np.where(good, np.cos(psi), 0.0)], axis=1).astype(np.float32)


def _dmove_block(rec, cfg_feat):
    """(n, 2) ``dmove_rows`` on a record: previous bout under the same stimulus condition and in the same run."""
    n = rec.n
    if not cfg_feat.get("stim_dmove") or n == 0:
        return np.zeros((n, 0), dtype=np.float32)
    rp = getattr(rec, "run_pos", None)
    nd = np.asarray(getattr(rec, "n_dots", np.zeros(n)), float)
    mv = np.asarray(rec.moving, float)
    seg = stim_segments(nd, mv, None if rp is None else (np.asarray(rp) == 0))
    dcx = np.asarray(getattr(rec, "dot_cx", np.full(n, np.nan)), float)
    dcy = np.asarray(getattr(rec, "dot_cy", np.full(n, np.nan)), float)
    h = (_travel_ref(rec) if cfg_feat.get("phi_ref", "ori") == "travel" else np.asarray(rec.f_ori_rad, float))
    ok = np.r_[False, seg[1:] == seg[:-1]] & (np.nan_to_num(mv) > 0) & (np.nan_to_num(nd) > 0)
    return dmove_rows(np.r_[np.nan, dcx[:-1]], np.r_[np.nan, dcy[:-1]], dcx, dcy, h, ok)


CLOCK_BLOCK_S = 60.0       # CLsemi stimulus block length (1800 fr @ 30 fps)
CLOCK_TAU_S = 4.0          # default decay for clock_mode='exp'


def _lagvalid_block(rec, cfg_feat):
    """(n, K) per-slot validity mask for the turn-history block, or (n, 1) scalar.

    ``M._history_block`` ZERO-PADS the first k rows of lag slot k-1. The turn lags are raw
    degrees over ``dth_scale``, so a padded slot is exactly 0.0 -- which is also what a
    perfectly straight bout gives. Without this channel the net cannot tell "there is no
    previous bout" from "the previous bout went straight", and straight-ish is the modal
    behaviour, so every sequence start injects a plausible fake observation.

    It matters far more once records are RUNS (cut at >10 s quiescence gaps) than it did with
    one record per fish: 979 sequence starts instead of 15, so ~5.5% of all K x n turn-lag
    entries are padded rather than 0.02%.

    ``cfg_feat['lag_valid']``: 'mask' -> (n, K) one flag per slot, unambiguous;
                               'scalar' -> (n, 1) how many slots are real, K times cheaper.
    """
    mode = str(cfg_feat.get("lag_valid", "") or "").lower()
    n, K = rec.n, int(cfg_feat["K"])
    if mode not in ("mask", "scalar"):
        return np.zeros((n, 0), dtype=np.float32)
    # `run_pos` counts from each RUN start, so it is the quantity that matches what
    # `_history_block` actually padded. Falling back to arange(n) counts from the record
    # start -- one per fish -- which would report "valid" for every cross-gap slot.
    rp = getattr(rec, "run_pos", None)
    pos = np.arange(n) if rp is None else np.asarray(rp)
    if mode == "scalar":
        return (np.minimum(pos, K) / max(K, 1)).reshape(n, 1).astype(np.float32)
    V = np.zeros((n, K), dtype=np.float32)
    for k in range(1, K + 1):
        V[pos >= k, k - 1] = 1.0
    return V


def _clock_block(rec, cfg_feat):
    """(n, 1) where the bout sits inside the current STIMULUS block; exactly 0 during a pause.

    MEASURED, holding the stimulus state fixed (range 40-85 px, |bearing| 45-135 deg, so it
    cannot be geometry in disguise): P(turn toward) is 0.80 in the first 5 s of a moving block
    against 0.70 for the remaining 55 s, +0.103 per fish in 14/15. No bearing, range, group-size
    or condition channel can carry this -- all of them are matched. Static shows 10/15, nothing,
    and ``dt`` inside the same cell is 8/15, so this is a TURN-head effect.

    Two encodings, because the effect is a STEP, not a line:

      'linear'  t / 60           a 0->1 ramp. Faithful to "where am I in the block", but the
                                 net must build a 5 s step out of a 60 s line.
      'exp'     exp(-t / tau)    an onset transient, tau = ``cfg_feat['clock_tau_s']`` (4 s),
                                 which is the shape actually measured.

    A pause bout is 0 under both -- and so is a late-block bout under 'exp' -- so the channel
    reads "a stimulus just started", not "a stimulus is on". The latter is already ``act``.

    ``t_pause_norm`` is deliberately NOT provided: its effects are real but small (dt 14/15,
    distance 13/15, r/R 13/15, all 3-6%) and it is a kinematic-side channel, so it is a
    cost-benefit call rather than a null. Add a second mode here if it is ever wanted.
    """
    mode = str(cfg_feat.get("clock_mode", "") or "").lower()
    n = rec.n
    if mode not in ("linear", "exp"):
        return np.zeros((n, 0), dtype=np.float32)
    start_fr, _bid, fr = block_onsets(rec)
    t = np.maximum((fr - start_fr) / _FPS, 0.0)
    nd = np.nan_to_num(np.asarray(getattr(rec, "n_dots", np.zeros(n)), float))
    on = (nd > 0).astype(float)                      # gate: 0 on every pause bout
    if mode == "linear":
        v = np.clip(t / CLOCK_BLOCK_S, 0.0, 1.0).reshape(n, 1)
    else:
        # One column per timescale. `clock2_tau_s` adds a SLOWER second clock: measured on the
        # active-only data, real fish keep turning hard toward moving dots for 6-8 s after onset
        # (31/25/22 deg on bouts 2-4 against 18 later) and P(turn toward) stays elevated for
        # ~18-20 s, while a 4 s clock is at 0.14 by 8 s -- the tail is not expressible without it.
        v = np.stack([np.exp(-t / tau) for tau in clock_taus(cfg_feat)], axis=1)
    return (v * on[:, None]).astype(np.float32)


def clock_taus(cfg_feat):
    """The exp-clock timescales in column order: ``clock_tau_s`` then, if set, ``clock2_tau_s``."""
    taus = [float(cfg_feat.get("clock_tau_s", CLOCK_TAU_S))]
    if cfg_feat.get("clock2_tau_s"):
        taus.append(float(cfg_feat["clock2_tau_s"]))
    return taus




def regime_weights(rec, w_onset=8.0, w_pursuit=8.0, onset_bouts=4, pursuit_deg=30.0,
                   pursuit_px=80.0, heads=None, include_static=False, w_static=1.0,
                   static_near_px=None, static_window=None,
                   w_pursuit2=None, pursuit2_deg=30.0, pursuit2_px=80.0, heads2=("turn", "dist"), drop_cond=None):
    """Per-bout training weights, 1 everywhere except in the two regimes Diagnostics.ipynb found
    UNDER-FIT in every model, both of them rare under a per-bout likelihood:

      onset    the first ``onset_bouts`` bouts of a MOVING block (measured 4.9 % of active bouts).
               Real fish turn 42 deg toward the dots on the first bout and 31/25/22 on the next
               three, against 18 later; the fitted law reaches 19-30 on the first, nothing after.
      pursuit  moving bouts with the dots ahead (|bearing| < ``pursuit_deg``) and >= ``pursuit_px``
               away (measured 1.5 %). Real fish go straight at them (mean |dtheta| 26 deg, against
               45 when close); the fitted law keeps turning 30-39 deg, which is two thirds of the
               closing shortfall in D4.

    At the default x8 the two regimes carry 27 % and 8 % of the loss (from 4.9 % and 1.5 %).
    Weighting rather than oversampling: the gradient is identical and the epoch does not grow.
    Returns ``(weights (n,) float32, info dict with the regime fractions and loss shares)``.

    ``heads`` (Model 16): which heads the weight applies to, e.g. ``("turn",)`` -- passed through
    in ``info`` for the trainer (``fit_mix_joint_re(weight_heads=...)``); the row loss of the
    non-latent trainer applies weights to all heads regardless. ``include_static``: the pursuit
    band also covers bouts with STATIC dots (Model 15's far-band weight on moving bouts only made
    the approach law bleed into static blocks: drift-away 0.35 vs 0.58 real)."""
    n = rec.n
    w = np.ones(n, dtype=np.float32)
    mv = np.asarray(rec.moving, bool)
    nd = np.nan_to_num(np.asarray(getattr(rec, "n_dots", np.zeros(n)), float))
    _s, bid, _f = block_onsets(rec)
    pos = np.zeros(n, dtype=int)                                # bout index within its block
    for t in range(1, n):
        pos[t] = pos[t - 1] + 1 if bid[t] == bid[t - 1] else 0
    onset = mv & (nd > 0) & (pos < int(onset_bouts))
    phi = getattr(rec, "stim_angle_travel", None)
    phi = np.asarray(phi if phi is not None else rec.stim_angle, float)
    rng = np.asarray(getattr(rec, "dot_dist_px", np.full(n, np.nan)), float)
    band = (nd > 0) if include_static else mv
    pursuit = (band & np.isfinite(phi) & np.isfinite(rng)
               & (np.degrees(np.abs(phi)) < float(pursuit_deg)) & (rng >= float(pursuit_px)))
    # STATIC NEAR band (Model 18): bouts with static dots within `static_near_px` -- where real fish
    # make the evasive turn (40-60 px ahead: toward -8.8 deg, |dtheta| 53) that the fit reaches only
    # a third of even with the static block (Model 17: -0.9 deg, 46).
    snear = np.zeros(n, dtype=bool)
    if static_near_px is not None:
        snear = ((~mv) & (nd > 0) & np.isfinite(phi) & np.isfinite(rng) & (rng < float(static_near_px))
                 & ~onset & ~pursuit)
        if static_window is not None:
            # Model 20: only the part of the block in which the turn-away happens (seconds since
            # the block onset), so the weight goes on the 3-deg signal and not on the onset
            # approach that Model 19's slow clock over-learned
            tb = (_f - _s) / _FPS
            snear &= (tb >= float(static_window[0])) & (tb < float(static_window[1]))
    w[onset] = float(w_onset)
    w[pursuit & ~onset] = float(w_pursuit)
    w[snear] = float(w_static)
    info = dict(frac_onset=float(onset.mean()), frac_pursuit=float((pursuit & ~onset).mean()),
                share_onset=float(w[onset].sum() / w.sum()),
                share_pursuit=float(w[pursuit & ~onset].sum() / w.sum()),
                frac_static=float(snear.mean()), share_static=float(w[snear].sum() / w.sum()),
                heads=(tuple(heads) if heads else None))
    if w_pursuit2 is not None:
        # SECOND, SHARP pursuit regime (Model 25) with its OWN head set: moving dots ahead within
        # `pursuit2_deg` and beyond `pursuit2_px` -- the exact cell in which the fit is measured short
        # (real |dtheta| 21-29 deg and 15.3-16.6 px steps, the fit 29-32 deg and 14.5-15.4 px). It
        # MULTIPLIES the first weight where both apply, so a bout in both regimes is weighted twice.
        # Per-head vectors go back in `info['per_head']`; the trainer reads them when present.
        p2 = (mv & np.isfinite(phi) & np.isfinite(rng)
              & (np.degrees(np.abs(phi)) < float(pursuit2_deg)) & (rng >= float(pursuit2_px)))
        w2 = np.ones(n, dtype=np.float32); w2[p2] = float(w_pursuit2)
        h1 = set(heads) if heads else {"turn", "dist", "dt"}
        h2 = set(heads2) if heads2 else {"turn", "dist", "dt"}
        per_head = {h: ((w if h in h1 else np.ones(n, np.float32)) * (w2 if h in h2 else np.ones(n, np.float32))).astype(np.float32)
                    for h in ("turn", "dist", "dt")}
        info.update(frac_pursuit2=float(p2.mean()),
                    share_pursuit2=float(per_head["turn"][p2].sum() / per_head["turn"].sum()),
                    heads2=tuple(h2), per_head=per_head)
    if drop_cond:
        # LEAVE-ONE-CONDITION-OUT (Final_report_missingPieces): bouts shown a held-out condition, given as
        # [(n_dots, moving), ...], get weight 0 on EVERY head, so the fit never learns from that condition; the bouts stay
        # in the records (the history of the bouts after them is real). Default None: nothing changes.
        drop = np.zeros(n, dtype=bool)
        for nd_c, mv_c in drop_cond:
            drop |= (nd == float(nd_c)) & (mv == bool(mv_c))
        h1 = set(heads) if heads else {"turn", "dist", "dt"}
        ph = info.get("per_head") or {h: (w if h in h1 else np.ones(n, np.float32)).astype(np.float32)
                                      for h in ("turn", "dist", "dt")}
        info["per_head"] = {h: np.where(drop, 0.0, v).astype(np.float32) for h, v in ph.items()}
        w = np.where(drop, 0.0, w).astype(np.float32)
        info["frac_dropped"] = float(drop.mean())
    return w, info


def build_feature_matrix(rec, cfg_feat):
    """(n, D) input matrix of a record: turn history, stimulus code, wall block, kinematic history, lag flag,
    block clock, run clock, aligned-and-far input, static block, jump phase, dot direction -- each when its
    ``cfg_feat`` key is on. Column order MUST match the generator's row (``joint.generate_mix_joint_loop``)."""
    off = [k for k in ("include_symbols", "use_adapt", "stim_static_clock", "stim_static_bump", "stim_onset",
                       "stim_lags") if cfg_feat.get(k)]
    if off or int(cfg_feat.get("K_stim", 1)) > 1:
        raise NotImplementedError(f"feature options not in this code: {off or 'K_stim > 1'}")
    blocks = [M._history_block(rec.dtheta, cfg_feat["K"], cfg_feat["dth_scale"],
                               run_pos=getattr(rec, "run_pos", None))]
    blocks.append(M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                                phi_ref=cfg_feat.get("phi_ref", "ori")))
    if cfg_feat.get("use_border"):
        blocks.append(_border_block(rec, cfg_feat))
    if cfg_feat.get("use_kin"):
        blocks.append(_kin_history_block(rec, cfg_feat))
    if cfg_feat.get("lag_valid"):
        blocks.append(_lagvalid_block(rec, cfg_feat))
    if cfg_feat.get("clock_mode"):
        blocks.append(_clock_block(rec, cfg_feat))
    if cfg_feat.get("run_clock_tau"):
        blocks.append(_runclock_block(rec, cfg_feat))
    if cfg_feat.get("stim_align"):
        blocks.append(_align_block(rec, cfg_feat))
    if cfg_feat.get("stim_static"):
        blocks.append(_static_block(rec, cfg_feat))
    if cfg_feat.get("stim_jump"):
        blocks.append(_jump_block(rec, cfg_feat))           # (Model 41)
    if cfg_feat.get("stim_dmove"):
        blocks.append(_dmove_block(rec, cfg_feat))          # LAST (Model 51)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _static_block(rec, cfg_feat):
    """(n, 8) ``stimulus.static_channels`` on the record (bearing in the ``phi_ref`` frame)."""
    phi = getattr(rec, "stim_angle_travel", None) if cfg_feat.get("phi_ref", "ori") == "travel" else None
    phi = np.asarray(phi if phi is not None else rec.stim_angle, float)
    n = rec.n
    return M.static_channels(phi, np.asarray(getattr(rec, "dot_dist_px", np.full(n, np.nan)), float),
                             np.asarray(getattr(rec, "n_dots", np.zeros(n)), float),
                             np.asarray(rec.moving, float)).astype(np.float32)


def tail_width(cfg_feat):
    """Width of the blocks appended AFTER the run clock (align, static) -- what the turn head's
    skip path reads together with the stimulus block."""
    return ((1 if cfg_feat.get("stim_align") else 0) + (8 if cfg_feat.get("stim_static") else 0)
            + (2 if cfg_feat.get("stim_jump") else 0)
            + (2 if cfg_feat.get("stim_dmove") else 0))


def align_feature(phi, r_px, moving, n_dots):
    """(n,) ``range x cos(bearing)`` for MOVING dots, in units of 100 px; 0 off-dot, in pauses
    and for static dots. The aligned-and-far interaction the shoal block cannot express: its
    range x bearing products are built on sin(bearing), which is ~0 exactly when the dots are
    ahead, so there the range only enters as a main effect. MEASURED (Model 11, teacher-forced):
    with the dots ahead the real |dtheta| falls 36.7 -> 31.8 -> 23.8 deg from 40-60 to 80-120 px
    while the fit stays at 39 -> 36 -> 34 -- the missing 3 mm of the approach dip."""
    phi = np.atleast_1d(np.asarray(phi, float)); r = np.atleast_1d(np.asarray(r_px, float))
    mv = np.atleast_1d(np.nan_to_num(np.asarray(moving, float)))
    nd = np.atleast_1d(np.nan_to_num(np.asarray(n_dots, float)))
    act = np.isfinite(phi) & np.isfinite(r) & (r > 0.0) & (mv > 0.0) & (nd > 0.0)
    return np.where(act, np.nan_to_num(r) / 100.0 * np.cos(np.nan_to_num(phi)), 0.0)


def _align_block(rec, cfg_feat):
    """(n, 1) ``align_feature`` on the record, with the bearing in the frame the stimulus block
    uses (``phi_ref``)."""
    phi = getattr(rec, "stim_angle_travel", None) if cfg_feat.get("phi_ref", "ori") == "travel" else None
    phi = np.asarray(phi if phi is not None else rec.stim_angle, float)
    n = rec.n
    return align_feature(phi, np.asarray(getattr(rec, "dot_dist_px", np.full(n, np.nan)), float),
                         np.asarray(rec.moving, float),
                         np.asarray(getattr(rec, "n_dots", np.zeros(n)), float)).reshape(n, 1).astype(np.float32)


def _runclock_block(rec, cfg_feat):
    """(n, 1) RUN CLOCK: ``exp(-run_pos / tau)`` -- 1 on the first bout after a > 10 s pause, gone
    within ~3 tau bouts, the same construction as the block clocks above.

    MEASURED (Data_cleanup_final.ipynb, step 5): masking the dead states removes the pauses but
    not their aftermath -- for the first ~60 bouts of a run the fish is slower (median dt +0.21 s,
    13/15 fish), takes shorter steps (-2.8 px, 13/15) and sits more centrally (r/R 0.41 vs 0.54,
    14/15) before it settles. Those are all ACTIVE bouts, 40 % of them within 60 bouts of a run
    start, and nothing told the model where they are: the lag-validity channel marks the first
    K = 8 bouts and is flat afterwards. A rollout is one run, so the generator uses exp(-t / tau)
    (``generate_mix_joint``) -- the transient is generated once, at t = 0, and the steady-state
    comparison is then made like for like (``generative_metrics.INTERIOR``)."""
    n = rec.n
    tau = float(cfg_feat["run_clock_tau"])
    rp = getattr(rec, "run_pos", None)
    pos = np.arange(n) if rp is None else np.asarray(rp, float)
    return np.exp(-pos / tau).reshape(n, 1).astype(np.float32)


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
    n = len(dist)
    H = np.zeros((n, 2 * Kk), dtype=np.float32)
    if cfg_feat.get("kin_dt_lags", True) == "centred":
        # CENTRED INTERVAL LAGS: each previous interval as its DEVIATION from the fish's own running
        # level -- an EMA of its past log intervals (tau `kin_dt_ema` bouts, init at the population
        # mean), computed identically on the emitted series in the generator. "Long relative to
        # what this fish does now" is what predicts the next interval; the level itself goes only
        # through z and the slow state, so a long interval can no longer reinforce the level in the
        # loop (the clustering Models 2-8 measured) while the real 0.07 bout-to-bout dependence,
        # which Model 9 lost by dropping the lags outright, is kept.
        lev = kin_dt_level(dtv, tm, float(cfg_feat.get("kin_dt_ema", KIN_DT_EMA)))
        lraw = np.log(np.clip(dtv, 1e-6, None))
        lraw = np.where(np.isfinite(lraw), lraw, lev)              # NaN -> deviation 0
        for k in range(1, Kk + 1):
            H[k:, k - 1] = ldist[:-k]
            H[k:, Kk + k - 1] = (lraw[:-k] - lev[k:]) / ts          # deviation from the CURRENT level
    else:
        ldt = np.nan_to_num((np.log(np.clip(dtv, 1e-6, None)) - tm) / ts)
        for k in range(1, Kk + 1):
            H[k:, k - 1] = ldist[:-k]                 # cols 0..Kk-1   = prev log-distance (lags 1..Kk)
            H[k:, Kk + k - 1] = ldt[:-k]              # cols Kk..2Kk-1 = prev log-dt
    rp = getattr(rec, "run_pos", None)
    if rp is not None:                                # never reach across a quiescence gap
        rp = np.asarray(rp)
        for k in range(1, Kk + 1):
            H[rp < k, k - 1] = 0.0
            H[rp < k, Kk + k - 1] = 0.0
    if not cfg_feat.get("kin_dt_lags", True):
        # INTERVAL LAGS OFF. Measured (Models 2-8): the interval lags carry the fish's tempo LEVEL
        # under teacher forcing (the latent recovered ~20% of the real between-fish interval spread
        # vs ~50% for step), and in the free run the inflated coefficients turn one long interval
        # into a run of them -- within-window lag-1 acf 0.12 vs real 0.07, per-window interval
        # spread 20% wide, unchanged by any gain or state. Real bout-to-bout interval dependence is
        # only 0.07, so dropping the lags loses almost nothing; the level then goes through z and
        # the slow state alone. Step lags stay (they came out honest: acf 0.196 vs 0.198).
        return H[:, :Kk]
    return H




KIN_DT_EMA = 250.0        # bouts; the level is a SLOW average (~4 windows), well above the drift's tau


def kin_dt_level(dt_s, tm, tau=KIN_DT_EMA):
    """(n,) the fish's running log-interval level BEFORE each bout: an EMA of its past log
    intervals with time constant ``tau`` bouts, initialised at the population mean ``tm``. Past
    only, so bout t's own interval never enters its own input; NaN intervals leave it unchanged."""
    lraw = np.log(np.clip(np.asarray(dt_s, float), 1e-6, None))
    a = 1.0 / max(float(tau), 1.0)
    lev = np.empty(len(lraw))
    e = float(tm)
    for t in range(len(lraw)):
        lev[t] = e
        if np.isfinite(lraw[t]):
            e += a * (lraw[t] - e)
    return lev




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




