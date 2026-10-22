"""Roll a trained bout model out over a NEW stimulus programme -- no real fish required.

The trained generators already react to the stimulus honestly (``stim_loop=True`` recomputes the
shoal bearing AND range from the fish's own generated pose each bout). What they could not do is
run on a stimulus that was never recorded: ``rec.dot_cx[t]`` is indexed by BOUT and materialised
before the rollout starts, while a programme is indexed by wall-clock FRAME and the bout times are
themselves generated. :class:`ProgramStimProvider` closes that loop -- it is handed the live clock
and pose each bout and resolves the programme on the spot (the ``stim_provider`` hook in
``fit_MLP_joint.generate_mix_joint_loop``).

Because ``CLsemi`` anchors the dot track to the fish's pose at block onset, and here that fish is
the simulated one, this is a genuine semi-closed-loop simulation: the same programme gives a
different absolute dot track on every rollout, exactly as it would with a different real fish.

Scope: the JOINT family (``MLPMixJointDensity`` and its ``MLPMixJointREDensity`` subclass), which
covers J10, its recalibrated/psi variants and J17. Other families raise rather than silently
teacher-force.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from twin import evaluate as E
from twin import fit_MLP_joint as JT
from twin import fit_MLP_mix as MX
from twin import paths
from twin.segment_bouts import PX_PER_MM, load_physical_dish

_DEFAULT_ROI = paths.ROI_PHYS
_DEFAULT_TRANSFORMED = paths.FRAMES

# Generous per-bout budget: real median dt is ~1.3 s, so 0.2 s/bout over-allocates ~6x. Only
# the array allocation scales with it (~0.4 ms/bout is paid per bout ACTUALLY generated). It was
# 0.5 until a latent + slow-state model drew one simulated fish fast enough to exhaust 21,100 bouts
# 300 s before the programme end (Model 11's profile run): such a fish is a tail of the model and
# must be scored, not crash the run.
_SEC_PER_BOUT_BUDGET = 0.2


# --------------------------------------------------------------------------- #
# The provider: programme -> per-bout stimulus, resolved online
# --------------------------------------------------------------------------- #
class ProgramStimProvider:
    """Callable passed to ``generate_mix_joint_loop(..., stim_provider=)``.

    ``(t, time_s, x, y, h) -> (dot_cx, dot_cy, moving, n_dots)``.

    ``time_s`` is the onset time of the bout about to be generated, so the programme is read at
    ``frame = round(time_s * fps)``. On entering a new block the CURRENT pose is latched as that
    block's anchor, which is what ``CLsemi`` does in the rig.

    Why latching at the first bout of a block is EXACT rather than approximate: under the
    dart-and-hold reconstruction the fish is stationary from ``onset(t-1) + bout_frames`` until
    ``onset(t)``, so its POSITION at the true block-onset frame is identical to the one handed in
    here -- even if a rare long interval swallows most of the block. Two known slivers of slack:
    a block onset landing inside the 3-frame dart itself (<= 0.1 s), and bounce bouts, where
    ``MB.advance(mode='reflect')`` leaves ``h`` as the post-bounce heading rather than the chord of
    the realised step (the same residual ``MB._travel_ref`` documents; a few percent of bouts).

    Contrast with a REAL fish, which keeps moving between bouts: there the same rule carries a
    median ~4 px / p90 ~81 px anchor error, dominated by heading change (the dots sit ~75 px out,
    so a 40 deg reorientation swings them ~50 px). ``verify_provider_parity`` measures both.
    """

    def __init__(self, program):
        self.p = program
        self.cur_block = -1
        self.anchor = (np.nan, np.nan, np.nan)
        self.anchors = {}          # block id -> (ax, ay, aori) actually used
        self.n_calls = 0
        self.n_out_of_range = 0

    def __call__(self, t, time_s, x, y, h):
        p = self.p
        self.n_calls += 1
        f = p.frame_of(time_s)
        if f < 0 or f >= p.n_frames:               # past the programme: no stimulus
            self.n_out_of_range += 1
            return np.nan, np.nan, 0.0, 0.0
        b = int(p.block_id[f])
        if b != self.cur_block:                    # NEW BLOCK -> latch this fish's own anchor
            self.cur_block = b
            self.anchor = (float(x), float(y), float(h))
            self.anchors[b] = self.anchor
        if not p.is_stim_of[b]:                    # inter-stimulus pause
            return np.nan, np.nan, 0.0, 0.0
        cx, cy = p.centroid(f, self.anchor)
        return cx, cy, float(p.moving_of[b]), float(p.n_dots_of[b])

    def dots(self, t, time_s, x, y, h):
        """PER-DOT form (stim_mode='shoal_pop'): ``(xs, ys, on, moving_k, n_dots)`` of the ACTIVE
        dots at this bout, anchored exactly as ``__call__`` (same latch, same block)."""
        p = self.p
        f = p.frame_of(time_s)
        self.last_time_s = f / p.fps          # when these dots were read (the dot-drift channels need it)
        if f < 0 or f >= p.n_frames:
            return _no_dots()
        b = int(p.block_id[f])
        if b != self.cur_block:
            self.cur_block = b
            self.anchor = (float(x), float(y), float(h))
            self.anchors[b] = self.anchor
        if not p.is_stim_of[b]:
            return _no_dots()
        X, Y = p.dots_absolute(f, self.anchor)
        return X, Y, np.ones(len(X), bool), np.full(len(X), float(p.moving_of[b])), float(p.n_dots_of[b])


def _no_dots():
    return np.empty(0), np.empty(0), np.empty(0, bool), np.empty(0), 0.0


class PhysicalStimProvider:
    """Drive a rollout from a FIXED physical programme (``stim_program.PhysicalProgram``).

    Where :class:`ProgramStimProvider` re-anchors the dot track to each simulated fish, this one
    hands back the absolute positions real fish N saw at that frame, whoever is swimming. So every
    rollout on the same programme sees dots in the same places, and differences between rollouts
    are behaviour rather than stimulus placement -- the point being to isolate individuality.

    Stateless (no anchor to latch), so one instance can serve every rollout. The model still
    computes bearing and range from its OWN pose, so it reacts honestly to where it actually is.
    """

    def __init__(self, phys):
        self.p = phys
        self.n_calls = 0
        self.n_out_of_range = 0

    def __call__(self, t, time_s, x, y, h):
        p = self.p
        self.n_calls += 1
        f = p.frame_of(time_s)
        if f < 0 or f >= p.n_frames:
            self.n_out_of_range += 1
            return np.nan, np.nan, 0.0, 0.0
        b = int(p.block_id[f])
        if not p.is_stim_of[b] or not np.isfinite(p.cx[f]):
            return np.nan, np.nan, 0.0, 0.0
        return p.cx[f], p.cy[f], float(p.moving_of[b]), float(p.n_dots_of[b])

    def dots(self, t, time_s, x, y, h):
        p = self.p
        f = p.frame_of(time_s)
        self.last_time_s = f / p.fps          # when these dots were read (the dot-drift channels need it)
        if f < 0 or f >= p.n_frames:
            return _no_dots()
        b = int(p.block_id[f])
        if not p.is_stim_of[b] or not np.isfinite(p.cx[f]):
            return _no_dots()
        X, Y = p.dots_absolute(f)
        return X, Y, np.ones(len(X), bool), np.full(len(X), float(p.moving_of[b])), float(p.n_dots_of[b])


class RecordAnchoredProvider:
    """Re-anchor a REAL record's own stimulus programme to the simulated fish, indexed by BOUT.

    For EVALUATION rollouts (Metric B panel, C2ST windows), which are compared with the real
    record bout for bout. :class:`ProgramStimProvider` reads the programme at the GENERATED clock,
    which drifts away from the record's -- a model trained on active bouts never pauses > 10 s --
    so here the frame for bout ``t`` is the record's own ``rec.frame_onset[t]``: same block, same
    condition and same dot-track phase as the real fish at that bout, so every mask built from
    ``rec`` still lines up. What changes is the ANCHOR: on entering a block the SIMULATED fish's
    pose is latched, exactly as the rig's CLsemi latches each real fish's, so the rollout starts
    every block at the programme's onset distance instead of wherever the real fish's dots were.

    Without it (the plain ``model.sample(rec, rng)`` path, which replays ``rec.dot_cx``), measured
    on S3_adt_rej held-out folds: block-onset distance median 121 px vs real 73, and 49.8% of
    block onsets beyond 121 px against 1.5% real.

    Heading: the rig anchors to the BODY axis; a rollout has none, so the travel heading the
    generator carries is used (as ProgramStimProvider does). Onset DISTANCE is exact under any
    heading (a rotation preserves it); the dots' SIDE is the approximation, and it is small.
    The generator's heading at a block onset is the direction of the bout JUST MADE, and the
    tracked body axis at rest sits a median 6.9 deg from that (p75 13.5, 3.0% > 90 deg, which are
    tracker head/tail flips), measured on 69,572 real bouts. The often-quoted "40 deg" is a
    different pair -- body axis at a bout's onset vs the direction of THAT bout, which contains
    the bout's own turn (median 40.4 deg) -- and does not apply to anchoring.
    """

    def __init__(self, program, rec):
        self.p = program
        self.fo = np.asarray(rec.frame_onset, int)
        self.cur_block = -1
        self.anchor = (np.nan, np.nan, np.nan)
        self.anchors = {}          # block id -> (ax, ay, aori) actually used

    def clock_s(self, t, time_s):
        """The wall-clock this provider reads bout ``t``'s stimulus at -- the record's own frame
        -- so the generator's block clock stays on the same grid as the stimulus it is shown.
        Past the record, fall back to the generated clock."""
        return float(self.fo[t]) / self.p.fps if t < len(self.fo) else float(time_s)

    def __call__(self, t, time_s, x, y, h):
        p = self.p
        if t >= len(self.fo):
            return np.nan, np.nan, 0.0, 0.0
        f = int(self.fo[t])
        if f < 0 or f >= p.n_frames:
            return np.nan, np.nan, 0.0, 0.0
        b = int(p.block_id[f])
        if b != self.cur_block:                    # NEW BLOCK -> latch the simulated fish's pose
            self.cur_block = b
            self.anchor = (float(x), float(y), float(h))
            self.anchors[b] = self.anchor
        if not p.is_stim_of[b]:
            return np.nan, np.nan, 0.0, 0.0
        cx, cy = p.centroid(f, self.anchor)
        return cx, cy, float(p.moving_of[b]), float(p.n_dots_of[b])

    def dots(self, t, time_s, x, y, h):
        """PER-DOT form (stim_mode='shoal_pop'), record-indexed and anchored exactly as ``__call__``."""
        p = self.p
        if t >= len(self.fo):
            return _no_dots()
        f = int(self.fo[t])
        self.last_time_s = f / p.fps          # when these dots were read (the dot-drift channels need it)
        if f < 0 or f >= p.n_frames:
            return _no_dots()
        b = int(p.block_id[f])
        if b != self.cur_block:
            self.cur_block = b
            self.anchor = (float(x), float(y), float(h))
            self.anchors[b] = self.anchor
        if not p.is_stim_of[b]:
            return _no_dots()
        X, Y = p.dots_absolute(f, self.anchor)
        return X, Y, np.ones(len(X), bool), np.full(len(X), float(p.moving_of[b])), float(p.n_dots_of[b])


# --------------------------------------------------------------------------- #
# A record with no fish behind it
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticRecord:
    """Duck-typed stand-in for ``evaluate.FishRecord`` for programme-driven rollouts.

    Verified empirically (see the plan): under ``closed_loop=True`` + ``stim_loop=True`` the only
    fields that reach the generated path are ``n``, the index-0 pose, the dish, and the per-bout
    stimulus -- and the stimulus now comes from the provider instead. Everything else is read only
    to compute the teacher-forced block WIDTHS (``_stim_block`` / ``_border_block``), whose values
    are discarded. Those fields are therefore filled with benign placeholders, chosen so the width
    computation runs without NaN warnings.

    ``dtheta[0] = 0`` and ``heading_rad[0] = f_ori_rad[0] = ori0`` make the start heading identical
    under both ``psi_ref='ori'`` (reads ``f_ori_rad[0]``) and ``psi_ref='travel'`` (reads
    ``heading_rad[0] - radians(dtheta[0])``).
    """
    fish_id: int
    dtheta: np.ndarray
    symbol: np.ndarray
    b: np.ndarray
    stim_angle: np.ndarray
    moving: np.ndarray
    dt_s: np.ndarray
    distance_px: np.ndarray
    u_raw: object = None
    stim_csv: object = None
    # attached by build_border_records in the real pipeline
    x_px: np.ndarray = field(default=None)
    y_px: np.ndarray = field(default=None)
    f_ori_rad: np.ndarray = field(default=None)
    heading_rad: np.ndarray = field(default=None)
    stim_angle_travel: np.ndarray = field(default=None)
    dist_to_border_norm: np.ndarray = field(default=None)
    frame_onset: np.ndarray = field(default=None)
    dot_dist_px: np.ndarray = field(default=None)
    dot_cx: np.ndarray = field(default=None)
    dot_cy: np.ndarray = field(default=None)
    dot_prox: np.ndarray = field(default=None)
    n_dots: np.ndarray = field(default=None)
    dot_xs: np.ndarray = field(default=None)      # per-dot positions (n, 8); NaN = inactive (shoal_pop)
    dot_ys: np.ndarray = field(default=None)
    dish_cx: float = 180.0
    dish_cy: float = 174.0
    dish_R: float = 183.5

    @property
    def n(self) -> int:
        return len(self.dtheta)


def make_synthetic_record(n, start_pose, dish, fish_id=0):
    """Allocate a :class:`SyntheticRecord` of ``n`` bouts seeded at ``start_pose = (x, y, ori)``."""
    x0, y0, ori0 = (float(v) for v in start_pose)
    cx, cy, R = (float(v) for v in dish)
    z = np.zeros(n)
    rec = SyntheticRecord(
        fish_id=int(fish_id),
        dtheta=z.copy(), symbol=np.zeros(n, np.int8), b=z.copy(),
        stim_angle=z.copy(), moving=z.copy(),
        dt_s=np.full(n, 1.3), distance_px=np.full(n, 10.0),
        x_px=np.full(n, x0), y_px=np.full(n, y0),
        f_ori_rad=np.full(n, ori0), heading_rad=np.full(n, ori0),
        stim_angle_travel=z.copy(),
        dist_to_border_norm=np.full(n, 0.5),
        frame_onset=np.arange(n, dtype=int),
        dot_dist_px=np.full(n, 100.0), dot_cx=np.full(n, np.nan), dot_cy=np.full(n, np.nan),
        dot_prox=np.full(n, 0.15), n_dots=z.copy(),
        dot_xs=np.full((n, 8), np.nan), dot_ys=np.full((n, 8), np.nan),
        dish_cx=cx, dish_cy=cy, dish_R=R)
    return rec


# --------------------------------------------------------------------------- #
# Where a fresh fish starts
# --------------------------------------------------------------------------- #
def default_dish(roi_path=_DEFAULT_ROI):
    """Median physical dish over the 15 recorded dishes (they differ by ~1%)."""
    d = load_physical_dish(roi_path)
    return (float(d["cx"].median()), float(d["cy"].median()), float(d["radius"].median()))


def sample_start_pose(records, rng, dish):
    """Draw a start pose from the REAL pooled per-bout occupancy, mapped into ``dish``.

    Sampling a real (x, y) directly would inherit that fish's own dish, and the dishes differ
    slightly; so the draw is taken in the source fish's NORMALISED dish coordinates and re-expanded
    into the target dish, which preserves the radial occupancy distribution (centre-weighted, peak
    r/R ~ 0.2-0.5) instead of, say, a uniform disc that would start many fish at the wall.
    """
    recs = [r for r in (records.values() if isinstance(records, dict) else records)
            if getattr(r, "x_px", None) is not None and getattr(r, "dish_R", None)]
    if not recs:
        raise ValueError("no usable records to draw a start pose from")
    r = recs[int(rng.integers(len(recs)))]
    i = int(rng.integers(len(r.x_px)))
    u = (float(r.x_px[i]) - r.dish_cx) / r.dish_R
    v = (float(r.y_px[i]) - r.dish_cy) / r.dish_R
    cx, cy, R = dish
    return cx + u * R, cy + v * R, float(r.f_ori_rad[i])


# --------------------------------------------------------------------------- #
# The rollout
# --------------------------------------------------------------------------- #
def rollout_program(model, program, rng, start_pose, dish, fish_id=0, n_max=None, provider=None,
                    on_budget="raise", block_seed=None):
    """Free-run ``model`` over ``program``. Returns the sample dict plus:

      ``t_onset_s``   per-bout onset time (s)
      ``frame_onset`` per-bout programme frame  = round(t_onset_s * fps)
      ``block_id``    per-bout programme block
      ``anchors``     {block id: (ax, ay, aori)} the fish latched -- the video needs these to
                      draw the dots it actually saw
      ``start_pose``, ``dish``, ``program_s``, ``end_time_s``
    """
    if not isinstance(model, JT.MLPMixJointDensity):
        raise TypeError(
            f"{type(model).__name__} is not a joint-family model. Programme rollouts need the "
            "stim_provider hook in fit_MLP_joint.generate_mix_joint_loop (J10 / J17 lineage: "
            "MLPMixJointDensity, MLPMixJointREDensity).")
    T = program.duration_s
    if n_max is None:
        n_max = int(T / _SEC_PER_BOUT_BUDGET) + 100
    rec = make_synthetic_record(n_max, start_pose, dish, fish_id=fish_id)
    prov = provider if provider is not None else ProgramStimProvider(program)

    # block_seed (report Model 11): per-block random streams in the generator (default None = one stream, unchanged)
    g = model.sample(rec, rng, stim_provider=prov, stop_time_s=T,
                     **({} if block_seed is None else {"block_seed": tuple(block_seed)}))

    n_gen = len(g["dtheta"])
    g["runaway"] = bool(n_gen >= n_max)
    if g["runaway"] and on_budget != "truncate":
        raise RuntimeError(
            f"bout budget exhausted ({n_gen} bouts) before the programme end "
            f"({g['end_time_s']:.1f}s of {T:.1f}s) -- pass a larger n_max.")
    # on_budget='truncate': a RUNAWAY fish (intervals collapsing -- Model 16's w3fta fold 2 emitted
    # 52,600 bouts in 6,180 s) is returned as far as it got, flagged, so a caller scoring many
    # rollouts can drop and COUNT it instead of losing the whole run.
    # end_time_s is the clock AFTER the last bout, i.e. where bout n+1 would have started -- past
    # the programme by construction. The realised extent of the trajectory is its LAST ONSET, which
    # sits below T by up to one inter-bout interval.
    g["last_onset_s"] = float(g["t_onset_s"][-1]) if n_gen else 0.0
    g["frame_onset"] = np.round(g["t_onset_s"] * program.fps).astype(int)
    g["block_id"] = program.block_id[np.clip(g["frame_onset"], 0, program.n_frames - 1)]
    g["anchors"] = getattr(prov, "anchors", {})      # empty for a physical programme
    g["start_pose"] = tuple(float(v) for v in start_pose)
    g["dish"] = tuple(float(v) for v in dish)
    g["program_s"] = float(T)
    return g


def rollout_many(model, program, n_rollouts, seed=0, records=None, dish=None,
                 start_poses=None, pose_sampler=None, verbose=True, on_budget="raise", block_seed=False):
    """``n_rollouts`` independent synthetic fish on the same programme.

    Rollout k is seeded ``default_rng([seed, k])`` and is therefore synthetic individual k,
    reproducibly: J10's per-trajectory ``indiv``/``vigor`` draws come off that stream.

    Start pose, in priority order: explicit ``start_poses[k]``; ``pose_sampler(rng, dish)`` (this is
    what the tool uses -- ``tool_reference.ReferenceStats.sample_start_pose``, so a clean clone and a
    full local checkout draw from the SAME occupancy histogram); ``records`` (real per-bout poses);
    else the dish centre with a random heading.
    """
    dish = dish if dish is not None else default_dish()
    out = []
    for k in range(int(n_rollouts)):
        rng = np.random.default_rng([int(seed), k])
        if start_poses is not None:
            pose = start_poses[k]
        elif pose_sampler is not None:
            pose = pose_sampler(rng, dish)
        elif records is not None:
            pose = sample_start_pose(records, rng, dish)
        else:
            pose = (dish[0], dish[1], float(rng.uniform(-np.pi, np.pi)))
        t0 = time.time()
        g = rollout_program(model, program, rng, pose, dish, fish_id=k, on_budget=on_budget,
                            block_seed=((int(seed), k) if block_seed else None))
        g["seed"] = [int(seed), k]
        g["wall_s"] = time.time() - t0
        out.append(g)
        if verbose:
            r = np.hypot(g["x_px"] - dish[0], g["y_px"] - dish[1]) / dish[2]
            print(f"  rollout {k}: {len(g['dtheta'])} bouts, last onset "
                  f"{g['last_onset_s'] / 60:.2f} of {g['program_s'] / 60:.2f} min "
                  f"(short by {g['program_s'] - g['last_onset_s']:.1f}s), "
                  f"median dt {np.median(g['dt_s']):.2f}s, max r/R {r.max():.3f}, "
                  f"{g['wall_s']:.1f}s")
    return out


def rollout_physical_many(model, phys, n_rollouts, seed=0, same_start=True, verbose=True):
    """``n_rollouts`` synthetic fish on ONE physical programme -- the individuality experiment.

    Both confounds are held fixed by default: every rollout sees the SAME absolute dot track (fish
    ``phys.fish_id``'s), and every rollout starts from that fish's real start pose in that fish's
    dish. What is left to vary is the model's own per-trajectory draws (``indiv``/``vigor``) and its
    emission noise -- so any divergence between rollouts IS the individuality.

    ``same_start=False`` keeps the shared stimulus but lets each rollout start wherever it likes,
    which re-introduces initial condition as a source of divergence.
    """
    prov = PhysicalStimProvider(phys)                 # stateless -> shared across rollouts
    out = []
    for k in range(int(n_rollouts)):
        rng = np.random.default_rng([int(seed), k])
        pose = phys.start_pose if same_start else (phys.dish[0], phys.dish[1],
                                                   float(rng.uniform(-np.pi, np.pi)))
        t0 = time.time()
        g = rollout_program(model, phys, rng, pose, phys.dish, fish_id=k, provider=prov)
        g["seed"] = [int(seed), k]
        g["wall_s"] = time.time() - t0
        g["physical_fish"] = int(phys.fish_id)
        out.append(g)
        if verbose:
            r = np.hypot(g["x_px"] - phys.dish[0], g["y_px"] - phys.dish[1]) / phys.dish[2]
            print(f"  rollout {k}: {len(g['dtheta'])} bouts, last onset "
                  f"{g['last_onset_s'] / 60:.2f} of {g['program_s'] / 60:.2f} min, "
                  f"median dt {np.median(g['dt_s']):.2f}s, max r/R {r.max():.3f}, "
                  f"{g['wall_s']:.1f}s")
    return out


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_provider_parity(program, records, fish_id=0,
                           transformed_dir=_DEFAULT_TRANSFORMED, verbose=True):
    """Check the provider's geometry against ``dot_cx``/``dot_cy`` in the bouts CSV.

    The reference was produced by an unrelated path (``segment_bouts.add_dist_to_dot`` reading the
    per-frame transformed CSVs), so agreement validates the whole chain: frame -> block ->
    active-dot rule -> anchored centroid.

    Two numbers are reported and they answer different questions:

    ``exact``  anchor = the REAL fish pose at the true block-onset frame. This is the rig's own
               definition, so it must agree to float precision.
    ``policy`` anchor = the real fish pose at the FIRST BOUT of the block, i.e. the latching rule
               :class:`ProgramStimProvider` uses. On a real fish this carries genuine error because
               a real fish keeps moving between bouts; on a synthetic one it is zero by
               construction (dart-and-hold). Reported to BOUND that modelling choice, not to pass
               a threshold.
    """
    import pandas as pd
    rec = records[fish_id]
    cands = sorted(Path(transformed_dir).glob(f"*_f{int(fish_id)}_transformed_abs.csv"))
    if not cands:
        raise FileNotFoundError(f"no transformed CSV for fish {fish_id}")
    tdf = pd.read_csv(cands[0], usecols=["frame", f"f{fish_id}_x", f"f{fish_id}_y",
                                         f"f{fish_id}_ori"]).sort_values("frame")
    fx = tdf[f"f{fish_id}_x"].to_numpy(float)
    fy = tdf[f"f{fish_id}_y"].to_numpy(float)
    fo = tdf[f"f{fish_id}_ori"].to_numpy(float)

    fo_b = np.asarray(rec.frame_onset, int)
    ref_cx = np.asarray(rec.dot_cx, float)
    ref_cy = np.asarray(rec.dot_cy, float)

    e_exact, e_policy, seen = [], [], {}
    for t, fr in enumerate(fo_b):
        if fr >= program.n_frames or not np.isfinite(ref_cx[t]):
            continue
        b = int(program.block_id[fr])
        if not program.is_stim_of[b]:
            continue
        s = int(program.starts[b])
        if s >= len(fx) or not np.isfinite(fx[s]):
            continue
        cx_e, cy_e = program.centroid(fr, (fx[s], fy[s], fo[s]))          # true block-onset pose
        e_exact.append(abs(cx_e - ref_cx[t])); e_exact.append(abs(cy_e - ref_cy[t]))
        if b not in seen:                                                  # first bout in block
            if fr >= len(fx) or not np.isfinite(fx[fr]):
                continue
            seen[b] = (fx[fr], fy[fr], fo[fr])
        cx_p, cy_p = program.centroid(fr, seen[b])
        e_policy.append(float(np.hypot(cx_p - ref_cx[t], cy_p - ref_cy[t])))

    mx = float(np.max(e_exact)) if e_exact else np.inf
    pol = np.asarray(e_policy, float)
    if verbose:
        print(f"provider parity (fish {fish_id}, {len(e_exact) // 2} bouts with an active dot):")
        print(f"  exact  (anchor = real pose at block-onset frame): max abs err {mx:.3e} px "
              f"-> {'PASS' if mx < 1e-9 else 'FAIL'}")
        if pol.size:
            print(f"  policy (anchor = real pose at first bout of block): median "
                  f"{np.median(pol):.1f} px, p90 {np.percentile(pol, 90):.1f} px "
                  f"-- real-fish drift between block onset and its next bout; "
                  f"exactly 0 for a synthetic fish (dart-and-hold).")
    return mx, pol


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    from twin import stim_program as SP

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default=str(_ROOT / "data" / "data" / "Original_data"
                                         / "2025.02.20-0957_trajectory_GsizeMotionSingle_180min.csv"))
    ap.add_argument("--model", default=str(_ROOT / "notebooks" / "_zip_full" / "ladder_b_indiv.pt"))
    ap.add_argument("--bouts", default=str(_ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="short programme, plus all checks")
    a = ap.parse_args()

    from twin import fit_MLP_border as MB

    print("loading programme ...")
    prog = SP.load_program(a.traj)
    print(prog.summary())
    print("\nanchoring check ...")
    SP.verify_against_transformed(prog)

    print("\nloading model + records ...")
    model = MX.load_mix(a.model)
    records = MB.build_border_records(E.load_bouts(a.bouts))
    print(f"  {model.name}")

    print("\nprovider parity ...")
    verify_provider_parity(prog, records)

    if a.smoke:                       # first 6 blocks only, so the whole path runs in seconds
        cut = int(prog.ends[5])
        prog = SP.StimulusProgram(
            dx=prog.dx[:cut], dy=prog.dy[:cut], active=prog.active[:cut],
            block_id=prog.block_id[:cut], labels=prog.labels[:6],
            starts=prog.starts[:6], ends=prog.ends[:6], n_dots_of=prog.n_dots_of[:6],
            moving_of=prog.moving_of[:6], side_of=prog.side_of[:6],
            is_stim_of=prog.is_stim_of[:6], fps=prog.fps, path=prog.path)
        print(f"\n[smoke] truncated to {prog.duration_s / 60:.1f} min")

    print(f"\nrolling out {a.n} synthetic fish ...")
    gs = rollout_many(model, prog, a.n, seed=a.seed, records=records)
    for g in gs:
        r = np.hypot(g["x_px"] - g["dish"][0], g["y_px"] - g["dish"][1]) / g["dish"][2]
        assert (r <= 1.0 + 1e-9).all(), "containment violated"
    print("containment: 0 bouts outside the dish  -> PASS")
