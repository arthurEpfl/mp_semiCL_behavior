"""Two moving targets at a controlled separation: does a simulated fish average them or choose one?

The paradigm of Sridhar et al. (2021, PNAS), where zebrafish following two virtual conspecifics moving back and forth in
parallel switch from swimming between them to following one as their lateral separation grows. Here the targets are two
dot groups (A and B) whose centroids move back and forth along two parallel tracks through the dish centre, at lateral
separation L, both at the same speed and in step; the trained models (no retraining) see them through one of the
combination rules of joint (effects, add, avg, nearest), and the references show only A, or nothing ('blind'),
on the same tracks.

GEOMETRY (per block, seeded by (fish, block)): the track axis u through the dish centre at a random angle; the two tracks
at +-L/2 along the normal n; both centroids at s(t) u + (+-L/2) n with s(t) a triangle wave of amplitude A_MM at SPEED_MM_S,
random phase; A on the +n side in even blocks and on the -n side in odd ones. Moving dot groups of the trained counts, seen
through the centroid code (centroid, count, moving flag), so the dots' spread inside a group plays no part. Distances in
mm at 3.60 px/mm. The block clock starts at each block's onset, as in training (the rig's block-onset structure: 100 s
lead-in pause, then N_BLOCKS x (60 s with the targets + 50 s pause)).

THE MAPS (figures/fig_bifurcation.py): per separation, the distribution of the fish's lateral position over the bouts
behind the pair, in a dish of twice the radius with an anchored (equidistant) start.

    python -m twin.two_targets validate               # one visible group: the four rules give the same bouts
    python -m twin.two_targets run-report [--jobs 14] # every map: M6 and F at both seeds, 4 pairs, L = 0-100 mm
    video("14f"), rules_video("14f")                   # one fish at L = 20 / 50 / 80 mm; the four rules at 80 mm

LARGE DISH. The training dish (radius 49 mm) holds tracks up to L = 50 mm; Sridhar et al. ran separations to
100 mm, so the dish radius is doubled (ARENA = 2, as multi_fish's 5x dish): every wall mechanism of the model is relative
to R, so the fish senses the training dish magnified (its preferred radius, ~0.45 R, moves from ~22 to ~44 mm); steps,
the tracks (A_MM) and the readout are absolute. The model's log-range input saturates at 280 px = 78 mm (multi_fish), so
a target farther away pulls as one at 78 mm: no range input leaves the trained values. 'blind' (no target shown, same
tracks and readout box) measures what the magnified dish alone does to the lateral positions.

ANCHORED ONSET (AnchoredParallelProvider). With the tracks fixed in the dish the fish starts each block
wherever the pause left it, often nearer one group, and that group is then the one followed in about 2/3 of committed
effects trials. Here every block is anchored on the fish instead: at the block's first bout (the fish's pose then is
its pose at the block onset; it has not moved since), the pair's midpoint is put ANCHOR_AHEAD_MM ahead of the fish
along the track axis, the fish exactly on the midline (equidistant from A and B), the pair starting a leg away from it.
The axis points from the fish toward the dish centre, turned by a random +-ANCHOR_JITTER_DEG (seeded per fish and
block), which keeps the tracks inside the dish; if a track end would still lie beyond 0.95 R, the pair starts further
ahead (5 mm steps). Everything else is unchanged.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from twin import model_figures as MF
from twin import paths
from twin import stim_program as SPm
from twin import stim_rollout as SR

OUT = paths.TWO_TARGETS
PX_MM = 3.60                                        # physical scale (Chapter 3)
SPEED_MM_S = 2.0                                    # both centroids; trained moving centroids 1.1 (8 dots) - 3.5 (1 dot)
A_MM = 20.0                                         # track half-length (inside the 49 mm dish with the widest L)
STIM_S, PAUSE_S, LEAD_S = 60.0, 50.0, 100.0
N_BLOCKS, N_FISH = 10, 15                           # 15 fish = 3 per fold model, 10 blocks each = 150 trials
PAIRS = {"1v1": (1, 1), "4v4": (4, 4), "1v4": (1, 4), "2v8": (2, 8)}
RULES = ("effects", "add", "avg", "nearest")
REFS = {"ref-A": (0,), "ref-B": (1,)}
GEOM_SEED = 7171
ARM_M6 = "S4x_adt_z2_cdt_tr_rc_al_w3ft"
ARM_F = "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"
MODELS = {"26e": ("M26e", ARM_M6), "30d": ("M30d", ARM_M6),
          "14f": ("M14f", ARM_F), "32b": ("M32b", ARM_F)}


@dataclass(frozen=True)
class PairScene:
    name: str
    nA: int
    nB: int
    L_mm: float


def scene(pair, L_mm):
    a, b = PAIRS[pair]
    return PairScene(f"{pair}_L{float(L_mm):g}", a, b, float(L_mm))


class ParallelProvider:
    """Two dot-group centroids moving back and forth on parallel tracks (see the module doc). ``groups()`` returns the
    visible groups for the multi-group generator; ``__call__`` the one-centroid view of the standard generator. Every
    stimulus-block call records the pose, both centroids and the targets' motion direction under the bout index."""

    def __init__(self, sc, fish_k, dish, n_blocks=N_BLOCKS, visible=None, fps=SPm.FPS):
        self.sc, self.fps, self.n_blocks = sc, float(fps), int(n_blocks)
        self.dish, self.fish_k = dish, int(fish_k)
        cx, cy, R = dish
        self.c0 = np.array([cx, cy], float)
        self.lead_f = int(round(LEAD_S * fps)); self.stim_f = int(round(STIM_S * fps))
        self.cyc_f = self.stim_f + int(round(PAUSE_S * fps))
        self.n_frames = self.lead_f + self.n_blocks * self.cyc_f
        self.T = self.n_frames / self.fps
        self.visible = (0, 1) if visible is None else tuple(visible)
        self.geom = []
        for j in range(self.n_blocks):
            rng = np.random.default_rng([GEOM_SEED, int(fish_k), j])
            ang = rng.uniform(0, 2 * np.pi)
            u = np.array([np.cos(ang), np.sin(ang)]); n = np.array([-u[1], u[0]])
            nA = n if j % 2 == 0 else -n                               # A alternates sides
            self.geom.append((u, nA, rng.uniform(0, 1)))
        self.amp = A_MM * PX_MM
        self.half = 0.5 * sc.L_mm * PX_MM
        self.period_s = 4.0 * A_MM / SPEED_MM_S                       # one full back-and-forth
        self.rec = {}

    def locate(self, f):
        if f < 0 or f >= self.n_frames or f < self.lead_f:
            return -1, -1
        j, r = divmod(f - self.lead_f, self.cyc_f)
        return (j, r) if r < self.stim_f else (-1, -1)

    def _state(self, j, k):
        """centroids of A and B (px) and the unit direction of motion at frame k of block j."""
        u, nA, ph = self.geom[j]
        tau = (k / self.fps / self.period_s + ph) % 1.0               # triangle wave in [-A, A]
        s = self.amp * (4 * tau - 1 if tau < 0.5 else 3 - 4 * tau)
        vdir = u if tau < 0.5 else -u
        mid = self.c0 + s * u
        return (mid + self.half * nA, mid - self.half * nA), vdir, mid, nA

    def groups(self, t, time_s, x, y, h):
        j, k = self.locate(int(round(float(time_s) * self.fps)))
        if j < 0:
            return []
        (pa, pb), vdir, mid, nA = self._state(j, k)
        self.rec[int(t)] = (j, k / self.fps, float(x), float(y), pa, pb, vdir, mid, nA)
        out = []
        for g in self.visible:
            p, nd = (pa, self.sc.nA) if g == 0 else (pb, self.sc.nB)
            out.append((float(p[0]), float(p[1]), 1.0, float(nd)))
        return out

    def __call__(self, t, time_s, x, y, h):
        gs = self.groups(t, time_s, x, y, h)
        if not gs:
            return np.nan, np.nan, 0.0, 0.0
        if len(gs) == 1:
            return gs[0]
        n = np.array([g[3] for g in gs])
        return (float(np.dot(n, [g[0] for g in gs]) / n.sum()), float(np.dot(n, [g[1] for g in gs]) / n.sum()),
                1.0, float(n.sum()))


ANCHOR_AHEAD_MM, ANCHOR_JITTER_DEG = 20.0, 30.0


class AnchoredParallelProvider(ParallelProvider):
    """ParallelProvider whose block geometry is set at each block's first bout from the fish's pose (module doc:
    ANCHORED ONSET). Block j's track centre, axis and phase are decided lazily; ``self.anchor[j]`` = (centre px, ahead mm,
    first-bout block time s)."""

    def __init__(self, sc, fish_k, dish, n_blocks=N_BLOCKS, visible=None, fps=SPm.FPS):
        super().__init__(sc, fish_k, dish, n_blocks, visible, fps)
        self.centre = [None] * self.n_blocks
        self.anchor = {}
        self.R = float(dish[2])

    def _anchor(self, j, k, x, y):
        rng = np.random.default_rng([GEOM_SEED + 1, int(self.fish_k), j])
        p = np.array([x, y], float)
        to_c = self.c0 - p
        ang = np.arctan2(to_c[1], to_c[0]) if np.hypot(*to_c) > 1e-6 else rng.uniform(0, 2 * np.pi)
        ang += np.radians(rng.uniform(-ANCHOR_JITTER_DEG, ANCHOR_JITTER_DEG))
        u = np.array([np.cos(ang), np.sin(ang)]); n = np.array([-u[1], u[0]])
        nA = n if j % 2 == 0 else -n
        # phase: the leg starts at the first bout, s = -A moving +u (away from the fish)
        ph = (-(k / self.fps) / self.period_s) % 1.0
        ahead = ANCHOR_AHEAD_MM
        while True:
            c = p + (ahead * PX_MM + self.amp) * u                       # track centre: start of the leg = ahead
            ends = [c + sg * self.amp * u + sd * self.half * nA for sg in (-1, 1) for sd in (-1, 1)]
            if max(np.hypot(*(e - self.c0)) for e in ends) <= 0.95 * self.R or ahead >= 100:
                break
            ahead += 5.0
        self.geom[j] = (u, nA, ph)
        self.centre[j] = c
        self.anchor[j] = (c, ahead, k / self.fps)

    def _state(self, j, k):
        u, nA, ph = self.geom[j]
        tau = (k / self.fps / self.period_s + ph) % 1.0
        s = self.amp * (4 * tau - 1 if tau < 0.5 else 3 - 4 * tau)
        vdir = u if tau < 0.5 else -u
        mid = (self.c0 if self.centre[j] is None else self.centre[j]) + s * u
        return (mid + self.half * nA, mid - self.half * nA), vdir, mid, nA

    def groups(self, t, time_s, x, y, h):
        j, k = self.locate(int(round(float(time_s) * self.fps)))
        if j >= 0 and self.centre[j] is None:
            self._anchor(j, k, x, y)
        return super().groups(t, time_s, x, y, h)


def _models(tag):
    d, arm = MODELS[tag]
    with contextlib.redirect_stdout(io.StringIO()):
        models, _ = MF.fold_models(d, arm)                            # the wall-rejection band merged in
    return [models[f] for f in sorted(models)]


def simulate(tag, sc, cond, n_fish=N_FISH, n_blocks=N_BLOCKS, seed=0, models=None, arena=1, onset="free"):
    """Bouts of the stimulus blocks of ``n_fish`` simulated fish (n_fish / 5 per fold model) on scene ``sc`` under a rule
    or a reference. Fish k is seeded (seed, k) with per-block random streams and its block geometry by (GEOM_SEED, k,
    block): start pose, latent, drift, randomness and tracks are the same in every condition (paired by fish)."""
    rule, vis = (cond, None) if cond in RULES else ("effects", () if cond == "blind" else REFS[cond])
    models = models or _models(tag)
    _, dish, ref = MF._programme()
    dish = (float(dish[0]), float(dish[1]), float(dish[2]) * float(arena))
    per = int(n_fish) // len(models)
    rows, n_run = [], 0
    for mi, m in enumerate(models):
        for r in range(per):
            k = mi * per + r
            rng = np.random.default_rng([int(seed), k])
            pose = ref.sample_start_pose(rng, dish)
            Prov = AnchoredParallelProvider if onset == "anchored" else ParallelProvider
            prov = Prov(sc, k, dish, n_blocks, visible=vis)
            n_max = int(prov.T / 0.2) + 100
            rec = SR.make_synthetic_record(n_max, pose, dish, fish_id=k)
            g = m.sample(rec, rng, stim_provider=prov, stop_time_s=prov.T, block_seed=(int(seed), k), multi_rule=rule)
            nb = len(g["dtheta"])
            if nb >= n_max:
                n_run += 1
                continue
            keep = sorted(t for t in prov.rec if t < nb)
            if not keep:
                continue
            R_ = [prov.rec[t] for t in keep]
            x = np.asarray(g["x_px"])[keep]; y = np.asarray(g["y_px"])[keep]
            if not (np.allclose(x, [q[2] for q in R_]) and np.allclose(y, [q[3] for q in R_])):
                raise RuntimeError("recorded pose differs from the generator's")
            P = np.stack([x, y], 1)
            pa = np.array([q[4] for q in R_]); pb = np.array([q[5] for q in R_])
            vd = np.array([q[6] for q in R_]); mid = np.array([q[7] for q in R_]); nA = np.array([q[8] for q in R_])
            rel = P - mid
            d = dict(fish=k, fold=mi, blk=[q[0] for q in R_], t_blk=[q[1] for q in R_],
                     dth=np.asarray(g["dtheta"])[keep],
                     lat_mm=(rel * nA).sum(1) / PX_MM, along_mm=(rel * vd).sum(1) / PX_MM,
                     dA_mm=np.hypot(*(P - pa).T) / PX_MM, dB_mm=np.hypot(*(P - pb).T) / PX_MM)
            rows.append(pd.DataFrame(d))
    b = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    b.attrs.update(runaway=n_run, tag=tag, scene=sc.name, cond=cond, n_fish=n_fish, seed=seed, L_mm=sc.L_mm,
                   nA=sc.nA, nB=sc.nB, arena=arena, onset=onset)
    return b


def _path(tag, sc, cond, arena=1, onset="free"):
    sub = (f"arena{arena:g}" if arena != 1 else "") + ("_anch" if onset == "anchored" else "")
    d = OUT / tag if not sub else OUT / sub.lstrip("_") / tag
    return d / f"{sc.name}__{cond}.pkl"


def run_one(job):
    tag, pair, L, cond = job[:4]
    arena = job[4] if len(job) > 4 else 1
    onset = job[5] if len(job) > 5 else "free"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    sc = scene(pair, L)
    p = _path(tag, sc, cond, arena, onset)
    if p.exists():
        return job, "cached", 0.0
    t0 = time.time()
    b = simulate(tag, sc, cond, arena=arena, onset=onset)
    p.parent.mkdir(parents=True, exist_ok=True)
    b.to_pickle(p)
    return job, f"{len(b)} bouts, runaway {b.attrs.get('runaway', 0)}", time.time() - t0


ARENA_BIG = 2.0                                                  # the large dish: twice the training radius (98 mm)
BIG_SEPS_MM = tuple(float(v) for v in np.arange(0.0, 100.01, 2.5))
REPORT_PAIRS = ("1v1", "4v4", "1v4", "2v8")
REPORT_CONDS = ("effects", "add", "avg", "nearest", "ref-A", "blind")


def report_jobs(tags=tuple(MODELS)):
    """Every map of the thesis: each pair x REPORT_CONDS, 2x dish, 0-100 mm in 2.5 mm steps, equidistant start. The
    unequal pairs have no blind run of their own: a blind fish sees no target, so it is the 4 v 4 blind run."""
    return [(tag, pair, L, c, ARENA_BIG, "anchored")
            for pair, tag, L, c in itertools.product(REPORT_PAIRS, tags, BIG_SEPS_MM, REPORT_CONDS)
            if not (c == "blind" and PAIRS[pair][0] != PAIRS[pair][1])]


def run_all(n_jobs=14):
    from concurrent.futures import ProcessPoolExecutor
    J = report_jobs()
    todo = [j for j in J if not _path(j[0], scene(j[1], j[2]), j[3], *j[4:]).exists()]
    print(f"{len(J)} jobs, {len(todo)} to run", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(n_jobs) as ex:
        for i, (job, msg, s) in enumerate(ex.map(run_one, todo), 1):
            print(f"[{i}/{len(todo)} {(time.time() - t0) / 60:.0f} min] {job}: {msg} ({s:.0f} s)", flush=True)
    print("ALL DONE", flush=True)


# ----------------------------------------------------------------------------------------------- readouts ------
def load(tag, pair, L, cond, arena=1, onset="free"):
    return pd.read_pickle(_path(tag, scene(pair, L), cond, arena, onset))


def report_runs(tag, pair, L, cond):
    """The bouts of one map column (2x dish, equidistant start): the stored run, else simulated now (seconds)."""
    p = _path(tag, scene(pair, L), cond, ARENA_BIG, "anchored")
    return pd.read_pickle(p) if p.exists() else simulate(tag, scene(pair, L), cond, arena=ARENA_BIG, onset="anchored",
                                                         models=_cached_models(tag))


# ------------------------------------------------------------------------------------------------- videos ------
# ONE simulated fish, the same fish and block at several separations (or under several rules) side by side. The fish and
# block are CHOSEN to show the pattern (an illustration, not a random sample): among the 15 fish x 10 blocks of the map's
# runs, the (fish, block) whose bouts behind the pair (20-60 s) sit closest to the midline at the smallest L shown and
# closest to one track at the largest. The fish is then re-simulated with the seeds of that run (same start pose,
# latent, drift, random streams and tracks) and checked against the stored run when there is one. Each group is drawn
# as 4 dots around its centroid (display only: the model sees a group as its centroid, count and moving flag); before
# the block's first bout the pair is held at its start (the model sees the targets only at its bouts).
VIDEO_DIR = paths.EXPERIMENTS / "two_dot_groups"
VIDEO_SPEED, VIDEO_TRAIL_S = 3.0, 10.0
DOT_OFFS_MM = np.array([[0.0, 3.5], [3.5, 0.0], [0.0, -3.5], [-3.5, 0.0]])
COL_A, COL_B, COL_FISH = "#d62728", "#1f77b4", "#111111"
TAGS = {("M6", 0): "26e", ("M6", 101): "30d", ("F", 0): "14f", ("F", 101): "32b"}
TAG_LABEL = {v: f"{m} seed {s}" for (m, s), v in TAGS.items()}
_MODELS = {}


def _cached_models(tag):
    if tag not in _MODELS:
        _MODELS[tag] = _models(tag)
    return _MODELS[tag]


def pick(tag, Ls, cond="effects", pair="4v4", min_behind=8):
    """(fish, block, table): score = |mean lateral| / (L/2) at max(Ls) minus mean |lateral| / (L/2) at min(Ls), over the
    bouts behind the pair 20-60 s after onset, both with at least ``min_behind`` such bouts."""
    lo, hi = min(Ls), max(Ls)
    per = {}
    for L in (lo, hi):
        b = report_runs(tag, pair, L, cond)
        b = b[(b.t_blk >= 20) & (b.along_mm <= 0)]
        per[L] = b.groupby(["fish", "blk"]).lat_mm.agg(n="size", absm=lambda v: np.abs(v).mean(), m="mean")
    T = per[lo].join(per[hi], lsuffix="_lo", rsuffix="_hi", how="inner")
    T = T[(T.n_lo >= min_behind) & (T.n_hi >= min_behind)]
    T["score"] = T.m_hi.abs() / (hi / 2) - T.absm_lo / (lo / 2)
    k, j = T.score.idxmax()
    return int(k), int(j), T.sort_values("score", ascending=False)


def rerun(tag, pair, L, k, cond="effects"):
    """Fish k of the map's run (``report_runs``) simulated again with its own seeds: (generator output, provider, dish)."""
    models = _cached_models(tag)
    m = models[k // (N_FISH // len(models))]
    _, dish, ref = MF._programme()
    dish = (float(dish[0]), float(dish[1]), float(dish[2]) * ARENA_BIG)
    rng = np.random.default_rng([0, k])
    pose = ref.sample_start_pose(rng, dish)
    rule, vis = (cond, None) if cond in RULES else ("effects", () if cond == "blind" else REFS[cond])
    prov = AnchoredParallelProvider(scene(pair, L), k, dish, N_BLOCKS, visible=vis)
    rec = SR.make_synthetic_record(int(prov.T / 0.2) + 100, pose, dish, fish_id=k)
    g = m.sample(rec, rng, stim_provider=prov, stop_time_s=prov.T, block_seed=(0, k), multi_rule=rule)
    p = _path(tag, scene(pair, L), cond, ARENA_BIG, "anchored")
    if p.exists():                                       # the stored run: the same fish, the same bouts
        b = pd.read_pickle(p)
        keep = sorted(t for t in prov.rec if t < len(g["dtheta"]))
        if not np.allclose(np.asarray(g["dtheta"])[keep], b[b.fish == k].dth.to_numpy()):
            raise RuntimeError("the re-simulated fish differs from the stored run")
    return g, prov, dish


def _panel(L, g, prov, dish, j, taus, label):
    """One dish of a two-group video (``twin.videos`` panel), in mm centred on the tracks, and its lateral trace."""
    from twin import videos as V
    t0 = (prov.lead_f + j * prov.cyc_f) / prov.fps
    fish = V.frames(g["t_onset_s"], g["x_px"], g["y_px"], t0 + taus)
    t_anchor = prov.anchor[j][2] if j in prov.anchor else 0.0
    A, B, lat = [], [], []
    for i, tb in enumerate(taus):
        (pa, pb), _, mid, nA = prov._state(j, int(round(max(tb, t_anchor) * prov.fps)))
        A.append(pa); B.append(pb); lat.append(float(np.dot(fish[i] - mid, nA)) / PX_MM)
    cx, cy, R = dish
    tc = prov.centre[j] if prov.centre[j] is not None else np.array([cx, cy])
    mm = lambda p: (np.asarray(p, float) - tc) / PX_MM                   # noqa: E731
    u, nA, _ = prov.geom[j]
    lines = [dict(x=[(sd * prov.half * nA + sg * prov.amp * u)[0] / PX_MM for sg in (-1, 1)],
                  y=[(sd * prov.half * nA + sg * prov.amp * u)[1] / PX_MM for sg in (-1, 1)], color=col, lw=6,
                  alpha=0.10) for sd, col in ((1, COL_A), (-1, COL_B))]          # the two tracks
    lines.append(dict(x=[-prov.amp * u[0] / PX_MM, prov.amp * u[0] / PX_MM],
                      y=[-prov.amp * u[1] / PX_MM, prov.amp * u[1] / PX_MM], color="0.5", lw=0.8, ls="--"))   # midline
    dc = mm([cx, cy])
    half = 75.0
    return dict(xy=mm(fish)[None], view=(-half, half, -half, half), dish=(dc[0], dc[1], R / PX_MM), label=label,
                title_size=12, colors=[COL_FISH], lines=lines, zoom_note=False,
                dots=[dict(xy=np.transpose(mm(np.array(P_))[:, None] + DOT_OFFS_MM, (1, 0, 2)), color=c, size=14)
                      for P_, c in ((A, COL_A), (B, COL_B))],
                trace=dict(t=taus, y=np.array(lat), lines=[(L / 2, COL_A), (-L / 2, COL_B), (0.0, "0.5")],
                           ylim=(-58, 58), ylabel="lateral (mm)", xlabel="time in block (s)", span=(0, 20)))


def _footer(taus):
    return [f"t = {tb:4.1f} s into the block (×{VIDEO_SPEED:g} speed)"
            + ("   [read: 20-60 s]" if tb >= 20 else "   [onset phase, not read]") for tb in taus]


def video(tag="14f", Ls=(20.0, 50.0, 80.0), cond="effects", pair="4v4", out_dir=VIDEO_DIR, fish_block=None,
          force=False):
    """The same fish and block at the separations ``Ls`` (one column each): mp4 path, (fish, block). An existing mp4
    is kept unless ``force``."""
    from twin import videos as V
    k, j = fish_block or pick(tag, Ls, cond, pair)[:2]
    taus = np.arange(0.0, STIM_S, VIDEO_SPEED / V.FPS)
    title = (f"{TAG_LABEL[tag]}, rule '{cond}', {pair}, dish of twice the radius, equidistant start: fish {k}, block {j} "
             f"at {len(Ls)} separations\nred = group A, blue = group B, black = the fish ({VIDEO_TRAIL_S:g} s trail); "
             "below: its lateral position (lines: the tracks; dashed: the midline; grey: 0-20 s, not read)")
    out = Path(out_dir) / f"{pair}_{tag}_{cond}_L{'-'.join(f'{L:g}' for L in Ls)}.mp4"
    if out.exists() and not force:
        return out, (k, j)
    panels = [_panel(L, *rerun(tag, pair, L, k, cond), j, taus, f"L = {L:g} mm") for L in Ls]
    V.write(panels, taus, title, _footer(taus), out, speed=VIDEO_SPEED, trail_s=VIDEO_TRAIL_S, dpi=100)
    return out, (k, j)


def rules_video(tag="14f", L=80.0, conds=RULES, pair="4v4", fish_block=None, out_dir=VIDEO_DIR, force=False):
    """The same fish and block at one separation under each rule (one column each): mp4 path, (fish, block). The fish
    and block are picked under 'effects' at L = 20 mm and ``L`` unless ``fish_block`` is given; an existing mp4 is kept
    unless ``force``."""
    from twin import videos as V
    k, j = fish_block or pick(tag, (20.0, L), "effects", pair)[:2]
    taus = np.arange(0.0, STIM_S, VIDEO_SPEED / V.FPS)
    title = (f"{TAG_LABEL[tag]}, {pair}, L = {L:g} mm, dish of twice the radius, equidistant start: fish {k}, block {j} "
             f"under each rule\nred = group A, blue = group B, black = the fish ({VIDEO_TRAIL_S:g} s trail); below: its "
             "lateral position (lines: the tracks; dashed: the midline; grey: 0-20 s, not read)")
    out = Path(out_dir) / f"{pair}_{tag}_rules_L{L:g}.mp4"
    if out.exists() and not force:
        return out, (k, j)
    panels = [_panel(L, *rerun(tag, pair, L, k, c), j, taus, f"rule '{c}'") for c in conds]
    V.write(panels, taus, title, _footer(taus), out, speed=VIDEO_SPEED, trail_s=VIDEO_TRAIL_S, dpi=100)
    return out, (k, j)


# ----------------------------------------------------------------------------------------------- validate ------
def validate(tag="26e"):
    """With only group A visible, the four rules reduce to that group's row: the same bouts under each."""
    sc = scene("4v4", 20)
    m = _models(tag)[0]
    _, dish, ref = MF._programme()
    out = {}
    for rule in RULES:
        rng = np.random.default_rng([0, 0])
        pose = ref.sample_start_pose(rng, dish)
        prov = ParallelProvider(sc, 0, dish, 2, visible=(0,))
        rec = SR.make_synthetic_record(int(prov.T / 0.2) + 100, pose, dish, fish_id=0)
        out[rule] = np.asarray(m.sample(rec, rng, stim_provider=prov, stop_time_s=prov.T, block_seed=(0, 0),
                                        multi_rule=rule)["dtheta"])
    same = all(len(v) == len(out["effects"]) and np.allclose(v, out["effects"], atol=1e-9) for v in out.values())
    print(f"one visible group: {', '.join(RULES)} give the same bouts: {same} ({len(out['effects'])} bouts)")
    return same


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "validate":
        sys.exit(0 if validate() else 1)
    elif cmd == "run-report":
        run_all(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 14)
    else:
        raise SystemExit(__doc__)
