"""Two moving targets at a controlled separation: does a simulated fish average them or choose one?

The paradigm of Sridhar et al. (2021, PNAS), where zebrafish following two virtual conspecifics moving back and forth in
parallel switch from swimming between them to following one as their lateral separation grows. Here the targets are two
dot groups (A and B) whose centroids move back and forth along two parallel tracks through the dish centre, at lateral
separation L, both at the same speed and in step; the trained models (no retraining) see them through one of the
combination rules of fit_MLP_joint (effects, add, avg, nearest), and the references show only A, or nothing ('blind'),
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
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from twin import fit_MLP_border as MB
from twin import fit_MLP_joint as JT
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
UNIT_LABEL = {"26e": ("M6", 0), "30d": ("M6", 101), "14f": ("F", 0), "32b": ("F", 101)}


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






COLS = ("absx", "x", "abslat", "lat", "closerA", "middle")




DIP_BW_MM = 3.0                                                  # kernel width of the lateral density (mm)
DIP_SPLIT = 0.9                                                  # a clear trough: the midline >= 10 % below the branches




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
