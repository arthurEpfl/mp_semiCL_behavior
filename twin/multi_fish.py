"""Several simulated fish in one dish, each seeing the others as moving dots: do they shoal?

The data never show a fish another fish, so this is an extrapolation of the dot response. Every fish runs the trained
generator unchanged (``fit_MLP_joint.generate_mix_joint_loop``, multi-group mode); only its stimulus provider is new:
at each bout onset it reads where the OTHER fish are and turns them into dot groups.

COUPLING. One thread per fish around the unchanged generator. The generator asks its provider for the stimulus exactly
once per bout, at the bout's onset, with the pre-bout pose. The provider blocks until it is that fish's turn: the fish
with the smallest (pending onset, index) goes next, so every fish sees each other fish where it is after that fish's
bouts with onset < t (or = t with a lower index). Bouts are instantaneous, so fish may overlap (dots overlap freely in
training too) unless the collision rule is on (``collision_mm``). Deterministic: the result does not depend on thread
timing.

PERCEPTION. A fish sees every other fish within RADIUS_PX = 280 px = 78 mm at any bearing, always as MOVING dots. Beyond
280 px the log-range input is saturated, so the model's pull would stay the same at any larger range (it does not fade:
P(toward) ~0.61 at 250-320 px vs 0.50 blind); the horizon is a choice that keeps every range input inside the trained
values, not a property of the model. In the training dish it hides only the rare neighbour 78-98 mm away. Neighbours close to each other are pooled into one dot group ('cluster' mapping):
agglomerative, closest centroids first, a merge kept only while every member stays within the spread a trained dot group
of that size had (ENV_PX: farthest moving dot from its centroid over the whole programme). A group's count is its size,
capped at 8 (the network does NOT clip the count above 8 itself). The groups are combined by the two-group rules
('effects' = each group's effect added; 'nearest' = the nearest group only; ``fit_MLP_joint.MULTI_RULES``).

DISH. The 5x dish passes its own (cx, cy, R) everywhere. Every wall mechanism of the model is relative to R (the wall input
(R - r)/R and the bearing to the centre, the 2-10 mm rejection band, the forbidden ring, start poses at the same r/R), so
the fish senses the wall as in the training dish magnified 5x; steps, the 78 mm view and the 25 mm shoal criterion are
absolute.

CLOCK. 'late': the block clock is held at 60 s (the largest block time in training, exp(-60/4) ~ 0): a neighbour has no
onset. Through the provider hook ``block_time_s`` (the generator's own rule would restart the clock whenever the
neighbours regroup). F's correction term is count-gated and acts only in the first 20 s of a block, so under 'late' it
is inactive; F's network weights are unchanged by the correction (checked: 16/16 tensors identical to its base).

    python -m twin.multi_fish --checks            # correctness checks (hook, parity, determinism, rule scenes)
    python -m twin.multi_fish --run-grid          # the shoal grid: 2 dishes x 2 / 5 / 10 fish x F and its variants x 2 seeds
    python -m twin.multi_fish --analyse-grid      # -> output/multi_fish/summary_grid.json
    python -m twin.multi_fish --run-pairdm        # pairs of the screen-trained D4, with and without the dot's direction
    python -m twin.multi_fish --analyse-pairdm    # -> output/multi_fish/summary_pairdm.json
    python -m twin.multi_fish --render NPZ MP4    # one video
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np

from twin import paths

OUT = paths.MULTI_FISH
PX_MM = 3.60                       # px per mm of the physical dish (the thesis' scale; distances are reported in mm)
RADIUS_PX = 280.0                  # perception horizon: the log-range input saturates here (70 px x 4 = 78 mm)
ENV_PX = {1: 0.0, 2: 37.0, 4: 59.2, 6: 67.4, 8: 74.3}   # trained spread envelope (moving groups, measured 2026-10-05)
N_CAP = 8                          # largest trained group size
LATE_S = 60.0                      # largest block time in training (60 s blocks)
TAIL_S = 10.0                      # every fish runs to T + 10 s (dt <= 10 s): every pose up to T is published
POSE_SEED, FISH_SEED, SCR_SEED = 7101, 7102, 7103
SHIFTS_S = tuple(60.0 * k for k in range(1, 11))   # Larsch / Kappel: 10 time shifts
CHUNK_S = 300.0                                     # Larsch / Kappel: 5-min chunks

# tag -> (registry key of the checkpoints, group-repulsion variant or None): F = 14f (seed 0) / 32b (seed 101); "<F>_<v>"
# = F with the group repulsion term of variant v (twin/grouprep_fit.py, ``variants``): calibrated turn (grT), strength
# x2 / x4 / x8, + collision rule (grTC). The screen-trained D4 without / with the dot's direction of motion (sD4, sD4dm)
# saw one dot, always moving, never a dish without one: run ONLY in PAIRS (single-partner mode), never blind.
GRID_VARS = ("F", "grT", "grT2", "grT4", "grT8", "grTC")
MODELS = {"14f": ("14f", None), "32b": ("32b", None)}
for _st in ("14f", "32b"):
    for _v in GRID_VARS[1:]:
        MODELS[f"{_st}_{_v}"] = (_st, _v)
SCREEN_ARMS = {"sD4": "S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st", "sD4dm": "S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st_dm"}


def _cond(tag, N, G, arena, T, rule=None, blind=False, scramble=False, t0=None, clock="late", start="free"):
    return dict(tag=tag, N=N, G=G, arena=arena, T=float(T), rule=rule or "effects", blind=blind, scramble=scramble,
                mapping="cluster", clock=clock, t_read=float(t0 if t0 is not None else (300.0 if arena == 1 else 600.0)),
                start=start)


# THE SHOAL GRID (thesis "Multiple fish"): every dish x group size x variant x seed. Fish (group g, index i) has the same
# start pose, random stream, latent and fold model (g % 5) in every condition of a seed, so conditions pair by group.
# Training dish: 30 min, readout from 5 min, free starts. 5x dish: 60 min, readout from 10 min, NEAR starts (each group
# starts within 25 mm of its fish 0, so the fish can see each other). Each dish has a blind pool of 10 fish per group.
GRID_N = {2: 50, 5: 40, 10: 40}
GRID_DISH = {"A1": dict(arena=1, T=1800, start="free"), "A5": dict(arena=5, T=3600, start="near")}
NEAR_MM = 25.0


def grid_name(dish_, N, var, seed):
    return f"grid_{dish_}_N{N}_{var}" + ("_s101" if seed == 101 else "")


CONDS = {}
for _seed, _st, _pre in ((0, "14f", ""), (101, "32b", "s101_")):
    CONDS[f"{_pre}A1_blind"] = _cond(_st, 10, 50, 1, 1800, blind=True)
    for _d, _dd in GRID_DISH.items():
        for _N, _G in GRID_N.items():
            for _v in GRID_VARS:
                CONDS[grid_name(_d, _N, _v, _seed)] = _cond(_st if _v == "F" else f"{_st}_{_v}", _N, _G, _dd["arena"],
                                                            _dd["T"], "effects", start=_dd["start"])
    CONDS["grid_A5_blind" + ("_s101" if _seed == 101 else "")] = _cond(_st, 10, 50, 5, 3600, blind=True, start="near")
GRID_CONDS = dict(CONDS)
# PAIRS WITH THE DOT'S DIRECTION OF MOTION (thesis "Adding alignment"): screen-trained D4 with and without the input, both
# seeds, the grid's two dishes and starts, 50 pairs; the partner is one moving dot, always in view.
PAIRDM_CONDS = {}
for _tag in ("sD4_s0", "sD4_s101", "sD4dm_s0", "sD4dm_s101"):
    for _d, _dd in GRID_DISH.items():
        PAIRDM_CONDS[f"pairdm_{_d}_{_tag}"] = dict(_cond(_tag, 2, 50, _dd["arena"], _dd["T"], start=_dd["start"]),
                                                   single=True)
CONDS.update(PAIRDM_CONDS)


def blind_of(cond):
    """The blind pool a condition pairs with: the blind condition of the same model and dish."""
    c = CONDS[cond]
    tag = c["tag"].split("_")[0]                         # a variant's blind fish are F's (the term needs neighbours)
    for k, b in CONDS.items():
        if (b["blind"] and b["tag"] == tag and b["arena"] == c["arena"] and b["G"] >= c["G"] and b["N"] >= c["N"]
                and b.get("start", "free") == c.get("start", "free")):
            return k
    raise KeyError(f"no blind pool for {cond} (model {c['tag']}, dish x{c['arena']})")


# --------------------------------------------------------------------------- #
# Loading (cached per process)
# --------------------------------------------------------------------------- #
_CACHE = {}


def _imports():
    if "MF" not in _CACHE:
        from twin import model_figures as MF
        from twin import stim_rollout as SR
        _CACHE.update(MF=MF, SR=SR)
    return _CACHE["MF"], _CACHE["SR"]


def models(tag):
    """The five fold models of a tag (the rej band merged in, as every rollout)."""
    key = ("models", tag)
    if key not in _CACHE:
        MF, _ = _imports()
        from twin import registry
        if tag.startswith("sD4"):
            name, seed = tag.split("_s")
            d, arm, var = paths.CKPTS / f"screen_s{seed}", SCREEN_ARMS[name], None
        else:
            k, var = MODELS[tag]
            d, arm = registry.ckpt_dir(k), registry.arm(k)
        with contextlib.redirect_stdout(io.StringIO()):
            ms, _ = MF.fold_models(d, arm)
        if var:
            from twin import grouprep_fit as GF
            gp, extra = GF.variants(MODELS[tag][0])[var]
            for m in ms.values():
                cf = dict(m._cfg_feat)
                cf["grouprep"] = dict(gp)
                cf.update(extra)
                m._cfg_feat = cf
        _CACHE[key] = [ms[f] for f in sorted(ms)]
    return _CACHE[key]


def dish(arena=1):
    MF, _ = _imports()
    _, d, _ = MF._programme()
    return (float(d[0]), float(d[1]), float(d[2]) * float(arena))


def start_pose(g, i, arena=1):
    MF, _ = _imports()
    _, _, ref = MF._programme()
    return ref.sample_start_pose(np.random.default_rng([POSE_SEED, int(g), int(i)]), dish(arena))


# --------------------------------------------------------------------------- #
# Perception: neighbours -> dot groups
# --------------------------------------------------------------------------- #
def envelope_px(n):
    """Largest member-to-centroid distance a trained moving group of ``n`` dots had (interpolated in log2 n)."""
    if n <= 1:
        return 0.0
    ks = sorted(ENV_PX)
    return float(np.interp(np.log2(min(n, N_CAP)), np.log2(ks), [ENV_PX[k] for k in ks]))


def cluster(pts):
    """Agglomerative grouping of neighbour positions (k, 2) px: closest centroids first, a merge kept only while its
    farthest member stays within ``envelope_px`` of the merged size (the 8-dot envelope beyond 8). Deterministic."""
    cl = [[k] for k in range(len(pts))]
    bad = set()
    while len(cl) > 1:
        best = None
        for a in range(len(cl)):
            ca = pts[cl[a]].mean(0)
            for b in range(a + 1, len(cl)):
                key = (frozenset(cl[a]), frozenset(cl[b]))
                if key in bad:
                    continue
                d = float(np.hypot(*(ca - pts[cl[b]].mean(0))))
                if best is None or d < best[0]:
                    best = (d, a, b, key)
        if best is None:
            break
        _, a, b, key = best
        m = cl[a] + cl[b]
        c = pts[m].mean(0)
        if float(np.max(np.hypot(*(pts[m] - c).T))) <= envelope_px(len(m)) + 1e-9:
            cl = [x for k, x in enumerate(cl) if k not in (a, b)] + [m]
        else:
            bad.add(key)
    return cl


def neighbour_groups(x, y, others, mapping="cluster"):
    """Dot groups ``[(cx, cy, moving=1, n)]`` a fish at (x, y) sees, from the other fish's (x, y) positions (k, 2), and a
    per-bout log record."""
    pts = np.asarray(others, float).reshape(-1, 2)
    if len(pts):
        pts = pts[np.hypot(pts[:, 0] - x, pts[:, 1] - y) <= RADIUS_PX]
    if not len(pts):
        return [], dict(n_vis=0, G=0, n_raw_max=0, spread_ratio=0.0, r_near=np.nan, near=[])
    cl = cluster(pts) if mapping == "cluster" else [[k] for k in range(len(pts))]
    out, ratio, near = [], 0.0, []
    for c in cl:
        cx, cy = pts[c].mean(0)
        q = pts[c][int(np.argmin(np.hypot(pts[c][:, 0] - x, pts[c][:, 1] - y)))]   # the group's nearest member
        near.append((float(q[0]), float(q[1])))
        if len(c) > 1:
            ratio = max(ratio, float(np.max(np.hypot(*(pts[c] - (cx, cy)).T))) / envelope_px(len(c)))
        out.append((float(cx), float(cy), 1.0, float(min(len(c), N_CAP))))
    r = [float(np.hypot(g[0] - x, g[1] - y)) for g in out]
    return out, dict(n_vis=len(pts), G=len(out), n_raw_max=max(len(c) for c in cl), spread_ratio=ratio,
                     r_near=min(r), near=near)


# --------------------------------------------------------------------------- #
# Coupling
# --------------------------------------------------------------------------- #
class Scheduler:
    """Token passing among N fish threads: the fish with the smallest (pending onset, index) runs; all others wait at
    their pending onset with their pre-bout pose published. One Event per fish (no thundering herd)."""

    def __init__(self, n, timeout=600.0):
        self.n, self.timeout = n, float(timeout)
        self.lock = threading.Lock()
        self.t = [np.inf] * n
        self.pose = [None] * n
        self.arrived = [False] * n
        self.done = [False] * n
        self.ev = [threading.Event() for _ in range(n)]
        self.err = None

    def _holder(self):
        best, h = None, None
        for j in range(self.n):
            if self.done[j]:
                continue
            if not self.arrived[j]:
                return None
            k = (self.t[j], j)
            if best is None or k < best:
                best, h = k, j
        return h

    def _take(self, i):
        self.arrived[i] = False
        ti = self.t[i]
        for j in range(self.n):                     # causality: nobody is read from its future
            if j != i and not self.done[j]:
                assert (self.t[j], j) > (ti, i), "scheduler: a neighbour is behind the reader"
        return [p for p in self.pose]

    def arrive(self, i, t, pose):
        with self.lock:
            if self.err:
                raise RuntimeError(f"group aborted: {self.err}")
            self.t[i], self.pose[i], self.arrived[i] = float(t), pose, True
            h = self._holder()
            if h == i:
                return self._take(i)
            if h is not None:
                self.ev[h].set()
            ev = self.ev[i]
        while True:
            if not ev.wait(self.timeout):
                self.fail(f"fish {i} timed out at t={t:.2f}")
            with self.lock:
                if self.err:
                    raise RuntimeError(f"group aborted: {self.err}")
                ev.clear()
                if self._holder() == i:
                    return self._take(i)

    def finish(self, i):
        with self.lock:
            self.done[i], self.arrived[i], self.t[i] = True, False, np.inf
            h = self._holder()
            if h is not None:
                self.ev[h].set()

    def fail(self, msg):
        with self.lock:
            if self.err is None:
                self.err = str(msg)
            for e in self.ev:
                e.set()
        raise RuntimeError(f"group aborted: {msg}")


class FishView:
    """Fish ``i``'s stimulus provider: the other fish as dot groups (``groups``), the block clock (``block_time_s``)."""

    def __init__(self, i, sched, sees, mapping="cluster", clock="late", scramble_rng=None, single=False):
        self.i, self.s, self.sees = i, sched, list(sees)
        self.single = bool(single)                     # pairs: the partner as ONE moving dot, always in view
        self.mapping, self.clock, self.scr = mapping, clock, scramble_rng
        self.t_entry, self.prev_vis = 0.0, False
        self.log = []                                  # (t_now, n_vis, G, n_raw_max, spread_ratio, r_near, n_out_dish)
        self.n_calls = 0
        self._others = []                              # this bout's neighbour positions (px), for dots()
        self._near = []                                # this bout's groups' nearest members (px), for group_nearest()

    def __call__(self, t, time_s, x, y, h):            # single-group interface: the single-partner mode only
        if not self.single:
            raise RuntimeError("FishView is a multi-group provider")
        self.n_calls += 1
        poses = self.s.arrive(self.i, time_s, (float(x), float(y), float(h)))
        px, py = poses[self.sees[0]][:2]
        self._others = [(px, py)]
        self.log.append((time_s, 1, 1, 1, 0.0, float(np.hypot(px - x, py - y)), 0))
        return float(px), float(py), 1.0, 1.0

    def groups(self, t, time_s, x, y, h):
        self.n_calls += 1
        if self.s is None or not self.sees:             # blind fish: nobody visible, no waiting
            self.prev_vis = False
            self._others = []
            self._near = []
            self.log.append((time_s, 0, 0, 0, 0.0, np.nan, 0))
            return []
        poses = self.s.arrive(self.i, time_s, (float(x), float(y), float(h)))
        others = [poses[j][:2] for j in self.sees]
        self._others = others
        out, lg = neighbour_groups(x, y, others, self.mapping)
        self._near = lg["near"]
        if out and not self.prev_vis:
            self.t_entry = float(time_s)
        self.prev_vis = bool(out)
        n_out = 0
        if self.scr is not None and out:                 # control: every group at its own range, random bearing
            sc = []
            for gx, gy, gm, gn in out:
                r = float(np.hypot(gx - x, gy - y)); a = float(self.scr.uniform(-np.pi, np.pi))
                sc.append((x + r * np.cos(a), y + r * np.sin(a), gm, gn))
            out = sc
        if self.dish is not None and out:
            cx, cy, R = self.dish
            n_out = int(sum(np.hypot(g[0] - cx, g[1] - cy) > R for g in out))
        self.log.append((time_s, lg["n_vis"], lg["G"], lg["n_raw_max"], lg["spread_ratio"], lg["r_near"], n_out))
        return out

    dish = None

    def group_nearest(self):
        """Each group's nearest member (px), aligned with this bout's groups() -- the group repulsion term."""
        return list(getattr(self, "_near", []))

    def neighbours_xy(self):
        """Every visible neighbour's position at this bout (px) -- the collision rule."""
        return np.asarray(getattr(self, "_others", []), float).reshape(-1, 2)

    def dots(self, t, time_s, x, y, h):
        """The INDIVIDUAL neighbours (not the groups) at this bout, for a per-dot term (the repulsion term). Reads the
        positions fetched by this bout's groups() call -- never the scheduler again, so the turn order is untouched."""
        o = self._others
        xs = np.array([q[0] for q in o], float)
        ys = np.array([q[1] for q in o], float)
        return xs, ys, np.ones(len(o), bool), np.ones(len(o)), float(len(o))

    def block_time_s(self, t, t_now):
        if self.clock == "late":
            return LATE_S
        return min(max(float(t_now) - self.t_entry, 0.0), LATE_S)   # 'entry': since the view last became non-empty


def _fish_out(o, prov):
    lg = np.array(prov.log, float).reshape(-1, 7)
    return dict(t=np.asarray(o["t_onset_s"], float), x=np.asarray(o["x_px"], float),
                y=np.asarray(o["y_px"], float), dth=np.asarray(o["dtheta"], np.float32),
                d=np.asarray(o["distance_px"], np.float32), dt=np.asarray(o["dt_s"], np.float32),
                end=float(o["end_time_s"]), hits=int(o.get("n_hits", 0)), log=lg.astype(np.float32),
                n_calls=int(prov.n_calls), n_collide=int(o.get("n_collide", 0)),
                n_redraw_bouts=int(o.get("n_redraw_bouts", 0)))


def run_group(spec, g, ms=None):
    """Simulate group ``g`` of condition ``spec``. Returns a list of per-fish dicts (see ``_fish_out``)."""
    _, SR = _imports()
    ms = ms or models(spec["tag"])
    m = ms[int(g) % len(ms)]
    N, T = int(spec["N"]), float(spec["T"]) + TAIL_S
    dsh = dish(spec["arena"])
    poses = [start_pose(g, i, spec["arena"]) for i in range(N)]
    if spec.get("start", "free") == "near":            # every fish within NEAR_MM of fish 0 (fish-specific draw)
        x0, y0 = poses[0][0], poses[0][1]
        for i in range(1, N):
            r_ = np.random.default_rng([POSE_SEED, int(g), i, 1])
            rr = NEAR_MM * PX_MM * np.sqrt(r_.uniform()); aa = r_.uniform(0, 2 * np.pi)
            poses[i] = (x0 + rr * np.cos(aa), y0 + rr * np.sin(aa), poses[i][2])
    blind = bool(spec.get("blind"))
    sched = None if blind else Scheduler(N)
    provs = []
    for i in range(N):
        p = FishView(i, sched, [] if blind else [j for j in range(N) if j != i], spec.get("mapping", "cluster"),
                     spec.get("clock", "late"),
                     np.random.default_rng([SCR_SEED, int(g), i]) if spec.get("scramble") else None,
                     single=bool(spec.get("single")))
        p.dish = dsh
        provs.append(p)
    outs, errs = [None] * N, []

    def work(i):
        try:
            rng = np.random.default_rng([FISH_SEED, int(g), i])
            rec = SR.make_synthetic_record(int(T / 0.05) + 100, poses[i], dsh, fish_id=1000 * int(g) + i)
            outs[i] = m.sample(rec, rng, stim_provider=provs[i], stop_time_s=T,
                               multi_rule=None if spec.get("single") else spec.get("rule", "effects"))
        except BaseException as e:                       # noqa: BLE001 -- reported below
            errs.append(f"fish {i}: {type(e).__name__}: {e}")
            if sched is not None:
                with contextlib.suppress(RuntimeError):
                    sched.fail(errs[-1])
        finally:
            if sched is not None:
                sched.finish(i)

    if blind:
        for i in range(N):
            work(i)
    else:
        th = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(N)]
        for x in th:
            x.start()
        for x in th:
            x.join()
    if errs:
        raise RuntimeError("; ".join(errs[:3]))
    res = [_fish_out(o, p) for o, p in zip(outs, provs)]
    for f in res:
        if f["end"] < T - 1e-9:
            raise RuntimeError(f"group {g}: a fish ended at {f['end']:.1f} s < {T:.1f} s (bout budget exhausted)")
        if f["n_calls"] != len(f["t"]):
            raise RuntimeError(f"group {g}: provider calls {f['n_calls']} != bouts {len(f['t'])}")
        if f["t"][-1] < float(spec["T"]):                # a finished fish is read stale only after the readout end
            raise RuntimeError(f"group {g}: last onset {f['t'][-1]:.1f} s < readout end {spec['T']:.1f} s")
    return res


# --------------------------------------------------------------------------- #
# Running the conditions
# --------------------------------------------------------------------------- #
def _worker_init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    with contextlib.suppress(Exception):
        import torch
        torch.set_num_threads(1)


def _task(args):
    cond, g = args
    t0 = time.time()
    res = run_group(CONDS[cond], g)
    return cond, g, res, time.time() - t0


def run_conditions(conds=None, workers=16, log=print):
    """Simulate every group of ``conds`` (all by default) in a process pool; one pickle per condition."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    conds = list(conds or CONDS)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(c, g) for c in conds for g in range(CONDS[c]["G"])]
    # the heaviest groups first, so the pool drains evenly
    tasks.sort(key=lambda cg: -(CONDS[cg[0]]["N"] ** 1.5 * CONDS[cg[0]]["T"] * (0.3 if CONDS[cg[0]]["blind"] else 1)))
    got = {c: {} for c in conds}
    secs = {c: 0.0 for c in conds}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_worker_init) as ex:
        futs = {ex.submit(_task, t): t for t in tasks}
        failed = []
        for k, f in enumerate(as_completed(futs), 1):
            try:
                c, g, res, el = f.result()
            except Exception as e:                           # noqa: BLE001 -- one bad group must not lose the run
                c, g = futs[f]
                failed.append((c, g, repr(e)))
                log(f"FAILED {c} group {g}: {e!r}")
                continue
            got[c][g] = res
            secs[c] += el
            if len(got[c]) == CONDS[c]["G"]:
                blob = dict(cond=c, spec=CONDS[c], groups=[got[c][i] for i in range(CONDS[c]["G"])],
                            cpu_s=secs[c], made=time.strftime("%Y-%m-%d %H:%M"))
                with open(OUT / f"{c}.pkl", "wb") as fh:
                    pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
                log(f"[{time.time() - t0:7.0f} s] {c}: {CONDS[c]['G']} groups done ({secs[c] / 3600:.2f} CPU-h)")
            elif k % 25 == 0:
                log(f"[{time.time() - t0:7.0f} s] {k}/{len(tasks)} groups")
    if failed:
        (OUT / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
        raise RuntimeError(f"{len(failed)} groups failed (see {OUT / 'failed.json'}); complete conditions were saved")
    return got


def load(cond):
    with open(OUT / f"{cond}.pkl", "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #
def pos_at(f, taus):
    """(len(taus), 2) px: where the fish is at times ``taus`` (after its last bout with onset <= tau)."""
    k = np.clip(np.searchsorted(f["t"], taus, side="right"), 0, len(f["t"]) - 1)
    return np.stack([f["x"][k], f["y"][k]], 1).astype(float)


def heading_at(f, taus):
    """Travel heading (rad) of the last bout before each tau."""
    k = np.clip(np.searchsorted(f["t"], taus, side="right"), 1, len(f["t"]) - 1)
    return np.arctan2(f["y"][k].astype(float) - f["y"][k - 1], f["x"][k].astype(float) - f["x"][k - 1])


def grid(spec, step=1.0):
    return np.arange(spec["t_read"], spec["T"], step)


def tracks(blob, n=None, step=1.0):
    """(groups, fish, frames, 2) mm positions on the readout grid."""
    spec = blob["spec"]
    taus = grid(spec, step)
    n = n or spec["N"]
    return np.array([[pos_at(f, taus) for f in grp[:n]] for grp in blob["groups"]]) / PX_MM, taus


def _nn1(P):
    """P (fish, frames, 2) -> (frames,) mean over fish of the nearest-neighbour distance."""
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    D[np.arange(len(P)), np.arange(len(P))] = np.inf
    return D.min(1).mean(0)




def _shuffled(T, n, n_pseudo=200, seed=0, mode="harpaz", folds=None, sep_s=600):
    """Pseudo-groups (n_pseudo, n, frames, 2) of fish that never saw each other.

    'harpaz' (PRIMARY, Harpaz et al. 2021): ``n`` fish from ``n`` DIFFERENT groups of the condition, overlaid in time.
    The groups mix fold models as real groups mix clutches; when fish see each other the fold models differ in where
    their groups sit, but that difference is itself caused by the shoaling (tight groups sit centrally), so it is not
    removed here.
    'fold' (sensitivity): fish from groups of ONE fold model (pseudo-group p uses fold p % 5); a fold has fewer groups
    than fish, so a group is reused with time shifts of k x ``sep_s`` frames (1 Hz grid, circular). Conservative: it
    absorbs the fold-level part of the shoaling and a reused group's persistent location."""
    rng = np.random.default_rng(seed)
    G, Nf = T.shape[0], T.shape[1]
    if mode == "harpaz":
        return np.array([np.stack([T[g, rng.integers(Nf)] for g in rng.choice(G, size=n, replace=False)])
                         for _ in range(n_pseudo)])
    folds = np.arange(G) % 5 if folds is None else np.asarray(folds)
    labs = np.unique(folds)
    out = []
    for p in range(n_pseudo):
        gk = np.flatnonzero(folds == labs[p % len(labs)])
        gs = np.resize(rng.permutation(gk), n)                       # distinct groups first, then reuse
        idx = {g: rng.permutation(Nf) for g in set(gs.tolist())}
        slot, fish = {}, []
        for g in gs:
            k = slot.get(g, 0); slot[g] = k + 1
            fish.append(np.roll(T[g, idx[g][k % Nf]], -k * sep_s, axis=0))
        out.append(np.stack(fish))
    return np.array(out)




def _boot_D(nn_g, T, n, n_boot=300, n_pseudo=25, seed=0):
    """95% CI of mean D (Harpaz null) resampling groups AND re-drawing the pseudo-groups from the distinct resampled
    groups: the denominator is estimated from the same groups, so holding it fixed would make the interval too narrow."""
    rng = np.random.default_rng(seed)
    G = len(nn_g)
    out = []
    for _ in range(n_boot):
        gi = rng.choice(G, G)
        pool = np.unique(gi)                             # pseudo-groups from distinct groups only (no duplicate copies)
        if len(pool) < n:
            continue
        S = _shuffled(T[pool], n, n_pseudo, seed=int(rng.integers(1 << 31)))
        out.append(np.mean(np.log(nn_g[gi] / np.mean([_nn1(p) for p in S]))))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def _boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v))).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def pair_ai(A, B, taus, t0, T):
    """Larsch / Kappel attraction index of one pair (A, B: (frames, 2) mm on the 1 Hz grid ``taus``): per 5-min chunk,
    (IAD_shifted - IAD) / IAD_shifted, IAD_shifted = mean over 10 circular time shifts of B (60..600 s); mean over
    chunks."""
    W = len(taus)
    iad = np.linalg.norm(A - B, axis=1)
    sh = np.stack([np.linalg.norm(A - np.roll(B, -int(round(s)), axis=0), axis=1) for s in SHIFTS_S])   # 1 Hz grid
    ais = []
    for c0 in np.arange(t0, T, CHUNK_S):
        sel = (taus >= c0) & (taus < c0 + CHUNK_S)
        if sel.sum() < CHUNK_S * 0.9:
            continue
        s_ = sh[:, sel].mean()
        ais.append((s_ - iad[sel].mean()) / s_)
    return float(np.mean(ais)), float(np.median(iad)), float(np.median(sh))




def per_fold(values, G):
    v = np.asarray(values, float)
    return [float(np.nanmean(v[np.arange(G) % 5 == k])) for k in range(5)]




PAIR_BINS = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 70], float)




def grid_readout(cond, n_pseudo=200, nb=500):
    """The grid's readouts for one condition (pairs and groups alike): attraction index (Larsch & Baier; groups: mean
    over the pairs of a group), dispersion D (Harpaz et al.; pseudo-groups of fish from different groups), median
    nearest-neighbour distance (mm), close-range shape (0-2 and 2-4 mm relative to 4-8 mm, over the same fish blind),
    alignment (Harpaz et al.: |mean unit travel heading|, per second, mean over the readout) against the same fish blind
    and against pseudo-groups; the collision rule's redraw share. 95 % intervals over groups."""
    spec = CONDS[cond]
    N, G = spec["N"], spec["G"]
    blob = load(cond)
    T, taus = tracks(blob)
    bc = blind_of(cond)
    bl = load(bc)
    TB, _ = tracks(bl, N)
    TB = TB[:G]
    t0, Tend = spec["t_read"], spec["T"]
    pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    ai = np.array([np.mean([pair_ai(T[g, a], T[g, b], taus, t0, Tend)[0] for a, b in pairs]) for g in range(G)])
    ai_b = np.array([np.mean([pair_ai(TB[g, a], TB[g, b], taus, t0, Tend)[0] for a, b in pairs]) for g in range(G)])
    nn = np.array([_nn1(T[g]) for g in range(G)])
    nn_s = np.array([_nn1(q) for q in _shuffled(T, N, n_pseudo)])
    D = np.log(nn.mean(1) / nn_s.mean())
    nn_b = np.array([_nn1(TB[g]) for g in range(G)])
    nn_bs = np.array([_nn1(q) for q in _shuffled(TB, N, n_pseudo, seed=1)])
    D_b = np.log(nn_b.mean(1) / nn_bs.mean())

    def nnd(X):                                          # every fish's nearest-neighbour distance, every second
        Dm = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=-1)
        Dm[:, np.arange(N), np.arange(N)] = np.inf
        return Dm.min(2).ravel()
    nn_all = nnd(T); nn_all_b = nnd(TB)

    def dens(X):
        Dm = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=-1)
        v = Dm[:, ~np.eye(N, dtype=bool)].ravel()
        return np.histogram(v, PAIR_BINS)[0].astype(float) / len(v)
    ratio = dens(T) / dens(TB)
    ref = float(np.nanmean(ratio[2:4]))
    H = np.array([[heading_at(f, taus) for f in grp[:N]] for grp in blob["groups"]])
    HB = np.array([[heading_at(f, taus) for f in grp[:N]] for grp in bl["groups"][:G]])
    pol = np.abs(np.exp(1j * H).mean(1)).mean(1)
    pol_b = np.abs(np.exp(1j * HB).mean(1)).mean(1)
    rng = np.random.default_rng(3)
    Hs = np.array([[H[g, rng.integers(N)] for g in rng.choice(G, N, replace=False)] for _ in range(n_pseudo)])
    pol_s = np.abs(np.exp(1j * Hs).mean(1)).mean(1)
    bouts = sum(len(f["t"]) for grp in blob["groups"] for f in grp)
    redraw = sum(int(f.get("n_redraw_bouts", 0)) for grp in blob["groups"] for f in grp)

    def m_ci(v):
        return dict(mean=float(np.mean(v)), ci=_boot(v, nb))
    return dict(cond=cond, blind=bc, N=N, G=G, arena=spec["arena"], tag=spec["tag"],
                ai=m_ci(ai), ai_blind=m_ci(ai_b), D=m_ci(D), D_ci=_boot_D(nn.mean(1), T, N, n_boot=200), D_blind=m_ci(D_b),
                nn_median_mm=float(np.median(nn_all)), nn_median_blind_mm=float(np.median(nn_all_b)),
                nn_le2=float(np.mean(nn_all <= 2)), nn_le4=float(np.mean(nn_all <= 4)),
                shape_0_2=float(ratio[0] / ref), shape_2_4=float(ratio[1] / ref), ratio=ratio.tolist(),
                pol=m_ci(pol), pol_blind=m_ci(pol_b), pol_shuf=float(np.mean(pol_s)),
                redraw_share=redraw / max(bouts, 1))


def run_grid(workers=16, log=print):
    todo = [c for c in GRID_CONDS if not (OUT / f"{c}.pkl").exists()]
    log(f"grid: {len(todo)} conditions to run")
    if todo:
        run_conditions(todo, workers=workers, log=log)


def pair_readout_pseudo(cond, n_pseudo=200, nb=500):
    """Pairs read WITHOUT blind fish: attraction index, dispersion D, median nearest distance, close-range shape and
    alignment, the last two against PSEUDO-PAIRS (fish from different dishes overlaid) instead of the same fish blind.
    Used for the screen-trained models (which swim abnormally blind) and, for comparison, for F's grid pairs."""
    spec = CONDS[cond]
    N, G = spec["N"], spec["G"]
    blob = load(cond)
    T, taus = tracks(blob)
    t0, Tend = spec["t_read"], spec["T"]
    ai = np.array([pair_ai(T[g, 0], T[g, 1], taus, t0, Tend)[0] for g in range(G)])
    nn = np.array([_nn1(T[g]) for g in range(G)])
    P = _shuffled(T, N, n_pseudo)
    nn_s = np.array([_nn1(q) for q in P])
    D = np.log(nn.mean(1) / nn_s.mean())

    def dens(X):
        Dm = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=-1)
        v = Dm[:, ~np.eye(N, dtype=bool)].ravel()
        return np.histogram(v, PAIR_BINS)[0].astype(float) / len(v)
    ratio = dens(T) / dens(np.asarray(P))
    ref = float(np.nanmean(ratio[2:4]))
    H = np.array([[heading_at(f, taus) for f in grp[:N]] for grp in blob["groups"]])
    pol = np.abs(np.exp(1j * H).mean(1)).mean(1)
    rng = np.random.default_rng(3)
    Hs = np.array([[H[g, rng.integers(N)] for g in rng.choice(G, N, replace=False)] for _ in range(n_pseudo)])
    pol_s = np.abs(np.exp(1j * Hs).mean(1)).mean(1)
    nn_all = np.linalg.norm(T[:, 0] - T[:, 1], axis=-1).ravel()

    def m_ci(v):
        return dict(mean=float(np.mean(v)), ci=_boot(v, nb))
    return dict(cond=cond, N=N, G=G, arena=spec["arena"], tag=spec["tag"], ai=m_ci(ai), D=m_ci(D),
                nn_median_mm=float(np.median(nn_all)), shape_0_2=float(ratio[0] / ref), shape_2_4=float(ratio[1] / ref),
                ratio=ratio.tolist(), pol=m_ci(pol), pol_pseudo=float(np.mean(pol_s)),
                pol_above_pseudo=float(np.mean(pol) - np.mean(pol_s)))


def run_pairdm(workers=16, log=print):
    todo = [c for c in PAIRDM_CONDS if not (OUT / f"{c}.pkl").exists()]
    log(f"pairdm: {len(todo)} conditions to run")
    if todo:
        run_conditions(todo, workers=workers, log=log)


def analyse_pairdm(out=None):
    conds = [c for c in PAIRDM_CONDS if (OUT / f"{c}.pkl").exists()]
    conds += [grid_name(d_, 2, "F", sd) for d_ in GRID_DISH for sd in (0, 101)]
    res = {c: pair_readout_pseudo(c) for c in conds if (OUT / f"{c}.pkl").exists()}
    Path(out or (OUT / "summary_pairdm.json")).write_text(json.dumps(res, indent=1, default=_json_default),
                                                          encoding="utf-8")
    for c, r in res.items():
        print(f"{c:28s} AI {r['ai']['mean']:.3f}  D {r['D']['mean']:+.3f}  NN {r['nn_median_mm']:.1f} mm  "
              f"shape {r['shape_0_2']:.2f}, {r['shape_2_4']:.2f}  align {r['pol']['mean']:.3f} "
              f"(pseudo {r['pol_pseudo']:.3f}, +{r['pol_above_pseudo']:.3f})")
    print(f"wrote summary_pairdm.json ({len(res)} conditions)")
    return res


def analyse_grid(out=None, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    conds = [c for c in GRID_CONDS if not GRID_CONDS[c]["blind"] and (OUT / f"{c}.pkl").exists()]
    with ProcessPoolExecutor(workers) as ex:
        res = dict(zip(conds, ex.map(grid_readout, conds)))
    Path(out or (OUT / "summary_grid.json")).write_text(json.dumps(res, indent=1, default=_json_default),
                                                        encoding="utf-8")
    print(f"wrote summary_grid.json ({len(res)} conditions)")
    return res






def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))




# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _same(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ("dtheta", "distance_px", "dt_s", "x_px", "y_px"))


def check_hook(tag="14f", T=600.0):
    """The provider hook ``block_time_s`` reproduces the generator's own clock bit for bit when it returns what the
    generator would compute (the recorded programme as a one-group scene)."""
    MF, SR = _imports()
    prog, dsh, ref = MF._programme()

    class ProgramGroups(SR.ProgramStimProvider):
        """The recorded programme as a one-group scene: [the dots' centroid] in a stimulus block, [] in a pause."""

        def groups(self, t, time_s, x, y, h):
            cx, cy, mv, nd = self(t, time_s, x, y, h)
            return [] if not np.isfinite(cx) else [(cx, cy, mv, nd)]

    class Mirror(ProgramGroups):
        """Same scene, plus the hook replaying the generator's rule: restart when (moving, n) changes."""
        _c, _t0, _last = None, 0.0, ()

        def block_time_s(self, t, t_now):
            g = self._last                                  # what groups() returned for this bout
            c = (g[0][3] * 10.0 + g[0][2]) if g else 0.0
            ts = float(self.clock_s(t, t_now)) if callable(getattr(self, "clock_s", None)) else float(t_now)
            if self._c is None or c != self._c:
                self._t0 = ts
            self._c = c
            return ts - self._t0

        def groups(self, t, time_s, x, y, h):
            self._last = super().groups(t, time_s, x, y, h)
            return self._last

    ok = []
    for k, m in enumerate(models(tag)[:3]):
        rng_seed = [11, k]
        pose = ref.sample_start_pose(np.random.default_rng(rng_seed), dsh)
        res = []
        for P in (ProgramGroups, Mirror):
            rec = SR.make_synthetic_record(int(T / 0.05) + 100, pose, dsh, fish_id=k)
            res.append(m.sample(rec, np.random.default_rng(rng_seed), stim_provider=P(prog), stop_time_s=T,
                                multi_rule="effects"))
        ok.append(_same(*res))
    return all(ok), ok


def check_parity(tag="14f", T=300.0, n_groups=3):
    """Coupled fish 0 of a pair == the generator run alone on a REPLAY of fish 1's track (same lookup rule), for both
    rules: the coupling layer adds nothing but the neighbour's position."""
    _, SR = _imports()
    ms = models(tag)
    ok = []
    for g in range(n_groups):
        for rule in ("effects", "nearest"):
            spec = _cond(tag, 2, 1, 1, T, rule)
            spec["T"] = T - TAIL_S
            res = run_group(spec, g, ms)
            B = res[1]
            tB, xB, yB = B["t"], B["x"].astype(float), B["y"].astype(float)

            class Replay:
                def groups(self, t, time_s, x, y, h):
                    k = min(int(np.searchsorted(tB, time_s, side="left")), len(xB) - 1)
                    out, _ = neighbour_groups(x, y, [(xB[k], yB[k])])
                    return out

                def __call__(self, *a):
                    raise RuntimeError

                def block_time_s(self, t, t_now):
                    return LATE_S

            rng = np.random.default_rng([FISH_SEED, g, 0])
            rec = SR.make_synthetic_record(int(T / 0.05) + 100, start_pose(g, 0), dish(1), fish_id=1000 * g)
            a = ms[g % len(ms)].sample(rec, rng, stim_provider=Replay(), stop_time_s=T, multi_rule=rule)
            A = res[0]
            ok.append(np.array_equal(A["t"], np.asarray(a["t_onset_s"], float))
                      and np.array_equal(A["x"], np.asarray(a["x_px"], float))
                      and np.array_equal(A["y"], np.asarray(a["y_px"], float))
                      and np.array_equal(A["dth"], np.asarray(a["dtheta"], np.float32)))
    return all(ok), ok


def check_determinism(tag="14f", T=300.0):
    spec = _cond(tag, 10, 1, 1, T - TAIL_S, "effects")
    a, b = run_group(spec, 0), run_group(spec, 0)
    return all(np.array_equal(x["t"], y["t"]) and np.array_equal(x["x"], y["x"]) for x, y in zip(a, b))


def check_blind_pairing(tag="14f", T=300.0):
    """A blind fish is the same fish whatever group it is in: fish 0-1 of a blind group of 10 == a blind pair."""
    a = run_group(_cond(tag, 10, 1, 1, T - TAIL_S, blind=True), 3)
    b = run_group(_cond(tag, 2, 1, 1, T - TAIL_S, blind=True), 3)
    return all(np.array_equal(x["t"], y["t"]) and np.array_equal(x["x"], y["x"]) for x, y in zip(a[:2], b))


class PhantomScene:
    """Rule check: ``k`` phantom moving neighbours re-placed every bout at 10-25 mm, bearing 60-120 deg on a random
    side, pooled and combined exactly as in the shoal runs; late clock."""

    def __init__(self, k, rng):
        self.k, self.rng, self.side = k, rng, []

    def __call__(self, *a):
        raise RuntimeError

    def groups(self, t, time_s, x, y, h):
        s = 1.0 if self.rng.random() < 0.5 else -1.0
        r = self.rng.uniform(10.0, 25.0, self.k) * PX_MM
        a = h + s * np.radians(self.rng.uniform(60.0, 120.0, self.k))
        out, _ = neighbour_groups(x, y, np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1))
        self.side.append(s)
        return out

    def block_time_s(self, t, t_now):
        return LATE_S


def check_rules(tag="14f", ks=(1, 2, 3, 4, 6, 9), T=600.0, per_fold=2):
    """P(turn toward the side the neighbours are on | |turn| > 3 deg), bout rate, median step and mean |turn|, per rule
    and number of phantom neighbours; with the same fish blind (k = 0) as the kinematic reference."""
    _, SR = _imports()
    ms = models(tag)
    rows = []
    for rule in ("effects", "nearest"):
        for k in (0,) + tuple(ks):
            tow, rate, step, turn = [], [], [], []
            for fi, m in enumerate(ms):
                for r in range(per_fold):
                    seed = [9, fi, r]
                    rng = np.random.default_rng(seed)
                    rec = SR.make_synthetic_record(int(T / 0.05) + 100, start_pose(900 + fi, r), dish(1), fish_id=fi)
                    if k == 0:
                        prov = FishView(0, None, [], clock="late")
                    else:
                        prov = PhantomScene(k, np.random.default_rng(seed + [1]))
                    o = m.sample(rec, rng, stim_provider=prov, stop_time_s=T, multi_rule=rule)
                    dth = np.asarray(o["dtheta"], float)
                    if k:
                        sd = np.asarray(prov.side)[:len(dth)]
                        sel = np.abs(dth) > 3.0
                        tow.append(np.mean(np.sign(dth[sel]) == sd[sel]))      # CCW-positive turn toward +side
                    rate.append(len(dth) / T); step.append(np.median(o["distance_px"]) / PX_MM)
                    turn.append(np.mean(np.abs(dth)))
            rows.append(dict(rule=rule, k=k, p_toward=float(np.mean(tow)) if tow else np.nan,
                             rate=float(np.mean(rate)), step=float(np.mean(step)), turn=float(np.mean(turn))))
    return rows


def real_toward_reference():
    """Real P(bout turns toward the dots' side | moving dots at 10-25 mm, bearing 60-120 deg, |turn| > 3 deg), per
    count, per-fish mean +- SEM -- the band the rule check is read against."""
    MF, _ = _imports()
    recs = MF.records()
    out = {}
    for n in (1, 2, 4, 6, 8):
        per = []
        for r in recs.values():
            dth = np.asarray(r.dtheta, float); sa = np.asarray(r.stim_angle_travel, float)
            sel = ((np.asarray(r.n_dots) == n) & (np.asarray(r.moving) > 0)
                   & (np.asarray(r.dot_dist_px, float) / PX_MM >= 10) & (np.asarray(r.dot_dist_px, float) / PX_MM < 25)
                   & (np.abs(np.degrees(sa)) >= 60) & (np.abs(np.degrees(sa)) <= 120) & (np.abs(dth) > 3))
            if sel.sum() >= 10:
                per.append(np.mean(np.sign(dth[sel]) == np.sign(sa[sel])))
        out[n] = (float(np.mean(per)), float(np.std(per, ddof=1) / np.sqrt(len(per))), len(per))
    return out


def run_checks(log=print):
    res = {}
    t0 = time.time()
    res["hook_bit_identical"] = check_hook()
    log(f"hook (provider clock == generator clock): {res['hook_bit_identical']}  [{time.time() - t0:.0f} s]")
    res["parity_coupled_vs_replay"] = check_parity()
    log(f"parity (coupled fish == generator on a replay of its partner): {res['parity_coupled_vs_replay']}  "
        f"[{time.time() - t0:.0f} s]")
    res["deterministic"] = check_determinism()
    log(f"determinism (group of 10 run twice): {res['deterministic']}  [{time.time() - t0:.0f} s]")
    res["blind_pairing"] = check_blind_pairing()
    log(f"blind fish identical in a group of 10 and in a pair: {res['blind_pairing']}  [{time.time() - t0:.0f} s]")
    res["real_toward"] = real_toward_reference()
    log(f"real P(toward), 10-25 mm, 60-120 deg: {res['real_toward']}")
    res["rules"] = check_rules()
    for r in res["rules"]:
        log(f"  rule check {r}")
    res["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    return res


# --------------------------------------------------------------------------- #
# Videos
# --------------------------------------------------------------------------- #
BOUT_S = 0.2                       # a bout is drawn as a 0.2 s glide (the model's bouts are instantaneous)
VIDEO_FPS = 30


def frames_xy(f, taus):
    """Positions (px) at video frame times: held between bouts, a linear 0.2 s glide through each bout."""
    t = f["t"]
    k = np.clip(np.searchsorted(t, taus, side="right") - 1, 0, len(t) - 2)    # last onset <= tau
    a = np.clip((taus - t[k]) / BOUT_S, 0.0, 1.0)
    x0, y0 = f["x"][k].astype(float), f["y"][k].astype(float)
    x1, y1 = f["x"][k + 1].astype(float), f["y"][k + 1].astype(float)
    return np.stack([x0 + a * (x1 - x0), y0 + a * (y1 - y0)], 1)




def render_npz(npz, out_mp4, trail_s=4.0, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    z = np.load(npz, allow_pickle=False)
    xy, taus, dishes, labels, speed = z["xy"], z["taus"], z["dishes"], z["labels"], float(z["speed"])
    P, N, F = xy.shape[0], xy.shape[1], xy.shape[2]
    cols = plt.get_cmap("tab10")(np.arange(N) % 10)
    fig, axes = plt.subplots(1, P, figsize=(5.2 * P, 5.6))
    axes = np.atleast_1d(axes)
    trail = max(1, int(trail_s * VIDEO_FPS / speed))
    arts = []
    for p, ax in enumerate(axes):
        cx, cy, R = dishes[p]
        x0, x1, y0, y1 = (float(v) for v in z["view"])
        ax.add_patch(plt.Circle((0, 0), R / PX_MM, fill=False, lw=1.2, color="0.35"))
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(str(labels[p]), fontsize=11)
        w = x1 - x0
        bar = 20.0 if w < 250 else 50.0
        ax.plot([x0 + 0.04 * w, x0 + 0.04 * w + bar], [y1 - 0.03 * w] * 2, color="k", lw=2)
        ax.text(x0 + 0.04 * w + bar / 2, y1 - 0.06 * w, f"{bar:.0f} mm", ha="center", fontsize=8)
        if w < 2 * R / PX_MM:                               # zoomed: say so
            ax.text(x1 - 0.02 * w, y0 + 0.03 * w, f"view {w:.0f} mm wide (dish {2 * R / PX_MM:.0f} mm)",
                    ha="right", va="top", fontsize=7, color="0.4")
        X = (xy[p, :, :, 0] - cx) / PX_MM
        Y = (xy[p, :, :, 1] - cy) / PX_MM                     # camera orientation (image y down)
        lines = [ax.plot([], [], lw=0.8, color=cols[i], alpha=0.6)[0] for i in range(N)]
        dots = ax.scatter(X[:, 0], Y[:, 0], s=22, c=cols, zorder=3, edgecolors="k", linewidths=0.3)
        bmm = float(z["body_mm"]) if "body_mm" in z.files else float("nan")
        if np.isfinite(bmm):                      # true-size fish: a dot bmm mm across in dish units (scales with the zoom)
            fig.canvas.draw()
            px_per_mm = ax.get_window_extent().width / (x1 - x0)
            dots.set_sizes([(bmm * px_per_mm * 72.0 / fig.dpi) ** 2] * len(X))
            dots.set_linewidths(0.0)
        arts.append((X, Y, lines, dots))
    ttl = fig.text(0.5, 0.02, "", ha="center", fontsize=10)

    def update(fr):
        lo = max(0, fr - trail)
        for X, Y, lines, dots in arts:
            for i, ln in enumerate(lines):
                ln.set_data(X[i, lo:fr + 1], Y[i, lo:fr + 1])
            dots.set_offsets(np.c_[X[:, fr], Y[:, fr]])
        m, s = divmod(float(taus[fr]), 60.0)
        ttl.set_text(f"t = {int(m):02d}:{s:04.1f}   (x{speed:g} speed)")
        return []

    anim = FuncAnimation(fig, update, frames=F, interval=1000.0 / VIDEO_FPS, blit=False)
    anim.save(str(out_mp4), writer="ffmpeg", fps=VIDEO_FPS, dpi=dpi)
    plt.close(fig)






# --------------------------------------------------------------------------- #
# Notebook helpers: tables and figures from summary.json / checks.json / the pickles
# --------------------------------------------------------------------------- #
VIDEO_DIR = OUT / "videos"






# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--render":
        render_npz(a[1], a[2])
    elif a and a[0] == "--checks":
        _worker_init(); run_checks()
    elif a and a[0] == "--run-grid":
        _worker_init(); run_grid(int(a[1]) if len(a) > 1 else 16)
    elif a and a[0] == "--analyse-grid":
        analyse_grid()
    elif a and a[0] == "--run-pairdm":
        _worker_init(); run_pairdm(int(a[1]) if len(a) > 1 else 16)
    elif a and a[0] == "--analyse-pairdm":
        analyse_pairdm()
    else:
        print(__doc__)
