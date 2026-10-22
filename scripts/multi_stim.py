"""Two dot groups at once: which group does a simulated fish approach? (report Model 31)

The recorded programme never shows two dot groups at the same time (0 % of 39,542 stimulus bouts have dots on both
sides), so a trained model's behaviour with two groups is not learned -- it is fixed by the rule that turns two groups
into the network's input. Four rules, all on the same trained weights, no retraining (``fit_MLP_joint.MULTI_RULES``):

  effects  each group alone through the whole network, plus a no-dots baseline; every head's output (before its link
           function) = baseline + the sum of the groups' effects. Every row the network sees is a valid single-group
           input. Hypothesis: independent pulls that add.
  add      one row with the groups' stimulus columns SUMMED (first-layer drives add; cos can exceed 1).
  avg      one row with the columns AVERAGED (close to the old one-centroid code).
  nearest  one row from the nearest group only (winner-take-all).

The profile term (pf) is evaluated on each group's own row and summed (effects / add), averaged (avg) or taken from the
nearest group; the aligned-and-far flag of a combined row is the max over groups; the block clock is shared.

SCENES (v2). Each simulated fish swims a lead-in pause (100 s), then 10 blocks of 60 s with two groups followed by a
50 s pause. At every block onset both groups are placed relative to the fish's pose (the rig's latch): group A at bearing
+theta (left) and distance d_A, group B at -theta, d_B; A swaps side every block. Each group replays a REAL recorded dot
group of its condition (count, moving or static), whole: rotated about the fish's onset point so its onset centroid sits
at the requested bearing, and SCALED about that point so it sits at the requested distance -- a similarity transform, so
the track's geometry relative to the fish (closest approach as a fraction of the onset distance, bearing sweep) is the
recorded one. Templates: every recorded block of the condition whose onset centroid lay 14-28 mm out (scale 0.5-1.9),
right-side recordings mirrored into the left half, so both sides draw from one pool. Two groups of the SAME condition are
exact mirror images (a symmetric scene is exactly symmetric); otherwise each group draws its own template. Draws are
random per (fish, block, group), seeded so a fish sees the same tracks under every rule, reference and scene.

REFERENCES. Every scene is also run with only A visible, only B visible and neither (the hidden groups' tracks are
computed but never shown): what a fish committed to one group, or ignoring both, produces on the SAME tracks. Every
two-group readout is read against them -- the side index has a geometric floor (two groups 30 deg apart are only 11 mm
apart) and the onset geometry alone favours some groups.

READOUTS (per block = trial): distances d_A(t), d_B(t); side index SI = (d_B - d_A) / (d_A + d_B); choice = sign of the
mean SI over 20-60 s, 'between' if |mean SI| < 0.2; late distance to the nearer and to the farther group (20-60 s);
decision time = first 2 s bin from which |SI| > 0.5 holds for 10 s, reported as the fraction of ALL trials decided by
time t (not a median over decided trials); the turn toward each group. Per scene and condition: P(A), P(B), P(between)
= the mean over simulated fish of each fish's fractions, 95 % CI by bootstrap over fish; differences against the
references paired by fish. Trials drop when a group's centroid starts outside the dish or is outside it for more than a
quarter of 20-60 s (fixed at onset by the placement, not by behaviour; counted).

    python multi_stim.py validate              # one group: every rule == the standard simulator, bit for bit
    python multi_stim.py run [--jobs 8]        # every (model, condition, scene) -> Model/multi_stim/*.pkl
    python multi_stim.py time                  # one short run per rule, for the timing
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import fit_MLP_joint as JT
import fit_MLP_border as MB
import model_figures as MF
import stim_program as SPm
import stim_rollout as SR

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("MULTI_STIM_OUT", ROOT / "Model" / "multi_stim"))
PX_PER_MM = MF.PX_TO_MM
BIN_S = 2.0
STIM_S, PAUSE_S, LEAD_S = 60.0, 50.0, 100.0
N_BLOCKS = 10                       # 5 with A on the left, 5 on the right
N_FISH = 50
LATE = (20.0, 60.0)                 # the choice window (s into the block)
LATE_IN_MIN = 0.75                  # a trial needs both groups inside the dish for >= 3/4 of the late window
SI_BETWEEN = 0.2                    # |mean SI| below this = 'between'
SI_DECIDED, DECIDE_S = 0.5, 10.0    # decision: |SI| above 0.5 for 10 s
R0_RANGE_MM = (14.0, 28.0)          # template onset distances kept (the placement scales them to the scene's)
TEMPLATE_SEED = 4242
RULES = JT.MULTI_RULES
REFS = ("ref-A", "ref-B", "ref-none")             # only A shown / only B shown / neither (both phantoms)
CONDS = RULES + REFS
VISIBLE = {"ref-A": (0,), "ref-B": (1,), "ref-none": ()}
COLOR = {"effects": "#1f77b4", "add": "#ff7f0e", "avg": "#2ca02c", "nearest": "#d62728",
         "ref-A": "#000000", "ref-B": "#555555", "ref-none": "#9e9e9e"}
RULE_COLOR = COLOR
REF_LABEL = {"ref-A": "only A shown", "ref-B": "only B shown", "ref-none": "neither shown",
             "chooser": "one group shown (mean of A only, B only)"}

# The two models (report Model 28b and 14f): (checkpoint dir under Model/, arm).
MODELS = {
    "28b": ("report12_pf_last/S4x_adt_z2_cdt_tr_rc_al_w3ft_pf5Bcr_last", "S4x_adt_z2_cdt_tr_rc_al_w3ft_pf5Bcr_last"),
    "14f": ("report8_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
            "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"),
    # Final_report_missingPieces (section 4): the predictions replicated on M6 itself (26e / 30d) and on F at seed 101
    "26e": ("simp18a_ladder_checkpoints", "S4x_adt_z2_cdt_tr_rc_al_w3ft"),
    "30d": ("simp18a_seed101_ckpts", "S4x_adt_z2_cdt_tr_rc_al_w3ft"),
    "32b": ("report13_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
            "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"),
}
MODEL_LABEL = {"28b": "28b: 26e + correction (no drift)", "14f": "14f: Model 17 + correction (best overall)",
               "26e": "26e: M6 (seed 0)", "30d": "30d: M6 (seed 101)", "32b": "32b: F (seed 101)"}


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scene:
    """``groups`` = ((n_dots, moving, bearing deg, onset distance px), ...) for the EVEN blocks (A first); the odd
    blocks mirror every bearing, so A alternates sides."""
    name: str
    groups: tuple
    sweep: str
    desc: str


def _g(n, mv, deg, d):
    return (int(n), bool(mv), float(deg), float(d))


D_STD, D_NEAR, D_FAR = 75.0, 50.0, 100.0          # px: real onset ~75 px (20 mm); 50 / 100 px = 14 / 27 mm


def _mm(px):
    return f"{px / PX_PER_MM:.0f}"


def scenes():
    S = []
    for sep in (30, 60, 90, 120, 180):            # angular separation 2 theta, 4 v 4 moving at 20 mm
        S.append(Scene(f"ang{sep}", (_g(4, 1, sep / 2, D_STD), _g(4, 1, -sep / 2, D_STD)), "angle",
                       f"4 v 4 moving, {sep} deg apart"))
    for a, b in ((2, 8), (2, 4), (4, 8)):         # count (A = the smaller group), moving, +-90 deg, 20 mm
        S.append(Scene(f"n{a}v{b}", (_g(a, 1, 90, D_STD), _g(b, 1, -90, D_STD)), "count", f"{a} v {b} moving"))
    S.append(Scene("near_far", (_g(4, 1, 90, D_NEAR), _g(4, 1, -90, D_FAR)), "distance",
                   f"4 v 4 moving, A near ({_mm(D_NEAR)} mm) v B far ({_mm(D_FAR)} mm)"))
    for da, db in itertools.product((D_NEAR, D_STD, D_FAR), repeat=2):   # 2 dots (A) at d_A v 8 dots (B) at d_B
        if (da, db) == (D_STD, D_STD):
            continue                              # = n2v8
        S.append(Scene(f"g2_{int(da)}_8_{int(db)}", (_g(2, 1, 90, da), _g(8, 1, -90, db)), "grid",
                       f"2 at {_mm(da)} mm v 8 at {_mm(db)} mm"))
    for n in (4, 8):                              # motion: A moving v B static, same count and distance
        S.append(Scene(f"mv{n}_st{n}", (_g(n, 1, 90, D_STD), _g(n, 0, -90, D_STD)), "motion",
                       f"{n} moving (A) v {n} static (B)"))
    return {s.name: s for s in S}


SCENES = scenes()


# --------------------------------------------------------------------------- #
# Real dot groups, placed
# --------------------------------------------------------------------------- #
class TemplateBank:
    """Every recorded stimulus block's dots, fish-relative (left, forward) px per frame and dot, pooled per
    (dot count, moving): right-side recordings mirrored into the left half; only blocks whose onset centroid lay
    R0_RANGE_MM out are kept."""

    def __init__(self, program, r0_range_mm=R0_RANGE_MM):
        self.pool, self.r0 = {}, {}
        for b in np.flatnonzero(program.is_stim_of):
            s, e = int(program.starts[b]), int(program.ends[b])
            m = program.active[s]
            if not (program.active[s:e] == m).all():
                raise ValueError(f"block {b}: the active dot set changes within the block")
            tr = np.stack([program.dx[s:e][:, m], program.dy[s:e][:, m]], axis=-1)       # (F, n, 2): left, forward
            if str(program.side_of[b]) == "right":
                tr = tr * np.array([-1.0, 1.0])                                            # mirror into the left half
            r0 = float(np.hypot(*tr[0].mean(axis=0))) / PX_PER_MM
            if not (r0_range_mm[0] <= r0 <= r0_range_mm[1]):
                continue
            key = (int(program.n_dots_of[b]), bool(program.moving_of[b]))
            self.pool.setdefault(key, []).append(tr)
            self.r0.setdefault(key, []).append(r0)

    def size(self, n, moving):
        return len(self.pool[(int(n), bool(moving))])

    def place(self, n, moving, deg, d_px, idx):
        """Template ``idx`` (cycled) of the condition, rotated about the fish's onset point so its onset centroid sits at
        bearing ``|deg|``, scaled about that point to distance ``d_px``, then mirrored if ``deg`` < 0."""
        lst = self.pool[(int(n), bool(moving))]
        tr = lst[int(idx) % len(lst)]
        L, F = tr[..., 0], tr[..., 1]
        c0l, c0f = float(L[0].mean()), float(F[0].mean())
        a = np.radians(abs(deg)) - np.arctan2(c0l, c0f)
        s = float(d_px) / float(np.hypot(c0l, c0f))
        Fr = s * (F * np.cos(a) - L * np.sin(a))
        Lr = s * (F * np.sin(a) + L * np.cos(a))
        return np.stack([-Lr if deg < 0 else Lr, Fr], axis=-1)


_BANK = {}


def template_bank():
    if "b" not in _BANK:
        program, _, _ = MF._programme()
        _BANK["b"] = TemplateBank(program)
    return _BANK["b"]


def template_index(fish_k, blk, gi):
    """Template draw of group ``gi`` in block ``blk`` of fish ``fish_k``: the same under every rule, reference and
    scene (a large integer, cycled over the pool in ``place``)."""
    return int(np.random.default_rng([TEMPLATE_SEED, int(fish_k), int(blk), int(gi)]).integers(1 << 30))


class SceneProvider:
    """Stimulus provider for one simulated fish on a scene. ``groups()`` (the multi-group generator) returns the
    VISIBLE groups; ``__call__`` (the standard generator) the pooled centroid of the visible dots. On entering a block
    the fish's pose is latched, as the rig does, and every group of the block is placed relative to it. Every call
    records the pose and ALL groups' centroids (visible or not) under the bout index, so the readouts can measure the
    distance to a group the model never saw (the references)."""

    def __init__(self, scene, bank, fish_k, n_blocks=N_BLOCKS, visible=None, dish=None, fps=SPm.FPS):
        self.scene, self.fps = scene, float(fps)
        self.n_blocks = int(n_blocks)
        self.lead_f = int(round(LEAD_S * fps)); self.stim_f = int(round(STIM_S * fps))
        self.cyc_f = self.stim_f + int(round(PAUSE_S * fps))
        self.n_frames = self.lead_f + self.n_blocks * self.cyc_f
        self.T = self.n_frames / self.fps
        G = len(scene.groups)
        self.visible = tuple(range(G)) if visible is None else tuple(int(v) for v in visible)
        self.dish = dish
        self.tracks, self.info = [], []
        for j in range(self.n_blocks):
            sg = 1.0 if j % 2 == 0 else -1.0                         # A on the left in even blocks
            idx = [template_index(fish_k, j, gi) for gi in range(G)]
            for gi in range(1, G):                                   # same condition as A -> A's exact mirror image
                if scene.groups[gi][:2] == scene.groups[0][:2]:
                    idx[gi] = idx[0]
            self.tracks.append([bank.place(n, mv, sg * deg, d, idx[gi])
                                for gi, (n, mv, deg, d) in enumerate(scene.groups)])
            self.info.append([(float(mv), float(n)) for n, mv, _, _ in scene.groups])
        self.cur_block = -1
        self.anchor = (np.nan, np.nan, np.nan)
        self.anchors = {}
        self.rec = {}                     # bout index -> (x, y, h, [(cx, cy) per group]) in stimulus blocks
        self.blk_ok = {}                  # stim block j -> (every group inside the dish at the latch, late inside frac)

    def locate(self, f):
        """frame -> (block id, stim index j or -1, frame within the stim block). Block ids: 0 lead-in, 1 + 2j stim j,
        2 + 2j the pause after it; past the programme -> (-1, -1, -1)."""
        if f < 0 or f >= self.n_frames:
            return -1, -1, -1
        if f < self.lead_f:
            return 0, -1, -1
        j, r = divmod(f - self.lead_f, self.cyc_f)
        return (1 + 2 * j, j, r) if r < self.stim_f else (2 + 2 * j, -1, -1)

    def _centroids(self, j, k):
        ax, ay, ao = self.anchor
        out = []
        for tr in self.tracks[j]:
            ox, oy = SPm.anchor_offsets(tr[k, :, 0], tr[k, :, 1], ao)
            out.append((float((ax + ox).mean()), float((ay + oy).mean())))
        return out

    def _flag_block(self, j, k0):
        """Whether every group starts inside the dish (at the latch frame) and its late-window inside fraction:
        functions of the placement alone (the track is fixed once the anchor is latched)."""
        if self.dish is None:
            self.blk_ok[j] = (True, 1.0)
            return
        cx, cy, R = self.dish
        ax, ay, ao = self.anchor
        lo, hi = int(LATE[0] * self.fps), min(int(LATE[1] * self.fps), self.stim_f)
        ins0, frac = True, 1.0
        for tr in self.tracks[j]:
            c = tr.mean(axis=1)                                      # centroid track (linear: = mean of the dots)
            ox, oy = SPm.anchor_offsets(c[:, 0], c[:, 1], ao)
            inside = np.hypot(ax + ox - cx, ay + oy - cy) <= R
            ins0 = ins0 and bool(inside[min(k0, len(inside) - 1)])
            frac = min(frac, float(inside[lo:hi].mean()))
        self.blk_ok[j] = (ins0, frac)

    def groups(self, t, time_s, x, y, h):
        b, j, k = self.locate(int(round(float(time_s) * self.fps)))
        if b < 0:
            return []
        if b != self.cur_block:                                       # NEW BLOCK -> latch this fish's pose
            self.cur_block = b
            self.anchor = (float(x), float(y), float(h))
            self.anchors[b] = self.anchor
            if j >= 0:
                self._flag_block(j, k)
        if j < 0:
            return []
        cs = self._centroids(j, k)
        self.rec[int(t)] = (float(x), float(y), float(h), cs)
        return [(cs[g][0], cs[g][1]) + tuple(self.info[j][g]) for g in self.visible]

    def __call__(self, t, time_s, x, y, h):
        """The standard simulator's view: ONE centroid. One visible group -> exactly that group; several -> every
        visible dot pooled (the old one-centroid code), moving if any group moves, the total count."""
        gs = self.groups(t, time_s, x, y, h)
        if not gs:
            return np.nan, np.nan, 0.0, 0.0
        if len(gs) == 1:
            return gs[0]
        n = np.array([g[3] for g in gs])
        return (float(np.dot(n, [g[0] for g in gs]) / n.sum()), float(np.dot(n, [g[1] for g in gs]) / n.sum()),
                float(max(g[2] for g in gs)), float(n.sum()))


class ProgramGroupsProvider(SR.ProgramStimProvider):
    """The RECORDED programme as a one-group scene (validation): ``groups()`` = [the dots' centroid] in a stimulus
    block, [] in a pause -- the multi-group generator then has to reproduce the standard one bit for bit."""

    def groups(self, t, time_s, x, y, h):
        cx, cy, mv, nd = self(t, time_s, x, y, h)
        return [] if not np.isfinite(cx) else [(cx, cy, mv, nd)]


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _models(tag):
    d, arm = MODELS[tag]
    with contextlib.redirect_stdout(io.StringIO()):
        models, _ = MF.fold_models(d, arm)                           # the rej band merged in, as every rollout
    return [models[f] for f in sorted(models)]


def _rule_visible(cond):
    if cond in RULES:
        return cond, None
    if cond in VISIBLE:
        return "effects", VISIBLE[cond]                               # 0 or 1 visible group: every rule is identical
    raise ValueError(f"condition {cond!r} not in {CONDS}")


def simulate(tag, scene, cond, n_fish=N_FISH, n_blocks=N_BLOCKS, seed=0, models=None):
    """Stimulus-block bouts of ``n_fish`` simulated fish (n_fish / 5 per fold model) on ``scene`` under ``cond`` (a
    rule, or a reference: 'ref-A' / 'ref-B' / 'ref-none'). Fish k is seeded ``(seed, k)`` with per-block random
    streams and draws its templates from ``(TEMPLATE_SEED, k, block, group)``, so its start pose, latent, drift,
    every block's randomness and every track are the same under every condition: everything is paired by fish."""
    rule, vis = _rule_visible(cond)
    models = models or _models(tag)
    _, dish, ref = MF._programme()
    bank = template_bank()
    per = int(n_fish) // len(models)
    G = len(scene.groups)
    rows, n_run = [], 0
    for mi, m in enumerate(models):
        for r in range(per):
            k = mi * per + r
            rng = np.random.default_rng([int(seed), k])
            pose = ref.sample_start_pose(rng, dish)
            prov = SceneProvider(scene, bank, k, n_blocks, visible=vis, dish=dish)
            n_max = int(prov.T / 0.2) + 100
            rec = SR.make_synthetic_record(n_max, pose, dish, fish_id=k)
            g = m.sample(rec, rng, stim_provider=prov, stop_time_s=prov.T, block_seed=(int(seed), k), multi_rule=rule)
            nb = len(g["dtheta"])
            if nb >= n_max:                                           # a runaway fish: counted, not scored
                n_run += 1
                continue
            fo = np.round(np.asarray(g["t_onset_s"]) * prov.fps).astype(int)
            loc = np.array([prov.locate(f) for f in fo]).reshape(-1, 3)
            keep = np.flatnonzero(loc[:, 1] >= 0)
            if not all(int(t) in prov.rec for t in keep):
                raise RuntimeError("a stimulus-block bout has no recorded scene: provider / generator out of step")
            P = np.array([prov.rec[int(t)][:3] for t in keep], float).reshape(-1, 3)          # x, y, h at onset
            C = np.array([prov.rec[int(t)][3] for t in keep], float).reshape(-1, G, 2)
            j = loc[keep, 1]
            d = dict(fish=k, fold=mi, blk=j, t_blk=loc[keep, 2] / prov.fps, dth=np.asarray(g["dtheta"])[keep],
                     x=P[:, 0], y=P[:, 1])
            if not (np.allclose(P[:, 0], np.asarray(g["x_px"])[keep]) and np.allclose(P[:, 1], np.asarray(g["y_px"])[keep])):
                raise RuntimeError("recorded pose differs from the generator's")
            for gi, lab in zip(range(G), "AB"):
                d[f"d{lab}"] = np.hypot(P[:, 0] - C[:, gi, 0], P[:, 1] - C[:, gi, 1]) / PX_PER_MM
                d[f"phi{lab}"] = MB._wrap_pi(np.arctan2(C[:, gi, 1] - P[:, 1], C[:, gi, 0] - P[:, 0]) - P[:, 2])
            ok = np.array([prov.blk_ok.get(int(jj), (False, 0.0)) for jj in j], dtype=object).reshape(-1, 2)
            d["onset_in"] = ok[:, 0].astype(bool); d["late_in"] = ok[:, 1].astype(float)
            rows.append(pd.DataFrame(d))
    b = pd.concat(rows, ignore_index=True)
    for c in b.columns:
        if b[c].dtype == np.float64:
            b[c] = b[c].astype(np.float32)
    b.attrs.update(runaway=n_run, tag=tag, scene=scene.name, cond=cond, n_fish=n_fish, seed=seed)
    return b


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #
def add_si(b):
    """+ the 2 s bin, the turn toward each group (deg, + = toward), the side index and the nearer / farther distance."""
    b = b.copy()
    b["bin"] = np.minimum((b.t_blk // BIN_S).astype(int), int(STIM_S / BIN_S) - 1)
    b["towA"] = b.dth * np.sign(b.phiA)
    if "dB" in b:
        b["SI"] = (b.dB - b.dA) / (b.dA + b.dB)
        b["towB"] = b.dth * np.sign(b.phiB)
        b["d_near"] = np.minimum(b.dA, b.dB)
        b["d_far"] = np.maximum(b.dA, b.dB)
    return b


def _decision_s(si_bins):
    """First 2 s bin from which |SI| > 0.5 with one sign for DECIDE_S (empty bins carry the last value)."""
    s = pd.Series(si_bins).ffill().to_numpy()
    w = int(DECIDE_S / BIN_S)
    for i in range(len(s) - w + 1):
        seg = s[i:i + w]
        if np.all(np.isfinite(seg)) and (np.all(seg > SI_DECIDED) or np.all(seg < -SI_DECIDED)):
            return i * BIN_S
    return np.nan


def trials(b):
    """One row per trial: kept (both groups start inside the dish and stay inside for >= 3/4 of 20-60 s), mean SI and
    distances over 20-60 s, the choice and the decision time."""
    b = add_si(b) if "SI" not in b else b
    nb = int(STIM_S / BIN_S)
    out = []
    for (f, j), g in b.groupby(["fish", "blk"], sort=True):
        g = g.sort_values("t_blk")
        ok = bool(g.onset_in.iloc[0]) and float(g.late_in.iloc[0]) >= LATE_IN_MIN
        late = g[(g.t_blk >= LATE[0]) & (g.t_blk < LATE[1])]
        enough = len(late) >= 3
        si = float(late.SI.mean()) if enough else np.nan
        ch = (None if not np.isfinite(si) else "A" if si > SI_BETWEEN else "B" if si < -SI_BETWEEN else "between")
        sib = g.groupby("bin").SI.mean().reindex(range(nb)).to_numpy()
        out.append(dict(fish=f, blk=j, ok=ok, onset_in=bool(g.onset_in.iloc[0]), late_in=float(g.late_in.iloc[0]),
                        n_late=len(late), si_late=si, choice=ch,
                        dA_late=float(late.dA.mean()) if enough else np.nan,
                        dB_late=float(late.dB.mean()) if enough else np.nan,
                        d_near_late=float(late.d_near.mean()) if enough else np.nan,
                        d_far_late=float(late.d_far.mean()) if enough else np.nan,
                        t_decide=_decision_s(sib)))
    return pd.DataFrame(out)


def _boot(per_fish, n=2000, seed=0):
    """(mean, lo, hi): 95 % bootstrap CI over fish."""
    v = np.asarray(per_fish, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fish_metrics(tr):
    """Per simulated fish, over its kept trials: P(A), P(B), P(between), late distances, decided fractions."""
    v = tr[tr.ok & tr.choice.notna()]
    g = v.groupby("fish")
    out = pd.DataFrame({c: g.choice.apply(lambda s, c=c: float((s == c).mean())) for c in ("A", "B", "between")})
    out.columns = ["P_A", "P_B", "P_between"]
    out["pref"] = out.P_A - out.P_B
    for c in ("d_near_late", "d_far_late", "dA_late", "dB_late", "si_late"):
        out[c] = g[c].mean()
    out["decided"] = g.t_decide.apply(lambda s: float(s.notna().mean()))
    out["decided_30s"] = g.t_decide.apply(lambda s: float((s <= 30.0).mean()))
    return out


def choice_summary(tr):
    """Fish-level means with 95 % bootstrap CIs over fish; trial counts and the dropped share."""
    fm = fish_metrics(tr)
    s = dict(n_trials=int((tr.ok & tr.choice.notna()).sum()), n_fish=int(len(fm)),
             dropped_pct=round(100.0 * (1.0 - tr.ok.mean()), 1))
    for c in ("P_A", "P_B", "P_between", "pref", "d_near_late", "d_far_late", "dA_late", "dB_late", "si_late",
              "decided", "decided_30s"):
        m, lo, hi = _boot(fm[c])
        s[c], s[c + "_lo"], s[c + "_hi"] = round(m, 3), round(lo, 3), round(hi, 3)
    return s


def _paired(a, b, n=2000, seed=0):
    """Mean of (a - b) over the fish present in both, with a 95 % bootstrap CI over fish."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return _boot(j.a - j.b, n=n, seed=seed)


# --------------------------------------------------------------------------- #
# Results on disk
# --------------------------------------------------------------------------- #
def _path(tag, cond, scene_name):
    return OUT / f"{tag}_{cond}_{scene_name}.pkl"


def run_one(tag, cond, scene_name, seed=0):
    p = _path(tag, cond, scene_name)
    if p.exists():
        return str(p), 0.0
    t0 = time.time()
    b = add_si(simulate(tag, SCENES[scene_name], cond, seed=seed))
    pd.to_pickle(dict(bouts=b, trials=trials(b), runaway=b.attrs.get("runaway", 0)), p)
    return str(p), time.time() - t0


def jobs(tags=("28b", "14f")):
    """Every (model, condition, scene): the four rules and the three references."""
    return [(tag, cond, sc) for tag in tags for sc in SCENES for cond in CONDS]


def run_all(tags=("28b", "14f"), n_jobs=8):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs(tags) if not _path(*j).exists()]
    print(f"{len(todo)} simulations to run ({len(jobs(tags)) - len(todo)} already on disk), {n_jobs} at once",
          flush=True)
    t0 = time.time()
    bad = []
    with ProcessPoolExecutor(n_jobs) as ex:
        fut = {ex.submit(run_one, *j): j for j in todo}
        for i, f in enumerate(as_completed(fut), 1):
            j = fut[f]
            try:
                _, secs = f.result()
                print(f"[{(time.time() - t0) / 60:.0f} min] {i}/{len(todo)} {j}: {secs:.0f} s", flush=True)
            except Exception as e:                                   # noqa: BLE001 -- report every failure, keep going
                bad.append(j)
                print(f"[{(time.time() - t0) / 60:.0f} min] {i}/{len(todo)} {j}: FAILED {e!r}", flush=True)
    if bad:
        raise SystemExit(f"{len(bad)} simulations failed: {bad}")


_LOADED = {}


def load(tag, cond, scene_name):
    key = (tag, cond, scene_name)
    if key not in _LOADED:
        p = _path(*key)
        _LOADED[key] = pd.read_pickle(p) if p.exists() else None
    return _LOADED[key]


def summary_table(tags=("28b", "14f")):
    """One row per (model, scene, condition): fish-level means and CIs, plus, for the rules, the differences paired
    by fish against the references:
      dbtw_chooser  P(between) minus that of a fish shown ONE group (mean of only-A and only-B): > 0 = more often
                    between the groups than a fish committed to one of them;
      dbtw_ignorer  P(between) minus that of a fish shown NEITHER: < 0 = less often between than the geometry alone;
      pref_corr     P(A) - P(B) minus the same for a fish shown neither -- the preference beyond what the placement
                    alone produces;
      dnear_*       late distance to the nearer group minus the references' (< 0 = closer than)."""
    rows = []
    for tag in tags:
        for sc, s in SCENES.items():
            fm = {}
            for cond in CONDS:
                r = load(tag, cond, sc)
                if r is not None:
                    fm[cond] = (fish_metrics(r["trials"]), choice_summary(r["trials"]), r["runaway"])
            for cond, (m, cs, nr) in fm.items():
                row = dict(model=tag, scene=sc, sweep=s.sweep, desc=s.desc, cond=cond, runaway=nr, **cs)
                if cond in RULES and all(x in fm for x in REFS):
                    chooser = (fm["ref-A"][0].P_between + fm["ref-B"][0].P_between) / 2
                    for key, a, bb in (("dbtw_chooser", m.P_between, chooser),
                                       ("dbtw_ignorer", m.P_between, fm["ref-none"][0].P_between),
                                       ("pref_corr", m.pref, fm["ref-none"][0].pref),
                                       ("dnear_chooser", m.d_near_late,
                                        (fm["ref-A"][0].dA_late + fm["ref-B"][0].dB_late) / 2),
                                       ("dnear_ignorer", m.d_near_late, fm["ref-none"][0].d_near_late)):
                        mu, lo, hi = _paired(a, bb)
                        row[key], row[key + "_lo"], row[key + "_hi"] = round(mu, 3), round(lo, 3), round(hi, 3)
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures (fish-level means, 95 % CI across simulated fish)
# --------------------------------------------------------------------------- #
def _fish_curve(b, col):
    """(bin centres s, mean over fish, lo, hi): per fish the mean over its trials in each 2 s bin."""
    f = b.groupby(["fish", "bin"])[col].mean().unstack()
    m = f.mean(0)
    se = f.std(0, ddof=1) / np.sqrt(f.notna().sum(0))
    t = (f.columns.to_numpy() + 0.5) * BIN_S
    return t, m.to_numpy(), (m - 1.96 * se).to_numpy(), (m + 1.96 * se).to_numpy()


def _kept_bouts(r):
    """Bouts of the kept trials, with each trial's choice."""
    tr = r["trials"]
    ch = tr[tr.ok & tr.choice.notna()].set_index(["fish", "blk"]).choice
    return r["bouts"].set_index(["fish", "blk"]).join(ch, how="inner").reset_index()


def _ang_rows(S, tag, cond):
    d = S[(S.model == tag) & (S.cond == cond) & (S.sweep == "angle")].copy()
    d["sep"] = d.scene.str[3:].astype(int)
    return d.sort_values("sep")


def plot_angle(S, tags=("28b", "14f")):
    """P(between) by angular separation per rule, against a fish shown one group and a fish shown neither (same
    tracks): the two references fall with separation too, so the readout is the gap to them, not the slope."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(tags), figsize=(5.4 * len(tags), 3.8), sharey=True, squeeze=False)
    for ax, tag in zip(axs[0], tags):
        a, bb, nn = (_ang_rows(S, tag, c) for c in REFS)
        if len(a) and len(bb):
            ch = (a.P_between.to_numpy() + bb.P_between.to_numpy()) / 2
            ax.plot(a.sep, ch, color="black", lw=1.2, ls="--", label=REF_LABEL["chooser"])
        if len(nn):
            ax.plot(nn.sep, nn.P_between, color=COLOR["ref-none"], lw=1.2, ls=":", label=REF_LABEL["ref-none"])
        for rule in RULES:
            d = _ang_rows(S, tag, rule)
            if d.empty:
                continue
            ax.errorbar(d.sep, d.P_between, yerr=[d.P_between - d.P_between_lo, d.P_between_hi - d.P_between],
                        color=COLOR[rule], marker="o", ms=4, lw=1.5, capsize=2, label=rule)
        ax.set_xticks([30, 60, 90, 120, 180]); ax.set_xlabel("angular separation of the two groups (deg)")
        ax.set_title(MODEL_LABEL[tag], fontsize=9); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("P(between): |mean SI 20-60 s| < 0.2")
    axs[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle("4 v 4 moving groups at 20 mm: how often the fish ends between them", fontsize=10)
    fig.tight_layout()
    return fig


def plot_late_distances(S, tags=("28b", "14f")):
    """Late (20-60 s) distance to the nearer and to the farther group by angular separation: choosing = near small and
    far large; averaging = both moderate; ignoring = both large."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(tags), 2, figsize=(10.5, 3.3 * len(tags)), sharex=True, squeeze=False)
    for ti, tag in enumerate(tags):
        for ci, (col, lab) in enumerate((("d_near_late", "nearer group"), ("d_far_late", "farther group"))):
            ax = axs[ti, ci]
            for cond in CONDS:
                d = _ang_rows(S, tag, cond)
                if d.empty:
                    continue
                ref = cond in REFS
                ax.errorbar(d.sep, d[col], yerr=[d[col] - d[col + "_lo"], d[col + "_hi"] - d[col]], color=COLOR[cond],
                            marker="s" if ref else "o", ms=3.5, lw=1.0 if ref else 1.5, ls="--" if ref else "-",
                            capsize=2, label=REF_LABEL.get(cond, cond))
            ax.set_title(f"{tag}: late distance to the {lab}", fontsize=9); ax.grid(alpha=0.3)
            ax.set_xticks([30, 60, 90, 120, 180])
            if ti == len(tags) - 1:
                ax.set_xlabel("angular separation (deg)")
            if ci == 0:
                ax.set_ylabel("mm (20-60 s)")
    axs[0, -1].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_preference(S, sweeps=("count", "distance", "motion"), tags=("28b", "14f")):
    """P(A) - P(B) beyond the placement: each rule's preference minus that of a fish shown neither group (paired by
    fish, 95 % CI), A = the first-named group (the smaller, the nearer, the moving one). The dots = the one-group
    references' preference (only A shown, only B shown) minus the same null: the most a fully committed fish shows."""
    import matplotlib.pyplot as plt
    sc = [s for s in SCENES.values() if s.sweep in sweeps]
    fig, axs = plt.subplots(len(tags), 1, figsize=(11, 3.2 * len(tags)), squeeze=False)
    for ax, tag in zip(axs[:, 0], tags):
        x, xt, xl = 0.0, [], []
        for s in sc:
            xs = []
            for rule in RULES:
                d = S[(S.model == tag) & (S.cond == rule) & (S.scene == s.name)]
                if d.empty or "pref_corr" not in d or not np.isfinite(d.iloc[0].get("pref_corr", np.nan)):
                    continue
                r = d.iloc[0]
                ax.bar(x, r.pref_corr, color=COLOR[rule], width=0.8)
                ax.errorbar(x, r.pref_corr, yerr=[[r.pref_corr - r.pref_corr_lo], [r.pref_corr_hi - r.pref_corr]],
                            color="black", lw=0.8, capsize=2)
                xs.append(x); x += 1
            nn = S[(S.model == tag) & (S.cond == "ref-none") & (S.scene == s.name)]
            for cond, mk in (("ref-A", "^"), ("ref-B", "v")):
                d = S[(S.model == tag) & (S.cond == cond) & (S.scene == s.name)]
                if len(d) and len(nn) and xs:
                    ax.plot(np.mean(xs), d.iloc[0].pref - nn.iloc[0].pref, marker=mk, color="black", ms=6, ls="none")
            if xs:
                xt.append(float(np.mean(xs))); xl.append(s.desc.replace(", ", "\n").replace(" (A)", "").replace(" (B)", ""))
                x += 1.2
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7.5)
        ax.set_ylabel("P(A) - P(B), minus\nthe neither-shown null")
        ax.set_title(f"{MODEL_LABEL[tag]} -- bars: effects, add, avg, nearest; triangles: only A / only B shown",
                     fontsize=9)
    fig.tight_layout()
    return fig


def plot_grid(S, tags=("28b", "14f")):
    """P(8 dots) - P(2 dots) beyond the placement (minus the neither-shown null) over both onset distances, per rule."""
    import matplotlib.pyplot as plt
    ds = (D_NEAR, D_STD, D_FAR)
    fig, axs = plt.subplots(len(tags), len(RULES), figsize=(3.0 * len(RULES), 2.9 * len(tags)), squeeze=False)
    im = None
    for ti, tag in enumerate(tags):
        for ri, rule in enumerate(RULES):
            M = np.full((3, 3), np.nan)
            for i, da in enumerate(ds):
                for j, db in enumerate(ds):
                    name = "n2v8" if (da, db) == (D_STD, D_STD) else f"g2_{int(da)}_8_{int(db)}"
                    d = S[(S.model == tag) & (S.cond == rule) & (S.scene == name)]
                    if len(d) and "pref_corr" in d:
                        M[i, j] = -d.iloc[0].pref_corr                # B = 8 dots
            ax = axs[ti, ri]
            im = ax.imshow(M, cmap="RdBu_r", vmin=-0.5, vmax=0.5, origin="lower")
            for i in range(3):
                for j in range(3):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8)
            lab = [_mm(d) for d in ds]
            ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_yticks(range(3)); ax.set_yticklabels(lab)
            ax.set_xlabel("8 dots at (mm)", fontsize=8); ax.set_ylabel("2 dots at (mm)", fontsize=8)
            ax.set_title(f"{tag} {rule}", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=0.6, label="P(8) - P(2), minus the neither-shown null")
    return fig


def plot_si_hist(tags=("28b", "14f"), scene_names=("ang60", "ang180", "n2v8", "mv4_st4")):
    """Per trial, the mean SI over 20-60 s, with the references on the same tracks (black: only A shown, grey: neither)."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(tags), len(scene_names), figsize=(3.6 * len(scene_names), 2.8 * len(tags)),
                            sharex=True, squeeze=False)
    bins = np.linspace(-1, 1, 25)
    for ti, tag in enumerate(tags):
        for si, sc in enumerate(scene_names):
            ax = axs[ti, si]
            for cond in RULES + ("ref-A", "ref-none"):
                r = load(tag, cond, sc)
                if r is None:
                    continue
                tr = r["trials"]
                v = tr[tr.ok].si_late.dropna()
                ref = cond in REFS
                ax.hist(v, bins=bins, histtype="step", lw=1.2 if ref else 1.6, ls="--" if ref else "-",
                        color=COLOR[cond], density=True, label=REF_LABEL.get(cond, cond))
            ax.axvspan(-SI_BETWEEN, SI_BETWEEN, color="#eeeeee", zorder=0)
            ax.set_title(f"{tag}: {SCENES[sc].desc}", fontsize=8.5)
            if ti == len(tags) - 1:
                ax.set_xlabel("mean SI 20-60 s  (+1 at A, -1 at B)")
    axs[0, 0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    return fig


def plot_si_heat(tag, scene_name="ang180"):
    """SI over the block, one row per kept trial, sorted by the late mean SI; the references on the same tracks last."""
    import matplotlib.pyplot as plt
    conds = RULES + ("ref-A", "ref-none")
    fig, axs = plt.subplots(1, len(conds), figsize=(2.7 * len(conds), 3.4))
    nb = int(STIM_S / BIN_S)
    for ax, cond in zip(axs, conds):
        r = load(tag, cond, scene_name)
        if r is None:
            ax.set_visible(False)
            continue
        b, tr = r["bouts"], r["trials"]
        ok = tr[tr.ok].set_index(["fish", "blk"])
        m = b.groupby(["fish", "blk", "bin"]).SI.mean().unstack().reindex(columns=range(nb))
        m = m.loc[m.index.isin(ok.index)]
        m = m.iloc[np.argsort(ok.loc[m.index].si_late.fillna(0).to_numpy())]
        ax.imshow(m.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6, extent=(0, STIM_S, len(m), 0),
                  interpolation="nearest")
        ax.set_title(REF_LABEL.get(cond, cond), fontsize=8.5); ax.set_xlabel("time in block (s)")
    axs[0].set_ylabel("trial (sorted by late SI)")
    fig.suptitle(f"{MODEL_LABEL[tag]}: SI over the block, {SCENES[scene_name].desc} (red = at A, blue = at B; "
                 "colour scale +-0.6)", fontsize=9.5)
    fig.tight_layout()
    return fig


def plot_dist_by_choice(tag, scene_name="ang180"):
    """Distance to the chosen and to the other group over the block, per rule, NEXT TO the same split of a fish shown
    only A (classified by the same SI rule, so both curves are selected the same way), and the 'between' trials."""
    import matplotlib.pyplot as plt
    conds = RULES + ("ref-A",)
    fig, axs = plt.subplots(1, len(conds), figsize=(3.1 * len(conds), 3.2), sharey=True)
    for ax, cond in zip(axs, conds):
        r = load(tag, cond, scene_name)
        if r is None:
            ax.set_visible(False)
            continue
        bb = _kept_bouts(r)
        bc = bb[bb.choice.isin(["A", "B"])].copy()
        bc["d_chosen"] = np.where(bc.choice == "A", bc.dA, bc.dB)
        bc["d_other"] = np.where(bc.choice == "A", bc.dB, bc.dA)
        c0 = COLOR[cond]
        if len(bc):
            for col, c, lab in (("d_chosen", c0, "chosen"), ("d_other", "#7f7f7f", "other")):
                t, m, lo, hi = _fish_curve(bc, col)
                ax.plot(t, m, color=c, lw=1.6, label=lab); ax.fill_between(t, lo, hi, color=c, alpha=0.2, lw=0)
        bt = bb[bb.choice == "between"]
        if len(bt):
            t, m, lo, hi = _fish_curve(bt.assign(d_mean=(bt.dA + bt.dB) / 2), "d_mean")
            ax.plot(t, m, color="#2b2b2b", lw=1.0, ls=":", label="between (mean of both)")
        tr = r["trials"]
        k = tr[tr.ok & tr.choice.notna()]
        ax.set_title(f"{REF_LABEL.get(cond, cond)}\nA {100 * (k.choice == 'A').mean():.0f} % / B "
                     f"{100 * (k.choice == 'B').mean():.0f} % / between {100 * (k.choice == 'between').mean():.0f} %",
                     fontsize=8.5)
        ax.set_xlabel("time in block (s)"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("distance to group (mm)")
    axs[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{MODEL_LABEL[tag]}: {SCENES[scene_name].desc}", fontsize=9.5)
    fig.tight_layout()
    return fig


def plot_decision(tags=("28b", "14f"), scene_names=("ang180", "n2v8", "mv4_st4")):
    """Fraction of ALL kept trials decided (|SI| > 0.5 for 10 s) by time t into the block, per rule and reference."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(tags), len(scene_names), figsize=(3.8 * len(scene_names), 2.9 * len(tags)),
                            sharey=True, squeeze=False)
    tt = np.arange(0, STIM_S - DECIDE_S + BIN_S, BIN_S)
    for ti, tag in enumerate(tags):
        for si, sc in enumerate(scene_names):
            ax = axs[ti, si]
            for cond in CONDS:
                r = load(tag, cond, sc)
                if r is None:
                    continue
                tr = r["trials"]
                k = tr[tr.ok & tr.choice.notna()]
                cdf = [(k.t_decide <= x).mean() for x in tt]
                ref = cond in REFS
                ax.plot(tt, cdf, color=COLOR[cond], lw=1.0 if ref else 1.6, ls="--" if ref else "-",
                        label=REF_LABEL.get(cond, cond))
            ax.set_title(f"{tag}: {SCENES[sc].desc}", fontsize=8.5); ax.grid(alpha=0.3)
            if ti == len(tags) - 1:
                ax.set_xlabel("decision time (s into the block)")
    axs[0, 0].set_ylabel("fraction of trials decided"); axs[-1, 0].set_ylabel("fraction of trials decided")
    axs[0, -1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    return fig


def plot_toward(tag, scene_name="n2v8"):
    """Turn toward each group over the block (deg per bout, + = toward), per rule and for a fish shown only A."""
    import matplotlib.pyplot as plt
    conds = RULES + ("ref-A",)
    fig, axs = plt.subplots(1, len(conds), figsize=(3.1 * len(conds), 3.0), sharey=True)
    s = SCENES[scene_name]
    for ax, cond in zip(axs, conds):
        r = load(tag, cond, scene_name)
        if r is None:
            ax.set_visible(False)
            continue
        b = _kept_bouts(r)
        for col, c, lab in (("towA", COLOR[cond] if cond != "ref-A" else "#1f77b4", f"toward A ({s.groups[0][0]} dots)"),
                            ("towB", "#7f7f7f", f"toward B ({s.groups[1][0]} dots)")):
            t, m, lo, hi = _fish_curve(b, col)
            ax.plot(t, m, color=c, lw=1.5, label=lab); ax.fill_between(t, lo, hi, color=c, alpha=0.2, lw=0)
        ax.axhline(0, color="black", lw=0.6); ax.set_title(REF_LABEL.get(cond, cond), fontsize=9)
        ax.set_xlabel("time in block (s)")
    axs[0].set_ylabel("turn toward (deg per bout)")
    axs[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{MODEL_LABEL[tag]}: {s.desc}", fontsize=9.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _roll(model, prov, pose, dish, rng_seed, T, rule):
    rng = np.random.default_rng(rng_seed)
    rec = SR.make_synthetic_record(int(T / 0.2) + 100, pose, dish, fish_id=0)
    kw = {} if rule is None else {"multi_rule": rule}
    return model.sample(rec, rng, stim_provider=prov, stop_time_s=T, block_seed=tuple(rng_seed), **kw)


def _same(g0, g1):
    keys = ("dtheta", "distance_px", "dt_s", "x_px", "y_px")
    return all(len(g0[k]) == len(g1[k]) and np.array_equal(g0[k], g1[k]) for k in keys)


def validate(tags=None, n_fish=3, T_prog=1800.0):
    """(1) the RECORDED programme (first 30 min) through every rule vs the standard simulator; (2) a scene with only
    group A shown, through every rule vs the standard simulator on the same scene -- both bit-identical; (3) two
    groups: finite and different; (4) symmetric scenes are exactly symmetric: at every block onset the fish is equally
    far from both groups; (5) a phantom group changes nothing: 'only A shown' equals a one-group scene of A."""
    program, dish, ref = MF._programme()
    bank = template_bank()
    res = {}
    for tag in (tags or list(MODELS)):
        ms = _models(tag)
        r = {}
        for k in range(n_fish):
            m = ms[k % len(ms)]
            pose = ref.sample_start_pose(np.random.default_rng([99, k]), dish)
            g0 = _roll(m, SR.ProgramStimProvider(program), pose, dish, [99, k], T_prog, None)
            for rule in RULES:
                g1 = _roll(m, ProgramGroupsProvider(program), pose, dish, [99, k], T_prog, rule)
                r.setdefault(f"programme/{rule}", []).append(_same(g0, g1))
            sc = SCENES["ang180"]
            T = SceneProvider(sc, bank, k).T
            g0 = _roll(m, SceneProvider(sc, bank, k, visible=(0,)), pose, dish, [99, k], T, None)
            for rule in RULES:
                g1 = _roll(m, SceneProvider(sc, bank, k, visible=(0,)), pose, dish, [99, k], T, rule)
                r.setdefault(f"A only/{rule}", []).append(_same(g0, g1))
            solo = Scene("soloA", sc.groups[:1], "solo", "A alone")
            g1 = _roll(m, SceneProvider(solo, bank, k), pose, dish, [99, k], T, "effects")
            r.setdefault("phantom B changes nothing", []).append(_same(g0, g1))
            for rule in RULES:
                p2 = SceneProvider(sc, bank, k, dish=dish)
                g2 = _roll(m, p2, pose, dish, [99, k], T, rule)
                fin = bool(np.all(np.isfinite(g2["dtheta"])) and np.all(np.isfinite(g2["x_px"])))
                r.setdefault(f"two groups run/{rule}", []).append(fin and not _same(g0, g2))
                if rule == "effects":                                # symmetric placement: equal distances at onset
                    gaps = []
                    for b, (ax_, ay_, ao_) in p2.anchors.items():
                        if b < 1 or b % 2 == 0:
                            continue
                        j = (b - 1) // 2
                        p2.anchor = (ax_, ay_, ao_)
                        (xa, ya), (xb, yb) = p2._centroids(j, 0)
                        gaps.append(abs(np.hypot(xa - ax_, ya - ay_) - np.hypot(xb - ax_, yb - ay_)))
                    r.setdefault("symmetric scene: |dA - dB| at onset < 1e-9 px", []).append(
                        bool(len(gaps)) and float(max(gaps)) < 1e-9)
            print(f"  {tag} fish {k}: {len(g0['dtheta'])} bouts on the scene", flush=True)
        res[tag] = {kk: bool(all(v)) for kk, v in r.items()}
        print(tag, json.dumps(res[tag], indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(res, indent=1))
    ok = all(all(v.values()) for v in res.values())
    print("VALIDATION", "PASSED" if ok else "FAILED", flush=True)
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    nj = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    tg = sys.argv[sys.argv.index("--models") + 1].split(",") if "--models" in sys.argv else None
    if cmd == "validate":
        sys.exit(0 if validate(tg) else 1)
    elif cmd == "run":
        run_all(tuple(tg or MODELS), n_jobs=nj)
    elif cmd == "time":
        for cond in CONDS:
            t0 = time.time()
            b = simulate(tg[0] if tg else "28b", SCENES["ang180"], cond, n_fish=5)
            print(cond, f"{time.time() - t0:.1f} s for 5 fish, {len(b)} stimulus bouts", flush=True)
