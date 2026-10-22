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

from twin import fit_MLP_joint as JT
from twin import fit_MLP_border as MB
from twin import model_figures as MF
from twin import stim_program as SPm
from twin import stim_rollout as SR

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




def jobs(tags=("28b", "14f")):
    """Every (model, condition, scene): the four rules and the three references."""
    return [(tag, cond, sc) for tag in tags for sc in SCENES for cond in CONDS]




_LOADED = {}




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
