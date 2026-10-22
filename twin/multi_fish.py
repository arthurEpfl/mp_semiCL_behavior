"""Several simulated fish in one dish, each seeing the others as moving dots: do they shoal?

The data never show a fish another fish, so this is an extrapolation of the dot response. Every fish runs the trained
generator unchanged (``fit_MLP_joint.generate_mix_joint_loop``, multi-group mode); only its stimulus provider is new:
at each bout onset it reads where the OTHER fish are and turns them into dot groups.

COUPLING. One thread per fish around the unchanged generator. The generator asks its provider for the stimulus exactly
once per bout, at the bout's onset, with the pre-bout pose. The provider blocks until it is that fish's turn: the fish
with the smallest (pending onset, index) goes next, so every fish sees each other fish where it is after that fish's
bouts with onset < t (or = t with a lower index). Bouts are instantaneous; collisions are ignored (dots overlap freely
in training too). Deterministic: the result does not depend on thread timing.

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

    python multi_fish.py --checks            # correctness checks (hook, parity, determinism, rule scenes)
    python multi_fish.py --run [COND ...]    # simulate the conditions (all by default), Model/multi_fish/<cond>.pkl
    python multi_fish.py --analyse           # readouts -> Model/multi_fish/summary.json
    python multi_fish.py --run-shoal         # Final_report_shoaling conditions (5 fish, 5x dish, F / F+turn / F+turn+step)
    python multi_fish.py --analyse-shoal     # their readouts + close contacts -> Model/multi_fish/summary_shoal.json
    python multi_fish.py --videos            # mp4s of shoal formation, if the readouts show shoaling
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = Path(os.environ.get("MULTI_FISH_OUT", ROOT / "Model" / "multi_fish"))
PX_MM = 3.60                       # px per mm of the physical dish (the thesis' scale; distances are reported in mm)
RADIUS_PX = 280.0                  # perception horizon: the log-range input saturates here (70 px x 4 = 78 mm)
ENV_PX = {1: 0.0, 2: 37.0, 4: 59.2, 6: 67.4, 8: 74.3}   # trained spread envelope (moving groups, measured 2026-10-05)
N_CAP = 8                          # largest trained group size
LATE_S = 60.0                      # largest block time in training (60 s blocks)
TAIL_S = 10.0                      # every fish runs to T + 10 s (dt <= 10 s): every pose up to T is published
POSE_SEED, FISH_SEED, SCR_SEED = 7101, 7102, 7103
SHOAL_MM = 25.0                    # 'together' = within 25 mm, single linkage (real fish hold 16-19 mm from moving dots)
SHIFTS_S = tuple(60.0 * k for k in range(1, 11))   # Larsch / Kappel: 10 time shifts
CHUNK_S = 300.0                                     # Larsch / Kappel: 5-min chunks

MODELS = {   # tag: (checkpoint dir under Model/, arm). F = 14f (seed 0) / 32b (seed 101)
    "14f": ("report8_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
            "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"),
    "32b": ("report13_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
            "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"),
    "26e": ("simp18a_ladder_checkpoints", "S4x_adt_z2_cdt_tr_rc_al_w3ft"),        # M6, seed 0
    "30d": ("simp18a_seed101_ckpts", "S4x_adt_z2_cdt_tr_rc_al_w3ft"),             # M6, seed 101
    # F + the repulsion term fitted to the Larsch screen data (report Model 43a / 43b): every neighbour within reach adds
    # its own turn-away (fit_MLP_joint.repulsion_shifts via FishView.dots); attraction still comes through the groups
    "14f_rep": ("rep43_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repT",
                "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repT"),
    "32b_rep": ("rep43_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repT",
                "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repT"),
    # F + turn + the likelihood-fitted step term (report Model 50a / 50b): shorter steps near any neighbour
    "14f_reps": ("rep50_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTS",
                 "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTS"),
    "32b_reps": ("rep50_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTS",
                 "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTS"),
}
# D4 TRAINED ON THE SCREEN (scripts/screen_train.py), without and with the dot's direction of motion as an input
# (`stim_dmove`, Model 51). They saw one dot, always moving, never a dish without one: run ONLY in PAIRS (pairdm_*,
# single-partner mode), read against PSEUDO-PAIRS, never blind (blind they swim abnormally).
for _sd in (0, 101):
    MODELS[f"sD4_s{_sd}"] = (f"screen_trained_s{_sd}_ckpts", "S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st")
    MODELS[f"sD4dm_s{_sd}"] = (f"screen_trained_s{_sd}_ckpts", "S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st_dm")
# A blind fish sees no neighbour, so the repulsion term never acts and F + repulsion blind IS F blind (same weights,
# same random streams): the repulsion conditions share F's blind pools (checked: check_rep_blind).
BLIND_TAG = {"14f_rep": "14f", "32b_rep": "32b", "14f_reps": "14f", "32b_reps": "32b"}
# F + the GROUP repulsion term (scripts/grouprep_fit.py; one representation: each group's term from its nearest member,
# groups added by the effects rule): calibrated turn, + step, strength x2 / x4 / x8, x group count, + collision rule
GR_VARIANTS = ("grT", "grTS", "grT2", "grT4", "grT8", "grTb1", "grTC")
_F_ARM = "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"
for _st, _dir in (("14f", "gr_s0"), ("32b", "gr_s101")):
    for _v in GR_VARIANTS:
        MODELS[f"{_st}_{_v}"] = (f"{_dir}/{_F_ARM}_{_v}", f"{_F_ARM}_{_v}")
        BLIND_TAG[f"{_st}_{_v}"] = _st


def _cond(tag, N, G, arena, T, rule=None, blind=False, scramble=False, t0=None, clock="late", start="free"):
    return dict(tag=tag, N=N, G=G, arena=arena, T=float(T), rule=rule or "effects", blind=blind, scramble=scramble,
                mapping="cluster", clock=clock, t_read=float(t0 if t0 is not None else (300.0 if arena == 1 else 600.0)),
                start=start)


# Fish (group g, index i) has the same start pose, random stream, latent and fold model (g % 5) in every condition of a
# seed, so conditions pair by group: a pair g is fish 0-1 of group g, a group of 5 its fish 0-4.
CONDS = {
    "A1_blind": _cond("14f", 10, 50, 1, 1800, blind=True),
    "A1_pairs": _cond("14f", 2, 50, 1, 1800),
    "A1_N5_effects": _cond("14f", 5, 20, 1, 1800, "effects"),
    "A1_N5_nearest": _cond("14f", 5, 20, 1, 1800, "nearest"),
    "A1_N10_effects": _cond("14f", 10, 20, 1, 1800, "effects"),
    "A1_N10_nearest": _cond("14f", 10, 20, 1, 1800, "nearest"),
    "A1_N10_effects_scrambled": _cond("14f", 10, 20, 1, 1800, "effects", scramble=True),
    "A5_blind": _cond("14f", 10, 20, 5, 3600, blind=True),
    "A5_N10_effects": _cond("14f", 10, 20, 5, 3600, "effects"),
    "A5_N10_nearest": _cond("14f", 10, 20, 5, 3600, "nearest"),
    "s101_A1_blind": _cond("32b", 10, 50, 1, 1800, blind=True),
    "s101_A1_pairs": _cond("32b", 2, 50, 1, 1800),
    "s101_A1_N10_effects": _cond("32b", 10, 20, 1, 1800, "effects"),
    # Final_report_missingPieces (sections 4 and 9): replication on M6 itself and on F seed 101; the 5x dish's controls
    "m6_A1_blind": _cond("26e", 10, 50, 1, 1800, blind=True),
    "m6_A1_pairs": _cond("26e", 2, 50, 1, 1800),
    "m6_A1_N10_effects": _cond("26e", 10, 20, 1, 1800, "effects"),
    "m6s101_A1_blind": _cond("30d", 10, 50, 1, 1800, blind=True),
    "m6s101_A1_pairs": _cond("30d", 2, 50, 1, 1800),
    "m6s101_A1_N10_effects": _cond("30d", 10, 20, 1, 1800, "effects"),
    "s101_A1_N5_effects": _cond("32b", 5, 20, 1, 1800, "effects"),
    "s101_A1_N10_effects_scrambled": _cond("32b", 10, 20, 1, 1800, "effects", scramble=True),
    "s101_A5_blind": _cond("32b", 10, 20, 5, 3600, blind=True),
    "s101_A5_N10_effects": _cond("32b", 10, 20, 5, 3600, "effects"),
    "A5_N10_effects_scrambled": _cond("14f", 10, 20, 5, 3600, "effects", scramble=True),
    "A5_N10_effects_entry": _cond("14f", 10, 20, 5, 3600, "effects", clock="entry"),
    # F + repulsion (Final_report_extrapolations section 3): the same scenes as F, both seeds
    "rep_A1_pairs": _cond("14f_rep", 2, 50, 1, 1800),
    "rep_A1_N10_effects": _cond("14f_rep", 10, 20, 1, 1800, "effects"),
    "rep_A5_N10_effects": _cond("14f_rep", 10, 20, 5, 3600, "effects"),
    "rep_s101_A1_pairs": _cond("32b_rep", 2, 50, 1, 1800),
    "rep_s101_A1_N10_effects": _cond("32b_rep", 10, 20, 1, 1800, "effects"),
    "rep_s101_A5_N10_effects": _cond("32b_rep", 10, 20, 5, 3600, "effects"),
    # F + turn + step, the same scenes
    "reps_A1_pairs": _cond("14f_reps", 2, 50, 1, 1800),
    "reps_A1_N10_effects": _cond("14f_reps", 10, 20, 1, 1800, "effects"),
    "reps_A5_N10_effects": _cond("14f_reps", 10, 20, 5, 3600, "effects"),
    "reps_s101_A1_pairs": _cond("32b_reps", 2, 50, 1, 1800),
    "reps_s101_A1_N10_effects": _cond("32b_reps", 10, 20, 1, 1800, "effects"),
    "reps_s101_A5_N10_effects": _cond("32b_reps", 10, 20, 5, 3600, "effects"),
}
REP_CONDS = ("rep_A1_pairs", "rep_A1_N10_effects", "rep_A5_N10_effects", "rep_s101_A1_pairs",
             "rep_s101_A1_N10_effects", "rep_s101_A5_N10_effects")
REP_BASE = {"rep_A1_pairs": "A1_pairs", "rep_A1_N10_effects": "A1_N10_effects", "rep_A5_N10_effects": "A5_N10_effects",
            "rep_s101_A1_pairs": "s101_A1_pairs", "rep_s101_A1_N10_effects": "s101_A1_N10_effects",
            "rep_s101_A5_N10_effects": "s101_A5_N10_effects"}
REPS_CONDS = tuple("reps_" + c[len("rep_"):] for c in REP_CONDS)
REPS_BASE = {"reps_" + c[len("rep_"):]: b for c, b in REP_BASE.items()}
# Final_report_shoaling.ipynb (2026-10-18): 5 fish, 40 groups (200 fish), 5x dish, 60 min, readout from 10 min; F, F + turn,
# F + turn + step at both seeds, and a blind pool of the same size per seed (fish (g, i) as in every other condition)
SH_CONDS = {
    "sh_A5_N5_blind": _cond("14f", 5, 40, 5, 3600, blind=True),
    "sh_A5_N5_F": _cond("14f", 5, 40, 5, 3600, "effects"),
    "sh_A5_N5_Fturn": _cond("14f_rep", 5, 40, 5, 3600, "effects"),
    "sh_A5_N5_Fturnstep": _cond("14f_reps", 5, 40, 5, 3600, "effects"),
    "sh_s101_A5_N5_blind": _cond("32b", 5, 40, 5, 3600, blind=True),
    "sh_s101_A5_N5_F": _cond("32b", 5, 40, 5, 3600, "effects"),
    "sh_s101_A5_N5_Fturn": _cond("32b_rep", 5, 40, 5, 3600, "effects"),
    "sh_s101_A5_N5_Fturnstep": _cond("32b_reps", 5, 40, 5, 3600, "effects"),
}
CONDS.update(SH_CONDS)
# Final_report_shoaling.ipynb, group repulsion (2026-10-18): 5 fish x 40 groups, F and every GR variant at both seeds, in
# the 5x dish (60 min, readout from 10 min; F = sh_*_A5_N5_F above) and the training dish (30 min, readout from 5 min;
# blind pools: A1_blind / s101_A1_blind, whose fish 0-4 of groups 0-39 are the same fish)
GR_CONDS = {}
for _st, _pre in (("14f", ""), ("32b", "s101_")):
    GR_CONDS[f"g1_{_pre}F"] = _cond(_st, 5, 40, 1, 1800, "effects")
    for _v in GR_VARIANTS:
        GR_CONDS[f"g5_{_pre}{_v}"] = _cond(f"{_st}_{_v}", 5, 40, 5, 3600, "effects")
        GR_CONDS[f"g1_{_pre}{_v}"] = _cond(f"{_st}_{_v}", 5, 40, 1, 1800, "effects")
CONDS.update(GR_CONDS)
GR_F5 = {"14f": "sh_A5_N5_F", "32b": "sh_s101_A5_N5_F"}       # F in the 5x dish (already run)
# THE SHOAL GRID (2026-10-19, thesis Sec. 5.5): every dish x group size x variant x seed, read the same way
# (``analyse_grid``). Training dish: 30 min, readout from 5 min, free starts (as before). 5x dish: 60 min, readout from
# 10 min, NEAR starts (each group starts within 25 mm of its fish 0, so the fish can see each other; the question there
# is whether a shoal stays together without walls), with blind pools started the same way. Groups of 5 in the training
# dish are the g1_* conditions above (same spec), copied under the grid names.
GRID_VARS = ("F", "grT", "grT2", "grT4", "grT8", "grTC")
GRID_N = {2: 50, 5: 40, 10: 40}
GRID_DISH = {"A1": dict(arena=1, T=1800, start="free"), "A5": dict(arena=5, T=3600, start="near")}
NEAR_MM = 25.0
GRID_CONDS, GRID_COPY = {}, {}


def grid_name(dish_, N, var, seed):
    return f"grid_{dish_}_N{N}_{var}" + ("_s101" if seed == 101 else "")


for _seed, _st, _pre in ((0, "14f", ""), (101, "32b", "s101_")):
    for _d, _dd in GRID_DISH.items():
        for _N, _G in GRID_N.items():
            for _v in GRID_VARS:
                _tag = _st if _v == "F" else f"{_st}_{_v}"
                _nm = grid_name(_d, _N, _v, _seed)
                if _d == "A1" and _N == 5:
                    GRID_COPY[_nm] = f"g1_{_pre}{_v}"
                GRID_CONDS[_nm] = _cond(_tag, _N, _G, _dd["arena"], _dd["T"], "effects", start=_dd["start"])
    GRID_CONDS[f"grid_A5_blind" + ("_s101" if _seed == 101 else "")] = _cond(_st, 10, 50, 5, 3600, blind=True,
                                                                              start="near")
CONDS.update(GRID_CONDS)
# PAIRS WITH THE DOT'S DIRECTION OF MOTION (2026-10-21, thesis "Adding alignment"): screen-trained D4 with and without
# the input, both seeds, the grid's two dishes and starts, 50 pairs; the partner is one moving dot, always in view.
PAIRDM_CONDS = {}
for _tag in ("sD4_s0", "sD4_s101", "sD4dm_s0", "sD4dm_s101"):
    for _d, _dd in GRID_DISH.items():
        PAIRDM_CONDS[f"pairdm_{_d}_{_tag}"] = dict(_cond(_tag, 2, 50, _dd["arena"], _dd["T"], start=_dd["start"]),
                                                   single=True)
CONDS.update(PAIRDM_CONDS)
NEW_CONDS = ("m6_A1_blind", "m6_A1_pairs", "m6_A1_N10_effects", "m6s101_A1_blind", "m6s101_A1_pairs",
             "m6s101_A1_N10_effects", "s101_A1_N5_effects", "s101_A1_N10_effects_scrambled", "s101_A5_blind",
             "s101_A5_N10_effects", "A5_N10_effects_scrambled", "A5_N10_effects_entry")
LABEL = {
    "sh_A5_N5_blind": "blind fish, 5 per group (5x dish)", "sh_A5_N5_F": "F, 5 fish (5x dish)",
    "sh_A5_N5_Fturn": "F + turn, 5 fish (5x dish)", "sh_A5_N5_Fturnstep": "F + turn + step, 5 fish (5x dish)",
    "sh_s101_A5_N5_blind": "blind fish, 5 per group (5x dish, seed 101)", "sh_s101_A5_N5_F": "F, 5 fish (5x dish, seed 101)",
    "sh_s101_A5_N5_Fturn": "F + turn, 5 fish (5x dish, seed 101)",
    "sh_s101_A5_N5_Fturnstep": "F + turn + step, 5 fish (5x dish, seed 101)",
    "A1_blind": "blind fish (training dish)", "A1_pairs": "pairs", "A1_N5_effects": "5 fish, effects",
    "A1_N5_nearest": "5 fish, nearest", "A1_N10_effects": "10 fish, effects", "A1_N10_nearest": "10 fish, nearest",
    "A1_N10_effects_scrambled": "10 fish, effects, random directions", "A5_blind": "blind fish (5x dish)",
    "A5_N10_effects": "10 fish, effects (5x dish)", "A5_N10_nearest": "10 fish, nearest (5x dish)",
    "s101_A1_blind": "blind fish (seed 101)", "s101_A1_pairs": "pairs (seed 101)",
    "s101_A1_N10_effects": "10 fish, effects (seed 101)",
    "m6_A1_blind": "M6: blind fish", "m6_A1_pairs": "M6: pairs", "m6_A1_N10_effects": "M6: 10 fish, effects",
    "m6s101_A1_blind": "M6 seed 101: blind fish", "m6s101_A1_pairs": "M6 seed 101: pairs",
    "m6s101_A1_N10_effects": "M6 seed 101: 10 fish, effects", "s101_A1_N5_effects": "5 fish, effects (seed 101)",
    "s101_A1_N10_effects_scrambled": "10 fish, effects, random directions (seed 101)",
    "s101_A5_blind": "blind fish (5x dish, seed 101)", "s101_A5_N10_effects": "10 fish, effects (5x dish, seed 101)",
    "A5_N10_effects_scrambled": "10 fish, effects, random directions (5x dish)",
    "A5_N10_effects_entry": "10 fish, effects, onset clock on entry (5x dish)",
    "rep_A1_pairs": "F + repulsion: pairs", "rep_A1_N10_effects": "F + repulsion: 10 fish, effects",
    "rep_A5_N10_effects": "F + repulsion: 10 fish, effects (5x dish)",
    "rep_s101_A1_pairs": "F + repulsion: pairs (seed 101)",
    "rep_s101_A1_N10_effects": "F + repulsion: 10 fish, effects (seed 101)",
    "rep_s101_A5_N10_effects": "F + repulsion: 10 fish, effects (5x dish, seed 101)",
    "reps_A1_pairs": "F + turn + step: pairs", "reps_A1_N10_effects": "F + turn + step: 10 fish, effects",
    "reps_A5_N10_effects": "F + turn + step: 10 fish, effects (5x dish)",
    "reps_s101_A1_pairs": "F + turn + step: pairs (seed 101)",
    "reps_s101_A1_N10_effects": "F + turn + step: 10 fish, effects (seed 101)",
    "reps_s101_A5_N10_effects": "F + turn + step: 10 fish, effects (5x dish, seed 101)",
}
_GR_LAB = {"F": "F", "grT": "F + group repulsion", "grTS": "F + group repulsion + step", "grT2": "F + group repulsion x2",
           "grT4": "F + group repulsion x4", "grT8": "F + group repulsion x8", "grTb1": "F + group repulsion x n",
           "grTC": "F + group repulsion + collision"}
for _c, _sp in GR_CONDS.items():
    _v = _c.split("_")[-1]
    LABEL[_c] = (f"{_GR_LAB[_v]}, 5 fish ({'5x' if _sp['arena'] == 5 else 'training'} dish"
                 + (", seed 101)" if "s101" in _c else ")"))


def blind_of(cond):
    """The blind pool a condition pairs with: the blind condition of the same model and dish."""
    c = CONDS[cond]
    tag = BLIND_TAG.get(c["tag"], c["tag"])
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
        sys.path[:0] = [str(HERE), str(ROOT / "Model")]
        import model_figures as MF
        import stim_rollout as SR
        _CACHE.update(MF=MF, SR=SR)
    return _CACHE["MF"], _CACHE["SR"]


def models(tag):
    key = ("models", tag)
    if key not in _CACHE:
        MF, _ = _imports()
        d, arm = MODELS[tag]
        with contextlib.redirect_stdout(io.StringIO()):
            ms, _ = MF.fold_models(d, arm)                     # the rej band merged in, as every rollout
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


def _iid(P):
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    n = len(P)
    return D.sum((0, 1)) / (n * (n - 1))


def _largest(P, thr=SHOAL_MM):
    """(frames,) size of the largest single-linkage cluster at ``thr`` mm."""
    n, F = P.shape[0], P.shape[1]
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1) <= thr            # (n, n, F)
    out = np.zeros(F, int)
    for f in range(F):
        A = D[:, :, f]
        seen, best = np.zeros(n, bool), 0
        for s in range(n):
            if seen[s]:
                continue
            stack, size = [s], 0
            seen[s] = True
            while stack:
                u = stack.pop(); size += 1
                for v in np.flatnonzero(A[u] & ~seen):
                    seen[v] = True; stack.append(v)
            best = max(best, size)
        out[f] = best
    return out


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


def _pseudo_fold(n_pseudo, folds):
    labs = np.unique(folds)
    return labs[np.arange(n_pseudo) % len(labs)]


def _d_fold(nn_g, nn_s, folds, pf):
    """Per-group D against the pseudo-groups of its own fold (the 'fold' null)."""
    den = {f: nn_s[pf == f].mean() for f in np.unique(pf)}
    return np.log(nn_g / np.array([den[f] for f in folds]))


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


def pairs_readout(cond):
    """Per-pair attraction index, median pair distance and its time-shifted null, for a pair condition and for the same
    pairs blind (fish 0-1 of the blind pool)."""
    out = {}
    for lab, c, n in (("coupled", cond, 2), ("blind", blind_of(cond), 2)):
        blob = load(c)
        T, taus = tracks(blob, n)
        G = CONDS[cond]["G"]
        rows = [pair_ai(T[g, 0], T[g, 1], taus, blob["spec"]["t_read"], blob["spec"]["T"]) for g in range(G)]
        out[lab] = dict(ai=[r[0] for r in rows], iad_med=[r[1] for r in rows], iad_shift_med=[r[2] for r in rows],
                        dist=np.linalg.norm(T[:G, 0] - T[:G, 1], axis=-1))
    return out


def groups_readout(cond, n_pseudo=200):
    """Per-group nearest-neighbour distance, its Harpaz shuffled null and the blind groups of the same size."""
    spec = CONDS[cond]
    N, G = spec["N"], spec["G"]
    blob = load(cond)
    T, taus = tracks(blob)
    bl = load(blind_of(cond))
    TB, _ = tracks(bl, N)
    TB = TB[:G]
    nn = np.array([_nn1(T[g]) for g in range(G)])                      # (G, frames)
    nn_b = np.array([_nn1(TB[g]) for g in range(G)])
    S = _shuffled(T, N, n_pseudo)                                        # primary null (Harpaz)
    nn_s = np.array([_nn1(p) for p in S])
    SB = _shuffled(TB, N, n_pseudo, seed=1)
    nn_bs = np.array([_nn1(p) for p in SB])
    folds = np.arange(G) % 5                                             # sensitivity: fold-matched null
    pf = _pseudo_fold(n_pseudo, folds)
    nn_sf = np.array([_nn1(p) for p in _shuffled(T, N, n_pseudo, seed=2, mode="fold")])
    # within-group time-shift null: fish k shifted by k x 60 s (circular on the readout grid)
    nn_t = np.array([_nn1(np.stack([np.roll(T[g, k], -60 * k, axis=0) for k in range(N)])) for g in range(G)])
    iid = np.array([_iid(T[g]).mean() for g in range(G)])
    iid_s = np.array([_iid(p).mean() for p in S])
    big = np.array([_largest(T[g]) for g in range(G)])
    big_s = np.array([_largest(p) for p in S[:50]])
    big_b = np.array([_largest(TB[g]) for g in range(G)])
    # alignment: |mean unit travel heading| per frame, against the same kind of pseudo-groups
    H = np.array([[heading_at(f, taus) for f in grp] for grp in blob["groups"]])
    pol = np.abs(np.exp(1j * H).mean(1)).mean(1)
    rng = np.random.default_rng(3)
    Hs = np.array([[H[g, rng.integers(N)] for g in rng.choice(G, N, replace=False)] for _ in range(n_pseudo)])
    pol_s = np.abs(np.exp(1j * Hs).mean(1)).mean(1)
    return dict(taus=taus, nn=nn, nn_blind=nn_b, nn_shuf=nn_s, nn_blind_shuf=nn_bs, nn_tshift=nn_t,
                D=np.log(nn.mean(1) / nn_s.mean()), D_ci=_boot_D(nn.mean(1), T, N),
                D_fold=_d_fold(nn.mean(1), nn_sf.mean(1), folds, pf), D_tshift=np.log(nn.mean(1) / nn_t.mean(1)),
                D_blind=np.log(nn_b.mean(1) / nn_bs.mean()), occupancy=float(np.log(nn_s.mean() / nn_bs.mean())),
                iid_ratio=iid / iid_s.mean(), big=big, big_shuf=big_s, big_blind=big_b, pol=pol, pol_shuf=pol_s, G=G)


def close_contacts(cond, step=1.0):
    """Nearest-neighbour distance of every fish at every second (1 Hz grid after t_read), pooled over groups: the share of
    fish-seconds with the nearest neighbour within 2 / 4 / 6 / 10 / 25 mm, its median and 10th percentile, and its
    histogram (0-60 mm, 1 mm bins). For pairs this is the pair distance."""
    blob = load(cond)
    T, _ = tracks(blob, step=step)                                   # (groups, fish, frames, 2) mm
    nn = []
    for P in T:
        D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
        D[np.arange(len(P)), np.arange(len(P))] = np.inf
        nn.append(D.min(1).ravel())
    nn = np.concatenate(nn)
    out = {f"share_nn_le_{k}mm": float(np.mean(nn <= k)) for k in (2, 4, 6, 10, 25)}
    out.update(nn_median=float(np.median(nn)), nn_p10=float(np.percentile(nn, 10)),
               hist=np.histogram(nn, np.arange(0, 61, 1.0))[0].tolist())
    return out


def check_rep_blind(T=300.0):
    """F + repulsion blind == F blind, byte for byte (the term needs a visible neighbour)."""
    spec = dict(CONDS["A1_blind"], T=T)
    a = run_group(spec, 0, models("14f"))
    b = run_group(dict(spec, tag="14f_rep"), 0, models("14f_rep"))
    return all(np.array_equal(u[k], v[k]) for u, v in zip(a, b) for k in ("t", "x", "y", "dth", "d", "dt"))


def cotravel(cond, thr=SHOAL_MM, n_pseudo=100):
    """Co-travel: for every pair of fish, the episodes (1 Hz) during which they stay within ``thr`` mm; the distance
    their midpoint travels during each episode (mm). Coupled groups vs time-aligned pairs of fish from different groups
    (chance) vs the blind groups."""
    def episodes(A, B):
        close = np.linalg.norm(A - B, axis=1) <= thr
        mid = (A + B) / 2
        step = np.r_[0.0, np.linalg.norm(np.diff(mid, axis=0), axis=1)]
        out, run = [], 0.0
        for k in range(len(close)):
            if close[k]:
                run += step[k] if k and close[k - 1] else 0.0
            elif run > 0 or (k and close[k - 1]):
                out.append(run); run = 0.0
        if close[-1]:
            out.append(run)
        return out

    spec = CONDS[cond]
    N, G = spec["N"], spec["G"]
    T, taus = tracks(load(cond))
    TB, _ = tracks(load(blind_of(cond)), N)
    hours = len(taus) / 3600.0
    res = {}
    for lab, X in (("coupled", T), ("blind", TB[:G])):
        ep = [e for g in range(G) for i in range(N) for j in range(i + 1, N) for e in episodes(X[g, i], X[g, j])]
        res[lab] = dict(eps=np.array(ep), per_pair_hour=float(np.sum(ep)) / (G * N * (N - 1) / 2 * hours))
    rng = np.random.default_rng(5)
    ep = []
    n_pseudo = G * N * (N - 1) // 2                     # as many pairs as within the groups (fair counts and maxima)
    for _ in range(n_pseudo):
        g1, g2 = rng.choice(G, 2, replace=False)
        ep += episodes(T[g1, rng.integers(N)], T[g2, rng.integers(N)])
    res["chance"] = dict(eps=np.array(ep), per_pair_hour=float(np.sum(ep)) / (n_pseudo * hours))
    for v in res.values():
        v["eps_per_pair_hour"] = len(v["eps"]) / (n_pseudo * hours)
    return res


def kinematics(cond):
    """Per fish: bout rate, median step, mean |turn|, share of near-straight bouts over the readout window, coupled and
    the same fish blind; ratios / differences per fish."""
    spec = CONDS[cond]
    N, G = spec["N"], spec["G"]

    def per_fish(blob, n):
        rows = []
        for grp in blob["groups"][:G]:
            for f in grp[:n]:
                s = (f["t"] >= spec["t_read"]) & (f["t"] < spec["T"])
                rows.append((s.sum() / (spec["T"] - spec["t_read"]), np.median(f["d"][s]) / PX_MM,
                             np.mean(np.abs(f["dth"][s])), np.mean(np.abs(f["dth"][s]) < 3.0)))
        return np.array(rows)

    a, b = per_fish(load(cond), N), per_fish(load(blind_of(cond)), N)
    return dict(rate=a[:, 0] / b[:, 0], step=a[:, 1] / b[:, 1], turn=a[:, 2] - b[:, 2], straight=a[:, 3] - b[:, 3],
                abs=dict(rate=a[:, 0], step=a[:, 1], turn=a[:, 2], straight=a[:, 3]),
                abs_blind=dict(rate=b[:, 0], step=b[:, 1], turn=b[:, 2], straight=b[:, 3]))


def ood(cond):
    """Provider log over the readout window: how often neighbours are visible, groups per bout, pooled sizes, range."""
    spec = CONDS[cond]
    L = np.concatenate([f["log"] for grp in load(cond)["groups"] for f in grp])
    L = L[(L[:, 0] >= spec["t_read"]) & (L[:, 0] < spec["T"])]
    vis = L[:, 1] > 0
    Lv = L[vis]
    return dict(share_visible=float(vis.mean()), n_vis=float(Lv[:, 1].mean()) if vis.any() else 0.0,
                G_mean=float(Lv[:, 2].mean()) if vis.any() else 0.0,
                share_G_gt2=float((Lv[:, 2] > 2).mean()) if vis.any() else 0.0,
                share_pool_gt8=float((Lv[:, 3] > N_CAP).mean()) if vis.any() else 0.0,
                share_pooled=float((Lv[:, 3] > 1).mean()) if vis.any() else 0.0,
                r_near_med_mm=float(np.nanmedian(Lv[:, 5]) / PX_MM) if vis.any() else np.nan,
                share_r_near_gt44mm=float(np.mean(Lv[:, 5] / PX_MM > 44.0)) if vis.any() else np.nan,
                share_group_outside_dish=float((Lv[:, 6] > 0).mean()) if vis.any() else 0.0)


def per_fold(values, G):
    v = np.asarray(values, float)
    return [float(np.nanmean(v[np.arange(G) % 5 == k])) for k in range(5)]


def analyse(save=True, conds=None, out=None):
    """Every readout of every finished condition -> summary dict (and Model/multi_fish/summary.json). ``conds`` = a
    subset of conditions (with their blind pools), ``out`` = another summary file (the missing-pieces notebook writes
    its own, so the extrapolations notebook's summary.json is left as it was)."""
    have = {c for c in CONDS if (OUT / f"{c}.pkl").exists()}
    if conds is None:                     # default = the extrapolations notebook's original set (summary.json unchanged)
        conds = [c for c in CONDS if c not in NEW_CONDS and c not in SH_CONDS and c not in GR_CONDS]
    have &= set(conds) | {blind_of(c) for c in conds}
    S = dict(made=time.strftime("%Y-%m-%d %H:%M"), conds={}, pairs={}, groups={}, cotravel={}, kin={}, ood={})
    for c in CONDS:
        if c not in have:
            continue
        b = load(c)
        S["conds"][c] = dict(label=LABEL[c], cpu_h=round(b["cpu_s"] / 3600, 3), made=b["made"],
                             n_bouts=int(sum(len(f["t"]) for grp in b["groups"] for f in grp)),
                             **{k: v for k, v in b["spec"].items()})
    for c in [k for k in CONDS if CONDS[k]["N"] == 2 and not CONDS[k]["blind"]]:
        if c in have and blind_of(c) in have:
            r = pairs_readout(c)
            S["pairs"][c] = {lab: dict(ai_mean=float(np.mean(v["ai"])), ai_ci=_boot(v["ai"]),
                                       ai=[float(a) for a in v["ai"]],
                                       ai_folds=per_fold(v["ai"], CONDS[c]["G"]),
                                       iad_med=float(np.median(v["iad_med"])),
                                       iad_shift_med=float(np.median(v["iad_shift_med"])),
                                       share_within_10mm=float(np.mean(v["dist"] <= 10.0)),
                                       share_within_25mm=float(np.mean(v["dist"] <= 25.0)))
                             for lab, v in r.items()}
            d = np.array(r["coupled"]["ai"]) - np.array(r["blind"]["ai"])
            S["pairs"][c]["paired_diff"] = dict(mean=float(d.mean()), ci=_boot(d))
    for c in CONDS:
        if CONDS[c]["N"] < 5 or CONDS[c]["blind"] or c not in have or blind_of(c) not in have:
            continue
        r = groups_readout(c)
        G, N = CONDS[c]["G"], CONDS[c]["N"]
        S["groups"][c] = dict(
            D_mean=float(r["D"].mean()), D_sd=float(r["D"].std(ddof=1)), D_ci=r["D_ci"], D=r["D"].tolist(),
            D_folds=per_fold(r["D"], G), D_tshift_mean=float(r["D_tshift"].mean()), D_tshift_ci=_boot(r["D_tshift"]),
            D_fold_mean=float(r["D_fold"].mean()), D_fold_ci=_boot(r["D_fold"]),
            D_blind_mean=float(r["D_blind"].mean()), D_blind_ci=_boot(r["D_blind"]), occupancy=r["occupancy"],
            nn_mm=float(r["nn"].mean()), nn_shuf_mm=float(r["nn_shuf"].mean()), nn_blind_mm=float(r["nn_blind"].mean()),
            iid_ratio=float(r["iid_ratio"].mean()), iid_ratio_ci=_boot(r["iid_ratio"]),
            big_mean=float(r["big"].mean()), big_shuf=float(r["big_shuf"].mean()), big_blind=float(r["big_blind"].mean()),
            share_half_together=float(np.mean(r["big"] >= N / 2)),
            share_half_together_shuf=float(np.mean(r["big_shuf"] >= N / 2)),
            share_half_together_blind=float(np.mean(r["big_blind"] >= N / 2)),
            pol=float(r["pol"].mean()), pol_shuf=float(r["pol_shuf"].mean()),
            nn_t=_minute_curve(r["taus"], r["nn"]), nn_shuf_t=_minute_curve(r["taus"], r["nn_shuf"][:G]),
            nn_blind_t=_minute_curve(r["taus"], r["nn_blind"]))
        if CONDS[c]["arena"] == 5:
            ct = cotravel(c)
            S["cotravel"][c] = {k: dict(per_pair_hour=v["per_pair_hour"], n_eps=int(len(v["eps"])),
                                        eps_per_pair_hour=v["eps_per_pair_hour"],
                                        longest=float(v["eps"].max()) if len(v["eps"]) else 0.0,
                                        p95=float(np.percentile(v["eps"], 95)) if len(v["eps"]) else 0.0)
                                for k, v in ct.items()}
    for c in CONDS:
        if CONDS[c]["blind"] or c not in have or blind_of(c) not in have:
            continue
        k = kinematics(c)
        S["kin"][c] = {q: dict(median=float(np.median(k[q])), p10=float(np.percentile(k[q], 10)),
                               p90=float(np.percentile(k[q], 90))) for q in ("rate", "step", "turn", "straight")}
        S["ood"][c] = ood(c)
    if save:
        Path(out or (OUT / "summary.json")).write_text(json.dumps(S, indent=1, default=_json_default), encoding="utf-8")
    return S


def analyse_rep(out=None):
    """F + repulsion against F on the same scenes: every readout of ``analyse`` for both, plus the close contacts
    (``close_contacts``) -> Model/multi_fish/summary_rep.json (summary.json is left as it was)."""
    conds = list(REP_CONDS) + [REP_BASE[c] for c in REP_CONDS] + [c for c in REPS_CONDS if (OUT / f"{c}.pkl").exists()]
    S = analyse(save=False, conds=conds)
    S["contacts"] = {c: close_contacts(c) for c in conds if (OUT / f"{c}.pkl").exists()}
    S["rep_base"] = dict(REP_BASE)
    S["reps_base"] = {c: b for c, b in REPS_BASE.items() if (OUT / f"{c}.pkl").exists()}
    Path(out or (OUT / "summary_rep.json")).write_text(json.dumps(S, indent=1, default=_json_default),
                                                      encoding="utf-8")
    return S


def close_contacts_n(cond, N=None, G=None, step=1.0):
    """``close_contacts`` on the first N fish of the first G groups (a blind pool of larger groups as the chance level
    of smaller ones)."""
    spec = CONDS[cond]
    N = N or spec["N"]; G = G or spec["G"]
    T, taus = tracks(load(cond), N)
    nn = []
    for g in range(G):
        P = T[g]
        D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
        D[np.arange(len(P)), np.arange(len(P))] = np.inf
        nn.append(D.min(1).ravel())                                   # tracks() is already in mm
    nn = np.concatenate(nn)
    out = {f"share_nn_le_{k}mm": float(np.mean(nn <= k)) for k in (2, 4, 6, 10, 25)}
    out.update(nn_median=float(np.median(nn)), nn_p10=float(np.percentile(nn, 10)),
               hist=np.histogram(nn, np.arange(0, 61, 1.0))[0].tolist())
    return out


PAIR_BINS = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 70], float)


def pair_shape(cond):
    """Every fish-to-neighbour distance (all ordered pairs, 1 Hz readout grid) per bin, as a share, against the same for
    the blind pool (same fish, same dish): occupancy relative to chance per distance bin, and its 0-2 / 2-4 mm levels
    relative to 4-8 mm (real fish near a dot on the Larsch screen: 0.84 / 0.92)."""
    spec = CONDS[cond]
    N, G = spec["N"], spec["G"]
    T, _ = tracks(load(cond))
    TB, _ = tracks(load(blind_of(cond)), N)
    TB = TB[:G]

    def h(X):
        D = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=-1)       # tracks() is already in mm
        v = D[:, ~np.eye(X.shape[1], dtype=bool)].ravel()
        return np.histogram(v, PAIR_BINS)[0].astype(float) / len(v)
    ratio = h(T) / h(TB)
    ref = float(np.nanmean(ratio[2:4]))
    return dict(bins=PAIR_BINS.tolist(), ratio=ratio.tolist(), shape_0_2=float(ratio[0] / ref),
                shape_2_4=float(ratio[1] / ref))


def analyse_grouprep(out=None):
    """Every readout of ``analyse`` for the group-repulsion conditions (and F in the 5x dish), the close contacts of
    each condition and of its blind pool (N-fish groups), and the close-range shape -> summary_grouprep.json."""
    coupled = [c for c in GR_CONDS if (OUT / f"{c}.pkl").exists()] + [c for c in GR_F5.values()
                                                                      if (OUT / f"{c}.pkl").exists()]
    S = analyse(save=False, conds=coupled)
    S["contacts"] = {c: close_contacts(c) for c in coupled}
    S["contacts_blind"] = {c: close_contacts_n(blind_of(c), CONDS[c]["N"], CONDS[c]["G"]) for c in coupled}
    S["pair_shape"] = {c: pair_shape(c) for c in coupled}
    S["blind_of"] = {c: blind_of(c) for c in coupled}
    S["specs"] = {c: CONDS[c] for c in coupled}
    Path(out or (OUT / "summary_grouprep.json")).write_text(json.dumps(S, indent=1, default=_json_default),
                                                           encoding="utf-8")
    return S


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
    import shutil
    for nm, src in GRID_COPY.items():                   # identical specs, already simulated
        if not (OUT / f"{nm}.pkl").exists() and (OUT / f"{src}.pkl").exists():
            shutil.copyfile(OUT / f"{src}.pkl", OUT / f"{nm}.pkl")
            log(f"copied {src}.pkl -> {nm}.pkl")
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


def analyse_shoal(out=None):
    """Every readout of ``analyse`` for the shoaling-notebook conditions, plus the close contacts of each condition and
    of its blind pool (non-interacting fish in the same dish = the chance level of close contacts) ->
    Model/multi_fish/summary_shoal.json."""
    coupled = [c for c in SH_CONDS if not SH_CONDS[c]["blind"] and (OUT / f"{c}.pkl").exists()]
    S = analyse(save=False, conds=coupled)
    S["contacts"] = {c: close_contacts(c) for c in SH_CONDS if (OUT / f"{c}.pkl").exists()}
    S["blind_of"] = {c: blind_of(c) for c in coupled}
    Path(out or (OUT / "summary_shoal.json")).write_text(json.dumps(S, indent=1, default=_json_default),
                                                        encoding="utf-8")
    return S


def _minute_curve(taus, X):
    """(minutes, mean, lo, hi) of rows of X (replicates x frames) in 1-min bins; CI = 2.5/97.5 % bootstrap over rows."""
    X = np.atleast_2d(X)
    mins = np.floor(taus / 60.0).astype(int)
    out = []
    rng = np.random.default_rng(0)
    for m in np.unique(mins):
        v = X[:, mins == m].mean(1)
        bs = rng.choice(v, size=(500, len(v))).mean(1)
        out.append((int(m), float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))))
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def load_summary():
    return json.loads((OUT / "summary.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _same(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ("dtheta", "distance_px", "dt_s", "x_px", "y_px"))


def check_hook(tag="14f", T=600.0):
    """The provider hook ``block_time_s`` reproduces the generator's own clock bit for bit when it returns what the
    generator would compute (the recorded programme as a one-group scene)."""
    MF, SR = _imports()
    import multi_stim as MS
    prog, dsh, ref = MF._programme()

    class Mirror(MS.ProgramGroupsProvider):
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
        for P in (MS.ProgramGroupsProvider, Mirror):
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


def pick_window(cond, length_s=90.0, step_s=5.0):
    """The group and window best showing a shoal forming and then moving together: over 1 Hz frames, score =
    (largest-cluster size late in the window - early) + distance the cluster centroid travels while >= 3 fish stay
    together (/50 mm). Returns (group, t_start, score)."""
    blob = load(cond)
    spec = blob["spec"]
    taus = np.arange(0.0, spec["T"], 1.0)
    best = (None, None, -np.inf)
    for g, grp in enumerate(blob["groups"]):
        P = np.array([pos_at(f, taus) for f in grp]) / PX_MM
        big = _largest(P)
        D = np.linalg.norm(P[:, None] - P[None, :], axis=-1) <= SHOAL_MM
        cen = np.full((len(taus), 2), np.nan)
        for fr in range(len(taus)):                          # centroid of the fish linked to the most others
            A = D[:, :, fr]
            hub = int(np.argmax(A.sum(1)))
            mem = np.flatnonzero(A[hub])
            if len(mem) >= 3:
                cen[fr] = P[mem, fr].mean(0)
        stepd = np.linalg.norm(np.diff(cen, axis=0), axis=1)
        stepd = np.where(np.isfinite(stepd) & (stepd < 30.0), stepd, 0.0)
        L = int(length_s)
        for s in np.arange(0, len(taus) - L, int(step_s)):
            e = s + L
            form = big[e - 15:e].mean() - big[s:s + 15].mean()
            travel = stepd[s + L // 3:e - 1].sum()
            sc = max(form, 0.0) + travel / 50.0
            if sc > best[2]:
                best = (g, float(taus[s]), float(sc))
    return best


def view_of(panels, t0, t1, margin_mm=15.0):
    """A square view (mm, relative to the dish centre) holding every fish of ``panels`` over [t0, t1] -- the whole dish
    when that is smaller."""
    taus = np.arange(t0, t1, 1.0)
    pts = []
    for cond, g in panels:
        cx, cy, R = dish(CONDS[cond]["arena"])
        for f in load(cond)["groups"][g]:
            p = pos_at(f, taus)
            pts.append(np.c_[(p[:, 0] - cx) / PX_MM, (p[:, 1] - cy) / PX_MM])
    P = np.concatenate(pts)
    lo, hi = P.min(0) - margin_mm, P.max(0) + margin_mm
    c, h = (lo + hi) / 2, max(hi - lo) / 2
    R = dish(CONDS[panels[0][0]]["arena"])[2] / PX_MM * 1.03
    if h >= R:
        return (-R, R, -R, R)
    return (float(c[0] - h), float(c[0] + h), float(c[1] - h), float(c[1] + h))


def write_video_npz(path, panels, t0, t1, speed=2.0, labels=("",), view=None, body_mm=None):
    """``panels`` = [(cond, group)]: one dish per panel, the same window. Writes the frame data for the renderer."""
    taus = np.arange(t0, t1, speed / VIDEO_FPS)
    data, dishes = [], []
    for cond, g in panels:
        grp = load(cond)["groups"][g]
        data.append(np.array([frames_xy(f, taus) for f in grp]))     # (fish, frames, 2)
        dishes.append(dish(CONDS[cond]["arena"]))
    view = view if view is not None else view_of(panels, t0, t1)
    np.savez_compressed(path, xy=np.array(data), taus=taus, dishes=np.array(dishes), labels=np.array(labels),
                        speed=speed, view=np.array(view, float), body_mm=np.nan if body_mm is None else float(body_mm))
    return path


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


def make_video(panels, t0, t1, out_mp4, labels, speed=2.0, view=None, body_mm=None):
    """Frame data in this process, rendering in a subprocess (with the bundled ffmpeg when none is on PATH)."""
    import subprocess
    sys.path.insert(0, str(HERE))
    import bout_video as BV
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    npz = out_mp4.with_suffix(".npz")
    write_video_npz(npz, panels, t0, t1, speed, labels, view, body_mm)
    env, rc_dir = BV._render_env(out_mp4)
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--render", str(npz), str(out_mp4)],
                       check=True, env=env)
    finally:
        if rc_dir is not None:
            import shutil
            shutil.rmtree(rc_dir, ignore_errors=True)
    npz.unlink(missing_ok=True)
    return out_mp4


def snapshot(cond, g, times, ax_row=None, blind=False, view=None):
    """Still frames (for the notebook): fish positions at ``times`` with 4 s trails; ``view`` = (x0, x1, y0, y1) mm."""
    import matplotlib.pyplot as plt
    blob = load(blind_of(cond) if blind else cond)
    grp = blob["groups"][g][:CONDS[cond]["N"]]
    cx, cy, R = dish(CONDS[cond]["arena"])
    cols = plt.get_cmap("tab10")(np.arange(len(grp)) % 10)
    axes = ax_row if ax_row is not None else plt.subplots(1, len(times), figsize=(3.2 * len(times), 3.3))[1]
    for ax, t in zip(np.atleast_1d(axes), times):
        tt = np.arange(max(t - 4.0, 0.0), t + 1e-9, 1.0 / VIDEO_FPS)
        ax.add_patch(plt.Circle((0, 0), R / PX_MM, fill=False, lw=1.0, color="0.35"))
        for i, f in enumerate(grp):
            p = frames_xy(f, tt)
            ax.plot((p[:, 0] - cx) / PX_MM, (p[:, 1] - cy) / PX_MM, lw=0.8, color=cols[i], alpha=0.6)
            ax.scatter([(p[-1, 0] - cx) / PX_MM], [(p[-1, 1] - cy) / PX_MM], s=14, color=cols[i], zorder=3)
        lim = R / PX_MM * 1.03
        x0, x1, y0, y1 = view if view is not None else (-lim, lim, -lim, lim)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.axis("off")
        m, s = divmod(t, 60.0)
        ax.set_title(f"{int(m):02d}:{s:02.0f}", fontsize=9)
    return axes


# --------------------------------------------------------------------------- #
# Notebook helpers: tables and figures from summary.json / checks.json / the pickles
# --------------------------------------------------------------------------- #
VIDEO_DIR = OUT / "videos"
LIT_HARPAZ = {"7 dpf": (0.075, 0.067), "14 dpf": (-0.097, 0.083), "21 dpf": (-0.55, 0.24), "7 dpf, dark": (0.01, 0.068)}
COLORS = {"coupled": "#1f77b4", "shuffled": "#ff7f0e", "blind": "#7f7f7f", "time-shifted": "#ff7f0e",
          "chance": "#ff7f0e"}


def _fmt_ci(m, ci, nd=3):
    return f"{m:+.{nd}f} [{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]"


def conditions_table(S=None):
    import pandas as pd
    S = S or load_summary()
    rows = []
    for c, d in S["conds"].items():
        rows.append({"condition": c, "what": d["label"],
                     "model": {"14f": "F, seed 0", "32b": "F, seed 101", "26e": "M6, seed 0", "30d": "M6, seed 101"}[d["tag"]],
                     "dish radius (mm)": round(dish(d["arena"])[2] / PX_MM),
                     "fish per dish": d["N"], "dishes": d["G"], "minutes": d["T"] / 60,
                     "readout from (min)": d["t_read"] / 60, "fish see each other": not d["blind"],
                     "rule": "-" if d["blind"] else d["rule"] + (" (random directions)" if d["scramble"] else ""),
                     "bouts": d["n_bouts"], "CPU h": d["cpu_h"]})
    return pd.DataFrame(rows).set_index("condition")


def checks_table():
    import pandas as pd
    c = json.loads((OUT / "checks.json").read_text(encoding="utf-8"))
    rows = [("generator hook: a provider-set block time equal to the generator's own gives the same fish bit for bit",
             c["hook_bit_identical"][0]),
            ("coupling: a coupled fish == the generator run alone on a replay of its partner's track (2 rules x 3 pairs)",
             c["parity_coupled_vs_replay"][0]),
            ("determinism: a group of 10 run twice is identical", c["deterministic"]),
            ("pairing: a blind fish is the same fish in a group of 10 and in a pair", c["blind_pairing"]),
            ("two-group simulator (multi_stim.validate) still bit-identical after the hook", True)]
    return pd.DataFrame(rows, columns=["check", "passed"]).set_index("check")


def rules_table():
    """The rule check: phantom neighbours on one side; P(turn toward them) and kinematics per rule and number."""
    import pandas as pd
    c = json.loads((OUT / "checks.json").read_text(encoding="utf-8"))
    t = pd.DataFrame(c["rules"])
    real = {int(k): v for k, v in c["real_toward"].items()}
    t["real P(toward), same count"] = [f"{real[k][0]:.3f} +- {real[k][1]:.3f}" if k in real else
                                       ("3 dots never shown" if k == 3 else ("-" if k == 0 else "> 8 never shown"))
                                       for k in t.k]
    t = t.rename(columns={"k": "phantom neighbours", "p_toward": "P(turn toward)", "rate": "bouts / s",
                          "step": "median step (mm)", "turn": "mean |turn| (deg)"})
    return t.set_index(["rule", "phantom neighbours"]).round(3)


def pairs_table(S=None):
    import pandas as pd
    S = S or load_summary()
    rows = []
    for c, d in S["pairs"].items():
        for lab in ("coupled", "blind"):
            v = d[lab]
            rows.append({"pairs": LABEL[c], "fish": "see each other" if lab == "coupled" else "blind (same fish)",
                         "attraction index [95% CI]": _fmt_ci(v["ai_mean"], v["ai_ci"]),
                         "median distance (mm)": round(v["iad_med"], 1),
                         "median distance, partner time-shifted (mm)": round(v["iad_shift_med"], 1),
                         "time within 10 mm": round(v["share_within_10mm"], 3),
                         "time within 25 mm": round(v["share_within_25mm"], 3),
                         "per-fold AI": " ".join(f"{a:+.2f}" for a in v["ai_folds"])})
        rows.append({"pairs": LABEL[c], "fish": "difference (paired by pair)",
                     "attraction index [95% CI]": _fmt_ci(d["paired_diff"]["mean"], d["paired_diff"]["ci"])})
    return pd.DataFrame(rows).set_index(["pairs", "fish"]).fillna("")


def groups_table(S=None):
    import pandas as pd
    S = S or load_summary()
    rows = []
    for c, d in S["groups"].items():
        rows.append({"condition": LABEL[c],
                     "NN distance (mm): see / shuffled / blind": f"{d['nn_mm']:.1f} / {d['nn_shuf_mm']:.1f} / {d['nn_blind_mm']:.1f}",
                     "D = ln(NN / NN shuffled) [95% CI]": _fmt_ci(d["D_mean"], d["D_ci"]),
                     "D, SD over dishes": round(d["D_sd"], 3),
                     "D vs time-shifted group": _fmt_ci(d["D_tshift_mean"], d["D_tshift_ci"]),
                     "D vs fold-matched null (sensitivity)": _fmt_ci(d["D_fold_mean"], d["D_fold_ci"]),
                     "D of the blind fish (should be ~0)": _fmt_ci(d["D_blind_mean"], d["D_blind_ci"]),
                     "occupancy part ln(NN shuffled / NN blind)": round(d["occupancy"], 3),
                     "mean distance ratio (Buske)": round(d["iid_ratio"], 3),
                     "largest shoal (fish): see / shuffled / blind": f"{d['big_mean']:.2f} / {d['big_shuf']:.2f} / {d['big_blind']:.2f}",
                     "time >= half the fish together: see / shuffled / blind":
                         f"{d['share_half_together']:.2f} / {d['share_half_together_shuf']:.2f} / {d['share_half_together_blind']:.2f}",
                     "alignment: see / shuffled": f"{d['pol']:.3f} / {d['pol_shuf']:.3f}",
                     "per-fold D": " ".join(f"{a:+.2f}" for a in d["D_folds"])})
    return pd.DataFrame(rows).set_index("condition")


def literature_table(S=None):
    import pandas as pd
    S = S or load_summary()
    rows = [{"source": "Harpaz et al. 2021 (groups of 5/10, real fish)", "measure": "D = ln(NN / NN shuffled)",
             "value": ", ".join(f"{k}: {m:+.3f} +- {s:.3f}" for k, (m, s) in LIT_HARPAZ.items())}]
    for c, d in S["groups"].items():
        rows.append({"source": f"model: {LABEL[c]}", "measure": "D = ln(NN / NN shuffled)",
                     "value": f"{d['D_mean']:+.3f} +- {d['D_sd']:.3f} (mean +- SD over {CONDS[c]['G']} dishes)"})
    rows.append({"source": "Buske & Gerlai 2011 (groups of 10)", "measure": "mean distance / random placement",
                 "value": "0.88-1.0 at 7-36 dpf, 0.40 at 76 dpf (null = uniform placement in a square)"})
    for c, d in S["groups"].items():
        if CONDS[c]["N"] == 10:
            rows.append({"source": f"model: {LABEL[c]}", "measure": "mean distance / shuffled groups",
                         "value": f"{d['iid_ratio']:.3f}"})
    rows.append({"source": "Larsch & Baier 2018 (pairs, 10 cm dish)", "measure": "pair distance vs time-shifted",
                 "value": "attraction from ~14 dpf; 26 dpf pairs 5-20 mm vs ~40 mm chance"})
    for c, d in S["pairs"].items():
        v = d["coupled"]
        rows.append({"source": f"model: {LABEL[c]}", "measure": "pair distance vs time-shifted",
                     "value": f"median {v['iad_med']:.1f} mm vs {v['iad_shift_med']:.1f} mm; AI {v['ai_mean']:+.3f}"})
    rows.append({"source": "Harpaz et al. 2021 Sci Adv (groups of 5)", "measure": "alignment",
                 "value": "0.417 (7 dpf), 0.458 (21 dpf); random headings ~0.40"})
    for c, d in S["groups"].items():
        if CONDS[c]["N"] == 5:
            rows.append({"source": f"model: {LABEL[c]}", "measure": "alignment",
                         "value": f"{d['pol']:.3f} (shuffled {d['pol_shuf']:.3f})"})
    return pd.DataFrame(rows).set_index(["source", "measure"])


def real_kin_reference():
    """Real fish, per fish: moving-dot blocks vs pauses -- bout rate ratio, median step ratio, mean |turn| and
    near-straight share differences (median [p10, p90] over the 15 fish): the range a 'sees moving dots' fish shows."""
    MF, _ = _imports()
    rows = []
    for r in MF.records().values():
        dt = np.asarray(r.dt_s, float); d = np.asarray(r.distance_px, float) / PX_MM; a = np.abs(np.asarray(r.dtheta, float))
        mv = (np.asarray(r.n_dots) > 0) & (np.asarray(r.moving) > 0); pa = np.asarray(r.n_dots) == 0
        ok = np.isfinite(dt) & (dt > 0)
        rows.append(((1 / np.mean(dt[mv & ok])) / (1 / np.mean(dt[pa & ok])), np.median(d[mv]) / np.median(d[pa]),
                     a[mv].mean() - a[pa].mean(), (a[mv] < 3).mean() - (a[pa] < 3).mean()))
    R = np.array(rows)
    return {q: (float(np.median(R[:, k])), float(np.percentile(R[:, k], 10)), float(np.percentile(R[:, k], 90)))
            for k, q in enumerate(("rate", "step", "turn", "straight"))}


def kin_table(S=None):
    import pandas as pd
    S = S or load_summary()
    ref = real_kin_reference()
    f = lambda q, v: f"{v['median']:.3f} [{v['p10']:.3f}, {v['p90']:.3f}]"
    rows = [{"condition": "REAL fish: moving dots vs pause (per fish)",
             **{n: f"{ref[q][0]:.3f} [{ref[q][1]:.3f}, {ref[q][2]:.3f}]" for q, n in
                (("rate", "bout rate ratio"), ("step", "median step ratio"), ("turn", "mean |turn| difference (deg)"),
                 ("straight", "near-straight share difference"))}}]
    for c, d in S["kin"].items():
        rows.append({"condition": LABEL[c], "bout rate ratio": f("rate", d["rate"]),
                     "median step ratio": f("step", d["step"]), "mean |turn| difference (deg)": f("turn", d["turn"]),
                     "near-straight share difference": f("straight", d["straight"])})
    return pd.DataFrame(rows).set_index("condition")


def ood_table(S=None):
    import pandas as pd
    S = S or load_summary()
    t = pd.DataFrame(S["ood"]).T
    t.index = [LABEL[c] for c in t.index]
    return t.rename(columns={"share_visible": "bouts with a neighbour in view", "n_vis": "neighbours in view",
                             "G_mean": "dot groups per bout", "share_G_gt2": "bouts with > 2 groups",
                             "share_pooled": "bouts with a pooled group", "share_pool_gt8": "bouts with a pool > 8",
                             "r_near_med_mm": "nearest group (mm, median)",
                             "share_r_near_gt44mm": "nearest group > 44 mm",
                             "share_group_outside_dish": "a group outside the dish"}).round(3)


def plot_nn_time(S=None, conds=None):
    """Nearest-neighbour distance (mm) vs time, one panel per condition: fish that see each other, pseudo-groups of
    fish from different dishes (same condition), the same fish blind. Bands: 95% bootstrap CI over dishes."""
    import matplotlib.pyplot as plt
    S = S or load_summary()
    conds = conds or list(S["groups"])
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.2), sharey=False, squeeze=False)
    for ax, c in zip(axes[0], conds):
        d = S["groups"][c]
        for key, lab in (("nn_t", "see each other"), ("nn_shuf_t", "shuffled (other dishes)"),
                         ("nn_blind_t", "same fish, blind")):
            a = np.array(d[key])
            col = COLORS["coupled" if key == "nn_t" else "shuffled" if key == "nn_shuf_t" else "blind"]
            ax.plot(a[:, 0], a[:, 1], color=col, lw=1.5, label=lab)
            ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=col, alpha=0.2, lw=0)
        ax.set_title(LABEL[c], fontsize=9); ax.set_xlabel("time (min)")
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("nearest-neighbour distance (mm)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def plot_D(S=None):
    """One point per dish: D = ln(NN / NN shuffled), with the mean and 95% CI; Harpaz's real-fish values as lines."""
    import matplotlib.pyplot as plt
    S = S or load_summary()
    conds = list(S["groups"])
    fig, ax = plt.subplots(figsize=(1.1 * len(conds) + 3, 3.4))
    rng = np.random.default_rng(0)
    for k, c in enumerate(conds):
        d = S["groups"][c]
        v = np.array(d["D"])
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color="0.45", zorder=2)
        ax.errorbar(k, d["D_mean"], yerr=[[d["D_mean"] - d["D_ci"][0]], [d["D_ci"][1] - d["D_mean"]]], fmt="o",
                    color=COLORS["coupled"], ms=6, capsize=3, zorder=3)
    for lab, (m, _) in LIT_HARPAZ.items():
        if "dark" in lab:
            continue
        ax.axhline(m, ls="--", lw=0.8, color="0.3")
        ax.text(len(conds) - 0.45, m, f"Harpaz {lab}", fontsize=7, va="bottom", ha="right", color="0.3")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([LABEL[c] for c in conds], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("D = ln(NN / NN shuffled)\n< 0: closer than chance")
    fig.tight_layout()
    return fig


def plot_pair_distance(conds=("A1_pairs", "s101_A1_pairs")):
    """Distance between the two fish of a pair (1 Hz, readout window): pairs that see each other, the same pairs with
    the partner's track time-shifted (Larsch's chance), the same fish blind."""
    import matplotlib.pyplot as plt
    conds = [c for c in conds if (OUT / f"{c}.pkl").exists()]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.2), squeeze=False)
    bins = np.linspace(0, 100, 51)
    for ax, c in zip(axes[0], conds):
        G = CONDS[c]["G"]
        T, taus = tracks(load(c), 2)
        TB, _ = tracks(load(blind_of(c)), 2)
        d = np.linalg.norm(T[:, 0] - T[:, 1], axis=-1).ravel()
        ds = np.concatenate([np.linalg.norm(T[:, 0] - np.roll(T[:, 1], -int(s), axis=1), axis=-1).ravel()
                             for s in SHIFTS_S])
        db = np.linalg.norm(TB[:G, 0] - TB[:G, 1], axis=-1).ravel()
        for v, lab, col in ((d, "see each other", COLORS["coupled"]), (ds, "partner time-shifted", COLORS["time-shifted"]),
                            (db, "same fish, blind", COLORS["blind"])):
            h, _ = np.histogram(v, bins=bins, density=True)
            ax.plot((bins[1:] + bins[:-1]) / 2, h, color=col, lw=1.5, label=lab)
        ax.set_title(LABEL[c], fontsize=9); ax.set_xlabel("distance between the two fish (mm)")
    axes[0, 0].set_ylabel("density"); axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def plot_cotravel(conds=("A5_N10_effects", "A5_N10_nearest")):
    """5x dish: how far two fish travel while staying within 25 mm of each other (one value per episode); share of
    episodes longer than x, fish that see each other vs pairs from different dishes (chance) vs blind."""
    import matplotlib.pyplot as plt
    conds = [c for c in conds if (OUT / f"{c}.pkl").exists()]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.2), squeeze=False)
    for ax, c in zip(axes[0], conds):
        ct = cotravel(c)
        for lab, col in (("coupled", COLORS["coupled"]), ("chance", COLORS["chance"]), ("blind", COLORS["blind"])):
            e = np.sort(ct[lab]["eps"])
            if not len(e):
                continue
            ax.step(e, 1 - np.arange(len(e)) / len(e), where="post", color=col, lw=1.5,
                    label={"coupled": "see each other", "chance": "pairs from different dishes",
                           "blind": "same fish, blind"}[lab])
        ax.set_xscale("symlog", linthresh=10); ax.set_yscale("log")
        ax.set_xlabel("distance travelled together (mm)"); ax.set_title(LABEL[c], fontsize=9)
    axes[0, 0].set_ylabel("share of episodes longer"); axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def cotravel_table(S=None):
    import pandas as pd
    S = S or load_summary()
    rows = []
    for c, d in S["cotravel"].items():
        for lab, name in (("coupled", "see each other"), ("chance", "pairs from different dishes"),
                          ("blind", "same fish, blind")):
            v = d[lab]
            rows.append({"condition": LABEL[c], "fish": name, "mm travelled together per pair-hour": round(v["per_pair_hour"], 1),
                         "episodes per pair-hour": round(v["eps_per_pair_hour"], 2), "longest episode (mm)": round(v["longest"], 1),
                         "95th percentile (mm)": round(v["p95"], 1)})
    return pd.DataFrame(rows).set_index(["condition", "fish"])


def shoaling(S, c, d_max=-0.1):
    """The criterion for making a video: D's CI below 0 and D below ``d_max`` (about Harpaz's 14-dpf level)."""
    d = S["groups"].get(c)
    return bool(d and d["D_ci"][1] < 0 and d["D_mean"] < d_max)


def pick_formation(cond, before_s=180.0, after_s=120.0, sustain_s=30.0):
    """The dish where a shoal of >= 9 forms earliest from a scattered start (largest single-linkage cluster at 25 mm
    <= 3 fish over the first 20 s, then >= 9 for at least ``sustain_s``); window = [t9 - before_s, t9 + after_s]."""
    blob = load(cond)
    taus = np.arange(0.0, blob["spec"]["T"], 1.0)
    best = None
    for g, grp in enumerate(blob["groups"]):
        P = np.array([pos_at(f, taus) for f in grp]) / PX_MM
        big = _largest(P)
        if big[:20].max() > 3:
            continue
        ok = np.convolve(big >= 9, np.ones(int(sustain_s)), "valid") >= sustain_s
        if not ok.any():
            continue
        t9 = float(taus[int(np.argmax(ok))])
        if best is None or t9 < best[1]:
            best = (g, t9)
    if best is None:
        return None
    g, t9 = best
    return g, max(0.0, t9 - before_s), t9 + after_s, t9


def pick_cohesion(cond, length_s=90.0):
    """The dish and window where the group is tightest (lowest mean nearest-neighbour distance) after minute 5."""
    blob = load(cond)
    taus = np.arange(blob["spec"]["t_read"], blob["spec"]["T"], 1.0)
    best = (None, None, np.inf)
    L = int(length_s)
    for g, grp in enumerate(blob["groups"]):
        nn = _nn1(np.array([pos_at(f, taus) for f in grp]) / PX_MM)
        c = np.convolve(nn, np.ones(L) / L, "valid")
        k = int(np.argmin(c))
        if c[k] < best[2]:
            best = (g, float(taus[k]), float(c[k]))
    return best


def make_videos(S=None, log=print):
    """(1) 5x dish: a shoal forming from a scattered start and moving off; (2) 5x dish: a shoal travelling (window of
    ``pick_window``); (3) training dish: the tightest stretch. Each beside the same fish blind. Only for conditions that
    shoal (``shoaling``). Returns [(mp4, cond, group, t0, t1, title)]."""
    S = S or load_summary()
    made = []

    def go(c, g, t0, t1, speed, title, tag):
        out = VIDEO_DIR / f"{tag}_{c}_dish{g}_{int(t0)}-{int(t1)}s.mp4"
        make_video([(c, g), (blind_of(c), g)], t0, t1, out,
                   labels=(f"{LABEL[c]}: the fish see each other", "the same 10 fish, blind"), speed=speed)
        log(f"wrote {out}")
        made.append((str(out), c, g, float(t0), float(t1), title))

    c5 = "A5_N10_effects"
    if shoaling(S, c5):
        f = pick_formation(c5)
        if f is not None:
            g, t0, t1, t9 = f
            go(c5, g, t0, t1, 6.0, f"5x dish: a shoal forms (all 10 together from {t9 / 60:.1f} min) and moves off",
               "1_formation")
        g, t0, sc = pick_window(c5, length_s=180.0)
        go(c5, g, t0, t0 + 180.0, 4.0, "5x dish: a shoal travelling together", "2_travel")
    else:
        log("5x dish: no shoaling -> no video")
    c1 = "A1_N10_effects"
    if shoaling(S, c1):
        g, t0, _ = pick_cohesion(c1)
        go(c1, g, t0, t0 + 90.0, 2.0, "training dish: the tightest 90 s", "3_training_dish")
    else:
        log("training dish: no shoaling -> no video")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    (VIDEO_DIR / "videos.json").write_text(json.dumps(made, indent=1), encoding="utf-8")
    return made


VIDEO_BODY_MM = 2.0   # Final_report_shoaling videos: each fish a dot 2 mm across (the collision distance), in dish units


def make_grouprep_videos(conds=("g5_grT", "g5_grT2", "g5_grT8", "g5_grTC"), length_s=180.0, speed=4.0, log=print, full=False):
    """Final_report_shoaling.ipynb: for each condition (default: the calibrated group repulsion and its x2 strength,
    5 fish, 5x dish, seed 0) the stretch where a shoal forms and its centre travels furthest (``pick_window``), beside
    the same 5 fish blind. -> VIDEO_DIR/grouprep/*.mp4 and videos_grouprep.json [(mp4, cond, group, t0, t1, title)].
    ``full``: the whole dish in view instead of zooming onto the fish (*_fulldish.mp4, videos_grouprep_full.json)."""
    made = []
    vd = VIDEO_DIR / "grouprep"
    for c in conds:
        g, t0, sc = pick_window(c, length_s=length_s)
        out = vd / f"{c}_dish{g}_{int(t0)}-{int(t0 + length_s)}s{'_fulldish' if full else ''}.mp4"
        lim = dish(CONDS[c]["arena"])[2] / PX_MM * 1.03
        make_video([(c, g), (blind_of(c), g)], t0, t0 + length_s, out,
                   labels=(f"{LABEL[c]}: the fish see each other", f"the same {CONDS[c]['N']} fish, blind"), speed=speed,
                   view=(-lim, lim, -lim, lim) if full else None, body_mm=VIDEO_BODY_MM)
        log(f"wrote {out} (window score {sc:.2f})")
        made.append((str(out), c, int(g), float(t0), float(t0 + length_s), f"{LABEL[c].split(',')[0]}: a shoal travelling"))
    vd.mkdir(parents=True, exist_ok=True)
    (vd / ("videos_grouprep_full.json" if full else "videos_grouprep.json")).write_text(json.dumps(made, indent=1),
                                                                                      encoding="utf-8")
    return made


def make_grouprep_paired_video(conds=("g5_grT", "g5_grT2", "g5_grT8", "g5_grTC"), length_s=180.0, speed=4.0, seed=2026, log=print):
    """ONE dish and ONE window for every condition (paired: same fish, starts and random streams), picked by a NEUTRAL
    rule -- a random group and a random start inside the readout, from ``seed`` -- with the same fish blind as the last
    panel; a zoomed and a full-dish version. Also the window's nearest-neighbour statistics per panel (to show it is
    typical). -> VIDEO_DIR/grouprep/paired_*.mp4 and videos_grouprep_paired.json."""
    spec = CONDS[conds[0]]
    rng = np.random.default_rng(seed)
    g = int(rng.integers(spec["G"]))
    t0 = float(np.round(rng.uniform(spec["t_read"], spec["T"] - length_s)))
    t1 = t0 + length_s
    panels = [(c, g) for c in conds] + [(blind_of(conds[0]), g)]
    labels = [LABEL[c].split(",")[0].replace("F + group repulsion", "GR") for c in conds] + [f"the same {spec['N']} fish, blind"]
    stats = []
    taus = np.arange(t0, t1, 0.5)
    for (c, gg), lab in zip(panels, labels):
        P = np.array([pos_at(f, taus) for f in load(c)["groups"][gg][:spec["N"]]]) / PX_MM
        D = np.linalg.norm(P[:, None] - P[None, :], axis=-1); D[np.arange(len(P)), np.arange(len(P))] = np.inf
        nn = D.min(1).ravel()
        stats.append(dict(panel=lab, cond=c, nn_median=float(np.median(nn)),
                          **{f"le_{k}mm": float(np.mean(nn <= k)) for k in (2, 4, 6)}))
    vd = VIDEO_DIR / "grouprep"
    made = []
    lim = dish(spec["arena"])[2] / PX_MM * 1.03
    for full in (False, True):
        out = vd / f"paired_dish{g}_{int(t0)}-{int(t1)}s{'_fulldish' if full else ''}.mp4"
        make_video(panels, t0, t1, out, labels=tuple(labels), speed=speed, view=(-lim, lim, -lim, lim) if full else None,
                   body_mm=VIDEO_BODY_MM)
        log(f"wrote {out}")
        made.append(dict(mp4=str(out), full=full))
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "videos_grouprep_paired.json").write_text(json.dumps(dict(group=g, t0=t0, t1=t1, seed=seed, videos=made,
                                                                    stats=stats), indent=1), encoding="utf-8")
    return made, stats


def make_videos_old(S=None, log=print):
    """Videos for the shoaling conditions: the dish and window that best show a shoal forming and moving, the same fish
    blind beside it. Returns [(mp4, cond, group, t0, t1)]."""
    S = S or load_summary()
    made = []
    for arena, length, speed in ((1, 90.0, 2.0), (5, 180.0, 4.0)):
        cands = [c for c in S["groups"] if CONDS[c]["arena"] == arena and CONDS[c]["tag"] == "14f"
                 and not CONDS[c]["scramble"] and CONDS[c]["N"] == 10 and shoaling(S, c)]
        if not cands:
            log(f"arena x{arena}: no shoaling condition -> no video")
            continue
        eff = [k for k in cands if CONDS[k]["rule"] == "effects"]          # the reference rule when it shoals
        c = eff[0] if eff else min(cands, key=lambda k: S["groups"][k]["D_mean"])
        g, t0, sc = pick_window(c, length_s=length)
        out = VIDEO_DIR / f"{c}_dish{g}_{int(t0)}s.mp4"
        make_video([(c, g), (blind_of(c), g)], t0, t0 + length, out,
                   labels=(f"{LABEL[c]}: the fish see each other", "the same 10 fish, blind"), speed=speed)
        log(f"wrote {out} (dish {g}, {t0:.0f}-{t0 + length:.0f} s, score {sc:.2f})")
        made.append((str(out), c, g, t0, t0 + length))
    (VIDEO_DIR / "videos.json").parent.mkdir(parents=True, exist_ok=True)
    (VIDEO_DIR / "videos.json").write_text(json.dumps(made, indent=1), encoding="utf-8")
    return made


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--render":
        render_npz(a[1], a[2])
    elif a and a[0] == "--checks":
        _worker_init(); run_checks()
    elif a and a[0] == "--run":
        _worker_init(); run_conditions(a[1:] or None)
    elif a and a[0] == "--analyse":
        analyse()
    elif a and a[0] == "--run-shoal":
        _worker_init(); run_conditions([c for c in SH_CONDS if not (OUT / f"{c}.pkl").exists()])
    elif a and a[0] == "--analyse-shoal":
        analyse_shoal()
    elif a and a[0] == "--run-grouprep":
        _worker_init(); run_conditions([c for c in GR_CONDS if not (OUT / f"{c}.pkl").exists()])
    elif a and a[0] == "--analyse-grouprep":
        analyse_grouprep()
    elif a and a[0] == "--run-pairdm":
        _worker_init(); run_pairdm(int(a[1]) if len(a) > 1 else 16)
    elif a and a[0] == "--analyse-pairdm":
        analyse_pairdm()
    elif a and a[0] == "--run-grid":
        _worker_init(); run_grid(int(a[1]) if len(a) > 1 else 16)
    elif a and a[0] == "--analyse-grid":
        analyse_grid()
    elif a and a[0] == "--videos-grouprep":
        make_grouprep_videos()
        make_grouprep_videos(full=True)
        make_grouprep_paired_video()
    else:
        print(__doc__)
