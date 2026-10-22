"""Chapter 5, section 5.4 (sec:r-dots): the per-bout turn rule, real fish against the fitted models.

(a-c) Response maps: mean toward-turn (deg) over |bearing| x range, moving dots, 2/4/6/8 dots, >= 10 s into the
      block, per fish then across the 15 fish; real fish, M6 seed 0, F seed 0 (teacher-forced, population latent).
      Source: Model/missing_pieces/laws.pkl via missing_pieces.law_maps (n = (2, 4, 6, 8)). A cell is masked when
      fewer than MIN_FISH fish have at least MIN_BOUTS bouts in it (the plotted values are law_maps' own). Every
      held-out bout whose dots have a side (|bearing| 10-170 deg); NO dot-centroid-inside-dish filter; ranges in mm
      at missing_pieces.PX_MM = 3.60 px/mm. Sequential single-hue scale 0-30 deg (no cell mean is negative; the
      script asserts it for all five maps, seed 101 included).
(d)   The law summary of missing_pieces.law_summary (onset gain, steady gain, count step, static gain), recomputed
      per fish so that a 95 % bootstrap CI over the 15 fish can be drawn (missing_pieces._boot_ci, 4000 resamples).
(e)   The three teacher-forced head gaps (model - real, deg) of Model/report_outputs/summary_M<key>.json along the
      main line and the drift branch, at both seeds (never pooled), as a forest plot with the 95 % t-interval of
      the paired per-fish gap across the fish that enter each cell. The per-fish gaps are recomputed with
      model_figures.head_gaps (teacher-forced only, no rollouts; 14 units, <= 8 processes) and cached; the script
      checks that the recomputed means give back the archived summary values. M2 is omitted: it has M1's weights,
      so its teacher-forced gaps are M1's by construction. Bouts: the profile bouts of model_figures.real_dot_profile
      (dot centroid inside the dish), all bearings, >= 10 bouts per fish, condition and phase.

Run:  uv run python scripts/report_figs/fig_rule.py                # build the head-gap cache if missing, check, draw
      uv run python scripts/report_figs/fig_rule.py --rebuild      # recompute the head gaps (14 jobs, <= 8 workers)
      (also writes $RF/figs/fig_rule_numbers.json)
"""
import sys
from pathlib import Path

from figures.style import *  # noqa: F401,F403,E402

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

from twin import missing_pieces as P  # noqa: E402

N_MAP = (2, 4, 6, 8)               # moving-dot counts pooled in the maps
MIN_BOUTS, MIN_FISH = 5, 10        # a fish counts in a cell with >= MIN_BOUTS bouts; cell shown with >= MIN_FISH fish
VMAX = 30.0
RFROOT = REPORT_FINAL_DIR
RF = RFROOT / "figs"
CACHE = RFROOT / "cache"
HG_CACHE = CACHE / "fig_rule_headgaps.pkl"

# public name -> (seed 0 key, seed 101 key); CONTEXT.md
GAP_UNITS = [("M1", "1", "34a"), ("M3", "3", "34c"), ("M4", "20f", "30a"), ("M5", "25d", "30b"),
             ("M6", "26e", "30d"), ("D4", "11e", "32a"), ("F", "14f", "32b")]
# summary key, head_gaps cell (cond, phase, qty), column label of (e); report_models._summary defines the keys
GAPS = [("gap_onset_N8", ("d8s3.5", "0-4 s", "toward"), "turn toward\n8 dots, 0–4 s"),
        ("gap_ahead_N8", ("d8s3.5", "20-60 s", "ahead"), "|Δθ|, 8 dots\nfar ahead, 20–60 s"),
        ("gap_toward_N2", ("d2s3.5", "20-60 s", "toward"), "turn toward\n2 dots, 20–60 s")]
SOURCES = [("real", "real fish", REAL_C, 0), ("M6 s0", "M6", MODEL_C["M6"], 0),
           ("M6 s101", "M6", MODEL_C["M6"], 101), ("F s0", "F", MODEL_C["F"], 0),
           ("F s101", "F", MODEL_C["F"], 101)]
LAWS = [("onset_gain", "onset\n0–4 s"), ("steady_gain", "steady\n≥ 10 s"),
        ("count_step", "count step\n2–8 vs 1 dot"), ("static_gain", "static\n≥ 10 s")]
OTHER_C = "#8a8a8a"                # main-line / drift-branch units other than M6 and F in (e)
UNIT_C = {"M6": MODEL_C["M6"], "F": MODEL_C["F"]}
WHITE = "#ffffff"
LABEL_DARK = "#000000"            # in-cell labels: black or white, whichever contrasts more (>= 4.58:1)




STEPS = [("M1", "M3"), ("M3", "M4"), ("M4", "M5"), ("M5", "M6"), ("M3", "M6"), ("D4", "F"), ("M6", "F")]




# --------------------------------------------------------------------------- drawing
# sequential, one hue light -> dark (ColorBrewer Purples, its near-white end trimmed); purple is not a model colour
CMAP = LinearSegmentedColormap.from_list("seqpurple", plt.get_cmap("Purples")(np.linspace(0.04, 0.95, 256)))




# --------------------------------------------------------------------------- the onset turn (report figure, 2026-10-21)
T_BINS = [(0, 2), (2, 4), (4, 7), (7, 10), (10, 20), (20, 40), (40, 60)]
B_EDGES, R_EDGES = [45, 75, 105, 135], [0, 10, 17.5, 25, 35, 47.5, 60]   # report: 0-60 mm (10-35 mm gives the same)


def _geo(t, R_EDGES=R_EDGES):
    """Moving dots, |bearing| 45-135 deg, range 10-35 mm, in 3 x 3 geometry cells; block time in T_BINS."""
    t = t[t.moving].copy()
    t["deg"] = np.degrees(np.abs(t.phi))
    t = t[(t.deg >= B_EDGES[0]) & (t.deg <= B_EDGES[-1]) & (t.rng_mm >= R_EDGES[0]) & (t.rng_mm < R_EDGES[-1])]
    t["cell"] = (pd.cut(t.deg, B_EDGES).cat.codes * 10 + pd.cut(t.rng_mm, R_EDGES, right=False).cat.codes).astype(int)
    t["tb"] = pd.cut(t.t_blk, [a for a, _ in T_BINS] + [T_BINS[-1][1]], right=False).cat.codes
    return t[t.tb >= 0]


def _std(t, col, w):
    """Geometry-matched turn toward per fish and time bin: the fish's mean in each cell, weighted by the common cell
    weights w (renormalised over the cells the fish has in that bin). -> DataFrame fish x time bin."""
    m = t.groupby(["fish", "tb", "cell"])[col].mean().rename("m").reset_index()
    m["w"] = m.cell.map(w)
    return (m.assign(wm=m.m * m.w).groupby(["fish", "tb"]).apply(lambda d: d.wm.sum() / d.w.sum(), include_groups=False)
            .unstack("tb"))


def onset_data(r_edges=R_EDGES):
    """Real fish and M6 / F (both seeds): geometry-matched turn toward the dots per fish and time bin, the expected
    model turn and its 20 single-draw replicates (laws_draw cache)."""
    td = _geo(P._toward_cols(P.load("laws_draw")), r_edges)
    sg = np.sign(td.phi)
    for j in range(P.N_DRAW):
        td[f"tow_d{j}"] = td[f"draw{j}"] * sg
    w = td[td.model == "M6 s0"].groupby("cell").size()
    w = w / w.sum()                                             # common geometry: all real bouts of the window
    out = {"real": _std(td[td.model == "M6 s0"], "tow_real", w)}
    for lab in P.HEAD:
        tt = td[td.model == lab]
        out[lab] = _std(tt, "tow_model", w)
        out[lab + "|draws"] = [_std(tt, f"tow_d{j}", w) for j in range(P.N_DRAW)]
    return out, w




