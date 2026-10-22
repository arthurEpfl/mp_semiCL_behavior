"""The model tables of the thesis, from the units' evaluation summaries (``unit_eval.evaluate``).

Every distance is converted from the profile code's nominal 3.67 px/mm to the physical 3.60 px/mm of the Data
chapter (``MM``). Each table lists seed 0 and seed 101 (never pooled).
"""
import glob
import json
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache

import numpy as np
import pandas as pd

from twin import paths, registry

MM = 3.67 / 3.60
MAIN = ["M1", "M2", "M3", "M4", "M5", "M5+R", "M5+T", "M6", "D1", "D2", "D3", "D4", "F"]
ABLATIONS = ["M6", "M6-stim", "M6-count", "M6-range", "M6-motion", "M6-turn4", "M6-turn3", "M6-turn1", "M6-turn0",
             "M6-kin5", "M6-kin4", "M6-kin2", "M6-kin1", "M6-kin0", "M6-block clock", "M6-run clock", "M6-x3",
             "M6-z"]
# the real-fish AUCs of the model tables: the floor of the matched classifier (each fish's real windows split into
# random halves, missing_pieces 'refs'), standard reading and without the radial features
REAL_AUC = {"auc_matched": 0.499, "auc_std": 0.516, "auc_matched_norad": 0.502}


@lru_cache(maxsize=1)
def real_readouts():
    """The real fish's profile readouts (dip, latency, plateaus, grading), computed from the bout table."""
    from twin.model_figures import _curves, metrics_from_curves, profile_summary, real_dot_profile
    pr, mr, sr = _curves(real_dot_profile())
    s = profile_summary(metrics_from_curves(pr, mr, sr, pr, mr))
    return {c: float(s[f"{c}_real"]) for c in ("dip", "lat", "plateau", "d1_plateau", "grading")}


def fmt(v, nd, half_up=True):
    """``v`` to ``nd`` decimals, halves rounded up, or with float formatting (``half_up=False``): the thesis tables
    were written by different scripts, each with one of the two."""
    if v is None or not np.isfinite(v):
        return "--"
    if not half_up:
        return f"{float(v):.{nd}f}"
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def summary(key):
    """The headline numbers of a unit (None if it has not been evaluated)."""
    p = paths.EVAL / f"summary_M{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


PROFILE_COLS = [("AUC match", "auc_matched", 1.0, 3), ("AUC std", "auc_std", 1.0, 3),
                ("AUC no-rad", "auc_matched_norad", 1.0, 3), ("dip (mm)", "dip", MM, 2), ("latency (s)", "lat", 1.0, 1),
                ("grading (mm)", "grading", MM, 2), ("plateau 2-8 (mm)", "plateau", MM, 2),
                ("plateau 1 (mm)", "d1_plateau", MM, 2), ("error moving", "mv_err_sem", 1.0, 2),
                ("error static", "st_err_sem", 1.0, 2)]


def _row(s, half_up=True):
    out = {}
    for lab, c, k, nd in PROFILE_COLS:
        v = None if s is None else s.get(c)
        out[lab] = fmt(v * k, nd, half_up) if v is not None else "--"
    return out


def profile_table(names, half_up=True):
    """AUC, dip, grading, plateaus and profile errors, one row per model and seed (tab:storyline, tab:abl-all)."""
    rows = {(n, f"s{s}"): _row(summary(registry.BY_NAME[(n, s)]), half_up) for n in names for s in (0, 101)}
    rows[("Real", "")] = {**_row({**REAL_AUC, **real_readouts()}, half_up), "error moving": "--",
                          "error static": "--"}
    return pd.DataFrame(rows).T


def individuality_table(names):
    """Spread of simulated 60-bout window means relative to real, between and within fish (tab:realism)."""
    cols = [(f"{w}_{c}", f"{'between' if w == 'bf' else 'within'} {lab}") for w in ("bf", "wf")
            for c, lab in (("turn", "turn"), ("dist", "disp."), ("dt", "interval"), ("rad", "radial"))]
    rows = {}
    for n in names:
        for s in (0, 101):
            S = summary(registry.BY_NAME[(n, s)]) or {}
            rows[(n, f"s{s}")] = {lab: fmt(S.get(c, np.nan), 2, half_up=False) for c, lab in cols}
    return pd.DataFrame(rows).T


def paired_table(rows, cols, half_up=True):
    """One row per model, each column 'seed 0 | seed 101'. ``cols`` = [(label, fn(summary, key) -> value, digits)]."""
    out = {}
    for n in rows:
        r = {}
        for lab, fn, nd in cols:
            v = [fn(summary(registry.BY_NAME[(n, s)]) or {}, registry.BY_NAME[(n, s)]) for s in (0, 101)]
            r[lab] = f"{fmt(v[0], nd, half_up)} | {fmt(v[1], nd, half_up)}"
        out[n] = r
    return pd.DataFrame(out).T


def window_corr(key, ch):
    """Lag-1 correlation of successive 60-bout window means of one channel (dt, dist, rad), simulated, mean over the
    held-out fish (the unit's panel table: real value + stored gap)."""
    f = glob.glob(str(paths.CKPTS / f"M{key}" / "panelB_*.csv"))
    d = pd.read_csv(f[0])
    return float((d[f"real_wacf_{ch}"] + d[f"wacf_{ch}_gap"]).mean())


def _get(c, k=1.0):
    return lambda s, key: np.nan if s.get(c) is None else s[c] * k


NEGRES_NET = (["M6", "M6, 3 layers", "M6, turn size", "D4", "D4, 3 layers", "D4, turn size", "F", "F, 3 layers",
               "F, turn size"],
              [("AUC", _get("auc_matched"), 3), ("AUC no-rad", _get("auc_matched_norad"), 3),
               ("dip", _get("dip", MM), 2), ("grading", _get("grading", MM), 2),
               ("moving error", _get("mv_err_sem"), 2), ("turn spread", _get("bf_turn"), 2)])
NEGRES_DRIFT = (["D4", "D4, tau 200", "D4, tau 200 + turn size", "D4, strength 0", "D4, tau 70, half",
                 "D4, tau 70, full", "D4, tau 200, half", "D4, tau 200, full", "F", "F, tau 200",
                 "F, tau 200 + turn size", "F on D4 at strength 0"],
                [("AUC", _get("auc_matched"), 3), ("AUC no-rad", _get("auc_matched_norad"), 3),
                 ("dip", _get("dip", MM), 2), ("moving error", _get("mv_err_sem"), 2),
                 ("static error", _get("st_err_sem"), 2),
                 ("window corr. interval", lambda s, k: window_corr(k, "dt"), 2),
                 ("window corr. displ.", lambda s, k: window_corr(k, "dist"), 2),
                 ("within displ.", _get("wf_dist"), 2)])
NEGRES_OTHER = (["M6", "M6, jump phase", "M6, floor 2 px", "D4", "D4, jump phase", "D4, floor 2 px", "F",
                 "F, jump phase", "F, floor 2 px"],
                [("lik. joint", _get("lik_joint"), 3), ("lik. interval", _get("lik_interval"), 3),
                 ("AUC", _get("auc_matched"), 3), ("dip", _get("dip", MM), 2),
                 ("moving error", _get("mv_err_sem"), 2)])
