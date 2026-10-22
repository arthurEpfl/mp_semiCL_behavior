"""Final_report_missingPieces: the analyses the thesis still lacked, one function family per notebook section.

Every heavy computation runs once (``python missing_pieces.py run <section>``) and is cached in Model/missing_pieces/;
the notebook (Model/final_report_missingpieces.py -> Final_report_missingPieces.ipynb) only reads the caches and draws.

Headline models: M6 = 26e (seed 0) / 30d (seed 101), the analysis model; F = 14f / 32b, the final model. Seeds are
separate cross-validation splits and are never pooled.

    1  realism_refs / likelihood_baselines   reference generators on the matched C2ST windowing; marginal / HMM likelihood
    2  properties / importances / nearest     models vs the real-data properties of Ch. 3; what the classifier uses; NRN
    3  response_maps                         the teacher-forced turn law vs real turns, binned; a compact parameter summary
    4  (multi_stim / multi_fish runners)     the extrapolations replicated on M6 (both seeds) and F seed 101
    5  stream_noise                          rollout-stream SD of the profile readouts (also the unfiltered profiles, 10)
    6  provenance                            the bout table regenerated with the current segmentation code, compared
    7  paired_ci                             paired-by-fish 95 % CIs of the key likelihood differences
    8  centroid_choice                       centroid vs nearest dot vs 1/r-weighted centroid as the turn target
    9  neighbour_maps                        multi-fish: where neighbours sit around a fish
    11 loco_condition_lik                    the leave-one-condition-out model on the held-out condition
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from twin import paths

OUT = paths.MISSING

PX_MM = 3.60                      # px per mm (Ch. 3); the profile code uses 3.67 (see section 10)
REJ = {"d0": 2.0, "d1": 10.0, "max_draws": 20}
ARM_M6 = "S4x_adt_z2_cdt_tr_rc_al_w3ft"
ARM_F = "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"
HEAD = {"M6 s0": ("M26e", ARM_M6, "26e"), "M6 s101": ("M30d", ARM_M6, "30d"),
        "F s0": ("M14f", ARM_F, "14f"), "F s101": ("M32b", ARM_F, "32b")}
W = 60
WIN_REPS = 8
TOWARD_BEARING = (10.0, 170.0)    # a bout has a side only when the target is not dead ahead / behind


def _threads1():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    with contextlib.suppress(Exception):
        import torch
        torch.set_num_threads(1)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.pkl", "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(name):
    with open(OUT / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def have(name):
    return (OUT / f"{name}.pkl").exists()


_REC = {}


def records():
    if "r" not in _REC:
        from twin import evaluate as E
        from twin import fit_MLP_border as MB
        with contextlib.redirect_stdout(io.StringIO()):
            _REC["r"] = MB.build_border_records(E.load_bouts())
    return _REC["r"]


def _ckpt(d):
    p = Path(d)
    return p if p.is_absolute() else paths.CKPTS / p


def manifest(d):
    return json.loads((_ckpt(d) / "manifest_model1.json").read_text())


def folds_of(d, arm):
    """[(fold file, held-out fish list)] of an arm, from its manifest (the exact CV split of that model)."""
    return [(e["file"], [int(f) for f in e["heldout"]]) for e in manifest(d) if e["arm"] == arm]


def load_model(d, file, rej=True):
    from twin import fit_MLP_mix as MX
    with contextlib.redirect_stdout(io.StringIO()):
        MX._numpy2_unpickle_compat()
        m = MX.load_mix(str(_ckpt(d) / file))
    if rej:
        m._cfg_feat = {**m._cfg_feat, "wall_reject": dict(REJ)}
    return m


def summary(key):
    p = paths.EVAL / f"summary_M{key}.json"
    return json.loads(p.read_text()) if p.exists() else None




def _boot_ci(v, n=4000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v))).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))




def _bout_arrays(rec):
    from twin import generative_metrics as GM
    dd = getattr(rec, "dot_dist_px", None)
    return dict(dth=np.asarray(rec.dtheta, float), sym=np.asarray(rec.symbol, int),
                dist=np.asarray(rec.distance_px, float), dt=np.asarray(rec.dt_s, float),
                dd=(np.full(rec.n, np.nan) if dd is None else np.asarray(dd, float)),
                rad=np.asarray(GM._radial_of(rec, rec.x_px, rec.y_px), float))




def _win(A, runs):
    from twin import generative_metrics as GM
    return GM.window_features_traj(A["dth"], A["sym"], A["dist"], A["dt"], A["dd"], W=W, runs=runs, radial=A["rad"])


def real_floor(seeds=5):
    """The matched classifier's floor: each fish's real matched windows split into random halves (``seeds`` splits), on
    all features, without the radial ones, and on the three heads only. Cached as 'refs'."""
    from twin import generative_metrics as GM
    _threads1()
    recs = records()
    res = {f: _win(_bout_arrays(recs[f]), getattr(recs[f], "run_id", None)) for f in sorted(recs)}
    names = GM.traj_names_for_dim(next(iter(res.values())).shape[1])
    sets = {"all features": list(range(len(names))),
            "no radial": [i for i, n in enumerate(names) if not n.startswith("rad")],
            "three heads": [i for i, n in enumerate(names) if not n.startswith(("rad", "dotdist"))]}
    floor = dict(generator="real vs real (random halves of each fish)")
    for lab, cols in sets.items():
        v = [GM.c2st_negative_control({f: (res[f][:, cols], res[f][:, cols]) for f in res}, seed=s)
             for s in range(seeds)]
        floor[lab] = float(np.mean(v))
        floor[lab + " se"] = float(np.std(v, ddof=1) / np.sqrt(len(v)))
    tab = pd.DataFrame([floor]).set_index("generator")
    save("refs", dict(table=tab, names=names))
    return tab




# =========================================================================== #
# 2. Models against the real-data properties; what the classifier uses; nearest real neighbour
# =========================================================================== #
WALL_BINS = (0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)
RAD_BINS = np.linspace(0, 1, 21)




def nearest(jobs=5):
    """Nearest-real-neighbour distance (unseen_fish_yardstick) and the feature-drop AUCs, per headline model."""
    from twin import unseen_fish_yardstick as Y
    out = {}
    for lab, (d, arm, key) in HEAD.items():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            per, nn_samp = Y.nearest_real(str(_ckpt(d)), arm, rej=REJ, jobs=jobs, plot=False)
            drop = Y.feature_drop_table(str(_ckpt(d)), [arm], rej=REJ, jobs=jobs)
        out[lab] = dict(per=per, nn_sampled=nn_samp, drop=drop, text=buf.getvalue())
    save("nearest", out)
    return out


# =========================================================================== #
# 3. The learned rule: teacher-forced turn law vs real turns
# =========================================================================== #
BEAR_BINS = (10, 45, 90, 135, 170)
RANGE_BINS_MM = (0, 10, 20, 35, 60, 120)




N_DRAW = 20                        # independent single-draw replicates of the model's turns per bout


def _law_draw_job(job):
    lab, d, file, fids = job
    _threads1()
    m = load_model(d, file)
    recs = records()
    parts = []
    for f in fids:
        rec = recs[f]
        rng = np.random.default_rng([11, list(HEAD).index(lab), int(f)])
        e_dth = np.asarray(m.conditional_turn(rec)[0], float)
        dr = m.conditional_turn_draws(rec, rng, N_DRAW)
        phi = np.asarray(getattr(rec, "stim_angle_travel", rec.stim_angle), float)
        fo = np.asarray(rec.frame_onset, float); pos = fo % 3300
        t_blk = np.where(pos >= 1500, pos - 1500, pos) / 30.0
        df = pd.DataFrame(dict(model=lab, fish=f, e_dth=e_dth, dth=np.asarray(rec.dtheta, float), phi=phi,
                               rng_mm=np.asarray(rec.dot_dist_px, float) / PX_MM,
                               n=np.nan_to_num(np.asarray(rec.n_dots, float)).astype(int),
                               moving=np.asarray(rec.moving, float) > 0, t_blk=t_blk))
        for j in range(N_DRAW):
            df[f"draw{j}"] = dr[:, j]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def response_draws(jobs=5):
    """As ``response_maps`` (same bouts, same e_dth), plus N_DRAW turns per bout drawn from the model's
    teacher-forced angle mixture, so the model's per-fish means carry the scatter of single turns as the real
    ones do. Cached as 'laws_draw'."""
    jl = [(lab, d, file, ho) for lab, (d, arm, key) in HEAD.items() for file, ho in folds_of(d, arm)]
    with ProcessPoolExecutor(jobs) as ex:
        tab = pd.concat(list(ex.map(_law_draw_job, jl)), ignore_index=True)
    save("laws_draw", tab)
    return tab


def _toward_cols(t):
    t = t[(t.n > 0) & np.isfinite(t.phi) & np.isfinite(t.rng_mm)].copy()
    ab = np.degrees(np.abs(t.phi))
    t = t[(ab > BEAR_BINS[0]) & (ab < BEAR_BINS[-1])]
    sg = np.sign(t.phi)
    t["tow_real"] = t.dth * sg
    t["tow_model"] = t.e_dth * sg
    t["bear"] = pd.cut(np.degrees(np.abs(t.phi)), BEAR_BINS)
    t["range"] = pd.cut(t.rng_mm, RANGE_BINS_MM)
    t["phase"] = np.where(t.t_blk < 4.0, "0-4 s", np.where(t.t_blk >= 10.0, ">= 10 s", "4-10 s"))
    return t




STREAM_COLS = ("dip", "lat", "plateau", "d1_plateau", "grading", "mv_err_sem", "st_err_sem", "static_drift")






# =========================================================================== #


def main(argv):
    """python -m twin.missing_pieces refs | nearest | laws_draw  (MP_JOBS sets the workers)"""
    _threads1()
    what = argv[1] if len(argv) > 1 else ""
    jobs = int(os.environ.get("MP_JOBS", "5"))
    t0 = time.time()
    if what == "refs":
        print(real_floor().round(3).to_string())
    elif what == "nearest":
        nearest(jobs=jobs)
    elif what == "laws_draw":
        response_draws(jobs=jobs)
    else:
        print(main.__doc__)
        return
    print(f"[{what}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv)
