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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "Model"
OUT = Path(os.environ.get("MISSING_PIECES_OUT", MODEL / "missing_pieces"))
for _p in (str(HERE), str(MODEL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PX_MM = 3.60                      # px per mm (Ch. 3); the profile code uses 3.67 (see section 10)
REJ = {"d0": 2.0, "d1": 10.0, "max_draws": 20}
ARM_M6 = "S4x_adt_z2_cdt_tr_rc_al_w3ft"
ARM_F = "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"
HEAD = {"M6 s0": ("report_ckpts/M26e", ARM_M6, "26e"), "M6 s101": ("report_ckpts/M30d", ARM_M6, "30d"),
        "F s0": ("report_ckpts/M14f", ARM_F, "14f"), "F s101": ("report_ckpts/M32b", ARM_F, "32b")}
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
        import evaluate as E
        import fit_MLP_border as MB
        with contextlib.redirect_stdout(io.StringIO()):
            _REC["r"] = MB.build_border_records(E.load_bouts())
    return _REC["r"]


def _ckpt(d):
    p = Path(d)
    return p if p.is_absolute() else MODEL / p


def manifest(d):
    return json.loads((_ckpt(d) / "manifest_model1.json").read_text())


def folds_of(d, arm):
    """[(fold file, held-out fish list)] of an arm, from its manifest (the exact CV split of that model)."""
    return [(e["file"], [int(f) for f in e["heldout"]]) for e in manifest(d) if e["arm"] == arm]


def load_model(d, file, rej=True):
    import fit_MLP_mix as MX
    with contextlib.redirect_stdout(io.StringIO()):
        MX._numpy2_unpickle_compat()
        m = MX.load_mix(str(_ckpt(d) / file))
    if rej:
        m._cfg_feat = {**m._cfg_feat, "wall_reject": dict(REJ)}
    return m


def summary(key):
    p = MODEL / "report_outputs" / f"summary_M{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def _boot_ci(v, n=4000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v))).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


# =========================================================================== #
# 1. Realism scale: reference generators (matched C2ST) and likelihood baselines
# =========================================================================== #
REF_KINDS = ("shuffled", "block3", "block10", "block30", "pooled", "hmm")


def _bout_arrays(rec):
    import generative_metrics as GM
    dd = getattr(rec, "dot_dist_px", None)
    return dict(dth=np.asarray(rec.dtheta, float), sym=np.asarray(rec.symbol, int),
                dist=np.asarray(rec.distance_px, float), dt=np.asarray(rec.dt_s, float),
                dd=(np.full(rec.n, np.nan) if dd is None else np.asarray(dd, float)),
                rad=np.asarray(GM._radial_of(rec, rec.x_px, rec.y_px), float))


def _take(A, idx):
    return {k: v[idx] for k, v in A.items()}


def _win(A, runs):
    import generative_metrics as GM
    return GM.window_features_traj(A["dth"], A["sym"], A["dist"], A["dt"], A["dd"], W=W, runs=runs, radial=A["rad"])


def _ref_job_real(job):
    """One held-out fish: its real matched windows and, per reference kind, WIN_REPS generated window sets cut at the
    same bout indices (the matched windowing). References re-order REAL bouts, each bout keeping all its attributes
    (turn, step, interval, distance to the dots, radial position): 'shuffled' = the fish's own bouts permuted;
    'blockL' = the fish's own bouts in random blocks of L consecutive bouts; 'pooled' = bouts drawn i.i.d. from the
    other 14 fish; 'hmm' = turns from a 3-state HMM fitted on the training fish of this fish's fold (seed 0 split), with
    step, interval, dot distance and radial position from a random bout of the other 14 fish."""
    fid, hmm_params, reps, seed = job
    _threads1()
    import fit_hmm as H
    recs = records()
    rec = recs[fid]
    A = _bout_arrays(rec)
    runs = getattr(rec, "run_id", None)
    n = rec.n
    real = _win(A, runs)
    others = [f for f in sorted(recs) if f != fid]
    pool = {k: np.concatenate([_bout_arrays(recs[f])[k] for f in others]) for k in A}
    out = {"real": real}
    for kind in REF_KINDS:
        gens = []
        for rep in range(reps):
            rng = np.random.default_rng([seed, fid, rep, REF_KINDS.index(kind)])
            if kind == "shuffled":
                G = _take(A, rng.permutation(n))
            elif kind.startswith("block"):
                L = int(kind[5:])
                starts = rng.integers(0, max(n - L, 0) + 1, size=int(np.ceil(n / L)))
                idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
                G = _take(A, idx)
            elif kind == "pooled":
                G = _take(pool, rng.integers(0, len(pool["dth"]), size=n))
            else:                                                    # hmm
                gen = H.generate_bouts(hmm_params, n, pool["dt"], pool["dist"], rng)
                G = _take(pool, rng.integers(0, len(pool["dth"]), size=n))
                G["dth"] = np.asarray(gen["angle_deg"], float)
                import evaluate as E
                G["sym"] = np.asarray(E.symbols_from_dtheta(G["dth"]), int)
            gens.append(_win(G, runs))
        out[kind] = gens
    return fid, out


def _fit_hmm_fold(job):
    k, train = job
    _threads1()
    import fit_hmm as H
    recs = records()
    with contextlib.redirect_stdout(io.StringIO()):
        m = H.fit_hmm_g3_sym([recs[f].dtheta for f in train], n_restarts=2, max_iter=200, seed=0, verbose=False)
    return k, m


def realism_refs(reps=WIN_REPS, seed=0, jobs=5):
    """Matched-windowing C2ST AUC of each reference generator (all features / no radial / three heads), the real-vs-real
    floor on the same feature sets, and the models' matched AUC from their saved windows. Cached as 'refs'."""
    import generative_metrics as GM
    _threads1()
    folds = folds_of(*HEAD["M6 s0"][:2])
    fish_all = sorted(records())
    with ProcessPoolExecutor(min(jobs, len(folds))) as ex:
        hmms = dict(ex.map(_fit_hmm_fold, [(k, [f for f in fish_all if f not in ho]) for k, (_, ho) in enumerate(folds)]))
    hmm_of = {f: hmms[k] for k, (_, ho) in enumerate(folds) for f in ho}
    with ProcessPoolExecutor(jobs) as ex:
        res = dict(ex.map(_ref_job_real, [(f, hmm_of[f], reps, seed) for f in fish_all]))
    dim = next(iter(res.values()))["real"].shape[1]
    names = GM.traj_names_for_dim(dim)
    sets = {"all features": list(range(dim)),
            "no radial": [i for i, n in enumerate(names) if not n.startswith("rad")],
            "three heads": [i for i, n in enumerate(names) if not n.startswith(("rad", "dotdist"))]}
    rows = []
    for kind in REF_KINDS:
        r = dict(generator=kind)
        for lab, cols in sets.items():
            wins = [{f: (res[f]["real"][:, cols], res[f][kind][j][:, cols]) for f in res
                     if len(res[f]["real"]) and len(res[f][kind][j])} for j in range(reps)]
            mean, sd, per = GM.c2st_auc_replicates(wins, seed=seed)
            r[lab] = mean
            r[lab + " se"] = sd / np.sqrt(max(len(per), 1))
        rows.append(r)
    floor = dict(generator="real vs real (random halves of each fish)")
    for lab, cols in sets.items():
        v = [GM.c2st_negative_control({f: (res[f]["real"][:, cols], res[f]["real"][:, cols]) for f in res}, seed=s)
             for s in range(5)]
        floor[lab] = float(np.mean(v))
        floor[lab + " se"] = float(np.std(v, ddof=1) / np.sqrt(len(v)))
    rows.append(floor)
    # the models' matched readings on the same three feature sets, from the windows their notebook units cached
    for lab, (d, arm, key) in HEAD.items():
        z = np.load(_ckpt(d) / f"protocol_windows_{arm}.npz", allow_pickle=True)["res"].item()
        r = dict(generator=f"{lab} (model)")
        for sl, cols in sets.items():
            wins = [{f: (z[f]["all"][0][:, cols], z[f]["all"][1][j][:, cols]) for f in z
                     if len(z[f]["all"][0]) and len(z[f]["all"][1][j])} for j in range(len(next(iter(z.values()))["all"][1]))]
            mean, sd, per = GM.c2st_auc_replicates(wins, seed=seed)
            r[sl] = mean
            r[sl + " se"] = sd / np.sqrt(max(len(per), 1))
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("generator")
    save("refs", dict(table=tab, names=names))
    return tab


def _baseline_job(job):
    """Fit the classical baselines on one fold's training fish and score its held-out fish (bits per bout)."""
    seed, k, train, held = job
    _threads1()
    import evaluate as E
    from sklearn.mixture import GaussianMixture
    recs = records()
    tr = [recs[f] for f in train]
    ctx = E.make_context(tr, {})
    out = []
    models = {}
    with contextlib.redirect_stdout(io.StringIO()):
        models["marginal (turn)"] = E.MarginalDensity().fit(tr, ctx)
        models["HMM, 3 states (turn)"] = E.SpontaneousHMM(seed=seed).fit(tr, ctx)
        models["HMM + stimulus (turn)"] = E.SimpleStimHMM(seed=seed).fit(tr, ctx)
    # i.i.d. pooled marginals of all three channels (turn in deg, step in px, interval in s), Gaussian mixtures
    th = np.concatenate([r.dtheta for r in tr])
    _d = np.concatenate([r.distance_px for r in tr]); _t = np.concatenate([r.dt_s for r in tr])
    ld = np.log(_d[np.isfinite(_d) & (_d > 0)])
    lt = np.log(_t[np.isfinite(_t) & (_t > 0)])
    gm = {c: GaussianMixture(n, random_state=0).fit(np.asarray(v)[np.isfinite(v)].reshape(-1, 1))
          for c, v, n in (("turn", th, 12), ("dist", ld, 8), ("dt", lt, 8))}
    for f in held:
        r = recs[f]
        row = dict(seed=seed, fold=k, fish=f, n=r.n)
        for lab, m in models.items():
            row[lab] = float(-np.nanmean(m.score(r)["logdens"]) / np.log(2))
        lp_t = gm["turn"].score_samples(np.asarray(r.dtheta, float).reshape(-1, 1))
        d = np.asarray(r.distance_px, float); t = np.asarray(r.dt_s, float)
        okd = np.isfinite(d) & (d > 0); okt = np.isfinite(t) & (t > 0)              # the MLP score's own masks
        lp_d = gm["dist"].score_samples(np.log(d[okd]).reshape(-1, 1)) - np.log(d[okd])     # density in px
        lp_t2 = gm["dt"].score_samples(np.log(t[okt]).reshape(-1, 1)) - np.log(t[okt])       # density in s
        row["pooled marginal (turn)"] = float(-np.mean(lp_t) / np.log(2))
        row["pooled marginal (step)"] = float(-np.mean(lp_d) / np.log(2))
        row["pooled marginal (interval)"] = float(-np.mean(lp_t2) / np.log(2))
        row["pooled marginal (joint)"] = row["pooled marginal (turn)"] + row["pooled marginal (step)"] + row["pooled marginal (interval)"]
        out.append(row)
    return out


def likelihood_baselines(jobs=5):
    """Held-out bits per bout of the classical baselines on EXACTLY the models' folds (both seeds), next to M6 and F
    (their likA tables, per fish). Cached as 'baselines'."""
    _threads1()
    fish = sorted(records())
    jl = []
    for seed, lab in ((0, "M6 s0"), (101, "M6 s101")):
        for k, (_, ho) in enumerate(folds_of(*HEAD[lab][:2])):
            jl.append((seed, k, [f for f in fish if f not in ho], ho))
    with ProcessPoolExecutor(min(jobs, len(jl))) as ex:
        rows = [r for part in ex.map(_baseline_job, jl) for r in part]
    base = pd.DataFrame(rows)
    mods = []
    for lab, (d, arm, key) in HEAD.items():
        l = pd.read_csv(_ckpt(d) / "likA_model1.csv").drop_duplicates(["arm", "fish"])
        l = l[l.arm == arm]
        for r in l.itertuples():
            mods.append(dict(model=lab, seed=0 if "s0" in lab else 101, fish=int(r.fish), turn=r.bits_per_bout,
                             step=r.bits_per_bout_dist, interval=r.bits_per_bout_dt, joint=r.bits_per_bout_joint))
    save("baselines", dict(base=base, models=pd.DataFrame(mods)))
    return base


def baselines_table():
    b = load("baselines")
    base, mods = b["base"], b["models"]
    rows = []
    for seed in (0, 101):
        bs = base[base.seed == seed].set_index("fish")
        for c in ("pooled marginal (turn)", "marginal (turn)", "HMM, 3 states (turn)", "HMM + stimulus (turn)"):
            rows.append(dict(seed=seed, model=c, turn=bs[c].mean()))
        rows.append(dict(seed=seed, model="pooled marginal (all three heads)", turn=bs["pooled marginal (turn)"].mean(),
                         step=bs["pooled marginal (step)"].mean(), interval=bs["pooled marginal (interval)"].mean(),
                         joint=bs["pooled marginal (joint)"].mean()))
        for lab in [l for l in HEAD if (("s0" in l) == (seed == 0))]:
            mm = mods[mods.model == lab].set_index("fish")
            dj = (bs["pooled marginal (joint)"] - mm.joint).values
            dt_ = (bs["HMM + stimulus (turn)"] - mm.turn).values
            cj, ct = _boot_ci(dj), _boot_ci(dt_)
            rows.append(dict(seed=seed, model=lab, turn=mm.turn.mean(), step=mm.step.mean(),
                             interval=mm.interval.mean(), joint=mm.joint.mean(),
                             saved_vs_pooled_joint=f"{dj.mean():.3f} [{cj[0]:.3f}, {cj[1]:.3f}]",
                             saved_vs_hmm_turn=f"{dt_.mean():.3f} [{ct[0]:.3f}, {ct[1]:.3f}]"))
    return pd.DataFrame(rows).set_index(["seed", "model"]).fillna("")


# =========================================================================== #
# 2. Models against the real-data properties; what the classifier uses; nearest real neighbour
# =========================================================================== #
WALL_BINS = (0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)
RAD_BINS = np.linspace(0, 1, 21)


def _acf_runs(v, runs, lags=(1, 2, 3, 4, 5)):
    v = np.asarray(v, float)
    runs = np.asarray(runs) if runs is not None else np.zeros(len(v), int)
    out = {}
    for L in lags:
        ok = (runs[L:] == runs[:-L]) & np.isfinite(v[L:]) & np.isfinite(v[:-L])
        a, b = v[L:][ok], v[:-L][ok]
        out[L] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else np.nan
    return out


def _props(dth, dist_px, dt, x, y, dish, runs):
    """The Ch. 3 properties of one trajectory (per-bout arrays; x, y = pre-bout positions)."""
    cx, cy, R = dish
    dth = np.asarray(dth, float); d = np.asarray(dist_px, float) / PX_MM; t = np.asarray(dt, float)
    x = np.asarray(x, float); y = np.asarray(y, float)
    a = np.abs(dth)
    p = dict(turn_median=np.median(a), turn_q90=np.quantile(a, .9), straight=np.mean(a < 3.0),
             step_median=np.median(d), step_q90=np.quantile(d, .9), interval_median=np.median(t),
             interval_q90=np.quantile(t, .9))
    for nm, v in (("turn", a), ("step", np.log(d)), ("interval", np.log(t))):
        for L, c in _acf_runs(v, runs).items():
            p[f"acf{L}_{nm}"] = c
    r = np.hypot(x - cx, y - cy) / R
    hd = np.r_[np.nan, np.arctan2(np.diff(y), np.diff(x))]                 # travel heading of the previous bout
    if runs is not None:                                                    # a run's first bout follows a pause
        rr = np.asarray(runs)
        hd[np.r_[True, rr[1:] != rr[:-1]]] = np.nan
    psi = _wrap(np.arctan2(cy - y, cx - x) - hd)                            # bearing of the dish centre
    side = (np.degrees(np.abs(psi)) > TOWARD_BEARING[0]) & (np.degrees(np.abs(psi)) < TOWARD_BEARING[1]) & (a > 3)
    tow = np.sign(dth) == np.sign(psi)
    for lo, hi in zip(WALL_BINS[:-1], WALL_BINS[1:]):
        s = side & (r >= lo) & (r < hi)
        p[f"wall_{lo:.1f}-{hi:.1f}"] = float(tow[s].mean()) if s.sum() >= 10 else np.nan
    h, _ = np.histogram(r, RAD_BINS)
    p["radial_hist"] = h / max(h.sum(), 1)
    p["radial_median"] = np.median(r)
    return p


def _prop_job(job):
    lab, d, file, fids, reps, seed = job
    _threads1()
    import generative_metrics as GM
    m = load_model(d, file)
    recs = records()
    rows = []
    for f in fids:
        rec = recs[f]
        dish = (rec.dish_cx, rec.dish_cy, rec.dish_R)
        runs = getattr(rec, "run_id", None)
        for rep in range(reps):
            g = m.sample(rec, np.random.default_rng([seed, f, rep]), **GM._stim_kw(m, rec, None))
            n = min(len(g["dtheta"]), rec.n)
            rows.append(dict(model=lab, fish=f, rep=rep,
                             **_props(g["dtheta"][:n], g["distance_px"][:n], g["dt_s"][:n], g["x_px"][:n], g["y_px"][:n],
                                      dish, None if runs is None else np.asarray(runs)[:n])))
    return rows


def properties(reps=2, seed=0, jobs=5):
    """Per fish: the real properties and those of 2 record-anchored rollouts of each headline model (each fish by the
    fold model that held it out). Cached as 'properties'."""
    recs = records()
    real = []
    for f, rec in sorted(recs.items()):
        real.append(dict(model="real", fish=f, rep=0, **_props(rec.dtheta, rec.distance_px, rec.dt_s, rec.x_px, rec.y_px,
                                                                (rec.dish_cx, rec.dish_cy, rec.dish_R),
                                                                getattr(rec, "run_id", None))))
    jl = [(lab, d, file, ho, reps, seed) for lab, (d, arm, key) in HEAD.items() for file, ho in folds_of(d, arm)]
    with ProcessPoolExecutor(jobs) as ex:
        rows = [r for part in ex.map(_prop_job, jl) for r in part]
    tab = pd.DataFrame(real + rows)
    save("properties", tab)
    return tab


def properties_table():
    t = load("properties")
    scal = [c for c in t.columns if c not in ("model", "fish", "rep", "radial_hist")]
    per = t.groupby(["model", "fish"])[scal].mean()
    rows = []
    for c in scal:
        r = dict(property=c, real=per.loc["real", c].mean())
        for lab in HEAD:
            diff = per.loc[lab, c] - per.loc["real", c]
            r[lab] = per.loc[lab, c].mean()
            ci = _boot_ci(diff.values)
            r[f"{lab} - real [95% CI]"] = f"{diff.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        rows.append(r)
    return pd.DataFrame(rows).set_index("property")


def importances(seed=0):
    """Permutation importance of each window feature for the matched classifier of each headline model (one
    replicate of its cached matched windows). Cached as 'importances'."""
    import generative_metrics as GM
    out = {}
    for lab, (d, arm, key) in HEAD.items():
        z = np.load(_ckpt(d) / f"protocol_windows_{arm}.npz", allow_pickle=True)["res"].item()
        dim = next(iter(z.values()))["all"][0].shape[1]
        names = GM.traj_names_for_dim(dim)
        wins = {f: (z[f]["all"][0], z[f]["all"][1][0]) for f in z if len(z[f]["all"][0])}
        out[lab] = GM.c2st_importances(wins, feature_names=names, seed=seed)
    tab = pd.DataFrame(out)
    save("importances", tab)
    return tab


def nearest(jobs=5):
    """Nearest-real-neighbour distance (unseen_fish_yardstick) and the feature-drop AUCs, per headline model."""
    import unseen_fish_yardstick as Y
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


def _law_job(job):
    lab, d, file, fids = job
    _threads1()
    m = load_model(d, file)
    recs = records()
    parts = []
    for f in fids:
        rec = recs[f]
        e_dth, e_abs, _ = m.conditional_turn(rec)
        phi = np.asarray(getattr(rec, "stim_angle_travel", rec.stim_angle), float)
        fo = np.asarray(rec.frame_onset, float); pos = fo % 3300                   # exact programme grid: 50 s + 60 s
        t_blk = np.where(pos >= 1500, pos - 1500, pos) / 30.0                     # = bouts_clean t_blk (not first bout)
        parts.append(pd.DataFrame(dict(model=lab, fish=f, e_dth=np.asarray(e_dth, float), e_abs=np.asarray(e_abs, float),
                                       dth=np.asarray(rec.dtheta, float), phi=phi,
                                       rng_mm=np.asarray(rec.dot_dist_px, float) / PX_MM,
                                       n=np.nan_to_num(np.asarray(rec.n_dots, float)).astype(int),
                                       moving=np.asarray(rec.moving, float) > 0, t_blk=t_blk)))
    return pd.concat(parts, ignore_index=True)


def response_maps(jobs=5):
    """Per bout of every held-out fish: the real turn and the model's teacher-forced expected turn (population latent),
    with the dots' bearing, range, count, motion and block time. Cached as 'laws'."""
    jl = [(lab, d, file, ho) for lab, (d, arm, key) in HEAD.items() for file, ho in folds_of(d, arm)]
    with ProcessPoolExecutor(jobs) as ex:
        tab = pd.concat(list(ex.map(_law_job, jl)), ignore_index=True)
    save("laws", tab)
    return tab


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


def law_maps(lab, moving=True, n=(2, 8), phase=">= 10 s"):
    """(real map, model map) of the mean toward-turn (deg), |bearing| x range, per fish then across fish."""
    t = _toward_cols(load("laws"))
    t = t[(t.model == lab) & (t.moving == moving) & (t.phase == phase) & (t.n.isin(list(n)))]
    def mp(col):
        g = t.groupby(["fish", "bear", "range"], observed=True)[col].mean().groupby(["bear", "range"], observed=True).mean()
        return g.unstack("range")
    return mp("tow_real"), mp("tow_model")


def law_summary():
    """A compact parameter summary of the turn law, real vs each model, on matched geometry (|bearing| 45-135 deg,
    range 10-35 mm unless stated): onset gain (toward-turn, first 4 s, moving), steady gain (>= 10 s, moving), the
    range at which the steady pull peaks, the count step (steady gain at 2-8 dots minus at 1 dot) and the static gain
    (>= 10 s, static). Means over fish of per-fish means (deg)."""
    t = _toward_cols(load("laws"))
    geo = (np.degrees(np.abs(t.phi)) >= 45) & (np.degrees(np.abs(t.phi)) <= 135)
    rows = []
    for lab in HEAD:
        tt = t[t.model == lab]
        g = geo[t.model == lab]
        for col, who in (("tow_real", "real"), ("tow_model", lab)):
            def f(sel):
                v = tt[sel].groupby("fish")[col].mean()
                return float(v.mean()) if len(v) else np.nan
            near = (tt.rng_mm >= 10) & (tt.rng_mm < 35)
            peak = (tt[g & tt.moving & (tt.phase == ">= 10 s")].groupby(["fish", "range"], observed=True)[col].mean()
                    .groupby("range", observed=True).mean())
            rows.append(dict(model=lab, source=who,
                             onset_gain=f(g & near & tt.moving & (tt.phase == "0-4 s")),
                             steady_gain=f(g & near & tt.moving & (tt.phase == ">= 10 s")),
                             peak_range=str(peak.idxmax()) if len(peak) else None,
                             count_step=f(g & near & tt.moving & (tt.phase == ">= 10 s") & (tt.n >= 2))
                             - f(g & near & tt.moving & (tt.phase == ">= 10 s") & (tt.n == 1)),
                             static_gain=f(g & near & ~tt.moving & (tt.phase == ">= 10 s"))))
    return pd.DataFrame(rows)


# =========================================================================== #
# 5 (+10). Rollout-stream noise of the profile readouts; the unfiltered profiles
# =========================================================================== #
def _stream_job(job):
    lab, d, arm, seed = job
    _threads1()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import model_figures as MF
    from input_ablation import profile_variants_matched
    MF.BLOCK_SEED = True                                     # the report units' per-block random streams
    with contextlib.redirect_stdout(io.StringIO()):
        models, _ = MF.fold_models(str(_ckpt(d)), arm, REJ)
        raw = pd.concat([MF.programme_bouts(m, k, 10, seed=seed) for k, (_, m) in enumerate(models.items())],
                        ignore_index=True)
        out = {}
        for filt in (True, False):
            MF.EXCLUDE_CENTROID_OUT = filt
            MF._CACHE.pop("st", None)
            gen = MF.finish_profile(raw.copy(), len(models), 10, progress=False)
            tab, _ = profile_variants_matched(str(_ckpt(d)), arm, {lab: {"arm": arm}}, rej=REJ, r_per_model=10,
                                              show_raw=False, gens={lab: gen})
            plt.close("all")
            att = None
            if filt:
                # the units' attraction used its own 6 rollouts per fold (r_per_model=6): the first 6 of these 10 are
                # those rollouts exactly (same streams, fish_id = fold * 10 + r), so stream 0 reproduces attr_N*
                gen6 = gen[(gen.fish_id % 10) < 6]
                att, _ = MF.attraction_table(str(_ckpt(d)), [arm], rej=REJ, r_per_model=6, gens={arm: gen6})
                plt.close("all")
            out["filtered" if filt else "unfiltered"] = dict(table=tab, attraction=att)
        MF.EXCLUDE_CENTROID_OUT = True
        MF._CACHE.pop("st", None)
    return (lab, seed), out


def stream_noise(seeds=(0, 1, 2, 3), jobs=5):
    """The matched profile and attraction of each headline model on 4 rollout streams (seed 0 = the archived stream).
    Each job also re-reads its rollouts without the centroid-outside-dish filter (section 10). Cached as 'streams'."""
    jl = [(lab, d, arm, s) for lab, (d, arm, key) in HEAD.items() for s in seeds]
    res = {}
    with ProcessPoolExecutor(jobs) as ex:
        for k, v in ex.map(_stream_job, jl):
            res[k] = v
    save("streams", res)
    return res


STREAM_COLS = ("dip", "lat", "plateau", "d1_plateau", "grading", "mv_err_sem", "st_err_sem", "static_drift")


def stream_table(filtered=True):
    res = load("streams")
    rows = []
    key = "filtered" if filtered else "unfiltered"
    for (lab, seed), v in sorted(res.items()):
        tab = v[key]["table"]
        r = tab.loc[[i for i in tab.index if "[matched]" in str(i)][0]]
        row = dict(model=lab, stream=seed, **{c: float(r[c]) for c in STREAM_COLS})
        if filtered and v[key]["attraction"] is not None:
            a = v[key]["attraction"]
            col = [c for c in a.columns if c[0] != "real" and c[1] == "attraction"][0]
            for cond, n in (("d1s3.5", 1), ("d2s3.5", 2), ("d4s3.5", 4), ("d6s3.5", 6), ("d8s3.5", 8)):
                row[f"attr_N{n}"] = float(a.loc[cond, col]) if cond in a.index else np.nan
        rows.append(row)
        if seed == 0:
            rr = tab.loc["REAL"]
            rows.append(dict(model="real", stream=lab, **{c: float(rr[c]) for c in STREAM_COLS}))
    return pd.DataFrame(rows)


# =========================================================================== #
# 6. Data provenance: the bout table regenerated with the current segmentation code
# =========================================================================== #
def provenance():
    """Run the CURRENT scripts/segment_bouts.py on the raw tracking into Model/missing_pieces/, then compare bout by
    bout with the table the training data were cleaned from (data/bouts_GsizeMotionSingle.csv) and with the training
    table (clean_data/bouts_clean.csv). Cached as 'provenance'."""
    OUT.mkdir(parents=True, exist_ok=True)
    new = OUT / "bouts_regenerated_current_code.csv"
    t0 = time.time()
    if not new.exists():
        r = subprocess.run([sys.executable, str(HERE / "segment_bouts.py"), "--output", str(new)], cwd=str(ROOT),
                           capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        log = r.stdout[-4000:] + r.stderr[-4000:]
        if r.returncode:
            save("provenance", dict(error=log))
            raise RuntimeError(log)
    else:
        log = "(regenerated table already on disk)"
    a = pd.read_csv(ROOT / "data" / "bouts_GsizeMotionSingle.csv")
    b = pd.read_csv(new)
    c = pd.read_csv(ROOT / "clean_data" / "bouts_clean.csv")
    fcol = next(x for x in ("frame_onset", "onset_frame", "frame") if x in a.columns)
    rows = []
    for f in sorted(set(a.fish_id) | set(b.fish_id)):
        fa = np.sort(a.loc[a.fish_id == f, fcol].to_numpy()); fb = np.sort(b.loc[b.fish_id == f, fcol].to_numpy())
        fc = np.sort(c.loc[c.fish_id == f, fcol].to_numpy()) if fcol in c.columns else np.array([])
        def match(x, y, tol=2):
            if not len(x) or not len(y):
                return 0
            i = np.clip(np.searchsorted(y, x), 1, len(y) - 1)
            return int((np.minimum(np.abs(y[i] - x), np.abs(y[i - 1] - x)) <= tol).sum())
        rows.append(dict(fish=f, old_table=len(fa), regenerated=len(fb), training_table=len(fc),
                         regenerated_matching_old=match(fb, fa, 0), regenerated_matching_old_2fr=match(fb, fa, 2),
                         training_matching_regenerated_2fr=match(fc, fb, 2)))
    per = pd.DataFrame(rows)
    # bout by bout on the matched bouts (old vs regenerated, onsets within 2 frames): how much each quantity moved
    diffs = []
    for f in sorted(set(a.fish_id) & set(b.fish_id)):
        A_ = a[a.fish_id == f].sort_values(fcol); B_ = b[b.fish_id == f].sort_values(fcol)
        ya = A_[fcol].to_numpy(); yb = B_[fcol].to_numpy()
        if not len(ya) or not len(yb):
            continue
        i = np.clip(np.searchsorted(ya, yb), 1, len(ya) - 1)
        j = np.where(np.abs(ya[i] - yb) <= np.abs(ya[i - 1] - yb), i, i - 1)
        ok = np.abs(ya[j] - yb) <= 2
        for col in ("dtheta_deg", "distance_px", "dt_s"):
            if col in A_.columns and col in B_.columns:
                d_ = B_[col].to_numpy(float)[ok] - A_[col].to_numpy(float)[j[ok]]
                diffs.append(dict(fish=f, quantity=col, matched=int(ok.sum()), median_abs_change=float(np.nanmedian(np.abs(d_))),
                                  share_changed=float(np.nanmean(np.abs(d_) > {"dtheta_deg": 1.0, "distance_px": 0.5, "dt_s": 0.04}[col]))))
    matched_changes = (pd.DataFrame(diffs).groupby("quantity")
                       .agg(matched=("matched", "sum"), median_abs_change=("median_abs_change", "median"),
                            share_changed=("share_changed", "mean")) if diffs else pd.DataFrame())
    stats = {}
    for name, df in (("old table", a), ("regenerated", b), ("training table", c)):
        s = {}
        for col in ("dtheta_deg", "distance_px", "dt_s"):
            if col in df.columns:
                v = df[col].to_numpy(float)
                v = np.abs(v) if col == "dtheta_deg" else v
                s[f"median |{col}|" if col == "dtheta_deg" else f"median {col}"] = float(np.nanmedian(v))
        s["bouts"] = int(len(df))
        stats[name] = s
    diff = subprocess.run(["git", "diff", "--", "scripts/segment_bouts.py"], cwd=str(ROOT), capture_output=True,
                          text=True).stdout
    last = subprocess.run(["git", "log", "-1", "--format=%h %ci", "--", "scripts/segment_bouts.py"], cwd=str(ROOT),
                          capture_output=True, text=True).stdout.strip()
    mt = {p: time.strftime("%Y-%m-%d %H:%M", time.localtime((ROOT / p).stat().st_mtime))
          for p in ("data/bouts_GsizeMotionSingle.csv", "clean_data/bouts_clean.csv", "scripts/segment_bouts.py")}
    out = dict(per_fish=per, stats=pd.DataFrame(stats).T, matched_changes=matched_changes, diff=diff, last_commit=last,
               mtimes=mt, log=log, seconds=time.time() - t0)
    save("provenance", out)
    return out


# =========================================================================== #
# 7. Paired-by-fish CIs of the key likelihood differences
# =========================================================================== #
PAIRS = [  # (label, (key A, key B) at seed 0, (key A, key B) at seed 101): difference = B - A (bits per bout)
    ("M3 vs M2: + per-fish latent", ("2", "3"), ("34b", "34c")),
    ("M6 vs F (final model)", ("26e", "14f"), ("30d", "32b")),
    ("D1 vs D2: + slow drift", ("18a", "24b"), ("24a", "24c")),
    ("D3 vs D4: + slow drift", ("16e", "11e"), ("30e", "32a")),
    ("M6 vs no stimulus", ("26e", "29e"), ("30d", "33u")),
    ("M6 vs no group size", ("26e", "29c"), ("30d", "33s")),
    ("M6 vs no kinematic history", ("26e", "33m"), ("30d", "33n")),
    ("M6 vs 1 kinematic lag", ("26e", "33k"), ("30d", "33l")),
    ("M6 vs 64 hidden units", ("26e", "34g"), ("30d", "34h")),
    ("M6 vs 256 hidden units", ("26e", "34i"), ("30d", "34j")),
    ("M6 vs trained without 6 moving dots", ("26e", "34e"), ("30d", "34f")),
]


def _lik_fish(key):
    import report_models as RM
    u = RM._unit(key)
    p = MODEL / "report_ckpts" / f"M{key}" / "likA_model1.csv"
    if not p.exists():
        return None
    l = pd.read_csv(p).drop_duplicates(["arm", "fish"])
    l = l[l.arm == u["arm"]].set_index("fish")
    return l[["bits_per_bout", "bits_per_bout_dist", "bits_per_bout_dt", "bits_per_bout_joint"]]


def paired_ci():
    rows = []
    for lab, s0, s1 in PAIRS:
        for seed, (ka, kb) in ((0, s0), (101, s1)):
            A, B = _lik_fish(ka), _lik_fish(kb)
            if A is None or B is None:
                rows.append(dict(comparison=lab, seed=seed, keys=f"{ka} -> {kb}", joint="not available"))
                continue
            dd = (B - A).dropna()
            r = dict(comparison=lab, seed=seed, keys=f"{ka} -> {kb}", fish=len(dd))
            for c, nm in (("bits_per_bout_joint", "joint"), ("bits_per_bout", "turn"), ("bits_per_bout_dist", "step"),
                          ("bits_per_bout_dt", "interval")):
                ci = _boot_ci(dd[c].values)
                r[nm] = f"{dd[c].mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
            r["fish where B is better (joint)"] = f"{int((dd.bits_per_bout_joint < 0).sum())}/{len(dd)}"
            rows.append(r)
    return pd.DataFrame(rows).set_index(["comparison", "seed"])


# =========================================================================== #
# 8. Which point of a dot group the fish turns toward
# =========================================================================== #
def centroid_choice(min_dots=2, rng_mm=(5.0, 60.0)):
    """Real data: on bouts with >= 2 dots, turning toward (a) the centroid, (b) the nearest dot, (c) the 1/r-weighted
    centroid, on the bouts where the candidates lie on DIFFERENT sides of the fish (the only bouts that can tell them
    apart). Per fish, moving and static separately. Cached as 'centroid'."""
    import fit_MLP_border as MB
    rows = []
    for f, rec in sorted(records().items()):
        xs, ys = np.asarray(rec.dot_xs, float), np.asarray(rec.dot_ys, float)
        x, y = np.asarray(rec.x_px, float), np.asarray(rec.y_px, float)
        hd = MB._travel_ref(rec)
        dth = np.asarray(rec.dtheta, float)
        mv = np.asarray(rec.moving, float) > 0
        dx, dy = xs - x[:, None], ys - y[:, None]
        r = np.hypot(dx, dy)
        ok_d = np.isfinite(r)
        nd = ok_d.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cen = np.c_[np.nanmean(dx, 1), np.nanmean(dy, 1)]
            k = np.nanargmin(np.where(ok_d, r, np.inf), 1)
            nea = np.c_[dx[np.arange(len(x)), k], dy[np.arange(len(x)), k]]
            wgt = np.where(ok_d, 1.0 / np.maximum(r, 1.0), 0.0)
            iw = np.c_[(np.nan_to_num(dx) * wgt).sum(1) / wgt.sum(1), (np.nan_to_num(dy) * wgt).sum(1) / wgt.sum(1)]
        side = {}
        for nm, v in (("centroid", cen), ("nearest dot", nea), ("1/r-weighted", iw)):
            phi = _wrap(np.arctan2(v[:, 1], v[:, 0]) - hd)
            side[nm] = np.where((np.degrees(np.abs(phi)) > TOWARD_BEARING[0]) & (np.degrees(np.abs(phi)) < TOWARD_BEARING[1]),
                                np.sign(phi), 0)
        dist_c = np.hypot(cen[:, 0], cen[:, 1]) / PX_MM
        base = (nd >= min_dots) & (np.abs(dth) > 3) & (dist_c >= rng_mm[0]) & (dist_c < rng_mm[1])
        for cond, m in (("moving", mv), ("static", ~mv)):
            for a_, b_ in (("centroid", "nearest dot"), ("centroid", "1/r-weighted"), ("nearest dot", "1/r-weighted")):
                dis = base & m & (side[a_] != 0) & (side[b_] != 0) & (side[a_] != side[b_])
                if dis.sum() < 5:
                    continue
                rows.append(dict(fish=f, condition=cond, pair=f"{a_} vs {b_}", bouts=int(dis.sum()),
                                 toward_first=float(np.mean(np.sign(dth[dis]) == side[a_][dis])),
                                 toward_second=float(np.mean(np.sign(dth[dis]) == side[b_][dis]))))
            for nm in side:
                s = base & m & (side[nm] != 0)
                rows.append(dict(fish=f, condition=cond, pair=f"{nm} (all bouts)", bouts=int(s.sum()),
                                 toward_first=float(np.mean(np.sign(dth[s]) == side[nm][s])), toward_second=np.nan))
    tab = pd.DataFrame(rows)
    save("centroid", tab)
    return tab


def centroid_table():
    t = load("centroid")
    rows = []
    for (cond, pair), g in t.groupby(["condition", "pair"]):
        few = len(g) < 5                                                     # a CI over < 5 fish means nothing
        ci1 = _boot_ci(g.toward_first.values)
        r = dict(condition=cond, comparison=pair, fish=len(g), bouts=int(g.bouts.sum()),
                 first=(f"{g.toward_first.mean():.3f}" + (" (too few fish)" if few else f" [{ci1[0]:.3f}, {ci1[1]:.3f}]")))
        if g.toward_second.notna().any():
            d = g.toward_first - g.toward_second
            ci = _boot_ci(d.values)
            r["second"] = f"{g.toward_second.mean():.3f}"
            r["first - second [95% CI]"] = (f"{d.mean():+.3f}" + (" (too few fish)" if few else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"))
        rows.append(r)
    return pd.DataFrame(rows).set_index(["condition", "comparison"])


# =========================================================================== #
# 9. Multi-fish: where the neighbours sit around a fish
# =========================================================================== #
def neighbour_map(cond, half_mm=40.0, bin_mm=2.0, shifts_s=(60, 120, 180, 240, 300, 360, 420, 480, 540, 600)):
    """2-D density of neighbour positions in the focal fish's frame (x = its last travel direction), 1 Hz frames over
    the readout window, divided by the same map with the neighbours' tracks shifted in time (chance). Returns (ratio
    map, bin edges, radial profile of the ratio)."""
    import multi_fish as MFish
    blob = MFish.load(cond)
    spec = blob["spec"]
    taus = MFish.grid(spec, 1.0)
    edges = np.arange(-half_mm, half_mm + 1e-9, bin_mm)
    H = np.zeros((len(edges) - 1, len(edges) - 1)); Hs = np.zeros_like(H)
    radial_e = np.arange(0, half_mm + 1e-9, 2.5)
    Rr = np.zeros(len(radial_e) - 1); Rs = np.zeros_like(Rr)
    for grp in blob["groups"]:
        P = np.array([MFish.pos_at(f, taus) for f in grp]) / PX_MM
        Hd = np.array([MFish.heading_at(f, taus) for f in grp])
        N = len(grp)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                for shift, (HH, RR) in [(0, (H, Rr))] + [(s, (Hs, Rs)) for s in shifts_s]:
                    q = np.roll(P[j], -int(shift), axis=0) if shift else P[j]
                    d = q - P[i]
                    c, s_ = np.cos(-Hd[i]), np.sin(-Hd[i])
                    u = d[:, 0] * c - d[:, 1] * s_
                    v = d[:, 0] * s_ + d[:, 1] * c
                    w = 1.0 if not shift else 1.0 / len(shifts_s)
                    h, _, _ = np.histogram2d(u, v, [edges, edges])
                    HH += w * h
                    rr, _ = np.histogram(np.hypot(u, v), radial_e)
                    RR += w * rr
    ratio = np.where(Hs > 0, H / np.maximum(Hs, 1e-9), np.nan)
    prof = pd.Series(Rr / np.maximum(Rs, 1e-9), index=(radial_e[1:] + radial_e[:-1]) / 2)
    return ratio, edges, prof


# =========================================================================== #
# 11. Leave one condition out: the held-out condition's likelihood
# =========================================================================== #
def _cond_lik_job(job):
    lab, d, file, fids = job
    _threads1()
    m = load_model(d, file)
    recs = records()
    rows = []
    for f in fids:
        rec = recs[f]
        sc = m.score(rec)
        nd = np.nan_to_num(np.asarray(rec.n_dots, float))
        mv = np.asarray(rec.moving, float) > 0
        cond = np.where(nd == 0, "no dots", np.where(mv, "moving ", "static ") + nd.astype(int).astype(str))
        lt, ldd, ldt = (np.asarray(sc.get(k), float) for k in ("logdens", "logdens_dist", "logdens_dt"))
        for c in np.unique(cond):
            s = cond == c
            b = [float(-np.nanmean(v[s]) / np.log(2)) for v in (lt, ldd, ldt)]   # likA's convention: per-channel means
            rows.append(dict(model=lab, fish=f, condition=c, bouts=int(s.sum()), turn=b[0], step=b[1], interval=b[2],
                             joint=float(np.nansum(b))))
    return rows


def loco_condition_lik(jobs=5):
    """Held-out bits per bout by stimulus condition for M6 and M6-trained-without-6-moving-dots, both seeds. Cached
    as 'loco_lik'."""
    units = {"M6 s0": ("report_ckpts/M26e", ARM_M6), "M6 s101": ("report_ckpts/M30d", ARM_M6),
             "LOCO s0": ("missing_ladder_checkpoints", ARM_M6 + "_lo6m"),
             "LOCO s101": ("missing_seed101_ckpts", ARM_M6 + "_lo6m")}
    jl = [(lab, d, file, ho) for lab, (d, arm) in units.items() for file, ho in folds_of(d, arm)]
    with ProcessPoolExecutor(jobs) as ex:
        rows = [r for part in ex.map(_cond_lik_job, jl) for r in part]
    tab = pd.DataFrame(rows)
    save("loco_lik", tab)
    return tab


def loco_table():
    t = load("loco_lik")
    rows = []
    for seed in ("s0", "s101"):
        a = t[t.model == f"M6 {seed}"].set_index(["fish", "condition"])
        b = t[t.model == f"LOCO {seed}"].set_index(["fish", "condition"])
        for c in sorted(t.condition.unique()):
            if c not in a.index.get_level_values(1):
                continue
            dd = (b.xs(c, level=1).joint - a.xs(c, level=1).joint).dropna()
            ci = _boot_ci(dd.values)
            rows.append(dict(seed=seed, condition=c, bouts=int(a.xs(c, level=1).bouts.sum()),
                             M6=a.xs(c, level=1).joint.mean(), LOCO=b.xs(c, level=1).joint.mean(),
                             **{"LOCO - M6 [95% CI]": f"{dd.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"},
                             held_out=(c == "moving 6")))
    return pd.DataFrame(rows).set_index(["seed", "condition"])


# =========================================================================== #
def main(argv):
    _threads1()
    what = argv[1] if len(argv) > 1 else ""
    jobs = int(os.environ.get("MP_JOBS", "5"))
    t0 = time.time()
    if what == "refs":
        print(realism_refs(jobs=jobs).round(3).to_string())
    elif what == "baselines":
        likelihood_baselines(jobs=jobs); print(baselines_table().round(4).to_string())
    elif what == "properties":
        properties(jobs=jobs); print(properties_table().round(3).to_string())
    elif what == "importances":
        print(importances().round(4).to_string())
    elif what == "nearest":
        nearest(jobs=jobs)
    elif what == "laws":
        response_maps(jobs=jobs); print(law_summary().round(2).to_string())
    elif what == "streams":
        stream_noise(jobs=jobs); print(stream_table().round(3).to_string())
    elif what == "provenance":
        o = provenance(); print(o["per_fish"].to_string()); print(o["stats"])
    elif what == "centroid":
        centroid_choice(); print(centroid_table().to_string())
    elif what == "loco":
        loco_condition_lik(jobs=jobs); print(loco_table().round(4).to_string())
    else:
        print(__doc__)
        return
    print(f"[{what}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv)
