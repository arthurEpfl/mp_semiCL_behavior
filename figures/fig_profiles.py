"""fig_profiles -- Chapter 5, sec:r-dots (label fig:profiles): distance to the dot centroid across the block.

Panels a-h: one per stimulus condition (1/2/4/6/8 moving dots, 1/4/8 static), real fish (mean over fish of each
fish's 2 s bin means, 95 % percentile bootstrap band over the 15 fish, as Chapter 3's fig_dot_profile) against M6 and
F at both seeds, the simulated side BLOCK-MATCHED (``model_figures.weighted_curves``: every generated bin mean taken
over the same programme blocks, in the same proportions, as the real curve), exactly the curves the archived matched
metrics (dip / latency / plateau / grading / profile error) are computed from.
(2026-10-12: the attraction panel i was dropped from the figure, its place taken by the legend. The attraction numbers
below are still computed and printed, and still checked against the archive.)
Attraction per moving-dot count (stimulus-blind null minus the 20-60 s plateau): real (recomputed here with
``model_figures.attraction_table``, with a 95 % percentile bootstrap over fish from the per-fish null and plateau,
``real_attraction_ci``), M6 and F (the archived ``attr_N*`` of the report units), and the M6 -stim ablation per seed
(``attr_N*`` of 29e | 33u) as the stimulus-blind zero point. ``latency_table`` writes the per-condition dip / latency
with the archived ``_dip`` and with an edge-free running mean (the archived one cuts at 20 s before smoothing).

Rollouts: the archived STREAM 0 of the four headline units, regenerated exactly as ``missing_pieces._stream_job``
does (MF.BLOCK_SEED = True; MF.fold_models with the wall-rejection band; MF.programme_bouts(m, k, 10, seed=0) per
fold; MF.finish_profile with the centroid-outside-dish filter on). The script checks that the regenerated curves
give back the archived matched dip / latency / plateau / 1-dot plateau / grading / profile errors and attraction of
summary_M26e / M30d / M14f / M32b, and refuses to draw if they do not.

Scale: every distance is drawn and printed at Chapter 3's physical 3.60 px/mm. The profile code divides px by the
nominal 3.67 (model_figures.PX_TO_MM, asserted), so curves, bands, attraction, its CI and the printed readouts are
multiplied by style.MM_3P67_TO_3P60 (= 3.67 / 3.60) at plotting time. The archive check runs on the UNconverted
values (the archived summaries are at 3.67); the caches stay at 3.67 and are written only by --rebuild (or when
missing). Latencies, SEM-unit errors and fish counts are not converted.

    uv run python -m figures.fig_profiles              # build the cache if missing, check, draw
    uv run python -m figures.fig_profiles --rebuild    # regenerate the rollouts (20 jobs, <= 8 workers)
"""


from figures.style import *  # noqa: F401,F403,E402  (sets rcParams, gives save(), panel(), colours)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RF = REPORT_FINAL_DIR
CACHE = RF / "cache"
CURVES = CACHE / "fig_profiles_curves.pkl"

REJ = {"d0": 2.0, "d1": 10.0, "max_draws": 20}
ARM_M6 = "S4x_adt_z2_cdt_tr_rc_al_w3ft"
ARM_F = "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last"
ARM_D4 = "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st"
# (public name, seed) -> (checkpoint folder under output/checkpoints/, arm, registry key)
HEAD = {("M6", 0): ("M26e", ARM_M6, "26e"), ("M6", 101): ("M30d", ARM_M6, "30d"),
        ("D4", 0): ("M11e", ARM_D4, "11e"), ("D4", 101): ("M32a", ARM_D4, "32a"),
        ("F", 0): ("M14f", ARM_F, "14f"), ("F", 101): ("M32b", ARM_F, "32b")}
ZERO = {0: "29e", 101: "33u"}                    # M6 -stim: the stimulus-blind zero point of attraction
N_FOLDS, R_PER_MODEL = 5, 10
CONDS = ["d1s3.5", "d2s3.5", "d4s3.5", "d6s3.5", "d8s3.5", "d1s0", "d4s0", "d8s0"]
COUNTS = (1, 2, 4, 6, 8)
N_BOOT, BOOT_SEED = 4000, 0


def _lab(name, seed):
    return f"{name} s{seed}"


# --------------------------------------------------------------------------------------------- compute ------
def _fold_job(job):
    """One fold model of one unit: its 10 programme rollouts of stream 0 (missing_pieces._stream_job, per fold)."""
    name, seed, d, arm, k = job
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    torch.set_num_threads(1)
    from twin import model_figures as MF
    MF.BLOCK_SEED = True                                  # the report units' per-block random streams
    with contextlib.redirect_stdout(io.StringIO()):
        models, _ = MF.fold_models(str(paths.CKPTS / d), arm, REJ)
        m = list(models.values())[k]                      # same (sorted-by-file) order as fold_models' callers
        raw = MF.programme_bouts(m, k, R_PER_MODEL, seed=0)
    return (name, seed, k), raw


def _boot_band(pr, cond):
    """Real mean and 95 % percentile bootstrap over fish (Chapter 3's fig_dot_profile, same n_boot and seed)."""
    M = pr.loc[cond].unstack("tbin").sort_index(axis=1)
    A = M.to_numpy(float)
    mu = np.nanmean(A, axis=0)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, A.shape[0], size=(N_BOOT, A.shape[0]))
    bs = np.array([np.nanmean(A[i], axis=0) for i in idx])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return pd.DataFrame({"mu": mu, "lo": lo, "hi": hi, "n_fish": np.sum(np.isfinite(A), axis=0)},
                        index=M.columns.to_numpy(float))


def build(workers=8, have=None):
    """Regenerate the rollouts of every unit in HEAD; with ``have`` (a loaded cache), only the units it lacks."""
    import matplotlib.pyplot as plt
    from twin import model_figures as MF
    from twin.input_ablation import profile_variants_matched
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = {u: v for u, v in HEAD.items() if have is None or u not in have["models"]}
    jobs = [(name, seed, d, arm, k) for (name, seed), (d, arm, _) in todo.items() for k in range(N_FOLDS)]
    parts = {}
    with ProcessPoolExecutor(min(workers, 8)) as ex:
        for key, raw in ex.map(_fold_job, jobs):
            parts[key] = raw
            print(f"  rollouts done: {key}  ({len(raw)} bouts, runaway {raw.attrs.get('runaway', 0)})", flush=True)

    MF.BLOCK_SEED = True
    MF.EXCLUDE_CENTROID_OUT = True
    MF._CACHE.pop("st", None)
    program, _, _ = MF._programme()
    st = MF.real_dot_profile()
    pr, mr, sr = MF._curves(st)
    w = MF.real_block_weights(st, program)
    out = {"px_per_mm": MF.PX_TO_MM, "bin_s": MF.BIN_S,
           "real": {"per_fish": pr, "mean": mr, "sem": sr,
                    "band": {c: _boot_band(pr, c) for c in CONDS}},
           "models": {}}
    if have is not None:                                  # the same real side (same code, same data): reuse it
        for cond in CONDS:
            assert np.allclose(have["real"]["band"][cond].to_numpy(float), out["real"]["band"][cond].to_numpy(float),
                               equal_nan=True), cond
        out["models"].update(have["models"])
    for (name, seed), (d, arm, key) in todo.items():
        lab = _lab(name, seed)
        raw = pd.concat([parts[(name, seed, k)] for k in range(N_FOLDS)], ignore_index=True)
        runaway = int(sum(int(parts[(name, seed, k)].attrs.get("runaway", 0)) for k in range(N_FOLDS)))
        with contextlib.redirect_stdout(io.StringIO()):
            gen = MF.finish_profile(raw.copy(), N_FOLDS, R_PER_MODEL, progress=False)
            tab, _ = profile_variants_matched(str(paths.CKPTS / d), arm, {lab: {"arm": arm}}, rej=REJ,
                                              r_per_model=R_PER_MODEL, show_raw=False, gens={lab: gen})
            plt.close("all")
            gen6 = gen[(gen.fish_id % R_PER_MODEL) < 6]   # the units' attraction used the first 6 rollouts per fold
            att, _ = MF.attraction_table(str(paths.CKPTS / d), [arm], rej=REJ, r_per_model=6, gens={arm: gen6})
            plt.close("all")
        pg1, mg1, sg1 = MF.weighted_curves(gen, w)        # the matched curves the [matched] metrics are read from
        gen.to_pickle(CACHE / f"fig_profiles_gen_{key}.pkl")
        out["models"][(name, seed)] = {"key": key, "matched_mean": mg1, "matched_per_fish": pg1,
                                       "matched_sem": sg1, "table": tab, "attraction": att,
                                       "n_sim_fish": int(gen.fish_id.nunique()), "runaway": runaway}
        print(f"  {lab} ({key}): {gen.fish_id.nunique()} simulated fish, {len(gen)} stimulus bouts kept")
    with open(CURVES, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cached {CURVES}")
    return out


def summary(key):
    return json.loads((paths.EVAL / f"summary_M{key}.json").read_text())


def drop_stale(c, head, path):
    """Forget the cached rollouts of the units of ``head`` retrained after ``path`` was written, so that ``build``
    rolls them out again."""
    from twin import registry
    for u, (d, _, _) in head.items():
        if u in c["models"] and registry.stale(path, [d]):
            print(f"  {_lab(*u)} was retrained after {path.name} was written: rolling it out again")
            del c["models"][u]
    return c


def gen_of(key):
    """A unit's cached profile rollouts (written by ``build``), refused if older than the unit's checkpoints."""
    from twin import registry
    p = CACHE / f"fig_profiles_gen_{key}.pkl"
    if registry.stale(p, [registry.ckpt_dir(key)]):
        raise SystemExit(f"{p.name} is missing or older than unit {key}'s checkpoints: "
                         "run figures.fig_profiles (or figures.fig_ablation_profiles) first")
    return pd.read_pickle(p)


def check(c):
    """The regenerated stream-0 readouts against the archived report units; returns the check table."""
    from twin import model_figures as MF
    from twin import registry
    old = [m["key"] for m in c["models"].values()
           if registry.stale(paths.EVAL / f"summary_M{m['key']}.json", [registry.ckpt_dir(m["key"])])]
    if old:
        raise SystemExit(f"unit(s) {', '.join(old)} retrained after their evaluation: run "
                         f"uv run python pipelines/evaluate.py --keys {','.join(old)} first")
    rows = []
    for (name, seed), m in c["models"].items():
        s = summary(m["key"])
        # recompute the matched metrics from the CACHED curves alone (no table): same code as the archive
        r = c["real"]
        summ = MF.profile_summary(MF.metrics_from_curves(r["per_fish"], r["mean"], r["sem"],
                                                         m["matched_per_fish"], m["matched_mean"]))
        mine = {"dip": summ["dip_gen"], "lat": summ["lat_gen"], "plateau": summ["plateau_gen"],
                "d1_plateau": summ["d1_plateau_gen"], "grading": summ["grading_gen"],
                "mv_err_sem": summ["mv_err_sem"], "st_err_sem": summ["st_err_sem"],
                "static_drift": summ["drift_gen"]}
        att = m["attraction"]
        col = [k for k in att.columns if k[0] != "real" and k[1] == "attraction"][0]
        for n in COUNTS:
            mine[f"attr_N{n}"] = float(att.loc[f"d{n}s3.5", col])
        for k, v in mine.items():
            rows.append(dict(model=_lab(name, seed), key=m["key"], metric=k, regenerated=float(v),
                             archived=float(s[k]), diff=float(v) - float(s[k])))
    t = pd.DataFrame(rows)
    # archived summaries are rounded to 3 decimals (attraction to 2)
    t["ok"] = t["diff"].abs() <= np.where(t.metric.str.startswith("attr"), 0.0051, 0.00051)
    return t


ATT_CI = CACHE / "fig_profiles_attr_ci.csv"
LAT_TAB = CACHE / "fig_profiles_latency.csv"


def _null_per_fish(df, rng, reps=6, cap=40000):
    """``model_figures._null_plateau`` line for line, but returning the PER-FISH null (its last line averages
    these over fish). Same rng consumption, so the fish mean is the archived null exactly (checked below)."""
    from twin import model_figures as MF
    g = (df[df.tbin < 60].groupby(["block_id", "cond_label", "tbin"], observed=True)[["dx", "dy"]]
         .mean().reset_index())
    lut = {(int(b), float(t)): (x, y) for b, t, x, y in zip(g.block_id, g.tbin, g.dx, g.dy)}
    blocks = {cc: np.array(sorted(v.block_id.unique())) for cc, v in g.groupby("cond_label", observed=True)}
    pos = df[(df.tbin >= 20) & (df.tbin < 60)][["fish_id", "block_id", "tbin", "fx", "fy"]]
    if len(pos) > cap:
        pos = pos.sample(cap, random_state=int(MF.seed_of(rng)))
    pos = pos.to_numpy()
    rows = []
    for cond, bl in blocks.items():
        for _ in range(int(reps)):
            pick = bl[rng.integers(0, len(bl), len(pos))]
            for (fid, own, t, fx, fy), b in zip(pos, pick):
                if int(b) == int(own):
                    continue
                p = lut.get((int(b), float(t)))
                if p is not None:
                    rows.append((cond, int(fid), float(np.hypot(fx - p[0], fy - p[1]) / MF.PX_TO_MM)))
    n = pd.DataFrame(rows, columns=["cond_label", "fish_id", "d"])
    return n.groupby(["cond_label", "fish_id"]).d.mean()


def real_attraction_ci(c, rebuild=False):
    """95 % percentile bootstrap over fish (4000 resamples, rng seed 0, Chapter 3's band convention) of the REAL
    attraction per moving-dot count: per fish, own null (``_null_per_fish``, rng seed 0, reps 6, exactly as
    ``attraction_table``) minus own 20-60 s plateau; resample fish, average. Checks that the fish means give back
    the cached ``attraction_table`` real null / plateau / attraction to 1e-9 and refuses otherwise."""
    if ATT_CI.exists() and not rebuild:
        return pd.read_csv(ATT_CI, index_col=0)
    from twin import model_figures as MF
    with contextlib.redirect_stdout(io.StringIO()):
        prog, _, _ = MF._programme()
        st = MF.real_dot_profile().copy()             # same preparation as attraction_table's real side
        recs = MF.records()
    st["block_id"] = prog.block_id[np.clip(np.asarray(st.frame_onset, int), 0, prog.n_frames - 1)]
    dish = pd.DataFrame({f: dict(cx=r.dish_cx, cy=r.dish_cy) for f, r in recs.items()}).T
    st = st.join(dish, on="fish_id")
    st["fx"], st["fy"] = st.x_px - st.cx, st.y_px - st.cy
    st["dx"], st["dy"] = st.dot_cx - st.cx, st.dot_cy - st.cy
    plateau = st[(st.tbin >= 20) & (st.tbin < 60)].groupby(["cond_label", "fish_id"]).dist_mm.mean()
    null = _null_per_fish(st, np.random.default_rng(0), reps=6)
    att = next(iter(c["models"].values()))["attraction"]
    rows = []
    for n in COUNTS:
        cond = f"d{n}s3.5"
        pl, nu = plateau.loc[cond], null.loc[cond]
        for got, want in ((nu.mean(), att.loc[cond, ("real", "null")]),
                          (pl.mean(), att.loc[cond, ("real", "plateau")]),
                          (nu.mean() - pl.mean(), att.loc[cond, ("real", "attraction")])):
            if abs(float(got) - float(want)) > 1e-9:
                raise SystemExit(f"per-fish real attraction does not reproduce attraction_table at {cond}")
        common = pl.index.intersection(nu.index)
        per = (nu[common] - pl[common]).to_numpy(float)
        rng = np.random.default_rng(BOOT_SEED)
        bs = per[rng.integers(0, len(per), size=(N_BOOT, len(per)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(n_dots=n, attraction=float(nu.mean() - pl.mean()), paired_mean=float(per.mean()),
                         lo=float(lo), hi=float(hi), n_fish=len(common), n_fish_plateau=len(pl),
                         n_fish_null=len(nu)))
    t = pd.DataFrame(rows).set_index("n_dots")
    t.to_csv(ATT_CI)
    return t


def _dip_edge_free(mu):
    """The archived ``model_figures._dip`` truncates the curve at < 20 s BEFORE the centred 3-bin running mean,
    so the 19 s point is a 2-bin mean (17, 19 s). Here the running mean is taken on the whole curve first and
    then restricted to < 20 s. For the latency note only; the archived metric is unchanged."""
    sm = mu.rolling(3, center=True, min_periods=1).mean()
    sm = sm[sm.index < 20]
    return float(mu[mu.index < 4].mean() - sm.min()), float(sm.idxmin())


def latency_table(c, write=False):
    """Per-condition dip and latency of every side, with the archived metric and with the edge-free variant.
    Dips in mm at the profile code's 3.67 px/mm (as the cache); ``to360_table`` converts them for printing."""
    from twin import model_figures as MF
    sides = {"real": c["real"]["mean"]}
    sides.update({_lab(*k): m["matched_mean"] for k, m in c["models"].items()})
    rows = []
    for side, mean in sides.items():
        for n in (2, 4, 6, 8):
            mu = mean[f"d{n}s3.5"].dropna()
            d0, l0 = MF._dip(mu)
            d1, l1 = _dip_edge_free(mu)
            rows.append(dict(side=side, n_dots=n, dip=d0, lat=l0, dip_edge_free=d1, lat_edge_free=l1))
    t = pd.DataFrame(rows)
    if write or not LAT_TAB.exists():
        t.to_csv(LAT_TAB, index=False)
    return t


def to360_table(t, cols):
    """A copy of ``t`` with the millimetre columns ``cols`` converted from 3.67 to 3.60 px/mm."""
    t = t.copy()
    for col in cols:
        t[col] = t[col] * MM_3P67_TO_3P60
    return t


def real_numbers(c):
    """The real readouts from the same code, for the caption / prose, in mm at 3.60 px/mm (latency in s)."""
    from twin import model_figures as MF
    f = MM_3P67_TO_3P60
    r = c["real"]
    any_m = next(iter(c["models"].values()))
    s = MF.profile_summary(MF.metrics_from_curves(r["per_fish"], r["mean"], r["sem"],
                                                  any_m["matched_per_fish"], any_m["matched_mean"]))
    att = any_m["attraction"]
    real_att = {n: float(att.loc[f"d{n}s3.5", ("real", "attraction")]) * f for n in COUNTS}
    real_null = {n: float(att.loc[f"d{n}s3.5", ("real", "null")]) * f for n in COUNTS}
    return {"dip": s["dip_real"] * f, "lat": s["lat_real"], "plateau": s["plateau_real"] * f,
            "d1_plateau": s["d1_plateau_real"] * f, "grading": s["grading_real"] * f,
            "static_drift_mm_per_s": s["drift_real"] * f, "attraction": real_att, "null": real_null}


def model_numbers(c):
    """Every unit's plotted readouts at 3.60 px/mm: the archived summary x factor (mm) and the latency / SEM errors
    unchanged; attraction per count as drawn in panel i (archived ``attr_N*`` x factor), plus the M6 -stim zero."""
    f = MM_3P67_TO_3P60
    rows = []
    for (name, seed), m in c["models"].items():
        s = summary(m["key"])
        rows.append(dict(side=_lab(name, seed), key=m["key"], dip=s["dip"] * f, lat=s["lat"],
                         plateau=s["plateau"] * f, d1_plateau=s["d1_plateau"] * f, grading=s["grading"] * f,
                         mv_err_sem=s["mv_err_sem"], st_err_sem=s["st_err_sem"],
                         **{f"attr_N{n}": s[f"attr_N{n}"] * f for n in COUNTS}))
    for seed, key in ZERO.items():
        s = summary(key)
        rows.append(dict(side=f"M6-stim s{seed}", key=key, **{f"attr_N{n}": s[f"attr_N{n}"] * f for n in COUNTS}))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- draw -------
TITLES = {"d1s3.5": "1 moving dot", "d2s3.5": "2 moving dots", "d4s3.5": "4 moving dots", "d6s3.5": "6 moving dots",
          "d8s3.5": "8 moving dots", "d1s0": "1 static dot", "d4s0": "4 static dots", "d8s0": "8 static dots"}
UNITS = ("M6", "D4", "F")                                 # drawn in this order, one block of panels per seed
UNIT_C = {"M6": MODEL_C["M6"], "D4": GREEN, "F": MODEL_C["F"]}
SEEDS = (0, 101)


Y_MOVING, Y_STATIC, Y_ATT = (10, 30), (17, 41), (-11, 10)   # mm at 3.60 px/mm; real bands: moving 10.6-29.6, static 18.1-40.2


def _inside(vals, lim, what):
    """Refuse to draw a value outside its axis (nothing may be clipped)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size and (v.min() < lim[0] or v.max() > lim[1]):
        raise SystemExit(f"{what}: values {v.min():.3f}..{v.max():.3f} mm fall outside the axis {lim}")


def _block(sf, c, seed, letters):
    """One seed: 2 rows x 4 panels, moving 1/2/4/6 dots on top, moving 8 then static 1/4/8 below. Moving panels share
    one y range, static panels another; real fish (mean, bootstrap band) against M6, D4 and F at this seed."""
    f = MM_3P67_TO_3P60                                   # every mm below: 3.67 -> 3.60 px/mm
    gs = sf.add_gridspec(2, 4, left=0.075, right=0.995, bottom=0.10, top=0.88, wspace=0.34, hspace=0.30)
    axes = [sf.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    for i, ax in enumerate(axes):
        if i:
            ax.sharex(axes[0])
        if 0 < i < 5:
            ax.sharey(axes[0])
        elif i > 5:
            ax.sharey(axes[5])
    for i, (ax, cond) in enumerate(zip(axes, CONDS)):
        lim = Y_MOVING if i < 5 else Y_STATIC
        b = c["real"]["band"][cond]
        b = b[b.index < 60]
        _inside(np.r_[b.lo, b.hi, b.mu] * f, lim, f"real {cond}")
        ax.fill_between(b.index, b.lo * f, b.hi * f, color=REAL_C, alpha=0.16, lw=0)
        ax.plot(b.index, b.mu * f, color=REAL_C, lw=1.6, zorder=4)
        for name in UNITS:
            mg = c["models"][(name, seed)]["matched_mean"]
            if cond in mg:
                y = mg[cond]
                y = y[y.index < 60]
                _inside(y.values * f, lim, f"{name} s{seed} {cond}")
                ax.plot(y.index, y.values * f, color=UNIT_C[name], lw=1.3, zorder=3)
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 20, 40, 60])
        ax.grid(axis="y"); ax.set_axisbelow(True)
        panel(ax, letters[i], TITLES[cond])
        ax.texts[-1].set_x(-0.06)                         # a little air between the letter and the title
        if i < 4:
            ax.tick_params(labelbottom=False)
        if i not in (0, 4, 5):
            ax.tick_params(labelleft=False)
        if i == 5:
            ax.set_ylabel("distance (mm)", labelpad=1.5)  # the static scale: its own label
    axes[0].set_ylim(*Y_MOVING); axes[0].set_yticks([10, 15, 20, 25, 30])
    axes[5].set_ylim(*Y_STATIC); axes[5].set_yticks([20, 25, 30, 35, 40])   # the real band spans 18.1-40.2 mm
    p0, p4 = axes[0].get_position(), axes[4].get_position()
    sf.text(0.008, (p0.y1 + p4.y0) / 2, "distance to dots (mm)", rotation=90, ha="left", va="center",
            fontsize=8)                                   # one moving-scale label for the block
    pe, pf = axes[4].get_position(), axes[5].get_position()
    xr = pe.x1 + 0.10 * (pf.x0 - pe.x1)                   # thin rule: 8 moving dots | static dots
    sf.add_artist(Line2D([xr, xr], [pe.y0 - 0.02, pe.y1 + 0.06], transform=sf.transSubfigure, color=FAINT,
                         lw=0.8))
    sf.text(0.54, 0.0, "time since block onset (s)", ha="center", va="bottom", fontsize=8)
    sf.text(0.0, 0.995, f"Seed {seed}", ha="left", va="top", fontsize=9.5, fontweight="bold")
    return axes


def _legend(fig, y):
    from matplotlib.legend_handler import HandlerTuple
    handles = [(Patch(facecolor=REAL_C, alpha=0.16, lw=0), Line2D([], [], color=REAL_C, lw=1.6))]
    handles += [Line2D([], [], color=UNIT_C[n], lw=1.3) for n in UNITS]
    labels = ["real fish (mean, 95 % CI)"] + list(UNITS)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, y), ncol=4, handlelength=2.4,
               columnspacing=1.6, frameon=False, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})


BLOCK_H = 4.0                                             # inches per seed block (both seeds = a full page)


def draw(c, ci, seeds=SEEDS):
    """Both seeds stacked (fig_profiles, panels a-h seed 0, i-p seed 101), or one seed alone."""
    lead = 0.25                                           # the legend row on top
    fig = plt.figure(figsize=(TEXTWIDTH_IN, BLOCK_H * len(seeds) + lead))
    top = fig.subfigures(2, 1, height_ratios=[lead, BLOCK_H * len(seeds)], hspace=0)
    _legend(top[0], 1.0)
    sfs = top[1].subfigures(len(seeds), 1, hspace=0) if len(seeds) > 1 else [top[1]]
    for j, (sf, seed) in enumerate(zip(sfs, seeds)):
        _block(sf, c, seed, "abcdefghijklmnop"[8 * j: 8 * j + 8])
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rebuild-ci", action="store_true", help="recompute the real attraction bootstrap only")
    a = ap.parse_args(argv)
    from twin import model_figures as MF
    assert_code_scale(MF.PX_TO_MM, "model_figures.PX_TO_MM")
    if a.rebuild or not CURVES.exists():
        c = build(a.workers)
    else:
        with open(CURVES, "rb") as fh:
            c = drop_stale(pickle.load(fh), HEAD, CURVES)
        if any(u not in c["models"] for u in HEAD):       # a unit added to HEAD or retrained: roll out only that one
            c = build(a.workers, have=c)
    assert_code_scale(c["px_per_mm"], "the cached curves' px_per_mm")
    t = check(c)                                          # UNconverted values (3.67) against the archive (3.67)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    if a.rebuild or not (CACHE / "fig_profiles_check.csv").exists():
        t.to_csv(CACHE / "fig_profiles_check.csv", index=False)
    if not t.ok.all():
        raise SystemExit("regenerated curves do NOT reproduce the archived summaries -- not drawing")
    print("check passed: regenerated stream-0 curves reproduce every archived matched metric and attraction")
    print(f"\nEVERYTHING BELOW IN mm AT 3.60 px/mm (x {MM_3P67_TO_3P60:.6f}); latencies in s, errors in real SEM")
    print("real:", {k: (round(v, 3) if isinstance(v, float) else {n: round(x, 3) for n, x in v.items()})
                    for k, v in real_numbers(c).items()})
    print("models as plotted (archived summary x factor):")
    print(model_numbers(c).round(3).to_string(index=False))
    ci = real_attraction_ci(c, rebuild=a.rebuild or a.rebuild_ci)
    print("real attraction, 95 % bootstrap over fish:")
    print(to360_table(ci, ["attraction", "paired_mean", "lo", "hi"]).round(3).to_string())
    lt = to360_table(latency_table(c, write=a.rebuild), ["dip", "dip_edge_free"])
    print("per-condition dip / latency, archived metric vs edge-free running mean:")
    print(lt.round(3).to_string(index=False))
    print(lt.groupby("side")[["dip", "lat", "dip_edge_free", "lat_edge_free"]].mean().round(3).to_string())
    save(draw(c, ci), "fig_profiles")
    for seed in SEEDS:                                    # one seed per file, if the report wants two figures
        save(draw(c, ci, seeds=(seed,)), f"fig_profiles_s{seed}")


if __name__ == "__main__":
    main()
