"""The two rollout-based figures that recur once per model in Model/Model_design_and_training.ipynb
(and Ablation_study/Stimulus_ablation.ipynb) -- what the panel / C2ST tables cannot show.

    individuality_plot(ckpt_dir, arm)   dist_std / dt_std / dt_acf1 / dist_acf1 -- the C2ST
                                        classifier's top features -- per fish and pooled, real vs
                                        each fish rolled out by the fold model that never saw it
    dot_profile(ckpt_dir, arm)          distance to the dots' centre over the block, per condition,
                                        real fish next to programme rollouts (the rig's re-anchoring
                                        rule), with the profile metrics printed under the plot

`ckpt_dir` is a checkpoint folder (relative to the notebook's cwd or to Model/), `arm` the arm whose
fold checkpoints to use, e.g. individuality_plot("stim_ladder_checkpoints", "S3_adt_w8").  `rej` is
the generation-time wall-rejection band merged into every model at load (rej=None for a model
without it).  The real-data pieces (border records, the real profile, the programme) are built once
per session and cached.
"""
import contextlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from twin import evaluate as E
from twin import fit_MLP_border as MB
from twin import fit_MLP_mix as MX
from twin import generative_metrics as GM
from twin import paths
from twin import stim_program as SP
from twin import stim_rollout as SR
from twin import tool_reference as TR

REJ_BAND = {"d0": 2.0, "d1": 10.0, "max_draws": 20}       # rej: redraw a bout landing 2-10 mm from the wall
FEATS = [("dist_std", "step spread in a window (px)"),
         ("dt_std", "interval spread in a window (s)"),
         ("dt_acf1", "interval lag-1 acf in a window"),
         ("dist_acf1", "step lag-1 acf in a window")]
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e1e0d9"

PX_TO_MM = 3.67               # same scale as the original AllFish plot
BIN_S = 2.0                   # bouts are sampled at their onsets (~1.2 s apart), so time is binned
EXCLUDE_CENTROID_OUT = True   # closest available analogue of the original `dots_out < 1` filter
STIMULI_ORDER = ["d1s3.5", "d2s3.5", "d4s3.5", "d6s3.5", "d8s3.5", "d1s0", "d4s0", "d8s0"]
MOVING_SHOAL = ["d2s3.5", "d4s3.5", "d6s3.5", "d8s3.5"]     # the shoal-approach conditions
PROGRAM = paths.PROGRAM
REFERENCE = paths.ASSETS / "real_reference.npz"

_CACHE = {}


# --- shared ----------------------------------------------------------------------------------------
def _ckpt_dir(ckpt_dir):
    """A unit's checkpoint folder: a path, or a folder name under paths.CKPTS (e.g. "M26e")."""
    p = Path(ckpt_dir)
    for q in (p, paths.CKPTS / p):
        if (q / "manifest_model1.json").exists():
            return q
    raise FileNotFoundError(f"no manifest_model1.json in {ckpt_dir} (nor under {paths.CKPTS})")


def records():
    """Border records of the real fish (physical dish geometry), built once."""
    if "recs" not in _CACHE:
        MX._numpy2_unpickle_compat()
        _CACHE["recs"] = MB.build_border_records(E.load_bouts())
    return _CACHE["recs"]


def fold_models(ckpt_dir, arm, rej=REJ_BAND):
    """({fold file: model}, {held-out fish: fold file}) for the arm's fold checkpoints, the rej band
    merged into each model at load (no retraining)."""
    d = _ckpt_dir(ckpt_dir)
    MX._numpy2_unpickle_compat()
    mani = json.loads((d / "manifest_model1.json").read_text())
    entries = sorted((e for e in mani if e["file"].startswith(arm + "_fold")), key=lambda e: e["file"])
    if not entries:
        raise ValueError(f"no {arm}_fold* checkpoints in {d}")
    models = {}
    for e in entries:
        m = MX.load_mix(str(d / e["file"]))
        if rej is not None:
            m._cfg_feat = {**m._cfg_feat, "wall_reject": dict(rej)}
        models[e["file"]] = m
    held = {int(f): e["file"] for e in entries for f in e["heldout"]}
    return models, held


def label(arm, rej=REJ_BAND):
    return arm + ("_rej" if rej is not None else "")


# --- 1. individuality: the C2ST classifier's top features, per fish and pooled -----------------------
def c2st_feature_windows(ckpt_dir, arm, rej=REJ_BAND, reps=None, seed=0, progress=True,
                         windows="all"):
    """Per-window C2ST 'traj' features for every fish: its real windows and the rollouts of the fold
    model that never saw it (re-anchored, rej band on). `reps` = rollouts stacked per fish; a latent
    model draws a NEW individual (z) every rollout, so stacking would average two individuals and
    shrink exactly the between-fish spread the plot measures -> 1 for latent arms, 2 otherwise.
    `windows='interior'` = the steady-state selection (GM.INTERIOR: real windows well inside a run,
    generated windows after the warm-up) instead of every run-aware window."""
    if reps is None:
        reps = 1 if "_z" in arm else 2
    if windows not in ("all", "interior"):
        raise ValueError("windows must be 'all' or 'interior'")
    key = "traj_int" if windows == "interior" else "traj"
    recs = records()
    models, held = fold_models(ckpt_dir, arm, rej)
    real, gen = {}, {}
    empty = []
    for i, (f, ck) in enumerate(sorted(held.items()), 1):
        w = GM.c2st_windows_for_fish(models[ck], recs[f], f, seed=seed, reps=reps)[key]
        if len(w[0][0]) == 0:                  # no interior window (a pause-heavy fish)
            empty.append(f)
            continue
        real[f], gen[f] = w[0][0], np.vstack([g for _, g in w])
        if progress:
            print(f"\r  rolled out fish {i}/{len(held)}", end="", flush=True)
    if progress:
        print()
    if empty:
        print(f"  no {windows} window for fish {empty} -- left out of the plot")
    names = GM.traj_names_for_dim(next(iter(real.values())).shape[1])
    return real, gen, names


def plot_c2st_features(real, gen, names, title, feats=FEATS):
    """Top row: one point per fish (dot = that fish's mean over its windows, bar = its window-to-window
    SD), EACH side sorted by its own mean -- a generated fish has no link to a real one. Bottom row:
    all windows pooled, which is what the classifier sees. Prints the per-feature table."""
    fish = sorted(real)
    fig, ax = plt.subplots(2, len(feats), figsize=(4.1 * len(feats), 7.4))
    print(f"{'feature':<11}{'between-fish SD':>17}{'':>7}{'within-fish SD':>16}{'':>7}"
          f"{'fish-mean level':>17}{'':>7}{'pooled SD':>11}{'':>7}")
    print(f"{'':<11}" + "".join(f"{'real':>10}{'gen':>7}" for _ in range(3)) + f"{'real':>11}{'gen':>7}")
    for j, (col, lab) in enumerate(feats):
        k = names.index(col)
        mr = np.array([real[f][:, k].mean() for f in fish]); sr = np.array([real[f][:, k].std() for f in fish])
        mg = np.array([gen[f][:, k].mean() for f in fish]);  sg = np.array([gen[f][:, k].std() for f in fish])
        o, og = np.argsort(mr), np.argsort(mg)
        x = np.arange(len(fish))
        a = ax[0, j]
        a.errorbar(x - .16, mr[o], yerr=sr[o], fmt="o", ms=5, color=BLUE, capsize=2, lw=1, label="real")
        a.errorbar(x + .16, mg[og], yerr=sg[og], fmt="o", ms=5, color=ORANGE, capsize=2, lw=1,
                   label="generated")
        a.set_xticks(x); a.set_xticklabels(x + 1, fontsize=7)
        a.set_xlabel("rank (each side sorted by its own mean)"); a.set_title(lab, fontsize=10)
        a.grid(axis="y", color=GRID, lw=.6); a.set_axisbelow(True)
        a.text(.02, .98, f"between-fish SD  real {mr.std(ddof=1):.3g} | gen {mg.std(ddof=1):.3g}\n"
                         f"within-fish SD    real {sr.mean():.3g} | gen {sg.mean():.3g}",
               transform=a.transAxes, va="top", fontsize=7.5, color=INK2)
        pr = np.concatenate([real[f][:, k] for f in fish])
        pg = np.concatenate([gen[f][:, k] for f in fish])
        lo, hi = np.quantile(pr, [.005, .995]); pad = .25 * (hi - lo)
        b = np.linspace(lo - pad, hi + pad, 40)              # bins from REAL only: same for every model
        a = ax[1, j]
        a.hist(np.clip(pr, b[0], b[-1]), b, density=True, histtype="step", lw=2, color=BLUE)
        a.hist(np.clip(pg, b[0], b[-1]), b, density=True, histtype="step", lw=2, color=ORANGE)
        a.set_xlabel(lab); a.set_yticks([])
        a.text(.02, .98, f"pooled SD  real {pr.std():.3g} | gen {pg.std():.3g}", transform=a.transAxes,
               va="top", fontsize=7.5, color=INK2)
        print(f"{col:<11}{mr.std(ddof=1):>10.3g}{mg.std(ddof=1):>7.3g}{sr.mean():>10.3g}{sg.mean():>7.3g}"
              f"{mr.mean():>10.3g}{mg.mean():>7.3g}{pr.std():>11.3g}{pg.std():>7.3g}")
    ax[0, 0].set_ylabel("per-fish mean ± within-fish SD")
    ax[1, 0].set_ylabel("density, all windows pooled")
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{title}, top C2ST features: per fish (top) and as the classifier sees them (bottom)",
                 x=.02, ha="left", y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


def individuality_plot(ckpt_dir, arm, rej=REJ_BAND, reps=None, seed=0, windows="all"):
    """Roll out, plot, print. Returns (real windows, gen windows, feature names) per fish.
    `windows='interior'` = the steady-state windows (see c2st_feature_windows)."""
    real, gen, names = c2st_feature_windows(ckpt_dir, arm, rej, reps, seed, windows=windows)
    plot_c2st_features(real, gen, names,
                       label(arm, rej) + (" [steady state]" if windows == "interior" else ""))
    plt.show()
    return real, gen, names


# --- 1b. every window statistic of every channel: real fish next to simulated individuals -------------------------
# The report notebook's individuality view (Report_models.ipynb): the distribution over the classifier's 60-bout windows
# of mean / median / SD / q90 / lag-1 autocorrelation of radial position, interval, |turn| and displacement, plus each
# fish's (each simulated individual's) mean, so both what the classifier sees and the between-fish spread are visible.
WS_CHANNELS = (("rad", "radial position (r/R)"), ("dt", "interval (s)"), ("absdth", "|turn| (deg)"),
               ("dist", "displacement (mm)"))
WS_STATS = ("mean", "median", "SD", "q90", "lag-1 acf")
WS_PX_PER_MM = 3.60           # the report's scale (Chapter 3), for displacement in mm


def _window_stats(series, runs=None, W=60):
    """{channel: (n_windows, 5) [mean, median, SD, q90, lag-1 acf]} over the classifier's windows: non-overlapping
    W-bout windows, inside runs when run ids are given (``generative_metrics._window_starts``, ``_acf1``)."""
    n = len(next(iter(series.values())))
    starts = [a for a in GM._window_starts(n, W, W, runs) if a + W <= n]
    out = {}
    for ch, x in series.items():
        x = np.asarray(x, float)
        rows = []
        for a in starts:
            w = x[a:a + W]
            w = w[np.isfinite(w)]
            rows.append([w.mean(), np.median(w), w.std(), np.quantile(w, 0.9), GM._acf1(w)] if len(w) else [np.nan] * 5)
        out[ch] = np.asarray(rows, float).reshape(-1, 5)
    return out


def window_stats(ckpt_dir, arm, rej=REJ_BAND, reps=4, seed=0, progress=True):
    """Per-window statistics of the four channels for every held-out fish (its real windows, inside runs) and for
    ``reps`` rollouts of the fold model that never saw it -- each rollout a NEW individual (fresh latent draw), the
    stimulus re-anchored and the RNG keyed as the classifier's (so these are its first ``reps`` replicates).
    Returns (real, gen): real[fish][channel], gen[(fish, rep)][channel], each (n_windows, 5)."""
    recs = records()
    models, held = fold_models(ckpt_dir, arm, rej)
    real, gen = {}, {}
    for i, (f, ck) in enumerate(sorted(held.items()), 1):
        rec, m = recs[f], models[ck]
        real[f] = _window_stats(dict(rad=GM._radial_of(rec, rec.x_px, rec.y_px), dt=rec.dt_s,
                                     absdth=np.abs(np.asarray(rec.dtheta, float)),
                                     dist=np.asarray(rec.distance_px, float) / WS_PX_PER_MM),
                                getattr(rec, "run_id", None))
        for r in range(int(reps)):
            g = m.sample(rec, np.random.default_rng([int(seed), int(f), int(r)]), **GM._stim_kw(m, rec, None))
            gen[(f, r)] = _window_stats(dict(rad=GM._radial_of(rec, g["x_px"], g["y_px"]), dt=g["dt_s"],
                                             absdth=np.abs(np.asarray(g["dtheta"], float)),
                                             dist=np.asarray(g["distance_px"], float) / WS_PX_PER_MM))
        if progress:
            print(f"\r  rolled out fish {i}/{len(held)}", end="", flush=True)
    if progress:
        print()
    return real, gen


def plot_window_stats(real, gen, title):
    """4 channels x 5 statistics. Each panel: the distribution over all windows, real (blue) against simulated
    (orange) -- what the classifier sees -- and underneath one tick per real fish / simulated individual at its mean
    -- the individuality; in the corner the between-fish and within-fish SD, simulated / real. Returns (fig, table)."""
    fish, keys = sorted(real), sorted(gen)
    fig, ax = plt.subplots(len(WS_CHANNELS), len(WS_STATS), figsize=(3.9 * len(WS_STATS), 2.9 * len(WS_CHANNELS)))
    rows = []
    for i, (ch, chl) in enumerate(WS_CHANNELS):
        for j, st in enumerate(WS_STATS):
            a = ax[i, j]
            if ch == "rad" and j == 4:
                a.axis("off")
                a.text(.5, .5, "radial lag-1 acf:\nnot a classifier feature", ha="center", va="center", fontsize=8,
                       color=INK2, transform=a.transAxes)
                continue
            pr = np.concatenate([real[f][ch][:, j] for f in fish]); pg = np.concatenate([gen[k][ch][:, j] for k in keys])
            pr, pg = pr[np.isfinite(pr)], pg[np.isfinite(pg)]
            lo, hi = np.quantile(np.r_[pr, pg], [.005, .995])
            bins = np.linspace(lo, hi, 36)
            hr, _ = np.histogram(pr, bins, density=True); hg, _ = np.histogram(pg, bins, density=True)
            a.stairs(hr, bins, color=BLUE, lw=1.6, label="real windows")
            a.stairs(hg, bins, color=ORANGE, lw=1.6, label="simulated windows")
            top = max(hr.max(), hg.max(), 1e-12)
            mr = np.array([np.nanmean(real[f][ch][:, j]) for f in fish])
            mg = np.array([np.nanmean(gen[k][ch][:, j]) for k in keys])
            sr = np.array([np.nanstd(real[f][ch][:, j]) for f in fish])
            sg = np.array([np.nanstd(gen[k][ch][:, j]) for k in keys])
            a.plot(mr, np.full(len(mr), -0.09 * top), "|", color=BLUE, ms=9, mew=1.5)
            a.plot(mg, np.full(len(mg), -0.21 * top), "|", color=ORANGE, ms=8, mew=1.0)
            a.set_ylim(-0.3 * top, 1.12 * top); a.set_yticks([])
            b_ratio = np.nanstd(mg, ddof=1) / np.nanstd(mr, ddof=1)
            w_ratio = np.nanmean(sg) / np.nanmean(sr)
            a.set_title(f"{chl}: window {st}", fontsize=9)
            a.text(.98, .97, f"between x{b_ratio:.2f}\nwithin x{w_ratio:.2f}", transform=a.transAxes, ha="right",
                   va="top", fontsize=7.5, color=INK2)
            a.grid(axis="x", color=GRID, lw=.6); a.set_axisbelow(True)
            for s_ in ("top", "right", "left"):
                a.spines[s_].set_visible(False)
            rows.append(dict(channel=chl, stat=st, real_median=np.median(pr), sim_median=np.median(pg),
                             between_real=np.nanstd(mr, ddof=1), between_sim=np.nanstd(mg, ddof=1),
                             between_ratio=b_ratio, within_ratio=w_ratio))
    ax[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(f"{title}: distributions over 60-bout windows ({len(fish)} real fish, {len(keys)} simulated "
                 f"individuals); ticks = each fish's / individual's mean; between / within = SD ratio, simulated / real",
                 fontsize=10)
    fig.tight_layout()
    return fig, pd.DataFrame(rows).set_index(["channel", "stat"])


def window_stats_grid(ckpt_dir, arm, rej=REJ_BAND, reps=4, seed=0):
    """Roll out, plot the 4 x 5 grid, print the table. Returns (real, gen, table)."""
    real, gen = window_stats(ckpt_dir, arm, rej, reps, seed)
    _, tab = plot_window_stats(real, gen, label(arm, rej))
    plt.show()
    print(tab.round(3).to_string())
    return real, gen, tab


# --- 2. distance to the dots over the block: real vs programme rollouts ------------------------------
def parse_stim(stim):
    m = re.match(r"d(\d+)s([\d\.]+)$", stim)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)


def stim_label(stim):
    n, speed = parse_stim(stim)
    return f"{n} dot{'s' if n != 1 else ''} - {'static' if speed == 0 else 'motion'}"


STATIC = [s for s in STIMULI_ORDER if parse_stim(s)[1] == 0]
MOTION = [s for s in STIMULI_ORDER if parse_stim(s)[1] != 0]


def _color_map():
    # PuBu motion, OrRd static, darker = more dots, sampled from 35% to 95% of each colormap
    pal = lambda name, n: [plt.get_cmap(name)(x) for x in np.linspace(0.35, 0.95, n)]
    return {**dict(zip(sorted(MOTION, key=lambda s: parse_stim(s)[0]), pal("PuBu", len(MOTION)))),
            **dict(zip(sorted(STATIC, key=lambda s: parse_stim(s)[0]), pal("OrRd", len(STATIC))))}


def real_dot_profile():
    """Real stimulus bouts (active only) with cond_label, t_blk (s since the EXACT block onset),
    dist_mm to the dots' centroid and the 2-s bin; the centroid-outside-dish rule applied. Cached."""
    if "st" in _CACHE:
        return _CACHE["st"]
    d = pd.read_csv(paths.BOUTS)
    a = d[d.active].copy()
    a["n_dots"] = (a.n_dots_left.fillna(0) + a.n_dots_right.fillna(0)).astype(int)
    st = a[(a.stimulus != "inter_stim_pause") & a.dot_dist_px.notna()].copy()
    st["cond_label"] = [f"d{n}s{'3.5' if m else '0'}" for n, m in zip(st.n_dots, st.moving)]
    if EXCLUDE_CENTROID_OUT:
        recs = records()
        dish = pd.DataFrame({f: dict(cx=r.dish_cx, cy=r.dish_cy, R=r.dish_R) for f, r in recs.items()}).T
        g = st.join(dish, on="fish_id")
        out = (np.hypot(g.dot_cx - g.cx, g.dot_cy - g.cy) > g.R).to_numpy()
        st = st[~out]
    st["dist_mm"] = st.dot_dist_px / PX_TO_MM
    st["tbin"] = (np.floor(st.t_blk / BIN_S) + 0.5) * BIN_S
    _CACHE["st"] = st
    return st


def _programme():
    """(programme, simulated dish, real start-pose sampler), loaded once."""
    if "program" not in _CACHE:
        _CACHE["program"] = SP.load_program(str(PROGRAM))
        _CACHE["dish_sim"] = SR.default_dish()                          # median PHYSICAL dish of the 15
        _CACHE["ref"] = TR.load_reference(str(REFERENCE))     # real start-pose sampler
    return _CACHE["program"], _CACHE["dish_sim"], _CACHE["ref"]


# PER-BLOCK RANDOM STREAMS for every programme rollout while switched on (report Model 11): each block's draws restart
# from (rollout seed, block), so two models that differ only in how they treat one condition draw identical
# randomness in every other block. Off by default -- every archived profile number is unchanged.
BLOCK_SEED = False


@contextlib.contextmanager
def block_seeding(on=True):
    """``with block_seeding(): dot_profile(...)`` -- the profile, matched profile and attraction rollouts inside use
    per-block random streams."""
    global BLOCK_SEED
    prev, BLOCK_SEED = BLOCK_SEED, bool(on)
    try:
        yield
    finally:
        BLOCK_SEED = prev


def programme_bouts(model, k, r_per_model=3, seed=0):
    """Raw bout table of r_per_model programme rollouts of ONE fold model (rollout seed k): each
    simulated fish swims the whole recorded programme on its OWN clock (the stimulus advances with
    the intervals the model emits), the dot track re-anchored to its pose at every block onset --
    the rig's rule. fish_id = k * r_per_model + r, so folds concatenate without collisions.
    ``seed`` offsets the rollout streams (0 = the archived streams, bit-identical) so a second,
    independent set of simulated fish can serve as the profile's rollout-noise floor."""
    program, dish_sim, ref = _programme()
    with contextlib.redirect_stdout(io.StringIO()):                # rollout_many prints a line per fish
        rolls = SR.rollout_many(model, program, r_per_model, seed=k + 1000 * int(seed), dish=dish_sim,
                                pose_sampler=ref.sample_start_pose, on_budget="truncate",
                                **({"block_seed": True} if BLOCK_SEED else {}))
    rows = []
    # A RUNAWAY rollout (intervals collapsing; flagged by rollout_program) is dropped and COUNTED
    # (`.attrs['runaway']`), not scored: its tens of thousands of bouts would otherwise own every
    # bin of the profile, and not crashing the run keeps the other 89 fish.
    n_run = sum(bool(g.get("runaway")) for g in rolls)
    for r, g in enumerate(rolls):
        if g.get("runaway"):
            continue
        b = np.asarray(g["block_id"])
        rows.append(pd.DataFrame(dict(
            fish_id=k * r_per_model + r,                           # replicate unit = one simulated fish
            block_id=b, frame_onset=np.asarray(g["frame_onset"]),  # kept so a bout can be put back on the programme's grid
            t_blk=(np.asarray(g["frame_onset"]) - program.starts[b]) / program.fps,
            is_stim=program.is_stim_of[b], n_dots=program.n_dots_of[b], moving=program.moving_of[b],
            gx=g["x_px"], gy=g["y_px"], gcx=g["dot_cx"], gcy=g["dot_cy"],
            # the bout's turn (deg) and the bearing of the dots it saw (rad, travel frame): toward-turn over block time
            dth=np.asarray(g["dtheta"], float),
            phi=np.asarray(g.get("stim_angle_gen", np.full(len(b), np.nan)), float))))
    out = (pd.concat(rows, ignore_index=True) if rows
           else pd.DataFrame(columns=["fish_id", "t_blk", "is_stim", "n_dots", "moving", "gx", "gy", "gcx", "gcy"]))
    out.attrs["runaway"] = int(n_run)
    return out


def finish_profile(gen, n_models, r_per_model, progress=True):
    """Raw bout tables (concatenated over folds) -> stimulus bouts with cond_label, dist_mm, tbin;
    the centroid-outside-dish rule applied as for the real data."""
    _, dish_sim, _ = _programme()
    gen = gen[gen.is_stim & gen.gcx.notna()].copy()
    gen["cond_label"] = [f"d{int(n)}s{'3.5' if mv else '0'}" for n, mv in zip(gen.n_dots, gen.moving)]
    if EXCLUDE_CENTROID_OUT:
        cx, cy, R = dish_sim
        out = (np.hypot(gen.gcx - cx, gen.gcy - cy) > R).to_numpy()
        if progress:
            print(f"dropped {100 * out.mean():.1f}% of generated stimulus bouts: dot centroid outside the dish")
        gen = gen[~out]
    gen["dist_mm"] = np.hypot(gen.gx - gen.gcx, gen.gy - gen.gcy) / PX_TO_MM
    gen["tbin"] = (np.floor(gen.t_blk / BIN_S) + 0.5) * BIN_S
    gen.attrs.update(n_models=n_models, r_per_model=r_per_model)
    return gen


def generated_dot_profile(ckpt_dir, arm, rej=REJ_BAND, r_per_model=3, progress=True, seed=0):
    """Programme rollouts of every fold model of the arm (each a population model trained on 12 of
    the 15 fish), r_per_model simulated fish each. Distance is to the dots each rollout actually
    saw. ``seed`` offsets the rollout streams (see programme_bouts)."""
    models, _ = fold_models(ckpt_dir, arm, rej)
    parts = []
    for k, (ck, m) in enumerate(models.items()):
        parts.append(programme_bouts(m, k, r_per_model, seed=seed))
        if progress:
            print(f"\r  {ck}: {r_per_model} rollouts", end="", flush=True)
    if progress:
        print()
    n_run = sum(int(b.attrs.get("runaway", 0)) for b in parts)
    if n_run and progress:
        print(f"  {n_run} RUNAWAY rollout(s) dropped (intervals collapsing -- a model defect of that fold)")
    gen = finish_profile(pd.concat(parts, ignore_index=True), len(models), r_per_model, progress)
    gen.attrs["runaway"] = n_run
    return gen


BAND_LABEL = {"sem": "mean ± SEM across fish",
              "boot": "mean with 95 % bootstrap over fish"}


def _band(per_fish, tbins, band, n_boot, seed):
    """(mu, lo, hi) across fish for one condition.

    ``sem``  -- mean ± 1 SEM. A ~68 % interval, and the historical default.
    ``boot`` -- percentile bootstrap over the FISH, which is the replicate unit.
                Measured on this dataset the two agree to within ~4 % of width,
                so the real difference between them is the stated coverage, not
                the method; ``boot`` is preferred where a figure is read as a
                confidence interval.
    """
    M = per_fish.set_index(["fish_id", "tbin"]).dist_mm.unstack("tbin").reindex(
        columns=tbins).to_numpy(float)
    mu = np.nanmean(M, axis=0)
    if band == "sem":
        n = np.sum(np.isfinite(M), axis=0)
        se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        return mu, mu - se, mu + se
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M.shape[0], size=(n_boot, M.shape[0]))
    bs = np.array([np.nanmean(M[i], axis=0) for i in idx])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return mu, lo, hi


def _draw(df, ax, title, color_map, band="sem", n_boot=4000, seed=0):
    for stim in STIMULI_ORDER:
        s = df[df.cond_label == stim]
        if s.empty:
            continue
        per_fish = s.groupby(["fish_id", "tbin"]).dist_mm.mean().reset_index()   # over that fish's blocks
        tbins = np.sort(per_fish.tbin.unique())
        mu, lo, hi = _band(per_fish, tbins, band, n_boot, seed)
        ax.plot(tbins, mu, label=stim_label(stim), color=color_map[stim], lw=1.8)
        ax.fill_between(tbins, lo, hi, alpha=0.3, color=color_map[stim], edgecolor="none")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time since block onset (s)")
    ax.set_xlim(0, 60)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def plot_dot_profiles(st, gen, title, band="sem", n_boot=4000):
    """Real (left) and generated (right): one line per condition, per-fish mean over that fish's
    blocks first, then a band across fish -- ``band='sem'`` (±1 SEM, the historical default) or
    ``band='boot'`` (95 % percentile bootstrap over fish). Both sides always use the same one."""
    cm = _color_map()
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    _draw(st, ax[0], f"Real fish (active bouts), {st.fish_id.nunique()} fish", cm, band, n_boot)
    _draw(gen, ax[1], title, cm, band, n_boot)
    ax[0].set_ylabel("Distance (mm) between fish and dots centre")
    ax[1].legend(title="Stimulus", loc="upper left", fontsize=9, bbox_to_anchor=(1.02, 1),
                 borderaxespad=0., frameon=False, title_fontsize=10)
    fig.suptitle(f"Experiment 2025.02.20 — {BAND_LABEL[band]}", y=1.0)
    fig.tight_layout()
    return fig


# --- the profile plot in numbers ----------------------------------------------------------------------
# Computed from the SAME per-fish curves the plot draws. Per condition:
#   err mm    mean |gen - real| over the 0-60 s bins; err/SEM = the same in units of the real
#             across-fish SEM at each bin ("how many real SEMs off"; < 1 = inside the real band)
#   dip, lat  moving blocks: amplitude = level at 0-4 s minus the minimum within 20 s of the plotted
#             mean curve (3-bin running mean, so one noisy bin cannot set it), mm; lat = bin of that
#             minimum, s. Real: ~5 mm at 11-13 s for 6-8 dots, 2.6 mm for 2 dots, none for 1 dot
#   drift     static blocks: slope of the mean curve over 0-20 s, mm/s (real ~0.6)
#   plateau   per-fish mean over 20-60 s, mean +/- SEM across fish, mm (the steady-state set point)
# Summary lines: moving 2-8 dots (the shoal-approach conditions), static, and the dot-count grading
# plateau(1 dot) - plateau(8 dots) within moving (real ~ +6 mm, absent in the models so far).
# A gen curve can score a small err by sitting flat at the right mean while missing the transient,
# hence dip and lat next to it. Noise floor: the 8 even-numbered REAL fish against all 15 score
# err 1.3 mm = 0.8 real SEM, dip 3.5 vs 4.2 mm -- differences smaller than that are not resolved.
def _curves(df):
    per_fish = df[df.t_blk < 60].groupby(["cond_label", "fish_id", "tbin"]).dist_mm.mean()
    by_bin = per_fish.groupby(["cond_label", "tbin"])
    return per_fish, by_bin.mean().unstack("cond_label"), by_bin.sem().unstack("cond_label")


def _dip(mu):
    sm = mu[mu.index < 20].rolling(3, center=True, min_periods=1).mean()
    return float(mu[mu.index < 4].mean() - sm.min()), float(sm.idxmin())


def _plateau(per_fish, c):
    pf = per_fish.loc[c]
    pf = pf[pf.index.get_level_values("tbin") >= 20].groupby(level="fish_id").mean()
    return float(pf.mean()), float(pf.sem())


def _drift(mu):
    e = mu[mu.index < 20].dropna()
    return float(np.polyfit(e.index.to_numpy(float), e.to_numpy(float), 1)[0])




def real_block_weights(st, program):
    """(cond_label, tbin, block_id) -> number of REAL bouts there: how much each programme block
    contributes to each point of the real curve.

    A real fish only contributes a bout to a bin when it bouts then, and it pauses: about half the
    (fish, block) cells feed the 0-2 s bin and 127 of 1425 feed nothing at all. Pooled over the 15 fish
    every block of a condition is still represented in every bin, but with very unequal weight, while a
    rollout bouts in nearly every block of every replicate. The two sides therefore average over the
    same blocks in DIFFERENT PROPORTIONS, and since the
    dots' placement distance differs from block to block, the STIMULUS differs between the samples
    before any behaviour does (measured: 0.77 mm of the 1.37 mm first-bin gap at 2 moving dots).
    ``weighted_curves`` uses these weights to average the generated side over the same blocks in the
    same proportions. Blocks are resolved by FRAME (verified: 100 % of real bouts land in a programme
    block of their own condition), so no index alignment is assumed."""
    b = program.block_id[np.clip(np.asarray(st.frame_onset, int), 0, program.n_frames - 1)]
    return st.assign(block_id=b).groupby(["cond_label", "tbin", "block_id"], observed=True).size().rename("w")


def weighted_curves(gen, w):
    """``_curves`` for a generated table, but each bin's mean is taken over the SAME programme blocks,
    in the same proportions, as the real curve (``real_block_weights``). Generated bouts in blocks the
    real curve never covers in that bin are dropped. Returns the same triple as ``_curves``."""
    g = gen[gen.t_blk < 60]
    cell = (g.groupby(["cond_label", "tbin", "block_id", "fish_id"], observed=True)
            .dist_mm.mean().rename("d").reset_index()
            .merge(w.reset_index(), on=["cond_label", "tbin", "block_id"], how="inner"))
    cell["wd"] = cell.d * cell.w
    agg = cell.groupby(["cond_label", "fish_id", "tbin"], observed=True)[["wd", "w"]].sum()
    per_fish = (agg.wd / agg.w).rename("dist_mm")
    by_bin = per_fish.groupby(["cond_label", "tbin"])
    return per_fish, by_bin.mean().unstack("cond_label"), by_bin.sem().unstack("cond_label")




def profile_metrics(real, gen):
    """One row per condition (see the block comment above); `_pr` / `_pg` = plateau means."""
    pr, mr, sr = _curves(real)
    pg, mg, _ = _curves(gen)
    return metrics_from_curves(pr, mr, sr, pg, mg)


def metrics_from_curves(pr, mr, sr, pg, mg):
    """``profile_metrics`` from precomputed curves, so a REWEIGHTED generated curve
    (``weighted_curves``) can be scored by exactly the same code."""
    rows = []
    for c in STIMULI_ORDER:
        r, g, s = mr[c], mg[c].reindex(mr.index), sr[c].replace(0, np.nan)
        ok = r.notna() & g.notna()
        err = (g - r).abs()[ok]
        mv = parse_stim(c)[1] != 0
        (dr, lr), (dg, lg) = (_dip(r), _dip(g)) if mv else ((np.nan, np.nan), (np.nan, np.nan))
        (p_r, e_r), (p_g, e_g) = _plateau(pr, c), _plateau(pg, c)
        rows.append({"cond": c, "err mm": err.mean(), "err/SEM": (err / s[ok]).mean(),
                     "dip r": dr, "dip g": dg, "lat r": lr, "lat g": lg,
                     "drift r": np.nan if mv else _drift(r), "drift g": np.nan if mv else _drift(g),
                     "plateau r": f"{p_r:.1f} +/- {e_r:.1f}", "plateau g": f"{p_g:.1f} +/- {e_g:.1f}",
                     "_pr": p_r, "_pg": p_g})
    return pd.DataFrame(rows).set_index("cond")


# The headline numbers of a profile table, one row per arm in the rollout panel. `_real` twins
# carry the real value on the same bouts so a table of arms can show the target once.
PROFILE_SUMMARY = ["mv_err_mm", "mv_err_sem", "dip_gen", "dip_real", "lat_gen", "lat_real",
                   "plateau_gen", "plateau_real", "d1_plateau_gen", "d1_plateau_real",
                   "grading_gen", "grading_real", "st_err_mm", "st_err_sem", "drift_gen", "drift_real"]


def profile_summary(m):
    sh, stc = m.loc[MOVING_SHOAL], m.loc[STATIC]
    return {"mv_err_mm": sh["err mm"].mean(), "mv_err_sem": sh["err/SEM"].mean(),
            "dip_gen": sh["dip g"].mean(), "dip_real": sh["dip r"].mean(),
            "lat_gen": sh["lat g"].mean(), "lat_real": sh["lat r"].mean(),
            "plateau_gen": sh._pg.mean(), "plateau_real": sh._pr.mean(),
            "d1_plateau_gen": m.at["d1s3.5", "_pg"], "d1_plateau_real": m.at["d1s3.5", "_pr"],
            "grading_gen": m.at["d1s3.5", "_pg"] - m.at["d8s3.5", "_pg"],
            "grading_real": m.at["d1s3.5", "_pr"] - m.at["d8s3.5", "_pr"],
            "st_err_mm": stc["err mm"].mean(), "st_err_sem": stc["err/SEM"].mean(),
            "drift_gen": stc["drift g"].mean(), "drift_real": stc["drift r"].mean()}


def print_profile_metrics(real, gen):
    m = profile_metrics(real, gen)
    print("the plot in numbers (mm): err = mean |gen - real| over 0-60 s; dip amplitude / latency (moving);\n"
          "drift over 0-20 s in mm/s (static); plateau = 20-60 s, mean +/- SEM across fish")
    print(m.drop(columns=["_pr", "_pg"]).round(2).fillna("").to_string())
    s = profile_summary(m)
    print(f"\nmoving 2-8 dots: err {s['mv_err_mm']:.1f} mm = {s['mv_err_sem']:.1f} real SEM; "
          f"dip real {s['dip_real']:.1f} / gen {s['dip_gen']:.1f} mm at "
          f"{s['lat_real']:.0f} / {s['lat_gen']:.0f} s; "
          f"plateau real {s['plateau_real']:.1f} / gen {s['plateau_gen']:.1f} mm")
    print(f"static: err {s['st_err_mm']:.1f} mm = {s['st_err_sem']:.1f} real SEM; "
          f"drift-away 0-20 s real {s['drift_real']:.2f} / gen {s['drift_gen']:.2f} mm/s")
    print(f"dot-count grading, plateau(1 dot) - plateau(8 dots), moving: "
          f"real {s['grading_real']:+.1f} / gen {s['grading_gen']:+.1f} mm")
    return m


def dot_profile(ckpt_dir, arm, rej=REJ_BAND, r_per_model=3, band="sem"):
    """Roll out the programme, plot real vs generated, print the metrics. Returns (real bouts,
    generated bouts, metrics table). ``band`` picks the shading -- see plot_dot_profiles."""
    st = real_dot_profile()
    gen = generated_dot_profile(ckpt_dir, arm, rej, r_per_model)
    plot_dot_profiles(st, gen, f"Generated: {label(arm, rej)}, {gen.attrs['n_models']} fold models x "
                               f"{r_per_model} = {gen.fish_id.nunique()} simulated fish", band=band)
    plt.show()
    return st, gen, print_profile_metrics(st, gen)


# --- 3. after a pause, and the dip: real-data splits and teacher-forced conditionals -----------------------
# Everything here is either the real data cut a different way or the FIT read at the real feature rows
# (no rollout): Metric-C style, so a difference is the model's law, not rollout noise.
import torch                                    # noqa: E402
from twin import fit_MLP_joint as JT                     # noqa: E402

CI_Z = 1.96
AQUA, GREEN, PURPLE = "#1baf7a", "#4c9a52", "#9b59b6"




def conditional_kin(model, rec):
    """TEACHER-FORCED kinematic law per bout: ``(E[log distance], E[log dt])`` under the model's
    conditional given the real feature row and the real within-bout conditioners (the real turn for
    the step head, the real turn and step for the interval head). Population fish (z = 0), no drift
    shift -- the same reading ``conditional_turn`` gives for the turn head. exp() of either is the
    geometric mean the model predicts; the interval head's truncation (``dt_trunc``) is honoured."""
    from scipy.stats import norm
    cf = model._cfg_feat
    X = MB.build_feature_matrix(rec, cf)
    az, lc, lt = JT._cond_cols(rec.dtheta, rec.distance_px, cf["dth_scale"], cf["kin_scales"], y_dt=rec.dt_s)
    with torch.no_grad():
        _, dpar, tpar = model._model.joint_forward(torch.tensor(X), torch.tensor(az), torch.tensor(lc),
                                                   torch.tensor(lt))
    lw, mu, sg = [q.numpy().astype(float) for q in dpar]
    e_ld = (np.exp(lw) * mu).sum(1)
    lw, mu, sg = [q.numpy().astype(float) for q in tpar]
    hi = cf.get("dt_trunc")
    if hi:
        a = (np.log(float(hi)) - mu) / sg
        Phi = norm.cdf(a)
        w = np.exp(lw) * Phi; w = w / w.sum(1, keepdims=True)
        e_lt = (w * (mu - sg * norm.pdf(a) / np.maximum(Phi, 1e-12))).sum(1)   # truncated-normal mean of log dt
    else:
        e_lt = (np.exp(lw) * mu).sum(1)
    return e_ld, e_lt








# --- Model 30: the profile plateau against a STIMULUS-BLIND baseline --------------------------------
def _null_plateau(df, rng, reps=6, cap=40000):
    """Decouple behaviour from stimulus and re-measure the plateau. ``df`` = one side's stimulus bouts
    with cond_label / tbin / block_id / fx, fy (fish, dish-centred) / dx, dy (dot centroid, same frame).

    For every condition, each position is scored against the dot geometry of a RANDOM OTHER block of
    that condition at the same within-block time. The same positions feed every condition, so whatever
    grading survives is geometry, not behaviour. MEASURED on the real fish: the null is flat at
    23.1-23.9 mm across all five moving conditions (grading 0.29 mm of the real 6.00), so the profile's
    dot-count grading IS behavioural -- and the flat baseline is then the useful part, because
    ``baseline - plateau`` turns each condition into an ATTRACTION in mm.
    """
    g = (df[df.tbin < 60].groupby(["block_id", "cond_label", "tbin"], observed=True)[["dx", "dy"]]
         .mean().reset_index())
    lut = {(int(b), float(t)): (x, y) for b, t, x, y in zip(g.block_id, g.tbin, g.dx, g.dy)}
    blocks = {c: np.array(sorted(v.block_id.unique())) for c, v in g.groupby("cond_label", observed=True)}
    pos = df[(df.tbin >= 20) & (df.tbin < 60)][["fish_id", "block_id", "tbin", "fx", "fy"]]
    if len(pos) > cap:
        pos = pos.sample(cap, random_state=int(seed_of(rng)))
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
                    rows.append((cond, int(fid), float(np.hypot(fx - p[0], fy - p[1]) / PX_TO_MM)))
    n = pd.DataFrame(rows, columns=["cond_label", "fish_id", "d"])
    return n.groupby(["cond_label", "fish_id"]).d.mean().groupby("cond_label").mean()


def seed_of(rng):
    return int(rng.integers(0, 2 ** 31 - 1))


def attraction_table(ckpt_dir, arms, rej=REJ_BAND, r_per_model=6, reps=6, seed=0, gens=None):
    """PLATEAU AS AN ATTRACTION: how much closer than a stimulus-blind fish, per dot count.

    The raw plateau mixes behaviour with where the dot centroid happens to sit, and the two are not
    separable by eye. Each side here gets its OWN null (its own positions, that side's geometry,
    decoupled), so ``null - plateau`` is what the stimulus is worth in mm for that model.

    THE READ (measured 2026-09-24, real vs the Model 29 best arm):
        N              1     2     4     6     8
        real         1.2   6.2   5.1   6.4   6.9     <- a STEP: nothing for one dot, ~6 flat from two
        popd_g3h    -1.3   2.4   4.0   5.9   6.7     <- a RAMP, fitted by -1.3 + 8 nz to R^2 ~ 0.99
    so the model reads dot count straight off the magnitude of its stimulus code. An arm that fixes
    this has N=2 well above 2.4 and stops being linear in ``nz``.
    """
    rng = np.random.default_rng(seed)
    prog, dish_sim, _ = _programme()
    st = real_dot_profile().copy()
    st["block_id"] = prog.block_id[np.clip(np.asarray(st.frame_onset, int), 0, prog.n_frames - 1)]
    recs = records()
    dish = pd.DataFrame({f: dict(cx=r.dish_cx, cy=r.dish_cy) for f, r in recs.items()}).T
    st = st.join(dish, on="fish_id")
    st["fx"], st["fy"] = st.x_px - st.cx, st.y_px - st.cy
    st["dx"], st["dy"] = st.dot_cx - st.cx, st.dot_cy - st.cy
    sides = {"real": st}
    gens = dict(gens or {})
    for a in ([arms] if isinstance(arms, str) else arms):
        gen = gens.get(a)
        if gen is None:
            gen = generated_dot_profile(ckpt_dir, a, rej, r_per_model, progress=False, seed=seed)
            gens[a] = gen
        cx, cy, _ = dish_sim
        gen = gen.assign(fx=gen.gx - cx, fy=gen.gy - cy, dx=gen.gcx - cx, dy=gen.gcy - cy,
                         fish_id=gen.get("fish_id", gen.get("rollout", 0)))
        sides[a] = gen
    out = {}
    for name, d in sides.items():
        plateau = (d[(d.tbin >= 20) & (d.tbin < 60)].groupby(["cond_label", "fish_id"]).dist_mm.mean()
                   .groupby("cond_label").mean())
        null = _null_plateau(d, np.random.default_rng(seed), reps=reps)
        out[(name, "plateau")] = plateau
        out[(name, "null")] = null
        out[(name, "attraction")] = null - plateau
    t = pd.DataFrame(out)
    t = t.loc[[c for c in STIMULI_ORDER if c in t.index and parse_stim(c)[1] != 0]]
    pd.set_option("display.width", 250)
    print("PLATEAU (20-60 s) vs a STIMULUS-BLIND NULL, mm. attraction = null - plateau = what the dots are "
          "worth.\nReal is a STEP (~0 for one dot, ~6 from two up); a code that reads count off magnitude "
          "gives a RAMP.")
    print(t.round(2).to_string())
    return t, gens


# --- Model 31: the two teacher-forced reads the local code / drift / slow clock are judged on first -----
def toward_by_n(ckpt_dir, arms, rej=REJ_BAND, windows=((0, 20), (4, 20)), min_n=15):
    """Turn TOWARD the moving dots by group size, teacher-forced on the held-out fish (z = 0), per fish then
    mean, in windows of time since the block onset. Two uses:
    * BEARING REGAINED: popl drops the global mv_sin/mv_cos (zeroing them in the fitted popn removed 73 %
      of its turn toward the dots) for per-direction [sin, cos] that sum back to them. A popl arm that
      learned them reads like popn_g3h at 0-20 s (9.8 / 11.3 / 13.5 / 14.1 / 16.5 deg at N = 1/2/4/6/8),
      not like popn with them zeroed (0.9 / 2.1 / 4.0 / 4.9 / 6.0).
    * THE DIP LASTING: the 4-20 s window is what a slow second block clock (tau 15 s) should lift toward
      real; the 4 s clock is down to 0.37 at 4 s and 0.14 by 8 s."""
    recs = records()
    arms = [arms] if isinstance(arms, str) else list(arms)
    out = {}
    for arm in arms:
        models, held = fold_models(ckpt_dir, arm, rej)
        rows = []
        for f, ck in sorted(held.items()):
            r, m = recs[f], models[ck]
            e, _, _ = m.conditional_turn(r)
            phi = np.asarray(getattr(r, "stim_angle_travel", r.stim_angle), float); sgn = np.sign(phi)
            mv = np.asarray(r.moving, bool); nd = np.nan_to_num(np.asarray(r.n_dots, float))
            dth = np.asarray(r.dtheta, float)
            s0, _, fr = MB.block_onsets(r); tb = (fr - s0) / MB._FPS
            ok = mv & (nd > 0) & np.isfinite(phi) & (sgn != 0)
            for lo, hi in windows:
                for N in (1, 2, 4, 6, 8):
                    k = ok & (nd == N) & (tb >= lo) & (tb < hi)
                    if k.sum() >= min_n:
                        rows.append(dict(win=f"{lo}-{hi} s", N=N, real=float(np.mean(dth[k] * sgn[k])),
                                         model=float(np.mean(e[k] * sgn[k]))))
        d = pd.DataFrame(rows).groupby(["win", "N"]).mean()
        out.setdefault("real", d.real)
        out[arm] = d.model
    t = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print("TURN TOWARD the moving dots by N (deg), teacher-forced, held-out fish, z = 0; per fish then mean.")
    print("Reference, 0-20 s: popn_g3h 9.8 / 11.3 / 13.5 / 14.1 / 16.5; popn with mv_sin/mv_cos ZEROED "
          "0.9 / 2.1 / 4.0 / 4.9 / 6.0 -- a popl arm near the first has regained the bearing.")
    print(t.round(2).to_string())
    return t




# --- Which head carries the approach gap: every stimulus condition, teacher-forced ------------------------------
# The FIT read at the real feature rows (no rollout), per held-out fish, per stimulus condition and block phase: the
# turn toward the dots, the turn size, the turn size with the dots AHEAD (|bearing| < 30 deg, >= 60 px), the step and
# the interval -- real against each model's conditional expectation on the SAME bouts (population fish, z = 0). Block
# time is the EXACT programme onset and the centroid-outside-dish rule is the profile's (``real_dot_profile``);
# 'none' = the pause bouts, the kinematic baseline without dots.
HEAD_PHASES = (("0-4 s", 0.0, 4.0), ("4-20 s", 4.0, 20.0), ("20-60 s", 20.0, 60.0))
HEAD_QTY = (("toward", "turn toward the dots (deg)"), ("absdth", "turn size |Δθ| (deg)"),
            ("ahead", "turn size, dots ahead ≥ 60 px (deg)"), ("step", "step (mm, geometric mean)"),
            ("dt", "interval (s, geometric mean)"))


def head_gap_rows(ckpt_dir, arm, rej=REJ_BAND, min_n=10):
    """Per (held-out fish, condition, phase): the real and the model means of the ``HEAD_QTY`` quantities."""
    recs = records()
    models, held = fold_models(ckpt_dir, arm, rej)
    st = real_dot_profile()
    d = pd.read_csv(paths.BOUTS, usecols=["fish_id", "frame_onset", "active", "stimulus"])
    pause = d[d.active & (d.stimulus == "inter_stim_pause")][["fish_id", "frame_onset"]]
    rows, n_st, n_hit = [], 0, 0
    for f, ck in sorted(held.items()):
        r, m = recs[f], models[ck]
        e_dth, e_abs, _ = m.conditional_turn(r)
        e_ld, e_lt = conditional_kin(m, r)
        phi = getattr(r, "stim_angle_travel", None)
        phi = np.asarray(phi if phi is not None else r.stim_angle, float)
        rng = np.asarray(r.dot_dist_px, float)
        per = pd.DataFrame(dict(
            frame_onset=np.asarray(r.frame_onset), dth=np.asarray(r.dtheta, float), e_dth=e_dth, e_abs=e_abs,
            sgn=np.sign(np.nan_to_num(phi)),
            ahead=(np.abs(np.degrees(np.nan_to_num(phi, nan=np.pi))) < 30.0) & (np.nan_to_num(rng) >= 60.0),
            lstep=np.log(np.clip(np.asarray(r.distance_px, float), 1e-6, None)), e_lstep=e_ld,
            lint=np.log(np.clip(np.asarray(r.dt_s, float), 1e-6, None)), e_lint=e_lt))
        sf = st[st.fish_id == f][["frame_onset", "cond_label", "t_blk"]]
        s = sf.merge(per, on="frame_onset")
        n_st += len(sf); n_hit += len(s)
        s["phase"] = None
        for lab, lo, hi in HEAD_PHASES:
            s.loc[(s.t_blk >= lo) & (s.t_blk < hi), "phase"] = lab
        s = s[s.phase.notna()]
        p = pause[pause.fish_id == f].merge(per, on="frame_onset").assign(cond_label="none", phase="pause")
        for (cond, ph), g in pd.concat([s, p], ignore_index=True).groupby(["cond_label", "phase"]):
            if len(g) < min_n:
                continue
            row = dict(fish=f, cond=cond, phase=ph, n=len(g),
                       absdth_real=np.mean(np.abs(g.dth)), absdth_model=g.e_abs.mean(),
                       step_real=np.exp(g.lstep.mean()) / WS_PX_PER_MM,
                       step_model=np.exp(g.e_lstep.mean()) / WS_PX_PER_MM,
                       dt_real=np.exp(g.lint.mean()), dt_model=np.exp(g.e_lint.mean()))
            if cond != "none":
                row.update(toward_real=np.mean(g.dth * g.sgn), toward_model=np.mean(g.e_dth * g.sgn))
                a = g[g.ahead.astype(bool)]
                if len(a) >= min_n:
                    row.update(ahead_real=np.mean(np.abs(a.dth)), ahead_model=a.e_abs.mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["matched_share"] = n_hit / max(n_st, 1)          # real stimulus bouts found in the records
    return out


def head_gaps(models, rej=REJ_BAND, min_n=10, plot=True):
    """``models`` = {label: (ckpt_dir, arm)}. Per model, condition, phase and quantity: the real mean across the
    held-out fish, the model's, and the PAIRED gap (model - real on the same bouts, per fish) with its 95 % CI.
    Prints where the gaps are and returns (per-fish rows, summary)."""
    parts = []
    for lab, (d, arm) in models.items():
        rr = head_gap_rows(d, arm, rej, min_n)
        print(f"{lab}: {rr.attrs['matched_share']:.1%} of the real stimulus bouts matched to the records")
        parts.append(rr.assign(model=lab))
    D = pd.concat(parts, ignore_index=True)
    out = []
    for (lab, cond, ph), g in D.groupby(["model", "cond", "phase"], sort=False):
        for q, _ in HEAD_QTY:
            if f"{q}_real" not in g:
                continue
            gg = g[[f"{q}_real", f"{q}_model"]].dropna()
            if len(gg) < 3:
                continue
            diff = gg[f"{q}_model"] - gg[f"{q}_real"]
            out.append(dict(model=lab, cond=cond, phase=ph, qty=q, fish=len(gg), real=gg[f"{q}_real"].mean(),
                            real_ci=CI_Z * gg[f"{q}_real"].std(ddof=1) / np.sqrt(len(gg)),
                            model_val=gg[f"{q}_model"].mean(), gap=diff.mean(),
                            gap_ci=CI_Z * diff.std(ddof=1) / np.sqrt(len(gg))))
    S = pd.DataFrame(out)
    S["gap_over_ci"] = S.gap / S.gap_ci
    pd.set_option("display.width", 250)
    conds = [c for c in STIMULI_ORDER + ["none"] if c in set(S.cond)]
    phases = [p for p, _, _ in HEAD_PHASES] + ["pause"]
    labs = list(models)
    for q, name in HEAD_QTY:
        tq = S[S.qty == q]
        if tq.empty:
            continue
        cols = {}
        for ph in phases:
            tp = tq[tq.phase == ph]
            if tp.empty:
                continue
            cols[(ph, "real")] = tp[tp.model == labs[0]].set_index("cond").real
            for lab in labs:
                cols[(ph, f"gap {lab}")] = tp[tp.model == lab].set_index("cond").gap
        tab = pd.DataFrame(cols).reindex(conds).dropna(how="all")
        print(f"\n{name}: real fish (mean over held-out fish) and each model's gap = model - real (paired, same bouts)")
        print(tab.round(2).to_string())
    for lab in labs:
        t = S[(S.model == lab) & (S.cond != "none")]
        big = t.reindex(t.gap_over_ci.abs().sort_values(ascending=False).index)
        print(f"\n{lab}: the 8 largest gaps relative to their 95 % CI (|ratio| > 1 = outside the CI)")
        print(big.head(8)[["cond", "phase", "qty", "real", "model_val", "gap", "gap_ci", "gap_over_ci"]]
              .round(2).to_string(index=False))
    if plot:
        plot_head_gaps(D, S, list(models))
        plt.show()
    return D, S


def plot_head_gaps(D, S, labels):
    """Rows = the HEAD_QTY quantities, columns = the block phases; x = stimulus condition. Real = black, mean and
    95 % CI across the held-out fish; each model = its conditional expectation on the same bouts. The pause
    baseline ('none') is drawn as horizontal lines (real solid, models dotted)."""
    conds = STIMULI_ORDER
    cols = [ORANGE, BLUE, AQUA, PURPLE]
    fig, axes = plt.subplots(len(HEAD_QTY), len(HEAD_PHASES), figsize=(4.6 * len(HEAD_PHASES), 2.6 * len(HEAD_QTY)),
                             sharey="row")
    x = np.arange(len(conds))
    for i, (q, name) in enumerate(HEAD_QTY):
        for j, (ph, _, _) in enumerate(HEAD_PHASES):
            ax = axes[i, j]
            s = S[(S.qty == q) & (S.phase == ph)]
            real = s[s.model == labels[0]].set_index("cond").reindex(conds)
            ax.errorbar(x, real.real, yerr=real.real_ci, fmt="o", color="black", ms=4, capsize=2, label="real")
            for k, lab in enumerate(labels):
                mm = s[s.model == lab].set_index("cond").reindex(conds)
                ax.plot(x + 0.15 * (k + 1), mm.model_val, "D", color=cols[k % len(cols)], ms=4, label=lab)
            base = S[(S.qty == q) & (S.cond == "none")]
            if len(base):
                ax.axhline(base.real.iloc[0], color="black", lw=0.8, alpha=0.5)
                for k, lab in enumerate(labels):
                    b = base[base.model == lab]
                    if len(b):
                        ax.axhline(b.model_val.iloc[0], color=cols[k % len(cols)], lw=0.8, ls=":", alpha=0.8)
            ax.axvline(4.5, color=GRID, lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace("s3.5", " mv").replace("s0", " st").replace("d", "") for c in conds],
                               fontsize=8)
            ax.grid(axis="y", color=GRID, lw=0.6)
            if i == 0:
                ax.set_title(f"{ph} after block onset", fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Each head against real fish, teacher-forced on the same bouts (held-out fish; x = dot count, "
                 "mv = moving, st = static; lines = pause baseline)", fontsize=10)
    fig.tight_layout()
    return fig
