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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

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

import missing_pieces as P  # noqa: E402

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


# --------------------------------------------------------------------------- head-gap compute
def _arm_of(key):
    arms = sorted(p.name[:-len("_fold0.pt")] for p in (MODEL / "report_ckpts" / f"M{key}").glob("*_fold0.pt"))
    assert len(arms) == 1, (key, arms)
    return arms[0]


def _hg_job(key):
    """One report unit: model_figures.head_gaps, teacher-forced on its held-out fish (as report_models' 'hgap' cell)."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    torch.set_num_threads(1)
    import model_figures as MF
    arm = _arm_of(key)
    with contextlib.redirect_stdout(io.StringIO()):
        D, S = MF.head_gaps({key: (str(MODEL / "report_ckpts" / f"M{key}"), arm)}, plot=False)
    return key, arm, D, S


def build_gaps(workers=7):
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = [k for _, k0, k1 in GAP_UNITS for k in (k0, k1)]
    parts = {}
    with ProcessPoolExecutor(min(workers, 8)) as ex:
        for key, arm, D, S in ex.map(_hg_job, keys):
            parts[key] = dict(arm=arm, D=D, S=S)
            print(f"  head gaps done: {key} ({arm}), {D.fish.nunique()} fish", flush=True)
    with open(HG_CACHE, "wb") as fh:
        pickle.dump(parts, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cached {HG_CACHE}")
    return parts


def gap_data(parts):
    """One row per unit x seed x gap: mean, fish, t-interval, head_gaps' own z-interval; checked against the
    archived summary_M<key>.json value (rounded to 2 dp there)."""
    rows = []
    for name, k0, k1 in GAP_UNITS:
        for seed, k in ((0, k0), (101, k1)):
            D = parts[k]["D"]
            s = P.summary(k)
            for g, (cond, ph, q), _ in GAPS:
                d = D[(D.cond == cond) & (D.phase == ph)][["fish", f"{q}_real", f"{q}_model"]].dropna()
                diff = (d[f"{q}_model"] - d[f"{q}_real"]).to_numpy(float)
                n = len(diff)
                m, sd = diff.mean(), diff.std(ddof=1)
                t = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
                Srow = parts[k]["S"]
                Srow = Srow[(Srow.cond == cond) & (Srow.phase == ph) & (Srow.qty == q)].iloc[0]
                assert abs(Srow.gap - m) < 1e-9 and Srow.fish == n
                assert abs(round(m, 2) - s[g]) < 0.0051, (k, g, m, s[g])      # = the archived summary
                rows.append(dict(model=name, seed=seed, key=k, gap=g, mean=m, n_fish=n, t_half=t,
                                 z_half=float(Srow.gap_ci), real=float(Srow.real), real_z_half=float(Srow.real_ci),
                                 model_val=float(Srow.model_val), archived=s[g],
                                 fish=sorted(int(f) for f in d.fish)))
    G = pd.DataFrame(rows)
    # the fish entering a cell are fixed by the REAL bouts (>= 10 per fish), so the same for every unit and seed
    for g, *_ in GAPS:
        assert G[G.gap == g].fish.map(tuple).nunique() == 1, g
    return G


STEPS = [("M1", "M3"), ("M3", "M4"), ("M4", "M5"), ("M5", "M6"), ("M3", "M6"), ("D4", "F"), ("M6", "F")]


def model_steps(parts):
    """Model-to-model change of each head gap, paired per fish within a seed (the real side cancels):
    mean and 95 % t half-width across the same fish. Not drawn; for the caption and the prose."""
    keys = {name: (k0, k1) for name, k0, k1 in GAP_UNITS}
    out = {}
    for a, b in STEPS:
        for si, seed in enumerate((0, 101)):
            for g, (cond, ph, q), _ in GAPS:
                def sel(k):
                    D = parts[k]["D"]
                    return D[(D.cond == cond) & (D.phase == ph)].set_index("fish")[f"{q}_model"]
                x = (sel(keys[b][si]) - sel(keys[a][si])).dropna().to_numpy(float)
                n = len(x)
                out[f"{b}-{a}|s{seed}|{g}"] = dict(
                    mean=float(x.mean()), t_half=float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)),
                    n_fish=n)
    return out


# --------------------------------------------------------------------------- maps and laws
def map_data():
    """{label: map} for real, M6 s0/s101, F s0/s101 (law_maps values) and the boolean mask of thin cells."""
    out = {}
    for lab in P.HEAD:
        real, model = P.law_maps(lab, True, N_MAP)
        out.setdefault("real", real)
        assert np.allclose(out["real"].values, real.values, equal_nan=True)   # same bouts at every label
        out[lab] = model
    t = P._toward_cols(P.load("laws"))
    t = t[(t.model == "M6 s0") & t.moving & (t.phase == ">= 10 s") & t.n.isin(list(N_MAP))]
    nb = t.groupby(["fish", "bear", "range"], observed=True).size()
    nfish = (nb >= MIN_BOUTS).groupby(["bear", "range"], observed=True).sum().unstack("range")
    nfish = nfish.reindex(index=out["real"].index, columns=out["real"].columns).fillna(0)
    mask = nfish < MIN_FISH
    keep = ~mask.all(axis=0)                          # a range column with no usable cell is not drawn
    out = {k: v.loc[:, keep] for k, v in out.items()}
    vals = np.concatenate([v.values[~mask.loc[:, keep].values].ravel() for v in out.values()])
    assert np.nanmin(vals) >= 0 and np.nanmax(vals) <= VMAX, (np.nanmin(vals), np.nanmax(vals))  # sequential 0-30
    return out, mask.loc[:, keep], nfish


def law_per_fish():
    """Per-fish values of the law_summary quantities; same selections as missing_pieces.law_summary."""
    t = P._toward_cols(P.load("laws"))
    deg = np.degrees(np.abs(t.phi))
    t = t[(deg >= 45) & (deg <= 135) & (t.rng_mm >= 10) & (t.rng_mm < 35)]
    per = {}
    for lab in P.HEAD:
        tt = t[t.model == lab]
        mv, st = tt.moving, tt.phase == ">= 10 s"
        for col, who in (("tow_real", "real"), ("tow_model", lab)):
            def pf(sel):
                return tt[sel].groupby("fish")[col].mean()
            d = pd.DataFrame(dict(onset_gain=pf(mv & (tt.phase == "0-4 s")), steady_gain=pf(mv & st),
                                  count_step=pf(mv & st & (tt.n >= 2)) - pf(mv & st & (tt.n == 1)),
                                  static_gain=pf(~mv & st)))
            if who == "real" and "real" in per:
                assert np.allclose(per["real"].values, d.values, equal_nan=True)
            per[who] = d
    if P.have("laws_draw"):                      # per-fish laws on single turns DRAWN from each model (one per draw)
        td = P._toward_cols(P.load("laws_draw"))
        deg = np.degrees(np.abs(td.phi))
        td = td[(deg >= 45) & (deg <= 135) & (td.rng_mm >= 10) & (td.rng_mm < 35)]
        for lab in P.HEAD:
            tt = td[td.model == lab]
            mv, st = tt.moving, tt.phase == ">= 10 s"
            sg = np.sign(tt.phi)
            reps = []
            for j in range(P.N_DRAW):
                col = tt[f"draw{j}"] * sg

                def pf(sel):
                    return col[sel].groupby(tt.fish[sel]).mean()
                reps.append(pd.DataFrame(dict(onset_gain=pf(mv & (tt.phase == "0-4 s")), steady_gain=pf(mv & st),
                                              count_step=pf(mv & st & (tt.n >= 2)) - pf(mv & st & (tt.n == 1)),
                                              static_gain=pf(~mv & st))))
            per[lab + "|draws"] = reps
    return per


# --------------------------------------------------------------------------- drawing
# sequential, one hue light -> dark (ColorBrewer Purples, its near-white end trimmed); purple is not a model colour
CMAP = LinearSegmentedColormap.from_list("seqpurple", plt.get_cmap("Purples")(np.linspace(0.04, 0.95, 256)))


def _lum(c):
    c = np.array(to_rgb(c))
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _bin_lab(iv):
    return f"{iv.left:g}–{iv.right:g}"


def draw_map(ax, M, mask, show_y, contrasts):
    v = M.values.astype(float)
    nr, nc = v.shape
    shown = np.ma.masked_array(v, mask=mask.values | ~np.isfinite(v))
    ax.set_facecolor("#ececec")
    im = ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), shown, cmap=CMAP, vmin=0, vmax=VMAX,
                       edgecolors="white", linewidth=0.6)
    for (i, j), x in np.ndenumerate(v):
        if mask.values[i, j] or not np.isfinite(x):
            ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", fontsize=7, color=MUTED)
            continue
        cell = CMAP(x / VMAX)
        ink, white = _contrast(cell, LABEL_DARK), _contrast(cell, WHITE)
        col = LABEL_DARK if ink >= white else WHITE
        contrasts.append(max(ink, white))
        ax.text(j + 0.5, i + 0.5, f"{x:.0f}", ha="center", va="center", fontsize=7, color=col)
    ax.set_xticks(np.arange(nc) + 0.5); ax.set_xticklabels([_bin_lab(c) for c in M.columns])
    ax.set_yticks(np.arange(nr) + 0.5); ax.set_yticklabels([_bin_lab(r) for r in M.index] if show_y else [])
    ax.set_xlim(0, nc); ax.set_ylim(0, nr)
    ax.tick_params(length=0, pad=2)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("range to the dots (mm)", labelpad=1.5)
    if show_y:
        ax.set_ylabel("|bearing| (deg)", labelpad=1.5)
    return im


def draw_gaps(fig, spec, G):
    """(e) forest plot: rows = units (main line, then the drift branch), one column per gap."""
    names = [u[0] for u in GAP_UNITS]
    y = np.arange(len(names), dtype=float)
    y[names.index("D4"):] += 0.6                      # gap between the main line and the drift branch
    sub = spec.subgridspec(1, len(GAPS), wspace=0.16)
    axes = []
    for j, (g, _, xlab) in enumerate(GAPS):
        ax = fig.add_subplot(sub[0, j])
        axes.append(ax)
        T = G[G.gap == g]
        for seed, dy in ((0, -0.17), (101, 0.17)):
            s = T[T.seed == seed].set_index("model").reindex(names)
            for i, nm in enumerate(names):
                c = UNIT_C.get(nm, OTHER_C)
                m, h = s.loc[nm, "mean"], s.loc[nm, "t_half"]
                ax.errorbar(m, y[i] + dy, xerr=h, fmt="o", color=c, ms=3.0, mfc=c if seed == 0 else "white",
                            mec=c, mew=0.8, elinewidth=0.7, capsize=0, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        ax.axhline((y[names.index("M6")] + y[names.index("D4")]) / 2, color=FAINT, lw=0.6, zorder=0)
        ax.set_ylim(y[-1] + 0.6, y[0] - 1.35)           # an empty top row for the fish count
        ax.set_yticks(y)
        ax.set_yticklabels(names if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", length=2, pad=1.5)
        ax.grid(axis="x"); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xlab, labelpad=1.5, fontsize=7)
        n = int(T.n_fish.iloc[0])
        ax.text(0.97, 0.985, f"{n} fish", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=MUTED,
                zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    lo = G.assign(lo=G["mean"] - G.t_half, hi=G["mean"] + G.t_half).groupby("gap")[["lo", "hi"]].agg(
        {"lo": "min", "hi": "max"})
    for ax, (g, *_) in zip(axes, GAPS):
        a, b = lo.loc[g, "lo"], lo.loc[g, "hi"]
        pad = 0.06 * (b - a)
        ax.set_xlim(min(a, 0) - pad, max(b, 0) + pad)
    return axes


def draw_laws(axd, per, legend_out=False):
    """(d) law summary: real fish as the per-fish mean with its 95 % bootstrap CI over fish (bar + band across the
    law's slot); M6/F at both seeds as the teacher-forced EXPECTED turn (population latent) on the same bouts. The
    model's interval is built like the real one, on single turns DRAWN from its angle mixture (one per bout, per-fish
    means, bootstrap over fish), averaged over P.N_DRAW draws as offsets from each draw's own mean and placed on the
    expected value. It carries the scatter of single turns but no between-fish differences (population latent)."""
    offs = np.linspace(-0.28, 0.28, len(SOURCES))
    law_vals = {}
    for i, (q, _) in enumerate(LAWS):
        for o, (src, name, c, seed) in zip(offs, SOURCES):
            v = per[src][q]
            m, (lo, hi) = float(v.mean()), P._boot_ci(v.values)
            law_vals[f"{src}|{q}"] = dict(mean=m, ci=[lo, hi], n_fish=int(v.notna().sum()))
            if src == "real":
                axd.fill_between([i - 0.42, i + 0.42], lo, hi, color=FAINT, alpha=0.55, lw=0, zorder=0.5)
                axd.errorbar(i + o, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=3.4, mfc=c, mec=c,
                             mew=0.8, elinewidth=0.8, capsize=0, zorder=3)
                continue
            reps = per[src + "|draws"]
            off = np.array([(lambda x: (lambda a, b: (a - x.mean(), b - x.mean()))(*P._boot_ci(x.values)))(r[q])
                            for r in reps])
            dlo, dhi = off.mean(0)
            means = np.array([r[q].mean() for r in reps])
            law_vals[f"{src}|{q}"].update(ci_draw=[m + dlo, m + dhi], draw_mean_avg=float(means.mean()),
                                          draw_mean_sd=float(means.std(ddof=1)), ci_expected=[lo, hi])
            axd.errorbar(i + o, m, yerr=[[-dlo], [dhi]], fmt="o", color=c, ms=3.4,
                         mfc=c if seed == 0 else "white", mec=c, mew=0.8, elinewidth=0.8, capsize=0, zorder=3)
    axd.axhline(0, color=MUTED, lw=0.6, zorder=0)
    axd.set_xticks(range(len(LAWS))); axd.set_xticklabels([l for _, l in LAWS])
    axd.tick_params(axis="x", length=0, pad=2)
    axd.set_xlim(-0.5, len(LAWS) - 0.5)
    axd.set_ylabel("turn toward (deg)")
    axd.grid(axis="y"); axd.set_axisbelow(True)
    h = [Line2D([], [], ls="", marker="o", color=c, mfc=c if s == 0 else "white", mec=c, mew=0.8, ms=3.4,
                label=("real fish" if src == "real" else f"{n} s{s}")) for src, n, c, s in SOURCES]
    if legend_out:
        axd.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, handletextpad=0.2,
                   borderaxespad=0.0, labelspacing=0.35)
    else:
        axd.legend(handles=h, loc="upper right", ncol=1, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
    return law_vals


def main_law_only():
    """The report's fig_rule: the law summary (panel d of the full figure) alone, no panel letter."""
    per = law_per_fish()
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.35))
    fig.subplots_adjust(left=0.09, right=0.86, top=0.97, bottom=0.17)
    law_vals = draw_laws(ax, per, legend_out=True)
    save(fig, "fig_rule")
    for k, v in law_vals.items():
        ci = v.get("ci_draw", v["ci"])
        extra = (f"   draws: mean {v['draw_mean_avg']:5.1f} (SD over draws {v['draw_mean_sd']:.2f})"
                 if "ci_draw" in v else "")
        print(f"  {k:28s} {v['mean']:6.1f}  [{ci[0]:5.1f}, {ci[1]:5.1f}]  n={v['n_fish']}{extra}")
    RF.mkdir(parents=True, exist_ok=True)
    (RF / "fig_rule_law_numbers.json").write_text(json.dumps(law_vals, indent=1, default=float))


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


def _excess(df):
    """Per-fish onset excess: mean over the 0-4 s bins minus mean over the bins from 10 s."""
    return df[[0, 1]].mean(axis=1) - df[[4, 5, 6]].mean(axis=1)


def main_onset(r_edges=R_EDGES, out_name="fig_rule"):
    D, w = onset_data(r_edges)
    xc = np.arange(len(T_BINS), dtype=float)                     # one evenly spaced slot per time bin
    fig, axa = plt.subplots(figsize=(TEXTWIDTH_IN, 2.6))
    fig.subplots_adjust(left=0.085, right=0.80, top=0.96, bottom=0.18)
    nums = {}
    r = D["real"]
    m = r.mean(axis=0).to_numpy()
    ci = np.array([P._boot_ci(r[c].to_numpy()) for c in r.columns])
    axa.fill_between(xc, ci[:, 0], ci[:, 1], color=FAINT, alpha=0.7, lw=0, zorder=1)
    axa.plot(xc, m, "-o", color=REAL_C, ms=3.2, lw=1.4, zorder=3, label="real fish")
    nums["real"] = dict(mean=m.tolist(), ci=ci.tolist())
    offs = {"M6 s0": -0.21, "M6 s101": -0.07, "F s0": 0.07, "F s101": 0.21}
    for src, name, c, seed in SOURCES[1:]:
        e = D[src].mean(axis=0).to_numpy()
        reps = D[src + "|draws"]
        off = np.array([[(lambda x: np.subtract(P._boot_ci(x), np.nanmean(x)))(rp[cc].to_numpy()) for cc in rp.columns]
                        for rp in reps]).mean(0)
        xs = xc + offs[src]
        axa.errorbar(xs, e, yerr=[-off[:, 0], off[:, 1]], fmt="o", ls="-", lw=0.9, color=c, ms=3.0,
                     mfc=c if seed == 0 else "white", mec=c, mew=0.8, elinewidth=0.7, capsize=0, zorder=4,
                     label=f"{name} s{seed}")
        nums[src] = dict(mean=e.tolist(), ci=(e[:, None] + off).tolist())
    axa.axvspan(-0.5, 1.5, color="#f2f2f2", lw=0, zorder=0)
    axa.text(0.5, 0.03, "onset", transform=axa.get_xaxis_transform(), ha="center", fontsize=6.5, color=MUTED)
    axa.set_xlim(-0.5, len(T_BINS) - 0.5)
    axa.set_xticks(xc); axa.set_xticklabels([f"{a}–{b}" for a, b in T_BINS], fontsize=7)
    axa.set_xlabel("time since the dots appeared (s)"); axa.set_ylabel("turn toward the dots (deg)")
    axa.set_ylim(0, None); axa.grid(axis="y"); axa.set_axisbelow(True)
    axa.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, handletextpad=0.3, borderaxespad=0.0,
               labelspacing=0.35, fontsize=7)

    for src in ["real"] + [x[0] for x in SOURCES[1:]]:            # per-fish onset excess (numbers only, not drawn)
        v = _excess(D[src]).dropna().to_numpy()
        nums[f"excess|{src}"] = dict(mean=float(v.mean()), ci=list(P._boot_ci(v)), n_fish=int(len(v)),
                                     positive=int((v > 0).sum()))
    save(fig, out_name)
    RF.mkdir(parents=True, exist_ok=True)
    (RF / f"{out_name}_onset_numbers.json").write_text(json.dumps(dict(nums, weights=w.to_dict(), t_bins=T_BINS),
                                                               indent=1, default=float))
    for k, v in nums.items():
        if k.startswith("excess"):
            print(f"  {k:18s} {v['mean']:6.1f}  [{v['ci'][0]:5.1f}, {v['ci'][1]:5.1f}]  {v['positive']}/{v['n_fish']} fish > 0")
    print("  real by bin:", np.round(nums["real"]["mean"], 1))
    for s in ("M6 s0", "M6 s101", "F s0", "F s101"):
        print(f"  {s:8s} by bin:", np.round(nums[s]["mean"], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="recompute the head gaps (teacher-forced, 14 units)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--law-only", action="store_true", help="draw only the law summary (the previous report figure)")
    ap.add_argument("--onset", action="store_true", help="draw the onset-turn figure (the report's fig_rule)")
    ap.add_argument("--onset60", action="store_true", help="the same over 0-60 mm (preview fig_rule_0-60mm)")
    a = ap.parse_args()
    if a.onset:
        return main_onset()
    if a.onset60:
        return main_onset([0, 10, 17.5, 25, 35, 47.5, 60], "fig_rule_0-60mm")
    if a.law_only:
        return main_law_only()
    if a.rebuild or not HG_CACHE.exists():
        parts = build_gaps(a.workers)
    else:
        with open(HG_CACHE, "rb") as fh:
            parts = pickle.load(fh)
    G = gap_data(parts)
    maps, mask, nfish = map_data()
    per = law_per_fish()

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.12], hspace=0.66, left=0.10, right=0.955,
                             top=0.935, bottom=0.135)
    top = outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.10)
    bot = outer[1].subgridspec(1, 2, width_ratios=[0.86, 1.14], wspace=0.24)

    # (a-c) response maps -------------------------------------------------------------------------
    contrasts = []
    for j, (lab, title, letter) in enumerate((("real", "Real fish", "a"), ("M6 s0", "M6, seed 0", "b"),
                                              ("F s0", "F, seed 0", "c"))):
        ax = fig.add_subplot(top[0, j])
        im = draw_map(ax, maps[lab], mask, show_y=(j == 0), contrasts=contrasts)
        panel(ax, letter, title)
    cax = fig.add_subplot(top[0, 3])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 10, 20, 30])
    cb.set_label("turn toward (deg)", labelpad=2)
    cb.outline.set_visible(False)
    cax.tick_params(length=2, pad=1.5)

    # (d) law summary -----------------------------------------------------------------------------
    axd = fig.add_subplot(bot[0, 0])
    law_vals = draw_laws(axd, per)
    panel(axd, "d", "Law summary")

    # (e) head gaps ---------------------------------------------------------------------------------
    axes = draw_gaps(fig, bot[0, 1], G)
    panel(axes[0], "e", "Head gaps, model − real (deg)")
    hs = [Line2D([], [], ls="", marker="o", color=OTHER_C, mfc=OTHER_C if s == 0 else "white", mec=OTHER_C,
                 mew=0.8, ms=3.0, label=f"s{s}") for s in (0, 101)]
    axes[-1].legend(handles=hs, loc="lower right", bbox_to_anchor=(1.02, 1.035), ncol=2, handletextpad=0.05,
                    columnspacing=0.5, borderaxespad=0.0, handlelength=1.0)

    save(fig, "fig_rule")

    # numbers for the notes ----------------------------------------------------------------------------
    RF.mkdir(parents=True, exist_ok=True)
    tabs = {}
    for lab in maps:
        tabs[lab] = maps[lab].round(2).astype(object).where(maps[lab].notna(), None).values.tolist()
    num = dict(
        maps=tabs, map_rows=[str(i) for i in maps["real"].index], map_cols=[str(c) for c in maps["real"].columns],
        map_nfish_ge5=nfish.astype(int).values.tolist(), map_mask=mask.values.tolist(),
        map_min_label_contrast=float(min(contrasts)),
        laws=law_vals,
        law_summary=P.law_summary().round(3).to_dict(orient="records"),
        paired_gap={f"{lab}|{q}": dict(mean=float((per[lab][q] - per["real"][q]).mean()),
                                        ci=list(P._boot_ci((per[lab][q] - per["real"][q]).values)))
                    for lab in P.HEAD for q, _ in LAWS},
        gaps=G.to_dict(orient="records"),
        model_steps=model_steps(parts),
        far_ahead_px=60.0, far_ahead_mm_at_3p60=60.0 / P.PX_MM)
    (RF / "fig_rule_numbers.json").write_text(json.dumps(num, indent=1, default=float))
    print(f"  numbers -> {RF / 'fig_rule_numbers.json'}")
    print(f"  min label contrast {min(contrasts):.2f}")
    print(G[["model", "seed", "gap", "mean", "n_fish", "t_half", "z_half", "archived"]].round(2).to_string())


if __name__ == "__main__":
    main()
