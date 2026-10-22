"""fig_ablation_profiles -- Appendix app:abl-profiles (label fig:abl-profiles): distance to the dots across the block for
M6 and its five stimulus ablations, in the style of Chapter 3's fig_dot_profile (all eight conditions in one panel,
blues moving, reds static, darker = more dots, 95 % bootstrap bands).

Panels: real fish (with the blind-fish curve of fig_dot_profile), then M6 and M6-stim, -pos, -count, -range, -motion at
seed 0 and seed 101. Model curves are exactly the archived matched profiles: the archived rollout stream 0 of every
unit regenerated as in fig_profiles (block-seeded programme rollouts, wall rejection, 10 per fold model, centroid-outside
-dish filter, block-matched to the real curve with model_figures.weighted_curves) and checked against each unit's
archived dip / latency / plateau / grading / profile errors / attraction (refuses to draw otherwise). Bands are
COMPARABLE between real and model (2026-10-14): 95 % percentile bootstrap of the mean over 15 fish -- for a model, 15
resampled from its 50 simulated fish, with the simulated bouts thinned at random to the real number per fish and bin
and re-block-matched (fig_profiles_bands.bands, first thinning); real: over the 15 fish (fig_profiles' cached band). Every mm at 3.60 px/mm (the profile code's 3.67 converted by MM_3P67_TO_3P60).

    uv run python scripts/report_figs/fig_ablation_profiles.py            # rollouts if missing (50 fold jobs), check, draw
"""
import pickle
import sys
from pathlib import Path

from figures.style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from figures import fig_profiles as FP  # noqa: E402
from figures import fig_profiles_bands as FPB  # noqa: E402

from figures import data_figures as RF  # noqa: E402

M6_ARM = "S4x_adt_z2_cdt_tr_rc_al_w3ft"
# (public name, seed) -> (checkpoint dir under Model/, arm, internal key); report_models MODELS k=26/29/30/33
UNITS = {("M6", 0): ("M26e", M6_ARM, "26e"), ("M6", 101): ("M30d", M6_ARM, "30d"),
         ("M6−stim", 0): ("M29e", "Snull_adt_z2_cdt_tr_rc_w3ft_nck", "29e"),
         ("M6−stim", 101): ("M33u", "Snull_adt_z2_cdt_tr_rc_w3ft_nck", "33u"),
         ("M6−pos", 0): ("M33a", "Scond_adt_z2_cdt_tr_rc_w3ft", "33a"),
         ("M6−pos", 101): ("M33b", "Scond_adt_z2_cdt_tr_rc_w3ft", "33b"),
         ("M6−count", 0): ("M29c", "S4xnn_adt_z2_cdt_tr_rc_al_w3ft", "29c"),
         ("M6−count", 101): ("M33s", "S4xnn_adt_z2_cdt_tr_rc_al_w3ft", "33s"),
         ("M6−range", 0): ("M33c", "S4xnr_adt_z2_cdt_tr_rc_w3ft", "33c"),
         ("M6−range", 101): ("M33d", "S4xnr_adt_z2_cdt_tr_rc_w3ft", "33d"),
         ("M6−motion", 0): ("M29d", "S4xnm_adt_z2_cdt_tr_rc_w3ft", "29d"),
         ("M6−motion", 101): ("M33t", "S4xnm_adt_z2_cdt_tr_rc_w3ft", "33t")}
ORDER = ["M6", "M6−stim", "M6−pos", "M6−count", "M6−range", "M6−motion"]
CURVES = FP.CACHE / "fig_ablation_profiles_curves.pkl"
COND_KEY = {"d1s3.5": (1, True), "d2s3.5": (2, True), "d4s3.5": (4, True), "d6s3.5": (6, True),
            "d8s3.5": (8, True), "d1s0": (1, False), "d4s0": (4, False), "d8s0": (8, False)}
YLIM = (8, 42)
N_BOOT = 4000


def curves():
    """Rollouts (only units missing from the cache), archive check, cache."""
    FP.HEAD = UNITS
    FP.CURVES = CURVES
    if CURVES.exists():
        with open(CURVES, "rb") as fh:
            c = pickle.load(fh)
    else:                                       # start from fig_profiles' cache: same real side, M6 already rolled out
        with open(FP.CACHE / "fig_profiles_curves.pkl", "rb") as fh:
            base = pickle.load(fh)
        c = dict(base, models={k: v for k, v in base["models"].items() if k[0] == "M6"})
    if any(u not in c["models"] for u in UNITS):
        c = FP.build(6, have=c)
    t = FP.check(c)
    bad = t[~t.ok]
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit("regenerated ablation curves do NOT reproduce the archived summaries -- not drawing")
    print(f"check passed: {t.model.nunique()} units reproduce their archived matched metrics and attraction")
    return c




def _draw_unit(ax, mean, bands, cm):
    f = MM_3P67_TO_3P60
    for cond, key in COND_KEY.items():
        if cond not in mean:
            continue
        y = mean[cond]
        y = y[y.index < 60]
        b = bands[cond]
        x, lo, hi = b.index.to_numpy(float), b.lo.to_numpy(float), b.hi.to_numpy(float)
        FP._inside(np.r_[lo, hi, y.values] * f, YLIM, cond)
        ax.fill_between(x, lo * f, hi * f, color=cm[key], alpha=0.18, lw=0)
        ax.plot(y.index, y.values * f, color=cm[key], lw=1.1)


def _draw_real(ax, c, cm, blind):
    f = MM_3P67_TO_3P60
    for cond, key in COND_KEY.items():
        b = c["real"]["band"][cond]
        b = b[b.index < 60]
        FP._inside(np.r_[b.lo, b.hi, b.mu] * f, YLIM, f"real {cond}")
        ax.fill_between(b.index, b.lo * f, b.hi * f, color=cm[key], alpha=0.18, lw=0)
        ax.plot(b.index, b.mu * f, color=cm[key], lw=1.1)
    x, mu, lo, hi = blind
    ax.plot(x, mu, color=INK, lw=1.3, ls=(0, (3, 1.6)), zorder=6)          # on top, no band: it must stay visible


def main():
    c = curves()
    B, ratio = FPB.bands(c)                     # 15 of 50 simulated fish, bouts thinned: comparable to the real band
    t = ratio.assign(unit=ratio.model + " s" + ratio.seed.astype(str)).pivot(index="cond", columns="unit", values="ratio")
    pd.set_option("display.width", 220)
    print("model / real band width (15 fish each, bouts thinned):")
    print(t.reindex(FP.CONDS).round(2).to_string())
    cm = RF._stim_colours()
    rc = RF.load()
    blind = RF._band(RF._blind_profile_frame(rc, keep_outside=False), 4000, np.random.default_rng(1))

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 7.6))
    gs = fig.add_gridspec(4, 4, left=0.075, right=0.995, top=0.955, bottom=0.07, wspace=0.12, hspace=0.42)
    slots = [(0, 0, "Real fish", None)] + [(0, 1, "M6, s0", ("M6", 0)), (0, 2, "M6, s101", ("M6", 101))]
    rest = [(n, s) for n in ORDER[1:] for s in (0, 101)]
    for i, (n, s) in enumerate(rest):
        r, col = 1 + i // 4, i % 4
        slots.append((r, col, f"{n}, s{s}", (n, s)))
    axes = []
    for k, (r, col, title, unit) in enumerate(slots):
        ax = fig.add_subplot(gs[r, col], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
        axes.append(ax)
        if unit is None:
            _draw_real(ax, c, cm, blind)
        else:
            m = c["models"][unit]
            _draw_unit(ax, m["matched_mean"], B[unit], cm)
        ax.set_xlim(0, 60); ax.set_xticks([0, 20, 40, 60])
        ax.set_ylim(*YLIM); ax.set_yticks([10, 20, 30, 40])
        ax.grid(axis="y"); ax.set_axisbelow(True)
        panel(ax, "abcdefghijklmn"[k], title)
        ax.texts[-1].set_x(-0.05)
        ax.title.set_fontsize(7.6)
        if col:
            ax.tick_params(labelleft=False)
        last_row = (r == 3) or (r == 2 and col >= 2)
        if not last_row:
            ax.tick_params(labelbottom=False)
    fig.text(0.012, 0.51, "distance to the dots (mm)", rotation=90, ha="left", va="center", fontsize=8)
    fig.text(0.54, 0.012, "time since block onset (s)", ha="center", va="bottom", fontsize=8)

    # legend: the free slots (top right, and bottom right two)
    lab = {k: f"{n} dot{'s' if n != 1 else ''}, {'moving' if mv else 'static'}" for k, (n, mv) in
           ((k, k) for k in RF.STIM_ORDER)}
    h = [(Line2D([], [], color=cm[k], lw=1.5), lab[k]) for k in RF.STIM_ORDER]
    h.append((Line2D([], [], color=INK, lw=1.3, ls=(0, (3, 1.6))), "blind fish (panel a)"))
    h.append((Patch(facecolor=MUTED, alpha=0.18, lw=0), "95 % bootstrap band over fish"))
    lax = fig.add_subplot(gs[3, 2:4]); lax.axis("off")
    lax.legend(*zip(*h), loc="center", ncol=2, frameon=False, fontsize=6.8, handlelength=1.8, columnspacing=1.0)
    save(fig, "fig_ablation_profiles")


if __name__ == "__main__":
    main()
