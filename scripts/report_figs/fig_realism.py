"""fig_realism (Chapter 5, sec:r-realism): M6 and F against the properties of the data chapter.

The classifier scale that used to be panel a is now fig_scale (scripts/report_figs/fig_scale.py, sec:r-reading).

a-c  The Chapter-3 properties as model - real, mean over the 15 fish of the per-fish paired difference, 95 % bootstrap
   CI over fish (missing_pieces._boot_ci, 4000 resamples, seed 0 -- exactly missing_pieces.properties_table).
   Source: Model/missing_pieces/properties.pkl (missing_pieces.properties: 2 record-anchored rollouts per held-out
   fish, each fish by the fold model that never saw it; steps at 3.60 px/mm).
   a  quantiles of |turn|, step, interval and the radial median, as % of the real mean over fish;
   b  share of near-straight bouts (|turn| < 3 deg) and the wall law P(turn toward the dish centre) by r/R, in
      percentage points (the r/R 0.9-1 bin is left out: only 3 real fish have >= 10 sideways bouts there; the
      0.8-0.9 bin has 14 paired fish); the wall bins are labelled at their edges, set apart from 'straight' by a gap;
   c  lag-1..5 correlations within a run of |turn|, log step and log interval.

    uv run python scripts/report_figs/fig_realism.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

import missing_pieces as MP  # noqa: E402

HEADS = [("M6", 0, "M6 s0"), ("M6", 101, "M6 s101"), ("F", 0, "F s0"), ("F", 101, "F s101")]
DODGE = np.array([-0.27, -0.09, 0.09, 0.27])


# --------------------------------------------------------------------------------------------- data
def property_diffs():
    """{property: {head label: (mean diff, lo, hi)}} and the real mean, as in missing_pieces.properties_table."""
    T = MP.load("properties")
    scal = [c for c in T.columns if c not in ("model", "fish", "rep", "radial_hist")]
    per = T.groupby(["model", "fish"])[scal].mean()
    out, real = {}, {}
    for c in scal:
        real[c] = float(per.loc["real", c].mean())
        out[c] = {}
        for _, _, lab in HEADS:
            diff = per.loc[lab, c] - per.loc["real", c]
            lo, hi = MP._boot_ci(diff.values)
            out[c][lab] = (float(diff.mean()), lo, hi, int(diff.notna().sum()))
    return out, real, per


# --------------------------------------------------------------------------------------------- drawing
def _mk(mod, seed):
    c = MODEL_C[mod]
    return dict(color=c, mec=c, mfc=c if seed == 0 else "white", mew=0.9)


def forest(ax, cats, D, scale=1.0, xs=None):
    """cats = [(property, tick label)]; one dodged point + CI per head and category (at xs, default 0, 1, ...)."""
    xs = list(range(len(cats))) if xs is None else list(xs)
    for i, (p, _) in enumerate(cats):
        for j, (mod, seed, lab) in enumerate(HEADS):
            m, lo, hi, _ = D[p][lab]
            x = xs[i] + DODGE[j]
            ax.plot([x, x], [lo * scale, hi * scale], color=MODEL_C[mod], lw=0.8, zorder=2,
                    solid_capstyle="butt")
            ax.plot(x, m * scale, "o", ms=3.0, zorder=3, **_mk(mod, seed))
    ax.axhline(0, color=INK, lw=0.7, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([t for _, t in cats])
    ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def group_labels(ax, groups, y_pt=-13.5, pad=None):
    """Second row of x labels under the ticks: [(first x, last x, text)], with a thin bracket that reaches pad[i]
    beyond the first / last category (default 0.38 = just past the dodged points). The bracket sits y_pt points
    below the axis and the text 1.5 pt under it, whatever the axes height."""
    tr_b = ax.get_xaxis_transform() + ScaledTranslation(0, y_pt / 72, ax.figure.dpi_scale_trans)
    tr_t = ax.get_xaxis_transform() + ScaledTranslation(0, (y_pt - 1.5) / 72, ax.figure.dpi_scale_trans)
    pad = [0.38] * len(groups) if pad is None else pad
    for (a, b, txt), p in zip(groups, pad):
        ax.plot([a - p, b + p], [0, 0], color=MUTED, lw=0.5, transform=tr_b, clip_on=False)
        ax.text((a + b) / 2, 0, txt, transform=tr_t, ha="center", va="top", fontsize=7, color=INK)


def letter(ax, txt, title):
    """style.panel, but with the letter on the title's baseline whatever the axes height (style.panel places it at a
    fixed fraction of the axes height, which floats far above a tall axes)."""
    ax.set_title(title, loc="left", pad=6)
    tr = ax.transAxes + ScaledTranslation(-0.06, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(0.0, 1.0, txt, transform=tr, fontsize=9, fontweight="bold", va="baseline", ha="right")


def main():
    D, real, per = property_diffs()

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.95, left=0.105, right=0.995,
                             top=0.865, bottom=0.135)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[7.4, 6.6], wspace=0.30)
    bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.07)

    # a: quantiles, % of the real mean over fish
    aa = fig.add_subplot(top[0])
    cats_a = [("turn_median", "50"), ("turn_q90", "90"), ("step_median", "50"), ("step_q90", "90"),
              ("interval_median", "50"), ("interval_q90", "90"), ("radial_median", "50")]
    Da = {p: {lab: tuple(v / real[p] if i < 3 else v for i, v in enumerate(D[p][lab])) for lab in D[p]}
          for p, _ in cats_a}
    forest(aa, cats_a, Da, scale=100.0)
    group_labels(aa, [(0, 1, r"$|\Delta\theta|$"), (2, 3, "step"), (4, 5, "interval"), (6, 6, "$r/R$")])
    for x in (1.5, 3.5, 5.5):
        aa.axvline(x, color=FAINT, lw=0.5, zorder=0)
    aa.set_ylabel("model $-$ real\n(% of real)")
    letter(aa, "a", "Percentiles 50 and 90")

    # b: proportions, percentage points (the r/R 0.9-1 wall bin is left out: 3 real fish only)
    ab = fig.add_subplot(top[1])
    wall = ["wall_0.0-0.4", "wall_0.4-0.6", "wall_0.6-0.7", "wall_0.7-0.8", "wall_0.8-0.9"]
    cats_b = [("straight", "")] + [(w, "") for w in wall]
    GAP = 0.6                                     # extra space between 'straight' and the wall bins
    xs_b = [0.0] + [k + GAP for k in range(1, len(wall) + 1)]
    forest(ab, cats_b, D, scale=100.0, xs=xs_b)
    ab.axvline(0.5, color=FAINT, lw=0.5, zorder=0)    # just right of the 'straight' points, well left of r/R = 0
    # the wall-law bins are labelled at their EDGES (r/R = 0, .4, .6, .7, .8, .9); the first edge sits clear of the
    # separator so the '0' reads as r/R = 0, not as part of 'straight'
    ab.set_xticks([x - 0.5 for x in xs_b[1:]] + [xs_b[-1] + 0.5])
    ab.set_xticklabels(["0", ".4", ".6", ".7", ".8", ".9"])
    ab.tick_params(axis="x", length=2, width=0.5)
    ab.set_xlim(-0.55, xs_b[-1] + 0.65)
    group_labels(ab, [(0, 0, "straight"), (xs_b[1], xs_b[-1], "turn to centre ($r/R$)")], pad=[0.38, 0.5])
    ab.set_ylabel("model $-$ real\n(% points)")
    letter(ab, "b", "Proportions")

    # c: lag correlations within a run
    chans = [("turn", r"$|\Delta\theta|$"), ("step", "log step"), ("interval", "log interval")]
    axc = [fig.add_subplot(bot[i]) for i in range(3)]
    lo = min(D[f"acf{L}_{ch}"][lab][1] for L in range(1, 6) for ch, _ in chans for _, _, lab in HEADS)
    hi = max(D[f"acf{L}_{ch}"][lab][2] for L in range(1, 6) for ch, _ in chans for _, _, lab in HEADS)
    for k, (a, (ch, title)) in enumerate(zip(axc, chans)):
        forest(a, [(f"acf{L}_{ch}", str(L)) for L in range(1, 6)], D)
        a.set_ylim(lo - 0.012, hi + 0.045)
        if k:
            a.tick_params(axis="y", labelleft=False)
        a.text(0.5, 0.98, title, transform=a.transAxes, ha="center", va="top", fontsize=7)
        a.set_xlabel("lag (bouts)", labelpad=1)
    axc[0].set_ylabel("model $-$ real\n(correlation)")
    letter(axc[0], "c", "Lag correlations within a run")

    # one legend for the four model units
    hl = [Line2D([], [], ls="", marker="o", ms=3.4, label=f"{mod} s{seed}", **_mk(mod, seed))
          for mod, seed, _ in HEADS]
    fig.legend(handles=hl, loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=4, handlelength=0.8,
               columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)
    save(fig, "fig_realism")
    return D, real


if __name__ == "__main__":
    D, real = main()
    print("\n(a-c) model - real [95 % CI] (n fish paired):")
    for p in D:
        s = "  ".join(f"{lab}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}" for lab, (m, lo, hi, n) in D[p].items())
        print(f"  {p:<16} real {real[p]:.3f}  {s}")
