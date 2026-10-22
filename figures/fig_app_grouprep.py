"""Appendix figure for the group repulsion term (app:grouprep): the three screen readouts the term is calibrated on,
by distance to the dot, for real fish (test half of the clutches) and F without the term and with it at 1, 2, 4 and 8
times the calibrated strength, one row per seed. From output/screen/grtest_<14f|32b>.json (twin/grouprep_fit.py test).

    python thesis_grouprep_fig.py [out_dir]      # default MSc_Thesis_local/figures -> fig_app_grouprep.pdf
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures import show_png
from twin import paths

TW = paths.SCREEN_OUT
OUTDIR = paths.FIGURES

def _save(fig, name):
    """figures/out/<name>.pdf + a PNG preview (shown inline in a notebook)."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{name}.pdf")
    (OUTDIR / "png").mkdir(exist_ok=True)
    fig.savefig(OUTDIR / "png" / f"{name}.png", dpi=170)
    plt.close(fig)
    show_png(OUTDIR / "png" / f"{name}.png")

BINS = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 70], float)
NB = 11                                           # bins up to 40 mm, as fig_screen_response
from figures.fig_screen import RESP_XLIM, RESP_YLIM  # noqa: E402  same axes as fig_screen_response
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})

VARS = (("F", "F", "#D55E00", 1.4), ("grT", r"$\times 1$", "#9ecae1", 1.6), ("grT2", r"$\times 2$", "#6baed6", 1.6),
        ("grT4", r"$\times 4$", "#2171b5", 1.6), ("grT8", r"$\times 8$", "#08306b", 1.6))
PANELS = (("tow", "turn toward the dot", "P(toward) minus chance", 0.0),
          ("gain", "distance gained per bout", "mm minus chance", 0.0),
          ("occ", "bouts at each distance", "over chance", 1.0))


def main():
    xc = ((BINS[:-1] + BINS[1:]) / 2)[:NB]
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.8), sharey="col", sharex=True)
    for r, (seed, key) in enumerate(((0, "14f"), (101, "32b"))):
        d = json.load(open(TW / f"grtest_{key}.json"))
        tg = d["targets"]
        for j, (q, title, yl, h) in enumerate(PANELS):
            ax = axs[r, j]
            y = np.array(tg[q], float)[:NB]; se = np.array(tg["se_" + q], float)[:NB]
            ax.fill_between(xc, y - 1.96 * se, y + 1.96 * se, color="0.75", lw=0)
            ax.plot(xc, y, "k-o", ms=3.5, lw=1.8, label="real (test half)")
            for v, lab, col, lw in VARS:
                ax.plot(xc, np.array(d["variants"][v][q], float)[:NB], "-o", ms=2.8, lw=lw, color=col, label=lab)
            ax.set_xlim(*RESP_XLIM); ax.set_ylim(*RESP_YLIM[j])
            ax.axhline(h, color="0.6", lw=0.8)
            ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"seed {seed}\n{yl}")
            else:
                ax.set_ylabel(yl)
            if r == 1:
                ax.set_xlabel("distance to the dot at bout onset (mm)")
            ax.text(-0.14, 1.04, "abcdef"[3 * r + j], transform=ax.transAxes, fontsize=12, fontweight="bold")
    axs[0, 2].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_app_grouprep")


if __name__ == "__main__":
    main()
