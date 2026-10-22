"""Thesis figures for the screen data (Results, Sec. 'A second data set'), from the saved results. Top row of each
figure: the zero-shot models (trained on our 15 fish); bottom row: M6 and D4 trained from scratch on the screen
(twin/screen_train.py), against the real fish of the 98 clutches not used in that training.
    output/screen/screen_eval.json          zero-shot response by distance       -> fig_screen_response.pdf (a-c)
    output/screen/screen_trained_eval.json  screen-trained response, profiles    -> fig_screen_response.pdf (d-f),
                                                                                       fig_screen_switch.pdf (c, d)
    output/screen/screen_profiles.json      zero-shot distance over the episode  -> fig_screen_switch.pdf (a, b)
    output/screen/screen_closerange.json    time near the dot, bout decomposition -> fig_closerange.pdf

    python thesis_screen_figs.py [out_dir]       # default MSc_Thesis_local/figures
"""
from __future__ import annotations

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

RESP_XLIM = (0.0, 37.0)
RESP_YLIM = ((-0.37, 0.19), (-1.15, 2.25), (0.4, 2.9))   # turn toward, distance gained, bouts at each distance
COL = {"M6": "#0072B2", "F": "#D55E00", "D4": "#009E73", "M6dm": "#CC79A7", "D4dm": "#E69F00"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})


def series(ax, x, d, row, color, ls="-", mfc=None, label=None, band=False, lw=1.4, alpha=1.0):
    y = np.array(d["est"], float); lo = np.array(d["lo"], float); hi = np.array(d["hi"], float)
    if row is not None:
        y, lo, hi = y[row], lo[row], hi[row]
    n = len(x)
    y, lo, hi = y[:n], lo[:n], hi[:n]
    if band:
        ax.fill_between(x, lo, hi, color=color, alpha=0.22 * alpha, lw=0)
        ax.plot(x, y, ls, color=color, lw=lw, label=label, alpha=alpha)
    else:
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="o", ls=ls, color=color, alpha=alpha,
                    mfc=mfc or color, ms=3.5, lw=lw, capsize=0, label=label)


def rowlabel(ax, s, x=-0.30):
    ax.text(x, 0.5, s, transform=ax.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center", ha="center")


def letter(ax, s):
    ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")




def fig_closerange():
    C = json.load(open(TW / "screen_closerange.json"))
    S = C["sides"]
    b = np.array(C["bins"]); xc = ((b[:-1] + b[1:]) / 2)
    n_occ, n_bd = 7, 8
    qs = (("tocc", "time near the dot", "time over chance", 1.0, n_occ),
          ("along_x", "moving along the dot's motion", "mm per bout, minus chance", 0.0, n_bd),
          ("toward_x", "steering toward the dot", "mm per bout, minus chance", 0.0, n_bd))
    rows = ((("F zero-shot", "F"),), (("M6 screen", "M6"), ("D4 screen", "D4")),
            (("M6dm screen", "M6dm"), ("D4dm screen", "D4dm")))
    if "M6dm screen s0" not in S:
        rows = rows[:2]
    lab = {"M6dm": "M6 + direction", "D4dm": "D4 + direction"}
    fig, axs = plt.subplots(len(rows), 3, figsize=(10.5, 2.9 * len(rows)), sharey="col")
    for r, units in enumerate(rows):
        for j, (q, t, yl, h, n) in enumerate(qs):
            ax = axs[r, j]
            series(ax, xc[:n], S["real (pool)"][q], None, "k", band=True, label="real (held-out clutches)", lw=1.8)
            for unit, m in units:
                for s, ls, mfc in ((0, "-", None), (101, "--", "white")):
                    series(ax, xc[:n] + (0.3 if s else -0.3), S[f"{unit} s{s}"][q], None, COL[m], ls=ls, mfc=mfc,
                           label=f"{lab.get(m, m)} s{s}")
            ax.axhline(h, color="0.6", lw=0.8); ax.set_ylabel(yl); ax.set_title(t, fontsize=9)
            letter(ax, "abcdefghi"[3 * r + j])
        axs[r, 0].legend(fontsize=7, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("distance to the dot (mm)")
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    for r_, nm in enumerate(("zero-shot", "trained on the screen", "+ dot direction")[:len(rows)]):
        rowlabel(axs[r_, 0], nm)
    _save(fig, "fig_closerange")


MERGED_ROWS = ((("F", "F zero-shot", "F"),), (("D4", "D4 screen", "D4"),), (("D4dm", "D4dm screen", "D4 + direction"),))
MERGED_LABELS = ("F, zero-shot", "D4, trained on the screen", "D4 + direction")


def fig_screen_merged():
    """THE THESIS'S SCREEN FIGURE (2026-10-21, replaces fig_screen_response + fig_closerange): one row per unit (F
    zero-shot, D4 trained on the screen, D4 + the dot's direction of motion), four readouts by distance to the dot, all
    against the rotated dot (chance) and the real fish of the 98 clutches not used in training, real band = 95 %
    bootstrap over CLUTCHES (turn toward: screen_tow_clutch.json; the rest: screen_closerange.json), models = 95 %
    bootstrap over their 200 simulated fish. M6 is not shown (dropped from the thesis's screen section)."""
    TR = json.load(open(TW / "screen_trained_eval.json"))
    TC = json.load(open(TW / "screen_tow_clutch.json"))
    C = json.load(open(TW / "screen_closerange.json"))
    S = C["sides"]
    b = np.array(TR["dbins"]); xc = ((b[:-1] + b[1:]) / 2)
    assert np.allclose(b, C["bins"]) and np.allclose(b, TC["bins"])
    cols = (("tow", "turn toward the dot", "P(toward) minus chance", 0.0, 11),
            ("tocc", "time near the dot", "time over chance", 1.0, 7),
            ("along_x", "moving along the dot's motion", "mm per bout, minus chance", 0.0, 8),
            ("toward_x", "moving toward the dot", "mm per bout, minus chance", 0.0, 8))
    fig, axs = plt.subplots(3, 4, figsize=(12.0, 7.6), sharey="col")
    for r, units in enumerate(MERGED_ROWS):
        for j, (q, t, yl, h, n) in enumerate(cols):
            ax = axs[r, j]
            real = TC["real"] if q == "tow" else S["real (pool)"][q]
            series(ax, xc[:n], real, None, "k", band=True, label="real", lw=1.8)
            for m, unit, lab in units:
                for s, ls, mfc in ((0, "-", None), (101, "--", "white")):
                    d = TR["curves"][f"{unit} s{s}"] if q == "tow" else S[f"{unit} s{s}"][q]
                    series(ax, xc[:n] + (0.3 if s else -0.3), d, 0 if q == "tow" else None, COL[m], ls=ls, mfc=mfc,
                           label=f"{lab} s{s}")
            ax.axhline(h, color="0.6", lw=0.8)
            ax.set_xlim(*(RESP_XLIM if q == "tow" else (0.0, float(b[n]))))   # each close-range column ends at its last bin
            if q == "tow":
                ax.set_ylim(*RESP_YLIM[0])
            ax.set_ylabel(yl)
            if r == 0:
                ax.set_title(t, fontsize=9.5)
            letter(ax, "abcdefghijkl"[4 * r + j])
        axs[r, 0].legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axs[-1]:
        ax.set_xlabel("distance to the dot (mm)")
    fig.tight_layout(rect=(0.035, 0, 1, 1))
    for r_, nm in enumerate(MERGED_LABELS):
        rowlabel(axs[r_, 0], nm, x=-0.40)
    _save(fig, "fig_screen_response")


def fig_switch_noM6():
    """fig_screen_switch: the bout-like episode from the switch on. F zero-shot, D4 trained on the screen, D4 + direction; every panel
    against the real fish of the 98 clutches not used in training (screen_trained_eval's profiles); F's blind run from
    screen_profiles.json (a model curve, independent of the real set)."""
    P = json.load(open(TW / "screen_profiles.json"))
    TR = json.load(open(TW / "screen_trained_eval.json"))
    assert np.allclose(P["bins"], TR["pbins"])
    b = np.array(TR["pbins"]); xc = (b[:-1] + b[1:]) / 2
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.0), sharey=True, sharex=True)
    panels = [("F", "F zero-shot s{s} sighted", P["prof"]["F s0 blind"], "F, zero-shot", "F s0 blind"),
              ("D4", "D4 screen s{s} sighted", TR["prof"]["D4 screen s0 decoupled"], "D4, trained on the screen",
               "D4 s0 decoupled"),
              ("D4dm", "D4dm screen s{s} sighted", TR["prof"]["D4dm screen s0 decoupled"], "D4 + direction",
               "D4 + dir. s0 decoupled")]
    nm_ = {"D4dm": "D4 + dir."}
    for ax, (m, km, ref, ttl, rlab), lt in zip(axs, panels, "abc"):
        series(ax, xc, TR["prof"]["real, rotated dot"], None, "0.45", ls="--", band=True,
               label="real, rotated dot (chance)", lw=1.2)
        series(ax, xc, TR["prof"]["real"], None, "k", band=True, label="real", lw=1.8)
        for s, ls in ((0, "-"), (101, "--")):
            series(ax, xc, TR["prof"][km.format(s=s)], None, COL[m], ls=ls, band=True, label=f"{nm_.get(m, m)} s{s}")
        series(ax, xc, ref, None, COL[m], ls=":", band=True, label=rlab, lw=1.0)
        ax.set_xlim(0.0, float(b[-1]))                       # the bout-like episode only (bins from the switch on)
        ax.set_ylim(11.0, 33.5); ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("time since the dot switched to bout-like motion (s)")
        letter(ax, lt)
        ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                  handlelength=1.8, columnspacing=1.0)
    axs[0].set_ylabel("distance to the dot (mm)")
    fig.tight_layout()
    _save(fig, "fig_screen_switch")


if __name__ == "__main__":
    fig_screen_merged(); fig_switch_noM6()
