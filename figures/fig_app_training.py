"""Training-curve figures for the methods-in-detail appendix, read from the report checkpoints (nothing re-run).

  fig_app_correction_trace  F (14f | 32b): the eight correction parameters over the 100 Adam steps of every fold
                            (theta = h, q, b_F, e_F, l_F, b_B, e_B, l_B in that order, the stim_turn.ahead and
                            stim_turn.coef_s rows; the count gates drawn as kappa = exp(clip(l, -3, 3)) on a log axis),
                            the profile term relative to its first step and the likelihood term's change from its first
                            step. profile_finetune['trace'] records the state after each step; the zero start is drawn
                            at step 0, the steps at 1-100.
  fig_app_drift_gain        D4 (11e | 32a): the per-fold rollout search of the drift's generation gains
                            (drift_report): lag-1 window correlation against the gain, raw points and the
                            monotone-smoothed curve, the fold's training-fish target (dashed) and the chosen gain.
  fig_app_latent_gain       M6 (26e | 30d): the moment-matching objective sum |ln(SD_sim / SD_real)| against the latent
                            gain g for every fold (re_gain_report['curve']), and the chosen g.

    uv run python scripts/report_figs/fig_app_training.py
"""
import glob
import sys
from pathlib import Path

from figures.style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FOLD_C = [BLUE, VERM, GREEN, ORANGE, PURPLE]
SEEDS = (0, 101)


def blobs(key):
    out = []
    for f in sorted(glob.glob(str(MODEL / "report_ckpts" / f"M{key}" / "*_fold[0-9].pt"))):
        out.append(torch.load(f, map_location="cpu", weights_only=False))
    assert len(out) == 5, key
    return out


def _seed_legend(fig, y, extra=()):
    h = [Line2D([], [], color=c, lw=1.2, label=f"fold {k}") for k, c in enumerate(FOLD_C)]
    h += [Line2D([], [], color=MUTED, lw=1.2, label="seed 0"),
          Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3, 1.6)), label="seed 101")] + list(extra)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(h), frameon=False, fontsize=6.8,
               handlelength=1.8, columnspacing=1.0)


# ------------------------------------------------------------------------------------------------ correction ----
NAMES = [("h", "$h$ (straight, level)"), ("q", "$q$ (straight, range)"), ("bF", "$b_\\mathrm{F}$ (side, front)"),
         ("eF", "$e_\\mathrm{F}$ (side, front, range)"), ("lF", "$\\kappa_\\mathrm{F}$ (count gate, front)"),
         ("bB", "$b_\\mathrm{B}$ (side, behind)"), ("eB", "$e_\\mathrm{B}$ (side, behind, range)"),
         ("lB", "$\\kappa_\\mathrm{B}$ (count gate, behind)")]


def fig_correction():
    fig, axes = plt.subplots(2, 5, figsize=(TEXTWIDTH_IN, 3.6), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.17, wspace=0.42, hspace=0.45)
    axes = axes.ravel()
    print("correction, final step (s0 | s101), per fold:")
    for seed, key in zip(SEEDS, ("14f", "32b")):
        ls = "-" if seed == 0 else (0, (3, 1.6))
        for k, b in enumerate(blobs(key)):
            tr = b["profile_finetune"]["trace"]
            steps = np.arange(0, len(tr) + 1)
            C = np.vstack([np.zeros(8), np.array([t["coef"] for t in tr], float)])
            for j, (nm, _) in enumerate(NAMES):
                y = C[:, j]
                if nm in ("lF", "lB"):
                    y = np.exp(np.clip(y, -3, 3))
                axes[j].plot(steps, y, color=FOLD_C[k], lw=0.9, ls=ls)
            prof = np.array([t["profile"] for t in tr], float)
            nll = np.array([t["nll"] for t in tr], float)
            run = lambda v: np.convolve(v, np.ones(10) / 10, mode="valid")      # 10-step running mean
            for ax, v in ((axes[8], prof / prof[0]), (axes[9], nll - nll[0])):
                ax.plot(steps[1:], v, color=FOLD_C[k], lw=0.4, ls=ls, alpha=0.25)
                ax.plot(steps[10:], run(v), color=FOLD_C[k], lw=1.0, ls=ls)
            print(f"  s{seed} fold {k}: profile {prof[0]:.1f} -> {prof[-1]:.1f} ({prof[-1] / prof[0] - 1:+.1%}), "
                  f"nll {nll[0]:.4f} -> {nll[-1]:.4f} ({nll[-1] - nll[0]:+.4f}), kappa F/B "
                  f"{np.exp(np.clip(C[-1, 4], -3, 3)):.2f} / {np.exp(np.clip(C[-1, 7], -3, 3)):.2f}")
    for j, (nm, title) in enumerate(NAMES):
        ax = axes[j]
        if nm in ("lF", "lB"):
            ax.set_yscale("log")
            for v in (np.exp(-3), np.exp(3)):
                ax.axhline(v, color=MUTED, lw=0.6, ls=":")
            ax.set_ylim(0.03, 35)
        else:
            ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(title, fontsize=7, loc="left", fontweight="normal")
    axes[8].set_title("profile term (relative)", fontsize=7, loc="left", fontweight="normal")
    axes[9].set_title("likelihood change", fontsize=7, loc="left", fontweight="normal")
    axes[9].axhline(0, color=MUTED, lw=0.6)
    for ax in axes:
        ax.axvline(50, color=FAINT, lw=0.8, zorder=0)            # learning rate halved
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(0, 100)
    for ax in axes[5:]:
        ax.set_xlabel("Adam step", fontsize=7)
    _seed_legend(fig, 0.0, [Line2D([], [], color=FAINT, lw=0.8, label="learning rate halved")])
    save(fig, "fig_app_correction_trace")


# ------------------------------------------------------------------------------------------------ drift gains ---
def fig_drift():
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.9), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.16, wspace=0.25, hspace=0.42)
    print("drift gains (D4), per fold: g_dist, g_dt, pick")
    for r, (seed, key) in enumerate(zip(SEEDS, ("11e", "32a"))):
        for k, b in enumerate(blobs(key)):
            d = b["drift_report"]
            grid = np.array(d["grid"], float)
            for c, (ch, curve, smooth, g) in enumerate((("dist", d["curve_dist"], d["smoothed_dist"], d["g_dist"]),
                                                        ("dt", d["curve_dt"], d["smoothed_dt"], d["g_dt"]))):
                ax = axes[r, c]
                raw = np.array([v[ch] for _, v in curve], float)
                sm = np.array(smooth, float)
                ax.plot(grid, raw, "o", ms=2.6, color=FOLD_C[k], alpha=0.8)
                ax.plot(grid, sm, color=FOLD_C[k], lw=1.0)
                ax.axhline(d["target"][ch], color=FOLD_C[k], lw=0.7, ls=(0, (3, 1.6)))
                ax.plot([g], [np.interp(np.log(g), np.log(grid), sm)], marker="D", ms=4.5, color=FOLD_C[k],
                        mec="white", mew=0.6, zorder=5)
            print(f"  s{seed} fold {k}: g_dist {d['g_dist']:.2f} ({d['pick']['dist']}), "
                  f"g_dt {d['g_dt']:.2f} ({d['pick']['dt']}), target dist {d['target']['dist']:.3f} "
                  f"dt {d['target']['dt']:.3f}")
    for r, seed in enumerate(SEEDS):
        for c, lab in enumerate(("displacement gain $g_\\mathrm{dist}$", "interval gain $g_\\mathrm{dt}$ (at the chosen $g_\\mathrm{dist}$)")):
            ax = axes[r, c]
            ax.set_xscale("log")
            ax.set_xticks([0.15, 0.3, 0.6, 1, 1.8, 3, 5])
            ax.set_xticklabels(["0.15", "0.3", "0.6", "1", "1.8", "3", "5"])
            ax.grid(axis="y"); ax.set_axisbelow(True)
            ax.set_title(f"{'Log displacement' if c == 0 else 'Log interval'}, seed {seed}", fontsize=8,
                         loc="left", fontweight="bold")
            if r == 1:
                ax.set_xlabel(lab, fontsize=7.5)
            if c == 0:
                ax.set_ylabel("lag-1 window\ncorrelation (simulated)", fontsize=7.5)
    h = [Line2D([], [], color=c, lw=1.0, marker="o", ms=2.6, label=f"fold {k}") for k, c in enumerate(FOLD_C)]
    h += [Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 1.6)), label="training-fish target"),
          Line2D([], [], color=MUTED, ls="none", marker="D", ms=4.5, mec="white", label="chosen gain")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=7, frameon=False, fontsize=6.8,
               handlelength=1.6, columnspacing=1.0)
    save(fig, "fig_app_drift_gain")


# ------------------------------------------------------------------------------------------------ latent gain ---
def fig_latent():
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.62, 2.3))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.30)
    chosen = set()
    for seed, key in zip(SEEDS, ("26e", "30d")):
        ls = "-" if seed == 0 else (0, (3, 1.6))
        for k, b in enumerate(blobs(key)):
            cur = b["re_gain_report"]["curve"]
            g = np.array([c[0] for c in cur], float)
            obj = np.array([c[1] for c in cur], float)
            ax.plot(g, obj, color=FOLD_C[k], lw=1.0, ls=ls, marker="o", ms=2.4,
                    mfc=FOLD_C[k] if seed == 0 else "white")
            chosen.add(float(b.get("re_gain")))
    print("latent gain chosen:", sorted(chosen))
    for gc in chosen:
        ax.axvline(gc, color=INK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.18, 0.28, 0.4, 0.7, 1.2, 1.9, 3, 4.5, 6.5])
    ax.set_xticklabels(["0.1", "0.18", "0.28", "0.4", "0.7", "1.2", "1.9", "3", "4.5", "6.5"], fontsize=6.5)
    ax.set_xlabel("latent gain $g$")
    ax.set_ylabel(r"$\sum_s |\ln(\mathrm{SD}_\mathrm{sim} / \mathrm{SD}_\mathrm{real})|$", fontsize=7.5)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    _seed_legend(fig, 0.0, [Line2D([], [], color=INK, lw=0.8, ls=":", label="chosen $g$")])
    save(fig, "fig_app_latent_gain")


if __name__ == "__main__":
    fig_correction()
    fig_drift()
    fig_latent()
