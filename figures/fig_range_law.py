"""fig_range_law -- Appendix app:range-law (label fig:range-law): turn toward the dots against their range.

Three panels (2-8 moving dots, 1 moving dot, static dots). Bouts >= 10 s into the block (after the onset turn), dots
clearly to one side (|bearing| 45-135 deg), dot centroid inside the dish; turn toward = turn x sign of the bearing
(deg). Range bins 0-10, 10-20, 20-30, 30-45, 45-70 mm at 3.60 px/mm.
  * Real fish: per fish (>= 5 bouts in the bin) then mean over fish, 95 % percentile bootstrap over the 15 fish
    (4000, seed 0); the stimulus bouts of make_report_figures._dot_profile_frame (Chapter 3's profile bouts).
  * Models M6, D4, F at seed 0 (filled, solid) and 101 (open, dashed): their own rollouts, the archived profile streams
    of fig_profiles (fig_profiles_gen_<key>.pkl; 50 simulated fish per unit), same selection, per simulated fish then
    mean over fish. The profile code is at 3.67 px/mm, so ranges are converted by MM_3P67_TO_3P60.
Writes figures/out/fig_range_law.pdf (+ PNG preview) and prints every plotted value.

    uv run python -m figures.fig_range_law
"""
import sys
from pathlib import Path

from figures.style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from figures import data_figures as RF  # noqa: E402

BINS = [0, 10, 20, 30, 45, 70]
LAB = ["0–10", "10–20", "20–30", "30–45", "45–70"]
XM = np.arange(5.0)                # bins drawn evenly spaced (unequal widths: 10, 10, 10, 15, 25 mm)
MIN_N, N_BOOT = 5, 4000
UNITS = [("M6", 0, "26e"), ("M6", 101, "30d"), ("D4", 0, "11e"), ("D4", 101, "32a"), ("F", 0, "14f"), ("F", 101, "32b")]
UNIT_C = {"M6": MODEL_C["M6"], "D4": GREEN, "F": MODEL_C["F"]}
OFF = {"M6": -0.09, "D4": 0.0, "F": 0.09}
PANELS = [("mv", "2–8 moving dots"), ("one", "1 moving dot"), ("st", "Static dots")]
CACHE = paths.FIG_CACHE / "cache"


def _bins(rng):
    return pd.cut(rng, BINS, labels=LAB)


def per_fish(d, fish, val):
    g = d.groupby([fish, "rb"], observed=True)[val]
    return g.mean().where(g.size() >= MIN_N).unstack("rb").reindex(columns=LAB)


def boot(M):
    A = M.to_numpy(float)
    idx = np.random.default_rng(0).integers(0, A.shape[0], (N_BOOT, A.shape[0]))
    bs = np.array([np.nanmean(A[i], 0) for i in idx])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], 0)
    return np.nanmean(A, 0), lo, hi, np.sum(np.isfinite(A), 0)


def real_side():
    c = RF.load()
    s = RF._dot_profile_frame(c, keep_outside=False)
    bear = np.degrees(s.stim_angle_travel)
    s = s[(s.t_blk >= 10) & bear.abs().between(45, 135)].copy()
    s["tow"] = s.dtheta_deg * np.sign(np.degrees(s.stim_angle_travel))
    s["rb"] = _bins(s.dot_dist_px / RF.PX_PER_MM)
    mv = s.moving.astype(bool)
    sel = {"mv": mv & (s.N >= 2), "one": mv & (s.N == 1), "st": ~mv}
    return {k: boot(per_fish(s[v], "fish_id", "tow")) for k, v in sel.items()}


def model_side(key):
    g = pd.read_pickle(CACHE / f"fig_profiles_gen_{key}.pkl")
    g = g[(g.t_blk >= 10) & np.degrees(np.abs(g.phi)).between(45, 135)].copy()
    g["tow"] = g.dth * np.sign(g.phi)
    g["rb"] = _bins(g.dist_mm * MM_3P67_TO_3P60)
    mv = g.moving.astype(bool)
    sel = {"mv": mv & (g.n_dots >= 2), "one": mv & (g.n_dots == 1), "st": ~mv}
    return {k: boot(per_fish(g[v], "fish_id", "tow")) for k, v in sel.items()}


def main():
    R = real_side()
    M = {(n, s): model_side(k) for n, s, k in UNITS}
    for p, title in PANELS:
        print(f"== {title}")
        mu, lo, hi, n = R[p]
        print("  real      " + "  ".join(f"{l}: {m:5.1f} [{a:5.1f}, {b:5.1f}] n={k}" for l, m, a, b, k in zip(LAB, mu, lo, hi, n)))
        for (name, seed), v in M.items():
            mu_, lo_, hi_, n_ = v[p]
            inside = (mu_ >= lo) & (mu_ <= hi)
            print(f"  {name} s{seed:<4d}" + "  ".join(f"{l}: {m:5.1f}{'' if i else ' *'}" for l, m, i in zip(LAB, mu_, inside))
                  + "   (* outside the real 95 % CI)")

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.5), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.80, bottom=0.18, wspace=0.10)
    for ax, (p, title), letter in zip(axes, PANELS, "abc"):
        mu, lo, hi, _ = R[p]
        ax.fill_between(XM, lo, hi, color=REAL_C, alpha=0.14, lw=0, zorder=1)
        ax.plot(XM, mu, color=REAL_C, lw=1.6, marker="o", ms=3.6, zorder=4)
        for (name, seed), v in M.items():
            c = UNIT_C[name]
            ax.plot(XM + OFF[name], v[p][0], color=c, lw=1.0, ls="-" if seed == 0 else (0, (3, 1.6)), marker="o",
                    ms=3.0, mfc=c if seed == 0 else "white", mec=c, mew=0.8, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
        ax.set_xticks(XM); ax.set_xticklabels(LAB, fontsize=7)
        ax.set_xlim(-0.4, 4.4)
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.set_xlabel("range to the dots (mm)")
        panel(ax, letter, title)
    axes[0].set_ylabel("turn toward the dots (deg)")
    h = [(Line2D([], [], color=REAL_C, lw=1.6, marker="o", ms=3.6), "real fish (95 % CI over fish)")]
    h += [(Line2D([], [], color=UNIT_C[n], lw=1.0, marker="o", ms=3.0), n) for n in ("M6", "D4", "F")]
    h += [(Line2D([], [], color=MUTED, lw=1.0, marker="o", ms=3.0), "seed 0"),
          (Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 1.6)), marker="o", ms=3.0, mfc="white"), "seed 101")]
    fig.legend(*zip(*h), loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=6, frameon=False, handlelength=2.2,
               columnspacing=1.2, fontsize=6.8)
    save(fig, "fig_range_law")


if __name__ == "__main__":
    main()
