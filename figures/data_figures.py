"""Figures for the thesis Data chapter.

Reads the real data only (no model output) and writes vector PDFs into
``figures/out/``. Every number quoted in Chapter 3 comes from here or
from ``--stats``, so the prose and the figures cannot drift apart.

    uv run python -m figures.data_figures            # all figures
    uv run python -m figures.data_figures --stats    # print the numbers only

Palette: Okabe-Ito blue / vermillion / green, validated for CVD separation
(worst all-pairs dE 11.0 deuteranopia, 18.7 normal vision).
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figures import show_png  # noqa: E402
from twin import paths  # noqa: E402

OUT = paths.FIGURES
CLEAN = paths.BOUTS
RAW = paths.BOUTS_RAW
TD = paths.FRAMES
REPAIRED = paths.REPAIRED

FPS = 30.0
PX_PER_MM = 3.60          # physical scale (raw ROIdef); 3.67 is the nominal one
PAUSE_FR, STIM_FR = 1500, 1800
QUIET_S = 10.0
IMPOSSIBLE_MM_S = 50.0

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, FAINT = "#1a1a1a", "#666666", "#c8c8c8"
COND_C = {"moving": BLUE, "static": VERM, "none": MUTED}

TEXTWIDTH_IN = 6.3        # \textwidth at a4paper, 2.5 cm margins

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "grid.color": FAINT, "grid.linewidth": 0.4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=6)
    ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="right")


PNG_DIR = paths.FIGURES / "png"


def _save(fig, name):
    """figures/out/<name>.pdf + a PNG preview (shown inline in a notebook)."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_DIR / f"{name}.png", dpi=170)
    plt.close(fig)
    show_png(PNG_DIR / f"{name}.png")


def load():
    c = pd.read_csv(CLEAN)
    c["active"] = c["active"].astype(bool)
    return c


ONSET_GEOM = paths.CACHE / "onset_geometry.csv"
STIM_NONE = {"none", "inter_stim_pause", "bout", "None", "nan"}


def onset_geometry(rebuild=False):
    """Fish-to-shoal distance and bearing at each stimulus-block onset frame.

    Cached: building it reads all fifteen per-frame files. The stimulus is
    semi-closed-loop, so the shoal is placed relative to the fish's own pose --
    the across-fish spread of both quantities is exactly zero within a block,
    which is what makes them controlled variables rather than outcomes.
    """
    if ONSET_GEOM.exists() and not rebuild:
        return pd.read_csv(ONSET_GEOM)
    starts = np.array([PAUSE_FR + (PAUSE_FR + STIM_FR) * k for k in range(95)])
    rows = []
    for p in sorted(TD.glob("*_transformed_abs.csv")):
        fid = int(p.stem.split("_f")[1].split("_")[0])
        cols = ["frame", "stimulus", f"f{fid}_x", f"f{fid}_y", f"f{fid}_ori"]
        cols += [f"d{k}_{s}" for k in range(8) for s in ("x", "y", "stim")]
        d = pd.read_csv(p, usecols=cols).iloc[starts]
        fx, fy = d[f"f{fid}_x"].to_numpy(float), d[f"f{fid}_y"].to_numpy(float)
        ori = d[f"f{fid}_ori"].to_numpy(float)
        DX = np.stack([d[f"d{k}_x"].to_numpy(float) for k in range(8)], 1)
        DY = np.stack([d[f"d{k}_y"].to_numpy(float) for k in range(8)], 1)
        ACT = np.stack([~d[f"d{k}_stim"].astype(str).isin(STIM_NONE).to_numpy()
                        & np.isfinite(DX[:, k]) for k in range(8)], 1)
        n = np.maximum(ACT.sum(1), 1)
        cx = np.nansum(np.where(ACT, DX, 0), 1) / n
        cy = np.nansum(np.where(ACT, DY, 0), 1) / n
        br = (np.arctan2(cy - fy, cx - fx) - ori + np.pi) % (2 * np.pi) - np.pi
        rows.append(pd.DataFrame(dict(
            fish=fid, blk=np.arange(95), dist_px=np.hypot(cx - fx, cy - fy),
            bearing_deg=np.degrees(br), label=d["stimulus"].to_numpy())))
        print(f"    onset geometry: fish {fid}", flush=True)
    t = pd.concat(rows, ignore_index=True)
    ONSET_GEOM.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(ONSET_GEOM, index=False)
    return t


# --------------------------------------------------------------------------- #
# Fig 1 -- the stimulus programme
# --------------------------------------------------------------------------- #
def fig_stimulus(c):
    # NOTE: this figure is not currently included in the report -- Chapter 3 was
    # reorganised and dropped it. Kept intact so it can be re-added unchanged.
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 5.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.7, 1.3, 1.4], hspace=1.0)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) block timeline, first 6 stimulus blocks
    lab = c.groupby("blk")["cond"].first()
    n_show = 13
    t = 0.0
    for b in range(n_show):
        dur = (PAUSE_FR if b % 2 == 0 else STIM_FR) / FPS
        cond = lab.get(b, "none")
        ax0.add_patch(plt.Rectangle((t, 0), dur, 1, facecolor=COND_C.get(cond, MUTED),
                                    alpha=0.85 if cond != "none" else 0.18, lw=0))
        t += dur
    ax0.set_xlim(0, t); ax0.set_ylim(0, 1)
    ax0.set_yticks([]); ax0.set_xlabel("time from session start (s)")
    ax0.spines["left"].set_visible(False)
    for c_, l_ in ((MUTED, "pause (50 s)"), (BLUE, "moving dots (60 s)"),
                   (VERM, "static dots (60 s)")):
        ax0.plot([], [], lw=5, color=c_, alpha=0.85 if c_ != MUTED else 0.3, label=l_)
    ax0.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.42))
    _panel(ax0, "A", "Block structure: 95 pause/stimulus pairs, identical for all fish")

    # (b) condition grid -- blocks per (N dots, motion), pooled over side
    g = (c.loc[c.dots_on].groupby("blk")
         .agg(n=("n_dots_left", lambda s: np.nan_to_num(s.iloc[0])),
              nr=("n_dots_right", lambda s: np.nan_to_num(s.iloc[0])),
              cond=("cond", "first")))
    g["N"] = (g.n + g.nr).astype(int)
    tab = g.groupby(["N", "cond"]).size().unstack(fill_value=0)
    Ns = sorted(tab.index)
    w = 0.38
    xs = np.arange(len(Ns))
    for i, cond in enumerate(["moving", "static"]):
        v = [tab.loc[n, cond] if cond in tab.columns else 0 for n in Ns]
        ax1.bar(xs + (i - 0.5) * (w + 0.02), v, w, color=COND_C[cond], label=cond)
    ax1.set_xticks(xs); ax1.set_xticklabels(Ns)
    ax1.set_xlabel("number of dots $N$"); ax1.set_ylabel("blocks in the session")
    ax1.grid(axis="y"); ax1.set_axisbelow(True)
    ax1.set_ylim(top=16)
    ax1.legend(loc="upper center", ncol=2)
    _panel(ax1, "B", "Conditions: $N$ crossed with motion (both sides pooled)")

    # (c) onset geometry -- prescribed, and identical for every fish
    og = onset_geometry()
    mv = og.label.str.contains("s3.5").to_numpy()
    for m, col, lab in ((mv, BLUE, "moving"), (~mv, VERM, "static")):
        ax2.scatter(og.bearing_deg[m], og.dist_px[m] / PX_PER_MM, s=9,
                    color=col, alpha=0.55, lw=0)
        ax2.scatter([], [], s=26, color=col, lw=0, label=lab)
    ax2.axvline(0, color=FAINT, lw=0.6)
    ax2.set_xlabel("bearing to the shoal at block onset (deg, $+$ = fish's left)")
    ax2.set_ylabel("range (mm)")
    ax2.set_xlim(-180, 180); ax2.set_xticks([-180, -90, 0, 90, 180])
    ax2.set_ylim(top=og.dist_px.max() / PX_PER_MM * 1.28)
    ax2.grid(axis="y"); ax2.set_axisbelow(True); ax2.legend(loc="upper center", ncol=2)
    _panel(ax2, "C", "Onset geometry is prescribed: all 15 fish coincide exactly")

    _save(fig, "fig_stimulus_programme")


# --------------------------------------------------------------------------- #
# Fig -- non-steady vs steady behaviour
# --------------------------------------------------------------------------- #
def fig_states(c):
    a = c[c.active].copy()
    a["pos"] = a.run_pos.astype(int)
    fish = sorted(a.fish_id.unique())
    POSMAX, STEADY = 25, 60

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.3),
                             gridspec_kw={"wspace": 0.48})
    ax0, ax1, ax2 = axes

    for ax, col, ylab, conv in (
            (ax0, "dt_s", "interbout interval (s)", 1.0),
            (ax1, "distance_px", "displacement (mm)", 1.0 / PX_PER_MM)):
        for f in fish:
            g = a[(a.fish_id == f) & (a.pos < POSMAX)]
            m = g.groupby("pos")[col].median() * conv
            ax.plot(m.index, m.values, lw=0.6, color=FAINT, zorder=1)
        m = a[a.pos < POSMAX].groupby("pos")[col].median() * conv
        ax.plot(m.index, m.values, lw=1.6, color=BLUE, zorder=3)
        steady = a.loc[a.pos >= STEADY, col].median() * conv
        ax.axhline(steady, color=VERM, lw=1.0, ls=(0, (3, 2)), zorder=2)
        ax.set_xlabel("bout position in run"); ax.set_ylabel(ylab)
        ax.set_xlim(0, POSMAX - 1)
    ax0.plot([], [], lw=1.6, color=BLUE, label="pooled")
    ax0.plot([], [], lw=1.0, color=VERM, ls=(0, (3, 2)), label="steady state")
    ax0.legend(loc="upper right")
    _panel(ax0, "A", "Tempo")
    _panel(ax1, "B", "Displacement")

    for cond, dur in (("moving", 60.0), ("static", 60.0), ("none", 50.0)):
        s = a[a.cond == cond]
        nb = c.loc[c.cond == cond, "blk"].nunique()
        bins = np.arange(0, dur + 2.5, 2.5)
        h, _ = np.histogram(s.t_blk, bins=bins)
        ax2.plot(0.5 * (bins[1:] + bins[:-1]), h / (2.5 * nb * len(fish)),
                 lw=1.3, color=COND_C[cond],
                 label="pause" if cond == "none" else cond)
    ax2.set_xlabel("time into block (s)"); ax2.set_ylabel("bout rate (Hz)")
    ax2.set_xlim(0, 60); ax2.legend(loc="lower right")
    _panel(ax2, "C", "Bout rate")
    _save(fig, "fig_states")


# --------------------------------------------------------------------------- #
# Fig 2 -- bout detection
# --------------------------------------------------------------------------- #
def fig_detection(c):
    fid, t0_s, span_s = 0, 600.0, 25.0
    p = next(TD.glob(f"*_f{fid}_transformed_abs.csv"))
    d = pd.read_csv(p, usecols=["frame", f"f{fid}_x", f"f{fid}_y"])
    a, b = int(t0_s * FPS), int((t0_s + span_s) * FPS)
    seg = d.iloc[a:b]
    x = gaussian_filter1d(seg[f"f{fid}_x"].to_numpy(float), 2.0)
    y = gaussian_filter1d(seg[f"f{fid}_y"].to_numpy(float), 2.0)
    v = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))] * FPS
    v = pd.Series(v).rolling(3, center=True, min_periods=1).mean().to_numpy()
    tt = seg["frame"].to_numpy() / FPS

    ons = c.loc[(c.fish_id == fid) & c.frame_onset.between(a, b), "frame_onset"].to_numpy()

    # Single panel. The turn-angle histogram that used to sit alongside was
    # dropped: its content is stated numerically in the text and the panel was
    # not carrying its own weight.
    fig, ax0 = plt.subplots(1, 1, figsize=(TEXTWIDTH_IN, 2.2))
    vm = v / PX_PER_MM
    ax0.plot(tt, vm, lw=0.9, color=INK, zorder=2)
    # Onsets sit at the local speed MINIMUM preceding each peak, so they mark the
    # foot of a rise, not its summit -- plotting them on the curve makes that
    # placement visible instead of hiding it behind a vertical rule.
    oi = np.clip(ons - a, 0, len(vm) - 1)
    # 2026-10-14 (Arthur's choice): ONE series, the trough-snap placement -- from the recorded onset, walk FORWARD
    # to the right edge of its flat plateau (segment_bouts._snap_to_trough), computed here, not re-detected. It
    # differs from the data set's onsets only where a bout starts from complete stillness (here 2 of 8 onsets).
    fixed = []
    for j in oi:
        k = int(j)
        while k + 1 < len(v) and v[k + 1] <= v[k]:
            k += 1
        fixed.append(k)
    fixed = np.array(fixed)
    print(f"  detection: {int((fixed != oi).sum())} of {len(oi)} onsets moved by the snap")
    ax0.scatter(tt[fixed], vm[fixed], s=16, color=VERM, zorder=4)
    ax0.set_xlabel("time (s)"); ax0.set_ylabel("speed (mm s$^{-1}$)")
    ax0.set_xlim(tt[0], tt[-1]); ax0.set_ylim(bottom=-0.6)
    ax0.set_title(f"Speed trace, fish {fid}", loc="left", pad=6)
    _save(fig, "fig_bout_detection")




# --------------------------------------------------------------------------- #
# Fig 3 -- per-fish marginals
# --------------------------------------------------------------------------- #
def fig_marginals(c):
    a = c[c.active]
    fish = sorted(a.fish_id.unique())
    specs = [("dtheta_deg", np.arange(0, 181, 5), r"$|\Delta\theta|$ (deg)", True),
             ("distance_px", np.arange(0, 16.1, 0.4), "bout displacement (mm)", False),
             ("dt_s", np.arange(0, 6.05, 0.15), "interbout interval (s)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.1))
    for ax, (col, bins, xlab, is_abs) in zip(axes, specs):
        for f in fish:
            v = a.loc[a.fish_id == f, col].to_numpy(float)
            if is_abs:
                v = np.abs(v)
            if col == "distance_px":
                v = v / PX_PER_MM
            h, e = np.histogram(v, bins=bins, density=True)
            ax.plot(0.5 * (e[1:] + e[:-1]), h, lw=0.6, color=FAINT, zorder=1)
        v = a[col].to_numpy(float)
        if is_abs:
            v = np.abs(v)
        if col == "distance_px":
            v = v / PX_PER_MM
        h, e = np.histogram(v, bins=bins, density=True)
        ax.plot(0.5 * (e[1:] + e[:-1]), h, lw=1.6, color=BLUE, zorder=3)
        ax.set_xlabel(xlab); ax.set_xlim(bins[0], bins[-1])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("density")
    axes[0].plot([], [], lw=0.6, color=FAINT, label="individual fish")
    axes[0].plot([], [], lw=1.6, color=BLUE, label="all fish pooled")
    axes[0].legend(loc="upper right")
    _save(fig, "fig_marginals")


# --------------------------------------------------------------------------- #
# Fig 4 -- quiescence
# --------------------------------------------------------------------------- #
def fig_quiescence(c):
    fish = sorted(c.fish_id.unique())
    share = (c.groupby("fish_id")
             .apply(lambda g: g.loc[g.dt_s > QUIET_S, "dt_s"].sum() / g.dt_s.sum(),
                    include_groups=False) * 100)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.4),
                                   width_ratios=[1, 1.25])
    ax0.bar(fish, [share[f] for f in fish], color=BLUE, width=0.72)
    ax0.set_xlabel("fish"); ax0.set_ylabel("session time static (%)")
    ax0.set_xticks(fish); ax0.grid(axis="y"); ax0.set_axisbelow(True)
    _panel(ax0, "A", "Static share, per fish")

    for i, f in enumerate(fish):
        g = c[c.fish_id == f]
        gaps = g.loc[g.dt_s > QUIET_S]
        for t0, dur in zip(gaps.t_onset_s, gaps.dt_s):
            ax1.add_patch(plt.Rectangle((t0 / 60, i - 0.42), dur / 60, 0.84,
                                        facecolor=VERM, lw=0))
    ax1.set_xlim(0, c.t_onset_s.max() / 60); ax1.set_ylim(-0.7, len(fish) - 0.3)
    ax1.set_yticks(fish); ax1.set_xlabel("session time (min)"); ax1.set_ylabel("fish")
    ax1.invert_yaxis()
    _panel(ax1, "B", "When the static stretches happen")
    _save(fig, "fig_quiescence")


# --------------------------------------------------------------------------- #
# Fig 5 -- cleaning: tracking excursions and impossible bouts
# --------------------------------------------------------------------------- #
def fig_cleaning(c):
    raw = pd.read_csv(RAW)
    v_raw = (raw.distance_px / PX_PER_MM / raw.dt_s).to_numpy()
    v_cl = (c.loc[c.active, "distance_px"] / PX_PER_MM
            / c.loc[c.active, "dt_s"]).to_numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.4),
                                   gridspec_kw={"wspace": 0.42})
    bins = np.logspace(-1, 2.6, 70)
    ax0.hist(v_raw[np.isfinite(v_raw) & (v_raw > 0)], bins=bins, color=FAINT,
             lw=0, label="as detected")
    ax0.hist(v_cl[np.isfinite(v_cl) & (v_cl > 0)], bins=bins, histtype="step",
             lw=1.2, color=BLUE, label="modelling set")
    ax0.axvline(IMPOSSIBLE_MM_S, color=VERM, lw=1.0, ls=(0, (3, 2)))
    ax0.text(IMPOSSIBLE_MM_S * 1.1, ax0.get_ylim()[1] * 0.75,
             "not swimming", color=VERM, fontsize=6.5)
    ax0.set_xscale("log"); ax0.set_yscale("log")
    ax0.set_xlabel("mean bout speed (mm s$^{-1}$)"); ax0.set_ylabel("bouts")
    ax0.legend(loc="upper left")
    _panel(ax0, "A", "Implied bout speed")

    # An excursion is a leap and return within <= 5 frames, so it reads as a spike
    # in frame-to-frame speed. Plotted against time rather than as an xy path: the
    # displacement is a few pixels, which an equal-aspect path renders as a smudge.
    rf = pd.read_csv(REPAIRED)
    fid = int(rf.fish_id.value_counts().idxmax())
    f0 = int(rf.loc[rf.fish_id == fid, "frame"].min())
    p = next(TD.glob(f"*_f{fid}_transformed_abs.csv"))
    d = pd.read_csv(p, usecols=["frame", f"f{fid}_x", f"f{fid}_y"])
    w = d.iloc[max(f0 - 60, 0):f0 + 90]
    xs = w[f"f{fid}_x"].to_numpy(float); ys = w[f"f{fid}_y"].to_numpy(float)
    sp = np.r_[0.0, np.hypot(np.diff(xs), np.diff(ys))] * FPS / PX_PER_MM
    tw = (w.frame.to_numpy() - f0) / FPS
    bad = w.frame.isin(set(rf.loc[rf.fish_id == fid, "frame"])).to_numpy()
    ax1.plot(tw, sp, lw=0.9, color=INK, zorder=2)
    ax1.scatter(tw[bad], sp[bad], s=18, color=VERM, zorder=3,
                label="flagged frame")
    ax1.axhline(100.0, color=VERM, lw=1.0, ls=(0, (3, 2)))
    ax1.text(tw[0] + 0.1, 112, "100 mm s$^{-1}$ threshold", color=VERM, fontsize=6.5)
    ax1.set_xlabel(f"time from frame {f0} (s)")
    ax1.set_ylabel("step speed (mm s$^{-1}$)")
    ax1.set_xlim(tw[0], tw[-1]); ax1.legend(loc="upper right")
    _panel(ax1, "B", f"An excursion (fish {fid})")
    _save(fig, "fig_cleaning")


# --------------------------------------------------------------------------- #
# Fig 6 -- real-data reference: turn-toward and radial occupancy
# --------------------------------------------------------------------------- #
def _turn_toward(g):
    """P(turn toward the dots) among turn bouts. + dtheta = left, + bearing = left."""
    m = g.dots_on & g.stim_angle_travel.notna() & (g.dtheta_deg.abs() >= 10)
    s = g.loc[m]
    if not len(s):
        return np.nan
    return float((np.sign(s.dtheta_deg) == np.sign(s.stim_angle_travel)).mean())


def fig_reference(c):
    a = c[c.active]
    fish = sorted(a.fish_id.unique())
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.3))

    w = 0.38
    xs = np.arange(len(fish))
    for i, cond in enumerate(["moving", "static"]):
        v = [_turn_toward(a[(a.fish_id == f) & (a.cond == cond)]) for f in fish]
        ax0.bar(xs + (i - 0.5) * (w + 0.02), v, w, color=COND_C[cond], label=cond)
    ax0.axhline(0.5, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax0.set_ylim(0.4, 0.80); ax0.set_xticks(xs); ax0.set_xticklabels(fish)
    ax0.set_xlabel("fish"); ax0.set_ylabel("P(turn toward dots)")
    ax0.grid(axis="y"); ax0.set_axisbelow(True)
    ax0.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    _panel(ax0, "A", "Turning toward the dots, per fish")

    bins = np.linspace(0, 1, 41)
    for f in fish:
        h, e = np.histogram(a.loc[a.fish_id == f, "r_norm"], bins=bins, density=True)
        ax1.plot(0.5 * (e[1:] + e[:-1]), h, lw=0.6, color=FAINT, zorder=1)
    h, e = np.histogram(a.r_norm, bins=bins, density=True)
    ax1.plot(0.5 * (e[1:] + e[:-1]), h, lw=1.6, color=BLUE, zorder=3)
    ax1.axvline(float(a.r_norm.median()), color=VERM, lw=1.0, ls=(0, (3, 2)))
    ax1.set_xlabel("radial position $r/R$  (0 = centre, 1 = wall)")
    ax1.set_ylabel("density"); ax1.set_xlim(0, 1)
    _panel(ax1, "B", "Radial occupancy")
    _save(fig, "fig_reference")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Sec 3.3 -- behavioural properties
# --------------------------------------------------------------------------- #
def _fish_mean_ci(per_fish, min_n=3):
    """Mean over fish with a 95% normal CI -- the fish is the replicate unit.

    Column-wise and nan-safe: a bin covered by only some fish still gets a mean
    from those that have it, and a bin with fewer than ``min_n`` fish is dropped
    rather than reported with a meaningless interval.
    """
    a = np.asarray(per_fish, float)
    n = np.sum(np.isfinite(a), axis=0)
    m = np.full(a.shape[1], np.nan)
    e = np.full(a.shape[1], np.nan)
    for k in np.flatnonzero(n >= min_n):
        v = a[:, k][np.isfinite(a[:, k])]
        m[k] = v.mean()
        e[k] = 1.96 * v.std(ddof=1) / np.sqrt(v.size)
    return m, e


# Definitions copied from the Data_Exploration notebook (§6B and §6C) so the report
# and the notebook cannot disagree. The BLOCK is the replicate unit for both bars.
DTHETA0, AHEAD_DEG, BEHIND_DEG = 10.0, 60.0, 120.0
MIN_TURNS, MIN_BLOCKS, MIN_EACH = 5, 8, 5


def _t_ci(v, min_n=MIN_BLOCKS):
    """Mean and a t-based 95% CI of a sample of block-level values."""
    from scipy.stats import t as _t
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < min_n:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), np.nan, np.nan
    h = _t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def _stim_bouts(a, cond=None):
    """Bouts with a dot on screen and a defined chord-referenced bearing."""
    s = a[(a.cond != "none") & np.isfinite(a.stim_angle_travel)]
    return s if cond is None else s[s.cond == cond]


def _toward_blocks(g, thr=0.0):
    """Per block, P(turn toward the shoal) over that block's bouts.

    ``thr`` is the forward/turn threshold on |dtheta|. The report uses 0, i.e.
    every bout counts. MEASURED: restricting to turns above 10 deg (the
    Data_Exploration convention, motivated by chord-angle sign being noisy at
    small angles) changes the departure from chance by 7 % under motion and 2 %
    under static, and moves no fish across chance -- so the result does not rest
    on the threshold and the simpler definition is preferred here.
    """
    t = g[g.dtheta_deg.abs() > thr]
    if not len(t):
        return np.array([])
    tw = (np.sign(t.dtheta_deg) == np.sign(t.stim_angle_travel)).to_numpy(float)
    per = (pd.DataFrame({"blk": t.blk.to_numpy(), "tw": tw})
           .groupby("blk").tw.agg(["mean", "size"]))
    return per.loc[per["size"] >= MIN_TURNS, "mean"].to_numpy()




BLIND_REPS, BLIND_SEED = 10, 0


def _heading_blocks(g, phi):
    """Per block, P(the bout leaves the fish heading within 90 deg of the shoal), all bouts.

    ``phi`` is the bearing to the shoal in the incoming-chord frame, so the outgoing
    bearing is wrap(phi - dtheta). A straight bout already aimed at the shoal counts.
    """
    ok = np.isfinite(phi)
    out = np.abs((phi - np.radians(g.dtheta_deg.to_numpy()) + np.pi) % (2 * np.pi) - np.pi)
    per = (pd.DataFrame({"blk": g.blk.to_numpy()[ok], "tw": (out[ok] < np.pi / 2).astype(float)})
           .groupby("blk").tw.agg(["mean", "size"]))
    return per.loc[per["size"] >= MIN_TURNS, "mean"].to_numpy()


def _phantom_phi(g, tracks, rng):
    """Bearing to a shoal the fish never saw: for each block of ``g`` (one fish), the
    shoal centroid of another of that fish's blocks of the same condition, at the
    nearest time in block. Same Data_Exploration §6B' construction."""
    th = g.heading_rad.to_numpy() - np.radians(g.dtheta_deg.to_numpy())
    cx, cy = np.full(len(g), np.nan), np.full(len(g), np.nan)
    pos = {b: np.flatnonzero(g.blk.to_numpy() == b) for b in np.unique(g.blk)}
    for b, idx in pos.items():
        others = [k for k in tracks if k != b]
        tr = tracks[others[rng.integers(len(others))]]
        t = g.t_blk.to_numpy()[idx]
        i = np.clip(np.searchsorted(tr[:, 0], t), 0, len(tr) - 1)
        j = np.clip(i - 1, 0, len(tr) - 1)
        k = np.where(np.abs(tr[j, 0] - t) < np.abs(tr[i, 0] - t), j, i)
        cx[idx], cy[idx] = tr[k, 1], tr[k, 2]
    return (np.arctan2(cy - g.y_px.to_numpy(), cx - g.x_px.to_numpy()) - th + np.pi) % (2 * np.pi) - np.pi


def fig_stim_bars(c):
    """P(the bout leaves the fish heading at the shoal) per fish, real vs a blind fish.

    One panel per condition, same y scale. The blind fish is the fish's own PAUSE
    bouts (nothing on screen) scored against the shoal of a random stimulus block of
    the same fish and condition, at the same time in block, averaged over draws: it
    keeps wall avoidance and the dish geometry, which put a dots-blind fish above 0.5
    (it turns toward the centre, and the shoal usually lies on the centre side).
    Both bars start from 0.5; whiskers are 95 % t-intervals across blocks.
    """
    a = c[c.active]
    fish = sorted(a.fish_id.unique())
    rng = np.random.default_rng(BLIND_SEED)
    conds = ["moving", "static"]
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 3.9), sharex=True, sharey=True)
    w, xs = 0.38, np.arange(len(fish))
    for ax, cond in zip(axes, conds):
        real, blind, real_b, blind_b = [], [], [], []
        for f in fish:
            s = _stim_bouts(a[a.fish_id == f], cond)
            real_b.append(np.asarray(_heading_blocks(s, s.stim_angle_travel.to_numpy()), float))
            real.append(_t_ci(real_b[-1]))
            tracks = {b: g.sort_values("t_blk")[["t_blk", "dot_cx", "dot_cy"]].dropna().to_numpy()
                      for b, g in s.groupby("blk")}
            p = a[(a.fish_id == f) & (a.cond == "none")]
            blind_b.append([np.asarray(_heading_blocks(p, _phantom_phi(p, tracks, rng)), float)
                            for _ in range(BLIND_REPS)])
            blind.append(np.nanmean([_t_ci(x) for x in blind_b[-1]], axis=0))
        bl = np.nanmean([_t_ci(np.concatenate([b[k] for b in blind_b])) for k in range(BLIND_REPS)], axis=0)
        print(f"  {cond}: P(heading at the dots) over stimulus blocks %.3f [%.3f, %.3f], blind fish %.3f [%.3f, %.3f]; "
              % (*_t_ci(np.concatenate(real_b)), *bl)
              + f"above the blind level in {sum(r[0] > b[0] for r, b in zip(real, blind))} of {len(fish)} fish")
        for i, (st, col, lab) in enumerate([(real, COND_C[cond], f"{cond} dots"),
                                            (blind, FAINT, "blind fish")]):
            st = np.asarray(st, float)
            x = xs + (i - 0.5) * (w + 0.02)
            ax.bar(x, st[:, 0] - 0.5, w, bottom=0.5, color=col, label=lab)
            ax.errorbar(x, st[:, 0], yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], fmt="none",
                        ecolor=INK, elinewidth=0.8, capsize=1.8)
        ax.axhline(0.5, color=INK, lw=1.0)
        ax.set_ylabel("P(heading at the shoal)")
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.12))
        ax.set_title(f"{cond.capitalize()} dots", loc="left", pad=6)
    ax = axes[-1]
    ax.set_xticks(xs); ax.set_xticklabels(fish)
    ax.set_xlim(-0.75, len(fish) - 0.3)
    ax.set_xlabel("fish")
    ax.set_ylim(0.30, 0.78); ax.set_yticks([0.4, 0.5, 0.6, 0.7])
    _save(fig, "fig_stim_bars")


# Group-size conditions and the model notebook's colour scheme (PuBu for motion,
# OrRd for static, darker = more dots), so the real-data figure here and the
# model comparison in Chapter 5 are read off the same key.
STIM_ORDER = [(1, True), (2, True), (4, True), (6, True), (8, True),
              (1, False), (4, False), (8, False)]


def _stim_colours():
    pal = lambda name, n: [plt.get_cmap(name)(x) for x in np.linspace(0.35, 0.95, n)]
    mv = [s for s in STIM_ORDER if s[1]]
    st = [s for s in STIM_ORDER if not s[1]]
    return {**dict(zip(mv, pal("PuBu", len(mv)))),
            **dict(zip(st, pal("OrRd", len(st))))}


def _dot_profile_frame(c, keep_outside=False, bin_s=2.0):
    """Stimulus bouts with distance (mm) and time bin; the centroid-outside-dish rule unless keep_outside."""
    from twin.segment_bouts import load_physical_dish
    dish = load_physical_dish(paths.ROI_PHYS)
    a = c[c.active].copy()
    a["N"] = (np.nan_to_num(a.n_dots_left) + np.nan_to_num(a.n_dots_right)).astype(int)
    s = a[(a.stimulus != "inter_stim_pause") & a.dot_dist_px.notna()].copy()
    g = s.join(dish.rename(columns={"radius": "R"}), on="fish_id")
    if not keep_outside:        # keep_outside=True: the same curves without the centroid-outside-dish rule
        s = s[~(np.hypot(g.dot_cx - g.cx, g.dot_cy - g.cy) > g.R).to_numpy()]
    s["dist_mm"] = s.dot_dist_px / PX_PER_MM
    s["tbin"] = (np.floor(s.t_blk / bin_s) + 0.5) * bin_s
    return s


def _blind_profile_frame(c, keep_outside=False, bin_s=2.0, reps=BLIND_REPS, seed=BLIND_SEED):
    """A blind fish for the distance profile: the fish's own PAUSE bouts (nothing on screen)
    scored against the dots of a random STATIC block of the programme, anchored on the fish's
    pose at the pause's first bout -- as the rig anchors every block on the fish at onset, and as
    the real curves' onset geometry is reproduced from the first stimulus bout (median distance
    error 1.4 px) -- and read at the same time since onset. Pauses last 50 s, so the curve
    stops there. Same bins, same centroid-outside-dish rule as ``_dot_profile_frame``; ``reps``
    static blocks per pause."""
    from twin import model_figures as MF
    from twin.segment_bouts import load_physical_dish
    prog, _, _ = MF._programme()
    dish = load_physical_dish(paths.ROI_PHYS)
    static = np.flatnonzero(prog.is_stim_of & ~prog.moving_of.astype(bool))
    rng = np.random.default_rng(seed)
    a = c[c.active & (c.stimulus == "inter_stim_pause")]
    rows = []
    for (f, b), g in a.groupby(["fish_id", "blk"]):
        g = g.sort_values("t_blk")
        a0 = g.iloc[0]
        anchor = (a0.x_px, a0.y_px, a0.f_ori_rad)
        off = g.frame_onset.to_numpy(int) - int(prog.starts[b])
        x, y, t = g.x_px.to_numpy(), g.y_px.to_numpy(), g.t_blk.to_numpy()
        d = dish.loc[f]
        for _ in range(reps):
            sb = static[rng.integers(len(static))]
            fr = int(prog.starts[sb]) + off
            for i in np.flatnonzero(fr < int(prog.ends[sb])):
                cx, cy = prog.centroid(int(fr[i]), anchor)
                if not np.isfinite(cx):
                    continue
                if not keep_outside and np.hypot(cx - d.cx, cy - d.cy) > d.radius:
                    continue
                rows.append((f, t[i], np.hypot(x[i] - cx, y[i] - cy) / PX_PER_MM))
    s = pd.DataFrame(rows, columns=["fish_id", "t_blk", "dist_mm"])
    s["tbin"] = (np.floor(s.t_blk / bin_s) + 0.5) * bin_s
    return s


def _band(sub, n_boot, rng):
    """Mean over fish of each fish's bin means, and a 95 % percentile bootstrap over fish."""
    M = sub.groupby(["fish_id", "tbin"]).dist_mm.mean().unstack("tbin").sort_index(axis=1)
    A = M.to_numpy(float)
    idx = rng.integers(0, A.shape[0], size=(n_boot, A.shape[0]))
    bs = np.array([np.nanmean(A[i], axis=0) for i in idx])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return M.columns.to_numpy(float), np.nanmean(A, axis=0), lo, hi


def _dot_profile_ax(ax, s, n_boot=4000, seed=0, blind=None):
    """Draw the per-condition curves of ``s`` on ``ax`` (and the blind fish of ``blind``, if given)."""
    cm = _stim_colours()
    rng = np.random.default_rng(seed)
    if blind is not None:       # under the condition curves, its own rng so they are unchanged
        x, mu, lo, hi = _band(blind, n_boot, np.random.default_rng(seed + 1))
        ax.plot(x, mu, lw=1.3, color=MUTED, ls=(0, (3, 1.6)), label="blind fish", zorder=1)
        ax.fill_between(x, lo, hi, color=MUTED, alpha=0.14, lw=0, zorder=0)
    for key in STIM_ORDER:
        N, moving = key
        sub = s[(s.N == N) & (s.moving.astype(bool) == moving)]
        if sub.empty:
            continue
        M = (sub.groupby(["fish_id", "tbin"]).dist_mm.mean()
             .unstack("tbin").sort_index(axis=1))
        A = M.to_numpy(float)
        mu = np.nanmean(A, axis=0)
        idx = rng.integers(0, A.shape[0], size=(n_boot, A.shape[0]))
        bs = np.array([np.nanmean(A[i], axis=0) for i in idx])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        x = M.columns.to_numpy(float)
        lab = f"{N} dot{'s' if N != 1 else ''}, {'motion' if moving else 'static'}"
        ax.plot(x, mu, lw=1.5, color=cm[key], label=lab)
        ax.fill_between(x, lo, hi, color=cm[key], alpha=0.18, lw=0)
    ax.set_xlim(0, 60)
    ax.set_xlabel("time into block (s)")
    ax.grid(axis="y"); ax.set_axisbelow(True)


def fig_dot_profile(c, n_boot=4000, seed=0, bin_s=2.0, keep_outside=False, name="fig_dot_profile_single",
                    save=True):
    """Distance to the shoal across the block, by group size -- ONE panel (the report uses the pair below).

    Matches model_figures.real_dot_profile: the centroid-outside-dish rule, a
    per-fish mean over that fish's blocks first, then a band across fish. The
    band is a 95% percentile bootstrap over fish, not +/- 1 SEM -- see the
    notebook, where the two agree to ~4% of width but differ twofold in coverage.
    Distances use the PHYSICAL scale of Sec. 3.1, not the 3.67 nominal one.
    """
    fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH_IN, 3.1))
    _dot_profile_ax(ax, _dot_profile_frame(c, keep_outside, bin_s), n_boot, seed)
    ax.set_ylabel("distance to the shoal (mm)")
    ax.legend(fontsize=6.6, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              handlelength=1.4, frameon=False, title="stimulus", title_fontsize=7)
    if not save:                # return the figure instead (e.g. to show it in a notebook)
        return fig
    _save(fig, name)


def fig_dot_profile_pair(c, n_boot=4000, seed=0, bin_s=2.0):
    """The report figure: (A) bouts with the shoal centroid outside the dish left out, as the
    profile readouts are defined; (B) the same curves with them kept. Shared y axis, one legend."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.0), sharey=True)
    for ax, keep, letter, title in [(axes[0], False, "A", "Outside-dish bouts left out"),
                                    (axes[1], True, "B", "Outside-dish bouts kept")]:
        _dot_profile_ax(ax, _dot_profile_frame(c, keep, bin_s), n_boot, seed,
                        blind=_blind_profile_frame(c, keep, bin_s))
        _panel(ax, letter, title)
    axes[0].set_ylabel("distance to the shoal (mm)")
    # legend: motion on the first row, static on the second (matplotlib fills column-major)
    h, lab = axes[0].get_legend_handles_labels()
    mv = [(hh, ll) for hh, ll in zip(h, lab) if "motion" in ll]
    st = [(hh, ll) for hh, ll in zip(h, lab) if "static" in ll and "blind" not in ll]
    st += [(hh, ll) for hh, ll in zip(h, lab) if "blind" in ll]
    blank = (mpl.lines.Line2D([], [], alpha=0), "")
    st += [blank] * (len(mv) - len(st))
    order = [x for pair in zip(mv, st) for x in pair]
    fig.legend(*zip(*order), ncol=len(mv), loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=6.6,
               handlelength=1.4, frameon=False, columnspacing=1.2)
    fig.tight_layout()
    _save(fig, "fig_dot_profile")


def _binned(x, y, bins):
    """Mean of y in bins of x; NaN where a bin is empty."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    idx = np.digitize(x, bins) - 1
    m = np.full(len(bins) - 1, np.nan)
    n = np.zeros(len(bins) - 1)
    for k in range(len(bins) - 1):
        sel = idx == k
        n[k] = sel.sum()
        if n[k] >= 3:
            m[k] = y[sel].mean()
    return m, n, idx


def fig_spatial(c):
    """Where the fish are, and the directed turn that keeps them there."""
    from twin.segment_bouts import load_physical_dish
    dish = load_physical_dish(paths.ROI_PHYS)
    a = c[c.active].copy()
    fish = sorted(a.fish_id.unique())

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 5.3))
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 1.5], hspace=0.45, wspace=0.18)
    for i, f in enumerate(fish):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        g = a[a.fish_id == f]
        cx, cy, R = dish.cx[f], dish.cy[f], dish.radius[f]
        u = (g.x_px - cx) / R
        v = (g.y_px - cy) / R
        ax.hexbin(u, v, gridsize=22, extent=(-1.05, 1.05, -1.05, 1.05),
                  cmap="magma_r", mincnt=1, linewidths=0)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec=INK, lw=0.8))
        ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"fish {f}", fontsize=6.5, pad=1.5, color=MUTED,
                     fontweight="normal")
    fig.text(0.008, 0.955, "A", fontsize=9, fontweight="bold", va="top")
    fig.text(0.06, 0.965, "Occupancy, each fish in its own dish "
             "(black circle = the dish wall)", fontsize=8.5, fontweight="bold",
             va="top")

    ax = fig.add_subplot(gs[3, :])
    t = a[(a.dtheta_deg.abs() >= 10) & a.wall_psi_rad.notna()].copy()
    t["toward"] = np.sign(t.dtheta_deg) == np.sign(t.wall_psi_rad)
    bins = np.linspace(0, 1, 21)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    per = []
    for f in fish:
        s = t[t.fish_id == f]
        m, _, _ = _binned(s.r_norm, s.toward.astype(float), bins)
        ax.plot(ctr, m, lw=0.6, color=FAINT, zorder=1)
        per.append(m)
    m, e = _fish_mean_ci(np.array(per))
    ax.plot(ctr, m, lw=1.7, color=BLUE, zorder=3)
    ax.fill_between(ctr, m - e, m + e, color=BLUE, alpha=0.18, lw=0, zorder=2)
    ax.axhline(0.5, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(0, 1); ax.set_xlabel("radial position $r/R$  (0 = centre, 1 = wall)")
    ax.set_ylabel("P(turn toward centre)")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    _panel(ax, "B", "Near the wall the turn is directed inward")
    _save(fig, "fig_spatial")


def fig_sequential(c):
    """One-bout memory, per channel.

    The signed turn is included deliberately as the negative case: its lag-1
    correlation has no consistent sign across fish, so turn DIRECTION carries no
    one-bout memory while turn MAGNITUDE does. A panel of consecutive large-turn
    reversals was tried and dropped -- only 88 such pairs exist in the whole
    dataset and 54 of them belong to one fish, so per-fish rates are noise.
    """
    a = c[c.active].copy()
    a["abs_dtheta"] = a.dtheta_deg.abs()
    fish = sorted(a.fish_id.unique())
    chans = [("distance_px", "A", "Displacement"),
             ("dt_s", "B", "Interval"),
             ("abs_dtheta", "C", r"$|\Delta\theta|$"),
             ("dtheta_deg", "D", r"signed $\Delta\theta$")]

    def lag1(g, col):
        v = g[col].to_numpy(float)
        r = g.run_id.to_numpy()
        ok = r[1:] == r[:-1]
        x, y = v[:-1][ok], v[1:][ok]
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 30 else np.nan

    LAGS = (1, 2, 3, 4, 5)

    def lagk(g, col, k):
        v = g[col].to_numpy(float)
        r = g.run_id.to_numpy()
        if len(v) <= k:
            return np.nan
        ok = r[k:] == r[:-k]
        x, y = v[:-k][ok], v[k:][ok]
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 30 else np.nan

    def med_ci(v, n_boot=2000, seed=1):
        rng = np.random.default_rng(seed)
        v = np.asarray(v, float)[np.isfinite(v)]
        bs = np.median(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
        return np.median(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 2.4), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, (col, letter, title) in zip(axes, chans):
        M = np.vstack([[lagk(a[a.fish_id == f], col, k) for f in fish] for k in LAGS])
        for j in range(M.shape[1]):
            ax.plot(LAGS, M[:, j], lw=0.6, color=FAINT, zorder=1)
        med = np.array([med_ci(M[i]) for i in range(len(LAGS))])
        ax.fill_between(LAGS, med[:, 1], med[:, 2], color=BLUE, alpha=0.20, lw=0, zorder=2)
        ax.plot(LAGS, med[:, 0], lw=1.7, color=BLUE, marker="o", ms=3.5, zorder=4)
        # A one-step (Markov) dependence would decay geometrically from lag 1.
        ax.plot(LAGS, [med[0, 0] ** k for k in LAGS], lw=1.1, color=VERM,
                ls=(0, (3, 2)), zorder=3)
        ax.axhline(0, color=INK, lw=0.8, zorder=2)
        ax.set_xticks(LAGS); ax.set_xlabel("lag (bouts)")
        ax.grid(axis="y"); ax.set_axisbelow(True)
        _panel(ax, letter, title)
    axes[0].set_ylabel("correlation within a run")
    axes[0].plot([], [], lw=1.7, color=BLUE, marker="o", ms=3.5, label="median over fish")
    axes[0].plot([], [], lw=1.1, color=VERM, ls=(0, (3, 2)), label="one-step decay")
    axes[0].plot([], [], lw=0.6, color=FAINT, label="individual fish")
    axes[0].legend(fontsize=5.8, loc="upper right")
    _save(fig, "fig_sequential")


def _window_means(a, win=60):
    """Per-window mean of each channel, windows lying inside one run.

    The same basis the model is scored on: `between_fish_metrics` computes its
    between-fish and within-fish spreads from per-fish window means, so the
    figure here and the model's score are the same decomposition.
    """
    a = a.copy()
    a["k"] = a.groupby("run_id").cumcount() // win
    full = a.groupby(["fish_id", "run_id", "k"]).filter(lambda g: len(g) == win)
    return (full.groupby(["fish_id", "run_id", "k"])
            .agg(dt=("dt_s", "mean"), dist=("dist_mm", "mean"),
                 adth=("abs_dtheta", "mean"), rad=("r_norm", "mean")).reset_index())


def fig_individuality(c, win=60):
    """Per-fish mean of each channel with that fish's own within-fish SD.

    The error bar is behavioural variability, NOT estimation uncertainty: with
    thousands of bouts per animal the mean is known almost exactly, so an
    estimation interval would make every fish trivially distinct and prove far
    less. Bars spanning less than the spread of the points is the claim -- fish
    differ from each other by more than they differ from themselves.
    """
    a = c[c.active].copy()
    a["dist_mm"] = a.distance_px / PX_PER_MM
    a["abs_dtheta"] = a.dtheta_deg.abs()
    fish = sorted(a.fish_id.unique())
    w = _window_means(a, win)

    PAN = [("dt", "interval (s)", "A"), ("dist", "displacement (mm)", "B"),
           ("adth", r"$|\Delta\theta|$ (deg)", "C"), ("rad", "radial $r/R$", "D")]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 4.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.70, wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, :])]

    for ax, (col, lab, letter) in zip(axes, PAN):
        m = w.groupby("fish_id")[col].mean().reindex(fish)
        s = w.groupby("fish_id")[col].std().reindex(fish)
        ax.errorbar(fish, m, yerr=s, fmt="o", color=BLUE, ms=3.2,
                    ecolor=MUTED, elinewidth=0.8, capsize=1.6)
        ax.axhline(float(m.mean()), color=VERM, lw=0.9, ls=(0, (3, 2)))
        ax.set_xticks(fish); ax.set_xticklabels(fish, fontsize=5.4)
        ax.set_xlim(-0.7, len(fish) - 0.3)
        ax.set_xlabel("fish"); ax.set_ylabel(lab)
        ax.grid(axis="y"); ax.set_axisbelow(True)
        _panel(ax, letter, "")
    _save(fig, "fig_individuality")


def stats(c):
    a = c[c.active]
    P = print
    P("=" * 64); P("NUMBERS QUOTED IN CHAPTER 3"); P("=" * 64)
    P(f"fish                     : {c.fish_id.nunique()}")
    P(f"frames per fish          : 313500  ({313500/FPS/60:.2f} min)")
    P(f"blocks                   : {c.blk.nunique()} "
      f"({PAUSE_FR/FPS:.0f} s pause + {STIM_FR/FPS:.0f} s stimulus, 95 pairs)")
    P(f"stimulus conditions      : {c.loc[c.dots_on, 'stimulus'].nunique()}")
    P(f"bouts detected           : {len(c):,}")
    P(f"active bouts             : {len(a):,}  ({100*len(a)/len(c):.2f}%)")
    P(f"runs                     : {c.run_id.nunique():,}  "
      f"(median {int(c.groupby('run_id').size().median())} bouts)")
    P(f"bouts with dots on       : {int(c.dots_on.sum()):,} "
      f"({100*c.dots_on.mean():.1f}%)")
    P(f"crosses_block            : {int(a.crosses_block.sum()):,} "
      f"({100*a.crosses_block.mean():.1f}% of active)")
    P("")
    P(f"dt_s      median {a.dt_s.median():.3f}  IQR "
      f"{a.dt_s.quantile(.25):.3f}-{a.dt_s.quantile(.75):.3f}  "
      f"95th {a.dt_s.quantile(.95):.2f}  -> {1/a.dt_s.median():.2f} Hz")
    P(f"dist  px  median {a.distance_px.median():.2f}  "
      f"mm {a.distance_px.median()/PX_PER_MM:.2f}  "
      f"95th {a.distance_px.quantile(.95)/PX_PER_MM:.2f} mm")
    P(f"|dtheta|  median {a.dtheta_deg.abs().median():.1f} deg  "
      f"95th {a.dtheta_deg.abs().quantile(.95):.1f} deg")
    P(f"bout type : " + ", ".join(f"{k}={v:.3f}" for k, v in
                                  a.bout_type.value_counts(normalize=True).items()))
    P(f"speed     median {(a.distance_px/PX_PER_MM/a.dt_s).median():.2f} mm/s")
    P("")
    q = (c.groupby("fish_id").apply(
        lambda g: g.loc[g.dt_s > QUIET_S, "dt_s"].sum() / g.dt_s.sum(),
        include_groups=False) * 100)
    P(f"quiescent share  pooled {100*c.loc[c.dt_s>QUIET_S,'dt_s'].sum()/c.dt_s.sum():.1f}%"
      f"  per-fish {q.min():.1f}-{q.max():.1f}% (fish {q.idxmin()}..{q.idxmax()})")
    P(f"bouts ending in a gap    : {100 * c.gap_after.mean():.1f}% (interval > {QUIET_S:.0f} s)")
    P(f"longest gap              : {c.dt_s.max():.1f} s")
    s = _dot_profile_frame(c, keep_outside=True)
    from twin.segment_bouts import load_physical_dish
    g = s.join(load_physical_dish(paths.ROI_PHYS).rename(columns={"radius": "R"}), on="fish_id")
    s = s.assign(out=(np.hypot(g.dot_cx - g.cx, g.dot_cy - g.cy) > g.R).to_numpy())
    per = s[s.moving.astype(bool)].groupby("N").out.mean()
    P(f"dot centroid outside dish: {100 * s.out.mean():.1f}% of stimulus bouts (moving: "
      + ", ".join(f"{n} dot{'s' if n > 1 else ''} {100 * v:.1f}%" for n, v in per.items()) + ")")
    P(f"bouts per fish           : {a.groupby('fish_id').size().min():,}-"
      f"{a.groupby('fish_id').size().max():,}")
    P("")
    for cond in ("moving", "static"):
        v = np.array([_turn_toward(a[(a.fish_id == f) & (a.cond == cond)])
                      for f in sorted(a.fish_id.unique())])
        P(f"P(turn toward) {cond:7s}: pooled {_turn_toward(a[a.cond==cond]):.3f}  "
          f"per-fish {np.nanmin(v):.3f}-{np.nanmax(v):.3f}  "
          f"above 0.5 in {int((v>0.5).sum())}/15")
    P(f"r_norm    median {a.r_norm.median():.3f}  "
      f"per-fish {a.groupby('fish_id').r_norm.median().min():.3f}-"
      f"{a.groupby('fish_id').r_norm.median().max():.3f}")
    # ---- Sec 3.3 -----------------------------------------------------------
    P("")
    P("-" * 64); P("SEC 3.3  BEHAVIOURAL PROPERTIES"); P("-" * 64)
    fish = sorted(a.fish_id.unique())
    for thr in (0.0, DTHETA0):
        for cond in ("moving", "static"):
            per = np.array([_t_ci(_toward_blocks(
                _stim_bouts(a[a.fish_id == f], cond), thr))[0] for f in fish])
            pooled = _t_ci(_toward_blocks(_stim_bouts(a, cond), thr), min_n=1)
            P(f"  P(turn toward) |dth|>{thr:>4.0f} {cond:7s}: "
              f"pooled {pooled[0]:.3f} [{pooled[1]:.3f}, {pooled[2]:.3f}]  "
              f"mean-of-fish {np.nanmean(per):.3f}  "
              f"per-fish {np.nanmin(per):.3f}-{np.nanmax(per):.3f}  "
              f"above 0.5 in {int((per > 0.5).sum())}/{np.isfinite(per).sum()}")
    P(f"  share of stimulus bouts with |dtheta| <= {DTHETA0:g} deg: "
      f"{(_stim_bouts(a).dtheta_deg.abs() <= DTHETA0).mean():.3f}")
    P("")
    a2 = a.copy()
    a2["N"] = (np.nan_to_num(a2.n_dots_left) + np.nan_to_num(a2.n_dots_right)).astype(int)
    for cond, Ns in (("moving", [1, 2, 4, 6, 8]), ("static", [1, 4, 8])):
        for N in Ns:
            s = a2[(a2.cond == cond) & (a2.N == N) & a2.dot_dist_px.notna()]
            e = s[s.t_blk < 8].dot_dist_px.mean() / PX_PER_MM
            l = s[s.t_blk > 45].dot_dist_px.mean() / PX_PER_MM
            P(f"  approach {cond:7s} N={N}: first 8 s {e:5.1f} mm -> last 15 s {l:5.1f} mm"
              f"   ({l - e:+5.1f})")
    P("")
    t = a[(a.dtheta_deg.abs() >= 10) & a.wall_psi_rad.notna()].copy()
    t["toward"] = np.sign(t.dtheta_deg) == np.sign(t.wall_psi_rad)
    for lab, m in (("centre  r/R<0.3", t.r_norm < 0.3),
                   ("mid     0.3-0.7", t.r_norm.between(0.3, 0.7)),
                   ("outer   r/R>0.8", t.r_norm > 0.8)):
        pf = t[m].groupby("fish_id").toward.mean()
        P(f"  P(turn toward centre) {lab}: {t[m].toward.mean():.3f}  "
          f"per-fish {pf.min():.3f}-{pf.max():.3f}  above 0.5 in {(pf > .5).sum()}/15")
    rr = a.groupby("fish_id").r_norm.max()
    P(f"  max r/R reached, per fish: {rr.min():.3f}-{rr.max():.3f} "
      f"(median {rr.median():.3f})  -> unoccupied outer rim")
    P("")

    def _lag1(g, col):
        v = g[col].to_numpy(float); r = g.run_id.to_numpy()
        ok = r[1:] == r[:-1]
        return float(np.corrcoef(v[:-1][ok], v[1:][ok])[0, 1])
    aa = a.copy(); aa["abs_dtheta"] = aa.dtheta_deg.abs()
    def _lagk(g, col, k):
        v = g[col].to_numpy(float); r = g.run_id.to_numpy()
        ok = r[k:] == r[:-k]
        return float(np.corrcoef(v[:-k][ok], v[k:][ok])[0, 1])
    for col, lab in (("distance_px", "displacement"), ("dt_s", "interval"),
                     ("abs_dtheta", "|dtheta|"), ("dtheta_deg", "signed dtheta")):
        med = []
        for k in (1, 2, 3, 4, 5):
            v = np.array([_lagk(aa[aa.fish_id == f], col, k) for f in fish])
            med.append(np.median(v))
            if k == 1:
                P(f"  lag-1 corr {lab:14s}: median {med[0]:+.3f}  "
                  f"range {v.min():+.3f}..{v.max():+.3f}  "
                  f"same sign in {max((v > 0).sum(), (v < 0).sum())}/15")
        P(f"    lags 1-5 {lab:13s}: "
          + "  ".join(f"{m:+.3f}" for m in med)
          + "   vs one-step decay "
          + "  ".join(f"{med[0] ** k:+.3f}" for k in (1, 2, 3, 4, 5)))
    P("")
    w = aa.copy()
    w["k"] = w.groupby("run_id").cumcount() // 60
    w["dist_mm"] = w.distance_px / PX_PER_MM
    full = w.groupby(["fish_id", "run_id", "k"]).filter(lambda g: len(g) == 60)
    ww = full.groupby(["fish_id", "run_id", "k"]).agg(
        sd_step=("dist_mm", "std"), sd_dt=("dt_s", "std")).reset_index()
    P(f"  60-bout windows: {len(ww)} across {ww.fish_id.nunique()} fish")
    for col, lab, unit in (("sd_step", "step SD", "mm"), ("sd_dt", "interval SD", "s")):
        m = ww.groupby("fish_id")[col].median()
        P(f"  within-window {lab:12s}: per-fish median {m.min():.2f}-{m.max():.2f} {unit}"
          f"  ({m.max() / m.min():.2f}x across fish)")
    P("")
    # the decomposition the individuality figure draws, on the same 60-bout windows
    aa2 = a.copy()
    aa2["dist_mm"] = aa2.distance_px / PX_PER_MM
    aa2["abs_dtheta"] = aa2.dtheta_deg.abs()
    W = _window_means(aa2, 60)
    P("individuality, per-fish window means (the Fig. 3.8 decomposition):")
    for col, lab, unit in (("dt", "interval", "s"), ("dist", "displacement", "mm"),
                           ("adth", "|dtheta|", "deg"), ("rad", "radial r/R", "")):
        m = W.groupby("fish_id")[col].mean()
        s = W.groupby("fish_id")[col].std()
        P(f"  {lab:12s}: per-fish mean {m.min():.3f}-{m.max():.3f} {unit}  "
          f"between-fish SD {m.std(ddof=1):.3f}  within-fish SD {s.mean():.3f}  "
          f"ratio B/W {m.std(ddof=1) / s.mean():.2f}")
    r = W.groupby("fish_id")[["dt", "dist", "adth", "rad"]].mean().rank()
    P("  per-fish RANK on each axis (1 = lowest), to show the axes are not one axis:")
    P("   " + r.astype(int).to_string().replace("\n", "\n   "))
    P("")

    rf = pd.read_csv(REPAIRED)
    P(f"repaired frames          : {len(rf)} across {rf.fish_id.nunique()} fish "
      f"(worst fish {rf.fish_id.value_counts().idxmax()}: "
      f"{rf.fish_id.value_counts().max()})")
    raw = pd.read_csv(RAW)
    vr = raw.distance_px / PX_PER_MM / raw.dt_s
    P(f"pre-repair bouts > {IMPOSSIBLE_MM_S:.0f} mm/s : {int((vr>IMPOSSIBLE_MM_S).sum())}  "
      f"(max {vr.max():.0f} mm/s)")
    va = a.distance_px / PX_PER_MM / a.dt_s
    P(f"post-clean max speed     : {va.max():.1f} mm/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="print numbers, no figures")
    ap.add_argument("--only", default=None, help="one figure name")
    args = ap.parse_args()
    c = load()
    if args.stats:
        stats(c)
        return
    figs = {"stimulus": fig_stimulus, "detection": fig_detection,
            "marginals": fig_marginals, "quiescence": fig_quiescence,
            "states": fig_states, "cleaning": fig_cleaning,
            "reference": fig_reference,
            # Sec 3.3
            "stim_bars": fig_stim_bars, "dot_profile": fig_dot_profile_pair,
            "spatial": fig_spatial, "sequential": fig_sequential,
            "individuality": fig_individuality}
    for k, fn in figs.items():
        if args.only and k != args.only:
            continue
        fn(c)


if __name__ == "__main__":
    main()
