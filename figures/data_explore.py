"""Real-data exploration beyond the Data chapter (notebooks/02_data_exploration.ipynb, sections 8bis-18): whether
10 s is the right cut for a pause, the whole session of one fish, where the fish is when still and when active, whether
behaviour changes over the session and whether its slow fluctuation is the fish or the schedule, and how far back a
bout remembers. Real fish only; it reads the segmenter's bout table (``data/bouts_GsizeMotionSingle.csv``), the 30 fps
tables (``data/frames``) and the dish geometry.

    df = load()
    plot_dt_spectrum(df); threshold_sweep(df)            # 8bis
    show(plot_session(df, 10))                           # 9
    show(plot_occupancy(df, "dead"))                      # 10
    trend_table(df); temporal_structure_table(df)        # 14 (and the rest of that section)
    lag_memory(df)                                       # 17
    window_acf_real(df); window_acf_vs_shuffles(df); fish_or_schedule(df)   # 18

The helpers are those of the exploration notebook the thesis was written from, unchanged.
"""
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as _spearman

from twin import paths

BOUTS = paths.BOUTS_RAW
FRAMES_DIR = paths.FRAMES
ROI_PHYSICAL = paths.ROI_PHYS
FISH = list(range(15))

# the exploration notebook's look (call style() before plotting)
RC = {"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
      "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "axes.titlecolor": "#0b0b0b",
      "axes.spines.top": False, "axes.spines.right": False, "text.color": "#0b0b0b", "xtick.color": "#898781",
      "ytick.color": "#898781", "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e",
      "axes.titlesize": 10, "axes.labelsize": 9, "font.size": 9, "legend.frameon": False, "figure.dpi": 110}


def style():
    plt.rcParams.update(RC)


def show(fig=None):
    """Display a figure in the notebook whatever matplotlib's backend (the figure scripts switch it to Agg, where
    plt.show() shows nothing), then close it."""
    import io

    from IPython.display import Image
    fig = fig if fig is not None else plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))


def _segment_bouts():
    """The segmenter that produced the bout table."""
    from twin import segment_bouts
    return segment_bouts


def load():
    """The segmenter's bout table with r/R, the condition, the bearing to the dots in the travel frame and the block
    clock (the rig's exact frame grid: 1500-frame pause + 1800-frame stimulus, x95, the same for every fish)."""
    df = pd.read_csv(BOUTS).sort_values(["fish_id", "segment_id", "bout_index"]).reset_index(drop=True)
    df["r_norm"] = 1.0 - df["dist_to_border_norm"]                      # 0 = centre, 1 = the physical wall
    stim_on = df["stimulus"].astype(str).str.startswith("CLsemi")
    df["cond"] = np.where(~stim_on, "none", np.where(df["moving"].astype(bool), "moving", "static"))
    # the bearing to the dots referenced to the INCOMING chord (theta_{t-1} = heading_t - dtheta_t), the frame of
    # dtheta and of the models, not the tracker's body axis
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi                  # noqa: E731
    theta_prev = wrap(df["heading_rad"] - np.radians(df["dtheta_deg"]))
    df["stim_angle_travel"] = wrap(np.arctan2(df["dot_cy"] - df["y_px"], df["dot_cx"] - df["x_px"]) - theta_prev)
    pos = df["frame_onset"] % CYCLE_FR
    in_stim = pos >= PAUSE_FR
    df["blk"] = df["frame_onset"] // CYCLE_FR * 2 + in_stim.astype(int)
    df["t_blk"] = np.where(in_stim, pos - PAUSE_FR, pos) / FPS
    return df


INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281"]


CONDS = {"none": "no stimulus", "moving": "moving dots", "static": "static dots"}


COND_COLOR = {"none": BLUE, "moving": ORANGE, "static": AQUA}


CYCLE_FR, PAUSE_FR, FPS = 3300, 1500, 30.0


def _bottom_xlabels(axes, label):
    """Put the x label on the lowest VISIBLE axes of each column (the last row can
    be part-empty when the fish count is not a multiple of ncols)."""
    A = np.atleast_2d(axes)
    for j in range(A.shape[1]):
        col = [A[i, j] for i in range(A.shape[0]) if A[i, j].get_visible()]
        if col:
            col[-1].set_xlabel(label)


DTHETA0 = 10.0                       # forward/turn threshold, deg


def stim(d, cond=None):
    """Bouts with a dot on screen and a defined bearing; optionally one condition."""
    s = d[(d["cond"] != "none") & np.isfinite(d["stim_angle_travel"])]
    return s if cond is None else s[s["cond"] == cond]


def _turns(g):
    """Turn bouts only. A forward bout has no direction to be 'toward' anything."""
    return g[g["dtheta_deg"].abs() > DTHETA0]


def _toward(g):
    return (np.sign(g["dtheta_deg"]) == np.sign(g["stim_angle_travel"])).to_numpy(float)


def follow_index(g, min_turns=25):
    """P(turn toward the dots) − 0.5 over turn bouts; NaN below `min_turns`.

    The scalar behind §6B's bars, pulled out so §15 can track it over session time.
    """
    t = _turns(g)
    return float(_toward(t).mean() - .5) if len(t) >= min_turns else np.nan


SPEED_K = 1.0                        # segment_bouts.py --speed-thresh-k default


def as_blk(b):
    """Accept either form of block name and return the internal grid id.

    THE TWO ARE NOT INTERCHANGEABLE and getting it wrong fails silently: the tables
    print a LABEL like ``"stim 9"`` (stimulus block 9 of 95), while the grid id counts
    pause and stimulus blocks together, so ``"stim 9"`` is grid id 19 -- and grid id 9
    is a different block entirely (``"stim 4"``). Every public function here takes
    either, so paste the label straight out of `worst_gaps` / `quiet_gaps` and it is
    right by construction.
    """
    if isinstance(b, str):
        kind, _, num = b.strip().partition(" ")
        kind = kind.lower()
        if kind not in ("stim", "pause") or not num.strip().isdigit():
            raise ValueError(f"block {b!r}: expected 'stim N' / 'pause N', or a grid int")
        return int(num) * 2 + (kind == "stim")
    return int(b)


def block_frame_range(blk):
    """(first, last+1) frame of a block, inverting the `blk` formula above."""
    cyc, in_stim = divmod(as_blk(blk), 2)
    f0 = cyc * CYCLE_FR + (PAUSE_FR if in_stim else 0)
    return f0, f0 + (CYCLE_FR - PAUSE_FR if in_stim else PAUSE_FR)


def _speed(x, y):
    """segment_bouts' instantaneous speed (px/s), lightly smoothed."""
    v = np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y))]) * FPS
    return pd.Series(v).rolling(3, center=True, min_periods=1).mean().to_numpy()


_SESSION_CACHE = {}


def session_trace(fish_id, k=SPEED_K):
    """Whole-session smoothed speed and detected bouts for one fish.

    The detection that every §6bis/§7/§10 figure sits on, run once over the ENTIRE
    continuous recording -- which is the only correct way to do it, since the threshold
    is mean(v) + k*SD over all 174 minutes (see block_traces). Cached per fish, because
    the per-frame CSV is ~115 MB and the block figures ask for it repeatedly.
    """
    if (fish_id, k) in _SESSION_CACHE:
        return _SESSION_CACHE[(fish_id, k)]
    from scipy.signal import find_peaks
    SB = _segment_bouts()

    prefix = f"f{fish_id}"
    path = next(FRAMES_DIR.glob(f"*_{prefix}_transformed_abs.csv"))
    fr = pd.read_csv(path, usecols=["frame", "stimulus", f"{prefix}_x", f"{prefix}_y"])
    g = fr[fr[f"{prefix}_x"].notna()].reset_index(drop=True)     # the one continuous segment

    x, y = SB.fish_xy_px(g, prefix)                              # smoothed: DETECTION only
    v = _speed(x, y)
    thr = float(np.nanmean(v) + k * np.nanstd(v))
    peaks, _ = find_peaks(v, height=thr, distance=SB.REFRACTORY_FR)
    onsets = np.array([SB._snap_to_trough(v, pk) for pk in peaks], dtype=int)
    keep = np.zeros(len(peaks), bool)
    for j, o in enumerate(onsets):                               # segment_bouts' displacement gate
        a, b = max(o - SB.DISP_WIN_FR, 0), min(o + SB.DISP_WIN_FR, len(x) - 1)
        d_ = np.hypot(x[b] - x[a], y[b] - y[a])
        keep[j] = SB.DISP_MIN_PX <= d_ <= SB.DISP_MAX_PX

    frames = g["frame"].to_numpy()
    out = dict(frames=frames, v=v, thr=thr, stims=g["stimulus"].to_numpy(),
               on_keep=frames[onsets[keep]], on_rej=frames[onsets[~keep]])
    _SESSION_CACHE[(fish_id, k)] = out
    return out


N_CYCLES = 95                                   # pause+stimulus cycles in the session


REC_S = N_CYCLES * CYCLE_FR / FPS               # 10450 s = 174.2 min per fish


QUIET_S = 10.0                                  # a "no-bout stretch" starts here


def block_inventory(d):
    """All 190 grid blocks (95 pause + 95 stimulus) with their condition.

    Taken by POOLING over fish, which is valid because the stimulus schedule is one
    programme shared by all 15. That matters: a fish can have zero bouts in a block it
    slept through, so a per-fish groupby would silently drop exactly the blocks this
    audit is about. Checked: all 190 are recovered (95 none / 60 moving / 35 static).
    """
    return d.groupby("blk").agg(stimulus=("stimulus", "first"), cond=("cond", "first"))


def plot_session(d, fish_id, row_min=15, thr=QUIET_S, k=SPEED_K, title=None):
    """The whole 174-minute session for one fish, wrapped over rows.

    Raw 30 fps speed — no downsampling, it renders in under a second — so a dead period
    is a flat line and an active one is a band of bout peaks, which is the contrast the
    figure exists to show. Two channels layered without fighting each other:

      * background tint = what was on screen (moving dots / static dots / pause). At this
        zoom there are ~8 blocks per row, so the periods are tinted but NOT labelled —
        the legend carries it instead.
      * black bar along the bottom = a stretch of more than `thr` s with no bout, i.e.
        exactly the gaps §8 counts.
    """
    S = session_trace(fish_id, k)
    inv = block_inventory(d)
    t = S["frames"] / FPS
    v = S["v"]
    gaps = d[(d["fish_id"] == fish_id) & (d["dt_s"] > thr)]
    g0 = gaps["t_onset_s"].to_numpy(float)
    g1 = g0 + gaps["dt_s"].to_numpy(float)

    row_s = row_min * 60.0
    nrows = int(np.ceil(t.max() / row_s))
    ymax = float(np.nanpercentile(v, 99.9))
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 1.25 * nrows + 1.2))
    axes = np.atleast_1d(axes)
    for r, ax in enumerate(axes):
        a, z = r * row_s, (r + 1) * row_s
        for b in inv.index:                                   # stimulus background
            f0, f1 = block_frame_range(b)
            s0, s1 = f0 / FPS, f1 / FPS
            if s1 > a and s0 < z:
                ax.axvspan((max(s0, a) - a) / 60, (min(s1, z) - a) / 60,
                           color=COND_COLOR[inv["cond"][b]], alpha=.13, lw=0, zorder=0)
        for s0, s1 in zip(g0, g1):                            # dead-period bar, bottom 6 %
            if s1 > a and s0 < z:
                ax.axvspan((max(s0, a) - a) / 60, (min(s1, z) - a) / 60,
                           ymin=0, ymax=.06, color=INK, lw=0, zorder=4)
        m = (t >= a) & (t < z)
        ax.plot((t[m] - a) / 60, v[m], lw=.3, color=INK2, zorder=3)
        ax.set_xlim(0, row_min)
        ax.set_ylim(-.06 * ymax / .94, ymax)                  # keep the bar clear of the trace
        ax.set_yticks([])
        ax.set_ylabel(f"{a / 60:.0f}–{min(z, t.max()) / 60:.0f}\nmin", fontsize=7.5,
                      color=INK2, rotation=0, ha="right", va="center", labelpad=14)
        ax.tick_params(labelsize=8)
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel(f"time within the row (min)  ·  each row is {row_min:g} min")
    quiet = float(gaps["dt_s"].sum()) / REC_S * 100
    fig.legend(handles=[Patch(facecolor=COND_COLOR[c], alpha=.4, label=l)
                        for c, l in CONDS.items()]
               + [Patch(facecolor=INK, label=f"no bout for > {thr:g} s"),
                  Line2D([], [], color=INK2, lw=.8, label="speed (px/s)")],
               loc="lower center", ncol=5, bbox_to_anchor=(.5, -.012), fontsize=8,
               labelcolor=INK2)
    fig.suptitle(title or f"fish {fish_id} — the whole session, {row_min:g} min per row  ·  "
                 f"{quiet:.0f} % of it with no bout for over {thr:g} s",
                 y=1.0, fontsize=11, color=INK)
    fig.tight_layout()
    return fig


_XY_CACHE = {}


def fish_xy(fish_id):
    """RAW per-frame position for one fish (tracked frames only) + the frame numbers.

    Raw, not `SB.fish_xy_px`: the Gaussian smoothing there exists to stop integer-pixel
    quantisation spawning fake speed peaks during DETECTION, and has no business moving a
    position that is being histogrammed. float32 keeps all 15 fish cacheable (~56 MB).
    """
    if fish_id in _XY_CACHE:
        return _XY_CACHE[fish_id]
    prefix = f"f{fish_id}"
    path = next(FRAMES_DIR.glob(f"*_{prefix}_transformed_abs.csv"))
    fr = pd.read_csv(path, usecols=["frame", f"{prefix}_x", f"{prefix}_y"]).dropna()
    out = (fr["frame"].to_numpy(np.int32),
           fr[f"{prefix}_x"].to_numpy(np.float32), fr[f"{prefix}_y"].to_numpy(np.float32))
    _XY_CACHE[fish_id] = out
    return out


def physical_dish():
    """Per-fish (cx, cy, radius) of the TRUE 100 mm dish, from the raw ROIdef."""
    return _segment_bouts().load_physical_dish(ROI_PHYSICAL)


def dead_active_masks(d, fish_id, frames, thr=QUIET_S):
    """Split a fish's tracked frames into (dead, active, unclassified).

    DEAD  = inside a stretch of more than `thr` s with no bout -- the same gaps §8 counts
            (`dt_s > thr`), taken strictly BETWEEN the two bouts that bracket them, since
            the endpoints themselves are bouts.
    ACTIVE= every other frame between the fish's first bout and the end of the last bout's
            interval.
    Frames outside that span are neither: the recording starts before the first bout and
    can end after the last, and those are not evidence either way.
    """
    sub = d[d["fish_id"] == fish_id]
    f0 = sub["frame_onset"].to_numpy(float)
    end = f0 + sub["dt_s"].to_numpy(float) * FPS
    dead = np.zeros(len(frames), bool)
    q = sub["dt_s"].to_numpy(float) > thr
    i0 = np.searchsorted(frames, f0[q], "right")
    i1 = np.searchsorted(frames, end[q], "left")
    for a, b in zip(i0, i1):
        dead[a:b] = True
    covered = (frames >= f0.min()) & (frames <= end.max())
    return dead & covered, covered & ~dead, ~covered


TREND_METRICS = {
    "quiescent %": ("all", lambda x, w: 100 * x.loc[x["dt_s"] > QUIET_S, "dt_s"].sum() / (w * 60)),
    "bouts / min": ("all", lambda x, w: len(x) / w),
    "median Δt (s)": ("active", lambda x, w: x["dt_s"].median()),
    "median dist (px)": ("active", lambda x, w: x["distance_px"].median()),
    "median |Δθ| (°)": ("active", lambda x, w: x["dtheta_deg"].abs().median()),
    "mean r/R": ("active", lambda x, w: (1 - x["dist_to_border_norm"]).mean()),
    "follow index": ("moving", lambda x, w: follow_index(x)),
}


def trend_series(d, bin_min=15.0, kinematics="active"):
    """Each metric as a per-fish time series over session-time bins.

    Returns {metric: DataFrame(bin x fish)}.

    `kinematics` decides which rows the per-bout metrics (dt, distance, |dtheta|, r/R,
    follow index) are computed on:
      'active' -- only rows whose two legs both stay under QUIET_S. `distance` and `dtheta`
                  on a gap-spanning row describe the GAP, so without this a drift in
                  quiescence leaks into every channel as a fake trend.
      'all'    -- let those rows back in.
    Running both and comparing is the test of whether a time course is really a quiescence
    effect wearing another channel's clothes. `quiescent %` and `bouts / min` are defined on
    all rows either way.
    """
    d = d.copy()
    d["dt_prev"] = d.groupby("fish_id")["dt_s"].shift(1)
    d["_tb"] = (d["t_onset_s"] / 60.0 // bin_min).astype(int)
    kin = (d[(d["dt_s"] <= QUIET_S) & (d["dt_prev"] <= QUIET_S)]
           if kinematics == "active" else d)
    src = {"all": d, "active": kin, "moving": stim(kin, "moving")}
    # the session is 174.2 min, so the final bin is PARTIAL -- a rate metric divided by
    # bin_min would dive at the end and read as a trend. Keep only whole bins.
    n_full = int(REC_S / 60 // bin_min)
    out = {}
    for name, (which, fn) in TREND_METRICS.items():
        s = src[which]
        t = pd.DataFrame({
            fid: gg.groupby("_tb").apply(lambda x: fn(x, bin_min), include_groups=False)
            for fid, gg in s.groupby("fish_id")})
        out[name] = t[t.index < n_full]
    return out


def trend_table(d, bin_min=15.0, kinematics="active"):
    """Per-metric: is there a trend, and do the fish agree on its direction?

    The across-fish SIGN TEST is the test that matters. A single fish drifting proves
    nothing -- with 15 fish, a real session effect should push most of them the same way.
    `rho` is each fish's own Spearman of the metric against time, so between-fish level
    differences (individuality) cannot produce it.
    """
    rows = []
    for name, S in trend_series(d, bin_min, kinematics).items():
        rhos = []
        for fid in S.columns:
            s = S[fid].dropna()
            if len(s) > 5:
                rhos.append(_spearman.spearmanr(s.index.to_numpy(), s.to_numpy()).statistic)
        rhos = np.asarray(rhos, float)
        rhos = rhos[np.isfinite(rhos)]
        pos = int((rhos > 0).sum())
        p = _spearman.binomtest(max(pos, len(rhos) - pos), len(rhos), .5).pvalue
        rows.append(dict(metric=name, mean_rho=round(float(rhos.mean()), 3),
                         same_sign=f"{max(pos, len(rhos)-pos)}/{len(rhos)}",
                         first_bin=round(float(S.iloc[0].mean()), 2),
                         last_bin=round(float(S.iloc[-1].mean()), 2),
                         sign_test_p=round(float(p), 4)))
    return pd.DataFrame(rows).set_index("metric")


def settling_table(d, early_min=30.0, bin_min=15.0, kinematics="active"):
    """Early vs the rest of the session, PAIRED within fish (Wilcoxon over the 15).

    A Spearman over the whole session tests for a monotone drift, which is the wrong
    instrument if the change is a settling-in effect confined to the opening. This
    contrasts the first `early_min` against everything after, fish by fish, so the pairing
    again removes individuality.
    """
    S = trend_series(d, bin_min, kinematics)
    k = int(early_min // bin_min)
    rows = []
    for name, T in S.items():
        e, l = T.iloc[:k].mean(axis=0), T.iloc[k:].mean(axis=0)
        diff = (e - l).dropna()
        if len(diff) < 10:
            continue
        rows.append(dict(metric=name, early=round(float(e.mean()), 2),
                         rest=round(float(l.mean()), 2), diff=round(float(diff.mean()), 2),
                         same_sign=f"{int(max((diff > 0).sum(), (diff < 0).sum()))}/{len(diff)}",
                         wilcoxon_p=round(float(_spearman.wilcoxon(diff).pvalue), 4)))
    return pd.DataFrame(rows).set_index("metric")


OVER_TIME_METRICS = {
    "median |Δθ| (°)": lambda g: g["dtheta_deg"].abs().median(),
    "median Δt (s)": lambda g: g["dt_s"].median(),
    "median dist (px)": lambda g: g["distance_px"].median(),
    "mean r/R": lambda g: (1 - g["dist_to_border_norm"]).mean(),
    "follow index": follow_index,
}


STIM_ONLY_METRICS = {"follow index"}


def plot_metric_over_time(d, metric="median |Δθ| (°)", bin_min=15.0, min_bouts=20,
                          kinematics="active", ncols=5, title=None):
    """ONE metric over session time, one panel per fish, one line per stimulus condition.

    Raw units on a shared axis -- with a panel per fish there is no need to normalise, and
    the level is worth seeing next to the change. The dashed line is that fish's own
    all-condition median, so a line sitting above or below it reads immediately.

    `kinematics='active'` drops rows whose two legs do not both stay under QUIET_S: on a
    gap-spanning row `distance` and `dtheta` describe the GAP, and quiescence falls over the
    session, so leaving them in manufactures a downward trend (see §15's contamination
    table). Bins with fewer than `min_bouts` bouts in a condition are left empty rather than
    drawn from noise -- splitting three ways thins each bin considerably.
    """
    fn = OVER_TIME_METRICS[metric]
    d = d.copy()
    d["dt_prev"] = d.groupby("fish_id")["dt_s"].shift(1)
    if kinematics == "active":
        d = d[(d["dt_s"] <= QUIET_S) & (d["dt_prev"] <= QUIET_S)]
    n_full = int(REC_S / 60 // bin_min)
    d = d[(d["t_onset_s"] / 60.0 // bin_min) < n_full]
    d["_tb"] = (d["t_onset_s"] / 60.0 // bin_min).astype(int)

    nrows = ceil(len(FISH) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.3 * nrows),
                             sharex=True, sharey=True)
    stim_only = metric in STIM_ONLY_METRICS
    conds = {k: v for k, v in CONDS.items() if not (stim_only and k == "none")}
    for ax, f in zip(axes.ravel(), FISH):
        sub = d[d["fish_id"] == f]
        ref = stim(sub) if stim_only else sub
        ax.axhline(fn(ref), color=MUTED, lw=.9, ls=(0, (4, 3)), zorder=1)
        if stim_only:
            ax.axhline(0, color=INK2, lw=1, zorder=2)          # chance
        for c, lab in conds.items():
            g = sub[sub["cond"] == c]
            if not len(g):
                continue
            s_ = g.groupby("_tb").apply(
                lambda x: fn(x) if len(x) >= min_bouts else np.nan, include_groups=False)
            ax.plot(s_.index.to_numpy() * bin_min + bin_min / 2, s_.to_numpy(),
                    lw=1.5, marker="o", ms=3, mec=SURF, mew=.6,
                    color=COND_COLOR[c], zorder=3)
        ax.text(.03, .95, f"fish {f}", transform=ax.transAxes, va="top", fontsize=8.5,
                color=INK, fontweight="bold")
        ax.set_xlim(0, REC_S / 60)
    for ax in axes.ravel()[len(FISH):]:
        ax.set_visible(False)
    _bottom_xlabels(axes, "time through the session (min)")
    for i in range(np.atleast_2d(axes).shape[0]):
        np.atleast_2d(axes)[i, 0].set_ylabel(metric, fontsize=8)
    fig.legend(handles=[Line2D([], [], color=COND_COLOR[c], lw=1.5, marker="o", ms=4,
                               label=l) for c, l in conds.items()]
               + [Line2D([], [], color=MUTED, lw=.9, ls=(0, (4, 3)),
                         label="that fish's overall level")]
               + ([Line2D([], [], color=INK2, lw=1, label="chance")] if stim_only else []),
               loc="lower center", ncol=5, bbox_to_anchor=(.5, -.015), fontsize=8,
               labelcolor=INK2)
    fig.suptitle(title or f"{metric} over the session · {bin_min:g} min bins · "
                 f"{kinematics.upper()} rows · split by stimulus",
                 y=1.0, fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def temporal_structure_table(d, bin_min=5.0, kinematics="active", seed=0):
    """Is the within-fish wobble a slow STATE, or just estimation noise?

    `time_variation_table` shows a fish varies over the session, but not whether that
    variation means anything. Two orthogonal tests, both per fish then averaged:

    * **acf1/2/3** -- autocorrelation of the fish's own binned series at 1, 2, 3 bins.
      Sampling noise is white, so acf ≈ 0; a slow state gives acf1 > 0 decaying with lag,
      and where it reaches 0 is the timescale. This is the per-fish analogue of
      `generative_metrics`' `rate_acf_1min/3min_gap`.
    * **split-half r** -- split each bin's bouts at random into two halves, compute the
      metric on each, correlate across all (fish, bin) cells. Assumption-free: it measures
      how much of a bin's value is reproducible at all. Spearman-Brown then gives the
      reliability of the FULL bin, i.e. the share of the observed variance that is signal.

    A metric can have a large within-fish SD and still be pure noise -- that is exactly
    what happens to `follow index`, whose SD is dominated by the binomial error on a few
    dozen turn bouts. Read this before believing any within/between ratio.
    """
    S = trend_series(d, bin_min, kinematics)
    d = d.copy()
    d["dt_prev"] = d.groupby("fish_id")["dt_s"].shift(1)
    act = (d[(d["dt_s"] <= QUIET_S) & (d["dt_prev"] <= QUIET_S)]
           if kinematics == "active" else d).copy()
    act["_tb"] = (act["t_onset_s"] / 60.0 // bin_min).astype(int)
    act["_h"] = np.random.default_rng(seed).integers(0, 2, len(act))

    half_fn = {"median |Δθ| (°)": lambda x: x["dtheta_deg"].abs().median(),
               "median Δt (s)": lambda x: x["dt_s"].median(),
               "median dist (px)": lambda x: x["distance_px"].median(),
               "mean r/R": lambda x: (1 - x["dist_to_border_norm"]).mean(),
               "follow index": lambda x: follow_index(stim(x), min_turns=25)}
    rows = []
    for name, T in S.items():
        r = np.nan
        if name in half_fn:
            fn, A, B = half_fn[name], [], []
            for _, gg in act.groupby(["fish_id", "_tb"]):
                a, b = fn(gg[gg["_h"] == 0]), fn(gg[gg["_h"] == 1])
                if np.isfinite(a) and np.isfinite(b):
                    A.append(a), B.append(b)
            if len(A) > 30:
                r = float(_spearman.pearsonr(A, B).statistic)
        # say which rows each metric actually used: the `kinematics` flag is PER-METRIC.
        # quiescent % and bouts / min are always computed on ALL rows -- they are the
        # metrics that measure the dead periods, so filtering them out would make the
        # first identically zero and turn the second into a rate with a shrunken numerator
        # over an unchanged denominator.
        which = TREND_METRICS[name][0]
        rows.append(dict(metric=name,
                         rows={"all": "all", "active": kinematics,
                               "moving": f"{kinematics} + moving"}[which],
                         acf1=round(float(np.nanmean([_acf_of(T[f], 1) for f in T.columns])), 3),
                         acf2=round(float(np.nanmean([_acf_of(T[f], 2) for f in T.columns])), 3),
                         acf3=round(float(np.nanmean([_acf_of(T[f], 3) for f in T.columns])), 3),
                         split_half_r=round(r, 3) if np.isfinite(r) else np.nan,
                         signal_share=round(2 * r / (1 + r), 3) if np.isfinite(r) else np.nan))
    return pd.DataFrame(rows).set_index("metric")


def _cond_binned(d, bin_min, min_bouts, seed=0):
    """Active rows, whole bins only, tagged with a time bin and a random half."""
    d = d.copy()
    d["dt_prev"] = d.groupby("fish_id")["dt_s"].shift(1)
    a = d[(d["dt_s"] <= QUIET_S) & (d["dt_prev"] <= QUIET_S)].copy()
    n_full = int(REC_S / 60 // bin_min)
    a = a[(a["t_onset_s"] / 60.0 // bin_min) < n_full]
    a["_tb"] = (a["t_onset_s"] / 60.0 // bin_min).astype(int)
    a["_h"] = np.random.default_rng(seed).integers(0, 2, len(a))
    return a


def _acf_of(s, k=1):
    """Autocorrelation at lag k of a bin-indexed series, pairing by the REAL bin index.

    ⚠ The obvious implementation -- drop the NaNs, then correlate x[:-k] against x[k:] --
    is wrong here. Dropping a bin makes its two neighbours adjacent in the compacted array,
    so a lag-2 pair is counted as lag 1, and the acf is pulled toward the longer-lag (i.e.
    smaller) value. It bites whenever bins are missing, which is exactly what happens when
    the data are split by stimulus condition: `static` keeps only ~18 of 34 bins against
    ~27 for the other two, so the compacted version under-reads static specifically.
    Pairing on the index costs nothing and removes the artefact.
    """
    s = pd.Series(s).dropna()
    if len(s) < k + 5:
        return np.nan
    v = s - s.mean()
    idx = set(s.index)
    pairs = [(v[i], v[i + k]) for i in s.index if (i + k) in idx]
    if len(pairs) < 6:
        return np.nan
    a, b = zip(*pairs)
    return float(np.corrcoef(a, b)[0, 1])


def temporal_structure_by_cond(d, bin_min=5.0, min_bouts=20, min_half=15, seed=0):
    """`temporal_structure_table` split by stimulus condition.

    ⚠ The bin size is FORCED to ~5 min and cannot be raised to buy back sample size: at
    15 min the autocorrelation is gone even pooled (0.11-0.13 against 0.16-0.28 at 5 min),
    because 15 min is already past the correlation time. So splitting three ways drops each
    bin from ~137 active bouts to 31-64, and `split_half_r` is the guard on whether the
    resulting acf means anything -- read it before the acf on every row.

    `quiescent %` and `bouts / min` are absent by design: both are rates over wall-clock
    time, and per-condition exposure varies bin to bin, so a per-condition version would
    measure the schedule as much as the fish.
    """
    a = _cond_binned(d, bin_min, min_bouts, seed)
    rows = []
    for name, fn in OVER_TIME_METRICS.items():
        for c, lab in CONDS.items():
            if name in STIM_ONLY_METRICS and c == "none":
                continue
            s = a[a["cond"] == c]
            T = pd.DataFrame({f: gg.groupby("_tb").apply(
                lambda x: fn(x) if len(x) >= min_bouts else np.nan, include_groups=False)
                for f, gg in s.groupby("fish_id")})
            A, B = [], []
            for _, gg in s.groupby(["fish_id", "_tb"]):
                h0, h1 = gg[gg["_h"] == 0], gg[gg["_h"] == 1]
                if len(h0) >= min_half and len(h1) >= min_half:
                    x, y = fn(h0), fn(h1)
                    if np.isfinite(x) and np.isfinite(y):
                        A.append(x), B.append(y)
            r = float(_spearman.pearsonr(A, B).statistic) if len(A) > 30 else np.nan
            rows.append(dict(metric=name, condition=lab,
                             bouts_per_bin=int(s.groupby(["fish_id", "_tb"]).size().median()),
                             acf1=round(float(np.nanmean([_acf_of(T[f]) for f in T.columns])), 3),
                             acf2=round(float(np.nanmean([_acf_of(T[f], 2) for f in T.columns])), 3),
                             split_half_r=round(r, 3) if np.isfinite(r) else np.nan))
    return pd.DataFrame(rows).set_index(["metric", "condition"])


def acf_condition_contrast(d, bin_min=5.0, a_cond="none", b_cond="moving",
                           min_bouts=20, seed=0):
    """Is the slow state weaker under one stimulus than another? PAIRED within fish.

    Each fish contributes its own acf1 in both conditions, so the contrast is immune to
    between-fish differences in how autocorrelated a fish is. `disatt_*` divides by that
    condition's split-half reliability, because a noisier estimate attenuates the acf
    toward 0 and could fake a difference on its own.

    ⚠ Several metrics are tested here with no multiplicity correction. Treat a p just
    under .05 as suggestive; the honest confirmatory test is a single pre-chosen metric
    (`mean r/R` is the best-measured one).
    """
    a = _cond_binned(d, bin_min, min_bouts, seed)
    rel = temporal_structure_by_cond(d, bin_min, min_bouts, seed=seed)["split_half_r"]
    rows = []
    for name, fn in OVER_TIME_METRICS.items():
        if name in STIM_ONLY_METRICS:
            continue
        per = {}
        for c in (a_cond, b_cond):
            s = a[a["cond"] == c]
            T = pd.DataFrame({f: gg.groupby("_tb").apply(
                lambda x: fn(x) if len(x) >= min_bouts else np.nan, include_groups=False)
                for f, gg in s.groupby("fish_id")})
            per[c] = {f: _acf_of(T[f]) for f in T.columns}
        t = pd.DataFrame(per).dropna()
        if len(t) < 8:
            continue
        diff = t[a_cond] - t[b_cond]
        ra = rel.get((name, CONDS[a_cond]), np.nan)
        rb = rel.get((name, CONDS[b_cond]), np.nan)
        rows.append(dict(
            metric=name, **{CONDS[a_cond]: round(float(t[a_cond].mean()), 3),
                            CONDS[b_cond]: round(float(t[b_cond].mean()), 3)},
            diff=round(float(diff.mean()), 3),
            same_sign=f"{int(max((diff > 0).sum(), (diff < 0).sum()))}/{len(diff)}",
            wilcoxon_p=round(float(_spearman.wilcoxon(diff).pvalue), 4),
            disatt_diff=round(float(t[a_cond].mean() / ra - t[b_cond].mean() / rb), 3)
            if np.isfinite(ra) and np.isfinite(rb) else np.nan))
    return pd.DataFrame(rows).set_index("metric")


_METRIC_COL = {"mean r/R": ("r_norm", np.mean),
               "median Δt (s)": ("dt_s", np.median),
               "median |Δθ| (°)": ("_ad", np.median),
               "median dist (px)": ("distance_px", np.median)}


def acf_stimulus_identity_control(d, bin_min=5.0, a_cond="none", b_cond="moving",
                                  min_bouts=20, seed=0):
    """Is the weaker acf under a stimulus real, or just the randomised protocol?

    THE ALTERNATIVE THIS RULES OUT. Stimulus parameters (group size, side, speed) are
    randomised block to block, and a fish's kinematics partly track *which* stimulus is on.
    That injects variation into the stimulus-condition series that is uncorrelated between
    bins, which dilutes the acf without any suppression of the fish's internal state. The
    split-half reliability cannot see this -- it splits bouts WITHIN a bin, so both halves
    carry the same stimulus mix.

    The control: residualise every bout on its own stimulus identity (fish x n_dots x side)
    before binning, which removes all between-bin variation attributable to the protocol.
    If the acf stays low, the suppression is the fish, not the schedule.
    """
    a = _cond_binned(d, bin_min, min_bouts, seed)
    a["_ad"] = a["dtheta_deg"].abs()
    a["_n_dots"] = a[["n_dots_left", "n_dots_right"]].max(axis=1)
    rows = []
    for name, (col, agg) in _METRIC_COL.items():
        per, raw_acf = {}, np.nan
        for cond in (a_cond, b_cond):
            s = a[a["cond"] == cond].copy()
            by = (["fish_id", "_n_dots", "side"] if cond == b_cond else ["fish_id"])
            s["_v"] = s[col] - s.groupby(by)[col].transform("mean")
            T = pd.DataFrame({f: gg.groupby("_tb")["_v"].apply(
                lambda x: agg(x) if len(x) >= min_bouts else np.nan)
                for f, gg in s.groupby("fish_id")})
            per[cond] = {f: _acf_of(T[f]) for f in T.columns}
            if cond == b_cond:
                R = pd.DataFrame({f: gg.groupby("_tb").apply(
                    lambda x: agg(x[col]) if len(x) >= min_bouts else np.nan,
                    include_groups=False) for f, gg in s.groupby("fish_id")})
                raw_acf = float(np.nanmean([_acf_of(R[f]) for f in R.columns]))
        t = pd.DataFrame(per).dropna()
        diff = t[a_cond] - t[b_cond]
        rows.append(dict(metric=name,
                         **{CONDS[a_cond]: round(float(t[a_cond].mean()), 3)},
                         **{f"{CONDS[b_cond]} raw": round(raw_acf, 3),
                            f"{CONDS[b_cond]} residualised": round(float(t[b_cond].mean()), 3)},
                         same_sign=f"{int(max((diff > 0).sum(), (diff < 0).sum()))}/{len(diff)}",
                         wilcoxon_p=round(float(_spearman.wilcoxon(diff).pvalue), 4)))
    return pd.DataFrame(rows).set_index("metric")


def between_fish_sd_by_cond(d, min_bouts=20, seed=0):
    """Does the stimulus compress INDIVIDUALITY, or only the within-fish drift?

    Two different quantities that are easy to conflate. The acf asks whether ONE fish's
    deviation persists over 5 min; this asks how far the fish are FROM EACH OTHER. A
    stimulus could suppress the first without touching the second -- and that is what
    happens here.
    """
    a = _cond_binned(d, 5.0, min_bouts, seed)
    a["_ad"] = a["dtheta_deg"].abs()
    rows = []
    for name, (col, agg) in _METRIC_COL.items():
        r = {CONDS[c]: float(a[a["cond"] == c].groupby("fish_id")[col].apply(agg).std(ddof=1))
             for c in CONDS}
        rows.append(dict(metric=name, **{k: round(v, 3) for k, v in r.items()},
                         ratio_moving_none=round(r[CONDS["moving"]] / r[CONDS["none"]], 2)))
    return pd.DataFrame(rows).set_index("metric")


def temporal_structure_by_fish(d, bin_min=5.0, min_bouts=20, min_half=15,
                               min_bins=6, min_pairs=8, seed=0):
    """`temporal_structure_by_cond`, one row per FISH instead of averaged over them.

    The pooled table cannot separate a condition effect that every fish shows from one
    that three fish carry. This keeps the 15 apart, at the cost of estimating each acf
    from a single fish's bins.

    WARNING -- POWER. Read this before any individual number. Each acf1 comes from that
    fish's ~34 five-minute bins minus the ones that condition did not fill with
    `min_bouts` bouts, so `n_bins` lands around 10-30 and the standard error on one acf1
    is roughly 1/sqrt(n_bins), i.e. 0.2-0.3. A single cell is NOT interpretable on its
    own. What the table is for is the PATTERN down a column -- which is exactly what the
    `same_sign` counts in `acf_condition_contrast` test formally.

    `split_half_r` per fish is noisier still and is a GUARD, not a result: where it is
    low or negative, that fish's acf on that metric is measuring nothing and the acf
    should be ignored rather than read as a small effect.

    NaN in `acf1` means the fish had fewer than `min_bins` usable bins in that condition.
    That is informative in itself -- it is the quiescent fish that run out, and `static`
    that runs out first.
    """
    a = _cond_binned(d, bin_min, min_bouts, seed)
    rows = []
    for name, fn in OVER_TIME_METRICS.items():
        for c, lab in CONDS.items():
            if name in STIM_ONLY_METRICS and c == "none":
                continue
            s = a[a["cond"] == c]
            for f, g in s.groupby("fish_id"):
                T = g.groupby("_tb").apply(
                    lambda x: fn(x) if len(x) >= min_bouts else np.nan, include_groups=False)
                A, B = [], []
                for _, hh in g.groupby("_tb"):
                    h0, h1 = hh[hh["_h"] == 0], hh[hh["_h"] == 1]
                    if len(h0) >= min_half and len(h1) >= min_half:
                        x, y = fn(h0), fn(h1)
                        if np.isfinite(x) and np.isfinite(y):
                            A.append(x), B.append(y)
                r = float(_spearman.pearsonr(A, B).statistic) if len(A) >= min_pairs else np.nan
                nb = int(T.notna().sum())
                rows.append(dict(
                    metric=name, condition=lab, fish_id=f, n_bins=nb,
                    bouts_per_bin=int(g.groupby("_tb").size().median()) if len(g) else 0,
                    acf1=round(_acf_of(T), 3) if nb >= min_bins else np.nan,
                    acf2=round(_acf_of(T, 2), 3) if nb >= min_bins else np.nan,
                    split_half_r=round(r, 3) if np.isfinite(r) else np.nan))
    return pd.DataFrame(rows).set_index(["metric", "condition", "fish_id"])


def plot_acf_by_fish(d, bin_min=5.0, min_bouts=20, seed=0, metrics=None,
                     rel_floor=0.3, title=None):
    """The 15 fish behind each pooled acf1 -- one panel per metric, one row per fish.

    The pooled table gives a mean; this shows whether the fish agree. Read it row-wise:
    if the `no stimulus` marker sits right of `moving dots` in most rows, the suppression
    is general. If it is a handful of fish and the rest scatter around zero, it is not.

    Identity is carried by MARKER as well as colour (o / s / ^), so the figure survives
    greyscale and colour-blind viewing. A HOLLOW marker means that fish's `split_half_r`
    for that metric and condition is below `rel_floor` or undefined -- the bin estimates
    are too noisy for the acf to mean anything, so hollow points should not be read.
    The dashed vertical rule in each condition's colour is the pooled mean from
    `temporal_structure_by_cond`, drawn over the fish it is an average of.
    """
    t = temporal_structure_by_fish(d, bin_min, min_bouts, seed=seed)
    names = list(metrics or OVER_TIME_METRICS)
    order = list(FISH)
    off = {"no stimulus": .26, "moving dots": 0.0, "static dots": -.26}
    mk = {"no stimulus": "o", "moving dots": "s", "static dots": "^"}

    fig, axes = plt.subplots(1, len(names), figsize=(2.75 * len(names), 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        sub = t.loc[name]
        for c, lab in CONDS.items():
            if lab not in sub.index.get_level_values("condition"):
                continue
            g = sub.loc[lab]
            v = g["acf1"].reindex(order)
            rel = g["split_half_r"].reindex(order)
            good = (rel >= rel_floor).fillna(False).to_numpy()
            y = np.array([len(order) - 1 - i for i in range(len(order))], float) + off[lab]
            m = np.isfinite(v.to_numpy())
            ax.scatter(v.to_numpy()[m & good], y[m & good], s=26, marker=mk[lab],
                       color=COND_COLOR[c], ec=SURF, lw=.6, zorder=4)
            ax.scatter(v.to_numpy()[m & ~good], y[m & ~good], s=26, marker=mk[lab],
                       facecolor="none", ec=COND_COLOR[c], lw=1.0, zorder=3)
            ax.axvline(float(np.nanmean(v.to_numpy())), color=COND_COLOR[c],
                       lw=1.1, ls=(0, (4, 3)), alpha=.85, zorder=2)
        ax.axvline(0, color=INK2, lw=1.0, zorder=1)
        ax.set_title(name)
        ax.set_xlabel("acf1  (lag = 5 min)")
        ax.grid(axis="x", color=GRID, lw=.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"fish {f}" for f in order][::-1])
    axes[0].set_ylim(-.8, len(order) - .2)
    handles = [Line2D([], [], marker=mk[l], color=COND_COLOR[c], ls="none", ms=6,
                      mec=SURF, mew=.6, label=l) for c, l in CONDS.items()]
    handles.append(Line2D([], [], marker="o", color=MUTED, ls="none", ms=6,
                          mfc="none", label=f"split-half r < {rel_floor:.1f} — do not read"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.035))
    fig.suptitle(title or "per-fish autocorrelation at 5 min, by stimulus condition", y=1.0)
    fig.tight_layout()
    return fig


def time_variation_table(d, bin_min=15.0, kinematics="active"):
    """How much a fish varies ACROSS the session, against how much fish differ FROM EACH OTHER.

    The trend tests ask whether change has a direction. This asks whether there is change
    at all. `within` is the mean over fish of that fish's SD across its own time bins;
    `between` is the SD across fish of their session means. A ratio near 1 says a single
    fish wanders as much over three hours as the fish differ from one another -- which
    makes any individuality estimated from a short window largely temporal noise.
    """
    rows = []
    for name, T in trend_series(d, bin_min, kinematics).items():
        w = float(T.std(axis=0).mean())
        b = float(T.mean(axis=0).std())
        rows.append(dict(metric=name, within_fish_sd=round(w, 3),
                         between_fish_sd=round(b, 3), ratio=round(w / max(b, 1e-9), 2)))
    return pd.DataFrame(rows).set_index("metric").sort_values("ratio", ascending=False)


def plot_time_trends(d, bin_min=15.0, ncols=4, kinematics="active", title=None):
    """Each metric over session time, one panel per metric.

    Every fish is CENTRED ON ITS OWN SESSION MEAN before plotting, so the panels show
    change, not level -- individuality is removed by construction and cannot masquerade as
    a trend. Thin grey = the 15 fish; bold = their mean with a 95 % CI across fish, which
    is the only line worth reading.
    """
    S = trend_series(d, bin_min, kinematics)
    nrows = ceil(len(S) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.5 * nrows), sharex=True)
    for ax, (name, T) in zip(axes.ravel(), S.items()):
        C = T.sub(T.mean(axis=0), axis=1)                 # centre each fish on itself
        t = C.index.to_numpy() * bin_min + bin_min / 2
        for fid in C.columns:
            ax.plot(t, C[fid], lw=.7, color=GRID, zorder=2)
        m = C.mean(axis=1).to_numpy()
        n = C.notna().sum(axis=1).to_numpy()
        se = C.std(axis=1, ddof=1).to_numpy() / np.sqrt(np.maximum(n, 1))
        h = _spearman.t.ppf(.975, np.maximum(n - 1, 1)) * se
        ax.fill_between(t, m - h, m + h, color=SEQ[3], alpha=.25, lw=0, zorder=3)
        ax.plot(t, m, color=SEQ[6], lw=2, zorder=4)
        ax.axhline(0, color=MUTED, lw=.9, ls=(0, (4, 3)), zorder=1)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xlim(0, REC_S / 60)
    for ax in axes.ravel()[len(S):]:
        ax.set_visible(False)
    _bottom_xlabels(axes, "time through the session (min)")
    for i in range(np.atleast_2d(axes).shape[0]):
        np.atleast_2d(axes)[i, 0].set_ylabel("change from that\nfish's own mean", fontsize=8)
    fig.suptitle(title or f"behaviour over the session · {bin_min:g} min bins · each fish "
                 f"centred on itself · band = 95 % CI across fish", y=1.0, fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def plot_occupancy(d, which="dead", thr=QUIET_S, bins=80, ncols=5, title=None):
    """Occupancy heat maps, one panel per fish, over the DEAD or the ACTIVE frames.

    ⚠ The two are not the same kind of picture. A dead frame is a stationary fish, so the
    dead map is dwell time at a few resting spots and its `p99/R` says how far out those
    spots are; the active map is where the fish actually swims. Each panel is normalised
    on its own (LogNorm), so compare SHAPE between the two figures, not intensity.

    Reads the 15 per-frame CSVs the first time (~1 min); cached after that.
    """
    dish = physical_dish()
    th = np.linspace(0, 2 * np.pi, 200)
    nrows = ceil(len(FISH) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    fills, shares = [], []
    for ax, fid in zip(axes.ravel(), FISH):
        frames, x, y = fish_xy(fid)
        dead, active, _ = dead_active_masks(d, fid, frames, thr)
        m = dead if which == "dead" else active
        shares.append(m.mean())
        cx, cy, R = (float(dish.loc[fid, k]) for k in ("cx", "cy", "radius"))
        xs, ys = x[m].astype(float), y[m].astype(float)
        r = np.hypot(xs - cx, ys - cy)
        fills.append(np.quantile(r, .99) / R if len(r) else np.nan)
        lim = 1.05 * R
        if len(xs):
            ax.hist2d(xs, ys, bins=bins, cmin=1, norm=LogNorm(),
                      range=[[cx - lim, cx + lim], [cy - lim, cy + lim]],
                      cmap="Blues" if which == "dead" else "Oranges")
        ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), "-", color="#d03b3b", lw=1.2)
        ax.set_xlim(cx - lim, cx + lim)
        ax.set_ylim(cy - lim, cy + lim)
        ax.set_title(f"f{fid}  p99/R={fills[-1]:.2f}  ({100 * shares[-1]:.0f} % of frames)",
                     fontsize=8, color=INK)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes.ravel()[len(FISH):]:
        ax.set_visible(False)
    what = ("DEAD frames — inside a no-bout stretch > %g s" % thr if which == "dead"
            else "ACTIVE frames — everything else between first and last bout")
    fig.suptitle(title or f"occupancy over {what}   (red = the physical 100 mm dish wall)",
                 y=1.0, fontsize=11, color=INK)
    fig.tight_layout()
    print(f"{which}: {100 * np.mean(shares):.1f} % of tracked frames on average, "
          f"mean p99/R = {np.nanmean(fills):.2f}")
    return fig


# --------------------------------------------------------------------------- #
#  8bis -- is QUIET_S = 10 the right cut?
# --------------------------------------------------------------------------- #
CANDIDATES = [2, 3, 5, 7.5, 10, 15, 20, 30, 60]


def plot_dt_spectrum(d, thr=QUIET_S, cands=(3, 10, 30), title=None):
    """(A) The interbout interval on a log axis, raw counts in log-spaced bins: is there a dip to put a threshold in?
    Prints the local minima of the smoothed log-interval density above 1 s."""
    dt = d["dt_s"].to_numpy(float)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    edges = np.logspace(np.log10(0.15), np.log10(max(dt.max(), 500)), 80)
    ctr = np.sqrt(edges[1:] * edges[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.4), gridspec_kw={"wspace": .24})
    for ax, logy in zip(axes, (False, True)):
        for _, g in d.groupby("fish_id"):
            v = g["dt_s"].to_numpy(float)
            v = v[np.isfinite(v) & (v > 0)]
            h, _ = np.histogram(v, bins=edges)
            ax.plot(ctr, h, color=GRID, lw=.7, zorder=1)
        h, _ = np.histogram(dt, bins=edges)
        ax.plot(ctr, h / d["fish_id"].nunique(), color=BLUE, lw=1.8, zorder=3)
        for c in cands:
            ax.axvline(c, color=ORANGE if c == thr else MUTED, lw=1.4 if c == thr else .8,
                       ls="-" if c == thr else (0, (3, 2)), zorder=2)
            ax.text(c, ax.get_ylim()[1], f" {c:g}s", fontsize=7.5, va="top", color=ORANGE if c == thr else MUTED)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("interbout interval (s)")
        ax.set_ylabel("bouts per bin" + (", log scale" if logy else ", per fish"))
    axes[0].set_title("linear counts: is there a dip?", fontsize=9.5, color=INK, loc="left")
    axes[1].set_title("log counts: the tail, where the cut lives", fontsize=9.5, color=INK, loc="left")
    fig.suptitle(title or "interval spectrum: grey = one fish, blue = pooled", y=1.03, fontsize=11, color=INK)
    show()
    lo = np.log10(dt)
    h, e = np.histogram(lo, bins=60)
    s = np.convolve(h, np.r_[1, 2, 3, 2, 1] / 9.0, mode="same")
    c = .5 * (e[1:] + e[:-1])
    dips = [(10 ** c[i], s[i]) for i in range(2, len(s) - 2)
            if s[i] < s[i - 1] and s[i] < s[i + 1] and 10 ** c[i] > 1.0]
    print("local minima in the smoothed log-interval density above 1 s:")
    print("  none: the tail is monotone, so no threshold is picked out by the data" if not dips else
          "\n".join(f"  {x:7.2f} s   (count {y:.0f})" for x, y in dips))


def threshold_sweep(d, cands=CANDIDATES, steady_pos=60, plot=True):
    """(B) Rebuild active / gap_before / run_id at each candidate threshold, as the cleaning does, and report what
    the models would inherit. ``transient_s`` = median interval over the first 5 bouts of a run minus the steady
    value (from bout ``steady_pos``): the non-steady effect the run clock carries."""
    out = []
    for thr in cands:
        t = d.copy()
        t["gap_after"] = t.dt_s > thr
        t["gap_before"] = t.groupby("fish_id").gap_after.shift(1, fill_value=False)
        t["active"] = ~(t.gap_after | t.gap_before)
        t["run_id"] = (t.gap_before | (t.fish_id != t.fish_id.shift())).cumsum()
        a = t[t.active].copy()
        a["pos"] = a.groupby("run_id").cumcount()
        early, late = a[a.pos < 5], a[a.pos >= steady_pos]
        out.append(dict(
            thr_s=thr, gaps=int(t.gap_after.sum()),
            pct_session=round(100 * t.loc[t.gap_after, "dt_s"].sum() / t.dt_s.sum(), 1),
            active=int(len(a)), pct_kept=round(100 * len(a) / len(t), 2),
            runs_all=int(t.run_id.nunique()), runs_act=int(a.run_id.nunique()),
            med_run=int(a.groupby("run_id").size().median()),
            dt_med=round(float(a.dt_s.median()), 3), dt_p99=round(float(a.dt_s.quantile(.99)), 2),
            transient_s=(round(float(early.dt_s.median() - late.dt_s.median()), 3) if len(late) > 20 else np.nan)))
    sweep = pd.DataFrame(out).set_index("thr_s")
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.9), gridspec_kw={"wspace": .32})
        for ax, col, lab in ((axes[0], "pct_kept", "bouts kept as active (%)"),
                             (axes[1], "med_run", "median run length (bouts)"),
                             (axes[2], "transient_s", "non-steady effect (s)")):
            ax.plot(sweep.index, sweep[col], "o-", color=BLUE, lw=1.5, ms=4.5)
            ax.axvline(QUIET_S, color=ORANGE, lw=1.4)
            ax.set_xscale("log"); ax.set_xlabel("threshold (s)"); ax.set_ylabel(lab)
            ax.set_xticks(cands); ax.set_xticklabels([f"{c:g}" for c in cands], fontsize=7)
        fig.suptitle("what the pipeline inherits from the threshold (orange = the 10 s in use)", y=1.04,
                     fontsize=11, color=INK)
        show()
    return sweep


def _active_runs(d, quiet_s=QUIET_S):
    """Active bouts with a run id (the raw table has neither), |dtheta| and r/R."""
    t = d.copy()
    t["gap_after"] = t.dt_s > quiet_s
    t["gap_before"] = t.groupby("fish_id").gap_after.shift(1, fill_value=False)
    t["run_id"] = (t.gap_before | (t.fish_id != t.fish_id.shift())).cumsum()
    a = t[~(t.gap_after | t.gap_before)].copy()
    a["abs_dtheta"] = a.dtheta_deg.abs()
    a["r_norm"] = 1.0 - a["dist_to_border_norm"]
    return a


# --------------------------------------------------------------------------- #
#  17 -- how far back does a bout remember? Lag-k correlation within runs, per fish, with a run bootstrap inside
#  each fish (the pairs of a run are not independent) and a bootstrap over fish for the median.
# --------------------------------------------------------------------------- #
LAGS = (1, 2, 3, 4, 5)
CHANS = [("distance_px", "displacement"), ("dt_s", "interval"),
         ("abs_dtheta", r"$|\Delta\theta|$"), ("dtheta_deg", r"signed $\Delta\theta$")]
N_BOOT = 2000


def _run_stats(g, col, k):
    """Per-run sufficient statistics of the (x_t, x_{t+k}) pairs inside that run."""
    v = g[col].to_numpy(float)
    r = g.run_id.to_numpy()
    if len(v) <= k:
        return np.empty((0, 6))
    ok = (r[k:] == r[:-k])
    x, y, rid = v[:-k][ok], v[k:][ok], r[:-k][ok]
    m = np.isfinite(x) & np.isfinite(y)
    x, y, rid = x[m], y[m], rid[m]
    if not len(x):
        return np.empty((0, 6))
    out = []
    for _, idx in pd.Series(np.arange(len(rid))).groupby(rid):
        i = idx.to_numpy()
        xa, ya = x[i], y[i]
        out.append((len(xa), xa.sum(), ya.sum(), (xa * xa).sum(), (ya * ya).sum(), (xa * ya).sum()))
    return np.array(out, float)


def _corr_from(S):
    """Pearson r from summed sufficient statistics."""
    n, sx, sy, sxx, syy, sxy = S
    if n < 5:
        return np.nan
    cov = sxy - sx * sy / n
    vx, vy = sxx - sx * sx / n, syy - sy * sy / n
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def lag_corr_ci(g, col, k, n_boot=N_BOOT, seed=0):
    """(r, lo, hi): point estimate and a 95 % interval from resampling the fish's runs."""
    S = _run_stats(g, col, k)
    if len(S) < 3:
        return np.nan, np.nan, np.nan
    r = _corr_from(S.sum(axis=0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), size=(n_boot, len(S)))
    bs = np.array([_corr_from(S[i].sum(axis=0)) for i in idx])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r, float(lo), float(hi)


def _median_ci(v, n_boot=N_BOOT, seed=1):
    """Bootstrap interval of the median ACROSS fish."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    bs = np.median(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def lag_memory(d, plot=True):
    """Lags 1-5 of displacement, interval, |dtheta| and signed dtheta, within runs: the table (median over fish with
    its bootstrap over fish; fish of the same sign; fish whose own run-bootstrap interval excludes 0), the decay with
    lag and the per-fish bars. Returns the table."""
    A = _active_runs(d)
    ACF, LO, HI = {}, {}, {}
    for col, _ in CHANS:
        for k in LAGS:
            trip = [lag_corr_ci(A[A.fish_id == f], col, k) for f in FISH]
            ACF[(col, k)], LO[(col, k)], HI[(col, k)] = (np.array([t[i] for t in trip]) for i in range(3))
    rows = []
    for col, nice in CHANS:
        for k in LAGS:
            v = ACF[(col, k)]
            med, mlo, mhi = _median_ci(v)
            rows.append(dict(channel=nice.replace("$", "").replace("\\", ""), lag=k, median=round(med, 3),
                             med_lo=round(mlo, 3), med_hi=round(mhi, 3),
                             same_sign=f"{max(int((v > 0).sum()), int((v < 0).sum()))}/{len(FISH)}",
                             fish_CI_excl_0=f"{int(((LO[(col, k)] > 0) | (HI[(col, k)] < 0)).sum())}/{len(FISH)}"))
    T = pd.DataFrame(rows)
    print(f"{len(FISH)} fish x {len(CHANS)} channels x {len(LAGS)} lags, {A.run_id.nunique()} runs, "
          f"{N_BOOT} run-resamples each")
    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3), sharey=True, gridspec_kw={"wspace": .10})
        for ax, (col, nice) in zip(axes, CHANS):
            M = np.vstack([ACF[(col, k)] for k in LAGS])
            for j in range(M.shape[1]):
                ax.plot(LAGS, M[:, j], lw=.7, color=GRID, marker="o", ms=2.5, zorder=1)
            med = np.array([_median_ci(ACF[(col, k)]) for k in LAGS])
            ax.fill_between(LAGS, med[:, 1], med[:, 2], color=BLUE, alpha=.22, lw=0, zorder=2)
            ax.plot(LAGS, med[:, 0], lw=2.0, color=BLUE, marker="o", ms=5, zorder=3, label="median over fish")
            ax.axhline(0, color=INK, lw=1.0, zorder=2)
            ax.set_xticks(LAGS); ax.set_xlabel("lag (bouts)")
            ax.set_title(nice, fontsize=10, color=INK, loc="left")
            ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
        axes[0].set_ylabel("correlation within a run")
        axes[0].legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper right")
        fig.suptitle("decay with lag: grey = one fish, band = 95 % bootstrap ACROSS fish", y=1.04, fontsize=11,
                     color=INK)
        show()
        ramp = [plt.get_cmap("PuBu")(x) for x in np.linspace(0.38, 0.95, len(LAGS))]
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 5.8), sharex=True)
        x, w = np.arange(len(FISH)), .165
        for ax, (col, nice) in zip(axes.ravel(), CHANS):
            for i, k in enumerate(LAGS):
                xs = x + (i - (len(LAGS) - 1) / 2) * w
                m, lo, hi = ACF[(col, k)], LO[(col, k)], HI[(col, k)]
                ax.bar(xs, m, w - .015, color=ramp[i], label=f"lag {k}", zorder=2)
                ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK2, elinewidth=.6, capsize=0,
                            zorder=4)
            ax.axhline(0, color=INK, lw=1.0, zorder=3)
            ax.set_xticks(x); ax.set_xticklabels([f"{f}" for f in FISH], fontsize=7)
            ax.set_ylabel("correlation"); ax.set_title(nice, fontsize=10, color=INK, loc="left")
            ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
        axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=5, loc="upper left")
        for ax in axes[1]:
            ax.set_xlabel("fish")
        fig.suptitle("per fish: whiskers = 95 % bootstrap over that fish's RUNS", y=1.0, fontsize=11, color=INK)
        fig.tight_layout(); show()
    return T


# --------------------------------------------------------------------------- #
#  18 -- the long timescale, and whether it is the fish or the schedule
# --------------------------------------------------------------------------- #
WCH = [("dt_s", "interval", True), ("distance_px", "displacement", True), ("r_norm", "radial position", False)]
WINS, BLOCK_B, N_SHUF = (5, 10, 20, 40, 80), 10, 20
MIN_BOUTS_BIN = 10


def _chan(g, col, take_log):
    v = g[col].to_numpy(float)
    return np.log(np.clip(v, 1e-6, None)) if take_log else v


def window_acf_real(d, W=60):
    """(A) ``generative_metrics.win_acf1`` (the statistic the models are scored on) on the real fish: log interval,
    log displacement and r/R, windows of W bouts inside one run, demeaned within the fish, lag-1 correlation between
    adjacent windows. A fish without two adjacent windows inside a run gets NaN."""
    from twin.generative_metrics import win_acf1
    A = _active_runs(d)
    rows = []
    for f in FISH:
        g = A[A.fish_id == f]
        r = {"fish": f, "active_bouts": len(g)}
        for col, lab, lg in WCH:
            r[lab] = win_acf1(_chan(g, col, lg), W=W, runs=g.run_id.to_numpy())
        rows.append(r)
    T = pd.DataFrame(rows).set_index("fish")
    labs = [lab for _, lab, _ in WCH]
    print(f"median over fish: {T[labs].median().round(3).to_dict()}; positive in "
          + str({lab: f"{int((T[lab] > 0).sum())}/{int(T[lab].notna().sum())}" for lab in labs}))
    print(f"median interval {A.dt_s.median():.2f} s, so a {W}-bout window is about {W * A.dt_s.median():.0f} s")
    return T


def _win_pairs(runs, W):
    xs, ys = [], []
    for v in runs:
        n = len(v) // W
        if n < 2:
            continue
        m = np.array([v[i * W:(i + 1) * W].mean() for i in range(n)])
        xs.append(m[:-1]); ys.append(m[1:])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def _corr(x, y, min_n=20):
    if len(x) < min_n:
        return np.nan
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() >= min_n else np.nan


def _shuffle_runs(runs, rng, B=None):
    """Pool a fish's runs, permute (single bouts, or blocks of B), re-split into runs of the same lengths."""
    lengths = [len(v) for v in runs]
    if B is None:
        pool = rng.permutation(np.concatenate(runs))
    else:
        blocks = [v[i:i + B] for v in runs for i in range(0, len(v), B)]
        rng.shuffle(blocks)
        pool = np.concatenate(blocks)
    out, i = [], 0
    for L in lengths:
        out.append(pool[i:i + L]); i += L
    return out


def window_acf_vs_shuffles(d):
    """(B) Window correlation against the window size W, against a block shuffle (B = 10: keeps every dependence
    shorter than 10 bouts, what a ten-lag model could at best reproduce) and a full shuffle (must be 0). Prints
    observed minus block-shuffled and the bouts each W uses (large W uses only long runs)."""
    A = _active_runs(d)
    runs = {(f, lab): [_chan(g, col, lg) for _, g in A[A.fish_id == f].groupby("run_id")]
            for f in FISH for col, lab, lg in WCH}
    OBS, BLK, FULL, COV = {}, {}, {}, {}
    rng = np.random.default_rng(0)
    for _, lab, _ in WCH:
        for W in WINS:
            o, b, u, cov = [], [], [], 0
            for f in FISH:
                rr = [v for v in runs[(f, lab)] if len(v) >= 2 * W]
                cov += sum(len(v) for v in rr)
                o.append(_corr(*_win_pairs(rr, W)))
                b.append(np.nanmean([_corr(*_win_pairs(_shuffle_runs(rr, rng, BLOCK_B), W)) for _ in range(N_SHUF)]))
                u.append(np.nanmean([_corr(*_win_pairs(_shuffle_runs(rr, rng, None), W)) for _ in range(N_SHUF)]))
            OBS[(lab, W)], BLK[(lab, W)], FULL[(lab, W)] = map(np.array, (o, b, u))
            COV[(lab, W)] = cov
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), sharey=True, gridspec_kw={"wspace": .10})
    for ax, (_, lab, _) in zip(axes, WCH):
        for key, colr, name, ls in ((OBS, BLUE, "observed", "-"),
                                    (BLK, ORANGE, f"block shuffle (B={BLOCK_B})", (0, (4, 2))),
                                    (FULL, MUTED, "full shuffle", (0, (1, 2)))):
            M = np.vstack([key[(lab, W)] for W in WINS])
            ax.plot(WINS, np.nanmedian(M, axis=1), lw=1.8, color=colr, ls=ls, marker="o", ms=4, label=name)
            if key is OBS:
                ax.fill_between(WINS, np.nanpercentile(M, 25, axis=1), np.nanpercentile(M, 75, axis=1), color=colr,
                                alpha=.18, lw=0)
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_xscale("log"); ax.set_xticks(WINS); ax.set_xticklabels([str(w) for w in WINS])
        ax.set_xlabel("window size $W$ (bouts)"); ax.set_title(lab, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("correlation between\nsuccessive windows")
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK2, loc="upper right")
    fig.suptitle("median over fish, band = IQR over fish", y=1.04, fontsize=11, color=INK)
    show()
    gap = pd.DataFrame({lab: [float(np.nanmedian(OBS[(lab, W)] - BLK[(lab, W)])) for W in WINS] for _, lab, _ in WCH},
                       index=[f"W={w}" for w in WINS])
    cov = pd.DataFrame({lab: [COV[(lab, W)] for W in WINS] for _, lab, _ in WCH}, index=[f"W={w}" for w in WINS])
    print("observed minus block-shuffled (median over fish): what ten lags cannot reach")
    display(gap.round(3))
    print(f"bouts contributing (of {len(A)} active)")
    display(cov)


def fish_or_schedule(d):
    """(C) The schedule is identical and synchronous for all 15 fish, so what it drives is COMMON to them. Bins of one
    110 s stimulus cycle (>= 10 active bouts), demeaned within the fish: each fish's own lag-1 correlation across
    cycles against its correlation with the mean of the other 14 at the same cycle. Shared = schedule, unshared =
    the animal."""
    A = _active_runs(d)
    cyc_s = CYCLE_FR / FPS
    A = A.assign(_cyc=(A.t_onset_s // cyc_s).astype(int))
    rows = []
    for col, lab, lg in WCH:
        M = {}
        for f in FISH:
            g = A[A.fish_id == f]
            s = g.assign(_v=_chan(g, col, lg)).groupby("_cyc").agg(v=("_v", "mean"), n=("_v", "size"))
            M[f] = s[s.n >= MIN_BOUTS_BIN]["v"]
        T = pd.DataFrame(M).sort_index()
        T = T - T.mean()
        own, shared = [], []
        for f in FISH:
            j = pd.concat([T[f].rename("x"), T.drop(columns=[f]).mean(axis=1).rename("o")], axis=1).dropna()
            if len(j) < 20:
                own.append(np.nan); shared.append(np.nan); continue
            xv = j.x.to_numpy()
            own.append(float(np.corrcoef(xv[:-1], xv[1:])[0, 1]))
            shared.append(float(np.corrcoef(j.x, j.o)[0, 1]))
        rows.append(dict(channel=lab, own_lag1=float(np.nanmedian(own)), cross_fish=float(np.nanmedian(shared)),
                         own_pos=f"{int(np.nansum(np.array(own) > 0))}/{len(FISH)}",
                         cross_pos=f"{int(np.nansum(np.array(shared) > 0))}/{len(FISH)}",
                         n_cycles=int(T.notna().sum().median())))
    R = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x, w = np.arange(len(WCH)), .36
    for i, (key, colr, lab2) in enumerate((("own_lag1", BLUE, "own lag-1 across cycles  (the fish)"),
                                           ("cross_fish", ORANGE, "correlation with the other 14  (the schedule)"))):
        ax.bar(x + (i - .5) * w, R[key], w - .02, color=colr, label=lab2, zorder=2)
    ax.axhline(0, color=INK, lw=1.0, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(list(R.channel)); ax.set_ylabel("correlation")
    ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper right")
    ax.set_title("the slow fluctuation is not shared between fish", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout(); show()
    return R
