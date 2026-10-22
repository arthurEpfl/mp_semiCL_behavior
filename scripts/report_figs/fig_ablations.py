"""Figure fig:ablations (Chapter 5, sec:r-ablations): what M6 needs.

Each row is M6 retrained without one ingredient (or, for "z frozen", M6 evaluated with every simulated fish set to
the population fish z = 0; the capacity rows retrain M6 with 64 or 256 hidden units instead of 128). Each panel is
the change from M6 at the SAME seed (seed 0 = unit 26e, seed 101 = unit 30d; seeds are separate CV splits and are
never pooled), read from Model/report_outputs/summary_M<key>.json:

    a  matched AUC                                   (auc_matched)
    b  joint held-out NLL, bits per bout             (lik_joint; z frozen = M6 by construction, not plotted)
    c  dip, mm at 3.60 px/mm                         (dip)
    d  grading, mm at 3.60 px/mm                     (grading)
    e  moving-profile error, real SEM                (mv_err_sem)

Scale: the archived dip and grading (summary files, stream table) are in mm at the profile code's nominal
3.67 px/mm; every millimetre here (values, M6 streams, bands, deltas() for tab:ablations, printed numbers) is
multiplied by style.MM_3P67_TO_3P60 (= 3.67 / 3.60), so the figure is at Chapter 3's physical 3.60 px/mm. The
stream-0 archive check compares unconverted values; AUC, bits and SEM-unit errors are not converted.

References. a, b: M6's archived value (AUC and likelihood have no rollout-stream replicates). c-e: the MEAN of M6's
four rollout streams (missing_pieces.stream_table(filtered=True); stream 0 = the archived stream), not the archived
stream alone: at seed 0 the archived stream is M6's deepest-dip / lowest-error stream, and the ablation readouts do
not share that luck (checked below: the near-null rows sit at about -(stream 0 - mean) from the archived stream).
The top row of c-e draws M6's four streams themselves (archived stream = diamond).

Bands: a +/-0.05 (claim threshold, Methods sec:m-stats); b +/-0.016 bits (likelihood seed SD, sec:m-stats);
c-e +/- SD * sqrt(1 + 1/4), the SD of one readout minus a 4-stream mean (SD = M6's stream SD at that seed, ddof=1):
seed 0 shaded, seed 101 dashed.

Axes: b and e are symmetric-log beyond +/-0.08 bits and +/-0.5 SEM (so -kin0, -stim, -pos, -motion are drawn, not
clipped); in c, -stim and -pos have no dip at all (the readout is 0 by definition), drawn as triangles labelled
"no dip" instead of a value.

The script asserts that every unit shares M6's held-out fish per fold at its seed, that stream 0 of the stream table
reproduces M6's archived summary, and that -stim / -pos have dip exactly 0. `deltas()` (archived-stream reference)
is kept: it is what tab:ablations prints ($RF/figs/check_ablations_table.py compares the two).

    uv run python scripts/report_figs/fig_ablations.py
"""
import json
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

SUMMARY = MODEL / "report_outputs"
CKPTS = MODEL / "report_ckpts"
SEEDS = (0, 101)
M6 = ("26e", "30d")
MINUS = "−"
EPOCH_LIMIT = 80          # train_model1_colab.py --epochs default; Methods: "for at most 80 epochs"

# (group, row label, (key s0, key s101)); top to bottom
ROWS = [
    ("stimulus", MINUS + "stim", ("29e", "33u")),
    ("stimulus", MINUS + "pos", ("33a", "33b")),
    ("stimulus", MINUS + "count", ("29c", "33s")),
    ("stimulus", MINUS + "range", ("33c", "33d")),
    ("stimulus", MINUS + "motion", ("29d", "33t")),
    ("history", MINUS + "turn4", ("33e", "33f")),
    ("history", MINUS + "turn1", ("33g", "33h")),
    ("history", MINUS + "turn0", ("33i", "33j")),
    ("history", MINUS + "kin1", ("33k", "33l")),
    ("history", MINUS + "kin0", ("33m", "33n")),
    ("onset", MINUS + "clock", ("33o", "33p")),
    ("individuality", r"$\mathbfit{z}$ frozen", ("33q", "33r")),
    ("capacity", "64 units", ("34g", "34h")),
    ("capacity", "256 units", ("34i", "34j")),
]
Z_FROZEN = ("33q", "33r")
NO_DIP = (MINUS + "stim", MINUS + "pos")              # dip readout 0 by definition (asserted)
NEAR_NULL = [MINUS + "turn4", MINUS + "turn1", MINUS + "turn0", MINUS + "kin1", MINUS + "kin0",
             "64 units", "256 units"]                # for the stream-luck check only

# summary column, x label, scale (None = linear, (linthresh, linscale) = symlog), x limits, x ticks, direction hint
COLS = [
    ("auc_matched", "Δ matched AUC", None, (-0.075, 0.225), (0, 0.1, 0.2), "worse →"),
    ("lik_joint", "Δ NLL\n(bits/bout)", (0.08, 1.0), (-0.04, 0.42), (0, 0.05, 0.3), "worse →"),
    ("dip", "Δ dip (mm)", None, (-2.5, 1.25), (-2, -1, 0, 1), "← shallower"),
    ("grading", "Δ grading (mm)", None, (-4.65, 1.95), (-4, -2, 0), "← weaker"),
    ("mv_err_sem", "Δ moving error\n(SEM)", (0.5, 1.25), (-0.3, 11.5), (0, 0.5, 8), "worse →"),
]
MINOR = {"lik_joint": (0.1, 0.2), "mv_err_sem": (1, 2, 4)}   # unlabelled ticks in the log part
BAND_FIXED = {"auc_matched": 0.05, "lik_joint": 0.016}      # Methods sec:m-stats (claim threshold / seed SD)
STREAM_COLS = ("dip", "grading", "mv_err_sem")
BAND_FACTOR = math.sqrt(1 + 1 / 4)                           # one readout minus the mean of 4 streams
MM_COLS = ("dip", "grading")                                 # millimetre readouts: 3.67 -> 3.60 px/mm


def to360(col, v):
    """A readout of column ``col`` at 3.60 px/mm: millimetre columns x MM_3P67_TO_3P60, the others unchanged."""
    return v * MM_3P67_TO_3P60 if col in MM_COLS else v


def load_summary(key):
    return json.loads((SUMMARY / f"summary_M{key}.json").read_text())


def manifest(key):
    return json.loads((CKPTS / f"M{key}" / "manifest_model1.json").read_text())


def heldout(key):
    return {m["fold"]: tuple(m["heldout"]) for m in manifest(key)}


def rnd(x, nd):
    """Round half up (as the chapter tables), signed, with a typographic minus."""
    d = Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    d = abs(d) if d == 0 else d
    return ("+" if d >= 0 else MINUS) + f"{abs(d):.{nd}f}"


def streams():
    """M6's 4 rollout streams per seed: {seed: {col: array(stream 0..3)}}, mm at 3.60 px/mm; stream 0 checked
    against the archive BEFORE converting (both at the profile code's 3.67)."""
    import missing_pieces as MP
    import model_figures as MF
    assert_code_scale(MF.PX_TO_MM, "model_figures.PX_TO_MM")
    t = MP.stream_table(filtered=True)
    out = {}
    for i, (s, lab) in enumerate(zip(SEEDS, ("M6 s0", "M6 s101"))):
        g = t[t.model == lab].sort_values("stream")
        assert list(g.stream) == [0, 1, 2, 3], (lab, list(g.stream))
        out[s] = {c: g[c].to_numpy(float) for c in STREAM_COLS}
        arch = load_summary(M6[i])                          # stream 0 = the archived stream
        for c in STREAM_COLS:
            assert abs(out[s][c][0] - arch[c]) < 6e-3, (lab, c, out[s][c][0], arch[c])
        out[s] = {c: to360(c, v) for c, v in out[s].items()}
    return out


def stream_sd(st=None):
    """M6's rollout-stream SD (4 streams, ddof=1) of dip, grading, moving error, per seed."""
    st = st or streams()
    return {s: {c: float(np.std(st[s][c], ddof=1)) for c in STREAM_COLS} for s in SEEDS}


def deltas():
    """Change from M6's ARCHIVED value at the same seed (the reference of tab:ablations), every column; dip and
    grading at 3.60 px/mm (the exact archived difference x MM_3P67_TO_3P60). ``base`` = the raw summaries."""
    base = [load_summary(k) for k in M6]
    ho = [heldout(k) for k in M6]
    D = {}
    for _, lab, keys in ROWS:
        for i, k in enumerate(keys):
            assert heldout(k) == ho[i], f"{k} does not share the folds of M6 at seed {SEEDS[i]}"
            s = load_summary(k)
            for c, *_ in COLS:
                D[(lab, SEEDS[i], c)] = to360(c, float(Decimal(str(s[c])) - Decimal(str(base[i][c]))))
    return D, base


def deltas_plot(st):
    """What the figure plots: a, b against M6's archived value; c-e against the mean of M6's 4 streams (3.60)."""
    D, base = deltas()
    mean = {s: {c: float(np.mean(st[s][c])) for c in STREAM_COLS} for s in SEEDS}
    P = {}
    for _, lab, keys in ROWS:
        for i, k in enumerate(keys):
            s = load_summary(k)
            for c, *_ in COLS:
                P[(lab, SEEDS[i], c)] = ((to360(c, s[c]) - mean[SEEDS[i]][c]) if c in STREAM_COLS
                                         else D[(lab, SEEDS[i], c)])
            if lab in NO_DIP:
                assert s["dip"] == 0, (k, s["dip"])
    return P, D, base, mean


def epochs(key):
    return [m["epochs_run"] for m in sorted(manifest(key), key=lambda m: m["fold"])]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    st = streams()
    sd = stream_sd(st)
    P, D, base, mean = deltas_plot(st)

    # y positions: reference row, then the ablation rows top to bottom, a gap between groups
    groups = [("reference", "M6 streams", None)] + ROWS
    ys, y, prev = [], 0.0, None
    for g, _, _ in groups:
        if prev is not None and g != prev:
            y += 0.7
        ys.append(y)
        y += 1.0
        prev = g
    ys = np.array(ys)
    y_ref, ys_rows = ys[0], ys[1:]
    off = {0: -0.2, 101: 0.2}                         # seed 0 above, seed 101 below (y axis inverted)

    fig, axes = plt.subplots(1, len(COLS), figsize=(TEXTWIDTH_IN, 3.4), sharey=True,
                             gridspec_kw=dict(wspace=0.17, width_ratios=(1, 1, 1, 1, 1.22)))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.885, bottom=0.125)

    for j, (ax, (col, xlab, scale, xl, xt, hint)) in enumerate(zip(axes, COLS)):
        lo, hi = xl
        if scale:
            ax.set_xscale("symlog", linthresh=scale[0], linscale=scale[1])
        # bands
        if col in BAND_FIXED:
            b = BAND_FIXED[col]
            ax.axvspan(-b, b, color=FAINT, alpha=0.55, lw=0, zorder=0)
        else:
            b0, b1 = BAND_FACTOR * sd[0][col], BAND_FACTOR * sd[101][col]
            ax.axvspan(-b0, b0, color=FAINT, alpha=0.55, lw=0, zorder=0)
            for x in (-b1, b1):
                ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (2.2, 1.6)), zorder=0.5)
        ax.axvline(0, color=INK, lw=0.7, zorder=1)
        # faint row guides and group separators
        for yy in ys:
            ax.axhline(yy, color=FAINT, lw=0.35, zorder=0.2)
        for r in range(1, len(groups)):
            if groups[r][0] != groups[r - 1][0]:
                ax.axhline((ys[r] + ys[r - 1]) / 2, color=MUTED, lw=0.5, zorder=0.3)

        # reference row: M6's own four rollout streams (c-e only), relative to their mean
        if col in STREAM_COLS:
            for s in SEEDS:
                d = st[s][col] - mean[s][col]
                kw = dict(color=MUTED, mec=MUTED, mew=0.7, fillstyle=SEED_MFC[s], ls="none", zorder=3)
                ax.plot(d[1:], [y_ref + off[s]] * 3, marker="o", ms=3.0, **kw)
                ax.plot(d[:1], [y_ref + off[s]], marker="D", ms=3.4, **kw)

        for r, (_, lab, keys) in enumerate(ROWS):
            yr = ys_rows[r]
            if col == "lik_joint" and keys == Z_FROZEN:
                ax.text(0.006, yr, "= M6", fontsize=7, color=MUTED, style="italic", va="center", ha="left")
                continue
            if col == "dip" and lab in NO_DIP:
                edge = lo + 0.02 * (hi - lo)
                for s in SEEDS:
                    ax.plot([edge], [yr + off[s]], marker="<", color=BLUE, mec=BLUE, ms=4.0, mew=0.8,
                            fillstyle=SEED_MFC[s], ls="none", zorder=3, clip_on=False)
                ax.text(lo + 0.075 * (hi - lo), yr, "no dip", fontsize=7, color=MUTED, style="italic",
                        va="center", ha="left", zorder=2)
                continue
            for s in SEEDS:
                v = P[(lab, s, col)]
                assert lo < v < hi, (lab, s, col, v, xl)
                ax.plot([v], [yr + off[s]], marker="o", color=BLUE, mec=BLUE, ms=3.6, mew=0.8,
                        fillstyle=SEED_MFC[s], ls="none", zorder=3)

        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_minor_locator(FixedLocator(MINOR[col]) if col in MINOR else NullLocator())
        ax.set_xticklabels([f"{t:g}".replace("-", MINUS) for t in xt])
        ax.set_xlabel(xlab, labelpad=2, linespacing=1.0)
        ax.text(1.0, 1.012, hint, transform=ax.transAxes, fontsize=7, color=MUTED, ha="right", va="bottom")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2.5, pad=1.5)
        ax.tick_params(axis="x", which="minor", length=1.5, width=0.5)
        ax.spines["left"].set_visible(False)
        panel(ax, "abcde"[j])

    ax0 = axes[0]
    ax0.set_ylim(ys[-1] + 0.6, ys[0] - 0.6)
    ax0.set_yticks(ys)
    ax0.set_yticklabels([lab for _, lab, _ in groups], fontsize=7, color=INK)
    ax0.get_yticklabels()[0].set_color(MUTED)
    # group labels left of the row labels
    for g in dict.fromkeys(gr for gr, _, _ in groups):
        if g == "reference":
            continue
        idx = [i for i, (gr, _, _) in enumerate(groups) if gr == g]
        ax0.annotate(g, xy=(0, ys[idx].mean()), xycoords=("axes fraction", "data"),
                     xytext=(-40, 0), textcoords="offset points", ha="right", va="center",
                     fontsize=7, color=MUTED, style="italic", annotation_clip=False)

    h_s0 = Line2D([], [], marker="o", color=BLUE, ls="none", ms=4)
    h_s1 = Line2D([], [], marker="o", color=BLUE, mfc="none", ls="none", ms=4, mew=0.8)
    h_st = Line2D([], [], marker="o", color=MUTED, ls="none", ms=3.2)
    h_ar = Line2D([], [], marker="D", color=MUTED, ls="none", ms=3.4)
    h_b0 = Patch(fc=FAINT, alpha=0.55, ec="none")
    h_b1 = Line2D([], [], color=MUTED, lw=0.6, ls=(0, (2.2, 1.6)))
    fig.legend(handles=[h_s0, h_s1, h_st, h_ar, (h_b0, h_b1)],
               labels=["seed 0", "seed 101", "M6 rollout streams", "archived stream", "bands (see caption)"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
               loc="upper center", ncol=5, bbox_to_anchor=(0.59, 1.0), handletextpad=0.4,
               columnspacing=1.2, borderaxespad=0.1)
    save(fig, "fig_ablations")

    # ---------------- what the figure shows (for the figure note) ----------------
    print(f"mm values (dip, grading) at 3.60 px/mm (x {MM_3P67_TO_3P60:.6f}); AUC, bits and SEM unchanged")
    print("M6 archived s0 | s101:", {c: (round(to360(c, base[0][c]), 4), round(to360(c, base[1][c]), 4))
                                     for c, *_ in COLS})
    print("M6 4-stream mean s0 | s101:", {c: (round(mean[0][c], 3), round(mean[101][c], 3)) for c in STREAM_COLS})
    print("M6 stream 0 - mean s0 | s101:",
          {c: (round(st[0][c][0] - mean[0][c], 3), round(st[101][c][0] - mean[101][c], 3)) for c in STREAM_COLS})
    print("M6 streams (0..3):", {s: {c: np.round(st[s][c], 3).tolist() for c in STREAM_COLS} for s in SEEDS})
    print("stream SD s0 | s101:", {c: (round(sd[0][c], 3), round(sd[101][c], 3)) for c in STREAM_COLS})
    print("plotted band c-e = SD*sqrt(5/4), s0 | s101:",
          {c: (round(BAND_FACTOR * sd[0][c], 3), round(BAND_FACTOR * sd[101][c], 3)) for c in STREAM_COLS})
    prec = {"auc_matched": 3, "lik_joint": 3}
    print("\nPLOTTED values (a, b vs archived M6; c-e vs M6 4-stream mean):")
    for _, lab, _ in ROWS:
        print(f"{lab:>22s} " + "  ".join(
            f"{c}: {rnd(P[(lab, 0, c)], prec.get(c, 2))}|{rnd(P[(lab, 101, c)], prec.get(c, 2))}"
            for c, *_ in COLS))

    # rows beyond the plotted band at both seeds with the same sign
    def beyond(col, band):
        out = []
        for _, lab, keys in ROWS:
            if (col == "lik_joint" and keys == Z_FROZEN) or (col == "dip" and lab in NO_DIP):
                continue
            v0, v1 = P[(lab, 0, col)], P[(lab, 101, col)]
            if abs(v0) > band[0] and abs(v1) > band[1] and np.sign(v0) == np.sign(v1):
                out.append(f"{lab}({'+' if v0 > 0 else MINUS})")
        return out
    print("\nbeyond the plotted band at both seeds, same sign:")
    for c, *_ in COLS:
        band = (BAND_FIXED[c],) * 2 if c in BAND_FIXED else (BAND_FACTOR * sd[0][c], BAND_FACTOR * sd[101][c])
        print(f"  {c}: band {tuple(round(b, 3) for b in band)} ->", beyond(c, band))
    for c in STREAM_COLS:
        print(f"  {c} at +/-1 SD (not plotted):", beyond(c, (sd[0][c], sd[101][c])))
    print("  no-dip rows (c):", list(NO_DIP))

    # stream-luck check: do the ablation readouts share the archived stream's deviation? (common random numbers)
    print("\nstream-luck check: mean change of the near-null rows", NEAR_NULL)
    for c in STREAM_COLS:
        for s in SEEDS:
            a = np.mean([D[(lab, s, c)] for lab in NEAR_NULL])
            m = np.mean([P[(lab, s, c)] for lab in NEAR_NULL])
            print(f"  {c} s{s}: vs archived {a:+.3f}, vs 4-stream mean {m:+.3f}; "
                  f"-(stream0 - mean) = {-(st[s][c][0] - mean[s][c]):+.3f}")

    # capacity training budget
    print("\nepochs run per fold (limit %d):" % EPOCH_LIMIT)
    for lab, keys in (("M6", M6), ("64 units", ("34g", "34h")), ("256 units", ("34i", "34j"))):
        e = [epochs(k) for k in keys]
        print(f"  {lab}: s0 {e[0]} | s101 {e[1]}; folds at the limit "
              f"{sum(x >= EPOCH_LIMIT for x in e[0])} | {sum(x >= EPOCH_LIMIT for x in e[1])}")


if __name__ == "__main__":
    main()
