"""fig_steps (Chapter 5, sec:r-building, label fig:steps): what each main-line step buys, at both seeds.

One row of four dot-plot panels sharing the step axis: change in matched AUC, joint held-out likelihood
(bits per bout), dip (mm at Chapter 3's physical 3.60 px/mm) and moving-profile error (real SEM), for the main-line
steps M1->M2 ... M5->M6 and the checks M5->M5+R, M5->M5+T. Each point is the later model minus the earlier
one, within a seed. Seed 0 filled, seed 101 open; never pooled.

Scale: the archived dips (summary files, stream table) are in mm at the profile code's nominal 3.67 px/mm; the dip
change, its band and the M6 stream whisker are multiplied by style.MM_3P67_TO_3P60 (= 3.67 / 3.60) at plotting
time, after the archive checks (which compare unconverted values). The claim-rule outcome is scale-free.

Grey bands = the noise of a DIFFERENCE between two units, on one footing in every panel:
  * (a) the claim bar of sec:m-stats, +/-0.05 (a little wider than sqrt(2) x the 0.028 matched-AUC seed SD = 0.040);
  * (b) sqrt(2) x the 0.016-bit seed SD of the likelihood (sec:m-stats);
  * (c, d) sqrt(2) x M6's rollout-stream SD (4 streams, ddof 1; the larger of the two seeds' SDs).
    Stream noise only: retraining noise is not in (c, d).
On the M5->M6 row of (c, d) a thin capped whisker spans M6's four rollout streams minus M5's archived value
(stream 0 = the archived M6 value = the marker).

Sources (read only):
  * Model/report_outputs/summary_M<key>.json  (lik_joint, auc_matched, dip, mv_err_sem) for every unit.
  * Model/missing_pieces/streams.pkl via missing_pieces.stream_table(): M6's dip and moving error on 4 streams
    per seed (rows "M6 s0", "M6 s101"); the real dip (rows "real").
At seed 0 the M1-M3 profiles were rolled out on an earlier stream; the M3->M4 step at seed 0 therefore uses
the M4 unit re-rolled on that stream (key "4", same networks and likelihood/AUC as key "20f").

    uv run python scripts/report_figs/fig_steps.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# (label, kind, (from, to) at seed 0, (from, to) at seed 101)
STEPS = [
    ("M1→M2  wall rejection", "main", ("1", "2"), ("34a", "34b")),
    ("M2→M3  per-fish latent", "main", ("2", "3"), ("34b", "34c")),
    ("M3→M4  count, motion", "main", ("3", "4"), ("34c", "30a")),     # s0: both on the earlier stream
    ("M4→M5  ×3 turn weight", "main", ("20f", "25d"), ("30a", "30b")),
    ("M5→M6  range + temporal", "main", ("25d", "26e"), ("30b", "30d")),
    ("M5→M5+R  range only", "check", ("25d", "29a"), ("30b", "30c")),
    ("M5→M5+T  temporal only", "check", ("25d", "29b"), ("30b", "34d")),
]
I_M6 = 4                                   # row of the M5->M6 step (gets the stream whisker in c, d)
M6_STREAM_ROW = {0: "M6 s0", 101: "M6 s101"}
METRICS = [  # (column, panel title, x label, better direction, arrow text)
    ("auc_matched", "Matched AUC", "Δ AUC", -1, "← better"),
    ("lik_joint", "Likelihood", "Δ bits per bout", -1, "← better"),
    ("dip", "Dip", "Δ dip (mm)", +1, "better →"),        # toward the real dip (all units below it)
    ("mv_err_sem", "Moving error", "Δ error (SEM)", -1, "← better"),
]
KIND_C = {"main": BLUE, "check": MUTED}
MM_COLS = ("dip",)        # millimetre readouts: converted 3.67 -> 3.60 px/mm for plotting and printing
AUC_BAR = 0.05            # claim bar for the matched AUC (sec:m-stats)
AUC_SEED_SD = 0.028       # matched-AUC seed SD (sec:m-stats), for the printed comparison only
LIK_SEED_SD = 0.016       # likelihood seed SD in bits per bout (sec:m-stats)
ANNOT_FS = 7


def summary(key):
    with open(MODEL / "report_outputs" / f"summary_M{key}.json") as f:
        return json.load(f)


def deltas():
    """{(step index, seed, column): to - from} straight from the summary files."""
    out = {}
    for i, (_, _, s0, s1) in enumerate(STEPS):
        for seed, (a, b) in ((0, s0), (101, s1)):
            A, B = summary(a), summary(b)
            for col, *_ in METRICS:
                out[(i, seed, col)] = float(B[col]) - float(A[col])
    return out


def streams():
    """M6's dip and moving error on its 4 rollout streams per seed (stream 0 = archived) and the real dip."""
    import missing_pieces as mp
    t = mp.stream_table()
    m6 = t[t.model.isin(M6_STREAM_ROW.values())].copy()
    m6["stream"] = m6["stream"].astype(int)
    real_dip = float(t.loc[t.model == "real", "dip"].iloc[0])
    return m6, real_dip


def noise(m6):
    """Half-width of the grey band per readout: the noise of a difference between two units."""
    sd = m6.groupby("model")[["dip", "mv_err_sem"]].std()      # 4 streams, ddof = 1, per seed
    r2 = np.sqrt(2.0)
    return {"auc_matched": AUC_BAR,
            "lik_joint": r2 * LIK_SEED_SD,
            "dip": r2 * float(sd["dip"].max()),
            "mv_err_sem": r2 * float(sd["mv_err_sem"].max())}, sd


def m6_stream_changes(m6):
    """{(seed, col): array over M6's 4 streams of (M6 stream value - M5 archived value)}."""
    out = {}
    for seed, (a, b) in ((0, STEPS[I_M6][2]), (101, STEPS[I_M6][3])):
        M5, M6 = summary(a), summary(b)
        rows = m6[m6.model == M6_STREAM_ROW[seed]].sort_values("stream")
        assert list(rows.stream) == [0, 1, 2, 3]
        for col in ("dip", "mv_err_sem"):
            v = rows[col].to_numpy()
            # stream 0 is the archived stream: it must reproduce the summary value (rounded to 3 decimals there)
            assert abs(v[0] - float(M6[col])) < 1e-3, (seed, col, v[0], M6[col])
            out[(seed, col)] = v - float(M5[col])
    return out


def main():
    d = deltas()
    m6, real_dip = streams()
    band, sd = noise(m6)
    sw = m6_stream_changes(m6)
    n = len(STEPS)
    y = np.arange(n)[::-1].astype(float)
    y[-2:] -= 0.35                         # small gap before the two checks
    off = {0: 0.13, 101: -0.13}

    # 'higher dip is better' only because every plotted unit is below the real dip
    dips = [float(summary(k)["dip"]) for _, _, s0, s1 in STEPS for k in (*s0, *s1)]
    assert max(dips) < real_dip and float(m6["dip"].max()) < real_dip, (max(dips), real_dip)

    # ---- 3.67 -> 3.60 px/mm for every millimetre, AFTER the archive checks (stream 0 = summary, both at 3.67)
    import model_figures as MF
    assert_code_scale(MF.PX_TO_MM, "model_figures.PX_TO_MM")
    f = MM_3P67_TO_3P60
    d = {k: (v * f if k[2] in MM_COLS else v) for k, v in d.items()}
    band = {k: (v * f if k in MM_COLS else v) for k, v in band.items()}
    sw = {k: (v * f if k[1] in MM_COLS else v) for k, v in sw.items()}
    sd = sd.copy()
    sd[list(MM_COLS)] = sd[list(MM_COLS)] * f
    real_dip, dips = real_dip * f, [x * f for x in dips]

    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 2.3), sharey=True,
                             gridspec_kw=dict(wspace=0.28))
    for k, (ax, (col, title, xlab, better, arrow)) in enumerate(zip(axes, METRICS)):
        ax.axvspan(-band[col], band[col], color=FAINT, alpha=0.45, lw=0, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
        ax.axhline((y[n - 3] + y[n - 2]) / 2, color=FAINT, lw=0.6, zorder=1)
        lo, hi = -band[col], band[col]
        for i, (_, kind, _, _) in enumerate(STEPS):
            c = KIND_C[kind]
            v0, v1 = d[(i, 0, col)], d[(i, 101, col)]
            ax.plot([v0, v1], [y[i] + off[0], y[i] + off[101]], color=c, lw=0.6, alpha=0.5, zorder=2)
            lo, hi = min(lo, v0, v1), max(hi, v0, v1)
            if i == I_M6 and (0, col) in sw:          # M6's four rollout streams minus M5's archived value
                for seed in (0, 101):
                    v = sw[(seed, col)]
                    ax.plot([v.min(), v.max()], [y[i] + off[seed]] * 2, color=c, lw=0.8, zorder=2.5,
                            solid_capstyle="butt")
                    ax.plot([v.min(), v.max()], [y[i] + off[seed]] * 2, ls="none", marker="|", ms=4.0,
                            mew=0.8, color=c, zorder=2.5)
                    lo, hi = min(lo, v.min()), max(hi, v.max())
            for seed, v in ((0, v0), (101, v1)):
                ax.plot(v, y[i] + off[seed], "o", ms=4.2, mec=c, mew=1.0,
                        mfc=c if seed == 0 else "white", zorder=3)
        pad = 0.12 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel(xlab, labelpad=2)
        ax.text(0.98 if better > 0 else 0.02, 0.012, arrow, transform=ax.transAxes,
                ha="right" if better > 0 else "left", va="bottom", fontsize=ANNOT_FS, color=MUTED)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator({"lik_joint": 3, "dip": 5}.get(col, 4),
                                                            steps=[1, 2, 2.5, 4, 5, 10]))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-9 else f"{v:g}".replace("-", "−")))
        panel(ax, "abcd"[k], title)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([s[0] for s in STEPS])
    for t, (_, kind, _, _) in zip(axes[0].get_yticklabels(), STEPS):
        t.set_color(INK if kind == "main" else MUTED)
    axes[0].set_ylim(y.min() - 0.95, y.max() + 0.5)

    # M1 -> M2 changes generation only: its likelihood change is exactly zero at both seeds
    ax = axes[1]
    assert d[(0, 0, "lik_joint")] == 0.0 and d[(0, 101, "lik_joint")] == 0.0
    ax.text(0.012, y[0], "≡ 0", fontsize=ANNOT_FS, color=MUTED, va="center", ha="left")  # caption: generation only

    handles = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=BLUE, mec=BLUE, label="seed 0"),
               Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=BLUE, mew=1.0, label="seed 101"),
               Line2D([], [], ls="none", marker="o", ms=4.2, mfc=MUTED, mec=MUTED, label="check"),
               Line2D([], [], color=BLUE, lw=0.8, marker="|", ms=4.0, mew=0.8, markevery=[0, -1],
                      label="M6, 4 rollout streams"),
               Patch(fc=FAINT, alpha=0.45, label="noise of a difference")]
    fig.subplots_adjust(bottom=0.27, top=0.88)
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.4, columnspacing=1.2, handlelength=1.6)
    save(fig, "fig_steps")

    # numbers for the notes / prose (printed, never hand-typed)
    print(f"mm values (dip) at 3.60 px/mm (x {f:.6f}); AUC, bits and SEM unchanged")
    print("band half-widths:", {k: round(v, 4) for k, v in band.items()})
    print(f"  sqrt(2) x matched-AUC seed SD = {np.sqrt(2) * AUC_SEED_SD:.4f}  (the claim bar used is {AUC_BAR})")
    print("M6 stream SD per seed (single unit):\n", sd.round(4).to_string())
    print(f"real dip (stream_table 'real'): {real_dip:.4f}; largest plotted dip {max(dips):.3f}")
    for i, (lab, *_rest) in enumerate(STEPS):
        lab = lab.replace("→", "->").replace("×", "x")
        print(f"{lab:30s}", "  ".join(f"{c}: {d[(i, 0, c)]:+.4f} | {d[(i, 101, c)]:+.4f}" for c, *_ in METRICS))
    print("M5->M6 per M6 stream (M6 stream - M5 archived), streams 0..3, and their mean:")
    for (seed, col), v in sorted(sw.items()):
        print(f"  s{seed:<3d} {col:10s}", " ".join(f"{x:+.3f}" for x in v), f"  mean {v.mean():+.3f}",
              f"  range [{v.min():+.3f}, {v.max():+.3f}]")
    print("claim rule per readout (same sign at both seeds AND |change| > band at each):")
    for col, *_ in METRICS:
        hits = [STEPS[i][0].split()[0].replace("→", "->") for i in range(n)
                if np.sign(d[(i, 0, col)]) == np.sign(d[(i, 101, col)]) != 0
                and min(abs(d[(i, 0, col)]), abs(d[(i, 101, col)])) > band[col]]
        print(f"  {col:12s}", ", ".join(hits))


if __name__ == "__main__":
    main()
