"""Figure fig:metrics (Chapter 5, section 5.6 'What each metric can see'): the three yardsticks disagree.

Every scored model of Chapters 4-6 as one point per seed (seed 0 filled, seed 101 open; never pooled):
  a  held-out joint likelihood (bits per bout, lower = better) against the matched C2ST AUC (lower = better);
  b  matched AUC against the moving-dot profile error (mean |sim - real| over 2-8 moving dots, in real SEM).
Plotted: main line M1-M6, checks M5+R / M5+T, drift branch D1-D4 and F, the 12 ablations of M6 (incl. z frozen).
Not plotted (switches below): M6+C (a two-group-scenes model, seed 0 only, it lies under M6 s0 in both panels) and the
LOCO / 64- / 256-unit variants (not introduced in Chapters 4-6 yet).
Arrows mark the informative comparisons, one per seed (solid seed 0, dashed seed 101), coloured by the base model's
family (blue main line / M6, vermillion drift branch):
  M2 -> M3 (latent), M6 -> M6-stim, M6 -> M6-kin0, M6 -> M6 with z frozen, D3 -> D4 (drift), D4 -> F (correction).
M6 and F are ringed. The likelihood axis of (a) is broken; the -kin0 arrows cross the break (marked on the arrows and
labelled with their true likelihood change).

Data: Model/report_outputs/summary_M<key>.json (keys lik_joint, auc_matched, mv_err_sem), read only.
Caption noise numbers: Model/missing_pieces/streams.pkl via scripts/missing_pieces.stream_table() (moving error of M6
and F on 4 rollout streams). The plotted table is written to $RF/figs/fig_metrics_data.csv.

    uv run python scripts/report_figs/fig_metrics.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

RF = REPORT_FINAL_DIR
SUMMARY = MODEL / "report_outputs"

SHOW_M6C = False        # M6+C: two-group-scenes model, hidden under M6 s0 in both panels
SHOW_CAP_LOCO = False   # LOCO, 64 / 256 units: plot only once Chapters 4-6 define them

# public name -> (family, key seed 0, key seed 101)  [CONTEXT.md name map]
MODELS = {
    "M1": ("main", "1", "34a"), "M2": ("main", "2", "34b"), "M3": ("main", "3", "34c"),
    "M4": ("main", "20f", "30a"), "M5": ("main", "25d", "30b"), "M6": ("main", "26e", "30d"),
    "M5+R": ("check", "29a", "30c"), "M5+T": ("check", "29b", "34d"),
    "D1": ("drift", "18a", "24a"), "D2": ("drift", "24b", "24c"), "D3": ("drift", "16e", "30e"),
    "D4": ("drift", "11e", "32a"), "F": ("drift", "14f", "32b"),
    "M6+C": ("m6c", "28b", None),
    "M6-stim": ("abl", "29e", "33u"), "M6-pos": ("abl", "33a", "33b"), "M6-count": ("abl", "29c", "33s"),
    "M6-range": ("abl", "33c", "33d"), "M6-motion": ("abl", "29d", "33t"), "M6-turn4": ("abl", "33e", "33f"),
    "M6-turn1": ("abl", "33g", "33h"), "M6-turn0": ("abl", "33i", "33j"), "M6-kin1": ("abl", "33k", "33l"),
    "M6-kin0": ("abl", "33m", "33n"), "M6-clock": ("abl", "33o", "33p"), "M6 z frozen": ("abl", "33q", "33r"),
    "LOCO": ("cap", "34e", "34f"), "M6 64 units": ("cap", "34g", "34h"), "M6 256 units": ("cap", "34i", "34j"),
}
FAMILY = {  # label, colour, marker
    "main": ("M1\u2013M6", BLUE, "o"),
    "check": ("M5+R, M5+T", PURPLE, "D"),
    "drift": ("D1\u2013D4, F", VERM, "s"),
    "m6c": ("M6+C", SKY, "v"),
    "abl": ("ablations of M6", MUTED, "^"),
    "cap": ("LOCO, 64 / 256 units", GREEN, "p"),
}
SHOWN = [f for f in FAMILY if (f != "m6c" or SHOW_M6C) and (f != "cap" or SHOW_CAP_LOCO)]
HEADLINE = ("M6", "F")  # ringed and drawn last
MIN_ARROW_PT = 5.0      # shorter: omitted (only D4->F at seed 0 in panel a, 3 pt)
SHORT_ARROW_PT = 14.0   # shorter: drawn centre to centre with almost no shrink
MSIZE = {"o": 4.2, "D": 3.4, "s": 3.8, "v": 4.2, "^": 4.2, "p": 4.6}
RING_MS = 7.6

# comparisons drawn as arrows: (from, to, label, colour of the base family)
ARROWS = [("M2", "M3", "latent (M2\u2192M3)", BLUE), ("M6", "M6-stim", "\u2212stim", BLUE),
          ("M6", "M6-kin0", "\u2212kin0", BLUE), ("M6", "M6 z frozen", "$z$ frozen", BLUE),
          ("D3", "D4", "drift (D3\u2192D4)", VERM), ("D4", "F", "correction (D4\u2192F)", VERM)]
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


def load():
    rows = []
    for name, (fam, k0, k1) in MODELS.items():
        for seed, key in ((0, k0), (101, k1)):
            if key is None:
                continue
            d = json.loads((SUMMARY / f"summary_M{key}.json").read_text())
            rows.append(dict(model=name, family=fam, seed=seed, key=key, lik_joint=d["lik_joint"],
                             auc_matched=d["auc_matched"], mv_err_sem=d["mv_err_sem"], plotted=fam in SHOWN))
    return pd.DataFrame(rows)


def stream_noise():
    """Moving-profile error of M6 and F on the 4 rollout streams (stream 0 = the archived one plotted here)."""
    import missing_pieces as mp
    t = mp.stream_table()
    t = t[t.model != "real"]
    g = t.groupby("model").mv_err_sem
    out = pd.DataFrame(dict(stream0=t[t.stream == 0].set_index("model").mv_err_sem, lo=g.min(), hi=g.max(),
                            sd=g.std(ddof=1)))
    out["range"] = out.hi - out.lo
    return out


# ---- broken likelihood axis: same scale on both sides, the empty stretch A..C removed --------------
A_BRK, C_BRK, GAP = 13.468, 13.655, 0.010


def _fwd(x):
    x = np.asarray(x, float)
    return np.where(x <= A_BRK, x, np.where(x >= C_BRK, x - (C_BRK - A_BRK) + GAP,
                                            A_BRK + (x - A_BRK) * GAP / (C_BRK - A_BRK)))


def _inv(y):
    y = np.asarray(y, float)
    return np.where(y <= A_BRK, y, np.where(y >= A_BRK + GAP, y + (C_BRK - A_BRK) - GAP,
                                            A_BRK + (y - A_BRK) * (C_BRK - A_BRK) / GAP))


def _slash(ax, xa, ya, hx=0.0045, hy=0.028):
    """A short break slash centred at axes coordinates (xa, ya)."""
    ax.plot([xa - hx, xa + hx], [ya - hy, ya + hy], transform=ax.transAxes, color=MUTED, lw=0.6,
            clip_on=False, zorder=6)


def _break_marks(ax):
    """Two short slashes over a white gap on the bottom spine at the axis break."""
    tr = ax.get_xaxis_transform()
    ax.plot([A_BRK + 0.001, C_BRK - 0.001], [0, 0], transform=tr, color="white", lw=2.2, clip_on=False,
            zorder=5, solid_capstyle="butt")
    for x0 in (A_BRK, C_BRK):
        xa = ax.transAxes.inverted().transform(ax.transData.transform((x0, 0)))[0]
        _slash(ax, xa, 0.0)


def _arrow_break(ax, p, q):
    """Mark where a straight arrow p -> q (data) crosses the axis break: white gap + two slashes on the shaft."""
    P, Q = ax.transAxes.inverted().transform(ax.transData.transform([p, q]))
    pts = []
    for x0 in (A_BRK, C_BRK):
        xa = ax.transAxes.inverted().transform(ax.transData.transform((x0, p[1])))[0]
        t = (xa - P[0]) / (Q[0] - P[0])
        pts.append((xa, P[1] + t * (Q[1] - P[1])))
    (x0, y0), (x1, y1) = pts
    ax.plot([x0, x1], [y0, y1], transform=ax.transAxes, color="white", lw=2.4, zorder=4.5,
            solid_capstyle="butt")
    for xa, ya in pts:
        _slash(ax, xa, ya, hy=0.022)


def _on_arrow(ax, p, q, rad=0.0, t=0.5):
    """Data coordinates of the point at parameter t on the (possibly arc3-curved) arrow p -> q."""
    P, Q = ax.transData.transform([p, q])
    d = Q - P
    C = (P + Q) / 2 + rad * np.array([d[1], -d[0]])   # matplotlib's arc3 control point
    B = (1 - t) ** 2 * P + 2 * t * (1 - t) * C + t ** 2 * Q
    return tuple(ax.transData.inverted().transform(B))


def _rad(sp, seed):
    r = sp.get("rad", 0.0)
    return r.get(seed, 0.0) if isinstance(r, dict) else r


def draw(ax, df, xcol, ycol, spec):
    """Points (headline models last, ringed), arrows and labels. spec: per-label options for this panel."""
    idx = df.set_index(["model", "seed"])
    for (fam, seed), g in df[~df.model.isin(HEADLINE)].groupby(["family", "seed"]):
        lab, col, mk = FAMILY[fam]
        ax.plot(g[xcol], g[ycol], ls="none", marker=mk, ms=MSIZE[mk], mew=0.8, color=col,
                mfc=col if seed == 0 else "none", alpha=0.95, zorder=3)
    for name in HEADLINE:
        fam = MODELS[name][0]
        _, col, mk = FAMILY[fam]
        for seed in (0, 101):
            r = idx.loc[(name, seed)]
            ax.plot(r[xcol], r[ycol], ls="none", marker="o", ms=RING_MS, mew=0.55, mec=INK, mfc="none",
                    zorder=3.4)
            ax.plot(r[xcol], r[ycol], ls="none", marker=mk, ms=MSIZE[mk], mew=0.8, color=col,
                    mfc=col if seed == 0 else "white", zorder=3.5)
    lengths = {}
    for a, b, lab, col in ARROWS:
        sp = spec.get(lab, {})
        for seed in (0, 101):
            p, q = idx.loc[(a, seed)], idx.loc[(b, seed)]
            P, Q = (p[xcol], p[ycol]), (q[xcol], q[ycol])
            d = ax.transData.transform([P, Q])
            length_pt = float(np.hypot(*(d[1] - d[0]))) * 72.0 / ax.figure.dpi
            lengths[(lab, seed)] = length_pt
            print(f"    {ax.get_label()} {a}->{b} s{seed}: {length_pt:.1f} pt")
            if length_pt < MIN_ARROW_PT:
                continue
            shr = 1.0 if length_pt < SHORT_ARROW_PT else 2.5
            nud = sp.get("nudge", {}).get(seed, 0.0)  # horizontal offset of the shaft (points), legibility only
            tr = mtrans.offset_copy(ax.transData, fig=ax.figure, x=nud, y=0, units="points") if nud else ax.transData
            rad = _rad(sp, seed)
            ax.annotate("", xy=Q, xytext=P, xycoords=tr, textcoords=tr, zorder=4,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.75,
                                        color=col, ls=SEED_LS[seed], shrinkA=shr, shrinkB=shr, mutation_scale=7,
                                        connectionstyle=f"arc3,rad={rad}"))
            if sp.get("break"):
                _arrow_break(ax, P, Q)
        if "xy" not in sp:
            continue
        kw = dict(fontsize=7, color=col, ha=sp.get("ha", "center"), va="center", zorder=6, path_effects=HALO,
                  linespacing=1.05)
        text = sp.get("label", lab)
        targets = sp.get("to", [])
        if not targets:
            ax.text(*sp["xy"], text, **kw)
        for k, tg in enumerate(targets):
            kind, seed = tg[0], tg[1]
            if kind == "at":        # point at fraction t along this comparison's arrow at that seed
                p, q = idx.loc[(a, seed)], idx.loc[(b, seed)]
                xy = _on_arrow(ax, (p[xcol], p[ycol]), (q[xcol], q[ycol]), _rad(sp, seed), tg[2])
            else:                   # a model's point
                r = idx.loc[(kind, seed)]
                xy = (r[xcol], r[ycol])
            ax.annotate(text if k == 0 else "", xy=xy, xytext=sp["xy"], **kw,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=1.5, shrinkB=2.5))
    return lengths


def name_label(ax, df, xcol, ycol, name, xy, col, targets=(0, 101), **kw):
    """A model name with thin leaders to its points at the given seeds."""
    idx = df.set_index(["model", "seed"])
    for k, seed in enumerate(targets):
        r = idx.loc[(name, seed)]
        ax.annotate(name if k == 0 else "", xy=(r[xcol], r[ycol]), xytext=xy, fontsize=7, color=col, ha="center",
                    va="center", zorder=6, path_effects=HALO,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=1.5, shrinkB=4.0), **kw)


def main():
    df_all = load()
    figs = RF / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(figs / "fig_metrics_data.csv", index=False)
    df = df_all[df_all.plotted].copy()
    idx = df.set_index(["model", "seed"])
    dk = {s: idx.loc[("M6-kin0", s), "lik_joint"] - idx.loc[("M6", s), "lik_joint"] for s in (0, 101)}
    noise = stream_noise()
    print("  moving error over 4 rollout streams:\n" + noise.round(3).to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.6))
    ax1.set_label("a"); ax2.set_label("b")
    fig.subplots_adjust(left=0.08, right=0.993, bottom=0.165, top=0.86, wspace=0.27)

    # ---- a: likelihood vs matched AUC ----------------------------------------------------------
    ax1.set_xscale("function", functions=(_fwd, _inv))
    ax1.set_xlim(13.318, 13.718)
    ax1.set_ylim(0.585, 0.900)
    ax1.set_xticks([13.35, 13.40, 13.45, 13.70])
    ax1.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    ax1.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    kin0_lab = f"\u2212kin0\n+{dk[0]:.2f} | +{dk[101]:.2f} bits"
    spec_a = {
        "latent (M2→M3)": dict(xy=(13.338, 0.785), label="latent\n(M2→M3)", to=[("at", 101, 0.7)]),
        "−stim": dict(xy=(13.452, 0.800), to=[("M6-stim", 0)]),
        "−kin0": dict(xy=(13.716, 0.718), ha="right", label=kin0_lab, **{"break": True}),
        "$z$ frozen": dict(xy=(13.380, 0.842), ha="right", nudge={0: -1.0, 101: 1.0}),
        "drift (D3→D4)": dict(xy=(13.436, 0.606), rad={0: 0.25}, to=[("at", 0, 0.15), ("at", 101, 0.35)]),
        "correction (D4→F)": dict(xy=(13.322, 0.597), ha="left", to=[("at", 101, 0.5)]),
    }
    draw(ax1, df, "lik_joint", "auc_matched", spec_a)
    _break_marks(ax1)
    ax1.set_xlabel("Joint likelihood (bits per bout; lower = better)")
    ax1.set_ylabel("Matched AUC")
    panel(ax1, "a")

    # ---- b: matched AUC vs moving profile error --------------------------------------------------
    ax2.set_yscale("log")
    ax2.set_xlim(0.595, 0.895)
    ax2.set_ylim(0.64, 10.5)
    ax2.set_yticks([0.7, 1, 2, 4, 8])
    ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    spec_b = {
        "latent (M2→M3)": dict(xy=(0.77, 2.95)),
        "−stim": dict(xy=(0.695, 4.2)),
        "−kin0": dict(xy=(0.80, 1.42)),
        "$z$ frozen": dict(xy=(0.855, 1.25), rad={101: 0.08}, to=[("M6 z frozen", 0)]),
        "drift (D3→D4)": dict(xy=(0.600, 1.75), ha="left", to=[("D4", 101)]),
        "correction (D4→F)": dict(xy=(0.600, 0.70), ha="left", to=[("F", 0)]),
    }
    draw(ax2, df, "auc_matched", "mv_err_sem", spec_b)
    name_label(ax2, df, "auc_matched", "mv_err_sem", "M6", (0.705, 0.80), BLUE, targets=(0,))
    name_label(ax2, df, "auc_matched", "mv_err_sem", "M6", (0.614, 0.88), BLUE, targets=(101,))
    r = idx.loc[("M6-pos", 0)]
    ax2.annotate("\u2212pos", xy=(r.auc_matched, r.mv_err_sem), xytext=(0.805, 6.3), fontsize=7, color=MUTED,
                 ha="center", va="center", zorder=6, path_effects=HALO,
                 arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=1.5, shrinkB=2.5))
    ax2.set_xlabel("Matched AUC")
    ax2.set_ylabel("Moving profile error (real SEM)")
    panel(ax2, "b")

    # ---- legend: families (filled = seed 0) + headline ring + seed key -------------------------------
    hs = [Line2D([], [], ls="none", marker=FAMILY[f][2], ms=MSIZE[FAMILY[f][2]], mew=0.8, color=FAMILY[f][1],
                 label=FAMILY[f][0]) for f in SHOWN]
    hs += [Line2D([], [], ls="none", marker="o", ms=RING_MS, mew=0.55, mec=INK, mfc="none", label="M6, F"),
           Line2D([], [], color=INK, ls="-", lw=0.75, marker="o", ms=3.8, mew=0.8, mfc=INK, label="seed 0"),
           Line2D([], [], color=INK, ls="--", lw=0.75, marker="o", ms=3.8, mew=0.8, mfc="none",
                  label="seed 101")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(hs), handlelength=1.5,
               columnspacing=1.0, handletextpad=0.35, borderaxespad=0.1)

    save(fig, "fig_metrics")


if __name__ == "__main__":
    main()
