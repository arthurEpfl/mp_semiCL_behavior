"""Figure fig_multifish (Chapter 5, sec:r-multifish): simulated fish that see each other as moving dots shoal.

(a) Pairs: distance between the two fish of a pair (1 Hz, readout window 5-30 min) for the fish that see each other and
    for the same pairs with the partner's track shifted in time (Larsch & Baier's chance); and the attraction index
    (mean over 50 pairs, 95 % bootstrap CI over pairs) for M6 and F at both seeds, with the same fish blind.
(b) Groups: D = ln(NN1 / NN1 of pseudo-groups of fish from different dishes, Harpaz et al.'s null), mean over dishes and
    95 % CI (groups resampled and pseudo-groups re-drawn), per model and seed where run (GROUP_CATS; the one run not
    shown is 5 fish under the nearest rule, F s0, D -0.285). Grey ticks: D against the two other chance references of
    ``multi_fish.groups_readout`` (fold-matched pseudo-groups; within-group time shift).
(c) A shoal in the 5x-radius dish (F seed 0, the dish where a shoal of >= 9 forms earliest, ``multi_fish.pick_formation``):
    positions and trails of the 10 fish, beside the same 10 fish blind to each other, at the same moment; inset = the
    shoal magnified.
(d) Neighbour density around a focal fish (heading up) over density with the neighbours' tracks time-shifted
    (``missing_pieces.neighbour_map``), 10 fish in the training dish, F seed 0; radial profile for all four units.

Sources (read only): Model/multi_fish/*.pkl (runs), Model/missing_pieces/multi_fish_summary.json (archived readouts:
pairs.*.ai_mean / ai_ci, groups.*.D_mean / D_ci / D_fold_mean / D_tshift_mean). Runs added for this figure (the
controls on M6 and the 5x-dish random-direction control on F s101; same simulator, same seeds) and their readouts:
$RF/figs/multi_fish_extra/*.pkl and summary_extra.json, made by $RF/figs/run_multifish_extra.py.
Distances in mm at 3.60 px/mm (multi_fish.PX_MM).
The heavy part (histograms, neighbour maps, formation pick) is cached in REPORT_FINAL_DIR (CACHE); missing keys are
computed on demand; delete the file to recompute everything.

    uv run python scripts/report_figs/fig_multifish.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import json  # noqa: E402
import pickle  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

RF = REPORT_FINAL_DIR
CACHE = RF / "figs" / "cache" / "fig_multifish.pkl"
SUMMARY = MODEL / "missing_pieces" / "multi_fish_summary.json"
SUMMARY_EXTRA = RF / "figs" / "multi_fish_extra" / "summary_extra.json"

UNITS = [("M6", 0), ("M6", 101), ("F", 0), ("F", 101)]
PAIRS = {("F", 0): "A1_pairs", ("F", 101): "s101_A1_pairs", ("M6", 0): "m6_A1_pairs", ("M6", 101): "m6s101_A1_pairs"}
MAPS = {("F", 0): "A1_N10_effects", ("F", 101): "s101_A1_N10_effects", ("M6", 0): "m6_A1_N10_effects",
        ("M6", 101): "m6s101_A1_N10_effects"}
SNAP_COND = "A5_N10_effects"            # F seed 0, 5x dish, 10 fish, effects rule
CENTRAL = ("A5_blind", "A5_N10_effects")  # how far from the centre the 5x-dish fish stay (caption of c)
# Panel (b): (tick label, dish, {unit: condition}); every condition run, nothing else.
GROUP_CATS = [
    ("5", 1, {("F", 0): "A1_N5_effects", ("F", 101): "s101_A1_N5_effects"}),
    ("10", 1, {("M6", 0): "m6_A1_N10_effects", ("M6", 101): "m6s101_A1_N10_effects",
               ("F", 0): "A1_N10_effects", ("F", 101): "s101_A1_N10_effects"}),
    ("10\nnearest", 1, {("F", 0): "A1_N10_nearest"}),
    ("10\nrandom\ndir.", 1, {("M6", 0): "m6_A1_N10_effects_scrambled", ("M6", 101): "m6s101_A1_N10_effects_scrambled",
                             ("F", 0): "A1_N10_effects_scrambled", ("F", 101): "s101_A1_N10_effects_scrambled"}),
    ("10", 5, {("M6", 0): "m6_A5_N10_effects", ("M6", 101): "m6s101_A5_N10_effects",
               ("F", 0): "A5_N10_effects", ("F", 101): "s101_A5_N10_effects"}),
    ("10\nnearest", 5, {("F", 0): "A5_N10_nearest"}),
    ("10\nrandom\ndir.", 5, {("M6", 0): "m6_A5_N10_effects_scrambled", ("M6", 101): "m6s101_A5_N10_effects_scrambled",
                             ("F", 0): "A5_N10_effects_scrambled", ("F", 101): "s101_A5_N10_effects_scrambled"}),
    ("10\nentry\nclock", 5, {("F", 0): "A5_N10_effects_entry"}),
]
DIST_BINS = np.arange(0.0, 100.0 + 1e-9, 2.5)      # mm


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _pair_job(cond):
    import multi_fish as MFish
    T, taus = MFish.tracks(MFish.load(cond), 2)
    G = MFish.CONDS[cond]["G"]
    d = np.linalg.norm(T[:G, 0] - T[:G, 1], axis=-1)                                   # (pairs, frames) mm
    ds = np.stack([np.linalg.norm(T[:G, 0] - np.roll(T[:G, 1], -int(s), axis=1), axis=-1) for s in MFish.SHIFTS_S])
    # recomputed attraction index (sanity check against the archived summary)
    ai = [MFish.pair_ai(T[g, 0], T[g, 1], taus, MFish.CONDS[cond]["t_read"], MFish.CONDS[cond]["T"])[0]
          for g in range(G)]
    return cond, dict(h=np.histogram(d.ravel(), DIST_BINS, density=True)[0],
                      h_shift=np.histogram(ds.ravel(), DIST_BINS, density=True)[0],
                      med=float(np.median(d)), med_shift=float(np.median(ds)),
                      iad_med=float(np.median(np.median(d, axis=1))), ai_mean=float(np.mean(ai)), n_pairs=G)


def _map_job(cond):
    import missing_pieces as MP
    ratio, edges, prof = MP.neighbour_map(cond)
    return cond, dict(ratio=ratio, edges=edges, prof_r=prof.index.to_numpy(float), prof=prof.to_numpy(float))


def _snap_job(cond):
    import multi_fish as MFish
    g, t0, t1, t9 = MFish.pick_formation(cond)
    out = dict(g=g, t0=t0, t1=t1, t9=t9, dish=MFish.dish(MFish.CONDS[cond]["arena"]),
               dish1=MFish.dish(1), px_mm=MFish.PX_MM)
    taus = np.arange(0.0, MFish.CONDS[cond]["T"], 1.0 / MFish.VIDEO_FPS)
    for lab, c in (("sees", cond), ("blind", MFish.blind_of(cond))):
        grp = MFish.load(c)["groups"][g][:MFish.CONDS[cond]["N"]]
        out[lab] = np.array([MFish.frames_xy(f, taus) for f in grp], np.float32)       # (fish, frames, 2) px
        big_t = np.arange(0.0, MFish.CONDS[cond]["T"], 1.0)
        P = np.array([MFish.pos_at(f, big_t) for f in grp]) / MFish.PX_MM
        out[lab + "_big"] = MFish._largest(P)                                          # 1 Hz largest shoal (25 mm)
        out[lab + "_nn"] = MFish._nn1(P)
    out["taus"] = taus
    return cond, out


def _central_job(cond):
    """Distance of every fish from the dish centre (mm), 1 Hz over the readout window, all dishes of ``cond``."""
    import multi_fish as MFish
    T, _ = MFish.tracks(MFish.load(cond))                                              # (G, N, frames, 2) mm
    cx, cy, _ = MFish.dish(MFish.CONDS[cond]["arena"])
    r = np.linalg.norm(T - np.array([cx, cy]) / MFish.PX_MM, axis=-1)
    return cond, dict(median=float(np.median(r)), p90=float(np.percentile(r, 90)), max=float(r.max()),
                      R_mm=float(MFish.dish(MFish.CONDS[cond]["arena"])[2] / MFish.PX_MM), n=int(r.size))


def compute(res=None):
    res = res or {"pairs": {}, "maps": {}, "snap": None, "central": {}}
    res.setdefault("central", {})
    jobs = ([(_pair_job, c) for c in PAIRS.values() if c not in res["pairs"]]
            + [(_map_job, c) for c in MAPS.values() if c not in res["maps"]]
            + ([(_snap_job, SNAP_COND)] if res["snap"] is None else [])
            + [(_central_job, c) for c in CENTRAL if c not in res["central"]])
    if not jobs:
        return res
    with ProcessPoolExecutor(min(8, len(jobs))) as ex:
        futs = [(fn.__name__, ex.submit(fn, c)) for fn, c in jobs]
        for name, f in futs:
            c, v = f.result()
            if name == "_pair_job":
                res["pairs"][c] = v
            elif name == "_map_job":
                res["maps"][c] = v
            elif name == "_central_job":
                res["central"][c] = v
            else:
                res["snap"] = v
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(res, fh)
    return res


def data():
    res = None
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            res = pickle.load(fh)
    return compute(res)


def summary():
    """Archived readouts, plus the readouts of the runs added for this figure (never overriding an archived one)."""
    S = json.loads(SUMMARY.read_text(encoding="utf-8"))
    S["_extra"] = set()
    if SUMMARY_EXTRA.exists():
        E = json.loads(SUMMARY_EXTRA.read_text(encoding="utf-8"))
        for c, v in E["groups"].items():
            if c not in S["groups"]:
                S["groups"][c] = v
                S["_extra"].add(c)
    return S


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def mark(ax, x, y, lo, hi, model, seed, ms=4.2, **kw):
    open_ = SEED_MFC[seed] == "none"          # open marker: white face so the CI line does not show through
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=MODEL_C[model], ms=ms, mew=0.9,
                mfc="white" if open_ else MODEL_C[model], elinewidth=0.9, capsize=0, zorder=3, **kw)


SNAP_T = 280.0          # s: 116 s after the group first holds >= 9 of 10 fish together for 30 s (t9 = 164 s)
TRAIL_S = 15.0          # s of trail behind each fish
FIG_H = 3.6            # in
UNIT_LAB = {("M6", 0): "M6 s0", ("M6", 101): "M6 s101", ("F", 0): "F s0", ("F", 101): "F s101"}
NULL_C = "#8c8c8c"      # grey ticks: D against the other two chance references


def ax_in(fig, x, y, w, h):
    """Axes from a rectangle in inches (origin bottom left)."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def letter_at(fig, x, y, s):
    """Panel letter with its baseline-bottom at (x, y) inches."""
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, s, fontsize=9, fontweight="bold", va="bottom", ha="left")


def panel_pairs(ax_d, ax_ai, S, R):
    # distance distribution, one curve per unit; partner time-shifted (Larsch and Baier's chance) in grey
    x = (DIST_BINS[1:] + DIST_BINS[:-1]) / 2
    for u in UNITS:
        v = R["pairs"][PAIRS[u]]
        ax_d.plot(x, v["h_shift"], color=MUTED, ls=SEED_LS[u[1]], lw=0.8, zorder=2)
        ax_d.plot(x, v["h"], color=MODEL_C[u[0]], ls=SEED_LS[u[1]], lw=1.0, zorder=3)
    ax_d.set_xlim(0, 100); ax_d.set_ylim(0, 0.05); ax_d.set_yticks([0, 0.02, 0.04])
    ax_d.set_xlabel("pair distance (mm)", labelpad=2)
    ax_d.set_ylabel("density (1/mm)", labelpad=2)
    ax_d.text(21, 0.049, "see each other", fontsize=7, color=INK, va="top")
    ax_d.text(35, 0.0275, "partner\ntime-shifted", fontsize=7, color=MUTED, va="bottom", linespacing=1.0)
    ax_d.grid(axis="y"); ax_d.set_axisbelow(True)

    # attraction index, 95 % CI over the 50 pairs (archived summary); grey squares = the same fish blind
    xs = {("M6", 0): -0.17, ("M6", 101): 0.17, ("F", 0): 0.83, ("F", 101): 1.17}
    for u in UNITS:
        p = S["pairs"][PAIRS[u]]
        c, b = p["coupled"], p["blind"]
        mark(ax_ai, xs[u], c["ai_mean"], *c["ai_ci"], *u)
        ax_ai.errorbar(xs[u], b["ai_mean"], yerr=[[b["ai_mean"] - b["ai_ci"][0]], [b["ai_ci"][1] - b["ai_mean"]]],
                       fmt="s", color=MUTED, ms=3.0, mew=0.8, mfc="white" if u[1] == 101 else MUTED, elinewidth=0.8,
                       zorder=3)
    ax_ai.axhline(0, color=INK, lw=0.6)
    ax_ai.set_xticks([0, 1]); ax_ai.set_xticklabels(["M6", "F"])
    ax_ai.tick_params(axis="x", length=0)
    ax_ai.set_xlim(-0.5, 1.5); ax_ai.set_ylim(-0.04, 0.4); ax_ai.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_ai.set_ylabel("attraction index", labelpad=2)
    ax_ai.text(0.5, 0.03, "blind", fontsize=7, color=MUTED, ha="center", va="bottom")
    ax_ai.grid(axis="y"); ax_ai.set_axisbelow(True)


def panel_groups(ax, S):
    xt, xl, lows = [], [], []
    for k, (lab, arena, units) in enumerate(GROUP_CATS):
        us = [u for u in UNITS if u in units and units[u] in S["groups"]]
        off = (np.arange(len(us)) - (len(us) - 1) / 2) * 0.2
        for o, u in zip(off, us):
            d = S["groups"][units[u]]
            # the two other chance references, as grey ticks behind the marker (hidden where they agree with D)
            ax.plot([k + o] * 2, [d["D_fold_mean"], d["D_tshift_mean"]], ls="none", marker="_", ms=5.5, mew=1.0,
                    color=NULL_C, zorder=2.5)
            mark(ax, k + o, d["D_mean"], *d["D_ci"], *u, ms=3.6)
            lows.append(d["D_ci"][0])
        xt.append(k); xl.append(lab)
    n1 = sum(1 for c in GROUP_CATS if c[1] == 1)
    ax.axvline(n1 - 0.5, color=FAINT, lw=0.8)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.8, linespacing=1.0, va="top")
    ax.tick_params(axis="x", length=0, pad=2.5)
    for t in ax.get_xticklabels():          # nudge neighbouring two-word labels apart (|shift| <= 3 pt)
        dx = -3.0 if "nearest" in t.get_text() else 3.0 if "entry" in t.get_text() else 0.0
        t.set_transform(t.get_transform() + ScaledTranslation(dx / 72, 0, ax.figure.dpi_scale_trans))
    y0 = min(-1.0, np.floor((min(lows) - 0.04) * 20) / 20)
    ax.set_xlim(-0.55, len(GROUP_CATS) - 0.45); ax.set_ylim(y0, 0.08)
    ax.set_yticks([v for v in (-1.0, -0.75, -0.5, -0.25, 0) if v >= y0])
    ax.set_ylabel(r"$D = \ln(\mathrm{NN} / \mathrm{NN}_0)$", labelpad=2)
    ax.set_xlabel("fish per dish; rule or control", labelpad=3)
    for x, lab in (((n1 - 1) / 2, "training dish"), (n1 + (len(GROUP_CATS) - n1 - 1) / 2, "dish 5× wider")):
        ax.text(x, y0 + 0.02, lab, ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.grid(axis="y"); ax.set_axisbelow(True)


def _trails(ax, X, col, s=8, lw=0.6):
    for i in range(X.shape[0]):
        ax.plot(X[i, :, 0], X[i, :, 1], color=col, lw=lw, alpha=0.6, solid_capstyle="round")
    ax.scatter(X[:, -1, 0], X[:, -1, 1], s=s, color=col, edgecolors="white", linewidths=0.3, zorder=3)


def _frame(ax, color=FAINT, lw=0.6):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(True); ax.spines[sp].set_color(color); ax.spines[sp].set_linewidth(lw)


def panel_snapshot(axs, ax_z, sn):
    cx, cy, Rp = sn["dish"]
    px = sn["px_mm"]
    k1 = int(round(SNAP_T * 30)); k0 = int(round((SNAP_T - TRAIL_S) * 30))
    P = {lab: (sn[lab][:, k0:k1 + 1] - np.array([cx, cy], np.float32)) / px for lab in ("sees", "blind")}
    ext = max(np.abs(P["sees"][:, -1]).max(), np.abs(P["blind"][:, -1]).max())   # current positions (trails may clip)
    h = 10.0 * np.ceil((ext + 20.0) / 10.0)                    # view centred on the dish centre, both panels
    for ax, lab, col, ttl in ((axs[0], "sees", VERM, "see each other"), (axs[1], "blind", MUTED, "same fish, blind")):
        _trails(ax, P[lab], col)
        ax.plot([0], [0], marker="+", ms=5, color=INK, mew=0.7, zorder=4)              # dish centre
        ax.set_xlim(-h, h); ax.set_ylim(h, -h)
        _frame(ax)
        ax.set_title(ttl, fontsize=7, fontweight="normal", color=INK, pad=2)
        big = int(sn[lab + "_big"][int(SNAP_T)])
        ax.text(0.04, 0.96, f"largest shoal {big}/10", transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    ax = axs[1]
    x1, y0 = h - 0.08 * h, h - 0.12 * h
    ax.plot([x1 - 50, x1], [y0, y0], color=INK, lw=1.2, solid_capstyle="butt")
    ax.text(x1 - 25, y0 - 0.05 * h, "50 mm", ha="center", va="bottom", fontsize=6.8)
    # inset: the shoal magnified (a square about the bounding box of the current positions, >= 4 mm margin),
    # its footprint marked in the main view
    cur = P["sees"][:, -1]
    zc = (cur.min(0) + cur.max(0)) / 2
    side = 5.0 * np.ceil((np.ptp(cur, axis=0).max() + 8.0) / 5.0)
    z0 = zc - side / 2
    axs[0].add_patch(Rectangle(z0, side, side, fill=False, ec=INK, lw=0.5, zorder=5))
    _trails(ax_z, P["sees"], VERM, s=12, lw=0.7)
    ax_z.set_xlim(z0[0], z0[0] + side); ax_z.set_ylim(z0[1] + side, z0[1])
    _frame(ax_z, INK, 0.5)
    ax_z.set_facecolor("white")
    ax_z.set_zorder(5)
    xb, yb = z0[0] + side - 0.07 * side - 10, z0[1] + 0.92 * side      # 10 mm bar, lower right
    ax_z.plot([xb, xb + 10], [yb, yb], color=INK, lw=1.0, solid_capstyle="butt", zorder=4)
    ax_z.text(xb + 5, yb - 0.04 * side, "10 mm", ha="center", va="bottom", fontsize=6.3, zorder=4,
              bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
    # what the inset hides of the main view (must be nothing but empty space)
    inv = axs[0].transData.inverted()
    bb = ax_z.get_position()
    (xa, ya), (xb_, yb_) = inv.transform(axs[0].figure.transFigure.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]]))
    pts = P["sees"].reshape(-1, 2)
    hid = ((pts[:, 0] >= min(xa, xb_)) & (pts[:, 0] <= max(xa, xb_)) & (pts[:, 1] >= min(ya, yb_))
           & (pts[:, 1] <= max(ya, yb_))).sum()
    print(f"inset covers x {min(xa, xb_):.0f}..{max(xa, xb_):.0f}, y {min(ya, yb_):.0f}..{max(ya, yb_):.0f} mm of "
          f"the main view; trail points hidden under it: {hid}")
    return h, zc, side


def panel_map(ax_m, cax, ax_p, R):
    m = R["maps"][MAPS[("F", 0)]]
    e = m["edges"]
    im = ax_m.imshow(m["ratio"], origin="lower", extent=[e[0], e[-1], e[0], e[-1]], cmap="magma_r", vmin=0.25,
                     vmax=1.75, interpolation="nearest")
    ctr = (e[1:] + e[:-1]) / 2
    ax_m.contour(ctr, ctr, m["ratio"], levels=[1.0], colors="white", linewidths=0.7, linestyles="--")   # chance
    ax_m.plot([0], [0], marker=(3, 0, 0), ms=5, color="white", mew=0)       # the focal fish, heading up
    ax_m.set_xlabel("lateral (mm)", labelpad=2); ax_m.set_ylabel("ahead (mm)", labelpad=1)
    ax_m.set_xticks([-40, 0, 40]); ax_m.set_yticks([-40, 0, 40])
    ax_m.set_title("F s0, 10 fish", fontsize=7, fontweight="normal", color=MUTED, pad=2)
    cb = plt.colorbar(im, cax=cax, ticks=[0.5, 1.0, 1.5])
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.axhline(1.0, color="white", lw=0.7, ls="--")
    cb.set_label("density / chance", fontsize=7, labelpad=2)
    for u in UNITS:
        v = R["maps"][MAPS[u]]
        ax_p.plot(v["prof_r"], v["prof"], color=MODEL_C[u[0]], ls=SEED_LS[u[1]], lw=1.0)
    ax_p.axhline(1, color=INK, lw=0.6)
    ax_p.set_xlim(0, 40); ax_p.set_ylim(0, 1.7); ax_p.set_xticks([0, 10, 20, 30, 40])
    ax_p.set_xlabel("distance to a\nneighbour (mm)", labelpad=2)
    ax_p.set_ylabel("density / chance", labelpad=2)
    ax_p.set_title("M6 and F, both seeds", fontsize=7, fontweight="normal", color=MUTED, pad=2, loc="right")
    ax_p.grid(axis="y"); ax_p.set_axisbelow(True)


def main():
    S = summary()
    R = data()
    sn = R["snap"]
    print("snapshot pick:", {k: sn[k] for k in ("g", "t0", "t1", "t9")})
    print("added runs plotted:", sorted(S["_extra"]))
    missing = [c for _, _, us in GROUP_CATS for c in us.values() if c not in S["groups"]]
    if missing:
        print("NOT YET RUN (left out of b):", missing)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, FIG_H))
    # rectangles in inches; the tight bbox (labels included) stays within 6.3 x 3.6 in
    yt, ht = 2.36, 0.92                       # top row
    ax_d = ax_in(fig, 0.50, yt, 0.97, ht)
    ax_ai = ax_in(fig, 1.87, yt, 0.60, ht)
    ax_b = ax_in(fig, 3.02, yt, 3.16, ht)
    sb, yb = 1.37, 0.10                      # (c): two square views
    ax_c1 = ax_in(fig, 0.04, yb, sb, sb)
    ax_c2 = ax_in(fig, 0.04 + sb + 0.06, yb, sb, sb)
    zs = 0.62                                 # (c) inset, lower-left corner of the first view
    ax_z = ax_in(fig, 0.04 + 0.03, yb + 0.03, zs, zs)
    hm, ym = 1.05, 0.50                       # (d)
    ax_m = ax_in(fig, 3.30, ym, hm, hm)
    cax = ax_in(fig, 3.30 + hm + 0.05, ym, 0.06, hm)
    ax_p = ax_in(fig, 5.30, ym, 0.89, hm)

    panel_pairs(ax_d, ax_ai, S, R)
    panel_groups(ax_b, S)
    h, zc, side = panel_snapshot((ax_c1, ax_c2), ax_z, sn)
    panel_map(ax_m, cax, ax_p, R)
    print(f"snapshot view: +-{h:.0f} mm about the dish centre, t = {SNAP_T:.0f} s; inset {side:.0f} mm about "
          f"({zc[0]:.1f}, {zc[1]:.1f}) mm")

    ytop_a = yt + ht + 0.03
    letter_at(fig, 0.10, ytop_a, "a"); letter_at(fig, 3.02 - 0.40, ytop_a, "b")
    ybot = ym + hm + 0.16                     # c and d on one line
    letter_at(fig, 0.0, ybot, "c"); letter_at(fig, 2.98, ybot, "d")

    hs = [Line2D([], [], color=MODEL_C[m], ls=SEED_LS[s], marker="o", ms=3.8, lw=1.0,
                 mfc="white" if SEED_MFC[s] == "none" else MODEL_C[m],
                 label=UNIT_LAB[(m, s)]) for m, s in UNITS]
    hs.append(Line2D([], [], color=MUTED, marker="s", ms=3.0, lw=0, label="same fish, blind"))
    hs.append(Line2D([], [], color=NULL_C, marker="_", ms=5.5, mew=1.0, lw=0, label="other chance refs."))
    fig.legend(handles=hs, ncol=6, loc="upper center", bbox_to_anchor=(0.53, 1.0), fontsize=7, handlelength=2.0,
               columnspacing=0.9, borderaxespad=0.1)
    save(fig, "fig_multifish")


if __name__ == "__main__":
    main()
