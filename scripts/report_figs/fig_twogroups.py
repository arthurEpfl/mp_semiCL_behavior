"""Figure fig_twogroups (Chapter 5, sec:r-twogroups): two dot groups at once, under the four combination rules.

(a) Excess P(between) over a fish shown ONE group (``dbtw_chooser``: P(between) minus the mean of the only-A and
    only-B references on the same tracks, paired by fish, 95 % bootstrap CI over the 49-50 simulated fish with kept trials) against the
    angular separation of two 4-dot moving groups at 21 mm, one line per rule. Left: M6 (26e); right: F (14f).
    Thin faint lines (no markers, no CI): seed 101 (M6 30d, F 32b, dashed). M6+C (28b, M6 with the profile
    correction, seed 0 only) agrees and is printed by --numbers only.
(b) Preference for group A beyond the neither-shown null (``pref_corr``: P(A) - P(B) minus the same for a fish shown
    neither group, paired by fish, 95 % CI) for the count scenes (2 v 8, 2 v 4, 4 v 8 moving at 21 mm), near v far
    (4 v 4 moving at 14 v 28 mm) and motion (4 moving v 4 static, 8 v 8). A = the smaller, nearer or moving group.
    Large markers: M6 (circle) and F (square), seed 0, with CI; small faint open markers (no CI), just outside their
    main marker: seed 101 of the same model.
(c) Count against distance: ``pref_corr`` for the 2-dot group (A) against 8 dots (B), the 2 dots at 14 / 21 / 28 mm
    and the 8 at 14 / 21 / 28 mm (the 21 v 21 cell is scene n2v8), for 'effects' and 'nearest', M6 and F (seed 0).
    Black outline and dark text = the 95 % CI excludes 0 (grey text: it includes 0). add / avg are not drawn
    (``--numbers`` prints them).

Sources (read only): Model/multi_stim/<tag>_<cond>_<scene>.pkl, summarised with ``multi_stim.summary_table`` (the same
function that wrote the archived Model/multi_stim/summary_table.csv for 28b / 14f; this script checks the match).
Distances: the scene code's 50 / 75 / 100 px, labelled at Chapter 3's physical 3.60 px/mm (13.9 / 20.8 / 27.8 mm,
shown as 14 / 21 / 28): the scene code's mm (multi_stim.PX_PER_MM = 3.67, asserted) x style.MM_3P67_TO_3P60. Nothing
else in this figure is a distance (probabilities and preferences are scale-free).
Seeds are never pooled: every marker is one model at one seed. The summary of all five models is cached in the
REPORT_FINAL_DIR (CACHE); delete it (or pass --recompute) to rebuild it from the pickles.

    uv run python scripts/report_figs/fig_twogroups.py [--recompute] [--numbers]   # --numbers prints every value
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import multi_stim as MS  # noqa: E402

RF = REPORT_FINAL_DIR
CACHE = RF / "figs" / "cache" / "fig_twogroups_summary.csv"
ARCHIVE = MODEL / "multi_stim" / "summary_table.csv"

TAGS = ("28b", "14f", "26e", "30d", "32b")
MAIN = {"M6": "26e", "F": "14f"}                         # seed 0: the analysis model and the final model
MARK = {"M6": "o", "F": "s"}
# Replications drawn faintly next to their family's main model: (tag, label, seed)
REPS = {"M6": [("30d", "M6 s101", 101)], "F": [("32b", "F s101", 101)]}   # M6+C (28b) agrees; printed only
REP_PRINT = ("30d", "32b", "28b")          # printed by --numbers next to the main models: s101 of each, and M6+C
RULES = MS.RULES                                          # effects, add, avg, nearest
ANGLES = (30, 60, 90, 120, 180)
assert_code_scale(MS.PX_PER_MM, "multi_stim.PX_PER_MM")
_NF = f"{MS.D_NEAR / MS.PX_PER_MM * MM_3P67_TO_3P60:.0f} v {MS.D_FAR / MS.PX_PER_MM * MM_3P67_TO_3P60:.0f}"
PREF_SCENES = [("n2v8", "2 v 8"), ("n2v4", "2 v 4"), ("n4v8", "4 v 8"), ("near_far", _NF),   # 50 v 100 px: 14 v 28
               ("mv4_st4", "4 v 4"), ("mv8_st8", "8 v 8")]
PREF_GROUPS = [("count (dots)", 0, 2), ("distance (mm)", 3, 3), ("moving v static", 4, 5)]
DS = (MS.D_NEAR, MS.D_STD, MS.D_FAR)                      # px
CMAP = LinearSegmentedColormap.from_list("bwv", [BLUE, "#f7f7f7", VERM])
VMAX = 0.5


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def summary(recompute=False):
    """multi_stim.summary_table for the five models (one tag at a time, freeing the loaded pickles)."""
    if CACHE.exists() and not recompute:
        return pd.read_csv(CACHE)
    parts = []
    for t in TAGS:
        parts.append(MS.summary_table(tags=(t,)))
        MS._LOADED.clear()
    S = pd.concat(parts, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    S.to_csv(CACHE, index=False)
    return S


def check_archive(S):
    """The recomputed rows of 28b / 14f must equal the archived summary_table.csv."""
    A = pd.read_csv(ARCHIVE)
    m = S.merge(A, on=["model", "scene", "cond"], suffixes=("", "_arch"))
    worst = 0.0
    for c in ("P_between", "pref", "dbtw_chooser", "dbtw_chooser_lo", "dbtw_chooser_hi", "pref_corr",
              "pref_corr_lo", "pref_corr_hi"):
        d = (m[c] - m[c + "_arch"]).abs().to_numpy(float)
        worst = max(worst, float(np.nanmax(d)) if np.isfinite(d).any() else 0.0)
    print(f"  archive check: {len(m)} rows of 28b/14f matched, max |diff| = {worst:.4g}")
    return worst


def row(S, tag, scene, cond):
    d = S[(S.model == tag) & (S.scene == scene) & (S.cond == cond)]
    if len(d) != 1:
        raise KeyError((tag, scene, cond))
    return d.iloc[0]


def grid_scene(da, db):
    return "n2v8" if (da, db) == (MS.D_STD, MS.D_STD) else f"g2_{int(da)}_8_{int(db)}"


def mm(px):
    """A scene distance (px) as a label in mm at 3.60 px/mm: the scene code's mm (px / 3.67) x MM_3P67_TO_3P60."""
    return f"{px / MS.PX_PER_MM * MM_3P67_TO_3P60:.0f}"


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
FAINT_A = 0.55                                            # alpha of the replications
A_OFF = dict(zip(RULES, (-6.0, -2.0, 2.0, 6.0)))          # deg, so the four intervals of one angle stay apart
B_SUB = {"M6": -0.22, "F": 0.22}                        # x of the main marker inside a rule slot of (b)
B_REP = 0.21                                              # replications this far outside their main marker


def letter_title(fig, x_fig, y_fig, letter, title):
    """Panel letter (right edge at x_fig) and bold title a FIXED 4 pt to its right, both hanging from y_fig."""
    dx = 4.0 / 72.0 / fig.get_figwidth()
    fig.text(x_fig, y_fig, letter, fontsize=9, fontweight="bold", ha="right", va="top")
    fig.text(x_fig + dx, y_fig, title, fontsize=8.5, fontweight="bold", ha="left", va="top")


def panel_angle(ax, S, fam, show_y):
    main = MAIN[fam]
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    for rule in RULES:
        c = RULE_C[rule]
        x = np.array(ANGLES) + A_OFF[rule]
        for tag, _, seed in REPS[fam]:                     # replications: thin faint lines, no markers
            y = [row(S, tag, f"ang{a}", rule).dbtw_chooser for a in ANGLES]
            ax.plot(x, y, color=c, alpha=FAINT_A, lw=0.6, ls=SEED_LS[seed], zorder=2)
        r = [row(S, main, f"ang{a}", rule) for a in ANGLES]
        y = np.array([q.dbtw_chooser for q in r])
        lo = np.array([q.dbtw_chooser_lo for q in r])
        hi = np.array([q.dbtw_chooser_hi for q in r])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=c, lw=1.1, elinewidth=0.7, capsize=0, marker=MARK[fam],
                    ms=3.4, mec="white", mew=0.4, zorder=3)
    ax.set_xticks(ANGLES)
    ax.set_xticklabels([str(a) for a in ANGLES])
    ax.set_xlim(20, 190)
    ax.set_ylim(-0.12, 0.50)
    ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("separation (deg)")
    if show_y:
        ax.set_ylabel(r"$\Delta P$(between)")
    else:
        ax.tick_params(labelleft=False)
    # which model is bold and which are faint, written on the axes itself
    faint = {"M6": "faint: M6 s101", "F": "faint: F s101"}[fam]
    ax.text(0.03, 0.985, f"{fam} s0", transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold")
    ax.text(0.03, 0.865, faint, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=MUTED)


def panel_pref(ax, S):
    xs_scene = []
    x0 = 0.0
    for si, (sc, _) in enumerate(PREF_SCENES):
        if si in (3, 4):
            x0 += 1.3                                      # extra gap between the scene families
        xs = x0 + np.arange(len(RULES), dtype=float)
        xs_scene.append(xs.mean())
        for ri, rule in enumerate(RULES):
            c = RULE_C[rule]
            for fam in ("M6", "F"):
                x = xs[ri] + B_SUB[fam]
                xr = x + (-B_REP if fam == "M6" else B_REP)         # outside: M6 to the left, F to the right
                for tag, _, seed in REPS[fam]:
                    q = row(S, tag, sc, rule)
                    ax.plot(xr, q.pref_corr, ls="none", marker=MARK[fam], ms=2.2, color=c, alpha=FAINT_A + 0.15,
                            mfc=c if seed == 0 else "white", mec=c, mew=0.5, zorder=4)
                q = row(S, MAIN[fam], sc, rule)
                ax.errorbar(x, q.pref_corr, yerr=[[q.pref_corr - q.pref_corr_lo], [q.pref_corr_hi - q.pref_corr]],
                            color=c, elinewidth=0.7, capsize=0, marker=MARK[fam], ms=3.4, mec="white", mew=0.4,
                            ls="none", zorder=3)
        x0 = xs[-1] + 1.45
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_xticks(xs_scene)
    ax.set_xticklabels([lab for _, lab in PREF_SCENES])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs_scene[0] - 2.1, xs_scene[-1] + 2.1)
    ax.set_ylim(-0.42, 1.0)
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("preference for A")
    for name, i0, i1 in PREF_GROUPS:                       # scene-family labels under the tick labels
        xm = (xs_scene[i0] + xs_scene[i1]) / 2
        ax.annotate(name, xy=(xm, 0), xycoords=("data", "axes fraction"), xytext=(0, -13),
                    textcoords="offset points", ha="center", va="top", fontsize=7, color=MUTED)
    return xs_scene


def panel_grid(ax, S, fam, rule, show_y):
    tag = MAIN[fam]
    n = len(DS)
    for i, da in enumerate(DS):                            # rows: the 2 dots (A)
        for j, db in enumerate(DS):                        # columns: the 8 dots (B)
            q = row(S, tag, grid_scene(da, db), rule)
            v = q.pref_corr
            sig = (q.pref_corr_lo > 0) or (q.pref_corr_hi < 0)
            fc = CMAP(0.5 + 0.5 * np.clip(v / VMAX, -1, 1))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="white", lw=1.6, zorder=1))
            if sig:
                ax.add_patch(Rectangle((j + 0.025, i + 0.03), 0.95, 0.94, fill=False, edgecolor=INK, lw=0.6,
                                       zorder=2))
            dark = 0.2126 * fc[0] + 0.7152 * fc[1] + 0.0722 * fc[2] < 0.40      # white text on the darkest cells
            ax.text(j + 0.5, i + 0.5, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=7,
                    color="white" if dark else (INK if sig else MUTED), zorder=3)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([mm(d) for d in DS])
    ax.set_yticks(np.arange(n) + 0.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("8 dots at (mm)", labelpad=1.5)
    if show_y:
        ax.set_yticklabels([mm(d) for d in DS])
        ax.set_ylabel("2 dots at (mm)", labelpad=1.5)
    else:
        ax.set_yticklabels([])
    ax.set_title(f"{fam} s0, {rule}", fontsize=7, fontweight="bold", pad=3)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make(S):
    import matplotlib as mpl
    fig = plt.figure(figsize=(TEXTWIDTH_IN - 0.06, 3.4))   # the tight bbox adds the outer labels back
    outer = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.66, left=0.075, right=0.995, top=0.925,
                             bottom=0.1)
    top = outer[0].subgridspec(1, 4, width_ratios=[1.12, 1.12, 0.36, 2.6], wspace=0.07)
    ax_a1 = fig.add_subplot(top[0])
    ax_a2 = fig.add_subplot(top[1], sharey=ax_a1)
    ax_b = fig.add_subplot(top[3])                         # top[2] = spacer for (b)'s y label
    panel_angle(ax_a1, S, "M6", True)
    panel_angle(ax_a2, S, "F", False)
    panel_pref(ax_b, S)

    # (b) keys, stacked at the top left: rules (colour only) above models (shape; faint small = replications)
    h_rules = [Line2D([], [], color=RULE_C[r], lw=2.2, label=r) for r in RULES]
    leg = ax_b.legend(handles=h_rules, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.3,
                      columnspacing=1.0, labelspacing=0.25, borderaxespad=0.15, handletextpad=0.5)
    ax_b.add_artist(leg)
    h_mod = [Line2D([], [], color=INK, ls="-", lw=1.1, marker="o", ms=3.4, mec="white", mew=0.4, label="M6 s0"),
             Line2D([], [], color=INK, alpha=FAINT_A, ls="--", lw=0.6, marker="o", ms=2.2, mfc="white",
                    label="M6 s101"),
             Line2D([], [], color=INK, ls="-", lw=1.1, marker="s", ms=3.4, mec="white", mew=0.4, label="F s0"),
             Line2D([], [], color=INK, alpha=FAINT_A, ls="--", lw=0.6, marker="s", ms=2.2, mfc="white",
                    label="F s101")]
    ax_b.legend(handles=h_mod, loc="upper left", bbox_to_anchor=(0.0, 0.775), ncol=2, handlelength=2.0,
                columnspacing=1.0, labelspacing=0.25, borderaxespad=0.15, handletextpad=0.5)

    # (c): M6 pair | gap | F pair | gap | colour bar
    bot = outer[1].subgridspec(1, 8, width_ratios=[1, 1, 0.14, 1, 1, 0.10, 0.07, 0.50], wspace=0.10)
    axs_c = []
    for k, (fam, rule) in zip((0, 1, 3, 4),
                              (("M6", "effects"), ("M6", "nearest"), ("F", "effects"), ("F", "nearest"))):
        ax = fig.add_subplot(bot[k])
        panel_grid(ax, S, fam, rule, show_y=(k == 0))
        axs_c.append(ax)
    cax = fig.add_subplot(bot[6])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-VMAX, VMAX), cmap=CMAP), cax=cax,
                      orientation="vertical", ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    cb.ax.set_yticklabels(["−0.5", "", "0", "", "+0.5"])
    cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
    cb.outline.set_visible(False)
    cb.set_label("preference for 2 dots", fontsize=7, labelpad=2)

    # panel letters and titles, at a fixed 4 pt gap (a and c share the letter x)
    fig.canvas.draw()
    pa, pb, pc = ax_a1.get_position(), ax_b.get_position(), axs_c[0].get_position()
    x_ac = pa.x0 - 0.30 / fig.get_figwidth()             # left of (a)'s y tick labels, as in Chapter 3
    y_ab = pa.y1 + 0.17 / fig.get_figheight()
    letter_title(fig, x_ac, y_ab, "a", "Ending between the two groups")
    letter_title(fig, pb.x0 - 0.30 / fig.get_figwidth(), y_ab, "b", "Preference for group A beyond the null")
    y_c = pc.y1 + 0.27 / fig.get_figheight()
    letter_title(fig, x_ac, y_c, "c", "Count against distance (2 v 8 moving dots)")
    fig.text(axs_c[-1].get_position().x1, y_c, "outlined: 95 % CI excludes 0", fontsize=7, color=MUTED,
             ha="right", va="top")
    return fig


def key_numbers(S):
    """Print what the figure shows (for the notes and the prose)."""
    R = S[S.cond.isin(RULES)]
    print("\n(a) dbtw_chooser [lo, hi] per rule, angle:")
    for fam, tag in MAIN.items():
        for rule in RULES:
            vals = [row(S, tag, f"ang{a}", rule) for a in ANGLES]
            print(f"  {fam:5s} {rule:8s} " + "  ".join(
                f"{a}:{q.dbtw_chooser:+.3f}[{q.dbtw_chooser_lo:+.3f},{q.dbtw_chooser_hi:+.3f}]" for a, q in
                zip(ANGLES, vals)))
    print("  replications (no CI shown):")
    for tag in REP_PRINT:
        for rule in RULES:
            print(f"  {tag} {rule:8s} " + "  ".join(
                f"{a}:{row(S, tag, f'ang{a}', rule).dbtw_chooser:+.3f}" for a in ANGLES))
    print("\n(b) pref_corr [lo, hi]:")
    for sc, lab in PREF_SCENES:
        for rule in RULES:
            s = "  ".join(f"{fam}:{row(S, t, sc, rule).pref_corr:+.3f}[{row(S, t, sc, rule).pref_corr_lo:+.3f},"
                          f"{row(S, t, sc, rule).pref_corr_hi:+.3f}]" for fam, t in MAIN.items())
            reps = " ".join(f"{t}:{row(S, t, sc, rule).pref_corr:+.3f}" for t in REP_PRINT)
            print(f"  {sc:9s} {rule:8s} {s}   reps {reps}")
    lab_d = "/".join(mm(d) for d in DS)
    print(f"\n(c) grid pref_corr (rows 2 dots at {lab_d}, cols 8 dots at {lab_d} mm, 3.60 px/mm), * = CI excludes 0:")
    for fam, tag in MAIN.items():
        for rule in RULES:
            print(f"  {fam} {rule}")
            for da in DS:
                cells = []
                for db in DS:
                    q = row(S, tag, grid_scene(da, db), rule)
                    sig = (q.pref_corr_lo > 0) or (q.pref_corr_hi < 0)
                    cells.append(f"{q.pref_corr:+.3f}{'*' if sig else ' '}")
                print(f"    2@{mm(da)}: " + "  ".join(cells))
    # sign agreement of the replications in the grid, effects and nearest
    print("\n(c) sign of every grid cell in all five models (28b, 14f, 26e, 30d, 32b), per rule:")
    for rule in RULES:
        agree, neg, tot, pos = 0, 0, 0, []
        for da in DS:
            for db in DS:
                sc = grid_scene(da, db)
                v = [row(S, t, sc, rule).pref_corr for t in TAGS]
                agree += int(len(set(np.sign(v))) == 1)
                neg += sum(x < 0 for x in v)
                tot += len(v)
                pos += [f"{t} 2@{mm(da)} v 8@{mm(db)} {x:+.3f}" for t, x in zip(TAGS, v) if x >= 0]
        print(f"  {rule:8s}: same sign in all five models in {agree}/9 cells; {neg}/{tot} estimates negative")
        if rule != "nearest" and pos:
            print("            non-negative: " + "; ".join(pos))
    print("  nearest, by geometry (all five models):")
    for lab, cmp in (("2 dots nearer", lambda a, b: a < b), ("8 dots as near or nearer", lambda a, b: a >= b)):
        v = [row(S, t, grid_scene(da, db), "nearest").pref_corr for t in TAGS for da in DS for db in DS if cmp(da, db)]
        print(f"    {lab}: {sum(x > 0 for x in v)}/{len(v)} positive, range {min(v):+.3f} to {max(v):+.3f}")
    print("  finding (6) count (8 non-centre cells x effects/add/avg), per model:")
    for t in TAGS:
        v = [row(S, t, grid_scene(da, db), r) for r in ("effects", "add", "avg") for da in DS for db in DS
             if (da, db) != (MS.D_STD, MS.D_STD)]
        nsig = sum((q.pref_corr_lo > 0) or (q.pref_corr_hi < 0) for q in v)
        print(f"    {t}: {sum(q.pref_corr < 0 for q in v)}/{len(v)} negative, {nsig} intervals exclude 0")
    print("\n(b) sign over all five models x four rules:")
    for sc, lab in PREF_SCENES:
        v = np.array([[row(S, t, sc, r).pref_corr for r in RULES] for t in TAGS])
        print(f"  {lab:8s}: {int((v > 0).sum())}/20 positive, range {v.min():+.3f} to {v.max():+.3f}")
    v = np.array([row(S, t, "near_far", r).pref_corr for t in TAGS for r in ("effects", "add", "avg")])
    print(f"  {_NF} without nearest: {v.min():+.3f} to {v.max():+.3f}")
    print("\n(a) replication ranges (" + ", ".join(REP_PRINT) + "):")
    for grp, rules in (("nearest", ("nearest",)), ("effects/add/avg", ("effects", "add", "avg"))):
        v = [row(S, t, f"ang{a}", r).dbtw_chooser for t in REP_PRINT for r in rules for a in ANGLES]
        print(f"  {grp}: {min(v):+.3f} to {max(v):+.3f}")
    print("\ntrials kept per rule and scene: " + ", ".join(
        f"{t} {int(R[R.model == t].n_trials.min())}-{int(R[R.model == t].n_trials.max())}" for t in TAGS)
        + f"; fish {int(R.n_fish.min())}-{int(R.n_fish.max())}")
    for t in TAGS:
        d = S[S.model == t]
        sp = d.groupby("scene").dropped_pct.agg(lambda x: x.max() - x.min())
        print(f"  dropped {t}: {d.dropped_pct.min():.1f}-{d.dropped_pct.max():.1f} %, "
              f"largest within-scene spread {sp.max():.1f} pp")
    return R


if __name__ == "__main__":
    S = summary(recompute="--recompute" in sys.argv)
    check_archive(S)
    fig = make(S)
    save(fig, "fig_twogroups")
    if "--numbers" in sys.argv:
        key_numbers(S)
