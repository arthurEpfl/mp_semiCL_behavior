"""fig_scale (Chapter 5, sec:r-reading): the two scales on which every model in the chapter is read.

a  Matched-windowing classifier AUC of the reference generators (re-ordered REAL bouts), the real-vs-real floor and
   the two headline models at both seeds, all 39 window features (circle) and without the four radial ones (diamond).
   Exactly the former fig_realism panel a. Source: Model/missing_pieces/refs.pkl ["table"]
   (missing_pieces.realism_refs); 95 % CI = mean +- t(0.975, 7) x SE over the 8 rollout / resampling replicates (the
   cached "se" column is SD / sqrt(8)); the real-vs-real floor is the mean of 5 random-halves splits, so its CI uses
   t(0.975, 4). Sorted by the all-feature AUC.
b  Held-out likelihood saved by M6 and F (bits per bout, baseline minus model, per fish), mean over the 15 fish and
   95 % bootstrap CI over fish (missing_pieces._boot_ci, 4000 resamples, seed 0 -- as missing_pieces.baselines_table):
     joint  = sum of the three heads, against the pooled per-head marginals (GMMs fitted on the training fish);
     turn   = turn head, against the symmetric 3-state HMM, and against the stimulus-aware (input-driven) HMM.
   Every baseline is fitted on the training fish of M6's folds at that seed and scored on the same held-out fish, so
   the difference is paired by fish within a seed. Source: Model/missing_pieces/baselines.pkl ("base": per-fish
   baseline bits; "models": per-fish M6/F bits from each unit's likA_model1.csv) (missing_pieces.likelihood_baselines).
   The script asserts that the joint and stimulus-HMM savings equal missing_pieces.baselines_table() to the printed
   precision.

    uv run python scripts/report_figs/fig_scale.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import missing_pieces as MP  # noqa: E402
from fig_realism import HEADS, _mk, letter  # noqa: E402
from scipy import stats  # noqa: E402

REF_LABEL = {"shuffled": "shuffled", "block3": "blocks of 3", "block10": "blocks of 10", "block30": "blocks of 30",
             "pooled": "pooled bouts", "hmm": "HMM, 3 states",
             "real vs real (random halves of each fish)": "real vs real"}
N_FLOOR_SEEDS = 5          # realism_refs: the floor is the mean of 5 random-halves splits (se over those 5)
REAL_LAB = "real vs real (random halves of each fish)"

# (key, tick label, baseline column in baselines.pkl "base", model column in "models"); top to bottom
SAVINGS = [("joint", "joint vs pooled\nmarginals", "pooled marginal (joint)", "joint"),
           ("hmm3", "turn vs\n3-state HMM", "HMM, 3 states (turn)", "turn"),
           ("hmms", "turn vs\nstimulus HMM", "HMM + stimulus (turn)", "turn")]
DODGE = np.array([-0.27, -0.09, 0.09, 0.27])


# --------------------------------------------------------------------------------------------- data
def auc_table():
    R = MP.load("refs")["table"].copy()
    rows = []
    for g, r in R.iterrows():
        t = stats.t.ppf(0.975, (N_FLOOR_SEEDS if g == REAL_LAB else MP.WIN_REPS) - 1)
        if g.endswith("(model)"):
            mod, seed = g.replace(" (model)", "").split()
            name, kind = f"{mod} {seed}", ("model", mod, int(seed[1:]))
        else:
            name, kind = REF_LABEL[g], ("real" if g == REAL_LAB else "ref", None, None)
        rows.append(dict(key=g, name=name, kind=kind[0], model=kind[1], seed=kind[2],
                         all=r["all features"], all_ci=t * r["all features se"],
                         norad=r["no radial"], norad_ci=t * r["no radial se"]))
    rows.sort(key=lambda d: (-d["all"], -d["norad"]))
    return rows


def savings():
    """{saving key: {head label: (mean, lo, hi, n fish, n fish where the model is better)}}, per fish paired."""
    b = MP.load("baselines")
    base, mods = b["base"], b["models"]
    out = {k: {} for k, *_ in SAVINGS}
    for mod, seed, lab in HEADS:
        bs = base[base.seed == seed].set_index("fish")
        mm = mods[mods.model == lab].set_index("fish")
        assert bs.index.is_unique and mm.index.is_unique and set(bs.index) == set(mm.index) and len(bs) == 15
        for k, _, bcol, mcol in SAVINGS:
            d = (bs[bcol] - mm[mcol]).values               # aligned on fish; > 0 = the model is better
            lo, hi = MP._boot_ci(d)
            out[k][lab] = (float(d.mean()), lo, hi, int(np.isfinite(d).sum()), int((d > 0).sum()))
    # the two savings that missing_pieces.baselines_table prints must agree to its 3 decimals
    T = MP.baselines_table()
    for mod, seed, lab in HEADS:
        for k, col in (("joint", "saved_vs_pooled_joint"), ("hmms", "saved_vs_hmm_turn")):
            m, lo, hi = out[k][lab][:3]
            assert T.loc[(seed, lab), col] == f"{m:.3f} [{lo:.3f}, {hi:.3f}]", (lab, k, T.loc[(seed, lab), col])
    return out


# --------------------------------------------------------------------------------------------- drawing
def main():
    rows = auc_table()
    S = savings()

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.40, left=0.125, right=0.985,
                          top=0.855, bottom=0.165)

    # ---------------------------------------------------------------- a: the AUC scale (as the old fig_realism a)
    ax = fig.add_subplot(gs[0])
    OFF_ALL, OFF_NR = -0.15, 0.17
    for k, r in enumerate(rows):
        if r["kind"] == "model":
            c, mk = MODEL_C[r["model"]], _mk(r["model"], r["seed"])
        else:
            c = REAL_C if r["kind"] == "real" else MUTED
            mk = dict(color=c, mec=c, mfc=c, mew=0.9)
        for v, ci, off, m, ms in ((r["all"], r["all_ci"], OFF_ALL, "o", 3.4),
                                  (r["norad"], r["norad_ci"], OFF_NR, "D", 2.8)):
            ax.plot([v - ci, v + ci], [k + off] * 2, color=c, lw=0.8, zorder=2, solid_capstyle="butt")
            ax.plot(v, k + off, m, ms=ms, zorder=3, **mk)
    ax.axvline(0.5, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows])
    for tl, r in zip(ax.get_yticklabels(), rows):
        tl.set_color(INK)
        if r["kind"] == "model":
            tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0.45, 1.03)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.5", "", "0.7", "", "0.9", "1"])
    ax.set_xlabel("classifier AUC (matched windows)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    # shape legend: hollow black outlines (neither the references' grey nor a model colour), right of the 0.5 line in
    # the empty space left of the top (AUC ~ 1) rows
    h = [Line2D([], [], ls="", marker="o", ms=3.4, mec=INK, mfc="none", mew=0.8, label="all features"),
         Line2D([], [], ls="", marker="D", ms=2.8, mec=INK, mfc="none", mew=0.8, label="no radial")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.11, 1.02), handletextpad=0.2, borderaxespad=0.1,
              labelspacing=0.25, handlelength=1.0)
    letter(ax, "a", "Classifier scale")

    # ---------------------------------------------------------------- b: likelihood saved over the baselines
    bx = fig.add_subplot(gs[1])
    for i, (k, _, _, _) in enumerate(SAVINGS):
        for j, (mod, seed, lab) in enumerate(HEADS):
            m, lo, hi, _, _ = S[k][lab]
            y = i + DODGE[j]
            bx.plot([lo, hi], [y, y], color=MODEL_C[mod], lw=0.8, zorder=2, solid_capstyle="butt")
            bx.plot(m, y, "o", ms=3.4, zorder=3, **_mk(mod, seed))
    bx.set_yticks(range(len(SAVINGS)))
    bx.set_yticklabels([t for _, t, _, _ in SAVINGS])
    for tl in bx.get_yticklabels():
        tl.set_color(INK)
    bx.tick_params(axis="y", length=0, pad=2)
    for y in (0.5, 1.5):
        bx.axhline(y, color=FAINT, lw=0.5, zorder=0)
    bx.set_ylim(len(SAVINGS) - 0.5, -0.5)
    bx.set_xlim(0, 1.08)
    bx.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bx.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    bx.set_xlabel("likelihood saved (bits per bout)")
    bx.grid(axis="x")
    bx.set_axisbelow(True)
    bx.spines["left"].set_visible(False)
    letter(bx, "b", "Likelihood scale")

    # one legend for the model units, the references and the real fish
    hl = [Line2D([], [], ls="", marker="o", ms=3.4, label=f"{mod} s{seed}", **_mk(mod, seed))
          for mod, seed, _ in HEADS]
    hl += [Line2D([], [], ls="", marker="o", ms=3.4, color=MUTED, label="references"),
           Line2D([], [], ls="", marker="o", ms=3.4, color=REAL_C, label="real")]
    fig.legend(handles=hl, loc="upper right", bbox_to_anchor=(0.985, 1.0), ncol=6, handlelength=0.8,
               columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)
    save(fig, "fig_scale")
    return rows, S


if __name__ == "__main__":
    rows, S = main()
    print("\n(a) matched AUC, all | no radial (95 % CI half-width):")
    for r in rows:
        print(f"  {r['name']:<16} {r['all']:.3f} +- {r['all_ci']:.3f} | {r['norad']:.3f} +- {r['norad_ci']:.3f}")
    print("\n(b) bits per bout saved, mean [95 % CI over fish] (fish where the model is better / n):")
    for k, lab, _, _ in SAVINGS:
        print("  " + lab.replace("\n", " "))
        for mod, seed, h in HEADS:
            m, lo, hi, n, nb = S[k][h]
            print(f"    {h:<8} {m:.3f} [{lo:.3f}, {hi:.3f}]  ({nb}/{n})")
