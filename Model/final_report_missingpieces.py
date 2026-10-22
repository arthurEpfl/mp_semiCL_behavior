"""Build Model/Final_report_missingPieces.ipynb: the analyses the thesis still lacked, one section per item, from the
caches of scripts/missing_pieces.py (Model/missing_pieces/), the saved units of report_models.py (Model 34 = the
retrained Part B units) and the multi-stim / multi-fish runs. Nothing is simulated here; the cells re-read and draw.

    python final_report_missingpieces.py
"""
import json
import sys

import final_report_common as C
from final_report_common import md, code

NB = C.HERE / "Final_report_missingPieces.ipynb"
sys.path.insert(0, str(C.HERE.parent / "scripts"))

INTRO = """# Final report — missing pieces

The analyses the work still lacked after the model sequence, the ablations and the extrapolations, one section per item.
**Part A** needs no retraining (references, real-data properties, the learned rule, replications, rollout noise,
provenance, intervals, the centroid choice, multi-fish extras, robustness numbers); **Part B** retrains (leave one
condition out, the main line at seed 101, a capacity check).

Headline models: **M6** = the analysis model (key 26e at seed 0, 30d at seed 101); **F** = the final model (14f / 32b).
Seed 0 and seed 101 are different cross-validation splits: never pooled, a claim needs the same direction at both.
Every AUC is the classifier test on held-out fish with the **matched** windowing unless stated (0.5 = indistinguishable;
real against real ≈ 0.52). Likelihoods are held-out bits per bout (lower = better).

Code: `scripts/missing_pieces.py` (computations, cached in `Model/missing_pieces/`), `scripts/multi_stim.py`,
`scripts/multi_fish.py`; the retrained units are Model 34 in `report_models.py` (keys 34a–34j)."""

TOC = """## Contents

**Part A**
1. [Realism scale: reference generators and classical baselines](#s1)
2. [What still gives the models away](#s2)
3. [The learned rule: response maps and a compact summary](#s3)
4. [The extrapolations replicated](#s4)
5. [Rollout-stream noise, and F against M6](#s5)
6. [Data provenance](#s6)
7. [Paired-by-fish intervals on the key likelihood differences](#s7)
8. [Which point of a dot group the fish turns toward](#s8)
9. [Multi-fish extras](#s9)
10. [Robustness numbers](#s10)

**Part B**
11. [Leave one condition out: predicting 6 moving dots](#s11)
12. [The main line at seed 101](#s12)
13. [Capacity: 64 and 256 hidden units](#s13)"""

SETUP = """# Setup: everything below reads saved results (scripts/missing_pieces.py caches, report_outputs/, multi_stim/, multi_fish/).
import sys, json, glob
sys.path.insert(0, "."); sys.path.insert(0, "../scripts")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
import missing_pieces as P

def S(key):
    \"\"\"The saved key numbers of a notebook unit (report_outputs/summary_M<key>.json), or {}.\"\"\"
    return P.summary(key) or {}"""

FINDINGS = """## Findings in short

Values are seed 0 / seed 101 unless stated. "Beyond the noise" means the same sign at both seeds with an interval that
excludes zero.

**Part A**

1. **Realism now has a scale.** On the matched windowing, the classifier separates real fish from:
   - their own bouts in random order: AUC 1.00;
   - the classical 3-state HMM: 1.00, and 0.99 on turn, step and interval alone;
   - their own bouts in random blocks of 10: 0.86;
   - their own bouts in random blocks of 30: 0.67.

   M6 scores 0.67 / 0.65 and F 0.64 / 0.64; real against real is 0.50. So the models are about as hard to tell from a
   fish as **the fish's own data cut into 30-bout pieces**. Without the position features they do better than that
   (M6 0.60 / 0.57 and F 0.56 / 0.56, against 0.66).

   On likelihood:
   - The models save **0.80–0.84 bits per bout** over pooled population marginals of the three heads.
   - They save **0.15–0.19 bits per bout** on the turn alone over the stimulus-aware HMM.
   - Both hold at both seeds, with intervals over fish well away from zero.
   - The classical HMMs barely beat a plain marginal (7.82–7.83 against 7.85 bits).
2. **What still gives the models away.** Both models match real fish within the per-fish intervals on:
   - the bout marginals (median and 90th percentile of turn, step and interval);
   - the wall law (turning toward the centre: 0.57 near the centre to 0.79 at the wall);
   - where in the dish the fish swims (median r/R 0.45).

   The systematic differences:
   - M6's interval correlations at lags 4–5 are too low (−0.05 to −0.06): slow persistence in timing is missing.
   - F's turns are slightly too persistent (+0.03 at lags 1–5, both seeds).

   The classifier uses dish position most (share of bouts near the wall, median radius), then interval timing.
   Simulated fish sit at the edge of the cloud of real individuals: their distance to the nearest real fish is
   0.83–0.86, against 0.84 between real fish.
3. **The learned rule** (toward-turn on matched geometry, real first):
   - **Steady pull, moving dots:** real 19.3°, M6 19.0–19.5° (matches), F 21.5–22.4°.
   - **Onset pull, first 4 s:** real 35.7°, M6 25.6–26.0°, F 30.9–31.1°. F's correction closes half the gap.
   - **Count step:** real 6.6°, M6 4.8°, F 6.0°.
   - **Strongest pull:** at 10–20 mm in both models, against 20–35 mm in real fish.
   - **Very close range:** real fish turn *away* from a single dot closer than 10 mm; neither model does.
4. **The extrapolations replicate.**
   - **Two dot groups:** every pattern holds on all five models and seeds (M6+C, F, M6 at seed 0; M6 and F at
     seed 101):
     - with two identical groups on opposite sides, "effects" ends between them in 44–48 % of trials (like no
       stimulus) while "nearest" commits (18–23 % between);
     - at equal distance the 8 dots win over 2 under both rules;
     - with 2 dots near and 8 far, "effects" still favours the 8 while "nearest" takes the near 2;
     - moving beats static everywhere.
   - **Multi-fish:** shoaling holds for M6 and F at both seeds:
     - pair attraction index +0.32 to +0.35;
     - groups of 10: D −0.18 to −0.20;
     - groups of 5: D −0.30 / −0.30 for F;
     - 5× dish, F seed 101: D −0.71, shoals of 8.7 fish on average, two fish travelling 4.6 m per hour within 25 mm
       of each other against 1.2 m by chance.
5. **Rollout noise, and F against M6.**
   - **One profile reading moves by:** 0.1–0.4 mm on the dip, 0.6–0.9 mm on the grading, 0.02–0.15 SEM on the
     moving error, 0.3–0.8 mm on attraction. Stream 0 reproduces every archived number exactly.
   - **F's deeper dip is beyond the noise at both seeds:** +0.4 / +0.6 mm, 4-stream means.
   - **Its moving-error advantage holds at seed 0 only; the grading differences are within noise.**
   - **M6's archived seed-0 dip (3.70 mm) is its luckiest stream:** the other three give 2.7–3.1 mm.
6. **Provenance.** The current segmentation code finds the same 71,933 bouts, with identical turns and steps. The
   uncommitted onset change shifts the timing of about 4 % of bouts (interval change > 0.04 s); every median is
   unchanged.
7. **Paired-by-fish intervals.**
   - **Beyond the noise at both seeds:**
     - the per-fish latent: −0.06 / −0.11 bits;
     - the slow drift: −0.05 to −0.06;
     - kinematic history: +0.28 / +0.32 bits without it, +0.04 with one lag instead of three.
   - **F against M6:** −0.041 [−0.079, −0.004] / −0.030 [−0.078, +0.011]. The turn head is better at both seeds;
     the joint likelihood is better at seed 0 only.
   - **Removing the stimulus:** costs turn likelihood at both seeds, but its joint cost crosses zero at seed 0.
   - **No significant change:** without group size, with 64 hidden units, or trained without 6 moving dots.
8. **The centroid is the right target.** When the centroid and the nearest dot lie on opposite sides of the fish,
   real fish turn toward the centroid 67 % of the time and toward the nearest dot 33 % (+0.34 [0.29, 0.39], 15 fish,
   1,780 bouts, moving dots). With static dots there is no preference. The centroid and a 1/r-weighted centroid almost
   never disagree.
9. **Multi-fish extras.**
   - **No short-range repulsion:** neighbour density peaks on top of each fish, whereas real juveniles keep a gap of
     5–15 mm (Kappel et al. 2022).
   - **5× dish, random directions:** the shoaling disappears (D −0.06; travelling together 0.30 m per pair-hour =
     chance). It is directed attraction.
   - **Onset clock restarted whenever a neighbour comes into view** (so F's correction acts): shoaling is unchanged
     (D −0.74).
10. **Robustness.**
    - **Without the centroid-outside-dish filter**, real fish show a 3.79 mm dip and 8.59 mm grading. The models stay
      shallower (2.4–3.0 mm) and less graded (4.9–7.1 mm), so the deficits are not an artefact of the filter.
    - **The real-vs-real floor without radial features** is 0.50.
    - **Profile, attraction and two-group millimetres** use 3.67 px/mm; multiply by 1.0194 for the thesis scale of
      3.60.

**Part B**

11. **Leave one condition out.** Trained without ever seeing 6 moving dots, M6 predicts those bouts almost as well as
    when trained on them: +0.012 [−0.011, +0.031] / +0.023 [−0.007, +0.054] bits per bout. That is within the
    retraining differences on the conditions both models saw, with AUC unchanged (0.66 / 0.63). Its 6-dot attraction
    is 4.7 / 4.6 mm, against 4.9 / 6.4 for M6 and 6.4 for real fish: a good prediction at seed 0, low at seed 101.
12. **The main line at seed 101.**
    - **Wall rejection** (0.86 → 0.83) and **the per-fish latent** (0.83 → 0.71) replicate their seed-0 steps.
    - **M3 → M4** is small at both seeds (−0.03 / −0.005).
    - **M5+T** again buys a deeper dip (+0.3 mm) at a slightly worse AUC.
13. **Capacity.**
    - **256 hidden units are worse:** +0.07 / +0.03 bits, AUC 0.73 / 0.69 — they overfit.
    - **64 units tie 128 units:** +0.02 / −0.02 bits, AUC 0.71 / 0.66. They often run into the epoch limit.

    One hidden layer of 128 is a sound choice."""

SECTIONS = []


def sec(sid, title, text, cells):
    SECTIONS.append(dict(id=sid, title=title, md=text, cells=cells))


# ---------------------------------------------------------------- 1
sec("s1", "1. Realism scale: reference generators and classical baselines", """**Question.** How good is a matched AUC of
~0.64? The classifier needs a scale: what does it give for generators that are obviously not the fish's model?

**What was done.** Each reference re-orders *real* bouts, every bout keeping all its attributes (turn, step, interval,
distance to the dots, position in the dish), and is cut at the real fish's own window positions (the matched windowing,
8 replicates per fish, the same classifier as the models):
- **shuffled** — the fish's own bouts in random order (its exact marginals and individuality, no sequence);
- **block L** — the fish's own bouts in random blocks of L = 3 / 10 / 30 consecutive bouts (short-range sequence kept);
- **pooled** — bouts drawn at random from the other 14 fish (a population marginal, no individuality, no sequence);
- **hmm** — turns from a 3-state hidden Markov model (the classical bout model, Dommanget-Kott-style; fitted on the
  training fish of each fold), every other attribute from a random bout of the other 14 fish.

**How to read.** A reference that keeps more of the real structure is harder to detect. The models are read against this
ladder; the *three heads* column (turn, step and interval only, no position or dot distance) is the fairest for the HMM,
which has no space. The likelihood table puts the same classical models (and a pooled-marginal density of all three
heads) on the models' exact folds.""",
    [("s1-refs", "cell", """# Matched C2ST AUC of the reference generators, the real-vs-real floor and the four headline models (same windows, same
# classifier). se = across the 8 replicates.
R = P.load("refs")["table"]
print(R[["all features", "all features se", "no radial", "no radial se", "three heads", "three heads se"]].round(3).to_string())"""),
     ("s1-models", "cell", """# The model sequence on the same reading (matched AUC, all features | no radial), from the saved units. M1-M3 and M5+T
# at seed 101 are the new Part B units (section 12).
rows = []
for name, k0, k1 in [("M1", "1", "34a"), ("M2", "2", "34b"), ("M3", "3", "34c"), ("M4", "20f", "30a"), ("M5", "25d", "30b"),
                     ("M5+R", "29a", "30c"), ("M5+T", "29b", "34d"), ("M6", "26e", "30d"), ("F", "14f", "32b"),
                     ("M6 without stimulus", "29e", "33u"), ("M6 latent frozen", "33q", "33r")]:
    a, b = S(k0), S(k1)
    rows.append({"model": name, "matched s0": a.get("auc_matched"), "matched s101": b.get("auc_matched"),
                 "no radial s0": a.get("auc_matched_norad"), "no radial s101": b.get("auc_matched_norad")})
print(pd.DataFrame(rows).set_index("model").round(3).to_string())"""),
     ("s1-fig", "cell", """fig, ax = plt.subplots(figsize=(9, 3.6))
R = P.load("refs")["table"]
lab = list(R.index)
ax.barh(range(len(lab)), R["all features"], xerr=R["all features se"], color=["0.6"] * (len(lab) - 4) + ["C0", "C0", "C3", "C3"])
ax.axvline(0.5, color="k", lw=0.8)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("matched C2ST AUC (all features)  — 0.5 = indistinguishable from real")
ax.set_xlim(0.45, 1.0); plt.tight_layout()"""),
     ("s1-lik", "df", """P.baselines_table().round(4)""")])

# ---------------------------------------------------------------- 2
sec("s2", "2. What still gives the models away", """**Question.** Where do M6 and F still differ from real fish, on the
properties Chapter 3 establishes, and what does the classifier use to tell them apart?

**What was done.** Two record-anchored rollouts per held-out fish (each fish by the fold model that never saw it):
the bout marginals (median and 90th percentile of |turn|, step, interval; share of near-straight bouts), the
correlations between successive bouts within runs (lags 1–5, |turn|, log step, log interval), the wall law (probability
of turning toward the dish centre by distance from the centre) and where in the dish the fish swims. Then the
classifier's permutation importances on the matched windows, and the nearest-real-neighbour distance (is each
simulated fish inside the cloud of real individuals?).

**How to read.** Differences are model − real per fish, mean over the 15 fish with a 95 % bootstrap interval over fish.""",
    [("s2-props", "df", """P.properties_table().round(3)"""),
     ("s2-scatter", "cell", """# One point per fish: real (x) against simulated (y), M6 and F at seed 0.
T = P.load("properties"); per = T.groupby(["model", "fish"]).mean(numeric_only=True)
props = ["interval_median", "step_median", "turn_median", "straight", "acf1_step", "acf1_interval"]
fig, axes = plt.subplots(1, len(props), figsize=(3.2 * len(props), 3.2))
for ax, p in zip(axes, props):
    r = per.loc["real", p]
    for lab, col in (("M6 s0", "C0"), ("F s0", "C3")):
        ax.scatter(r, per.loc[lab, p].reindex(r.index), s=18, color=col, label=lab)
    lo, hi = np.nanmin([r.min(), per[p].min()]), np.nanmax([r.max(), per[p].max()])
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8); ax.set_title(p, fontsize=9); ax.set_xlabel("real fish")
axes[0].set_ylabel("simulated (same fish's fold model)"); axes[0].legend(fontsize=7)
plt.tight_layout()"""),
     ("s2-wall", "cell", """# The wall law and where in the dish the fish are: real (mean +- 95 % CI over fish) against the models.
T = P.load("properties"); per = T.groupby(["model", "fish"]).mean(numeric_only=True)
wc = [c for c in per.columns if c.startswith("wall_")]
x = [np.mean([float(v) for v in c[5:].split("-")]) for c in wc]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
r = per.loc["real", wc]
ci = np.array([P._boot_ci(r[c].dropna().values) for c in wc])
a1.fill_between(x, ci[:, 0], ci[:, 1], color="0.8"); a1.plot(x, r.mean(), "k-o", label="real")
for lab, col in zip(P.HEAD, ("C0", "C1", "C3", "C4")):
    a1.plot(x, per.loc[lab, wc].mean(), "-o", color=col, ms=3, label=lab)
a1.set_xlabel("distance from the dish centre (r / R)"); a1.set_ylabel("P(turn toward the centre)"); a1.legend(fontsize=7)
H = T[T.rep == 0].groupby("model").radial_hist.apply(lambda s: np.mean(np.stack(s.values), 0))
edges = P.RAD_BINS; mid = (edges[1:] + edges[:-1]) / 2
for lab, col in [("real", "k")] + list(zip(P.HEAD, ("C0", "C1", "C3", "C4"))):
    a2.plot(mid, H[lab], color=col, lw=2 if lab == "real" else 1, label=lab)
a2.set_xlabel("r / R"); a2.set_ylabel("share of bouts"); a2.legend(fontsize=7)
plt.tight_layout()"""),
     ("s2-imp", "cell", """# What the matched classifier uses: permutation importance (drop in held-out AUC when a feature is shuffled), top 12.
I = P.load("importances")
print(I.assign(mean=I.mean(1)).sort_values("mean", ascending=False).head(12).round(4).to_string())"""),
     ("s2-nrn", "cell", """# Nearest real neighbour: is a simulated fish as close to some real fish as real fish are to each other?
N = P.load("nearest")
for lab, v in N.items():
    print(f"== {lab}")
    print("\\n".join(l for l in v["text"].splitlines() if l.strip().startswith(("real fish", "sampled fish", "mean distance", "the sampled"))))
    print("feature-drop AUC, PLAIN harness windowing (not matched; section 1 has the matched readings):")
    print(v["drop"].round(3).to_string())""")])

# ---------------------------------------------------------------- 3
sec("s3", "3. The learned rule: response maps and a compact summary", """**Question.** What rule did the model learn for
turning toward the dots, as a function of where they are, how many, whether they move and how long ago they appeared —
and is it the real fish's rule?

**What was done.** On every bout of every held-out fish, the model's teacher-forced expected turn (its prediction from
the real history and the real stimulus, population latent) next to the fish's real turn, both as a **toward-turn**
(turn × sign of the dots' bearing, deg). Binned by |bearing| × range, per count, motion and phase of the block. Then
five numbers per source on matched geometry (|bearing| 45–135°): the onset gain (first 4 s, moving, 10–35 mm), the
steady gain (≥ 10 s), the range of the strongest steady pull, the count step (2–8 dots minus 1 dot) and the static gain.

**How to read.** Real and model maps are on the same bouts, so a difference is the fitted rule's error, not sampling.""",
    [("s3-sum", "df", """P.law_summary().round(2)"""),
     ("s3-maps", "cell", """# Toward-turn (deg): |bearing| (rows) x range (columns), moving dots, >= 10 s into the block, per fish then over fish.
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
fig.subplots_adjust(hspace=0.6, wspace=0.45)
for j, (n, title) in enumerate((((1,), "1 moving dot"), ((2, 4), "2-4 moving dots"), ((6, 8), "6-8 moving dots"))):
    real, m6 = P.law_maps("M6 s0", True, n)
    _, f = P.law_maps("F s0", True, n)
    for i, (M, lab) in enumerate(((real, "real"), (m6, "M6 s0 (model)"), (f, "F s0 (model)"))):
        ax = axes[i, j]
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=-25, vmax=25, aspect="auto")
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([str(c) for c in M.columns], fontsize=7, rotation=30)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([str(c) for c in M.index], fontsize=7)
        ax.set_title(f"{lab}: {title}", fontsize=9)
        for (a, b), v in np.ndenumerate(M.values):
            if np.isfinite(v):
                ax.text(b, a, f"{v:.0f}", ha="center", va="center", fontsize=6)
for ax in axes[-1]:
    ax.set_xlabel("range to the dots (mm)")
for ax in axes[:, 0]:
    ax.set_ylabel("|bearing| (deg)")
fig.colorbar(im, ax=axes, shrink=0.6, label="toward-turn (deg)")"""),
     ("s3-lines", "cell", """# Toward-turn against range by dot count, real (solid) vs M6 (dashed), moving dots, |bearing| 45-135 deg, >= 10 s.
t = P._toward_cols(P.load("laws"))
g = (np.degrees(np.abs(t.phi)) >= 45) & (np.degrees(np.abs(t.phi)) <= 135) & t.moving & (t.phase == ">= 10 s") & (t.model == "M6 s0")
fig, ax = plt.subplots(figsize=(7, 3.6))
for n, col in zip((1, 2, 4, 6, 8), plt.get_cmap("PuBu")(np.linspace(0.35, 0.95, 5))):
    s = t[g & (t.n == n)]
    for c, ls in (("tow_real", "-"), ("tow_model", "--")):
        v = s.groupby(["fish", "range"], observed=True)[c].mean().groupby("range", observed=True).mean()
        ax.plot([str(i) for i in v.index], v.values, ls, color=col, label=f"{n} dot{'s' if n > 1 else ''}" if c == "tow_real" else None)
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("range (mm)"); ax.set_ylabel("toward-turn (deg)"); ax.legend(fontsize=7)
plt.tight_layout()""")])

# ---------------------------------------------------------------- 4
sec("s4", "4. The extrapolations replicated", """**Question.** Do the two extrapolations (two dot groups at once; several
fish seeing each other) hold on M6 itself and at the second seed, as the claim rule requires?

**What was done.** The two-group scenes (v2: references, mirror pairs) re-run on M6 at both seeds (26e, 30d) and on F at
seed 101 (32b), next to the original M6+C (28b) and F seed 0 (14f). The multi-fish runs repeated on M6 at both seeds
(pairs, groups of 10) and, for F at seed 101, on the 5× dish, the random-direction control and groups of 5.

**How to read.** A pattern is claimed only if it has the same sign for every model and seed.""",
    [("s4-two", "cell", """# Two groups: key readouts per model (fish-level means). P_between = share of trials ending between the groups;
# pref = P(choose A) - P(choose B) (A = the first group of the scene).
import multi_stim as MS
tags = [t for t in ("28b", "14f", "26e", "30d", "32b") if all(MS._path(t, c, s).exists() for c in ("effects", "nearest") for s in ("ang180", "n2v8", "g2_50_8_100", "mv4_st4"))]
T2 = MS.summary_table(tags=tuple(tags))
key = [("ang180", "P_between", "4 v 4 opposite: share between"), ("n2v8", "pref", "2 v 8 same distance: pref for 2"),
       ("g2_50_8_100", "pref", "2 near v 8 far: pref for 2"), ("mv4_st4", "pref", "4 moving v 4 static: pref for moving")]
rows = []
for sc, col, lab in key:
    for rule in ("effects", "nearest"):
        r = {"readout": lab, "rule": rule}
        for t in tags:
            v = T2[(T2.model == t) & (T2.scene == sc) & (T2.cond == rule)]
            r[MS.MODEL_LABEL.get(t, t).split(":")[0]] = float(v[col].iloc[0]) if len(v) else np.nan
        vals = [r[k] for k in r if k not in ("readout", "rule") and np.isfinite(r[k])]
        r["same sign everywhere"] = (len(vals) > 1 and (all(v > 0 for v in vals) or all(v < 0 for v in vals))) if col == "pref" else "-"
        rows.append(r)
print("models:", tags)
print(pd.DataFrame(rows).set_index(["readout", "rule"]).round(3).to_string())"""),
     ("s4-fish", "cell", """# Multi-fish: pairs (attraction index) and groups (D = ln(NN / NN of fish from different dishes)) per model and seed.
import multi_fish as MFish
Sm = json.load(open(P.OUT / "multi_fish_summary.json"))
rows = []
for c, d in Sm["pairs"].items():
    rows.append({"condition": MFish.LABEL[c], "readout": "pair attraction index",
                 "value [95% CI]": f"{d['coupled']['ai_mean']:+.3f} [{d['coupled']['ai_ci'][0]:+.3f}, {d['coupled']['ai_ci'][1]:+.3f}]",
                 "blind (same fish)": f"{d['blind']['ai_mean']:+.3f}"})
for c, d in Sm["groups"].items():
    rows.append({"condition": MFish.LABEL[c], "readout": "D (groups)",
                 "value [95% CI]": f"{d['D_mean']:+.3f} [{d['D_ci'][0]:+.3f}, {d['D_ci'][1]:+.3f}]",
                 "blind (same fish)": f"{d['D_blind_mean']:+.3f}"})
print(pd.DataFrame(rows).set_index(["condition", "readout"]).to_string())""")])

# ---------------------------------------------------------------- 5
sec("s5", "5. Rollout-stream noise, and F against M6", """**Question.** How much do the profile readouts move when the same
model is simply rolled out again with other random numbers? Is F's advantage over M6 on the profiles beyond that noise?

**What was done.** The matched profile (dip, latency, plateau, grading, moving and static error; 10 simulated fish per
fold model) and the attraction per count (the first 6 of those per fold model, as the units' own attraction cell), for
each headline model on 4 rollout streams. Stream 0 is the archived stream, so its row must reproduce the saved numbers;
streams 1–3 are new. The stream SD is the noise of ONE reading; the archived F − M6 difference (one stream each) is read
against √2 × SD.""",
    [("s5-tab", "cell", """St = P.stream_table(filtered=True)
cols = ["dip", "lat", "plateau", "d1_plateau", "grading", "mv_err_sem", "st_err_sem", "attr_N1", "attr_N2", "attr_N8"]
M = St[St.model != "real"]
print("per stream:"); print(M.set_index(["model", "stream"])[cols].round(2).to_string())
print("\\nmean and SD over the 4 streams:")
print(M.groupby("model")[cols].agg(["mean", "std"]).round(2).T.to_string())
print("\\nstream 0 against the archived numbers (must agree):")
for lab, (d, arm, key) in P.HEAD.items():
    s = S(key); r = M[(M.model == lab) & (M.stream == 0)].iloc[0]
    print(f"  {lab}: dip {r.dip:.3f} vs {s.get('dip')}, grading {r.grading:.3f} vs {s.get('grading')}, moving err {r.mv_err_sem:.3f} vs "
          f"{s.get('mv_err_sem')}, attraction 2 / 8 dots {r.attr_N2:.2f} / {r.attr_N8:.2f} vs {s.get('attr_N2')} / {s.get('attr_N8')}")"""),
     ("s5-fvm6", "cell", """# F - M6 per seed against the stream noise. Two readings: the ARCHIVED difference (stream 0, one reading per model), whose
# noise is sqrt(2) x the single-reading SD; and the difference of the 4-stream MEANS, whose noise is that divided by 2.
# Ratio > 2 = beyond the noise.
St = P.stream_table(filtered=True); M = St[St.model != "real"]
cols = ["dip", "grading", "mv_err_sem", "st_err_sem", "attr_N1", "attr_N2", "attr_N8"]
mean = M.groupby("model")[cols].mean(); sd = M.groupby("model")[cols].std()
s0 = M[M.stream == 0].set_index("model")
rows = []
for seed in ("s0", "s101"):
    a, b = f"M6 {seed}", f"F {seed}"
    for c in cols:
        noise = np.sqrt((sd.loc[a, c] ** 2 + sd.loc[b, c] ** 2) / 2)          # SD of ONE reading
        d0 = s0.loc[b, c] - s0.loc[a, c]; dm = mean.loc[b, c] - mean.loc[a, c]
        rows.append({"seed": seed, "readout": c, "archived F - M6": d0, "ratio (archived)": abs(d0) / (np.sqrt(2) * noise),
                     "4-stream F - M6": dm, "ratio (4-stream means)": abs(dm) / (np.sqrt(2) * noise / 2), "SD of one reading": noise})
print(pd.DataFrame(rows).set_index(["seed", "readout"]).round(3).to_string())""")])

# ---------------------------------------------------------------- 6
sec("s6", "6. Data provenance", """**Question.** Does the current segmentation code reproduce the bout table the training data
were cleaned from? The training table was built before the last changes to `scripts/segment_bouts.py`.

**What was done.** The current code run on the raw tracking (into `Model/missing_pieces/`), compared bout by bout with
`data/bouts_GsizeMotionSingle.csv` (the segmentation output the training table came from) and with
`clean_data/bouts_clean.csv` (the training table), onsets matched within 0 and 2 frames.""",
    [("s6-tab", "cell", """O = P.load("provenance")
if "error" in O:
    print(O["error"])
else:
    print("file dates:", O["mtimes"]); print("last commit touching segment_bouts.py:", O["last_commit"])
    pf = O["per_fish"]
    print(pf.to_string(index=False))
    tot = pf.sum(numeric_only=True)
    print(f"\\nTOTAL: old {int(tot.old_table)}, regenerated {int(tot.regenerated)}, training {int(tot.training_table)}; "
          f"regenerated onsets identical to the old table {100 * tot.regenerated_matching_old / tot.regenerated:.2f} %, "
          f"within 2 frames {100 * tot.regenerated_matching_old_2fr / tot.regenerated:.2f} %; "
          f"training bouts with a regenerated onset within 2 frames {100 * tot.training_matching_regenerated_2fr / max(tot.training_table, 1):.2f} %")
    print(O["stats"].round(4).to_string())
    print("\\nmatched bouts (old vs regenerated, onsets within 2 frames): share whose value changed beyond 1 deg / 0.5 px / 0.04 s")
    print(O["matched_changes"].round(4).to_string())"""),
     ("s6-diff", "cell", """# The uncommitted change to the segmentation code since the last commit (what differs from the code that built the table).
print(P.load("provenance").get("diff", "")[:6000] or "(no uncommitted change)")""")])

# ---------------------------------------------------------------- 7
sec("s7", "7. Paired-by-fish intervals on the key likelihood differences", """**Question.** Are the likelihood
differences the story rests on beyond the fish-to-fish variation?

**What was done.** For each comparison, the per-fish difference in held-out bits per bout (B − A: negative = B better),
mean over the 15 fish with a 95 % bootstrap interval over fish, per head, at each seed.""",
    [("s7-tab", "df", """P.paired_ci()""")])

# ---------------------------------------------------------------- 8
sec("s8", "8. Which point of a dot group the fish turns toward", """**Question.** The stimulus code describes a dot group by its
centroid. Do real fish turn toward the centroid, the nearest dot, or a distance-weighted centroid?

**What was done.** Real data only: bouts with ≥ 2 dots, |turn| > 3°, centroid 5–60 mm away. The candidates usually lie
on the same side of the fish; only bouts where two of them lie on **different** sides can tell them apart. On those, the
share of turns toward each, per fish (95 % CI over fish), moving and static separately.""",
    [("s8-tab", "df", """P.centroid_table()""")])

# ---------------------------------------------------------------- 9
sec("s9", "9. Multi-fish extras", """**Question.** Where do neighbours sit around a fish (real juveniles keep a short-range gap
and an attraction ring at 10–30 mm, Kappel et al. 2022)? Does the shoaling in the 5× dish survive the random-direction
control, and what changes when the 'dots just appeared' clock restarts each time a neighbour comes into view (so F's
correction acts)?

**What was done.** Neighbour positions in each fish's own frame (x = its travel direction), 1 Hz over the readout window,
divided by the same map with the neighbours' tracks shifted in time (chance = 1). New runs in the 5× dish: random
directions; clock restarted on entry.""",
    [("s9-maps", "cell", """fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, c in zip(axes[:3], ("A1_pairs", "A1_N10_effects", "A5_N10_effects")):
    ratio, edges, prof = P.neighbour_map(c)
    im = ax.imshow(ratio.T, origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]], cmap="RdBu_r", vmin=0, vmax=3)
    ax.set_title(c, fontsize=9); ax.set_xlabel("ahead (mm)"); ax.set_ylabel("left (mm)")
    axes[3].plot(prof.index, prof.values, label=c)
axes[3].axhline(1, color="k", lw=0.6); axes[3].set_xlabel("distance to a neighbour (mm)"); axes[3].set_ylabel("density / chance")
axes[3].legend(fontsize=7); fig.colorbar(im, ax=axes[:3], shrink=0.8, label="density / chance")"""),
     ("s9-tab", "cell", """import multi_fish as MFish
Sm = json.load(open(P.OUT / "multi_fish_summary.json"))
rows = []
for c in ("A5_N10_effects", "A5_N10_effects_scrambled", "A5_N10_effects_entry", "s101_A5_N10_effects", "A1_N10_effects", "A1_N10_effects_scrambled", "s101_A1_N10_effects_scrambled"):
    d = Sm["groups"].get(c)
    if d is None:
        continue
    ct = Sm["cotravel"].get(c, {})
    rows.append({"condition": MFish.LABEL[c], "D [95% CI]": f"{d['D_mean']:+.3f} [{d['D_ci'][0]:+.3f}, {d['D_ci'][1]:+.3f}]",
                 "largest shoal (of 10)": round(d["big_mean"], 2), ">= half together": round(d["share_half_together"], 2),
                 "co-travel m / pair-h (chance)": (f"{ct['coupled']['per_pair_hour'] / 1000:.2f} ({ct['chance']['per_pair_hour'] / 1000:.2f})" if ct else "-")})
print(pd.DataFrame(rows).set_index("condition").to_string())""")])

# ---------------------------------------------------------------- 10
sec("s10", "10. Robustness numbers", """Four small checks the text needs.
- **Unfiltered profiles.** The profiles drop bouts whose dot centroid lies outside the dish (as for the real data). The
  same rollouts read without the filter, real and model.
- **No-radial floor.** The real-vs-real AUC without the four radial features (the floor of the no-radial reading).
- **Bits saved over the pooled marginals** (from section 1).
- **Two length scales.** The profile, attraction and two-group code converts with 3.67 px/mm, everything else with 3.60.""",
    [("s10-unf", "cell", """U = P.stream_table(filtered=False); F_ = P.stream_table(filtered=True)
cols = ["dip", "plateau", "d1_plateau", "grading", "mv_err_sem", "st_err_sem"]
a = F_[F_.stream.astype(str).isin(["0"]) | (F_.model == "real")].assign(reading="filtered")
b = U[U.stream.astype(str).isin(["0"]) | (U.model == "real")].assign(reading="unfiltered")
print(pd.concat([a, b]).set_index(["reading", "model", "stream"])[cols].round(2).to_string())"""),
     ("s10-floor", "cell", """R = P.load("refs")["table"]
print(R.loc[["real vs real (random halves of each fish)"], ["all features", "no radial", "three heads"]].round(3).to_string())
B = P.baselines_table().reset_index()
print(B[B.model.isin(list(P.HEAD))][["seed", "model", "joint", "saved_vs_pooled_joint"]].to_string(index=False))"""),
     ("s10-px", "cell", """print(pd.DataFrame([
    {"quantity": "profiles: dip, plateau, grading, errors in mm; attraction; two-group distances", "px per mm": 3.67,
     "to the thesis scale (3.60)": "multiply by 3.67 / 3.60 = 1.0194"},
    {"quantity": "Ch. 3, individuality, multi-fish, missing-pieces sections 2, 3, 8", "px per mm": 3.60, "to the thesis scale (3.60)": "-"},
]).to_string(index=False))
print("real profile targets at 3.60: dip", round(4.23 * 1.0194, 2), "mm, plateau", round(17.26 * 1.0194, 2), "mm, grading", round(6.00 * 1.0194, 2), "mm")""")])

# ---------------------------------------------------------------- 11
sec("s11", "11. Leave one condition out: predicting 6 moving dots", """**Question.** Can the model predict a stimulus it was never
trained on? The only test of the extrapolation idea the data allow.

**What was done.** M6's exact recipe retrained at both seeds with every bout shown **6 moving dots** given zero weight in
the loss (8.0 % of bouts; the bouts stay in the records, so the history after them is real). The dot count enters as
log2(n)/3, so 6 lies between the trained 4 and 8: an interpolation in count. Read on the held-out fish: the likelihood
of the 6-moving-dot bouts (and of every other condition, as a control), the attraction per count and the matched
profile.

**How to read.** If the model generalises, the held-out condition costs little likelihood and its 6-dot response sits
where the trained one does.""",
    [("s11-lik", "df", """P.loco_table().round(4)"""),
     ("s11-sum", "cell", """cols = ["lik_joint", "auc_matched", "dip", "grading", "mv_err_sem", "attr_N4", "attr_N6", "attr_N8"]
rows = [dict(model=l, **{c: S(k).get(c) for c in cols}) for l, k in (("M6 s0", "26e"), ("trained without 6 moving, s0", "34e"),
                                                                     ("M6 s101", "30d"), ("trained without 6 moving, s101", "34f"))]
print(pd.DataFrame(rows).set_index("model").round(3).to_string())
print("real attraction (mm): 4 dots 5.19, 6 dots 6.43, 8 dots 6.94")"""),
     ("s11-prof", "cell", """# The matched profiles as saved by the units (M6 and the held-out model, seed 0): the 6-moving-dot panel is the prediction.
import base64, io
from PIL import Image
for key in ("26e", "34e"):
    blob = json.load(open(f"report_outputs/M{key}.json"))
    cell = next(c for c in blob["cells"] if c["cid"] == "matched")
    img = next(o for o in cell["outputs"] if o.get("output_type") == "display_data")
    im = Image.open(io.BytesIO(base64.b64decode(img["data"]["image/png"])))
    fig, ax = plt.subplots(figsize=(15, 6.6)); ax.imshow(im); ax.axis("off"); ax.set_title(f"unit {key}", fontsize=10)""")])

# ---------------------------------------------------------------- 12
sec("s12", "12. The main line at seed 101", """**Question.** The main line's first steps (M1–M3) and the check M5+T existed at seed
0 only. Do they hold at seed 101?

**What was done.** M1 (= M2's weights without wall rejection), M3 and M5+T trained at seed 101 on the same recipes
(units 34a–34d), next to their seed-0 units. Steps are read within a seed. At seed 0 the M1–M3 profile numbers come
from the old rollout streams (no per-block-stream units exist for them), so across the M1–M4 rows only the likelihood
and AUC columns compare like with like at seed 0; at seed 101 every unit is on per-block streams.""",
    [("s12-tab", "cell", """cols = [("lik_joint", 3), ("auc_matched", 3), ("auc_std", 3), ("dip", 2), ("grading", 2), ("mv_err_sem", 2), ("st_err_sem", 2)]
line = [("M1", "1", "34a"), ("M2", "2", "34b"), ("M3", "3", "34c"), ("M4", "20f", "30a"), ("M5", "25d", "30b"),
        ("M5+T", "29b", "34d"), ("M6", "26e", "30d")]
rows = []
for name, k0, k1 in line:
    a, b = S(k0), S(k1)
    rows.append({"model": name, **{f"{c} s0 | s101": f"{a.get(c, np.nan) if a.get(c) is not None else np.nan:.{nd}f} | {b.get(c, np.nan) if b.get(c) is not None else np.nan:.{nd}f}" for c, nd in cols}})
print(pd.DataFrame(rows).set_index("model").to_string())
print()
steps = [("M1 -> M2 (wall rejection)", "1", "2", "34a", "34b"), ("M2 -> M3 (+ latent)", "2", "3", "34b", "34c"),
         ("M3 -> M4 (s0: both on the old stream)", "3", "4", "34c", "30a"), ("M5 -> M5+T", "25d", "29b", "30b", "34d")]
rows = []
for lab, a0, b0, a1, b1 in steps:
    r = {"step": lab}
    for c, nd in cols:
        d0 = (S(b0).get(c) or np.nan) - (S(a0).get(c) or np.nan); d1 = (S(b1).get(c) or np.nan) - (S(a1).get(c) or np.nan)
        r[c] = f"{d0:+.{nd}f} | {d1:+.{nd}f}" + ("" if not (np.isfinite(d0) and np.isfinite(d1)) else ("" if np.sign(d0) == np.sign(d1) else "  (sign differs)"))
    rows.append(r)
print("steps (s0 | s101):"); print(pd.DataFrame(rows).set_index("step").to_string())""")])

# ---------------------------------------------------------------- 13
sec("s13", "13. Capacity: 64 and 256 hidden units", """**Question.** Was one hidden layer of 128 units a reasonable size?

**What was done.** M6's exact recipe with 64 and with 256 hidden units, both seeds (units 34g–34j), against M6. Note: the
64-unit fits often run into the 80-epoch training limit (printed below) while the 256-unit fits stop early, so part of a
64-unit deficit can be the training budget rather than the capacity.""",
    [("s13-tab", "cell", """cols = ["lik_turn", "lik_displacement", "lik_interval", "lik_joint", "auc_matched", "auc_std", "dip", "grading", "mv_err_sem", "st_err_sem"]
rows = []
for lab, k in (("64 units, s0", "34g"), ("128 units (M6), s0", "26e"), ("256 units, s0", "34i"),
               ("64 units, s101", "34h"), ("128 units (M6), s101", "30d"), ("256 units, s101", "34j")):
    rows.append(dict(model=lab, **{c: S(k).get(c) for c in cols}))
print(pd.DataFrame(rows).set_index("model").round(3).to_string())
import missing_pieces as _P
for d, lab in (("missing_ladder_checkpoints", "s0"), ("missing_seed101_ckpts", "s101")):
    try:
        man = _P.manifest(d)
        man = pd.DataFrame(man)
        print(f"epochs run ({lab}):", man.groupby("arm").epochs_run.apply(list).to_dict())
        print(f"  folds stopped by the 80-epoch limit ({lab}):", man.assign(cap=man.epochs_run >= 80).groupby("arm").cap.sum().to_dict())
    except Exception as e:
        print(lab, e)""")])


def build():
    ns = C.new_ns()
    cells = [md(INTRO, "intro"), md(FINDINGS, "findings"), md(TOC, "toc"), code(SETUP, "setup", C.run_cell(ns, SETUP))]
    for s in SECTIONS:
        cells.append(md(f'<a id="{s["id"]}"></a>\n## {s["title"]}\n\n{s["md"]}', f'{s["id"]}-md'))
        for cid, kind, src in s["cells"]:
            outs = C.run_df(ns, src) if kind == "df" else C.run_cell(ns, src)
            cells.append(code(src, cid, outs))
    C.write_nb(NB, cells)


if __name__ == "__main__":
    build()
