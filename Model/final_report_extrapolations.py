"""Build Model/Final_report_extrapolations.ipynb: the in-silico experiments beyond the recorded data -- two dot groups at
once (report Model 31) and several simulated fish seeing each other as dots (planned) -- from the saved results of
report_models.py. Nothing is re-run.

    python final_report_extrapolations.py
"""
import json
import sys
from pathlib import Path

import final_report_common as C
from final_report_common import md, code

NB = C.HERE / "Final_report_extrapolations.ipynb"

INTRO = f"""# Final report extrapolations

Stimuli the fish were never shown, simulated with the trained models: **two dot groups at once** and **several simulated
fish in one dish, each seeing the others as dots**. The data contain neither, so these are predictions: what the models
do is fixed by the rule that turns the scene into the network's input, and every result is read against references on
the same scenes (a fish shown one group, a fish shown neither).

{C.LINKS}"""

TOC = """## Contents

1. [Two dot groups at once](#two-groups)
2. [Several fish: do they shoal?](#multi-fish) — yes; pairs, groups of 5 and 10, a 5× dish, videos
3. [With repulsion](#repulsion) — F + the turn-away term fitted to the Larsch screen data, the same scenes"""

MULTI_FISH_XX = """<a id="multi-fish"></a>
## 2. Several fish: do they shoal?

XX — not run yet (scripts/multi_fish.py)."""

MF_INTRO = """<a id="multi-fish"></a>
## 2. Several fish: do they shoal?

**Question.** If simulated fish see each other the way the real fish saw the projected dots, do they form shoals — and
do the shoals move together?

**How each fish sees the others** (`scripts/multi_fish.py`). Every fish runs the final model F unchanged; only what it is
shown is new. At each of its bouts it sees every other fish within 78 mm, at any bearing, always as **moving** dots. 78 mm
is where the model's log-range input saturates: beyond it the model's pull would stay constant rather than fade, so the
horizon is our choice (it keeps every input inside the trained values); in the training dish it hides only the rare
neighbour 78–98 mm away. Neighbours that are close to each other are pooled into one dot group of that
size — 3 fish close together = a 3-dot group at their centre — as long as they are no more spread out than a trained dot
group of that size was (≤ 10 mm from the centre for 2 dots … ≤ 21 mm for 8); the count is capped at 8. The groups are
combined by the two rules of the two-group experiment: **effects** (each group's effect added) and **nearest** (the
nearest group only). The 'dots just appeared' clock is held at its late-block value (a neighbour has no onset), so F's
correction term — which acts only in the first 20 s of a block — is inactive here; F's network weights are its own.

**Coupling.** The fish are simulated together: the fish whose next bout is earliest moves next and sees the others where
they are at that moment. Collisions are ignored (dots overlapped freely in training). Bouts are instantaneous.

**Pairing.** Every fish of a seed is fixed by (dish, fish): the same start, latent and random stream in every condition,
so conditions pair dish by dish, and a pair is fish 0–1 of the corresponding blind dish of 10.

**How to read it.**
- **Nulls are recomputed from our own simulations**, as in the papers: for pairs, each fish against its partner's track
  shifted in time by 1–10 min (Larsch & Baier 2018); for groups, pseudo-groups of fish taken from *different* dishes of
  the same condition, overlaid in time (Harpaz et al. 2021). Both keep each fish's own space use and kinematics.
- **D = ln(nearest-neighbour distance / that of the pseudo-groups)**: < 0 = closer together than chance. The blind fish
  give the occupancy part: fish that see moving neighbours also change where they swim in the dish, which moves
  distances even without attraction to particular fish.
- **Random directions** (control): the same scene, but every neighbour group is shown at a random bearing, redrawn every
  bout (same range, count and number of groups). Shoaling that survives this is not attraction.
- Readouts start at minute 5 (training dish) or 10 (5× dish). Seed 0 and seed 101 are separate fold sets, never pooled.
- Distances in mm at 3.60 px/mm."""

MF_UNITS = [
    ("mf-setup", None, "cell", """# Several fish: everything below is read from Model/multi_fish/ (scripts/multi_fish.py --run / --analyse)
import json
import multi_fish as MFish
S = MFish.load_summary()"""),
    ("mf-conds", "### 2.1 The conditions", "df", "MFish.conditions_table(S)"),
    ("mf-checks", """### 2.2 Checks

The simulator itself: the new clock option changes nothing when unused, a coupled fish is exactly the generator run on a
replay of its partner, runs are reproducible, and fish pair across conditions.""", "df", "MFish.checks_table()"),
    ("mf-rules", """**Rule check.** Phantom moving neighbours on one side (10–25 mm, 60–120°, re-placed every bout), pooled and combined
exactly as in the shoal runs: does the fish turn toward them as real fish turn toward moving dots, and do its kinematics
stay sane? (k = 0: the same fish with nothing in view.)""", "df", "MFish.rules_table()"),
    ("mf-pairs", "### 2.3 Pairs", "df", "MFish.pairs_table(S)"),
    ("mf-pairs-fig", None, "cell", "MFish.plot_pair_distance();"),
    ("mf-groups", "### 2.4 Groups of 5 and 10 (training dish)", "df", "MFish.groups_table(S)"),
    ("mf-nn-fig", None, "cell", """MFish.plot_nn_time(S, [c for c in S["groups"] if MFish.CONDS[c]["arena"] == 1 and MFish.CONDS[c]["tag"] == "14f"]);"""),
    ("mf-D-fig", None, "cell", "MFish.plot_D(S);"),
    ("mf-big", """### 2.5 The 5× dish (radius 244 mm, 60 min)

Blind fish use the inner part of this dish (90 % of bouts within ~100 mm of the centre, against 36 mm in the training
dish), so fish can lose sight of each other (78 mm) and meet again, and a shoal has room to travel. Every wall input of the
model is relative to the dish radius (the distance (R − r)/R, the rejection band, the start positions), so the fish
senses the wall as in the training dish magnified 5×; steps, the 78 mm view and the 25 mm shoal criterion are absolute.""", "cell",
     """MFish.plot_nn_time(S, [c for c in S["groups"] if MFish.CONDS[c]["arena"] == 5]);"""),
    ("mf-cotravel", "**Travelling together.** Episodes in which two fish stay within 25 mm of each other; the distance their midpoint covers.",
     "df", "MFish.cotravel_table(S)"),
    ("mf-cotravel-fig", None, "cell", "MFish.plot_cotravel();"),
    ("mf-lit", """### 2.6 Next to the literature

Same definitions, nulls recomputed from our simulations. The papers' fish are real conspecifics of known age; ours are a
dot-trained model, and the age of the recorded fish is unknown (the stimulus protocol and the real dot response suggest
juveniles, roughly 2–3 weeks).""", "df", "MFish.literature_table(S)"),
    ("mf-kin", """### 2.7 Kinematics and inputs

Each fish against itself blind (same start, latent, random stream), next to what real fish do when moving dots appear
(moving blocks vs pauses, per fish, median [10th, 90th percentile]).""", "df", "MFish.kin_table(S)"),
    ("mf-ood", "What the fish were shown (provider log, readout window).", "df", "MFish.ood_table(S)"),
]

MF_RESULTS = """### In short

**Yes — the model's fish shoal when they see each other, and in a large dish the shoals travel together.** Final model F
at seed 0 unless stated; every result below holds at seed 101 where it was run.

- **Pairs stay together.** The two fish are a median 14.5 mm apart, against 26.1 mm when the partner's track is shifted
  in time; attraction index +0.35 [0.33, 0.36] (seed 101: +0.32 [0.30, 0.34]). The same two fish blind: 29.2 mm, index
  0.00. Every fold of both seeds is positive (+0.27 to +0.37).
- **Groups are closer together than chance.** Training dish: D = −0.30 (5 fish) and −0.20 (10 fish), the same under
  'nearest' (−0.29, −0.20); seed 101 (10 fish): −0.18. Blind fish: D ≈ 0. The interval of every coupled condition is
  below 0, and so is D against a within-group time shift and against a fold-matched null.
- **It is attraction, not a side effect of the rule.** With every neighbour group shown in a random direction (same
  range, count and number of groups), D = −0.01 [−0.02, 0.00]: the aggregation needs the neighbours' direction. On top
  of the attraction, fish that see neighbours gather toward the dish centre (occupancy part −0.12 to −0.33 in ln
  distance); the null keeps that part out of D.
- **In the 5× dish, shoals form from a scattered start and move as one.** D = −0.76 [−0.89, −0.57] ('effects') and
  −0.40 [−0.47, −0.31] ('nearest'). Under 'effects' the largest shoal holds 8.3 of the 10 fish on average (fish of
  different dishes overlaid: 5.1; blind: 3.8), and half the fish or more are together 88 % of the time (59 %, 29 %). Two
  fish travel 4.7 m per pair-hour while staying within 25 mm of each other, against 1.3 m for fish of different dishes
  and 0.75 m blind; single episodes reach 0.5 m. In the formation video all 10 fish are together within 3 min of a
  scattered start.
- **Rules.** 'effects' (each neighbour group adds its pull) gives tighter, more persistent shoals than 'nearest' in the
  large dish; in the training dish the two tie. The fold models differ in how tight their shoals get in the large dish
  (D per fold −0.31 to −1.15), all negative.

**Next to the literature** (same definitions, nulls recomputed from our simulations; table 2.6). The training-dish D
(−0.18 to −0.30) lies between Harpaz et al.'s 14-dpf (−0.10) and 21-dpf (−0.55) real groups and far from their 7-dpf
larvae, which are slightly over-dispersed (+0.08). The pair attraction is weaker than Larsch & Baier's 26-dpf pairs (14.5
against 26 mm here; 5–20 against 40 mm there). Alignment barely exceeds chance (5 fish: 0.418 against 0.402), the level
of Harpaz's 7-dpf fish — the model has no input for a neighbour's heading, so it cannot align except by following. This
is juvenile-like aggregation, consistent with what the protocol and the real dot response suggest (roughly 2–3 weeks);
an age cannot be read off it, because D depends on the arena (−0.2 here, −0.4 to −0.76 in the 5× dish) and on the rule.

**Limits.** To the model a neighbour is a moving dot: no body, no heading, never pausing (real fish are still ~30 % of
the time), and collisions are ignored. In groups of 10 in the training dish the nine neighbours are often pooled into
one group, and on 39 % of bouts that pool has 9 fish shown as 8 dots — beyond the trained counts. Fish in groups take
steps 7–16 % longer than blind (real fish +3 % when moving dots appear, 90th percentile +8 %). The 78 mm view is a
choice (the model's pull does not fade there). In the 5× dish the model's wall cues are those of the training dish
magnified 5×, and no experiment has a counterpart. One model at two seeds; F's correction term is inactive with the
late clock."""
MF_VIDEO_MD = """Each video shows one dish: **left**, the 10 fish seeing each other; **right**, the same 10 fish (same starts, latents
and random streams) blind. Bouts are drawn as 0.2 s glides (the model's bouts are instantaneous) with the last 4 s as a
trail; the 5× videos are zoomed onto the fish (the width of the view is printed in the corner). (1) the dish where all
10 fish first come together from a scattered start; (2) the stretch where a shoal's centre travels furthest; (3) training
dish: the 90 s with the smallest nearest-neighbour distance after minute 5. The videos play when the notebook is opened
from the `Model` folder; stills of the same windows are below."""
MF_NO_VIDEO_MD = """No condition met the video criterion (95 % interval of D below 0 and D below −0.1)."""


def multi_fish_cells():
    sys.path.insert(0, str(C.HERE.parent / "scripts"))
    import multi_fish as MFish
    if not (MFish.OUT / "summary.json").exists():
        return [md(MULTI_FISH_XX, "multi-fish-md")]
    cells = [md(MF_INTRO, "multi-fish-md"), md(MF_RESULTS, "mf-results")]
    ns = C.new_ns()
    for cid, text, kind, src in MF_UNITS:
        if text:
            cells.append(md(text, cid + "-md"))
        outs = C.run_df(ns, src) if kind == "df" else C.run_cell(ns, src)
        cells.append(code(src, cid, outs))
    vids = MFish.VIDEO_DIR / "videos.json"
    V = json.loads(vids.read_text(encoding="utf-8")) if vids.exists() else []
    if V:
        tags = "\n\n".join(f'**{v[5]}** — {MFish.LABEL[v[1]]}, dish {v[2]}, {v[3] / 60:.1f}–{v[4] / 60:.1f} min '
                           f'(`{Path(v[0]).relative_to(C.HERE).as_posix()}`)\n\n'
                           f'<video src="{Path(v[0]).relative_to(C.HERE).as_posix()}" controls width="980"></video>'
                           for v in V)
        cells.append(md("### 2.8 Videos\n\n" + MF_VIDEO_MD + "\n\n" + tags, "mf-videos-md"))
        src = """import matplotlib.pyplot as plt
V = json.load(open(MFish.VIDEO_DIR / "videos.json"))
for mp4, c, g, t0, t1, title in V:   # stills of each video window: top = the fish see each other, bottom = the same fish blind
    view = MFish.view_of([(c, g), (MFish.blind_of(c), g)], t0, t1)
    fig, ax = plt.subplots(2, 5, figsize=(16, 6.9))
    MFish.snapshot(c, g, list(np.linspace(t0, t1, 5)), ax[0], view=view)
    MFish.snapshot(c, g, list(np.linspace(t0, t1, 5)), ax[1], blind=True, view=view)
    fig.suptitle(f"{title} -- top: the fish see each other; bottom: the same fish blind (dish {g})", fontsize=11)"""
        cells.append(code(src, "mf-video-stills", C.run_cell(ns, src)))
    else:
        cells.append(md("### 2.8 Videos\n\n" + MF_NO_VIDEO_MD, "mf-videos-md"))
    return cells

REP_INTRO = """<a id="repulsion"></a>
## 3. With repulsion: F + the turn-away term fitted to the Larsch screen data

**What changes.** The shoals of section 2 are made by F, which has no short-range repulsion: on the Larsch lab's
single-dot screen (Final_report_extra_data.ipynb, section 2) the models keep turning *toward* a dot closer than 4 mm,
where real fish turn away. Section 4 of that notebook fits a fixed-form turn-away to those data (network and profile
correction untouched; fitted on half the screen's clutches, tested on the other half) and retains **F + turn**: a turn
away from any dot within reach, *s* = −0.3 Σ_j sin φ_j σ((*r_c* − *d_j*)/1 mm) with *r_c* = 8 mm (seed 0) / 9 mm
(seed 101). It passes on the held-out clutches and leaves the likelihood and AUC on our data unchanged.

**How it enters the shoal.** Every simulated fish adds the term for **each neighbour individually** (the individual
positions, `FishView.dots`), while attraction still comes through the dot-group code exactly as in section 2. Same
scenes, start poses, latents and random streams as F. The blind pools are F's: a blind fish sees no neighbour, so the
term never acts (checked: byte-identical).

**Expected, written before the runs.** (1) Fewer close contacts — the share of time a fish's nearest neighbour is within
a few mm should drop clearly. (2) Aggregation (attraction index, D, largest cluster, co-travel) similar or slightly
weaker, with the nearest-neighbour distance shifted up by a few mm — on the screen data the term did not make the fish
hover near the dot. (3) No polarisation: the term carries no heading. (4) Neighbours sum and a fish is a larger stimulus
than a 2 mm dot, so the effect may be stronger than on the screen."""

REP_TABLE = """import json, numpy as np, pandas as pd
import multi_fish as MFish
SR = json.load(open(MFish.OUT / "summary_rep.json"))
CC, BASE = SR["contacts"], SR["rep_base"]
rows = []
for rc, bc in BASE.items():
    seed = "101" if "s101" in rc else "0"
    scene = "pairs" if "pairs" in rc else ("10 fish, 5x dish" if "A5" in rc else "10 fish, training dish")
    sc = "reps_" + rc[len("rep_"):]
    has_s = sc in CC
    def add(measure, f, r, q=None, nd=3):
        row = dict(scene=scene, seed=seed, measure=measure, F=round(f, nd), **{"F + turn": round(r, nd)})
        if has_s and q is not None:
            row["F + turn + step"] = round(q, nd)
        rows.append(row)
    for k, lab in (("share_nn_le_2mm", "nearest neighbour <= 2 mm (share of time)"),
                   ("share_nn_le_4mm", "nearest neighbour <= 4 mm"), ("share_nn_le_6mm", "nearest neighbour <= 6 mm"),
                   ("share_nn_le_10mm", "nearest neighbour <= 10 mm"), ("nn_p10", "nearest neighbour, 10th percentile (mm)"),
                   ("nn_median", "nearest neighbour, median (mm)")):
        add(lab, CC[bc][k], CC[rc][k], CC[sc][k] if has_s else None, 3 if "share" in k else 1)
    if rc in SR["pairs"] and bc in SR["pairs"]:
        f, r = SR["pairs"][bc]["coupled"], SR["pairs"][rc]["coupled"]
        q = SR["pairs"].get(sc, {}).get("coupled")
        add("attraction index (pairs)", f["ai_mean"], r["ai_mean"], q and q["ai_mean"])
        add("pair within 25 mm (share of time)", f["share_within_25mm"], r["share_within_25mm"], q and q["share_within_25mm"])
    if rc in SR["groups"] and bc in SR["groups"]:
        f, r = SR["groups"][bc], SR["groups"][rc]
        q = SR["groups"].get(sc)
        add("D (log NND / shuffled; < 0 = shoaling)", f["D_mean"], r["D_mean"], q and q["D_mean"])
        add("largest cluster (fish within 25 mm)", f["big_mean"], r["big_mean"], q and q["big_mean"], 2)
        add("half the fish together (share of time)", f["share_half_together"], r["share_half_together"],
            q and q["share_half_together"])
        add("polarisation (|mean heading|)", f["pol"], r["pol"], q and q["pol"])
    if rc in SR["cotravel"] and bc in SR["cotravel"]:
        q = SR["cotravel"].get(sc)
        add("co-travel episodes per pair-hour", SR["cotravel"][bc]["coupled"]["eps_per_pair_hour"],
            SR["cotravel"][rc]["coupled"]["eps_per_pair_hour"], q and q["coupled"]["eps_per_pair_hour"], 2)
T3 = pd.DataFrame(rows).set_index(["scene", "seed", "measure"])
T3"""

REP_FIG = """import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharey=False)
order = [("A1_pairs", "pairs"), ("A1_N10_effects", "10 fish, training dish"), ("A5_N10_effects", "10 fish, 5x dish")]
x = np.arange(60) + 0.5
for a, (stem, title) in zip(ax, order):
    for seed, ls in (("", "-"), ("s101_", "--")):
        for c, col, lab in ((seed + stem, "C0", "F"), ("rep_" + seed + stem, "C3", "F + turn"),
                            ("reps_" + seed + stem, "C2", "F + turn + step")):
            if c not in CC:
                continue
            h = np.array(CC[c]["hist"], float); h = h / h.sum()
            a.plot(x, h, ls, color=col, label=f"{lab}, seed {'101' if seed else '0'}")
    a.axvspan(0, 8.5, color="r", alpha=.05); a.set_xlim(0, 40); a.set_title(title)
    a.set_xlabel("nearest-neighbour distance (mm)"); a.set_ylabel("share of fish-seconds")
ax[0].legend(fontsize=8); plt.tight_layout()"""

REP_SNAP = """import matplotlib.pyplot as plt
c0, c1, g = "A1_N10_effects", "rep_A1_N10_effects", 0     # the same dish, the same 10 fish (starts, latents, streams)
times = list(np.linspace(600, 1800, 5))
view = MFish.view_of([(c0, g), (c1, g)], times[0], times[-1])
fig, ax = plt.subplots(2, 5, figsize=(16, 6.9))
MFish.snapshot(c0, g, times, ax[0], view=view)
MFish.snapshot(c1, g, times, ax[1], view=view)
fig.suptitle("training dish, seed 0, dish 0 -- top: F; bottom: F + repulsion (same fish, same starts)", fontsize=11)"""


def rep_cells():
    sys.path.insert(0, str(C.HERE.parent / "scripts"))
    import multi_fish as MFish
    if not (MFish.OUT / "summary_rep.json").exists():
        return [md(REP_INTRO + "\n\nXX — not run yet (multi_fish.run_conditions(REP_CONDS); multi_fish.analyse_rep()).",
                   "rep-md")]
    ns = C.new_ns()
    cells = [md(REP_INTRO, "rep-md"), md("### 3.1 F, F + turn and F + turn + step, scene by scene", "rep-table-md"),
             code(REP_TABLE, "rep-table", C.run_df(ns, REP_TABLE)),
             md("### 3.2 Nearest-neighbour distance", "rep-fig-md"), code(REP_FIG, "rep-fig", C.run_cell(ns, REP_FIG)),
             md("### 3.3 The same dish with and without repulsion", "rep-snap-md"),
             code(REP_SNAP, "rep-snap", C.run_cell(ns, REP_SNAP))]
    if REP_SUMMARY:
        cells.append(md(REP_SUMMARY, "rep-summary"))
    return cells


REP_SUMMARY = """### 3.4 Verdict

**Repulsion changes the shoals, but only modestly: fewer close contacts, the same shoaling.** Every prediction written
before the runs holds in direction, at both seeds; the size is small.
- **Close contacts drop by a fifth.** In groups of 10, the time a fish's nearest neighbour is within 4 mm falls
  29 → 24–25 % (training dish) and 20 → 16 % (5× dish), within 2 mm 8.5 → 6.6–6.9 % and 5.7 → 4.4 %; the nearest
  neighbour's median moves out by 0.5–0.7 mm (5.9 → 6.4 mm; 7.6–7.8 → 8.3–8.4 mm). Pairs: within 4 mm 5.3–5.5 → 4.5–4.6 %.
- **Shoaling is unchanged.** Pair attraction index 0.35 → 0.33 (seed 0) and 0.32 → 0.32 (seed 101), intervals
  overlapping; D −0.20 → −0.17 and −0.18 → −0.16 in the training dish (slightly weaker), −0.76 → −0.75 and −0.71 → −0.70
  in the 5× dish; largest cluster and co-travel (≈123 episodes per pair-hour) unchanged.
- **No polarisation** (0.30–0.31 against 0.28 for shuffled groups, as without the term): the term carries no heading.

**Why so small.** The term was fitted to a 2 mm dot under the fish and reaches ~8–9 mm; a neighbour inside a dense group
is also pulled back by the attraction of every other neighbour (the group code), and the fish still approach without
braking (Final_report_extra_data, section 4: the backing off and the hovering are not reproduced). The groups stay
tight — in the training dish the median nearest neighbour is ~6 mm, about one body length of a larva.

**For the report.** The over-clustering that F's shoals show is reduced but not removed by the repulsion the screen data
support; a stronger spacing would need either the missing backing off (being diagnosed) or a larger reach for a
neighbour than for a 2 mm dot — not measurable from single-dot data.

**Adding the step term (F + turn + step, Model 50) partly undoes the spacing gain.** Shorter steps near any neighbour
make a fish linger where it already is, so close contacts come back part of the way: nearest neighbour within 4 mm in
groups of 10, training dish, F 29 % → F + turn 24.5 % → F + turn + step 27 % (seed 0) and 29 → 24 → 25 % (seed 101); 5×
dish 20 → 16 → 18.6 % and 20 → 16 → 17.5 %; median nearest neighbour 5.9 → 6.4 → 6.0 mm. Shoaling is again unchanged
(pair attraction 0.35 | 0.32 for all three; D −0.20 → −0.17 → −0.16 in the training dish), co-travel in the 5× dish drops
by ~8 % (123 → 114–116 episodes per pair-hour: the slower steps near neighbours shorten the joint trips), no
polarisation. For the shoal, **F + turn is the better of the two**: the step term, which fixes the single-dot backing
off, works against spacing when every neighbour slows the fish down."""


def build():
    m31 = next(m for m in C.RM.MODELS if m["k"] == 31)
    cells = [md(INTRO, "intro"), md(TOC, "toc"), code(C.RM.SETUP, "setup"),
             md('<a id="two-groups"></a>\n' + m31["md"].replace("## Model 31 — ", "## 1. "), "two-groups-md")]
    p = C.OUTS / "M31.json"
    n = 0
    if p.exists():
        blob = json.loads(p.read_text(encoding="utf-8"))
        cells += [code(c["source"], f"u31-{c['cid']}", c["outputs"]) for c in blob["cells"]]
        n = 1
    else:
        cells.append(md("XX — not evaluated yet.", "two-groups-xx"))
    cells += multi_fish_cells()
    cells += rep_cells()
    C.write_nb(NB, cells, n, [] if n else ["31"])


if __name__ == "__main__":
    build()
