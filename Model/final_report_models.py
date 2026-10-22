"""Build Model/Final_report_models.ipynb: ONLY the models of the report's storyline, organised by section, each with the
usual cells (held-out likelihood, AUC and individuality, the fairer AUC windowings, the individuality grid, the dot
profile, the matched profile, attraction, turn toward, head analysis) at seed 0 and, where it exists, seed 101.

Nothing is re-run: every model block is copied, outputs included, from the saved results of report_models.py
(report_outputs/M<key>.json). Only the summary tables at the top are computed here, from report_outputs/summary_M*.json.
A missing result is marked XX. When report_models.py adds a unit (e.g. 14f at seed 101), add its key below and re-run:

    python final_report_models.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "scripts"))
import report_models as RM                       # noqa: E402  (SETUP, _run_cell, the saved outputs' location)

NB = HERE / "Final_report_models.ipynb"
OUTS = RM.OUTS

# --------------------------------------------------------------------------- #
# The storyline. units = [(notebook key, seed)]; missing = text for an XX placeholder.
# --------------------------------------------------------------------------- #
S0, S101 = "seed 0", "seed 101"
STORY = [
    dict(id="main", title="1. Main line — closed-loop profiles improve at constant AUC", md="""Each step adds one thing to
the previous model. The AUC gains stop after the latent (Model 3); from Model 4 on, what improves — at both seeds — is the
closed-loop response to the dots (dip, grading, moving and static errors). The main line ends at **26e, the ablation
model** (no drift, no correction).""", models=[
        dict(name="Model 1", anchor="model-1", change="the simplest stimulus code (S0), no latent", units=[("1", S0)],
             missing=[(S101, "not trained at seed 101 (optional: the steps 1 → 3 are 3–5× the seed noise)")],
             md="Baseline. *Profile numbers on the old single random stream (see the note at the top).*"),
        dict(name="Model 2", anchor="model-2", change="+ wall rejection (generation only, the same weights as Model 1)",
             units=[("2", S0)], missing=[(S101, "not needed: the same weights as Model 1")],
             md="Removes the wall artefacts: the gain is in the radial features. *Old stream.*"),
        dict(name="Model 3", anchor="model-3", change="+ per-fish latent", units=[("3", S0)],
             missing=[(S101, "not trained at seed 101 (optional)")],
             md="Individuality: the largest single gain (matched AUC −0.14, 3–5× the noise); between-fish spread "
                "0.1–0.2 → 0.9–1.2 of real. *Old stream.*"),
        dict(name="Model 4", anchor="model-4", change="+ count × bearing and moving × bearing (S4i code)",
             units=[("20f", S0), ("30a", S101)],
             md="The response to the dots appears (dip 0.9 → 2.1 mm, static error 2.8 → 1.1); the AUC change is within "
                "noise. Seed 0 is shown on per-block streams (key 20f); key '4' is the same checkpoint on the old "
                "stream (identical likelihood and AUC), used only for the 3 → 4 step."),
        dict(name="25d", anchor="m25d", change="+ ×3 turn weight on far moving bouts", units=[("25d", S0), ("30b", S101)],
             md="The one component that helps on its own: grading +1.7 | +1.9 mm and moving error −0.3 | −0.4 replicate "
                "at both seeds; matched AUC −0.02 | −0.03."),
        dict(name="26e — the ablation model", anchor="m26e",
             change="+ range × bearing products (S4x) and the temporal set (centred interval lags with the 10 s "
                    "truncation, run clock, aligned-and-far flag)", units=[("26e", S0), ("30d", S101)],
             md="The dip deepens at both seeds (2.7 → 3.7 | 2.7 → 3.0 mm); AUC flat at seed 0, better at seed 101. "
                "The two components only work together — see the checks below."),
        dict(name="Check: 29a — range products alone", anchor="m29a", change="25d + S4x", level=4,
             units=[("29a", S0), ("30c", S101)],
             md="Worse at both seeds (matched +0.07 | +0.08)."),
        dict(name="Check: 29b — temporal set alone", anchor="m29b", change="25d + the temporal set (S4i code)", level=4,
             units=[("29b", S0)], missing=[(S101, "not trained at seed 101 (optional)")],
             md="No AUC gain (matched +0.04), most of the dip (3.4 mm)."),
        dict(name="Check: 35a — 26e without the run clock", anchor="m35a",
             change="26e − run clock (the rest of the temporal set and S4x kept)", level=4,
             units=[("35a", S0), ("35b", S101)],
             md="The run clock alone, removed from 26e. Matched AUC worse at both seeds (+0.034 | +0.078, beyond the "
                "0.05 bar at seed 101 only), likelihood unchanged (−0.005 | −0.007 bits), dip −1.16 | +0.12 mm (not "
                "replicated), grading unchanged, static error +0.40 | +0.45. Same checkpoints as the ablation notebook."),
    ]),
    dict(id="drift", title="2. Drift branch — best likelihood and slow structure (not for ablations)", md="""The AR(1)
slow state (drift) improves the held-out likelihood and reproduces the slow window-to-window correlation at both seeds,
but costs dip. With the drift, the bump code beats S4x at both seeds; whether the drift improves AUC on the bump code
depends on the seed (−0.05 | +0.003).""", models=[
        dict(name="18a", anchor="m18a", change="26e + static block (a control)", units=[("18a", S0), ("24a", S101)],
             md="The static block has no replicated effect; 18a only makes the next comparisons differ by one thing."),
        dict(name="18a + drift", anchor="m24b", change="+ AR(1) slow state on the S4x code",
             units=[("24b", S0), ("24c", S101)],
             md="Likelihood better (−0.065 | −0.047 bits) and the slow window correlation reproduced; matched AUC and "
                "dip worse at both seeds."),
        dict(name="Model 27", anchor="model-27", change="the bump code (S4) instead of S4x, no drift",
             units=[("16e", S0), ("30e", S101)],
             md="Without the drift, bumps vs S4x depends on the seed (worst at seed 0, best of all at seed 101)."),
        dict(name="Model 17", anchor="model-17", change="Model 27 + drift", units=[("11e", S0), ("32a", S101)],
             md="Best likelihood (−0.062 | −0.046 bits). Both seeds on per-block streams: seed 0 = key 11e (9a / 12a are "
                "the same checkpoint on the old stream), seed 101 = key 32a (12b's checkpoints re-evaluated)."),
        dict(name="14f — the final model", anchor="m14f",
             change="Model 17 + profile correction (pf-B, 20 s cut-off, final-step numbers)",
             units=[("14f", S0), ("32b", S101)],
             md="Best likelihood and the only model with slow structure, with the profiles restored by the correction "
                "(dip +0.8 mm, grading +1.8 mm, moving error −0.3); AUC tied with the no-drift models."),
    ]),
    dict(id="scale", title="3. Scaling up — three hidden layers instead of one", md="""M6, D4 and F retrained with three
hidden layers of 128 units in the shared body instead of one, everything else unchanged (same recipe, folds and seeds;
F = the three-layer D4 + the same profile correction). If simply making the network bigger helped, these would beat
their one-layer versions at both seeds. The width check (64 / 256 units, Final_report_missingPieces) is the other
direction of the same question.""", models=[
        dict(name="M6 (26e), three layers", anchor="m36a", change="26e with depth 3",
             units=[("36a", S0), ("36b", S101)], md=""),
        dict(name="D4 (Model 17), three layers", anchor="m36c", change="Model 17 with depth 3",
             units=[("36c", S0), ("36d", S101)], md=""),
        dict(name="F (14f), three layers", anchor="m36e", change="the three-layer D4 + profile correction",
             units=[("36e", S0), ("36f", S101)], md=""),
    ]),
    dict(id="rzlo", title="4. Lower-clip check — the log range down to 2 px", md="""M6, D4 and F retrained with the
log range rz clipped below at the 2 px floor (log4(2 / 70) = −2.56) instead of −1 (17.5 px, 4.9 mm), everything else
unchanged (same recipe, folds and seeds; F = the lower-clip D4 + the same profile correction). 4.5 % of the bouts with dots
have the centroid closer than 17.5 px, almost all with moving groups. If the clip hid useful information, these would beat
their clipped versions at both seeds.""", models=[
        dict(name="M6 (26e), lower clip", anchor="m42a", change="26e with rz clipped at 2 px",
             units=[("42a", S0), ("42b", S101)],
             md="No effect: likelihood −0.009 | −0.008 bits, matched AUC 0.000 | +0.010, dip −0.11 | 0.00 mm, grading "
                "−0.11 | +0.16 mm; the static error +0.41 at seed 0 does not replicate (0.00)."),
        dict(name="D4 (Model 17), lower clip", anchor="m42c", change="Model 17 with rz clipped at 2 px",
             units=[("42c", S0), ("42d", S101)],
             md="No effect beyond noise: likelihood +0.004 | −0.004 bits, matched AUC +0.033 | +0.017 (under the bar; "
                "the drift-gain recalibration alone moves it ~0.03), grading +0.94 | +0.79 mm (within ±1 mm), dip "
                "−0.01 | −0.21 mm."),
        dict(name="F (14f), lower clip", anchor="m42e", change="the lower-clip D4 + profile correction",
             units=[("42e", S0), ("42f", S101)],
             md="No effect: likelihood +0.005 | −0.005 bits, matched AUC +0.020 | −0.013, dip +0.18 | +0.12 mm, grading "
                "−0.28 | +0.07 mm, moving error 0.00 | +0.14, static error +0.15 | +0.07."),
    ]),
    dict(id="tau", title="5. Slow-state time constant — tau = 200 bouts instead of 70", md="""D4 and F retrained
with the AR(1) drift's time constant at 200 bouts instead of 70, everything else unchanged (same recipe, folds and
seeds; F = the tau-200 D4 + the same profile correction). In the real data the correlation between 60-bout window means
falls by a factor of about 0.75 per window for log displacement and 0.7 for radial position (tau ~220 and ~160 bouts;
not determined for the interval), whereas tau = 70 forgets by 0.42 per window. If the shorter time constant limited the
drift, these would beat their tau-70 versions at both seeds.""", models=[
        dict(name="D4 (Model 17), tau = 200", anchor="m37a", change="Model 17 with tau = 200 bouts",
             units=[("37a", S0), ("37b", S101)], md=""),
        dict(name="F (14f), tau = 200", anchor="m37c", change="the tau-200 D4 + profile correction",
             units=[("37c", S0), ("37d", S101)], md=""),
    ]),
    dict(id="tsize", title="6. Turn-size individuality — the latent also on the turning scales", md="""M6, D4 and F
retrained with the per-fish latent loading also on the scales of the four turning components (one shared loading), so
how far a fish turns becomes a trait alongside how often it swims straight; everything else unchanged (same recipe,
folds and seeds, and the same initial weights apart from the new loading; F = the turn-size D4 + the same profile
correction). In the data the mean turn size spans 42-64 deg across fish (split-half r = 0.95) and carries most of the
between-fish spread of mean |dtheta|, while the models' between-fish turn spread is 0.70-0.83 of the real one. If the
loading closes that gap, the between-fish turn spread should move toward 1 at both seeds.""", models=[
        dict(name="M6 (26e), turn-size latent", anchor="m38a", change="26e + turn-size loading",
             units=[("38a", S0), ("38b", S101)], md=""),
        dict(name="D4 (Model 17), turn-size latent", anchor="m38c", change="Model 17 + turn-size loading",
             units=[("38c", S0), ("38d", S101)], md=""),
        dict(name="F (14f), turn-size latent", anchor="m38e", change="the turn-size D4 + profile correction",
             units=[("38e", S0), ("38f", S101)], md=""),
    ]),
    dict(id="tsz200", title="7. Turn-size latent with tau = 200 — both changes together", md="""D4 and F
retrained with both changes of sections 5 and 6 at once: the slow state's time constant at 200 bouts and the per-fish
latent also on the scales of the four turning components; everything else unchanged (same recipe, folds and seeds;
F = this D4 + the same profile correction). Compared with Model 17 / 14f (neither change), with the tau-200 units
(37a-d) and with the turn-size units (38c-f) at the same seed.""", models=[
        dict(name="D4 (Model 17), turn-size latent + tau = 200", anchor="m46a",
             change="Model 17 + turn-size loading + tau = 200", units=[("46a", S0), ("46b", S101)], md=""),
        dict(name="F (14f), turn-size latent + tau = 200", anchor="m46c",
             change="the turn-size tau-200 D4 + profile correction", units=[("46c", S0), ("46d", S101)], md=""),
    ]),
    dict(id="driftfix", title="8. Fixed drift strength — tau 70 against tau 200 at equal strength", md="""D4's drift
with its generation strength FIXED for every fold instead of calibrated per fold (same checkpoints, gains rewritten so
that gain x |loading| is the same in every fold model; no retraining, likelihood unchanged). The per-fold calibration
gives the interval the same effective strength in every fold, but leaves folds whose likelihood learned a near-zero
displacement loading with no displacement drift (two of five in the tau-200 seed-101 unit), which can split the seeds.
Strength 0 (no drift at generation), half and full of the median the tau-70 calibration reached (displacement 0.05,
interval 0.13), at tau 70 and tau 200: at equal strength only the drift's speed differs. Read paired by seed against
each other and against Model 17 | 32a (tau 70, calibrated) and 37a | 37b (tau 200, calibrated).""", models=[
        dict(name="D4, no drift at generation", anchor="m47a", change="Model 17 with drift strength 0",
             units=[("47a", S0), ("47f", S101)], md=""),
        dict(name="D4, tau 70, half strength", anchor="m47b", change="Model 17, fixed half strength",
             units=[("47b", S0), ("47g", S101)], md=""),
        dict(name="D4, tau 70, full strength", anchor="m47c", change="Model 17, fixed full strength",
             units=[("47c", S0), ("47h", S101)], md=""),
        dict(name="D4, tau 200, half strength", anchor="m47d", change="37a | 37b, fixed half strength",
             units=[("47d", S0), ("47i", S101)], md=""),
        dict(name="D4, tau 200, full strength", anchor="m47e", change="37a | 37b, fixed full strength",
             units=[("47e", S0), ("47j", S101)], md=""),
        dict(name="F on the strength-0 D4", anchor="m48a",
             change="47a | 47f + the profile correction (fitted without drift)", units=[("48a", S0), ("48b", S101)],
             md=""),
    ]),
    dict(id="app", title="Appendix — the correction on other bases", md="""The same correction (pf-B, 8 numbers, 100 steps)
on 26e, on Model 27 and always-on on Model 17 (seed 0). It never changes AUC. On 26e it does nothing measurable and the
static error gets worse, which is why the main line ends at 26e.""", models=[
        dict(name="28a / 28b — 26e + correction, 20 s cut-off", anchor="m28b", change="keep-best step / final step",
             units=[("28a", S0), ("28b", S0)], md="Null: dip 3.70 → 3.91 / 3.79, static error 0.80 → 1.02 / 1.22."),
        dict(name="28c / 28d — 26e + correction, always on", anchor="m28d", change="keep-best step / final step",
             units=[("28c", S0), ("28d", S0)], md="Grading 6.67 / 5.84 (real 6.00), 8-dot plateau 1.1–1.6 mm closer."),
        dict(name="16b — Model 27 + correction", anchor="m16b", change="20 s cut-off, final step", units=[("16b", S0)],
             md="The best profile shape of all (dip 4.04, moving error 0.76) but Model 27's worse AUC."),
        dict(name="28e / 28f — Model 17 + correction, always on", anchor="m28f", change="keep-best step / final step",
             units=[("28e", S0), ("28f", S0)], md="Within noise of 14f (the cut-off) on every metric."),
    ]),
]

# The storyline table rows: (label, seed-0 key, seed-101 key)
ROWS = [
    ("1  Model 1", "1", None), ("2  Model 2", "2", None), ("3  Model 3", "3", None), ("4  Model 4", "20f", "30a"),
    ("5  25d", "25d", "30b"), ("   29a (check)", "29a", "30c"), ("   29b (check)", "29b", None),
    ("   35a (check: 26e − run clock)", "35a", "35b"),
    ("6  26e = ablation model", "26e", "30d"), ("7  18a", "18a", "24a"), ("8  18a + drift", "24b", "24c"),
    ("9  Model 27", "16e", "30e"), ("10 Model 17", "11e", "32a"), ("11 14f = final model", "14f", "32b"),
    ("S  26e, three layers", "36a", "36b"), ("S  Model 17, three layers", "36c", "36d"), ("S  14f, three layers", "36e", "36f"),
    ("C  26e, lower clip", "42a", "42b"), ("C  Model 17, lower clip", "42c", "42d"), ("C  14f, lower clip", "42e", "42f"),
    ("T  Model 17, tau 200", "37a", "37b"), ("T  14f, tau 200", "37c", "37d"),
    ("Z  26e, turn size", "38a", "38b"), ("Z  Model 17, turn size", "38c", "38d"), ("Z  14f, turn size", "38e", "38f"),
    ("ZT Model 17, turn size + tau 200", "46a", "46b"), ("ZT 14f, turn size + tau 200", "46c", "46d"),
    ("G  D4 drift strength 0", "47a", "47f"), ("G  D4 tau 70 half", "47b", "47g"), ("G  D4 tau 70 full", "47c", "47h"),
    ("G  D4 tau 200 half", "47d", "47i"), ("G  D4 tau 200 full", "47e", "47j"),
    ("G  F on strength-0 D4", "48a", "48b"),
    ("A  28b (26e + corr.)", "28b", None), ("A  28d (26e + corr., always on)", "28d", None), ("A  16b (M27 + corr.)", "16b", None),
]
# The steps: (label, (seed-0 from, to), (seed-101 from, to) or None)
STEPS = [
    ("1 -> 2  + wall rejection", ("1", "2"), None), ("2 -> 3  + latent", ("2", "3"), None),
    ("3 -> 4  + S4i products (old stream)", ("3", "4"), None), ("4 -> 25d  + x3 turn weight", ("20f", "25d"), ("30a", "30b")),
    ("25d -> 26e  + range products + temporal set", ("25d", "26e"), ("30b", "30d")),
    ("25d -> 29a  range products alone", ("25d", "29a"), ("30b", "30c")), ("29a -> 26e  + temporal set", ("29a", "26e"), ("30c", "30d")),
    ("25d -> 29b  temporal set alone", ("25d", "29b"), None), ("29b -> 26e  + range products", ("29b", "26e"), None),
    ("26e -> 35a  run clock removed", ("26e", "35a"), ("30d", "35b")),
    ("26e -> 18a  + static block", ("26e", "18a"), ("30d", "24a")),
    ("18a -> 18a + drift", ("18a", "24b"), ("24a", "24c")), ("18a -> Model 27  bumps", ("18a", "16e"), ("24a", "30e")),
    ("Model 27 -> Model 17  + drift", ("16e", "11e"), ("30e", "32a")),
    ("18a + drift -> Model 17  bumps, with drift", ("24b", "11e"), ("24c", "32a")),
    ("Model 17 -> 14f  + correction", ("11e", "14f"), ("32a", "32b")), ("26e -> 28b  + correction", ("26e", "28b"), None),
    ("26e -> Model 17  (head to head)", ("26e", "11e"), ("30d", "32a")),
    ("26e -> 14f  (head to head)", ("26e", "14f"), ("30d", "32b")),
    ("26e -> three layers", ("26e", "36a"), ("30d", "36b")), ("Model 17 -> three layers", ("11e", "36c"), ("32a", "36d")),
    ("14f -> three layers", ("14f", "36e"), ("32b", "36f")),
    ("26e -> lower clip", ("26e", "42a"), ("30d", "42b")), ("Model 17 -> lower clip", ("11e", "42c"), ("32a", "42d")),
    ("14f -> lower clip", ("14f", "42e"), ("32b", "42f")),
    ("Model 17 -> tau 200", ("11e", "37a"), ("32a", "37b")), ("14f -> tau 200", ("14f", "37c"), ("32b", "37d")),
    ("26e -> turn size", ("26e", "38a"), ("30d", "38b")), ("Model 17 -> turn size", ("11e", "38c"), ("32a", "38d")),
    ("14f -> turn size", ("14f", "38e"), ("32b", "38f")),
    ("Model 17 -> turn size + tau 200", ("11e", "46a"), ("32a", "46b")), ("14f -> turn size + tau 200", ("14f", "46c"), ("32b", "46d")),
    ("tau 200 D4 -> + turn size", ("37a", "46a"), ("37b", "46b")), ("turn-size D4 -> + tau 200", ("38c", "46a"), ("38d", "46b")),
    ("fixed full: tau 70 -> tau 200", ("47c", "47e"), ("47h", "47j")), ("fixed half: tau 70 -> tau 200", ("47b", "47d"), ("47g", "47i")),
    ("tau 70: strength 0 -> full", ("47a", "47c"), ("47f", "47h")), ("Model 17 -> fixed full strength", ("11e", "47c"), ("32a", "47h")),
    ("strength-0 D4 -> + correction", ("47a", "48a"), ("47f", "48b")), ("14f -> F on strength-0 D4", ("14f", "48a"), ("32b", "48b")),
]

INTRO = """# Final report models

Only the models of the report's storyline, in the order the report tells it, each with the usual measurements (held-out
likelihood; window-classifier AUC and individuality; the AUC under fairer windowings; the individuality grid; the
distance-to-dots profile; the matched profile; attraction; turn toward the dots; the head analysis). The full development
history — every other model and arm — stays in `Report_models.ipynb`; the ablation study is in
`Final_report_ablations.ipynb` and the two-group and multi-fish experiments in `Final_report_extrapolations.ipynb`.

**How to read it.**
- **Seeds.** Seed 0 and seed 101 are different cross-validation splits. Models are compared only on the same seed; the
  second seed is a second paired comparison, never pooled. Differences under ~0.05 AUC at one seed are within the seed
  noise (~0.03 per seed for these recipes); a claim needs the same direction at both seeds.
- **Headline AUC = matched** (generated windows cut at the real fish's own window positions, 1:1); 0.5 = indistinguishable,
  real against real ~0.52. Standard and no-radial readings alongside.
- **Profiles** (dip, grading, moving / static error, attraction) are from programme rollouts on per-block random streams,
  except Models 1–3, which are on the old single stream (marked; their AUC and likelihood are unaffected). Profile numbers carry rollout noise: grading ±1 mm, attraction ±1–2 mm, static error ±0.3.
- Real fish: dip 4.23 mm at 13.5 s, plateau 17.26 mm, 1-dot plateau 22.26 mm, grading 6.00 mm, attraction
  1.16 / 6.19 / 5.19 / 6.43 / 6.94 mm (1 / 2 / 4 / 6 / 8 dots).
- **XX** = a result that does not exist yet.

The cells of each model can be re-run on their own after the setup cell (they read `report_ckpts/M<key>`)."""

TOC_MD = """## Contents

1. [Summary tables](#summary) — every storyline model at both seeds; every step's change at both seeds
2. [Main line](#main) — [Model 1](#model-1) · [Model 2](#model-2) · [Model 3](#model-3) · [Model 4](#model-4) ·
   [25d](#m25d) · [26e](#m26e) (checks: [29a](#m29a), [29b](#m29b), [35a](#m35a))
3. [Drift branch](#drift) — [18a](#m18a) · [18a + drift](#m24b) · [Model 27](#model-27) · [Model 17](#model-17) ·
   [14f](#m14f)
4. [Scaling up: three hidden layers](#scale) — [M6](#m36a) · [D4](#m36c) · [F](#m36e)
5. [Lower-clip check: log range down to 2 px](#rzlo) — [M6](#m42a) · [D4](#m42c) · [F](#m42e)
6. [Slow-state time constant: tau = 200](#tau) — [D4](#m37a) · [F](#m37c)
7. [Turn-size individuality](#tsize) — [M6](#m38a) · [D4](#m38c) · [F](#m38e)
8. [Turn-size latent with tau = 200](#tsz200) — [D4](#m46a) · [F](#m46c)
9. [Fixed drift strength](#driftfix) — [0](#m47a) · [tau 70](#m47b) · [tau 200](#m47d) · [F on strength 0](#m48a)
10. [Appendix: the correction on other bases](#app) — [28a/b](#m28b) · [28c/d](#m28d) · [16b](#m16b) · [28e/f](#m28f)
11. [Supporting numbers](#support) — seed noise (Model 12); components alone on Model 4 (Model 25) and removed from 18a
   (Model 26)"""

_TAB_COMMON = '''import json, os
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)

def _s(key):
    """Key numbers of one notebook unit (None if it does not exist yet). A trailing * = profile numbers on the old stream."""
    p = f"report_outputs/summary_M{key.rstrip('*')}.json" if key else None
    return json.load(open(p)) if p and os.path.exists(p) else None
'''

TABLE = _TAB_COMMON + '''
ROWS = %r
COLS = [("likelihood (bits/bout)", "lik_joint", 3), ("matched AUC", "auc_matched", 3), ("std AUC", "auc_std", 3),
        ("no-radial AUC", "auc_matched_norad", 3), ("dip (mm)", "dip", 2), ("dip latency (s)", "lat", 1),
        ("grading (mm)", "grading", 2), ("moving err (SEM)", "mv_err_sem", 2), ("static err (SEM)", "st_err_sem", 2),
        ("2-dot attraction", "attr_N2", 2), ("8-dot attraction", "attr_N8", 2)]
PROFILE = {"dip", "lat", "grading", "mv_err_sem", "st_err_sem", "attr_N2", "attr_N8"}

def _cell(key, c, nd):
    s = _s(key)
    if s is None or s.get(c) is None:
        return "XX"
    if key.endswith("*") and c in PROFILE:              # old-stream profile numbers do not pair
        return f"({s[c]:.{nd}f})"
    return f"{s[c]:.{nd}f}"

T = pd.DataFrame([{lab: f"{_cell(k0, c, nd)} | {_cell(k1, c, nd)}" for lab, c, nd in COLS}
                  for _, k0, k1 in ROWS], index=[r[0] for r in ROWS])
T.insert(0, "keys (s0 | s101)", [f"{k0} | {k1 or 'XX'}" for _, k0, k1 in ROWS])
print("Every storyline model: seed 0 | seed 101 (XX = not run; (x) = profile number on the old stream, not paired)")
print("Real fish: dip 4.23 @ 13.5 s, grading 6.00, attraction 2 dots 6.19 / 8 dots 6.94; real-vs-real AUC ~0.52.")
T''' % (ROWS,)

STEP_TABLE = _TAB_COMMON + '''
STEPS = %r
D = [("d likelihood", "lik_joint", 3), ("d matched AUC", "auc_matched", 3), ("d std AUC", "auc_std", 3),
     ("d no-radial AUC", "auc_matched_norad", 3), ("d dip", "dip", 2), ("d grading", "grading", 2),
     ("d moving err", "mv_err_sem", 2), ("d static err", "st_err_sem", 2)]
PROFILE = {"dip", "grading", "mv_err_sem", "st_err_sem"}

def _d(pair, c):
    if pair is None:
        return None
    a, b = _s(pair[0]), _s(pair[1])
    if a is None or b is None or a.get(c) is None or b.get(c) is None:
        return None
    if c in PROFILE and (pair[0].endswith("*") or pair[1].endswith("*")):
        return None                                      # mixes random streams: not a paired profile change
    return b[c] - a[c]

rows = []
for lab, p0, p1 in STEPS:
    r = {"step": lab}
    for name, c, nd in D:
        d0, d1 = _d(p0, c), _d(p1, c)
        f = lambda v: "XX" if v is None else f"{v:+.{nd}f}"
        r[name] = f"{f(d0)} | {f(d1)}"
    m0, m1 = _d(p0, "auc_matched"), _d(p1, "auc_matched")
    r["matched: same sign at both seeds?"] = ("one seed" if (m0 is None or m1 is None) else
                                             ("yes" if np.sign(m0) == np.sign(m1) else "NO (seed-dependent)"))
    rows.append(r)
print("Every step: change seed 0 | seed 101 (negative AUC / likelihood / errors = better; XX = not run or not paired)")
pd.DataFrame(rows).set_index("step")''' % (STEPS,)

SUPPORT = _TAB_COMMON + '''
GROUPS = [
    ("Seed noise: Model 17's recipe at four seeds (Model 12; profiles on the old stream)",
     [("12a seed 0", "12a"), ("12b seed 101", "12b"), ("12c seed 202", "12c"), ("12d seed 303", "12d")]),
    ("Each component ALONE on Model 4 (Model 25, seed 0; control 20f)",
     [("20f Model 4", "20f"), ("25a + centred lags & truncation", "25a"), ("25b + run clock", "25b"),
      ("25c + aligned-and-far flag", "25c"), ("25d + x3 turn weight", "25d"), ("25e + static block", "25e"),
      ("25f + range products (S4x)", "25f")]),
    ("Each component REMOVED from 18a (Model 26, seed 0; control 18a)",
     [("18a", "18a"), ("26a - centred lags & truncation", "26a"), ("26b - run clock", "26b"),
      ("26c - aligned-and-far flag", "26c"), ("26d - x3 turn weight", "26d"), ("26e - static block", "26e")]),
]
COLS = [("likelihood", "lik_joint", 3), ("matched AUC", "auc_matched", 3), ("std AUC", "auc_std", 3),
        ("no-radial AUC", "auc_matched_norad", 3), ("dip", "dip", 2), ("grading", "grading", 2),
        ("moving err", "mv_err_sem", 2), ("static err", "st_err_sem", 2)]
for title, items in GROUPS:
    print(title)
    print(pd.DataFrame([{n: ("XX" if _s(k) is None or _s(k).get(c) is None else round(_s(k)[c], nd)) for n, c, nd in COLS}
                        for _, k in items], index=[i[0] for i in items]).to_string())
    print()'''


def md(text, cid):
    return dict(cell_type="markdown", id=cid, metadata={}, source=text.splitlines(True))


def code(src, cid, outs=()):
    return dict(cell_type="code", id=cid, execution_count=None, metadata={}, source=src.splitlines(True),
                outputs=list(outs))


def _df_outputs(ns, src):
    """Run a cell whose last line is an expression and embed the DataFrame as text (stdout + repr)."""
    lines = src.rstrip().splitlines()
    body, last = "\n".join(lines[:-1]), lines[-1]
    outs, _ = RM._run_cell(ns, body + f"\n_df_ = {last}\nprint(_df_.to_string())")
    return outs


def build():
    import os
    os.chdir(HERE)
    ns = {"__name__": "__cell__"}
    exec(compile(RM.SETUP, "<setup>", "exec"), ns)
    cells = [md(INTRO, "intro"), md(TOC_MD, "toc"), code(RM.SETUP, "setup")]
    cells += [md('<a id="summary"></a>\n## Summary tables', "summary-md"),
              code(TABLE, "summary-models", _df_outputs(ns, TABLE)),
              code(STEP_TABLE, "summary-steps", _df_outputs(ns, STEP_TABLE))]
    n_units, missing = 0, []
    for sec in STORY:
        cells.append(md(f'<a id="{sec["id"]}"></a>\n## {sec["title"]}\n\n{sec["md"]}', f'{sec["id"]}-md'))
        for m in sec["models"]:
            h = "#" * m.get("level", 3)
            cells.append(md(f'<a id="{m["anchor"]}"></a>\n{h} {m["name"]}\n\n**Change:** {m["change"]}\n\n{m["md"]}',
                            f'{m["anchor"]}-md'))
            for key, seed in m["units"]:
                p = OUTS / f"M{key}.json"
                hh = "#" * (m.get("level", 3) + 1)
                if not p.exists():
                    cells.append(md(f"{hh} {m['name']} — {seed} (key {key})\n\nXX — not evaluated yet.", f"u{key}-xx"))
                    missing.append(key)
                    continue
                blob = json.loads(p.read_text(encoding="utf-8"))
                cells.append(md(f"{hh} {m['name']} — {seed} (key `{key}`)", f"u{key}-sub"))
                cells += [code(c["source"], f"u{key}-{c['cid']}", c["outputs"]) for c in blob["cells"]]
                n_units += 1
            for seed, why in m.get("missing", []):
                cells.append(md(f"{'#' * (m.get('level', 3) + 1)} {m['name']} — {seed}\n\nXX — {why}",
                                f'{m["anchor"]}-{seed.replace(" ", "")}-xx'))
    cells += [md('<a id="support"></a>\n## Supporting numbers\n\nKey numbers only; the full blocks are in '
                 '`Report_models.ipynb` (Models 12, 25, 26).', "support-md"),
              code(SUPPORT, "support", RM._run_cell(ns, SUPPORT)[0])]
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                         language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {NB.name}: {len(cells)} cells, {n_units} model units, {NB.stat().st_size / 1e6:.1f} MB"
          + (f"; missing outputs: {missing}" if missing else ""), flush=True)


if __name__ == "__main__":
    build()
