"""Build Model/Report_models.ipynb: one block per model that ends up in the report -- a markdown cell (what the model
is and why), then the same code cells for every model (held-out likelihood; AUC and individuality; the AUC under two
fairer windowings, with and without the radial features; the individuality grid; the distance-to-dots profile; the
matched profile numbers; attraction; turn toward by dot count). A summary cell
at the end puts every model in one table (the report's Table 5.1).

The heavy work runs OUTSIDE the notebook, one process per model, all at once; the notebook is then written with every
cell's output embedded, so it opens fully rendered.

    python report_models.py evaluate K      # run unit K's cells (K = 3, 6a, ...); save outputs + key numbers
    python report_models.py assemble        # write Report_models.ipynb from the saved outputs
    python report_models.py all             # evaluate every unit in parallel, then assemble
    python report_models.py run 6a,6b       # evaluate only those units in parallel, then assemble
    python report_models.py refresh indv    # recompute only those cells (comma-separated ids), every unit, reassemble

A block is one model: one markdown cell, then the cells of each of its ARMS (a "unit": Models 1-5 have one, Models 6
and 7 two, `6a` / `6b`). Each unit reads only its own folder (report_ckpts/M<key>, a copy of that arm's five fold
checkpoints), so any unit can be re-run on its own in the notebook. Adding a model = one entry in MODELS.
"""
import base64
import io
import json
import re
import subprocess
import textwrap
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "Report_models.ipynb"
CKPTS = HERE / "report_ckpts"
OUTS = HERE / "report_outputs"

INTRO = """# Report models

The models that appear in the report, one block each: a markdown cell saying what the model is and why, then the same
measurements for every model:

- **held-out likelihood** per head (bits per bout, lower is better);
- **distinguishability**: the window classifier's AUC (0.5 = indistinguishable; real against real scores ~0.52 standard,
  ~0.49 steady-state) and the spread of simulated individuals relative to real fish;
- the same classifier under two **fairer windowings** of the same rollouts -- one generated window per real one, and
  generated windows cut at the real fish's own window positions (matched to the real runs, 1:1) -- each with all 39
  features and without the four radial-position features (mean +/- SE over the 8 rollout replicates);
- **individuality**: the distribution over 60-bout windows of every window statistic (mean, median, SD, q90, lag-1
  autocorrelation) of radial position, interval, |turn| and displacement, real against simulated;
- the **distance-to-dots profile** across the block (15 simulated fish), then the same with 50 simulated fish **matched**
  to the real block composition -- the numbers to read;
- **attraction** per dot count (stimulus-blind null minus plateau) and the **turn toward the dots** by dot count.

Every number is on held-out fish: five-fold cross-validation by fish, the same folds and seeds for every model, so two
models differ only by what was changed between them. Built by `report_models.py`; each block reads only its own folder
(`report_ckpts/M<k>`) and can be re-run on its own. Differences in AUC below ~0.03 are within the noise.

Models 6 and 7 hold two arms each, side by side: **a** = without the range bumps (the Model 4 line), **b** = with them
(the Model 5 line). Model 8 holds four arms (8a–8d), all built on 6a; Model 9 four arms (9a–9d) on the development
ladder's Model 17; Model 10 four arms (10a–10d) simplifying Model 17; Model 11 four profile fine-tunes of Model 17
plus its control (11a–11e). From Model 10 on, every arm ends with a head analysis: each head's law against real fish,
per stimulus condition and block phase.
"""

SETUP = """# Setup: the evaluation code lives in ../scripts.
import sys; sys.path.insert(0, "."); sys.path.insert(0, "../scripts")
import numpy as np, pandas as pd
pd.set_option("display.width", 250)
from rollout_model1_local import run
from model_figures import window_stats_grid, dot_profile, attraction_table, toward_by_n
from input_ablation import profile_variants_matched
REJ = {"wall_reject": {"d0": 2.0, "d1": 10.0, "max_draws": 20}}   # generation-time wall containment (Models 2+)"""

MODELS = [
    dict(k=1, src="report1_ladder_checkpoints", arm="S0_adt", rej=False, md="""## Model 1 — Base model

The simplest model that can respond to the dots. One hidden layer of 128 units feeds three chained heads: the turn,
then the displacement given the turn, then the interval given both.

**Inputs (25):** the last 8 turns; the last 3 displacements and 3 intervals; lag availability; the wall (distance to
it, bearing to the dish centre); the block clock; and the simplest stimulus code (6 columns): dots present, moving or
static, log dot count, log range to the dots' centroid, and the sine and cosine of the bearing to it.

No per-fish latent: every simulated fish is the same population model. No wall containment: the only constraint is a
guardrail that stops a step at the wall."""),
    dict(k=2, src="report1_ladder_checkpoints", arm="S0_adt", rej=True, md="""## Model 2 — + wall containment

The same trained model as Model 1; only generation changes. Real fish never enter an outer annulus of the dish, where
the water is too shallow: a physical limit rather than a behaviour, so it is imposed on position. A drawn bout that
would land within 2 mm of the wall is always redrawn; between 2 and 10 mm, with a probability falling linearly to zero;
up to 20 redraws. The turn toward the centre near the wall is still learned from the wall inputs."""),
    dict(k=3, src="report1_ladder_checkpoints", arm="S0_adt_z2", rej=True, md="""## Model 3 — + per-fish latent

Fish differ in their typical displacement, tempo and turning, stably across the session. Each fish now has a
2-dimensional latent (prior N(0, I)) that shifts the level of each head: how often it swims straight, its typical
displacement, its tempo. Training fits one latent per training fish; a held-out fish's likelihood integrates it out;
every rollout draws a new one, so a simulated fish is a new individual, never a copy of a training fish. One gain on the
loadings, set on training-fish rollouts, sets how widely the simulated individuals spread."""),
    dict(k=4, src="report1_ladder_checkpoints", arm="S4i_adt_z2", rej=True, md="""## Model 4 — + interaction terms

The stimulus code gains three products (9 columns, 28 inputs): log count × sine of the bearing, so the lateral pull can
grow with the number of dots; and moving × sine and moving × cosine of the bearing, so moving and static dots can pull
differently. The network could in principle learn these products itself; handing them over tests whether it can from
15 fish."""),
    dict(k=5, src="base1_ladder_checkpoints", arm="S4_adt_z2", rej=True, md="""## Model 5 — + range bumps (the full centroid code, S4)

The bearing terms are expanded on four Gaussian bumps in log range, centred at 6.9, 13.9, 25.0 and 44.4 mm and
normalised to sum to one (18 columns, 37 inputs). The pull toward the dots can now vary non-monotonically with
distance, including reversing very close to them, and it extrapolates flat beyond the outer bumps. (Same checkpoints as
Model 39's `S4_adt_z2`: training is deterministic given the seed.)"""),
    dict(k=6, md="""## Model 6 — + per-head recalibration of the individual spread

Models 4 and 5 again, with no retraining: only the latent's loading gains change. Until now ONE gain, picked per fold
from a coarse grid (0.4, 0.7, 1.2, ...) on 16 simulated trajectories, set how widely the simulated individuals spread,
and it landed differently between folds and models (Model 4: 0.7 / 0.7 / 0.7 / 0.7 / 0.4; Model 5: 0.4 / 1.2 / 0.4 /
0.7 / 0.7). Here each fold's loadings are unfolded and refitted with one continuous gain per head (turn, displacement,
interval), each set so that the spread of simulated individuals matches the between-fish spread of that fold's 12
training fish (held-out fish never enter). The gain is folded into the network's weights, so the held-out likelihood
is recomputed too.

- **6a**: Model 4 (no range bumps) + recalibration.
- **6b**: Model 5 (range bumps) + recalibration.

With the grid lottery gone, 6a and 6b differ only in the bumps.""",
         arms=[dict(key="6a", src="report2_g3h/S4i_adt_z2_g3h", arm="S4i_adt_z2_g3h", rej=True,
                    sub="6a — no bumps: Model 4 + per-head recalibration"),
               dict(key="6b", src="report2_g3h/S4_adt_z2_g3h", arm="S4_adt_z2_g3h", rej=True,
                    sub="6b — bumps: Model 5 + per-head recalibration")]),
    dict(k=7, md="""## Model 7 — separate range laws for static and moving dots

Both Model 6 arms retrained with stimulus terms that let static and moving dots pull differently depending on range,
then recalibrated per head as in Model 6. Real fish turn away from static dots once within about 60 px (17 mm) and are
neutral further out, while moving dots pull them in. In Models 4 to 6 the only condition-specific stimulus terms are
moving × sin and moving × cos of the bearing, so the static law can only be the moving law shifted, never a sign flip.

- **7a** (no bumps; 14 columns, 33 inputs): Model 4's code + five products, linear in log range: moving × log range ×
  sin and × cos, static × log range × sin and × cos, and log count × log range × sin. The pull can grow or shrink with
  range, with its own slope for moving and static dots, so it can flip sign once per condition.
- **7b** (bumps; 26 columns, 45 inputs): Model 5's code + the static bearing block: the four range bumps × sin and × cos
  of the bearing, non-zero only for static dots.

After this step both arms carry the same kinds of terms (a range law per condition, and count × range), so 7a against 7b
tests the range shape: linear in log range against four bumps. These terms carry no time, so the timing of the static
response (a glance toward the dots at onset, a turn away at 12-30 s) is not addressed.""",
         arms=[dict(key="7a", src="report2_g3h/S4x_adt_z2_g3h", arm="S4x_adt_z2_g3h", rej=True,
                    sub="7a — no bumps: + range × bearing per condition, linear in log range"),
               dict(key="7b", src="report2_g3h/S4_adt_z2_st_g3h", arm="S4_adt_z2_st_g3h", rej=True,
                    sub="7b — bumps: + the static bearing block")]),
    dict(k=8, md="""## Model 8 — + run state and interval handling (on Model 6a)

Four arms on Model 6a's code (no range bumps), each retrained and then recalibrated per head as in Model 6: a 2 × 2 on
two additions, then a turn weight on top of both.

- **Run clock**: exp(−k / 30), k = bouts since the fish last resumed swimming after a pause longer than 10 s. Real fish
  swim differently just after resuming; without it the network cannot tell a fresh run from a long one (the
  lag-availability flag only covers a run's first 8 bouts). The generator treats a rollout as one continuous run (the
  model never pauses), so in rollouts the clock only decays.
- **Interval handling**: each past interval enters as its deviation from the fish's own running level (an average of
  its past log intervals over ~250 bouts) instead of its raw value, so one long interval no longer drags the next ones
  up (the interval lag-1 autocorrelation is about 0.13 in the models so far against 0.06 real); and the interval
  density is truncated at 10 s, the longest interval active data contain (a longer gap ends a run).
- **Turn weight**: the turn head's loss counts × 3 on bouts with moving dots at least 60 px (17 mm) away, at any
  bearing (about 17 % of bouts), the band where the approach is decided and where the fit was measured to under-fit
  (real fish swim straight at far dots ahead, |Δθ| 26°, against the models' 30–39°).

- **8a**: + run clock (29 inputs) — **8b**: + interval handling (28) — **8c**: + both (29) — **8d**: 8c + turn weight (29).""",
         arms=[dict(key="8a", src="report3_g3h/S4i_adt_z2_rc_g3h", arm="S4i_adt_z2_rc_g3h", rej=True,
                    sub="8a — + run clock"),
               dict(key="8b", src="report3_g3h/S4i_adt_z2_cdt_tr_g3h", arm="S4i_adt_z2_cdt_tr_g3h", rej=True,
                    sub="8b — + interval handling (centred interval lags, 10 s truncation)"),
               dict(key="8c", src="report3_g3h/S4i_adt_z2_cdt_tr_rc_g3h", arm="S4i_adt_z2_cdt_tr_rc_g3h", rej=True,
                    sub="8c — + run clock + interval handling"),
               dict(key="8d", src="report3_g3h/S4i_adt_z2_cdt_tr_rc_w3ft_g3h", arm="S4i_adt_z2_cdt_tr_rc_w3ft_g3h",
                    rej=True, sub="8d — 8c + ×3 turn weight on the far moving band")]),
    dict(k=9, md="""## Model 9 — what makes Model 17 work

Model 17 is the development ladder's best drift model (standard AUC 0.672; matched 0.634, 0.559 without the radial
block): the S4 code with the static block and aligned-and-far, centred interval lags, 10 s interval truncation, run
clock, ×3 far-band turn weight, one global loading gain, and the AR(1) drift. Removing only the drift (development Model
27) took its AUC to 0.700, worse than 6a. Four arms ask where its edge comes from: 9a and 9d are evaluated from their
existing checkpoints, 9b and 9c are trained here with the same seeds. The drift is left out of the report's model; this
block measures what it is worth.""",
         arms=[dict(key="9a", src="dip4_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True,
                    sub="""9a — Model 17 as it is

`S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st`, 47 inputs: the S4 centroid code with the range bumps (18), the static bearing
block (8) and aligned-and-far (1); the last 8 turns; the last 3 displacements and the last 3 intervals (centred on the
fish's running level); the wall (3); the lag-availability flag, the block clock and the run clock. Interval head
truncated at 10 s; turn loss ×3 on the far moving band; per-fish latent (2 dimensions, one global loading gain); AR(1)
drift (τ = 70 bouts) on all three heads. Existing checkpoints (development Model 17), evaluated here."""),
               dict(key="9b", src="report4_ladder_checkpoints", arm="S4x_adt_z2_drift_cdt_tr_rc_al_w3ft", rej=True,
                    sub="""9b — Model 17 without the range bumps

Model 17's recipe retrained on 7a's code: the centroid code without bumps plus range × bearing per condition, linear in
log range (14 columns); its static terms replace the bump-based static block. 35 inputs, one global loading gain as in
9a. Paired with 9a: only the range shape differs."""),
               dict(key="9c", src="report4_g3h/S4i_adt_z2_drift_g3h", arm="S4i_adt_z2_drift_g3h", rej=True,
                    sub="""9c — Model 6a + the drift

Model 6a (no bumps, 28 inputs) with only the AR(1) drift added: a slow within-fish state (τ = 70 bouts) on all three
heads; then the per-head recalibration, as for 6a. Paired with 6a: the drift is the only difference."""),
               dict(key="9d", src="gain1_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_g3h", rej=True,
                    sub="""9d — Model 17 with per-head gains

Model 17's checkpoints with the per-head loading recalibration (one gain per head, matched to the training fish's
between-fish spread), the same step as in Models 6–8 (development Model 23). Existing checkpoints, evaluated here.
Paired with 9a: only the loading gains differ.""")]),
    dict(k=10, md="""## Model 10 — simplifying Model 17

Four arms retrained on Model 17's recipe, all without the aligned-and-far input (measured inert) and with the drift at
its likelihood-fitted size: no generation-time drift-gain search, which picks gains of 0.15–5.0 per fold and hits both
bounds (Model 9) — a lottery that would swamp these comparisons. One global loading gain, as in Model 17; same seeds as
every other arm. 10a is the control; each other arm changes one thing against it. The head-analysis cell at the end of
each arm shows where the approach is lost, condition by condition.""",
         arms=[dict(key="10a", src="report5_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_w3ft_st_g1", rej=True,
                    sub="""10a — Model 17 − aligned-and-far (the control)

46 inputs: Model 17's without aligned-and-far; drift gains 1."""),
               dict(key="10b", src="report5_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_w3ft_st_g1", rej=True,
                    sub="""10b — 10a − run clock

45 inputs. Without the drift the run clock cost AUC (Model 8a); this asks whether Model 17 needs it."""),
               dict(key="10c", src="report5_ladder_checkpoints", arm="S4_adt_z2_drift_rc_w3ft_st_g1", rej=True,
                    sub="""10c — 10a − centred interval lags − 10 s truncation

46 inputs (raw interval lags, same width). Model 17 is the only model whose interval lag-1 autocorrelation matches
real; this asks whether the centred lags are why."""),
               dict(key="10d", src="report5_ladder_checkpoints", arm="S4c_adt_z2_drift_cdt_tr_rc_w3ft_st_g1", rej=True,
                    sub="""10d — 10a + count × range × cos

50 inputs: the four range bumps × log count × cos(bearing), the mirror of the existing count × range × sin — so "how
many dots are ahead" has an input (with the dots far ahead, real fish straighten with the count and Model 17 does
not).""")]),
    dict(k=11, md="""## Model 11 — profile fine-tunes on Model 17

Model 17 as it is (9a) with a small term added to the turn head, fitted from rollouts to the matched dot-profile
curves; the network stays frozen and the likelihood is the anchor. R8's recipe, run longer: profile weight ×3, 200
steps, 16 rollouts per training block, common random numbers, warm starts, learning rate 0.1 halved at step 100 —
without the κ prior, with a weak pull on the straight term's range slope, and keeping the best running-mean step.
(⚠ The warm-start rollouts of this run met the block ~15 mm too far from the dots — see Model 13.) Every piece is
multiplied by a count curve that is **zero at one dot**, so the 1-dot behaviour (Model 17's best moving curve) is
untouched by construction.

With n the dot count, n_z = log₂n / 3, ρ = log₂(range / 70 px), g_κ(n) = 1 − e^(−(n−1)/κ), and "ahead" = within ±30°:
the straight term f is added to the straight-bout logit, the sideways term s to turns toward the dots and subtracted
from turns away.

- **11a pf-S** — straight only: f = ahead · n_z · (h + q·ρ) (2 numbers).
- **11b pf-B** — + sideways: s = sin φ · g_κ(n) · (b + e·ρ), front and behind separately (8).
- **11c pf-BS** — as 11b, but the straight term's count curve is g_κa(n) with its own κ instead of n_z (9).
- **11d pf-B+st** — 11b + the same pieces for static dots with their own numbers, and the static curves added to the
  objective (13).
- **11e** — Model 17 itself, evaluated the same way: the control.

The distance-to-dots profiles of this block use **per-block random streams**: every programme block's draws restart
from the same seed in every arm, so a static curve can only move through the state the fish carries into the block.
(For this drift model the fine-tune's likelihood anchor is taken at the drift's mean — an approximation.)""",
         arms=[dict(key="11a", src="report6_pf/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5S",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5S", rej=True, block_seed=True,
                    sub="11a — pf-S: straight at dots ahead, growing with the count"),
               dict(key="11b", src="report6_pf/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5B",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5B", rej=True, block_seed=True,
                    sub="11b — pf-B: + the sideways pull, count-gated"),
               dict(key="11c", src="report6_pf/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5BS",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5BS", rej=True, block_seed=True,
                    sub="11c — pf-BS: as pf-B, the straight term's count curve fitted (own κ)"),
               dict(key="11d", src="report6_pf/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bst",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bst", rej=True, block_seed=True,
                    sub="11d — pf-B+st: pf-B + static terms, static curves in the objective"),
               dict(key="11e", src="dip4_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st",
                    rej=True, block_seed=True, sub="11e — Model 17, evaluated the same way (the control)")]),
    dict(k=12, md="""## Model 12 — Model 17's recipe, retrained with new seeds

Is Model 17 a better recipe or a lucky draw? Every retrain of its recipe with a changed input has landed worse
(standard AUC 0.70–0.77 against 0.672), and single drift fits vary by 0.03–0.1 AUC from the draw alone (network
initialisation, loading-gain grid, drift-gain search). Here the recipe is retrained **unchanged** (development `dip4`
arm, drift-gain search included): 12a with the original seed — which should reproduce Model 17 (9a) exactly if the
pipeline is unchanged — and 12b–12d with three new seeds. A new seed also reshuffles which fish are held out
together, so each is a full repeat of the five-fold cross-validation. The spread of 12a–12d is Model 17's real level.""",
         arms=[dict(key=f"12{c}", src=f"seed17_ckpts/seed{s}", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True,
                    sub=f"12{c} — Model 17's recipe, seed {s}" + (" (the original: should reproduce 9a)" if s == 0 else ""))
               for c, s in (("a", 0), ("b", 101), ("c", 202), ("d", 303))]),
    dict(k=13, md="""## Model 13 — the profile fine-tunes redone with re-anchored warm starts

Model 11's fine-tunes, rerun unchanged except for one fix. Half of the fine-tune's rollouts are **warm starts**: the
simulated fish swims its last 16 bouts before the block itself, so it meets the block in its own state, as in the
evaluation. But the dots were replayed where they had been for the **real** fish, so these fish met the block about
15 mm too far from the dots (32 mm against 17), could never match the early curve, and kept pushing the fit toward
more pull. That pull, active all block long, is what kept Model 11's pf-B / pf-BS fish ~1.5 mm too close to the dots
after the dip (switching the term off after 20 s restored the late level; switching it off near the dots did not).
Here the replayed dots are **moved onto the simulated fish at the block onset** — the rig's own rule — and each warm
start is scored against the dots it saw. Same recipe as Model 11 otherwise (200 steps, rate halved at 100, same seeds),
so 13a–c pair with 11a–c; Model 17 on the same streams is 11e.

- **13a pf-S**, **13b pf-B**, **13c pf-BS** — as 11a–c.
- **13d pf-BS + late factor** — the sideways pull multiplied by 1 + tanh(γ)·(1 − e^(−t/20 s)), t = time since the
  block's first bout (read off the model's own 4 s block clock): one more number, the pull can relax late in the
  block but never reverse (10 numbers).""",
         arms=[dict(key=f"13{c}", src=f"report7_pf/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5{v}r",
                    arm=f"S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5{v}r", rej=True, block_seed=True, sub=sub)
               for c, v, sub in (("a", "S", "13a — pf-S, re-anchored warm starts"),
                                 ("b", "B", "13b — pf-B, re-anchored warm starts"),
                                 ("c", "BS", "13c — pf-BS, re-anchored warm starts"),
                                 ("d", "BSl", "13d — pf-BS + late-block factor, re-anchored warm starts"))]),
    dict(k=14, md="""## Model 14 — the profile term switched off after 20 s, on Model 17 and on 6a

Model 13 fixed the warm starts but left Model 11's other defect: the added pull acts all block long, so after the dip
the fish stay ~1 mm too close to the dots (40–60 s level 16.9 against 18.0 real), and a fitted late factor (13d) was not
pinned down by the objective. Switching pf-B's term off 20 s into the block restored the late level (18.4) and the
rebound while keeping the dip — but that was applied after training. Here the **cut-off is built into the term and fitted
with it in place**: the whole term (straight, sideways and static pieces) acts only while t < 20 s, t = time since the
block's first bout read off the model's own 4 s block clock; after 20 s the model is its base model again. 20 s is also
where the static curves flatten. No new number is fitted.

Same recipe as Model 13 (re-anchored warm starts, 16 rollouts per block, profile weight ×3, keep-best), but **100 steps**
with the rate halved at 50 (every earlier run converged within ~30 steps) and **no pull on q**.

- **14a** — Model 17 + pf-B with the 20 s cut-off (pairs with 13b).
- **14b** — Model 17 + pf-B + the static pieces, cut-off on all of them, static curves in the objective (pairs with 14a:
  isolates the static part; cf. 11d).
- **14c** — **6a** (no drift) + pf-B, always on — the fine-tune on the report-eligible base.
- **14d** — **6a** + pf-B with the 20 s cut-off — the report candidate (pairs with 14c).
- **14e** — 6a itself, evaluated on the same per-block streams (the control for 14c / 14d).

Expected trade-off (from the 20 s diagnostic): better 40–60 s level and rebound, but a worse 20–40 s window (real fish
are still close then) and lower grading / attraction, which read the 20–60 s level.""",
         arms=[dict(key="14a", src="report8_pf/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr", rej=True, block_seed=True,
                    sub="14a — Model 17 + pf-B, off after 20 s"),
               dict(key="14b", src="report8_pf/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bstcr",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bstcr", rej=True, block_seed=True,
                    sub="14b — Model 17 + pf-B + static pieces, off after 20 s, static curves in the objective"),
               dict(key="14c", src="report8_pf/S4i_adt_z2_g3h_pf5Br", arm="S4i_adt_z2_g3h_pf5Br", rej=True,
                    block_seed=True, sub="14c — 6a + pf-B, always on"),
               dict(key="14d", src="report8_pf/S4i_adt_z2_g3h_pf5Bcr", arm="S4i_adt_z2_g3h_pf5Bcr", rej=True,
                    block_seed=True, sub="14d — 6a + pf-B, off after 20 s (the report candidate)"),
               dict(key="14e", src="report2_g3h/S4i_adt_z2_g3h", arm="S4i_adt_z2_g3h", rej=True, block_seed=True,
                    sub="14e — 6a itself on the same per-block streams (the control)"),
               dict(key="14f", src="report8_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", rej=True, block_seed=True,
                    sub="""14f — 14a with every parameter at its FINAL value (step 100) instead of the kept step

Keep-best keeps the step with the lowest 20-step running mean of the training objective; once the loss is flat that
step is close to random (36–89 here) while several parameters still wander (straight term, behind count curve). These
two arms test whether the 14a / 14b results depend on where along that wandering the numbers were taken."""),
               dict(key="14g", src="report8_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bstcr_last",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bstcr_last", rej=True, block_seed=True,
                    sub="14g — 14b (+ static pieces) with every parameter at its FINAL value (step 100)")]),
    dict(k=15, md="""## Model 15 — simplifying Model 17, one component at a time

Model 17's exact recipe, retrained with **one component removed per arm** and everything else unchanged — including
the drift-gain search and seed 0, so every arm pairs with Model 17 itself (9a; 12a showed seed 0 reproduces it
exactly). Model 10 attempted this but also fixed the drift gains at 1 in every arm, so none of its costs could be
attributed to the component removed.

How to read it: retraining the same recipe with a new seed moves the standard AUC by ±0.014 (Model 12), so a
difference under ~0.04 is within noise. 15d and 15e keep Model 17's input width, so they start from the same initial
weights and pair more tightly; 15a–c change the width. A removal that looks free is worth confirming with a second seed.

- **15a** — without the aligned-and-far flag (1 input; inert per bout, removable at rollout time).
- **15b** — without the static block (8 inputs; worth 0.017 bits/bout).
- **15c** — without the run clock (1 input; +0.041 bits/bout, but cost AUC on 6a).
- **15d** — without the centred interval lags and the 10 s truncation (designed together with the drift).
- **15e** — without the ×3 turn weight on far moving bouts (training loss only).""",
         arms=[dict(key=f"15{c}", src="simp17_ladder_checkpoints", arm=arm, rej=True, sub=sub)
               for c, arm, sub in (("a", "S4_adt_z2_drift_cdt_tr_rc_w3ft_st", "15a — Model 17 without the aligned-and-far flag"),
                                   ("b", "S4_adt_z2_drift_cdt_tr_rc_al_w3ft", "15b — Model 17 without the static block"),
                                   ("c", "S4_adt_z2_drift_cdt_tr_al_w3ft_st", "15c — Model 17 without the run clock"),
                                   ("d", "S4_adt_z2_drift_rc_al_w3ft_st", "15d — Model 17 without centred interval lags and truncation"),
                                   ("e", "S4_adt_z2_drift_cdt_tr_rc_al_st", "15e — Model 17 without the ×3 turn weight"))]),
    dict(k=16, md="""## Model 16 — the profile correction on Model 17 without the drift

Model 27 is Model 17's recipe retrained without the AR(1) drift (same seeds). Its dot profiles are better than Model
17's (moving error 0.97 against 1.14, a deeper 2-dot dip, grading 4.9 against 4.0) but with different defects (4
moving dots too close late, 1 moving dot falling away at the end), and worse than 14f (0.82). Here 14a's correction is
applied to it unchanged: pf-B, switched off 20 s into the block, re-anchored warm starts, 100 steps with the rate halved
at 50, no pull on q.

The same correction always on (no cut-off) is fitted alongside, to see whether the cut-off matters on this base too.
Since the keep-best step turned out to be a lottery (14a vs 14f), every fit is evaluated at both its kept and its final
step:

- **16a** — Model 27 + pf-B with the 20 s cut-off, keep-best step (pairs with 14a).
- **16b** — the same with every parameter at its final value (pairs with 14f).
- **16c** — Model 27 + pf-B always on, keep-best step.
- **16d** — the same, final-step numbers.
- **16e** — Model 27 itself on the same per-block streams (the control).""",
         arms=[dict(key="16a", src="report9_pf/S4_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Bcr",
                    arm="S4_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Bcr", rej=True, block_seed=True,
                    sub="16a — Model 27 + pf-B, off after 20 s (keep-best step)"),
               dict(key="16b", src="report9_pf_last/S4_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
                    arm="S4_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", rej=True, block_seed=True,
                    sub="16b — Model 27 + pf-B, off after 20 s (final-step numbers)"),
               dict(key="16c", src="report9_pf/S4_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Br",
                    arm="S4_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Br", rej=True, block_seed=True,
                    sub="16c — Model 27 + pf-B, always on (keep-best step)"),
               dict(key="16d", src="report9_pf_last/S4_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Br_last",
                    arm="S4_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Br_last", rej=True, block_seed=True,
                    sub="16d — Model 27 + pf-B, always on (final-step numbers)"),
               dict(key="16e", src="nodrift1_ladder_checkpoints", arm="S4_adt_z2_cdt_tr_rc_al_w3ft_st", rej=True,
                    block_seed=True, sub="16e — Model 27 (Model 17 without the drift) on the same per-block streams")]),
    dict(k=18, md="""## Model 18 — simplifying Model 27's stimulus representation

Model 27 (Model 17 without the drift, the report's R5) retrained with **one change to its stimulus inputs per arm**, seed 0
and everything else unchanged, so each arm pairs with Model 27 itself (16e, the same per-block streams). Model 27's
centroid code S4 has 18 columns: presence, motion, log count, log range, the bearing (sin / cos) on four Gaussian bumps in
log range, the same bumps × log count × sin, and moving-dot sin / cos; plus an 8-column static block and the 1-column
aligned-and-far flag.

- **18a** — range bumps replaced by linear range × bearing products (S4x, 14 columns).
- **18b** — no range dependence in the bearing terms at all (S4i, 9 columns: 6a's code).
- **18c** — the simplest code (S0, 6 columns: no count × bearing) — the floor.
- **18d** — without the static block.
- **18e** — without the aligned-and-far flag.

A single-seed difference under ~0.04 standard AUC is within noise (Model 12).""",
         arms=[dict(key=f"18{c}", src="simp27s_ladder_checkpoints", arm=arm, rej=True, block_seed=True, sub=sub)
               for c, arm, sub in (("a", "S4x_adt_z2_cdt_tr_rc_al_w3ft_st", "18a — range bumps → linear range × bearing (S4x)"),
                                   ("b", "S4i_adt_z2_cdt_tr_rc_al_w3ft_st", "18b — no range in the bearing terms (S4i)"),
                                   ("c", "S0_adt_z2_cdt_tr_rc_al_w3ft_st", "18c — the simplest code (S0)"),
                                   ("d", "S4_adt_z2_cdt_tr_rc_al_w3ft", "18d — without the static block"),
                                   ("e", "S4_adt_z2_cdt_tr_rc_w3ft_st", "18e — without the aligned-and-far flag"))]),
    dict(k=19, md="""## Model 19 — simplifying Model 27's other components

Model 27 retrained with **one non-stimulus component removed per arm**, seed 0, everything else unchanged (pairs with
16e). Model 15 did this on Model 17 (with the drift), where the run clock and the centred lags + truncation turned out
to be needed and the ×3 turn weight free; here the same questions without the drift, with the lag / truncation pair
split. 19b–19d keep Model 27's input width, so they start from its exact initial weights.

- **19a** — without the run clock.
- **19b** — plain interval lags instead of centred ones (truncation kept).
- **19c** — without the 10 s interval truncation (centred lags kept).
- **19d** — without the ×3 turn weight on far moving bouts.
- **19e** — without the lag-availability flag.""",
         arms=[dict(key=f"19{c}", src="simp27o_ladder_checkpoints", arm=arm, rej=True, block_seed=True, sub=sub)
               for c, arm, sub in (("a", "S4_adt_z2_cdt_tr_al_w3ft_st", "19a — without the run clock"),
                                   ("b", "S4_adt_z2_tr_rc_al_w3ft_st", "19b — plain interval lags (truncation kept)"),
                                   ("c", "S4_adt_z2_cdt_rc_al_w3ft_st", "19c — without the 10 s truncation (centred lags kept)"),
                                   ("d", "S4_adt_z2_cdt_tr_rc_al_st", "19d — without the ×3 turn weight"),
                                   ("e", "S4_adt_z2_cdt_tr_rc_al_w3ft_st_nlv", "19e — without the lag-availability flag"))]),
    dict(k=20, md="""## Model 20 — the chain from Model 4 to the ablation model D

One change per step, cumulative, starting from Model 4 (S4i code + per-fish latent; seed 0, so every step pairs with
it), each step justified by the leave-one-out results on Model 27 (Models 18 / 19):

- **20a (A)** — Model 4 without the lag-availability flag (19e: AUC and moving error better without it).
- **20b (B)** — A + the static block (18d: the static curves need it).
- **20c (C)** — B + the run clock (19a: the grading needs it).
- **20d (D)** — C + the ×3 turn weight on far moving bouts (19d: the grading collapses without it) = **the ablation
  model**: no drift, no profile correction, no range bumps; plain interval lags and no truncation (Model 4's defaults,
  19b / 19c).
- **20e** — D retrained with a second seed (101: new folds, new initial weights) — its seed spread.
- **20f** — Model 4 itself on the same per-block streams (the control).""",
         arms=[dict(key="20a", src="chainD_ladder_checkpoints", arm="S4i_adt_z2_nlv", rej=True, block_seed=True,
                    sub="20a — A: Model 4 without the lag-availability flag"),
               dict(key="20b", src="chainD_ladder_checkpoints", arm="S4i_adt_z2_st_nlv", rej=True, block_seed=True,
                    sub="20b — B: + static block"),
               dict(key="20c", src="chainD_ladder_checkpoints", arm="S4i_adt_z2_rc_st_nlv", rej=True, block_seed=True,
                    sub="20c — C: + run clock"),
               dict(key="20d", src="chainD_ladder_checkpoints", arm="S4i_adt_z2_rc_w3ft_st_nlv", rej=True,
                    block_seed=True, sub="20d — D: + ×3 turn weight = the ablation model"),
               dict(key="20e", src="chainD_seed101_ckpts", arm="S4i_adt_z2_rc_w3ft_st_nlv", rej=True, block_seed=True,
                    sub="20e — D, second seed (101)"),
               dict(key="20f", src="report1_ladder_checkpoints", arm="S4i_adt_z2", rej=True, block_seed=True,
                    sub="20f — Model 4 on the same per-block streams (the control)"),
               ]),
    dict(k=24, md="""## Model 24 — 18a: a second seed, the drift and the profile correction

18a is Model 27 with the range bumps replaced by linear range × bearing products (S4x): among the no-drift models its
matched AUC without radial is the best (0.568), with the same dot profiles as 18b. Here its seed spread, the drift step
and the correction step:

- **24a** — 18a retrained with seed 101 (new folds and initial weights; pairs with 18a for the spread).
- **24b** — 18a + Model 17's AR(1) drift, seed 0 (pairs with 18a). With the drift, Model 17 without its bumps did badly
  (9b, single seed, no static block) — does the drift need the bumps?
- **24c** — the same at seed 101 (pairs with 24a).
- **24d** — 18a + the profile correction (pf-B on moving dots, off after 20 s, re-anchored warm starts, 100 steps,
  final-step numbers), pairs with 18a.""",
         arms=[dict(key="24a", src="simp27s_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_st", rej=True,
                    block_seed=True, sub="24a — 18a, seed 101"),
               dict(key="24b", src="s4xdrift_ladder_checkpoints", arm="S4x_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True,
                    block_seed=True, sub="24b — 18a + drift, seed 0"),
               dict(key="24c", src="s4xdrift_seed101_ckpts", arm="S4x_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True,
                    block_seed=True, sub="24c — 18a + drift, seed 101"),
               dict(key="24d", src="report11_pf_last/S4x_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
                    arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", rej=True, block_seed=True,
                    sub="24d — 18a + profile correction (final-step numbers)")]),
    dict(k=25, md="""## Model 25 — each component of 18a ALONE on Model 4

Model 4 (S4i code + per-fish latent, global gain) plus exactly one of the components 18a adds, seed 0 (pairs with
Model 4, 20f). With Model 26 (each removed from 18a) this measures every component both on its own and with all the
others present -- the components interact, so a chain built one step at a time from Model 4 does not have to improve at
every step (Model 20).

- **25a** — + centred interval lags and the 10 s truncation (report3's fit).
- **25b** — + run clock (report3's fit).
- **25c** — + aligned-and-far flag.
- **25d** — + ×3 turn weight on far moving bouts.
- **25e** — + static block.
- **25f** — + linear range × bearing products (the S4x code; report2's fit).""",
         arms=[dict(key="25a", src="report3_ladder_checkpoints", arm="S4i_adt_z2_cdt_tr", rej=True, block_seed=True,
                    sub="25a — Model 4 + centred lags and truncation"),
               dict(key="25b", src="report3_ladder_checkpoints", arm="S4i_adt_z2_rc", rej=True, block_seed=True,
                    sub="25b — Model 4 + run clock"),
               dict(key="25c", src="m4plus_ladder_checkpoints", arm="S4i_adt_z2_al", rej=True, block_seed=True,
                    sub="25c — Model 4 + aligned-and-far flag"),
               dict(key="25d", src="m4plus_ladder_checkpoints", arm="S4i_adt_z2_w3ft", rej=True, block_seed=True,
                    sub="25d — Model 4 + ×3 turn weight"),
               dict(key="25e", src="m4plus_ladder_checkpoints", arm="S4i_adt_z2_st", rej=True, block_seed=True,
                    sub="25e — Model 4 + static block"),
               dict(key="25f", src="report2_ladder_checkpoints", arm="S4x_adt_z2", rej=True, block_seed=True,
                    sub="25f — Model 4 + linear range × bearing products (S4x)")]),
    dict(k=26, md="""## Model 26 — each component REMOVED from 18a

18a minus exactly one component, seed 0 (pairs with 18a). 18a minus the range products is 18b (Model 18).

- **26a** — without the centred interval lags and the 10 s truncation.
- **26b** — without the run clock.
- **26c** — without the aligned-and-far flag.
- **26d** — without the ×3 turn weight.
- **26e** — without the static block.""",
         arms=[dict(key=f"26{c}", src="simp18a_ladder_checkpoints", arm=arm, rej=True, block_seed=True, sub=sub)
               for c, arm, sub in (("a", "S4x_adt_z2_rc_al_w3ft_st", "26a — 18a without centred lags and truncation"),
                                   ("b", "S4x_adt_z2_cdt_tr_al_w3ft_st", "26b — 18a without the run clock"),
                                   ("c", "S4x_adt_z2_cdt_tr_rc_w3ft_st", "26c — 18a without the aligned-and-far flag"),
                                   ("d", "S4x_adt_z2_cdt_tr_rc_al_st", "26d — 18a without the ×3 turn weight"),
                                   ("e", "S4x_adt_z2_cdt_tr_rc_al_w3ft", "26e — 18a without the static block"))]),
    dict(k=28, md="""## Model 28 — which profile correction: the 20 s cut-off or always on, kept or final step

pf-B (8 numbers: straight h, q; sideways b, e, κ for dots in front and behind; moving dots only, zero at one dot) fitted
with the same recipe everywhere — re-anchored warm starts, 100 steps with the rate halved at 50, no pull on q, seed 0 —
and every fit evaluated twice: at the keep-best step and with every parameter at its final value.

- **28a / 28b** — 26e (the no-drift ablation model) + pf-B with the 20 s cut-off: kept / final.
- **28c / 28d** — 26e + pf-B always on: kept / final.
- **28e / 28f** — Model 17 + pf-B always on: kept / final — 14a / 14f's exact recipe without the cut-off.""",
         arms=[dict(key=k, src=f"{d}/{a}{'_last' if last else ''}", arm=f"{a}{'_last' if last else ''}", rej=True,
                    block_seed=True, sub=sub)
               for k, d, a, last, sub in (
                   ("28a", "report12_pf", "S4x_adt_z2_cdt_tr_rc_al_w3ft_pf5Bcr", False, "28a — 26e + pf-B, 20 s cut-off, keep-best step"),
                   ("28b", "report12_pf_last", "S4x_adt_z2_cdt_tr_rc_al_w3ft_pf5Bcr", True, "28b — 26e + pf-B, 20 s cut-off, final step"),
                   ("28c", "report12_pf", "S4x_adt_z2_cdt_tr_rc_al_w3ft_pf5Br", False, "28c — 26e + pf-B always on, keep-best step"),
                   ("28d", "report12_pf_last", "S4x_adt_z2_cdt_tr_rc_al_w3ft_pf5Br", True, "28d — 26e + pf-B always on, final step"),
                   ("28e", "report12_pf", "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Br", False, "28e — Model 17 + pf-B always on, keep-best step"),
                   ("28f", "report12_pf_last", "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Br", True, "28f — Model 17 + pf-B always on, final step"))]),
    dict(k=29, md="""## Model 29 — the missing main-line steps, and the ablations of 26e

The main line Model 4 → 25d (+ ×3 turn weight) → **+ range products** → **+ the temporal set** (centred lags + 10 s
truncation, run clock, aligned-and-far flag) = 26e, the no-drift ablation model (18a without the static block). Seed 0.

- **29a** — 25d + the range products (S4x code): Model 4 + ×3 turn weight on S4x.
- **29b** — 25d + the temporal set, i.e. the steps in the other order (= 18b without the static block).
- **29c** — 26e without the dot count (no log count, count × sin, count × range × sin).
- **29d** — 26e without the moving indicator (no moving flag, moving × sin / cos, moving and static range × bearing
  products, no aligned-and-far flag, which reads the moving flag).
- **29e** — 26e without any stimulus information (no stimulus columns, no aligned-and-far flag, no block clock).""",
         arms=[dict(key=f"29{c}", src="run1a_ladder_checkpoints", arm=arm, rej=True, block_seed=True, sub=sub)
               for c, arm, sub in (("a", "S4x_adt_z2_w3ft", "29a — 25d + range products (S4x)"),
                                   ("b", "S4i_adt_z2_cdt_tr_rc_al_w3ft", "29b — 25d + the temporal set (other order)"),
                                   ("c", "S4xnn_adt_z2_cdt_tr_rc_al_w3ft", "29c — 26e without the dot count"),
                                   ("d", "S4xnm_adt_z2_cdt_tr_rc_w3ft", "29d — 26e without the moving indicator"),
                                   ("e", "Snull_adt_z2_cdt_tr_rc_w3ft_nck", "29e — 26e without any stimulus information"))]),
    dict(k=30, md="""## Model 30 — the main line at a second seed (101)

Seed 101 redraws the cross-validation folds and the initial weights, so these pair only with each other and with the
other seed-101 fits (Model 17's 12b, 18a's 24a) — never with the seed-0 models (different folds).

- **30a** Model 4, **30b** 25d (+ ×3 turn weight), **30c** 29a (+ range products), **30d** 26e (+ temporal set = the
  ablation model), **30e** Model 27.""",
         arms=[dict(key=f"30{c}", src=src, arm=arm, rej=True, block_seed=True, sub=sub)
               for c, src, arm, sub in (("a", "report1_seed101_ckpts", "S4i_adt_z2", "30a — Model 4, seed 101"),
                                        ("b", "m4plus_seed101_ckpts", "S4i_adt_z2_w3ft", "30b — 25d, seed 101"),
                                        ("c", "run1a_seed101_ckpts", "S4x_adt_z2_w3ft", "30c — 25d + range products, seed 101"),
                                        ("d", "simp18a_seed101_ckpts", "S4x_adt_z2_cdt_tr_rc_al_w3ft", "30d — 26e, seed 101"),
                                        ("e", "nodrift1_seed101_ckpts", "S4_adt_z2_cdt_tr_rc_al_w3ft_st", "30e — Model 27, seed 101"))]),
    dict(k=31, md="""## Model 31 — two dot groups at once: averaging or choosing?

The recorded programme never shows two dot groups at the same time (0 % of 39,542 stimulus bouts have dots on both
sides), so how a trained model responds to two groups is **not learned**: it is fixed by the rule that turns two groups
into the network's input. Four rules, all on the same trained weights, no retraining:

| Rule | What the network is given | Inputs stay valid? | Hypothesis |
|---|---|---|---|
| **effects** (main) | each group alone, plus a no-dots baseline; every head's output = baseline + the sum of the groups' effects | yes | independent pulls that add |
| **add** | one row with the groups' stimulus columns summed | no (cos > 1 etc.) | drives add before the nonlinearity, so groups can interact |
| **avg** | one row with the columns averaged | yes | averaging (close to the old one-centroid code) |
| **nearest** | the nearest group only | yes | winner-take-all |

With one group every rule gives exactly the standard input, so every model's single-group behaviour is unchanged:
checked bit for bit (below). The profile correction is evaluated on each group's own features and summed (effects /
add), averaged (avg) or taken from the nearest group; the aligned-and-far flag of a combined row is the max over groups;
the block clock is shared (time since the scene's onset).

**Models:** 28b (26e + pf-B with the 20 s cut-off, final step: the no-drift model) and 14f (Model 17 + the same
correction: the best model overall). 50 simulated fish per scene and condition (10 per fold model), each with its own
latent draw; the same fish (start pose, latent, drift, per-block random streams, dot tracks) under every condition, so
every comparison is paired by fish.

**Scenes:** a 100 s lead-in pause, then 10 blocks of 60 s with two groups and a 50 s pause. At each block onset group A
is placed at bearing +θ and distance d_A from the fish, group B at −θ, d_B; A swaps sides every block. Each group is a
whole recorded dot group of its condition, rotated about the fish's onset point to the bearing and **scaled** about it
to the distance (its path relative to the fish keeps the recorded shape); templates = every recorded block of the
condition that started 14–28 mm out, right-side recordings mirrored, drawn at random per fish, block and group; two
groups of the same condition are exact mirror images, so a symmetric scene is exactly symmetric. Sweeps: angular
separation 2θ = 30–180° (4 v 4 moving, 20 mm); counts 2 v 8, 2 v 4, 4 v 8 (moving, ±90°, 20 mm); near (14 mm) v far
(27 mm); a 2-dot v 8-dot grid over both distances (14 / 20 / 27 mm); moving v static (4 v 4, 8 v 8).

**References, on the same tracks:** every scene is also run with only A shown, only B shown and neither (the hidden
groups are computed but never given to the model). The side index has a geometric floor — two groups 30° apart are only
11 mm apart, so even a fish committed to one of them often scores "between" — and the placement alone favours some
groups; every two-group number is therefore read **against these references**, paired by fish.

**Readouts per trial (block):** side index SI = (d_B − d_A)/(d_A + d_B) (+1 at A, −1 at B); **choice** = sign of the
mean SI over 20–60 s, *between* if |mean SI| < 0.2; late (20–60 s) distance to the **nearer** and the **farther** group
(choosing = near small and far large; averaging = both moderate; ignoring = both large); **decision** = |SI| > 0.5 for
10 s, as the fraction of all trials decided by time t; the turn toward each group. P(A), P(B), P(between) = mean over
simulated fish of each fish's fractions, 95 % bootstrap CI over fish. Trials are kept when both groups start inside the
dish and stay inside for ≥ 3/4 of 20–60 s (`dropped_pct`; set by the placement, not by behaviour).

These are predictions, not fits: any of the four rules is consistent with every recorded bout. A two-group experiment
would decide between them. (Version 2: the first version translated the tracks instead of scaling them, used 6 fixed
track pairs and had no references, which biased the side choices and the distance and grid readouts; its outputs are
kept in `multi_stim_v1/`.)""",
         cells=[
             ("val", """# Model 31: validation -- with ONE group shown, every rule reproduces the standard simulator bit for bit (the
# recorded programme, first 30 min, and a scene with only A shown, 3 fish per model); a hidden group changes nothing;
# symmetric scenes are exactly symmetric at onset; every two-group run is finite and differs.
import json
import multi_stim as MS
print(json.dumps(json.load(open(MS.OUT / "validation.json")), indent=1))"""),
             ("tab", """# Model 31: per scene and condition (the four rules, then only A shown / only B shown / neither shown).
# P_* = mean over simulated fish of each fish's fraction of kept trials; d_near / d_far = late (20-60 s) distance to the
# nearer / farther group (mm); decided = fraction of trials with |SI| > 0.5 for 10 s. For the rules, paired by fish
# against the references: dbtw_chooser = P(between) minus a fish shown one group (> 0: more often between than a
# committed fish); pref_corr = P(A) - P(B) minus that of a fish shown neither (the preference beyond the placement);
# [lo, hi] = 95 % CI over fish.
S31 = MS.summary_table()
S31.to_csv(MS.OUT / "summary_table.csv", index=False)
pd.set_option("display.max_rows", 400)
_c = ["P_A", "P_between", "P_B", "d_near_late", "d_far_late", "decided", "n_trials", "dropped_pct",
      "dbtw_chooser", "dbtw_chooser_lo", "dbtw_chooser_hi", "pref_corr", "pref_corr_lo", "pref_corr_hi"]
for _tag in ("28b", "14f"):
    _t = S31[S31.model == _tag].set_index(["sweep", "scene", "cond"])[_c]
    print(MS.MODEL_LABEL[_tag]); print(_t.round(3).to_string()); print()"""),
             ("ang", """# Model 31: angular separation (4 v 4 moving at 20 mm) -- P(between) per rule against a fish shown one group (dashed)
# and a fish shown neither (dotted), on the same tracks. Both references fall with separation (geometry): read the gap.
MS.plot_angle(S31);"""),
             ("late", """# Model 31: late distance to the nearer and to the farther group by separation: choosing = near small and far large;
# averaging = both moderate (closer to both than a committed fish); ignoring = both large (the neither-shown reference).
MS.plot_late_distances(S31);"""),
             ("pref", """# Model 31: count, distance and motion -- P(A) - P(B) beyond the placement (minus a fish shown neither), A = the
# first-named group (the smaller, the nearer, the moving one); triangles = a fish shown only A / only B (the most a
# fully committed fish shows on these tracks).
MS.plot_preference(S31);"""),
             ("grid", """# Model 31: count against distance -- P(8 dots) - P(2 dots) beyond the placement, each group at 14, 20 or 27 mm.
MS.plot_grid(S31);"""),
             ("hist", """# Model 31: per trial, the mean SI over 20-60 s, with a fish shown only A (black dashed) and shown neither (grey).
MS.plot_si_hist(scene_names=("ang60", "ang180", "n2v8", "mv4_st4"));"""),
             ("heat", """# Model 31: SI over the block, one row per kept trial (sorted), 4 v 4 at 180 deg, with the references.
for _tag in ("28b", "14f"):
    MS.plot_si_heat(_tag, "ang180")"""),
             ("dist", """# Model 31: distance to the chosen and to the other group over the block, next to the same split of a fish shown only
# A (selected by the same SI rule, so the curves compare fairly); dotted = the trials that stayed between.
for _tag in ("28b", "14f"):
    MS.plot_dist_by_choice(_tag, "ang180")
    MS.plot_dist_by_choice(_tag, "ang60")"""),
             ("decide", """# Model 31: fraction of all kept trials decided by time t (|SI| > 0.5 for 10 s), per rule and reference.
MS.plot_decision();"""),
             ("tow", """# Model 31: the turn toward each group over the block (deg per bout), 2 v 8 and moving v static; last panel = a fish
# shown only A (its turn toward B, which it cannot see, is the null).
for _tag in ("28b", "14f"):
    MS.plot_toward(_tag, "n2v8")
    MS.plot_toward(_tag, "mv4_st4")"""),
         ]),
    dict(k=32, md="""## Model 32 — Model 17 and 14f at seed 101, on per-block streams

The two seed-101 results the storyline was missing. **32a** is Model 17 at seed 101: the checkpoints of 12b, evaluated
again on per-block random streams like every other seed-101 model (12b used the old single stream: its likelihood and
AUC are identical, its profile numbers did not pair). **32b** is 14f at seed 101: the same correction as 14f (pf-B, 20 s
cut-off, re-anchored warm starts, 100 steps halved at 50, no pull on q, final-step numbers) fitted on Model 17 seed 101.
Both pair with 30a–30e, 24a and 24c (seed 101) — never with the seed-0 models.""",
         arms=[dict(key="32a", src="seed17_ckpts/seed101", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True,
                    block_seed=True, sub="32a — Model 17, seed 101, per-block streams (12b's checkpoints)"),
               dict(key="32b", src="report13_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", rej=True, block_seed=True,
                    sub="32b — 14f at seed 101: Model 17 seed 101 + pf-B, 20 s cut-off, final step")]),
    dict(k=33, nb=False, md="""## Model 33 — the ablation study on 26e

Shown only in Final_report_ablations.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="33a", src="abl26e_ladder_checkpoints", arm="Scond_adt_z2_cdt_tr_rc_w3ft", rej=True, block_seed=True, sub="33a — moving vs static only, seed 0"),
               dict(key="33b", src="abl26e_seed101_ckpts", arm="Scond_adt_z2_cdt_tr_rc_w3ft", rej=True, block_seed=True, sub="33b — moving vs static only, seed 101"),
               dict(key="33c", src="abl26e_ladder_checkpoints", arm="S4xnr_adt_z2_cdt_tr_rc_w3ft", rej=True, block_seed=True, sub="33c — range removed, seed 0"),
               dict(key="33d", src="abl26e_seed101_ckpts", arm="S4xnr_adt_z2_cdt_tr_rc_w3ft", rej=True, block_seed=True, sub="33d — range removed, seed 101"),
               dict(key="33e", src="abl26e_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_h4", rej=True, block_seed=True, sub="33e — turn history 4 bouts, seed 0"),
               dict(key="33f", src="abl26e_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_h4", rej=True, block_seed=True, sub="33f — turn history 4 bouts, seed 101"),
               dict(key="33g", src="abl26e_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_h1", rej=True, block_seed=True, sub="33g — turn history 1 bout, seed 0"),
               dict(key="33h", src="abl26e_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_h1", rej=True, block_seed=True, sub="33h — turn history 1 bout, seed 101"),
               dict(key="33i", src="abl26e_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_h0", rej=True, block_seed=True, sub="33i — no turn history, seed 0"),
               dict(key="33j", src="abl26e_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_h0", rej=True, block_seed=True, sub="33j — no turn history, seed 101"),
               dict(key="33k", src="abl26e_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk1", rej=True, block_seed=True, sub="33k — kinematic history 1 bout, seed 0"),
               dict(key="33l", src="abl26e_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk1", rej=True, block_seed=True, sub="33l — kinematic history 1 bout, seed 101"),
               dict(key="33m", src="abl26e_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk0", rej=True, block_seed=True, sub="33m — no kinematic history, seed 0"),
               dict(key="33n", src="abl26e_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk0", rej=True, block_seed=True, sub="33n — no kinematic history, seed 101"),
               dict(key="33o", src="abl26e_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_nck", rej=True, block_seed=True, sub="33o — no block clock, seed 0"),
               dict(key="33p", src="abl26e_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_nck", rej=True, block_seed=True, sub="33p — no block clock, seed 101"),
               dict(key="33q", src="abl_z0_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_z0", rej=True, block_seed=True, sub="33q — per-fish latent frozen (z = 0), seed 0"),
               dict(key="33r", src="abl_z0_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_z0", rej=True, block_seed=True, sub="33r — per-fish latent frozen (z = 0), seed 101"),
               dict(key="33s", src="run1a_seed101_ckpts", arm="S4xnn_adt_z2_cdt_tr_rc_al_w3ft", rej=True, block_seed=True, sub="33s — group size removed (29c), seed 101"),
               dict(key="33t", src="run1a_seed101_ckpts", arm="S4xnm_adt_z2_cdt_tr_rc_w3ft", rej=True, block_seed=True, sub="33t — moving indicator removed (29d), seed 101"),
               dict(key="33u", src="run1a_seed101_ckpts", arm="Snull_adt_z2_cdt_tr_rc_w3ft_nck", rej=True, block_seed=True, sub="33u — no stimulus (29e), seed 101")]),
    dict(k=34, nb=False, md="""## Model 34 — the missing pieces (Part B: retrained units)

Shown only in Final_report_missingPieces.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="34a", src="report1_seed101_ckpts", arm="S0_adt", rej=False, block_seed=True, sub="34a — Model 1 (M1), seed 101"),
               dict(key="34b", src="report1_seed101_ckpts", arm="S0_adt", rej=True, block_seed=True, sub="34b — Model 2 (M2 = M1 + wall rejection), seed 101"),
               dict(key="34c", src="report1_seed101_ckpts", arm="S0_adt_z2", rej=True, block_seed=True, sub="34c — Model 3 (M3), seed 101"),
               dict(key="34d", src="run1a_seed101_ckpts", arm="S4i_adt_z2_cdt_tr_rc_al_w3ft", rej=True, block_seed=True, sub="34d — 29b (M5+T), seed 101"),
               dict(key="34e", src="missing_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_lo6m", rej=True, block_seed=True, sub="34e — 26e trained without 6 moving dots, seed 0"),
               dict(key="34f", src="missing_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_lo6m", rej=True, block_seed=True, sub="34f — 26e trained without 6 moving dots, seed 101"),
               dict(key="34g", src="missing_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_w64", rej=True, block_seed=True, sub="34g — 26e with 64 hidden units, seed 0"),
               dict(key="34h", src="missing_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_w64", rej=True, block_seed=True, sub="34h — 26e with 64 hidden units, seed 101"),
               dict(key="34i", src="missing_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_w256", rej=True, block_seed=True, sub="34i — 26e with 256 hidden units, seed 0"),
               dict(key="34j", src="missing_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_w256", rej=True, block_seed=True, sub="34j — 26e with 256 hidden units, seed 101")]),
    dict(k=35, nb=False, md="""## Model 35 — the ablation study on 26e, second round

26e minus the components the first round never removed from 26e itself (the temporal set and the ×3 turn weight).
Shown only in Final_report_ablations.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="35a", src="abl26e2_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_al_w3ft", rej=True, block_seed=True, sub="35a — run clock removed, seed 0"),
               dict(key="35b", src="abl26e2_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_al_w3ft", rej=True, block_seed=True, sub="35b — run clock removed, seed 101"),
               dict(key="35c", src="abl26e2_ladder_checkpoints", arm="S4x_adt_z2_rc_al_w3ft", rej=True, block_seed=True, sub="35c — centred interval lags and 10 s truncation removed, seed 0"),
               dict(key="35d", src="abl26e2_seed101_ckpts", arm="S4x_adt_z2_rc_al_w3ft", rej=True, block_seed=True, sub="35d — centred interval lags and 10 s truncation removed, seed 101"),
               dict(key="35e", src="abl26e2_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_w3ft", rej=True, block_seed=True, sub="35e — aligned-and-far flag removed, seed 0"),
               dict(key="35f", src="abl26e2_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_w3ft", rej=True, block_seed=True, sub="35f — aligned-and-far flag removed, seed 101"),
               dict(key="35g", src="abl26e2_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al", rej=True, block_seed=True, sub="35g — ×3 turn weight removed, seed 0"),
               dict(key="35h", src="abl26e2_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al", rej=True, block_seed=True, sub="35h — ×3 turn weight removed, seed 101")]),
    dict(k=36, nb=False, md="""## Model 36 — scaling up: three hidden layers

M6, D4 and F with three hidden layers of 128 units instead of one, everything else unchanged.
Shown only in Final_report_models.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="36a", src="deep3_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_d3", rej=True, block_seed=True, sub="36a — M6 (26e) with three hidden layers, seed 0"),
               dict(key="36b", src="deep3_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_d3", rej=True, block_seed=True, sub="36b — M6 (26e) with three hidden layers, seed 101"),
               dict(key="36c", src="deep3_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d3", rej=True, block_seed=True, sub="36c — D4 (Model 17) with three hidden layers, seed 0"),
               dict(key="36d", src="deep3_seed101_ckpts", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d3", rej=True, block_seed=True, sub="36d — D4 (Model 17) with three hidden layers, seed 101"),
               dict(key="36e", src="deep3_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d3_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d3_pf5Bcr_last", rej=True, block_seed=True, sub="36e — F (14f) with three hidden layers: 36c + pf-B, 20 s cut-off, final step, seed 0"),
               dict(key="36f", src="deep3_s101_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d3_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d3_pf5Bcr_last", rej=True, block_seed=True, sub="36f — F (14f) with three hidden layers: 36d + pf-B, 20 s cut-off, final step, seed 101")]),
    dict(k=37, nb=False, md="""## Model 37 — the slow state's time constant: tau = 200 bouts (D4, F)

D4 and F with the AR(1) drift's time constant at 200 bouts instead of 70, everything else unchanged; the real
window-to-window correlation decays with tau ~160-220 bouts (displacement, radial position).
Shown only in Final_report_models.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="37a", src="tau200_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200", rej=True, block_seed=True, sub="37a — D4 with tau = 200 bouts, seed 0"),
               dict(key="37b", src="tau200_seed101_ckpts", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200", rej=True, block_seed=True, sub="37b — D4 with tau = 200 bouts, seed 101"),
               dict(key="37c", src="tau200_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_pf5Bcr_last", rej=True, block_seed=True, sub="37c — F with tau = 200 bouts: 37a + pf-B, 20 s cut-off, final step, seed 0"),
               dict(key="37d", src="tau200_s101_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_pf5Bcr_last", rej=True, block_seed=True, sub="37d — F with tau = 200 bouts: 37b + pf-B, 20 s cut-off, final step, seed 101")]),
    dict(k=38, nb=False, md="""## Model 38 — turn-size individuality (M6, D4, F)

M6, D4 and F with the per-fish latent ALSO loading on the scales of the four turning components (re_tsize), so how
far a fish turns is a trait alongside how often it swims straight; everything else unchanged.
Shown only in Final_report_models.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="38a", src="tsize_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_tz", rej=True, block_seed=True, sub="38a — M6 + turn-size latent, seed 0"),
               dict(key="38b", src="tsize_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_tz", rej=True, block_seed=True, sub="38b — M6 + turn-size latent, seed 101"),
               dict(key="38c", src="tsize_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_tz", rej=True, block_seed=True, sub="38c — D4 + turn-size latent, seed 0"),
               dict(key="38d", src="tsize_seed101_ckpts", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_tz", rej=True, block_seed=True, sub="38d — D4 + turn-size latent, seed 101"),
               dict(key="38e", src="tsize_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_tz_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_tz_pf5Bcr_last", rej=True, block_seed=True, sub="38e — F + turn-size latent: 38c + pf-B, 20 s cut-off, final step, seed 0"),
               dict(key="38f", src="tsize_s101_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_tz_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_tz_pf5Bcr_last", rej=True, block_seed=True, sub="38f — F + turn-size latent: 38d + pf-B, 20 s cut-off, final step, seed 101")]),
    dict(k=46, nb=False, md="""## Model 46 — turn-size latent with tau = 200 (D4, F)

D4 and F with BOTH changes of Models 37 and 38: the AR(1) slow state at tau = 200 bouts and the per-fish latent also
loading on the scales of the four turning components (re_tsize); everything else unchanged.
Shown only in Final_report_models.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="46a", src="tsize200_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_tz", rej=True, block_seed=True, sub="46a — D4 + turn-size latent, tau = 200, seed 0"),
               dict(key="46b", src="tsize200_seed101_ckpts", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_tz", rej=True, block_seed=True, sub="46b — D4 + turn-size latent, tau = 200, seed 101"),
               dict(key="46c", src="tsize200_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_tz_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_tz_pf5Bcr_last", rej=True, block_seed=True, sub="46c — F + turn-size latent, tau = 200: 46a + pf-B, 20 s cut-off, final step, seed 0"),
               dict(key="46d", src="tsize200_s101_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_tz_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_tz_pf5Bcr_last", rej=True, block_seed=True, sub="46d — F + turn-size latent, tau = 200: 46b + pf-B, 20 s cut-off, final step, seed 101")]),
    dict(k=47, nb=False, md="""## Model 47 — D4 with the drift strength fixed instead of calibrated per fold

The drift's generation gains are normally re-calibrated per fold model by a coarse search; in some folds the likelihood
learns a near-zero displacement loading and the search's cap leaves them with no displacement drift. Here every fold
gets the same EFFECTIVE strength (gain x |loading|), fixed in advance at the median the tau-70 calibration reached
(displacement 0.05, interval 0.13), half of it, or 0, at tau 70 and tau 200 (same checkpoints as Model 17 / 32a and
37a / 37b, gains rewritten; no retraining; likelihood unchanged by construction).
Shown only in Final_report_models.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="47a", src="driftfix/amp0_s0", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True, block_seed=True, sub="47a — D4, no drift at generation (strength 0), seed 0"),
               dict(key="47b", src="driftfix/t70_half_s0", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True, block_seed=True, sub="47b — D4, tau 70, half strength (displacement 0.025, interval 0.065), seed 0"),
               dict(key="47c", src="driftfix/t70_full_s0", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True, block_seed=True, sub="47c — D4, tau 70, full strength (displacement 0.05, interval 0.13), seed 0"),
               dict(key="47d", src="driftfix/t200_half_s0", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200", rej=True, block_seed=True, sub="47d — D4, tau 200, half strength, seed 0"),
               dict(key="47e", src="driftfix/t200_full_s0", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200", rej=True, block_seed=True, sub="47e — D4, tau 200, full strength, seed 0"),
               dict(key="47f", src="driftfix/amp0_s101", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True, block_seed=True, sub="47f — D4, no drift at generation (strength 0), seed 101"),
               dict(key="47g", src="driftfix/t70_half_s101", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True, block_seed=True, sub="47g — D4, tau 70, half strength (displacement 0.025, interval 0.065), seed 101"),
               dict(key="47h", src="driftfix/t70_full_s101", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True, block_seed=True, sub="47h — D4, tau 70, full strength (displacement 0.05, interval 0.13), seed 101"),
               dict(key="47i", src="driftfix/t200_half_s101", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200", rej=True, block_seed=True, sub="47i — D4, tau 200, half strength, seed 101"),
               dict(key="47j", src="driftfix/t200_full_s101", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200", rej=True, block_seed=True, sub="47j — D4, tau 200, full strength, seed 101")]),
    dict(k=48, nb=False, md="""## Model 48 — F built on D4 generated without its drift (strength 0)

47a | 47f (D4's weights, the drift switched off at generation) + the same profile correction as F (pf-B, 20 s cut-off,
final step), fitted on rollouts that also run without the drift.
Shown only in Final_report_models.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="48a", src="driftfix0_s0_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", rej=True, block_seed=True, sub="48a — F on strength-0 D4, seed 0"),
               dict(key="48b", src="driftfix0_s101_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", rej=True, block_seed=True, sub="48b — F on strength-0 D4, seed 101")]),
    dict(k=41, nb=False, md="""## Model 41 — the dot-jump phase input (M6, D4, F)

Shown only in Model_design_and_training.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="41a", src="jump_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_jp", rej=True, block_seed=True, sub="41a — M6 + jump phase, seed 0"),
               dict(key="41b", src="jump_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_jp", rej=True, block_seed=True, sub="41b — M6 + jump phase, seed 101"),
               dict(key="41c", src="jump_ladder_checkpoints", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_jp", rej=True, block_seed=True, sub="41c — D4 + jump phase, seed 0"),
               dict(key="41d", src="jump_seed101_ckpts", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_jp", rej=True, block_seed=True, sub="41d — D4 + jump phase, seed 101"),
               dict(key="41e", src="report14_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_jp_pf5Bcr_last",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_jp_pf5Bcr_last", rej=True, block_seed=True,
                    sub="41e — F + jump phase (41c + pf-B, 20 s cut-off, final step), seed 0"),
               dict(key="41f", src="report15_pf_last/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_jp_pf5Bcr_last",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_jp_pf5Bcr_last", rej=True, block_seed=True,
                    sub="41f — F + jump phase (41d + pf-B, 20 s cut-off, final step), seed 101")]),
    dict(k=42, nb=False, md="""## Model 42 — the lower-clip check (M6, D4, F)

The log range rz clipped below at the 2 px floor (−2.56) instead of −1 (17.5 px); everything else exactly as M6, D4, F.
Shown only in Final_report_models.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="42a", src="rzlo_ladder_checkpoints", arm="S4xlo_adt_z2_cdt_tr_rc_al_w3ft", rej=True, block_seed=True, sub="42a — M6, lower clip at 2 px, seed 0"),
               dict(key="42b", src="rzlo_seed101_ckpts", arm="S4xlo_adt_z2_cdt_tr_rc_al_w3ft", rej=True, block_seed=True, sub="42b — M6, lower clip at 2 px, seed 101"),
               dict(key="42c", src="rzlo_ladder_checkpoints", arm="S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True, block_seed=True, sub="42c — D4, lower clip at 2 px, seed 0"),
               dict(key="42d", src="rzlo_seed101_ckpts", arm="S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rej=True, block_seed=True, sub="42d — D4, lower clip at 2 px, seed 101"),
               dict(key="42e", src="rzlo_pf_last/S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
                    arm="S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", rej=True, block_seed=True,
                    sub="42e — F, lower clip at 2 px (42c + pf-B, 20 s cut-off, final step), seed 0"),
               dict(key="42f", src="rzlo_s101_pf_last/S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last",
                    arm="S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last", rej=True, block_seed=True,
                    sub="42f — F, lower clip at 2 px (42d + pf-B, 20 s cut-off, final step), seed 101")]),
    dict(k=43, nb=False, md="""## Model 43 — F + a repulsion term fitted to the Larsch screen data

A fixed-form per-dot term (fit_MLP_joint.repulsion_shifts; network and profile correction untouched), fitted on half the
screen's clutches (scripts/repulsion_fit.py): T = turn-away only, TB = turn-away + brake (shorter steps toward a near dot).
Shown only in Final_report_extra_data.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="43a", src="rep43_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repT",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repT", rej=True, block_seed=True,
                    sub="43a — F + repulsion (turn), seed 0"),
               dict(key="43b", src="rep43_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repT",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repT", rej=True, block_seed=True,
                    sub="43b — F + repulsion (turn), seed 101"),
               dict(key="43c", src="rep43_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTB",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTB", rej=True, block_seed=True,
                    sub="43c — F + repulsion (turn + brake), seed 0"),
               dict(key="43d", src="rep43_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTB",
                    arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTB", rej=True, block_seed=True,
                    sub="43d — F + repulsion (turn + brake), seed 101")]),
    dict(k=44, nb=False, md="""## Model 44 — the ablation study on 26e: kinematic history longer than 3

26e with the step and interval lags 3 -> 4 / 5 / 6 (the shorter histories 1 / 0 are 33k-33n).
Shown only in Final_report_ablations.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="44a", src="abl26e3_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk4", rej=True, block_seed=True, sub="44a — kinematic history 4 bouts, seed 0"),
               dict(key="44b", src="abl26e3_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk4", rej=True, block_seed=True, sub="44b — kinematic history 4 bouts, seed 101"),
               dict(key="44c", src="abl26e3_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk5", rej=True, block_seed=True, sub="44c — kinematic history 5 bouts, seed 0"),
               dict(key="44d", src="abl26e3_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk5", rej=True, block_seed=True, sub="44d — kinematic history 5 bouts, seed 101"),
               dict(key="44e", src="abl26e3_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk6", rej=True, block_seed=True, sub="44e — kinematic history 6 bouts, seed 0"),
               dict(key="44f", src="abl26e3_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk6", rej=True, block_seed=True, sub="44f — kinematic history 6 bouts, seed 101")]),
    dict(k=45, nb=False, md="""## Model 45 — the ablation study on 26e: the latent removed (retrained without z)

26e retrained without the per-fish random effect (the plain joint model), everything else identical. The first round froze
the trained 26e's latent at 0 at simulation time (33q / 33r).
Shown only in Final_report_ablations.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="45a", src="abl26e4_ladder_checkpoints", arm="S4x_adt_cdt_tr_rc_al_w3ft", rej=True, block_seed=True, sub="45a — latent removed (retrained without z), seed 0"),
               dict(key="45b", src="abl26e4_seed101_ckpts", arm="S4x_adt_cdt_tr_rc_al_w3ft", rej=True, block_seed=True, sub="45b — latent removed (retrained without z), seed 101")]),
    dict(k=49, nb=False, md="""## Model 49 — the ablation study on 26e: kinematic history 2, turn history 3

26e with the step and interval lags 3 -> 2, and (separately) the turn lags 8 -> 3, everything else identical.
Shown only in Final_report_ablations.ipynb (nb=False keeps it out of Report_models.ipynb).""",
         arms=[
               dict(key="49a", src="abl26e5_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk2", rej=True, block_seed=True, sub="49a — kinematic history 2 bouts, seed 0"),
               dict(key="49b", src="abl26e5_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_kk2", rej=True, block_seed=True, sub="49b — kinematic history 2 bouts, seed 101"),
               dict(key="49c", src="abl26e5_ladder_checkpoints", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_h3", rej=True, block_seed=True, sub="49c — turn history 3 bouts, seed 0"),
               dict(key="49d", src="abl26e5_seed101_ckpts", arm="S4x_adt_z2_cdt_tr_rc_al_w3ft_h3", rej=True, block_seed=True, sub="49d — turn history 3 bouts, seed 101")]),
    dict(k=50, nb=False, md="""## Model 50 — F + turn + a likelihood-fitted step term (the backing-off diagnosis)

F + turn (Model 43a / 43b) plus a step term fitted by the teacher-forced likelihood of the Larsch screen bouts (fit half):
shorter steps near any dot (scripts/repulsion_lik.py). Shown only in Final_report_extra_data.ipynb.""",
         arms=[
               dict(key="50a", src="rep50_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTS", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTS", rej=True, block_seed=True, sub="50a — F + turn + step, seed 0"),
               dict(key="50b", src="rep50_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTS", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_repTS", rej=True, block_seed=True, sub="50b — F + turn + step, seed 101")]),
    dict(k=51, nb=False, md="""## Model 51 — F + the GROUP repulsion term (one representation for attraction and repulsion)

Each visible moving group's term comes from its nearest member, in the single-dot form calibrated on the Larsch screen
(scripts/grouprep_fit.py); groups add by the effects rule. Shown only in Final_report_shoaling.ipynb. The teacher-forced
cells (likelihood, tow, hgap) are computed WITHOUT the term (the torch path does not see it): read the likelihood from
Model/grouprep/lik.csv instead.""",
         arms=[
               dict(key="51a", src="gr_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT", rej=True, block_seed=True, sub="51a — F + group repulsion (calibrated turn), seed 0"),
               dict(key="51b", src="gr_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT", rej=True, block_seed=True, sub="51b — F + group repulsion (calibrated turn), seed 101"),
               dict(key="51c", src="gr_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grTS", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grTS", rej=True, block_seed=True, sub="51c — F + group repulsion (calibrated turn + step), seed 0"),
               dict(key="51d", src="gr_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grTS", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grTS", rej=True, block_seed=True, sub="51d — F + group repulsion (calibrated turn + step), seed 101"),
               dict(key="51e", src="gr_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT2", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT2", rej=True, block_seed=True, sub="51e — F + group repulsion (turn x2), seed 0"),
               dict(key="51f", src="gr_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT2", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT2", rej=True, block_seed=True, sub="51f — F + group repulsion (turn x2), seed 101"),
               dict(key="51g", src="gr_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT4", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT4", rej=True, block_seed=True, sub="51g — F + group repulsion (turn x4), seed 0"),
               dict(key="51h", src="gr_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT4", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT4", rej=True, block_seed=True, sub="51h — F + group repulsion (turn x4), seed 101"),
               dict(key="51i", src="gr_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT8", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT8", rej=True, block_seed=True, sub="51i — F + group repulsion (turn x8), seed 0"),
               dict(key="51j", src="gr_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT8", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grT8", rej=True, block_seed=True, sub="51j — F + group repulsion (turn x8), seed 101"),
               dict(key="51k", src="gr_s0/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grTb1", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grTb1", rej=True, block_seed=True, sub="51k — F + group repulsion (turn x group count), seed 0"),
               dict(key="51l", src="gr_s101/S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grTb1", arm="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_pf5Bcr_last_grTb1", rej=True, block_seed=True, sub="51l — F + group repulsion (turn x group count), seed 101")]),
]


def _units_of(m):
    """The evaluation units of one MODELS block: its arms, or the block itself when it has one arm."""
    if "arms" in m:
        return [dict(a, block=m["k"]) for a in m["arms"]]
    if "cells" in m:                         # a block with its OWN cells (Model 31): no checkpoint copy, no summary row
        return [dict(key=str(m["k"]), src=None, arm="", rej=False, sub=None, block=m["k"], cells=m["cells"])]
    return [dict(key=str(m["k"]), src=m["src"], arm=m["arm"], rej=m["rej"], sub=None, block=m["k"])]


def units():
    return [u for m in MODELS for u in _units_of(m)]


def _unit(key):
    return next(u for u in units() if u["key"] == str(key))

CELLS = [
    ("lik", """# Model {k}: held-out likelihood per head, bits per bout (lower = better), mean over the 15 held-out fish.
_l = pd.read_csv("{d}/likA_model1.csv").drop_duplicates(["arm", "fish"])
lik_{k} = _l[_l.arm == "{arm}"][["bits_per_bout", "bits_per_bout_dist", "bits_per_bout_dt", "bits_per_bout_joint"]].mean()
lik_{k}.index = ["turn", "displacement", "interval", "joint"]
print(lik_{k}.round(3).to_string())"""),
    ("auc", """# Model {k}: distinguishability -- the window classifier's AUC, standard and steady-state readings, and the spread of
# simulated individuals relative to real fish (bf = between fish, wf = within fish); 8 rollouts per held-out fish.
panel_{k}, c2st_{k} = run(ckpt_dir="{d}", arms="{arm}", R=20, jobs=4, progress=False{runrej})"""),
    ("auc2", """# Model {k}: the same classifier under two fairer windowings of the SAME rollouts (8 per held-out fish, the seeds of
# the AUC above). "1 per real": the plain tiling cut down to one generated window per real one, which removes the extra
# classifier power of the ~1.6x more generated windows. "matched": the rollout cut at the real fish's own window
# positions (identical bouts, conditions and phase, 1:1), which also removes the difference in composition. All 39
# features, and without the four radial-position features.
from c2st_protocol import matched_c2st_table
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    _t = matched_c2st_table("{d}", "{arm}", rej={rej}, jobs=4)
_t = _t[(_t.windows == "all") & _t.generated.isin(["plain tiling, 1 generated window per real one",
                                                    "matched to the real runs (same bouts, 1:1)"])]
auc2_{k} = _t.set_index("generated")[["all features", "all features se", "no radial", "no radial se"]]
print(auc2_{k}.round(3).to_string())"""),
    ("indv", """# Model {k}: individuality -- every window statistic (mean, median, SD, q90, lag-1 autocorrelation) of radial position,
# interval, |turn| and displacement over the classifier's 60-bout windows: the distribution over all windows, real (blue)
# against simulated (orange), and underneath one tick per real fish / simulated individual at its mean. 4 rollouts per
# held-out fish, each a new individual; in each panel the between- and within-fish SD, simulated / real.
from model_figures import window_stats_grid
real_w, gen_w, ws_{k} = window_stats_grid("{d}", "{arm}", rej={rej}, reps=4)"""),
    ("prof", """# Model {k}: distance to the dots across the block, real fish next to programme rollouts (each simulated fish swims
# the whole programme on its own clock, the dots re-anchored to it at every block onset), 5 fold models x 3 fish.
st, gen_p, prof = dot_profile("{d}", "{arm}", rej={rej})"""),
    ("matched", """# Model {k}: the same profile with 50 simulated fish, matched to the real block composition -- the numbers to read
# (dip = approach within 20 s and its latency; plateau 20-60 s; grading = 1-dot minus 8-dot plateau; errors in real SEM).
varm_{k}, _ = profile_variants_matched("{d}", "{arm}", {{"Model {k}": {{"arm": "{arm}"}}}}, rej={rej}, r_per_model=10,
                                       show_raw=False)"""),
    ("attr", """# Model {k}: attraction = stimulus-blind null minus the 20-60 s plateau, per dot count (mm). Real fish: a step
# (~1 mm for one dot, ~6 mm from two up).
attr_{k}, _ = attraction_table("{d}", ["{arm}"], rej={rej}, r_per_model=6)"""),
    ("tow", """# Model {k}: turn toward the moving dots by dot count (deg), predicted bout by bout from the real history (held-out
# fish, population latent), over the first 20 s of a block and over 4-20 s.
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    tow_{k} = toward_by_n("{d}", ["{arm}"], rej={rej})
print(tow_{k}.round(2).to_string())"""),
    ("hgap", """# Model {k}: each head against real fish, per stimulus condition and block phase (teacher-forced on the held-out
# fish's real bouts; gap = model - real on the same bouts, 95 % CI across fish): where the approach is lost.
from model_figures import head_gaps
hg_rows_{k}, hg_{k} = head_gaps({{"Model {k}": ("{d}", "{arm}")}}, rej={rej})"""),
]

SUMMARY_MD = """## Summary

Every model above in one table (Models 6 and 7 as two columns each: a = without the range bumps, b = with them; Models 8
to 10 as four each, Model 11 as five), from the key numbers each block saved (`report_outputs/summary_M<k>.json`); the
head-analysis gaps (`gap_*`, model − real, deg) where a block has that cell: held-out
likelihood (bits per bout); the AUC -- standard and steady-state (the harness), then one generated window per real one
(`auc_1per`) and matched to the real runs (`auc_matched`), each also without the radial features (`_norad`); the spread
of simulated individuals (gen / real; bf = between fish, wf = within); the matched profile (dip, its latency, plateau,
1-dot plateau, grading, moving and static error in real SEM); attraction per dot count. Real fish: dip 4.23 mm at
13.5 s, plateau 17.26 mm, 1-dot plateau 22.26 mm, grading 6.00 mm, attraction 1.16 / 6.19 / 5.19 / 6.43 / 6.94 mm."""

SUMMARY = """# Every model in one table.
import json, glob
S = pd.DataFrame([json.load(open(p)) for p in sorted(glob.glob("report_outputs/summary_M*.json"))]).set_index("model")
_first = [c for c in S.columns if c.startswith(("lik_", "auc_"))]      # likelihood and every AUC reading together
S = S[_first + [c for c in S.columns if c not in _first]]
print(S.T.to_string())"""


def _fmt(src, u):
    d = f"report_ckpts/M{u['key']}"
    return src.format(k=u["key"], d=d, arm=u["arm"], rej='REJ["wall_reject"]' if u["rej"] else "None",
                      runrej=', cfg_override=REJ, tag="rej"' if u["rej"] else "")


def _run_cell(ns, src):
    import matplotlib.pyplot as plt
    buf, outs, t0 = io.StringIO(), [], time.time()
    try:
        with redirect_stdout(buf):
            exec(compile(src, "<cell>", "exec"), ns)
    except Exception:
        buf.write("\n" + traceback.format_exc())
    if buf.getvalue():
        outs.append(dict(output_type="stream", name="stdout", text=buf.getvalue().splitlines(True)))
    for num in plt.get_fignums():
        png = io.BytesIO()
        plt.figure(num).savefig(png, format="png", dpi=100, bbox_inches="tight")
        outs.append(dict(output_type="display_data", metadata={},
                         data={"image/png": base64.b64encode(png.getvalue()).decode(), "text/plain": ["<Figure>"]}))
    plt.close("all")
    return outs, time.time() - t0


def _summary(ns, u):
    """The key numbers of unit k, from the variables its cells left behind (None where a cell failed)."""
    import numpy as np
    k, arm = u["key"], u["arm"]
    s = {"model": f"Model {k}"}
    try:
        s.update({f"lik_{c}": round(float(v), 4) for c, v in ns[f"lik_{k}"].items()})
    except Exception:
        pass
    try:
        c = ns[f"c2st_{k}"]; row = c[c.arm.isin([arm, arm + "_rej"])].iloc[0]
        s.update(auc_std=round(float(row.c2st_traj), 4), auc_steady=round(float(row.c2st_traj_int), 4))
        for ch in ("turn", "dist", "dt", "rad"):
            s[f"bf_{ch}"] = round(float(row[f"bf_sd_gen_{ch}"] / row[f"bf_sd_real_{ch}"]), 3)
            s[f"wf_{ch}"] = round(float(row[f"wf_sd_gen_{ch}"] / row[f"wf_sd_real_{ch}"]), 3)
    except Exception:
        pass
    try:
        t = ns[f"auc2_{k}"]
        for row, tag in (("plain tiling, 1 generated window per real one", "1per"),
                         ("matched to the real runs (same bouts, 1:1)", "matched")):
            s[f"auc_{tag}"] = round(float(t.loc[row, "all features"]), 4)
            s[f"auc_{tag}_norad"] = round(float(t.loc[row, "no radial"]), 4)
    except Exception:
        pass
    try:
        v = ns[f"varm_{k}"]; r = v.loc[[i for i in v.index if "[matched]" in str(i)][0]]
        s.update({c: round(float(r[c]), 3) for c in ("dip", "lat", "plateau", "d1_plateau", "grading", "mv_err_sem",
                                                     "st_err_sem", "static_drift")})
    except Exception:
        pass
    try:
        a = ns[f"attr_{k}"]
        col = [c for c in a.columns if c[0] != "real" and c[1] == "attraction"][0]
        for cond, n in (("d1s3.5", 1), ("d2s3.5", 2), ("d4s3.5", 4), ("d6s3.5", 6), ("d8s3.5", 8)):
            s[f"attr_N{n}"] = round(float(a.loc[cond, col]), 2)
    except Exception:
        pass
    try:                                     # the head analysis' three key gaps (model - real, deg)
        h = ns[f"hg_{k}"]
        for key, cond, ph, q in (("gap_ahead_N8", "d8s3.5", "20-60 s", "ahead"),
                                 ("gap_toward_N2", "d2s3.5", "20-60 s", "toward"),
                                 ("gap_onset_N8", "d8s3.5", "0-4 s", "toward")):
            r = h[(h.cond == cond) & (h.phase == ph) & (h.qty == q)]
            s[key] = round(float(r.gap.iloc[0]), 2) if len(r) else None
    except Exception:
        pass
    return {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv) for kk, vv in s.items()}


def prepare(u):
    """report_ckpts/M<key> = a copy of the arm's five fold checkpoints, manifest and held-out likelihood table."""
    import shutil
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "scripts"))
    from rollout_model1_local import merge_ckpt_dirs
    d = CKPTS / f"M{u['key']}"
    if d.exists():
        shutil.rmtree(d)
    merge_ckpt_dirs(d, HERE / u["src"], arms=[u["arm"]])
    n = len(list(d.glob(f"{u['arm']}_fold[0-9].pt")))
    if n != 5:
        raise SystemExit(f"Model {u['key']}: expected 5 fold checkpoints of {u['arm']} in {u['src']}, found {n}")


def evaluate(k, only=None):
    """Run unit k's cells and save them. ``only`` = cell ids to (re)compute into an existing saved block, keeping
    the others as they are (e.g. a changed plot); the summary numbers are kept unless they come from those cells."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    os.chdir(HERE)
    k = str(k)
    m = _unit(k)
    OUTS.mkdir(exist_ok=True)
    old = json.loads((OUTS / f"M{k}.json").read_text(encoding="utf-8")) if only else None
    custom = m.get("cells")
    if not only and not custom:
        prepare(m)
    ns = {"__name__": "__cell__"}
    exec(compile(SETUP, "<setup>", "exec"), ns)
    cells = []
    for cid, tpl in (custom or CELLS):
        if only and cid not in only:
            prev = next((c for c in old["cells"] if c["cid"] == cid), None)
            if prev is not None:
                cells.append(prev)
            continue
        src = tpl if custom else _fmt(tpl, m)
        if m.get("block_seed") and cid in ("prof", "matched", "attr"):   # programme rollouts: per-block streams
            src = ("from model_figures import block_seeding\n"
                   "with block_seeding(True):   # per-block random streams: a fair comparison of the static curves\n"
                   + textwrap.indent(src, "    "))
        outs, secs = _run_cell(ns, src)
        cells.append(dict(cid=cid, source=src, outputs=outs))
        print(f"  Model {k} {cid}: {secs:.0f} s", flush=True)
    (OUTS / f"M{k}.json").write_text(json.dumps(dict(k=k, arm=m["arm"], cells=cells)), encoding="utf-8")
    if custom:
        return
    s, p = _summary(ns, m), OUTS / f"summary_M{k}.json"
    if only and p.exists():                  # keep the numbers of the cells that were not re-run
        s = {**json.loads(p.read_text(encoding="utf-8")), **{kk: vv for kk, vv in s.items() if kk != "model"}}
    p.write_text(json.dumps(s, indent=1), encoding="utf-8")


def assemble():
    def md(text, cid):
        return dict(cell_type="markdown", id=cid, metadata={}, source=text.splitlines(True))

    def code(src, cid, outs=()):
        return dict(cell_type="code", id=cid, execution_count=None, metadata={}, source=src.splitlines(True),
                    outputs=list(outs))
    # Cells added BY HAND (any id the builder does not generate) are kept, each re-inserted after the generated cell
    # it followed, so a rebuild never deletes a note or an analysis written into the notebook.
    gen_id = re.compile(r"^(intro|setup|summary(-md)?|m\d+[a-z]?-.+)$")
    kept = []
    if NB.exists():
        anchor = None
        for c in json.loads(NB.read_text(encoding="utf-8")).get("cells", []):
            if gen_id.match(str(c.get("id", ""))):
                anchor = c.get("id")
            else:
                kept.append((anchor, c))
    cells = [md(INTRO, "intro"), code(SETUP, "setup")]
    for m in MODELS:
        if m.get("nb") is False:                     # evaluated here, shown in another notebook (e.g. the ablations)
            continue
        cells.append(md(m["md"], f"m{m['k']}-md"))
        for u in _units_of(m):
            if u["sub"]:
                cells.append(md(f"### {u['sub']}", f"m{u['key']}-sub"))
            p = OUTS / f"M{u['key']}.json"
            if not p.exists():
                cells.append(md("*(not evaluated yet)*", f"m{u['key']}-pending"))
                continue
            blob = json.loads(p.read_text(encoding="utf-8"))
            cells += [code(c["source"], f"m{u['key']}-{c['cid']}", c["outputs"]) for c in blob["cells"]]
    import os
    os.chdir(HERE)
    ns = {"__name__": "__cell__"}
    exec(compile(SETUP, "<setup>", "exec"), ns)
    outs, _ = _run_cell(ns, SUMMARY)
    cells += [md(SUMMARY_MD, "summary-md"), code(SUMMARY, "summary", outs)]
    placed = set()
    for anchor, c in kept:
        i = next((k for k, x in enumerate(cells) if x.get("id") == anchor), None)
        i = len(cells) - 3 if i is None else i                       # unknown anchor: just before the summary
        j = i + 1
        while j < len(cells) and cells[j].get("id") in placed:       # after the kept cells already placed there
            j += 1
        cells.insert(j, c)
        placed.add(c.get("id"))
    if kept:
        print(f"kept {len(kept)} hand-added cell(s)", flush=True)
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                         language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {NB.name}: {len(cells)} cells", flush=True)


def evaluate_all(only=None, keys=None):
    """Evaluate the units ``keys`` (default: every unit) in parallel, one process each, then assemble."""
    t0 = time.time()
    keys = [str(k) for k in keys] if keys else [u["key"] for u in units()]
    extra = ["--only", ",".join(only)] if only else []
    procs = [(k, subprocess.Popen([sys.executable, "-u", "-W", "ignore", str(Path(__file__).resolve()), "evaluate",
                                   k] + extra, cwd=str(HERE))) for k in keys]
    bad = []
    for k, p in procs:
        rc = p.wait()
        print(f"Model {k}: exit {rc}  [{(time.time() - t0) / 60:.0f} min]", flush=True)
        if rc:
            bad.append(k)
    assemble()
    if bad:
        raise SystemExit(f"evaluation failed for models {bad} (their blocks show the error)")


if __name__ == "__main__":      # REQUIRED on Windows: run()'s process pool re-imports this module
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    keys = sys.argv[sys.argv.index("--keys") + 1].split(",") if "--keys" in sys.argv else None
    if cmd == "evaluate":
        evaluate(sys.argv[2], only=only)
    elif cmd == "assemble":
        assemble()
    elif cmd == "run":                           # python report_models.py run 6a,6b: those units, then assemble
        evaluate_all(keys=sys.argv[2].split(","))
    elif cmd == "refresh":                       # python report_models.py refresh indv[,prof] [--keys 6a,6b]
        evaluate_all(only=sys.argv[2].split(","), keys=keys)
    else:
        evaluate_all()
