"""Build Model/Final_report_ablations.ipynb: the ablation study on the analysis model 26e (no drift, no correction), from
the saved results of report_models.py. Nothing is re-run; a planned ablation that has not been run yet shows as XX. When
an ablation is evaluated, put its notebook key in place of the None below and re-run:

    python final_report_ablations.py
"""
from pathlib import Path

import final_report_common as C
from final_report_common import S0, S101, md, code

NB = C.HERE / "Final_report_ablations.ipynb"

# units = [(notebook key or None if not run yet, seed)]
STORY = [
    dict(id="ref", title="0. The ablation model", md="""Every ablation below is 26e retrained without one ingredient (same recipe,
same folds and seeds; seed 0 pairs with 26e, seed 101 with 30d), except the latent ablation, which freezes the latent of
the trained 26e at simulation time. 26e has no drift and no profile correction, so every component is interpretable and
it retrains stably; the full blocks of the reference are repeated here so the profiles can be compared side by side.""",
         models=[dict(name="26e — the ablation model", anchor="m26e", change="(reference)",
                      units=[("26e", S0), ("30d", S101)],
                      md="S4x code + per-fish latent + ×3 turn weight + centred lags with the 10 s truncation, run clock, "
                         "aligned-and-far flag, block clock (the 'dots just appeared' feature).")]),
    dict(id="stim", title="1. Stimulus information", md="""What the response to the dots needs: none of the stimulus, only
whether the dots move, everything but the group size, everything but the range. Attraction has a ~−9 mm zero point for a
stimulus-blind model, so a negative attraction is not repulsion.""", models=[
        dict(name="No stimulus information", anchor="abl-none",
             change="no stimulus columns, no aligned-and-far flag, no block clock", units=[("29e", S0), ("33u", S101)],
             md="+0.12 AUC; no approach at all (dip 0)."),
        dict(name="Moving vs static only", anchor="abl-mvonly",
             change="only the presence and moving flags: no bearing, no range, no count (and no aligned-and-far flag)",
             units=[("33a", S0), ("33b", S101)],
             md="The model knows that dots are shown and whether they move, not where they are or how many."),
        dict(name="Group size removed", anchor="abl-nocount",
             change="without log count, count × sin, count × range × sin", units=[("29c", S0), ("33s", S101)],
             md="AUC unchanged; grading halves (4.67 → 2.53 mm) and the count step is lost (seed 0)."),
        dict(name="Range removed", anchor="abl-norange",
             change="without log range, the range × bearing products and the aligned-and-far flag (which reads the range)",
             units=[("33c", S0), ("33d", S101)], md="The model knows where the dots are (bearing), how many and whether they "
             "move, not how far."),
        dict(name="Moving indicator removed", anchor="abl-nomoving",
             change="without the moving flag, moving × sin / cos, the moving and static range products and the "
                    "aligned-and-far flag", units=[("29d", S0), ("33t", S101)],
             md="The moving and static profiles collapse (moving error 0.90 → 2.66, static 0.80 → 4.99; seed 0)."),
    ]),
    dict(id="hist", title="2. History: how much memory each decision uses", md="""The history inputs (the last turns, the last
displacements and intervals) feed one shared hidden layer that all three heads read, so the history length is shortened
for the whole bout (turn, step and interval together), separately for the turn history and for the kinematic history.""",
         models=[
        dict(name="Turn history 4 bouts", anchor="abl-k4", change="the last 4 turns instead of 8 (kinematic history kept)",
             units=[("33e", S0), ("33f", S101)], md=""),
        dict(name="Turn history 3 bouts", anchor="abl-k3", change="the last 3 turns, instead of 8",
             units=[("49c", S0), ("49d", S101)], md=""),
        dict(name="Turn history 1 bout", anchor="abl-k1", change="the last turn only", units=[("33g", S0), ("33h", S101)],
             md=""),
        dict(name="No turn history", anchor="abl-k0", change="no previous turns", units=[("33i", S0), ("33j", S101)],
             md=""),
        dict(name="Kinematic history 2 bouts", anchor="abl-kk2",
             change="the last 2 displacements and intervals, instead of 3 (turn history kept)",
             units=[("49a", S0), ("49b", S101)], md=""),
        dict(name="Kinematic history 1 bout", anchor="abl-kk1",
             change="the last displacement and interval only, instead of 3 (turn history kept)",
             units=[("33k", S0), ("33l", S101)], md=""),
        dict(name="No kinematic history", anchor="abl-kk0", change="no previous displacements or intervals",
             units=[("33m", S0), ("33n", S101)], md=""),
        dict(name="Kinematic history 4 bouts", anchor="abl-kk4",
             change="the last 4 displacements and intervals, instead of 3 (turn history kept)",
             units=[("44a", S0), ("44b", S101)], md=""),
        dict(name="Kinematic history 5 bouts", anchor="abl-kk5", change="the last 5 displacements and intervals",
             units=[("44c", S0), ("44d", S101)], md=""),
        dict(name="Kinematic history 6 bouts", anchor="abl-kk6", change="the last 6 displacements and intervals",
             units=[("44e", S0), ("44f", S101)], md=""),
    ]),
    dict(id="latent", title="3. Individuality: the per-fish latent removed", md="""How much of the between-fish difference —
in tempo, turning, space and in the response to the dots — the per-fish latent is needed for. **Main test: 26e retrained
without the latent** (the plain joint model, everything else identical): the fair ablation, since the rest of the network
can then reorganise without it. For comparison, the trained 26e simulated with every fish's latent frozen at the
population average (z = 0), which only shows what the trained latent carries. (The slow within-fish AR(1) state is not
part of the ablation model.)""",
         models=[dict(name="Latent removed (retrained without z)", anchor="abl-noz",
                      change="26e retrained without the per-fish random effect (plain joint model)",
                      units=[("45a", S0), ("45b", S101)],
                      md="A held-out fish is scored directly by the population model (there is no latent to integrate "
                         "out), so the likelihood row is a fair comparison with 26e."),
                 dict(name="Latent frozen (z = 0), for comparison", anchor="abl-z0",
                      change="the trained 26e, latent frozen at simulation only, no retraining",
                      units=[("33q", S0), ("33r", S101)],
                      md="The held-out likelihood is unchanged by construction (the latent is frozen only in "
                         "simulation), so its row repeats 26e's; read the AUC, the individuality spread and the "
                         "profiles.")]),
    dict(id="onset", title="4. Onset transient: the 'dots just appeared' feature removed", md="""26e carries a block clock,
exp(−t / 4 s) with t the time since the first bout of the current stimulus condition: 1 on the first bout after the dots
appear, 0.37 after 4 s, ~0 after 15 s, 0 in pauses. Without it the model only knows where the dots are now, not that they
are new. If the approach dip survives, the range and bearing of the dots produce it on their own; if it disappears, the
approach is made in the first bouts after onset, i.e. the fish react more strongly when the dots are new.""",
         models=[dict(name="No block clock", anchor="abl-noclock", change="without the block clock (run clock kept)",
                      units=[("33o", S0), ("33p", S101)], md="")]),
    dict(id="temporal", title="5. The temporal set and the ×3 turn weight", md="""The components of 26e that the first round
removed only from other models (D1, D3, D4; Report_models Models 25-26), here removed from 26e itself: the run clock
(exp(−k / 30), k the position in the current run: the first bouts after a quiescent gap), the centred interval lags with
the 10 s truncation (removed together, as in Model 26), the aligned-and-far flag on its own (the stimulus ablations above
drop it only together with what it reads) and the ×3 weight on the turn of far moving-dot bouts in the training loss.
The likelihood is the unweighted one in every row, so the turn-weight ablation is scored on the same objective as 26e.""",
         models=[
        dict(name="Run clock removed", anchor="abl-norc", change="without the run clock (block clock kept)",
             units=[("35a", S0), ("35b", S101)], md=""),
        dict(name="Centred lags and truncation removed", anchor="abl-nocdt",
             change="plain interval lags, no 10 s truncation", units=[("35c", S0), ("35d", S101)], md=""),
        dict(name="Aligned-and-far flag removed", anchor="abl-noal", change="without the aligned-and-far flag only",
             units=[("35e", S0), ("35f", S101)], md=""),
        dict(name="×3 turn weight removed", anchor="abl-now3", change="every bout weighted 1 in the training loss",
             units=[("35g", S0), ("35h", S101)], md=""),
    ]),
]

# summary rows: (label, seed-0 key, seed-101 key) -- None = not run yet
ROWS = [("26e (reference)", "26e", "30d")] + [
    (m["name"], m["units"][0][0], m["units"][1][0] if len(m["units"]) > 1 else None)
    for sec in STORY[1:] for m in sec["models"]]
STEPS = [(f"26e -> {lab}", ("26e", k0) if k0 else None, ("30d", k1) if k1 else None) for lab, k0, k1 in ROWS[1:]]

INTRO = f"""# Final report ablations

The ablation study on the analysis model **26e** (no drift, no correction): what the model needs from the stimulus, how
much history each decision uses, how much of the individuality the per-fish latent carries, and whether the response
depends on the dots being new, and which of the temporal inputs and the turn weight it needs. Each ablation has the usual measurements at both seeds, paired with 26e on the same folds.

{C.SEED_NOTE}

{C.LINKS}"""

TOC = """## Contents

1. [Summary tables](#summary) — each ablation at both seeds; its change from 26e at both seeds
2. [The ablation model](#ref) — 26e at both seeds
3. [Stimulus information](#stim) — [none](#abl-none) · [moving vs static only](#abl-mvonly) · [group size](#abl-nocount) ·
   [range](#abl-norange) · [moving indicator](#abl-nomoving)
4. [History](#hist) — turn history [4](#abl-k4) / [3](#abl-k3) / [1](#abl-k1) / [0](#abl-k0) · kinematic history [2](#abl-kk2) / [1](#abl-kk1) / [0](#abl-kk0) / [4](#abl-kk4) / [5](#abl-kk5) / [6](#abl-kk6)
5. [Individuality: latent removed](#latent) — [retrained without z](#abl-noz) · [frozen at z = 0](#abl-z0)
6. [Onset transient: no block clock](#onset)
7. [The temporal set and the ×3 turn weight](#temporal) — [run clock](#abl-norc) · [centred lags and truncation](#abl-nocdt) · [aligned-and-far flag](#abl-noal) · [×3 turn weight](#abl-now3)"""


def build():
    ns = C.new_ns()
    t1 = C.models_table_src(ROWS, "Each ablation: seed 0 | seed 101 (XX = not run yet)")
    t2 = C.steps_table_src(STEPS, "Change from 26e: seed 0 | seed 101 (negative AUC / likelihood / errors = better)")
    cells = [md(INTRO, "intro"), md(TOC, "toc"), code(C.RM.SETUP, "setup"),
             md('<a id="summary"></a>\n## Summary tables', "summary-md"),
             code(t1, "summary-models", C.run_df(ns, t1)), code(t2, "summary-steps", C.run_df(ns, t2))]
    body, n, missing = C.story_cells(STORY)
    C.write_nb(NB, cells + body, n, missing)


if __name__ == "__main__":
    build()
