# Working in this repository

A guide for AI coding agents (and anyone new to the code). It is the code of an MSc thesis: generative models of
larval zebrafish swimming, bout by bout, around groups of moving or static dots, trained on 15 fish, simulated in closed
loop and evaluated on held-out fish. `README.md` says what the repository is for; this file says how it is built and
the rules to keep.

## Setup

```bash
uv sync                                  # Python >= 3.13, every dependency from uv.lock
# unzip twin_data.zip and twin_output.zip into the repository root (data/ and output/)
uv run python pipelines/check.py         # 15 s: data, a checkpoint and the code give back stored numbers
```

Run everything with `uv run` from the repository root. Paths come from `twin/paths.py` (override with `TWIN_DATA`,
`TWIN_OUTPUT`, `TWIN_FIGURES`); never hard-code a path. There is no test suite: `pipelines/check.py` is the smoke test
and the reproducibility checks below are the regression tests.

## Map

| Where | What |
|---|---|
| `twin/paths.py` | every location: data, checkpoints, evaluations, caches, figures |
| `twin/registry.py` | every thesis model: `RECIPES` (name -> ingredients) and `UNITS` (key -> name, seed, how it is made) |
| `twin/frames.py`, `segment_bouts.py`, `cleaning.py` | rig files -> 30 fps tables -> bout table -> training table |
| `twin/stimulus.py`, `features.py`, `emission.py`, `joint.py`, `latent.py` | the model: stimulus codes, input row, the three chained heads (turn, step, interval), the per-fish latent and the slow drift |
| `twin/training.py`, `correction.py`, `profile_finetune.py`, `rollout_batch.py` | fitting a unit (5 folds), the units made from others (F = D4 + profile correction) |
| `twin/stim_program.py`, `stim_rollout.py` | a stimulus programme (the rig's file, or one built in code) and closed-loop rollouts on it |
| `twin/unit_eval.py`, `rollout_eval.py`, `generative_metrics.py`, `evaluate.py`, `c2st_protocol.py`, `model_figures.py`, `tables.py` | evaluation on held-out fish: classifier AUC, individuality, distance profiles, slow structure; the thesis tables |
| `twin/missing_pieces.py` | turn rule, references, nearest-real-neighbour (cached in `output/missing_pieces/`) |
| `twin/two_targets.py` | two dot groups at a separation L: rules, the bifurcation maps' runs, single-fish videos |
| `twin/multi_fish.py`, `shoal_videos.py` | several simulated fish in one dish (shoals) and their videos |
| `twin/twin_test.py`, `screen_*.py`, `grouprep_fit.py` | the second data set (Larsch lab screen) and the group repulsion term |
| `twin/videos.py`, `rollout_video.py` | the one video renderer, and programme-rollout videos |
| `figures/` | one script per thesis figure (`python -m figures.<name>`), `data_figures.py` (Data chapter), `data_explore.py` (real-data exploration) |
| `pipelines/` | `train.py`, `evaluate.py`, `reproduce.py`, `check.py` |
| `notebooks/01-09` | one per thesis part; read the saved results and redraw every table and figure |
| `experiments/` | runnable explorations: two dot groups, shoaling, the rollout tool (any model on any programme) |
| `assets/` | the rig's stimulus programme, the dish geometry, a histogram summary of the real fish (no animal data) |

## Vocabulary

- **bout**: one swim event; the data and the models are sequences of bouts (turn `dtheta`, step `distance`, interval
  `dt`). Models are bout-indexed: the stimulus is read once per bout onset, the fish is blind in between.
- **block**: 60 s of dots (1, 2, 4, 6 or 8; moving or static) after a 50 s pause; 95 blocks, 175 min, the same exact
  frame grid for every fish. **CLsemi**: each block is anchored to the fish's pose at its onset, then plays open loop.
- **run**: bouts between pauses; a pause (quiescence) is a gap of more than 10 s (`QUIET_S`). Active bouts = not
  adjacent to a pause. The run clock counts bouts since the run started.
- **unit**: one model at one cross-validation seed = five fold models, each trained with three fish held out.
  **key**: the unit's id (`"26e"` = M6 seed 0, `"14f"` = F seed 0); checkpoints in `output/checkpoints/M<key>/`.
  **arm**: the checkpoint file stem, derived from the recipe. **seed** 0 or 101 = two different fold splits.
- **models**: M1-M6 the main line (M6 = analysis model), D1-D4 the slow-drift branch, F = D4 + the profile
  correction (final model); ablations `M6-<part>`; variants in the appendix ("M6, 3 layers", ...).
- **rollout**: a free-running simulation of a model on a programme; **matched** = the real fish's windowing.
- **rules** (two or more dot groups): `effects` (each group's effect summed), `add`, `avg`, `nearest`.
- **profile**: distance to the dot centroid over the block; **attraction**: blind null minus the 20-60 s plateau.
- **screen**: the second data set (Larsch lab, 2819 fish, one dot); unpublished, see its `DATA_USE.md`.
- Scale: 3.60 px/mm (physical dish). The profile code divides by the nominal 3.67 and the figures convert
  (`figures/style.MM_3P67_TO_3P60`); keep the two apart.

## How to

- **Train / evaluate a thesis model**: `pipelines/train.py M6 --seed 0` then `pipelines/evaluate.py M6 --seed 0`;
  `unit_eval.show("26e")` prints every measurement. Retraining in place replaces the thesis checkpoints: use `--out`.
- **A new variant**: add a recipe in `registry.RECIPES` (see `recipe()`'s ingredients) and a unit line with unused
  keys for both seeds; train both seeds, evaluate both, compare with `tables.profile_table([...])`.
- **Simulate a model on a programme**: `SR.rollout_program(model, program, rng, pose, dish)` with a model from
  `model_figures.fold_models(registry.ckpt_dir(key), registry.arm(key))`; a programme from
  `stim_program.load_program(path)` or `build_program(blocks, path)` (dots relative to the fish at block onset, mm,
  x = left, y = ahead). `experiments/rollout_tool.ipynb` does all of it.
- **A video**: build panels (fish positions in mm per frame via `videos.frames`, view, dish, optional dots, lines and
  a trace) and call `videos.write(panels, taus, title, footer, out_mp4, speed)`. Every video in the repository goes
  through `twin/videos.py`; do not write another renderer.
- **Two groups / shoals**: `two_targets.simulate`, `video`, `rules_video`; `multi_fish.run_group(spec, g)`,
  `shoal_videos.video(name)`. The full runs are cached per condition and are not in the output zip.
- **A thesis figure**: `uv run python -m figures.<name>` writes `figures/out/<name>.pdf` (+ PNG).
- **Everything**: `pipelines/reproduce.py --level figures|evaluate|train --dry-run` lists the steps and their cost.

## Rules

1. **Seeds are never pooled.** Compare two models on the same seed (the same folds), then check that seed 101 gives
   the same sign; a difference that flips between seeds is noise.
2. **Rollouts are the evaluation.** The held-out likelihood guides training only; results are measured on simulated
   fish (classifier AUC on matched windows, distance profiles, turn rule, slow structure), never on the likelihood.
3. **Determinism is a feature.** Training is deterministic given the seed and evaluation reproduces the saved
   summaries exactly; after a change to shared code, re-evaluate a unit (or run `pipelines/check.py`) and compare.
   A change that moves a thesis number must be intended and said.
4. **Do not change a thesis recipe or unit.** Add a new one. Options the thesis models never used were removed; a
   configuration asking for one is refused, keep it that way rather than reviving dead paths.
5. **Caches are guarded, not trusted.** Results computed from checkpoints check `registry.stale` and are redone or
   refused when the checkpoints are newer; keep that guard on anything new that caches rollouts.
6. **Nothing computed goes into git**: `data/`, `output/`, `figures/out/`, checkpoints, videos and zips are ignored.
   The screen data and its caches (`schizscreen_twin_v1*`, `output/screen/`) must never be committed; its figures and
   notebook outputs may be.
7. **Windows**: scripts that start process pools need the `if __name__ == "__main__":` guard (workers re-import);
   video rendering runs in a subprocess with the ffmpeg of `imageio-ffmpeg`.
8. Match the surrounding code: dense docstrings that say what a function measures and why, mm and seconds in names,
   no new dependencies without need.
