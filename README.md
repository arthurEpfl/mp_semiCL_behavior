# A behavioural digital twin of larval zebrafish

Fifteen larval zebrafish, each alone in a dish, were shown groups of one to eight moving or static dots in a
semi-closed-loop assay (Perrino, 2026). Their swimming was cut into bouts, and generative models of the bout sequence
were trained on it: a trained model is a simulated fish that swims, bout by bout, around any stimulus. The models were
evaluated against held-out fish and taken beyond the data: two groups of dots, several fish in one dish, a second data
set.

This repository holds the code, the notebooks that reproduce every table and figure of the thesis, and tools to
simulate the trained models on stimuli of your own.

## Quick start

```bash
uv sync                                   # Python >= 3.13 and every dependency, from uv.lock
# unzip twin_data.zip and twin_output.zip into this folder (they create data/ and output/)
uv run python pipelines/check.py          # 15 s: the data, a trained model and the code give back stored numbers
uv run jupyter lab                        # then open notebooks/ or experiments/
```

The recordings and the trained models are not in git: they come as these two zips. Without them the code runs but
has nothing to read; `data/README.md` describes the recordings.

## What you can do

**Read the thesis results.** One notebook per part of the thesis, with its outputs; each reads the saved results and
runs in minutes.

| Notebook | Thesis | What it reproduces |
|---|---|---|
| `notebooks/01_bout_discretisation` | Data | from the rig's raw files to the 30 fps tables and the bout table (fig:detection, fig:bouttraj) |
| `notebooks/02_data_exploration` | Data | every figure and number of the Data chapter, and further real-data analyses: the pause threshold, a whole session, where fish rest, change over the session, how far back a bout remembers, the fish or the schedule |
| `notebooks/03_models` | Methods, Results | the main line M1-M6 and the drift branch D1-D4, F: tab:storyline, tab:realism, tab:app-imp-*, tab:profile-spread; fig:slow, fig:profiles, fig:app-bands-*, fig:app-latent-gain, fig:app-drift-gain, fig:app-correction-trace; every model's measurements |
| `notebooks/04_other_results` | Appendix | tab:app-negres-net, tab:app-negres-drift, tab:app-negres-other |
| `notebooks/05_ablations` | Results | tab:abl-all, tab:abl-indiv; fig:abl-profiles |
| `notebooks/06_turn_rule_and_individuals` | Results, Appendix | fig:rule, fig:range-law; tab:range-law, tab:nrn; the classifier's real-against-real floor |
| `notebooks/07_two_groups` | Results, Appendix | fig:bifurcation, fig:app-bif-* |
| `notebooks/08_second_data_set` | Results, Appendix | tab:screen-auc, tab:app-screen-kin, tab:app-screen-importances, tab:app-grouprep; fig:screen-response, fig:screen-switch, fig:app-grouprep |
| `notebooks/09_shoaling` | Results | tab:shoal-grid, tab:shoal-pairs; the shoaling videos |

Notebooks 07 and 09 need simulations the output zip leaves out (7 and 49 CPU-hours): each has a `RUN = True` switch.

**Simulate the models.** The notebooks in `experiments/` explain an experiment, run a small version of it and make
videos, all from the checkpoints:

| Notebook | What it does |
|---|---|
| `experiments/rollout_tool` | any thesis model on any stimulus programme (the rig's, a file in its format, or one built in code, e.g. 4 dots on a circle): N simulated fish, their distance to the dots per block, trajectories, a check against the real fish and a video next to real fish |
| `experiments/two_dot_groups` | two groups of dots at a separation L (Sridhar et al., 2021): how it is built, a small bifurcation map, one fish at three separations, the four combination rules side by side, M6 against F |
| `experiments/shoaling` | several simulated fish in one dish seeing each other as dots: how it is built, a small group, one video per thesis condition with what to watch for |

**Train and evaluate a model**, a thesis model or a variant of your own (see [Models](#models)).

**Reproduce everything**: `uv run python pipelines/reproduce.py --level figures | evaluate | train --dry-run` lists
what would run and how long it takes: from the saved results (the notebooks, plus the simulations the zip leaves
out), from the checkpoints (re-evaluating every model, about 15 min each), or from the data (retraining every model,
days of CPU). It is resumable and, from the checkpoints on, compares the recomputed evaluations with the saved ones.

## Layout

```
twin/            the library: data processing, the models, training, simulation, evaluation, the experiments
  paths.py       where everything below lives (override with TWIN_DATA, TWIN_OUTPUT, TWIN_FIGURES)
  registry.py    every thesis model: its name, recipe, seeds and checkpoint folder
  videos.py      the one renderer behind every video
figures/         one script per thesis figure (uv run python -m figures.<name>), writing figures/out/
pipelines/       train.py, evaluate.py, reproduce.py, check.py
notebooks/       01-09, one per part of the thesis, with outputs
experiments/     the three simulation notebooks above
assets/          the rig's stimulus programme, the dish geometry, a histogram summary of the real fish
data/            from twin_data.zip (not in git)
  raw/             the rig's files: tracking, stimulus programme, dish geometry
  frames/          one 30 fps table per fish
  bouts_GsizeMotionSingle.csv, bouts_clean.csv, repaired_frames.csv    the bout tables
  schizscreen_twin_v1/, schizscreen_twin_v1.zip                        the second data set (Larsch lab screen)
output/          from twin_output.zip, and everything computed later (not in git)
  checkpoints/     M<key>/: the five fold models of each thesis unit; screen_s0/, screen_s101/
  evaluation/      M<key>.json and summary_M<key>.json: every measurement of each unit
  figures_cache/, cache/, missing_pieces/, screen/, grouprep/       rollouts and numbers behind the figures
  two_targets/, multi_fish/                                         simulated by notebooks 07 and 09 (RUN = True)
  experiments/     what the experiments notebooks simulate and render
```

| Zip | Unzips to | Enables |
|---|---|---|
| `twin_data.zip` (7.3 GB) | `data/` | notebooks 01-02; training; the real fish in every comparison; the second data set (notebook 08) |
| `twin_output.zip` (2.1 GB) | `output/` | the trained models (106 thesis units and the screen-trained D4), their evaluations and the caches: notebooks 03-06 and 08, the experiments, `check.py` |

`AGENTS.md` is a guide to the code for coding agents and new contributors: the module map, the vocabulary, how to
do common tasks and the rules the code keeps.

## Models

A model is a *unit*: five fold models, each trained with three fish held out, at one cross-validation seed (0 or
101; seeds are never pooled). `twin/registry.py` lists every thesis model with its recipe;
`uv run python pipelines/train.py --list` prints them.

```bash
uv run python pipelines/train.py M6 --seed 0      # train (five folds in parallel, one thread each)
uv run python pipelines/train.py F --seed 0       # F = the profile correction fitted on D4 (needs D4's unit)
uv run python pipelines/evaluate.py M6 F --seed 0 101
```

Training takes 1-3 min per fold without the per-fish latent (M1), 10-60 min with it and 1-2 h with the slow drift;
the correction 45-80 min per fold; evaluating a unit about 15 min. Training is deterministic given the seed, and so
is evaluation: re-evaluating a unit reproduces its saved numbers exactly. On another machine or with other library
versions, retrained models differ within the seed-to-seed noise.

### Train and evaluate a model of your own

A model is a recipe in `twin/registry.py` plus a unit (a key per seed). To try a variant, add both, for example M6
with six bouts of step and interval history:

```python
# twin/registry.py
RECIPES = { ...,
    "M6, kin 6": recipe(**M6, Kk=6),        # the ingredients: see recipe()'s docstring
}
for args in [ ...,
    ("M6, kin 6", "x1", "x2"),              # unit keys at seed 0 and seed 101 (any unused names)
]:
```

then train it, evaluate it on the held-out fish and look at the results:

```bash
uv run python pipelines/train.py "M6, kin 6" --seed 0        # -> output/checkpoints/Mx1/
uv run python pipelines/evaluate.py "M6, kin 6" --seed 0     # -> output/evaluation/Mx1.json, summary_Mx1.json
```

```python
from twin import tables, unit_eval
unit_eval.show("x1")                                 # every measurement: classifier, individuality, profiles, ...
tables.profile_table(["M6", "M6, kin 6"])            # next to M6 in the thesis table (seed 101 once trained)
```

To retrain a thesis model without replacing its checkpoints, train into another folder and evaluate that folder;
the results go to `output/evaluation/<folder name>/`:

```bash
uv run python pipelines/train.py M6 --seed 0 --out output/checkpoints/M26e_retrain
uv run python pipelines/evaluate.py M6 --seed 0 --ckpts output/checkpoints/M26e_retrain
```

and `unit_eval.show("26e", eval_dir="output/evaluation/M26e_retrain")`. Both commands end by printing the next one.

Retraining a thesis unit in place (no `--out`) replaces its checkpoints. What was computed from the old ones is then
redone or refused, never used silently: the classifier's window cache is recomputed by the next evaluation; the tables
refuse a unit evaluated before it was retrained until it is evaluated again; the profile figures roll the retrained
units out again; the caches behind the turn-rule figures (`twin/missing_pieces.py`) name the call that rebuilds them.
The simulations beyond the data (two groups, shoals, the second data set) are cached per condition under
`output/two_targets/`, `output/multi_fish/` and `output/screen/` and are not checked: delete a condition's files to
simulate it again.

## How this was checked

The code was cleaned from the development repository (tag `pre-cleanup`) and checked against it:

- the training recipes against the ones that trained every thesis checkpoint, and every checkpoint loads;
- two-epoch fits bit-identical to the original code (M1, M6, D4, and the 3-layer, turn-size, dot-direction and
  position-only variants), and a two-step profile correction of D4 likewise;
- rollouts bit-identical on every path the thesis uses (evaluation, the correction, stimulus programmes, the second
  data set, shoals, two groups), and the evaluations of M6 and F reproducing their saved summaries exactly;
- the 30 fps tables rebuilt from the rig's files, and the bout table and the training table from them;
- every thesis figure pixel-identical, and the thesis tables reproduced from the saved evaluations;
- the cached shoal and two-group simulations reproduced by the new code, a single group or fish simulated alone
  identical to the same one in the full runs, and the shoal videos frame-identical after moving to `twin/videos.py`;
- the train-then-evaluate path itself: M1 retrained from scratch into a new folder gave bit-identical weights and
  its evaluation the saved summary; M6's checkpoints evaluated from a fresh folder (the classifier's windows
  recomputed) gave the saved summary, the saved panel and, for the new turn column, the cache behind the turn panel
  of the slow-state figure;
- the real-data analyses added to notebook 02 give the numbers of the development notebook they come from.

Options the thesis models never used were removed; a configuration that asks for one is refused rather than run.

## Reference

Arthur Pollet, *A Generative Model of Larval Zebrafish Swimming Behaviour*, MSc thesis, EPFL (LCN), 2026; supervised
by Johanni Brea and Johannes Larsch. The recordings: Matilde Perrino, *Visual perception of social groups in
zebrafish*, PhD thesis, University of Trento, 2026.
