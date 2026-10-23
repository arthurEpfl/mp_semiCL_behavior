# A behavioural digital twin of larval zebrafish

Fifteen fish, each alone in a dish, were shown groups of one to eight moving or static dots in a
semi-closed-loop assay (Perrino, 2026); their swimming was cut into bouts, and generative models of the bout sequence
were trained on it, simulated in closed loop, evaluated against held-out fish and taken beyond the data (two groups of
dots, several fish in one dish, a second data set).

Everything in the thesis is reproduced by the notebooks in `notebooks/`, from the code in `twin/` (the library) and
`figures/` (one script per thesis figure).

## Layout

| Path | What it holds |
|---|---|
| `twin/` | the library: preprocessing (`segment_bouts`, `cleaning`), the models (`stimulus`, `features`, `emission`, `joint`, `latent`), training (`training`, `profile_finetune`, `correction`), simulation (`stim_rollout`, `rollout_*`), evaluation (`unit_eval`, `rollout_eval`, `generative_metrics`, `model_figures`, `tables`), the analyses beyond the data (`missing_pieces`, `two_targets`, `multi_fish`, the `screen_*` modules) |
| `twin/registry.py` | every thesis model: its name, recipe, seeds and checkpoint folder |
| `twin/paths.py` | where the data, the trained models and every result live |
| `figures/` | one script per thesis figure, writing `figures/out/<name>.pdf` |
| `pipelines/` | the command lines: train and evaluate a model |
| `notebooks/` | one notebook per part of the thesis, with its outputs |
| `assets/` | the stimulus programme the rig played and the dish geometry (no animal data) |
| `data/` | the recordings (not in git, see `data/README.md`) |
| `output/` | checkpoints, evaluations and caches (not in git) |

## Setup

```bash
uv sync                      # Python >= 3.13 and every dependency, from uv.lock
```

Put the recordings in `data/` as described in `data/README.md`. Trained checkpoints and computed results go to
`output/`; to use folders elsewhere, set `TWIN_DATA`, `TWIN_OUTPUT` and `TWIN_FIGURES`. Open the notebooks with
`uv run jupyter lab`.

## The notebooks

| Notebook | Thesis | What it reproduces |
|---|---|---|
| `01_bout_discretisation` | Data | from the rig's raw files to the 30 fps tables and on to the bout table (Figures fig:detection, fig:bouttraj), checked identical to the training table |
| `02_data_exploration` | Data | every figure and number of the Data chapter |
| `03_models` | Methods, Results | the main line M1-M6 and the drift branch D1-D4, F: Tables tab:storyline, tab:realism, tab:app-imp-*, tab:profile-spread; Figures fig:slow, fig:profiles, fig:app-bands-*, fig:app-latent-gain, fig:app-drift-gain, fig:app-correction-trace; every model's measurements |
| `04_other_results` | Appendix (negative results) | Tables tab:app-negres-net, tab:app-negres-drift, tab:app-negres-other |
| `05_ablations` | Results | Tables tab:abl-all, tab:abl-indiv; Figure fig:abl-profiles |
| `06_turn_rule_and_individuals` | Results, Appendix | Figures fig:rule, fig:range-law; Tables tab:range-law, tab:nrn; the classifier's real-against-real floor |
| `07_two_groups` | Results, Appendix | Figures fig:bifurcation, fig:app-bif-* |
| `08_second_data_set` | Results, Appendix | Tables tab:screen-auc, tab:app-screen-kin, tab:app-screen-importances, tab:app-grouprep; Figures fig:screen-response, fig:screen-switch, fig:app-grouprep |
| `09_shoaling` | Results | Tables tab:shoal-grid, tab:shoal-pairs; the shoaling videos |

The notebooks read saved results and say how to recompute them; each runs in minutes once `output/` holds the
checkpoints and caches. Every figure script also runs on its own, e.g. `uv run python -m figures.fig_profiles`.

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
- the cached shoal and two-group simulations reproduced by the new code.
- the train-then-evaluate path itself: M1 retrained from scratch into a new folder gave bit-identical weights and
  its evaluation the saved summary; M6's checkpoints evaluated from a fresh folder (the classifier's windows
  recomputed) gave the saved summary, the saved panel and, for the new turn column, the cache behind the turn panel
  of the slow-state figure.

Options the thesis models never used were removed; a configuration that asks for one is refused rather than run.
