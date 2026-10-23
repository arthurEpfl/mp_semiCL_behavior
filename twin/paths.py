"""Where the recordings, the trained models and every computed result live.

Nothing here is in git. Three roots, each overridable by an environment variable:

    TWIN_DATA     the recordings and the tables made from them         (default: <repo>/data, see data/README.md)
    TWIN_OUTPUT   checkpoints, evaluations and the caches of analyses   (default: <repo>/output)
    TWIN_FIGURES  the thesis figures                                    (default: <repo>/figures/out)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def _root(var, default):
    return Path(os.environ.get(var) or default)


DATA = _root("TWIN_DATA", ROOT / "data")
OUTPUT = _root("TWIN_OUTPUT", ROOT / "output")
FIGURES = _root("TWIN_FIGURES", ROOT / "figures" / "out")

# ---- recordings (data/README.md) and the tables built from them (notebooks/01_bout_discretisation.ipynb)
RAW = DATA / "raw"                                   # the rig's files
POSITIONS = RAW / "PositionTxt_allROI2025-02-20T15_37_59.csv"
TRAJECTORY = RAW / "2025.02.20-0957_trajectory_GsizeMotionSingle_180min.csv"
ROI_PHYS = RAW / "ROIdef2025-02-20T12_09_16.csv"     # dish geometry on the projector (the physical 100 mm dish)
ROI_CAM = RAW / "ROIdef.csv"                         # the same dishes in camera pixels
FRAMES = DATA / "frames"                             # per fish, 30 fps: *_f<N>_transformed_abs.csv
BOUTS_RAW = DATA / "bouts_GsizeMotionSingle.csv"     # the segmenter's output
BOUTS = DATA / "bouts_clean.csv"                     # the training table: every model reads this
REPAIRED = DATA / "repaired_frames.csv"
SCREEN = DATA / "schizscreen_twin_v1"                # Larsch lab screen (unpublished; its own DATA_USE.md)
SCREEN_ZIP = DATA / "schizscreen_twin_v1.zip"
PROGRAM = ASSETS / "stimulus" / "2025.02.20-0957_trajectory_GsizeMotionSingle_180min.csv.gz"

# ---- outputs
CKPTS = OUTPUT / "checkpoints"                       # one folder per model unit: M<key>/ (five fold checkpoints)
EVAL = OUTPUT / "evaluation"                         # per unit: M<key>.json (cells) and summary_M<key>.json
TRAIN_RUNS = OUTPUT / "training"                     # per-fold training logs
CACHE = OUTPUT / "cache"                             # small caches of the data figures and of training inputs
MISSING = OUTPUT / "missing_pieces"                  # turn rule, individuality, references (missing_pieces.py)
MULTI_FISH = OUTPUT / "multi_fish"                   # shoals (multi_fish.py)
TWO_TARGETS = OUTPUT / "two_targets"                 # two dot groups (two_targets.py)
SCREEN_OUT = OUTPUT / "screen"                       # the second data set (twin_test.py and screen_*.py)
GROUPREP = OUTPUT / "grouprep"                       # the group repulsion term (grouprep_fit.py)
FIG_CACHE = OUTPUT / "figures_cache"                 # rollouts and numbers behind the result figures
EXPERIMENTS = OUTPUT / "experiments"                 # what the notebooks in experiments/ simulate and render
