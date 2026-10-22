"""Shared style for the thesis figures of Chapters 4-6 (Methods, Results, Discussion).

Same look as ``scripts/make_report_figures.py`` (Chapter 3): DejaVu Sans 8 pt, recessive axes,
Okabe-Ito colours, vector PDFs written into ``Final_Report/figures/`` at the width they are shown.

Usage in a figure script (one script per figure, ``scripts/report_figs/fig_<name>.py``):

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from style import *          # noqa: F401,F403  (sets rcParams, gives save(), panel(), colours)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.4))
    ...
    save(fig, "fig_results_profiles")      # -> Final_Report/figures/fig_results_profiles.pdf (+ PNG preview)
"""

import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
MODEL = ROOT / "Model"
OUT = ROOT / "Final_Report" / "figures"
# Caches behind the figures (rollout curves, two-group / multi-fish summaries, the extra multi-fish runs) and the PNG
# previews live in Model/report_figs_cache/ (untracked data, like the rest of Model/). Override with REPORT_FINAL_DIR /
# REPORT_FIG_PNG.
REPORT_FINAL_DIR = Path(os.environ.get("REPORT_FINAL_DIR", MODEL / "report_figs_cache"))
PNG_DIR = Path(os.environ.get("REPORT_FIG_PNG", REPORT_FINAL_DIR / "png"))

for p in (str(SCRIPTS), str(MODEL)):
    if p not in sys.path:
        sys.path.insert(0, p)

TEXTWIDTH_IN = 6.3          # \textwidth at a4paper with 2.5 cm margins

# ---- one scale for every millimetre: the physical 3.60 px/mm (Chapter 3) ---------------------------
# The profile, attraction, profile-correction and two-group code divides px by the projector's nominal 3.67 px/mm
# (model_figures.PX_TO_MM, multi_stim.PX_PER_MM); Chapter 3, the turn laws, the centroid test and the multi-fish code
# use the physical 3.60. Every millimetre that comes from the 3.67 code is multiplied by this factor at plotting time
# (axes, bands, CIs, annotations, printed numbers), so every figure is at 3.60 px/mm: mm_3.60 = mm_3.67 x 3.67/3.60.
# Archive checks compare the UNconverted values with the archive (or converted values with archived x factor).
# Never applied to latencies (s), errors in real-SEM units, probabilities, preferences, AUCs, likelihoods or angles.
MM_3P67_TO_3P60 = 3.67 / 3.60


def assert_code_scale(px_per_mm, name):
    """The factor is right only for code that divides px by 3.67 px/mm: refuse to convert anything else."""
    if abs(float(px_per_mm) - 3.67) > 1e-12:
        raise SystemExit(f"{name} = {px_per_mm} px/mm, not 3.67: MM_3P67_TO_3P60 does not apply")

# ---- colours (Okabe-Ito; the Chapter 3 trio first) ---------------------------------------------
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, SKY, PURPLE, YELLOW = "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"
INK, MUTED, FAINT = "#1a1a1a", "#666666", "#c8c8c8"

# The two headline models and the real fish. Seed 0 = solid line / filled marker,
# seed 101 = dashed line / open marker (never pooled, always shown side by side).
REAL_C = INK
MODEL_C = {"M6": BLUE, "F": VERM, "M6+C": SKY}
SEED_LS = {0: "-", 101: "--"}
SEED_MFC = {0: "full", 101: "none"}      # matplotlib fillstyle
# Stimulus conditions, as in Chapter 3.
COND_C = {"moving": BLUE, "static": VERM, "none": MUTED}
# Two-group combination rules (distinct from the model colours on purpose).
RULE_C = {"effects": GREEN, "add": ORANGE, "avg": PURPLE, "nearest": SKY,
          "reference": MUTED}

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "grid.color": FAINT, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def panel(ax, letter, title=None):
    """Bold panel letter at the top left, optional left-aligned bold title (Chapter 3 layout)."""
    if title:
        ax.set_title(title, loc="left", pad=6)
    ax.text(-0.02, 1.16 if title else 1.06, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="right")


def save(fig, name):
    """Write Final_Report/figures/<name>.pdf and a PNG preview in PNG_DIR."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_DIR / f"{name}.png", dpi=170)
    plt.close(fig)
    print(f"  wrote {OUT / (name + '.pdf')}  (preview {PNG_DIR / (name + '.png')})")
