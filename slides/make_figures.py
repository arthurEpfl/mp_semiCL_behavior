"""Generate every figure used by the model-lineage slide deck.

All figures come from artefacts already on disk -- no GPU, no refitting, no ffmpeg.
Each figure is cached: delete the PNG to force a rebuild.

    python slides/make_figures.py              # build what's missing
    python slides/make_figures.py --force      # rebuild everything
    python slides/make_figures.py --only fig2  # one figure

PROVENANCE. The deck stops at the `conv` ladder (e250). runs/mix_loop/samples/ is
k8-derived (written minutes before ladder_k8.mp4) and k8 post-dates that cutoff, so
step 0 regenerates paired trajectories from ladder_e250.pt into slides/samples_e250/
rather than reusing them. The existing k8 samples are never touched.
"""
import argparse
import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Artefacts were pickled on Colab under numpy 2; this environment is numpy 1.
if not hasattr(np, "_core"):
    import numpy.core as _nc
    sys.modules["numpy._core"] = _nc
    for _s in ("multiarray", "umath", "numeric", "_multiarray_umath", "numerictypes"):
        try:
            sys.modules[f"numpy._core.{_s}"] = __import__(f"numpy.core.{_s}", fromlist=[_s])
        except Exception:
            pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

FIGS = Path(__file__).resolve().parent / "figs"
SAMPLES = Path(__file__).resolve().parent / "samples_e250"
BOUTS = ROOT / "data" / "bouts_GsizeMotionSingle.csv"

# deck palette (matches model_lineage.tex)
WIN, LOSE, NEUT, REAL, GEN = "#006E3C", "#AA1E1E", "#787878", "#3C3C3C", "#1F3B8C"
DPI = 200

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# ladder dir -> (label, control-first order preserved in results.csv)
LADDER_SRC = {
    "stim": ROOT / "ladder_out", "wall": ROOT / "wall_out", "kin": ROOT / "kin_out",
    "disp": ROOT / "disp_out", "contain": ROOT / "contain_out",
    "tail": ROOT / "ladder_tail_results.zip", "far": ROOT / "far_out",
    "cal": ROOT / "ladder_cal_results.zip", "conv": ROOT / "conv_out",
}
FLOOR = 0.47610883928568803


# --------------------------------------------------------------------------- #
# shared loaders
# --------------------------------------------------------------------------- #
def load_ladder(src):
    """-> (results_df, artifacts_dict) from a *_out dir or a results zip."""
    if src.suffix == ".zip":
        z = zipfile.ZipFile(src)
        r = [n for n in z.namelist() if n.endswith("results.csv")][0]
        a = [n for n in z.namelist() if n.endswith("artifacts.pkl")][0]
        return pd.read_csv(io.BytesIO(z.read(r)), index_col=0), pickle.loads(z.read(a))
    with open(src / "artifacts.pkl", "rb") as fh:
        art = pickle.load(fh)
    return pd.read_csv(src / "results.csv", index_col=0), art


_RECORDS = None


def records():
    """Bout records with dish + dot geometry (cached across figures)."""
    global _RECORDS
    if _RECORDS is None:
        import evaluate as E
        import fit_MLP_border as MB
        print("  loading bouts + building records ...", flush=True)
        _RECORDS = MB.build_border_records(E.load_bouts(str(BOUTS)))
    return _RECORDS


def dish():
    recs = records()
    dd = [(r.dish_cx, r.dish_cy, r.dish_R) for r in recs.values() if hasattr(r, "dish_cx")]
    if not dd:
        return 180.0, 174.0, 183.5
    a = np.asarray(dd, float)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1])), float(np.median(a[:, 2]))


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1e3:.0f} KB)", flush=True)


# --------------------------------------------------------------------------- #
# step 0 -- regenerate the paired trajectories at e250
# --------------------------------------------------------------------------- #
FISH_IDS = [0, 3, 6, 9]          # same fish the ladder_e250.mp4 render used


def step0_samples(force=False):
    if SAMPLES.exists() and len(list(SAMPLES.glob("trial_*_real.csv"))) >= len(FISH_IDS) and not force:
        print("step0: samples_e250/ present, skipping regeneration")
        return
    import bout_video as BV
    import fit_MLP_mix as MX
    ckpt = ROOT / "ladder_e250.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt} -- needed to regenerate e250 trajectories")
    print(f"step0: regenerating e250 trajectories from {ckpt.name} (fish {FISH_IDS}) ...", flush=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    model = MX.load_mix(str(ckpt))
    n = BV.make_trials(model, records(), SAMPLES, fish_ids=FISH_IDS,
                       rng=np.random.default_rng(0), trial_length_s=60.0, verbose=True)
    print(f"step0: wrote {n} paired trials -> {SAMPLES.relative_to(ROOT)}", flush=True)


# --------------------------------------------------------------------------- #
# fig1 -- real vs generated trajectories in the dish
# --------------------------------------------------------------------------- #
def fig1_trajectories():
    cx, cy, R = dish()
    trials = sorted(SAMPLES.glob("trial_[0-9][0-9][0-9].csv"))
    if not trials:
        raise SystemExit("no trials in slides/samples_e250 -- run step 0 first")
    trials = trials[:4]
    fig, axes = plt.subplots(2, len(trials), figsize=(3.1 * len(trials), 6.4))
    axes = np.atleast_2d(axes)
    for j, tp in enumerate(trials):
        gen = pd.read_csv(tp)
        real = pd.read_csv(tp.with_name(tp.stem + "_real.csv"))
        for i, (df, lab, col) in enumerate(((real, "real", REAL), (gen, "generated", GEN))):
            ax = axes[i, j]
            ax.add_patch(plt.Circle((cx, cy), R, fill=False, ls="--", lw=1.0, color="0.55"))
            # active dots, faint, behind the path
            for k in range(8):
                s = df.get(f"d{k}_stim")
                if s is None:
                    continue
                m = np.asarray(s, float) > 0
                if m.any():
                    ax.scatter(df[f"d{k}_x"][m], df[f"d{k}_y"][m], s=0.7,
                               color="#E08A1E", alpha=0.20, linewidths=0, zorder=1)
            ax.plot(df["f_x"], df["f_y"], lw=0.8, color=col, alpha=0.9, zorder=2)
            ax.plot(df["f_x"].iloc[0], df["f_y"].iloc[0], "o", ms=4,
                    color=col, mec="white", mew=0.8, zorder=3)
            ax.set_aspect("equal")
            ax.set_xlim(cx - R * 1.08, cx + R * 1.08)
            ax.set_ylim(cy - R * 1.08, cy + R * 1.08)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(lab, fontsize=13, color=col, labelpad=8)
            if i == 0:
                ax.set_title(f"fish {FISH_IDS[j] if j < len(FISH_IDS) else j}", fontsize=11)
    fig.suptitle("60 s in the dish — real (top) vs generated (bottom), same stimulus",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig1_trajectories")


# --------------------------------------------------------------------------- #
# fig2 -- the progress ladder
# --------------------------------------------------------------------------- #
ADOPTED = [("base\n(dotb)", "stim", 0), ("shoal", "stim", 3), ("+drift", "kin", 1),
           ("+kfix", "kin", 2), ("+contain", "contain", 1), ("+tail", "tail", 1),
           ("+e250", "conv", 1)]


def fig2_progress():
    xs, ys = [], []
    for lab, lad, idx in ADOPTED:
        res, _ = load_ladder(LADDER_SRC[lad])
        ys.append(float(res["c2st_traj_auc"].iloc[idx]))
        xs.append(lab)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(range(len(xs)), ys, "-o", color=GEN, lw=2.0, ms=7, zorder=3)
    for i, v in enumerate(ys):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=10, color=GEN)
    ax.axhline(FLOOR, ls="--", lw=1.6, color=LOSE)
    ax.annotate(f"real-vs-real floor  {FLOOR:.4f}", (0, FLOOR), textcoords="offset points",
                xytext=(4, 7), fontsize=10, color=LOSE)
    closed = (ys[0] - ys[-1]) / (ys[0] - FLOOR)
    ax.annotate(f"{closed:.0%} of the closable gap removed",
                (len(xs) - 1, ys[-1]), textcoords="offset points", xytext=(-8, -26),
                ha="right", fontsize=11, color=WIN, fontweight="bold")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=10)
    ax.set_ylabel("trajectory C2ST AUC")
    ax.set_ylim(0.44, 0.93)
    ax.grid(axis="y", color="0.90")
    ax.set_axisbelow(True)
    save(fig, "fig2_progress")


# --------------------------------------------------------------------------- #
# fig3 -- dispersion, start vs current
# --------------------------------------------------------------------------- #
def _dispersion(lad, idx):
    import generative_metrics as GM
    res, art = load_ladder(LADDER_SRC[lad])
    name = list(res.index)[idx]
    rep = GM.dispersion_report(art["wins_traj"][name])
    return rep["var_ratio"], rep["between_fish_share"]


def fig3_dispersion():
    groups = ["turn", "distance", "dt", "dot-dist", "radial"]
    a, bfa = _dispersion("stim", 3)      # shoal, the lineage start
    b, bfb = _dispersion("conv", 1)      # e250, current
    labels = ["turn", "distance", "interval", "dot-dist", "radial"]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.bar(x - w / 2, [a.get(g, np.nan) for g in groups], w,
           color="0.72", edgecolor="0.45", label="shoal (lineage start)")
    ax.bar(x + w / 2, [b.get(g, np.nan) for g in groups], w,
           color=GEN, edgecolor="#16295E", label="e250 (current)")
    ax.axhline(1.0, ls="--", lw=1.6, color=LOSE)
    ax.annotate("target 1.0", (len(groups) - 0.45, 1.0), textcoords="offset points",
                xytext=(0, 6), ha="right", fontsize=10, color=LOSE)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("gen variance ÷ real variance")
    ax.set_title(f"between-fish share: real {bfb['real']:.3f}   "
                 f"generated {bfb['gen']:.3f}   (the individuality gap)", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(axis="y", color="0.90"); ax.set_axisbelow(True)
    save(fig, "fig3_dispersion")


# --------------------------------------------------------------------------- #
# fig4 -- range profile: the evidence behind `stim`
# --------------------------------------------------------------------------- #
def fig4_range(reps=2):
    import diag_range_profile as DR
    import fit_MLP_mix as MX
    recs = records()
    print("  fig4: real side ...", flush=True)
    real = DR.real_side(recs)
    out = {}
    for lab, ck in (("generated (dotb: bearing only)", ROOT / "ladder_out" / "ladder_dotb.pt"),
                    ("generated (shoal: range basis)", ROOT / "ladder_out" / "ladder_shoal.pt")):
        print(f"  fig4: rolling out {ck.name} ({reps} reps x {len(recs)} fish) ...", flush=True)
        out[lab] = DR.gen_side(MX.load_mix(str(ck)), recs, reps=reps, seed=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs = np.arange(len(DR.R_LABELS))
    ax.plot(xs, real["toward"], "-o", lw=2.4, ms=7, color=REAL, label="real fish", zorder=4)
    for (lab, g), col, ls in zip(out.items(), (LOSE, GEN), ("--", "-")):
        ax.plot(xs, g["toward"], ls, marker="s", ms=5, lw=1.8, color=col, label=lab)
    ax.axhline(0.5, color="0.75", lw=1.0, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(DR.R_LABELS)
    ax.set_xlabel("fish-to-shoal centroid distance (px)")
    ax.set_ylabel("P(turn toward the shoal)")
    ax.set_title("Real following is an inverted-U in range; a bearing-only model is flat", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", color="0.92"); ax.set_axisbelow(True)
    save(fig, "fig4_range")


# --------------------------------------------------------------------------- #
# fig5 -- radial occupancy: the evidence behind `contain`
# --------------------------------------------------------------------------- #
def fig5_radial(reps=2):
    import fit_MLP_mix as MX
    recs = records()
    edges = np.linspace(0, 1.0, 26)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    real = np.concatenate([np.hypot(np.asarray(r.x_px, float) - r.dish_cx,
                                    np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R
                           for r in recs.values()])
    curves = {"real fish": (np.histogram(real, edges, density=True)[0], REAL, "-", 2.4)}
    for lab, ck, col, ls in (("generated, before contain", ROOT / "contain_out" / "ladder_shoal_kfix.pt", LOSE, "--"),
                             ("generated, after contain", ROOT / "contain_out" / "ladder_contain.pt", WIN, "-")):
        print(f"  fig5: rolling out {ck.name} ({reps} reps x {len(recs)} fish) ...", flush=True)
        m = MX.load_mix(str(ck))
        acc = []
        for r in recs.values():
            for rep in range(reps):
                g = m.sample(r, np.random.default_rng([0, int(r.fish_id), rep]))
                acc.append(np.hypot(np.asarray(g["x_px"], float) - r.dish_cx,
                                    np.asarray(g["y_px"], float) - r.dish_cy) / r.dish_R)
        curves[lab] = (np.histogram(np.concatenate(acc), edges, density=True)[0], col, ls, 1.8)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for lab, (h, col, ls, lw) in curves.items():
        ax.plot(ctr, h, ls, lw=lw, color=col, label=lab)
    ax.axvline(0.85, color="0.6", lw=1.0, ls=":")
    ax.annotate("wall shell\nr/R > 0.85", (0.855, ax.get_ylim()[1] * 0.72), fontsize=9, color="0.35")
    ax.set_xlabel("radial position  r / R   (0 = dish centre, 1 = wall)")
    ax.set_ylabel("density of bouts")
    ax.set_title("Generated fish piled up on the wall; the shell-gated bias removed it", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(axis="y", color="0.92"); ax.set_axisbelow(True)
    save(fig, "fig5_radial")


# --------------------------------------------------------------------------- #
# fig6 -- what still makes generated windows detectable
# --------------------------------------------------------------------------- #
def fig6_importances():
    import generative_metrics as GM
    res, art = load_ladder(LADDER_SRC["conv"])
    name = list(res.index)[1]                       # e250
    wins = art["wins_traj"][name]
    print("  fig6: permutation importances (this fits forests, ~1-3 min) ...", flush=True)
    imp = GM.c2st_importances(wins, kind="permutation", n_repeats=5)
    imp = imp.clip(lower=0)
    tot = float(imp.sum()) or 1.0
    grp = GM.traj_groups(list(imp.index))
    shares = {g: float(imp[list(cols)].sum()) / tot for g, cols in grp.items() if len(cols)}
    top = imp.sort_values(ascending=False)[:8] / tot

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    gs = sorted(shares.items(), key=lambda kv: -kv[1])
    cols = [LOSE if k == "radial" else GEN for k, _ in gs]
    a1.barh([k for k, _ in gs][::-1], [v for _, v in gs][::-1], color=cols[::-1], edgecolor="none")
    a1.set_xlabel("share of remaining detectability")
    a1.set_title("by feature group", fontsize=11)
    a1.grid(axis="x", color="0.92"); a1.set_axisbelow(True)

    a2.barh(list(top.index)[::-1], list(top.values)[::-1], color="0.55", edgecolor="none")
    a2.set_xlabel("share of remaining detectability")
    a2.set_title("top individual features", fontsize=11)
    a2.grid(axis="x", color="0.92"); a2.set_axisbelow(True)
    fig.suptitle("What still tells generated windows apart (e250, permutation importances)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_importances")


# --------------------------------------------------------------------------- #
BUILDERS = {"fig1": fig1_trajectories, "fig2": fig2_progress, "fig3": fig3_dispersion,
            "fig4": fig4_range, "fig5": fig5_radial, "fig6": fig6_importances}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the PNG exists")
    ap.add_argument("--only", nargs="+", default=None, choices=list(BUILDERS) + ["step0"])
    args = ap.parse_args()

    want = args.only or (["step0"] + list(BUILDERS))
    if "step0" in want:
        step0_samples(force=args.force)
    for k in BUILDERS:
        if k not in want:
            continue
        outs = list(FIGS.glob(f"{k}_*.png"))
        if outs and not args.force:
            print(f"{k}: cached ({outs[0].name})")
            continue
        print(f"{k}: building ...", flush=True)
        BUILDERS[k]()
    print("\nfigures in", FIGS.relative_to(ROOT))


if __name__ == "__main__":
    main()
