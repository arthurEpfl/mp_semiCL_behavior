"""Rebuild colab_bundle.zip from the CURRENT working tree.

The bundle is training/eval only: scripts + the bouts CSV + the physical ROIdef + the notebook.
It deliberately EXCLUDES data/data/df_transformed/*.csv (15 x 313k rows, hundreds of MB) -- those
are only needed to render comparison videos, which must be done locally.

Run:  python make_colab_bundle.py            # writes colab_bundle.zip, prints a manifest
      python make_colab_bundle.py --check    # verify the existing zip matches the tree, no write
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "colab_bundle.zip"

# Everything the eval/train path imports, plus the two runnable entry points.
SCRIPTS = [
    "common.py", "preprocess.py", "train.py", "sample.py", "compare.py", "visualize.py",
    "build_abs_dict.py", "segment_bouts.py", "add_dot_geometry.py",
    "fit_hmm.py", "fit_hmm_stim.py",
    "fit_MLP.py", "fit_MLP_border.py", "fit_MLP_mix.py", "fit_MLP_joint.py",
    "fit_MLP_fishre.py", "fit_GRU.py",
    "evaluate.py", "generative_metrics.py", "exchangeability.py",
    "calibrate.py", "calibrate_vigor.py",
    # rollout-aware lineage. rollout_aware.py (energy distance + REINFORCE) is kept for the record
    # but returned a null; rollout_torch/rollout_gan are the differentiable-simulator replacement.
    "rollout_aware.py", "rollout_torch.py", "rollout_gan.py",
    "time_state_fit.py", "eval_pair.py",                    # batch/lr/chunk cost measurement for the GRU fit
    "recalibrate_containment.py", "add_rad_indiv.py", "add_rad_gate.py",   # post-fit drivers
    "recalibrate_boundary.py",                                            # shallow-border retro-fit
    "recalibrate_gains.py", "diag_variance_split.py",                                     # per-channel latent gain retro-fit
    "bout_video.py",
    # diagnostics that are useful to run next to a fresh fit
    "diagnose_joint.py", "diagnose_wander.py", "diag_kin_gap.py", "diag_artr.py",
    "diag_range_profile.py", "diag_border_profile.py",
    # smoke tests (cheap structural guards worth having on the remote too)
    "smoke_shoal.py", "smoke_kin_trunk.py", "smoke_vigor.py", "smoke_artr.py",
    "smoke_kin_memory.py", "smoke_state_layer.py", "smoke_indiv.py", "smoke_joint.py",
    "smoke_fishre.py", "smoke_psi_frame.py", "smoke_bgain.py", "smoke_gruz.py", "smoke_srnn.py",
    "smoke_wall_reject.py", "smoke_noise_aug.py", "smoke_bit_identity.py",
    "smoke_dt_latent.py", "smoke_rollout_aware.py",
]

OTHER = [
    # The CLEANED table: tracking excursions repaired, plus run_id / active / run_pos and the
    # modelling features. 41 MB against the old 24 MB -- the extra 19 columns. The old
    # data/bouts_GsizeMotionSingle.csv is NOT bundled: it has no `active`, so
    # build_border_records now raises on it rather than silently training on gap rows.
    "clean_data/bouts_clean.csv",
    "data/data/Original_data/ROIdef2025-02-20T12_09_16.csv",
    "notebooks/evaluate_models.ipynb",
    "colab_run_ladder.py",
    # Model 1: trains the 4 stimulus representations, saves 5 folds + 1 all-data each, and
    # emits Metric A + the train/val curves. Rollouts and Metric B are deliberately NOT here
    # -- they are CPU work, run locally against these checkpoints.
    "train_model1_colab.py",
]


def manifest():
    files = [f"scripts/{s}" for s in SCRIPTS] + OTHER
    missing = [f for f in files if not (ROOT / f).exists()]
    return files, missing


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    args = ap.parse_args()

    files, missing = manifest()
    if missing:
        print("MISSING (not bundled):")
        for m in missing:
            print("   ", m)
        if not args.check:
            print("\nrefusing to build an incomplete bundle -- create the files or edit SCRIPTS/OTHER")
            return 1

    if args.check:
        if not OUT.exists():
            print("no colab_bundle.zip yet"); return 1
        z = zipfile.ZipFile(OUT)
        inzip = set(z.namelist())
        stale = []
        for f in files:
            if f not in inzip:
                stale.append(f"{f}: NOT IN ZIP")
            elif hashlib.sha256(z.read(f)).hexdigest()[:12] != _sha(ROOT / f):
                stale.append(f"{f}: DIFFERS from working tree")
        extra = sorted(inzip - set(files))
        for s in stale:
            print("  STALE ", s)
        for e in extra:
            print("  EXTRA ", e)
        print("bundle is up to date" if not stale else f"\n{len(stale)} stale entr(y/ies) -- rerun without --check")
        return 1 if stale else 0

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f)
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.name}  ({len(files)} entries, {size:.2f} MB)")
    for f in files:
        print(f"   {(ROOT / f).stat().st_size / 1e3:9.1f} KB  {f}")
    print("\nEXCLUDED on purpose: data/data/df_transformed/*.csv (video rendering only, too big)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
