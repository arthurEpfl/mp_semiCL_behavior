"""Does the new code still reproduce the ARCHIVED checkpoints, byte for byte?

This is the check that stands in for a control arm. The 2026-08-24 ladder block spends all eight
Colab slots on new mechanisms and reads them against the archived J23 number (c2st_traj 0.6739),
which is only legitimate if the code that produced that number still behaves identically when the
new flags are absent. Four mechanisms were threaded through the shared feature builder, the
shared rollout and the shared calibrator dispatch; every one of them is key-gated, and this
asserts that the gating actually holds.

THE REFERENCE IS ``baseline_scripts_psire2.zip`` -- a FROZEN copy of ``scripts/`` as it stood when
the psire2/psire3 ladders ran, i.e. literally the code that produced
``psire2_out/ladder_re_swap_psib.pt``. Running both versions side by side is stronger than any
assertion about what "should" be unchanged, and it is the only option available: the fit_MLP_*
modules are UNTRACKED in git, so there is no HEAD to diff against.

It must be a frozen archive and NOT ``colab_bundle.zip``. The bundle is rebuilt on every upload,
so pointing this at it makes the check compare the new code against itself and pass vacuously --
which is exactly what happened the first time this was written. ``colab_bundle.zip`` is still
accepted as a fallback so the check degrades to "at least it runs" rather than vanishing, but it
prints a loud warning when it is used.

    python scripts/smoke_bit_identity.py
    python scripts/smoke_bit_identity.py --ckpt psij2_out/ladder_b_indiv_psib.pt
    python scripts/smoke_bit_identity.py --ref some_other_snapshot.zip

Compares, over all 15 fish: the training feature matrix, a full rollout (dtheta / distance / dt /
x / y) and all four ``score`` channels. NaN is compared as equal to NaN -- the distance channel
masks the ten bouts with distance exactly 0, so a plain ``array_equal`` reports a spurious
mismatch there (it did, on first writing).
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline_scripts_psire2.zip"     # frozen; never regenerate from the live tree
BUNDLE = ROOT / "colab_bundle.zip"                  # fallback only -- rebuilt on every upload
ROI = "data/data/Original_data/ROIdef2025-02-20T12_09_16.csv"

_WORKER = r'''
import sys, pickle, numpy as np
sys.path.insert(0, sys.argv[1])
import evaluate as E, fit_MLP_border as MB, fit_MLP_mix as MX
df = E.load_bouts(sys.argv[3])
recs = MB.build_border_records(df, roi_path=sys.argv[4])
m = MX.load_mix(sys.argv[5])
out = {}
for fid in sorted(recs):
    r = recs[fid]
    out[("R", fid)] = {k: np.asarray(v) for k, v in m.sample(r, np.random.default_rng([0, fid])).items()
                       if isinstance(v, np.ndarray)}
    out[("X", fid)] = MB.build_feature_matrix(r, m._cfg_feat)
    out[("S", fid)] = {k: np.asarray(v) for k, v in m.score(r).items()}
pickle.dump(out, open(sys.argv[2], "wb"))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="psire2_out/ladder_re_swap_psib.pt")
    ap.add_argument("--bouts", default="data/bouts_GsizeMotionSingle.csv")
    ap.add_argument("--ref", default=None, help=f"reference snapshot (default {BASELINE.name})")
    args = ap.parse_args()
    ckpt = ROOT / args.ckpt
    if not ckpt.exists():
        print(f"SKIP: {ckpt} not present (download it from the ladder zip first)")
        return 0
    ref = Path(args.ref) if args.ref else (BASELINE if BASELINE.exists() else BUNDLE)
    if not ref.exists():
        print(f"SKIP: no reference snapshot ({BASELINE.name} / {BUNDLE.name}) -- nothing to "
              "compare against")
        return 0
    if ref.resolve() == BUNDLE.resolve():
        print(f"WARNING: falling back to {BUNDLE.name}, which is REBUILT on every upload. If it "
              "already contains the current code this check passes vacuously.")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(ref) as z:
            n = 0
            for f in z.namelist():
                if f.startswith("scripts/") and f.endswith(".py"):
                    z.extract(f, td); n += 1
        print(f"reference: {ref.name} ({n} scripts)   checkpoint: {ckpt.name}")
        w = td / "worker.py"
        w.write_text(_WORKER, encoding="utf-8")
        blobs = {}
        for tag, scripts in (("old", td / "scripts"), ("new", ROOT / "scripts")):
            out = td / f"{tag}.pkl"
            r = subprocess.run([sys.executable, str(w), str(scripts), str(out),
                                str(ROOT / args.bouts), str(ROOT / ROI), str(ckpt)],
                               cwd=str(ROOT), capture_output=True, text=True)
            if r.returncode:
                print(f"FAIL: the {tag} worker crashed\n{r.stderr[-2000:]}")
                return 1
            blobs[tag] = pickle.load(open(out, "rb"))
            print(f"  {tag} pass done")

    a, b = blobs["old"], blobs["new"]
    if set(a) != set(b):
        print("FAIL: different record sets")
        return 1
    diffs, nan_only, worst = [], 0, {}
    for k in a:
        va, vb = a[k], b[k]
        items = va.items() if isinstance(va, dict) else [(k[0], va)]
        for kk, arr in items:
            arr2 = vb[kk] if isinstance(vb, dict) else vb
            tag = f"{k[0]}:{kk}"
            fa, fb = np.asarray(arr, float), np.asarray(arr2, float)
            if fa.size:
                worst[tag] = max(worst.get(tag, 0.0), float(np.nanmax(np.abs(fa - fb))))
            if np.array_equal(arr, arr2):
                continue
            if np.array_equal(fa, fb, equal_nan=True):
                nan_only += 1                      # identical values AND identical NaN masks
            else:
                diffs.append((k, kk))

    print("\narchived checkpoint under old vs new code, 15 fish:")
    for t in sorted(worst):
        print(f"   {t:<24} max|diff| = {worst[t]:.3e}")
    print(f"\n   arrays equal only under equal_nan (masked bouts): {nan_only}")
    ok = not diffs
    print(f"   genuinely different arrays: {len(diffs)} {diffs[:5] if diffs else ''}")
    print("\n" + ("BIT-IDENTICAL -- the archived baseline is still valid"
                  if ok else "CHANGED -- the archived baseline is NOT a valid reference"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
