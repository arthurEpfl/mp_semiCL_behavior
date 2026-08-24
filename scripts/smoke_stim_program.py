"""Smoke tests for the stimulus-programme rollout tool.

Run: ``python scripts/smoke_stim_program.py``  (needs the recorded experiment on disk, because
several checks compare the tool against ground truth built by unrelated code paths).

The load-bearing one is check 3: the ``stim_provider``/``stop_time_s`` hook added to
``fit_MLP_joint.generate_mix_joint_loop`` must leave the ordinary rollout BIT-IDENTICAL, or every
archived number in the project silently moves.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as E
import fit_MLP_border as MB
import fit_MLP_mix as MX
import stim_program as SP
import stim_rollout as SR

_ROOT = Path(__file__).resolve().parents[1]
TRAJ = _ROOT / "data/data/Original_data/2025.02.20-0957_trajectory_GsizeMotionSingle_180min.csv"
MODEL = _ROOT / "notebooks/_zip_full/ladder_b_indiv.pt"
BOUTS = _ROOT / "data/bouts_GsizeMotionSingle.csv"
TRANSFORMED = _ROOT / "data/data/df_transformed"

# Recorded BEFORE the hook was added (default path, default_rng([7, fid]), sha1 of
# dtheta|dt_s|distance_px|x_px|y_px). These are the regression anchors.
BASELINE = {0: "b62d6c9b4eecfe7e", 3: "6186a7d7f12e13da", 6: "6b2dc02c778adbfe"}

_n_fail = 0


def check(name, ok, detail=""):
    global _n_fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        _n_fail += 1
    return ok


def _hash(g):
    h = hashlib.sha1()
    for k in ("dtheta", "dt_s", "distance_px", "x_px", "y_px"):
        h.update(np.asarray(g[k], float).tobytes())
    return h.hexdigest()[:16]


def _truncate(prog, n_blocks):
    """First ``n_blocks`` blocks only, so the whole path runs in seconds."""
    cut = int(prog.ends[n_blocks - 1])
    return SP.StimulusProgram(
        dx=prog.dx[:cut], dy=prog.dy[:cut], active=prog.active[:cut],
        block_id=prog.block_id[:cut], labels=prog.labels[:n_blocks],
        starts=prog.starts[:n_blocks], ends=prog.ends[:n_blocks],
        n_dots_of=prog.n_dots_of[:n_blocks], moving_of=prog.moving_of[:n_blocks],
        side_of=prog.side_of[:n_blocks], is_stim_of=prog.is_stim_of[:n_blocks],
        fps=prog.fps, path=prog.path)


def main():
    print("loading ...")
    prog = SP.load_program(TRAJ)
    model = MX.load_mix(str(MODEL))
    records = MB.build_border_records(E.load_bouts(str(BOUTS)))
    short = _truncate(prog, 10)

    print("\n1. programme parsing")
    check("frame count is a whole number of 30 fps seconds", prog.n_frames % 30 == 0,
          f"{prog.n_frames} frames = {prog.duration_s / 60:.1f} min")
    check("trailing 'bout' block dropped", "bout" not in set(prog.labels))
    check("every stimulus block has N >= 1",
          bool((prog.n_dots_of[prog.is_stim_of] >= 1).all()))
    check("no dots active during pauses",
          not prog.active[~prog.is_stim_of[prog.block_id]].any())

    print("\n2. anchoring transform vs df_transformed (independent code path)")
    mx, n_cmp = SP.verify_against_transformed(prog, transformed_dir=TRANSFORMED, verbose=False)
    check("max abs error < 1e-9 px", mx < 1e-9, f"{mx:.3e} px over {n_cmp} (frame, dot) pairs")

    print("\n3. NO REGRESSION: default path bit-identical to pre-hook")
    for fid, exp in BASELINE.items():
        got = _hash(model.sample(records[fid], np.random.default_rng([7, fid])))
        check(f"fish {fid} sha1", got == exp, f"{got} vs {exp}")
    keys = set(model.sample(records[0], np.random.default_rng([7, 0])))
    check("default sample dict keys unchanged",
          keys == {"dtheta", "symbol", "dt_s", "distance_px", "x_px", "y_px",
                   "n_hits", "n_bounces", "stim_angle_gen"}, str(sorted(keys)))

    print("\n4. provider parity vs bouts-CSV dot_cx/dot_cy (independent code path)")
    pmx, _pol = SR.verify_provider_parity(prog, records, transformed_dir=TRANSFORMED, verbose=False)
    check("max abs error < 1e-9 px", pmx < 1e-9, f"{pmx:.3e} px")

    print("\n5. rollout on the programme")
    gs = SR.rollout_many(model, short, 3, seed=0, records=records, verbose=False)
    dish = gs[0]["dish"]
    for k, g in enumerate(gs):
        rr = np.hypot(g["x_px"] - dish[0], g["y_px"] - dish[1]) / dish[2]
        check(f"rollout {k} contained", int((rr > 1.0).sum()) == 0, f"max r/R {rr.max():.3f}")
        short_by = g["program_s"] - g["last_onset_s"]
        check(f"rollout {k} ends within one interval of the programme",
              0 <= short_by <= float(g["dt_s"][-1]) + 1e-6,
              f"short by {short_by:.1f}s, final dt {g['dt_s'][-1]:.1f}s")
        check(f"rollout {k} bout frames inside the programme",
              bool((g["frame_onset"] < short.n_frames).all()))

    print("\n6. the stimulus actually drives the fish")
    # Same seed, mirrored programme (dot x negated = shoal swapped to the other side). If the
    # provider were inert the two rollouts would be identical; a genuinely stimulus-driven fish
    # must diverge. This is the direct test that the hook is wired, not just present.
    flip = SP.StimulusProgram(**{**short.__dict__, "dx": -short.dx})
    a = SR.rollout_many(model, short, 1, seed=11, records=records, verbose=False)[0]
    b = SR.rollout_many(model, flip, 1, seed=11, records=records, verbose=False)[0]
    n = min(len(a["dtheta"]), len(b["dtheta"]))
    check("mirrored shoal changes the trajectory",
          not np.array_equal(a["dtheta"][:n], b["dtheta"][:n]),
          f"first differing bout {int(np.argmax(a['dtheta'][:n] != b['dtheta'][:n]))}")
    # ... and the bearing it reacted to must mirror at the first stimulus bout, before the paths
    # diverge enough to decorrelate.
    pa, pb = a["stim_angle_gen"], b["stim_angle_gen"]
    i = int(np.flatnonzero(np.isfinite(pa[:n]) & np.isfinite(pb[:n]))[0])
    check("bearing mirrors on the first stimulus bout",
          abs(pa[i] + pb[i]) < 1e-9, f"phi {pa[i]:+.4f} vs {pb[i]:+.4f} rad")

    print("\n7. following index is graded, not flat (full programme, 6 fish)")
    gs6 = SR.rollout_many(model, prog, 6, seed=0, records=records, verbose=False)
    d = np.concatenate([g["dtheta"] for g in gs6])
    p = np.concatenate([g["stim_angle_gen"] for g in gs6])
    nd = np.concatenate([prog.n_dots_of[g["block_id"]] for g in gs6])
    mv = np.concatenate([prog.moving_of[g["block_id"]].astype(float) for g in gs6])
    ok = np.isfinite(p) & (np.abs(d) > 10.0) & (mv > 0)
    fi = {int(k): float(np.mean(np.sign(d[m]) == np.sign(p[m])))
          for k in sorted(set(nd[ok].astype(int))) for m in [ok & (nd == k)] if m.sum() > 50}
    check("P(turn toward shoal) > 0.5 at every N", all(v > 0.5 for v in fi.values()),
          str({k: round(v, 3) for k, v in fi.items()}))
    check("large groups followed more than N=1", fi[max(fi)] > fi[min(fi)],
          f"N={min(fi)}: {fi[min(fi)]:.3f} -> N={max(fi)}: {fi[max(fi)]:.3f}")

    print(f"\n{'ALL CHECKS PASSED' if not _n_fail else f'{_n_fail} CHECK(S) FAILED'}")
    return 1 if _n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
