"""Extra multi-fish runs for fig_multifish (review follow-up): the controls on M6 (the analysis model) and F s101.

Runs the UNCHANGED simulator (``multi_fish.run_group``) for conditions that were never run, with the same per-(group, fish)
seeds as every archived condition (start pose, random stream, latent, fold model g % 5), and writes the pickles to
Model/report_figs_cache/figs/multi_fish_extra/<cond>.pkl (no archived file under Model/ is touched). Then computes the archived readouts
(``multi_fish.groups_readout``, the same fields as ``multi_fish.analyse``) for them, with ``multi_fish.load`` patched in
this process to read the new pickles first and the archived ones (blind pools of existing conditions) otherwise.

    uv run python <this file> --run       # simulate (8 worker processes)
    uv run python <this file> --analyse   # readouts -> Model/report_figs_cache/figs/multi_fish_extra/summary_extra.json
"""
import contextlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "Model")]
import numpy as np  # noqa: E402

import multi_fish as MFish  # noqa: E402

OUT = ROOT / "Model" / "report_figs_cache" / "figs" / "multi_fish_extra"
_c = MFish._cond
EXTRA = {
    # training dish, random directions (scramble), M6 both seeds
    "m6_A1_N10_effects_scrambled": _c("26e", 10, 20, 1, 1800, "effects", scramble=True),
    "m6s101_A1_N10_effects_scrambled": _c("30d", 10, 20, 1, 1800, "effects", scramble=True),
    # 5x dish (radius x5), M6 both seeds: blind pool, effects, random directions
    "m6_A5_blind": _c("26e", 10, 20, 5, 3600, blind=True),
    "m6_A5_N10_effects": _c("26e", 10, 20, 5, 3600, "effects"),
    "m6_A5_N10_effects_scrambled": _c("26e", 10, 20, 5, 3600, "effects", scramble=True),
    "m6s101_A5_blind": _c("30d", 10, 20, 5, 3600, blind=True),
    "m6s101_A5_N10_effects": _c("30d", 10, 20, 5, 3600, "effects"),
    "m6s101_A5_N10_effects_scrambled": _c("30d", 10, 20, 5, 3600, "effects", scramble=True),
    # 5x dish random directions, F seed 101 (its blind pool s101_A5_blind is archived)
    "s101_A5_N10_effects_scrambled": _c("32b", 10, 20, 5, 3600, "effects", scramble=True),
}
MFish.CONDS.update(EXTRA)          # this process only
for k in EXTRA:
    MFish.LABEL.setdefault(k, k)
_orig_load = MFish.load


def load(cond):
    p = OUT / f"{cond}.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    return _orig_load(cond)


MFish.load = load


def _task(args):
    cond, spec, g = args
    t0 = time.time()
    res = MFish.run_group(spec, g)
    return cond, g, res, time.time() - t0


def run(conds, workers=8):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    OUT.mkdir(parents=True, exist_ok=True)
    conds = [c for c in conds if not (OUT / f"{c}.pkl").exists()]
    tasks = [(c, EXTRA[c], g) for c in conds for g in range(EXTRA[c]["G"])]
    tasks.sort(key=lambda t: -(t[1]["N"] ** 1.5 * t[1]["T"] * (0.3 if t[1]["blind"] else 1)))
    got, secs, t0 = {c: {} for c in conds}, {c: 0.0 for c in conds}, time.time()
    with ProcessPoolExecutor(workers, initializer=MFish._worker_init) as ex:
        futs = {ex.submit(_task, t): t for t in tasks}
        for k, f in enumerate(as_completed(futs), 1):
            c, g, res, el = f.result()
            got[c][g] = res
            secs[c] += el
            if len(got[c]) == EXTRA[c]["G"]:
                blob = dict(cond=c, spec=EXTRA[c], groups=[got[c][i] for i in range(EXTRA[c]["G"])], cpu_s=secs[c],
                            made=time.strftime("%Y-%m-%d %H:%M"))
                with open(OUT / f"{c}.pkl", "wb") as fh:
                    pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[{time.time() - t0:7.0f} s] {c}: done ({secs[c] / 3600:.2f} CPU-h)", flush=True)
            elif k % 20 == 0:
                print(f"[{time.time() - t0:7.0f} s] {k}/{len(tasks)} groups", flush=True)


def _group_job(c):
    r = MFish.groups_readout(c)
    G = MFish.CONDS[c]["G"]
    return c, dict(D_mean=float(r["D"].mean()), D_ci=list(r["D_ci"]), D=r["D"].tolist(),
                   D_fold_mean=float(r["D_fold"].mean()), D_fold_ci=list(MFish._boot(r["D_fold"])),
                   D_tshift_mean=float(r["D_tshift"].mean()), D_tshift_ci=list(MFish._boot(r["D_tshift"])),
                   D_blind_mean=float(r["D_blind"].mean()), D_blind_ci=list(MFish._boot(r["D_blind"])),
                   nn_mm=float(r["nn"].mean()), nn_shuf_mm=float(r["nn_shuf"].mean()),
                   nn_blind_mm=float(r["nn_blind"].mean()), big_mean=float(r["big"].mean()),
                   big_blind=float(r["big_blind"].mean()), G=G, blind_pool=MFish.blind_of(c))


def analyse(conds, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    conds = [c for c in conds if not MFish.CONDS[c]["blind"] and (OUT / f"{c}.pkl").exists()]
    with ProcessPoolExecutor(min(workers, len(conds))) as ex:
        res = dict(ex.map(_group_job, conds))
    meta = {c: dict(cpu_h=round(load(c)["cpu_s"] / 3600, 3), made=load(c)["made"], **EXTRA[c])
            for c in EXTRA if (OUT / f"{c}.pkl").exists()}
    S = dict(made=time.strftime("%Y-%m-%d %H:%M"), conds=meta, groups=res)
    (OUT / "summary_extra.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    for c, v in res.items():
        print(f"{c:36s} D {v['D_mean']:+.3f} [{v['D_ci'][0]:+.3f}, {v['D_ci'][1]:+.3f}]  fold {v['D_fold_mean']:+.3f}"
              f"  tshift {v['D_tshift_mean']:+.3f}  blind {v['D_blind_mean']:+.3f}  big {v['big_mean']:.2f}")


if __name__ == "__main__":
    with contextlib.suppress(Exception):
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if "--run" in sys.argv:
        run(list(EXTRA))
    if "--analyse" in sys.argv:
        analyse(list(EXTRA))
