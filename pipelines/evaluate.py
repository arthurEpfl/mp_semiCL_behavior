"""Evaluate thesis models: every measurement of twin/unit_eval.py on held-out fish, saved in output/evaluation/.

    uv run python pipelines/evaluate.py M6 --seed 0          # one unit (about 15 min on 4 cores)
    uv run python pipelines/evaluate.py M6 F --seed 0 101    # several, one process per unit, in parallel
    uv run python pipelines/evaluate.py --keys 26e,14f       # by unit key
    uv run python pipelines/evaluate.py M6 --seed 0 --ckpts output/checkpoints/M26e_retrain
                                                             # other checkpoints of M6's recipe (train.py --out)

Each unit writes M<key>.json (the outputs shown in the models notebooks and by unit_eval.show) and summary_M<key>.json
(the numbers of every model table). Re-running a unit reproduces its summary exactly. With --ckpts the results go to
output/evaluation/<folder name>/ (or --out), never over the thesis evaluation of the unit.
"""
import argparse
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="thesis model names (see pipelines/train.py --list)")
    ap.add_argument("--seed", type=int, nargs="+", choices=(0, 101), default=[0, 101])
    ap.add_argument("--keys", default="", help="comma-separated unit keys instead of names")
    ap.add_argument("--ckpts", default=None, help="score the checkpoints in this folder instead of the unit's own "
                                                  "(one unit; the folder must hold the unit's arm)")
    ap.add_argument("--out", default=None, help="evaluation folder (default output/evaluation, with --ckpts "
                                                "output/evaluation/<folder name>)")
    a = ap.parse_args()
    from pathlib import Path

    from twin import registry, unit_eval
    keys = [k for k in a.keys.split(",") if k] or [registry.BY_NAME[(n, s)] for n in a.names for s in a.seed
                                                    if (n, s) in registry.BY_NAME]
    if not keys:
        ap.error("no unit to evaluate")
    if a.ckpts and len(keys) != 1:
        ap.error("--ckpts scores one unit: give one name and one --seed, or one --keys")
    if len(keys) == 1:
        unit_eval.evaluate(keys[0], out_dir=Path(a.out) if a.out else None, ckpt_dir=a.ckpts)
        return
    t0 = time.time()
    extra = ["--out", a.out] if a.out else []
    procs = [(k, subprocess.Popen([sys.executable, "-u", "-W", "ignore", __file__, "--keys", k] + extra))
             for k in keys]
    bad = [k for k, p in procs if p.wait()]
    print(f"{len(keys)} units in {(time.time() - t0) / 60:.0f} min" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":      # required on Windows: the rollout workers re-import this module
    main()
