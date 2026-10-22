"""The whole group-repulsion run, stage by stage, resumable (2026-10-18). Every stage leaves a marker in
Model/grouprep/status/ and is skipped once done; the stages themselves also skip finished work. A failing stage is logged
and the run moves on to the stages that do not need it (the shoal runs go ahead whatever the checks show).

    uv run --with pyarrow python grouprep_pipeline.py      # (pyarrow: the Larsch screen's parquet files)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOUT = ROOT / "Model" / "grouprep"
STATUS = GOUT / "status"
LOG = GOUT / "pipeline.log"
PY = sys.executable

STAGES = [
    ("check", [PY, "-u", "grouprep_fit.py", "check"], HERE, ()),
    ("fit_14f", [PY, "-u", "grouprep_fit.py", "fit", "14f"], HERE, ("check",)),
    ("fit_32b", [PY, "-u", "grouprep_fit.py", "fit", "32b"], HERE, ("check",)),
    ("build_14f", [PY, "-u", "grouprep_fit.py", "build", "14f"], HERE, ("fit_14f",)),
    ("build_32b", [PY, "-u", "grouprep_fit.py", "build", "32b"], HERE, ("fit_32b",)),
    ("test_14f", [PY, "-u", "grouprep_fit.py", "test", "14f"], HERE, ("build_14f",)),
    ("test_32b", [PY, "-u", "grouprep_fit.py", "test", "32b"], HERE, ("build_32b",)),
    ("lik", [PY, "-u", "grouprep_fit.py", "lik"], HERE, ("build_14f", "build_32b")),
    ("scenes", [PY, "-u", "repulsion_scenes.py", "gr"], HERE, ("build_14f", "build_32b")),
    ("shoal_run", [PY, "-u", "multi_fish.py", "--run-grouprep"], HERE, ("build_14f", "build_32b")),
    ("shoal_analyse", [PY, "-u", "multi_fish.py", "--analyse-grouprep"], HERE, ("shoal_run",)),
    ("notebook_1", [PY, "-u", "final_report_shoaling.py"], ROOT / "Model", ()),
    ("auc_a", [PY, "-u", "report_models.py", "run", "51a,51b,51c,51d,51e,51f"], ROOT / "Model", ("build_14f", "build_32b")),
    ("auc_b", [PY, "-u", "report_models.py", "run", "51g,51h,51i,51j,51k,51l"], ROOT / "Model", ("build_14f", "build_32b")),
    ("notebook_2", [PY, "-u", "final_report_shoaling.py"], ROOT / "Model", ()),
]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    STATUS.mkdir(parents=True, exist_ok=True)
    failed = set()
    for name, cmd, cwd, needs in STAGES:
        done = STATUS / f"{name}.done"
        if done.exists() and not name.startswith("notebook"):
            log(f"{name}: already done")
            continue
        miss = [n for n in needs if not (STATUS / f"{n}.done").exists()]
        if miss:
            log(f"{name}: SKIPPED (needs {miss})")
            failed.add(name)
            continue
        log(f"{name}: start  ({' '.join(str(c) for c in cmd[2:])})")
        t0 = time.time()
        with open(GOUT / f"{name}.log", "a", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
                                 env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        el = (time.time() - t0) / 60
        if rc == 0:
            done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"{name}: done in {el:.0f} min")
        else:
            failed.add(name)
            log(f"{name}: FAILED (exit {rc}) after {el:.0f} min -- see {GOUT / (name + '.log')}")
    log(f"pipeline finished; failed / skipped: {sorted(failed) or 'none'}")


if __name__ == "__main__":
    main()
