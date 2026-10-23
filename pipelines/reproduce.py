"""Reproduce the thesis results, from the saved results, from the checkpoints, or from the data.

    uv run python pipelines/reproduce.py --level figures --dry-run    # what would run, and roughly how long
    uv run python pipelines/reproduce.py --level figures              # every table and figure from the saved results
    uv run python pipelines/reproduce.py --level evaluate --jobs 4    # + re-evaluate every unit, rebuild the caches
    uv run python pipelines/reproduce.py --level train --yes          # + retrain every unit first (days of CPU)

Levels (each includes the ones below it):

    figures   needs the output zip. Simulates what the notebooks need and the zip lacks (the two-group runs and the
              shoals: output/two_targets/ and output/multi_fish/, skipped when present), then executes notebooks
              01-09 into output/reproduce/notebooks/; every figure script writes figures/out/.
    evaluate  re-evaluates every thesis unit (pipelines/evaluate.py; deterministic, so the saved summaries come back
              exactly: the old ones are kept in output/reproduce/backup/ and compared at the end), rebuilds the
              profile and turn-rule rollouts, the turn-rule references and the whole second-data-set pipeline.
    train     retrains every unit in place first (bases before the models made from them), and the screen-trained
              D4. This REPLACES the thesis checkpoints (needs --yes); everything computed from them is then redone.

Resumable: finished steps are recorded in output/reproduce/state.json and skipped when the command is run again
(--restart forgets them). A failed step stops the run; fix it and run the same command again.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import paths, registry  # noqa: E402

PY = [sys.executable, "-u", "-W", "ignore"]
STATE_DIR = paths.OUTPUT / "reproduce"
STATE = STATE_DIR / "state.json"
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0*.ipynb"))
LEVELS = ("figures", "evaluate", "train")
NOTEBOOK_MIN = {"01": 2, "02": 2, "03": 4, "04": 0.2, "05": 1, "06": 0.2, "07": 3, "08": 0.2, "09": 0.1}   # measured


def step(name, level, cmd, minutes=None, group=None, needed=None, note=""):
    """``minutes``: wall-clock estimate (None = not measured); ``group``: steps of one group may run in parallel
    (--jobs); ``needed``: a callable, False when the step's outputs are already there."""
    return dict(name=name, level=level, cmd=cmd, minutes=minutes, group=group, needed=needed, note=note)


def _train_minutes(u):
    r = registry.RECIPES.get(u["name"], {})
    if u["make"] == "correct":
        return 80
    if u["make"] in ("same", "driftfix"):
        return 1
    return 120 if r.get("drift") else (60 if r.get("latent") else 3)


def _ordered_units():
    """Every unit with its base first."""
    done, out = set(), []

    def visit(k):
        if k in done:
            return
        b = registry.UNITS[k]["base"]
        if b:
            visit(b)
        done.add(k); out.append(k)
    for k in registry.UNITS:
        visit(k)
    return out


def _missing_two_targets():
    from twin import two_targets as TT
    return any(not TT._path(j[0], TT.scene(j[1], j[2]), j[3], *j[4:]).exists() for j in TT.report_jobs())


def _missing_shoals():
    from twin import multi_fish as MF
    return (any(not (MF.OUT / f"{c}.pkl").exists() for c in list(MF.GRID_CONDS) + list(MF.PAIRDM_CONDS))
            or not (MF.OUT / "summary_grid.json").exists() or not (MF.OUT / "summary_pairdm.json").exists())


def steps():
    S = []
    for k in _ordered_units():                                         # ---- train
        u = registry.UNITS[k]
        S.append(step(f"train {u['name']} s{u['seed']} ({k})", "train",
                      PY + [str(ROOT / "pipelines" / "train.py"), u["name"], "--seed", str(u["seed"])],
                      _train_minutes(u)))
    S += [step("screen: bout table", "train", PY + ["-m", "twin.screen_train", "build"]),
          step("screen: train D4 and D4 + direction", "train", PY + ["-m", "twin.screen_train", "run"])]
    for k in registry.UNITS:                                           # ---- evaluate
        u = registry.UNITS[k]
        S.append(step(f"evaluate {u['name']} s{u['seed']} ({k})", "evaluate",
                      PY + [str(ROOT / "pipelines" / "evaluate.py"), "--keys", k], 15, group="evaluate"))
    S += [step("profile rollouts", "evaluate", PY + ["-m", "figures.fig_profiles", "--rebuild"]),
          step("turn-rule head gaps", "evaluate", PY + ["-m", "figures.fig_rule", "--rebuild"])]
    S += [step(f"turn rule, references: {s}", "evaluate", PY + ["-m", "twin.missing_pieces", s])
          for s in ("refs", "nearest", "laws_draw")]
    for args in (["twin.twin_test", "extract-meta"], ["twin.twin_test", "real"], ["twin.screen_eval", "real"],
                 ["twin.screen_eval", "sim"], ["twin.screen_eval", "analyse"], ["twin.screen_train", "sim"],
                 ["twin.screen_train", "analyse"], ["twin.screen_matched"],
                 ["twin.screen_report_extras", "importances"], ["twin.screen_report_extras", "closerange"],
                 ["twin.grouprep_fit", "fit", "14f"], ["twin.grouprep_fit", "test", "14f"],
                 ["twin.grouprep_fit", "fit", "32b"], ["twin.grouprep_fit", "test", "32b"]):
        S.append(step("screen: " + " ".join(args[0].split(".")[1:] + args[1:]), "evaluate", PY + ["-m"] + args))
    S += [step("two groups: the map runs (missing only)", "figures", PY + ["-m", "twin.two_targets", "run-report"],
               35, needed=_missing_two_targets, note="7 CPU-hours on 14 workers"),
          step("shoals: grid (missing only)", "figures", PY + ["-m", "twin.multi_fish", "--run-grid"], 210,
               needed=_missing_shoals, note="48 CPU-hours on 16 workers"),
          step("shoals: grid readout", "figures", PY + ["-m", "twin.multi_fish", "--analyse-grid"], None,
               needed=_missing_shoals),
          step("shoals: pairs with direction (missing only)", "figures", PY + ["-m", "twin.multi_fish", "--run-pairdm"],
               5, needed=_missing_shoals),
          step("shoals: pairs readout", "figures", PY + ["-m", "twin.multi_fish", "--analyse-pairdm"], None,
               needed=_missing_shoals)]
    out = STATE_DIR / "notebooks"
    for nb in NOTEBOOKS:                                               # ---- figures
        S.append(step(f"notebook {nb.stem}", "figures",
                      PY + ["-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb),
                            "--output-dir", str(out), "--ExecutePreprocessor.timeout=-1"],
                      NOTEBOOK_MIN.get(nb.stem[:2])))
    return S


def _load_state(restart):
    if restart or not STATE.exists():
        return {}
    return json.loads(STATE.read_text(encoding="utf-8"))


def _save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def _run(s):
    t0 = time.time()
    print(f"\n=== {s['name']}\n    {' '.join(s['cmd'][len(PY):])}", flush=True)
    rc = subprocess.call(s["cmd"], cwd=str(ROOT))
    return s, rc, (time.time() - t0) / 60


def _compare_summaries(backup):
    """The re-evaluated summaries against the ones saved before."""
    same, diff = 0, []
    for p in sorted(backup.glob("summary_M*.json")):
        q = paths.EVAL / p.name
        if not q.exists():
            diff.append((p.name, "missing after the run"))
            continue
        a, b = json.loads(p.read_text()), json.loads(q.read_text())
        bad = [k for k in a if k in b and a[k] != b[k] and not (a[k] != a[k] and b[k] != b[k])]   # NaN == NaN
        if bad:
            diff.append((p.name, ", ".join(bad[:5]) + ("..." if len(bad) > 5 else "")))
        else:
            same += 1
    print(f"\nre-evaluated summaries: {same} identical to the saved ones, {len(diff)} different")
    for n, why in diff:
        print(f"  {n}: {why}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=LEVELS, default="figures")
    ap.add_argument("--dry-run", action="store_true", help="list the steps and the time estimates, run nothing")
    ap.add_argument("--jobs", type=int, default=1, help="evaluations run in parallel (each uses about 4 cores)")
    ap.add_argument("--restart", action="store_true", help="forget the finished steps")
    ap.add_argument("--yes", action="store_true", help="confirm --level train (it replaces the thesis checkpoints)")
    a = ap.parse_args()
    upto = LEVELS.index(a.level)
    todo = [s for s in steps() if LEVELS.index(s["level"]) <= upto]
    state = _load_state(a.restart)
    plan = []
    for s in todo:
        if s["name"] in state:
            continue
        if s["needed"] is not None and not s["needed"]():
            continue
        plan.append(s)
    known = sum(s["minutes"] or 0 for s in plan if s["group"] != "evaluate")
    ev = [s for s in plan if s["group"] == "evaluate"]
    known += sum(s["minutes"] for s in ev) / max(a.jobs, 1)
    print(f"--level {a.level}: {len(plan)} steps to run ({len(todo) - len(plan)} done or not needed)")
    for s in plan:
        est = f"~{s['minutes']:g} min" if s["minutes"] else "not measured"
        print(f"  [{s['level']:8s}] {s['name']:60s} {est}" + (f"  ({s['note']})" if s["note"] else ""))
    print(f"measured part: about {known / 60:.1f} h" + (f" (evaluations {a.jobs} at a time)" if ev else ""))
    if a.dry_run or not plan:
        return
    if a.level == "train" and not a.yes:
        ap.error("--level train retrains every unit IN PLACE (the thesis checkpoints are replaced); add --yes")
    if upto >= LEVELS.index("evaluate"):
        backup = STATE_DIR / "backup" / "evaluation"
        if not backup.exists():
            backup.mkdir(parents=True)
            for p in paths.EVAL.glob("*.json"):
                shutil.copy2(p, backup / p.name)
            print(f"saved the current evaluations in {backup}")
    i = 0
    while i < len(plan):
        g = plan[i]["group"]
        batch = [plan[i]]
        while g and i + len(batch) < len(plan) and plan[i + len(batch)]["group"] == g:
            batch.append(plan[i + len(batch)])
        with ThreadPoolExecutor(a.jobs if g else 1) as ex:
            res = list(ex.map(_run, batch))
        for s, rc, mins in res:
            if rc:
                _save_state(state)
                sys.exit(f"FAILED: {s['name']} (exit {rc}); fix it and run the same command again")
            state[s["name"]] = dict(finished=time.strftime("%Y-%m-%d %H:%M"), minutes=round(mins, 1))
            print(f"--- done: {s['name']} ({mins:.1f} min)", flush=True)
        _save_state(state)
        i += len(batch)
    if upto >= LEVELS.index("evaluate"):
        _compare_summaries(STATE_DIR / "backup" / "evaluation")
    print(f"\nall steps done; executed notebooks in {STATE_DIR / 'notebooks'}, figures in {paths.FIGURES}")


if __name__ == "__main__":
    main()
