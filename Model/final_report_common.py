"""Shared pieces of the three final-report notebook builders (final_report_models.py, final_report_ablations.py,
final_report_extrapolations.py). Every model block is copied, outputs included, from the saved results of
report_models.py (report_outputs/M<key>.json): nothing is re-run. Summary tables are computed from
report_outputs/summary_M<key>.json when a notebook is built. A missing result is shown as XX."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "scripts"))
import report_models as RM                       # noqa: E402  (SETUP, _run_cell, the saved outputs' location)

OUTS = RM.OUTS
S0, S101 = "seed 0", "seed 101"

SEED_NOTE = """**How to read it.**
- **Seeds.** Seed 0 and seed 101 are different cross-validation splits. Models are compared only on the same seed; the
  second seed is a second paired comparison, never pooled. Differences under ~0.05 AUC at one seed are within the seed
  noise (~0.03 per seed for these recipes); a claim needs the same direction at both seeds.
- **Headline AUC = matched** (generated windows cut at the real fish's own window positions, 1:1); 0.5 = indistinguishable,
  real against real ~0.52. Standard and no-radial readings alongside.
- **Profiles** (dip, grading, moving / static error, attraction) are from programme rollouts on per-block random streams.
  Profile numbers carry rollout noise: grading ±1 mm, attraction ±1–2 mm, static error ±0.3.
- Real fish: dip 4.23 mm at 13.5 s, plateau 17.26 mm, 1-dot plateau 22.26 mm, grading 6.00 mm, attraction
  1.16 / 6.19 / 5.19 / 6.43 / 6.94 mm (1 / 2 / 4 / 6 / 8 dots).
- **XX** = a result that does not exist yet."""

LINKS = """The three final-report notebooks: `Final_report_models.ipynb` (the model sequence), `Final_report_ablations.ipynb`
(the ablation study on the analysis model 26e), `Final_report_extrapolations.ipynb` (two dot groups, several fish). The full
development history stays in `Report_models.ipynb`."""


def md(text, cid):
    return dict(cell_type="markdown", id=cid, metadata={}, source=text.splitlines(True))


def code(src, cid, outs=()):
    return dict(cell_type="code", id=cid, execution_count=None, metadata={}, source=src.splitlines(True),
                outputs=list(outs))


def new_ns():
    os.chdir(HERE)
    ns = {"__name__": "__cell__"}
    exec(compile(RM.SETUP, "<setup>", "exec"), ns)
    return ns


def run_df(ns, src):
    """Run a cell whose last line is a DataFrame expression; embed it as text (the notebook shows it as a table when
    re-run)."""
    lines = src.rstrip().splitlines()
    body, last = "\n".join(lines[:-1]), lines[-1]
    outs, _ = RM._run_cell(ns, body + f"\n_df_ = {last}\nprint(_df_.to_string())")
    return outs


def run_cell(ns, src):
    return RM._run_cell(ns, src)[0]


def story_cells(story):
    """Markdown + the saved cells of every unit in ``story`` (sections -> models -> units); returns (cells, n_units,
    missing keys). A unit key of None, or a key without saved outputs, becomes an XX placeholder."""
    cells, n_units, missing = [], 0, []
    for sec in story:
        cells.append(md(f'<a id="{sec["id"]}"></a>\n## {sec["title"]}\n\n{sec["md"]}', f'{sec["id"]}-md'))
        for m in sec["models"]:
            lvl = m.get("level", 3)
            h, hh = "#" * lvl, "#" * (lvl + 1)
            cells.append(md(f'<a id="{m["anchor"]}"></a>\n{h} {m["name"]}\n\n**Change:** {m["change"]}\n\n{m["md"]}',
                            f'{m["anchor"]}-md'))
            for key, seed in m.get("units", []):
                p = OUTS / f"M{key}.json" if key else None
                if p is None or not p.exists():
                    cells.append(md(f"{hh} {m['name']} — {seed}\n\nXX — not run yet.",
                                    f'{m["anchor"]}-{seed.replace(" ", "")}-xx'))
                    missing.append(key or m["anchor"])
                    continue
                blob = json.loads(p.read_text(encoding="utf-8"))
                cells.append(md(f"{hh} {m['name']} — {seed} (key `{key}`)", f"u{key}-sub"))
                cells += [code(c["source"], f"u{key}-{c['cid']}", c["outputs"]) for c in blob["cells"]]
                n_units += 1
            for seed, why in m.get("missing", []):
                cells.append(md(f"{hh} {m['name']} — {seed}\n\nXX — {why}", f'{m["anchor"]}-{seed.replace(" ", "")}-xx'))
    return cells, n_units, missing


TAB_COMMON = '''import json, os
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)

def _s(key):
    """Key numbers of one notebook unit (None if it does not exist yet)."""
    p = f"report_outputs/summary_M{key}.json" if key else None
    return json.load(open(p)) if p and os.path.exists(p) else None
'''


def models_table_src(rows, title):
    return TAB_COMMON + '''
ROWS = %r
COLS = [("likelihood (bits/bout)", "lik_joint", 3), ("matched AUC", "auc_matched", 3), ("std AUC", "auc_std", 3),
        ("no-radial AUC", "auc_matched_norad", 3), ("dip (mm)", "dip", 2), ("dip latency (s)", "lat", 1),
        ("grading (mm)", "grading", 2), ("moving err (SEM)", "mv_err_sem", 2), ("static err (SEM)", "st_err_sem", 2),
        ("2-dot attraction", "attr_N2", 2), ("8-dot attraction", "attr_N8", 2)]

def _cell(key, c, nd):
    s = _s(key)
    return "XX" if s is None or s.get(c) is None else f"{s[c]:.{nd}f}"

T = pd.DataFrame([{lab: f"{_cell(k0, c, nd)} | {_cell(k1, c, nd)}" for lab, c, nd in COLS}
                  for _, k0, k1 in ROWS], index=[r[0] for r in ROWS])
T.insert(0, "keys (s0 | s101)", [f"{k0 or 'XX'} | {k1 or 'XX'}" for _, k0, k1 in ROWS])
print(%r)
print("Real fish: dip 4.23 @ 13.5 s, grading 6.00, attraction 2 dots 6.19 / 8 dots 6.94; real-vs-real AUC ~0.52.")
T''' % (rows, title)


def steps_table_src(steps, title):
    return TAB_COMMON + '''
STEPS = %r
D = [("d likelihood", "lik_joint", 3), ("d matched AUC", "auc_matched", 3), ("d std AUC", "auc_std", 3),
     ("d no-radial AUC", "auc_matched_norad", 3), ("d dip", "dip", 2), ("d grading", "grading", 2),
     ("d moving err", "mv_err_sem", 2), ("d static err", "st_err_sem", 2)]

def _d(pair, c):
    if pair is None:
        return None
    a, b = _s(pair[0]), _s(pair[1])
    if a is None or b is None or a.get(c) is None or b.get(c) is None:
        return None
    return b[c] - a[c]

rows = []
for lab, p0, p1 in STEPS:
    r = {"step": lab}
    for name, c, nd in D:
        d0, d1 = _d(p0, c), _d(p1, c)
        f = lambda v: "XX" if v is None else f"{v:+.{nd}f}"
        r[name] = f"{f(d0)} | {f(d1)}"
    m0, m1 = _d(p0, "auc_matched"), _d(p1, "auc_matched")
    r["matched: same sign at both seeds?"] = ("not run" if (m0 is None and m1 is None) else
                                             "one seed" if (m0 is None or m1 is None) else
                                             ("yes" if np.sign(m0) == np.sign(m1) else "NO (seed-dependent)"))
    rows.append(r)
print(%r)
pd.DataFrame(rows).set_index("step")''' % (steps, title)


def write_nb(path, cells, n_units=0, missing=()):
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                         language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.name}: {len(cells)} cells, {n_units} model units, {path.stat().st_size / 1e6:.1f} MB"
          + (f"; XX placeholders: {len(missing)}" if missing else ""), flush=True)
