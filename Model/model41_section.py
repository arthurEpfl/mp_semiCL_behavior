"""Append (or refresh) the Model 41 section of Model_design_and_training.ipynb: the dot-jump phase input on M6, D4 and F.

Idempotent: every cell this script writes has an id starting with 'm41-'; a re-run removes those cells and appends the
section again at the end of the notebook. Nothing else in the notebook is touched. The comparison cells are executed
here (report_models' namespace) and saved with their outputs; the per-unit evaluation cells are the ones
report_models.py saved in report_outputs/M41a..f.json.

    python model41_section.py            # after RUN 41 (training, evaluation, F refit, jump_timing readouts)
"""
import json
import os
from pathlib import Path

import report_models as RM

HERE = Path(__file__).resolve().parent
NB = HERE / "Model_design_and_training.ipynb"
FINDINGS = HERE / "jump41" / "findings.md"          # the closing summary, written once the results are read


def md(text, cid):
    return dict(cell_type="markdown", id=cid, metadata={}, source=text.splitlines(True))


def code(src, cid, outs=()):
    return dict(cell_type="code", id=cid, execution_count=None, metadata={}, source=src.splitlines(True),
                outputs=list(outs))


INTRO = """## Model 41 — the dot-jump phase input: bout timing locked to the dots' jumps

**Why.** Moving dots do not glide: each dot jumps by about 2.1 mm every 18 frames (0.6 s) and is still in between. In the
real data, single-moving-dot blocks show that fish time their bouts to these jumps: onsets are suppressed 2–4 frames
(0.07–0.13 s) after a jump, at 0.71–0.76 of the uniform rate, and enriched 10–14 frames (0.33–0.47 s) after it, at
1.10–1.21 (5 388 bouts; Rayleigh z = 48; 9 of 15 fish individually). Every model so far sees the dots' motion only as a
flag, so it cannot produce this.

**The input.** `stim_jump` (`fit_MLP_border._jump_block`): at each bout onset, the frames since the last jump of any shown
dot (the block's first frame counts as a jump), as [sin, cos] of 2π·age/18, gated to moving-dot bouts (0 on pauses and
static dots). The jump timeline is identical for every fish (the semi-closed loop only moves a block's track rigidly; checked
frame by frame), so the phase is a pure function of the programme frame, like the block clock: one cached table
(`clean_data/dot_jump_age.npy`) serves training, the record-indexed and programme rollouts and the batched engine. With one
dot it is the exact 18-frame clock; with several dots the jumps interleave and the phase is short and irregular. Appended
LAST, so every archived layout is unchanged.

**Checks before training** (scratchpad `jump41/parity_jump.py`): archived 26e / 11e layouts unchanged (35 / 47 inputs);
training rows = simulator rows under forcing (max difference 0.0) on the M6 and D4 layouts; the record-anchored closed-loop
path's jump columns = the table at the provider clock (0.0); the batched engine = the builder (0.0); the programme the
simulator plays jumps on exactly the table's frames (0 of 72 000 frames differ); the table reproduces the real
measurement (5 388 bouts, same rates). Recipes: each new arm differs from its base only by `stim_jump`.

**The units** (seeds never pooled; each new unit pairs with its base at the same seed, same folds; adding two inputs
redraws the first layer's initialisation, so small differences can be retraining noise):

| unit | model | base |
|---|---|---|
| 41a / 41b | M6 + jump phase, seed 0 / 101 | 26e / 30d |
| 41c / 41d | D4 + jump phase (drift-gain search included) | 11e / 32a |
| 41e / 41f | F + jump phase = 41c / 41d + Model 14's correction (pf-B, 20 s cut-off, final step) | 14f / 32b |

**The new readout** (`scripts/jump_timing.py`): in single-moving-dot blocks of PROGRAMME rollouts (each simulated fish swims
the programme on its own clock, so its own intervals decide where its onsets fall relative to the jumps; 10 per fold model,
50 simulated fish), the onset rate by frames since the last jump, per fish relative to its mean; suppression = mean rate at
2–4 frames, enrichment = at 10–14, modulation = enrichment − suppression; phase locking R and Rayleigh z on the 18-frame
cycle."""

COMPARE = r'''# Model 41: each new unit next to its base at the same seed (archived key numbers of every unit). Likelihood in bits
# per bout and AUC: lower = better; dip / latency / grading / attraction: closer to the real fish (printed below).
import os, json, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
PAIRS41 = [("M6", 0, "26e", "41a"), ("M6", 101, "30d", "41b"), ("D4", 0, "11e", "41c"), ("D4", 101, "32a", "41d"),
           ("F", 0, "14f", "41e"), ("F", 101, "32b", "41f")]
COLS41 = ["lik_turn", "lik_displacement", "lik_interval", "lik_joint", "auc_matched", "auc_std", "auc_matched_norad",
          "dip", "lat", "grading", "mv_err_sem", "st_err_sem", "attr_N1", "attr_N2", "attr_N8"]
def _s41(k):
    p = f"report_outputs/summary_M{k}.json"
    return json.load(open(p)) if os.path.exists(p) else {}
rows41 = []
for name, seed, kb, kn in PAIRS41:
    b, n = _s41(kb), _s41(kn)
    rows41.append({"model": f"{name} s{seed}", "unit": f"base {kb}", **{c: b.get(c, np.nan) for c in COLS41}})
    rows41.append({"model": f"{name} s{seed}", "unit": f"+jump {kn}", **{c: n.get(c, np.nan) for c in COLS41}})
    rows41.append({"model": f"{name} s{seed}", "unit": "change",
                   **{c: n.get(c, np.nan) - b.get(c, np.nan) if (c in n and c in b and n[c] is not None and b[c] is not None)
                      else np.nan for c in COLS41}})
cmp41 = pd.DataFrame(rows41).set_index(["model", "unit"]).astype(float)
print("real fish: dip 4.23 mm at 13.5 s, grading 6.00 mm, N=2 attraction 6.19 mm (matched profile, attraction_table)")
print(cmp41.round(3).to_string())'''

PAIRED = r'''# Model 41: held-out likelihood per head, paired by fish (new - base, bits per bout; 95 % bootstrap interval over the 15
# held-out fish). The jump phase should help the INTERVAL head if anything: it tells the model when the next bout is due.
import numpy as np, pandas as pd
def _lik41(key):
    d = f"report_ckpts/M{key}"
    l = pd.read_csv(f"{d}/likA_model1.csv").drop_duplicates(["arm", "fish"])
    arm = json.load(open(f"{d}/manifest_model1.json"))[0]["arm"]
    return l[l.arm == arm].set_index("fish")
def _ci41(v, n=4000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)
out41 = []
for name, seed, kb, kn in PAIRS41:
    try:
        b, n = _lik41(kb), _lik41(kn)
    except FileNotFoundError:
        continue
    for col, head in (("bits_per_bout", "turn"), ("bits_per_bout_dist", "step"), ("bits_per_bout_dt", "interval"),
                      ("bits_per_bout_joint", "joint")):
        if col not in b or col not in n:
            continue
        d = (n[col] - b[col]).dropna()
        lo, hi = _ci41(d)
        out41.append(dict(model=f"{name} s{seed}", head=head, diff=d.mean(), lo=lo, hi=hi,
                          fish_better=int((d < 0).sum()), n=len(d)))
lik41 = pd.DataFrame(out41)
print(lik41.round(4).to_string(index=False))'''

TIMING = r'''# Model 41: bout timing locked to the dots' jumps (single moving dot, programme rollouts). Per fish: onset rate by frames
# since the last jump relative to that fish's mean (1 = uniform); suppression (2-4 frames), enrichment (10-14 frames),
# modulation = enrichment - suppression; phase locking z (Rayleigh, 18-frame cycle). Real fish: 15; models: 50 simulated
# fish per unit. Code: scripts/jump_timing.py, results in Model/jump41/.
import pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
J41 = {}
for lab in ["real", "M6 s0", "M6+J s0", "D4 s0", "D4+J s0", "F s0", "F+J s0",
            "M6 s101", "M6+J s101", "D4 s101", "D4+J s101", "F s101", "F+J s101"]:
    p = f"jump41/{lab}.pkl"
    if os.path.exists(p):
        J41[lab] = pickle.load(open(p, "rb"))
rows = []
for lab, r in J41.items():
    per, pooled = r["per"], r["pooled"]
    lo, hi = np.percentile(np.random.default_rng(0).choice(per.modulation.to_numpy(), size=(4000, len(per))).mean(1), [2.5, 97.5])
    rows.append(dict(source=lab, fish=len(per), bouts=pooled["n"], suppression=pooled["suppression"],
                     enrichment=pooled["enrichment"], modulation=pooled["modulation"],
                     modulation_per_fish=per.modulation.mean(), ci_lo=lo, ci_hi=hi, z_pooled=pooled["z"],
                     fish_z_gt_3=int((per.z > 3).sum())))
timing41 = pd.DataFrame(rows).set_index("source")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
ph = np.arange(18) / 30.0
real = np.vstack(J41["real"]["per"].rate.to_numpy()) if "real" in J41 else None
for ax, base in zip(axes, ("M6", "D4", "F")):
    if real is not None:
        m = real.mean(0); s = real.std(0, ddof=1) / np.sqrt(len(real))
        ax.fill_between(ph, m - 1.96 * s, m + 1.96 * s, color="0.8", label="real fish (95 % CI)")
        ax.plot(ph, m, color="k", lw=1.5)
    for lab, c, ls in ((f"{base} s0", "tab:blue", "-"), (f"{base} s101", "tab:blue", "--"),
                       (f"{base}+J s0", "tab:red", "-"), (f"{base}+J s101", "tab:red", "--")):
        if lab in J41:
            ax.plot(ph, np.vstack(J41[lab]["per"].rate.to_numpy()).mean(0), color=c, ls=ls, lw=1.2, label=lab)
    ax.axhline(1, color="0.5", lw=0.6); ax.set_xlabel("time since the dot's last jump (s)"); ax.set_title(base)
    ax.legend(fontsize=7, frameon=False)
axes[0].set_ylabel("bout onset rate / mean")
plt.tight_layout(); plt.show()
print(timing41.round(3).to_string())'''


def main():
    os.chdir(HERE)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    keep = [c for c in nb["cells"] if not str(c.get("id", "")).startswith("m41-")]
    ns = {"__name__": "__cell__"}
    exec(compile(RM.SETUP, "<setup>", "exec"), ns)
    cells = [md(INTRO, "m41-intro")]
    cells.append(md("### Model 41 at a glance: each new unit against its base", "m41-cmp-md"))
    cells.append(code(COMPARE, "m41-cmp", RM._run_cell(ns, COMPARE)[0]))
    cells.append(code(PAIRED, "m41-paired", RM._run_cell(ns, PAIRED)[0]))
    cells.append(md("### The target: bout timing locked to the dots' jumps", "m41-timing-md"))
    cells.append(code(TIMING, "m41-timing", RM._run_cell(ns, TIMING)[0]))
    if FINDINGS.exists():
        cells.append(md(FINDINGS.read_text(encoding="utf-8"), "m41-findings"))
    for u in RM._units_of(next(m for m in RM.MODELS if m["k"] == 41)):
        cells.append(md(f"### {u['sub']}", f"m41-{u['key']}-sub"))
        p = RM.OUTS / f"M{u['key']}.json"
        if not p.exists():
            cells.append(md("*(not evaluated yet)*", f"m41-{u['key']}-pending"))
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        cells += [code(c["source"], f"m41-{u['key']}-{c['cid']}", c["outputs"]) for c in blob["cells"]]
    nb["cells"] = keep + cells
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {NB.name}: {len(keep)} existing + {len(cells)} Model 41 cells")


if __name__ == "__main__":
    main()
