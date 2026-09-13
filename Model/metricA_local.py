"""Metric A from checkpoints already on disk -- seconds, no Colab re-run needed.

    python metricA_local.py --ckpt-dir <where you extracted model1_checkpoints.zip>

`score()` returns PER-BOUT log-density arrays; they are aggregated here as
`-mean(logdens) / ln 2`, the same convention as evaluate, so these sit on the same scale as
every archived bits_per_bout. `bits_per_bout` is the TURN term alone (the historical reported
number); `bits_per_bout_joint` sums all three channels, which is what the model optimises.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT if (OUT / "scripts").exists() else OUT.parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as E, fit_MLP_border as MB, fit_MLP_mix as MX      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", type=str, default=str(OUT))
ap.add_argument("--out", type=str, default="likA_model1.csv")
a = ap.parse_args()
d = Path(a.ckpt_dir)

recs = MB.build_border_records(E.load_bouts())
rows = []
for e in json.loads((d / "manifest_model1.json").read_text()):
    if e.get("insample_only") or not e["heldout"]:
        continue                                   # all-data model has no held-out fish
    m = MX.load_mix(str(d / e["file"]))
    for f in e["heldout"]:
        sc = m.score(recs[f])
        r = dict(rep=e["rep"], fold=e["fold"], fish=int(f), n_bouts=int(recs[f].n))
        for k, nm in (("logdens", "bits_per_bout"), ("logdens_dist", "bits_per_bout_dist"),
                      ("logdens_dt", "bits_per_bout_dt")):
            v = np.asarray(sc.get(k, np.nan), float)
            r[nm] = float(-np.nanmean(v) / np.log(2.0)) if v.size else np.nan
        r["bits_per_bout_joint"] = float(np.nansum([r["bits_per_bout"], r["bits_per_bout_dist"],
                                                    r["bits_per_bout_dt"]]))
        rows.append(r)
p = pd.DataFrame(rows)
p.to_csv(d / a.out, index=False)
cols = ["bits_per_bout", "bits_per_bout_dist", "bits_per_bout_dt", "bits_per_bout_joint"]
print(f"{len(p)} held-out (fold, fish) scores -> {a.out}\n")
print("Metric A, held-out bits per bout (lower = better):")
print(p.groupby("rep")[cols].mean().round(4).to_string())
