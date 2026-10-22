"""Model 41: bout timing locked to the dots' jumps, in real fish and in simulated ones.

Moving dots jump about 2.1 mm every 18 frames (0.6 s); real fish start fewer bouts just after a jump and more about
0.33-0.47 s after it (thesis Ch. 3). The jump timeline is the same for every fish (fit_MLP_border.jump_age_table), so a
bout's phase is the number of frames since the last jump at its onset frame. The readout uses single-moving-dot bouts
only, where one dot jumps on a clean 18-frame clock.

Simulated bouts come from PROGRAMME rollouts (model_figures.programme_bouts): each simulated fish swims the programme on
its own clock, so its own intervals decide where its onsets fall relative to the jumps. Record-indexed rollouts cannot
show this (their bout times are the real fish's).

Readouts, per fish (real fish or simulated fish) and pooled:
  rate[k]        onset count at phase k (0..17 frames since the last jump) / mean count over phases (1 = uniform)
  suppression    mean rate over phases 2-4 (0.07-0.13 s after a jump)
  enrichment     mean rate over phases 10-14 (0.33-0.47 s)
  modulation     enrichment - suppression (0 = no locking)
  R, z           phase-locking vector length on the 18-frame cycle and its Rayleigh z (n R^2)

    python jump_timing.py real                          # the real fish
    python jump_timing.py unit <ckpt dir> <arm> <label>  # one model unit, 10 programme rollouts per fold
"""
from __future__ import annotations

import contextlib
import io
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "Model"
OUT = MODEL / "jump41"
for _p in (str(HERE), str(MODEL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PERIOD = 18
SUPP = (2, 3, 4)
ENRICH = (10, 11, 12, 13, 14)
REJ = {"d0": 2.0, "d1": 10.0, "max_draws": 20}
MIN_BOUTS = 30


def phases(frames):
    import fit_MLP_border as MB
    a = MB.jump_age_at(np.asarray(frames, float))
    return a[a < PERIOD].astype(int)


def _fish_row(ph):
    c = np.bincount(ph, minlength=PERIOD)[:PERIOD].astype(float)
    rate = c / c.mean() if c.mean() > 0 else np.full(PERIOD, np.nan)
    a = 2 * np.pi * ph / PERIOD
    R = float(np.hypot(np.cos(a).mean(), np.sin(a).mean())) if len(ph) else np.nan
    return dict(n=len(ph), rate=rate, suppression=float(rate[list(SUPP)].mean()),
                enrichment=float(rate[list(ENRICH)].mean()),
                modulation=float(rate[list(ENRICH)].mean() - rate[list(SUPP)].mean()),
                R=R, z=float(len(ph) * R * R))


def summarise(table, label):
    """table: one row per bout with fish_id and frame_onset (single moving dot only). Per-fish rows + pooled row."""
    rows = []
    for f, g in table.groupby("fish_id"):
        ph = phases(g.frame_onset)
        if len(ph) >= MIN_BOUTS:
            rows.append(dict(source=label, fish=f, **_fish_row(ph)))
    per = pd.DataFrame(rows)
    pooled = _fish_row(phases(table.frame_onset))
    pooled.update(source=label, fish="pooled")
    return per, pooled


def real_table():
    b = pd.read_csv(ROOT / "clean_data" / "bouts_clean.csv",
                    usecols=["fish_id", "frame_onset", "stimulus", "moving", "active"])
    b = b[b.active.astype(bool) & b.stimulus.astype(str).str.startswith("CLsemi")]
    n = b.stimulus.str.extract(r"_g(\d+)-(\d+)").astype(float).sum(axis=1)
    return b[(n == 1) & b.moving.astype(bool)][["fish_id", "frame_onset"]]


def _unit_fold(job):
    d, arm, k, r_per = job
    d = str(Path(d) if Path(d).is_absolute() else MODEL / d)
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    import model_figures as MF
    with contextlib.redirect_stdout(io.StringIO()):
        models, _ = MF.fold_models(d, arm, REJ)
        m = models[sorted(models)[k]]
        MF.BLOCK_SEED = True
        g = MF.programme_bouts(m, k, r_per, seed=0)
    one = g[(g.n_dots == 1) & g.moving.astype(bool) & g.is_stim.astype(bool)]
    return one[["fish_id", "frame_onset"]], int(g.attrs.get("runaway", 0)), len(g)


def unit_table(d, arm, r_per=10, jobs=5):
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(_unit_fold, [(d, arm, k, r_per) for k in range(5)]))
    t = pd.concat([r[0] for r in res], ignore_index=True)
    t.attrs.update(runaway=sum(r[1] for r in res), n_bouts=sum(r[2] for r in res))
    return t


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.pkl", "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(name):
    with open(OUT / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def main(argv):
    if argv[0] == "real":
        per, pooled = summarise(real_table(), "real")
        save("real", dict(per=per, pooled=pooled))
        print(f"real: {pooled['n']} bouts, modulation {pooled['modulation']:+.3f}, z {pooled['z']:.1f}, "
              f"fish with z > 3: {int((per.z > 3).sum())}/{len(per)}")
    elif argv[0] == "unit":
        d, arm, label = argv[1], argv[2], argv[3]
        t = unit_table(d, arm)
        per, pooled = summarise(t, label)
        save(label, dict(per=per, pooled=pooled, runaway=t.attrs["runaway"], n_bouts=t.attrs["n_bouts"]))
        print(f"{label}: {pooled['n']} single-dot bouts from {len(per)} simulated fish, modulation "
              f"{pooled['modulation']:+.3f}, z {pooled['z']:.1f}; runaway {t.attrs['runaway']}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
