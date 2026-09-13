"""Which (batch, lr, chunk) trains the recurrent model FASTEST PER UNIT WALL TIME?

WHY THIS EXISTS
---------------
The GRU fit is latency-bound, not compute-bound: ``state_chain`` walks one timestep at a time and
costs ~2.6 ms/timestep almost regardless of batch size, so an A100 sits near idle. The tempting
conclusion is "raise the batch, it's free" -- and that is WRONG in the way that matters. The step
count is ``len(tr) / (chunk * batch)``, so a bigger batch buys fewer, larger updates. Measured on
CPU at 4 epochs:

    B=16 chunk=64 -> 7.2 s/epoch, val 9.94
    B=48 chunk=64 -> 5.1 s/epoch, val 14.77      faster per epoch, clearly WORSE trained

So batch alone trades updates for speed. The thing that might genuinely win is batch AND learning
rate together (standard batch/LR scaling), and whether it does is an empirical question about THIS
model on THIS hardware -- which is what this script answers.

WHAT IT MEASURES, AND WHY IT IS VAL-VS-TIME
-------------------------------------------
Comparing configs at equal EPOCHS is meaningless when an epoch means different amounts of work.
Comparing at equal WALL TIME is the question you actually have. Each fit already records a
per-epoch validation NLL, so one run per config gives the whole val-vs-time curve for free; the
table then reports each config's val at a COMMON budget (the shortest total across configs), which
is an apples-to-apples "best model per minute".

    python scripts/time_state_fit.py --epochs 8 --fish 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                     # noqa: E402
import fit_MLP_border as MB              # noqa: E402
import fit_MLP_fishre as FR              # noqa: E402

# (seq_batch, lr, chunk). The lr column is the point: batch without lr is the trap above.
GRID = [
    (16, 1e-3, 64),      # the measured-sound baseline
    (64, 1e-3, 64),      # batch alone -- expected faster/epoch but under-trained
    (64, 2e-3, 64),      # batch + sqrt-ish lr scaling
    (256, 4e-3, 64),     # big batch + linear-ish lr scaling
    (64, 2e-3, 128),     # longer gradient horizon at the same batch, for the acf question
]


def run(records, ctx, seq_batch, lr, chunk, epochs, hidden, state, seed=0):
    m = FR.MLPMixJointREDensity(
        K=8, K_kin=3, stim_mode="shoal", hidden=hidden, epochs=epochs, n_dist=2, n_dt=3,
        re_m=2, re_ghq_k=3, re_gain_search=False, dt_snap=False, state=state,
        contain=False, kin_fb_calib=False, psi_ref="travel", use_border=True,
        border_mode="basis", re_zero=True, seq_batch=seq_batch, state_chunk=chunk,
        lr=lr, seed=seed)
    m.name = f"B{seq_batch}_lr{lr:g}_c{chunk}"
    t0 = time.time()
    m.fit(records, ctx)
    el = time.time() - t0
    hist = [float(h) for h in m.history]
    return el, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=str(Path(__file__).parent.parent / "data" /
                                           "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--fish", type=int, default=12, help="match a ladder fold's TRAIN size")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--state", type=int, default=64)
    a = ap.parse_args()

    recs = MB.build_border_records(pd.read_csv(a.bouts))
    tr = [recs[f] for f in sorted(recs)[:a.fish]]
    ctx = E.make_context(tr, dict(mlp_epochs=a.epochs))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}   {len(tr)} fish, {sum(int(r.n) for r in tr)} bouts   "
          f"{a.epochs} epochs/config, hidden {a.hidden}, state {a.state}\n")

    rows = []
    for sb, lr, ch in GRID:
        el, hist = run(tr, ctx, sb, lr, ch, a.epochs, a.hidden, a.state)
        rows.append(dict(sb=sb, lr=lr, ch=ch, total=el, spe=el / max(len(hist), 1), hist=hist))
        print(f"  B={sb:<4d} lr={lr:<7g} chunk={ch:<4d}  {el / max(len(hist), 1):6.1f} s/epoch  "
              f"{el:7.1f} s total  val {hist[-1]:8.3f}   curve {[round(x, 2) for x in hist]}",
              flush=True)

    # val at a COMMON wall-clock budget -- the only comparison that answers "best model per minute"
    budget = min(r["total"] for r in rows)
    print(f"\nvalidation NLL at a COMMON budget of {budget:.0f} s (lower is better):")
    print(f"  {'config':<26}{'s/epoch':>9}{'epochs in budget':>19}{'val @ budget':>14}")
    best = None
    for r in rows:
        k = max(1, int(budget // max(r["spe"], 1e-9)))
        v = r["hist"][min(k, len(r["hist"])) - 1]
        tag = f"B={r['sb']} lr={r['lr']:g} chunk={r['ch']}"
        print(f"  {tag:<26}{r['spe']:>9.1f}{k:>19d}{v:>14.3f}")
        if best is None or v < best[1]:
            best = (tag, v)
    print(f"\nbest per unit time: {best[0]}  (val {best[1]:.3f})")
    print("NOTE a config that wins here but whose curve is still falling steeply may win by more "
          "with a longer run -- read the curves, not only the endpoint.")


if __name__ == "__main__":
    main()
