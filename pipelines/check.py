"""One-minute check that the installation works: the data, a trained model and the code give back stored numbers.

    uv run python pipelines/check.py

1. the training table loads (data zip: data/);
2. F (seed 0) loads and scores the held-out fish of its first fold: the bits per bout its training run wrote
   (output/checkpoints/M14f/likA_model1.csv) come back;
3. F swims the rig's programme (175 min) as simulated fish 0: its bouts are the stored ones (the rollouts behind
   the distance profiles, output/figures_cache/).
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin import paths  # noqa: E402

KEY = "14f"                  # F, seed 0


def ok(msg):
    print(f"  ok    {msg}", flush=True)


def fail(msg, hint):
    print(f"  FAIL  {msg}\n        {hint}", flush=True)
    sys.exit(1)


def main():
    t0 = time.time()
    print("1. data", flush=True)
    if not paths.BOUTS.exists():
        fail(f"no {paths.BOUTS}", "unzip twin_data.zip into the repository root (data/README.md)")
    from twin import training
    recs = training.records()
    ok(f"training table: {sum(r.n for r in recs.values()):,} bouts of {len(recs)} fish")

    print("2. a trained model", flush=True)
    import pandas as pd

    from twin import model_figures as MF
    from twin import registry
    d = registry.ckpt_dir(KEY)
    if not (d / "manifest_model1.json").exists():
        fail(f"no checkpoints in {d}", "unzip twin_output.zip into the repository root")
    with contextlib.redirect_stdout(io.StringIO()):
        models, held = MF.fold_models(d, registry.arm(KEY))
    first = sorted(models)[0]
    fish = sorted(f for f, m in held.items() if m == first)
    L = pd.read_csv(d / "likA_model1.csv")
    L = L[L.arm == registry.arm(KEY)].set_index("fish")
    for f in fish:
        got = training.metricA(models[first], recs[f])["bits_per_bout"]
        if abs(got - L.loc[f, "bits_per_bout"]) > 1e-6:
            fail(f"fish {f}: {got:.9f} bits per bout, the training run wrote {L.loc[f, 'bits_per_bout']:.9f}",
                 "the checkpoints, the data or the libraries differ from the ones that made them")
    ok(f"F fold 0 scores its held-out fish {fish} as its training run did "
       f"({', '.join(f'{L.loc[f, 'bits_per_bout']:.4f}' for f in fish)} bits per bout)")

    print("3. a simulation", flush=True)
    cache = paths.FIG_CACHE / "cache" / f"fig_profiles_gen_{KEY}.pkl"
    if not cache.exists():
        fail(f"no {cache}", "unzip twin_output.zip into the repository root")
    MF.BLOCK_SEED = True                                  # the thesis units' per-block random streams
    with contextlib.redirect_stdout(io.StringIO()):
        raw = MF.programme_bouts(models[first], 0, 1, seed=0)
    old = pd.read_pickle(cache)
    old = old[old.fish_id == 0]
    new = raw[raw.frame_onset.isin(old.frame_onset)]      # the stored table keeps the stimulus bouts the profile reads
    if len(old) == 0 or len(new) != len(old) or not np.allclose(new[["gx", "gy", "dth"]].to_numpy(float),
                                                                old[["gx", "gy", "dth"]].to_numpy(float)):
        fail("the simulated bouts differ from the stored ones",
             "the checkpoints, the code or the libraries differ from the ones that made them")
    ok(f"simulated fish 0: {len(raw):,} bouts over the programme, its {len(old):,} stored stimulus bouts identical")
    print(f"\nall checks passed in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
