"""Make the checkpoints of a thesis model at one seed (see twin/registry.py for every name).

    uv run python pipelines/train.py M6 --seed 0            # train the five folds of M6 at seed 0 (unit 26e)
    uv run python pipelines/train.py M6 --seed 0 --fold 2   # one fold only
    uv run python pipelines/train.py F --seed 101           # the correction on D4 at seed 101 (needs D4's unit)
    uv run python pipelines/train.py --list                 # every model, its keys and how it is made

A model trained from scratch takes 1-3 min per fold without the latent (M1), 10-60 min with it and 1-2 h with the
drift (D2, D4), on one CPU thread per fold; the five folds run in parallel. The correction takes 45-80 min per fold.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="thesis model name, e.g. M6, D4, F, 'M6-turn4' (quote names with spaces)")
    ap.add_argument("--seed", type=int, choices=(0, 101), default=0)
    ap.add_argument("--fold", type=int, default=None, help="train one fold only (0-4)")
    ap.add_argument("--epochs", type=int, default=None, help="at most this many epochs (default 80)")
    ap.add_argument("--jobs", type=int, default=5, help="folds trained in parallel")
    ap.add_argument("--out", default=None, help="unit folder (default: output/checkpoints/M<key>)")
    ap.add_argument("--list", action="store_true", help="list every model and exit")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from twin import registry, training
    if a.list or not a.name:
        for k, u in registry.UNITS.items():
            print(f"{u['name']:26s} seed {u['seed']:<3d} key {k:4s} {u['make']:8s} {registry.arm(u)}"
                  + (f"  (from {u['base']})" if u["base"] else ""))
        return
    u = registry.unit(a.name, a.seed)
    fit = {} if a.epochs is None else dict(epochs=a.epochs)
    if a.fold is not None or a.out:
        training.train_unit(a.name, a.seed, folds=[a.fold] if a.fold is not None else range(5), jobs=a.jobs,
                            out_dir=a.out, **fit)
    else:
        training.make_unit(a.name, a.seed, jobs=a.jobs, **fit)
    print(f"{a.name} seed {a.seed}: unit {u['key']} -> {registry.ckpt_dir(u)}")


if __name__ == "__main__":      # required on Windows: the fold workers re-import this module
    main()
