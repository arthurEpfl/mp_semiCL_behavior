"""Train a thesis model: five folds of cross-validation by fish, each fold a separate fit.

A model's recipe is in ``registry.RECIPES``; ``train_unit("M6", 0)`` trains the five fold models of M6 at seed 0 into
``paths.CKPTS / "M26e"``, with the manifest (held-out fish, epochs), the held-out likelihood per fish
(``likA_model1.csv``) and the training curves. Fold k holds out three fish (``make_folds``) and is initialised from
seed + k, so every model at one seed sees the same folds and initialisations: two models differ only by what was
changed between them. Each fold runs in its own process on one thread; training is deterministic given the seed.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from twin import evaluate as E
from twin import fit_MLP_border as MB
from twin import fit_MLP_fishre as FR
from twin import fit_MLP_joint as JT
from twin import fit_MLP_mix as MX
from twin import registry

# stimulus codes (registry recipe ``rep``) -> the feature builder's stimulus mode (fit_MLP_border)
REPS = {"S0": "shoal_simple",   # 6  presence, motion, log count, log range, bearing sin / cos
        "S4i": "shoal_int",     # 9  + count x sin, moving x sin, moving x cos
        "S4x": "shoal_rx",      # 14 + range x bearing per condition, linear in log range
        "S4": "shoal",          # 18 the range bumps: bearing terms on four Gaussian bumps in log range
        "Snull": "shoal_null",  # 0  no stimulus columns
        "S4xnn": "shoal_rx_nn",  # 11 S4x without the dot count
        "S4xnr": "shoal_rx_nr",  # 8  S4x without the range
        "S4xnm": "shoal_rx_nm",  # 7  S4x without the moving indicator
        "S4xlo": "shoal_rx_rzlo",  # 14 S4x with the log range clipped at the 2 px floor instead of 17.5 px
        "S4lo": "shoal_rzlo"}      # 18 S4, the same lower clip

# use_kin is forced True by the joint class. clock_mode / clock_tau_s / lag_valid are cfg_feat keys with no
# constructor argument, so they ride in via cfg_feat_extra and are saved with the checkpoint.
BASE = dict(K=8, K_kin=3, hidden=128, use_border=True, border_mode="raw",
            n_dist=2, n_dt=3,
            phi_ref="travel", psi_ref="travel",
            closed_loop=True, guardrail=True, stim_loop=True,
            contain=False, vigor=False, indiv=False, K_stim=1, noise_aug=None,
            cfg_feat_extra=dict(clock_mode="exp", clock_tau_s=4.0, lag_valid="scalar"))
FIT = dict(epochs=80, patience=8, batch_size=1024, lr=2e-3, gain_grid="wide_up")


def make_model(r, seed, epochs=FIT["epochs"], patience=FIT["patience"], batch_size=FIT["batch_size"],
               lr=FIT["lr"], gain_grid=FIT["gain_grid"]):
    """The unfitted model of recipe ``r`` (``registry.recipe``): the joint model (``fit_MLP_joint``), or with a
    per-fish latent (``latent`` > 0) the random-effect model (``fit_MLP_fishre``), whose wall rejection is baked in so
    that the latent's loading-gain search rolls out under the wall rule used in evaluation."""
    kw = dict(BASE, stim_mode=REPS[r["rep"]], head_order="adt", epochs=epochs, patience=patience,
              batch_size=batch_size, lr=lr, seed=seed, dt_trunc=r["dt_trunc"],
              cfg_feat_extra=dict(BASE["cfg_feat_extra"], **(r["extra"] or {})))
    if r["hist"] is not None:                         # the history ablations: (turn lags K, kinematic lags K_kin)
        kw["K"], kw["K_kin"] = int(r["hist"][0]), int(r["hist"][1])
    if not r["latent"]:
        return JT.MLPMixJointDensity(regime_w=r["regime_w"], **kw)
    kw["cfg_feat_extra"] = dict(kw["cfg_feat_extra"], wall_reject=dict(registry.REJ))
    return FR.MLPMixJointREDensity(re_m=int(r["latent"]), re_gain_grid=gain_grid, re_gain_mode="global",
                                   regime_w=r["regime_w"], drift=r["drift"], depth=int(r["depth"] or 1),
                                   re_tsize=bool(r["tsize"]), **kw)


def metricA(model, rec):
    """Held-out cross-entropy in BITS PER BOUT, per head and jointly (lower = better). ``bits_per_bout`` is the
    turn head; ``joint`` the sum of the three, which is what the model optimises."""
    sc = model.score(rec)
    out = {}
    for key, name in (("logdens", "bits_per_bout"), ("logdens_dist", "bits_per_bout_dist"),
                      ("logdens_dt", "bits_per_bout_dt")):
        v = np.asarray(sc.get(key, np.nan), float)
        out[name] = float(-np.nanmean(v) / np.log(2.0)) if v.size else np.nan
    out["bits_per_bout_joint"] = float(np.nansum([out["bits_per_bout"],
                                                  out["bits_per_bout_dist"],
                                                  out["bits_per_bout_dt"]]))
    out["n_bouts"] = int(rec.n)
    return out


def make_folds(fish, n_folds=5, seed=0):
    """Deterministic GroupKFold over fish ids -> [(train_fish, heldout_fish), ...]."""
    f = np.array(sorted(fish))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(f))
    # plain ints, not np.int64 -- the fold membership is written to JSON in the manifest
    return [([int(x) for x in f[np.setdiff1d(order, order[i::n_folds])]],
             [int(x) for x in f[order[i::n_folds]]]) for i in range(n_folds)]


def curve_png(hists, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(hists)
    fig, ax = plt.subplots(1, n, figsize=(3.1 * n, 3.0), sharey=True)
    ax = np.atleast_1d(ax)
    for a, (tr, va, lab) in zip(ax, hists):
        if tr:
            a.plot(range(1, len(tr) + 1), tr, color="#2a78d6", lw=1.5, label="train")
        a.plot(range(1, len(va) + 1), va, color="#eb6834", lw=1.5, label="val")
        if va:
            k = int(np.argmin(va))
            a.axvline(k + 1, color="#52514e", ls="--", lw=1)
            a.plot([k + 1], [va[k]], "o", color="#52514e", ms=5)
        a.set_xlabel("epoch"); a.set_title(lab, fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#e1e0d9", lw=.6); a.set_axisbelow(True)
    ax[0].set_ylabel("joint NLL  (nats / bout)")
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def records():
    """The training records: one per fish, active bouts only, from ``paths.BOUTS``."""
    return MB.build_border_records(E.load_bouts(), min_len=10)


def train_fold(name, seed, fold, out_dir, recs=None, **fit):
    """Fit fold ``fold`` of model ``name`` at ``seed`` and save it in ``out_dir`` as ``<arm>_fold<k>.pt``.
    Returns (manifest entry, held-out likelihood rows, training curves)."""
    torch.set_num_threads(1)
    r = registry.RECIPES[name]
    recs = records() if recs is None else recs
    tr_f, ho_f = make_folds(sorted(recs), 5, seed=seed)[fold]
    m = make_model(r, seed=seed + fold, **fit)
    m.name = f"{r['arm']}_fold{fold}"
    train = [recs[f] for f in tr_f]
    m.fit(train, E.make_context(train, dict(mlp_epochs=fit.get("epochs", FIT["epochs"]),
                                            mlp_patience=fit.get("patience", FIT["patience"]))))
    lik = [dict(arm=r["arm"], rep=r["rep"], order="adt", latent=r["latent"], fold=fold, fish=f,
                **metricA(m, recs[f])) for f in ho_f]
    p = Path(out_dir) / f"{r['arm']}_fold{fold}.pt"
    MX.save_mix(m, str(p))
    blob = torch.load(p, map_location="cpu", weights_only=False)     # what evaluation needs to reproduce the run
    blob["heldout_fish"] = ho_f
    blob["train_fish"] = tr_f
    blob["train_seed"] = seed + fold
    blob["rollout_seed"] = 0
    blob["phi_ref"], blob["psi_ref"] = "travel", "travel"
    blob["cfg_feat_extra"] = dict(getattr(m, "cfg_feat_extra", BASE["cfg_feat_extra"]))
    torch.save(blob, p)
    entry = dict(file=p.name, arm=r["arm"], rep=r["rep"], order="adt", stim_mode=REPS[r["rep"]], fold=fold,
                 latent=r["latent"], heldout=ho_f, epochs_run=len(m.history), regime_w=r["regime_w"],
                 extra=r["extra"], drift=r["drift"], dt_trunc=r["dt_trunc"], depth=r["depth"], hist=r["hist"],
                 tsize=r["tsize"])
    curve = (list(getattr(m._model, "_hist_train", [])), list(m.history), f"fold {fold} (held out {ho_f})")
    print(f"  {name} s{seed} fold {fold}: {len(m.history)} epochs, val {min(m.history):.4f}, held out {ho_f}"
          + (f", latent loading gain {getattr(m, 're_gain', None)}" if r["latent"] else ""), flush=True)
    return entry, lik, curve


def _fold_job(job):
    return train_fold(*job[:4], **job[4])


def train_unit(name, seed, folds=range(5), jobs=5, out_dir=None, **fit):
    """Train the folds of model ``name`` at ``seed`` in parallel and write the unit folder (default
    ``registry.ckpt_dir``): checkpoints, ``manifest_model1.json``, ``likA_model1.csv``, ``curve_<arm>.png``."""
    u = registry.unit(name, seed)
    out = Path(out_dir) if out_dir else registry.ckpt_dir(u)
    out.mkdir(parents=True, exist_ok=True)
    arm, t0 = registry.RECIPES[name]["arm"], time.time()
    jl = [(name, seed, k, str(out), fit) for k in folds]
    with ProcessPoolExecutor(max_workers=min(jobs, len(jl))) as ex:
        res = list(ex.map(_fold_job, jl))
    (out / "manifest_model1.json").write_text(json.dumps([e for e, _, _ in res], indent=1))
    pd.DataFrame([row for _, lik, _ in res for row in lik]).to_csv(out / "likA_model1.csv", index=False)
    curve_png([c for _, _, c in res], out / f"curve_{arm}.png", f"{name} s{seed} - {arm}")
    print(f"{name} s{seed} ({u['key']}): {len(res)} folds -> {out}  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    return out
