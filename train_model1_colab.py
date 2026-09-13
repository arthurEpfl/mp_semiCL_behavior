"""COLAB CELL — train Model 1 over stimulus representation x joint-likelihood ORDERING.

Two representations (S1, S3) x the six exact factorisations of the bout joint = 12 arms, each
5 GroupKFold folds, plus Metric A on held-out fish, the train/val curves as PNG, and a
manifest. Rollouts and Metric B are NOT run here -- they are CPU work and belong on the local
machine (see rollout_model1_local.py).

    !python train_model1_colab.py                     # full: 12 arms x 5 folds = 60 fits
    !python train_model1_colab.py --smoke             # ~5 min wiring check
    !python train_model1_colab.py --orders adt,tad    # a subset of the sweep
    !python train_model1_colab.py --all-data          # + one in-sample model per arm (video only)
    !python train_model1_colab.py --latent 2          # Model 2 + per-fish latent z (m=2), S1_adt + S3_adt

LATENT (--latent M). Model 2 with a per-fish random effect z ~ N(0, I_M) loading on all three
heads (fit_MLP_fishre.MLPMixJointREDensity): fitted by likelihood, integrated out EXACTLY at score
time (so Metric A stays comparable), drawn fresh per rollout at generation. Its loading gain is set
inside every fold's fit by method of moments against the TRAIN fish's between-fish spread -- that
step rolls out (~5 grid points x 8 fish x 2 draws) on the CPU, so a latent fold takes a few
minutes longer than a plain one. rej's wall band is baked into these checkpoints (cfg_feat_extra)
so the gain search rolls out under the same wall rule the local evaluation uses. Runs S1_adt and
S3_adt: the latent's heads are hard-wired in the adt order (the ordering sweep showed order is
worth 0.015 AUC end to end), and S1 vs S3 is a stimulus-response TRADE worth keeping both sides of
(S3 follows static dots better, S1 approaches better; tied on the AUC).

WHAT THE ORDERING SWEEP TESTS. The chain rule is exact in all six orderings, so every arm
targets the SAME joint density and `bits_per_bout_joint` is comparable straight across the
table. What differs is capacity: whichever channel is emitted first must be predicted from the
feature vector alone, and whichever is last also sees the other two outcomes of its own bout.
The EDA points at dt first -- |dtheta| scales with the time a bout spans (rho = 0.17) while
distance barely does (rho = -0.01) -- so "draw the interval, then turn and travel inside it"
may be the cheaper description. `adt` is the archived chain and is the control.

The three per-channel columns are NOT comparable across orderings on their own (a channel
gets cheaper simply by being later); only the joint sum is.

WHY THE SPLIT. Every metric change this month (dropping _late, the _h pair, run-aware
helpers, replacing between_fish_share) would have cost a full retrain under the coupled
design. Decoupled, a metric change is a CPU re-run over saved checkpoints.

SEEDS. Both the training seed and the rollout seed go into the blob. Arm-to-arm differences
here are small -- rollout_aware came back null because they sat under within-arm noise -- so
the local rollouts must be able to reproduce common random numbers across models trained in
separate runs. Without the recorded seed that is impossible. Every arm shares the per-fold
seed, so the 12 arms are PAIRED fold for fold.
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent          # artifacts land next to this script
ROOT = OUT if (OUT / "scripts").exists() else OUT.parent   # repo root, for imports only
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as E                     # noqa: E402
import fit_MLP_border as MB              # noqa: E402
import fit_MLP_joint as JT               # noqa: E402
import fit_MLP_mix as MX                 # noqa: E402
import fit_MLP_fishre as FR              # noqa: E402

# The two representations this sweep crosses with the orderings. S2 and S4 are dropped: on the
# R=20 Metric B panel S2 tracked S1 and S4 tracked S3 closely, so carrying all four would triple
# the GPU cost to re-measure a difference that is already known to be small.
REPS = {"S1": "s1",          # 6  [act, sin, cos, g, moving, nz]      monotone proximity
        "S3": "shoalr"}      # 13 range BASIS instead of the scalar g; no group size

# The six exact factorisations of p(dtheta, distance, dt | x). 'a' = dtheta, 'd' = distance,
# 't' = dt; the string is the EMISSION order, so 'tad' means dt is drawn first from x alone and
# dtheta last, conditioned on both. See fit_MLP_joint's module docstring.
ORDERS = ["adt", "atd", "dat", "dta", "tad", "tda"]

# use_kin is forced True by the joint class -- one x carrying everything -- so it is not a
# constructor arg. clock_mode / clock_tau_s / lag_valid are cfg_feat keys with no arg of their
# own, so they ride in via cfg_feat_extra and are saved with the checkpoint.
BASE = dict(K=8, K_kin=3, hidden=128, use_border=True, border_mode="raw",
            n_dist=2, n_dt=3,
            phi_ref="travel", psi_ref="travel",          # defaults now, set anyway for the record
            closed_loop=True, guardrail=True, stim_loop=True,
            contain=False, vigor=False, indiv=False, K_stim=1, noise_aug=None,
            cfg_feat_extra=dict(clock_mode="exp", clock_tau_s=4.0, lag_valid="scalar"))

# rej's generation-time wall rejection (redraw any bout landing 2-10 mm from the wall). Applied at
# rollout for the plain arms; BAKED IN for latent arms so their in-fit gain search sees it.
REJ_BAND = {"d0": 2.0, "d1": 10.0, "max_draws": 20}


def make_model(args, mode, order, seed):
    """Model 2 (MLPMixJointDensity), or Model 2 + per-fish latent when --latent > 0."""
    kw = dict(BASE, stim_mode=mode, head_order=order, epochs=args.epochs, patience=args.patience,
              batch_size=args.batch_size, lr=args.lr, seed=seed)
    if not args.latent:
        return JT.MLPMixJointDensity(**kw)
    kw["cfg_feat_extra"] = dict(BASE["cfg_feat_extra"], wall_reject=dict(REJ_BAND))
    return FR.MLPMixJointREDensity(re_m=int(args.latent), re_gain_grid=args.gain_grid, **kw)


def metricA(model, rec):
    """Held-out cross-entropy in BITS PER BOUT, per channel and jointly (lower = better).

    `score` returns PER-BOUT log densities, not scalars -- aggregate here with the same
    convention as evaluate (`-mean(logdens) / ln 2`) so these numbers sit on the same scale
    as every archived bits_per_bout. `joint` is the sum of the three channels, which is what
    the joint model actually optimises; `bits_per_bout` alone is the turn term only.
    """
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
    """Train/val NLL per epoch, one panel per fold. `hists` = [(train, val, label), ...]."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)   # GPU: big batches, see note below
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--out", type=str, default=None,
                    help="zip name (default model1_checkpoints.zip, model2z_checkpoints.zip "
                         "with --latent)")
    ap.add_argument("--orders", type=str, default=None,
                    help="comma-separated emission orders (default: all six; 'adt' with --latent)")
    ap.add_argument("--reps", type=str, default=None,
                    help="comma-separated representation tags (default: S1,S3)")
    ap.add_argument("--latent", type=int, default=0,
                    help="per-fish latent dimension m (0 = off, the plain Model 2; 2 = rej's)")
    ap.add_argument("--gain-grid", type=str, default="wide_up", choices=("std", "wide", "wide_up"),
                    help="candidate latent loading gains searched per fold (--latent only): std = "
                         "0.4..3.0 (5), wide = 0.1..3.0 (8), wide_up = 0.1..6.5 (10, default)")
    # OFF by default for this sweep. The all-data model has no held-out fish, so it is scored by
    # nothing and skipped by rollout_model1_local.py -- it exists only to render video. Training
    # one per arm would add 12 full fits to a run whose whole purpose is the 12-way comparison.
    ap.add_argument("--all-data", action="store_true",
                    help="also train one in-sample model per arm (generation artifact only)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.patience, args.folds = 3, 2, 2
    if args.orders is None:
        args.orders = "adt" if args.latent else ",".join(ORDERS)
    if args.reps is None:
        args.reps = ",".join(REPS)
    if args.out is None:
        args.out = "model2z_checkpoints.zip" if args.latent else "model1_checkpoints.zip"
    orders = [JT.order_tag(o) for o in args.orders.split(",") if o.strip()]
    reps = {t: REPS[t] for t in (s.strip() for s in args.reps.split(",")) if t}
    if args.latent and any(o != "adt" for o in orders):
        # fail before any GPU time is spent, not at the first fit
        raise SystemExit(f"--latent supports head order 'adt' only (the random-effect heads are "
                         f"hard-wired); got {orders}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}"
          + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "  -- NO GPU"))
    if dev == "cuda":
        # Big batches + TF32 are the two free wins here: the net is tiny (128 hidden) so the
        # bottleneck is kernel-launch overhead, not FLOPs. A 1024 batch cuts the number of
        # launches 4x against the 256 default at no cost to the fit -- the gradient is an
        # average either way, and lr is scaled with it. TF32 matmuls are ~free on Ampere+.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    df = E.load_bouts()                                   # clean_data/bouts_clean.csv
    recs = MB.build_border_records(df, min_len=args.min_len)   # active_only=True by default
    fish = sorted(recs)
    print(f"{len(df):,} rows -> {len(fish)} fish, "
          f"{sum(r.n for r in recs.values()):,} active bouts")
    folds = make_folds(fish, args.folds, seed=args.seed)

    arms = [(t, mode, o) for t, mode in reps.items() for o in orders]
    n_fit = len(arms) * (args.folds + int(args.all_data))
    print(f"{len(arms)} arms ({'/'.join(reps)} x {'/'.join(orders)}) x "
          f"{args.folds} folds{' + all-data' if args.all_data else ''} = {n_fit} fits")

    likA, manifest, curves, t0 = [], [], {}, time.time()
    for ai, (tag, mode, order) in enumerate(arms):
        arm = f"{tag}_{order}" + (f"_z{args.latent}" if args.latent else "")
        hists = []
        for k, (tr_f, ho_f) in enumerate(folds):
            # SAME seed per fold across every arm, so the 12 arms are paired: an arm-to-arm
            # difference is then the ordering, not a different initialisation.
            m = make_model(args, mode, order, seed=args.seed + k)
            m.name = f"{arm}_fold{k}"
            train = [recs[f] for f in tr_f]
            ctx = E.make_context(train, dict(mlp_epochs=args.epochs,
                                             mlp_patience=args.patience))
            m.fit(train, ctx)
            # Metric A on the fish this fold never saw
            for f in ho_f:
                likA.append(dict(rep=tag, order=order, latent=int(args.latent), fold=k, fish=f,
                                 **metricA(m, recs[f])))
            p = OUT / f"{arm}_fold{k}.pt"
            MX.save_mix(m, str(p))
            # everything the local half needs to reproduce the run
            blob = torch.load(p, map_location="cpu", weights_only=False)
            blob["heldout_fish"] = ho_f
            blob["train_fish"] = tr_f
            blob["train_seed"] = args.seed + k
            blob["rollout_seed"] = args.rollout_seed
            blob["phi_ref"], blob["psi_ref"] = "travel", "travel"
            blob["cfg_feat_extra"] = dict(getattr(m, "cfg_feat_extra", BASE["cfg_feat_extra"]))
            torch.save(blob, p)
            hists.append((list(getattr(m._model, "_hist_train", [])),
                          list(m.history), f"fold {k} (held out {ho_f})"))
            manifest.append(dict(file=p.name, rep=tag, order=order, stim_mode=mode, fold=k,
                                 latent=int(args.latent), heldout=ho_f,
                                 epochs_run=len(m.history)))
            if args.latent:
                print(f"    {arm} fold {k}: latent loading gain {getattr(m, 're_gain', None)} "
                      f"(grid {args.gain_grid}: {FR.gain_grid_for(args.gain_grid)})")
            print(f"  {arm} fold {k}: {len(m.history)} epochs, "
                  f"val {min(m.history):.4f}, held out {ho_f}  [{time.time()-t0:.0f}s]")

        if args.all_data:
            # generation artifact ONLY -- no held-out fish, so never a score
            m = make_model(args, mode, order, seed=args.seed)
            m.name = f"{arm}_alldata"
            train = [recs[f] for f in fish]
            m.fit(train, E.make_context(train, dict(mlp_epochs=args.epochs,
                                                    mlp_patience=args.patience)))
            p = OUT / f"{arm}_alldata.pt"
            MX.save_mix(m, str(p))
            blob = torch.load(p, map_location="cpu", weights_only=False)
            blob.update(heldout_fish=[], train_fish=fish, train_seed=args.seed,
                        rollout_seed=args.rollout_seed, insample_only=True,
                        phi_ref="travel", psi_ref="travel")
            torch.save(blob, p)
            manifest.append(dict(file=p.name, rep=tag, order=order, stim_mode=mode, fold=-1,
                                 heldout=[], epochs_run=len(m.history), insample_only=True))
            hists.append((list(getattr(m._model, "_hist_train", [])), list(m.history),
                          "all data (in-sample)"))
        curves[arm] = f"curve_{arm}.png"
        curve_png(hists, OUT / curves[arm], f"Model 2 · {arm} ({mode})")
        done = (ai + 1) / len(arms)
        print(f"  [{ai+1}/{len(arms)} arms] {arm} done -> {curves[arm]}   "
              f"eta {(time.time()-t0)*(1-done)/done:.0f}s")

    lik = pd.DataFrame(likA)
    lik.to_csv(OUT / "likA_model1.csv", index=False)
    (OUT / "manifest_model1.json").write_text(json.dumps(manifest, indent=1))
    with zipfile.ZipFile(OUT / args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for e in manifest:
            z.write(OUT / e["file"], e["file"])
        for f in curves.values():
            z.write(OUT / f, f)
        z.write(OUT / "likA_model1.csv", "likA_model1.csv")
        z.write(OUT / "manifest_model1.json", "manifest_model1.json")
    mb = (OUT / args.out).stat().st_size / 1e6

    # The comparison the sweep exists for, printed so a killed download is not a lost run.
    # Only the JOINT column is comparable across orderings: a channel emitted late is cheaper
    # simply because it sees more, so the per-channel columns move against each other by
    # construction and only their sum is the same quantity in every arm.
    if not lik.empty:
        piv = lik.pivot_table(index="order", columns="rep", values="bits_per_bout_joint")
        print("\nMetric A, bits per bout, JOINT (all three channels; lower = better):")
        print(piv.round(4).to_string())
        print("\nper channel (turn / dist / dt) -- NOT comparable across orders, only within:")
        print(lik.groupby(["rep", "order"])[["bits_per_bout", "bits_per_bout_dist",
                                             "bits_per_bout_dt"]].mean().round(4).to_string())
        best = piv.mean(axis=1).idxmin()
        print(f"\nbest ordering on the mean of the {piv.shape[1]} representations: {best}"
              f"   (archived chain 'adt' = {piv.mean(axis=1).get('adt', float('nan')):.4f})")
    print(f"\nwrote {args.out} ({mb:.1f} MB) in {time.time()-t0:.0f}s -- DOWNLOAD THIS.")
    print("Metric B is NOT in here: run rollout_model1_local.py against these checkpoints.")


if __name__ == "__main__":
    main()
