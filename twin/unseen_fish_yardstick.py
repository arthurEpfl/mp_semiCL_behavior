"""Yardstick for the "realistic UNSEEN fish" claim: C2ST between INDIVIDUALS.

The ladder's C2ST (``generative_metrics.c2st_auc``) pools every fish and splits by fish, so its
classifier must find what separates generated from real windows ACROSS fish -- it is forbidden to learn
the individual. That is the right test for "is the generated population real", and it cannot answer
"is a sampled individual a plausible fish". Nor can it compare two real fish: labelling fish g's
windows as fish f's "generated" set leaks identity with the wrong sign (g's other windows sit in the
training set with the other label), so the AUC comes out BELOW 0.5 by construction.

This module asks the individual-level question with one PAIRWISE protocol for every row: two
individuals A and B, each a time-ordered set of the harness's W=60 windows (``window_features_traj``,
dot-distance and radial included); the harness's RandomForest(200) is trained on the first half of each
individual's session and tested on the second halves, then the reverse; the row is the mean ROC-AUC.
An individual is one real fish's session, or ONE full-session rollout (one z draw).

    pairwise_table(ckpt_dir, {"st": arm, ...}, rej=...)   # floor / real-real / real vs one sampled / pooled
    nearest_real(ckpt_dir, arm, rej=...)                   # distance to the nearest real fish, real vs sampled

MEASURED 2026-09-19 on Model 17's ``S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st`` (held-out fold models, rej):
real vs real 0.960, real vs one sampled fish 0.969 -- individuals are that distinct, the pairwise number
saturates for any model; the usable form is the nearest-real-neighbour distance: real fish 0.838,
sampled fish 0.860 (sampled fish lie inside the cloud of real individuals, with a ~10 % tail outside).
Windows and pair AUCs are cached next to the checkpoints (``yardstick_*``), so a re-run is seconds.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from twin import paths


W_REPS, W_SEED = 8, 0


def _ckpt_dir(d):
    d = Path(d)
    for q in (d, paths.CKPTS / d):
        if (q / "manifest_model1.json").exists():
            return q
    raise FileNotFoundError(f"no manifest_model1.json under {d}")


def _windows_job(job):
    """(ckpt, fish, rej, seed, reps) -> (fish, real windows, [sampled windows per rollout])."""
    ckpt, fid, rej, seed, reps = job
    import torch
    torch.set_num_threads(1)
    from twin import evaluate as E
    from twin import fit_MLP_border as MB
    from twin import fit_MLP_mix as MX
    from twin import generative_metrics as GM
    recs = MB.build_border_records(E.load_bouts())
    m = MX.load_mix(str(ckpt))
    if rej:
        m._cfg_feat = {**m._cfg_feat, "wall_reject": dict(rej)}
    w = GM.c2st_windows_for_fish(m, recs[fid], fid, seed=seed, reps=reps, interior=None)
    return fid, w["traj"][0][0], [g for _, g in w["traj"]]


def individual_windows(ckpt_dir, arm, rej=None, reps=W_REPS, seed=W_SEED, jobs=10):
    """real {fish: (n, d)} and sampled {fish: [(n_r, d)] * reps} window features for ``arm``: each fish
    rolled out by the checkpoint that holds it out, the ladder's rollout streams ([seed, fish, rep])."""
    d = _ckpt_dir(ckpt_dir)
    cache = d / f"yardstick_windows_{arm}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["real"].item(), z["sampled"].item()
    mani = json.loads((d / "manifest_model1.json").read_text())
    jl = [(d / e["file"], int(f), rej, seed, reps) for e in mani if e["arm"] == arm for f in e["heldout"]]
    if not jl:
        raise ValueError(f"no checkpoints for arm {arm} in {d}")
    with ProcessPoolExecutor(max_workers=min(jobs, len(jl))) as ex:
        res = list(ex.map(_windows_job, jl))
    real = {fid: r for fid, r, _ in res}
    sampled = {fid: g for fid, _, g in res}
    np.savez(cache, real=np.array(real, dtype=object), sampled=np.array(sampled, dtype=object))
    return real, sampled


def pair_auc(A, B, seed=0, n_jobs=1):
    """Pairwise C2ST between two individuals (see the module docstring)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    A = np.asarray(A, float); B = np.asarray(B, float)
    ha, hb = len(A) // 2, len(B) // 2
    out = []
    for tr, te in (((A[:ha], B[:hb]), (A[ha:], B[hb:])), ((A[ha:], B[hb:]), (A[:ha], B[:hb]))):
        X = np.vstack(tr); y = np.r_[np.ones(len(tr[0])), np.zeros(len(tr[1]))]
        Xt = np.vstack(te); yt = np.r_[np.ones(len(te[0])), np.zeros(len(te[1]))]
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=n_jobs).fit(X, y)
        out.append(roc_auc_score(yt, clf.predict_proba(Xt)[:, 1]))
    return float(np.mean(out))


def _pair_job(job):
    ka, A, kb, B = job
    return ka, kb, pair_auc(A, B)


def _pairs(jobs, workers=10):
    if len(jobs) < 40:
        return [_pair_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_pair_job, jobs, chunksize=20))


def pairwise_table(ckpt_dir, arms, rej=None, reps=W_REPS, seed=W_SEED, jobs=10):
    """The pairwise C2ST table. ``arms`` = {label: arm}. Rows: ``floor`` (a real fish vs its own
    interleaved windows), ``real-real`` (every pair of real fish), and per label ``<label>`` (a real fish
    vs ONE full-session rollout of its held-out checkpoint, ``reps`` per fish) and ``<label>_pool`` (the
    same fish vs its ``reps`` rollouts pooled -- the harness's mixture, for reference).
    Returns (summary, per_fish, rows)."""
    d = _ckpt_dir(ckpt_dir)
    t0 = time.time()
    feats = {lab: individual_windows(d, arm, rej, reps, seed, jobs) for lab, arm in arms.items()}
    first = next(iter(feats))
    real = feats[first][0]
    fish = sorted(real)
    cache = d / f"yardstick_pairs_{'_'.join(arms)}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        jl = [(("floor", f, f), real[f][0::2], ("", f), real[f][1::2]) for f in fish]
        jl += [(("real-real", f, g), real[f], ("", g), real[g]) for i, f in enumerate(fish) for g in fish[i + 1:]]
        for lab in arms:
            for f in fish:
                r, gens = feats[lab][0][f], feats[lab][1][f]
                jl += [((lab, f, k), r, ("", f), g) for k, g in enumerate(gens)]
                jl.append(((lab + "_pool", f, -1), r, ("", f), np.vstack(gens)))
        res = _pairs(jl, jobs)
        df = pd.DataFrame([dict(kind=ka[0], a=ka[1], b=ka[2], auc=a) for ka, _, a in res])
        df.to_csv(cache, index=False)
    s = df.groupby("kind").auc.agg(["mean", "std", "count"])
    s["se"] = s["std"] / np.sqrt(s["count"])
    order = ["floor", "real-real"] + [x for lab in arms for x in (lab, lab + "_pool")]
    s = s.reindex(order)
    rr = df[df.kind == "real-real"]
    per = pd.DataFrame({"vs_other_real": (pd.concat([rr.groupby("a").auc.sum(), rr.groupby("b").auc.sum()])
                                          .groupby(level=0).sum() / (len(fish) - 1))})
    for lab in arms:
        per["vs_" + lab] = df[df.kind == lab].groupby("a").auc.mean()
    per["n_real_windows"] = pd.Series({f: len(real[f]) for f in fish})
    pd.set_option("display.width", 200)
    print("PAIRWISE C2ST BETWEEN INDIVIDUALS (one real fish vs one other individual; RF200 on the harness's "
          "window features, trained on the first half of each session, tested on the second, and the reverse):")
    print("  floor = the same real fish, interleaved windows; real-real = two real fish; <arm> = a real fish vs ONE "
          "full-session rollout of the checkpoint that never saw it (one z draw); <arm>_pool = vs its rollouts pooled")
    print(s.round(3).to_string())
    print("\nper fish: the mean pairwise AUC against the other 14 real fish, and against its own sampled fish")
    print(per.round(3).to_string())
    for lab in arms:
        n = int((per["vs_" + lab] < per.vs_other_real).sum())
        print(f"  fish closer to its {lab} sampled fish than to the average other real fish: {n} / {len(fish)}")
    print(f"  [{time.time() - t0:.0f}s]")
    return s, per, df


def feature_drop_table(ckpt_dir, arms, rej=None, reps=W_REPS, seed=W_SEED, jobs=10):
    """The held-out C2ST of each arm with feature blocks REMOVED from the same rollouts: all 39, without
    the radial block, the three heads only (turn + step + interval), turn only. ``arms`` = list of arm
    names or {label: arm}. Same GroupKFold-by-fish classifier as the ladder (``c2st_auc_replicates``)."""
    from twin import generative_metrics as GM
    d = _ckpt_dir(ckpt_dir)
    arms = arms if isinstance(arms, dict) else {a: a for a in arms}
    rows = []
    for lab, arm in arms.items():
        real, sampled = individual_windows(d, arm, rej, reps, seed, jobs)
        dim = next(iter(real.values())).shape[1]
        names = GM.traj_names_for_dim(dim)
        sets = {"all": list(range(dim)),
                "no radial": [i for i, n in enumerate(names) if not n.startswith("rad")],
                "three heads": [i for i, n in enumerate(names) if not n.startswith(("rad", "dotdist"))],
                "turn only": [i for i, n in enumerate(names) if not n.startswith(("rad", "dotdist", "dist", "dt"))]}
        r = dict(arm=lab)
        for k, cols in sets.items():
            wins = [{f: (real[f][:, cols], sampled[f][j][:, cols]) for f in real} for j in range(len(next(iter(sampled.values()))))]
            mean, sd, per = GM.c2st_auc_replicates(wins, seed=seed)
            r[k] = mean; r[k + "_se"] = sd / np.sqrt(max(len(per), 1))
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("arm")
    pd.set_option("display.width", 200)
    print("held-out C2ST with feature blocks removed (same rollouts, 8 per fish; 0.5 = indistinguishable):")
    print(tab.round(3).to_string())
    return tab


def nearest_real(ckpt_dir, arm, rej=None, reps=W_REPS, seed=W_SEED, jobs=10, plot=True):
    """Distance to the NEAREST real fish, same pairwise AUC: for each real fish the closest OTHER real
    fish, for each sampled fish (every rollout of ``arm``) the closest real fish. Sampled fish inside
    the cloud of real individuals read like real fish here; outside it they are farther from every real
    fish than any real fish is from its neighbour. Returns (per_fish, nn_sampled)."""
    d = _ckpt_dir(ckpt_dir)
    t0 = time.time()
    real, sampled = individual_windows(d, arm, rej, reps, seed, jobs)
    fish = sorted(real)
    samp = {(f, k): g for f in fish for k, g in enumerate(sampled[f])}
    cache = d / f"yardstick_nearest_{arm}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        jl = [(("r", f, -1), real[f], ("r", g, -1), real[g]) for i, f in enumerate(fish) for g in fish[i + 1:]]
        jl += [(("r", g, -1), real[g], ("s", f, k), samp[(f, k)]) for g in fish for (f, k) in samp]
        res = _pairs(jl, jobs)
        df = pd.DataFrame([dict(ka=ka[0], fa=ka[1], ra=ka[2], kb=kb[0], fb=kb[1], rb=kb[2], auc=a)
                           for ka, kb, a in res])
        df.to_csv(cache, index=False)
    auc = {}
    for r in df.itertuples():
        auc[((r.ka, r.fa, r.ra), (r.kb, r.fb, r.rb))] = r.auc
        auc[((r.kb, r.fb, r.rb), (r.ka, r.fa, r.ra))] = r.auc
    R = lambda f: ("r", f, -1)  # noqa: E731
    nn_real = pd.Series({f: min(auc[(R(f), R(g))] for g in fish if g != f) for f in fish})
    who_real = pd.Series({f: min(((auc[(R(f), R(g))], g) for g in fish if g != f))[1] for f in fish})
    nn_samp = pd.Series({s: min(auc[(R(g), ("s",) + s)] for g in fish) for s in samp})
    who_samp = pd.Series({s: min(((auc[(R(g), ("s",) + s)], g) for g in fish))[1] for s in samp})
    all_real = pd.Series({f: np.mean([auc[(R(f), R(g))] for g in fish if g != f]) for f in fish})
    all_samp = pd.Series({s: np.mean([auc[(R(g), ("s",) + s)] for g in fish]) for s in samp})
    per = pd.DataFrame({"nearest_real": nn_real, "which": who_real,
                        "nearest_real_of_its_sampled_fish": pd.Series({f: np.mean([nn_samp[(f, k)] for k in range(len(sampled[f]))]) for f in fish}),
                        "n_windows": pd.Series({f: len(real[f]) for f in fish})})
    own = float(np.mean([who_samp[s] == s[0] for s in samp]))
    med = float(nn_real.median())
    inside = int((nn_samp < med).sum())
    print("DISTANCE TO THE NEAREST REAL FISH (pairwise AUC; lower = closer):")
    print(f"  real fish    -> nearest OTHER real fish: mean {nn_real.mean():.3f}  se {nn_real.std() / np.sqrt(len(nn_real)):.3f}  "
          f"median {med:.3f}  10th-90th pct {nn_real.quantile(.1):.3f}-{nn_real.quantile(.9):.3f}   (n = {len(nn_real)})")
    print(f"  sampled fish -> nearest real fish:       mean {nn_samp.mean():.3f}  se {nn_samp.std() / np.sqrt(len(nn_samp)):.3f}  "
          f"median {nn_samp.median():.3f}  10th-90th pct {nn_samp.quantile(.1):.3f}-{nn_samp.quantile(.9):.3f}   (n = {len(nn_samp)})")
    print(f"  mean distance to ALL real fish: real {all_real.mean():.3f}, sampled {all_samp.mean():.3f}")
    print(f"  sampled fish whose nearest real fish is closer than the median real fish's ({med:.3f}): {inside} / {len(nn_samp)}"
          f"  (a population exactly inside the cloud: ~{len(nn_samp) // 2})")
    print(f"  the sampled fish's nearest real fish is the one it was rolled out for: {100 * own:.0f}%  (chance {100 / len(fish):.0f}%)")
    print("\nper fish:")
    print(per.round(3).to_string())
    print(f"  [{time.time() - t0:.0f}s]")
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(fish))
        ax.bar(x, [nn_real[f] for f in fish], color="0.35", width=0.6, label="real fish: nearest other real fish")
        for i, f in enumerate(fish):
            ys = [nn_samp[(f, k)] for k in range(len(sampled[f]))]
            ax.plot(np.full(len(ys), i) + np.linspace(-0.18, 0.18, len(ys)), ys, "o", color="tab:orange", ms=4.5,
                    alpha=0.85, label="sampled fish (this fish's rollouts): nearest real fish" if i == 0 else None)
        ax.axhline(nn_real.mean(), color="0.35", lw=1, ls="--")
        ax.axhline(nn_samp.mean(), color="tab:orange", lw=1, ls="--")
        ax.axhline(0.5, color="0.8", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(fish); ax.set_xlabel("fish")
        ax.set_ylabel("pairwise AUC to the nearest real fish\n(lower = closer; 0.5 = same individual)")
        ax.set_ylim(0.45, 1.02)
        ax.set_title(f"How far is an individual from its nearest real fish?  real {nn_real.mean():.3f}  vs  sampled "
                     f"{nn_samp.mean():.3f}  (dashed = means; {arm})", fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
    return per, nn_samp
