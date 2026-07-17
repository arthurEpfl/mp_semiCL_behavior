"""
Request 1 -- is leave-fish-out (LOFO) cross-validation legitimate?

LOFO does NOT need the 15 fish to be identical; it needs the model RANKING to be
stable across fish, and it helps if between-fish heterogeneity is modest. This module
provides the checks from the approved plan (Part 1), in order of decision-relevance:

  A. rank_stability      -- do paired per-fish model deltas agree in sign? (the test
                            you actually depend on; >=13/15 sign-agreement green-lights)
  B. pooling_gap         -- personalized (train on self) vs pooled (train on others)
                            held-out density, with a DATA-MATCHED pooled control that
                            separates individuality from training-data volume.
  C. icc_table           -- between- vs within-fish variance (ICC) of summary stats on
                            CONTIGUOUS blocks (>~0.6 = strong individuality).
  D. transition_homogeneity -- pairwise Jensen-Shannon between per-fish transition
                            matrices, CALIBRATED against a within-fish (split-half) null
                            (ratio ~1 => shared structure). Effect size, not a p-value.
  E. learning_curve      -- held-out score vs number of training fish (plateau before
                            12 => population well covered).
  F. cluster_fish        -- do fish form behavioural clusters? (=> stratify folds).

Reuses evaluate.py's records/adapters; the density metric is mean nats/bout.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E
import generative_metrics as G


# =========================================================================== #
#  Per-fish / per-block summary statistics
# =========================================================================== #
def summary_stats(dtheta, symbol, dt_s):
    """Behavioural summary of a bout chunk (used for ICC and clustering)."""
    dtheta = np.asarray(dtheta, float); sym = np.asarray(symbol, int)
    dt = np.asarray(dt_s, float); dt = dt[np.isfinite(dt)]
    T = G._transition_matrix(sym)
    turn = sym != 0
    return dict(
        forward_fraction=float(np.mean(sym == 0)),
        abs_dtheta_median=float(np.median(np.abs(dtheta))),
        bout_rate_hz=float(1.0 / np.median(dt)) if len(dt) else np.nan,
        lr_bias=float(np.mean(sym == 2) - np.mean(sym == 1)),
        turn_stay=float((T[1, 1] + T[2, 2]) / 2),         # P(keep turning same way)
        p_ff=float(T[0, 0]),                              # forward stickiness
    )


def _contiguous_blocks(n, k):
    """Indices of k near-equal contiguous blocks over range(n)."""
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(k) if edges[i + 1] > edges[i]]


# =========================================================================== #
#  C. ICC -- between- vs within-fish variance decomposition
# =========================================================================== #
def _icc1(values_by_group):
    """One-way random-effects ICC(1) from balanced-ish groups (list of arrays)."""
    groups = [np.asarray(v, float) for v in values_by_group if len(v) > 0]
    groups = [v[np.isfinite(v)] for v in groups]
    groups = [v for v in groups if len(v) > 0]
    if len(groups) < 2:
        return np.nan
    k = np.mean([len(v) for v in groups])
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(len(v) * (v.mean() - grand) ** 2 for v in groups)
    ssw = sum(np.sum((v - v.mean()) ** 2) for v in groups)
    dfb = len(groups) - 1
    dfw = len(allv) - len(groups)
    if dfb <= 0 or dfw <= 0:
        return np.nan
    msb, msw = ssb / dfb, ssw / dfw
    denom = msb + (k - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else np.nan


def icc_table(records, n_blocks=8, stats=("forward_fraction", "abs_dtheta_median",
                                          "bout_rate_hz", "lr_bias", "turn_stay", "p_ff")):
    """ICC per summary statistic across fish, using contiguous within-fish blocks.

    ICC ~<0.3: near-exchangeable for that stat; ~>0.6: strong individuality
    (between-fish variance dominates -> pooling loses fidelity for that stat)."""
    import pandas as pd
    per_stat = {s: [] for s in stats}
    for rec in records.values():
        blocks = _contiguous_blocks(rec.n, n_blocks)
        vals = {s: [] for s in stats}
        for lo, hi in blocks:
            st = summary_stats(rec.dtheta[lo:hi], rec.symbol[lo:hi], rec.dt_s[lo:hi])
            for s in stats:
                vals[s].append(st[s])
        for s in stats:
            per_stat[s].append(np.array(vals[s]))
    rows = [dict(statistic=s, icc=_icc1(per_stat[s]),
                 interpretation=("near-exchangeable" if (_icc1(per_stat[s]) or 1) < 0.3
                                 else "moderate" if (_icc1(per_stat[s]) or 1) < 0.6
                                 else "strong-individuality"))
            for s in stats]
    return pd.DataFrame(rows)


# =========================================================================== #
#  D. Transition-matrix homogeneity vs a within-fish null
# =========================================================================== #
def transition_homogeneity(records):
    """Pairwise between-fish JS of symbol transition matrices, vs a within-fish
    (split-half) null. Returns (summary_dict, dissimilarity_matrix, fish_ids)."""
    fids = sorted(records)
    Tmats = {f: G._transition_matrix(records[f].symbol) for f in fids}

    def _mat_js(A, B):
        return float(np.mean([G._js_divergence(A[i], B[i]) for i in range(3)]))

    n = len(fids)
    D = np.zeros((n, n))
    between = []
    for i in range(n):
        for j in range(i + 1, n):
            d = _mat_js(Tmats[fids[i]], Tmats[fids[j]])
            D[i, j] = D[j, i] = d
            between.append(d)
    within = []
    for f in fids:
        s = records[f].symbol
        h = len(s) // 2
        within.append(_mat_js(G._transition_matrix(s[:h]), G._transition_matrix(s[h:])))
    between, within = np.array(between), np.array(within)
    summary = dict(between_mean=float(between.mean()), between_median=float(np.median(between)),
                   within_mean=float(within.mean()), within_median=float(np.median(within)),
                   ratio_mean=float(between.mean() / (within.mean() + 1e-12)),
                   ratio_median=float(np.median(between) / (np.median(within) + 1e-12)))
    return summary, D, fids


# =========================================================================== #
#  F. Fish clustering (-> stratified folds if clusters exist)
# =========================================================================== #
def fish_feature_matrix(records, stats=("forward_fraction", "abs_dtheta_median",
                                        "bout_rate_hz", "lr_bias", "turn_stay", "p_ff")):
    import pandas as pd
    fids = sorted(records)
    rows = {f: summary_stats(records[f].dtheta, records[f].symbol, records[f].dt_s) for f in fids}
    df = pd.DataFrame([rows[f] for f in fids], index=fids)[list(stats)]
    return df


def cluster_fish(records, n_clusters=2):
    """Hierarchical (Ward) clustering of fish on standardised summary stats.
    Returns (labels_dict, linkage_matrix, feature_df)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    df = fish_feature_matrix(records)
    X = df.to_numpy(float)
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12)
    X = np.nan_to_num(X)
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return {f: int(l) for f, l in zip(df.index, labels)}, Z, df


def strata_from_clusters(records, n_clusters=2):
    """Convenience: a fish_id->label dict for make_folds(strata=...)."""
    labels, _, _ = cluster_fish(records, n_clusters=n_clusters)
    return labels


# =========================================================================== #
#  B. Pooling gap -- personalized vs pooled (with data-matched control)
# =========================================================================== #
def _slice(rec, lo, hi):
    """Slice a FishRecord to bouts [lo:hi] keeping all channels aligned."""
    return replace(rec,
                   dtheta=rec.dtheta[lo:hi], symbol=rec.symbol[lo:hi], b=rec.b[lo:hi],
                   stim_angle=rec.stim_angle[lo:hi], moving=rec.moving[lo:hi],
                   dt_s=rec.dt_s[lo:hi], distance_px=rec.distance_px[lo:hi],
                   u_raw=None if rec.u_raw is None else rec.u_raw[lo:hi])


def _mean_nats(model, rec):
    ld = model.score(rec).get("logdens")
    return float(np.mean(ld)) if ld is not None else np.nan


def pooling_gap(records, model_factory=None, split=0.5, seed=0, cfg=None):
    """Per fish: personalized vs pooled held-out density (nats/bout), same test half.

    personalized  : fit on the fish's EARLY bouts, score its LATE bouts.
    pooled_full   : fit on ALL OTHER fish, score this fish's LATE bouts.
    pooled_matched: fit on ~one-fish-worth of OTHER fish's bouts, score LATE bouts
                    (controls for training-data volume -> isolates individuality).
    gap_full/matched = personalized - pooled_*  (>0 => personalising helps).
    """
    import pandas as pd
    model_factory = model_factory or (lambda: E.SpontaneousHMM(restarts=1, max_iter=120))
    cfg = cfg or {}
    rng = np.random.default_rng(seed)
    fids = sorted(records)
    rows = []
    for f in fids:
        rec = records[f]
        h = int(rec.n * split)
        early, late = _slice(rec, 0, h), _slice(rec, h, rec.n)
        # personalized
        mp = model_factory(); mp.fit([early], E.make_context([early], cfg))
        L_self = _mean_nats(mp, late)
        # pooled full
        others = [records[g] for g in fids if g != f]
        mf = model_factory(); mf.fit(others, E.make_context(others, cfg))
        L_pool = _mean_nats(mf, late)
        # pooled matched (~ h bouts from ONE random other fish)
        og = records[rng.choice([g for g in fids if g != f])]
        og_slice = _slice(og, 0, min(h, og.n))
        mm = model_factory(); mm.fit([og_slice], E.make_context([og_slice], cfg))
        L_match = _mean_nats(mm, late)
        rows.append(dict(fish_id=f, n_test=late.n,
                         personalized=L_self, pooled_full=L_pool, pooled_matched=L_match,
                         gap_full=L_self - L_pool, gap_matched=L_self - L_match))
    df = pd.DataFrame(rows)
    return df


# =========================================================================== #
#  E. Learning curve vs number of training fish
# =========================================================================== #
def learning_curve(records, sizes=(1, 2, 4, 8, 12), n_holdout=3, n_draws=4,
                   model_factory=None, seed=0, cfg=None):
    """Held-out density (nats/bout) vs number of training fish. A plateau before the
    max means the population is well covered (exchangeability behaviourally supported)."""
    import pandas as pd
    model_factory = model_factory or (lambda: E.SpontaneousHMM(restarts=1, max_iter=120))
    cfg = cfg or {}
    rng = np.random.default_rng(seed)
    fids = sorted(records)
    holdout = list(rng.choice(fids, size=n_holdout, replace=False))
    pool = [f for f in fids if f not in holdout]
    rows = []
    for m in sizes:
        if m > len(pool):
            continue
        for d in range(n_draws):
            train_ids = list(rng.choice(pool, size=m, replace=False))
            train = [records[g] for g in train_ids]
            model = model_factory(); model.fit(train, E.make_context(train, cfg))
            score = float(np.mean([_mean_nats(model, records[g]) for g in holdout]))
            rows.append(dict(n_train_fish=m, draw=d, holdout_nats_per_bout=score))
    df = pd.DataFrame(rows)
    agg = df.groupby("n_train_fish")["holdout_nats_per_bout"].agg(["mean", "std", "count"]).reset_index()
    return df, agg, holdout


# =========================================================================== #
#  A. Rank stability (the decision-relevant check)
# =========================================================================== #
def rank_stability(likelihood_df, pairs, metric="nats_per_bout", higher_better=True):
    """For each (model_a, model_b) pair, the paired per-fish sign agreement + Wilcoxon.

    ``likelihood_df`` is the output of evaluate.cross_validate_likelihood (one row per
    fish per model). Returns a DataFrame; high ``prefer_a``/N (e.g. >=13/15) with a
    consistent winner means LOFO ranking is trustworthy regardless of individuality.
    """
    import pandas as pd
    rows = []
    for a, b in pairs:
        rows.append(E.paired_compare(likelihood_df, a, b, metric, higher_better=higher_better))
    return pd.DataFrame(rows)
