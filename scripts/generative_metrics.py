"""
Metric B -- generative similarity for bout-sequence models.

One-step predictive likelihood (Metric A, evaluate.py) rewards good conditional
predictions but is blind to ROLLOUT DRIFT: a model can predict each next bout well
yet, run free, accumulate small biases into sequences whose long-range structure
(autocorrelation, streaks) is wrong. Metric B compares FREE-RUN sampled sequences to
the real held-out ones. It is also the only option for a black-box sampler with no
tractable density.

PROTOCOL (leak-free, stimulus-matched)
--------------------------------------
For each held-out fish we teacher-force the STIMULUS (feed the model that fish's real
per-bout stimulus covariates) and free-run the BEHAVIOUR, drawing ``R`` independent
rollouts matched to the real bout count. Every metric is computed per fish (real vs
generated), then aggregated across fish -- never pooling bouts as i.i.d.

PANEL (each a per-fish scalar; lower = closer to real unless noted)
  1  |dtheta| Wasserstein-1          turn-magnitude fidelity
  2  symbol distribution JS          marginal F/L/R bias
  3  inter-bout interval Wasserstein  bout rate (dt resampled from pool -> realism of pool for this fish)
  4  travel distance Wasserstein     path length (likewise pool-based)
  5  transition-matrix row-JS        one-step dependence
  6  turn-sign ACF L2                MULTI-STEP structure / rollout drift  (primary drift detector)
  7  run-length Wasserstein          persistent-turning / streak blow-up
  8  bigram-entropy |gap|            sequential predictability beyond unigram
  9  stimulus tuning curve MSE + following-index gap   the functional stimulus test

OMNIBUS (model-agnostic, see only samples)
  * classifier two-sample test (C2ST) AUC on windowed features, GroupKFold BY FISH
    (0.5 = indistinguishable, ->1 = distinguishable).
  * MMD^2 between per-fish feature vectors with a fish-level permutation test.

Reuses the model adapters and records from evaluate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E

EPS = 1e-12


# =========================================================================== #
#  Distribution helpers
# =========================================================================== #
def _js_divergence(p, q, base=2):
    """Jensen-Shannon divergence between two discrete distributions (in bits)."""
    p = np.asarray(p, float) + EPS; p /= p.sum()
    q = np.asarray(q, float) + EPS; q /= q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return np.sum(a * (np.log(a) - np.log(b)))
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(js / np.log(base))


def _symbol_freq(sym, k=3):
    h = np.bincount(np.asarray(sym, int), minlength=k)[:k].astype(float)
    return h / max(h.sum(), 1)


def _transition_matrix(sym, k=3):
    P = np.zeros((k, k))
    s = np.asarray(sym, int)
    for i in range(len(s) - 1):
        P[s[i], s[i + 1]] += 1
    row = P.sum(1, keepdims=True)
    return P / np.maximum(row, 1)


def _turn_sign(sym):
    """+1 for Right (2), -1 for Left (1), 0 for Forward (0)."""
    s = np.asarray(sym, int)
    out = np.zeros(len(s))
    out[s == 2] = 1.0
    out[s == 1] = -1.0
    return out


def _acf(x, lags):
    """Autocorrelation of a 1-D signal at the given lags (mean-removed)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom < EPS:
        return np.zeros(len(lags))
    return np.array([np.sum(x[:-lag] * x[lag:]) / denom if lag < len(x) else 0.0
                     for lag in lags])


def _run_lengths(sym):
    """Lengths of maximal runs of identical symbols."""
    s = np.asarray(sym, int)
    if len(s) == 0:
        return np.array([0])
    change = np.flatnonzero(np.diff(s) != 0) + 1
    bounds = np.concatenate([[0], change, [len(s)]])
    return np.diff(bounds).astype(float)


def _bigram_cond_entropy(sym, k=3):
    """Conditional entropy H(x_t | x_{t-1}) in bits."""
    P = np.zeros((k, k))
    s = np.asarray(sym, int)
    for i in range(len(s) - 1):
        P[s[i], s[i + 1]] += 1
    row = P.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(P > 0, P / np.maximum(row, 1), 0)
        logc = np.where(cond > 0, np.log2(cond), 0)
    prow = (row.ravel() / max(row.sum(), 1))
    return float(-np.sum(prow[:, None] * cond * logc))


# =========================================================================== #
#  Stimulus tuning curve (the functional following test)
# =========================================================================== #
def tuning_curve(dtheta, stim_angle, moving, n_bins=7, bin_edges=None):
    """P(rightward turn | binned egocentric stimulus bearing) on MOVING bouts.

    A following fish turns toward the dots: sign(dtheta) tracks sign(stim_angle)
    (+ = dots on the fish's left favour dtheta>0). Returns (bin_centers, curve,
    following_index) where curve[i] = P(dtheta>0 | bearing in bin i) and
    following_index = P(sign(dtheta)==sign(stim_angle)) - 0.5 over moving turn bouts.
    """
    dtheta = np.asarray(dtheta, float)
    phi = np.asarray(stim_angle, float)
    mv = np.asarray(moving, bool) & np.isfinite(phi)
    if bin_edges is None:
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    curve = np.full(len(centers), np.nan)
    if mv.sum() >= 5:
        idx = np.clip(np.digitize(phi[mv], bin_edges) - 1, 0, len(centers) - 1)
        right = (dtheta[mv] > 0).astype(float)
        for i in range(len(centers)):
            sel = idx == i
            if sel.sum() >= 3:
                curve[i] = right[sel].mean()
    turn = mv & (np.abs(dtheta) > E.DTHETA0)
    if turn.sum() >= 5:
        toward = (np.sign(dtheta[turn]) == np.sign(phi[turn])).mean()
        fidx = float(toward - 0.5)
    else:
        fidx = np.nan
    return centers, curve, fidx


# =========================================================================== #
#  Sampling
# =========================================================================== #
def sample_rollouts(model, rec, rng, R=20):
    """R stimulus-matched free-run rollouts for one fish (each length rec.n)."""
    return [model.sample(rec, rng) for _ in range(R)]


# =========================================================================== #
#  Per-fish panel
# =========================================================================== #
def panel_for_fish(model, rec, rng, R=20, acf_lags=(1, 2, 3, 4, 5, 8, 12)):
    """Compute the Metric-B panel for one held-out fish (mean over R rollouts)."""
    real_dth, real_sym = rec.dtheta, rec.symbol
    real_dt = rec.dt_s[np.isfinite(rec.dt_s)]
    real_dist = rec.distance_px[np.isfinite(rec.distance_px)]
    real_T = _transition_matrix(real_sym)
    real_acf = _acf(_turn_sign(real_sym), acf_lags)
    real_rl = _run_lengths(real_sym)
    real_H = _bigram_cond_entropy(real_sym)
    _, real_curve, real_fidx = tuning_curve(real_dth, rec.stim_angle, rec.moving)
    # static-stimulus split: stimulus present but NOT moving (exclude no-stimulus bouts).
    # tuning_curve's `moving` arg is just a boolean selector, so this is the same
    # following index restricted to static-dot bouts.
    static_mask = np.isfinite(rec.stim_angle) & ~np.asarray(rec.moving, bool)
    _, _, real_fidx_static = tuning_curve(real_dth, rec.stim_angle, static_mask)

    rolls = sample_rollouts(model, rec, rng, R=R)
    acc = {k: [] for k in ["abs_dtheta_w1", "abs_dtheta_ks", "symbol_js", "dt_w1",
                            "dist_w1", "trans_js", "acf_l2", "runlen_w1",
                            "bigram_H_gap", "tuning_mse", "following_gap"]}
    gcurves = []
    for g in rolls:
        gd, gs = g["dtheta"], g["symbol"]
        acc["abs_dtheta_w1"].append(wasserstein_distance(np.abs(real_dth), np.abs(gd)))
        acc["abs_dtheta_ks"].append(ks_2samp(np.abs(real_dth), np.abs(gd)).statistic)
        acc["symbol_js"].append(_js_divergence(_symbol_freq(real_sym), _symbol_freq(gs)))
        gdt = g["dt_s"][np.isfinite(g["dt_s"])]; gdist = g["distance_px"][np.isfinite(g["distance_px"])]
        acc["dt_w1"].append(wasserstein_distance(real_dt, gdt) if len(gdt) and len(real_dt) else np.nan)
        acc["dist_w1"].append(wasserstein_distance(real_dist, gdist) if len(gdist) and len(real_dist) else np.nan)
        gT = _transition_matrix(gs)
        acc["trans_js"].append(np.mean([_js_divergence(real_T[i], gT[i]) for i in range(3)]))
        acc["acf_l2"].append(float(np.sqrt(np.mean((real_acf - _acf(_turn_sign(gs), acf_lags)) ** 2))))
        acc["runlen_w1"].append(wasserstein_distance(real_rl, _run_lengths(gs)))
        acc["bigram_H_gap"].append(abs(real_H - _bigram_cond_entropy(gs)))
        _, gcurve, gfidx = tuning_curve(gd, rec.stim_angle, rec.moving)
        gcurves.append(gcurve)
        both = np.isfinite(real_curve) & np.isfinite(gcurve)
        acc["tuning_mse"].append(float(np.mean((real_curve[both] - gcurve[both]) ** 2)) if both.any() else np.nan)
        acc["following_gap"].append(abs(real_fidx - gfidx) if np.isfinite(real_fidx) and np.isfinite(gfidx) else np.nan)

    row = dict(fish_id=rec.fish_id, model=model.name, n_bouts=rec.n,
               real_following_index=real_fidx,
               gen_following_index=float(np.nanmean([tuning_curve(g["dtheta"], rec.stim_angle, rec.moving)[2] for g in rolls])),
               real_following_index_static=real_fidx_static,
               gen_following_index_static=float(np.nanmean([tuning_curve(g["dtheta"], rec.stim_angle, static_mask)[2] for g in rolls])))
    for k, v in acc.items():
        row[k] = float(np.nanmean(v))
        row[k + "_std"] = float(np.nanstd(v))
    return row


# =========================================================================== #
#  Windowed features (for the C2ST / MMD omnibus)
# =========================================================================== #
def window_features(dtheta, sym, W=60, step=None, acf_lags=(1, 2, 3)):
    """Feature vectors for sliding windows: symbol uni/bigram freqs + |dtheta|
    summary + turn-sign ACF + mean run-length. Returns (n_windows, d)."""
    step = step or W
    dtheta = np.asarray(dtheta, float); sym = np.asarray(sym, int)
    feats = []
    for a in range(0, max(1, len(sym) - W + 1), step):
        w_sym = sym[a:a + W]; w_dth = np.abs(dtheta[a:a + W])
        if len(w_sym) < W:
            break
        uni = _symbol_freq(w_sym)
        Tm = _transition_matrix(w_sym).ravel()
        mag = [w_dth.mean(), w_dth.std(), np.median(w_dth), np.quantile(w_dth, 0.9)]
        acf = _acf(_turn_sign(w_sym), acf_lags)
        rl = [_run_lengths(w_sym).mean()]
        feats.append(np.concatenate([uni, Tm, mag, acf, rl]))
    return np.array(feats) if feats else np.zeros((0, 3 + 9 + 4 + len(acf_lags) + 1))


def c2st_auc(windows_by_fish, seed=0):
    """Classifier two-sample test: can a classifier tell real from generated windows,
    in a way that GENERALISES to a held-out fish? GroupKFold by fish, mean ROC-AUC.

    windows_by_fish : dict fish_id -> (real_feats (nr,d), gen_feats (ng,d)).
    0.5 => indistinguishable (good); ->1 => the model's samples are detectable.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    X, y, grp = [], [], []
    for fid, (rf, gf) in windows_by_fish.items():
        if len(rf) == 0 or len(gf) == 0:
            continue
        X.append(rf); y.append(np.ones(len(rf))); grp.append(np.full(len(rf), fid))
        X.append(gf); y.append(np.zeros(len(gf))); grp.append(np.full(len(gf), fid))
    if len(X) < 2:
        return np.nan, []
    X = np.vstack(X); y = np.concatenate(y); grp = np.concatenate(grp)
    n_groups = len(np.unique(grp))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    aucs = []
    for tr, te in gkf.split(X, y, grp):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return (float(np.mean(aucs)) if aucs else np.nan), aucs


# Names for the window_features vector, IN ORDER: concat([uni(3), Tm(9), mag(4),
# acf(3), rl(1)]) = 20 dims (window_features default acf_lags=(1,2,3)). Tm is the
# row-major flatten of the 3x3 P[from, to] symbol-transition matrix (F=0,L=1,R=2).
C2ST_FEATURE_NAMES = [
    "uni_F", "uni_L", "uni_R",
    "T_FF", "T_FL", "T_FR", "T_LF", "T_LL", "T_LR", "T_RF", "T_RL", "T_RR",
    "absdth_mean", "absdth_std", "absdth_median", "absdth_q90",
    "acf_lag1", "acf_lag2", "acf_lag3",
    "runlen_mean",
]


def c2st_importances(windows_by_fish, feature_names=None, seed=0):
    """Mean RandomForest feature importances across the SAME GroupKFold-by-fish folds
    as c2st_auc. Returns a pandas Series (index = feature names; values sum to ~1).

    A high-importance feature is the axis on which real and generated windows differ
    most -- i.e. where the model's samples are least realistic. Averaging over folds
    (each RF trained on ~12 fish) mirrors c2st_auc, so the importances correspond to
    the reported AUC. Caveat: sklearn impurity importances are biased toward
    high-variance features -- read as "where the classifier looks", not an effect size.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    X, y, grp = [], [], []
    for fid, (rf, gf) in windows_by_fish.items():
        if len(rf) == 0 or len(gf) == 0:
            continue
        X.append(rf); y.append(np.ones(len(rf))); grp.append(np.full(len(rf), fid))
        X.append(gf); y.append(np.zeros(len(gf))); grp.append(np.full(len(gf), fid))
    names = feature_names or C2ST_FEATURE_NAMES
    if len(X) < 2:
        return pd.Series(np.full(len(names), np.nan), index=names, name="importance")
    X = np.vstack(X); y = np.concatenate(y); grp = np.concatenate(grp)
    n_groups = len(np.unique(grp))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    imps = []
    for tr, te in gkf.split(X, y, grp):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        imps.append(clf.feature_importances_)
    if not imps:
        return pd.Series(np.full(len(names), np.nan), index=names, name="importance")
    mean_imp = np.mean(imps, axis=0)
    if len(names) != len(mean_imp):                       # guard if acf_lags customised
        names = [f"f{i}" for i in range(len(mean_imp))]
    return pd.Series(mean_imp, index=names, name="importance")


def c2st_negative_control(windows_by_fish, seed=0):
    """Protocol sanity check: relabel each fish's REAL windows into two RANDOM halves
    and run the same GroupKFold-by-fish classifier. Both 'classes' are real windows
    from the same distribution, so the AUC should be ~0.5. A markedly higher value
    would mean the C2ST pipeline separates real-from-real (fish idiosyncrasy leaking
    across the fold split), which would inflate the real-vs-generated AUC. The split
    is RANDOM (not temporal) on purpose -- a temporal split would test within-fish
    nonstationarity instead. Returns the mean AUC (float)."""
    rng = np.random.default_rng(seed)
    pseudo = {}
    for fid, (rf, gf) in windows_by_fish.items():
        if len(rf) < 4:
            continue
        idx = rng.permutation(len(rf))
        h = len(rf) // 2
        pseudo[fid] = (rf[idx[:h]], rf[idx[h:]])
    auc, _ = c2st_auc(pseudo, seed=seed)
    return auc


def mmd2_permutation(real_feats, gen_feats, gamma=None, n_perm=2000, seed=0):
    """Unbiased MMD^2 between two feature sets (per-fish vectors) with an RBF kernel
    and a label-permutation test. Returns (mmd2, p_value)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(real_feats, float); Y = np.asarray(gen_feats, float)
    if len(X) < 2 or len(Y) < 2:
        return np.nan, np.nan
    Z = np.vstack([X, Y])
    Z = (Z - Z.mean(0)) / (Z.std(0) + EPS)
    n, m = len(X), len(Y)
    if gamma is None:
        # median heuristic on pairwise sq distances
        from scipy.spatial.distance import pdist
        d2 = pdist(Z, "sqeuclidean")
        gamma = 1.0 / (np.median(d2) + EPS)
    K = np.exp(-gamma * _sq_dists(Z))

    def _mmd2(idx_x, idx_y):
        Kxx = K[np.ix_(idx_x, idx_x)]; Kyy = K[np.ix_(idx_y, idx_y)]
        Kxy = K[np.ix_(idx_x, idx_y)]
        nx, ny = len(idx_x), len(idx_y)
        sx = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1))
        sy = (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1))
        sxy = Kxy.mean()
        return sx + sy - 2 * sxy

    obs = _mmd2(np.arange(n), np.arange(n, n + m))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n + m)
        if _mmd2(perm[:n], perm[n:]) >= obs:
            count += 1
    return float(obs), float((count + 1) / (n_perm + 1))


def _sq_dists(Z):
    sq = np.sum(Z * Z, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)


# =========================================================================== #
#  CV driver for Metric B
# =========================================================================== #
def cross_validate_generative(records, model_factories, folds, cfg=None, R=20,
                              seed=0, W=60, collect_windows=True, verbose=True,
                              progress=None):
    """Leave-3-fish-out generative-similarity CV.

    Returns (panel_df, windows_by_model). ``panel_df`` has one row per (fish, model)
    with the panel scalars; ``windows_by_model`` maps model name -> {fish_id:
    (real_feats, gen_feats)} for the C2ST/MMD omnibus. Models are refit per fold so
    the generator for a fish never trained on that fish.
    """
    import pandas as pd
    cfg = cfg or {}
    rng = np.random.default_rng(seed)
    rows = []
    windows_by_model = {}
    total = len(folds) * len(model_factories)      # for the optional progress callback
    done = 0
    for j, (train_ids, test_ids) in enumerate(folds):
        train = [records[f] for f in train_ids]
        ctx = E.make_context(train, cfg)
        for factory in model_factories:
            model = factory()
            skip = None
            if not getattr(model, "can_sample", True):   # baselines don't generate
                skip = "not generative"
            elif model.requires_absdict and any(records[f].u_raw is None for f in train_ids):
                skip = "rich features unavailable"
            if skip is None:
                try:
                    model.fit(train, ctx)
                except Exception as ex:            # e.g. rich without abs_dict
                    skip = str(ex)
            if skip is None:
                wm = windows_by_model.setdefault(model.name, {})
                for f in test_ids:
                    rec = records[f]
                    rows.append(panel_for_fish(model, rec, rng, R=R))
                    if collect_windows:
                        real_w = window_features(rec.dtheta, rec.symbol, W=W)
                        gen = model.sample(rec, rng)
                        gen_w = window_features(gen["dtheta"], gen["symbol"], W=W)
                        wm[f] = (real_w, gen_w)
            elif verbose:
                print(f"  [fold {j}] skip {model.name}: {skip}")
            done += 1
            if progress is not None:
                progress(done, total, f"fold {j} · {model.name}")
        if verbose:
            print(f"  fold {j}: generative panel for test fish {test_ids}")
    return pd.DataFrame(rows), windows_by_model


PANEL_METRICS = ["abs_dtheta_w1", "symbol_js", "dt_w1", "dist_w1", "trans_js",
                 "acf_l2", "runlen_w1", "bigram_H_gap", "tuning_mse", "following_gap"]


def summarize_panel(panel_df, metrics=PANEL_METRICS, n_boot=10000, seed=0):
    """Per-model mean of each panel metric across fish with fish-level bootstrap CI
    (lower = closer to real)."""
    import pandas as pd
    out = []
    for model, g in panel_df.groupby("model"):
        row = dict(model=model, n_fish=len(g))
        for mt in metrics:
            if mt in g.columns:
                mean, lo, hi = E.fish_bootstrap_ci(g[mt].to_numpy(), n_boot=n_boot, seed=seed)
                row[mt] = mean; row[mt + "_lo"] = lo; row[mt + "_hi"] = hi
        out.append(row)
    return pd.DataFrame(out)
