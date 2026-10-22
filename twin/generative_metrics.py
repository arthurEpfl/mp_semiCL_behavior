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

from twin import paths
from twin import evaluate as E

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


def _transition_matrix(sym, k=3, runs=None):
    """Row-normalised symbol transition matrix. ``runs`` drops cross-boundary transitions --
    a bout before a five-minute stop does not 'transition' into the one after it."""
    P = np.zeros((k, k))
    s = np.asarray(sym, int)
    ok = (np.ones(max(len(s) - 1, 0), bool) if runs is None
          else np.diff(np.asarray(runs)) == 0)
    for i in range(len(s) - 1):
        if ok[i]:
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


def _pair_mask(runs, lag, n):
    """Boolean over the ``n - lag`` lag-``lag`` pairs: True where both ends share a run.

    ``runs`` is a per-bout run id (see ``clean_data/bouts_clean.csv``). A run is a maximal
    stretch with no >10 s quiescence gap, so a pair that straddles a boundary joins bouts
    separated by minutes of stillness -- adjacent in the array, unrelated in the fish.

    WHY THIS MATTERS AND WHY IT IS ASYMMETRIC: a rollout is one continuous sequence with no
    boundaries, so its acf uses every pair. Masking the real side to `active` and then
    handing a SPLICED array to a flat estimator biases only the real reference. Measured on
    dtheta: lag 1 reads +0.0161 spliced against -0.0052 run-aware -- the bias is larger than
    the signal and flips its sign, and by lag 12 12.3% of pairs straddle a break.
    """
    r = np.asarray(runs)
    return r[:-lag] == r[lag:] if lag < n else np.zeros(0, bool)


def _acf(x, lags, runs=None):
    """Autocorrelation of a 1-D signal at the given lags (mean-removed).

    With ``runs``, pairs that straddle a run boundary are dropped and the sum is rescaled by
    ``(n - lag) / n_valid`` so the normalisation matches the flat convention. With a single
    run that rescale is 1.0, so ``runs=None`` and an all-one-run ``runs`` agree exactly and
    every archived number is reproduced.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.sum(x * x)
    n = len(x)
    if denom < EPS:
        return np.zeros(len(lags))
    if runs is None:
        return np.array([np.sum(x[:-lag] * x[lag:]) / denom if lag < n else 0.0
                         for lag in lags])
    out = []
    for lag in lags:
        if lag >= n:
            out.append(0.0)
            continue
        m = _pair_mask(runs, lag, n)
        nv = int(m.sum())
        if nv == 0:
            out.append(0.0)
            continue
        s = np.sum(x[:-lag][m] * x[lag:][m]) * ((n - lag) / nv)
        out.append(s / denom)
    return np.array(out)


def _run_lengths(sym, runs=None):
    """Lengths of maximal runs of identical symbols.

    With ``runs``, symbol runs are not allowed to span a quiescence boundary -- otherwise a
    fish that stops mid-run and resumes on the same symbol reads as one long run.
    """
    s = np.asarray(sym, int)
    if len(s) == 0:
        return np.array([0])
    if runs is None:
        change = np.flatnonzero(np.diff(s) != 0) + 1
        bounds = np.concatenate([[0], change, [len(s)]])
        return np.diff(bounds).astype(float)
    r = np.asarray(runs)
    change = np.flatnonzero((np.diff(s) != 0) | (np.diff(r) != 0)) + 1
    bounds = np.concatenate([[0], change, [len(s)]])
    return np.diff(bounds).astype(float)


def _bigram_cond_entropy(sym, k=3, runs=None):
    """Conditional entropy H(x_t | x_{t-1}) in bits. ``runs`` drops cross-boundary bigrams."""
    P = np.zeros((k, k))
    s = np.asarray(sym, int)
    ok = (np.ones(max(len(s) - 1, 0), bool) if runs is None
          else np.diff(np.asarray(runs)) == 0)
    for i in range(len(s) - 1):
        if ok[i]:
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
# --------------------------------------------------------------------------- #
# RE-ANCHORED stimulus for evaluation rollouts
# --------------------------------------------------------------------------- #
# The rig anchors each block's dot track to the fish's own pose at block onset, so every real
# fish starts every block at the programme's onset distance (median ~73 px). A rollout that
# REPLAYS the recorded dots (rec.dot_cx[t]) puts the simulated fish -- which is somewhere else --
# at whatever distance those dots happen to be: measured on S3_adt_rej, median 121 px and 49.8% of
# block onsets beyond 121 px (real 1.5%), so every fish-to-dot metric scored the placement as
# much as the behaviour. Evaluation rollouts are therefore re-anchored to the SIMULATED fish at
# each block onset (stim_rollout.RecordAnchoredProvider), read at the record's own frame per bout
# so the block/condition masks still line up with rec. `reanchor=False` reproduces the old replay.
REANCHOR_EVAL = True
_PROGRAM_CACHE = {}
_PROGRAM_FILES = (paths.PROGRAM, paths.TRAJECTORY)


def eval_program():
    """The recorded stimulus programme (identical for all 15 fish), loaded once per process. None
    if the file is absent -- sampling then falls back to replaying the recorded dots, loudly."""
    if "p" not in _PROGRAM_CACHE:
        path = next((c for c in _PROGRAM_FILES if c.exists()), None)
        if path is None:
            import warnings
            warnings.warn("stimulus programme not found -- evaluation rollouts will REPLAY the "
                          "recorded dot positions (block-onset placement artefact NOT removed)")
            _PROGRAM_CACHE["p"] = None
        else:
            from twin import stim_program as SP
            _PROGRAM_CACHE["p"] = SP.load_program(str(path))
    return _PROGRAM_CACHE["p"]


def _stim_kw(model, rec, reanchor=None):
    """``model.sample`` kwargs: a FRESH re-anchoring provider (it latches per-rollout anchors) when
    the model can take one, else nothing. Consumes no RNG, so a rollout's random streams are
    unchanged -- only where the dots are changes."""
    if not (REANCHOR_EVAL if reanchor is None else reanchor):
        return {}
    cf = getattr(model, "_cfg_feat", None) or {}
    if not (cf.get("joint") and getattr(model, "closed_loop", False)
            and getattr(model, "stim_loop", False) and int(cf.get("K_stim", 1)) == 1
            and getattr(rec, "frame_onset", None) is not None):
        return {}
    from twin import fit_MLP_mix as MX
    if cf.get("stim_mode") not in MX.ONLINE_STIM_MODES:
        return {}
    prog = eval_program()
    if prog is None:
        return {}
    from twin import stim_rollout as SR
    return {"stim_provider": SR.RecordAnchoredProvider(prog, rec)}


def sample_rollouts(model, rec, rng, R=20, reanchor=None):
    """R stimulus-matched free-run rollouts for one fish (each length rec.n).

    Models that set ``batched_sample`` generate all R in ONE batched call. That is not an
    optimisation detail for the recurrent family, it is the difference between usable and not: a
    GRU rollout is 4700 SEQUENTIAL steps and at batch 1 it is Python-dispatch bound, so the panel
    alone took longer than the whole fit with the GPU idle. Every other model keeps the original
    loop over the SAME rng, so their draws are unchanged bout for bout.
    """
    if getattr(model, "batched_sample", False):
        return model.sample_batch(rec, R, rng)
    return [model.sample(rec, rng, **_stim_kw(model, rec, reanchor)) for _ in range(R)]


# --- approach / proximity to the dots (trajectory-level; needs positions + dot geometry) --- #
NEAR_DOT_PX = 40.0     # "near a dot" threshold (~11 mm) for time_near_dot
FAR_DOT_PX = 120.0     # "far from a dot" threshold for time_far / wander


def _mean_approach(dist_to_dot, use):
    """Mean per-bout REDUCTION in fish-to-dot distance over consecutive usable bouts
    (positive = the fish closes on the dot). ``use`` selects bouts (dot active + finite)."""
    d = np.asarray(dist_to_dot, float)
    u = np.asarray(use, bool)
    pair = u[:-1] & u[1:] & np.isfinite(d[:-1]) & np.isfinite(d[1:])
    if not pair.any():
        return np.nan
    return float(np.mean(d[:-1][pair] - d[1:][pair]))


# =========================================================================== #
#  Per-fish panel
# =========================================================================== #
LATE_BLOCK_S = 40.0        # seconds into a stimulus block before dot-distance is informative
_FPS = 30.0


def _time_in_block(rec):
    """(n,) seconds since the current stimulus block began, per bout.

    A block boundary is any change in the stimulus condition (N, moving). Needed because the
    shoal is PLACED relative to the real fish at block onset, so early-block fish-to-shoal
    distance is an initial condition, not behaviour.

    Delegates to ``fit_MLP_border.block_onsets``, which is also what the K_stim FEATURES are
    built from -- so the block structure a model is told about and the block structure this
    metric judges it on cannot drift apart."""
    from twin import fit_MLP_border as _MB
    start_fr, _bid, fr = _MB.block_onsets(rec, fps=_FPS)
    return (fr - start_fr) / _FPS         # ONE division: bit-identical to the pre-refactor code


QUIESC_S = (5.0, 20.0, 60.0)      # inter-bout gaps that count as quiescence
RATE_BIN_S = 60.0                 # wall-clock bin for the bout-rate autocorrelation


def _quiesc_frac(dt, thr):
    """Fraction of ELAPSED TIME spent in inter-bout gaps longer than ``thr`` seconds.

    Time-weighted, not bout-weighted: real fish put 24.4% of the session into gaps > 20 s
    from only ~0.6% of their bouts, so a bout-weighted version reads as a rounding error.
    """
    dt = np.asarray(dt, float); dt = dt[np.isfinite(dt)]
    tot = float(dt.sum())
    return float(dt[dt > thr].sum() / tot) if tot > 0 else np.nan


def _rate_acf(dt, lags_min=(1, 2, 3), bin_s=RATE_BIN_S):
    """Autocorrelation of the bout COUNT per ``bin_s`` bin, at lags in minutes.

    The slow arousal band lives here and nowhere in the per-bout features: real fish give
    0.51/0.33/0.22 at 1/2/3 min (0.41/0.27/0.17 once the 110 s stimulus cycle is removed),
    the shipped model 0.24/0.05/0.00.
    """
    dt = np.asarray(dt, float); dt = dt[np.isfinite(dt)]
    if len(dt) < 10:
        return [np.nan] * len(lags_min)
    t = np.cumsum(dt)
    r = np.histogram(t, bins=np.arange(0, t[-1] + bin_s, bin_s))[0].astype(float)
    r -= r.mean(); v = r.var()
    step = max(int(round(60.0 / bin_s)), 1)
    return [float((r[:-L * step] * r[L * step:]).mean() / v)
            if v > 0 and L * step < len(r) else np.nan for L in lags_min]


def win_acf1(series, W=60, runs=None):
    """Lag-1 autocorrelation of the WINDOW-MEAN series, WITHIN one trajectory.

    The statistic the `drift` mechanism exists to move, and the one every previous individuality
    mechanism scores exactly zero on: demeaning within the fish removes any static per-trajectory
    latent, so fish_re z / rad_indiv / rad_gate / re_bgain cannot touch it by construction.

    Measured (15 fish, generated over 3 replicates):
        channel     REAL             GEN              gap
        dt        +0.332 +- 0.050  -0.001 +- 0.016   6.3 sigma
        distance  +0.337 +- 0.046  -0.002 +- 0.018   6.9 sigma
        radial    +0.377 +- 0.050  +0.043 +- 0.016   6.4 sigma

    ``runs`` (per-bout run id) keeps each window inside one quiescence-free run and only pairs
    windows that are CONSECUTIVE within the same run -- otherwise, once the real side is masked
    to `active`, a window can straddle a multi-minute gap and two 'adjacent' windows can be far
    apart in time. A rollout has no gaps, so leaving it out biases only the real reference of
    the very statistic this metric exists to measure.

    NOTE ON REPLICATES: on a single replicate of 6 fish the radial value swung -0.035 to +0.117
    across seeds, which is larger than the effect being measured. Average over replicates before
    reading it -- the panel already does (R rollouts per held-out fish).
    """
    ch = np.asarray(series, float)
    W = int(W)
    if runs is None:
        nw = len(ch) // W
        if nw < 12:
            return np.nan
        w = ch[:nw * W].reshape(nw, W)
        w = w[np.isfinite(w).all(1)]
        if len(w) < 12:
            return np.nan
        x = w.mean(1); x = x - x.mean()
        d = float((x * x).sum())
        return float((x[1:] * x[:-1]).sum() / d) if d > 0 else np.nan

    # Run-aware: windows lie entirely inside one run, and only CONSECUTIVE windows of the
    # SAME run form a lag-1 pair. Demeaning stays global (within the fish), which is what
    # removes a static per-trajectory latent -- the property this statistic exists for.
    starts = _window_starts(len(ch), W, W, runs)
    r = np.asarray(runs)
    keep = [(a, ch[a:a + W]) for a in starts if np.isfinite(ch[a:a + W]).all()]
    if len(keep) < 12:
        return np.nan
    x = np.array([w.mean() for _, w in keep]) - np.mean([w.mean() for _, w in keep])
    a_idx = np.array([a for a, _ in keep])
    # a pair is valid when the two windows are adjacent in the kept list AND in the same run
    ok = (r[a_idx[1:]] == r[a_idx[:-1]]) & (a_idx[1:] - a_idx[:-1] == W)
    d = float((x * x).sum())
    if d <= 0 or ok.sum() < 6:
        return np.nan
    scale = (len(x) - 1) / max(int(ok.sum()), 1)
    return float((x[1:][ok] * x[:-1][ok]).sum() * scale / d)


def _win_chan_corr(dt, dist, dtheta, W=60):
    """Within-fish, ACROSS-window correlations of the per-window channel means (log units).

    Real fish: corr(log dt, log dist) = -0.386 and corr(log dt, |dth|) = +0.308 -- one slow
    state moves timing, step size and turning together. The three vigor latents are
    independent per channel by construction, so the model sits at ~0 on both.

    ⚠ DEPRECATED as a SCORED metric under the active-only / re-anchored setup, and dropped
    from the reported panel. It correlates per-window channel MEANS across windows, so it
    needs windows that are (a) inside one quiescence-free run and (b) adjacent in time rather
    than merely adjacent in the array -- neither of which holds once the real side is masked
    to `active` and spliced. It also reads the very coupling that the quiescence mask removes:
    the shared slow state moves dt, distance and turning together largely THROUGH stopping,
    and a model that never stops cannot express it. Left in place because it is still a useful
    DIAGNOSTIC on unmasked real data; it is `timing_report` (`cpl_dt_dist_gap`,
    `cpl_dt_turn_gap`) that should be ignored, not this function.
    """
    dt = np.asarray(dt, float); dist = np.abs(np.asarray(dist, float))
    th = np.abs(np.asarray(dtheta, float))
    n = min(len(dt), len(dist), len(th))
    a = [[np.mean(np.log(dt[i:i + W] + EPS)), np.mean(np.log(dist[i:i + W] + EPS)),
          np.mean(th[i:i + W])] for i in range(0, n - W + 1, W)]
    if len(a) < 4:
        return np.nan, np.nan
    M = np.asarray(a, float)
    M = M[np.isfinite(M).all(1)]
    if len(M) < 4 or np.any(M.std(0) <= 0):
        return np.nan, np.nan
    C = np.corrcoef(M, rowvar=False)
    return float(C[0, 1]), float(C[0, 2])


def panel_for_fish(model, rec, rng, R=20, acf_lags=(1, 2, 3, 4, 5, 8, 12), reanchor=None):
    """Compute the Metric-B panel for one held-out fish (mean over R rollouts)."""
    real_dth, real_sym = rec.dtheta, rec.symbol
    real_dt = rec.dt_s[np.isfinite(rec.dt_s)]
    real_dist = rec.distance_px[np.isfinite(rec.distance_px)]
    # --- TIMING / REALISM targets. These exist because the metric has now twice rewarded a
    # trade that degraded an uncounted property: kin_feedback won -0.0407 while halving the
    # distance dispersion, and e250 won -0.0113 while cutting quiescent time from 13.0% to
    # 6.9% of the session (real 24.4%) and finishing the 174-min stimulus programme in 142.
    real_quiesc = [_quiesc_frac(real_dt, s) for s in QUIESC_S]
    real_elapsed = float(np.sum(real_dt))
    # Run ids for the REAL side only. A rollout is one continuous sequence with no
    # quiescence gaps, so its flat estimate is already correct; the real side, once masked
    # to `active`, is spliced and must not pair bouts across a boundary. Passing runs only
    # here is what keeps the two comparable. None -> previous behaviour, bit-identical.
    _runs = getattr(rec, "run_id", None)
    real_rate_acf = _rate_acf(real_dt)
    real_cpl_dd, real_cpl_dth = _win_chan_corr(rec.dt_s, rec.distance_px, real_dth)
    # SLOW-STATE targets: the across-window autocorrelation the `drift` mechanism targets. Reported
    # as signed gen-minus-real gaps beside the AUC, because a mechanism that moves these while the
    # AUC stays put is the masking effect, not a dead mechanism -- and that distinction is
    # unreadable if the number lives only in a one-off script.
    _rr = _radial_of(rec, rec.x_px, rec.y_px)
    real_wacf = (win_acf1(np.log(np.clip(rec.dt_s, 1e-6, None)), runs=_runs),
                 win_acf1(np.log(np.clip(rec.distance_px, 1e-6, None)), runs=_runs),
                 win_acf1(_rr, runs=_runs) if _rr is not None else np.nan)
    real_dist_std = float(np.std(np.abs(real_dist))) if len(real_dist) else np.nan
    real_T = _transition_matrix(real_sym, runs=_runs)
    real_acf = _acf(_turn_sign(real_sym), acf_lags, runs=_runs)
    real_rl = _run_lengths(real_sym, runs=_runs)
    real_H = _bigram_cond_entropy(real_sym, runs=_runs)
    _, real_curve, real_fidx = tuning_curve(real_dth, rec.stim_angle, rec.moving)
    # static-stimulus split: stimulus present but NOT moving (exclude no-stimulus bouts).
    # tuning_curve's `moving` arg is just a boolean selector, so this is the same
    # following index restricted to static-dot bouts.
    static_mask = np.isfinite(rec.stim_angle) & ~np.asarray(rec.moving, bool)
    _, _, real_fidx_static = tuning_curve(real_dth, rec.stim_angle, static_mask)
    # HONEST-BEARING real reference. The gen side of the *_h metrics uses stim_angle_gen, which
    # the closed loop computes against the TRAVEL heading; rec.stim_angle is f_ori-referenced
    # (40.6 deg median apart, 10.3% of bouts >90 deg). Comparing indices across frames put a
    # nonzero floor under following_gap_h for a model that matches the real travel-frame law
    # exactly. stim_angle_travel (build_border_records) is the same bearing in the gen frame;
    # its NaN pattern is identical, so static_mask carries over. Metric definition change
    # (2026-08-11): *_h columns are not comparable with zips produced before it.
    _sa_t = getattr(rec, "stim_angle_travel", None)
    if _sa_t is not None:
        _sa_t = np.asarray(_sa_t, float)
        real_fidx_h = tuning_curve(real_dth, _sa_t, rec.moving)[2]
        real_fidx_static_h = tuning_curve(real_dth, _sa_t, static_mask)[2]
    else:
        real_fidx_h, real_fidx_static_h = real_fidx, real_fidx_static

    # fish-to-dot (approach/proximity) targets -- only if the record carries per-bout dot
    # geometry (add_dot_geometry.py) AND the rollout returns positions (loop models).
    has_dotgeom = getattr(rec, "dot_cx", None) is not None
    if has_dotgeom:
        dot_cx = np.asarray(rec.dot_cx, float); dot_cy = np.asarray(rec.dot_cy, float)
        real_dd = np.asarray(rec.dot_dist_px, float)
        dmask = np.isfinite(real_dd)
        moving_dot = np.asarray(rec.moving, bool) & dmask
        real_appr = _mean_approach(real_dd, moving_dot)
        # LATE-BLOCK mask. The rig places the shoal ~75 px from the REAL fish at each block
        # onset; a free-running generated fish is ~134 px away at that instant (= the
        # swapped-track null), and a 60 s block cannot close that at 2-4 px/bout. So the raw
        # dot-distance gap is ~2/3 an initial-condition artefact: +23.4 px over all bouts vs
        # +8.8 px conditioned on >=40 s into a block. Read the *_late metrics.
        late_mask = dmask & (_time_in_block(rec) >= LATE_BLOCK_S)

    rolls = sample_rollouts(model, rec, rng, R=R, reanchor=reanchor)
    acc = {k: [] for k in ["abs_dtheta_w1", "abs_dtheta_ks", "symbol_js", "dt_w1",
                            "dist_w1", "trans_js", "acf_l2", "runlen_w1",
                            "bigram_H_gap", "tuning_mse", "following_gap",
                            "following_gap_static", "following_gap_h", "following_gap_static_h",
                            "dot_dist_w1", "time_near_dot_gap", "approach_gap",
                            "dot_dist_bias", "time_far_gap",
                            "dot_dist_bias_late", "dot_dist_w1_late", "wall_frac_gap",
                            # --- timing / realism guards (see the real-side block above) ---
                            "quiesc5_gap", "quiesc20_gap", "quiesc60_gap", "clock_ratio",
                            "dt_tail20_gap", "rate_acf_1min_gap", "rate_acf_3min_gap",
                            "wacf_dt_gap", "wacf_dist_gap", "wacf_rad_gap",
                            "cpl_dt_dist_gap", "cpl_dt_turn_gap", "dist_std_ratio",
                            "gen_fwd_frac", "gen_absdth_median", "bounce_rate"]}
    gcurves = []
    for g in rolls:
        gd, gs = g["dtheta"], g["symbol"]
        acc["abs_dtheta_w1"].append(wasserstein_distance(np.abs(real_dth), np.abs(gd)))
        acc["abs_dtheta_ks"].append(ks_2samp(np.abs(real_dth), np.abs(gd)).statistic)
        acc["symbol_js"].append(_js_divergence(_symbol_freq(real_sym), _symbol_freq(gs)))
        gdt = g["dt_s"][np.isfinite(g["dt_s"])]; gdist = g["distance_px"][np.isfinite(g["distance_px"])]
        acc["dt_w1"].append(wasserstein_distance(real_dt, gdt) if len(gdt) and len(real_dt) else np.nan)
        acc["dist_w1"].append(wasserstein_distance(real_dist, gdist) if len(gdist) and len(real_dist) else np.nan)
        # --- TIMING / REALISM guards: signed gen-minus-real, so the SIGN says which way a
        # rung moved the property. Read these on the same line as the AUC.
        for _s, _r, _k in zip(QUIESC_S, real_quiesc,
                              ("quiesc5_gap", "quiesc20_gap", "quiesc60_gap")):
            acc[_k].append(_quiesc_frac(gdt, _s) - _r)
        acc["clock_ratio"].append(float(np.sum(gdt) / real_elapsed) if real_elapsed > 0 else np.nan)
        acc["dt_tail20_gap"].append(float(np.mean(gdt > 20.0) - np.mean(real_dt > 20.0))
                                    if len(gdt) and len(real_dt) else np.nan)
        _ga = _rate_acf(gdt)
        acc["rate_acf_1min_gap"].append(_ga[0] - real_rate_acf[0])
        acc["rate_acf_3min_gap"].append(_ga[2] - real_rate_acf[2])
        _cdd, _cdth = _win_chan_corr(g["dt_s"], g["distance_px"], gd)
        acc["cpl_dt_dist_gap"].append(_cdd - real_cpl_dd)
        acc["cpl_dt_turn_gap"].append(_cdth - real_cpl_dth)
        # Signed gen-minus-real across-window acf gaps. Averaged over the R panel rollouts, which
        # is what makes them readable: a single replicate of the radial one is pure noise.
        _gr = _radial_of(rec, g.get("x_px"), g.get("y_px")) if "x_px" in g else None
        acc["wacf_dt_gap"].append(win_acf1(np.log(np.clip(g["dt_s"], 1e-6, None))) - real_wacf[0])
        acc["wacf_dist_gap"].append(
            win_acf1(np.log(np.clip(g["distance_px"], 1e-6, None))) - real_wacf[1])
        acc["wacf_rad_gap"].append((win_acf1(_gr) - real_wacf[2])
                                   if _gr is not None else np.nan)
        # dist_std is the model's dominant tell AND the defect that MASKS every other one
        # (an identical injected defect is worth +0.042 AUC alone, +0.001 alongside a 20%
        # marginal error). Track it as a ratio with an unambiguous target of 1.0.
        acc["dist_std_ratio"].append(float(np.std(np.abs(gdist)) / real_dist_std)
                                     if len(gdist) and real_dist_std > 0 else np.nan)
        # per-fish turn profile -> the ACROSS-fish spread of these is the individuality the
        # model does not have (real sd 0.054 / 6.7 deg, generated 0.004 / 0.6).
        acc["gen_fwd_frac"].append(float(np.mean(np.abs(gd) < E.DTHETA0)))
        acc["gen_absdth_median"].append(float(np.median(np.abs(gd))))
        acc["bounce_rate"].append(float(g.get("n_bounces", np.nan)) / max(len(gd), 1)
                                  if g.get("n_bounces") is not None else np.nan)
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
        gfidx_static = tuning_curve(gd, rec.stim_angle, static_mask)[2]
        acc["following_gap_static"].append(abs(real_fidx_static - gfidx_static)
                                           if np.isfinite(real_fidx_static) and np.isfinite(gfidx_static) else np.nan)
        # --- honest-bearing following: score turns vs the bearing the MODEL ACTUALLY REACTED TO
        # (the generated bearing for stim_loop models; the teacher-forced real bearing otherwise).
        # Fair to closed-loop-stimulus models, which react to their own bearing, not rec.stim_angle.
        # Frame-matched real reference: stim_angle_gen is TRAVEL-framed, so compare it against
        # the travel-framed real index; the teacher-forced fallback stays f_ori vs f_ori.
        gb = g.get("stim_angle_gen")
        rf_h, rfs_h = ((real_fidx, real_fidx_static) if gb is None
                       else (real_fidx_h, real_fidx_static_h))
        gb = rec.stim_angle if gb is None else gb
        gfidx_h = tuning_curve(gd, gb, rec.moving)[2]
        acc["following_gap_h"].append(abs(rf_h - gfidx_h)
                                      if np.isfinite(rf_h) and np.isfinite(gfidx_h) else np.nan)
        gfidx_static_h = tuning_curve(gd, gb, static_mask)[2]
        acc["following_gap_static_h"].append(abs(rfs_h - gfidx_static_h)
                                             if np.isfinite(rfs_h) and np.isfinite(gfidx_static_h) else np.nan)
        # --- approach / proximity: generated fish position vs the dots IT was shown ---------
        # A re-anchored rollout returns the dots it actually used (g['dot_cx']); only a replayed
        # one falls back to the real fish's track. The provider reads the programme at the
        # record's own frame, so the same bouts carry a dot on both sides; `gm` only guards a rare
        # bout where the two disagree. Real and generated are compared as distributions.
        if has_dotgeom and "x_px" in g and "y_px" in g and dmask.any():
            gdd = np.hypot(np.asarray(g["x_px"], float) - np.asarray(g.get("dot_cx", dot_cx), float),
                           np.asarray(g["y_px"], float) - np.asarray(g.get("dot_cy", dot_cy), float))
            gm = dmask & np.isfinite(gdd)
            acc["dot_dist_w1"].append(wasserstein_distance(real_dd[dmask], gdd[gm]))
            acc["time_near_dot_gap"].append(abs(np.mean(real_dd[dmask] < NEAR_DOT_PX)
                                                - np.mean(gdd[gm] < NEAR_DOT_PX)))
            acc["approach_gap"].append(abs(real_appr - _mean_approach(gdd, moving_dot & gm))
                                       if np.isfinite(real_appr) else np.nan)
            # DIRECTED wander metrics (approach_gap/dot_dist_w1 hide direction): + bias = gen further
            acc["dot_dist_bias"].append(float(np.mean(gdd[gm]) - np.mean(real_dd[dmask])))
            acc["time_far_gap"].append(abs(np.mean(real_dd[dmask] > FAR_DOT_PX)
                                           - np.mean(gdd[gm] > FAR_DOT_PX)))
            # LATE-BLOCK versions: the placement artefact has decayed, so these are the ones
            # that reflect the stimulus response rather than the block's initial condition.
            _lm = late_mask & gm
            if late_mask.any() and _lm.any():
                acc["dot_dist_bias_late"].append(
                    float(np.mean(gdd[_lm]) - np.mean(real_dd[late_mask])))
                acc["dot_dist_w1_late"].append(
                    wasserstein_distance(real_dd[late_mask], gdd[_lm]))
            else:
                acc["dot_dist_bias_late"].append(np.nan)
                acc["dot_dist_w1_late"].append(np.nan)
            # radial occupancy: a CONTAINMENT defect that otherwise leaks into dotdist_*
            grad = _radial_of(rec, g["x_px"], g["y_px"])
            rrad = _radial_of(rec, rec.x_px, rec.y_px)
            if grad is not None and rrad is not None:
                acc["wall_frac_gap"].append(float(np.mean(grad > 0.85) - np.mean(rrad > 0.85)))
            else:
                acc["wall_frac_gap"].append(np.nan)
        else:                                            # no dot geometry or no positions (non-loop)
            for _k in ("dot_dist_w1", "time_near_dot_gap", "approach_gap",
                       "dot_dist_bias", "time_far_gap",
                       "dot_dist_bias_late", "dot_dist_w1_late", "wall_frac_gap"):
                acc[_k].append(np.nan)

    row = dict(fish_id=rec.fish_id, model=model.name, n_bouts=rec.n,
               real_following_index=real_fidx,
               gen_following_index=float(np.nanmean([tuning_curve(g["dtheta"], rec.stim_angle, rec.moving)[2] for g in rolls])),
               real_following_index_static=real_fidx_static,
               gen_following_index_static=float(np.nanmean([tuning_curve(g["dtheta"], rec.stim_angle, static_mask)[2] for g in rolls])))
    for k, v in acc.items():
        # An all-NaN accumulator is a LEGITIMATE state, not a failure: the window-acf family
        # needs at least two non-overlapping 60-bout windows INSIDE one quiescence-free run,
        # and the most quiescent fish (4, 6, 10 on the active-only set) never have them. numpy
        # is right to warn about the mean of an empty slice, but it fires once per fish per
        # metric per model and buries the report under identical RuntimeWarnings. The value is
        # NaN either way; take it directly instead of provoking the warning to produce it.
        _v = np.asarray(v, float)
        _ok = np.isfinite(_v).any()
        row[k] = float(np.nanmean(_v)) if _ok else np.nan
        row[k + "_std"] = float(np.nanstd(_v)) if _ok else np.nan
    if has_dotgeom and dmask.any():                      # raw direction (real vs gen mean dist-to-dot)
        row["real_dot_dist_mean"] = float(np.mean(real_dd[dmask]))
        row["gen_dot_dist_mean"] = row["real_dot_dist_mean"] + row["dot_dist_bias"]
    else:
        row["real_dot_dist_mean"] = row["gen_dot_dist_mean"] = np.nan
    # real-side timing targets, so timing_report can form BETWEEN-FISH spreads and so the
    # gaps above can be read back as absolute levels rather than only as differences.
    row["real_quiesc5"], row["real_quiesc20"], row["real_quiesc60"] = real_quiesc
    row["real_wacf_dt"], row["real_wacf_dist"], row["real_wacf_rad"] = real_wacf
    row["real_fwd_frac"] = float(np.mean(np.abs(real_dth) < E.DTHETA0))
    row["real_absdth_median"] = float(np.median(np.abs(real_dth)))
    return row


def timing_report(panel_df, model=None):
    """Realism guards that the C2ST cannot see, summarised ACROSS fish.

    Companion to ``dispersion_report``. That one caught a rung buying the AUC by narrowing
    the model; this one catches a rung buying it by making the fish's TIMING wrong. Both
    failures have now happened (``kin_feedback`` and ``e250`` respectively), so read all
    three -- AUC, dispersion, timing -- on the same line.

    Everything here is a mean over held-out fish except ``*_sd_gen`` / ``*_sd_real``, which
    are spreads ACROSS fish: those are the individuality channel, and a per-fish mean cannot
    show it. NEVER read them off a 3-fish subset -- between-fish spread is 7-12x on window
    variance and that trap has already fired twice in this project.

    Targets: every ``*_gap`` -> 0.0, ``clock_ratio`` -> 1.0, ``dist_std_ratio`` -> 1.0,
    ``fwd_frac_sd_gen`` -> ``fwd_frac_sd_real`` (~0.054), ``absdth_median_sd_gen`` -> ~6.7.
    """
    df = panel_df if model is None else panel_df[panel_df["model"] == model]
    if not len(df):
        return {}
    out = {}
    for k in ("quiesc5_gap", "quiesc20_gap", "quiesc60_gap", "clock_ratio", "dt_tail20_gap",
              "rate_acf_1min_gap", "rate_acf_3min_gap", "cpl_dt_dist_gap", "cpl_dt_turn_gap",
              "wacf_dt_gap", "wacf_dist_gap", "wacf_rad_gap",
              "dist_std_ratio", "bounce_rate"):
        if k in df:
            out[k] = float(np.nanmean(df[k].values))
    for k, lab in (("fwd_frac", "fwd_frac"), ("absdth_median", "absdth_median")):
        for src, tag in ((f"gen_{k}", "gen"), (f"real_{k}", "real")):
            if src in df:
                out[f"{lab}_sd_{tag}"] = float(np.nanstd(df[src].values))
    return out


# =========================================================================== #
#  Windowed features (for the C2ST / MMD omnibus)
# =========================================================================== #
def _window_starts(n, W, step, runs=None, lead=0, tail=0):
    """Window start indices. With ``runs``, only windows lying ENTIRELY inside one run.

    A window that straddles a >10 s quiescence gap mixes two stretches of behaviour minutes
    apart and reports them as one 60-bout sample. A rollout has no gaps, so leaving them in
    biases only the real side. Measured cost of the restriction at W = 60: 267 runs still
    yield 42,195 windows covering 83.3 % of active bouts, and all 15 fish are represented.

    ``lead`` / ``tail``: keep only windows that start >= ``lead`` bouts into their run and end
    >= ``tail`` bouts before it ends (the STEADY-STATE selection, see ``INTERIOR``); without
    ``runs`` the whole series is one run.
    """
    starts = range(0, max(1, n - W + 1), step)
    if runs is None:
        return [a for a in starts if a + W <= n and a >= lead and n - (a + W) >= tail]
    r = np.asarray(runs)
    pos = np.zeros(n, dtype=int)                     # bouts since the run began
    for t in range(1, n):
        pos[t] = pos[t - 1] + 1 if r[t] == r[t - 1] else 0
    rem = np.zeros(n, dtype=int)                     # bouts left before the run ends
    for t in range(n - 2, -1, -1):
        rem[t] = rem[t + 1] + 1 if r[t] == r[t + 1] else 0
    return [a for a in starts if a + W <= n and r[a] == r[a + W - 1]
            and pos[a] >= lead and rem[a + W - 1] >= tail]


# STEADY-STATE window selection (lead, tail) in bouts. MEASURED (Data_cleanup_final.ipynb step 5):
# the first ~60 bouts after a > 10 s pause are slower, shorter-stepping and more central, and
# the last ~30 before one are more central again; a third of the run-aware windows are the FIRST
# window of a run (per-window dt_std 1.09 s vs 0.74 interior). A rollout has no pauses, so the
# ``traj_int`` C2ST pairs real windows starting >= 120 bouts into a run and ending >= 60 before
# it ends with generated windows starting at t >= 120 (the run clock, if any, has decayed).
INTERIOR = (120, 60)


def window_features(dtheta, sym, W=60, step=None, acf_lags=(1, 2, 3), runs=None):
    """Feature vectors for sliding windows: symbol uni/bigram freqs + |dtheta|
    summary + turn-sign ACF + mean run-length. Returns (n_windows, d).

    ``runs`` (per-bout run id) keeps every window inside a single quiescence-free run; see
    ``_window_starts``. ``runs=None`` reproduces the previous behaviour exactly."""
    step = step or W
    dtheta = np.asarray(dtheta, float); sym = np.asarray(sym, int)
    feats = []
    for a in _window_starts(len(sym), W, step, runs):
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


def _summ4(w):
    """[mean, std, median, q90] of the finite entries of a window (zeros if empty)."""
    w = np.asarray(w, float); w = w[np.isfinite(w)]
    return [w.mean(), w.std(), np.median(w), np.quantile(w, 0.9)] if len(w) else [0.0, 0.0, 0.0, 0.0]


def _acf1(w):
    """Lag-1 autocorrelation of a window (0 if degenerate). The traj features summarised
    distance/dt with moments ONLY, so burstiness was invisible to the classifier."""
    w = np.asarray(w, float); w = w[np.isfinite(w)]
    if len(w) < 3:
        return 0.0
    w = w - w.mean()
    d = float(np.sum(w * w))
    return float(np.sum(w[1:] * w[:-1]) / d) if d > 0 else 0.0


def window_features_traj(dtheta, sym, distance, dt, dot_dist=None, W=60, step=None,
                         acf_lags=(1, 2, 3), radial=None, runs=None, lead=0, tail=0):
    """TRAJECTORY C2ST features: everything ``window_features`` has (turns) PLUS per-window
    KINEMATIC summaries -- |distance| (mean/std/median/q90 + lag-1 acf), dt (same 5), and, when
    ``dot_dist`` is given (loop models with dot geometry), fish-to-dot distance (same 5). So this
    classifier 'sees' step sizes, timing and proximity, which the turn-only C2ST is blind to.
    Same windowing as ``window_features`` (aligned).

    ``radial`` (optional, loop models) = per-bout r/R in the dish, summarised as
    [mean, median, q90, frac>0.85]. Without it the classifier has NO occupancy channel, and a
    model that hugs the wall is only detectable indirectly through ``dotdist_*`` -- which
    misattributes a CONTAINMENT defect to the stimulus model. Measured: real fish are at
    r/R>0.85 on 1.2% of bouts, the dotb rollout on 8.5% (and worse during no-dot pauses)."""
    step = step or W
    dtheta = np.asarray(dtheta, float); sym = np.asarray(sym, int)
    dist = np.abs(np.asarray(distance, float)); dts = np.asarray(dt, float)
    dd = None if dot_dist is None else np.asarray(dot_dist, float)
    rad = None if radial is None else np.asarray(radial, float)
    feats = []
    for a in _window_starts(len(sym), W, step, runs, lead=lead, tail=tail):
        w_sym = sym[a:a + W]; w_dth = np.abs(dtheta[a:a + W])
        if len(w_sym) < W:
            break
        turn = np.concatenate([_symbol_freq(w_sym), _transition_matrix(w_sym).ravel(),
                               [w_dth.mean(), w_dth.std(), np.median(w_dth), np.quantile(w_dth, 0.9)],
                               _acf(_turn_sign(w_sym), acf_lags), [_run_lengths(w_sym).mean()]])
        kin = (_summ4(dist[a:a + W]) + [_acf1(dist[a:a + W])]
               + _summ4(dts[a:a + W]) + [_acf1(dts[a:a + W])])
        if dd is not None:
            kin = kin + _summ4(dd[a:a + W]) + [_acf1(dd[a:a + W])]
        if rad is not None:
            wr = rad[a:a + W]; wr = wr[np.isfinite(wr)]
            kin = kin + ([wr.mean(), np.median(wr), np.quantile(wr, 0.9), float(np.mean(wr > 0.85))]
                         if len(wr) else [0.0, 0.0, 0.0, 0.0])
        feats.append(np.concatenate([turn, kin]))
    ndim = (3 + 9 + 4 + len(acf_lags) + 1 + 10
            + (5 if dot_dist is not None else 0) + (4 if radial is not None else 0))
    return np.array(feats) if feats else np.zeros((0, ndim))


def rank_map(gen, real):
    """Map ``gen`` onto ``real``'s marginal by rank, preserving the generated ORDER.

    Removes the marginal as an explanatory variable so the residual AUC is about temporal
    structure alone. Needed because the block-bootstrap reference ladder resamples real bouts
    and therefore has EXACT marginals, while the model does not: raw, the shipped model looks
    like a surrogate preserving ~1 bout of structure; rank-equalised it looks like ~2.8. The
    difference is entirely marginals, and reading the raw number as 'memory' is a mistake.

    It is also the split the headline AUC cannot make on its own -- 'the model gets levels
    wrong' and 'the model gets within-fish temporal structure wrong' are different defects
    with different fixes. Report both numbers, never this one alone: an equalised AUC is not
    a model score, since a real model has to get its marginals right too.
    """
    g = np.asarray(gen, float); r = np.asarray(real, float)
    out = np.array(g, float)
    fg = np.isfinite(g); fr = np.isfinite(r)
    if fg.sum() < 2 or fr.sum() < 2:
        return out
    src = np.sort(r[fr])
    n, m = int(fg.sum()), len(src)
    # map rank i of n onto ORDER STATISTIC i*(m-1)/(n-1) of the real sample, not onto an
    # interpolated quantile: this makes the map exactly the identity when the two samples are
    # the same multiset, so a surrogate that already has correct marginals is left untouched.
    ranks = np.argsort(np.argsort(g[fg]))
    out[fg] = src[np.rint(ranks * (m - 1) / max(n - 1, 1)).astype(int)] if n > 1 else src[0]
    return out


def _as_reps(w):
    """Normalise either form to a list of replicate window-dicts.

    ``cross_validate_generative`` now returns a LIST of per-replicate dicts. Accepting both
    here keeps every existing caller (notebook cells, diag scripts) working unchanged."""
    return list(w) if isinstance(w, (list, tuple)) else [w]


def c2st_auc(windows_by_fish, seed=0):
    """Classifier two-sample test: can a classifier tell real from generated windows,
    in a way that GENERALISES to a held-out fish? GroupKFold by fish, mean ROC-AUC.

    windows_by_fish : dict fish_id -> (real_feats (nr,d), gen_feats (ng,d)), OR a list of
    such dicts (one per rollout replicate), in which case the per-replicate AUCs are
    averaged -- see ``c2st_auc_replicates`` for the spread as well.
    0.5 => indistinguishable (good); ->1 => the model's samples are detectable.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    if isinstance(windows_by_fish, (list, tuple)):
        m, _, per = c2st_auc_replicates(windows_by_fish, seed=seed)
        return m, per
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


def c2st_auc_replicates(windows_reps, seed=0):
    """Average the C2ST AUC over INDEPENDENT rollout replicates -> (mean, sd, per_rep).

    ``windows_reps`` : list of ``windows_by_fish`` dicts, one per replicate rollout.

    Why this and not "pool all replicates into one generated class": pooling makes the
    generated class R times larger than the real one, and the resulting class imbalance
    SHIFTS the AUC level (measured 0.790 -> 0.822 at R=4). Averaging balanced single-rollout
    AUCs leaves the level alone and cuts the standard error as 1/sqrt(R).

    Rollout randomness alone gives c2st_traj_auc an SD of 0.0173 at R=1 (13 seeds, one fitted
    checkpoint), i.e. SE of a two-model difference = 0.0245. So R=1 cannot resolve anything
    below ~0.035 AUC; R>=8 gives SE_diff ~ 0.012 and R>=24 resolves 0.01.
    """
    vals = [c2st_auc(w, seed=seed)[0] for w in _as_reps(windows_reps)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return np.nan, np.nan, []
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, vals


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

# window_features_traj = C2ST_FEATURE_NAMES + distance(4) + dt(4) [+ fish-to-dot(4) for loop
# models]. Use with c2st_importances(wins_traj[name], C2ST_TRAJ_FEATURE_NAMES) to see WHICH
# channel (e.g. dotdist_mean = the wandering) makes the trajectory most detectable.
C2ST_TRAJ_FEATURE_NAMES = C2ST_FEATURE_NAMES + [
    "dist_mean", "dist_std", "dist_median", "dist_q90", "dist_acf1",
    "dt_mean", "dt_std", "dt_median", "dt_q90", "dt_acf1",
    "dotdist_mean", "dotdist_std", "dotdist_median", "dotdist_q90", "dotdist_acf1",  # loop only
    "rad_mean", "rad_median", "rad_q90", "rad_frac_wall",                            # loop only
]


def traj_feature_names(has_dot=True, has_radial=True):
    """The names matching what ``window_features_traj`` actually returned, so grouped
    importances never silently fall back to f0..fN (which then KeyErrors on the caller)."""
    names = list(C2ST_FEATURE_NAMES) + [
        "dist_mean", "dist_std", "dist_median", "dist_q90", "dist_acf1",
        "dt_mean", "dt_std", "dt_median", "dt_q90", "dt_acf1"]
    if has_dot:
        names += ["dotdist_mean", "dotdist_std", "dotdist_median", "dotdist_q90", "dotdist_acf1"]
    if has_radial:
        names += ["rad_mean", "rad_median", "rad_q90", "rad_frac_wall"]
    return names


def traj_names_for_dim(d):
    """Resolve the traj feature names from the observed width, so a non-loop model (no
    positions -> no dot-distance and no radial block) never silently falls back to f0..fN."""
    for has_dot in (True, False):
        for has_rad in (True, False):
            nm = traj_feature_names(has_dot, has_rad)
            if len(nm) == d:
                return nm
    return [f"f{i}" for i in range(d)]


def traj_groups(names):
    """Partition traj feature names into the reporting groups (turn / distance / dt /
    dot-dist / radial). Radial is its own group because it is a CONTAINMENT defect, not a
    stimulus one -- lumping it into dot-dist is exactly the misattribution to avoid."""
    dist = [n for n in names if n.startswith("dist_")]
    dt = [n for n in names if n.startswith("dt_")]
    dd = [n for n in names if n.startswith("dotdist_")]
    rad = [n for n in names if n.startswith("rad_")]
    turn = [n for n in names if n not in dist + dt + dd + rad]
    return dict(turn=turn, distance=dist, dt=dt, **{"dot-dist": dd, "radial": rad})


def c2st_importances(windows_by_fish, feature_names=None, seed=0, kind="permutation",
                     n_repeats=5):
    """Mean feature importances across the SAME GroupKFold-by-fish folds as c2st_auc.
    Returns a pandas Series (index = feature names).

    ``kind='permutation'`` (default) scores each feature by the DROP IN HELD-OUT AUC when
    that feature is shuffled in the test fold -- an effect size on the reported metric.
    ``kind='impurity'`` is the old sklearn mean-decrease-in-Gini: cheaper, but biased toward
    high-variance/high-cardinality features and computed on the TRAINING split, so it says
    "where the forest looked", not "what made the samples detectable". The two disagree
    whenever features are correlated, which these are (moments of the same channel).

    Permutation importances are per-feature AUC drops; they do NOT sum to 1 and may be
    slightly negative for irrelevant features. Normalise before comparing group shares.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    windows_by_fish = _as_reps(windows_by_fish)[0]      # importances from one replicate
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
    if kind == "permutation":
        rng = np.random.default_rng(seed)
        for tr, te in gkf.split(X, y, grp):
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            base = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
            drop = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                d = 0.0
                for _ in range(n_repeats):
                    Xp = X[te].copy()
                    Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                    d += base - roc_auc_score(y[te], clf.predict_proba(Xp)[:, 1])
                drop[j] = d / n_repeats
            imps.append(drop)
        if imps:
            return pd.Series(np.mean(imps, axis=0),
                             index=(names if len(names) == X.shape[1]
                                    else [f"f{i}" for i in range(X.shape[1])]),
                             name="importance")
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


def dispersion_report(windows_by_fish, feature_names=None):
    """Per-group variance ratios (gen/real) and the between-fish variance share.

    The C2ST says HOW detectable a model is; it does not say whether the model is too
    NARROW or too WIDE, and those need opposite fixes. This does, in two numbers with
    unambiguous targets.

    Why it exists: the kin ladder's winning rung improved the AUC by -0.0407 while HALVING
    the generated distance dispersion (1.04 -> 0.52 against a real target of 1.00), because
    shrinking an AR history damps persistence and spread together. The AUC rewarded that
    trade; nothing reported at the time could see it. Read this next to every AUC.

    Returns dict with
      ``var_ratio``  {group: sum(var(gen)) / sum(var(real))} over z-scored window features;
                     1.0 = matched, <1 under-dispersed (too typical), >1 over-dispersed.
      ``between_fish_share`` {'real','gen'}: fraction of total window variance carried by
                     the per-fish means. A pooled model that samples per-trajectory latents
                     from the right prior should match this; a shortfall is individuality the
                     model fails to produce, and it is a direction that GENERALISES across
                     fish, so it inflates the C2ST.
      ``within_ratio`` {group: gen/real ratio of the WITHIN-fish, window-to-window variance}
      ``between_ratio`` {group: gen/real ratio of the BETWEEN-fish variance}

    WHY THE SPLIT EXISTS. ``var_ratio`` is POOLED variance -- ``R`` and ``G`` are vstacked over all
    fish -- so by the law of total variance it sums the within-fish window-to-window term and the
    between-fish term, which need OPPOSITE fixes. A model can match the pooled number by making each
    fish wander more over time instead of by making different fish differ, and nothing reported
    before this could tell those apart.

    That is not hypothetical. Measured on ``psib_gruz``: var_ratio distance 3.13 (3x too wide) while
    its between-fish level spread was too NARROW (0.115 vs a real 0.156). The algebra then forces
    W_gen/W_real > 3.13/(1-b) ~ 4.2 -- i.e. MORE than all of the excess is within-fish, and adding a
    between-fish latent on top of it can only make the pooled number worse. Read ``within_ratio``
    next to ``win_acf1``: excess within-fish variance with acf ~ +0.33 is genuine slow drift, the
    same excess with acf ~ 0 is white window-to-window jitter, and only the pair distinguishes them.

    THE gen 0.078 FIGURE THIS USED TO QUOTE IS STALE and was being read as a standing fact about
    the current family. Re-measured 2026-08-25 on ``rej`` (ladder_psib_rej.pt, 15 fish):
    **gen 0.2151 vs real 0.2551** -- a 0.04 gap, not 0.18. On that model the large deficit is not
    the between-fish share at all but POOLED under-dispersion: var_ratio turn 0.858, distance
    0.626, dt 0.442, dot-dist 1.028, radial 0.563. Read the two numbers together, and re-measure
    rather than quoting either.
    """
    d = _as_reps(windows_by_fish)[0]
    if not d:
        return {}
    R = np.vstack([v[0] for v in d.values()])
    G = np.vstack([v[1] for v in d.values()])
    names = feature_names or traj_names_for_dim(R.shape[1])
    mu, sd = R.mean(0), R.std(0) + EPS               # z-score against REAL, so real == 1.0
    Rz, Gz = (R - mu) / sd, (G - mu) / sd
    out = {"var_ratio": {}}
    for grp, cols in traj_groups(names).items():
        if not cols:
            continue
        ix = [names.index(c) for c in cols if c in names]
        rv = float(np.sum(np.var(Rz[:, ix], 0)))
        out["var_ratio"][grp] = float(np.sum(np.var(Gz[:, ix], 0)) / max(rv, EPS))
    share = {}
    Zs = {}
    for lab, sel in (("real", 0), ("gen", 1)):
        Z = {f: (v[sel] - mu) / sd for f, v in d.items()}
        Zs[lab] = Z
        fm = np.stack([z.mean(0) for z in Z.values()])
        tot = np.vstack(list(Z.values()))
        share[lab] = float(np.sum(np.var(fm, 0)) / max(float(np.sum(np.var(tot, 0))), EPS))
    out["between_fish_share"] = share
    # THE DECOMPOSITION, per group. Same fm/tot construction as the share above, restricted to the
    # group's columns: T = pooled, B = variance of the per-fish means, W = T - B. Reported as gen/real
    # ratios so 1.0 is the target on both, exactly as var_ratio is.
    out["within_ratio"], out["between_ratio"] = {}, {}
    for grp, cols in traj_groups(names).items():
        if not cols:
            continue
        ix = [names.index(c) for c in cols if c in names]
        wb = {}
        for lab in ("real", "gen"):
            per = [z[:, ix] for z in Zs[lab].values()]
            fm = np.stack([p.mean(0) for p in per])
            # W = mean WITHIN-fish variance, estimated per fish and averaged (not T - B, which
            # inherits B's bias).
            W = float(np.mean([np.sum(np.var(p, 0, ddof=1)) for p in per if len(p) > 1]))
            # B = Var(per-fish means) MINUS W/n. The variance of a fish's mean carries the
            # within-fish variance divided by its window count, so the naive Var(fm) is biased UP by
            # exactly W/n -- and that bias is largest precisely when W is large, which is the case
            # this split exists to diagnose. Measured on synthetic data with a known 4x within-fish
            # excess: the naive estimator read 0.34 against a true 0.25, this one reads it correctly.
            # Same one-way-ANOVA correction as exchangeability._icc1.
            n = float(np.mean([len(p) for p in per]))
            B = max(float(np.sum(np.var(fm, 0, ddof=1))) - W / max(n, 1.0), 0.0)
            wb[lab] = (W, B)
        out["within_ratio"][grp] = float(wb["gen"][0] / max(wb["real"][0], EPS))
        out["between_ratio"][grp] = float(wb["gen"][1] / max(wb["real"][1], EPS))
    return out


def between_fish_metrics(windows_by_fish, feature_names=None):
    """Three ABSOLUTE between-fish statistics, replacing the ``between_fish_share`` ratio.

    WHY NOT THE RATIO. ``between_fish_share = B / (B + W)`` can be exactly right while both
    parts are wrong -- a model with half the between-fish spread AND half the within-fish
    spread scores a perfect share. Reporting numerator and denominator separately, in the
    feature's own units, cannot balance out that way.

      ``bf_sd``  SD across fish of the per-fish window mean.  "how different are the fish"
      ``wf_sd``  mean over fish of that fish's SD across its own windows.  "how variable is
                 one fish" -- the denominator the ratio was hiding
      ``bf_w1``  Wasserstein-1 between the real and generated sets of per-fish means. Catches
                 SHAPE, not just spread: a model can have the right bf_sd and still miss that
                 real fish include an outlier like fish 14 (8.2% of bouts turning >150°
                 against 0.37-2.31% for the rest).

    ``windows_by_fish``: {fish_id: (real_windows, gen_windows)}, each (n_windows, d).
    Returns {'bf_sd': {'real': (d,), 'gen': (d,)}, 'wf_sd': {...}, 'bf_w1': (d,), 'names': [...]}.
    """
    fish = [f for f, (r, g) in windows_by_fish.items() if len(r) and len(g)]
    if len(fish) < 3:
        return {}
    out = {}
    for side, idx in (("real", 0), ("gen", 1)):
        mu = np.stack([windows_by_fish[f][idx].mean(0) for f in fish])        # (n_fish, d)
        sd = np.stack([windows_by_fish[f][idx].std(0) for f in fish])
        out.setdefault("bf_sd", {})[side] = mu.std(0, ddof=1)
        out.setdefault("wf_sd", {})[side] = sd.mean(0)
        out.setdefault("_mu", {})[side] = mu
    d = out["_mu"]["real"].shape[1]
    out["bf_w1"] = np.array([wasserstein_distance(out["_mu"]["real"][:, j],
                                                  out["_mu"]["gen"][:, j]) for j in range(d)])
    out["n_fish"] = len(fish)
    out["names"] = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(d)]
    out.pop("_mu")
    return out


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
    for fid, (rf, gf) in _as_reps(windows_by_fish)[0].items():
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
                              progress=None, c2st_reps=8):
    """Leave-3-fish-out generative-similarity CV.

    Returns (panel_df, windows_by_model, windows_by_model_traj). ``panel_df`` has one row
    per (fish, model) with the panel scalars; ``windows_by_model`` maps model name ->
    LIST of ``c2st_reps`` dicts {fish_id: (real_feats, gen_feats)}, one per independent
    rollout replicate, for the TURN-only C2ST/MMD omnibus; ``windows_by_model_traj`` is the
    same with TRAJECTORY features. Feed either to ``c2st_auc_replicates``; ``c2st_auc``
    still accepts a single dict (use ``[0]`` for one replicate).

    Models are refit per fold so the generator for a fish never trained on that fish.

    RANDOMNESS. Every rollout draws from a stream derived from
    ``(seed, model_idx, fish_id, rep)``, NOT from one generator threaded through the whole
    run. Two consequences, both required for a valid A/B:
      * COMMON RANDOM NUMBERS -- model i and model j see the same draws for the same fish
        and replicate, so their difference is paired and lower-variance.
      * ORDER INDEPENDENCE -- a model's numbers no longer depend on which other models are
        in ``model_factories`` or on ``R``. Previously the shared stream made the same
        config score 0.760 in one run and 0.7946 in another purely from list composition.
    ``c2st_reps`` replicates cut the AUC standard error as 1/sqrt(R): at reps=1 the SD is
    0.0173 (SE of a difference 0.0245), at 8 it is ~0.006 (SE_diff ~0.009).

    The stream key is (seed, fish_id, rep) and deliberately does NOT include the model. If it
    did -- e.g. by using the model's index in ``model_factories`` -- then reordering or adding
    a model would change every other model's draws, which is the exact order dependence this
    replaces, and the replicates would not be paired across models. Excluding it means model A
    and model B see the SAME draws for the same (fish, rep), so replicate r of A and replicate
    r of B are comparable and their difference can be tested pairwise.
    """
    import pandas as pd
    cfg = cfg or {}
    rows = []
    windows_by_model = {}
    windows_by_model_traj = {}
    windows_traj_eq = {}          # marginal-equalised twin, returned via panel_df.attrs
    total = len(folds) * len(model_factories)      # for the optional progress callback
    done = 0
    for j, (train_ids, test_ids) in enumerate(folds):
        train = [records[f] for f in train_ids]
        ctx = E.make_context(train, cfg)
        for mi, factory in enumerate(model_factories):
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
                wm = windows_by_model.setdefault(model.name, [{} for _ in range(c2st_reps)])
                wmt = windows_by_model_traj.setdefault(model.name, [{} for _ in range(c2st_reps)])
                wmte = windows_traj_eq.setdefault(model.name, [{} for _ in range(c2st_reps)])
                for f in test_ids:
                    rec = records[f]
                    # panel: its own stream, so changing c2st_reps cannot move the panel
                    rows.append(panel_for_fish(
                        model, rec, np.random.default_rng([seed, int(f), 10_000]), R=R))
                    if not collect_windows:
                        continue
                    _ru = getattr(rec, "run_id", None)
                    real_w = window_features(rec.dtheta, rec.symbol, W=W, runs=_ru)
                    real_rad = _radial_of(rec, rec.x_px, rec.y_px)
                    real_dd_all = getattr(rec, "dot_dist_px", None)
                    real_t = None
                    # Batched models generate all replicates at once; everyone else keeps the
                    # per-rep RNG keyed on (seed, fish, rep), which is what makes replicate r of
                    # model A paired with replicate r of model B.
                    _batched = (model.sample_batch(rec, c2st_reps,
                                                   np.random.default_rng([seed, int(f)]))
                                if getattr(model, "batched_sample", False) else None)
                    for rep in range(c2st_reps):
                        gen = (_batched[rep] if _batched is not None else
                               model.sample(rec, np.random.default_rng([seed, int(f), rep]),
                                            **_stim_kw(model, rec)))
                        wm[rep][f] = (real_w, window_features(gen["dtheta"], gen["symbol"], W=W))
                        has_pos = ("x_px" in gen) and (getattr(rec, "dot_cx", None) is not None)
                        gx = np.asarray(gen["x_px"], float) if "x_px" in gen else None
                        gy = np.asarray(gen["y_px"], float) if "x_px" in gen else None
                        real_dd = real_dd_all if has_pos else None
                        gen_dd = (np.hypot(gx - np.asarray(gen.get("dot_cx", rec.dot_cx), float),
                                           gy - np.asarray(gen.get("dot_cy", rec.dot_cy), float))
                                  if has_pos else None)
                        gen_rad = _radial_of(rec, gx, gy) if gx is not None else None
                        if real_t is None:          # real side is replicate-invariant
                            real_t = window_features_traj(rec.dtheta, rec.symbol, rec.distance_px,
                                                          rec.dt_s, real_dd, W=W, runs=_ru,
                                                          radial=(real_rad if gen_rad is not None else None))
                        wmt[rep][f] = (real_t,
                                       window_features_traj(gen["dtheta"], gen["symbol"],
                                                            gen["distance_px"], gen["dt_s"], gen_dd,
                                                            W=W, radial=gen_rad))
                        # MARGINAL-EQUALISED twin: same rollout, each behavioural channel
                        # rank-mapped onto this fish's real marginal. Separates "wrong levels"
                        # from "wrong temporal structure" -- see rank_map. Positions are not
                        # remapped (dot-distance and radial are geometric consequences of the
                        # path, not channels the model emits), so those columns are unchanged.
                        eq_d = rank_map(gen["distance_px"], rec.distance_px)
                        eq_t = rank_map(gen["dt_s"], rec.dt_s)
                        eq_a = np.sign(gen["dtheta"]) * rank_map(np.abs(gen["dtheta"]),
                                                                 np.abs(rec.dtheta))
                        wmte[rep][f] = (real_t,
                                        window_features_traj(eq_a, E.symbols_from_dtheta(eq_a),
                                                             eq_d, eq_t, gen_dd,
                                                             W=W, radial=gen_rad))
            elif verbose:
                print(f"  [fold {j}] skip {model.name}: {skip}")
            done += 1
            if progress is not None:
                progress(done, total, f"fold {j} · {model.name}")
        if verbose:
            print(f"  fold {j}: generative panel for test fish {test_ids}")
    panel_df = pd.DataFrame(rows)
    # carried on .attrs rather than in the return tuple so every existing 3-tuple caller
    # (quick_results, the notebook cells) keeps working unchanged.
    panel_df.attrs["windows_traj_eq"] = windows_traj_eq
    return panel_df, windows_by_model, windows_by_model_traj


def c2st_windows_for_fish(model, rec, fish_id, seed=0, reps=8, W=60, reanchor=None,
                          interior=INTERIOR):
    """The C2ST window pairs for ONE fish -- the same construction ``cross_validate_generative``
    does inline, factored out so a run that owns its own rollout loop (the local Metric-B script)
    cannot drift away from the ladder's feature set.

    Returns ``dict(turn=[...], traj=[...], traj_eq=[...], traj_int=[...])``, each a list of
    ``reps`` entries ``(real_feats, gen_feats)`` -- exactly the value a ``windows_by_fish`` dict
    holds, so replicate ``r`` of every fish assembles into the r-th dict that
    ``c2st_auc_replicates`` consumes. ``traj_int`` is the STEADY-STATE twin of ``traj`` from the
    SAME rollouts: real windows >= ``interior[0]`` bouts into a run and >= ``interior[1]`` before
    it ends, generated windows starting at t >= ``interior[0]`` and subsampled to the fish's
    standard gen:real window ratio (see ``INTERIOR`` and the note in the loop); None skips it.

    THE RNG IS ``default_rng([seed, fish_id, rep])``, the ladder's convention and deliberately
    NOT the stream the Metric-B panel draws on: changing ``reps`` must not be able to move the
    panel. Two models trained under the same rollout seed therefore see matched streams replicate
    for replicate, which is what makes an arm-to-arm AUC difference a model difference rather
    than rollout noise (SD 0.0173 at reps=1; ~1/sqrt(reps) from there).
    """
    _runs = getattr(rec, "run_id", None)
    real_w = window_features(rec.dtheta, rec.symbol, W=W, runs=_runs)
    real_rad = _radial_of(rec, rec.x_px, rec.y_px)
    real_dd_all = getattr(rec, "dot_dist_px", None)
    real_t = real_i = None
    out = dict(turn=[], traj=[], traj_eq=[])
    if interior:
        out["traj_int"] = []
        lead, tail = int(interior[0]), int(interior[1])
    for rep in range(int(reps)):
        gen = model.sample(rec, np.random.default_rng([int(seed), int(fish_id), int(rep)]),
                           **_stim_kw(model, rec, reanchor))
        out["turn"].append((real_w, window_features(gen["dtheta"], gen["symbol"], W=W)))
        has_pos = ("x_px" in gen) and (getattr(rec, "dot_cx", None) is not None)
        gx = np.asarray(gen["x_px"], float) if "x_px" in gen else None
        gy = np.asarray(gen["y_px"], float) if "x_px" in gen else None
        real_dd = real_dd_all if has_pos else None
        gen_dd = (np.hypot(gx - np.asarray(gen.get("dot_cx", rec.dot_cx), float),
                           gy - np.asarray(gen.get("dot_cy", rec.dot_cy), float)) if has_pos else None)
        gen_rad = _radial_of(rec, gx, gy) if gx is not None else None
        if real_t is None:                       # the real side is replicate-invariant
            real_t = window_features_traj(rec.dtheta, rec.symbol, rec.distance_px, rec.dt_s,
                                          real_dd, W=W, runs=_runs,
                                          radial=(real_rad if gen_rad is not None else None))
        out["traj"].append((real_t,
                            window_features_traj(gen["dtheta"], gen["symbol"], gen["distance_px"],
                                                 gen["dt_s"], gen_dd, W=W, radial=gen_rad)))
        if interior:
            if real_i is None:
                real_i = window_features_traj(rec.dtheta, rec.symbol, rec.distance_px, rec.dt_s,
                                              real_dd, W=W, runs=_runs, lead=lead, tail=tail,
                                              radial=(real_rad if gen_rad is not None else None))
            gi = window_features_traj(gen["dtheta"], gen["symbol"], gen["distance_px"], gen["dt_s"],
                                      gen_dd, W=W, radial=gen_rad, lead=lead)
            # The selection keeps ~40 % of the real windows and ~97 % of the generated ones, so
            # left alone the interior classifier would see a 4:1 class imbalance against the
            # standard reading's ~1.6:1 -- and imbalance alone moves the AUC (measured +0.03 for
            # a 4:1 pool). Subsample the generated side to THIS fish's standard gen:real ratio,
            # so the two AUCs differ by the window selection and nothing else. Seeded per
            # (seed, fish, rep): deterministic, replicate-matched across arms.
            gen_t = out["traj"][-1][1]
            n_keep = min(len(gi), int(round(len(real_i) * len(gen_t) / max(len(real_t), 1))))
            if n_keep < len(gi):
                keep = np.sort(np.random.default_rng([int(seed), int(fish_id), int(rep), 1])
                               .choice(len(gi), n_keep, replace=False))
                gi = gi[keep]
            out["traj_int"].append((real_i, gi))
        # MARGINAL-EQUALISED twin: same rollout, each behavioural channel rank-mapped onto this
        # fish's real marginal. Separates "wrong levels" from "wrong temporal structure". The
        # positions are NOT remapped -- dot-distance and radial are geometric consequences of the
        # path, not channels the model emits.
        eq_d = rank_map(gen["distance_px"], rec.distance_px)
        eq_t = rank_map(gen["dt_s"], rec.dt_s)
        eq_a = np.sign(gen["dtheta"]) * rank_map(np.abs(gen["dtheta"]), np.abs(rec.dtheta))
        out["traj_eq"].append((real_t,
                               window_features_traj(eq_a, E.symbols_from_dtheta(eq_a), eq_d, eq_t,
                                                    gen_dd, W=W, radial=gen_rad)))
    return out


def _radial_of(rec, x, y):
    """Per-bout r/R in the physical dish, or None if the record has no dish geometry."""
    if x is None or getattr(rec, "dish_R", None) is None:
        return None
    return np.hypot(np.asarray(x, float) - rec.dish_cx,
                    np.asarray(y, float) - rec.dish_cy) / rec.dish_R


# Reported panel. DROPPED under the active-only / re-anchored setup, and why:
#   following_gap, following_gap_static  -- score gen turns against rec.stim_angle, the REAL
#       fish's bearing track, which a re-anchored rollout never saw. Correct only for
#       teacher-forced models; the _h pair is the closed-loop version.
#   dot_dist_bias_late, dot_dist_w1_late -- existed to cut a block-onset placement artefact
#       (median onset 124 px vs real 73). Evaluation rollouts are now RE-ANCHORED at each block
#       onset (_stim_kw -> stim_rollout.RecordAnchoredProvider), which removes it at source.
#       WARNING: until 2026-09-11 this comment claimed that while the rollouts still REPLAYED the
#       recorded dots (measured: median onset 121 px, 49.8% beyond 121 px), so every dot-distance
#       number produced before that date carries the artefact.
# Still COMPUTED by panel_for_fish (so a teacher-forced run can ask for them by name); just
# not reported by default.
PANEL_METRICS = ["abs_dtheta_w1", "symbol_js", "dt_w1", "dist_w1", "trans_js",
                 "acf_l2", "runlen_w1", "bigram_H_gap", "tuning_mse",
                 "following_gap_h", "following_gap_static_h",
                 "dot_dist_w1", "time_near_dot_gap", "approach_gap",
                 "dot_dist_bias", "time_far_gap", "wall_frac_gap"]
PANEL_METRICS_TEACHER_FORCED = ["following_gap", "following_gap_static"]


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
