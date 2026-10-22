"""
Model-agnostic evaluation harness for generative models of zebrafish BOUT sequences.

This module implements **Metric A (predictive likelihood)** and the cross-validation
machinery of the approved methodology
(see .claude/plans/i-want-you-to-velvet-wilkinson.md). Metric B (generative
similarity) lives in ``generative_metrics.py``; the exchangeability diagnostics
(Request 1) live in ``exchangeability.py``. All three share the record/adapter
contract defined *here*.

WHAT THIS EVALUATES
-------------------
Each fish f is one ordered sequence of bouts; the modeled observable is the signed
reorientation angle ``dtheta_deg``. We score how well a model assigns probability to
held-out real next-bouts. By the chain rule the held-out joint log-likelihood per
bout equals the mean one-step predictive log-score -- exactly the forward
algorithm's ``mean_t log c_t`` -- so the metric is defined on each model's
*predictive distribution*, not its parametric form, which is what makes HMMs and
future autoregressive models comparable.

TWO TRACKS (locked design decisions)
------------------------------------
* PRIMARY  -- continuous-angle density: ``log p(dtheta_t | past, stim)`` in
  **nats/bout** (report bits/bout = /ln2). This is ``log c_t`` from the scaled
  forward pass. Every model in this track must emit a density over ``dtheta`` in
  DEGREES w.r.t. Lebesgue measure (future AR models need a continuous head).
* SECONDARY -- 3-symbol categorical log-loss over {Forward, Left, Right}: convert a
  continuous model's one-step predictive emission to a proper PMF by integrating
  each state's emission over the +/-10 deg bins (Normal/Gamma CDFs) mixed by the
  predicted state distribution. Lets a purely categorical AR model be compared on a
  coarser but universal space; also a resolution-robustness check on the primary.

Channels: only the TURN angle enters the likelihood. ``dt_s`` (timing) and
``distance_px`` are resampled from empirical pools by the generators and carry no
model signal, so they are excluded here and compared only in Metric B.

CV: **leave-3-fish-out, 5 stratified folds** (each fish tested exactly once ->
15 paired per-fish scores). Everything that could leak (dt/dist pools, rich-feature
standardisation scales, baseline counts) is computed on the TRAINING fold only.

Model adapters reuse the existing implementations in ``fit_hmm.py`` /
``fit_hmm_stim.py`` verbatim (forward-backward, emissions, sampling, EM fits).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import erf, gammainc          # normal / gamma CDFs for the symbol track

sys.path.insert(0, str(Path(__file__).parent))
import fit_hmm as H
import fit_hmm_stim as S

LN2 = np.log(2.0)
DTHETA0 = H.DTHETA0_DEG                            # +/-10 deg forward/turn threshold
SQRT2 = np.sqrt(2.0)


# =========================================================================== #
#  1. Per-fish records  (the canonical data structure every model consumes)
# =========================================================================== #
@dataclass
class FishRecord:
    fish_id: int
    dtheta: np.ndarray            # (n,) signed reorientation angle, deg   (the observation)
    symbol: np.ndarray            # (n,) 0=F, 1=L(dtheta<-10), 2=R(dtheta>10)
    b: np.ndarray                 # (n,) simple-model scalar bias sin(phi) on moving bouts, else 0
    stim_angle: np.ndarray        # (n,) egocentric bearing to dots (rad), NaN where undefined
    moving: np.ndarray            # (n,) bool: a moving stimulus present at this bout
    dt_s: np.ndarray              # (n,) inter-bout interval to next bout (s); NaN allowed
    distance_px: np.ndarray       # (n,) travel distance to next bout (px); NaN allowed
    u_raw: np.ndarray | None = None   # (n,5) raw rich stimulus features [Lp,Lm,Sp,Sm,Fp] or None
    stim_csv: np.ndarray | None = None  # (n,7) bout-level rich stim from CSV (stim_mode='rich_csv') or None

    @property
    def n(self) -> int:
        return len(self.dtheta)


def symbols_from_dtheta(dtheta, dtheta0=DTHETA0):
    """0=Forward (|dtheta|<dtheta0), 1=Left (dtheta<-dtheta0), 2=Right (dtheta>+dtheta0).
    Matches fit_hmm.threshold_classifier (left = negative)."""
    sym = np.zeros(len(dtheta), dtype=np.int8)
    sym[dtheta < -dtheta0] = 1
    sym[dtheta > +dtheta0] = 2
    return sym


def load_bouts(path=None):
    """Load the bout table. Defaults to the CLEANED table (tracking excursions repaired,
    plus run_id / active / the modelling features). The old
    ``data/bouts_GsizeMotionSingle.csv`` has neither ``run_id`` nor ``active``, so every
    run-aware metric path silently falls back to its flat form -- pass it explicitly only
    when reproducing an archived number.

    ``path=None`` resolves against the REPO ROOT, not the working directory -- the previous
    relative default only worked when run from the root."""
    import pandas as pd
    from pathlib import Path as _P
    if path is None:
        path = _P(__file__).resolve().parents[1] / "clean_data" / "bouts_clean.csv"
    return pd.read_csv(path)


def build_records(df, min_len=10, group_cols=("fish_id", "segment_id"),
                  order_col="bout_index", abs_dict=None, roi=None):
    """Split the tidy bout dataframe into per-fish :class:`FishRecord`s.

    One record per (fish_id, segment_id) group (there is one segment per fish),
    ordered by bout_index, keeping only bouts with finite dtheta. If ``abs_dict``
    (per-frame dot positions) and ``roi`` are given, the raw rich stimulus feature
    vector is attached per bout (aligned to the same bouts; missing frames -> zeros).
    """
    records = {}
    cols = list(group_cols)
    fish_arrs = {}                                  # fish_id -> (arrays, frame->row map) cache
    for keys, g in df.sort_values(cols + [order_col]).groupby(cols, sort=False):
        fid = int(keys[0] if isinstance(keys, tuple) else keys)
        g = g.sort_values(order_col)
        dth = g["dtheta_deg"].to_numpy(float)
        good = np.isfinite(dth)
        dth = dth[good]
        if len(dth) < min_len:
            continue
        phi = g["stim_angle_rad"].to_numpy(float)[good]
        moving = (g["moving"].to_numpy(bool)[good] if "moving" in g.columns
                  else np.zeros(len(dth), bool))
        present = np.isfinite(phi) & moving
        b = S.bias_from_phi(phi, present)
        dt_s = (g["dt_s"].to_numpy(float)[good] if "dt_s" in g.columns
                else np.full(len(dth), np.nan))
        dist = (g["distance_px"].to_numpy(float)[good] if "distance_px" in g.columns
                else np.full(len(dth), np.nan))

        u_raw = None
        if abs_dict is not None and roi is not None:
            u_raw = _rich_features_for_group(g[good], fid, abs_dict, roi, fish_arrs)

        # bout-level rich stimulus block (stim_mode='rich_csv') -- from CSV columns, no abs_dict.
        # all-zero on no-stimulus bouts; carries bearing + group size + per-side speed + moving
        # (present on static-dot bouts too, unlike the moving-only simple `b`).
        stim_csv = None
        if "n_dots_left" in g.columns:
            def _c(name):
                return (np.nan_to_num(g[name].to_numpy(float)[good])
                        if name in g.columns else np.zeros(len(dth)))
            stim_csv = np.stack([
                np.nan_to_num(np.sin(phi)), np.nan_to_num(np.cos(phi)),
                _c("n_dots_left") / 8.0, _c("n_dots_right") / 8.0,
                _c("speed_left") / 3.5, _c("speed_right") / 3.5,
                moving.astype(float),
            ], axis=1).astype(float)

        records[fid] = FishRecord(
            fish_id=fid, dtheta=dth, symbol=symbols_from_dtheta(dth), b=b,
            stim_angle=phi, moving=moving, dt_s=dt_s, distance_px=dist, u_raw=u_raw,
            stim_csv=stim_csv)
    return records


def _rich_features_for_group(g, fid, abs_dict, roi, cache):
    """Raw rich stimulus features (n,5) for one fish group, aligned to its bouts.

    Mirrors fit_hmm_stim.build_rich_sequences' per-bout feature computation but
    keeps alignment with the record (missing frame -> zero feature vector) instead
    of dropping the bout, and returns UN-standardised features so cross-validation
    can standardise with training-fold scales.
    """
    key = f"f{fid}"
    if key not in abs_dict:
        return None
    if fid not in cache:
        fdf = abs_dict[key]
        arrs = S._fish_arrays(fdf)
        fmap = {int(fr): i for i, fr in enumerate(fdf["frame"].to_numpy())}
        cache[fid] = (arrs, fmap)
    (FX, FY, FORI, DX, DY, ACT), fmap = cache[fid]
    s_f = S.S_F                                     # global nominal mm/px (roi unused)
    frames = g["frame_onset"].to_numpy(int)
    U = np.zeros((len(frames), 5))
    for j, fo in enumerate(frames):
        i = fmap.get(int(fo))
        if i is None:
            continue
        ip = i - 1 if i > 0 else i
        U[j] = S._covariates(FX[i], FY[i], FORI[i], DX[i], DY[i], ACT[i],
                             DX[ip], DY[ip], ACT[ip], s_f)
    return U


# =========================================================================== #
#  2. Forward-scoring core  (per-bout predictive density + symbol mass)
# =========================================================================== #
def forward_logdens_states(pi, T, B):
    """Scaled forward pass returning PER-BOUT predictive log-density and the
    one-step predicted state distribution.

    pi : (3,) initial distribution.
    T  : (3,3) fixed transition, OR (n,3,3) time-varying (T[t] = transition INTO
         bout t, matching fit_hmm_stim._io_forward_backward).
    B  : (n,3) emission PROBABILITIES (densities) at each bout.

    Returns
    -------
    logdens : (n,)  log p(dtheta_t | dtheta_{1:t-1}, stim) = log c_t.
    states  : (n,3) predicted state distribution s_t BEFORE seeing bout t
              (s_0 = pi; s_t = normalised-alpha_{t-1} @ T_t). c_t = sum_j s_t[j] B[t,j].
    """
    n = B.shape[0]
    time_varying = (T.ndim == 3)
    logdens = np.empty(n)
    states = np.empty((n, 3))
    states[0] = pi
    a = pi * B[0]
    c = a.sum(); c = c if c > 0 else 1e-300
    logdens[0] = np.log(c)
    alpha = a / c
    for t in range(1, n):
        Tt = T[t] if time_varying else T
        pred = alpha @ Tt                            # predicted state dist at t (sums to 1)
        states[t] = pred
        a = pred * B[t]
        c = a.sum(); c = c if c > 0 else 1e-300
        logdens[t] = np.log(c)
        alpha = a / c
    return logdens, states


def symbol_logmass(states, sigma, alpha, theta, mu=0.0, dtheta0=DTHETA0, eps=1e-300):
    """One-step predictive log-PMF over {F,L,R} for a continuous 3-state emission.

    Integrates each state's emission over the +/-dtheta0 bins (exact via CDFs) and
    mixes by the predicted state distribution ``states`` (n,3):

      state 0 (forward)  Normal(mu, sigma)      -> mass to all three bins
      state 1 (turn-)    Gamma on -dtheta (<0)  -> [P(0<mag<dtheta0), P(mag>dtheta0), 0]
      state 2 (turn+)    Gamma on +dtheta (>0)  -> [P(0<mag<dtheta0), 0, P(mag>dtheta0)]

    ``mu`` may be a scalar (spontaneous, mu=0) or an (n,) array (stimulus shifts the
    forward mean). Returns (n,3) log-probabilities, rows sum to 1 in prob space.
    The forward state gives strictly positive mass to every bin, so log is finite.
    """
    n = states.shape[0]
    mu = np.broadcast_to(np.asarray(mu, float), (n,))
    Gc = gammainc(alpha, dtheta0 / theta)            # P(turn magnitude < dtheta0), scalar
    z_hi = (dtheta0 - mu) / sigma
    z_lo = (-dtheta0 - mu) / sigma
    Phi_hi = 0.5 * (1.0 + erf(z_hi / SQRT2))
    Phi_lo = 0.5 * (1.0 + erf(z_lo / SQRT2))
    nR = 1.0 - Phi_hi                                # forward-state mass in Right bin
    nL = Phi_lo                                      # forward-state mass in Left bin
    nF = Phi_hi - Phi_lo                             # forward-state mass in Forward bin
    s0, s1, s2 = states[:, 0], states[:, 1], states[:, 2]
    qF = s0 * nF + s1 * Gc + s2 * Gc
    qL = s0 * nL + s1 * (1.0 - Gc)
    qR = s0 * nR + s2 * (1.0 - Gc)
    q = np.stack([qF, qL, qR], axis=1)
    q = np.maximum(q, eps)
    q /= q.sum(axis=1, keepdims=True)
    return np.log(q)


# =========================================================================== #
#  3. Fit context  (everything computed on the TRAINING fold, passed to models)
# =========================================================================== #
@dataclass
class FitContext:
    """Training-fold quantities shared with model adapters (kept leak-free)."""
    dt_pool: np.ndarray                  # inter-bout intervals from TRAIN fish (for generation)
    dist_pool: np.ndarray                # travel distances from TRAIN fish
    rich_scales: np.ndarray | None = None   # (5,) RMS feature scales from TRAIN fish
    cfg: dict = field(default_factory=dict)


def make_context(train_records, cfg):
    dt = np.concatenate([r.dt_s[np.isfinite(r.dt_s)] for r in train_records])
    dist = np.concatenate([r.distance_px[np.isfinite(r.distance_px)] for r in train_records])
    scales = None
    if all(r.u_raw is not None for r in train_records):
        allU = np.concatenate([r.u_raw for r in train_records], axis=0)
        scales = np.sqrt(np.mean(allU ** 2, axis=0))
        scales[scales < 1e-9] = 1.0
    return FitContext(dt_pool=dt, dist_pool=dist, rich_scales=scales, cfg=dict(cfg))


# =========================================================================== #
#  4. Model adapters  (the harness never inspects a model's internals)
# =========================================================================== #
class GenerativeModel:
    """Adapter contract. Implement fit / score (/ sample for Metric B)."""
    name = "model"
    has_density = True                    # emits a continuous dtheta density (Metric A primary)?
    has_symbol = True                     # emits a {F,L,R} predictive PMF (secondary)?
    can_sample = True                     # can free-run rollouts (Metric B)? baselines cannot
    requires_absdict = False

    def fit(self, train_records, ctx):    # -> self
        raise NotImplementedError

    def score(self, rec):
        """Return dict(logdens=(n,) or None, symbol_logq=(n,3) or None)."""
        raise NotImplementedError

    def sample(self, rec, rng):
        """Return dict(dtheta, symbol, dt_s, distance_px) conditioned on rec's stimulus."""
        raise NotImplementedError


class SpontaneousHMM(GenerativeModel):
    name = "spontaneous"

    def __init__(self, restarts=2, max_iter=200, seed=0):
        self.restarts, self.max_iter, self.seed = restarts, max_iter, seed
        self.m = None

    def fit(self, train_records, ctx):
        seqs = [r.dtheta for r in train_records]
        self.m = H.fit_hmm_g3_sym(seqs, n_restarts=self.restarts,
                                  max_iter=self.max_iter, seed=self.seed, verbose=False)
        self._dt_pool, self._dist_pool = ctx.dt_pool, ctx.dist_pool
        return self

    def score(self, rec):
        m = self.m
        B = np.exp(H.emission_logprob(rec.dtheta, m))
        logdens, states = forward_logdens_states(m["pi"], m["T"], B)
        slq = symbol_logmass(states, m["sigma"], m["alpha"], m["theta"], mu=0.0)
        return dict(logdens=logdens, symbol_logq=slq)

    def sample(self, rec, rng):
        gen = H.generate_bouts(self.m, rec.n, self._dt_pool, self._dist_pool, rng)
        return dict(dtheta=gen["angle_deg"], symbol=symbols_from_dtheta(gen["angle_deg"]),
                    dt_s=gen["dt_s"], distance_px=gen["dist_px"])


class SimpleStimHMM(GenerativeModel):
    name = "simple_stim"

    def __init__(self, restarts=2, max_iter=150, seed=0):
        self.restarts, self.max_iter, self.seed = restarts, max_iter, seed
        self.m = None

    def fit(self, train_records, ctx):
        seqs = [(r.dtheta, r.b) for r in train_records]
        self.m = S.fit_io_hmm(seqs, n_restarts=self.restarts, max_iter=self.max_iter,
                              seed=self.seed, verbose=False)
        self._dt_pool, self._dist_pool = ctx.dt_pool, ctx.dist_pool
        return self

    def score(self, rec):
        m = self.m
        L0 = S._L0_from_params(m["u"])
        Tseq = S._build_T_seq(rec.b, L0, m["kappa"])
        pi_vec = S.init_dist(m["pi0"], m["kappa"], rec.b[0])
        B = S.emission_probs(rec.dtheta, rec.b, m["sigma"], m["alpha"], m["theta"], m["g"])
        logdens, states = forward_logdens_states(pi_vec, Tseq, B)
        mu = m["g"] * rec.b
        slq = symbol_logmass(states, m["sigma"], m["alpha"], m["theta"], mu=mu)
        return dict(logdens=logdens, symbol_logq=slq)

    def sample(self, rec, rng):
        m = self.m
        L0 = S._L0_from_params(m["u"]); kappa = m["kappa"]
        n = rec.n
        states = np.empty(n, int)
        states[0] = rng.choice(3, p=S.init_dist(m["pi0"], kappa, rec.b[0]))
        for t in range(1, n):
            T = S.trans_matrix(L0, kappa, rec.b[t])
            states[t] = rng.choice(3, p=T[states[t - 1]])
        dth = np.array([S.sample_dtheta(int(s), rec.b[t], m, rng)
                        for t, s in enumerate(states)])
        return dict(dtheta=dth, symbol=symbols_from_dtheta(dth),
                    dt_s=rng.choice(self._dt_pool, size=n),
                    distance_px=rng.choice(self._dist_pool, size=n))


class RichStimHMM(GenerativeModel):
    name = "rich_stim"
    requires_absdict = True

    def __init__(self, restarts=2, max_iter=120, seed=0):
        self.restarts, self.max_iter, self.seed = restarts, max_iter, seed
        self.m = None

    def fit(self, train_records, ctx):
        if ctx.rich_scales is None:
            raise RuntimeError("RichStimHMM needs rich features (abs_dict); ctx.rich_scales is None")
        self._scales = ctx.rich_scales
        seqs = []
        for r in train_records:
            Us = r.u_raw / self._scales
            seqs.append((r.dtheta, Us[:, 0:2], Us[:, 2:4], Us[:, 4]))
        self.m = S.fit_io_hmm_rich(seqs, self._scales, n_restarts=self.restarts,
                                   max_iter=self.max_iter, seed=self.seed, verbose=False)
        self._dt_pool, self._dist_pool = ctx.dt_pool, ctx.dist_pool
        return self

    def _steer_prop(self, rec):
        Us = rec.u_raw / self.m["scales"]
        steer = Us[:, 0:2] @ self.m["beta_odd"]
        prop = Us[:, 2:4] @ self.m["beta_even"]
        return steer, prop

    def score(self, rec):
        m = self.m
        steer, prop = self._steer_prop(rec)
        L0 = S._L0_from_params(m["u"])
        Tseq = S._build_T_seq_g(steer, prop, L0)
        B = S._emission_probs_g(rec.dtheta, steer, m["sigma"], m["alpha"], m["theta"], m["g"])
        logdens, states = forward_logdens_states(m["pi0"], Tseq, B)
        mu = m["g"] * steer
        slq = symbol_logmass(states, m["sigma"], m["alpha"], m["theta"], mu=mu)
        return dict(logdens=logdens, symbol_logq=slq)

    def sample(self, rec, rng):
        m = self.m
        steer, prop = self._steer_prop(rec)
        L0 = S._L0_from_params(m["u"])
        n = rec.n
        states = np.empty(n, int)
        states[0] = rng.choice(3, p=m["pi0"])
        for t in range(1, n):
            T = S._trans_scalars(L0, steer[t], prop[t])
            states[t] = rng.choice(3, p=T[states[t - 1]])
        dth = np.empty(n)
        for t, s in enumerate(states):
            if s == 0:
                dth[t] = rng.normal(m["g"] * steer[t], m["sigma"])
            else:
                mag = rng.gamma(m["alpha"], m["theta"])
                dth[t] = -mag if s == 1 else mag
        return dict(dtheta=dth, symbol=symbols_from_dtheta(dth),
                    dt_s=rng.choice(self._dt_pool, size=n),
                    distance_px=rng.choice(self._dist_pool, size=n))


# =========================================================================== #
#  5. Baselines  (fit on TRAIN fold; the bar every model must clear)
# =========================================================================== #
class MarginalDensity(GenerativeModel):
    """i.i.d. mixture density of dtheta with NO sequential structure and NO stimulus.
    p(dtheta) = wF Normal(0,sigma) + wL Gamma(-dtheta) + wR Gamma(+dtheta), weights =
    training symbol fractions. The reference for the density skill score: the gap to
    this baseline is the value of the model's *dynamics* (and stimulus)."""
    name = "marginal_density"
    has_symbol = True
    can_sample = False

    def fit(self, train_records, ctx):
        dth = np.concatenate([r.dtheta for r in train_records])
        sym = np.concatenate([r.symbol for r in train_records])
        self.w = np.array([(sym == k).mean() for k in range(3)])
        fwd = dth[np.abs(dth) < DTHETA0]
        turn = np.abs(dth[np.abs(dth) >= DTHETA0])
        self.sigma = float(max(np.std(fwd) if len(fwd) > 5 else 8.0, 0.5))
        m_, v_ = (turn.mean(), turn.var()) if len(turn) > 5 else (40.0, 400.0)
        self.alpha = float(max(m_ * m_ / max(v_, 1e-12), H.ALPHA_MIN))
        self.theta = float(max(v_ / max(m_, 1e-12), 1.0))
        return self

    def score(self, rec):
        prm = dict(sigma=self.sigma, alpha=self.alpha, theta=self.theta)
        B = np.exp(H.emission_logprob(rec.dtheta, prm))     # (n,3) per-state densities
        dens = B @ self.w                                   # mixture density
        logdens = np.log(np.maximum(dens, 1e-300))
        # symbol PMF is the (constant) unigram; states = w broadcast
        states = np.broadcast_to(self.w, (rec.n, 3))
        slq = symbol_logmass(states, self.sigma, self.alpha, self.theta, mu=0.0)
        return dict(logdens=logdens, symbol_logq=slq)


class UnigramSymbol(GenerativeModel):
    """Symbol-only baseline: predict the training marginal {F,L,R} frequencies."""
    name = "unigram"
    has_density = False
    can_sample = False

    def fit(self, train_records, ctx):
        sym = np.concatenate([r.symbol for r in train_records])
        p = np.array([(sym == k).mean() for k in range(3)])
        self.logp = np.log(np.maximum(p, 1e-300))
        return self

    def score(self, rec):
        return dict(logdens=None, symbol_logq=np.broadcast_to(self.logp, (rec.n, 3)).copy())


class BigramSymbol(GenerativeModel):
    """First-order Markov symbol baseline (paper's counted transition matrix)."""
    name = "bigram"
    has_density = False
    can_sample = False

    def fit(self, train_records, ctx):
        P = np.zeros((3, 3))
        for r in train_records:
            s = r.symbol
            for i in range(len(s) - 1):
                P[s[i], s[i + 1]] += 1
        self.logT = np.log(np.maximum(P / np.maximum(P.sum(1, keepdims=True), 1e-300), 1e-300))
        p0 = np.array([P.sum(1)[k] / P.sum() for k in range(3)])
        self.logp0 = np.log(np.maximum(p0, 1e-300))
        return self

    def score(self, rec):
        s = rec.symbol
        slq = np.empty((rec.n, 3))
        slq[0] = self.logp0
        slq[1:] = self.logT[s[:-1]]
        return dict(logdens=None, symbol_logq=slq)


# =========================================================================== #
#  6. Folds  (leave-3-fish-out, 5 stratified folds; each fish tested once)
# =========================================================================== #
def make_folds(fish_ids, n_folds=5, strata=None, seed=0):
    """Partition fish into ``n_folds`` test groups; return [(train_ids, test_ids), ...].

    Each fish is in exactly one test group. With 15 fish and n_folds=5 that is
    leave-3-fish-out. If ``strata`` (dict fish_id->label) is given, folds are
    balanced so each fold's test group spans the strata (round-robin within stratum).
    """
    fish_ids = list(fish_ids)
    rng = np.random.default_rng(seed)
    assign = {}
    if strata is None:
        order = list(rng.permutation(fish_ids))
        for i, f in enumerate(order):
            assign[f] = i % n_folds
    else:
        by = {}
        for f in fish_ids:
            by.setdefault(strata[f], []).append(f)
        k = 0
        for lab in sorted(by):
            grp = list(rng.permutation(by[lab]))
            for f in grp:
                assign[f] = k % n_folds
                k += 1
    folds = []
    for j in range(n_folds):
        test = [f for f in fish_ids if assign[f] == j]
        train = [f for f in fish_ids if assign[f] != j]
        folds.append((train, test))
    return folds


# =========================================================================== #
#  7. Metric A: per-fish predictive scores + CV driver
# =========================================================================== #
def _score_fish(model, rec, dtheta0=DTHETA0):
    """Turn a model's per-bout scores into per-fish summary numbers (nats/bits)."""
    out = model.score(rec)
    row = dict(fish_id=rec.fish_id, model=model.name, n_bouts=rec.n)
    ld = out.get("logdens")
    if ld is not None:
        stim = rec.moving
        row["nats_per_bout"] = float(np.mean(ld))
        row["bits_per_bout"] = float(-np.mean(ld) / LN2)      # cross-entropy bits (lower=better)
        row["nats_stim"] = float(np.mean(ld[stim])) if stim.any() else np.nan
        row["nats_nostim"] = float(np.mean(ld[~stim])) if (~stim).any() else np.nan
        row["n_stim"] = int(stim.sum())
    # EXTRA CHANNELS. Models that fit a joint emission (turn + travel + interval) can return
    # `logdens_dist` / `logdens_dt` alongside the angle `logdens`. Historically only the angle was
    # reported, which meant ~2/3 of what those models optimise was never scored. Adapters that do
    # not emit them are unaffected -- the keys are simply absent.
    # nanmean, NOT mean: the last bout of every fish has no forward-looking distance or dt, so a
    # plain mean would return NaN for every fish. `n_*` records how many bouts actually counted.
    for key, suf in (("logdens_dist", "dist"), ("logdens_dt", "dt")):
        v = out.get(key)
        if v is None:
            continue
        v = np.asarray(v, float); ok = np.isfinite(v)
        row[f"nats_per_bout_{suf}"] = float(np.nanmean(v)) if ok.any() else np.nan
        row[f"bits_per_bout_{suf}"] = float(-np.nanmean(v) / LN2) if ok.any() else np.nan
        row[f"n_{suf}"] = int(ok.sum())
    slq = out.get("symbol_logq")
    if slq is not None:
        idx = rec.symbol.astype(int)
        chosen = slq[np.arange(rec.n), idx]
        row["symbol_nats_per_bout"] = float(np.mean(chosen))
        row["symbol_bits_per_bout"] = float(-np.mean(chosen) / LN2)
        row["symbol_perplexity"] = float(np.exp(-np.mean(chosen)))
    return row


def cross_validate_likelihood(records, model_factories, folds, cfg=None, verbose=True,
                              progress=None):
    """Run leave-3-fish-out CV for Metric A.

    Parameters
    ----------
    records : dict fish_id -> FishRecord.
    model_factories : list of zero-arg callables each returning a fresh
        GenerativeModel (fresh per fold so nothing leaks across folds).
    folds : output of :func:`make_folds`.
    cfg : dict forwarded into the FitContext.

    Returns a tidy pandas DataFrame: one row per (fish, model), with nats/bits per
    bout (overall + stimulus-present/absent strata) and the secondary symbol scores.
    """
    import pandas as pd
    cfg = cfg or {}
    # SKIP THE GENERATION-TIME CALIBRATORS HERE. `score()` reads dth_scale, kin_scales and the
    # fitted latents -- it never touches centre_bias, wall_reject, kin_feedback or kin_innov. So
    # every calibrator fitted during this pass is fitted and then never read, and each one rolls the
    # whole fish set out once per grid point. On a recurrent arm that is ~30 min per fold spent on
    # nothing. Metric B refits from scratch anyway (quick_results runs two independent CV passes),
    # so the calibrated model is still built exactly once, where it is actually used.
    #
    # Opt-out rather than opt-in, and set HERE rather than inside fit(): a model must not have to
    # guess which pass it is in, and any future scorer that DOES read a calibrator would silently
    # get the wrong answer from a guess.
    cfg = dict(cfg)
    cfg.setdefault("skip_calibrators", True)
    rows = []
    total = len(folds) * len(model_factories)      # for the optional progress callback
    done = 0
    for j, (train_ids, test_ids) in enumerate(folds):
        train = [records[f] for f in train_ids]
        ctx = make_context(train, cfg)
        for factory in model_factories:
            model = factory()
            if model.requires_absdict and any(records[f].u_raw is None for f in train_ids):
                if verbose:
                    print(f"  [fold {j}] skip {model.name}: rich features unavailable")
            else:
                model.fit(train, ctx)
                for f in test_ids:
                    row = _score_fish(model, records[f])
                    row["fold"] = j
                    rows.append(row)
            done += 1
            if progress is not None:
                progress(done, total, f"fold {j} · {model.name}")
        if verbose:
            print(f"  fold {j}: trained on {len(train_ids)} fish, scored {test_ids}")
    return pd.DataFrame(rows)


# =========================================================================== #
#  8. Aggregation, skill scores and paired tests
# =========================================================================== #
def add_skill_scores(df, density_ref="marginal_density", symbol_ref="bigram"):
    """Add per-fish skill = bits saved/bout vs a reference baseline (higher=better).

    density_skill = bits_per_bout(ref) - bits_per_bout(model)  (continuous density)
    symbol_skill  = symbol_bits(ref)    - symbol_bits(model)   (3-symbol track)
    """
    df = df.copy()
    ref_d = df[df.model == density_ref].set_index("fish_id")["bits_per_bout"] \
        if "bits_per_bout" in df.columns else None
    ref_s = df[df.model == symbol_ref].set_index("fish_id")["symbol_bits_per_bout"] \
        if "symbol_bits_per_bout" in df.columns else None
    if ref_d is not None:
        df["density_skill_bits"] = df.apply(
            lambda r: ref_d.get(r.fish_id, np.nan) - r.get("bits_per_bout", np.nan)
            if np.isfinite(r.get("bits_per_bout", np.nan)) else np.nan, axis=1)
    if ref_s is not None:
        df["symbol_skill_bits"] = df.apply(
            lambda r: ref_s.get(r.fish_id, np.nan) - r.get("symbol_bits_per_bout", np.nan)
            if np.isfinite(r.get("symbol_bits_per_bout", np.nan)) else np.nan, axis=1)
    return df


def fish_bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0, agg=np.mean):
    """Fish-level (cluster) bootstrap CI of an aggregate over per-fish values."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    point = float(agg(v))
    boot = np.array([agg(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
    lo, hi = float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))
    # keep the point estimate inside [lo, hi] so downstream error bars are non-negative
    return point, min(lo, point), max(hi, point)


def summarize(df, metric="bits_per_bout", n_boot=10000, seed=0):
    """Per-model aggregate of a metric across fish, with fish-level bootstrap CI."""
    import pandas as pd
    out = []
    for model, g in df.groupby("model"):
        if metric not in g.columns:
            continue
        mean, lo, hi = fish_bootstrap_ci(g[metric].to_numpy(), n_boot=n_boot, seed=seed)
        med, mlo, mhi = fish_bootstrap_ci(g[metric].to_numpy(), n_boot=n_boot, seed=seed, agg=np.median)
        out.append(dict(model=model, metric=metric, mean=mean, mean_lo=lo, mean_hi=hi,
                        median=med, n_fish=int(g[metric].notna().sum())))
    return pd.DataFrame(out).sort_values("mean")


def paired_compare(df, model_a, model_b, metric, n_boot=10000, seed=0, higher_better=True):
    """Paired per-fish comparison of two models on one metric.

    Returns sign-test agreement (how many of N fish prefer A), Wilcoxon signed-rank
    p-value, median paired delta (A-B) with a fish-level bootstrap CI, and the number
    of fish. This is the decision-relevant test: 15 folds collapse to ONE paired test.
    """
    from scipy.stats import wilcoxon, binomtest
    a = df[df.model == model_a].set_index("fish_id")[metric]
    b = df[df.model == model_b].set_index("fish_id")[metric]
    common = [f for f in a.index if f in b.index and np.isfinite(a[f]) and np.isfinite(b[f])]
    delta = np.array([a[f] - b[f] for f in common])          # A - B
    pref = delta if higher_better else -delta                # >0 means A preferred
    n = len(delta)
    n_a = int((pref > 0).sum()); n_b = int((pref < 0).sum())
    med, lo, hi = fish_bootstrap_ci(delta, n_boot=n_boot, seed=seed, agg=np.median)
    try:
        w_p = float(wilcoxon(delta).pvalue) if n > 0 and np.any(delta != 0) else np.nan
    except ValueError:
        w_p = np.nan
    sign_p = float(binomtest(max(n_a, n_b), n_a + n_b).pvalue) if (n_a + n_b) > 0 else np.nan
    return dict(model_a=model_a, model_b=model_b, metric=metric, n_fish=n,
                prefer_a=n_a, prefer_b=n_b, ties=n - n_a - n_b,
                median_delta=med, delta_lo=lo, delta_hi=hi,
                wilcoxon_p=w_p, sign_test_p=sign_p)


# =========================================================================== #
#  9. Identity check  (correctness unit test for the scoring core)
# =========================================================================== #
def identity_check(records, spont_model=None, simple_model=None, rich_model=None,
                   abs_dict=None, roi=None):
    """Confirm the per-bout decomposition sum_t log c_t reproduces the reference
    forward-backward joint log-likelihood, for whichever loaded models are given.
    Returns a list of dict rows with the max abs discrepancy per model."""
    import pandas as pd
    rows = []
    recs = list(records.values())
    if spont_model is not None:
        worst = 0.0; total = 0.0
        for r in recs:
            B = np.exp(H.emission_logprob(r.dtheta, spont_model))
            ll_ref, *_ = H._forward_backward(spont_model["pi"], spont_model["T"], B)
            ld, _ = forward_logdens_states(spont_model["pi"], spont_model["T"], B)
            worst = max(worst, abs(ld.sum() - ll_ref)); total += ld.sum()
        rows.append(dict(model="spontaneous", max_abs_diff=worst, total_loglik=total,
                         stored_loglik=spont_model.get("loglik", np.nan)))
    if simple_model is not None:
        m = simple_model; L0 = S._L0_from_params(m["u"]); worst = 0.0; total = 0.0
        for r in recs:
            Tseq = S._build_T_seq(r.b, L0, m["kappa"])
            pi_vec = S.init_dist(m["pi0"], m["kappa"], r.b[0])
            B = S.emission_probs(r.dtheta, r.b, m["sigma"], m["alpha"], m["theta"], m["g"])
            ll_ref, *_ = S._io_forward_backward(pi_vec, Tseq, B)
            ld, _ = forward_logdens_states(pi_vec, Tseq, B)
            worst = max(worst, abs(ld.sum() - ll_ref)); total += ld.sum()
        rows.append(dict(model="simple_stim", max_abs_diff=worst, total_loglik=total,
                         stored_loglik=m.get("loglik", np.nan)))
    if rich_model is not None:
        m = rich_model; L0 = S._L0_from_params(m["u"]); worst = 0.0; total = 0.0
        for r in recs:
            if r.u_raw is None:
                continue
            Us = r.u_raw / m["scales"]
            steer = Us[:, 0:2] @ m["beta_odd"]; prop = Us[:, 2:4] @ m["beta_even"]
            Tseq = S._build_T_seq_g(steer, prop, L0)
            B = S._emission_probs_g(r.dtheta, steer, m["sigma"], m["alpha"], m["theta"], m["g"])
            ll_ref, *_ = S._io_forward_backward(m["pi0"], Tseq, B)
            ld, _ = forward_logdens_states(m["pi0"], Tseq, B)
            worst = max(worst, abs(ld.sum() - ll_ref)); total += ld.sum()
        rows.append(dict(model="rich_stim", max_abs_diff=worst, total_loglik=total,
                         stored_loglik=m.get("loglik", np.nan)))
    return pd.DataFrame(rows)


# =========================================================================== #
#  Convenience default model set
# =========================================================================== #
def default_model_factories(restarts=2, max_iter=None, include_rich=False, seed=0):
    """The standard comparison set as fresh-model factories (one per fold).

    ``max_iter`` (if given) caps EM iterations for every HMM adapter -- lower it to
    keep the CV responsive in the notebook, raise it for publication-quality fits.
    """
    kw = {} if max_iter is None else dict(max_iter=max_iter)
    facs = [
        lambda: MarginalDensity(),
        lambda: UnigramSymbol(),
        lambda: BigramSymbol(),
        lambda: SpontaneousHMM(restarts=restarts, seed=seed, **kw),
        lambda: SimpleStimHMM(restarts=restarts, seed=seed, **kw),
    ]
    if include_rich:
        facs.append(lambda: RichStimHMM(restarts=restarts, seed=seed, **kw))
    return facs
