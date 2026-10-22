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

from dataclasses import dataclass, field

import numpy as np

from twin import fit_hmm as H
from twin import fit_hmm_stim as S

DTHETA0 = H.DTHETA0_DEG                            # +/-10 deg forward/turn threshold


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

    ``path=None`` is ``paths.BOUTS``."""
    import pandas as pd
    from twin import paths
    return pd.read_csv(paths.BOUTS if path is None else path)


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






