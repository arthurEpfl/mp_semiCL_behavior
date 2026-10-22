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

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DTHETA0 = 10.0                                     # +/-10 deg forward/turn threshold


# =========================================================================== #
#  1. Per-fish records  (the canonical data structure every model consumes)
# =========================================================================== #
@dataclass
class FishRecord:
    fish_id: int
    dtheta: np.ndarray            # (n,) signed reorientation angle, deg   (the observation)
    symbol: np.ndarray            # (n,) 0=F, 1=L(dtheta<-10), 2=R(dtheta>10)
    stim_angle: np.ndarray        # (n,) egocentric bearing to dots (rad), NaN where undefined
    moving: np.ndarray            # (n,) bool: a moving stimulus present at this bout
    dt_s: np.ndarray              # (n,) inter-bout interval to next bout (s); NaN allowed
    distance_px: np.ndarray       # (n,) travel distance to next bout (px); NaN allowed

    @property
    def n(self) -> int:
        return len(self.dtheta)


def symbols_from_dtheta(dtheta, dtheta0=DTHETA0):
    """0=Forward (|dtheta|<dtheta0), 1=Left (dtheta<-dtheta0), 2=Right (dtheta>+dtheta0)."""
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


def build_records(df, min_len=10, group_cols=("fish_id", "segment_id"), order_col="bout_index"):
    """Split the tidy bout dataframe into per-fish :class:`FishRecord`s.

    One record per (fish_id, segment_id) group (there is one segment per fish),
    ordered by bout_index, keeping only bouts with finite dtheta.
    """
    records = {}
    cols = list(group_cols)
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
        dt_s = (g["dt_s"].to_numpy(float)[good] if "dt_s" in g.columns
                else np.full(len(dth), np.nan))
        dist = (g["distance_px"].to_numpy(float)[good] if "distance_px" in g.columns
                else np.full(len(dth), np.nan))

        records[fid] = FishRecord(
            fish_id=fid, dtheta=dth, symbol=symbols_from_dtheta(dth),
            stim_angle=phi, moving=moving, dt_s=dt_s, distance_px=dist)
    return records


# =========================================================================== #
#  3. Fit context  (everything computed on the TRAINING fold, passed to models)
# =========================================================================== #
@dataclass
class FitContext:
    """Training-fold quantities shared with model adapters (kept leak-free)."""
    dt_pool: np.ndarray                  # inter-bout intervals from TRAIN fish (for generation)
    dist_pool: np.ndarray                # travel distances from TRAIN fish
    rich_scales: np.ndarray | None = None   # unused (None; kept in the saved configs)
    cfg: dict = field(default_factory=dict)


def make_context(train_records, cfg):
    dt = np.concatenate([r.dt_s[np.isfinite(r.dt_s)] for r in train_records])
    dist = np.concatenate([r.distance_px[np.isfinite(r.distance_px)] for r in train_records])
    return FitContext(dt_pool=dt, dist_pool=dist, cfg=dict(cfg))


# =========================================================================== #
#  4. Model adapters  (the harness never inspects a model's internals)
# =========================================================================== #
class GenerativeModel:
    """Adapter contract. Implement fit / score (/ sample for Metric B)."""
    name = "model"
    has_density = True                    # emits a continuous dtheta density (Metric A primary)?
    has_symbol = True                     # emits a {F,L,R} predictive PMF (secondary)?
    can_sample = True                     # can free-run rollouts (Metric B)? baselines cannot

    def fit(self, train_records, ctx):    # -> self
        raise NotImplementedError

    def score(self, rec):
        """Return dict(logdens=(n,) or None, symbol_logq=(n,3) or None)."""
        raise NotImplementedError

    def sample(self, rec, rng):
        """Return dict(dtheta, symbol, dt_s, distance_px) conditioned on rec's stimulus."""
        raise NotImplementedError






