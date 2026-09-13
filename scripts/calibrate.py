"""Magnitude-calibration DIAGNOSTIC (oracle upper bound).

Wraps any generative model and, at sample time, quantile-maps the generated |dtheta|
onto the REAL |dtheta| marginal of the held-out fish (signs kept). This makes the
generated turn-magnitude distribution match real *by construction*, so the resulting
C2ST tells you how low it could go if the emission's magnitude were perfect -- the
residual then comes only from the sign/temporal features (turn-sign ACF, transitions).

It is an ORACLE (uses the test fish's real magnitudes), so it is a diagnostic upper
bound, NOT a deployable model. `score()` (Metric A) is delegated unchanged, since the
calibration is sampling-only.
"""
import numpy as np

import evaluate as E


class MagCalibrated(E.GenerativeModel):
    has_density = True
    has_symbol = True
    can_sample = True
    requires_absdict = False

    def __init__(self, base_factory, name):
        self._base_factory = base_factory
        self.name = name

    def fit(self, train_records, ctx):
        self._base = self._base_factory().fit(train_records, ctx)
        self.requires_absdict = getattr(self._base, "requires_absdict", False)
        return self

    def score(self, rec):
        return self._base.score(rec)                 # density untouched (calibration is sampling-only)

    def sample(self, rec, rng):
        g = dict(self._base.sample(rec, rng))
        gen = np.asarray(g["dtheta"], float)
        real = np.abs(np.asarray(rec.dtheta, float))
        real = real[np.isfinite(real)]
        if len(gen) and len(real):
            ranks = np.argsort(np.argsort(np.abs(gen)))     # 0..n-1, ordered by magnitude
            q = (ranks + 0.5) / len(gen)
            g["dtheta"] = np.sign(gen) * np.quantile(real, q)   # ORACLE: match the real |dtheta| marginal
            g["symbol"] = E.symbols_from_dtheta(g["dtheta"])
        return g
