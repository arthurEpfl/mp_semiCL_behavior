"""Per-fish random effect: individuality as a FITTED variance component, not a sampler constant.

THE DEFECT. ``between_fish_share`` is 0.061 generated against 0.255 real; the across-fish SD of
forward fraction is 0.0061 against a real 0.0537 (8.8x too narrow) and of median |dtheta| is 1.30 deg
against 6.73 (5.2x). Every generated fish turns like the average fish. This is not a mistuned
mechanism -- there is no mechanism: ``vigor`` covers the interval and distance LEVEL only, and
``_calibrate_vigor`` hard-sets its turn term to zero.

WHY IT MATTERS BEYOND INDIVIDUALITY. Decomposing the missing window variance, ~65% of the missing
turn variance and ~87% of the missing distance variance is BETWEEN-fish. So the under-dispersion
deficit (distance 0.442, dt 0.466 against a target of 1.0) is largely a manifestation of this one,
and under-dispersion is the measured MASKING defect -- an identical injected coupling defect is worth
+0.001 alongside a 20% marginal error and +0.049 once the marginals are repaired. Repairing it here
unmasks everything measured afterwards.

WHY IT ESCAPES THE sigma^2/(1-phi^2) TRAP. A per-trajectory CONSTANT adds window-MEAN variance and
contributes exactly zero to ``dist_acf1``/``dt_acf1``, because ``_acf1`` subtracts the window mean
(generative_metrics.py:491). It does not touch the coupled knob at all. That is the difference from
``nofb``, which fixed the spread perfectly (0.51 -> 0.98) and lost 0.068 AUC because the free-run
lag-1 autocorrelation blew to 0.394 against a real 0.218.

THE MODEL. Latent z_i in R^m, m=2, prior z ~ N(0, I). It enters as a bias offset on the three head
pre-activations and NOWHERE else -- head variances and mixture shapes are shared across z:

    turn:  o[0]           += B_turn z_i     (the FORWARD-mixture weight logit only)
    dist:  do[nd:2nd]     += B_dist z_i     (all log-normal mu shifted equally)
    dt:    to[ndt:2ndt]   += B_dt   z_i

The shared body does NOT see z (``use_body=False``). It did in the first version, and that version
measured +0.0299 WORSE than its own anchor: 90% of the latent's turn effect, 74% of its distance
effect and 59% of its interval effect were flowing through the body rather than the level loadings,
so the mechanism was an unconstrained per-fish reshaping of every conditional rather than the level
shift it was designed to be. The pooled marginals broke (abs_dtheta_w1 6.83 -> 10.29, dist_w1
3.33 -> 4.93 -- a level shift cannot do that), distance dispersion overshot 0.40 -> 2.03, wall
contact rose with it, and radial went from 0.22 to 0.56 of all remaining detectability. The body
path is retained behind a flag purely so those checkpoints still load.

Level-only is not a shortcut: ``smoke_state_layer`` recorded that "with a free per-state emission the
chain spends its states on the MARGINAL instead". And the turn loading sits on the forward-mixture
WEIGHT, not on a scale -- scaling |dtheta| is exactly how ``vigor_turn`` failed, and forward fraction
spans 4.5x across the 15 fish while median |dtheta| spans only 2x, so weight-tilting is the axis real
fish actually vary on and it leaves the pooled turn marginal alone. m=2 comes from the measured ICCs:
one turn-style axis (forward fraction 0.835, |dtheta| median 0.796), one tempo axis (bout rate 0.893).

THREE-STAGE ESTIMATION.
  A. FIT. z_i are learnable embedding rows trained jointly with the net on the existing shuffled-row
     minibatches, with the Gaussian prior penalty added in every batch. ~187 optimiser steps/epoch --
     no sequence batching, so none of the step-collapse risk a recursion would bring.
     The prior is FIXED at N(0, I) and the scale lives in the loadings. That is the standard factor
     parameterisation and it is not cosmetic: with both the prior SD and the loadings free, only
     their product is identified, and re-estimating Sigma by EM afterwards then diverges -- measured,
     on 3 fish: sd 0.49 -> 27.0 over five iterations, because each iteration widens the quadrature
     grid that the next iteration's second moment is computed on.
  B. DE-SHRINK. Joint MAP still shrinks the fitted product toward zero, so the generated between-fish
     spread would come out too narrow -- and ``vigor``'s intercept is moment-matched to the real
     spread by construction, so an uncorrected swap would REGRESS rather than merely fail to help.
     The correction is a single global gain on the loadings, chosen on a BOUNDED grid by marginal
     likelihood. One dimension, cannot run away, and the fitted gain is the reportable number.
  C. SCORE. A held-out fish has no embedding, so ``logdens`` is the exact marginal predictive
     log p(y_t | y_{1:t-1}, x) = log int p(y_t|z) p(z|y_{1:t-1}) dz. Because z is STATIC this is a
     cumsum, not a loop, and it is exact up to Gauss-Hermite quadrature.

FITTED AND SCORED, which is the point. ``vigor`` and ``_search_indiv`` are sequentially-fitted
generation-time constants that ``score()`` never reads; this replaces them with one prior estimated by
maximum likelihood. It is the largest single reduction of the "two objects, one likelihood" surface
available in this project.

GENERATION. z is drawn once per trajectory from the fitted prior -- the honest "new fish" claim, and
the only option that cannot leak, since the evaluated fish is held out. The draw lives in
``JT.generate_mix_joint_loop`` (keyed off ``cfg_feat['fish_re']``) rather than here, so the shared
calibrators reach it through ``MX._gen_for`` unchanged and automatically calibrate against PRIOR
draws -- which is what they must do, since they roll out train fish and the fitted z_i would
calibrate against a different law than the one evaluated.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from twin import evaluate as E                    # noqa: E402
from twin import fit_MLP as M                     # noqa: E402
from twin import fit_MLP_border as MB             # noqa: E402
from twin import fit_MLP_joint as JT              # noqa: E402
from twin import fit_MLP_mix as MX                # noqa: E402


# =========================================================================== #
#  Net
# =========================================================================== #
class MDNMixJointRE(JT.MDNMixJoint):
    """``MDNMixJoint`` plus a per-fish latent loading level-only on all three heads.

    ``joint_forward`` is inherited and runs at z = 0, so every existing consumer (the numpy
    predictor, ``predict_params_mix``, the symbol track) keeps working on the population-average
    fish. The random effect is reached through ``joint_forward_re``.
    """

    def __init__(self, n_features, hidden, n_dist=2, n_dt=3, n_fish=1, m=2, sigma_init=0.5,
                 use_body=False, border_gain=False, border_slice=None, dt_latent=None,
                 block=False, drift=None, state=0, h0_dim=0, disp=False, lag_slice=None,
                 tail=False, skip_slices=None, depth=1, tsize=False):
        super().__init__(n_features, hidden, n_dist=n_dist, n_dt=n_dt, skip_slices=skip_slices,
                         depth=depth)
        if skip_slices and int(state):
            raise ValueError("the turn-head skip path is not implemented on the recurrent path")
        # RECURRENT STATE. A GRU cell carried through the rollout, updated every bout, feeding all
        # three heads. It replaces the whole hand-built latent zoo -- drift's pinned AR(1),
        # fish_re's static z, kin_feedback's shrinkage -- with ONE object whose timescales are
        # LEARNED. Under MLE those timescales were not identifiable (the likelihood put dt_latent's
        # tau at 107-152 against a much shorter real value); under a rollout objective they are,
        # because the statistic being scored IS the long-timescale one.
        #
        # The cell reads the feature row PLUS the previous emission, because in a free run the row
        # already carries lagged turns and kinematics but not this bout's own outcome.
        self.state_hidden = int(state)
        # PER-FISH SEED FOR THE RECURRENT STATE (h0_dim > 0). Measured on gru1: with h_0 pinned at
        # 0 the state produces excellent WITHIN-fish slow structure (var_ratio distance 0.63 ->
        # 1.04, dt 0.44 -> 0.91) but essentially no BETWEEN-fish spread -- between_fish_share
        # collapsed to 0.071 against a real 0.255, worse even than the MLP's 0.215. Individuality
        # did not emerge endogenously from the sampled history, so it has to be given.
        #
        # A LOW-DIMENSIONAL latent projected into the state, not a free 64-d h_0, and that is what
        # makes it affordable: Metric A must marginalise it out for a held-out fish, and a 2-d
        # latent is a 49-node GHQ grid whose nodes BATCH through state_chain -- so the exact
        # marginal costs about the same as a single pass, because the loop is dispatch-bound.
        # Marginalising a full 64-d h_0 would be impossible; this is the same trick fish_re uses.
        self.h0_dim = int(h0_dim) if self.state_hidden else 0
        if self.h0_dim:
            self.h0_map = nn.Linear(self.h0_dim, self.state_hidden, bias=False)
            self.h0_emb = nn.Embedding(int(n_fish), self.h0_dim)
            nn.init.normal_(self.h0_emb.weight, std=0.5)
            nn.init.normal_(self.h0_map.weight, std=0.3)
        if self.state_hidden:
            self.state_cell = nn.GRUCell(n_features + 3, self.state_hidden)
            w = hidden + self.state_hidden          # heads read [body(x), h]
            self.head = nn.Linear(w, 14)
            self.dist_head = nn.Linear(w + 1, 3 * int(n_dist))
            self.dt_head = nn.Linear(w + 2, 3 * int(n_dt))
        # DRIFT: one AR(1) state loading on ALL THREE heads (see DRIFT_DEFAULT for the measurement
        # that motivates it). Parameters exist only when enabled, so archived state_dicts load
        # unchanged and `dt_latent` arms are untouched.
        self.drift = dict(DRIFT_DEFAULT, **drift) if drift else None
        if self.drift:
            if dt_latent and not self.drift.get("dt_off"):
                raise ValueError("drift and dt_latent are the same kind of object (an AR(1) state "
                                 "on a head location) and would compete for the same variance; "
                                 "drift already loads on dt. Enable one -- or set drift['dt_off'] "
                                 "for the TWO-STATE design (slow chain on turn + step, dt_latent "
                                 "as the faster interval chain).")
            # Same saddle argument as the fishre loadings: start SMALL BUT NONZERO. At lam = 0 every
            # level gives an identical likelihood, the posterior never leaves the prior, and the
            # state is silently inert for the whole fit.
            li = float(self.drift["lam_init"])
            if self.drift.get("turn_off"):
                # TURN LOADING OFF. Model 9: without the interval lags the state became a large
                # "vigour" state (turn loading +-0.5-0.8) whose swings the step lags amplified into a
                # 3x window-to-window step variance on the S3 arms; the real turn slow-acf is weak
                # (+0.13). A buffer, exactly 0, so the state is kinematic only.
                self.register_buffer("d_turn", torch.tensor(0.0))
            else:
                self.d_turn = nn.Parameter(torch.tensor(li))
            self.d_dist = nn.Parameter(torch.tensor(li))
            if self.drift.get("dt_off"):
                # TWO-STATE DESIGN. The real window-acf profile decays slowly for step / radial
                # (tau ~70+) and fast for the interval (tau ~20-40), so one shared timescale is right
                # for the step and wrong for the interval: the dt loading of the shared state never
                # settled (-0.04..+0.10 across folds) while the step loading did (+0.10..+0.14). Here
                # the slow chain loads on turn + step only and the interval gets its own chain
                # (dt_latent with a PINNED phi). A buffer, not a Parameter: no gradient, saved as
                # exactly 0, and every consumer of d_dt (`_params_at`, the published cfg) is unchanged.
                self.register_buffer("d_dt", torch.tensor(0.0))
            else:
                self.d_dt = nn.Parameter(torch.tensor(li))
            # d_bord is NOT a Parameter and that is deliberate. It scales `centre_bias`, which is a
            # GENERATION-TIME term applied inside the rollout -- it appears nowhere in the
            # likelihood, so there is no gradient to fit it with and pretending otherwise would give
            # a parameter that silently never moves. It is fitted by the rollout search
            # (MX._search_drift_gain) against the measured radial acf, and carried as a plain float.
            self.d_bord = float(self.drift.get("bord", 0.0))
        # SLOW TIMING STATE (see the tauchen section above). Scoped to the interval head's
        # location; the parameters exist only when enabled, so archived state_dicts load unchanged.
        self.dt_latent = dict(DT_LATENT_DEFAULT, **dt_latent) if dt_latent else None
        if self.dt_latent:
            # lam starts SMALL BUT NONZERO, not at 0. lam = 0 is not a saddle for lam itself (the
            # gradient is E_j[s_j * d logpdf/d mu], generally nonzero), but it IS one for phi: with
            # lam = 0 every level gives an identical likelihood, the posterior never leaves the
            # prior, and d/dphi is exactly zero. Starting lam off zero gives the timescale gradient
            # from the first step instead of making it wait for lam to drift.
            self.s_lam = nn.Parameter(torch.tensor(float(self.dt_latent["lam_init"])))
            # phi_fixed: PIN the timescale at an independently measured value and fit only the
            # amplitude. Round 2a fitted phi freely and it settled at tau = 107-152 bouts against a
            # measured real slow timescale of ~55 -- slow enough that on a 4700-bout record the
            # state stops being a within-trajectory phenomenon and starts impersonating the
            # per-fish latent, which is why it inflated EVERY head (dist 143%, dt 158%) and pushed
            # between_fish_share to 0.304 against a real 0.255. Pinning removes that competition by
            # construction rather than by a bound the optimiser can crawl toward, and it is the
            # same discipline the loading gain follows: timescale measured, everything else ML.
            # No s_phi_raw parameter is created at all, so it cannot drift and cannot be restored
            # into a free state by a later refit.
            if self.dt_latent.get("phi_fixed") is None:
                p0 = float(np.clip(self.dt_latent["phi_init"], PHI_LO + 1e-4, PHI_HI - 1e-4))
                u = (p0 - PHI_LO) / (PHI_HI - PHI_LO)
                self.s_phi_raw = nn.Parameter(torch.tensor(float(np.log(u / (1.0 - u)))))
        self.m = int(m)
        self.n_fish = int(n_fish)
        self.use_body = bool(use_body)
        # LIKELIHOOD-VISIBLE spatial individuality (re_bgain): a per-fish GAIN on the border
        # input columns, g = exp(clamp(b.z, -1.5, 1.5)) -- how strongly this fish responds to
        # the wall covariate is itself a fitted trait. This is the CONSTRAINED entry point the
        # body path was not: one scalar direction that can only amplify/attenuate the wall
        # response, not reshape arbitrary conditionals. The parameter exists only when enabled,
        # so archived state_dicts load unchanged. Use with re_m=3: at m=2 the two axes are
        # already claimed (turn style, tempo) and a border gain through them would force spatial
        # personality to be a linear combination of the other two, which the ICCs contradict.
        self.border_gain = bool(border_gain)
        self.border_slice = tuple(border_slice) if border_slice is not None else None
        # BLOCK-STRUCTURED LATENT. Without it every loading reads every dimension, so "re_m=3 gives
        # re_bgain a spatial axis" was a hope, not a constraint -- three free dimensions fed four
        # loadings and the extra capacity went to noise, which is exactly what the m=3 arm measured
        # (fitted gain 0.28-0.40, the pre-registered fitting-noise band). With block=True the first
        # `n_core = m - 1` dimensions feed turn/dist/dt and the LAST one feeds the border gain and
        # nothing else.
        #
        # Enforced by the WEIGHT SHAPES, not by slicing at call time: there is simply no parameter
        # connecting the spatial dimension to the turn head, so leakage is impossible rather than
        # merely absent. (A runtime slice with full-width Linears would leave the weights there,
        # one refactor away from being read again.) Archived checkpoints are unaffected because
        # block defaults False and the shapes stay (m, 1).
        self.block = bool(block)
        if self.block and not self.border_gain:
            raise ValueError("block=True is meaningless without border_gain: there would be no "
                             "second block for the reserved dimension to feed.")
        if self.block and self.m < 2:
            raise ValueError(f"block=True needs m >= 2 (got {self.m}): one dimension is reserved "
                             "for the border gain, leaving none for turn/dist/dt.")
        self.n_core = (self.m - 1) if self.block else self.m
        n_sp = self.m - self.n_core if self.block else self.m
        self.emb = nn.Embedding(self.n_fish, self.m)
        self.z_body = nn.Linear(self.m, hidden, bias=False)
        self.z_turn = nn.Linear(self.n_core, 1, bias=False)
        self.z_dist = nn.Linear(self.n_core, 1, bias=False)
        self.z_dt = nn.Linear(self.n_core, 1, bias=False)
        if self.border_gain:
            if self.border_slice is None:
                raise ValueError("border_gain=True needs border_slice=(start, stop)")
            self.z_bord = nn.Linear(n_sp, 1, bias=False)
        # BOTH factors must start nonzero. z enters only as (loading x embedding), so zeroing both
        # -- the tempting "start exactly at the population model" init -- is an exact saddle: the
        # gradient w.r.t. the embedding is proportional to the loadings and vice versa, so both stay
        # at zero forever and the mechanism is silently inert for the whole fit. (Measured: the MAP
        # embedding spread came out exactly [0, 0].) The embedding is drawn from the prior, which is
        # what an empirical-Bayes initialisation should be; the loadings get a small random init so
        # the function still starts within noise of the population model.
        nn.init.normal_(self.emb.weight, std=float(sigma_init))
        for lin in (self.z_turn, self.z_dist, self.z_dt):
            nn.init.normal_(lin.weight, std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)
        if self.border_gain:
            # same saddle-avoidance as the other loadings: zero-init would never leave zero
            nn.init.normal_(self.z_bord.weight,
                            std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)

        # DISPERSION AXIS (re_disp, Model 12). z also loads on the interval head's pre-softplus
        # WIDTHS, one loading per mixture component. MEASURED need: the between-fish spread of the
        # step is a level effect (SD/median 0.43-0.58 across the 15 fish) and the location loading
        # reproduces it, but the interval's is not (SD/median 0.74-1.20; fish 0 has 3x the typical
        # rate of > 5 s intervals on top of being the slowest), so the top fish on interval spread
        # is unreachable by a location shift. Same saddle rule as the other loadings: small nonzero
        # init. Applied identically in the numpy generator (JT._numpy_predictor_joint).
        self.disp = bool(disp)
        if self.disp:
            self.z_dt_sig = nn.Linear(self.n_core, int(n_dt), bias=False)
            nn.init.normal_(self.z_dt_sig.weight, std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)
        # TURN SIZE (re_tsize, Model 38): z also loads, through ONE shared loading, on the pre-softplus SCALES of the
        # four turning Gamma components (turn-head outputs 7, 9, 11, 13), so how far a fish turns when it turns is a
        # trait alongside how often it swims straight (the forward logit, z_turn). MEASURED (clean data): mean turn
        # size of the turning bouts spans 42-64 deg across the 15 fish, split-half r = 0.95, and accounts for most of
        # the between-fish spread of mean |dtheta|, which the forward logit alone cannot reach. Symmetric left/right,
        # and linear in z, so its mean over N(0, I) is zero: no re-centring. Same saddle rule as the other loadings.
        # Applied identically in the numpy generator (JT._numpy_predictor_joint).
        self.tsize = bool(tsize)
        if self.tsize:
            # the global RNG is restored afterwards, so every OTHER weight starts exactly as in the same model without
            # the loading (a paired comparison, not one confounded by an initialisation redraw)
            _rng_state = torch.get_rng_state()
            self.z_tsize = nn.Linear(self.n_core, 1, bias=False)
            nn.init.normal_(self.z_tsize.weight, std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)
            torch.set_rng_state(_rng_state)
        # LAG SLOPE (re_lag, Model 12): a z x interval-lag interaction on the interval head's
        # location -- mu_dt += sum_k (L z)_k * x[lag_k] over the interval-lag columns of the feature
        # row (``lag_slice``), so how strongly one interval predicts the next is itself a fish trait
        # (burstiness: fish 6's within-window lag-1 acf is 0.19 against 0.02-0.09 for the rest).
        # Not on the recurrent path: the GRU trunk does not expose the feature row.
        self.lag_slice = tuple(int(v) for v in lag_slice) if lag_slice else None
        if self.lag_slice:
            if self.state_hidden:
                raise ValueError("re_lag is not implemented on the recurrent (state) path")
            self.z_lag = nn.Linear(self.n_core, self.lag_slice[1] - self.lag_slice[0], bias=False)
            nn.init.normal_(self.z_lag.weight, std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)
        # TAIL WEIGHT (re_tail, Model 13): z loads on the interval mixture's component LOGITS, so
        # how OFTEN a fish produces long intervals is a trait, independent of its level. MEASURED
        # (Model 11 S3 arm, teacher-forced at fish 0's own fitted z): the model puts 2.5-5.0 % of
        # intervals above 5 s against the fish's real 6.2 % (population fish 2-3.6 %), and fish 6
        # (real 0.9 %) comes out 1.1-2.6 % -- a location shift cannot express the tail rate; the
        # dispersion loading (Model 12) widened every fish instead. Bounded by the softmax, so no
        # re-centring; gets its OWN rollout gain (search_tail_gain, matched to the between-fish
        # spread of the window interval spread), not the interval gain that is matched to the mean.
        self.tail = bool(tail)
        if self.tail:
            self.z_dt_w = nn.Linear(self.n_core, int(n_dt), bias=False)
            nn.init.normal_(self.z_dt_w.weight, std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)

        # z_body stays EXACTLY zero unless explicitly enabled, so it receives no gradient and the
        # mechanism is genuinely level-only. See the class docstring for why this defaults off.
        if self.use_body:
            nn.init.normal_(self.z_body.weight,
                            std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)
        else:
            nn.init.zeros_(self.z_body.weight)

    def _z_core(self, z):
        """The dimensions that feed turn / distance / interval."""
        return z[:, :self.n_core] if self.block else z

    def _z_spatial(self, z):
        """The dimensions that feed the border gain (all of them unless block-structured)."""
        return z[:, self.n_core:] if self.block else z

    def s_phi(self):
        """Fitted AR(1) persistence, bounded into [0.50, 0.999].

        The upper bound is not cosmetic: at phi = 1 the chain is a random walk, the stationary
        variance the whole parameterisation assumes stops existing, and the free-run wanders --
        the same near-unit-root failure the kin memory work measured at K = 32 (acf@50 +0.57
        against a real +0.042) with no detector anywhere in the pipeline."""
        pf = (self.dt_latent or {}).get("phi_fixed")
        if pf is not None:
            return torch.tensor(float(pf), dtype=torch.float32,
                                device=self.s_lam.device)     # constant: no gradient, no drift
        return PHI_LO + (PHI_HI - PHI_LO) * torch.sigmoid(self.s_phi_raw)

    def h0_from(self, w):
        """(B, state_hidden) initial GRU state from the per-fish latent ``w`` (B, h0_dim).

        ``tanh`` because a GRU hidden unit lives in (-1, 1): an unbounded seed would start the
        trajectory outside the range the cell's own dynamics can produce, and the first bouts would
        be spent relaxing back rather than expressing the fish.
        """
        return torch.tanh(self.h0_map(w))

    def state_chain(self, X, a_z, ld_c, u, ya, yd, yt, h0=None, z=None):
        """(la, ld, lt, h_T): teacher-forced per-bout log-densities along B ordered sequences.

        THE POINT OF TEACHER FORCING. With the targets observed the GRU is deterministic, so
        ``log p(y_1..T | latent) = sum_t log p(y_t | y_<t, latent)`` is exact with no marginalisation
        INSIDE the chain -- no forward recursion, no product grid over time. Any per-trajectory
        latent is therefore STATIC given the sequence, and integrating it out is a plain reweighting
        of B parallel chains (see ``score``). That is what keeps Metric A exact for a recurrent
        model, and it is why a full 64-d random ``h_0`` is not offered: it could not be integrated
        affordably, and putting the headline number behind an approximation is not worth it.

        TWO LATENTS, TWO JOBS, AND THEY ARE NOT INTERCHANGEABLE.
          * ``z`` (fish_re) shifts the three head LOCATIONS -- a static per-fish offset. This is
            BETWEEN-fish identity, and it is the quantity gru1 measured at 0.071 against a real
            0.255 with no latent at all.
          * the GRU state ``h`` evolves bout to bout. This is WITHIN-fish slow structure, and it is
            the quantity gru1 fixed (var_ratio distance 0.63 -> 1.04, dt 0.44 -> 0.91).
        gru1 moved those two independently, which is the measurement saying they are ORTHOGONAL --
        so the "three mechanisms competing for one variance" objection that kept fish_re off does
        not apply to this pair. ``z`` is applied EXACTLY as ``joint_forward_re`` applies it, and
        exactly as ``rollout_torch.rollout`` already did at generation time; before this the
        recurrent path silently bypassed the latent on BOTH the train and score sides while the
        generator was still applying it, which is the two-objects trap in its purest form.

        Shapes: ``X`` (B, T, D); ``a_z`` / ``ld_c`` / ``ya`` / ``yd`` / ``yt`` (B, T); ``u`` (B, T, 3)
        is the GRU's emission input and MUST be built the same way the rollout builds it
        (``[dtheta/dth_scale, ld_c, (log dt - tm)/ts]``) -- if the two disagree, the state means one
        thing at train time and another at generation time, which is this project's oldest trap.
        ``z`` is (B, m), ONE latent per sequence, broadcast over T because it is static.
        Returns the final state so a long record can be scored in chunks that carry it across.
        """
        B, T, _ = X.shape
        h = torch.zeros((B, self.state_hidden), device=X.device) if h0 is None else h0
        # None rather than a zero vector when the latent is off: `z_turn(0)` is numerically the same
        # but not the same GRAPH, and the pinned-latent arm (gru1) has to stay bit-identical.
        zc = None if z is None else self._z_core(z)
        la, ld, lt = [], [], []
        for t in range(T):
            x = X[:, t]
            trunk = torch.cat([self.body(x), h], dim=1)
            out, do, to = self._state_heads(trunk, a_z[:, t:t + 1], ld_c[:, t:t + 1], zc)
            ang, dpar, tpar = self._params_at(out, do, to)
            la.append(MX.mix_logdens(ya[:, t], ang))
            ld.append(MX._lnk_logpdf(torch.clamp(yd[:, t], min=1e-6), *dpar))
            lt.append(MX._dt_logpdf(torch.clamp(yt[:, t], min=1e-6), *tpar, hi=self.dt_trunc))
            h = self.state_cell(torch.cat([x, u[:, t]], dim=1), h)
        return (torch.stack(la, 1), torch.stack(ld, 1), torch.stack(lt, 1), h)

    def _state_heads(self, trunk, a_z_t, ld_c_t, zc):
        """Raw head outputs ``(out, do, to)`` from a recurrent trunk ``[body(x), h]``.

        Factored out ONLY so ``state_chain`` and ``state_chain_levels`` cannot drift apart: they must
        apply the per-fish latent identically or the likelihood and its level-marginalised version
        would describe different models. Returns the RAW outputs so ``_params_at`` -- shared with the
        non-recurrent path -- owns the drift shift and the softplus/softmax, in one place.
        """
        k, kt = self.n_dist, self.n_dt
        out = self.head(trunk)
        do = self.dist_head(torch.cat([trunk, a_z_t], dim=1))
        to = self.dt_head(torch.cat([trunk, a_z_t, ld_c_t], dim=1))
        if zc is not None:
            out = self._turn_with_z(torch.cat([out[:, :1] + self.z_turn(zc), out[:, 1:]], dim=1), zc)
            do = torch.cat([do[:, :k], do[:, k:2 * k] + self.z_dist(zc), do[:, 2 * k:3 * k]], dim=1)
            to = self._dt_with_z(to, zc)
        return out, do, to

    def _dt_with_z(self, to, zc, x=None):
        """The interval head's raw outputs with EVERY z loading applied, in one place so the
        non-recurrent and recurrent paths cannot drift apart: the location shift (z_dt), the
        dispersion loading on the pre-softplus widths (re_disp) and the lag-slope term (re_lag:
        z x the interval-lag columns of ``x``, on the location)."""
        kt = self.n_dt
        lg = to[:, :kt]
        mu = to[:, kt:2 * kt] + self.z_dt(zc)
        sig = to[:, 2 * kt:3 * kt]
        if getattr(self, "tail", False):
            lg = lg + self.z_dt_w(zc)
        if getattr(self, "disp", False):
            sig = sig + self.z_dt_sig(zc)
        if getattr(self, "lag_slice", None) is not None:
            if x is None:
                raise ValueError("re_lag needs the feature row on this path")
            s0, s1 = self.lag_slice
            mu = mu + (self.z_lag(zc) * x[:, s0:s1]).sum(dim=1, keepdim=True)
        return torch.cat([lg, mu, sig], dim=1)

    def state_trunks(self, X, u, h0=None):
        """(B, T, hidden + state_hidden): the recurrent trunk ``[body(x_t), h_t]`` at every bout.

        THE COST ARGUMENT, and the reason a product grid is affordable on a recurrent model at all.
        The GRU reads only the feature row and the OBSERVED previous emission -- it sees neither the
        per-fish latent ``z`` nor the drift level ``s``. So the recurrent pass is computed ONCE per
        record and every one of the Q x L (z-node, drift-level) combinations reuses it; the grid then
        costs three small head evaluations per combination, not a re-run of the state.

        That is what turns 49 x 9 = 441 states per bout into the same order of cost the `dt_latent`
        arm already paid, rather than another 3969-state refusal.
        """
        B, T, _ = X.shape
        h = torch.zeros((B, self.state_hidden), device=X.device) if h0 is None else h0
        tr = []
        for t in range(T):
            x = X[:, t]
            tr.append(torch.cat([self.body(x), h], dim=1))
            h = self.state_cell(torch.cat([x, u[:, t]], dim=1), h)
        return torch.stack(tr, dim=1)

    def state_chain_levels(self, X, a_z, ld_c, u, ya, yd, yt, nodes, h0=None, z=None,
                           trunks=None, split=False):
        """Per-bout log-likelihood at each drift level, with the GRU state carried.

        ``split=False`` (the trainer) returns (B, T, L) with turn + distance + interval SUMMED and
        the kinematic masks folded in -- what the forward recursion consumes. The drift loads on all
        three, which is the whole point of it over ``dt_latent`` and the only way to reproduce the
        real cross-channel coupling (corr(log dt, log dist) = -0.386, corr(log dt, |dth|) = +0.308)
        that three independent per-channel latents sit at ~0 on by construction.

        ``split=True`` (the scorer) returns the three channels SEPARATELY, each (B, T, L), unmasked.
        They have to be apart there because the scorer reweights the belief WITHIN a bout, in the
        same chained order the model factorises: distance is scored under a posterior that has
        already seen this bout's turn, and the interval under one that has also seen its distance.
        Summing first would double-count the evidence and the three would no longer chain-rule to
        log p(dtheta, l, tau | past).

        Masked kinematic bouts contribute 0 to EVERY level in the summed form -- not NaN, and not a
        real density -- so the chain still transitions through them without the belief being updated
        by a bout the model was never fitted on.
        """
        B, T, _ = X.shape
        if trunks is None:
            trunks = self.state_trunks(X, u, h0)
        # FLATTEN (B, T) -> rows and evaluate the heads ONCE over every bout. Only the GRU needs a
        # Python loop over time (it is a recursion); the heads are position-independent given the
        # trunk, so looping them per bout costs Q x T x L iterations where Q x L suffices. Measured
        # the wrong way round first: scoring one 3257-bout record was ~1.4M head calls.
        tr = trunks.reshape(B * T, -1)
        zc = None
        if z is not None:
            # one z per SEQUENCE, so repeat it down the flattened rows -- interleaved to match
            # reshape's row-major order, or fish i's latent would be applied to fish j's bouts.
            zc = self._z_core(z).repeat_interleave(T, dim=0)
        out, do, to = self._state_heads(tr, a_z.reshape(-1, 1), ld_c.reshape(-1, 1), zc)
        ydc = torch.clamp(yd.reshape(-1), min=1e-6)
        ytc = torch.clamp(yt.reshape(-1), min=1e-6)
        yaf = ya.reshape(-1)
        pa, pd, pt = [], [], []
        for j in range(nodes.shape[0]):
            ang, dpar, tpar = self._params_at(out, do, to, nodes[j])
            pa.append(MX.mix_logdens(yaf, ang))
            pd.append(MX._lnk_logpdf(ydc, *dpar))
            pt.append(MX._dt_logpdf(ytc, *tpar, hi=self.dt_trunc))
        la = torch.stack(pa, 1).reshape(B, T, -1)
        ld = torch.stack(pd, 1).reshape(B, T, -1)
        lt = torch.stack(pt, 1).reshape(B, T, -1)
        if split:
            return la, ld, lt
        md = (torch.isfinite(yd) & (yd > 0)).float().unsqueeze(2)
        mt = (torch.isfinite(yt) & (yt > 0)).float().unsqueeze(2)
        return la + md * ld + mt * lt

    def d_phi(self):
        """Drift persistence -- a CONSTANT, pinned from ``tau``, never a parameter.

        There is deliberately no ``d_phi_raw``: the timescale cannot drift, cannot be restored into
        a free state by a later refit, and cannot trade against the per-fish latent the way
        ``dt_latent``'s fitted phi did (it settled at tau = 107-152 against a measured ~22). See
        DRIFT_DEFAULT for the measurement that sets tau.
        """
        return torch.tensor(drift_phi(self.drift["tau"]), dtype=torch.float32,
                            device=self.d_turn.device)

    def dt_level_logliks(self, tpar, y_dt, nodes):
        """(n, L) ``log p(tau | ..., s = nodes[j])``, reusing ONE dt-head forward pass.

        The latent only shifts the interval mixture's LOCATION, so the head output is computed
        once and each level costs a single log-normal-mixture evaluation rather than a full
        forward pass. That is what keeps an L-level marginalisation affordable inside training."""
        logw, mu, sig = tpar
        y = torch.clamp(y_dt, min=1e-6)
        return torch.stack([MX._dt_logpdf(y, logw, mu + self.s_lam * nodes[j], sig, hi=self.dt_trunc)
                            for j in range(nodes.shape[0])], dim=1)

    def joint_forward_re(self, x, a_z, ld_c, z):
        """Same chain as ``joint_forward`` with the latent added. ``z`` is (n, m).

        The body path is OFF by default and that is a measured decision, not a default chosen for
        tidiness. With it on, 90% of the latent's effect on the turn head, 74% on the distance head
        and 59% on the interval head flowed through ``z_body`` rather than the level loadings -- so
        the mechanism was not level-only at all, but an unconstrained per-fish reshaping of every
        conditional. Two things followed in the first full ladder: the pooled marginals degraded
        (abs_dtheta_w1 6.83 -> 10.29, dist_w1 3.33 -> 4.93, which a level shift cannot cause), and
        the analytic re-centring in ``apply_loading_gain`` was correcting only the level path and so
        touched roughly a quarter of the distance effect. Radial detectability went 0.22 -> 0.56 of
        total importance and the arm finished +0.0299 WORSE than its anchor.
        """
        out, do, to = self._trunk(x, a_z, ld_c, z)
        return self._params_at(out, do, to)

    def _trunk(self, x, a_z, ld_c, z):
        """Raw head outputs ``(out, do, to)`` with the per-FISH latent already added.

        Split out so ``drift_level_logliks`` can evaluate L latent levels from ONE body pass: the
        drift only shifts locations, so each extra level costs a softmax and a log-density rather
        than a forward pass. That is what keeps an L-level marginalisation affordable in training.
        """
        b, zc, out, x = self._trunk_turn(x, z)
        return out, self._trunk_dist(b, zc, x, a_z), self._trunk_dt(b, zc, x, a_z, ld_c)

    # The three stages of ``_trunk``, exposed so a batched ROLLOUT (scripts/rollout_batch.py) can run the body once per
    # bout and each later head only on its own conditioners -- ``_trunk`` is their composition, so the two cannot drift.
    # turn-head outputs that are the four turning Gamma components' pre-softplus SCALES (L1, L2, R1, R2)
    _TSIZE_COLS = (7, 9, 11, 13)

    def _turn_with_z(self, out, zc):
        """The turn-size loading (re_tsize) on the four Gamma scales; identity when it is off, so every other
        model is bit-identical."""
        if not getattr(self, "tsize", False):
            return out
        sh = self.z_tsize(zc)                                            # (n, 1), one shift for all four scales
        cols = [out[:, k:k + 1] + sh if k in self._TSIZE_COLS else out[:, k:k + 1] for k in range(out.shape[1])]
        return torch.cat(cols, dim=1)

    def _trunk_turn(self, x, z):
        """(body activation, core latent, turn head's raw outputs, x as the heads read it)."""
        if self.border_gain:
            # bounded per-fish gain on the border columns only: exp(+-1.5) caps the swing at
            # 0.22-4.5x, and clamp(0)=0 keeps the z=0 path (joint_forward, the population fish)
            # bit-identical to a net without the mechanism.
            g = torch.exp(torch.clamp(self.z_bord(self._z_spatial(z)), -1.5, 1.5))
            s0, s1 = self.border_slice
            x = torch.cat([x[:, :s0], x[:, s0:s1] * g, x[:, s1:]], dim=1)
        b = torch.relu(self.body[0](x) + self.z_body(z)) if self.use_body else self.body(x)
        zc = self._z_core(z)
        h = self.head(b) + self.skip_term(x)
        out = self._turn_with_z(torch.cat([h[:, :1] + self.z_turn(zc), h[:, 1:]], dim=1), zc)
        return b, zc, out, x

    def _trunk_dist(self, b, zc, x, a_z):
        a = a_z.reshape(-1, 1)
        k = self.n_dist
        do = self.dist_head(torch.cat([b, a], dim=1))
        return torch.cat([do[:, :k], do[:, k:2 * k] + self.z_dist(zc) + self.step_term(x), do[:, 2 * k:3 * k]], dim=1)

    def _trunk_dt(self, b, zc, x, a_z, ld_c):
        to = self.dt_head(torch.cat([b, a_z.reshape(-1, 1), ld_c.reshape(-1, 1)], dim=1))
        return self._dt_with_z(to, zc, x)

    def _params_at(self, out, do, to, s=None):
        """Emission parameters, optionally with the drift state at value ``s``.

        ``s=None`` means NO drift shift and reproduces ``joint_forward_re`` exactly -- which is what
        keeps every non-drift arm bit-identical, since the drift branch is then never entered.

        The drift shifts the SAME three quantities the per-fish latent shifts -- the first turn
        mixture logit and the two kinematic locations -- so it is level-only by construction, for
        the reason spelled out above ``DT_LATENT_DEFAULT``: a chain with a free per-state emission
        spends its states on the marginal and becomes a mixture model wearing a latent's clothes.
        """
        k, kt = self.n_dist, self.n_dt
        if s is None:
            angle = self._angle_params(out)
            dmu, tmu = do[:, k:2 * k], to[:, kt:2 * kt]
        else:
            angle = self._angle_params(torch.cat([out[:, :1] + self.d_turn * s, out[:, 1:]], dim=1))
            dmu = do[:, k:2 * k] + self.d_dist * s
            tmu = to[:, kt:2 * kt] + self.d_dt * s
        dist = (torch.log_softmax(do[:, :k], dim=1), dmu,
                torch.nn.functional.softplus(do[:, 2 * k:3 * k]) + M.EPS)
        dtp = (torch.log_softmax(to[:, :kt], dim=1), tmu,
               torch.nn.functional.softplus(to[:, 2 * kt:3 * kt]) + M.EPS)
        return angle, dist, dtp

    def drift_level_logliks(self, x, a_z, ld_c, z, ya, yd, yt, nodes, weights=None):
        """(n, L) TOTAL per-bout log-likelihood at each drift level, over the three fitted channels.

        Unlike ``dt_level_logliks`` this sums turn + distance + interval, because the drift loads on
        all three -- that is the whole point of it over ``dt_latent``. Masked kinematic bouts
        contribute 0 to every level (not NaN, and not a real density), so the chain still transitions
        through them without the belief being updated by a bout the model was never fitted on.

        ``weights`` = ``(w_turn, w_dist, w_dt)`` per-bout tensors (n,) multiplying the three terms
        (tempered emissions, Model 15/16); None is bit-identical to unweighted.
        """
        out, do, to = self._trunk(x, a_z, ld_c, z)
        md = (torch.isfinite(yd) & (yd > 0)).float()
        mt = (torch.isfinite(yt) & (yt > 0)).float()
        if weights is not None:
            wa, wd, wt_ = weights
            md, mt = md * wd, mt * wt_
        ydc, ytc = torch.clamp(yd, min=1e-6), torch.clamp(yt, min=1e-6)
        lev = []
        for j in range(nodes.shape[0]):
            ang, dpar, tpar = self._params_at(out, do, to, nodes[j])
            la = MX.mix_logdens(ya, ang)
            if weights is not None:
                la = la * wa
            lev.append(la + md * MX._lnk_logpdf(ydc, *dpar)
                       + mt * MX._dt_logpdf(ytc, *tpar, hi=self.dt_trunc))
        return torch.stack(lev, dim=1)

    def two_chain_logliks(self, x, a_z, ld_c, z, ya, yd, yt, nodes_d, nodes_s):
        """TWO-STATE design (drift with ``dt_off`` + ``dt_latent``): per-bout log-likelihood of
        turn + step at each SLOW-chain level, (n, L_d), and of the interval at each FAST-chain
        level, (n, L_s), from ONE trunk pass. Masked kinematic bouts contribute 0 to every level.

        Given z the two chains see disjoint channels (the within-bout conditioners |dtheta| and
        log l are observed, so the interval head's dependence on them is not a dependence on the
        slow state), so the joint marginal is the SUM of two independent forward recursions --
        2 x L levels, not L^2. That is what makes the second state affordable in the scorer.
        """
        out, do, to = self._trunk(x, a_z, ld_c, z)
        md = (torch.isfinite(yd) & (yd > 0)).float()
        mt = (torch.isfinite(yt) & (yt > 0)).float()
        ydc = torch.clamp(yd, min=1e-6)
        lev = []
        for j in range(nodes_d.shape[0]):
            ang, dpar, _ = self._params_at(out, do, to, nodes_d[j])
            lev.append(MX.mix_logdens(ya, ang) + md * MX._lnk_logpdf(ydc, *dpar))
        tpar = self._params_at(out, do, to)[2]             # d_dt is 0: no slow-chain shift on dt
        lt = mt.unsqueeze(1) * self.dt_level_logliks(tpar, yt, nodes_s)
        return torch.stack(lev, dim=1), lt


# =========================================================================== #
#  Per-bout log-likelihood, shared by the trainer, the EM step and the scorer
# =========================================================================== #
def state_gru_input(y_ang, y_dist, y_dt, dth_scale, kin_scales):
    """(n, 3) the GRU's emission input: ``[dtheta/dth_scale, ld_c, (log dt - tm)/ts]``.

    ONE definition, used by both the teacher-forced chain and the torch rollout, so the state
    cannot mean different things at train and generation time.
    """
    dm, ds, tm, ts = kin_scales
    a = np.asarray(y_ang, float) / float(dth_scale)
    d = (np.log(np.clip(np.asarray(y_dist, float), 1e-6, None)) - dm) / ds
    t = (np.log(np.clip(np.asarray(y_dt, float), 1e-6, None)) - tm) / ts
    return np.nan_to_num(np.stack([a, d, t], axis=1)).astype(np.float32)


def _chan_logliks(net, Xt, az, lc, ya, yd, yt, z):
    """(la, ld, lt, md, mt): per-bout log-densities for the three channels at latent ``z``.

    ``md``/``mt`` are the finite-and-positive masks the training loss uses. They matter: 10 of the
    70590 bouts have distance exactly 0 and the last bout of every fish has no forward-looking
    distance or interval, and a log-normal at the 1e-6 clamp returns about -850 nats.
    """
    ang, dpar, tpar = net.joint_forward_re(Xt, az, lc, z)
    la = MX.mix_logdens(ya, ang)
    md = torch.isfinite(yd) & (yd > 0)
    mt = torch.isfinite(yt) & (yt > 0)
    ld = MX._lnk_logpdf(torch.clamp(yd, min=1e-6), *dpar)
    lt = MX._dt_logpdf(torch.clamp(yt, min=1e-6), *tpar, hi=getattr(net, "dt_trunc", None))
    return la, ld, lt, md, mt


# =========================================================================== #
#  Slow TIMING latent: an AR(1) state entering the interval head's location ONLY
# =========================================================================== #
# WHY THIS EXISTS, AND WHY THE PER-FISH LATENT CANNOT DO IT. `z_re` is drawn once per trajectory
# (fit_MLP_joint.py:363, before the bout loop at :470), so it is a CONSTANT offset on the three
# head levels. It shifts every window of a fish equally and therefore contributes exactly zero to
# within-fish, window-to-window variance -- which is what `rate_acf_1min` (gap -0.268),
# `quiesc20` (-0.112) and the dt dispersion (0.55 against a target of 1.0) all measure. That it has
# nearly closed `between_fish_share` (0.250 vs real 0.255) while leaving every timing statistic
# untouched is the same fact seen from the other side: it is individuality, not state.
#
# WHY NOT JUST MORE LAGS. K_kin 3 -> 8 is the cheap alternative and it is a TRADE, measured paired
# per fish across jkin3/jk8both: dt -0.0218 bits (p 0.005, 13/15 fish) but turn +0.0169 bits
# (p 0.0001, 0/15 fish). In the joint family the kin columns feed the SHARED body, so extra lags
# reach the angle head. A level-only loading on mu_tau touches the timing head and nothing else,
# which is the whole architectural argument for paying for this.
#
# LEVEL-ONLY IS NOT A STYLE CHOICE. smoke_state_layer.py measured that a chain with a free
# per-state emission spends its states on the MARGINAL instead -- it learns a "long pause" mode
# and a "short bout" mode, dwell collapses to a couple of bouts, and what looks like a latent is
# really a mixture model. Constrain it to shift the level and nothing else.
PHI_LO, PHI_HI = 0.50, 0.999
DT_LATENT_DEFAULT = {"n_levels": 9, "chunk": 256, "phi_init": 0.97, "span": 3.0, "lam_init": 0.15}

# =========================================================================== #
#  DRIFT: the same AR(1) chain, but loading on EVERY channel instead of only dt
# =========================================================================== #
# WHY THIS EXISTS, AND WHY SIX PREVIOUS MECHANISMS COULD NOT DO IT.
#
# Measured on 15 fish at W = 60 bouts, comparing REAL against the rej rollout. `acf1` is the lag-1
# autocorrelation of the series of WINDOW MEANS, computed WITHIN a fish -- so any static
# per-trajectory latent is removed by the demeaning and contributes exactly zero by construction:
#
#     channel     REAL acf1     GEN acf1        REAL excess   GEN excess
#     dt            +0.332       -0.062             4.8          3.4
#     distance      +0.337       -0.022             3.0          2.3
#     radial        +0.377       -0.035            21.6         16.0
#     |dtheta|      +0.128       +0.064             2.0          1.8
#
# (SE ~0.05, so the real values are ~7 sigma from zero and the generated ones are indistinguishable
# from it.) The model carries 72-77% of the real EXCESS window variance but essentially NONE of the
# across-window correlation: its excess comes from K=8 turn lags and K_kin=3 kinematic lags, which
# inflate variance inside a window and are dead long before the next one starts.
#
# `fish_re` z, `rad_indiv`, `rad_gate` and `re_bgain` are ALL static per trajectory. A static latent
# shifts every window of a fish by the same amount, so none of them can move `acf1` at all -- which
# is why six mechanisms in a row hit their own target and left the C2ST where it was. Only
# `dt_latent` is dynamic, and it loads on the interval head alone, never on the spatial channel that
# the classifier actually keys on.
#
# TAU IS PINNED, NOT FITTED -- and it is pinned from a ROLLOUT sweep, not from the pure-AR algebra.
# A bare AR(1) reaches adjacent-window-mean correlation 0.31 at tau = 20 and 0.50 at tau = 40, which
# would suggest tau ~ 22. That is WRONG here, because the drift's contribution is diluted by every
# other source of window variance in the closed loop. Measured end-to-end on the rej rollout, all
# loadings fixed at 0.2 (8 fish x 3 replicates, SE ~0.02):
#
#     tau     dt      dist    radial            (targets: +0.332  +0.337  +0.377)
#      22   +0.207   +0.041   +0.105
#      40   +0.364   +0.145   +0.183
#      70   +0.493   +0.217   +0.226
#     120   +0.609   +0.322   +0.345
#
# At a FIXED loading the three channels appear to want different timescales, and that is real: the
# measured lag-profile decays fast for dt (0.332, 0.103, 0.065, 0.045) and slowly for distance and
# radial (0.337, 0.185, 0.149, 0.118). But the AMPLITUDES are fitted per channel, so one state at a
# longer tau can still land all three -- the likelihood is free to choose a smaller d_dt and a
# larger d_dist. tau = 70 keeps every target reachable with moderate loadings.
#
# WHY NOT TWO STATES, one fast for dt and one slow for distance/radial. It is the better model and
# the likelihood would even factorise (turn/dist depend on one chain, dt on the other, and both
# conditioners are observed) -- but `score` marginalises the PRODUCT grid with the static z, which
# would go from Q*L = 49*9 = 441 states per bout to 49*81 = 3969, inside a 4700-iteration Python
# recursion per fish. Metric A is a headline number and must stay exact, so this is one state.
#
# Fitting tau freely is what went wrong in round 2a: the likelihood settled dt_latent at tau =
# 107-152 while the real DT timescale is short, so the state stopped being a within-trajectory
# phenomenon and started impersonating the per-fish latent -- inflating every head and pushing
# between_fish_share to 0.304 against a real 0.255. Pinning removes that degeneracy by construction.
#
# THE SPATIAL CHANNEL IS REFUTED, and `bord` therefore defaults to 0. It was built (an additive
# wandering term on centre_bias, so the fish's preferred radius moves over minutes) and MEASURED on
# the rej checkpoint, 8 fish x 2 replicates. It does not work, by two independent routes:
#
#   bulk  bord   radial acf1   occ>0.85   r_mean        (real: +0.371, 0.0082, 0.436)
#      0     0      +0.005       0.0137    0.465
#      0     4      +0.116       0.0188    0.469        acf partly moves, occupancy 2.3x real
#      4     2      +0.042       0.0080    0.422        bulk floor kills the acf gain
#      8     4      +0.039       0.0058    0.385        ... and drags r_mean off
#     16     8      +0.117       0.0027    0.334
#
# Without a `bulk` floor the additive term pushes OUTWARD half the time, and a sustained outward
# push at tau = 70 beats the rejection ramp (acceptance is only suppressed to ~46% at r/R = 0.85,
# not blocked) -- an end-to-end fit at bord = 22 reached radial acf +0.341 and occ 0.163, i.e. 20x
# real, with the boundary search pinned at its grid edge. With a floor, containment dominates and
# the acf gain disappears while r_mean drifts off. The turn loading (a bounded route: a straighter
# fish covers more dish) does not work either -- at turn = 0.076 -> 1.6 the forward fraction moves
# only 0.173 -> 0.177 because the softmax saturates, and radial stays 0.06-0.15 with no trend.
#
# So this arm closes the dt and distance gaps and leaves radial open. That is worth saying plainly:
# radial is the top C2ST discriminator, so the headline AUC may not move much even though two of the
# three 6-sigma defects are fixed.
DRIFT_TAU_BOUTS = 70.0
DRIFT_DEFAULT = {"n_levels": 9, "chunk": 256, "span": 3.0, "lam_init": 0.15,
                 "tau": DRIFT_TAU_BOUTS, "bord": 0.0,
                 "g_dist": 1.0, "g_dt": 1.0, "dist_c": 0.0, "dt_c": 0.0}


# =========================================================================== #
#  SCHEDULED SAMPLING on the lag inputs (Bengio et al. 2015)
# =========================================================================== #
# WHY. Real steps and intervals persist bout to bout for three reasons -- a fish-level tempo, a slow
# drift over minutes, and genuine bout-to-bout dependence -- and a teacher-forced fit has only the
# third, so its lag coefficients absorb the other two (omitted-variable bias). In the free run the
# inflated coefficients show up as too-high WITHIN-window acf (dist_acf1 0.263 vs real 0.198 before
# the per-fish latent, dt_acf1 0.144 vs 0.073) and too-wide interval spread. The drift above puts
# the missing slow state INTO the likelihood; scheduled sampling instead trains the conditional on
# the inputs it will actually see in a rollout: with probability 1 - p the lagged (dtheta, distance,
# dt) a bout is conditioned on are the MODEL'S OWN one-step draws for those earlier bouts rather than
# the recorded values, p decaying linearly from 1 (teacher forcing) to `p_min` over `decay` epochs
# after `warmup` epochs of plain fitting. Targets stay real; stimulus, border, lag-validity and
# clock inputs stay teacher-forced (the pose is not replayed); the VALIDATION loss stays clean --
# so early stopping and Metric A are the same teacher-forced density as every other arm, but the
# fitted objective is no longer an exact likelihood.
# ONE-STEP form: the draw for bout t-k is made from bout t-k's real context, not from a free run
# (a 60k-step sequential pass per epoch); the standard approximation for a fixed-window
# autoregressive emission, and the same construction as MLPBorderLoopDensity's scheduled mode,
# extended to all three channels and to the random-effect net (each fish's fitted z is used).
SS_DEFAULT = {"p_min": 0.5, "warmup": 5, "decay": 40}


def ss_teacher_p(ep, ss):
    """Teacher-forcing probability at epoch ``ep``: 1 through the warmup, then linear to p_min."""
    w, d, pm = int(ss["warmup"]), max(int(ss["decay"]), 1), float(ss["p_min"])
    if ep < w:
        return 1.0
    return float(max(pm, 1.0 - (ep - w + 1) / d * (1.0 - pm)))


@torch.no_grad()
def ss_draw(net, X, a_z, ld_c, z, dth_scale, kin_scales, kin_clip):
    """One draw of (dtheta, distance, dt) per row from the CURRENT net through the adt chain --
    dtheta from x alone, distance given the DRAWN |dtheta|, dt given both -- exactly the
    generator's order, so the lag a later bout sees is a value the model would have emitted. The
    drift / state shift is off (``s=None``): this is the fast lag feedback; the slow state is the
    drift's job. Draws use the global torch RNG (seeded by the trainer) through rollout_torch's
    samplers, which clamp at the same ``kin_clip`` the generator uses."""
    from twin import rollout_torch as RT
    dm, ds, _tm, _ts = kin_scales
    ang, _d, _t = net.joint_forward_re(X, a_z, ld_c, z)
    ya = RT.sample_angle(ang, dth_scale, 1.0, reparam=False)
    az = torch.abs(ya) / float(dth_scale)
    _a, dpar, _t = net.joint_forward_re(X, az, ld_c, z)
    yd = RT.sample_lognorm_mix(dpar, kin_clip[0], 1.0, reparam=False)
    lc = (torch.log(yd) - dm) / ds
    _a, _d, tpar = net.joint_forward_re(X, az, lc, z)
    yt = RT.sample_lognorm_mix(tpar, kin_clip[1], 1.0, reparam=False,
                               trunc=getattr(net, "dt_trunc", None))
    return ya, yd, yt


def drift_phi(tau):
    """Per-bout AR(1) coefficient for a correlation time of ``tau`` bouts."""
    return float(np.exp(-1.0 / max(float(tau), 1e-6)))




def ar1_chain(phi, n_levels, span=None):
    """``(nodes (L,), logP (L,L), log_pi0 (L,))`` for ``s_t = phi s_{t-1} + sqrt(1-phi^2) eps``.

    ROUWENHORST, not Tauchen, and the difference is load-bearing here. Tauchen bins the normal
    kernel onto a fixed +-3 sd grid, which is fine at low persistence and degrades badly at high:
    measured stationary variance on a 9-node grid was 1.06 at phi=0.5, **1.22 at 0.9 and 1.49 at
    0.97**, against a target of 1. Two things break at that point. (a) ``lam`` stops being the only
    amplitude parameter -- the effective sd becomes lam*sqrt(var(phi)), so lam and phi trade off
    and the timescale is no longer identified. (b) The ROLLOUT integrates the exact continuous
    AR(1), whose stationary variance really is 1, so training and generation would be running
    different processes -- this project's recurring two-objects trap, in a new place.

    Rouwenhorst matches mean 0, stationary variance EXACTLY 1 and lag-1 autocorrelation EXACTLY
    phi, for any L >= 2 and any phi, which is precisely why it is the standard choice for
    persistent chains. Its stationary law is Binomial(L-1, 1/2) in closed form, so the prior needs
    no eigen-solve either.

    ``phi`` is a TENSOR and P is differentiable in it (the recursion is polynomial in p=(1+phi)/2);
    the nodes and the prior are constants. ``span`` is accepted and ignored -- Rouwenhorst fixes
    the grid width at sqrt(L-1), which is what makes the variance exact.
    """
    L = int(n_levels)
    dev = phi.device
    if L < 2:
        z = torch.zeros(1, dtype=torch.float32, device=dev)
        return z, torch.zeros(1, 1, device=dev), torch.zeros(1, device=dev)
    p = 0.5 * (1.0 + phi)
    P = torch.stack([torch.stack([p, 1 - p]), torch.stack([1 - p, p])])
    for n in range(3, L + 1):
        z_row = torch.zeros(n - 1, 1, dtype=P.dtype, device=dev)
        z_col = torch.zeros(1, n, dtype=P.dtype, device=dev)
        A = torch.cat([torch.cat([P, z_row], dim=1), z_col], dim=0)
        B = torch.cat([torch.cat([z_row, P], dim=1), z_col], dim=0)
        C = torch.cat([z_col, torch.cat([P, z_row], dim=1)], dim=0)
        D = torch.cat([z_col, torch.cat([z_row, P], dim=1)], dim=0)
        P = p * A + (1 - p) * B + (1 - p) * C + p * D
        # interior rows receive mass from two parents and must be halved (Rouwenhorst's step)
        P = torch.cat([P[:1], P[1:-1] * 0.5, P[-1:]], dim=0)
    psi = float(np.sqrt(L - 1))                      # grid half-width that makes var == 1 exactly
    nodes = torch.linspace(-psi, psi, L, dtype=torch.float32, device=dev)
    # math.comb, NOT np.math.comb: the np.math alias was removed in numpy 2, which is what Colab
    # runs -- this would have imported fine locally and crashed on the remote.
    logp0 = (np.log(np.array([float(math.comb(L - 1, i)) for i in range(L)]))
             - (L - 1) * np.log(2.0))                # Binomial(L-1, 1/2), in closed form
    p0 = torch.tensor(logp0, dtype=P.dtype, device=dev)
    return nodes, torch.log(torch.clamp(P, min=1e-30)), p0


def forward_recursion(L_lev, logP, log_pi0):
    """Exact ``log p(y_1..T)`` marginalising the latent PATH. ``L_lev`` is (B, T, L).

    ``log a_t(j) = L_t(j) + logsumexp_i [ log a_{t-1}(i) + log P(i->j) ]``, batched over B so the
    Python loop runs T times per optimiser step rather than once per bout. Returns (B,).

    This is the piece the static latent does not need: `_marginal_predictive` gets away with a
    cumsum precisely because z never moves, i.e. its transition matrix is the identity. Everything
    else about the two is the same recursion.
    """
    loga = log_pi0.reshape(1, -1) + L_lev[:, 0, :]
    for t in range(1, L_lev.shape[1]):
        loga = L_lev[:, t, :] + torch.logsumexp(loga.unsqueeze(2) + logP.unsqueeze(0), dim=1)
    return torch.logsumexp(loga, dim=1)


# =========================================================================== #
#  Gauss-Hermite quadrature over a diagonal Gaussian prior
# =========================================================================== #
def _ghq(m, sd, k=7):
    """(nodes (Q,m), log-weights (Q,)) for E_{z~N(0,diag(sd^2))}[.] on a tensor product grid.

    ``hermegauss`` is the PROBABILISTS' Hermite rule, i.e. weight exp(-x^2/2), so the nodes scale by
    sd directly and the weights normalise to 1 with no sqrt(2) fiddling. Q = k^m; at m=2, k=7 that
    is 49 nodes, exact for polynomials up to degree 13 per dimension -- far beyond what the
    log-likelihood surface of a single fish needs. Keep m <= 3: at m >= 5 a tensor grid is no longer
    affordable and importance sampling over ~4700 bouts has degenerate weights, at which point the
    "exactly scored" claim that motivates the whole mechanism stops being true.
    """
    m = int(m)
    sd = np.asarray(sd, float).reshape(-1)
    if not np.any(sd > 0):
        # Degenerate prior (the Sigma-pinned control): the marginal IS the z = 0 plug-in. One node.
        return np.zeros((1, m), dtype=np.float32), np.zeros(1)
    x, w = np.polynomial.hermite_e.hermegauss(k)
    logw = np.log(w / w.sum())
    grids = np.meshgrid(*([x] * m), indexing="ij")
    nodes = np.stack([g.reshape(-1) for g in grids], axis=1) * sd.reshape(1, -1)
    lgrids = np.meshgrid(*([logw] * m), indexing="ij")
    lw = np.sum([g.reshape(-1) for g in lgrids], axis=0)
    return nodes.astype(np.float32), lw




def _marginal_predictive(L, log_pi):
    """Per-bout marginal predictive log-density from per-node log-likelihoods.

    ``L`` is (n, Q) TOTAL per-bout log-likelihood, ``log_pi`` the (Q,) prior log-weights. Because z
    is static, the posterior after t-1 bouts is just the prior times the running product, so the
    filter is a cumsum rather than a recursion:

        log w_q(t) = log pi_q + sum_{s<t} L[s,q]
        log c_t    = logsumexp_q(log w_q(t) + L[t,q]) - logsumexp_q(log w_q(t))

    and sum_t log c_t telescopes to log p(y_{1:n}) exactly. Returns (logdens (n,), logZ float).
    """
    L = np.asarray(L, float)
    n = L.shape[0]
    cum = np.zeros_like(L)
    if n > 1:
        cum[1:] = np.cumsum(L[:-1], axis=0)
    logw = log_pi.reshape(1, -1) + cum                      # (n, Q), un-normalised log posterior
    from scipy.special import logsumexp
    denom = logsumexp(logw, axis=1)
    numer = logsumexp(logw + L, axis=1)
    return numer - denom, float(logsumexp(log_pi + L.sum(axis=0)))


# =========================================================================== #
#  Training
# =========================================================================== #
def fit_mix_joint_re(X, y_ang, y_dist, y_dt, fish_idx, n_fish, hidden, dth_scale, kin_scales, *,
                     m=2, epochs=80, lr=1e-3, batch_size=256, patience=8, val_frac=0.15, seed=0,
                     device=None, n_dist=2, n_dt=3, weight_decay=1e-5, sigma_init=0.5,
                     freeze_sigma_zero=False, shuffle_labels=False, use_body=False,
                     border_gain=False, border_slice=None, dt_latent=None, block=False, drift=None,
                     state=0, seq_batch=1, state_chunk=64, h0_dim=0,
                     noise_builder=None, noise_mode="concat", noise_frac=0.5,
                     ss=None, ss_builder=None, kin_clip=(400.0, 600.0), weights=None,
                     dt_trunc=None, disp=False, lag_slice=None, tail=False, weight_heads=None,
                     weights_per_head=None,
                     skip_slices=None, depth=1, tsize=False):
    """Stage A. Joint MAP fit of the net and the per-fish latents under a FIXED N(0, I) prior.

    ``weight_heads`` (Model 16): the heads ``weights`` applies to, e.g. ``("turn",)``; the other
    heads keep weight 1. None = all three (Model 15's form). Each head's weights are normalised
    by their mean over the training rows, so the emphasis is WITHIN a channel (the band against
    the rest) and the channels' relative scale is untouched.

    ``weights`` = per-row training weights (MB.regime_weights): a weighted MEAN per channel, the
    same form as JT.fit_mix_joint, so the loss keeps its scale and ``weights=None`` is bit-identical.
    Early stopping is on the WEIGHTED validation loss (the objective being fitted); Metric A is
    scored unweighted afterwards. Not implemented on the sequence paths (drift / dt_latent / state).

    ``ss`` (dict, see SS_DEFAULT) turns on SCHEDULED SAMPLING: from epoch ``warmup`` on, a
    Bernoulli(1 - p) subset of the TRAINING rows has its recorded (dtheta, distance, dt) replaced
    by the current net's one-step draws (``ss_draw``) before the lag blocks are rebuilt by
    ``ss_builder(dtheta, distance, dt) -> X`` -- so a bout's history carries what the model would
    have emitted, not what the fish did. Targets and the validation split stay clean.

    The prior scale is fixed and the between-fish magnitude is carried by the loadings, because only
    the product (loading x latent) is identified -- see the module docstring for the measured
    divergence that results from letting both float and then re-estimating Sigma.

    The penalty ``sum_i ||z_i||^2 / 2`` is divided by the TOTAL number of training rows and added to
    EVERY minibatch. Both halves matter: dividing by rows (not by fish, not by batch size) puts the
    prior on the same per-row scale as the mean log-likelihood, and adding it in every batch rather
    than once per epoch keeps it from being weaker than intended by a factor of the batch count.

    VALIDATION SPLIT. Contiguous per-fish blocks, not random rows. A random row split puts every
    fish in both halves, so a per-fish parameter can improve the validation loss purely by memorising
    identity and early stopping runs too long. Holding out the LAST ``val_frac`` of each fish keeps
    every fish represented (so z_i is still identified) while making the held-out rows temporally
    disjoint from the ones that fitted it.

    TWO CONTROL ARMS, and they answer different questions.
      ``freeze_sigma_zero`` pins Sigma at 0. Because the loadings are zero-initialised and receive no
        gradient at z = 0, the whole mechanism stays inert -- so this arm is the TRAINING-LOOP
        control: same trainer, same contiguous validation split, no random effect. It is what stops
        a win being credited to the mechanism when it belongs to the reshuffle fix or the new split.
      ``shuffle_labels`` keeps the mechanism fully live but assigns each ROW a random fish index, so
        the embedding has identical capacity and identical gradient flow while fish identity carries
        no information. It is the CAPACITY control: it answers "did it use the between-fish
        structure, or just the extra parameters?", which pinning Sigma cannot, because a pinned arm
        has no free parameters to overfit with.
    """
    M._seed_everything(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a_z, ld_c = JT._cond_cols(y_ang, y_dist, dth_scale, kin_scales)
    Xt = torch.tensor(np.asarray(X, np.float32), device=dev)
    ya = torch.tensor(np.asarray(y_ang, np.float32), device=dev)
    yd = torch.tensor(np.asarray(y_dist, np.float32), device=dev)
    yt = torch.tensor(np.asarray(y_dt, np.float32), device=dev)
    az = torch.tensor(a_z, device=dev); lc = torch.tensor(ld_c, device=dev)
    fish_idx = np.asarray(fish_idx, np.int64)
    # The validation split below is always built from the TRUE fish blocks, so the permutation
    # control changes only what the embedding is allowed to learn, never how the model is selected.
    lbl = fish_idx
    if shuffle_labels:
        lbl = np.random.default_rng(seed).permutation(fish_idx)
    fi = torch.tensor(lbl, device=dev)

    if (dt_latent or drift) and noise_builder is not None:
        raise ValueError("a latent state + noise_aug is an untested combination; noise augmentation "
                         "was resolved WORSE twice in round 1 (+0.111 / +0.137 c2st_traj).")
    if ss and ss_builder is None:
        raise ValueError("scheduled sampling needs ss_builder (MLPMixJointREDensity supplies it)")
    if ss and (state or noise_builder is not None):
        raise ValueError("scheduled sampling with the GRU state or noise_aug is untested; one "
                         "input-side treatment at a time")
    wt = None if weights is None else torch.tensor(np.asarray(weights, np.float32), device=dev)
    # PER-HEAD weights (Model 16): the row loss normalises each channel by its own weight sum, the
    # drift recursion cannot (it needs the per-bout SUM over channels), so there each head's
    # weights are normalised by their mean and the block by its row count -- identical to the
    # single-weight form for all-heads weights, and bit-identical to unweighted for all-ones.
    _heads = set(weight_heads or ("turn", "dist", "dt"))
    if weights_per_head is not None:
        # explicit per-head vectors (MB.regime_weights with a second regime, Model 25)
        w_a, w_d, w_t = (torch.tensor(np.asarray(weights_per_head[h], np.float32), device=dev)
                         for h in ("turn", "dist", "dt"))
        wt = w_a                                  # non-None: the weighted paths below switch on
        w_seq = tuple(v / v.mean() for v in (w_a, w_d, w_t))
    elif wt is not None:
        _ones = torch.ones_like(wt)
        w_a = wt if "turn" in _heads else _ones
        w_d = wt if "dist" in _heads else _ones
        w_t = wt if "dt" in _heads else _ones
        w_seq = tuple(v / v.mean() for v in (w_a, w_d, w_t))
    if wt is not None and (dt_latent or state):
        raise NotImplementedError("regime weights are per-bout; the dt_latent / state sequence "
                                  "losses marginalise over blocks and are unweighted (the drift "
                                  "loss supports them: see nll_seq_drift)")
    if wt is not None and noise_builder is not None:
        raise ValueError("regime weights + noise_aug: the noisy copy has no weights; not combined")
    net = MDNMixJointRE(Xt.shape[1], hidden, n_dist=n_dist, n_dt=n_dt,
                        n_fish=int(n_fish), m=int(m), sigma_init=sigma_init,
                        use_body=use_body, border_gain=border_gain,
                        border_slice=border_slice, dt_latent=dt_latent,
                        block=block, drift=drift, state=state, h0_dim=h0_dim,
                        disp=disp, lag_slice=lag_slice, tail=tail, skip_slices=skip_slices,
                        depth=depth, tsize=tsize).to(dev)
    net.dt_trunc = (float(dt_trunc) if dt_trunc else None)
    # No weight decay on the embedding: it already has an explicit N(0, I) prior in the loss, and
    # decaying it as well would apply that prior twice at an unstated strength.
    emb_params = list(net.emb.parameters())
    if os.environ.get("FISHRE_NO_WD_LOADINGS") == "1":
        # TEST SWITCH (off by default, so every archived fit is reproduced bit-for-bit). The same
        # argument as above applies to the LOADINGS: the random effect is c.z, the N(0, I) prior in
        # the loss is on z, so decaying c shrinks the individuality AMPLITUDE under a prior nobody
        # stated. Motivated by popd fold 2, whose z_dist fell from an init norm of ~0.2 to 0.0029
        # while its z_turn / z_dt stayed healthy -- a direction the likelihood barely constrains,
        # where Adam turns even a 1e-5 decay into a full-size step toward zero.
        emb_params += [p for n_, p in net.named_parameters() if n_.startswith("z_")]
    emb_ids = {id(p) for p in emb_params}
    opt = torch.optim.Adam(
        [dict(params=[p for p in net.parameters() if id(p) not in emb_ids],
              weight_decay=weight_decay),
         dict(params=emb_params, weight_decay=0.0)], lr=lr)

    n = Xt.shape[0]
    zero_z = torch.zeros((1, int(m)), device=dev)

    def _z_for(ix):
        return zero_z.expand(ix.shape[0], -1) if freeze_sigma_zero else net.emb(fi[ix])

    def _nll_arrays(ix, Xb, azb, lcb, yab, ydb, ytb):
        z = _z_for(ix)
        la, ld, lt, md, mt = _chan_logliks(net, Xb, azb, lcb, yab, ydb, ytb, z)
        zt = torch.zeros((), device=dev)
        if wt is None:
            ll = (la.mean() + (ld[md].mean() if md.any() else zt)
                  + (lt[mt].mean() if mt.any() else zt))
        else:
            wa, wd, wt_ = w_a[ix], w_d[ix], w_t[ix]
            ll = ((wa * la).sum() / wa.sum()
                  + ((wd[md] * ld[md]).sum() / wd[md].sum() if md.any() else zt)
                  + ((wt_[mt] * lt[mt]).sum() / wt_[mt].sum() if mt.any() else zt))
        if freeze_sigma_zero:
            return -ll
        return -ll + (0.5 * net.emb.weight ** 2).sum() / float(n)

    def nll(ix):
        return _nll_arrays(ix, Xt[ix], az[ix], lc[ix], ya[ix], yd[ix], yt[ix])

    def nll_seq(blocks):
        """Loss on B CONTIGUOUS blocks of T bouts, with the timing latent marginalised exactly.

        ``blocks`` is (B, T) row indices, each row ordered and within one fish. Turn and distance
        are per-bout as before; the interval channel goes through the forward recursion, which is
        the only reason the batch has to be ordered at all. Dividing the sequence log-likelihood by
        the number of VALID interval bouts keeps it on the same per-bout scale as the other two
        channels, so the three-way weighting of the loss is unchanged from the base model."""
        B, T = blocks.shape
        ix = blocks.reshape(-1)
        z = _z_for(ix)
        ang, dpar, tpar = net.joint_forward_re(Xt[ix], az[ix], lc[ix], z)
        la = MX.mix_logdens(ya[ix], ang)
        ydb, ytb = yd[ix], yt[ix]
        md = torch.isfinite(ydb) & (ydb > 0)
        mt = torch.isfinite(ytb) & (ytb > 0)
        ld = MX._lnk_logpdf(torch.clamp(ydb, min=1e-6), *dpar)
        nodes, logP, log_pi0 = ar1_chain(net.s_phi(), int(net.dt_latent["n_levels"]),
                                       span=float(net.dt_latent["span"]))
        lt_lev = net.dt_level_logliks(tpar, ytb, nodes)                       # (B*T, L)
        # A masked bout must contribute log-weight 0 to EVERY level -- not NaN, and not a real
        # density -- so the chain still transitions through it without the belief being updated by
        # a bout the model was never fitted on.
        lt_lev = torch.where(mt.unsqueeze(1), lt_lev, torch.zeros_like(lt_lev))
        ll_dt = forward_recursion(lt_lev.reshape(B, T, -1), logP, log_pi0).sum()
        zt = torch.zeros((), device=dev)
        ll = (la.mean() + (ld[md].mean() if md.any() else zt)
              + ll_dt / torch.clamp(mt.sum(), min=1))
        if freeze_sigma_zero:
            return -ll
        return -ll + (0.5 * net.emb.weight ** 2).sum() / float(n)

    def nll_seq_drift(blocks):
        """Same contiguous-block loss, but the DRIFT path is marginalised across ALL THREE channels.

        That is the whole difference from ``nll_seq``: ``dt_latent`` shifts only the interval
        location, so turn and distance stay per-bout; the drift shifts all three, so the forward
        recursion has to run on their SUM. ``drift_level_logliks`` already folds the kinematic masks
        in (a masked bout contributes 0 to every level), so the chain transitions through it without
        updating the belief.

        Normalisation differs slightly from the base loss by construction: this is the true joint
        per-bout log-likelihood, whereas the base sums three separately-averaged channels. With 10
        invalid distance bouts in 70590 the two agree to well under a millinat, and the joint form
        is the one the recursion actually defines.

        PER-BOUT WEIGHTS (Model 15: MB.regime_weights on the drift arms). Applied as TEMPERED
        EMISSIONS -- bout t's per-level log-likelihood is multiplied by w_t before the recursion,
        which for a fixed state path is exactly the weighted per-bout NLL the row loss uses, and
        the marginal over paths is its natural sequence form; the sum is divided by the total
        weight so the scale stays per-bout. All-ones weights reproduce the unweighted loss exactly.
        """
        B, T = blocks.shape
        ix = blocks.reshape(-1)
        z = _z_for(ix)
        nodes, logP, log_pi0 = ar1_chain(net.d_phi(), int(net.drift["n_levels"]))
        lev = net.drift_level_logliks(Xt[ix], az[ix], lc[ix], z, ya[ix], yd[ix], yt[ix], nodes,
                                      weights=(None if wt is None else tuple(v[ix] for v in w_seq)))
        ll = forward_recursion(lev.reshape(B, T, -1), logP, log_pi0).sum() / float(len(ix))
        if freeze_sigma_zero:
            return -ll
        return -ll + (0.5 * net.emb.weight ** 2).sum() / float(n)

    def nll_seq_two(blocks):
        """TWO-STATE design: the slow chain (turn + step) and the fast interval chain are
        marginalised by two independent forward recursions on the same contiguous blocks (see
        ``two_chain_logliks``). Joint per-bout normalisation, as ``nll_seq_drift``."""
        B, T = blocks.shape
        ix = blocks.reshape(-1)
        z = _z_for(ix)
        nd, logP_d, pi0_d = ar1_chain(net.d_phi(), int(net.drift["n_levels"]))
        ns, logP_s, pi0_s = ar1_chain(net.s_phi(), int(net.dt_latent["n_levels"]),
                                      span=float(net.dt_latent["span"]))
        lev_td, lev_t = net.two_chain_logliks(Xt[ix], az[ix], lc[ix], z,
                                              ya[ix], yd[ix], yt[ix], nd, ns)
        ll = (forward_recursion(lev_td.reshape(B, T, -1), logP_d, pi0_d).sum()
              + forward_recursion(lev_t.reshape(B, T, -1), logP_s, pi0_s).sum()) / float(len(ix))
        if freeze_sigma_zero:
            return -ll
        return -ll + (0.5 * net.emb.weight ** 2).sum() / float(n)

    ugru = (torch.tensor(state_gru_input(y_ang, y_dist, y_dt, dth_scale, kin_scales), device=dev)
            if state else None)

    def nll_seq_state(blocks):
        """Teacher-forced sequence loss for the RECURRENT model, on B contiguous blocks.

        No marginalisation of any kind: with the targets observed the GRU is deterministic, so this
        IS the exact sequence log-likelihood. h_0 = 0 for every block -- the blocks are random
        windows, so carrying a state across them would attach it to sampling order rather than to
        the fish. The cost is that the first bouts of a block are scored from a cold state; with
        chunk >> the state's useful memory that is a small, constant handicap applied equally to
        every arm.

        The fish_re latent ``z``, when it is on, is the OPPOSITE of that: static per fish, so it is
        the one thing that legitimately survives across blocks and it is read from the embedding by
        the block's fish. See ``state_chain`` for why it and the GRU are not redundant.
        """
        ix = blocks.reshape(-1)
        B, T = blocks.shape
        # every row of a block comes from ONE fish (that is what makes it a sequence), so the
        # block's fish is just the fish of its first row.
        h0 = (net.h0_from(net.h0_emb(fi[blocks[:, 0]])) if net.h0_dim else None)
        # None, NOT _z_for's zero vector: the pinned arm must stay bit-identical (see state_chain).
        zb = None if freeze_sigma_zero else net.emb(fi[blocks[:, 0]])
        la, ld, lt, _ = net.state_chain(
            Xt[ix].reshape(B, T, -1), az[ix].reshape(B, T), lc[ix].reshape(B, T),
            ugru[ix].reshape(B, T, 3), ya[ix].reshape(B, T),
            yd[ix].reshape(B, T), yt[ix].reshape(B, T), h0=h0, z=zb)
        ydb, ytb = yd[ix].reshape(B, T), yt[ix].reshape(B, T)
        md = torch.isfinite(ydb) & (ydb > 0)
        mt = torch.isfinite(ytb) & (ytb > 0)
        zt = torch.zeros((), device=dev)
        ll = (la.mean() + (ld[md].mean() if md.any() else zt)
              + (lt[mt].mean() if mt.any() else zt))
        # Explicit N(0, I) prior on each latent that is on: the latent is generated from that prior
        # at rollout time, so it has to be fitted under it or the trained values and the sampled
        # ones describe different fish. Both terms are the same expression the non-recurrent losses
        # use, so the penalty is on the same per-row scale as `_nll_arrays` and `nll_seq_drift`.
        pen = torch.zeros((), device=dev)
        if net.h0_dim:
            pen = pen + (0.5 * net.h0_emb.weight ** 2).sum() / float(n)
        if not freeze_sigma_zero:
            pen = pen + (0.5 * net.emb.weight ** 2).sum() / float(n)
        return -ll + pen

    def nll_seq_state_drift(blocks):
        """SRNN Stage 1: the recurrent chain with the slow AR(1) marginalised EXACTLY.

        Three variance components, one mechanism each, all fitted under this one loss:
          * ``z``   static per fish        -> BETWEEN-fish   (embedding, prior N(0, I))
          * ``s_t`` pinned AR(1), tau > W  -> WITHIN-fish slow, marginalised by forward recursion
          * ``h_t`` GRU                    -> fast context, deterministic given y

        The recursion runs on the SUM of the three channels (``state_chain_levels``), because the
        drift loads on all three -- so unlike a per-channel latent it can reproduce the real
        cross-channel coupling. ``h_t`` does not depend on ``s``, so the GRU pass is computed once
        and the L levels cost only the heads.

        Normalisation matches ``nll_seq_drift`` rather than ``nll_seq_state``: this is the true joint
        per-bout log-likelihood, where the plain state loss sums three separately-averaged channels.
        The two agree to well under a millinat (10 invalid distance bouts in 70590), but the joint
        form is the one the recursion defines, and mixing them would make the two arms' bits
        incomparable for a reason that has nothing to do with the model.
        """
        ix = blocks.reshape(-1)
        B, T = blocks.shape
        h0 = (net.h0_from(net.h0_emb(fi[blocks[:, 0]])) if net.h0_dim else None)
        zb = None if freeze_sigma_zero else net.emb(fi[blocks[:, 0]])
        nodes, logP, log_pi0 = ar1_chain(net.d_phi(), int(net.drift["n_levels"]))
        lev = net.state_chain_levels(
            Xt[ix].reshape(B, T, -1), az[ix].reshape(B, T), lc[ix].reshape(B, T),
            ugru[ix].reshape(B, T, 3), ya[ix].reshape(B, T),
            yd[ix].reshape(B, T), yt[ix].reshape(B, T), nodes, h0=h0, z=zb)     # (B, T, L), summed
        ll = forward_recursion(lev, logP, log_pi0).sum() / float(len(ix))
        pen = torch.zeros((), device=dev)
        if net.h0_dim:
            pen = pen + (0.5 * net.h0_emb.weight ** 2).sum() / float(n)
        if not freeze_sigma_zero:
            pen = pen + (0.5 * net.emb.weight ** 2).sum() / float(n)
        return -ll + pen

    seq_loss = ((nll_seq_state_drift if drift else nll_seq_state) if state
                else (nll_seq_two if (drift and dt_latent) else (nll_seq_drift if drift else nll_seq)))
    seq_on = bool(dt_latent or drift or state)

    # Contiguous per-fish validation blocks (see docstring).
    idx_all = np.arange(n)
    tr_list, va_list = [], []
    for f in range(int(n_fish)):
        rows = idx_all[np.asarray(fish_idx) == f]
        cut = max(int(len(rows) * (1.0 - val_frac)), 1)
        tr_list.append(rows[:cut]); va_list.append(rows[cut:])
    tr = torch.tensor(np.concatenate(tr_list), dtype=torch.long, device=dev)
    va = torch.tensor(np.concatenate([v for v in va_list if len(v)]), dtype=torch.long, device=dev)

    # ---- contiguous-chunk sampler, used only when the timing latent is on ------------------- #
    # The forward recursion needs ORDERED rows, so batches become contiguous windows instead of
    # shuffled rows. Step count and rows-per-step are deliberately left EXACTLY as they were
    # (chunk == batch_size, one block per step), so this arm's optimiser budget matches the base
    # and the comparison is not confounded by a different number of updates -- the failure that
    # made psire3's gain search unreadable. The one real difference is that a batch is now
    # correlated rather than i.i.d.; the windows are redrawn (random fish, random offset) every
    # step, so the "same 256 bouts for all 80 epochs" trap this repo hit before cannot recur.
    _rng_ck = np.random.default_rng(seed + 991)
    _tr_by_fish = [r for r in tr_list if len(r) >= 2]
    _T = 0
    if seq_on:
        # ONE chunk length for every block, so they stack. Clamped by the SHORTEST fish rather
        # than per-fish, or np.stack would raise on a ragged batch the moment one fish is short.
        # CHUNK LENGTH IS THE COST KNOB, and this is the whole reason it is separate for the
        # recurrent model. `state_chain` walks T timesteps SEQUENTIALLY and is dispatch-bound, so
        # wall time ~ (optimiser steps) x T, essentially INDEPENDENT of batch size (measured: 2.0
        # ms/timestep at B=1, 2.6 ms at B=24). dt_latent's 256 was chosen for a forward recursion
        # that had no such loop; inheriting it made a GRU epoch take 125 s with the GPU idle.
        # 64 keeps well over the state's useful memory at a quarter of the cost -- and the big
        # batch then makes each of those cheaper steps far better informed, which is free here.
        _cfg_seq = (drift if drift else dt_latent) or {}
        _T = int(state_chunk) if state else int(_cfg_seq.get("chunk", DT_LATENT_DEFAULT["chunk"]))
        _T = max(2, min(_T, min(len(r) for r in _tr_by_fish)))

    def _sample_blocks(nb):
        out = []
        for _ in range(nb):
            rows = _tr_by_fish[_rng_ck.integers(len(_tr_by_fish))]
            o = int(_rng_ck.integers(0, len(rows) - _T + 1))
            out.append(rows[o:o + _T])
        return torch.tensor(np.stack(out), dtype=torch.long, device=dev)

    # SCHEDULED SAMPLING state. `Xt` is REBOUND per epoch (mixed for the training steps, clean for
    # validation); every loss closure above reads the name at call time, so they all follow it.
    Xt_clean = Xt
    _is_tr = torch.zeros(n, dtype=torch.bool, device=dev)
    _is_tr[tr] = True
    _ss_last, _ss_epochs = None, 0

    best, best_state, bad, hist = np.inf, None, 0, []
    _gstats = {}
    for _ep in range(epochs):
        net.train()
        Xt = Xt_clean
        if ss and ss_teacher_p(_ep, ss) < 1.0:
            p_tf = ss_teacher_p(_ep, ss)
            net.eval()
            with torch.no_grad():
                ix_all = torch.arange(n, device=dev)
                sa, sd, st_ = ss_draw(net, Xt_clean, az, lc, _z_for(ix_all), dth_scale,
                                      kin_scales, kin_clip)
            net.train()
            mix = (torch.rand(n, device=dev) < (1.0 - p_tf)) & _is_tr     # validation rows stay clean
            mixed = [torch.where(mix & torch.isfinite(y_), s_, y_)
                     for y_, s_ in ((ya, sa), (yd, sd), (yt, st_))]
            Xt = torch.tensor(np.asarray(ss_builder(*[m_.cpu().numpy() for m_ in mixed]), np.float32),
                              device=dev)
            _ss_last, _ss_epochs = (p_tf, float(mix.float().sum() / _is_tr.float().sum())), _ss_epochs + 1
        if seq_on:
            # ``seq_batch`` blocks per step, with ``steps`` scaled down so the epoch still covers
            # the same number of ROWS -- the optimiser sees fewer, larger batches, not more data.
            #
            # WHY IT IS 1 FOR dt_latent/drift AND >1 FOR THE GRU. Those arms were tuned at one
            # block per step and their archived numbers are only comparable at that optimiser
            # budget, so they stay put. The recurrent model has no such history and cannot afford
            # it: `state_chain` walks T timesteps SEQUENTIALLY, so at B=1 every step is 256 tiny
            # ops on a batch of one and the whole fit is Python-dispatch bound -- measured on an
            # A100 with the GPU sitting at ~0% while an epoch took minutes. Batching the blocks is
            # the same fix that took generation from 89 s to 12 s for 8 rollouts.
            nb = max(1, int(seq_batch))
            steps = max(1, int(np.ceil(len(tr) / max(_T * nb, 1))))
            for _s in range(steps):
                opt.zero_grad()
                JT.guarded_step(opt, seq_loss(_sample_blocks(nb)), net.parameters(), stats=_gstats)
            Xt = Xt_clean                                    # validation is teacher-forced
            net.eval()
            with torch.no_grad():
                # Validation blocks are ALREADY contiguous per fish (that is why they were built
                # that way), so each is a valid sequence for the recursion.
                if nb > 1:
                    # Batched: stack the per-fish blocks truncated to the shortest, so validation is
                    # ONE call instead of one per fish. Truncation makes it a fixed, equal-per-fish
                    # slice -- a consistent early-stopping signal, which is all it has to be.
                    rows = [r for r in va_list if len(r) >= 2]
                    tv = min(len(r) for r in rows)
                    blk = torch.tensor(np.stack([r[:tv] for r in rows]), dtype=torch.long,
                                       device=dev)
                    v = float(seq_loss(blk))
                else:
                    # one fish at a time, length-weighted, because they differ in length
                    num = 0.0; den = 0
                    for r in va_list:
                        if len(r) < 2:
                            continue
                        b_ = torch.tensor(r.reshape(1, -1), dtype=torch.long, device=dev)
                        num += float(seq_loss(b_)) * len(r); den += len(r)
                    v = num / max(den, 1)
            hist.append(v)
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
            continue
        # NOISE AUGMENTATION -- see JT.fit_mix_joint. The per-fish latent index `ix` is UNCHANGED
        # for the noisy copy (same rows, same fish), so z_i keeps its meaning and the augmentation
        # teaches the same fish to correct its own drift. Validation stays CLEAN.
        NT = (JT._noise_tensors(noise_builder, _ep, dth_scale, kin_scales, dev)
              if noise_builder is not None else None)
        tr_ep = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(tr_ep), batch_size):
            ix = tr_ep[i:i + batch_size]
            if NT is not None and noise_mode == "mask":
                mk = torch.rand(len(ix), device=dev) < float(noise_frac)
                opt.zero_grad()
                loss = _nll_arrays(ix, torch.where(mk.unsqueeze(1), NT[0][ix], Xt[ix]),
                                   torch.where(mk, NT[1][ix], az[ix]),
                                   torch.where(mk, NT[2][ix], lc[ix]),
                                   torch.where(mk, NT[3][ix], ya[ix]),
                                   torch.where(mk, NT[4][ix], yd[ix]),
                                   torch.where(mk, NT[5][ix], yt[ix]))
            else:
                opt.zero_grad(); loss = nll(ix)
            JT.guarded_step(opt, loss, net.parameters(), stats=_gstats)
        if NT is not None and noise_mode != "mask":       # 'concat': a second, noisy pass
            tr_ep2 = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(tr_ep2), batch_size):
                ix = tr_ep2[i:i + batch_size]
                opt.zero_grad()
                loss = _nll_arrays(ix, NT[0][ix], NT[1][ix], NT[2][ix],
                                   NT[3][ix], NT[4][ix], NT[5][ix])
                JT.guarded_step(opt, loss, net.parameters(), stats=_gstats)
        Xt = Xt_clean                                        # validation is teacher-forced
        net.eval()
        with torch.no_grad():
            v = float(nll(va))
        hist.append(v)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if ss:
        if _ss_last is None:
            print(f"    scheduled sampling: NEVER ACTIVE -- the fit stopped after {len(hist)} epochs, "
                  f"inside the {ss['warmup']}-epoch warmup")
        else:
            print(f"    scheduled sampling: {_ss_epochs} mixed epochs of {len(hist)}; last epoch "
                  f"teacher p {_ss_last[0]:.2f} (floor {ss['p_min']}), {100 * _ss_last[1]:.0f}% of "
                  f"train rows carried model-drawn lags")
    JT._warn_unstable(_gstats, hist)
    if best_state is not None:
        net.load_state_dict(best_state)
    return net.to("cpu").eval(), hist


# Spans a 10x range in 5 points. Every point costs `max_fish x n_draws` full rollouts inside EVERY
# fold's fit, so the grid is a real slice of ladder wall-clock -- the first full run spent 11,253 s
# on Metric A, a large part of it here. Five points plus the smoothness of the objective (a clean
# interior U: measured 3.44 / 2.43 / 1.72 / 1.06 / 1.11 / 1.57 / 2.48 on the earlier 7-point grid)
# is enough to locate the minimum to within the noise of the SD estimate itself.
# The bottom end reaches well below 1 because the label-shuffled CAPACITY control fits noise and can
# come out too WIDE; the top reaches 3 because on real labels it has been observed too NARROW.
GAIN_GRID = (0.4, 0.7, 1.2, 1.9, 3.0)
# WIDE grid, selected per-arm by ``re_gain_grid='wide'`` so every archived arm keeps the 5-point
# grid and stays bit-identical. Extends DOWNWARD, which is the direction the evidence points:
# 0.4 is the FLOOR, and a fold that pins there is saying the generated between-fish spread is too
# WIDE and it wants to shrink further than the grid allows. That is what the m=3 arm reported.
# It is a per-FOLD instability rather than a systematic offset -- the all-fish standalone fits of
# both psire2 and psire3 landed on 0.7, an interior point -- so the fix is to let a pinned fold
# find its own minimum and to REPORT the fitted gain, not to move the whole grid.
# Reading rule, pre-registered: if m=3 settles at 0.10-0.28 the third latent dimension is fitting
# noise across 12 training fish, and m=2 is the right dimensionality. That is an answer.
GAIN_GRID_WIDE = (0.10, 0.18, 0.28, 0.40, 0.70, 1.20, 1.90, 3.00)
# WIDE + headroom ABOVE 3.0, for the active-only pipeline. 'wide' only extends downward, but the
# first active-only latent smoke pinned a fold at the TOP (3.0), and real fish are more different
# from each other once quiescence is removed (between-fish share 0.34 vs 0.25), so the gain may
# genuinely need to exceed 3. A new name, not an edit to 'wide', so a refit of an archived 'wide'
# arm keeps searching exactly the grid it was fitted on.
# NO NEGATIVE POINTS, by construction: z ~ N(0, I) is symmetric, so a gain -g generates the same
# population of fish as +g (and scores the same marginal likelihood). They would only double the
# search cost and return each answer twice.
GAIN_GRID_WIDE_UP = GAIN_GRID_WIDE + (4.50, 6.50)
# PER-CHANNEL grid (re_gain_mode='per_channel'): five points spanning 0.25-2.5, which brackets every
# per-channel value the global searches have wanted (step ~0.4, interval / forward ~1-1.5) at a
# third of the wide grid's cost -- the per-channel search sweeps it 3 channels x 2 sweeps.
GAIN_GRID_PC = (0.25, 0.45, 0.8, 1.4, 2.5)


def apply_drift_gains(m, gd, gt, report):
    """Publish the rollout-fitted drift gains into the density's cfg (what the generator reads)
    with the log-normal re-centring, and print the outcome. Shared by ``fit`` and the local
    recalibration (rollout_model1_local.recalibrate_drift), so a re-searched checkpoint carries
    exactly what a freshly fitted one would.

    TWO-STATE design (``report['dt_via_latent']``): the interval chain is dt_latent, so the dt gain
    scales ITS amplitude (lam) and the slow chain's dt loading -- 0 -- gets no gain."""
    cf = m._cfg_feat
    _dd = float(cf["drift"].get("dist", 0.0))
    _dt = float(cf["drift"].get("dt", 0.0))
    if report.get("dt_via_latent"):
        _lam = float(cf["dt_latent"]["lam"])
        cf["drift"].update(g_dist=gd, g_dt=1.0, dist_c=0.5 * (gd * gd - 1.0) * _dd * _dd, dt_c=0.0)
        cf["dt_latent"].update(g_lam=gt, lam_c=0.5 * (gt * gt - 1.0) * _lam * _lam)
        m.drift.update(g_dist=gd, g_dt=1.0)
        m.dt_latent.update(g_lam=gt)
    else:
        cf["drift"].update(g_dist=gd, g_dt=gt,
                           dist_c=0.5 * (gd * gd - 1.0) * _dd * _dd,
                           dt_c=0.5 * (gt * gt - 1.0) * _dt * _dt)
        m.drift.update(g_dist=gd, g_dt=gt)
    m.drift_report = report
    got, tgt = report["got"], report["target"]
    pk = report.get("pick", {})
    print(f"    [{m.name}] drift gains dist {gd:.2f} ({pk.get('dist', '?')}) "
          f"{'dt-chain' if report.get('dt_via_latent') else 'dt'} {gt:.2f} ({pk.get('dt', '?')}): "
          f"window-acf1 dt {got['dt']:+.3f}/{tgt['dt']:+.3f}  "
          f"dist {got['dist']:+.3f}/{tgt['dist']:+.3f}  "
          f"rad {got['rad']:+.3f}/{tgt['rad']:+.3f} (REPORTED, not fitted)   (got/real)")


def gain_grid_for(name):
    """``'wide'`` -> 8 points, ``'wide_up'`` -> those + 4.5 and 6.5, else the archived 5-point grid."""
    name = str(name)
    return (GAIN_GRID_WIDE_UP if name == "wide_up" else
            GAIN_GRID_WIDE if name == "wide" else GAIN_GRID)


def _fish_level_stats(dth, dist, dt):
    """The three per-fish levels with the highest measured ICC: mean log travel, mean log interval,
    and forward fraction (bout rate 0.893, forward fraction 0.835, |dtheta| median 0.796). Forward
    fraction rather than a |dtheta| scale on purpose -- scaling |dtheta| is exactly how ``vigor_turn``
    failed, and forward fraction spans 4.5x across the 15 fish where median |dtheta| spans only 2x."""
    ld = np.log(np.clip(np.asarray(dist, float), 1e-6, None)); ld = ld[np.isfinite(ld)]
    lt = np.log(np.clip(np.asarray(dt, float), 1e-6, None)); lt = lt[np.isfinite(lt)]
    a = np.abs(np.asarray(dth, float)); a = a[np.isfinite(a)]
    return (float(ld.mean()) if len(ld) else np.nan,
            float(lt.mean()) if len(lt) else np.nan,
            float((a < M.DTHETA0).mean()) if len(a) else np.nan)


def state_rollout_batch(net, rec, cfg_feat, n, rng, n_cap=None):
    """``n`` independent free-runs of one record through the TORCH rollout, as sample dicts.

    THE ONE PLACE the recurrent free-run is built. Every caller that rolls out a state model --
    ``sample_batch`` for the metrics, ``search_loading_gain`` for the latent calibration -- comes
    through here, because the alternative is the trap this repo keeps hitting: ``MX._gen_for``
    returns the NUMPY generator, which has no GRU, so a calibrator that used it would tune the
    latent of a stateless model and hand the result to a stateful one.

    ``n_cap`` truncates the rollout to that many bouts. DIAGNOSTIC USE ONLY -- it is for the gain
    search, where the quantity estimated is a BETWEEN-fish spread driven by the latent draw rather
    than by within-trajectory sampling noise, so a few hundred bouts already pins each trajectory's
    level and the cost falls linearly. Never pass it from the metrics: those compare generated
    trajectories with real ones bout for bout, and a short rollout would change the comparison
    rather than just its precision. Default None = the record's own length.
    """
    from twin import rollout_torch as RT
    dev = torch.device("cpu")
    n = int(n)
    static = RT.build_static([rec], cfg_feat, dev)
    gen = torch.Generator(device="cpu").manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
    idx = torch.zeros(n, dtype=torch.long, device=dev)          # n copies of the same record
    with torch.no_grad():
        h0 = None
        if int(getattr(net, "h0_dim", 0)):
            w = torch.randn((n, net.h0_dim), device=dev, generator=gen)
            h0 = net.h0_from(w)
        st = RT.RolloutState(static, idx, cfg_feat, dev, hidden=int(net.state_hidden),
                             z=RT.draw_latent(cfg_feat, n, dev, gen), h0=h0)
        T = int(rec.n) if n_cap is None else min(int(rec.n), int(n_cap))
        o = RT.rollout(net, static, st, cfg_feat, T, tau=1.0, reparam=False, generator=gen)
    out = []
    for i in range(n):
        dth = o["dtheta"][i].numpy().astype(float)
        out.append(dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                        distance_px=o["dist"][i].numpy().astype(float),
                        dt_s=o["dt"][i].numpy().astype(float),
                        x_px=o["x"][i].numpy().astype(float),
                        y_px=o["y"][i].numpy().astype(float),
                        n_hits=0, n_bounces=0))
    return out


def search_loading_gain(net, records, cfg_feat, *, grid=GAIN_GRID, seed=0, max_fish=12, n_draws=2,
                        verbose=False, hold_zero=(), **gen_kw):
    """Stage B. One global gain on the latent loadings, matched to the REAL between-fish spread.

    WHY A CORRECTION IS NEEDED. Joint MAP shrinks: the prior pulls every z_i toward 0 and the
    loadings do not fully compensate, so the generated between-fish spread comes out too narrow.
    ``vigor``'s intercept is moment-matched to the real across-fish spread by construction, so an
    uncorrected swap would REGRESS rather than merely fail to help.

    WHY NOT BY LIKELIHOOD. The obvious estimator -- maximise the marginal likelihood of the training
    fish with z integrated out -- is monotone in the gain and has no interior optimum, measured: over
    a 0.5..3.0 grid the log-likelihood rose by 54k nats end to end and the argmax pinned at the top,
    then pinned again at 2.4 on a second pass. The reason is that the loadings were FITTED on these
    fish, so joint MAP has already placed each z_i where that fish's own data wants it; every
    training fish therefore sits far out along its own loading direction and a wider prior always
    scores better. A leak-free likelihood estimator would need leave-one-FISH-out refits (12 x 6 =
    72 fits per fold), which does not fit the budget. So the loading SCALE -- a single population
    hyperparameter -- is set by method of moments and everything else stays maximum likelihood.
    Note the status carefully for the write-up: this is a hyperparameter of a model that remains a
    normalised density and is fully SCORED, not an unscored transform applied after the fact. That
    is the distinction from the generation-time constants it replaces.

    MATCHED QUANTITY. The across-fish SD of the three highest-ICC per-fish levels, computed on
    TRAIN fish only. The objective is the sum of |log(generated SD / real SD)| -- a ratio, not a
    difference, because the three statistics live on different scales and the record already
    documents an unnormalised multi-objective calibration letting a ~1% term dominate a ~0.5 term
    115:1.

    SAMPLE SIZE. The generated SD is a POPULATION quantity, so it is estimated over
    ``max_fish x n_draws`` trajectories rather than one per fish. This matters: at 4 fish the
    objective is flat to within its own noise and the argmax pins at a grid edge; at 6 it is already
    a clean interior U (measured 4.19 / 3.37 / 2.80 / 1.05 / 0.62 / 1.28 over the grid). This is the
    project's own 3-fish trap -- per-fish spread estimates need many units -- so never run this on a
    small subset, and check the returned gain is not sitting on an endpoint.

    The z draws are keyed on (seed, fish_id, draw) and are therefore IDENTICAL across grid points,
    so the comparison between gains is paired and the grid noise largely cancels.
    """
    records = list(records)[:max_fish]
    real = np.array([_fish_level_stats(r.dtheta, r.distance_px, r.dt_s) for r in records])
    tgt = np.nanstd(real, axis=0, ddof=1)
    base = {n_: p.detach().clone() for n_, p in net.named_parameters() if n_.startswith("z_")}
    # HOLD OTHER BLOCKS AT ZERO while measuring this one. With a block-structured latent the border
    # gain also moves trajectory statistics -- a fish that responds differently to the wall covers
    # different ground -- so leaving it live means its between-fish contribution is attributed to
    # the core block, and the core gain shrinks to compensate. Measured symptom: the core gain
    # pinned at the grid FLOOR on the first bgblk run, the same signature m=3 produced before the
    # block existed. Each block must be measured against its own moment with the other neutral.
    hold = {n_: base[n_] for n_ in base if any(n_.startswith(h) for h in hold_zero)}
    # THE GENERATOR MUST MATCH THE MODEL. `MX._gen_for` returns the NUMPY generator, which cannot
    # carry a GRU state -- calibrating the latent on that and shipping it on a recurrent model is
    # exactly the two-objects trap. With `state` on we go through the torch rollout instead, and
    # batch the n_draws of each fish into ONE call: the rollout is dispatch-bound, so the draws
    # are close to free while the fish loop is not.
    _state = int(cfg_feat.get("state", 0))
    gen_fn = None if _state else MX._gen_for(cfg_feat)
    curve = []
    for g in grid:
        with torch.no_grad():
            for n_, p in net.named_parameters():
                if n_ in hold:
                    p.zero_()
                elif n_ in base:
                    p.copy_(base[n_] * float(g))
        stats = []
        if _state:
            # ONE batched call for every (record, draw). This is the calibrator that CANNOT be
            # skipped in the Metric A pass -- the gain is folded into the weights, so it changes the
            # scored model -- which made it the single largest remaining cost per fold: on the old
            # per-record path it was ~13 min on 4 fish, and it ran in BOTH CV passes. Batched and
            # capped it is well under a minute.
            #
            # The cap is safe here for the same reason it is elsewhere: the estimand is a
            # BETWEEN-fish spread of three per-fish level means, which converges long before a
            # record's full length; `_level_se` reports the residual within-trajectory floor.
            for out in state_rollout_many(net, records, cfg_feat, int(n_draws),
                                          np.random.default_rng([seed, 0]),
                                          n_cap=MX.STATE_CALIB_CAP):
                stats.append(_fish_level_stats(out["dtheta"], out["distance_px"], out["dt_s"]))
        else:
            for d in range(int(n_draws)):
                for r in records:
                    out = gen_fn(net, r, cfg_feat,
                                 np.random.default_rng([seed, int(r.fish_id), d]), **gen_kw)
                    stats.append(_fish_level_stats(out["dtheta"], out["distance_px"], out["dt_s"]))
        arr = np.array(stats)
        sd = np.nanstd(arr, axis=0, ddof=1)
        # ONE job for the gain: match the SPREAD, in log-ratio units (the three statistics live on
        # different scales and the record documents an unnormalised multi-objective calibration
        # letting a ~1% term dominate a ~0.5 term 115:1).
        #
        # The pooled LEVEL is held by the analytic re-centring in apply_loading_gain, NOT here, and
        # is reported below only as a diagnostic. Putting it in the objective was tried and is wrong:
        # it measures the model's overall bias rather than the gain's effect on it, so on an
        # under-trained fit it came out ~18 SD units and almost flat across the grid (18.5 -> 16.7)
        # and simply dragged the argmax to the top edge, drowning a spread term with a clean
        # interior minimum (3.44 -> 1.06 -> 2.48).
        err_sd = float(np.nansum(np.abs(np.log(np.maximum(sd, 1e-9) / np.maximum(tgt, 1e-9)))))
        err_mu = float(np.nansum(np.abs(np.nanmean(arr, axis=0) - np.nanmean(real, axis=0))
                                 / np.maximum(tgt, 1e-9)))
        err = err_sd
        curve.append((float(g), err, sd, err_sd, err_mu))
        if verbose:
            print(f"    gain {g:>4}: gen sd {np.round(sd, 4)} vs real {np.round(tgt, 4)}"
                  f"  err {err:.3f} (spread {err_sd:.3f} + level {err_mu:.3f})")
    with torch.no_grad():                                    # restore; the caller applies the winner
        for n_, p in net.named_parameters():
            if n_ in base:
                p.copy_(base[n_])
    best = min(curve, key=lambda c: c[1])
    return float(best[0]), dict(target_sd=tgt, target_mean=np.nanmean(real, axis=0),
                                curve=[(c[0], c[1], c[2].tolist(), c[3], c[4]) for c in curve])


# level index -> the loading whose spread it measures. `_fish_level_stats` returns
# (mean log travel, mean log interval, forward fraction), and the three loadings are read out of
# the SAME per-fish latent independently, so each has its own moment to be matched against.
CHANNEL_LEVEL = (("turn", 2), ("dist", 0), ("dt", 1))
LEVEL_NAME = ("mean log travel", "mean log interval", "forward fraction")


def _level_se(dth, dist, dt):
    """WITHIN-trajectory standard error of each level -- the noise floor under the between-fish sd.

    Each level is a mean over bouts, so a finite rollout estimates it with error, and that error
    adds IN QUADRATURE to the between-fish spread being fitted. At full length it is negligible;
    with ``n_cap`` set it is not, and the forward fraction is the channel that suffers first (it is
    a proportion, so its SE falls only as 1/sqrt(n) with no variance reduction from averaging a
    tight quantity). Reporting it is what stops a short rollout from being read as real spread --
    the trap the record already names for the 3-fish per-fish estimates.
    """
    ld = np.log(np.clip(np.asarray(dist, float), 1e-6, None)); ld = ld[np.isfinite(ld)]
    lt = np.log(np.clip(np.asarray(dt, float), 1e-6, None)); lt = lt[np.isfinite(lt)]
    a = np.abs(np.asarray(dth, float)); a = a[np.isfinite(a)]
    p = float((a < M.DTHETA0).mean()) if len(a) else np.nan
    return (float(ld.std(ddof=1) / np.sqrt(len(ld))) if len(ld) > 1 else np.nan,
            float(lt.std(ddof=1) / np.sqrt(len(lt))) if len(lt) > 1 else np.nan,
            float(np.sqrt(max(p * (1.0 - p), 0.0) / len(a))) if len(a) else np.nan)


def state_rollout_many(net, records, cfg_feat, n_draws, rng, n_cap=None, static=None):
    """``n_draws`` free-runs of EVERY record, in ONE batched rollout. Returns a flat list.

    WHY NOT JUST LOOP ``state_rollout_batch``. The rollout is dispatch-bound -- 4700 sequential
    steps of ~30 tiny ops -- so wall-clock tracks the number of CALLS far more than the batch width.
    Measured on the gain search: 6 records x 2 draws as six batch-2 calls cost 43 s per grid point,
    where the same 12 trajectories as ONE batch-12 call is a single pass over the timesteps. The
    fish loop, not the draw loop, was the expensive one.

    EQUAL LENGTH, and that is a feature here rather than a compromise. Every trajectory runs for
    ``T = min(record length)`` bouts, so each simulated fish contributes a level estimate of the
    same precision -- which is what the between-fish spread this feeds wants. A ragged batch would
    weight long fish more, and ``build_static`` pads with NaN dot centroid past a record's own
    length, so running to the max would silently roll short fish out with no stimulus.

    NOT for the metrics: those compare generated trajectories against real ones bout for bout, and
    a common truncated T would change the comparison rather than its precision. Use
    ``state_rollout_batch`` there.

    ``static`` may be a prebuilt ``RT.build_static`` for the SAME records, hoisted by a caller that
    rolls out repeatedly -- it depends only on the records and cfg, so rebuilding it once per grid
    point (12 records x up to 6686 bouts of stimulus arrays) is pure repeated work.
    """
    from twin import rollout_torch as RT
    dev = torch.device("cpu")
    records = list(records)
    nd = int(n_draws)
    T = min(int(r.n) for r in records)
    if n_cap is not None:
        T = min(T, int(n_cap))
    if static is None:
        static = RT.build_static(records, cfg_feat, dev)
    elif int(static["B"]) != len(records):
        raise ValueError(f"prebuilt static has B={int(static['B'])} but {len(records)} records "
                         "were passed -- it must be built from the SAME list, or `idx` below "
                         "would select the wrong fish's dish and stimulus programme.")
    gen = torch.Generator(device="cpu").manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
    # row -> record. repeat_interleave so a record's draws are contiguous, which keeps the returned
    # order (record 0 draw 0, record 0 draw 1, record 1 draw 0, ...) readable at the call site.
    idx = torch.arange(len(records), device=dev).repeat_interleave(nd)
    B = int(idx.shape[0])
    with torch.no_grad():
        h0 = None
        if int(getattr(net, "h0_dim", 0)):
            h0 = net.h0_from(torch.randn((B, net.h0_dim), device=dev, generator=gen))
        st = RT.RolloutState(static, idx, cfg_feat, dev, hidden=int(net.state_hidden),
                             z=RT.draw_latent(cfg_feat, B, dev, gen), h0=h0)
        o = RT.rollout(net, static, st, cfg_feat, T, tau=1.0, reparam=False, generator=gen)
    out = []
    for i in range(B):
        dth = o["dtheta"][i].numpy().astype(float)
        out.append(dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                        distance_px=o["dist"][i].numpy().astype(float),
                        dt_s=o["dt"][i].numpy().astype(float),
                        x_px=o["x"][i].numpy().astype(float),
                        y_px=o["y"][i].numpy().astype(float),
                        n_hits=0, n_bounces=0))
    return out


TAIL_GAIN_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


def _fish_dt_spread(dt, W=60, runs=None):
    """Per-fish mean, over 60-bout windows, of the window SD of the interval (s) -- the C2ST's
    `dt_std` feature. With ``runs`` only windows inside one run count (the real side)."""
    dt = np.asarray(dt, float)
    n = len(dt)
    if runs is None:
        starts = range(0, max(1, n - W + 1), W)
    else:
        r = np.asarray(runs)
        starts = [a for a in range(0, max(1, n - W + 1), W) if a + W <= n and r[a] == r[a + W - 1]]
    vals = [np.nanstd(dt[a:a + W]) for a in starts if a + W <= n]
    return float(np.mean(vals)) if vals else np.nan


def search_tail_gain(net, records, cfg_feat, *, grid=TAIL_GAIN_GRID, seed=0, max_fish=8, n_draws=2,
                     **gen_kw):
    """The ONE gain of the tail loading (re_tail), matched to the between-fish spread of the window
    interval spread (sd over fish of `_fish_dt_spread`, real 0.216 s over the 15 fish) -- the
    moment the loading exists to produce, and NOT the mean-interval spread the interval gain is
    matched to (sharing that gain is how the dispersion loading of Model 12 widened every fish).
    Same rollout stream as the level search ([seed, fish, draw]) so the z draws are common across
    grid points and the curve is monotone up to rollout noise; picked by ``MX._pick_gain``
    (monotone smoothing, log-interpolation, plateau stop). Returns (gain, report)."""
    records = list(records)[:max_fish]
    if int(cfg_feat.get("state", 0)):
        raise NotImplementedError("search_tail_gain: recurrent path not implemented")
    tgt = float(np.nanstd([_fish_dt_spread(r.dt_s, runs=getattr(r, "run_id", None)) for r in records], ddof=1))
    base = net.z_dt_w.weight.detach().clone()
    gen_fn = MX._gen_for(cfg_feat)
    curve = []
    for x in grid:
        with torch.no_grad():
            net.z_dt_w.weight.copy_(base * float(x))
        gs = [gen_fn(net, r, cfg_feat, np.random.default_rng([seed, int(r.fish_id), d]), **gen_kw)
              for r in records for d in range(int(n_draws))]
        curve.append(float(np.nanstd([_fish_dt_spread(o["dt_s"]) for o in gs], ddof=1)))
    with torch.no_grad():
        net.z_dt_w.weight.copy_(base)
    g, status, sm = MX._pick_gain(grid, curve, tgt, plateau_tol=0.05 * max(tgt, 1e-9))
    return float(g), dict(grid=list(grid), curve=curve, smoothed=[float(v) for v in sm],
                          target=tgt, gain=float(g), status=status)


def search_loading_gain_per_channel(net, records, cfg_feat, *, grid=GAIN_GRID, seed=0, max_fish=8,
                                    n_draws=2, sweeps=2, n_cap=None, verbose=True, **gen_kw):
    """Fit THREE gains -- one per head -- each matched to its own between-fish moment.

    WHY ONE SCALAR IS NOT ENOUGH. ``search_loading_gain`` multiplies all three core loadings by a
    single number, on the implicit assumption that joint-MAP shrinkage is uniform across channels.
    Measured on ``psib_gruz`` it is not, and not by a little: travel spread came out 3.13x too WIDE
    while forward-fraction spread was ~4x too NARROW (gen 0.0129 vs real 0.0537). The two want the
    gain to move in OPPOSITE directions, so a single knob cannot express the correction -- and
    because the objective sums |log sd-ratio| over the three levels, the blown-up travel term
    dominated, drove the gain to its grid FLOOR on 4 of 5 folds, still failed to fix travel, and
    crushed the turn loading that needed to grow. This is that diagnosis turned into a measurement.

    COORDINATE DESCENT, NOT THREE INDEPENDENT SEARCHES. The heads are chained (turn -> distance |
    turn -> dt | turn, distance), so ``z_turn`` moves ``a_z``, which feeds both downstream heads.
    Sweeping in chain order with the others held at their CURRENT value -- not at zero -- is what
    accounts for that coupling; holding them at zero would measure each channel in a model that is
    not the one being shipped.

    THE GAIN = 0 ROW IS THE POINT, not a formality. It switches the latent off entirely and reports
    what the model produces on its OWN. For a recurrent model that is the direct test of whether the
    GRU is already generating between-fish spread from its own sampled history -- in which case z is
    redundant and the two are double-counting, which is precisely what a single global gain can
    never reveal because it can scale all three together but never test zero.

    Returns ``({'turn': g, 'dist': g, 'dt': g}, report)``. The gains are RAW scalings; the caller
    folds them in with ``apply_loading_gain``, which adds the per-channel re-centring.
    """
    records = list(records)[:max_fish]
    real = np.array([_fish_level_stats(r.dtheta, r.distance_px, r.dt_s) for r in records])
    tgt = np.nanstd(real, axis=0, ddof=1)
    base = {n_: p.detach().clone() for n_, p in net.named_parameters() if n_.startswith("z_")}
    if not base:
        raise ValueError("net has no z_* loadings -- this is not a fish_re model")
    _state = int(cfg_feat.get("state", 0))
    gen_fn = None if _state else MX._gen_for(cfg_feat)
    # HOISTED: depends only on (records, cfg), so building it per grid point was ~30 rebuilds of
    # 12 records' worth of stimulus arrays for nothing.
    _static = None
    if _state:
        from twin import rollout_torch as _RT
        _static = _RT.build_static(records, cfg_feat, torch.device("cpu"))
    cache = {}

    def _levels(g):
        """(sd over fish of the 3 levels, mean over fish) at gains ``g``, cached on the triple."""
        key = tuple(round(float(g[k]), 6) for k, _ in CHANNEL_LEVEL)
        if key in cache:
            return cache[key]
        with torch.no_grad():                       # RAW scaling; re-centring shifts a level, not
            for n_, p in net.named_parameters():    # a spread, so it cannot change what is measured
                if n_ in base:
                    p.copy_(base[n_] * float(g[_owner(n_)]))
        if _state:
            # ONE batched call for every (record, draw) -- the rollout is dispatch-bound, so the
            # fish loop is what costs, not the batch width. See state_rollout_many.
            gs = state_rollout_many(net, records, cfg_feat, int(n_draws),
                                    np.random.default_rng([seed, 0]), n_cap=n_cap, static=_static)
        else:
            gs = [gen_fn(net, r, cfg_feat, np.random.default_rng([seed, int(r.fish_id), d]),
                         **gen_kw) for r in records for d in range(int(n_draws))]
        stats, ses = [], []
        for o in gs:
            stats.append(_fish_level_stats(o["dtheta"], o["distance_px"], o["dt_s"]))
            ses.append(_level_se(o["dtheta"], o["distance_px"], o["dt_s"]))
        arr = np.array(stats)
        # every trajectory draws its OWN latent, so these ARE independent simulated fish and the
        # pooled sd is the population quantity -- the same pooling `search_loading_gain` documents.
        out = (np.nanstd(arr, axis=0, ddof=1), np.nanmean(arr, axis=0),
               np.sqrt(np.nanmean(np.asarray(ses, float) ** 2, axis=0)))
        cache[key] = out
        return out

    def _owner(pname):
        """Which gain scales this parameter. z_body has no moment of its own, so it follows the
        turn channel -- the one the objective pins directly (it is zero unless re_body is on)."""
        for k, _ in CHANNEL_LEVEL:
            if pname.startswith("z_" + k):
                return k
        return "turn"

    g = {k: 1.0 for k, _ in CHANNEL_LEVEL}
    trace = []
    for s in range(int(sweeps)):
        # CHAIN ORDER: turn first, so each downstream sweep sees a settled upstream.
        for key, li in CHANNEL_LEVEL:
            def _err(x, key=key, li=li):
                gg = dict(g, **{key: float(x)})
                sd = _levels(gg)[0][li]
                return abs(np.log(max(float(sd), 1e-9) / max(float(tgt[li]), 1e-9)))
            # INTERPOLATED pick (Model 13). The raw grid argmin could only land within a factor
            # 1.34 of the target on GAIN_GRID_PC (spacing 1.8x): Model 12's g3 went from 1.5x over
            # to 0.77x under on the step spread. The spread curve is monotone in the gain up to
            # rollout noise, so the same rule the drift gain uses applies: PAV-smooth it,
            # interpolate in log-gain to the target, stop at the plateau if it is unreachable.
            curve = [float(_levels(dict(g, **{key: float(x)}))[0][li]) for x in grid]
            g[key], _st, _ = MX._pick_gain(grid, curve, float(tgt[li]),
                                           plateau_tol=0.05 * max(float(tgt[li]), 1e-9))
            trace.append((s, key, g[key], _err(g[key]), _st))
            if verbose:
                sd, mu, se = _levels(g)
                # flush: this loop is minutes long and stdout is block-buffered off a tty,
                # so without it a Colab cell shows nothing at all until the very end.
                print(f"    sweep {s} {key:>4}: gain {g[key]:>5}  "
                      f"sd({LEVEL_NAME[li]}) {sd[li]:.4f} vs real {tgt[li]:.4f}"
                      + ("   <- GRID EDGE" if g[key] in (grid[0], grid[-1]) else ""), flush=True)
    sd_fit, mu_fit, se_fit = _levels(g)
    sd_one, _, _ = _levels({k: 1.0 for k, _ in CHANNEL_LEVEL})    # the un-gained fit, for contrast
    sd_off, _, se_off = _levels({k: 0.0 for k, _ in CHANNEL_LEVEL})   # THE DOUBLE-COUNTING READ
    # REACHABLE SPAN per channel: the spread at each end of the grid, others held at their fitted
    # values. Mostly cached from the last sweep. This is what separates "the grid is too narrow"
    # from "this loading cannot produce that spread at ANY gain", and those need opposite actions.
    span = {k: (float(_levels(dict(g, **{k: grid[0]}))[0][li]),
                float(_levels(dict(g, **{k: grid[-1]}))[0][li])) for k, li in CHANNEL_LEVEL}
    with torch.no_grad():                                         # restore; the caller applies
        for n_, p in net.named_parameters():
            if n_ in base:
                p.copy_(base[n_])
    # WARN ON THE RESULT, NOT ON GRID POSITION -- the lesson `_search_wall_reject` already records
    # ("~83% of fits warned by chance and the warning carried no information"). A channel sitting on
    # an endpoint whose fitted ratio is 1.08 is MATCHED; one at 0.30 is not, wherever it sits.
    edge = [k for k, li in CHANNEL_LEVEL
            if abs(np.log(max(float(sd_fit[li]), 1e-9) / max(float(tgt[li]), 1e-9))) > 0.22]
    at_edge = [k for k, _ in CHANNEL_LEVEL if g[k] in (grid[0], grid[-1])]
    # SATURATED: even the largest gain in the grid cannot reach the target. Widening the grid is
    # then the WRONG advice -- the loading has no more spread to give.
    sat = [k for k, li in CHANNEL_LEVEL if max(span[k]) < 0.6 * float(tgt[li])]
    if verbose:
        for k, li in CHANNEL_LEVEL:
            if k in sat:
                print(f"    ** {k}: SATURATED. Over the whole grid its level spread only spans "
                      f"{span[k][0]:.4f}-{span[k][1]:.4f} against a real {tgt[li]:.4f}. No gain "
                      f"reaches that, so this is not a calibration problem -- widening the grid "
                      f"will not help and the loading needs replacing. **")
            elif k in edge:
                print(f"    WARNING: {k} missed its target ({sd_fit[li]:.4f} vs {tgt[li]:.4f})"
                      + (" at a grid edge -- widen the grid" if k in at_edge else ""))
    return g, dict(target_sd=tgt, target_mean=np.nanmean(real, axis=0), got_sd=sd_fit,
                   got_mean=mu_fit, sd_gain1=sd_one, sd_latent_off=sd_off, gains=dict(g),
                   grid_edge=edge, at_grid_edge=at_edge, saturated=sat, span=span,
                   trace=trace, level_names=LEVEL_NAME,
                   # the within-trajectory noise floor under every sd above; `sampling_floor` is
                   # the one that matters for the latent-OFF row, since that is where a spurious
                   # "the model already has spread" reading would come from.
                   noise_floor=se_fit, noise_floor_off=se_off, n_cap=n_cap,
                   n_units=int(len(records) * int(n_draws)))


# Geometric, spanning 22x, and it goes DOWN as well as up on purpose. Up is the expected need --
# `rej` produces an across-fish sd(mean r/R) of 0.0246 against a real 0.0779, so ~3x more spread is
# wanted -- but the wiring check pinned this at the lower end on an under-trained model, and a grid
# that cannot shrink would have clipped that silently. A method-of-moments search must be able to
# go both ways, and the loading-gain history in this file is a catalogue of edge-pinned searches
# that were read as results. 6 points x 8 fish x 2 draws = 96 rollouts per fit, paid for by the
# ~108 the pinned d0 saves in the boundary search.
SPATIAL_GAIN_GRID = (0.4, 0.8, 1.5, 2.8, 5.0, 9.0)


def real_radial_sd(records):
    """Across-fish SD of mean r/R on the REAL data -- the moment the spatial block is matched to.

    This is the one statistic `search_loading_gain`'s objective does not contain, which is exactly
    why a block-structured latent needs its own gain: the three existing moments (mean log travel,
    mean log interval, forward fraction) say nothing about where in the dish a fish prefers to sit,
    so a single global gain would scale the spatial dimension by whatever makes turn and tempo
    spread right. Measured on all 15 fish: 0.0779, against a generated 0.025-0.043."""
    u = [float(np.mean(np.hypot(np.asarray(r.x_px, float) - r.dish_cx,
                                np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R))
         for r in records]
    return float(np.std(u, ddof=1))


def search_spatial_gain(net, records, cfg_feat, *, grid=SPATIAL_GAIN_GRID, seed=0, max_fish=8,
                        n_draws=2, verbose=False, **gen_kw):
    """Method-of-moments gain for the SPATIAL block, matched to the across-fish SD of mean r/R.

    Same shape and same justification as ``search_loading_gain`` -- the loading scale cannot be
    fitted by in-sample likelihood (it is monotone in the gain, because joint MAP has already put
    each z_i where that fish's data wants it), so one population hyperparameter is set by moments
    while everything else stays maximum likelihood.

    Runs AFTER the core gain has been applied, so the trajectories it measures are the ones that
    will actually be scored. 8 fish x 2 draws = 16 trajectories, matching the core search: the
    project's standing lesson is that per-fish spread estimates need many units (at 4 fish the
    objective is flat within its own noise), and a radial SD is no exception.

    Reports clamp saturation. ``z_bord`` feeds ``g = exp(clamp(., +-1.5))``, so past some gain the
    extra scale does nothing but push fish against the bound -- at which point the CLAMP is the
    binding constraint and the right conclusion is 'widen the clamp', not 'the mechanism is dead'.
    """
    records = list(records)[:max_fish]
    tgt = real_radial_sd(records)
    base = net.z_bord.weight.detach().clone()
    gen_fn = MX._gen_for(cfg_feat)
    curve = []
    for g in grid:
        with torch.no_grad():
            net.z_bord.weight.copy_(base * float(g))
        mus, sat = [], []
        for d in range(int(n_draws)):
            for r in records:
                out = gen_fn(net, r, cfg_feat,
                             np.random.default_rng([seed, int(r.fish_id), d]), **gen_kw)
                u = np.hypot(np.asarray(out["x_px"], float) - r.dish_cx,
                             np.asarray(out["y_px"], float) - r.dish_cy) / r.dish_R
                mus.append(float(np.mean(u)))
        # fraction of the PRIOR that the clamp bites on, at this gain
        w = float(torch.linalg.vector_norm(net.z_bord.weight).item())
        sat = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(1.5 / max(w, 1e-9) / np.sqrt(2.0)))))
        sd = float(np.std(mus, ddof=1))
        curve.append((float(g), abs(np.log(max(sd, 1e-9) / max(tgt, 1e-9))), sd, sat))
        if verbose:
            print(f"    spatial gain {g:>4}: gen sd(mean r/R) {sd:.4f} vs real {tgt:.4f}  "
                  f"clamped {100 * sat:.1f}%")
    with torch.no_grad():
        net.z_bord.weight.copy_(base)
    # INTERPOLATED pick (Model 19), the same rule the core and drift gains use since Models 8 / 13:
    # the radial spread is monotone in the gain up to rollout noise, so PAV-smooth the curve and
    # interpolate in log-gain to the target instead of taking the nearest of six grid points
    # (2x apart). The raw argmin is what pinned this search at its floor on the old m=3 arm.
    g_pick, status, sm = MX._pick_gain([c[0] for c in curve], [c[2] for c in curve], tgt,
                                       plateau_tol=0.05 * max(tgt, 1e-9))
    j = int(np.argmin([abs(np.log(c[0] / max(g_pick, 1e-9))) for c in curve]))   # nearest grid point, for the report
    return float(g_pick), dict(target_sd=tgt, got_sd=curve[j][2], clamp_frac=curve[j][3], status=status,
                               smoothed=[float(v) for v in sm],
                               curve=[(c[0], c[1], c[2], c[3]) for c in curve])


def apply_loading_gain(net, gain, which="all"):
    """Fold the fitted gain into the loadings, RE-CENTRING so the pooled marginals do not move.

    ``which`` selects the block: 'all' (archived behaviour, bit-identical), 'core'
    (turn/dist/dt + body) or 'spatial' (the border gain only). The spatial block needs NO
    re-centring: it scales an INPUT column rather than a log-normal location, so there is no
    exp(gamma^2/2) inflation to cancel, and its effect is bounded by the +-1.5 clamp anyway.

    Folding rather than threading a constant through: every downstream path -- the numpy rollout
    predictor, the quadrature scorer, the saved checkpoint -- then sees one self-consistent model,
    with no chance of one of them missing the correction.

    THE RE-CENTRING IS NOT OPTIONAL. The kinematic heads are log-normal, so a latent shift on mu
    carries into the ARITHMETIC mean: E[exp(mu + gamma z)] = exp(mu + gamma^2/2) at z ~ N(0, I).
    Training fits mu with the latent already present, so in-sample that is absorbed -- but rescaling
    gamma -> g gamma afterwards inflates every generated travel and interval by
    exp((g^2 - 1) gamma^2 / 2) with nothing to cancel it. Measured on the first end-to-end run
    without this: dist_w1 3.04 -> 6.78 and dot_dist_bias 11.4 -> 20.4 px, i.e. the mechanism bought
    between-fish spread by destroying the pooled marginal. This is the same de-biasing `vigor` does
    with its -var/2 term ("keeps E[exp(z)] = 1 so the population mean is unchanged").

    The turn head needs no analogue: its loading tilts the forward-mixture LOGIT rather than a
    location, so it is bounded by the softmax and cannot inflate a marginal without bound. Its
    pooled drift is instead held by the gain objective, which matches forward fraction directly.

    PER-CHANNEL GAINS. ``gain`` may instead be a dict ``{'turn': .., 'dist': .., 'dt': ..}`` (any
    subset; missing keys default to 1.0), which scales the three core loadings SEPARATELY. That is
    the `gruz` finding: MAP shrinkage is not uniform across channels, and one scalar for three of
    them is a knob that cannot express the correction. Measured on psib_gruz -- travel spread came
    out 3.13x too WIDE while forward-fraction spread was ~4x too NARROW, so the single gain drove
    to its grid floor on 4 of 5 folds, failed to fix travel, and crushed the turn loading on the
    way. See ``search_loading_gain(per_channel=True)``.

    Each channel re-centres with its OWN gamma^2, which is the whole reason this cannot be done by
    calling the scalar version three times: gamma^2 must be read from the ORIGINAL loading, and a
    second call would read the already-scaled one and double-correct.
    """
    if which == "spatial":
        with torch.no_grad():
            net.z_bord.weight.mul_(float(gain))
        return net
    if not isinstance(gain, dict):
        gain = {"turn": float(gain), "dist": float(gain), "dt": float(gain),
                # 'all' folds the border gain in with the rest, exactly as it always did; 'core'
                # leaves it for the spatial pass. Kept as a key so both paths are ONE code path.
                **({"bord": float(gain)} if which != "core" else {})}
    g = {k: float(gain.get(k, 1.0)) for k in ("turn", "dist", "dt", "bord")}
    # the TAIL loading (re_tail) has its own gain (search_tail_gain) and follows neither the scalar
    # nor the interval gain: only an explicit {'tail': g} moves it
    g["tail"] = float(gain.get("tail", 1.0))
    with torch.no_grad():
        if getattr(net, "tail", False) and g["tail"] != 1.0:
            net.z_dt_w.weight.mul_(g["tail"])
        for key, lin, head, n_c in (("dist", net.z_dist, net.dist_head, net.n_dist),
                                    ("dt", net.z_dt, net.dt_head, net.n_dt)):
            gamma2 = float((lin.weight ** 2).sum())          # from the ORIGINAL loadings
            head.bias[n_c:2 * n_c] -= 0.5 * (g[key] * g[key] - 1.0) * gamma2
        net.z_turn.weight.mul_(g["turn"])
        if getattr(net, "tsize", False):
            # re_tsize: the turn-size loading follows the TURN gain; linear in z (zero mean), so no re-centring
            net.z_tsize.weight.mul_(g["turn"])
        net.z_dist.weight.mul_(g["dist"])
        net.z_dt.weight.mul_(g["dt"])
        # the interval head's extra z loadings (re_disp / re_lag) follow the dt gain; neither is a
        # log-normal location, so neither needs the re-centring above
        if getattr(net, "disp", False):
            net.z_dt_sig.weight.mul_(g["dt"])
        if getattr(net, "lag_slice", None):
            net.z_lag.weight.mul_(g["dt"])
        # z_body is part of the core block and has no moment of its own, so it follows the TURN
        # gain -- it is zero unless re_body is on, in which case it feeds all three heads and the
        # turn channel is the one the objective pins directly.
        net.z_body.weight.mul_(g["turn"])
        if hasattr(net, "z_bord") and g["bord"] != 1.0:
            net.z_bord.weight.mul_(g["bord"])
    return net


# =========================================================================== #
#  Adapter
# =========================================================================== #
class MLPMixJointREDensity(JT.MLPMixJointDensity):
    """Joint model with a fitted per-fish random effect, marginalised exactly at score time."""

    def __init__(self, *args, re_m=2, re_sigma_init=0.5, re_gain_search=True, re_ghq_k=7,
                 re_zero=False, re_shuffle=False, re_body=False, contain_decouple=False,
                 contain_graded=False, rad_indiv=False, re_bgain=False,
                 re_gain_grid="std", dt_latent=None, re_block=False, drift=None,
                 drift_gain_search=True, state=0, seq_batch=64, state_chunk=64, h0_dim=0,
                 ss=None, re_gain_mode="global", re_disp=False, re_lag=False, re_tail=False,
                 turn_skip=False, depth=1, re_tsize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.re_tail = bool(re_tail)            # Model 13: z on the interval component weights
        self.re_tsize = bool(re_tsize)          # Model 38: z on the turning components' scales (turn size)
        # Model 17: a skip path from the stimulus inputs to the turn head / a deeper body
        self.turn_skip = bool(turn_skip)
        self.depth = int(depth)
        # SCHEDULED SAMPLING on the lag inputs (see SS_DEFAULT): True or {} takes the defaults, a
        # dict overrides p_min / warmup / decay. Training-time only; nothing at generation.
        self.ss = (dict(SS_DEFAULT) if ss is True else (dict(SS_DEFAULT, **ss) if ss else None))
        # Model 12: z on the interval head's widths (re_disp) / on its lag slopes (re_lag) -- see
        # MDNMixJointRE.__init__ for the measurements behind each.
        self.re_disp = bool(re_disp)
        self.re_lag = bool(re_lag)
        self.re_gain_grid = str(re_gain_grid)
        # LOADING GAIN, one scalar over the three loadings ('global', every archived arm) or THREE
        # ('per_channel': search_loading_gain_per_channel, each matched to its own between-fish
        # moment). Measured on S3_adt_z2: the single gain lands on 0.7 on every fold as a compromise
        # -- step spread 1.5-3x too wide, interval / forward spread 0.6-0.9x too narrow.
        self.re_gain_mode = str(re_gain_mode)
        if self.re_gain_mode not in ("global", "per_channel"):
            raise ValueError(f"re_gain_mode must be 'global' or 'per_channel', got {re_gain_mode!r}")
        # Block-structured latent: reserve the LAST dimension for the border gain so
        # re_bgain finally has an axis it cannot lose to turn/tempo. Needs re_bgain.
        self.re_block = bool(re_block)
        self.spatial_gain = None
        self.spatial_gain_report = None
        # Slow WITHIN-trajectory timing state (None = off, byte-identical to every archived arm).
        # True or {} takes DT_LATENT_DEFAULT; a dict overrides n_levels / chunk / phi_init / span.
        self.dt_latent = (dict(DT_LATENT_DEFAULT) if dt_latent is True
                          else (dict(DT_LATENT_DEFAULT, **dt_latent) if dt_latent else None))
        # `tau` (bouts) is the readable way to PIN the interval chain's timescale: it becomes
        # phi_fixed, so no s_phi_raw parameter exists and the timescale cannot be refitted.
        if self.dt_latent and self.dt_latent.get("tau") is not None \
                and self.dt_latent.get("phi_fixed") is None:
            self.dt_latent["phi_fixed"] = drift_phi(self.dt_latent["tau"])
        # re_bgain: LIKELIHOOD-VISIBLE spatial individuality -- a fitted per-fish gain on the
        # border input columns (see MDNMixJointRE). Built because every generation-time radial
        # lever is measured dead on the psi/basis models: the multiplicative swing is inert on
        # bulk=0 biases, and the gate offset probed 0.0335 -> 0.0335 on psib (the net's own,
        # fish-identical wall response dominates the weakened patch). Spatial personality must
        # be LEARNED. Pair with re_m=3 -- at m=2 both axes are already claimed by turn/tempo.
        self.re_bgain = bool(re_bgain)
        self.re_m = int(re_m)
        self.re_sigma_init = float(re_sigma_init)
        self.re_gain_search = bool(re_gain_search)
        self.re_ghq_k = int(re_ghq_k)
        self.re_zero = bool(re_zero)          # training-loop control: prior pinned at 0, arm inert
        self.re_shuffle = bool(re_shuffle)    # capacity control: same capacity, fish labels permuted
        self.re_body = bool(re_body)          # OFF: level-only. See joint_forward_re's docstring.
        # 2-D containment search (bulk against the MEAN radius, shell against the TAIL) instead of
        # one scale over both. The 1-D form ties a bias that acts EVERYWHERE to one that acts only
        # inside the shell, so it can only buy wall occupancy by dragging the whole distribution
        # inward. Measured on re_swap: rad_mean 0.392 vs real 0.449 and rad_median 0.385 vs 0.456,
        # both ~0.45 sd low and the top radial tells, while rad_q90 (-0.12 sd) and rad_frac_wall
        # (+0.19 sd) were nearly right -- i.e. the wall was fine and the CENTRE was over-corrected.
        self.contain_decouple = bool(contain_decouple)
        # Wide graded containment: bulk pinned at 0, gate opened to mid-dish so the bias acts on
        # APPROACH rather than only inside the shell. Beats the shipped 1-D form on BOTH the mean
        # radius and wall occupancy simultaneously; see _search_containment's graded branch.
        self.contain_graded = bool(contain_graded)
        # Radial individuality: the likelihood latent z loads on fwd logit / mu_l / mu_tau and has
        # NO spatial channel, so every fishre rollout has the same radial personality (measured
        # between-fish SD of mean r/R 0.030 vs real 0.075). rad_indiv adds a generation-time
        # per-trajectory multiplicative swing on the containment bias (fish_re['rad_sd'], moments-
        # calibrated by MX._search_rad_indiv, drawn from a SPAWNED stream so CRN pairing holds).
        self.rad_indiv = bool(rad_indiv)
        # DRIFT: a within-trajectory AR(1) state loading on all three heads plus (via the rollout
        # search) the containment bias. The one kind of object never tried -- every previous
        # individuality mechanism is STATIC per trajectory and so contributes exactly zero to the
        # across-window autocorrelation, which is the largest measured gap. See DRIFT_DEFAULT.
        self.drift = (dict(DRIFT_DEFAULT) if drift is True
                      else (dict(DRIFT_DEFAULT, **drift) if drift else None))
        self.drift_gain_search = bool(drift_gain_search)
        self.drift_report = None
        # RECURRENT STATE (see MDNMixJointRE). Mutually exclusive with drift/dt_latent/fish_re: it
        # is meant to REPLACE them, and running them together would put three mechanisms in
        # competition for the same variance -- the trap that produced the dt_latent/z
        # double-counting (between_fish_share 0.304 against a real 0.255).
        self.state = int(state)
        # MEASURED defaults (6 fish, hidden 128, 4 epochs, s/epoch and val NLL):
        #   B=1  chunk=256 -> 114.6 s,  9.51     <- the inherited dt_latent setting
        #   B=16 chunk=64  ->   7.2 s,  9.94
        #   B=48 chunk=64  ->   5.1 s, 14.77     <- batch WITHOUT lr: faster, under-trained
        #   B=48 chunk=32  ->   3.8 s, 10.66
        # Re-measured on an A100, 12 fish, 8 epochs (s/epoch, val, and whether it had converged):
        #   B=16  lr 1e-3 c64  -> 30.7 s,  9.76  CONVERGED (flat 5 epochs), 51% of steps NaN-skipped
        #   B=64  lr 2e-3 c64  ->  8.9 s,  9.98  still falling      <- CHOSEN
        #   B=256 lr 4e-3 c64  ->  3.6 s, 12.14  PLATEAUED (stuck)
        #   B=64  lr 2e-3 c128 ->  9.0 s, 11.28  no faster, trains worse
        # 3.4x faster per epoch and still descending, so it passes B=16 well inside B=16's budget.
        # The NaN-skip rate also falls with batch (51% -> 29% -> 21%): one extreme bout can blow up
        # a batch of ONE sequence but is diluted across 64.
        # Cost is per-TIMESTEP, not per-row (2.0 ms/timestep at B=1, 2.6 ms at B=24), so the
        # chunk is the real knob and the batch is close to free -- it just makes each of the
        # cheaper steps far better informed.
        self.seq_batch = int(seq_batch) if state else 1
        self.state_chunk = int(state_chunk)
        # per-fish seed for the recurrent state; 0 reproduces gru1 exactly
        self.h0_dim = int(h0_dim) if state else 0
        if self.state:
            # DRIFT + GRU IS NOW ALLOWED, and the guard that refused it was built on a premise that
            # has since been measured false. It read "the GRU replaces them" -- but gru1 measured
            # between_fish_share 0.071 against a real 0.255 (worse than the memoryless MLP's 0.215),
            # and var_ratio distance 3.13 on gruz with win_acf1 ~ 0, i.e. the GRU produces white
            # window-to-window jitter rather than the slow correlated drift it was supposed to
            # subsume. It replaces neither the between-fish latent nor the slow one.
            #
            # The two are ORTHOGONAL by construction, not merely by measurement: win_acf1 demeans
            # WITHIN a fish, so a static latent scores exactly zero on it; and a zero-mean AR(1)
            # shifts every window of a fish by a different amount, so it contributes nothing to the
            # between-fish term. One mechanism per variance component.
            #
            # dt_latent STAYS refused: it is a slow state on the interval head alone, so it and the
            # drift genuinely do compete for the same variance -- that is the double-counting the
            # original guard was right about.
            if self.dt_latent:
                raise ValueError("state (GRU) with dt_latent: both are slow states on the interval "
                                 "channel, competing for the same variance. Use drift, which loads "
                                 "on all three heads.")
            # fish_re (re_zero=False) IS allowed alongside the GRU, and is the `gruz` arm: the GRU
            # carries WITHIN-fish slow structure, z carries BETWEEN-fish identity, and gru1 measured
            # those moving independently (var_ratio fixed, between_fish_share collapsed to 0.071).
            # What is refused is BOTH static latents at once -- h_0 and z parameterise the same
            # between-fish quantity, so they would fit against each other, and scoring them would
            # need the product quadrature grid.
            if self.h0_dim and not self.re_zero:
                raise ValueError("h0_dim>0 with the fish_re latent live (re_zero=False): two static "
                                 "per-fish latents for one between-fish variance. Pick one -- "
                                 "h0_dim=2 with re_zero=True is `gru2`, h0_dim=0 with "
                                 "re_zero=False is `gruz`.")
        self.re_sd = None
        self.re_gain = None
        self.re_gain_report = None

    def _ss_builder(self, train_records):
        """Feature-matrix rebuilder for scheduled sampling: the clean matrix with ONLY the two
        history blocks (turn lags, kinematic lags) recomputed from the mixed real / model-drawn
        series. Everything else is copied from the clean matrix on purpose -- the border block
        under ``psi_ref='travel'`` derives the previous heading from the CURRENT bout's dtheta
        (``heading - dtheta``), so rebuilding it from a mixed series would leak a model draw into a
        teacher-forced input. Column offsets follow build_feature_matrix's block order:
        [K hist][(3K symbols)][stim][stimhist][border][adapt][kin][lag_valid][clock][run_clock]."""
        import copy
        cfg = self._cfg_feat
        r0 = train_records[0]
        K = int(cfg["K"])
        k0 = K + (3 * K if cfg["include_symbols"] else 0)
        k0 += M._stim_block(r0, cfg["stim_mode"], cfg["rich_scales"],
                            phi_ref=cfg.get("phi_ref", "ori")).shape[1]
        if int(cfg.get("K_stim", 1)) > 1:
            k0 += MB._stimhist_block(r0, cfg).shape[1]
        if cfg.get("use_border"):
            k0 += MB._border_block(r0, cfg).shape[1]
        if cfg.get("use_adapt"):
            k0 += MB._adapt_block(r0, cfg).shape[1]
        kin_w = MB.kin_width(cfg)
        offs = np.cumsum([0] + [len(r.dtheta) for r in train_records])
        Xc = np.concatenate([MB.build_feature_matrix(r, cfg) for r in train_records], 0)

        def build(ya, yd, yt):
            X = Xc.copy()
            for i, r in enumerate(train_records):
                sl = slice(int(offs[i]), int(offs[i + 1]))
                X[sl, :K] = M._history_block(ya[sl], K, cfg["dth_scale"],
                                             run_pos=getattr(r, "run_pos", None))
                r2 = copy.copy(r)
                r2.distance_px, r2.dt_s = yd[sl], yt[sl]
                X[sl, k0:k0 + kin_w] = MB._kin_history_block(r2, cfg)
            return X
        return build

    # ---- the pieces score() and the EM step both need, built once per record --------------- #
    def _rec_arrays(self, rec):
        X = MB.build_feature_matrix(rec, self._cfg_feat)
        a_z, ld_c = JT._cond_cols(rec.dtheta, rec.distance_px, self._cfg_feat["dth_scale"],
                                  self._cfg_feat["kin_scales"])
        y = (rec.dtheta, np.asarray(rec.distance_px, float), np.asarray(rec.dt_s, float))
        return X, (a_z, ld_c), y

    def fit(self, train_records, ctx):
        c = ctx.cfg or {}
        self.K = int(c.get("mlp_K", self.K))
        self.stim_mode = c.get("mlp_stim_mode", self.stim_mode)
        self.hidden = int(c.get("mlp_hidden", self.hidden))
        self.epochs = int(getattr(self, "epochs_override", None)
                          if getattr(self, "epochs_override", None) is not None
                          else c.get("mlp_epochs", self.epochs))
        self.patience = int(getattr(self, "patience_override", None)
                            if getattr(self, "patience_override", None) is not None
                            else c.get("mlp_patience", self.patience))
        self.lr = float(c.get("mlp_lr", self.lr))
        self._dt_pool, self._dist_pool = ctx.dt_pool, ctx.dist_pool
        self._rich_scales = ctx.rich_scales
        alld = np.concatenate([r.dtheta for r in train_records])
        self._dth_scale = float(max(np.std(alld), 1.0))
        adist = np.concatenate([r.distance_px for r in train_records])
        adist = adist[np.isfinite(adist) & (adist > 0)]
        adt = np.concatenate([r.dt_s for r in train_records]); adt = adt[np.isfinite(adt) & (adt > 0)]
        self._kin_scales = (float(np.log(adist).mean()), float(max(np.log(adist).std(), 1e-3)),
                            float(np.log(adt).mean()), float(max(np.log(adt).std(), 1e-3)))
        self._kin_clip = (float(np.quantile(adist, 0.999)), float(np.quantile(adt, 0.999)))
        # Mirrors MLPMixJointDensity.fit's cfg build -- every flag added there MUST be added here
        # too, or the fishre family silently trains/generates without it.
        self._cfg_feat = dict(K=self.K, stim_mode=self.stim_mode, include_symbols=self.include_symbols,
                              dth_scale=self._dth_scale, rich_scales=self._rich_scales,
                              use_border=self.use_border, border_mode=self.border_mode,
                              wall_mode=self.wall_mode, use_adapt=False, adapt_lambda=0.6,
                              use_kin=True, joint=True, kin_scales=self._kin_scales,
                              K_kin=self.K_kin, kin_clip=self._kin_clip,
                              n_dist=self.n_dist, n_dt=self.n_dt,
                              dt_snap=self.dt_snap, psi_ref=self.psi_ref, phi_ref=self.phi_ref,
                              K_stim=self.K_stim)
        if self.noise_aug:
            self._cfg_feat["noise_aug"] = dict(self.noise_aug)
        if getattr(self, "dt_trunc", None):
            self._cfg_feat["dt_trunc"] = float(self.dt_trunc)
        # cfg_feat_extra (clock_mode / clock_tau_s / lag_valid / wall_reject ...) -- merged LAST,
        # exactly as MLPMixJointDensity.fit does. It was missing here, which is the drift the
        # comment above warns about: a latent model trained on the active-only pipeline silently
        # lost its block clock and lag-validity features while its no-latent twin kept them, so
        # "+ latent" would have been compared against a different feature set.
        self._cfg_feat.update(getattr(self, "cfg_feat_extra", None) or {})
        # The random-effect net's chained heads (joint_forward_re) are hard-wired dtheta ->
        # distance -> dt. Refuse any other order rather than train one and label it another.
        if tuple(getattr(self, "head_order", JT.DEFAULT_ORDER)) != JT.DEFAULT_ORDER:
            raise ValueError("MLPMixJointREDensity supports head_order 'adt' only "
                             "(joint_forward_re is hard-wired); got "
                             f"{JT.order_tag(self.head_order)!r}")
        # Per-bout regime weights (MB.regime_weights), the same recipe MLPMixJointDensity.fit uses.
        weights = None
        weights_per_head = None
        if getattr(self, "regime_w", None):
            ws, infos = zip(*[MB.regime_weights(r, **self.regime_w) for r in train_records])
            weights = np.concatenate(ws)
            if "per_head" in infos[0]:            # a second regime with its own head set (Model 25)
                weights_per_head = {h: np.concatenate([i["per_head"][h] for i in infos])
                                    for h in ("turn", "dist", "dt")}
            self.regime_info = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]
                                if k not in ("heads", "heads2", "per_head")}
            print(f"    [{self.name}] regime weights {self.regime_w}: onset "
                  f"{100 * self.regime_info['frac_onset']:.1f}% of bouts -> "
                  f"{100 * self.regime_info['share_onset']:.0f}% of the loss, pursuit "
                  f"{100 * self.regime_info['frac_pursuit']:.1f}% -> "
                  f"{100 * self.regime_info['share_pursuit']:.0f}%")

        bg_slice = None
        if self.re_bgain:
            # column range of the border block inside x:
            # [K hist][(3K symbols)][stim][stimhist][border][kin]
            r0 = train_records[0]
            stim_w = M._stim_block(r0, self.stim_mode, self._rich_scales,
                                   phi_ref=self._cfg_feat.get("phi_ref", "ori")).shape[1]
            # K_stim inserts its block BETWEEN stim and border, so the offset must include it or
            # re_bgain would silently scale the wrong columns.
            sh_w = MB._stimhist_block(r0, self._cfg_feat).shape[1]
            bord_w = MB._border_block(r0, self._cfg_feat).shape[1]
            s0 = self.K + (3 * self.K if self.include_symbols else 0) + stim_w + sh_w
            bg_slice = (int(s0), int(s0 + bord_w))
        lag_slice = None
        if self.re_lag:
            # column range of the INTERVAL lags inside x, the same offset arithmetic `_ss_builder`
            # uses: [K hist][(3K symbols)][stim][stimhist][border][adapt][step lags][interval lags]
            if not self._cfg_feat.get("kin_dt_lags", True):
                raise ValueError("re_lag needs the interval lags (kin_dt_lags True or 'centred')")
            r0 = train_records[0]
            k0 = self.K + (3 * self.K if self.include_symbols else 0)
            k0 += M._stim_block(r0, self.stim_mode, self._rich_scales,
                                phi_ref=self._cfg_feat.get("phi_ref", "ori")).shape[1]
            if int(self._cfg_feat.get("K_stim", 1)) > 1:
                k0 += MB._stimhist_block(r0, self._cfg_feat).shape[1]
            if self._cfg_feat.get("use_border"):
                k0 += MB._border_block(r0, self._cfg_feat).shape[1]
            if self._cfg_feat.get("use_adapt"):
                k0 += MB._adapt_block(r0, self._cfg_feat).shape[1]
            Kk = int(self._cfg_feat.get("K_kin", self.K))
            lag_slice = (int(k0 + Kk), int(k0 + 2 * Kk))
        per_fish = [MB.build_feature_matrix(r, self._cfg_feat) for r in train_records]
        X = np.concatenate(per_fish, 0)
        fish_idx = np.concatenate([np.full(len(m_), i, dtype=np.int64)
                                   for i, m_ in enumerate(per_fish)])
        skip_slices = None
        if self.turn_skip:
            # the stimulus block [K hist][(3K symbols)][stim ...] and the tail blocks appended after
            # the run clock (align, static): both straight onto the turn head's outputs
            r0 = train_records[0]
            k_s = self.K + (3 * self.K if self.include_symbols else 0)
            stim_w = M._stim_block(r0, self.stim_mode, self._rich_scales,
                                   phi_ref=self._cfg_feat.get("phi_ref", "ori")).shape[1]
            skip_slices = [(int(k_s), int(k_s + stim_w))]
            tw = MB.tail_width(self._cfg_feat)
            if tw:
                skip_slices.append((int(X.shape[1] - tw), int(X.shape[1])))
        y_ang = np.concatenate([r.dtheta for r in train_records])
        y_dist = np.concatenate([np.asarray(r.distance_px, float) for r in train_records])
        y_dt = np.concatenate([np.asarray(r.dt_s, float) for r in train_records])

        # ---- Stage A --------------------------------------------------------------------- #
        self._model, self.history = fit_mix_joint_re(
            X, y_ang, y_dist, y_dt, fish_idx, len(train_records), self.hidden,
            self._dth_scale, self._kin_scales, m=self.re_m,
            epochs=self.epochs, lr=self.lr, patience=self.patience,
            batch_size=self.batch_size, seed=self.seed, n_dist=self.n_dist, n_dt=self.n_dt,
            sigma_init=self.re_sigma_init, freeze_sigma_zero=self.re_zero,
            shuffle_labels=self.re_shuffle, use_body=self.re_body,
            border_gain=self.re_bgain, border_slice=bg_slice, dt_latent=self.dt_latent,
            block=self.re_block, drift=self.drift, state=self.state,
            seq_batch=(self.seq_batch if self.state else 1), state_chunk=self.state_chunk,
            h0_dim=self.h0_dim,
            ss=self.ss, ss_builder=(self._ss_builder(train_records) if self.ss else None),
            kin_clip=self._kin_clip, weights=weights, dt_trunc=getattr(self, "dt_trunc", None),
            disp=self.re_disp, lag_slice=lag_slice, tail=self.re_tail,
            weight_heads=(self.regime_w or {}).get("heads"), weights_per_head=weights_per_head,
            skip_slices=skip_slices, depth=self.depth, tsize=self.re_tsize,
            **JT._noise_kwargs(self, train_records))
        if self.turn_skip:
            with torch.no_grad():
                print(f"    [{self.name}] turn-head skip path on columns {skip_slices}: "
                      f"|W| per output {np.round(self._model.skip.weight.norm(dim=1).numpy(), 3)}")
        if self.re_tail:
            with torch.no_grad():
                print(f"    [{self.name}] tail loading per component "
                      + str(np.round(self._model.z_dt_w.weight.norm(dim=1).numpy(), 3)))
        if self.re_disp or self.re_lag:
            with torch.no_grad():
                msg = []
                if self.re_disp:
                    msg.append("dispersion loading per component "
                               + str(np.round(self._model.z_dt_sig.weight.norm(dim=1).numpy(), 3)))
                if self.re_lag:
                    msg.append(f"lag-slope loading per lag (cols {lag_slice}) "
                               + str(np.round(self._model.z_lag.weight.norm(dim=1).numpy(), 3)))
            print(f"    [{self.name}] " + "; ".join(msg))
        if self.state:
            # The GRU is the generator's state, so generation MUST go through the torch rollout --
            # the numpy generator has no way to carry it. Recorded in cfg_feat so `sample` and
            # every calibrator dispatch on the same flag.
            self._cfg_feat["state"] = int(self.state)
            self._cfg_feat["h0_dim"] = int(self.h0_dim)
        if self.drift:
            # Publish the FITTED loadings into cfg_feat BEFORE the calibrators, which all roll out.
            # Same discipline dt_latent and fish_re follow, and for the same reason: the generator
            # and every calibrator must see the object the likelihood was actually fitted with.
            with torch.no_grad():
                self._cfg_feat["drift"] = dict(
                    self.drift, phi=drift_phi(self.drift["tau"]),
                    turn=float(self._model.d_turn), dist=float(self._model.d_dist),
                    dt=float(self._model.d_dt), bord=float(self._model.d_bord))
            d = self._cfg_feat["drift"]
            print(f"    [{self.name}] drift: tau {d['tau']:.0f} bouts (phi {d['phi']:.4f}, PINNED)  "
                  f"loadings turn {d['turn']:+.4f}  dist {d['dist']:+.4f}  dt {d['dt']:+.4f}")
        if self.dt_latent:
            # Publish the FITTED (lam, phi) into cfg_feat so the generator and every calibrator
            # that rolls out see the same object the likelihood was fitted with. Doing this here --
            # before Stage B and the calibrators, all of which roll out -- is the same discipline
            # `fish_re` follows just below, and for the same reason.
            with torch.no_grad():
                _phi = float(self._model.s_phi()); _lam = float(self._model.s_lam)
            self._cfg_feat["dt_latent"] = dict(self.dt_latent, lam=_lam, phi=_phi)
            print(f"    [{self.name}] timing latent: lam {_lam:+.4f}  phi {_phi:.4f}  "
                  f"(tau {-1.0 / np.log(max(_phi, 1e-9)):.1f} bouts, "
                  f"{self.dt_latent['n_levels']} levels)")

        # ---- Stage B --------------------------------------------------------------------- #
        # The prior is N(0, I) throughout; `re_sd` exists only so the generator and the quadrature
        # grid read the same object, and is zeroed for the pinned control.
        self.re_sd = np.zeros(self.re_m) if self.re_zero else np.ones(self.re_m)
        # The generator reads this; JT.generate_mix_joint_loop draws z ~ N(0, diag(sd^2)) per
        # trajectory from a SPAWNED stream. Setting it BEFORE the gain search and the calibrators is
        # deliberate: they all roll out train fish through MX._gen_for, so they see PRIOR draws,
        # which is the law that will actually be evaluated. Using the fitted z_i there would
        # calibrate against a different model than the one scored.
        self._cfg_feat["fish_re"] = dict(m=int(self.re_m), sd=np.asarray(self.re_sd, float))
        # GENERATION-TIME CALIBRATORS: every one below is invisible to `score()`, which reads only
        # dth_scale / kin_scales / state / drift. `cross_validate_likelihood` therefore sets this so
        # the Metric A pass does not spend ~30 min per fold fitting constants it will never read.
        # The loading gain is the ONE exception and is NOT skipped: it is folded into the weights
        # (apply_loading_gain), so it changes the scored model, and dropping it here would make
        # Metric A describe a different object from Metric B.
        _skip_cal = bool((ctx.cfg or {}).get("skip_calibrators", False))
        self.re_gain, self.re_gain_report = 1.0, None
        if not self.re_zero and self.re_gain_search:
            # max_fish=8, matching _search_kin_feedback's precedent: this runs inside every fold's
            # fit, so it is charged 5 grid points x max_fish x n_draws rollouts every time. 8 fish
            # x 2 draws = 16 trajectories is what the smoke suite showed gives a clean interior
            # minimum; 12 fish was measurably no better and cost 50% more wall-clock.
            _grid = gain_grid_for(self.re_gain_grid)
            _gen_kw = dict(closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                           stim_loop=self.stim_loop)
            if self.re_gain_mode == "per_channel":
                if self.re_block:
                    raise NotImplementedError("re_gain_mode='per_channel' with re_block: the "
                                              "spatial block has no per-channel path")
                # THREE gains by coordinate descent in chain order (turn -> step -> interval), each
                # matched to its own between-fish moment; ~3x the rollouts of the single search.
                self.re_gain, self.re_gain_report = search_loading_gain_per_channel(
                    self._model, train_records, self._cfg_feat, grid=GAIN_GRID_PC, seed=self.seed,
                    max_fish=min(self.contain_fish, 8), n_draws=2, sweeps=2, **_gen_kw)
                apply_loading_gain(self._model, dict(self.re_gain), which="all")
                _r = self.re_gain_report
                print(f"    [{self.name}] loading gains per head "
                      f"{ {k: round(v, 3) for k, v in self.re_gain.items()} } (grid {GAIN_GRID_PC}, "
                      f"2 sweeps); spread gen/real: "
                      + "  ".join(f"{LEVEL_NAME[li]} {_r['got_sd'][li] / max(_r['target_sd'][li], 1e-9):.2f}"
                                  for _, li in CHANNEL_LEVEL))
            else:
                self.re_gain, self.re_gain_report = search_loading_gain(
                    self._model, train_records, self._cfg_feat, grid=_grid, seed=self.seed,
                    max_fish=min(self.contain_fish, 8), **_gen_kw,
                    hold_zero=(("z_bord",) if self.re_block else ()))
                apply_loading_gain(self._model, self.re_gain,
                                   which=("core" if self.re_block else "all"))
                # ALWAYS print the fitted gain. Until now only the pinned case was visible, so a fold
                # that landed at an interior-but-implausible value left no trace anywhere -- the
                # results CSV has no column for it and the per-fold report is discarded.
                print(f"    [{self.name}] loading gain {self.re_gain} "
                      f"(grid {self.re_gain_grid}, {len(_grid)} pts)")
            self.tail_gain, self.tail_gain_report = 1.0, None
            if self.re_tail:
                # The tail loading's OWN gain, searched after the core gains are folded so it
                # measures the model that will be scored; matched to the between-fish spread of the
                # window interval spread (see search_tail_gain), never to the mean-interval spread.
                self.tail_gain, self.tail_gain_report = search_tail_gain(
                    self._model, train_records, self._cfg_feat, grid=TAIL_GAIN_GRID, seed=self.seed,
                    max_fish=min(self.contain_fish, 8), n_draws=2, **_gen_kw)
                apply_loading_gain(self._model, {"tail": self.tail_gain}, which="all")
                _r = self.tail_gain_report
                print(f"    [{self.name}] tail gain {self.tail_gain:.3f} ({_r['status']}; grid "
                      f"{TAIL_GAIN_GRID}): sd over fish of window dt_std "
                      + " ".join(f"{v:.3f}" for v in _r["curve"]) + f" vs real {_r['target']:.3f}")
            if self.re_block:
                # SECOND, SEPARATE gain for the spatial block.NOT optional -- the core objective (mean log
                # travel, mean log interval, forward fraction) contains no radial statistic, so a
                # single gain would scale the spatial dimension by moments it has nothing to do
                # with. Runs after the core gain is applied so it measures the model that will
                # actually be scored.
                self.spatial_gain, self.spatial_gain_report = search_spatial_gain(
                    self._model, train_records, self._cfg_feat, seed=self.seed,
                    max_fish=min(self.contain_fish, 8), closed_loop=self.closed_loop,
                    guardrail=self.guardrail, eps=self.eps, stim_loop=self.stim_loop)
                apply_loading_gain(self._model, self.spatial_gain, which="spatial")
                _r = self.spatial_gain_report
                print(f"    [{self.name}] spatial gain {self.spatial_gain}  "
                      f"sd(mean r/R) {_r['got_sd']:.4f} vs real {_r['target_sd']:.4f}  "
                      f"clamped {100 * _r['clamp_frac']:.1f}%")
                if self.spatial_gain in (SPATIAL_GAIN_GRID[0], SPATIAL_GAIN_GRID[-1]):
                    print(f"    WARNING [{self.name}]: spatial gain pinned at a grid edge "
                          f"({self.spatial_gain})")
                if _r["clamp_frac"] > 0.25:
                    print(f"    WARNING [{self.name}]: the +-1.5 border-gain clamp bites on "
                          f"{100 * _r['clamp_frac']:.0f}% of the prior -- the CLAMP is the binding "
                          "constraint here, not the mechanism")
            if self.re_gain_mode == "global" and self.re_gain in (_grid[0], _grid[-1]):
                print(f"    WARNING [{self.name}]: loading gain pinned at the grid edge "
                      f"({self.re_gain}); the between-fish spread could not be matched, so widen "
                      f"the grid (re_gain_grid='wide') before trusting this arm"
                      + ("  (expected for the label-shuffled control)" if self.re_shuffle else ""))
        self._re_nodes, self._re_log_pi = _ghq(self.re_m, self.re_sd, k=self.re_ghq_k)

        # ---- the inherited calibration pipeline, unchanged and in the same order ----------- #
        if not _skip_cal and self.vigor:
            vg = MX._calibrate_vigor(train_records)
            vg["apply_turn"] = False
            self._cfg_feat["vigor"] = vg
            if self.vigor_drift_calib:
                MX.MLPMixKinDensity._rescale_vigor_drift(self, train_records)
        if not _skip_cal and self.kin_fb_calib:
            tg, vtg = MX._kin_acf_targets(train_records)
            kfb, _kvs = MX._search_kin_feedback(
                self._model, train_records, self._cfg_feat, tg, vtg,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
            self._cfg_feat["kin_feedback"] = kfb
        if not _skip_cal and self.contain:
            self._cfg_feat["wall_mode"] = "reflect"
            self._cfg_feat.update(MX._search_wall(
                self._model, train_records, self._cfg_feat, mode=self.contain_mode,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed,
                max_fish=self.contain_fish, w_occ=self.contain_w_occ,
                decouple=self.contain_decouple, graded=self.contain_graded))
        if not _skip_cal and self.innov_comp:
            self._cfg_feat["kin_innov"] = MX._search_innov_comp(
                self._model, train_records, self._cfg_feat,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
        if not _skip_cal and self.indiv:
            self._cfg_feat["indiv"] = MX._search_indiv(
                self._model, train_records, self._cfg_feat, MX._indiv_targets(train_records),
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
        if self.rad_indiv:
            # AFTER indiv/innov (it widens on top of whatever they left) and BEFORE the contain
            # re-fit below -- widening mechanisms drive wall contact, so containment must be
            # re-fit with the widening active or the occupancy calibration is against the wrong
            # model (the contain_vc lesson).
            self._rad_indiv_report = MX._search_rad_indiv(
                self._model, train_records, self._cfg_feat,
                MX._indiv_targets(train_records)[1],
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
            self._cfg_feat["fish_re"]["rad_sd"] = self._rad_indiv_report["rad_sd"]
        if not _skip_cal and self.drift and self.drift_gain_search:
            # AFTER containment (bord SCALES centre_bias, so there must be a fitted bias to scale)
            # and BEFORE the re-fit below -- a wandering containment strength changes wall contact,
            # so occupancy has to be re-calibrated with the drift live or it is calibrated against
            # a model that no longer exists. Same ordering lesson as contain_vc / rad_indiv.
            _gd, _gt, self.drift_report = MX._search_drift_gain(
                self._model, train_records, self._cfg_feat,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
            apply_drift_gains(self, _gd, _gt, self.drift_report)
        _rad_widened = bool(self.rad_indiv
                            and self._cfg_feat.get("fish_re", {}).get("rad_sd", 0.0) > 0)
        if not _skip_cal and self.contain and (self.indiv or self.innov_comp or _rad_widened
                             or (self.drift and self.drift_gain_search)):
            self._cfg_feat.update(MX._search_wall(
                self._model, train_records, self._cfg_feat, mode=self.contain_mode,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed + 1,
                max_fish=min(self.contain_fish, 8), w_occ=self.contain_w_occ,
                decouple=self.contain_decouple, graded=self.contain_graded))
        self.config = dict(mode="joint_re", hidden=self.hidden, n_features=X.shape[1], K=self.K,
                           K_kin=self.K_kin, stim_mode=self.stim_mode, use_border=self.use_border,
                           use_kin=True, n_dist=self.n_dist, n_dt=self.n_dt,
                           re_m=self.re_m, n_fish=len(train_records),
                           head_order=JT.order_tag(JT.DEFAULT_ORDER))
        return self

    def score(self, rec):
        """The marginal predictive (``_score``), with the repulsion term's per-bout side input when there is one."""
        with JT.repulsion_side(self, rec):
            return self._score(rec)

    def _score(self, rec):
        """THREE channels, each the EXACT marginal predictive with z integrated out.

        The held-out fish has no fitted embedding, so what is reported is
        log p(y_t | y_{1:t-1}, x) under the fitted population prior -- the same object every other
        model in the zoo reports, and the same object the harness's own HMM path
        (``evaluate.forward_logdens_states``) reports for a latent-variable model. No adapter change
        is needed and the bits/bout column stays comparable.

        The three channels are marginalised in the SAME chained order the model factorises them, by
        reweighting the belief WITHIN the bout: the distance channel is scored under the posterior
        that has already seen this bout's turn, and the interval channel under the posterior that has
        also seen this bout's distance. Using one shared belief for all three would double-count the
        evidence and the three would no longer chain-rule to log p(dtheta, l, tau | past).
        """
        from scipy.special import logsumexp
        X, (a_z, ld_c), (ya, yd, yt) = self._rec_arrays(rec)
        if int(getattr(self, "state", 0)) and getattr(self._model, "drift", None):
            # SRNN STAGE 1: GRU + pinned AR(1) drift (+ optionally the static z).
            #
            # Structurally IDENTICAL to the non-recurrent drift branch below -- same product grid
            # g = (q, j), same `I (x) P` transition (z never moves, s does), same chained
            # within-bout reweighting -- with one substitution: the emission parameters come from
            # the recurrent trunk instead of `_trunk`. Written to mirror that code deliberately;
            # a second, differently-shaped implementation of the same recursion is how two objects
            # start diverging.
            #
            # The GRU pass is hoisted OUT of the grid (`state_trunks`), because the state sees
            # neither z nor s. One recurrent pass per record, Q x L head evaluations on top.
            from scipy.special import logsumexp as _lse
            net = self._model
            s_nodes, logP_t, log_pi0_t = ar1_chain(net.d_phi(), int(net.drift["n_levels"]))
            logP = logP_t.numpy().astype(float); log_pi0 = log_pi0_t.numpy().astype(float)
            L = int(s_nodes.shape[0])
            live_z = bool(self._state_latent_dim()) and not int(net.h0_dim)
            zn, zlp = ((self._re_nodes, self._re_log_pi) if live_z
                       else (np.zeros((1, max(int(self.re_m), 1))), np.zeros(1)))
            Q, n = zn.shape[0], X.shape[0]
            La = np.empty((n, Q, L)); Ld = np.empty((n, Q, L)); Lt = np.empty((n, Q, L))
            W = np.empty((n, Q, 5)); SHP = np.empty((9, n, Q))
            with torch.no_grad():
                t1 = lambda a, d=np.float32: torch.tensor(np.asarray(a, d)).reshape(1, -1)
                u = torch.tensor(state_gru_input(ya, yd, yt, self._cfg_feat["dth_scale"],
                                                 self._cfg_feat["kin_scales"])).reshape(1, n, 3)
                Xq = torch.tensor(np.asarray(X, np.float32)).reshape(1, n, -1)
                trunks = net.state_trunks(Xq, u)                       # ONCE, shared by all Q x L
                for q in range(Q):
                    zq = (torch.tensor(np.asarray(zn[q:q + 1], np.float32)) if live_z else None)
                    la, ld_, lt_ = net.state_chain_levels(
                        Xq, t1(a_z), t1(ld_c), u, t1(ya), t1(yd), t1(yt), s_nodes,
                        z=zq, trunks=trunks, split=True)
                    La[:, q, :] = la[0].numpy(); Ld[:, q, :] = ld_[0].numpy()
                    Lt[:, q, :] = lt_[0].numpy()
                    # symbol track at the drift's prior MEAN (s = 0), as the non-recurrent drift
                    # branch does: with the drift on, the turn emission depends on s, so a single
                    # symbol PMF cannot be exact. The approximation is confined to this column --
                    # the three density channels are marginalised exactly, and they are Metric A.
                    zc = None if zq is None else net._z_core(zq).repeat_interleave(n, dim=0)
                    o0, d0_, t0_ = net._state_heads(trunks.reshape(n, -1),
                                                    t1(a_z).reshape(-1, 1),
                                                    t1(ld_c).reshape(-1, 1), zc)
                    ang0 = net._params_at(o0, d0_, t0_)[0]
                    W[:, q, :] = np.exp(ang0[0].numpy())
                    for j in range(9):
                        SHP[j, :, q] = ang0[j + 1].numpy()

            G = Q * L
            La = La.reshape(n, G); Ld = Ld.reshape(n, G); Lt = Lt.reshape(n, G)
            log_pi_g = (zlp.reshape(-1, 1) + log_pi0.reshape(1, -1)).reshape(-1)
            dv = np.asarray(yd, float); tv = np.asarray(yt, float)
            md = np.isfinite(dv) & (dv > 0); mt = np.isfinite(tv) & (tv > 0)
            Ld_w = np.where(md.reshape(-1, 1), Ld, 0.0)
            Lt_w = np.where(mt.reshape(-1, 1), Lt, 0.0)
            Ltot = La + Ld_w + Lt_w
            # Running belief over (z, s): the prior times everything seen strictly BEFORE bout t,
            # propagated through I (x) P.
            lw0 = np.empty((n, G))
            lw0[0] = log_pi_g
            for t in range(1, n):
                prev = (lw0[t - 1] + Ltot[t - 1]).reshape(Q, L)
                lw0[t] = _lse(prev[:, :, None] + logP[None, :, :], axis=1).reshape(-1)
            d0 = _lse(lw0, axis=1)
            logdens = _lse(lw0 + La, axis=1) - d0                      # p(dtheta | past)
            lw1 = lw0 + La; d1 = _lse(lw1, axis=1)
            ldv = _lse(lw1 + Ld, axis=1) - d1                          # p(l | past, dtheta)
            lw2 = lw1 + Ld_w; d2 = _lse(lw2, axis=1)
            ltv = _lse(lw2 + Lt, axis=1) - d2                          # p(tau | past, dtheta, l)
            pw = np.exp(lw0 - d0.reshape(-1, 1)).reshape(n, Q, L).sum(axis=2)
            slq_q = np.stack([MX.symbol_logq_mix(W[:, q, :], *[SHP[j, :, q] for j in range(9)])
                              for q in range(Q)], axis=1)
            slq = _lse(slq_q + np.log(np.maximum(pw, 1e-300))[:, :, None], axis=1)
            slq = slq - _lse(slq, axis=1, keepdims=True)
            return dict(logdens=logdens, symbol_logq=slq,
                        logdens_dist=np.where(md, ldv, np.nan),
                        logdens_dt=np.where(mt, ltv, np.nan))
        if int(getattr(self, "state", 0)) and self._state_latent_dim():
            # RECURRENT MODEL WITH A STATIC PER-FISH LATENT -- either the GRU seed (h0_dim) or the
            # fish_re offset z. It must be integrated out, because a held-out fish has no fitted
            # embedding. Given the latent the GRU is still deterministic, so the latent is STATIC
            # over the sequence and the belief update is the same cumsum `fish_re` uses -- no
            # forward recursion needed. The Q quadrature nodes are run as a BATCH through
            # state_chain, which is why this costs about one pass rather than Q of them: the loop
            # is dispatch-bound, so the batch dimension is nearly free.
            #
            # ONE latent, never both. Two static latents would need the PRODUCT grid (2-d x 2-d =
            # 2401 nodes at k=7) and, worse, they would be fitting the same between-fish variance
            # against each other -- the dt_latent/z double-counting trap. The constructor refuses
            # the combination; this branch therefore only ever has one to integrate.
            from scipy.special import logsumexp as _lse
            net = self._model
            if int(net.h0_dim):
                nodes, log_pi = _ghq(net.h0_dim, np.ones(net.h0_dim), k=int(self.re_ghq_k))
            else:
                # the SAME grid the non-recurrent fish_re path uses, built from the fitted prior sd
                nodes, log_pi = self._re_nodes, self._re_log_pi
            Q, n = nodes.shape[0], X.shape[0]
            with torch.no_grad():
                t1 = lambda a, d=np.float32: torch.tensor(np.asarray(a, d)).reshape(1, -1).expand(Q, -1)
                u = torch.tensor(state_gru_input(ya, yd, yt, self._cfg_feat["dth_scale"],
                                                 self._cfg_feat["kin_scales"])).reshape(1, n, 3).expand(Q, -1, -1)
                Xq = torch.tensor(np.asarray(X, np.float32)).reshape(1, n, -1).expand(Q, -1, -1)
                nt = torch.tensor(np.asarray(nodes, np.float32))
                h0 = net.h0_from(nt) if int(net.h0_dim) else None
                zq = None if int(net.h0_dim) else nt
                la_q, ld_q, lt_q, _ = net.state_chain(Xq, t1(a_z), t1(ld_c), u,
                                                      t1(ya), t1(yd), t1(yt), h0=h0, z=zq)
            La = la_q.numpy().T.astype(float)                      # (n, Q)
            Ld = ld_q.numpy().T.astype(float); Lt = lt_q.numpy().T.astype(float)
            dv = np.asarray(yd, float); tv = np.asarray(yt, float)
            md = np.isfinite(dv) & (dv > 0); mt = np.isfinite(tv) & (tv > 0)
            Ld_w = np.where(md.reshape(-1, 1), Ld, 0.0)
            Lt_w = np.where(mt.reshape(-1, 1), Lt, 0.0)
            cum = np.zeros_like(La)
            if n > 1:
                cum[1:] = np.cumsum((La + Ld_w + Lt_w)[:-1], axis=0)
            lw0 = log_pi.reshape(1, -1) + cum
            d0 = _lse(lw0, axis=1)
            logdens = _lse(lw0 + La, axis=1) - d0
            lw1 = lw0 + La; d1 = _lse(lw1, axis=1)
            ldv = _lse(lw1 + Ld, axis=1) - d1
            lw2 = lw1 + Ld_w; d2 = _lse(lw2, axis=1)
            ltv = _lse(lw2 + Lt, axis=1) - d2
            # symbol track at the prior MEAN latent (zero for both parameterisations) -- an
            # approximation confined to that column; the three density channels above are
            # marginalised exactly and they are what Metric A is.
            slq = self._state_symbol_logq(X, ya, yd, yt)
            return dict(logdens=logdens, symbol_logq=slq,
                        logdens_dist=np.where(md, ldv, np.nan),
                        logdens_dt=np.where(mt, ltv, np.nan))
        if int(getattr(self, "state", 0)):
            # RECURRENT MODEL, h_0 = 0: no latent to integrate out. With the targets observed the
            # GRU is deterministic, so the chain rule IS the exact marginal predictive.
            with torch.no_grad():
                t = lambda a, d=np.float32: torch.tensor(np.asarray(a, d)).reshape(1, -1)
                u = torch.tensor(state_gru_input(ya, yd, yt, self._cfg_feat["dth_scale"],
                                                 self._cfg_feat["kin_scales"])).reshape(1, -1, 3)
                la, ld_, lt_, _ = self._model.state_chain(
                    torch.tensor(np.asarray(X, np.float32)).reshape(1, X.shape[0], -1),
                    t(a_z), t(ld_c), u, t(ya), t(yd), t(yt))
                trunk_lo = la.reshape(-1).numpy()
                ldv = ld_.reshape(-1).numpy(); ltv = lt_.reshape(-1).numpy()
            md = np.isfinite(np.asarray(yd, float)) & (np.asarray(yd, float) > 0)
            mt = np.isfinite(np.asarray(yt, float)) & (np.asarray(yt, float) > 0)
            # symbol PMF from the SAME per-bout angle params the density used
            with torch.no_grad():
                slq = self._state_symbol_logq(X, ya, yd, yt)
            return dict(logdens=trunk_lo, symbol_logq=slq,
                        logdens_dist=np.where(md, ldv, np.nan),
                        logdens_dt=np.where(mt, ltv, np.nan))
        nodes, log_pi = self._re_nodes, self._re_log_pi
        n, Q = X.shape[0], nodes.shape[0]

        Xt = torch.tensor(np.asarray(X, np.float32))
        azt = torch.tensor(a_z); lct = torch.tensor(ld_c)
        yat = torch.tensor(np.asarray(ya, np.float32))
        ydt = torch.tensor(np.asarray(yd, np.float32))
        ytt = torch.tensor(np.asarray(yt, np.float32))
        # THE GRID. Without the timing latent it is the Q z-nodes, exactly as before. With it, the
        # belief lives on the PRODUCT grid (z_q, s_j), flattened as g = q*L + j, and the transition
        # is I (x) P -- z never moves, s does. Every line below is shared; only the propagation
        # differs, which is the entire structural difference between a static latent and a state.
        DTL = getattr(self._model, "dt_latent", None)
        DRF = getattr(self._model, "drift", None)
        if DTL and DRF:
            return self._score_two_chain(X, a_z, ld_c, ya, yd, yt, nodes, log_pi)
        if DTL or DRF:
            with torch.no_grad():
                if DRF:
                    s_nodes, logP_t, log_pi0_t = ar1_chain(self._model.d_phi(),
                                                           int(DRF["n_levels"]))
                else:
                    s_nodes, logP_t, log_pi0_t = ar1_chain(self._model.s_phi(),
                                                           int(DTL["n_levels"]),
                                                           span=float(DTL["span"]))
            logP = logP_t.numpy().astype(float); log_pi0 = log_pi0_t.numpy().astype(float)
            L = int(s_nodes.shape[0])
        else:
            s_nodes, logP, log_pi0, L = None, None, np.zeros(1), 1
        # With DRIFT the state shifts ALL THREE channels, so every channel's log-density is
        # (n, Q, L) rather than only the interval's -- that is the sole structural difference here.
        La = np.empty((n, Q, L)) if DRF else np.empty((n, Q))
        Ld = np.empty((n, Q, L)) if DRF else np.empty((n, Q))
        Lt = np.empty((n, Q, L))
        W = np.empty((n, Q, 5))
        SHP = np.empty((9, n, Q))          # sigmaF, (alpha, theta) x 4, in symbol_logq_mix's order
        with torch.no_grad():
            for q in range(Q):
                zq = torch.tensor(nodes[q]).reshape(1, -1).expand(n, -1)
                if DRF:
                    out_, do_, to_ = self._model._trunk(Xt, azt, lct, zq)
                    for j in range(L):
                        ang_j, dp_j, tp_j = self._model._params_at(out_, do_, to_, s_nodes[j])
                        La[:, q, j] = MX.mix_logdens(yat, ang_j).numpy()
                        Ld[:, q, j] = MX._lnk_logpdf(torch.clamp(ydt, min=1e-6), *dp_j).numpy()
                        Lt[:, q, j] = MX._dt_logpdf(torch.clamp(ytt, min=1e-6), *tp_j,
                                                    hi=self._model.dt_trunc).numpy()
                    # The symbol track and the reported mixture shape are population quantities and
                    # are read at the drift's PRIOR MEAN (s = 0), matching joint_forward_re.
                    ang = self._model._params_at(out_, do_, to_)[0]
                else:
                    ang, dpar, tpar = self._model.joint_forward_re(Xt, azt, lct, zq)
                    La[:, q] = MX.mix_logdens(yat, ang).numpy()
                    Ld[:, q] = MX._lnk_logpdf(torch.clamp(ydt, min=1e-6), *dpar).numpy()
                    if DTL:
                        Lt[:, q, :] = self._model.dt_level_logliks(tpar, ytt, s_nodes).numpy()
                    else:
                        Lt[:, q, 0] = MX._dt_logpdf(torch.clamp(ytt, min=1e-6), *tpar,
                                                    hi=self._model.dt_trunc).numpy()
                W[:, q, :] = np.exp(ang[0].numpy())
                for j in range(9):
                    SHP[j, :, q] = ang[j + 1].numpy()

        G = Q * L
        La = (La.reshape(n, G) if DRF
              else np.broadcast_to(La[:, :, None], (n, Q, L)).reshape(n, G))
        Ld = (Ld.reshape(n, G) if DRF
              else np.broadcast_to(Ld[:, :, None], (n, Q, L)).reshape(n, G))
        Lt = Lt.reshape(n, G)
        log_pi_g = (log_pi.reshape(-1, 1) + log_pi0.reshape(1, -1)).reshape(-1)

        # Masks: exactly the bouts the training loss masks, so the reported likelihood is over the
        # support the model was fitted on (10 bouts have distance 0; the last bout of each fish has
        # no forward-looking distance or interval). A masked channel must contribute log-weight 0 to
        # the running belief -- NOT NaN, which would poison every later bout's posterior.
        dv = np.asarray(yd, float); tv = np.asarray(yt, float)
        md = np.isfinite(dv) & (dv > 0); mt = np.isfinite(tv) & (tv > 0)
        Ld_w = np.where(md.reshape(-1, 1), Ld, 0.0)
        Lt_w = np.where(mt.reshape(-1, 1), Lt, 0.0)

        # Running belief: the prior times everything seen strictly before bout t.
        Ltot = La + Ld_w + Lt_w
        if DTL or DRF:
            lw0 = np.empty((n, G))
            lw0[0] = log_pi_g
            for t in range(1, n):
                prev = (lw0[t - 1] + Ltot[t - 1]).reshape(Q, L)
                lw0[t] = logsumexp(prev[:, :, None] + logP[None, :, :], axis=1).reshape(-1)
        else:
            # z is static, so the transition is the identity and the recursion degenerates to a
            # cumsum. Kept as the literal cumsum rather than a loop with an identity matrix so the
            # archived arms stay bit-identical.
            cum = np.zeros_like(Ltot)
            if n > 1:
                cum[1:] = np.cumsum(Ltot[:-1], axis=0)
            lw0 = log_pi_g.reshape(1, -1) + cum

        d0 = logsumexp(lw0, axis=1)
        logdens = logsumexp(lw0 + La, axis=1) - d0                     # p(dtheta | past)
        lw1 = lw0 + La                                                 # ... having seen dtheta
        d1 = logsumexp(lw1, axis=1)
        ld = logsumexp(lw1 + Ld, axis=1) - d1                          # p(l | past, dtheta)
        lw2 = lw1 + Ld_w                                               # ... having seen l
        d2 = logsumexp(lw2, axis=1)
        lt = logsumexp(lw2 + Lt, axis=1) - d2                          # p(tau | past, dtheta, l)
        ld = np.where(md, ld, np.nan)
        lt = np.where(mt, lt, np.nan)

        # Symbol PMF under the SAME belief as the density beside it. Running symbol_logq_mix on the
        # z = 0 parameters, as the parent class does, would report a symbol column conditioned on a
        # different object than the density column and the two would silently disagree.
        # Under dt_latent the turn emission does not depend on s, so summing the product-grid
        # posterior over levels leaves exactly the belief about z. Under DRIFT the turn emission
        # DOES depend on s, so this reports the symbol track at the drift's prior mean (s = 0) --
        # an approximation, and confined to the symbol column: the three density channels above are
        # marginalised exactly, and they are what Metric A reports.
        pw = np.exp(lw0 - d0.reshape(-1, 1)).reshape(n, Q, L).sum(axis=2)      # (n, Q)
        slq_q = np.stack([MX.symbol_logq_mix(W[:, q, :], *[SHP[j, :, q] for j in range(9)])
                          for q in range(Q)], axis=1)                  # (n, Q, 3)
        slq = logsumexp(slq_q + np.log(np.maximum(pw, 1e-300))[:, :, None], axis=1)
        slq = slq - logsumexp(slq, axis=1, keepdims=True)
        return dict(logdens=logdens, symbol_logq=slq, logdens_dist=ld, logdens_dt=lt)

    def _score_two_chain(self, X, a_z, ld_c, ya, yd, yt, nodes, log_pi):
        """TWO-STATE design: the slow chain (turn + step) and the fast interval chain marginalised
        EXACTLY, z integrated out on the quadrature grid.

        Given z the two chains see disjoint channels, so their posteriors factorise: the belief is
        (Q, L_d) + (Q, L_s), never Q x L_d x L_s. Within a bout the reweighting follows the chained
        order, as the one-chain branches do: the turn is scored under the slow-chain posterior with
        the fast chain's evidence folded into the z-weights; the step under the slow-chain posterior
        that has also seen this bout's turn; the interval under the fast-chain posterior with the
        slow chain's evidence INCLUDING this bout's turn and step folded into the z-weights. The
        three conditionals multiply to p(dtheta, l, tau | past), so the record's sum telescopes to
        the exact joint marginal -- checked against the trainer's two-recursion loss at z = 0.
        """
        from scipy.special import logsumexp as lse
        net = self._model
        with torch.no_grad():
            nd, logPd_t, pi0d_t = ar1_chain(net.d_phi(), int(net.drift["n_levels"]))
            ns, logPs_t, pi0s_t = ar1_chain(net.s_phi(), int(net.dt_latent["n_levels"]),
                                            span=float(net.dt_latent["span"]))
        logPd, pi0d = logPd_t.numpy().astype(float), pi0d_t.numpy().astype(float)
        logPs, pi0s = logPs_t.numpy().astype(float), pi0s_t.numpy().astype(float)
        Ld, Ls = len(pi0d), len(pi0s)
        n, Q = X.shape[0], nodes.shape[0]
        Xt = torch.tensor(np.asarray(X, np.float32))
        azt = torch.tensor(a_z); lct = torch.tensor(ld_c)
        yat = torch.tensor(np.asarray(ya, np.float32))
        ydt = torch.tensor(np.asarray(yd, np.float32))
        ytt = torch.tensor(np.asarray(yt, np.float32))
        La = np.empty((n, Q, Ld)); Ldd = np.empty((n, Q, Ld)); Lt = np.empty((n, Q, Ls))
        W = np.empty((n, Q, 5)); SHP = np.empty((9, n, Q))
        with torch.no_grad():
            for q in range(Q):
                zq = torch.tensor(nodes[q]).reshape(1, -1).expand(n, -1)
                out_, do_, to_ = net._trunk(Xt, azt, lct, zq)
                for j in range(Ld):
                    ang_j, dp_j, _ = net._params_at(out_, do_, to_, nd[j])
                    La[:, q, j] = MX.mix_logdens(yat, ang_j).numpy()
                    Ldd[:, q, j] = MX._lnk_logpdf(torch.clamp(ydt, min=1e-6), *dp_j).numpy()
                ang, _, tpar = net._params_at(out_, do_, to_)     # d_dt = 0: dt sees no slow shift
                Lt[:, q, :] = net.dt_level_logliks(tpar, ytt, ns).numpy()
                W[:, q, :] = np.exp(ang[0].numpy())
                for j in range(9):
                    SHP[j, :, q] = ang[j + 1].numpy()
        dv = np.asarray(yd, float); tv = np.asarray(yt, float)
        md = np.isfinite(dv) & (dv > 0); mt = np.isfinite(tv) & (tv > 0)
        Ld_w = np.where(md[:, None, None], Ldd, 0.0)
        Lt_w = np.where(mt[:, None, None], Lt, 0.0)
        # Forward recursions per z-node: the belief over each chain's level BEFORE bout t.
        ad = np.empty((n, Q, Ld)); at = np.empty((n, Q, Ls))
        ad[0] = pi0d[None, :]; at[0] = pi0s[None, :]
        for t in range(1, n):
            ad[t] = lse((ad[t - 1] + La[t - 1] + Ld_w[t - 1])[:, :, None] + logPd[None, :, :], axis=1)
            at[t] = lse((at[t - 1] + Lt_w[t - 1])[:, :, None] + logPs[None, :, :], axis=1)
        Ct = lse(at, axis=2)                                        # (n, Q) fast-chain evidence < t
        w0 = log_pi[None, :, None] + ad + Ct[:, :, None]           # belief over (z, s_slow)
        d0 = lse(w0, axis=(1, 2))
        logdens = lse(w0 + La, axis=(1, 2)) - d0                    # p(dtheta | past)
        w1 = w0 + La
        d1 = lse(w1, axis=(1, 2))
        ldv = lse(w1 + Ldd, axis=(1, 2)) - d1                       # p(l | past, dtheta)
        Dinc = lse(ad + La + Ld_w, axis=2)                          # slow-chain evidence <= t
        v0 = log_pi[None, :, None] + at + Dinc[:, :, None]          # belief over (z, s_fast)
        e0 = lse(v0, axis=(1, 2))
        ltv = lse(v0 + Lt, axis=(1, 2)) - e0                        # p(tau | past, dtheta, l)
        # symbol track at both chains' prior mean, under the belief about z (as the drift branch)
        pw = np.exp(w0 - d0[:, None, None]).sum(axis=2)             # (n, Q)
        slq_q = np.stack([MX.symbol_logq_mix(W[:, q, :], *[SHP[j, :, q] for j in range(9)])
                          for q in range(Q)], axis=1)
        slq = lse(slq_q + np.log(np.maximum(pw, 1e-300))[:, :, None], axis=1)
        slq = slq - lse(slq, axis=1, keepdims=True)
        return dict(logdens=logdens, symbol_logq=slq,
                    logdens_dist=np.where(md, ldv, np.nan), logdens_dt=np.where(mt, ltv, np.nan))

    def _state_latent_dim(self):
        """Width of the STATIC per-fish latent the recurrent scorer has to integrate out, or 0.

        Exactly one of the two can be live (the constructor refuses both), so this is a single
        number rather than a pair. ``re_sd`` is all-zero for the pinned control, which is what makes
        gru1 fall through to the cheap deterministic branch untouched.
        """
        if int(getattr(self._model, "h0_dim", 0)):
            return int(self._model.h0_dim)
        sd = getattr(self, "re_sd", None)
        return int(self.re_m) if sd is not None and float(np.max(np.abs(sd))) > 0.0 else 0

    @torch.no_grad()
    def _state_symbol_logq(self, X, ya, yd, yt):
        """(n, 3) symbol log-PMF for the recurrent model, from the same per-bout angle params.

        Re-runs the state forward to collect the angle parameters, then hands them to the SAME
        ``MX.symbol_logq_mix`` the rest of the family uses -- so the symbol column is conditioned on
        exactly the object the density column is, rather than on a z=0 stand-in.
        """
        net = self._model
        u = torch.tensor(state_gru_input(ya, yd, yt, self._cfg_feat["dth_scale"],
                                         self._cfg_feat["kin_scales"]))
        Xt = torch.tensor(np.asarray(X, np.float32))
        h = torch.zeros((1, net.state_hidden))
        W = np.empty((len(X), 5)); SHP = np.empty((9, len(X)))
        for t in range(len(X)):
            trunk = torch.cat([net.body(Xt[t:t + 1]), h], dim=1)
            ang = net._angle_params(net.head(trunk))
            W[t] = np.exp(ang[0][0].numpy())
            for j in range(9):
                SHP[j, t] = float(ang[j + 1][0])
            h = net.state_cell(torch.cat([Xt[t:t + 1], u[t:t + 1]], dim=1), h)
        return MX.symbol_logq_mix(W, *[SHP[j] for j in range(9)])

    def sample(self, rec, rng, **kw):
        """Free-run one trajectory.

        The RECURRENT model must go through the torch rollout: the numpy generator has no way to
        carry a GRU state, so dispatching to it would silently roll out a stateless model while the
        likelihood reported a stateful one -- exactly the two-objects trap this repo keeps hitting.
        """
        if int(self._cfg_feat.get("state", 0)):
            return self.sample_batch(rec, 1, rng)[0]
        return self._sample_legacy(rec, rng, **kw)

    @property
    def batched_sample(self):
        """True when ``sample_batch`` is worth calling -- see its docstring for why it matters."""
        return bool(int(self._cfg_feat.get("state", 0)))

    def sample_batch(self, rec, n, rng, **kw):
        """``n`` independent free-runs of one record, generated in ONE batched rollout.

        WHY THIS EXISTS. A recurrent rollout is 4700 SEQUENTIAL steps, each ~30 small torch ops.
        At batch 1 that is entirely Python-dispatch bound -- measured on an A100, Metric B (which
        needs R=12 panel rollouts per held-out fish plus the C2ST replicates, ~80 rollouts for a
        2-fold run) took longer than the whole teacher-forced fit, with the GPU essentially idle.
        Batching the replicates costs almost nothing per extra trajectory because the dispatch is
        amortised over the batch dimension.

        The trajectories are independent: nothing in the rollout couples rows (feature blocks,
        heads, GRU cell and samplers are all elementwise per row), and ``advance_t``'s early break
        on ``inside.all()`` only skips iterations that are masked no-ops for the rows already
        finished. Batching therefore changes WHICH variates each trajectory draws -- a different
        realisation of the same law -- not the law itself.
        """
        if not self.batched_sample:
            # legacy path unchanged: successive draws off the SAME rng, exactly as the callers did
            return [self._sample_legacy(rec, rng, **kw) for _ in range(int(n))]
        self._model.to(torch.device("cpu")).eval()
        return state_rollout_batch(self._model, rec, self._cfg_feat, int(n), rng)

    def _sample_legacy(self, rec, rng, **kw):
        # **kw forwards the tool-mode knobs (stim_provider / stop_time_s); see JT.sample.
        cfg = self._cfg_feat
        ztw = getattr(self, "z_twin", None)
        if ztw and getattr(rec, "fish_id", None) is not None and int(rec.fish_id) in ztw:
            # TWIN MODE: an in-sample checkpoint (trained on this fish) rolls the fish out under
            # its OWN fitted embedding instead of a prior draw. Only `z_twin` checkpoints written
            # by the all-data builder carry the table; a fold checkpoint never has it, so the
            # held-out "new fish" claim of every ladder number is untouched. The cfg copy keeps
            # the checkpoint's cfg (and the calibrators, which read it) on prior draws.
            cfg = dict(cfg, fish_re=dict(cfg["fish_re"], z_fixed=np.asarray(ztw[int(rec.fish_id)], float)))
        return JT.generate_mix_joint_loop(self._model, rec, cfg, rng,
                                          closed_loop=self.closed_loop, guardrail=self.guardrail,
                                          eps=self.eps, stim_loop=self.stim_loop, **kw)


# =========================================================================== #
#  Serialisation
# =========================================================================== #
def save_fishre(model, path):
    """Persist a fitted random-effect model. Must carry n_fish, m and the fitted Sigma: without
    them ``load`` would rebuild the net with a RANDOM embedding and every downstream diagnostic
    would silently score a different model."""
    if not hasattr(model, "_model"):
        raise RuntimeError("model is not fitted; call .fit(...) before save_fishre")
    torch.save(dict(cls="MLPMixJointREDensity", state_dict=model._model.state_dict(),
                    n_features=int(model.config["n_features"]), hidden=int(model.config["hidden"]),
                    cfg_feat=model._cfg_feat, config=model.config, name=model.name,
                    dt_pool=np.asarray(model._dt_pool), dist_pool=np.asarray(model._dist_pool),
                    history=getattr(model, "history", None),
                    K=model.K, K_kin=model.K_kin, stim_mode=model.stim_mode,
                    include_symbols=model.include_symbols, use_border=model.use_border,
                    border_mode=model.border_mode, wall_mode=model._cfg_feat.get("wall_mode", "clip"),
                    n_dist=model.n_dist, n_dt=model.n_dt,
                    vigor=model.vigor, vigor_drift_calib=model.vigor_drift_calib,
                    kin_fb_calib=model.kin_fb_calib,
                    indiv=model.indiv, innov_comp=model.innov_comp,
                    re_m=int(model.re_m), re_sd=np.asarray(model.re_sd, float),
                    re_zero=bool(model.re_zero), re_shuffle=bool(model.re_shuffle),
                    re_body=bool(model.re_body), contain_graded=bool(model.contain_graded),
                    contain_decouple=bool(getattr(model, "contain_decouple", False)),
                    rad_indiv=bool(getattr(model, "rad_indiv", False)),
                    re_bgain=bool(getattr(model, "re_bgain", False)),
                    re_border_slice=(tuple(model._model.border_slice)
                                     if getattr(model._model, "border_gain", False) else None),
                    re_ghq_k=int(model.re_ghq_k), re_gain=model.re_gain,
                    re_gain_grid=str(getattr(model, "re_gain_grid", "std")),
                    re_gain_mode=str(getattr(model, "re_gain_mode", "global")),
                    re_block=bool(getattr(model, "re_block", False)),
                    re_disp=bool(getattr(model, "re_disp", False)),
                    re_lag=bool(getattr(model, "re_lag", False)),
                    re_tail=bool(getattr(model, "re_tail", False)),
                    re_tsize=bool(getattr(model, "re_tsize", False)),
                    turn_skip=bool(getattr(model, "turn_skip", False)),
                    skip_slices=(list(model._model.skip_slices) if getattr(model._model, "skip_slices", None) else None),
                    depth=int(getattr(model._model, "depth", 1)),
                    tail_gain=getattr(model, "tail_gain", None),
                    tail_gain_report=getattr(model, "tail_gain_report", None),
                    re_lag_slice=(tuple(model._model.lag_slice)
                                  if getattr(model._model, "lag_slice", None) else None),
                    spatial_gain=getattr(model, "spatial_gain", None),
                    spatial_gain_report=getattr(model, "spatial_gain_report", None),
                    re_gain_report=getattr(model, "re_gain_report", None),
                    dt_latent=getattr(model, "dt_latent", None),
                    drift=getattr(model, "drift", None),
                    state=int(getattr(model, "state", 0)),
                    h0_dim=int(getattr(model, "h0_dim", 0)),
                    drift_report=getattr(model, "drift_report", None),
                    ss=getattr(model, "ss", None),            # training-only; for the record
                    regime_w=getattr(model, "regime_w", None),   # training-only; for the record
                    K_stim=int(getattr(model, "K_stim", 1)),
                    noise_aug=getattr(model, "noise_aug", None),
                    contain_mode=str(getattr(model, "contain_mode", "bias")),
                    n_fish=int(model.config["n_fish"]),
                    z_twin=getattr(model, "z_twin", None),   # in-sample twin table, else None
                    stim_turn=(model._model.stim_turn.spec()
                               if getattr(model._model, "stim_turn", None) is not None else None),
                    loop=dict(closed_loop=model.closed_loop, guardrail=model.guardrail,
                              eps=model.eps, stim_loop=model.stim_loop)), path)
    return path


def load_fishre(path, map_location="cpu"):
    """Reload a model saved by ``save_fishre``, ready to ``.sample`` / ``.score``."""
    MX._numpy2_unpickle_compat()
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    lp = blob["loop"]
    m = MLPMixJointREDensity(K=blob["K"], K_kin=blob["K_kin"], stim_mode=blob["stim_mode"],
                             include_symbols=blob["include_symbols"], hidden=blob["hidden"],
                             use_border=blob["use_border"], border_mode=blob["border_mode"],
                             wall_mode=blob.get("wall_mode", "clip"),
                             closed_loop=lp["closed_loop"], guardrail=lp["guardrail"],
                             eps=lp["eps"], stim_loop=lp["stim_loop"],
                             n_dist=blob["n_dist"], n_dt=blob["n_dt"],
                             re_m=blob["re_m"], re_zero=blob.get("re_zero", False),
                             re_shuffle=blob.get("re_shuffle", False),
                             re_body=blob.get("re_body", True),   # pre-flag ckpts HAD the body
                             contain_graded=blob.get("contain_graded", False),
                             contain_decouple=blob.get("contain_decouple", False),
                             rad_indiv=blob.get("rad_indiv", False),
                             re_bgain=blob.get("re_bgain", False),
                             re_gain_grid=blob.get("re_gain_grid", "std"),
                             re_gain_mode=blob.get("re_gain_mode", "global"),
                             re_block=blob.get("re_block", False),
                             dt_latent=blob.get("dt_latent"),
                             drift=blob.get("drift"),
                             state=int(blob.get("state", 0)),
                             h0_dim=int(blob.get("h0_dim", 0)),
                             ss=blob.get("ss"),
                             re_disp=blob.get("re_disp", False), re_lag=blob.get("re_lag", False),
                             re_tail=blob.get("re_tail", False), re_tsize=blob.get("re_tsize", False),
                             turn_skip=blob.get("turn_skip", False), depth=int(blob.get("depth", 1)),
                             re_ghq_k=blob.get("re_ghq_k", 7))
    net = MDNMixJointRE(blob["n_features"], blob["hidden"], n_dist=blob["n_dist"],
                        n_dt=blob["n_dt"], n_fish=blob["n_fish"], m=blob["re_m"],
                        use_body=blob.get("re_body", True),
                        border_gain=blob.get("re_bgain", False),
                        border_slice=blob.get("re_border_slice"),
                        dt_latent=blob.get("dt_latent"),
                        block=blob.get("re_block", False),
                        drift=blob.get("drift"),
                        state=int(blob.get("state", 0)),
                        h0_dim=int(blob.get("h0_dim", 0)),
                        disp=blob.get("re_disp", False), lag_slice=blob.get("re_lag_slice"),
                        tail=blob.get("re_tail", False), skip_slices=blob.get("skip_slices"),
                        depth=int(blob.get("depth", 1)), tsize=blob.get("re_tsize", False))
    if blob.get("stim_turn") is not None:          # profile fine-tune: rebuild the term before loading
        net.stim_turn = JT.stim_turn_from_spec(blob["stim_turn"])
    net.load_state_dict(blob["state_dict"]); net.eval()
    m.tail_gain = blob.get("tail_gain"); m.tail_gain_report = blob.get("tail_gain_report")
    net.dt_trunc = blob["cfg_feat"].get("dt_trunc")
    m.dt_trunc = net.dt_trunc
    m._model = net
    m._cfg_feat = blob["cfg_feat"]
    m._rich_scales = blob["cfg_feat"].get("rich_scales")
    m._dt_pool = blob["dt_pool"]; m._dist_pool = blob["dist_pool"]
    m.config = blob["config"]; m.name = blob["name"]; m.history = blob.get("history")
    m._dth_scale = blob["cfg_feat"]["dth_scale"]; m._kin_scales = blob["cfg_feat"]["kin_scales"]
    # Restore the treatment flags. Generation reads _cfg_feat, not these, so a rollout is
    # correct either way -- but a reloaded checkpoint that reports contain=False while its
    # cfg_feat carries a fitted centre_bias is a trap for anything that inspects the object.
    for _f in ("vigor", "vigor_drift_calib", "kin_fb_calib", "indiv", "innov_comp"):
        if _f in blob:
            setattr(m, _f, bool(blob[_f]))
    m.contain = m._cfg_feat.get("centre_bias") is not None   # not stored; infer from the cfg
    # cfg-carried flags added 2026-08: a refit of the loaded object rebuilds _cfg_feat from
    # these attributes, so they must round-trip or a psit/dt_snap recipe silently reverts.
    m.dt_snap = bool(m._cfg_feat.get("dt_snap", False))
    m.psi_ref = str(m._cfg_feat.get("psi_ref", "ori"))
    m.phi_ref = str(m._cfg_feat.get("phi_ref", "ori"))
    # Same fallback order as load_joint: cfg_feat is what generation reads, the top-level keys
    # are the compatibility path for checkpoints written before these treatments existed.
    m.K_stim = int(m._cfg_feat.get("K_stim", blob.get("K_stim", 1)))
    m.noise_aug = m._cfg_feat.get("noise_aug", blob.get("noise_aug"))
    m.contain_mode = str(blob.get("contain_mode",
                                  "reject" if m._cfg_feat.get("wall_reject") else "bias"))
    m.re_sd = np.asarray(blob["re_sd"], float)
    # The gain is already folded into the saved loadings; kept for reporting only. Re-applying it
    # here would square it.
    m.re_gain = blob.get("re_gain")
    m.re_gain_report = blob.get("re_gain_report")
    # Round-trip the active-only extras so a REFIT of a loaded object rebuilds the same features
    # (generation reads _cfg_feat and is correct either way).
    m.cfg_feat_extra = {k: m._cfg_feat[k] for k in ("clock_mode", "clock_tau_s", "clock2_tau_s",
                                                    "lag_valid", "wall_reject", "kin_dt_lags",
                                                    "run_clock_tau", "stim_align", "stim_static",
                                                    "stim_static_clock", "stim_static_bump", "stim_onset",
                                                    "stim_lags", "stim_jump", "stim_dmove")
                        if k in m._cfg_feat}
    # cfg_feat is what generation reads and already carries the fitted (lam, phi); this keeps the
    # attribute in step so a refit of a loaded checkpoint rebuilds the same architecture.
    m.dt_latent = m._cfg_feat.get("dt_latent", blob.get("dt_latent"))
    # cfg_feat is what the generator reads, so it wins; the blob key is the fallback. d_bord lives
    # only in cfg_feat (it is fitted by the rollout search, not by the likelihood), so it has to be
    # copied back onto the net or a reloaded checkpoint would roll out with the drift's spatial
    # channel silently switched off.
    m.state = int(m._cfg_feat.get("state", blob.get("state", 0)))
    m.h0_dim = int(m._cfg_feat.get("h0_dim", blob.get("h0_dim", 0)))
    m.drift = m._cfg_feat.get("drift", blob.get("drift"))
    m.drift_report = blob.get("drift_report")
    m.regime_w = blob.get("regime_w")
    if m.drift is not None and getattr(net, "drift", None) is not None:
        net.d_bord = float(m.drift.get("bord", 0.0))
    m.spatial_gain = blob.get("spatial_gain")
    m.spatial_gain_report = blob.get("spatial_gain_report")
    # twin table (fish_id -> fitted z_i) of an in-sample checkpoint; None on every fold checkpoint
    m.z_twin = ({int(k): [float(v) for v in np.asarray(z).reshape(-1)] for k, z in blob["z_twin"].items()}
                if blob.get("z_twin") else None)
    m._re_nodes, m._re_log_pi = _ghq(m.re_m, m.re_sd, k=m.re_ghq_k)
    return m
