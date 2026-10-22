"""The per-fish latent and the slow drift: individuality and slow state as fitted parts of the likelihood.

THE LATENT. z_i in R^m (m = 2), prior N(0, I), enters as a bias offset on the three heads and nowhere else (the
body does not see z): the forward-mixture weight logit of the turn head, and the locations of the step and the
interval (``turn`` / ``dist`` / ``dt`` loadings; with ``re_tsize`` also the four turning components' scales).
Level-only on purpose: real fish differ mostly in level, and a body path let the latent reshape every conditional.

ESTIMATION.
  A. FIT. z_i are embedding rows trained jointly with the net, with the N(0, I) prior penalty in every batch. The
     prior is FIXED and the scale lives in the loadings (with both free only their product is identified).
     Both factors start nonzero: zero loadings x zero embedding is an exact saddle.
  B. DE-SHRINK. Joint MAP shrinks the between-fish spread, so one global gain on the loadings is chosen by rollout
     to match the real between-fish spread of three per-fish levels (``search_loading_gain``; the marginal
     likelihood is monotone in the gain and cannot set it), and folded into the weights with a log-normal
     re-centring (``apply_loading_gain``).
  C. SCORE. A held-out fish has no embedding: ``score`` reports the exact marginal predictive
     log p(y_t | y_<t, x), z integrated out by Gauss-Hermite quadrature.

THE DRIFT (``drift``). An AR(1) state s_t, timescale pinned (tau = 70 bouts by default), loading on the same three
quantities; trained on contiguous per-fish blocks with s marginalised by a forward recursion over a discretised
chain, scored on the product grid (z, s). Its generation gains are searched by rollout against the real
across-window autocorrelations (``emission._search_drift_gain``).

GENERATION. z is drawn once per trajectory from the prior (a new fish), the drift advanced once per bout; both in
``joint.generate_mix_joint_loop``, keyed off ``cfg_feat['fish_re']`` and ``cfg_feat['drift']``.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from twin import stimulus as M                     # noqa: E402
from twin import features as MB             # noqa: E402
from twin import joint as JT              # noqa: E402
from twin import emission as MX                # noqa: E402


# =========================================================================== #
#  Net
# =========================================================================== #
class MDNMixJointRE(JT.MDNMixJoint):
    """``MDNMixJoint`` plus a per-fish latent loading level-only on all three heads.

    ``joint_forward`` is inherited and runs at z = 0, so every existing consumer (the numpy
    predictor, ``predict_params_mix``, the symbol track) keeps working on the population-average
    fish. The random effect is reached through ``joint_forward_re``.
    """

    def __init__(self, n_features, hidden, n_dist=2, n_dt=3, n_fish=1, m=2, sigma_init=0.5, drift=None,
                 skip_slices=None, depth=1, tsize=False):
        super().__init__(n_features, hidden, n_dist=n_dist, n_dt=n_dt, skip_slices=skip_slices,
                         depth=depth)
        # DRIFT: one AR(1) state loading on ALL THREE heads (see DRIFT_DEFAULT for the measurement
        # that motivates it). Parameters exist only when enabled.
        self.drift = dict(DRIFT_DEFAULT, **drift) if drift else None
        if self.drift:
            # Same saddle argument as the fishre loadings: start SMALL BUT NONZERO. At lam = 0 every
            # level gives an identical likelihood, the posterior never leaves the prior, and the
            # state is silently inert for the whole fit.
            li = float(self.drift["lam_init"])
            self.d_turn = nn.Parameter(torch.tensor(li))
            self.d_dist = nn.Parameter(torch.tensor(li))
            self.d_dt = nn.Parameter(torch.tensor(li))
            # d_bord (the drift's loading on a containment bias, a generation-time term) is 0 in every model here
            self.d_bord = float(self.drift.get("bord", 0.0))
        self.m = int(m)
        self.n_fish = int(n_fish)
        self.emb = nn.Embedding(self.n_fish, self.m)
        self.z_body = nn.Linear(self.m, hidden, bias=False)
        self.z_turn = nn.Linear(self.m, 1, bias=False)
        self.z_dist = nn.Linear(self.m, 1, bias=False)
        self.z_dt = nn.Linear(self.m, 1, bias=False)
        # BOTH factors must start nonzero. z enters only as (loading x embedding), so zeroing both
        # -- the tempting "start exactly at the population model" init -- is an exact saddle: the
        # gradient w.r.t. the embedding is proportional to the loadings and vice versa, so both stay
        # at zero forever and the mechanism is silently inert for the whole fit. The embedding is
        # drawn from the prior, which is what an empirical-Bayes initialisation should be; the
        # loadings get a small random init so the function still starts within noise of the
        # population model.
        nn.init.normal_(self.emb.weight, std=float(sigma_init))
        for lin in (self.z_turn, self.z_dist, self.z_dt):
            nn.init.normal_(lin.weight, std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)
        # TURN SIZE (re_tsize, Model 38): z also loads, through ONE shared loading, on the pre-softplus SCALES of the
        # four turning Gamma components (turn-head outputs 7, 9, 11, 13), so how far a fish turns when it turns is a
        # trait alongside how often it swims straight (the forward logit, z_turn). Symmetric left/right and linear
        # in z, so its mean over N(0, I) is zero: no re-centring. Applied identically in the numpy generator.
        self.tsize = bool(tsize)
        if self.tsize:
            # the global RNG is restored afterwards, so every OTHER weight starts exactly as in the same model without
            # the loading (a paired comparison, not one confounded by an initialisation redraw)
            _rng_state = torch.get_rng_state()
            self.z_tsize = nn.Linear(self.m, 1, bias=False)
            nn.init.normal_(self.z_tsize.weight, std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)
            torch.set_rng_state(_rng_state)
        # z_body stays EXACTLY zero, so it receives no gradient and the latent is level-only (see
        # joint_forward_re); kept because the checkpoints carry it.
        nn.init.zeros_(self.z_body.weight)

    def _z_core(self, z):
        """The dimensions that feed turn / distance / interval (all of them)."""
        return z

    def _dt_with_z(self, to, zc, x=None):
        """The interval head's raw outputs with the latent's location shift (z_dt)."""
        kt = self.n_dt
        lg = to[:, :kt]
        mu = to[:, kt:2 * kt] + self.z_dt(zc)
        sig = to[:, 2 * kt:3 * kt]
        return torch.cat([lg, mu, sig], dim=1)

    def d_phi(self):
        """Drift persistence -- a CONSTANT, pinned from ``tau``, never a parameter.

        There is deliberately no ``d_phi_raw``: the timescale cannot drift, cannot be restored into
        a free state by a later refit, and cannot trade against the per-fish latent the way
        ``dt_latent``'s fitted phi did (it settled at tau = 107-152 against a measured ~22). See
        DRIFT_DEFAULT for the measurement that sets tau.
        """
        return torch.tensor(drift_phi(self.drift["tau"]), dtype=torch.float32,
                            device=self.d_turn.device)

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
        b = self.body(x)
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
# (joint.py:363, before the bout loop at :470), so it is a CONSTANT offset on the three
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
                     device=None, n_dist=2, n_dt=3, weight_decay=1e-5, sigma_init=0.5, drift=None,
                     weights=None, dt_trunc=None, weight_heads=None, weights_per_head=None, depth=1,
                     tsize=False):
    """Stage A. Joint MAP fit of the net and the per-fish latents under a FIXED N(0, I) prior.

    ``weight_heads`` (Model 16): the heads ``weights`` applies to, e.g. ``("turn",)``; the other
    heads keep weight 1. None = all three (Model 15's form). Each head's weights are normalised
    by their mean over the training rows, so the emphasis is WITHIN a channel (the band against
    the rest) and the channels' relative scale is untouched.

    ``weights`` = per-row training weights (MB.regime_weights): a weighted MEAN per channel, the
    same form as JT.fit_mix_joint, so the loss keeps its scale and ``weights=None`` is bit-identical.
    Early stopping is on the WEIGHTED validation loss (the objective being fitted); Metric A is
    scored unweighted afterwards. With the drift the loss is the sequence likelihood of contiguous per-fish
    blocks, the drift marginalised by the forward recursion (``nll_seq_drift``).

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
    fi = torch.tensor(fish_idx, device=dev)
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
    net = MDNMixJointRE(Xt.shape[1], hidden, n_dist=n_dist, n_dt=n_dt,
                        n_fish=int(n_fish), m=int(m), sigma_init=sigma_init, drift=drift,
                        depth=depth, tsize=tsize).to(dev)
    net.dt_trunc = (float(dt_trunc) if dt_trunc else None)
    # No weight decay on the embedding: it already has an explicit N(0, I) prior in the loss, and
    # decaying it as well would apply that prior twice at an unstated strength.
    emb_params = list(net.emb.parameters())
    emb_ids = {id(p) for p in emb_params}
    opt = torch.optim.Adam(
        [dict(params=[p for p in net.parameters() if id(p) not in emb_ids],
              weight_decay=weight_decay),
         dict(params=emb_params, weight_decay=0.0)], lr=lr)

    n = Xt.shape[0]

    def _z_for(ix):
        return net.emb(fi[ix])

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
        return -ll + (0.5 * net.emb.weight ** 2).sum() / float(n)

    def nll(ix):
        return _nll_arrays(ix, Xt[ix], az[ix], lc[ix], ya[ix], yd[ix], yt[ix])

    def nll_seq_drift(blocks):
        """Loss on B CONTIGUOUS blocks of T bouts with the DRIFT marginalised exactly across ALL THREE
        channels: the drift shifts all three, so the forward recursion runs on their SUM. ``drift_level_logliks`` already folds the kinematic masks
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
        return -ll + (0.5 * net.emb.weight ** 2).sum() / float(n)

    seq_loss = nll_seq_drift
    seq_on = bool(drift)

    # Contiguous per-fish validation blocks (see docstring).
    idx_all = np.arange(n)
    tr_list, va_list = [], []
    for f in range(int(n_fish)):
        rows = idx_all[np.asarray(fish_idx) == f]
        cut = max(int(len(rows) * (1.0 - val_frac)), 1)
        tr_list.append(rows[:cut]); va_list.append(rows[cut:])
    tr = torch.tensor(np.concatenate(tr_list), dtype=torch.long, device=dev)
    va = torch.tensor(np.concatenate([v for v in va_list if len(v)]), dtype=torch.long, device=dev)

    # ---- contiguous-chunk sampler, used only with the drift ---------------------------------- #
    # The forward recursion needs ORDERED rows, so batches become contiguous windows instead of
    # shuffled rows: one block of ``chunk`` bouts per step, as many steps as rows / chunk, so the
    # optimiser budget matches the per-row loss. The windows are redrawn (random fish, random
    # offset) every step.
    _rng_ck = np.random.default_rng(seed + 991)
    _tr_by_fish = [r for r in tr_list if len(r) >= 2]
    _T = 0
    if seq_on:
        # ONE chunk length for every block, clamped by the SHORTEST fish so the blocks stack.
        _T = int(drift.get("chunk", DRIFT_DEFAULT["chunk"]))
        _T = max(2, min(_T, min(len(r) for r in _tr_by_fish)))

    def _sample_blocks(nb):
        out = []
        for _ in range(nb):
            rows = _tr_by_fish[_rng_ck.integers(len(_tr_by_fish))]
            o = int(_rng_ck.integers(0, len(rows) - _T + 1))
            out.append(rows[o:o + _T])
        return torch.tensor(np.stack(out), dtype=torch.long, device=dev)

    best, best_state, bad, hist = np.inf, None, 0, []
    _gstats = {}
    for _ep in range(epochs):
        net.train()
        if seq_on:
            nb = 1
            steps = max(1, int(np.ceil(len(tr) / max(_T * nb, 1))))
            for _s in range(steps):
                opt.zero_grad()
                JT.guarded_step(opt, seq_loss(_sample_blocks(nb)), net.parameters(), stats=_gstats)
            net.eval()
            with torch.no_grad():
                # Validation blocks are ALREADY contiguous per fish (that is why they were built
                # that way), so each is a valid sequence for the recursion.
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
        tr_ep = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(tr_ep), batch_size):
            ix = tr_ep[i:i + batch_size]
            opt.zero_grad(); loss = nll(ix)
            JT.guarded_step(opt, loss, net.parameters(), stats=_gstats)
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


def apply_drift_gains(m, gd, gt, report):
    """Publish the rollout-fitted drift gains into the density's cfg (what the generator reads)
    with the log-normal re-centring, and print the outcome. Shared by ``fit`` and the local
    and the drift's fixed-strength units (twin/correction.py), so a re-searched checkpoint carries exactly what a
    freshly fitted one would."""
    cf = m._cfg_feat
    _dd = float(cf["drift"].get("dist", 0.0))
    _dt = float(cf["drift"].get("dt", 0.0))
    cf["drift"].update(g_dist=gd, g_dt=gt,
                       dist_c=0.5 * (gd * gd - 1.0) * _dd * _dd,
                       dt_c=0.5 * (gt * gt - 1.0) * _dt * _dt)
    m.drift.update(g_dist=gd, g_dt=gt)
    m.drift_report = report
    got, tgt = report["got"], report["target"]
    pk = report.get("pick", {})
    print(f"    [{m.name}] drift gains dist {gd:.2f} ({pk.get('dist', '?')}) "
          f"dt {gt:.2f} ({pk.get('dt', '?')}): "
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
    gen_fn = MX._gen_for(cfg_feat)
    curve = []
    for g in grid:
        with torch.no_grad():
            for n_, p in net.named_parameters():
                if n_ in hold:
                    p.zero_()
                elif n_ in base:
                    p.copy_(base[n_] * float(g))
        stats = []
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

    def __init__(self, *args, re_m=2, re_sigma_init=0.5, re_gain_search=True, re_ghq_k=7, re_gain_grid="std",
                 drift=None, drift_gain_search=True, re_gain_mode="global", depth=1, re_tsize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.re_tsize = bool(re_tsize)          # Model 38: z on the turning components' scales (turn size)
        self.depth = int(depth)                 # Model 17: a deeper body
        self.re_gain_grid = str(re_gain_grid)
        # LOADING GAIN: one scalar over the three loadings, matched by rollout to the real between-fish spread
        # (``search_loading_gain``).
        if str(re_gain_mode) != "global":
            raise ValueError(f"re_gain_mode must be 'global', got {re_gain_mode!r}")
        self.re_gain_mode = "global"
        self.re_m = int(re_m)
        self.re_sigma_init = float(re_sigma_init)
        self.re_gain_search = bool(re_gain_search)
        self.re_ghq_k = int(re_ghq_k)
        # DRIFT: a within-trajectory AR(1) state loading on the three heads; its generation gains are searched by
        # rollout (``MX._search_drift_gain``). See DRIFT_DEFAULT.
        self.drift = (dict(DRIFT_DEFAULT) if drift is True
                      else (dict(DRIFT_DEFAULT, **drift) if drift else None))
        self.drift_gain_search = bool(drift_gain_search)
        self.drift_report = None
        self.re_sd = None
        self.re_gain = None
        self.re_gain_report = None

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

        per_fish = [MB.build_feature_matrix(r, self._cfg_feat) for r in train_records]
        X = np.concatenate(per_fish, 0)
        fish_idx = np.concatenate([np.full(len(m_), i, dtype=np.int64)
                                   for i, m_ in enumerate(per_fish)])
        y_ang = np.concatenate([r.dtheta for r in train_records])
        y_dist = np.concatenate([np.asarray(r.distance_px, float) for r in train_records])
        y_dt = np.concatenate([np.asarray(r.dt_s, float) for r in train_records])

        # ---- Stage A --------------------------------------------------------------------- #
        self._model, self.history = fit_mix_joint_re(
            X, y_ang, y_dist, y_dt, fish_idx, len(train_records), self.hidden,
            self._dth_scale, self._kin_scales, m=self.re_m,
            epochs=self.epochs, lr=self.lr, patience=self.patience,
            batch_size=self.batch_size, seed=self.seed, n_dist=self.n_dist, n_dt=self.n_dt,
            sigma_init=self.re_sigma_init, drift=self.drift,
            weights=weights, dt_trunc=getattr(self, "dt_trunc", None),
            weight_heads=(self.regime_w or {}).get("heads"), weights_per_head=weights_per_head,
            depth=self.depth, tsize=self.re_tsize)
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
        # ---- Stage B --------------------------------------------------------------------- #
        # The prior is N(0, I) throughout; `re_sd` exists only so the generator and the quadrature
        # grid read the same object.
        self.re_sd = np.ones(self.re_m)
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
        if self.re_gain_search:
            # max_fish=8, matching _search_kin_feedback's precedent: this runs inside every fold's
            # fit, so it is charged 5 grid points x max_fish x n_draws rollouts every time. 8 fish
            # x 2 draws = 16 trajectories is what the smoke suite showed gives a clean interior
            # minimum; 12 fish was measurably no better and cost 50% more wall-clock.
            _grid = gain_grid_for(self.re_gain_grid)
            _gen_kw = dict(closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                           stim_loop=self.stim_loop)
            self.re_gain, self.re_gain_report = search_loading_gain(
                self._model, train_records, self._cfg_feat, grid=_grid, seed=self.seed,
                max_fish=min(self.contain_fish, 8), **_gen_kw)
            apply_loading_gain(self._model, self.re_gain, which="all")
            print(f"    [{self.name}] loading gain {self.re_gain} (grid {self.re_gain_grid}, {len(_grid)} pts)")
            if self.re_gain in (_grid[0], _grid[-1]):
                print(f"    WARNING [{self.name}]: loading gain pinned at the grid edge "
                      f"({self.re_gain}); the between-fish spread could not be matched, so widen "
                      f"the grid (re_gain_grid='wide') before trusting this arm")
        self._re_nodes, self._re_log_pi = _ghq(self.re_m, self.re_sd, k=self.re_ghq_k)

        # ---- the drift's generation gains, searched by rollout ---------------------------- #
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
        self.config = dict(mode="joint_re", hidden=self.hidden, n_features=X.shape[1], K=self.K,
                           K_kin=self.K_kin, stim_mode=self.stim_mode, use_border=self.use_border,
                           use_kin=True, n_dist=self.n_dist, n_dt=self.n_dt,
                           re_m=self.re_m, n_fish=len(train_records),
                           head_order=JT.order_tag(JT.DEFAULT_ORDER))
        return self

    def score(self, rec):
        """The marginal predictive (``_score``)."""
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
        nodes, log_pi = self._re_nodes, self._re_log_pi
        n, Q = X.shape[0], nodes.shape[0]

        Xt = torch.tensor(np.asarray(X, np.float32))
        azt = torch.tensor(a_z); lct = torch.tensor(ld_c)
        yat = torch.tensor(np.asarray(ya, np.float32))
        ydt = torch.tensor(np.asarray(yd, np.float32))
        ytt = torch.tensor(np.asarray(yt, np.float32))
        # THE GRID. Without the drift it is the Q z-nodes. With it, the belief lives on the PRODUCT grid (z_q, s_j),
        # flattened as g = q*L + j, and the transition is I (x) P -- z never moves, s does. Every line below is
        # shared; only the propagation differs.
        DRF = getattr(self._model, "drift", None)
        if DRF:
            with torch.no_grad():
                s_nodes, logP_t, log_pi0_t = ar1_chain(self._model.d_phi(), int(DRF["n_levels"]))
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
        if DRF:
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
        # Under the DRIFT the turn emission depends on s, so this reports the symbol track at the drift's prior mean
        # (s = 0) --
        # an approximation, and confined to the symbol column: the three density channels above are
        # marginalised exactly, and they are what Metric A reports.
        pw = np.exp(lw0 - d0.reshape(-1, 1)).reshape(n, Q, L).sum(axis=2)      # (n, Q)
        slq_q = np.stack([MX.symbol_logq_mix(W[:, q, :], *[SHP[j, :, q] for j in range(9)])
                          for q in range(Q)], axis=1)                  # (n, Q, 3)
        slq = logsumexp(slq_q + np.log(np.maximum(pw, 1e-300))[:, :, None], axis=1)
        slq = slq - logsumexp(slq, axis=1, keepdims=True)
        return dict(logdens=logdens, symbol_logq=slq, logdens_dist=ld, logdens_dt=lt)

    def sample(self, rec, rng, **kw):
        """Free-run one trajectory (``JT.generate_mix_joint_loop``)."""
        return self._sample_legacy(rec, rng, **kw)

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
                    re_m=int(model.re_m), re_sd=np.asarray(model.re_sd, float), re_body=False,
                    re_ghq_k=int(model.re_ghq_k), re_gain=model.re_gain,
                    re_gain_grid=str(getattr(model, "re_gain_grid", "std")), re_gain_mode="global",
                    re_tsize=bool(getattr(model, "re_tsize", False)),
                    skip_slices=(list(model._model.skip_slices) if getattr(model._model, "skip_slices", None) else None),
                    depth=int(getattr(model._model, "depth", 1)),
                    re_gain_report=getattr(model, "re_gain_report", None),
                    drift=getattr(model, "drift", None),
                    drift_report=getattr(model, "drift_report", None),
                    regime_w=getattr(model, "regime_w", None),   # training-only; for the record
                    K_stim=int(getattr(model, "K_stim", 1)),
                    n_fish=int(model.config["n_fish"]),
                    z_twin=getattr(model, "z_twin", None),   # in-sample twin table, else None
                    stim_turn=(model._model.stim_turn.spec()
                               if getattr(model._model, "stim_turn", None) is not None else None),
                    loop=dict(closed_loop=model.closed_loop, guardrail=model.guardrail,
                              eps=model.eps, stim_loop=model.stim_loop)), path)
    return path


# options of earlier models that this code does not have: a checkpoint that uses one is refused
_REMOVED = ("re_zero", "re_shuffle", "contain_graded", "contain_decouple", "rad_indiv", "re_bgain", "re_block",
            "re_disp", "re_lag", "re_tail", "turn_skip", "dt_latent", "ss", "noise_aug", "state", "h0_dim")


def load_fishre(path, map_location="cpu"):
    """Reload a model saved by ``save_fishre``, ready to ``.sample`` / ``.score``."""
    MX._numpy2_unpickle_compat()
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    off = [k for k in _REMOVED if blob.get(k)] + (["re_body"] if blob.get("re_body", True) else [])
    if off or str(blob.get("re_gain_mode", "global")) != "global":
        raise ValueError(f"{path}: the checkpoint uses options this code does not have: {off}")
    lp = blob["loop"]
    m = MLPMixJointREDensity(K=blob["K"], K_kin=blob["K_kin"], stim_mode=blob["stim_mode"],
                             include_symbols=blob["include_symbols"], hidden=blob["hidden"],
                             use_border=blob["use_border"], border_mode=blob["border_mode"],
                             wall_mode=blob.get("wall_mode", "clip"),
                             closed_loop=lp["closed_loop"], guardrail=lp["guardrail"],
                             eps=lp["eps"], stim_loop=lp["stim_loop"],
                             n_dist=blob["n_dist"], n_dt=blob["n_dt"],
                             re_m=blob["re_m"], re_gain_grid=blob.get("re_gain_grid", "std"),
                             drift=blob.get("drift"), re_tsize=blob.get("re_tsize", False),
                             depth=int(blob.get("depth", 1)), re_ghq_k=blob.get("re_ghq_k", 7))
    net = MDNMixJointRE(blob["n_features"], blob["hidden"], n_dist=blob["n_dist"],
                        n_dt=blob["n_dt"], n_fish=blob["n_fish"], m=blob["re_m"],
                        drift=blob.get("drift"), skip_slices=blob.get("skip_slices"),
                        depth=int(blob.get("depth", 1)), tsize=blob.get("re_tsize", False))
    if blob.get("stim_turn") is not None:          # profile fine-tune: rebuild the term before loading
        net.stim_turn = JT.stim_turn_from_spec(blob["stim_turn"])
    net.load_state_dict(blob["state_dict"]); net.eval()
    net.dt_trunc = blob["cfg_feat"].get("dt_trunc")
    m.dt_trunc = net.dt_trunc
    m._model = net
    m._cfg_feat = blob["cfg_feat"]
    m._rich_scales = blob["cfg_feat"].get("rich_scales")
    m._dt_pool = blob["dt_pool"]; m._dist_pool = blob["dist_pool"]
    m.config = blob["config"]; m.name = blob["name"]; m.history = blob.get("history")
    m._dth_scale = blob["cfg_feat"]["dth_scale"]; m._kin_scales = blob["cfg_feat"]["kin_scales"]
    # the cfg-carried flags a refit of the loaded object rebuilds _cfg_feat from
    m.dt_snap = bool(m._cfg_feat.get("dt_snap", False))
    m.psi_ref = str(m._cfg_feat.get("psi_ref", "ori"))
    m.phi_ref = str(m._cfg_feat.get("phi_ref", "ori"))
    m.K_stim = int(m._cfg_feat.get("K_stim", blob.get("K_stim", 1)))
    m.re_sd = np.asarray(blob["re_sd"], float)
    # The gain is already folded into the saved loadings; kept for reporting only. Re-applying it
    # here would square it.
    m.re_gain = blob.get("re_gain")
    m.re_gain_report = blob.get("re_gain_report")
    # Round-trip the active-only extras so a REFIT of a loaded object rebuilds the same features
    # (generation reads _cfg_feat and is correct either way).
    m.cfg_feat_extra = {k: m._cfg_feat[k] for k in ("clock_mode", "clock_tau_s", "lag_valid", "wall_reject",
                                                    "kin_dt_lags", "run_clock_tau", "stim_align", "stim_static",
                                                    "stim_jump", "stim_dmove")
                        if k in m._cfg_feat}
    # cfg_feat is what the generator reads, so it wins; the blob key is the fallback
    m.drift = m._cfg_feat.get("drift", blob.get("drift"))
    m.drift_report = blob.get("drift_report")
    m.regime_w = blob.get("regime_w")
    if m.drift is not None and getattr(net, "drift", None) is not None:
        net.d_bord = float(m.drift.get("bord", 0.0))
    # twin table (fish_id -> fitted z_i) of an in-sample checkpoint; None on every fold checkpoint
    m.z_twin = ({int(k): [float(v) for v in np.asarray(z).reshape(-1)] for k, z in blob["z_twin"].items()}
                if blob.get("z_twin") else None)
    m._re_nodes, m._re_log_pi = _ghq(m.re_m, m.re_sd, k=m.re_ghq_k)
    return m
