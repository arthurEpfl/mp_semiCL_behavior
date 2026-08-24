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

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP as M                     # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_joint as JT              # noqa: E402
import fit_MLP_mix as MX                # noqa: E402


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
                 use_body=False, border_gain=False, border_slice=None):
        super().__init__(n_features, hidden, n_dist=n_dist, n_dt=n_dt)
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
        self.emb = nn.Embedding(self.n_fish, self.m)
        self.z_body = nn.Linear(self.m, hidden, bias=False)
        self.z_turn = nn.Linear(self.m, 1, bias=False)
        self.z_dist = nn.Linear(self.m, 1, bias=False)
        self.z_dt = nn.Linear(self.m, 1, bias=False)
        if self.border_gain:
            if self.border_slice is None:
                raise ValueError("border_gain=True needs border_slice=(start, stop)")
            self.z_bord = nn.Linear(self.m, 1, bias=False)
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
        # z_body stays EXACTLY zero unless explicitly enabled, so it receives no gradient and the
        # mechanism is genuinely level-only. See the class docstring for why this defaults off.
        if self.use_body:
            nn.init.normal_(self.z_body.weight,
                            std=0.1 / max(float(sigma_init), 1e-6) / self.m ** 0.5)
        else:
            nn.init.zeros_(self.z_body.weight)

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
        if self.border_gain:
            # bounded per-fish gain on the border columns only: exp(+-1.5) caps the swing at
            # 0.22-4.5x, and clamp(0)=0 keeps the z=0 path (joint_forward, the population fish)
            # bit-identical to a net without the mechanism.
            g = torch.exp(torch.clamp(self.z_bord(z), -1.5, 1.5))
            s0, s1 = self.border_slice
            x = torch.cat([x[:, :s0], x[:, s0:s1] * g, x[:, s1:]], dim=1)
        b = torch.relu(self.body[0](x) + self.z_body(z)) if self.use_body else self.body(x)
        out = self.head(b)
        out = torch.cat([out[:, :1] + self.z_turn(z), out[:, 1:]], dim=1)
        angle = self._angle_params(out)
        a = a_z.reshape(-1, 1)
        k = self.n_dist
        do = self.dist_head(torch.cat([b, a], dim=1))
        dist = (torch.log_softmax(do[:, :k], dim=1),
                do[:, k:2 * k] + self.z_dist(z),
                torch.nn.functional.softplus(do[:, 2 * k:3 * k]) + M.EPS)
        kt = self.n_dt
        to = self.dt_head(torch.cat([b, a, ld_c.reshape(-1, 1)], dim=1))
        dtp = (torch.log_softmax(to[:, :kt], dim=1),
               to[:, kt:2 * kt] + self.z_dt(z),
               torch.nn.functional.softplus(to[:, 2 * kt:3 * kt]) + M.EPS)
        return angle, dist, dtp


# =========================================================================== #
#  Per-bout log-likelihood, shared by the trainer, the EM step and the scorer
# =========================================================================== #
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
    lt = MX._lnk_logpdf(torch.clamp(yt, min=1e-6), *tpar)
    return la, ld, lt, md, mt


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


@torch.no_grad()
def _node_logliks(net, X, a_z, ld_c, y_ang, y_dist, y_dt, nodes):
    """(n, Q) total per-bout log-likelihood at each quadrature node, masked channels excluded."""
    Xt = torch.tensor(np.asarray(X, np.float32))
    az = torch.tensor(a_z); lc = torch.tensor(ld_c)
    ya = torch.tensor(np.asarray(y_ang, np.float32))
    yd = torch.tensor(np.asarray(y_dist, np.float32))
    yt = torch.tensor(np.asarray(y_dt, np.float32))
    n, Q = Xt.shape[0], nodes.shape[0]
    out = np.empty((n, Q), dtype=np.float64)
    for q in range(Q):
        zq = torch.tensor(nodes[q]).reshape(1, -1).expand(n, -1)
        la, ld, lt, md, mt = _chan_logliks(net, Xt, az, lc, ya, yd, yt, zq)
        tot = la + torch.where(md, ld, torch.zeros_like(ld)) + torch.where(mt, lt, torch.zeros_like(lt))
        out[:, q] = tot.numpy().astype(np.float64)
    return out


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
                     border_gain=False, border_slice=None,
                     noise_builder=None, noise_mode="concat", noise_frac=0.5):
    """Stage A. Joint MAP fit of the net and the per-fish latents under a FIXED N(0, I) prior.

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

    net = MDNMixJointRE(Xt.shape[1], hidden, n_dist=n_dist, n_dt=n_dt,
                        n_fish=int(n_fish), m=int(m), sigma_init=sigma_init,
                        use_body=use_body, border_gain=border_gain,
                        border_slice=border_slice).to(dev)
    # No weight decay on the embedding: it already has an explicit N(0, I) prior in the loss, and
    # decaying it as well would apply that prior twice at an unstated strength.
    emb_params = list(net.emb.parameters())
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
        ll = (la.mean() + (ld[md].mean() if md.any() else zt) + (lt[mt].mean() if mt.any() else zt))
        if freeze_sigma_zero:
            return -ll
        return -ll + (0.5 * net.emb.weight ** 2).sum() / float(n)

    def nll(ix):
        return _nll_arrays(ix, Xt[ix], az[ix], lc[ix], ya[ix], yd[ix], yt[ix])

    # Contiguous per-fish validation blocks (see docstring).
    idx_all = np.arange(n)
    tr_list, va_list = [], []
    for f in range(int(n_fish)):
        rows = idx_all[np.asarray(fish_idx) == f]
        cut = max(int(len(rows) * (1.0 - val_frac)), 1)
        tr_list.append(rows[:cut]); va_list.append(rows[cut:])
    tr = torch.tensor(np.concatenate(tr_list), dtype=torch.long, device=dev)
    va = torch.tensor(np.concatenate([v for v in va_list if len(v)]), dtype=torch.long, device=dev)

    best, best_state, bad, hist = np.inf, None, 0, []
    for _ep in range(epochs):
        net.train()
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
            loss.backward(); opt.step()
        if NT is not None and noise_mode != "mask":       # 'concat': a second, noisy pass
            tr_ep2 = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(tr_ep2), batch_size):
                ix = tr_ep2[i:i + batch_size]
                opt.zero_grad()
                loss = _nll_arrays(ix, NT[0][ix], NT[1][ix], NT[2][ix],
                                   NT[3][ix], NT[4][ix], NT[5][ix])
                loss.backward(); opt.step()
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


def gain_grid_for(name):
    """``'wide'`` -> the 8-point grid, anything else -> the archived 5-point grid."""
    return GAIN_GRID_WIDE if str(name) == "wide" else GAIN_GRID


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
                        verbose=False, **gen_kw):
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
    gen_fn = MX._gen_for(cfg_feat)
    curve = []
    for g in grid:
        with torch.no_grad():
            for n_, p in net.named_parameters():
                if n_ in base:
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


def apply_loading_gain(net, gain):
    """Fold the fitted gain into the loadings, RE-CENTRING so the pooled marginals do not move.

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
    """
    g = float(gain)
    with torch.no_grad():
        for lin, head, n_c in ((net.z_dist, net.dist_head, net.n_dist),
                               (net.z_dt, net.dt_head, net.n_dt)):
            gamma2 = float((lin.weight ** 2).sum())          # from the ORIGINAL loadings
            head.bias[n_c:2 * n_c] -= 0.5 * (g * g - 1.0) * gamma2
        for n_, p in net.named_parameters():
            if n_.startswith("z_"):
                p.mul_(g)
    return net


# =========================================================================== #
#  Adapter
# =========================================================================== #
class MLPMixJointREDensity(JT.MLPMixJointDensity):
    """Joint model with a fitted per-fish random effect, marginalised exactly at score time."""

    def __init__(self, *args, re_m=2, re_sigma_init=0.5, re_gain_search=True, re_ghq_k=7,
                 re_zero=False, re_shuffle=False, re_body=False, contain_decouple=False,
                 contain_graded=False, rad_indiv=False, re_bgain=False,
                 re_gain_grid="std", **kwargs):
        super().__init__(*args, **kwargs)
        self.re_gain_grid = str(re_gain_grid)
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
            sigma_init=self.re_sigma_init, freeze_sigma_zero=self.re_zero,
            shuffle_labels=self.re_shuffle, use_body=self.re_body,
            border_gain=self.re_bgain, border_slice=bg_slice,
            **JT._noise_kwargs(self, train_records))

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
        self.re_gain, self.re_gain_report = 1.0, None
        if not self.re_zero and self.re_gain_search:
            # max_fish=8, matching _search_kin_feedback's precedent: this runs inside every fold's
            # fit, so it is charged 5 grid points x max_fish x n_draws rollouts every time. 8 fish
            # x 2 draws = 16 trajectories is what the smoke suite showed gives a clean interior
            # minimum; 12 fish was measurably no better and cost 50% more wall-clock.
            _grid = gain_grid_for(self.re_gain_grid)
            self.re_gain, self.re_gain_report = search_loading_gain(
                self._model, train_records, self._cfg_feat, grid=_grid, seed=self.seed,
                max_fish=min(self.contain_fish, 8), closed_loop=self.closed_loop,
                guardrail=self.guardrail, eps=self.eps, stim_loop=self.stim_loop)
            apply_loading_gain(self._model, self.re_gain)
            # ALWAYS print the fitted gain. Until now only the pinned case was visible, so a fold
            # that landed at an interior-but-implausible value left no trace anywhere -- the
            # results CSV has no column for it and the per-fold report is discarded.
            print(f"    [{self.name}] loading gain {self.re_gain} "
                  f"(grid {self.re_gain_grid}, {len(_grid)} pts)")
            if self.re_gain in (_grid[0], _grid[-1]):
                print(f"    WARNING [{self.name}]: loading gain pinned at the grid edge "
                      f"({self.re_gain}); the between-fish spread could not be matched, so widen "
                      f"the grid (re_gain_grid='wide') before trusting this arm"
                      + ("  (expected for the label-shuffled control)" if self.re_shuffle else ""))
        self._re_nodes, self._re_log_pi = _ghq(self.re_m, self.re_sd, k=self.re_ghq_k)

        # ---- the inherited calibration pipeline, unchanged and in the same order ----------- #
        if self.vigor:
            vg = MX._calibrate_vigor(train_records)
            vg["apply_turn"] = False
            self._cfg_feat["vigor"] = vg
            if self.vigor_drift_calib:
                MX.MLPMixKinDensity._rescale_vigor_drift(self, train_records)
        if self.kin_fb_calib:
            tg, vtg = MX._kin_acf_targets(train_records)
            kfb, _kvs = MX._search_kin_feedback(
                self._model, train_records, self._cfg_feat, tg, vtg,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
            self._cfg_feat["kin_feedback"] = kfb
        if self.contain:
            self._cfg_feat["wall_mode"] = "reflect"
            self._cfg_feat.update(MX._search_wall(
                self._model, train_records, self._cfg_feat, mode=self.contain_mode,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed,
                max_fish=self.contain_fish, w_occ=self.contain_w_occ,
                decouple=self.contain_decouple, graded=self.contain_graded))
        if self.innov_comp:
            self._cfg_feat["kin_innov"] = MX._search_innov_comp(
                self._model, train_records, self._cfg_feat,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
        if self.indiv:
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
        _rad_widened = bool(self.rad_indiv
                            and self._cfg_feat.get("fish_re", {}).get("rad_sd", 0.0) > 0)
        if self.contain and (self.indiv or self.innov_comp or _rad_widened):
            self._cfg_feat.update(MX._search_wall(
                self._model, train_records, self._cfg_feat, mode=self.contain_mode,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed + 1,
                max_fish=min(self.contain_fish, 8), w_occ=self.contain_w_occ,
                decouple=self.contain_decouple, graded=self.contain_graded))
        self.config = dict(mode="joint_re", hidden=self.hidden, n_features=X.shape[1], K=self.K,
                           K_kin=self.K_kin, stim_mode=self.stim_mode, use_border=self.use_border,
                           use_kin=True, n_dist=self.n_dist, n_dt=self.n_dt,
                           re_m=self.re_m, n_fish=len(train_records))
        return self

    def score(self, rec):
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
        La = np.empty((n, Q)); Ld = np.empty((n, Q)); Lt = np.empty((n, Q))
        W = np.empty((n, Q, 5))
        SHP = np.empty((9, n, Q))          # sigmaF, (alpha, theta) x 4, in symbol_logq_mix's order
        with torch.no_grad():
            for q in range(Q):
                zq = torch.tensor(nodes[q]).reshape(1, -1).expand(n, -1)
                ang, dpar, tpar = self._model.joint_forward_re(Xt, azt, lct, zq)
                La[:, q] = MX.mix_logdens(yat, ang).numpy()
                Ld[:, q] = MX._lnk_logpdf(torch.clamp(ydt, min=1e-6), *dpar).numpy()
                Lt[:, q] = MX._lnk_logpdf(torch.clamp(ytt, min=1e-6), *tpar).numpy()
                W[:, q, :] = np.exp(ang[0].numpy())
                for j in range(9):
                    SHP[j, :, q] = ang[j + 1].numpy()

        # Masks: exactly the bouts the training loss masks, so the reported likelihood is over the
        # support the model was fitted on (10 bouts have distance 0; the last bout of each fish has
        # no forward-looking distance or interval). A masked channel must contribute log-weight 0 to
        # the running belief -- NOT NaN, which would poison every later bout's posterior.
        dv = np.asarray(yd, float); tv = np.asarray(yt, float)
        md = np.isfinite(dv) & (dv > 0); mt = np.isfinite(tv) & (tv > 0)
        Ld_w = np.where(md.reshape(-1, 1), Ld, 0.0)
        Lt_w = np.where(mt.reshape(-1, 1), Lt, 0.0)

        # Running belief over z: the prior times everything seen strictly before bout t.
        Ltot = La + Ld_w + Lt_w
        cum = np.zeros_like(Ltot)
        if n > 1:
            cum[1:] = np.cumsum(Ltot[:-1], axis=0)
        lw0 = log_pi.reshape(1, -1) + cum

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
        pw = np.exp(lw0 - d0.reshape(-1, 1))                           # (n, Q) posterior weights
        slq_q = np.stack([MX.symbol_logq_mix(W[:, q, :], *[SHP[j, :, q] for j in range(9)])
                          for q in range(Q)], axis=1)                  # (n, Q, 3)
        slq = logsumexp(slq_q + np.log(np.maximum(pw, 1e-300))[:, :, None], axis=1)
        slq = slq - logsumexp(slq, axis=1, keepdims=True)
        return dict(logdens=logdens, symbol_logq=slq, logdens_dist=ld, logdens_dt=lt)

    def sample(self, rec, rng, **kw):
        # **kw forwards the tool-mode knobs (stim_provider / stop_time_s); see JT.sample.
        return JT.generate_mix_joint_loop(self._model, rec, self._cfg_feat, rng,
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
                    re_gain_report=getattr(model, "re_gain_report", None),
                    K_stim=int(getattr(model, "K_stim", 1)),
                    noise_aug=getattr(model, "noise_aug", None),
                    contain_mode=str(getattr(model, "contain_mode", "bias")),
                    n_fish=int(model.config["n_fish"]),
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
                             re_ghq_k=blob.get("re_ghq_k", 7))
    net = MDNMixJointRE(blob["n_features"], blob["hidden"], n_dist=blob["n_dist"],
                        n_dt=blob["n_dt"], n_fish=blob["n_fish"], m=blob["re_m"],
                        use_body=blob.get("re_body", True),
                        border_gain=blob.get("re_bgain", False),
                        border_slice=blob.get("re_border_slice"))
    net.load_state_dict(blob["state_dict"]); net.eval()
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
    m._re_nodes, m._re_log_pi = _ghq(m.re_m, m.re_sd, k=m.re_ghq_k)
    return m
