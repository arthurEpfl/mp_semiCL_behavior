"""Joint-likelihood bout model: one feature vector, one body, chained heads.

WHAT IS DIFFERENT FROM ``fit_MLP_mix.MLPMixKinDensity``, and why it is worth measuring.

1. THE FACTORISATION MATCHES THE DATA. The kin model trains
   ``p(dtheta|x_ang) . p(l|x_kin) . p(tau|x_kin, l)`` -- the distance head sees no turn angle at
   all, and neither does the dt head. Here the emission is the full chain

       L = -[ log p(dtheta | x) + log p(l | x, dtheta) + log p(tau | x, dtheta, l) ]

   so each channel is conditioned on everything sampled before it within the same bout.

2. ONE FEATURE VECTOR. The kin model hard-codes ``use_kin=False`` and builds a SEPARATE 6-column
   kin trunk, so the angle body never sees kinematic history and the kin trunk never sees the
   stimulus, the border or the turn history. Here ``use_kin=True`` puts everything in one ``x``
   that a single shared body consumes -- ``MB.build_feature_matrix`` already supports this, the
   switch was simply never flipped.

3. METRIC A COVERS ALL THREE CHANNELS. ``MLPMixKinDensity`` inherits a ``score`` that evaluates
   only the angle mixture, so about two thirds of the fitted emission is optimised against a
   likelihood that is never reported. ``score`` here returns ``logdens`` (angle, same key as
   before) plus ``logdens_dist`` and ``logdens_dt``.

DELIBERATELY ABSENT: vigor, artr, kin_feedback, kin_var_comp, qmap, indiv, innov_comp. This is a
clean baseline, not the accumulated lineage. ``contain`` is available because it corrects the
radial OCCUPANCY distribution; the thing that actually keeps the fish in the dish is the
``guardrail``, which is on regardless and is not a "fix".

CONDITIONER CONVENTION. The heads receive ``|dtheta|``, not signed dtheta: left and right should
not change step size or timing by symmetry, and feeding the sign would break a symmetry the angle
mixture otherwise enforces. Every conditioner is scaled the same way the corresponding history
block is (``/dth_scale``, and centred by ``kin_scales``) so the heads see inputs on the scale they
were trained at.

HEAD ORDER (``head_order``, default ``"a,d,t"`` = the chain written above). The chain rule is exact
in all SIX orderings of (dtheta, distance, dt), so they all target the same joint density and their
summed bits/bout ARE comparable. What differs is CAPACITY: a channel emitted first is predicted
from ``x`` alone, a channel emitted last sees the other two outcomes of the same bout. Which
ordering wins is therefore an empirical question, and the EDA points at ``t`` first -- |dtheta|
scales with the time the bout spans (rho = 0.17) while distance barely does (rho = -0.01), so
"draw the interval, then turn and travel within it" may be the cheaper description.

``"a,d,t"`` is bit-identical to the pre-ordering code, RNG draw for RNG draw, and is what every
archived checkpoint reloads as. A head at position ``i`` is a ``Linear(hidden + i, ...)``; the
conditioners are appended in the ordering's own order, so position is the only thing that changes.

COMPARABILITY WARNING: because the angle body now also reads kinematic history, this model's
angle ``bits_per_bout`` is NOT comparable to the historical turn-only series from the kin zoo.
Within this family it is fine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP as M                     # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402

_EMPTY = np.zeros(0)                    # allocation-free stand-in for an absent feature block


# =========================================================================== #
#  Net
# =========================================================================== #
#: The three emission channels, by their one-letter code. 'a' = dtheta (angle), 'd' = distance,
#: 't' = dt (interval). An ORDER is a permutation of these three.
CHANNELS = ("a", "d", "t")
DEFAULT_ORDER = ("a", "d", "t")


def parse_order(order):
    """``"t,a,d"`` / ``"tad"`` / ``('t','a','d')`` -> ``('t','a','d')``. ``None`` -> the default.

    Rejects anything that is not a permutation of the three channels, LOUDLY: a typo that silently
    fell back to the default would run a whole sweep arm under the wrong name and the CSV would be
    indistinguishable from a real result."""
    if order is None:
        return DEFAULT_ORDER
    if not isinstance(order, str):
        order = ",".join(str(s) for s in order)
    ks = tuple(s for s in order.replace(",", " ").replace("-", " ").replace(">", " ").split() if s)
    if len(ks) == 1 and len(ks[0]) == 3:        # compact form: "tad"
        ks = tuple(ks[0])
    ks = tuple(s.strip().lower()[:1] for s in ks)
    if sorted(ks) != sorted(CHANNELS):
        raise ValueError(f"head_order must be a permutation of {CHANNELS}; got {order!r} -> {ks}")
    return ks


def order_tag(order):
    """``('t','a','d')`` -> ``'tad'``. The short label the sweep uses in file names and the CSV."""
    return "".join(parse_order(order))


class MDNMixJoint(MX.MDNMix):
    """Shared body -> three chained heads, emitted in ``order`` (default angle, distance, dt).

    Each head is a ``Linear(hidden + i, ...)`` where ``i`` is its position in the chain: it reads
    the shared body plus the conditioner of every channel emitted BEFORE it, appended in the
    ordering's own order. So the first head sees the body alone and the last sees the body plus
    two within-bout outcomes.

    ``forward`` is inherited from ``MDNMix`` and returns the ANGLE parameters only, so
    ``predict_params_mix`` / ``symbol_logq_mix`` / the symbol track keep working unchanged -- but
    ONLY when the angle head is first, because otherwise those parameters are not defined without
    conditioners. It raises rather than returning a quietly-wrong plug-in. The chained heads are
    reached through ``joint_forward``.
    """

    def __init__(self, n_features, hidden, n_dist=2, n_dt=3, order=None):
        super().__init__(n_features, hidden)
        self.n_dist, self.n_dt = int(n_dist), int(n_dt)
        self.order = parse_order(order)
        self.head_pos = {c: i for i, c in enumerate(self.order)}
        # Only REPLACE the inherited angle head when it actually needs to widen. Rebuilding it
        # unconditionally would draw a fresh init from the same generator and shift every
        # subsequent parameter's random numbers, so the default order would no longer reproduce
        # the archived fits bit for bit.
        if self.head_pos["a"]:
            self.head = nn.Linear(hidden + self.head_pos["a"], 14)
        self.dist_head = nn.Linear(hidden + self.head_pos["d"], 3 * self.n_dist)
        self.dt_head = nn.Linear(hidden + self.head_pos["t"], 3 * self.n_dt)

    def forward(self, x):
        if self.head_pos["a"]:
            raise RuntimeError(
                f"forward() is the unconditional angle head, but this net emits {self.order}, so "
                "the angle parameters depend on conditioners drawn earlier in the bout. Use "
                "joint_forward(x, a_z, ld_c, lt_c) and take its first return value.")
        return super().forward(x)

    def joint_forward(self, x, a_z, ld_c, lt_c=None):
        """(angle params, (logw,mu,sigma) for distance, same for dt) -- ALWAYS in that return
        order, whatever the emission order is.

        ``a_z`` = |dtheta|/dth_scale, ``ld_c`` = (log distance - dm)/ds, ``lt_c`` = (log dt - tm)/ts,
        all teacher-forced at train time. ``lt_c`` may be None only when dt is emitted LAST (the
        default), because then nothing conditions on it."""
        b = self.body(x)
        cond = {"a": a_z, "d": ld_c, "t": lt_c}
        out, prev = {}, []
        for i, ch in enumerate(self.order):
            inp = torch.cat([b] + prev, dim=1) if prev else b
            if ch == "a":
                out["a"] = self._angle_params(self.head(inp))
            else:
                k = self.n_dist if ch == "d" else self.n_dt
                o = (self.dist_head if ch == "d" else self.dt_head)(inp)
                out[ch] = (F.log_softmax(o[:, :k], dim=1), o[:, k:2 * k],
                           F.softplus(o[:, 2 * k:3 * k]) + M.EPS)
            if i < len(self.order) - 1:          # the last channel conditions nothing
                c = cond[ch]
                if c is None:
                    raise ValueError(f"emission order {self.order} needs the {ch!r} conditioner "
                                     f"at position {i}, but it was not supplied")
                prev.append(c.reshape(-1, 1))
        return out["a"], out["d"], out["t"]


def _cond_cols(y_ang, y_dist, dth_scale, kin_scales, y_dt=None):
    """The within-bout conditioners, on the scale the history blocks use.

    Two by default (|dtheta|, centred log distance) so every pre-ordering caller is unchanged;
    pass ``y_dt`` to also get the centred log interval, which the non-default orders need."""
    dm, ds, tm, ts = kin_scales
    a_z = np.abs(np.asarray(y_ang, float)) / float(dth_scale)
    ld = np.log(np.clip(np.asarray(y_dist, float), 1e-6, None))
    ld_c = (ld - dm) / ds
    out = (np.nan_to_num(a_z).astype(np.float32),
           np.nan_to_num(ld_c).astype(np.float32))
    if y_dt is None:
        return out
    lt = np.log(np.clip(np.asarray(y_dt, float), 1e-6, None))
    return out + (np.nan_to_num((lt - tm) / ts).astype(np.float32),)


def _noise_tensors(noise_builder, ep, dth_scale, kin_scales, dev):
    """Build one epoch's NOISY copy as device tensors, in the same row order as the clean one."""
    Xn, yan, ydn, ytn = noise_builder(ep)
    azn, lcn, ltn = _cond_cols(yan, ydn, dth_scale, kin_scales, y_dt=ytn)
    return (torch.tensor(np.asarray(Xn, np.float32), device=dev),
            torch.tensor(azn, device=dev), torch.tensor(lcn, device=dev),
            torch.tensor(ltn, device=dev),
            torch.tensor(np.asarray(yan, np.float32), device=dev),
            torch.tensor(np.asarray(ydn, np.float32), device=dev),
            torch.tensor(np.asarray(ytn, np.float32), device=dev))


# Measured, not guessed. Gradient-norm distribution over 402 steps of a 6-epoch fit (identical
# for plain m=2 and block-structured m=3): median 4.5, p90 28, p99 47, max 102. The first value
# tried, 10.0, sat between the median and p90 and clipped ~25% of steps -- that is not a backstop,
# it is a change of optimiser, and it would have made every future arm incomparable to the archived
# lineage while claiming to be a safety net. 150 is above every healthy gradient seen, so it binds
# only on a genuine excursion.
#
# NOTE the hard guarantee against NaN is the isfinite check in guarded_step, NOT this number: a
# non-finite loss or gradient is skipped outright and never written into the weights. The clip is
# a softener for large-but-finite steps, and _warn_unstable makes whatever is left visible.
GRAD_CLIP = 150.0


def guarded_step(opt, loss, params, clip=GRAD_CLIP, stats=None):
    """One optimiser step that refuses to poison the weights with a non-finite update.

    WHY THIS EXISTS. Neither joint trainer clipped gradients, and on an L4 the bgain smoke went
    ``26.86 -> nan`` in four epochs while the IDENTICAL code on CPU went ``26.82 -> 10.92``. Same
    config, same data -- CUDA's accumulation order is enough to trip an emission with lgamma and
    log terms in it. (rollout_finetune.py has clipped at 5.0 since it was written; the trainers
    that do all the actual work never did.)

    The silent-failure mode is what makes this urgent rather than cosmetic. Early stopping tests
    ``v < best - 1e-5``; with ``v = nan`` that is False, so a NaN counts as "no improvement",
    patience expires, and the fit returns ``best_state`` from whatever epoch it last improved on.
    A run that diverged at epoch 2 of 80 therefore finishes, saves, and reports as a completed
    5-hour arm while containing a 2-epoch model.

    Threshold 10.0 is deliberately loose: on a healthy fit it should never bind, so this is a
    backstop rather than a change of optimiser. It IS counted, and ``fit`` warns if it fires --
    an uninstrumented guard that quietly reshapes training would be its own trap.
    """
    if stats is not None:
        stats["n_steps"] = stats.get("n_steps", 0) + 1
    if not torch.isfinite(loss):
        if stats is not None:
            stats["n_bad_loss"] = stats.get("n_bad_loss", 0) + 1
        opt.zero_grad(set_to_none=True)
        return False
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, clip)
    if not torch.isfinite(gn):
        if stats is not None:
            stats["n_bad_grad"] = stats.get("n_bad_grad", 0) + 1
        opt.zero_grad(set_to_none=True)
        return False
    if stats is not None and float(gn) > clip:
        stats["n_clipped"] = stats.get("n_clipped", 0) + 1
    opt.step()
    return True


def _warn_unstable(stats, hist):
    """Say out loud that a fit was unstable, and how far it actually got.

    Without this a diverged fit is INVISIBLE: a non-finite validation loss fails
    ``v < best - 1e-5``, so it is counted as "no improvement", patience runs out, and the run
    returns the best state from before the blow-up. The arm completes, saves a checkpoint and
    prints a normal-looking results row while containing a model trained for a fraction of the
    requested budget. Measured on an L4: the bgain smoke went 26.86 -> nan in four epochs.
    """
    n_bad = int(stats.get("n_bad_loss", 0)) + int(stats.get("n_bad_grad", 0))
    n_clip = int(stats.get("n_clipped", 0))
    n_steps = max(int(stats.get("n_steps", 0)), 1)
    n_nan_val = int(np.sum(~np.isfinite(np.asarray(hist, float)))) if len(hist) else 0
    # TWO DIFFERENT SITUATIONS, and conflating them is how an alarm gets ignored. A non-finite
    # VALIDATION epoch means the fit actually broke and best_state is truncated -- that is the
    # five-hours-wasted case. A few skipped MINIBATCH steps with every validation epoch finite
    # means the guard did its job and the fit is healthy: measured on an L4, one skipped step out
    # of ~268 was the whole difference between 26.86 -> nan and 26.86 -> 10.94.
    if n_nan_val:
        print(f"    WARNING: UNSTABLE FIT -- {n_nan_val} of {len(hist)} epochs had a non-finite "
              f"validation loss ({n_bad} of {n_steps} minibatch steps also skipped). The returned "
              f"model is the best state from BEFORE the blow-up, so this arm is trained on far "
              f"less than the requested budget. Do not read it as a completed run.")
    elif n_bad:
        # SPLIT the count, and escalate above a few percent. A handful of skips is the guard doing
        # its job; HALF the updates being discarded is a different animal -- measured 192 of 376 on
        # a GRU fit at seq_batch=16, which is a large silent handicap that the old wording called
        # "healthy". Naming loss-vs-gradient is what makes it diagnosable without another run.
        frac = n_bad / n_steps
        lead = "note" if frac < 0.05 else "WARNING"
        print(f"    {lead}: the NaN guard skipped {n_bad} of {n_steps} minibatch step(s) "
              f"({100 * frac:.0f}%) -- {int(stats.get('n_bad_loss', 0))} non-finite LOSS, "
              f"{int(stats.get('n_bad_grad', 0))} non-finite GRADIENT. Every validation epoch was "
              f"finite, so the returned model is complete"
              + ("." if frac < 0.05 else
                 ", but it was trained on far fewer updates than the budget implies -- a larger "
                 "batch dilutes the offending bout and lowers this rate."))
    elif n_clip:
        print(f"    note: gradient clipping bound on {n_clip} step(s) (threshold {GRAD_CLIP}) -- "
              "the fit was steered but did not diverge")


def _noise_kwargs(model, train_records):
    """Translate a model's ``noise_aug`` dict into the trainer's noise kwargs ({} when off).

    ONE rng for the whole fit, advanced by every call, so each epoch sees a FRESH perturbation of
    the same records rather than a fixed augmented dataset -- which would merely be a bigger
    training set and would let the net memorise the particular jitter. Shared by the joint and
    fishre trainers so the two cannot drift."""
    na = getattr(model, "noise_aug", None)
    if not na:
        return {}
    rng = np.random.default_rng([int(model.seed), 7919])
    cfg, recs = model._cfg_feat, list(train_records)
    return dict(noise_builder=lambda _ep: MB.build_noisy_sequences(recs, cfg, rng),
                noise_mode=str(na.get("mode", "concat")),
                noise_frac=float(na.get("frac", 0.5)))


def fit_mix_joint(X, y_ang, y_dist, y_dt, hidden, dth_scale, kin_scales, *, epochs=80, lr=1e-3,
                  batch_size=256, patience=8, val_frac=0.15, seed=0, device=None,
                  n_dist=2, n_dt=3, weight_decay=1e-5, order=None,
                  noise_builder=None, noise_mode="concat", noise_frac=0.5):
    """Train ``MDNMixJoint`` on the exact chained joint NLL. Distance/dt NaN-masked per channel
    (the last bout of every fish has no forward-looking dt or distance); early stopping is on the
    JOINT validation loss, so all three channels share one stopping criterion.

    ``noise_builder(epoch) -> (X~, dtheta~, distance~, dt)`` enables NOISE AUGMENTATION: a fresh
    pose-perturbed copy of the whole training set each epoch, with the targets corrected back to
    the true next onset (``MB.build_noisy_sequences``). ``noise_mode='concat'`` trains on clean
    and noisy rows both (~2x epoch cost, the faithful version); ``'mask'`` replaces a
    Bernoulli(``noise_frac``) subset of each clean batch instead (same cost, half the clean data).
    THE VALIDATION LOSS IS ALWAYS CLEAN -- otherwise early stopping optimises a different
    objective from the one reported, and the loss is not comparable to any archived arm."""
    M._seed_everything(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a_z, ld_c, lt_c = _cond_cols(y_ang, y_dist, dth_scale, kin_scales, y_dt=y_dt)
    Xt = torch.tensor(np.asarray(X, np.float32), device=dev)
    ya = torch.tensor(np.asarray(y_ang, np.float32), device=dev)
    yd = torch.tensor(np.asarray(y_dist, np.float32), device=dev)
    yt = torch.tensor(np.asarray(y_dt, np.float32), device=dev)
    az = torch.tensor(a_z, device=dev); lc = torch.tensor(ld_c, device=dev)
    lt_t = torch.tensor(lt_c, device=dev)
    net = MDNMixJoint(Xt.shape[1], hidden, n_dist=n_dist, n_dt=n_dt, order=order).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    def nll_on(Xb, azb, lcb, ltb, yab, ydb, ytb):
        ang, dpar, tpar = net.joint_forward(Xb, azb, lcb, ltb)
        la = MX.mix_logdens(yab, ang).mean()
        md = torch.isfinite(ydb) & (ydb > 0); mt = torch.isfinite(ytb) & (ytb > 0)
        ld = MX._lnk_logpdf(torch.clamp(ydb, min=1e-6), *dpar)
        lt = MX._lnk_logpdf(torch.clamp(ytb, min=1e-6), *tpar)
        z = torch.zeros((), device=dev)
        return -(la + (ld[md].mean() if md.any() else z) + (lt[mt].mean() if mt.any() else z))

    def nll(ix):
        return nll_on(Xt[ix], az[ix], lc[ix], lt_t[ix], ya[ix], yd[ix], yt[ix])

    n = Xt.shape[0]; nval = max(int(val_frac * n), 1)
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g).to(dev)
    tr, va = perm[nval:], perm[:nval]
    best, best_state, bad, hist, hist_tr = np.inf, None, 0, [], []
    _gstats = {}
    for _ep in range(epochs):
        net.train()
        # RESHUFFLE EVERY EPOCH. `perm` above fixes the train/val SPLIT (which must stay fixed);
        # the batch COMPOSITION must not. Slicing `tr` directly, as this loop used to, meant every
        # minibatch contained the same 256 bouts for all 80/250 epochs. `fit_mix` (:118) and
        # `fit_mix_kin` both reshuffle; this was the only trainer in the repo that did not.
        NT = (_noise_tensors(noise_builder, _ep, dth_scale, kin_scales, dev)
              if noise_builder is not None else None)
        tr_ep = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(tr_ep), batch_size):
            ix = tr_ep[i:i + batch_size]
            if NT is not None and noise_mode == "mask":
                m = torch.rand(len(ix), device=dev) < float(noise_frac)
                Xb = torch.where(m.unsqueeze(1), NT[0][ix], Xt[ix])
                azb = torch.where(m, NT[1][ix], az[ix]); lcb = torch.where(m, NT[2][ix], lc[ix])
                ltb = torch.where(m, NT[3][ix], lt_t[ix])
                yab = torch.where(m, NT[4][ix], ya[ix]); ydb = torch.where(m, NT[5][ix], yd[ix])
                ytb = torch.where(m, NT[6][ix], yt[ix])
                opt.zero_grad(); loss = nll_on(Xb, azb, lcb, ltb, yab, ydb, ytb)
            else:
                opt.zero_grad(); loss = nll(ix)
            guarded_step(opt, loss, net.parameters(), stats=_gstats)
        if NT is not None and noise_mode != "mask":       # 'concat': a second, noisy pass
            tr_ep2 = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(tr_ep2), batch_size):
                ix = tr_ep2[i:i + batch_size]
                opt.zero_grad()
                loss = nll_on(*(t[ix] for t in NT))
                guarded_step(opt, loss, net.parameters(), stats=_gstats)
        net.eval()
        with torch.no_grad():
            v = float(nll(va))
            # Training NLL on the SAME criterion, for the train/val curve. One extra forward
            # pass per epoch over the train split -- negligible on GPU, and it is the only way
            # to see the generalisation gap rather than just the early-stopping trace.
            t_ = float(nll(tr))
        hist.append(v)
        hist_tr.append(t_)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    _warn_unstable(_gstats, hist)
    if best_state is not None:
        net.load_state_dict(best_state)
    net = net.to("cpu").eval()
    net._hist_train = list(hist_tr)      # `hist` stays val-only so nothing downstream changes
    return net, hist


# =========================================================================== #
#  numpy rollout
# =========================================================================== #
def _numpy_predictor_joint(model, z=None):
    """Hand-rolled forward pass for the free-run loop (the torch graph is far too slow per bout).

    ``z`` is the per-trajectory random effect, or None. It loads LEVEL-ONLY and enters every head
    additively, so the whole mechanism folds into four bias vectors ONCE here and costs nothing per
    bout. Level-only means: the forward-mixture weight logit for the turn head, and the location
    parameters (not the scales, not the mixture shapes) for the two kinematic heads.
    """
    sd = model.state_dict()
    W1, b1 = sd["body.0.weight"].numpy(), sd["body.0.bias"].numpy()
    W2, b2 = sd["head.weight"].numpy(), sd["head.bias"].numpy()
    Wd, bd = sd["dist_head.weight"].numpy(), sd["dist_head.bias"].numpy()
    Wt, bt = sd["dt_head.weight"].numpy(), sd["dt_head.bias"].numpy()
    nd, ndt = int(model.n_dist), int(model.n_dt)
    if z is not None and "z_body.weight" in sd:
        # KEEP THE DTYPE. `b1 + (W_z @ z)` promotes float32 -> float64 the moment z is float64, and
        # every subsequent bout is then computed in a different precision from the control's. The
        # divergence is ~1e-8 per bout, which is invisible until it flips an rng.choice and the two
        # trajectories separate completely -- so a z = 0 arm would NOT reproduce the control and the
        # paired ladder would be silently unpaired.
        z = np.asarray(z, np.float32).reshape(-1)
        # BLOCK-STRUCTURED LATENT: turn/dist/dt read the leading `n_core` dimensions, the border
        # gain reads the trailing ones. Without this the dot products below would shape-mismatch
        # the moment block=True, because those loadings are (1, n_core) while z is length m.
        _ncore = int(getattr(model, "n_core", z.shape[0]))

        def _load1(key, core=True):
            """Scalar offset from a loading. The RESHAPE IS NOT COSMETIC: `W @ z` on a
            (1, m) weight returns a length-1 ARRAY, and converting that to a scalar is a
            deprecation under numpy 1.x but a hard TypeError under numpy 2 -- so the same code
            ran on one interpreter and crashed on another. Reduce to a float explicitly."""
            w = sd[key].numpy().reshape(-1)
            zz = z[:_ncore] if core else z[_ncore:]
            if w.shape[0] != zz.shape[0]:          # not block-structured: the loading reads all of z
                zz = z
            return float(np.dot(w, zz))

        b1 = (b1 + sd["z_body.weight"].numpy().reshape(len(b1), -1) @ z).astype(b1.dtype)
        b2 = b2.copy(); b2[0] = b2.dtype.type(b2[0] + _load1("z_turn.weight"))
        bd = bd.copy()
        bd[nd:2 * nd] = (bd[nd:2 * nd] + _load1("z_dist.weight")).astype(bd.dtype)
        bt = bt.copy()
        bt[ndt:2 * ndt] = (bt[ndt:2 * ndt] + _load1("z_dt.weight")).astype(bt.dtype)
        if "z_bord.weight" in sd:
            # re_bgain: the per-fish gain on the border INPUT columns folds into W1's border
            # columns once (W1 @ diag(g on slice) == scale those columns), zero per-bout cost.
            # Same clamp as the torch path; dtype preserved for the CRN reasons above.
            _g = float(np.exp(np.clip(_load1("z_bord.weight", core=False), -1.5, 1.5)))
            _s0, _s1 = model.border_slice
            W1 = W1.copy()
            W1[:, _s0:_s1] = (W1[:, _s0:_s1] * _g).astype(W1.dtype)

    def sp(z):
        return np.logaddexp(0.0, z)

    def _sm(lw):
        w = np.exp(lw - lw.max()); return w / w.sum()

    # Head positions. Each closure below takes the conditioners of the channels emitted before it
    # as trailing scalars -- so on the default order the calls stay exactly `body_angle(x)`,
    # `dist_params(b, a_z)`, `dt_params(b, a_z, ld_c)` and every existing caller is unchanged.
    pos = getattr(model, "head_pos", None) or {c: i for i, c in enumerate(DEFAULT_ORDER)}

    def _inp(b, prev, ch):
        if len(prev) != pos[ch]:
            raise ValueError(f"head {ch!r} sits at position {pos[ch]} of {getattr(model, 'order', DEFAULT_ORDER)} "
                             f"and needs that many conditioners; got {len(prev)}")
        # float64 on purpose: the old code wrote `np.concatenate([b, [a_z]])`, and a Python-float
        # list promotes the float32 body to float64 for the whole matmul. Forcing b.dtype here
        # would be "cleaner" and would silently change every archived rollout -- see the CRN note
        # on dtype above. Keep the promotion.
        return np.concatenate([b, np.asarray(prev, float)]) if prev else b

    def body(x):
        return np.maximum(W1 @ x + b1, 0.0)

    def angle_params(b, *prev):
        o = W2 @ _inp(b, prev, "a") + b2
        return (_sm(o[:5]), sp(o[5]) + M.EPS,
                sp(o[6]) + M.ALPHA_MIN, sp(o[7]) + M.EPS, sp(o[8]) + M.ALPHA_MIN, sp(o[9]) + M.EPS,
                sp(o[10]) + M.ALPHA_MIN, sp(o[11]) + M.EPS,
                sp(o[12]) + M.ALPHA_MIN, sp(o[13]) + M.EPS)

    def body_angle(x, *prev):
        """(body activation, angle params). Only meaningful in one call when the angle head is
        first; otherwise pass its conditioners, or call ``body`` and ``angle_params`` separately."""
        b = body(x)
        return b, angle_params(b, *prev)

    def dist_params(b, *prev):
        do = Wd @ _inp(b, prev, "d") + bd
        return _sm(do[:nd]), do[nd:2 * nd], sp(do[2 * nd:3 * nd]) + M.EPS

    def dt_params(b, *prev):
        to = Wt @ _inp(b, prev, "t") + bt
        return _sm(to[:ndt]), to[ndt:2 * ndt], sp(to[2 * ndt:3 * ndt]) + M.EPS

    # `body` and `angle_params` ride along as attributes rather than extra return values, so the
    # arity stays 3 for smoke_bgain / smoke_rollout_aware and the fishre family.
    body_angle.body = body
    body_angle.angle_params = angle_params
    return body_angle, dist_params, dt_params


def generate_mix_joint_loop(model, rec, cfg_feat, rng, closed_loop=True, guardrail=True, eps=1e-3,
                            stim_loop=True, keep_features=False,
                            stim_provider=None, stop_time_s=None):
    """Free-run one trajectory. Same contract as ``MX.generate_mix_kin_loop`` so the calibrators
    and the harness can use either.

    Closed-loop from the model's own pose: the border covariate and (when ``stim_loop``) the
    shoal bearing/range are recomputed each bout from the generated position. The turn history and
    BOTH kinematic history lags live in the same feature row, matching training.

    ROLLOUT ON A *NEW* STIMULUS PROGRAM (``stim_provider`` / ``stop_time_s``, both default off):
    ``rec.dot_cx[t]`` is indexed by BOUT, and is materialised as an array before this loop starts,
    so a fixed stimulus timeline cannot be pre-baked into it -- the bout times are themselves
    generated here. ``stim_provider(t, time_s, x, y, h) -> (dot_cx, dot_cy, moving, n_dots)`` is
    called instead, with the live clock and pose, so the program can be resolved online (see
    ``scripts/stim_rollout.py``). ``stop_time_s`` ends the run at the last bout whose ONSET is
    still inside the programme and truncates every returned array to that length.

    Both are pure additions: with ``stim_provider=None`` and ``stop_time_s=None`` the reads and
    the RNG draw order are exactly as before, so output stays bit-identical (asserted by
    ``scripts/smoke_stim_program.py``).
    """
    n = rec.n
    K = cfg_feat["K"]; Kk = int(cfg_feat.get("K_kin", 3))
    dth_scale = cfg_feat["dth_scale"]
    dm, ds, tm, ts = cfg_feat["kin_scales"]
    dist_hi, dt_hi = cfg_feat.get("kin_clip", (400.0, 600.0))
    stim = M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                         phi_ref=cfg_feat.get("phi_ref", "ori"))
    use_border = cfg_feat.get("use_border")
    tf_border = MB._border_block(rec, cfg_feat) if use_border else np.zeros((n, 0))
    # K_stim block: pure function of the record's own stimulus programme, so it is indexed by
    # bout exactly as the teacher-forced stim block is. (The whole rollout is indexed by bout
    # number rather than wall-clock -- the known clock_ratio caveat -- so this is consistent with
    # how `moving` and `n_dots` are already fed.)
    tf_shist = (MB._stimhist_block(rec, cfg_feat) if int(cfg_feat.get("K_stim", 1)) > 1
                else np.zeros((n, 0)))
    recompute_stim = (stim_loop and closed_loop
                      and cfg_feat["stim_mode"] in MX.ONLINE_STIM_MODES)
    if stim_provider is not None and not recompute_stim:
        # Fail loudly: silently teacher-forcing the record's stimulus while the caller believes a
        # new programme is driving the fish would look like a working tool and be a replay.
        raise ValueError(
            "stim_provider needs the online stimulus path: require stim_loop=True, "
            f"closed_loop=True and stim_mode in {MX.ONLINE_STIM_MODES}; got "
            f"stim_loop={stim_loop}, closed_loop={closed_loop}, "
            f"stim_mode={cfg_feat['stim_mode']!r}.")
    if stim_provider is not None and tf_shist.shape[1]:
        # Same reasoning as above: the K_stim channels describe the RECORD's block programme, so
        # replaying them under a new programme would be a silent replay of the old one.
        raise ValueError("stim_provider is incompatible with K_stim > 1: the stimulus-memory "
                         "block would still describe the record's own programme. Resolve the "
                         "previous onsets from the provider's programme before enabling both.")
    if recompute_stim:
        dot_cx_a = np.asarray(getattr(rec, "dot_cx", np.full(n, np.nan)), float)
        dot_cy_a = np.asarray(getattr(rec, "dot_cy", np.full(n, np.nan)), float)
        mv_a = np.asarray(rec.moving, float); phi_gen = np.full(n, np.nan)
        nd_a = np.asarray(getattr(rec, "n_dots", np.zeros(n)), float)
        dcx_used = np.full(n, np.nan); dcy_used = np.full(n, np.nan)
    # PER-FISH RANDOM EFFECT. One draw per TRAJECTORY from the fitted prior -- the honest "new fish"
    # claim, and the only option that cannot leak: the evaluated fish is held out, so it has no
    # fitted embedding. Sigma is diagonal by construction, so a scaled standard normal is the draw.
    #
    # SPAWNED STREAM, and this is load-bearing. generative_metrics keys rollouts on (seed, fish, rep)
    # with NO model index, deliberately, so replicate r of model A and model B are paired -- which
    # requires every arm to consume the same variates from `rng` in the same order. Drawing z from
    # `rng` itself would desynchronise every emission draw after it and the paired t-test would
    # still print RESOLVED while being invalid. SeedSequence.spawn bumps the parent's child counter
    # but does not advance its stream, so the emission draws stay byte-identical to the control.
    # Asserted in smoke_fishre.py, not assumed.
    _re = cfg_feat.get("fish_re")
    z_re = None
    u_rad_re = 0.0
    if _re is not None:
        _sd_re = np.asarray(_re["sd"], float).reshape(-1)
        z_re = rng.spawn(1)[0].normal(size=_sd_re.shape[0]) * _sd_re
        if "rad_sd" in _re:
            # RADIAL individuality for the fishre family (rad_indiv): a per-trajectory
            # multiplicative swing on the containment bias, mirroring indiv's u_rad -- an
            # additive offset leaks into the turn marginal and can flip a fish outward.
            # SPAWN ORDER IS LOAD-BEARING: z stays child 0 (a cfg without the key draws exactly
            # as before), this is child 1, and the main emission stream is never touched, so the
            # CRN pairing against the no-rad control arm holds bout for bout.
            # Gate on KEY PRESENCE, not truthiness: _search_rad_indiv's base pass sets
            # rad_sd=0.0 and shares one rng across its fish, so both passes must consume the
            # SAME spawn count per rollout -- a truthiness gate shifts every later fish's z_re
            # to a different spawn child and the two-point solve reads z-resampling noise.
            # rad_sd=0.0 still yields u_rad_re == 0.0, i.e. a bit-identical rollout.
            u_rad_re = float(rng.spawn(1)[0].normal() * float(_re["rad_sd"]))
    # PER-FISH GATE OFFSET (rad_gate): each trajectory shifts the containment gate u0 by a fixed
    # draw -- a personal HOME RADIUS, moving where wall avoidance engages rather than how hard.
    # Built because the multiplicative swing (u_rad/rad_sd) is measured INERT on graded bulk=0
    # biases: scaling a bias that only acts in the outer ring cannot move per-fish MEAN r/R,
    # while moving the gate 0.60->0.45 shifts it by ~0.03. Same CRN contract as rad_sd: spawned
    # stream (never the main one), key-presence gated so the calibrator's base pass stays paired.
    u_gate = 0.0
    _rg = cfg_feat.get("rad_gate")
    if _rg is not None and "sd" in _rg:
        u_gate = float(rng.spawn(1)[0].normal() * float(_rg["sd"]))
    # SLOW TIMING STATE. Unlike every latent above it, this one EVOLVES: s_t = phi s_{t-1} +
    # sqrt(1-phi^2) eps, entering the interval head's location as mu_tau + lam*s. That is the
    # whole point -- a per-trajectory constant shifts every window equally and so contributes
    # nothing to rate_acf_1min / quiesc20 / dt dispersion, which is exactly what the fishre latent
    # cannot fix. Drawn from a SPAWNED stream (same contract as rad_indiv/rad_gate) so a control
    # arm without the key consumes the identical main-stream variates and stays CRN-paired.
    _dtl = cfg_feat.get("dt_latent")
    s_lam = s_phi = 0.0
    s_state = 0.0
    s_rng = None
    if _dtl:
        s_lam = float(_dtl["lam"]); s_phi = float(_dtl["phi"])
        s_rng = rng.spawn(1)[0]
        s_state = float(s_rng.normal())          # stationary N(0,1), matching tauchen's prior
    # DRIFT. The same AR(1) machinery as dt_latent, but loading on ALL FOUR channels -- turn logit,
    # distance location, interval location, and the containment-bias scale. The last one is why this
    # exists: every previous individuality mechanism is static per trajectory and therefore cannot
    # move the across-window autocorrelation at all (real radial +0.377, generated -0.035). Drawn
    # from a SPAWNED stream, same contract as rad_indiv/rad_gate/dt_latent, so a control arm without
    # the key consumes identical main-stream variates and stays CRN-paired.
    _dr = cfg_feat.get("drift")
    d_turn = d_dist = d_dt = d_bord = 0.0
    d_dist_c = d_dt_c = 0.0
    d_phi = 0.0
    d_state = 0.0
    d_rng = None
    if _dr:
        if _dtl:
            raise ValueError("drift and dt_latent both set: two AR(1) states competing for the "
                             "same variance. drift already loads on the interval head.")
        # `gain` scales the LIKELIHOOD-FITTED loadings so the rollout search can put each channel's
        # across-window acf on target; `dist_c`/`dt_c` are the matching re-centring constants.
        # THE RE-CENTRING IS NOT OPTIONAL, for exactly the reason apply_loading_gain documents: the
        # kinematic heads are log-normal, so a zero-mean state of unit variance carries into the
        # ARITHMETIC mean as E[exp(d s)] = exp(d^2/2). Training absorbs that at the fitted d; scaling
        # d -> g d afterwards inflates every generated travel and interval by exp((g^2-1) d^2/2)
        # with nothing to cancel it -- which is how the fishre gain destroyed dist_w1 (3.04 -> 6.78)
        # the first time. The turn loading needs no analogue: it tilts a logit, so the softmax bounds
        # it and it cannot inflate a marginal without bound.
        # PER-CHANNEL gains. A single shared gain cannot serve both kinematic channels: measured on
        # rej, dt and distance reach their targets at different loadings, and the one gain that put
        # dt on target left distance at +0.118 against +0.253. They are near-independent (each
        # channel's across-window acf responds to its own loading), so they are searched separately.
        d_turn = float(_dr.get("turn", 0.0))
        d_dist = float(_dr.get("dist", 0.0)) * float(_dr.get("g_dist", 1.0))
        d_dt = float(_dr.get("dt", 0.0)) * float(_dr.get("g_dt", 1.0))
        d_bord = float(_dr.get("bord", 0.0))
        d_dist_c = float(_dr.get("dist_c", 0.0)); d_dt_c = float(_dr.get("dt_c", 0.0))
        d_phi = float(_dr["phi"])
        d_rng = rng.spawn(1)[0]
        d_state = float(d_rng.normal())          # stationary N(0,1), matching the chain's prior
    body_angle, dist_params, dt_params = _numpy_predictor_joint(model, z=z_re)
    cx, cy, R = rec.dish_cx, rec.dish_cy, rec.dish_R
    # Initial heading: in psi_ref='travel' the whole rollout runs in the travel frame, so seed it
    # with theta_{-1} (exact from the first bout's dtheta), NOT f_ori -- 10.3% of bouts carry an
    # uncorrected 180-deg body-axis flip, which would start one rollout in ten facing backwards.
    h0 = (float(MB._travel_ref(rec)[0]) if cfg_feat.get("psi_ref") == "travel"
          else float(rec.f_ori_rad[0]))
    x, y, h = float(rec.x_px[0]), float(rec.y_px[0]), h0
    dth = np.zeros(n); dist = np.zeros(n); dtg = np.zeros(n)
    xs = np.empty(n); ys = np.empty(n)
    ld_hist = np.zeros(n); lt_hist = np.zeros(n)
    # Running wall-clock. t_now is ALWAYS the onset time of the bout about to be generated, so it
    # is what a stimulus programme must be sampled at. Costs one float add per bout and consumes
    # no RNG, so the default path is unaffected.
    t_now = 0.0; t_onset = np.zeros(n); n_done = n
    wall_stats = {}
    Xrec = (np.zeros((n, int(model.body[0].in_features)), dtype=np.float32)
            if keep_features else None)
    # POLICY-GRADIENT HANDLES (keep_features only; pure recording, no RNG consumed, so the default
    # path stays bit-identical). A rollout-aware objective must differentiate the log-density of the
    # value the emission ACTUALLY drew, and that is NOT what the sample dict reports: ``dtheta`` is
    # post-containment-bias, ``distance_px`` is post-vigor, ``dt_s`` is post-vigor AND post-dt_snap.
    # Scoring those evaluates the mixture at a point it never sampled -- a silently wrong score
    # function, which is one of the defects that made rollout_finetune.py unusable.
    # See scripts/rollout_aware.py.
    dth_base = np.zeros(n) if keep_features else None
    dist_base = np.zeros(n) if keep_features else None
    dt_base = np.zeros(n) if keep_features else None
    s_rec = np.zeros(n) if keep_features else None
    d_rec = np.zeros(n) if keep_features else None
    cb_cfg = cfg_feat.get("centre_bias", 0.0)
    # SHALLOW-BORDER REJECTION (key absent => rd0 is None => the loop below is byte-identical to
    # every archived arm, which is the only bit-identity guarantee offered: rejection consumes a
    # VARIABLE number of variates, so a reject arm can never be CRN-paired bout-for-bout against
    # a non-reject one. The ladder pairs at replicate level, which is unaffected.)
    _wr = cfg_feat.get("wall_reject")
    rd0 = rd1 = None
    rmax = 0
    R_g = R                    # radius the GUARDRAIL acts at (the border FEATURES always use R)
    if _wr:
        rd0 = float(_wr.get("d0", MB.WALL_REJECT_DEFAULT["d0"]))
        rd1 = float(_wr.get("d1", MB.WALL_REJECT_DEFAULT["d1"]))
        rmax = max(int(_wr.get("max_draws", MB.WALL_REJECT_DEFAULT["max_draws"])), 1)
        # THE HARD FLOOR. Rejection SHAPES the distribution across the ramp band, but on its own
        # it can only make d < d0 improbable, not impossible -- exhaust the draw budget and
        # something has to be emitted. So the guardrail moves inward to the forbidden-ring
        # boundary and does the same job there it already does at the physical wall: 'reflect'
        # preserves the step-length marginal and leaves the heading inward, unlike a clip.
        # Note the FEATURES still see the true R: the fish's distance to the real wall is real,
        # and only its reachable set is smaller.
        R_g = R * (1.0 - rd0 / (0.5 * MB.DISH_MM))
        _u0 = np.hypot(x - cx, y - cy) / R_g
        if _u0 > 1.0:
            # Seeded inside the ring (real fish do reach 0.5 mm on ~0.02% of bouts, and those
            # are the tracking-glitch bouts anyway). Pull the seed onto the boundary, or
            # _ray_to_wall returns 0 and the trajectory freezes in place for the whole rollout.
            x = cx + (x - cx) / _u0 * (1.0 - 1e-9)
            y = cy + (y - cy) / _u0 * (1.0 - 1e-9)
    # KIN_FEEDBACK: shrink the kinematic history ON READ. The block is z-scored (mean 0), so a
    # factor < 1 pulls every lag toward the population mean, which rescales the effective AR
    # coefficient of the free-running loop and damps the exposure-bias over-persistence.
    # NOTE this model differs from the kin family: those columns feed the SHARED body, so the
    # shrinkage reaches the turn head too. Watch the turn panel metrics, not just the kinematic ones.
    # LAG-VALIDITY and BLOCK-CLOCK columns, appended by MB.build_feature_matrix AFTER the kin
    # block. Both are pure functions of the bout index and the clock -- not of the generated
    # behaviour -- so they are precomputed here rather than rebuilt per bout.
    #   lag_valid: a rollout is ONE run, so run_pos == t and the scalar is min(t, K)/K.
    #   clock:     time into the current stimulus block on the fixed 110 s grid, 0 during a
    #              pause, matching MB._clock_block. exp(-t/tau) or the 0->1 ramp.
    _lv_mode = str(cfg_feat.get("lag_valid", "") or "").lower()
    _clk_mode = str(cfg_feat.get("clock_mode", "") or "").lower()
    _lagv_a = None
    if _lv_mode == "scalar":
        _lagv_a = (np.minimum(np.arange(n), K) / max(K, 1)).astype(np.float32).reshape(n, 1)
    elif _lv_mode == "mask":
        _lagv_a = (np.arange(n)[:, None] >= np.arange(1, K + 1)[None, :]).astype(np.float32)
    _clk_tau = float(cfg_feat.get("clock_tau_s", MB.CLOCK_TAU_S))

    def _clock_of(t_s):
        """Block clock at wall-clock ``t_s``. NOT precomputable: t_now advances by the
        GENERATED dt, so where a bout falls in the 110 s cycle is a consequence of the
        rollout's own timing. 0 during a pause, matching MB._clock_block."""
        if _clk_mode not in ("linear", "exp"):
            return None
        pos = int(round(float(t_s) * MB._FPS)) % 3300
        if pos < 1500:                                   # inter-stimulus pause
            return np.zeros((1,), dtype=np.float32)
        tb = (pos - 1500) / MB._FPS
        v = (min(tb / MB.CLOCK_BLOCK_S, 1.0) if _clk_mode == "linear"
             else float(np.exp(-tb / _clk_tau)))
        return np.array([v], dtype=np.float32)
    _kfb = cfg_feat.get("kin_feedback", 1.0)
    kfb_d, kfb_t = ((float(_kfb["dist"]), float(_kfb["dt"])) if isinstance(_kfb, dict)
                    else (float(_kfb), float(_kfb)))
    # VIGOR: per-trajectory log-modulation of the emitted step / interval. u = a fixed intercept
    # (individuality), d = a slow stationary AR(1) (tempo drift). Applied OUTSIDE the loop -- the
    # history keeps the unmodulated value, or the mean-reverting body would push back against z
    # and it would compound. -var/2 keeps E[exp(z)] = 1 so the population mean is unchanged.
    # INDIV: per-trajectory individuality for the two channels vigor does not cover.
    # u_fw tilts the FORWARD mixture weight -- changing how OFTEN the fish turns, not how hard,
    # which is the axis real fish actually vary on (forward fraction spans 4.5x across the 15,
    # median |dtheta| only 2x) and which leaves the pooled turn marginal alone.
    # u_rad SCALES the containment bias. An additive offset was tried and is wrong twice over:
    # centre_bias steers by adding cb*sin(psi) to dtheta so an offset leaks into the turn
    # marginal, and a negative draw flips a fish OUTWARD and triples the wall fraction.
    _iv = cfg_feat.get("indiv"); u_fw = u_rad = 0.0
    if _iv:
        u_fw = float(rng.normal(0.0, float(_iv.get("fw", 0.0))))
        u_rad = float(rng.normal(0.0, float(_iv.get("rad", 0.0))))
    # INNOV_COMP: widen the innovation about THIS bout's conditional mean, restoring the spread
    # kin_feedback removes. For a linear AR, scaling the innovation multiplies the stationary
    # variance while leaving acf(k)=phi^k untouched -- so lambda sets the acf and c the spread.
    # NOT kin_var_comp, which rescaled about the fixed TRAINING log-mean with the history left
    # unscaled: a pure output transform that widened the marginal without restoring the loop.
    _kin = cfg_feat.get("kin_innov")
    ci_d = float(_kin.get("dist", 1.0)) if _kin else 1.0
    ci_t = float(_kin.get("dt", 1.0)) if _kin else 1.0
    vg = cfg_feat.get("vigor"); z_di = z_dt = 0.0
    if vg:
        su_di, sd_di, ph_di, var_di = vg["dist"]; su_dt, sd_dt, ph_dt, var_dt = vg["dt"]
        c_di = float(np.sqrt(1.0 - ph_di * ph_di)); c_dt = float(np.sqrt(1.0 - ph_dt * ph_dt))
        u_di = rng.normal(0.0, su_di); u_dt = rng.normal(0.0, su_dt)
        d_di = rng.normal(0.0, sd_di); d_dt = rng.normal(0.0, sd_dt)

    # EMISSION ORDER. Read off the net, so a checkpoint generates in the order it was trained in
    # and an old one (no attribute) keeps the archived chain. When the angle head is first it takes
    # no conditioners, so the body+angle forward pass stays OUTSIDE the redraw loop exactly as it
    # was -- the shallow-border rejection then still costs no matmul per redraw.
    _ORDER = parse_order(getattr(model, "order", None))
    _a_first = _ORDER[0] == "a"
    _body_fn = getattr(body_angle, "body", None)
    _angle_fn = getattr(body_angle, "angle_params", None)
    if not _a_first and (_body_fn is None or _angle_fn is None):
        raise RuntimeError(f"emission order {_ORDER} needs a conditioned angle head, but this "
                           "numpy predictor only exposes the unconditional one")

    def _adj_w(w):
        """The mixture weights after the two forward-weight tilts. Both multiply the FIRST weight
        and renormalise, which is exactly the torch path's `out[:, :1] + shift` then softmax --
        doing it on the weights keeps the numpy predictor's single forward pass intact."""
        if u_fw:
            w = np.array([w[0] * float(np.exp(u_fw)), w[1], w[2], w[3], w[4]]); w = w / w.sum()
        if d_turn:
            w = np.array([w[0] * float(np.exp(d_turn * d_state)), w[1], w[2], w[3], w[4]])
            w = w / w.sum()
        return w

    for t in range(n):
        xs[t], ys[t] = x, y
        if use_border and closed_loop:
            r = np.hypot(x - cx, y - cy); psi = MB._wrap_pi(np.arctan2(cy - y, cx - x) - h)
            border_t = MB.border_channels((R - r) / R, psi, cfg_feat.get("border_mode", "raw"))[0]
            assert border_t.shape[0] == tf_border.shape[1], "border row width != block width"
        else:
            border_t = tf_border[t]
        if recompute_stim:
            if stim_provider is None:
                dcx, dcy, mv_t, nd_t = dot_cx_a[t], dot_cy_a[t], mv_a[t], nd_a[t]
            else:                            # resolve the programme from the live clock + pose
                dcx, dcy, mv_t, nd_t = stim_provider(t, t_now, x, y, h)
                dcx = float(dcx); dcy = float(dcy)
                mv_t = float(mv_t); nd_t = float(nd_t)
            dcx_used[t] = dcx; dcy_used[t] = dcy
            if np.isfinite(dcx):
                phi_gen[t] = MB._wrap_pi(np.arctan2(dcy - y, dcx - x) - h)
            stim_t = MX._dot_stim_row(cfg_feat["stim_mode"], dcx, dcy,
                                      x, y, h, mv_t, nd_t, dish=(cx, cy, R))
            assert stim_t.shape[0] == stim.shape[1], "closed-loop stim row width != block width"
        else:
            stim_t = stim[t]
        hist = np.zeros(K)
        for k in range(1, K + 1):
            if t - k >= 0:
                hist[k - 1] = dth[t - k]
        kin = np.zeros(2 * Kk)
        for k in range(1, Kk + 1):
            if t - k >= 0:
                kin[k - 1] = kfb_d * (ld_hist[t - k] - dm) / ds
                kin[Kk + k - 1] = kfb_t * (lt_hist[t - k] - tm) / ts
        # block order MUST match MB.build_feature_matrix:
        #   history, stim, stimhist, border, kin, lag_valid, clock
        _tail = [kin]
        if _lagv_a is not None:
            _tail.append(_lagv_a[t])
        _ck = _clock_of(t_now)
        if _ck is not None:
            _tail.append(_ck)
        row = np.concatenate([hist / dth_scale, stim_t, tf_shist[t] if tf_shist.shape[1] else _EMPTY,
                              border_t] + _tail).astype(np.float32)
        assert row.shape[0] == int(model.body[0].in_features), "joint row width != net input dim"
        if keep_features:
            Xrec[t] = row
        if _a_first:
            b, ang = body_angle(row)
        else:
            b, ang = _body_fn(row), None
        _w0 = _adj_w(ang[0]) if _a_first else None

        def _draw_a(prev):
            """dtheta | x, (whatever was emitted before it). Returns (post-bias, mixture draw)."""
            if _a_first:
                w = _w0
                sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = ang[1:]
            else:
                _a = _angle_fn(b, *prev)
                w = _adj_w(_a[0])
                sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = _a[1:]
            c = rng.choice(5, p=w)
            if c == 0:
                v = rng.normal(0.0, sF)
            elif c == 1:
                v = -rng.gamma(aL1, tL1)
            elif c == 2:
                v = -rng.gamma(aL2, tL2)
            elif c == 3:
                v = +rng.gamma(aR1, tR1)
            else:
                v = +rng.gamma(aR2, tR2)
            dth_t = float(np.clip(v, -180.0, 180.0))
            # The MIXTURE draw, before the deterministic containment shift below. This is the
            # value ``MX.mix_logdens`` is the density of; the shift is a function of the state,
            # not of the sample, so it carries no score-function term.
            dth_raw = dth_t
            cb = cb_cfg
            if isinstance(cb, dict):                      # shell-gated containment bias
                _u = np.hypot(x - cx, y - cy) / R
                _u0 = float(cb.get("u0", 0.75))
                if u_gate:
                    # clip keeps a personal gate physically sane: never inside r/R 0.20 (a bias
                    # acting over the whole dish) and never beyond 0.90 (no containment at all).
                    _u0 = float(np.clip(_u0 + u_gate, 0.20, 0.90))
                cb = (float(cb.get("bulk", 0.0))
                      + float(cb.get("shell", 0.0)) * max(0.0, (_u - _u0) / max(1.0 - _u0, 1e-6)))
            if u_rad:
                cb = float(cb) * float(np.exp(u_rad))
            if u_rad_re:
                cb = float(cb) * float(np.exp(u_rad_re))
            if d_bord:
                # THE LOAD-BEARING SHIFT: a slowly wandering radial SET POINT, so the fish's
                # preferred radius moves over minutes instead of being pinned. Real radial window
                # means autocorrelate at +0.377, generated at -0.035, and nothing static can change
                # that.
                #
                # ADDITIVE, NOT MULTIPLICATIVE, and that is measured rather than stylistic. The
                # multiplicative form (`cb * exp(d_bord * s)`, which is what u_rad/u_rad_re do) is
                # exactly INERT here: the rej family calibrates to bulk = 0 and shell = 0, carrying
                # its containment entirely in the rejection sampler, so scaling the bias scales
                # zero. Measured on rej with the multiplicative form -- radial acf1 went 0.117 ->
                # 0.036, i.e. it did nothing but add noise. Same trap already documented for
                # rad_indiv ("inert on graded bulk=0 biases").
                #
                # An additive zero-mean term is also the right shape for this job: it cancels in the
                # POOLED turn marginal (E[s] = 0) while leaving the WINDOW-level statistics moved,
                # which is precisely the split this mechanism needs. Negative values push outward;
                # the rejection sampler still enforces the physical boundary, so that is safe.
                cb = float(cb) + d_bord * d_state
            if cb:
                psi_c = MB._wrap_pi(np.arctan2(cy - y, cx - x) - h)
                dth_t = float(np.clip(dth_t + cb * np.sin(psi_c), -180.0, 180.0))
            return dth_t, dth_raw

        def _draw_d(prev):
            """distance | x, (whatever was emitted before it)."""
            dw, dmu, dsig = dist_params(b, *prev)
            j = rng.choice(len(dw), p=dw)
            # clip the 'average fish' step FIRST, then scale by the vigor -- otherwise a genuinely
            # long-stepping trajectory is squashed back into the pooled q99.9 cap and loses exactly
            # the large steps that make it long-stepping.
            _ld = rng.normal(dmu[j] + d_dist * d_state - d_dist_c, dsig[j])
            if ci_d != 1.0:
                # Re-centre on the conditional mean INCLUDING the drift shift, for the same reason
                # the interval channel does below: innov_comp widens about THIS bout's mean, and
                # with the drift on, that mean has moved.
                _m = float(np.dot(dw, dmu)) + d_dist * d_state - d_dist_c
                _ld = _m + ci_d * (_ld - _m)
            return float(np.clip(np.exp(_ld), 1e-3, dist_hi))

        def _draw_t(prev):
            """dt | x, (whatever was emitted before it)."""
            tw, tmu, tsig = dt_params(b, *prev)
            jt = rng.choice(len(tw), p=tw)
            # The timing latent shifts the LOCATION of the whole interval mixture, exactly as it
            # does in dt_level_logliks at train and score time.
            _lt = rng.normal(tmu[jt] + s_lam * s_state + d_dt * d_state - d_dt_c, tsig[jt])
            if ci_t != 1.0:
                # Re-centre on the conditional mean INCLUDING the latent shift -- innov_comp widens
                # about this bout's mean, and with the latent on that mean has moved.
                _m = float(np.dot(tw, tmu)) + s_lam * s_state + d_dt * d_state - d_dt_c
                _lt = _m + ci_t * (_lt - _m)
            return float(np.clip(np.exp(_lt), 1e-3, dt_hi))

        def _emit():
            """One draw of the chained bout triple ``(dtheta, base distance, base dt)``, emitted in
            the net's own ``order`` and each channel conditioned on the ones before it.

            Factored out ONLY so the shallow-border rejection can redraw it. On the default order
            the body forward pass has already happened above, so a redraw is a handful of rng calls
            and no matmul -- which is why a max_draws of 20 costs nothing. Nothing here depends on
            the draw index, so calling it once reproduces the original code exactly, variate for
            variate.

            The conditioners are the SAME quantities the trainer teacher-forces (``_cond_cols``):
            |dtheta| after the containment shift, and the BASE distance / interval before vigor --
            so distance individuality cannot leak into the interval channel (real between-fish corr
            of the two levels ~ 0)."""
            got, prev = {}, []
            for _i, _ch in enumerate(_ORDER):
                if _ch == "a":
                    dth_t, dth_raw = _draw_a(prev)
                    got["a"] = (dth_t, dth_raw)
                    _c = abs(dth_t) / dth_scale
                elif _ch == "d":
                    got["d"] = _draw_d(prev)
                    _c = (np.log(max(got["d"], 1e-6)) - dm) / ds
                else:
                    got["t"] = _draw_t(prev)
                    _c = (np.log(max(got["t"], 1e-6)) - tm) / ts
                if _i < len(_ORDER) - 1:
                    prev.append(_c)
            return got["a"][0], got["d"], got["t"], got["a"][1]

        def _vigor_step():
            nonlocal d_di, d_dt, z_di, z_dt
            d_di = ph_di * d_di + sd_di * c_di * rng.normal()
            d_dt = ph_dt * d_dt + sd_dt * c_dt * rng.normal()
            z_di = u_di + d_di - 0.5 * var_di
            z_dt = u_dt + d_dt - 0.5 * var_dt

        if rd0 is None:
            dth_t, base_di, base_dt, dth_raw = _emit()
            if vg:
                _vigor_step()
        else:
            # SHALLOW-BORDER REJECTION. Redraw the whole triple (the heads are chained, so
            # keeping the turn and redrawing only the step would sample from a different law)
            # until the LANDING POINT is accepted by MB.accept_prob.
            #
            # This terminates. The dish is convex, so heading straight inward always gives the
            # largest reachable wall distance d + L, and a draw therefore succeeds whenever
            # L >= d0 - d; with d0 = 2 mm = 7.2 px and a median real step of 15.0 px, 95% of
            # steps clear that even from ON the wall, where roughly half the heading circle is
            # still available. There is no corner to be trapped in. The ramp is also inactive for
            # ~95% of bouts (only 0.07% of real bouts are inside 4 mm), so the expected cost is
            # ~1 draw. All of which is INSTRUMENTED below rather than assumed.
            #
            # The vigor AR state advances ONCE per bout, not once per draw -- otherwise a rejected
            # bout would silently age the tempo latent.
            if vg:
                _vigor_step()
            _exp_di = float(np.exp(z_di))
            _best = None; _best_u = np.inf
            for _k in range(rmax):
                dth_t, base_di, base_dt, dth_raw = _emit()
                _hh = h + np.radians(dth_t)
                _dv = base_di * _exp_di
                _ul = np.hypot(x + _dv * np.cos(_hh) - cx, y + _dv * np.sin(_hh) - cy) / R
                _a = MB.accept_prob(_ul, rd0, rd1)
                if _a >= 1.0 or rng.random() < _a:
                    break
                if _best is None or _ul < _best_u:     # remember the least-bad candidate
                    _best_u = _ul; _best = (dth_t, base_di, base_dt, dth_raw)
            else:
                # BUDGET EXHAUSTED -> take the BEST of the draws, not the last one. Keeping the
                # last was the first implementation and it is strictly worse: it throws away up
                # to 19 better candidates for no reason, and it was measured putting bouts
                # 0.011 mm from the wall on a deliberately under-trained model. The hard floor is
                # then still guaranteed by the guardrail, which runs at R_g (the forbidden-ring
                # boundary) rather than the physical wall -- see below.
                dth_t, base_di, base_dt, dth_raw = _best
                wall_stats["n_reject_fallback"] = wall_stats.get("n_reject_fallback", 0) + 1
            wall_stats["n_draws"] = wall_stats.get("n_draws", 0) + _k + 1
            if _k:
                wall_stats["n_redraw_bouts"] = wall_stats.get("n_redraw_bouts", 0) + 1
        dth[t] = dth_t
        if keep_features:
            # s_rec[t] is the state that shifted THIS bout's interval mixture, recorded before the
            # AR step below advances it -- scoring against the advanced value would put every dt
            # log-density one bout out of phase.
            dth_base[t] = dth_raw; dist_base[t] = base_di
            dt_base[t] = base_dt; s_rec[t] = s_state; d_rec[t] = d_state
        dval = base_di * float(np.exp(z_di)); tval = base_dt * float(np.exp(z_dt))
        if cfg_feat.get("dt_snap"):
            # EMITTED value only; ld_hist/lt_hist keep the continuous base below, so the loop
            # dynamics, the RNG stream and every other channel stay byte-identical to the
            # unsnapped control -- the contrast is attributable to the measurement operator alone.
            tval = MB.snap_dt(tval)
        dist[t] = dval; dtg[t] = tval
        t_onset[t] = t_now; t_now += tval          # t_now is now bout t+1's onset time
        # history keeps the BASE: the body must stay in average-fish world (see the vigor note above)
        # -- note base_dt ALREADY carries the timing latent's shift, and that is deliberate. vigor
        # was a post-hoc multiplier the body was never trained under, so its history had to stay
        # unmodulated; lam*s is inside the fitted likelihood, i.e. the model explains the REAL
        # interval as mu + lam*s, so the emitted interval is what _kin_history_block saw at train
        # time. Stripping it here would train one object and roll out another.
        ld_hist[t] = np.log(max(base_di, 1e-6)); lt_hist[t] = np.log(max(base_dt, 1e-6))
        if _dtl:                                  # advance ONCE per bout, never per reject-redraw
            s_state = s_phi * s_state + np.sqrt(max(1.0 - s_phi * s_phi, 0.0)) * s_rng.normal()
        if _dr:                                   # same contract: one step per BOUT, not per draw
            d_state = d_phi * d_state + np.sqrt(max(1.0 - d_phi * d_phi, 0.0)) * d_rng.normal()
        h = h + np.radians(dth[t])
        if guardrail:
            x, y, h = MB.advance(x, y, h, dval, cx, cy, R_g, eps,
                                 mode=cfg_feat.get("wall_mode", "clip"), stats=wall_stats)
        else:
            x = x + dval * np.cos(h); y = y + dval * np.sin(h)
        if stop_time_s is not None and t_now >= stop_time_s:
            # The test is on the NEXT onset, so the last kept bout is the last one starting
            # strictly inside the programme; the bout that would start past the end (and whose
            # stimulus lookup would be out of range) is never generated. The realised end
            # therefore undershoots stop_time_s by up to one inter-bout interval.
            n_done = t + 1
            break
    if n_done < n:                                  # truncate every per-bout array together
        dth, dist, dtg = dth[:n_done], dist[:n_done], dtg[:n_done]
        xs, ys, t_onset = xs[:n_done], ys[:n_done], t_onset[:n_done]
        if recompute_stim:
            phi_gen = phi_gen[:n_done]; dcx_used = dcx_used[:n_done]; dcy_used = dcy_used[:n_done]
        if keep_features:
            Xrec = Xrec[:n_done]
            dth_base = dth_base[:n_done]; dist_base = dist_base[:n_done]
            dt_base = dt_base[:n_done]; s_rec = s_rec[:n_done]; d_rec = d_rec[:n_done]
    out = dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth), dt_s=dtg, distance_px=dist,
               x_px=xs, y_px=ys,
               n_hits=wall_stats.get("n_hit", 0), n_bounces=wall_stats.get("n_bounce", 0))
    if rd0 is not None:
        # Instrumented, not assumed. An uninstrumented quantity a downstream calibrator can
        # absorb is exactly the shape of the kin_feedback trap, so count the draws.
        out["n_draws"] = wall_stats.get("n_draws", 0)
        out["n_redraw_bouts"] = wall_stats.get("n_redraw_bouts", 0)
        out["n_reject_fallback"] = wall_stats.get("n_reject_fallback", 0)
    if recompute_stim:
        out["stim_angle_gen"] = phi_gen
    if stim_provider is not None or stop_time_s is not None:
        # Tool-mode extras only -- the default sample dict keeps exactly the keys it always had.
        out["t_onset_s"] = t_onset
        out["end_time_s"] = float(t_now)
        if recompute_stim:
            out["dot_cx"] = dcx_used; out["dot_cy"] = dcy_used
    if keep_features:
        out["X"] = Xrec
        out["dth_base"] = dth_base            # the mixture draw, pre-containment-bias
        out["dist_base"] = dist_base          # pre-vigor
        out["dt_base"] = dt_base              # pre-vigor, pre-dt_snap
        out["s_state"] = s_rec                # timing-latent value that shifted each dt mixture
        out["d_state"] = d_rec                # drift value that shifted all four channels
        # THE LATENT THAT ACTED. joint_forward_re(..., z) has to be evaluated here, not at z=0:
        # the old fine-tuner scored at z=0 and therefore computed a valid-looking gradient for a
        # DIFFERENT policy than the one that produced the actions.
        out["z_re"] = (np.zeros(0) if z_re is None else np.asarray(z_re, float))
    return out


# =========================================================================== #
#  Adapter
# =========================================================================== #
class MLPMixJointDensity(MX.MLPMixLoopDensity):
    """Joint-likelihood model with a single feature vector and chained heads.

    ``score`` reports THREE channels, which is the point: the kin family optimises distance and dt
    against a likelihood it never reports.
    """

    def __init__(self, K=8, K_kin=3, stim_mode="shoal", include_symbols=False, hidden=128,
                 epochs=80, lr=1e-3, patience=8, batch_size=256, seed=0,
                 use_border=True, border_mode="raw", wall_mode="clip",
                 closed_loop=True, guardrail=True, eps=1e-3, stim_loop=True,
                 n_dist=2, n_dt=3, contain=False, contain_fish=12, contain_w_occ=2.6,
                 vigor=False, vigor_drift_calib=False, kin_fb_calib=False,
                 indiv=False, innov_comp=False,
                 # travel frame by DEFAULT for new models: the body axis sits a median
                 # 40.6 deg from the travel heading and a rollout has no body axis to
                 # query. The load path keeps its own "ori" fallback, so a checkpoint
                 # saved without these keys still reloads in the frame it was trained in.
                 dt_snap=False, psi_ref="travel", phi_ref="travel",
                 # Which of the six exact factorisations of the joint to fit. See the module
                 # docstring: same density, different capacity per channel.
                 head_order="a,d,t",
                 K_stim=1, noise_aug=None, contain_mode="bias",
                 # Escape hatch for cfg_feat keys that have no constructor arg of their own
                 # (clock_mode, clock_tau_s, lag_valid, use_adapt, ...). Merged LAST in fit,
                 # so it can also override a computed default -- and it is saved with the
                 # checkpoint, so a reloaded model rebuilds the same feature matrix.
                 cfg_feat_extra=None, **kwargs):
        super().__init__(K=K, stim_mode=stim_mode, include_symbols=include_symbols, hidden=hidden,
                         epochs=epochs, lr=lr, patience=patience, batch_size=batch_size, seed=seed,
                         use_border=use_border, border_mode=border_mode, wall_mode=wall_mode,
                         closed_loop=closed_loop, guardrail=guardrail, eps=eps,
                         stim_loop=stim_loop, **kwargs)
        self.K_kin = int(K_kin)
        self.n_dist, self.n_dt = int(n_dist), int(n_dt)
        self.contain = bool(contain)
        self.contain_fish, self.contain_w_occ = int(contain_fish), float(contain_w_occ)
        # Ported from the kin lineage so the treatments can be compared like-for-like.
        # vigor      = per-trajectory intercept (+ optional slow AR drift) on the EMITTED step/interval.
        # kin_fb_calib = shrink the kinematic history on read, damping the free-run over-persistence.
        self.vigor, self.vigor_drift_calib = bool(vigor), bool(vigor_drift_calib)
        self.kin_fb_calib = bool(kin_fb_calib)
        # indiv      = per-trajectory turning style + spatial preference (the channels vigor misses)
        # innov_comp = restore the spread kin_feedback removes, without disturbing the acf
        self.indiv, self.innov_comp = bool(indiv), bool(innov_comp)
        # dt_snap  = quantise the EMITTED interval to the 1/30 s camera lattice (MB.snap_dt)
        # psi_ref  = angular reference of the border bearing: 'ori' (archived behaviour, raw body
        #            orientation) or 'travel' (previous travel heading -- the frame the closed
        #            loop actually queries). phi_ref = the same choice for the dot bearing.
        self.dt_snap = bool(dt_snap)
        self.psi_ref, self.phi_ref = str(psi_ref), str(phi_ref)
        self.head_order = parse_order(head_order)
        self.cfg_feat_extra = dict(cfg_feat_extra or {})
        # K_stim = how many stimulus ONSETS the net sees (1 = the current snapshot only, i.e.
        #          every archived arm; 3 = plus the two previous presentations). MB._stimhist_block.
        # noise_aug = pose-perturbed training pairs with the target corrected back to the truth
        #          (dict: sigma_pos px, sigma_ori deg, mode 'concat'|'mask', frac). None = off.
        # contain_mode = 'bias' (the archived shell-gated centre_bias) or 'reject' (shallow-border
        #          rejection replaces the SHELL ramp; the bulk term stays a fitted knob with 0 in
        #          its grid, so the search decides whether any turn bias is needed at all).
        self.K_stim = int(K_stim)
        self.noise_aug = dict(noise_aug) if noise_aug else None
        self.contain_mode = str(contain_mode)
        self.use_kin = True

    def fit(self, train_records, ctx):
        c = ctx.cfg
        self.K = int(c.get("mlp_K", self.K))
        self.stim_mode = c.get("mlp_stim_mode", self.stim_mode)
        self.hidden = int(c.get("mlp_hidden", self.hidden))
        # epochs_override / patience_override are how colab_run_ladder's --force-epochs runs a PAIRED
        # budget contrast inside one select. They were read only from ctx.cfg here, so they never
        # reached any joint rung and a budget ladder silently ran both arms at the same budget.
        # Same precedence as MLPMixKinDensity.fit (fit_MLP_mix.py:1552).
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
        # use_kin=True is the whole point: ONE x carrying turn history, stimulus, border AND
        # kinematic history. joint=True tells the shared calibrators which generator to use.
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
        self._cfg_feat.update(self.cfg_feat_extra)      # last word, and saved with the blob
        X = np.concatenate([MB.build_feature_matrix(r, self._cfg_feat) for r in train_records], 0)
        y_ang = np.concatenate([r.dtheta for r in train_records])
        y_dist = np.concatenate([np.asarray(r.distance_px, float) for r in train_records])
        y_dt = np.concatenate([np.asarray(r.dt_s, float) for r in train_records])
        self._model, self.history = fit_mix_joint(
            X, y_ang, y_dist, y_dt, self.hidden, self._dth_scale, self._kin_scales,
            epochs=self.epochs, lr=self.lr, patience=self.patience,
            batch_size=self.batch_size, seed=self.seed, n_dist=self.n_dist, n_dt=self.n_dt,
            order=self.head_order, **_noise_kwargs(self, train_records))
        # SAME ORDER as MLPMixKinDensity.fit, so the treatments are comparable to the lineage:
        # vigor -> drift rescale -> kin_feedback -> containment. (The known ordering quirk that the
        # drift is sized before the shrinkage is preserved deliberately; its impact here is small
        # because these models have almost no window-level drift of their own to subtract.)
        if self.vigor:
            vg = MX._calibrate_vigor(train_records)
            vg["apply_turn"] = False                     # turn individuality is a separate mechanism
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
                max_fish=self.contain_fish, w_occ=self.contain_w_occ))
        # innov BEFORE indiv: it changes the per-bout spread that the individuality search then
        # measures on top of. Both AFTER contain, then contain is RE-FIT, because both widen the
        # model and movement spread drives wall contact (corr +0.59) while rad_frac_wall is the
        # top C2ST feature. Not re-fitting is how contain_vc lost.
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
        if self.contain and (self.indiv or self.innov_comp):
            self._cfg_feat.update(MX._search_wall(
                self._model, train_records, self._cfg_feat, mode=self.contain_mode,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed + 1,
                max_fish=min(self.contain_fish, 8), w_occ=self.contain_w_occ))
        self.config = dict(mode="joint", hidden=self.hidden, n_features=X.shape[1], K=self.K,
                           K_kin=self.K_kin, stim_mode=self.stim_mode, use_border=self.use_border,
                           use_kin=True, n_dist=self.n_dist, n_dt=self.n_dt,
                           head_order=order_tag(self.head_order))
        return self

    def score(self, rec):
        """THREE channels. ``logdens`` keeps the angle meaning so every existing consumer is
        unaffected; the harness picks up the two extra keys and reports them separately.

        Units differ per channel (nats per degree / pixel / second), so the three numbers are not
        comparable to EACH OTHER in absolute level -- only across models within a channel.
        """
        X = MB.build_feature_matrix(rec, self._cfg_feat)
        a_z, ld_c, lt_c = _cond_cols(rec.dtheta, rec.distance_px, self._cfg_feat["dth_scale"],
                                     self._cfg_feat["kin_scales"], y_dt=rec.dt_s)
        with torch.no_grad():
            Xt = torch.tensor(X)
            ang, dpar, tpar = self._model.joint_forward(Xt, torch.tensor(a_z), torch.tensor(ld_c),
                                                        torch.tensor(lt_c))
            logdens = MX.mix_logdens(torch.tensor(rec.dtheta.astype(np.float32)), ang).numpy()
            yd = torch.tensor(np.asarray(rec.distance_px, np.float32))
            yt = torch.tensor(np.asarray(rec.dt_s, np.float32))
            ld = MX._lnk_logpdf(torch.clamp(yd, min=1e-6), *dpar).numpy()
            lt = MX._lnk_logpdf(torch.clamp(yt, min=1e-6), *tpar).numpy()
        # Mask EXACTLY the bouts the training loss masks (`isfinite & > 0`), so the reported
        # likelihood is over the same support the model was fitted on. This is not cosmetic: 10
        # of the 70590 bouts have distance exactly 0, and a log-normal evaluated at the 1e-6 clamp
        # returns about -850 nats. Ten such bouts would move a fish's bits/bout by ~0.26 bits --
        # small enough to look like a real effect and entirely an artefact.
        dv = np.asarray(rec.distance_px, float); tv = np.asarray(rec.dt_s, float)
        ld = np.where(np.isfinite(dv) & (dv > 0), ld, np.nan)
        lt = np.where(np.isfinite(tv) & (tv > 0), lt, np.nan)
        # Symbol PMF from the angle parameters that were just computed, rather than a second
        # unconditional forward pass. On the default order those are the SAME numbers (the angle
        # head reads nothing but the body), so this is bit-identical to the old
        # `predict_params_mix` call -- but on an order where dtheta is emitted last, the
        # unconditional head does not exist and the plug-in would be silently wrong.
        params = (np.exp(ang[0].numpy()),) + tuple(z.numpy().astype(float) for z in ang[1:])
        return dict(logdens=logdens, symbol_logq=MX.symbol_logq_mix(*params),
                    logdens_dist=ld, logdens_dt=lt)

    def sample(self, rec, rng, **kw):
        # **kw forwards the tool-mode knobs (stim_provider / stop_time_s / keep_features); the
        # harness only ever calls sample(rec, rng), so the adapter contract is unchanged.
        return generate_mix_joint_loop(self._model, rec, self._cfg_feat, rng,
                                       closed_loop=self.closed_loop, guardrail=self.guardrail,
                                       eps=self.eps, stim_loop=self.stim_loop, **kw)


def save_joint(model, path):
    """Persist a fitted joint model. Mirrors ``MX.save_mix``; kept separate because the net has a
    different head layout and ``load_mix`` rebuilds from ``config``."""
    if not hasattr(model, "_model"):
        raise RuntimeError("model is not fitted; call .fit(...) before save_joint")
    torch.save(dict(cls="MLPMixJointDensity", state_dict=model._model.state_dict(),
                    n_features=int(model.config["n_features"]), hidden=int(model.config["hidden"]),
                    cfg_feat=model._cfg_feat, config=model.config, name=model.name,
                    dt_pool=np.asarray(model._dt_pool), dist_pool=np.asarray(model._dist_pool),
                    history=getattr(model, "history", None),
                    K=model.K, K_kin=model.K_kin, stim_mode=model.stim_mode,
                    include_symbols=model.include_symbols, use_border=model.use_border,
                    border_mode=model.border_mode, wall_mode=model._cfg_feat.get("wall_mode", "clip"),
                    n_dist=model.n_dist, n_dt=model.n_dt,
                    head_order=order_tag(getattr(model, "head_order", DEFAULT_ORDER)),
                    vigor=model.vigor, vigor_drift_calib=model.vigor_drift_calib,
                    kin_fb_calib=model.kin_fb_calib,
                    indiv=model.indiv, innov_comp=model.innov_comp,
                    K_stim=int(getattr(model, "K_stim", 1)),
                    noise_aug=getattr(model, "noise_aug", None),
                    contain_mode=str(getattr(model, "contain_mode", "bias")),
                    loop=dict(closed_loop=model.closed_loop, guardrail=model.guardrail,
                              eps=model.eps, stim_loop=model.stim_loop)), path)
    return path


def load_joint(path, map_location="cpu"):
    """Reload a joint model saved by ``save_joint``, ready to ``.sample`` / ``.score``."""
    MX._numpy2_unpickle_compat()
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    lp = blob["loop"]
    m = MLPMixJointDensity(K=blob["K"], K_kin=blob["K_kin"], stim_mode=blob["stim_mode"],
                           include_symbols=blob["include_symbols"], hidden=blob["hidden"],
                           use_border=blob["use_border"], border_mode=blob["border_mode"],
                           wall_mode=blob.get("wall_mode", "clip"),
                           closed_loop=lp["closed_loop"], guardrail=lp["guardrail"],
                           eps=lp["eps"], stim_loop=lp["stim_loop"],
                           n_dist=blob["n_dist"], n_dt=blob["n_dt"],
                           # absent on every pre-ordering checkpoint, which all used the default
                           head_order=blob.get("head_order", DEFAULT_ORDER))
    net = MDNMixJoint(blob["n_features"], blob["hidden"], n_dist=blob["n_dist"], n_dt=blob["n_dt"],
                      order=m.head_order)
    net.load_state_dict(blob["state_dict"]); net.eval()
    m._model = net
    m._cfg_feat = blob["cfg_feat"]
    m._rich_scales = blob["cfg_feat"].get("rich_scales")
    m._dt_pool = blob["dt_pool"]; m._dist_pool = blob["dist_pool"]
    m.config = blob["config"]; m.name = blob["name"]; m.history = blob.get("history")
    m._dth_scale = blob["cfg_feat"]["dth_scale"]; m._kin_scales = blob["cfg_feat"]["kin_scales"]
    # Restore the treatment flags (mirrors load_fishre). Generation/scoring read _cfg_feat and
    # are correct either way, but a re-save or a refit of the loaded object reads THESE -- a
    # psit/dt_snap checkpoint refit without them would silently train an 'ori'-frame, unsnapped
    # model under the same name.
    for _f in ("vigor", "vigor_drift_calib", "kin_fb_calib", "indiv", "innov_comp"):
        if _f in blob:
            setattr(m, _f, bool(blob[_f]))
    m.contain = m._cfg_feat.get("centre_bias") is not None
    m.dt_snap = bool(m._cfg_feat.get("dt_snap", False))
    m.psi_ref = str(m._cfg_feat.get("psi_ref", "ori"))
    m.phi_ref = str(m._cfg_feat.get("phi_ref", "ori"))
    # Read the new treatments from cfg_feat first (it is the object generation actually uses) and
    # fall back to the top-level keys, so a checkpoint written before these existed still loads.
    m.K_stim = int(m._cfg_feat.get("K_stim", blob.get("K_stim", 1)))
    m.noise_aug = m._cfg_feat.get("noise_aug", blob.get("noise_aug"))
    m.contain_mode = str(blob.get("contain_mode",
                                  "reject" if m._cfg_feat.get("wall_reject") else "bias"))
    return m
