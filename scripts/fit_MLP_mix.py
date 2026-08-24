"""Richer-emission MLP: mixture of 2 Gammas per turn direction.

c2st_importances showed the residual gap on the best model is the turn-MAGNITUDE
distribution (|dtheta| median/std/q90/mean). A single Gamma per side can't fit the
body AND the tail at once, so this replaces the 3-component emission
(F=Normal, L=-Gamma, R=+Gamma) with a **5-component** one:

    F = Normal(0, sigmaF)
    L = wL1 . -Gamma(aL1,tL1) + wL2 . -Gamma(aL2,tL2)      (short + long left turns)
    R = wR1 . +Gamma(aR1,tR1) + wR2 . +Gamma(aR2,tR2)      (short + long right turns)

Only the EMISSION changes: the input features (history + stim + optional border) reuse
``fit_MLP_border.build_feature_matrix`` verbatim, and the Gamma/Normal math reuses
``fit_MLP``. So ``mlp_border_full_h128`` vs the mix variant is a clean emission-only A/B.
full mode only (the mix is a per-bout head).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate as E
import fit_MLP as M
import fit_MLP_border as MB
from segment_bouts import PX_PER_MM, R0_MM     # for the honest closed-loop dot-proximity recompute

_LOG2PI = float(math.log(2.0 * math.pi))


# =========================================================================== #
#  5-component MDN head (feedforward body + 2 Gammas per side)
# =========================================================================== #
class MDNMix(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 14)     # 5 weight logits + sigmaF + 4 (alpha,theta) pairs

    @staticmethod
    def _angle_params(out):
        logw = F.log_softmax(out[:, :5], dim=1)
        sp = F.softplus
        return (logw,
                sp(out[:, 5]) + M.EPS,                               # sigmaF
                sp(out[:, 6]) + M.ALPHA_MIN, sp(out[:, 7]) + M.EPS,  # aL1, tL1
                sp(out[:, 8]) + M.ALPHA_MIN, sp(out[:, 9]) + M.EPS,  # aL2, tL2
                sp(out[:, 10]) + M.ALPHA_MIN, sp(out[:, 11]) + M.EPS,  # aR1, tR1
                sp(out[:, 12]) + M.ALPHA_MIN, sp(out[:, 13]) + M.EPS)  # aR2, tR2

    def forward(self, x):
        return self._angle_params(self.head(self.body(x)))


def mix_logdens(dtheta, params):
    """Per-bout log predictive density of the 5-component mixture (nats over degrees)."""
    logw, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = params
    logNF = -0.5 * (_LOG2PI + 2.0 * torch.log(sF)) - dtheta ** 2 / (2.0 * sF ** 2)
    gL1 = M._gamma_logpdf_torch(-dtheta, aL1, tL1)
    gL2 = M._gamma_logpdf_torch(-dtheta, aL2, tL2)
    gR1 = M._gamma_logpdf_torch(+dtheta, aR1, tR1)
    gR2 = M._gamma_logpdf_torch(+dtheta, aR2, tR2)
    comp = torch.stack([logw[:, 0] + logNF, logw[:, 1] + gL1, logw[:, 2] + gL2,
                        logw[:, 3] + gR1, logw[:, 4] + gR2], dim=1)
    return torch.logsumexp(comp, dim=1)


@torch.no_grad()
def predict_params_mix(model, X):
    """numpy per-bout params (w (n,5), sigmaF, aL1,tL1,aL2,tL2,aR1,tR1,aR2,tR2 each (n,))."""
    model.eval()
    t = model(torch.tensor(X))
    return (np.exp(t[0].numpy()),) + tuple(z.numpy().astype(float) for z in t[1:])


def symbol_logq_mix(w, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2, dtheta0=None, eps=1e-300):
    """(n,3) predictive log-PMF over {F,L,R}: integrate each component over +/-dtheta0
    (generalises fit_MLP.symbol_logq_full to 2 Gammas per side)."""
    from scipy.special import erf, gammainc
    d0 = M.DTHETA0 if dtheta0 is None else dtheta0
    sqrt2 = math.sqrt(2.0)
    Phi_hi = 0.5 * (1.0 + erf((d0 / sF) / sqrt2))
    Phi_lo = 0.5 * (1.0 + erf((-d0 / sF) / sqrt2))
    nF, nL, nR = Phi_hi - Phi_lo, Phi_lo, 1.0 - Phi_hi
    GL1 = gammainc(aL1, d0 / tL1); GL2 = gammainc(aL2, d0 / tL2)
    GR1 = gammainc(aR1, d0 / tR1); GR2 = gammainc(aR2, d0 / tR2)
    qF = w[:, 0] * nF + w[:, 1] * GL1 + w[:, 2] * GL2 + w[:, 3] * GR1 + w[:, 4] * GR2
    qL = w[:, 0] * nL + w[:, 1] * (1.0 - GL1) + w[:, 2] * (1.0 - GL2)
    qR = w[:, 0] * nR + w[:, 3] * (1.0 - GR1) + w[:, 4] * (1.0 - GR2)
    q = np.maximum(np.stack([qF, qL, qR], axis=1), eps)
    return np.log(q / q.sum(axis=1, keepdims=True))


# =========================================================================== #
#  Training (batched one-step NLL; mirrors fit_MLP.fit_mlp with the 5-comp loss)
# =========================================================================== #
def fit_mix(X, y, hidden, *, epochs=60, lr=1e-3, batch_size=256, patience=8,
            val_frac=0.15, seed=0, device=None):
    device = device or M._get_device()
    M._seed_everything(seed)
    N = len(X); idx = np.random.permutation(N); nval = max(1, int(val_frac * N))
    vi, ti = idx[:nval], idx[nval:]
    Xt = torch.tensor(X[ti], device=device); yt = torch.tensor(y[ti], device=device)
    Xv = torch.tensor(X[vi], device=device); yv = torch.tensor(y[vi], device=device)
    model = MDNMix(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def nll(m, Xb, yb):
        return -mix_logdens(yb, m(Xb)).mean()

    best_val, best_state, no_imp = float("inf"), None, 0
    hist = {"train": [], "val": []}
    nb = int(np.ceil(len(ti) / batch_size))
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(ti), device=device); tot = 0.0
        for b in range(nb):
            bi = perm[b * batch_size:(b + 1) * batch_size]
            loss = nll(model, Xt[bi], yt[bi])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.item())
        model.eval()
        with torch.no_grad():
            vloss = float(nll(model, Xv, yv).item())
        hist["train"].append(tot / max(1, nb)); hist["val"].append(vloss)
        if vloss < best_val - 1e-4:
            best_val, no_imp = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model = model.to("cpu")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


# =========================================================================== #
#  Autoregressive sampling (numpy; border-aware; clip +/-180)
# =========================================================================== #
def _numpy_predictor_mix(model):
    sd = model.state_dict()
    W1 = sd["body.0.weight"].numpy(); b1 = sd["body.0.bias"].numpy()
    W2 = sd["head.weight"].numpy();   b2 = sd["head.bias"].numpy()

    def sp(z):
        return np.logaddexp(0.0, z)

    def fwd(x):
        h = np.maximum(W1 @ x + b1, 0.0); o = W2 @ h + b2
        lw = o[:5]; w = np.exp(lw - lw.max()); w /= w.sum()
        return (w, sp(o[5]) + M.EPS,
                sp(o[6]) + M.ALPHA_MIN, sp(o[7]) + M.EPS, sp(o[8]) + M.ALPHA_MIN, sp(o[9]) + M.EPS,
                sp(o[10]) + M.ALPHA_MIN, sp(o[11]) + M.EPS, sp(o[12]) + M.ALPHA_MIN, sp(o[13]) + M.EPS)
    return fwd


def generate_mix(model, rec, cfg_feat, dt_pool, dist_pool, rng):
    n = rec.n
    K = cfg_feat["K"]
    dth_scale = cfg_feat["dth_scale"]
    stim = M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                         phi_ref=cfg_feat.get("phi_ref", "ori"))
    border = MB._border_block(rec, cfg_feat) if cfg_feat.get("use_border") else np.zeros((n, 0))
    adapt = MB._adapt_block(rec, cfg_feat) if cfg_feat.get("use_adapt") else np.zeros((n, 0))
    fwd = _numpy_predictor_mix(model)
    dth = np.zeros(n)
    for t in range(n):
        hist = np.zeros(K)
        for k in range(1, K + 1):
            if t - k >= 0:
                hist[k - 1] = dth[t - k]
        row = [hist / dth_scale]
        if cfg_feat["include_symbols"]:
            sb = np.zeros(3 * K)
            for k in range(1, K + 1):
                if t - k >= 0:
                    dd = dth[t - k]
                    s = 0 if abs(dd) < M.DTHETA0 else (1 if dd < 0 else 2)
                    sb[3 * (k - 1) + s] = 1.0
            row.append(sb)
        row.append(stim[t]); row.append(border[t]); row.append(adapt[t])
        w, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = fwd(np.concatenate(row).astype(np.float32))
        c = rng.choice(5, p=w)
        if c == 0:
            dth[t] = rng.normal(0.0, sF)
        elif c == 1:
            dth[t] = -rng.gamma(aL1, tL1)
        elif c == 2:
            dth[t] = -rng.gamma(aL2, tL2)
        elif c == 3:
            dth[t] = +rng.gamma(aR1, tR1)
        else:
            dth[t] = +rng.gamma(aR2, tR2)
        dth[t] = float(np.clip(dth[t], -180.0, 180.0))
    return dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                dt_s=rng.choice(dt_pool, size=n), distance_px=rng.choice(dist_pool, size=n))


def _dot_stim_row(stim_mode, dot_cx_t, dot_cy_t, x, y, h, moving_t, n_dots_t=0.0):
    """Honest closed-loop recompute of the 'dot'/'dotb'/'shoal' stim row from the CURRENT generated
    pose (x, y, h) and the REAL dot centroid (dot_cx_t, dot_cy_t) -- so the fish reacts to its OWN
    bearing to the real dots (mirrors the wall recompute), NOT the teacher-forced real-fish bearing.
    Same column layout as ``fit_MLP._stim_block`` for one bout; zeros (bearing/prox) off-dot.

    For 'shoal'/'shoalr' both the bearing AND the range are recomputed from the generated pose,
    via the SAME ``fit_MLP.shoal_channels`` used at train time (so the layout cannot drift);
    N / moving / active stay teacher-forced block constants."""
    if stim_mode in ("shoal", "shoalr", "shoalf"):
        kw = dict(use_n=(stim_mode != "shoalr"), far=(stim_mode == "shoalf"))
        if not np.isfinite(dot_cx_t):
            return M.shoal_channels([np.nan], [np.nan], [n_dots_t], [moving_t], **kw)[0]
        phi = MB._wrap_pi(np.arctan2(dot_cy_t - y, dot_cx_t - x) - h)
        r_px = float(np.hypot(dot_cx_t - x, dot_cy_t - y))
        return M.shoal_channels([phi], [r_px], [n_dots_t], [moving_t], **kw)[0]
    if not np.isfinite(dot_cx_t):                       # no active dot at this bout onset
        return (np.array([0.0, 0.0, 0.0, 0.0, 0.0, moving_t], float) if stim_mode == "dot"
                else np.array([0.0, 0.0, moving_t], float))
    phi = MB._wrap_pi(np.arctan2(dot_cy_t - y, dot_cx_t - x) - h)
    s, c = np.sin(phi), np.cos(phi)
    if stim_mode == "dotb":
        return np.array([s, c, moving_t], float)
    r_px = np.hypot(dot_cx_t - x, dot_cy_t - y)          # proximity from the CURRENT pose
    g = 1.0 / (1.0 + (r_px / PX_PER_MM) / R0_MM)
    return np.array([s, c, g * s, g * c, g, moving_t], float)


def generate_mix_loop(model, rec, cfg_feat, dt_pool, dist_pool, rng,
                      closed_loop=True, guardrail=True, eps=1e-3, stim_loop=False):
    """CLOSED-LOOP free-run with the 5-component mix emission + hard containment guardrail.

    Combines ``fit_MLP_border.generate_mlp_loop`` (absolute-pose tracking, wall covariate
    recomputed from the GENERATED position, ray-to-wall guardrail) with this module's mix
    predictor (2 Gammas/side). Two modes:

      closed_loop=True  -- HONEST simulator: the wall covariate is recomputed from the
                           fish's OWN generated pose each bout, so its turns react to where
                           it actually is. This is the 'real' C2ST.
      closed_loop=False -- guardrail-only: the wall covariate is teacher-forced from ``rec``
                           (the same input ``generate_mix`` sees), so the TURN distribution
                           matches the non-loop mix model (~same C2ST); only the rendered
                           PATH is contained by the guardrail.

    ``stim_loop=True`` additionally makes the STIMULUS honest: for 'dot'/'dotb' modes the dot
    bearing (and proximity) are recomputed from the fish's OWN generated pose + the real dot
    position each bout (mirrors the wall recompute), so a drifted fish sees the dot where it
    actually is relative to itself -- the only lever that lets it steer back. Default False keeps
    the stimulus teacher-forced. Step length is pool-resampled (no speed head yet). Returns the
    usual sample dict PLUS the tracked ``x_px, y_px``."""
    n = rec.n
    K = cfg_feat["K"]
    dth_scale = cfg_feat["dth_scale"]
    stim = M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                         phi_ref=cfg_feat.get("phi_ref", "ori"))   # teacher-forced
    use_border = cfg_feat.get("use_border")
    tf_border = MB._border_block(rec, cfg_feat) if use_border else np.zeros((n, 0))
    recompute_stim = (stim_loop and closed_loop
                      and cfg_feat["stim_mode"] in ("dot", "dotb", "shoal", "shoalr", "shoalf"))
    use_adapt = cfg_feat.get("use_adapt")
    adapt_online = bool(use_adapt and closed_loop)       # advance the accumulator from the GENERATED pose
    if recompute_stim or adapt_online:                   # need the real dot positions in the loop
        dot_cx_a = np.asarray(getattr(rec, "dot_cx", np.full(n, np.nan)), float)
        dot_cy_a = np.asarray(getattr(rec, "dot_cy", np.full(n, np.nan)), float)
    if recompute_stim:
        mv_a = np.asarray(rec.moving, float)
        nd_a = np.asarray(getattr(rec, "n_dots", np.zeros(n)), float)   # teacher-forced group size
        phi_gen = np.full(n, np.nan)                     # bearing the honest fish actually reacted to
    if use_adapt:
        tf_adapt = MB._adapt_block(rec, cfg_feat)        # teacher-forced fallback (closed_loop=False)
        lam_ad = float(cfg_feat.get("adapt_lambda", 0.6)); a_acc = None
    fwd = _numpy_predictor_mix(model)
    cx, cy, R = rec.dish_cx, rec.dish_cy, rec.dish_R
    x, y, h = float(rec.x_px[0]), float(rec.y_px[0]), float(rec.f_ori_rad[0])
    dth = np.zeros(n); dist = np.zeros(n); xs = np.empty(n); ys = np.empty(n)
    for t in range(n):
        xs[t], ys[t] = x, y
        if use_border and closed_loop:                 # wall covariate from the CURRENT pose
            r = np.hypot(x - cx, y - cy)
            psi = MB._wrap_pi(np.arctan2(cy - y, cx - x) - h)
            border_t = MB.border_channels((R - r) / R, psi,
                                          cfg_feat.get('border_mode', 'raw'))[0]
            assert border_t.shape[0] == tf_border.shape[1], 'border row width != block width'
        else:                                          # teacher-forced (guardrail-only) or none
            border_t = tf_border[t]
        if recompute_stim:                               # honest dot bearing from the CURRENT pose
            if np.isfinite(dot_cx_a[t]):
                phi_gen[t] = MB._wrap_pi(np.arctan2(dot_cy_a[t] - y, dot_cx_a[t] - x) - h)
            stim_t = _dot_stim_row(cfg_feat["stim_mode"], dot_cx_a[t], dot_cy_a[t],
                                   x, y, h, mv_a[t], nd_a[t])
            # the online row and the trained block must agree in width for EVERY stim_mode --
            # the layout is replicated by hand in several places with no other check
            assert stim_t.shape[0] == stim.shape[1], "closed-loop stim row width != block width"
        else:
            stim_t = stim[t]
        if adapt_online:                                 # proximity + adaptation from the CURRENT pose
            if np.isfinite(dot_cx_a[t]):
                r_px = np.hypot(dot_cx_a[t] - x, dot_cy_a[t] - y)
                g_now = 1.0 / (1.0 + (r_px / PX_PER_MM) / R0_MM)
            else:
                g_now = 0.0
            a_acc = g_now if a_acc is None else lam_ad * a_acc + (1.0 - lam_ad) * g_now
            adapt_t = np.array([g_now, g_now - a_acc], dtype=np.float32)
        elif use_adapt:
            adapt_t = tf_adapt[t]                         # teacher-forced (guardrail-only)
        else:
            adapt_t = np.zeros(0, dtype=np.float32)
        hist = np.zeros(K)
        for k in range(1, K + 1):
            if t - k >= 0:
                hist[k - 1] = dth[t - k]
        row = [hist / dth_scale]
        if cfg_feat["include_symbols"]:
            sb = np.zeros(3 * K)
            for k in range(1, K + 1):
                if t - k >= 0:
                    dd = dth[t - k]
                    s = 0 if abs(dd) < M.DTHETA0 else (1 if dd < 0 else 2)
                    sb[3 * (k - 1) + s] = 1.0
            row.append(sb)
        row.append(stim_t); row.append(border_t); row.append(adapt_t)   # SAME order as build_feature_matrix
        w, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = fwd(np.concatenate(row).astype(np.float32))
        c = rng.choice(5, p=w)
        if c == 0:
            dth[t] = rng.normal(0.0, sF)
        elif c == 1:
            dth[t] = -rng.gamma(aL1, tL1)
        elif c == 2:
            dth[t] = -rng.gamma(aL2, tL2)
        elif c == 3:
            dth[t] = +rng.gamma(aR1, tR1)
        else:
            dth[t] = +rng.gamma(aR2, tR2)
        dth[t] = float(np.clip(dth[t], -180.0, 180.0))
        # integrate pose: turn first, then step along the NEW heading (as reconstruct_xy does)
        h = h + np.radians(dth[t])
        L = float(rng.choice(dist_pool))
        dist[t] = L
        if guardrail:
            x, y, h = MB.advance(x, y, h, L, cx, cy, R, eps,
                                 mode=cfg_feat.get('wall_mode', 'clip'))
        else:
            x = x + L * np.cos(h); y = y + L * np.sin(h)
    out = dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
               dt_s=rng.choice(dt_pool, size=n), distance_px=dist, x_px=xs, y_px=ys)
    if recompute_stim:
        out["stim_angle_gen"] = phi_gen                  # the bearing the honest fish reacted to
    return out


# =========================================================================== #
#  Evaluation-pipeline adapter
# =========================================================================== #
class MLPMixDensity(E.GenerativeModel):
    """MDN-MLP with the 5-component (2-Gamma/side) emission; full mode only. Inputs
    (history + stim + optional border) come from fit_MLP_border.build_feature_matrix."""
    has_density = True
    has_symbol = True
    can_sample = True

    def __init__(self, K=8, stim_mode="simple", include_symbols=False, hidden=128,
                 epochs=60, lr=1e-3, patience=8, batch_size=256, seed=0, use_border=True,
                 border_mode='raw', wall_mode='clip',
                 use_adapt=False, adapt_lambda=0.6):
        self.mode = "full"
        self.name = "mlp_border_mix" if use_border else "mlp_mix"
        self.K = K
        self.stim_mode = stim_mode
        self.include_symbols = include_symbols
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.use_border = use_border
        self.border_mode = border_mode      # 'raw' (3 ch) or 'basis' (9 ch, distance-tuned)
        self.wall_mode = wall_mode          # 'clip' (legacy) or 'reflect' (billiard bounce)
        self.use_adapt = use_adapt          # sensory-adaptation covariate (leaky-integrated proximity)
        self.adapt_lambda = adapt_lambda
        self.requires_absdict = (stim_mode == "rich")

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
        self._cfg_feat = dict(K=self.K, stim_mode=self.stim_mode,
                              include_symbols=self.include_symbols, dth_scale=self._dth_scale,
                              rich_scales=self._rich_scales, use_border=self.use_border,
                              border_mode=self.border_mode, wall_mode=self.wall_mode,
                              use_adapt=self.use_adapt, adapt_lambda=self.adapt_lambda)
        X = np.concatenate([MB.build_feature_matrix(r, self._cfg_feat) for r in train_records], 0)
        y = np.concatenate([r.dtheta.astype(np.float32) for r in train_records])
        self._model, self.history = fit_mix(
            X, y, self.hidden, epochs=self.epochs, lr=self.lr, patience=self.patience,
            batch_size=self.batch_size, seed=self.seed)
        self.config = dict(mode="full", hidden=self.hidden, n_features=X.shape[1], K=self.K,
                           stim_mode=self.stim_mode, use_border=self.use_border, dth_scale=self._dth_scale)
        return self

    def score(self, rec):
        X = MB.build_feature_matrix(rec, self._cfg_feat)
        params = predict_params_mix(self._model, X)
        with torch.no_grad():
            logdens = mix_logdens(torch.tensor(rec.dtheta.astype(np.float32)),
                                  self._model(torch.tensor(X))).numpy()
        return dict(logdens=logdens, symbol_logq=symbol_logq_mix(*params))

    def sample(self, rec, rng):
        return generate_mix(self._model, rec, self._cfg_feat,
                            self._dt_pool, self._dist_pool, rng)


class MLPMixLoopDensity(MLPMixDensity):
    """Mix-emission MLP that free-runs CLOSED-LOOP with a hard containment guardrail.

    ``fit`` / ``score`` are inherited from ``MLPMixDensity`` UNCHANGED, so Metric A
    (bits/bout) is identical to ``mlp_border_mix``; only ``sample`` differs:

      closed_loop=True  -> honest self-contained simulator (the 'real' C2ST), 0% out-of-dish.
      closed_loop=False -> guardrail-only: wall covariate teacher-forced, so the turn
                           distribution (and thus C2ST) matches ``mlp_border_mix`` while the
                           rendered path is still contained.
    """

    def __init__(self, *args, closed_loop=True, guardrail=True, eps=1e-3, stim_loop=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"{self.name}_{'loop' if closed_loop else 'guard'}"
        self.closed_loop = closed_loop
        self.guardrail = guardrail
        self.eps = eps
        self.stim_loop = stim_loop     # honest: recompute the dot bearing from the generated pose

    def sample(self, rec, rng):
        return generate_mix_loop(self._model, rec, self._cfg_feat,
                                 self._dt_pool, self._dist_pool, rng,
                                 closed_loop=self.closed_loop,
                                 guardrail=self.guardrail, eps=self.eps,
                                 stim_loop=self.stim_loop)


# =========================================================================== #
#  Checkpoint save / load  (train in Colab -> download -> render/score locally)
# =========================================================================== #
def save_mix(model, path):
    """Persist a FITTED ``MLPMixDensity`` / ``MLPMixLoopDensity`` so it can be reloaded and
    used (``.sample`` / ``.score``) WITHOUT retraining. Stores the MDN weights + the feature
    config + the dt/distance resample pools + the wrapper flags -- everything the numpy
    sampler and the density scorer need. Records are NOT stored (rebuild them locally with
    ``fit_MLP_border.build_border_records``). Typical use: fit on all fish in Colab, save,
    download the (~1 MB) file, then ``load_mix`` locally and feed it to ``bout_video``."""
    import torch
    if not hasattr(model, "_model"):
        raise RuntimeError("model is not fitted; call .fit(...) before save_mix")
    # Different head layout -> own serialiser. isinstance, NOT a class-name string: a SUBCLASS of
    # the joint model (e.g. the per-fish random-effect variant) used to fall straight through this
    # test to the generic branch below, which writes a checkpoint with none of its extra parameters
    # -- and that checkpoint then LOADS WITHOUT ERROR as a plain joint model. Silent and plausible.
    import fit_MLP_joint as _J
    if isinstance(model, _J.MLPMixJointDensity):
        try:
            import fit_MLP_fishre as _FR
        except ImportError:
            _FR = None
        if _FR is not None and isinstance(model, _FR.MLPMixJointREDensity):
            return _FR.save_fishre(model, path)
        return _J.save_joint(model, path)
    is_loop = isinstance(model, MLPMixLoopDensity)
    is_kin = isinstance(model, MLPMixKinDensity)
    blob = dict(
        cls="MLPMixKinDensity" if is_kin else ("MLPMixLoopDensity" if is_loop else "MLPMixDensity"),
        state_dict=model._model.state_dict(),
        n_features=int(model.config["n_features"]),
        hidden=int(model.config["hidden"]),
        cfg_feat=model._cfg_feat,
        dt_pool=np.asarray(model._dt_pool),
        dist_pool=np.asarray(model._dist_pool),
        config=model.config,
        name=model.name,
        K=model.K, stim_mode=model.stim_mode,
        include_symbols=model.include_symbols, use_border=model.use_border,
        border_mode=getattr(model, 'border_mode', 'raw'),
        history=getattr(model, "history", None),   # so early-stopping can be checked post hoc
        wall_mode=getattr(model, 'wall_mode', 'clip'),
        use_adapt=getattr(model, "use_adapt", False),
        adapt_lambda=getattr(model, "adapt_lambda", 0.6),
        loop=(dict(closed_loop=model.closed_loop, guardrail=model.guardrail, eps=model.eps,
                   stim_loop=getattr(model, "stim_loop", False))
              if is_loop else None),
    )
    torch.save(blob, path)
    return path


def _numpy2_unpickle_compat():
    """Let a numpy-1.x runtime unpickle arrays saved under numpy-2.x. numpy 2.0 renamed
    ``numpy.core`` -> ``numpy._core``, so a .pt saved in Colab (numpy 2) references
    ``numpy._core.*`` which a local numpy 1 lacks. Alias the missing modules to their
    ``numpy.core`` equivalents (no-op on numpy 2). Enables train-in-Colab -> load-locally."""
    import importlib
    import sys as _sys
    try:
        importlib.import_module("numpy._core")           # present on numpy >= 2
        return
    except ModuleNotFoundError:
        pass
    for name in ("numpy._core", "numpy._core.multiarray", "numpy._core.numeric",
                 "numpy._core.umath", "numpy._core._multiarray_umath"):
        try:
            _sys.modules[name] = importlib.import_module(name.replace("numpy._core", "numpy.core"))
        except Exception:
            pass


def load_mix(path, map_location="cpu", closed_loop=None, guardrail=None):
    """Reload a model saved by ``save_mix``, ready to ``.sample`` / ``.score`` (no fit).
    ``closed_loop`` / ``guardrail`` override the saved sampling flags if given (handy to
    reload a trained model as either the honest loop or the guardrail-only variant)."""
    import torch
    _numpy2_unpickle_compat()                            # Colab (numpy 2) -> local (numpy 1) compat
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                                   # older torch: no weights_only kwarg
        blob = torch.load(path, map_location=map_location)
    _cls = blob.get("cls")
    if _cls in ("MLPMixJointDensity", "MLPMixJointREDensity"):
        # joint checkpoints have chained heads (dist_head/dt_head hang off the SHARED body), and
        # the random-effect subclass adds an embedding plus four loading matrices on top, so the
        # MDNMix rebuild below cannot take either state_dict. Delegate, mirroring save_mix.
        # Doing this here means every downstream consumer -- render_video, bout_video, the diag
        # scripts, the report notebook -- accepts either checkpoint with no change of its own.
        # NOTE both branches must be kept in step with save_mix: a class that can be SAVED by a
        # specialised serialiser and not LOADED by one fails only at the call site, far away.
        if _cls == "MLPMixJointREDensity":
            import fit_MLP_fishre as _J
            m = _J.load_fishre(path, map_location=map_location)
        else:
            import fit_MLP_joint as _J
            m = _J.load_joint(path, map_location=map_location)
        if closed_loop is not None:
            m.closed_loop = bool(closed_loop)
        if guardrail is not None:
            m.guardrail = bool(guardrail)
        return m
    kw = dict(K=blob["K"], stim_mode=blob["stim_mode"], include_symbols=blob["include_symbols"],
              hidden=blob["hidden"], use_border=blob["use_border"],
              border_mode=blob.get("border_mode", "raw"),
              wall_mode=blob.get("wall_mode", "clip"),
              use_adapt=blob.get("use_adapt", False), adapt_lambda=blob.get("adapt_lambda", 0.6))
    if blob["cls"] == "MLPMixKinDensity":
        kw.update(n_dist=blob["config"].get("n_dist", 1), n_dt=blob["config"].get("n_dt", 2))
    if blob["cls"] in ("MLPMixKinDensity", "MLPMixLoopDensity") or closed_loop is not None:
        lp = blob["loop"] or dict(closed_loop=True, guardrail=True, eps=1e-3)
        Wrapper = MLPMixKinDensity if blob["cls"] == "MLPMixKinDensity" else MLPMixLoopDensity
        m = Wrapper(
            **kw,
            closed_loop=lp["closed_loop"] if closed_loop is None else closed_loop,
            guardrail=lp["guardrail"] if guardrail is None else guardrail,
            eps=lp["eps"], stim_loop=lp.get("stim_loop", False))
    else:
        m = MLPMixDensity(**kw)
    if blob["cls"] == "MLPMixKinDensity":
        cf = blob["config"]
        net = MDNMixKin(blob["n_features"], blob["hidden"], cf["kin_features"], cf.get("kin_hidden", 32),
                        n_dist=cf.get("n_dist", 1), n_dt=cf.get("n_dt", 2))
    else:
        net = MDNMix(blob["n_features"], blob["hidden"])
    net.load_state_dict(blob["state_dict"]); net.eval()
    m._model = net
    m._cfg_feat = blob["cfg_feat"]
    m._rich_scales = blob["cfg_feat"].get("rich_scales")
    m._dt_pool = blob["dt_pool"]; m._dist_pool = blob["dist_pool"]
    m.config = blob["config"]; m.name = blob["name"]
    m.history = blob.get("history")
    return m


# =========================================================================== #
#  Joint AR kinematics: distance (log-normal) + dt (2-comp log-normal | distance)
# =========================================================================== #
def _ln_logpdf(x, mu, sigma):                            # torch log-normal log-density (x > 0)
    lx = torch.log(x)
    return -lx - torch.log(sigma) - 0.5 * _LOG2PI - (lx - mu) ** 2 / (2.0 * sigma ** 2)


def _lnk_logpdf(x, logw, mu, sigma):                     # K-component log-normal; logw/mu/sigma (n,K)
    """Mixture log-normal for the kinematic channels.

    Why more than one component. The real upper tails are far heavier than a single log-normal
    can carry: dt q99/q99.9/max = 18.7/103.6/426.5 s against a 1.23 s median, and distance
    44/85/218 px against a 15 px median. A 1-comp distance head and a 2-comp dt head reproduce
    the BULK correctly (gen dt_median 1.292 vs real 1.281) but truncate the tail -- gen dt q99.9
    is 24.6 s vs real 103.6, and only 0.055% of generated bouts exceed 30 s vs 0.63% of real.
    That collapses every tail-sensitive window feature: across-window variance ratios of
    dt_std 0.04, dt_mean 0.08, dt_q90 0.12 (while dt_median, a bulk statistic, sits at 0.46).
    The clip is NOT the cause -- it binds on 0.001% of generated bouts.
    An extra component gives the head a slow/rare mode to place the long pauses in."""
    lx = torch.log(x).reshape(-1, 1)
    comp = logw + (-lx - torch.log(sigma) - 0.5 * _LOG2PI - (lx - mu) ** 2 / (2.0 * sigma ** 2))
    return torch.logsumexp(comp, dim=1)


def _ln2_logpdf(x, logw, mu, sigma):                     # 2-component log-normal; logw/mu/sigma (n,2)
    lx = torch.log(x).reshape(-1, 1)
    comp = logw + (-lx - torch.log(sigma) - 0.5 * _LOG2PI - (lx - mu) ** 2 / (2.0 * sigma ** 2))
    return torch.logsumexp(comp, dim=1)


class MDNMixKin(MDNMix):
    """MDNMix + a JOINT AR kinematics emission on a SEPARATE trunk. The angle 5-comp mix (``body``+``head``)
    is inherited unchanged and reads the full dotb features (so ``forward``/``score`` stay TURN-ONLY and the
    angle emission == dotb_hloop). A dedicated ``kin_body`` reads ONLY the kinematic-history block and feeds the
    distance (log-normal) + dt (2-comp log-normal | log-distance) heads -- so distance/dt depend only on their
    OWN history (+ the within-bout distance->dt coupling), NOT the autocorrelated position/turn representation
    (which caused the acf over-shoot when the heads shared the angle body)."""

    def __init__(self, n_features, hidden, kin_features, kin_hidden=32, n_dist=1, n_dt=2):
        super().__init__(n_features, hidden)                       # angle body+head on the full dotb features
        self.kin_body = nn.Sequential(nn.Linear(kin_features, kin_hidden), nn.ReLU())
        # n_dist/n_dt components per channel. Defaults keep the historical 1-comp distance /
        # 2-comp dt exactly; heavy-tail variants pass n_dist=2, n_dt=3.
        self.n_dist, self.n_dt = int(n_dist), int(n_dt)
        self.dist_head = nn.Linear(kin_hidden, 2 if self.n_dist == 1 else 3 * self.n_dist)
        self.dt_head = nn.Linear(kin_hidden + 1, 3 * self.n_dt)

    def kin_forward(self, x, xk, log_dist):
        """(angle_params from ``x``; (dmu,dsig), (dt_logw,dt_mu,dt_sig) from the kin trunk ``xk``)."""
        angle = self._angle_params(self.head(self.body(x)))
        bk = self.kin_body(xk)
        do = self.dist_head(bk)
        if self.n_dist == 1:
            dist = (do[:, 0], F.softplus(do[:, 1]) + M.EPS)            # legacy 1-comp: (mu, sigma)
        else:
            k = self.n_dist
            dist = (F.log_softmax(do[:, :k], dim=1), do[:, k:2 * k],
                    F.softplus(do[:, 2 * k:3 * k]) + M.EPS)
        to = self.dt_head(torch.cat([bk, log_dist.reshape(-1, 1)], dim=1))
        k = self.n_dt
        return angle, dist, (F.log_softmax(to[:, :k], dim=1), to[:, k:2 * k],
                             F.softplus(to[:, 2 * k:3 * k]) + M.EPS)


def fit_mix_kin(X, Xk, y_ang, y_dist, y_dt, hidden, *, kin_hidden=32, epochs=60, lr=1e-3, batch_size=256,
                patience=8, val_frac=0.15, seed=0, device=None, kin_wd=1e-4, n_dist=1, n_dt=2):
    """Train MDNMixKin with the JOINT one-step NLL = angle mix (from ``X``) + distance log-normal + dt 2-comp
    log-normal (from the kin trunk ``Xk``); distance/dt targets NaN-masked. ``kin_wd`` = weight-decay on the
    kin trunk + heads only (mild; the marginal is fixed by the clip, not by shrinking)."""
    device = device or M._get_device()
    M._seed_everything(seed)
    N = len(X); idx = np.random.permutation(N); nval = max(1, int(val_frac * N))
    vi, ti = idx[:nval], idx[nval:]
    tt = lambda a, i: torch.tensor(np.asarray(a)[i], device=device)
    Xt, Xv = tt(X, ti), tt(X, vi)
    Xkt, Xkv = tt(Xk, ti), tt(Xk, vi)
    yat, yav = tt(y_ang, ti), tt(y_ang, vi)
    ydt_, ydv = tt(y_dist, ti), tt(y_dist, vi)
    ytt, ytv = tt(y_dt, ti), tt(y_dt, vi)
    model = MDNMixKin(X.shape[1], hidden, Xk.shape[1], kin_hidden,
                      n_dist=n_dist, n_dt=n_dt).to(device)
    kin_p = (list(model.kin_body.parameters()) + list(model.dist_head.parameters())
             + list(model.dt_head.parameters()))
    kin_ids = {id(p) for p in kin_p}
    base_p = [p for p in model.parameters() if id(p) not in kin_ids]
    opt = torch.optim.Adam([{"params": base_p, "weight_decay": 1e-5},
                            {"params": kin_p, "weight_decay": kin_wd}], lr=lr)

    def nll(m, Xb, Xkb, ya, yd, yt):
        lin = torch.nan_to_num(torch.log(torch.clamp(yd, min=1e-6)))      # dt conditioning (clean of NaN)
        ang, dpar, (tw, tmu, tsig) = m.kin_forward(Xb, Xkb, lin)
        la = mix_logdens(ya, ang).mean()
        md = torch.isfinite(yd) & (yd > 0); mt = torch.isfinite(yt) & (yt > 0)
        # dpar is (mu, sigma) for the legacy 1-comp distance head, (logw, mu, sigma) otherwise
        ld = (_ln_logpdf(torch.clamp(yd, min=1e-6), *dpar) if len(dpar) == 2
              else _lnk_logpdf(torch.clamp(yd, min=1e-6), *dpar))
        lt = _lnk_logpdf(torch.clamp(yt, min=1e-6), tw, tmu, tsig)
        z = torch.zeros((), device=Xb.device)
        ld = ld[md].mean() if md.any() else z
        lt = lt[mt].mean() if mt.any() else z
        return -(la + ld + lt)

    best_val, best_state, no_imp = float("inf"), None, 0
    hist = {"train": [], "val": []}
    nb = int(np.ceil(len(ti) / batch_size))
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(ti), device=device); tot = 0.0
        for bidx in range(nb):
            bi = perm[bidx * batch_size:(bidx + 1) * batch_size]
            loss = nll(model, Xt[bi], Xkt[bi], yat[bi], ydt_[bi], ytt[bi])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.item())
        model.eval()
        with torch.no_grad():
            vloss = float(nll(model, Xv, Xkv, yav, ydv, ytv).item())
        hist["train"].append(tot / max(1, nb)); hist["val"].append(vloss)
        if vloss < best_val - 1e-4:
            best_val, no_imp = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model = model.to("cpu")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def _numpy_predictor_kin(model):
    sd = model.state_dict()
    W1, b1 = sd["body.0.weight"].numpy(), sd["body.0.bias"].numpy()
    W2, b2 = sd["head.weight"].numpy(), sd["head.bias"].numpy()
    Wk, bk = sd["kin_body.0.weight"].numpy(), sd["kin_body.0.bias"].numpy()
    Wd, bd = sd["dist_head.weight"].numpy(), sd["dist_head.bias"].numpy()
    Wt, bt = sd["dt_head.weight"].numpy(), sd["dt_head.bias"].numpy()

    def sp(z):
        return np.logaddexp(0.0, z)

    def angle_fn(x):                                      # angle mix from the full dotb features
        o = W2 @ np.maximum(W1 @ x + b1, 0.0) + b2
        lw = o[:5]; w = np.exp(lw - lw.max()); w /= w.sum()
        return (w, sp(o[5]) + M.EPS, sp(o[6]) + M.ALPHA_MIN, sp(o[7]) + M.EPS,
                sp(o[8]) + M.ALPHA_MIN, sp(o[9]) + M.EPS, sp(o[10]) + M.ALPHA_MIN, sp(o[11]) + M.EPS,
                sp(o[12]) + M.ALPHA_MIN, sp(o[13]) + M.EPS)

    n_dist = int(getattr(model, "n_dist", 1)); n_dt = int(getattr(model, "n_dt", 2))

    def kin_fn(xk):                                       # distance params from the SEPARATE kin trunk
        bkin = np.maximum(Wk @ xk + bk, 0.0)
        do = Wd @ bkin + bd
        if n_dist == 1:
            return bkin, (do[0], sp(do[1]) + M.EPS)
        lw = do[:n_dist]; w = np.exp(lw - lw.max()); w /= w.sum()
        return bkin, (w, do[n_dist:2 * n_dist], sp(do[2 * n_dist:3 * n_dist]) + M.EPS)

    def dt_params(bkin, log_dist):
        to = Wt @ np.concatenate([bkin, [log_dist]]) + bt
        tlw = to[:n_dt]; tw = np.exp(tlw - tlw.max()); tw /= tw.sum()
        return tw, to[n_dt:2 * n_dt], sp(to[2 * n_dt:3 * n_dt]) + M.EPS

    return angle_fn, kin_fn, dt_params


def generate_mix_kin_loop(model, rec, cfg_feat, rng, closed_loop=True, guardrail=True, eps=1e-3,
                          stim_loop=False, keep_features=False):
    """``generate_mix_loop`` + JOINT AR kinematics: distance & dt are drawn from the log-normal heads
    AUTOREGRESSIVELY (generated kin history fed back), replacing the i.i.d. pool draws. Distance is still
    guardrail-capped for the POSITION step, but the returned ``distance_px`` is the MODEL's (uncapped) value
    so its marginal/acf reflect the model. Angle/border/stim/adapt handling identical to generate_mix_loop."""
    n = rec.n
    K = cfg_feat["K"]
    Kk = int(cfg_feat.get("K_kin", K))                   # short kinematic AR lag (default = K)
    dth_scale = cfg_feat["dth_scale"]
    dm, ds, tm, ts = cfg_feat["kin_scales"]
    dist_hi, dt_hi = cfg_feat.get("kin_clip", (400.0, 600.0))   # realistic caps (train q99.9)
    stim = M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                         phi_ref=cfg_feat.get("phi_ref", "ori"))
    use_border = cfg_feat.get("use_border")
    tf_border = MB._border_block(rec, cfg_feat) if use_border else np.zeros((n, 0))
    recompute_stim = (stim_loop and closed_loop
                      and cfg_feat["stim_mode"] in ("dot", "dotb", "shoal", "shoalr", "shoalf"))
    use_adapt = cfg_feat.get("use_adapt")
    adapt_online = bool(use_adapt and closed_loop)
    if recompute_stim or adapt_online:
        dot_cx_a = np.asarray(getattr(rec, "dot_cx", np.full(n, np.nan)), float)
        dot_cy_a = np.asarray(getattr(rec, "dot_cy", np.full(n, np.nan)), float)
    if recompute_stim:
        mv_a = np.asarray(rec.moving, float); phi_gen = np.full(n, np.nan)
        nd_a = np.asarray(getattr(rec, "n_dots", np.zeros(n)), float)   # teacher-forced group size
    if use_adapt:
        tf_adapt = MB._adapt_block(rec, cfg_feat); lam_ad = float(cfg_feat.get("adapt_lambda", 0.6)); a_acc = None
    angle_fn, kin_fn, dt_params = _numpy_predictor_kin(model)
    cx, cy, R = rec.dish_cx, rec.dish_cy, rec.dish_R
    x, y, h = float(rec.x_px[0]), float(rec.y_px[0]), float(rec.f_ori_rad[0])
    dth = np.zeros(n); dist = np.zeros(n); dtg = np.zeros(n); xs = np.empty(n); ys = np.empty(n)
    dth_base = np.zeros(n)                               # UNSCALED turn (feeds the angle head's history; see below)
    wall_stats = {}                                      # guardrail interventions, see `out` at the end
    # width == the net's input dim by construction; using it makes a layout drift fail loudly
    Xrec = (np.zeros((n, int(model.body[0].in_features)), dtype=np.float32)
            if keep_features else None)
    ld_hist = np.zeros(n); lt_hist = np.zeros(n)         # generated log(distance) / log(dt) for the AR history
    # EXPOSURE-BIAS shrinkage on the kinematic feedback. Teacher-forced training fits the head's
    # response to the REAL history; at generation the head is fed its OWN noisy samples, so the
    # noise is re-amplified each bout and the rollout over-persists -- measured gen log-distance
    # lag-1 acf +0.42 vs real +0.23 (dt +0.35 vs +0.23). The kin block is z-scored (mean 0), so
    # scaling it shrinks each lag toward the population mean by exactly kfb, damping the
    # self-feedback loop without touching the emission's conditional shape or its marginal.
    # 1.0 = unchanged (default).
    _qm = cfg_feat.get("qmap")               # monotone emitted-marginal correction
    _kfb = cfg_feat.get("kin_feedback", 1.0)
    # per channel: distance and dt need different shrinkage (measured optima ~0.52 and ~0.85)
    kfb_d, kfb_t = ((float(_kfb["dist"]), float(_kfb["dt"])) if isinstance(_kfb, dict)
                    else (float(_kfb), float(_kfb)))
    vg = cfg_feat.get("vigor"); z_dt = z_di = z_tn = 0.0  # per-trajectory swim-vigor latent (log-space modulation)
    if vg:
        su_dt, sd_dt, ph_dt, var_dt = vg["dt"]; su_di, sd_di, ph_di, var_di = vg["dist"]
        c_dt = float(np.sqrt(1.0 - ph_dt * ph_dt)); c_di = float(np.sqrt(1.0 - ph_di * ph_di))
        u_dt = rng.normal(0.0, su_dt); u_di = rng.normal(0.0, su_di)     # intercept = anonymous 'individuality'
        d_dt = rng.normal(0.0, sd_dt); d_di = rng.normal(0.0, sd_di)     # AR(1) slow drift, started stationary
    apply_turn = bool(vg and vg.get("apply_turn"))       # turn-magnitude individuality (scales |dtheta|)
    if apply_turn:
        su_tn, sd_tn, ph_tn, var_tn = vg["turn"]
        c_tn = float(np.sqrt(1.0 - ph_tn * ph_tn))
        u_tn = rng.normal(0.0, su_tn); d_tn = rng.normal(0.0, sd_tn)
    artr = cfg_feat.get("artr")                          # ARTR slow L/R turn-bias oscillator (direction persistence)
    if artr:
        rho_ar = float(artr["rho"]); beta_ar = float(artr["beta"])
        c_ar = float(np.sqrt(1.0 - rho_ar * rho_ar)); m_ar = rng.normal()
    # PER-TRAJECTORY INDIVIDUALITY on the two channels vigor does not cover: turning style and
    # spatial preference. Between-fish variance share is 0.074 against a real 0.255 and has not
    # moved in eleven ladders, because there is no mechanism for it -- not because the mechanism
    # is mistuned. Anonymous draws from a population prior fitted on the train fold, so this
    # generalises to held-out fish exactly like the vigor intercept; no fish identity anywhere.
    _iv = cfg_feat.get("indiv"); u_fw = u_rad = 0.0
    if _iv:
        u_fw = float(rng.normal(0.0, float(_iv.get("fw", 0.0))))     # forward/turn balance
        u_rad = float(rng.normal(0.0, float(_iv.get("rad", 0.0))))   # inward-bias offset, deg
    # INNOVATION COMPENSATION for the kin_feedback shrinkage. The mem ladder proved the shrinkage
    # is NECESSARY -- without it free-run log-distance acf@1 blows to 0.394 against a real 0.218
    # and dist_acf1 becomes the top C2ST tell -- but damping an AR's feedback damps its spread too
    # (stationary variance sigma^2/(1-phi^2)), leaving dist_std_ratio 0.83 and distance dispersion
    # 0.52 against a target of 1.0. Widening the INNOVATION restores the variance and leaves the
    # normalised acf untouched, since acf(k)=phi^k is scale-free.
    # NOT kin_var_comp, which lost twice: that rescaled the emitted value about the fixed TRAINING
    # log-mean and kept the history unscaled, so it widened the marginal symmetrically without
    # restoring the loop's dynamics. This rescales about the PER-BOUT conditional mean and feeds
    # the widened value back into the history, which is where the lost variance actually went.
    _kin = cfg_feat.get("kin_innov")
    ci_d = float(_kin.get("dist", 1.0)) if _kin else 1.0
    ci_t = float(_kin.get("dt", 1.0)) if _kin else 1.0
    for t in range(n):
        xs[t], ys[t] = x, y
        if use_border and closed_loop:
            r = np.hypot(x - cx, y - cy); psi = MB._wrap_pi(np.arctan2(cy - y, cx - x) - h)
            border_t = MB.border_channels((R - r) / R, psi,
                                          cfg_feat.get('border_mode', 'raw'))[0]
            assert border_t.shape[0] == tf_border.shape[1], 'border row width != block width'
        else:
            border_t = tf_border[t]
        if recompute_stim:
            if np.isfinite(dot_cx_a[t]):
                phi_gen[t] = MB._wrap_pi(np.arctan2(dot_cy_a[t] - y, dot_cx_a[t] - x) - h)
            stim_t = _dot_stim_row(cfg_feat["stim_mode"], dot_cx_a[t], dot_cy_a[t],
                                   x, y, h, mv_a[t], nd_a[t])
            # the online row and the trained block must agree in width for EVERY stim_mode --
            # the layout is replicated by hand in several places with no other check
            assert stim_t.shape[0] == stim.shape[1], "closed-loop stim row width != block width"
        else:
            stim_t = stim[t]
        if adapt_online:
            if np.isfinite(dot_cx_a[t]):
                r_px = np.hypot(dot_cx_a[t] - x, dot_cy_a[t] - y); g_now = 1.0 / (1.0 + (r_px / PX_PER_MM) / R0_MM)
            else:
                g_now = 0.0
            a_acc = g_now if a_acc is None else lam_ad * a_acc + (1.0 - lam_ad) * g_now
            adapt_t = np.array([g_now, g_now - a_acc], dtype=np.float32)
        elif use_adapt:
            adapt_t = tf_adapt[t]
        else:
            adapt_t = np.zeros(0, dtype=np.float32)
        hist = np.zeros(K)
        for k in range(1, K + 1):
            if t - k >= 0:
                hist[k - 1] = dth_base[t - k]            # UNSCALED history -> angle AR stays a stationary avg-fish
        row = [hist / dth_scale]
        if cfg_feat["include_symbols"]:
            sb = np.zeros(3 * K)
            for k in range(1, K + 1):
                if t - k >= 0:
                    dd = dth_base[t - k]; s = 0 if abs(dd) < M.DTHETA0 else (1 if dd < 0 else 2)
                    sb[3 * (k - 1) + s] = 1.0
            row.append(sb)
        row.append(stim_t); row.append(border_t); row.append(adapt_t)   # ANGLE row (no kin block)
        if keep_features:
            # Record the exact angle row the emission conditioned on, so a rollout-aware
            # fine-tune can recompute log p(dtheta_t | x_t) in torch WITHOUT reimplementing the
            # feature assembly. That reuse is the point: every previous duplication of this
            # layout has been a column-order hazard, and there are already ~7 hand-copies of it.
            _r = np.concatenate(row)
            assert _r.shape[0] == Xrec.shape[1], (_r.shape, Xrec.shape)
            Xrec[t] = _r
        kin_row = np.zeros(2 * Kk, dtype=np.float32)      # SEPARATE kin-trunk input: gen log-dist/log-dt history
        for k in range(1, Kk + 1):
            if t - k >= 0:
                kin_row[k - 1] = kfb_d * (ld_hist[t - k] - dm) / ds
                kin_row[Kk + k - 1] = kfb_t * (lt_hist[t - k] - tm) / ts
        ang = angle_fn(np.concatenate(row).astype(np.float32))
        bkin, _dp = kin_fn(kin_row)
        if len(_dp) == 2:
            dmu, dsig = _dp
            d_condmu = float(dmu)
        else:
            _dw, _dmus, _dsigs = _dp
            _ci = rng.choice(len(_dw), p=_dw)
            dmu, dsig = _dmus[_ci], _dsigs[_ci]
            d_condmu = float(np.dot(_dw, _dmus))         # mixture mean of log-distance for THIS bout
        w = ang[0]; sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = ang[1:]
        if u_fw:
            # TURN INDIVIDUALITY. Real fish differ enormously in how often they go straight --
            # forward fraction 0.062..0.277 across the 15 (sd 0.054) -- but the generated spread is
            # 0.005, because vigor gives a per-trajectory latent to dt and distance only and
            # _calibrate_vigor hard-sets the turn term to zero. So every rollout turns like the
            # average fish, and uni_F is the single worst individuality feature (real between-fish
            # variance share 0.617 vs 0.004 generated).
            # Tilt the FORWARD mixture weight, do NOT scale |dtheta|: that changes how OFTEN the
            # fish turns while leaving the shape of a turn alone, so the pooled |dtheta| marginal
            # is preserved. The old vigor_turn multiplied |dtheta| and wrecked abs_dtheta_w1
            # (6.67 -> 8.43) for exactly this reason.
            w = np.array([w[0] * float(np.exp(u_fw)), w[1], w[2], w[3], w[4]]); w = w / w.sum()
        if artr:                                         # slow L/R bias: m>0 favours right, m<0 left (w=[F,L1,L2,R1,R2])
            m_ar = rho_ar * m_ar + c_ar * rng.normal()
            be = float(np.exp(beta_ar * m_ar))
            w = np.array([w[0], w[1] / be, w[2] / be, w[3] * be, w[4] * be]); w = w / w.sum()
        c = rng.choice(5, p=w)
        if c == 0:
            dth_base[t] = rng.normal(0.0, sF)
        elif c == 1:
            dth_base[t] = -rng.gamma(aL1, tL1)
        elif c == 2:
            dth_base[t] = -rng.gamma(aL2, tL2)
        elif c == 3:
            dth_base[t] = +rng.gamma(aR1, tR1)
        else:
            dth_base[t] = +rng.gamma(aR2, tR2)
        dth_base[t] = float(np.clip(dth_base[t], -180.0, 180.0))
        if apply_turn:                                   # scale the EMITTED turn magnitude by the vigor (history=base)
            d_tn = ph_tn * d_tn + sd_tn * c_tn * rng.normal()
            z_tn = u_tn + d_tn - 0.5 * var_tn            # log multiplier on |dtheta|; -var/2 keeps E[|dtheta|]
        dth[t] = float(np.clip(dth_base[t] * float(np.exp(z_tn)), -180.0, 180.0))   # z_tn=0 -> dth == dth_base
        if _qm:
            dth[t] = float(np.clip(_apply_qmap(_qm, "dtheta", dth[t]), -180.0, 180.0))
        cb = cfg_feat.get("centre_bias", 0.0)            # generation-only inward steering bias (deg)
        if isinstance(cb, dict):
            # SHELL-GATED containment. The wall excess decomposes ~evenly into two factors:
            # entry rate 2.6x real (0.0139 vs 0.0054 per bout) and dwell 2.1x real (4.09 vs 1.93
            # bouts), multiplying to the 4.7x occupancy excess. A single constant bias can only
            # attack entry -- it matches the MEAN r/R at ~6 deg while leaving frac>0.85 at 0.033
            # vs real 0.012, because the tail is dwell. So: a small BULK bias for the entry rate
            # plus a ramp that switches on inside the shell for the escape.
            _u = np.hypot(x - cx, y - cy) / R
            _u0 = float(cb.get("u0", 0.75))
            cb = (float(cb.get("bulk", 0.0))
                  + float(cb.get("shell", 0.0)) * max(0.0, (_u - _u0) / max(1.0 - _u0, 1e-6)))
        # Spatial individuality SCALES the whole containment bias rather than offsetting it.
        # Offsetting was tried and is wrong twice over: centre_bias steers by ADDING cb*sin(psi)
        # to dtheta, so a per-trajectory offset leaks straight into the turn marginal
        # (abs_dtheta_w1 2.61 -> 3.89 at the offset the calibrator needed), and a large negative
        # offset flips a fish OUTWARD, tripling the wall fraction. Scaling keeps the sign, keeps
        # the bias zero near the centre where it would otherwise distort turns most, and gives a
        # fish that is uniformly weaker or stronger at staying in.
        if u_rad:
            cb = float(cb) * float(np.exp(u_rad))
        if cb:
            # One-step MLE fits the CONDITIONAL turn well but does not control the rollout's
            # STATIONARY radial distribution: gen sits at mean r/R 0.50 vs real 0.45 and spends
            # 6-7% of bouts beyond r/R 0.85 vs real 1.2%. The per-bout deficit is tiny (~1-2
            # points of P(turn toward centre) at every radius) but it integrates. This adds a
            # small constant inward bias, calibrated like vigor/ARTR, to correct the stationary
            # distribution without touching the emission's conditional shape.
            psi_c = MB._wrap_pi(np.arctan2(cy - y, cx - x) - h)
            dth[t] = float(np.clip(dth[t] + cb * np.sin(psi_c), -180.0, 180.0))
        if vg:                                           # advance the slow drift, recompute the log-modulation
            d_dt = ph_dt * d_dt + sd_dt * c_dt * rng.normal()
            d_di = ph_di * d_di + sd_di * c_di * rng.normal()
            z_dt = u_dt + d_dt - 0.5 * var_dt            # -var/2 keeps E[exp(z)]=1 (marginal mean preserved)
            z_di = u_di + d_di - 0.5 * var_di
        # clip the 'average-fish' step FIRST (the clip's real job = cap within-traj runaway), then scale by the
        # vigor multiplicatively -- so a high-vigor trajectory keeps its larger steps instead of being compressed
        # into the pooled q99.9 clip. dt|dist conditions on the CENTRED distance so the two channels stay
        # independent (real between-fish corr ~ 0). vigor OFF (z=0) -> identical to the plain kin model.
        _ld = dmu + dsig * rng.normal()
        if ci_d != 1.0:                                  # widen the innovation about THIS bout's conditional mean
            _ld = d_condmu + ci_d * (_ld - d_condmu)
        base_di = float(np.clip(np.exp(_ld), 1e-3, dist_hi))
        tw, tmu, tsig = dt_params(bkin, float(np.log(max(base_di, 1e-6))))
        ci = rng.choice(len(tw), p=tw)
        _lt = tmu[ci] + tsig[ci] * rng.normal()
        if ci_t != 1.0:
            _lt = float(np.dot(tw, tmu)) + ci_t * (_lt - float(np.dot(tw, tmu)))
        base_dt = float(np.clip(np.exp(_lt), 1e-3, dt_hi))
        # VARIANCE COMPENSATION for the kin_feedback shrinkage. Shrinking the AR history damps
        # persistence AND spread together: measured across the kin ladder, distance dispersion went
        # 1.04 (drift-only) -> 0.52 once kin_feedback was applied, and dt 0.47 -> 0.34, against a
        # real target of 1.00. Rescaling a series about its mean by a constant leaves its
        # AUTOCORRELATION exactly unchanged while scaling its variance by c^2, so this knob is
        # ORTHOGONAL to kin_feedback: lambda sets the acf, c restores the spread. Applied to the
        # EMITTED value only -- ld_hist/lt_hist keep the unscaled base, exactly as vigor does, so
        # the trunk still sees a stationary average-fish history and c cannot feed back on itself.
        emit_di, emit_dt = base_di, base_dt
        _kvs = cfg_feat.get("kin_var_scale")
        if _kvs:
            cdi = float(_kvs.get("dist", 1.0)); cdt = float(_kvs.get("dt", 1.0))
            if cdi != 1.0:
                emit_di = float(np.clip(np.exp(dm + cdi * (np.log(max(base_di, 1e-6)) - dm)),
                                        1e-3, dist_hi))
            if cdt != 1.0:
                emit_dt = float(np.clip(np.exp(tm + cdt * (np.log(max(base_dt, 1e-6)) - tm)),
                                        1e-3, dt_hi))
        dval = emit_di * float(np.exp(z_di)); tval = emit_dt * float(np.exp(z_dt))
        if _qm:
            # marginal correction applied INSIDE the loop: the mapped step is what moves the fish,
            # so the pose, border covariate and recomputed shoal bearing all stay consistent with
            # what was actually emitted. Post-processing a finished rollout could not do this.
            dval = _apply_qmap(_qm, "dist", dval); tval = _apply_qmap(_qm, "dt", tval)
        if cfg_feat.get("dt_snap"):
            # EMITTED value only; the base kept in ld_hist/lt_hist below stays continuous, so the
            # loop dynamics and RNG stream are byte-identical to the unsnapped control.
            tval = MB.snap_dt(tval)
        dist[t] = dval; dtg[t] = tval
        # trunk reads the CENTRED (vigor-free) history -> it stays a stationary 'average-fish' AR, no z feedback.
        ld_hist[t] = np.log(max(base_di, 1e-6)); lt_hist[t] = np.log(max(base_dt, 1e-6))
        h = h + np.radians(dth[t])
        if guardrail:
            # wall_mode='reflect' bounces instead of parking the fish on the boundary;
            # it also leaves the heading INWARD, so the next bout is not blocked again.
            x, y, h = MB.advance(x, y, h, dval, cx, cy, R, eps,
                                 mode=cfg_feat.get('wall_mode', 'clip'), stats=wall_stats)
        else:
            x = x + dval * np.cos(h); y = y + dval * np.sin(h)
    out = dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth), dt_s=dtg, distance_px=dist, x_px=xs, y_px=ys)
    # guardrail interventions per rollout. Emitted dtheta is computed BEFORE the guardrail, so
    # without this the turn features cannot see how often the wall silently rewrote the path.
    out["n_hits"] = wall_stats.get("n_hit", 0)
    out["n_bounces"] = wall_stats.get("n_bounce", 0)
    if recompute_stim:
        out["stim_angle_gen"] = phi_gen
    if keep_features:
        out["X"] = Xrec                 # (n, D) angle features actually used
        out["dth_base"] = dth_base      # the raw emission SAMPLE (pre-vigor/artr/centre_bias),
                                        # i.e. the variable whose density the model defines
    return out


def _calibrate_vigor(records, W=60, seed=0):
    """Estimate the per-trajectory 'swim-vigor' latent from REAL data. The kin trunk has NO fish signal, so
    every rollout gets the ~same distance/dt distribution -- but real fish differ a lot in baseline step-size
    and timing (between-fish level) AND drift slowly in tempo within a trajectory. We model each channel c in
    {dt, dist} in LOG-units as  z_c[t] = u_c + d_c[t],  u_c ~ N(0, su_c^2) a per-trajectory INTERCEPT (fixes
    between-fish level [B]) and d_c[t] a stationary AR(1) slow DRIFT (persistence phi_c, std sd_c; fixes the
    within-fish window-median tempo drift [C2]). Injected at GENERATION only (no net retrain): the between-fish
    variance is entirely in the residual the trunk can't see, so adding it back is purely corrective.

    Calibration (all leak-free, on the train fold):
      su_c  = std ACROSS fish of each fish's mean(log value)                       -> [B] exactly.
      phi_c = grid-picked so a unit AR(1)'s window-median lag-1 acf matches real    -> [C2] persistence.
      sd_c  = scaled so the drift's window-median variance matches real            -> [C2] amplitude.
    Returns {'dt': [su,sd,phi,var], 'dist': [...], 'turn': [...]} (var=su^2+sd^2, for the exp de-bias at
    generation). The 'turn' channel scales |dtheta| per trajectory = between-fish turn-magnitude individuality;
    it is APPLIED at generation only when cfg_feat['vigor']['apply_turn'] (set by MLPMixKinDensity.fit).
    """
    rng = np.random.default_rng(seed)

    def _fmean(x):
        x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
        return float(np.log(x).mean()) if len(x) else np.nan

    def _winmed_centered(x):
        lx = np.log(np.clip(np.asarray(x, float), 1e-6, None))
        m = [np.nanmedian(lx[a:a + W]) for a in range(0, max(1, len(lx) - W + 1), W) if len(lx[a:a + W]) == W]
        m = np.asarray(m, float)
        return (m - m.mean()) if len(m) >= 3 else np.zeros(0)

    def _pooled_acf_var(series):
        num = den = 0.0; vs = []
        for s in series:
            s = np.asarray(s, float); s = s[np.isfinite(s)]
            if len(s) < 3:
                continue
            x = s - s.mean(); num += float((x[1:] * x[:-1]).sum()); den += float((x * x).sum()); vs.append(float(np.var(s)))
        return (num / den if den else np.nan), (float(np.mean(vs)) if vs else np.nan)

    lengths = [int(r.n) for r in records if int(r.n) >= W]

    def _sim_drift(phi):                                    # unit-stationary AR(1) over the real lengths
        series = []
        c = float(np.sqrt(1.0 - phi * phi))
        for n in lengths:
            e = rng.standard_normal(n); d = np.empty(n); d[0] = rng.standard_normal()
            for t in range(1, n):
                d[t] = phi * d[t - 1] + c * e[t]
            m = np.array([np.median(d[a:a + W]) for a in range(0, n - W + 1, W)])
            if len(m) >= 3:
                series.append(m - m.mean())
        return _pooled_acf_var(series)

    TURN_EPS = 1e-2                                       # deg floor so log|dtheta| is finite on forward bouts

    def _channel(getter, agg="logbout"):
        # agg='logbout' (dt/dist): per-fish mean(log x), window series median(log x) -- geometric level.
        # agg='meanlog' (turn |dtheta|): per-fish log(mean|x|), window series log(mean|x|) -- matches the
        #   absdth_mean C2ST feature; z scales |dtheta| multiplicatively so log(mean|dtheta|) shifts by z.
        if agg == "meanlog":
            def _lvl(r):
                x = np.abs(np.asarray(getter(r), float)); x = x[np.isfinite(x)]
                return float(np.log(max(x.mean(), TURN_EPS))) if len(x) else np.nan

            def _winser(r):
                x = np.abs(np.asarray(getter(r), float))
                m = [np.log(max(np.nanmean(x[a:a + W]), TURN_EPS))
                     for a in range(0, max(1, len(x) - W + 1), W) if len(x[a:a + W]) == W]
                m = np.asarray(m, float)
                return (m - m.mean()) if len(m) >= 3 else np.zeros(0)
        else:
            _lvl = lambda r: _fmean(getter(r))
            _winser = lambda r: _winmed_centered(getter(r))
        su = float(np.nanstd([_lvl(r) for r in records]))                    # [B]
        tgt_acf, tgt_var = _pooled_acf_var([_winser(r) for r in records])
        best = None
        for phi in (0.95, 0.97, 0.98, 0.985, 0.99, 0.993, 0.996, 0.998):
            a, v = _sim_drift(phi)
            score = abs(a - tgt_acf)
            if best is None or score < best[0]:
                best = (score, phi, v)
        _, phi, v = best
        sd = float(np.sqrt(max(tgt_var, 1e-9) / max(v, 1e-9)))
        return [su, sd, phi, su * su + sd * sd]

    def _tgt_var(getter):
        return _pooled_acf_var([_winmed_centered(getter(r)) for r in records])[1]

    turn = _channel(lambda r: r.dtheta, agg="meanlog")
    # turn is INTERCEPT-ONLY: the feature is log(mean|dtheta|), and a time-varying drift inside the convex
    # mean(exp(z)) amplifies the between-traj spread (gen over-shoots). A constant per-traj z factors out of the
    # mean, so log(mean|dtheta|) shifts by EXACTLY the intercept -> exact [B] match. (The tell is a level feature.)
    turn[1] = 0.0; turn[3] = turn[0] * turn[0]
    out = dict(dt=_channel(lambda r: r.dt_s), dist=_channel(lambda r: r.distance_px), turn=turn)
    # Target window-median variance per channel, kept so the drift can later be re-sized against the
    # rollout's OWN baseline. sd is fitted here so the INJECTED drift alone carries the full real
    # variance -- but the rollout already varies window-to-window on its own, and the two ADD.
    # Measured on a fitted model: real 0.0194 (dist) with a no-vigor baseline of 0.0135 gives 0.0349
    # once the drift is added, i.e. 1.80x real. See MLPMixKinDensity._rescale_vigor_drift.
    out["tgt_var"] = {"dt": _tgt_var(lambda r: r.dt_s), "dist": _tgt_var(lambda r: r.distance_px)}
    return out


def _turnsign_acf1(dth):
    """lag-1 autocorrelation of the turn SIGN over turning bouts (|dtheta| > DTHETA0); nan if too few."""
    d = np.asarray(dth, float); s = np.sign(d[np.abs(d) > M.DTHETA0]); s = s[s != 0]
    if len(s) < 10:
        return np.nan
    x = s - s.mean(); den = float((x * x).sum())
    return float((x[1:] * x[:-1]).sum() / den) if den else np.nan


def _gen_for(cfg_feat):
    """Which free-run generator this cfg belongs to.

    The generation-time calibrators (containment, vigor, artr, ...) all roll the model out, and
    the joint model has a different rollout because its heads are chained. Dispatching on the cfg
    rather than threading a `gen_fn` through every calibrator keeps the call sites unchanged and
    means a model can never be calibrated with the wrong generator.
    """
    if cfg_feat.get("joint"):
        import fit_MLP_joint as _J
        return _J.generate_mix_joint_loop
    return generate_mix_kin_loop


def _search_containment(model, records, cfg_feat, grid=(0.0, 1.0, 1.75, 2.5, 3.25), seed=0,
                        max_fish=12, u0=0.75, base=(4.0, 35.0), decouple=False, w_occ=2.6,
                        bulk_grid=(0.0, 2.0, 4.0, 6.0), shell_grid=(0.0, 25.0, 50.0, 80.0),
                        graded=False, u0_grid=(0.45, 0.55, 0.60, 0.70),
                        graded_shell_grid=(40.0, 80.0, 140.0, 200.0), tail_max=None, **gen_kw):
    """Scale the shell-gated inward bias so generated wall occupancy matches real.

    ``u0`` may be a TUPLE in the 1-D branch, in which case the gate position is swept jointly
    with the scale (the u0=0.65 sweep in the lineage doc lands occupancy exactly on real with
    the mean nearly flat -- the shipped 0.75 leaves it ~1/3 high). ``tail_max`` (if given)
    rejects any candidate whose generated P(r/R > 0.95) exceeds it -- real fish NEVER cross
    0.95, so mass there is free detectability no moment objective sees.

    The wall excess factorises as entry-rate x dwell, measured 2.6x and 2.1x real respectively
    (occupancy 4.7x). A CONSTANT inward bias only attacks entry -- it matches the mean r/R while
    leaving frac>0.85 at ~0.033 vs real 0.012. Gating part of the bias on being inside the shell
    attacks dwell as well. A 1-D scale over (bulk, shell) = s*(4, 35) deg spans the useful range:
    s=1 gave occupancy 0.028 (from 0.060, real 0.012), mean r/R 0.447 (real 0.456), the turn
    marginal untouched, and NO cost to the stimulus response (following curvature +0.065 -> +0.064,
    restoring-force zero-crossing 40.8 -> 42.0 px against real 41.4).

    Calibrated on TRAIN fish only. Returns the centre_bias dict.
    """
    def _ru(r):
        return np.hypot(np.asarray(r.x_px, float) - r.dish_cx,
                        np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R
    _occ = [float(np.mean(_ru(r) > 0.85)) for r in records]
    _mn = [float(np.mean(_ru(r))) for r in records]
    tgt, tgt_mean = float(np.mean(_occ)), float(np.mean(_mn))
    # Scale each objective by its BETWEEN-FISH sd, i.e. score the error in units of normal
    # individual variation -- which is what the window classifier actually sees, since its
    # features are z-scored across windows. Relative error is the wrong yardstick here: on a
    # ~1% occupancy it is enormous, on a mean near 0.5 it is tiny, so an unweighted sum of
    # relative errors is dominated ~115:1 by occupancy and runs the bias to the grid ceiling,
    # producing mean r/R 0.377 against real 0.509 -- a WORSE absolute error than doing nothing.
    sd_occ = max(float(np.std(_occ)), 1e-4)
    sd_mean = max(float(np.std(_mn)), 1e-4)
    # Wall occupancy is a ~1% probability whose per-fish spread is 7-12x, so a 6-fish estimate is
    # far too noisy to calibrate against: it picked scale 1.5, which measures 1.60x real occupancy
    # on all 15 fish, when 2.0-2.5 lands at 1.02-1.12x. Since rad_frac_wall is ~30% of ALL remaining
    # detectability, that under-shoot is the single most expensive estimation error in the pipeline.
    recs = list(records)[:max_fish]
    cf = dict(cfg_feat)

    def _roll_stats(cb):
        """(mean wall occupancy, mean r/R, mean P(r/R>0.95)) over the calibration fish."""
        cf["centre_bias"] = cb
        o_, m_, p_ = [], [], []
        for r in recs:
            g = _gen_for(cf)(model, r, cf, np.random.default_rng([seed, int(r.fish_id)]), **gen_kw)
            u = np.hypot(np.asarray(g["x_px"], float) - r.dish_cx,
                         np.asarray(g["y_px"], float) - r.dish_cy) / r.dish_R
            o_.append(float(np.mean(u > 0.85))); m_.append(float(np.mean(u)))
            p_.append(float(np.mean(u > 0.95)))
        return float(np.mean(o_)), float(np.mean(m_)), float(np.mean(p_))

    def _obj(o_, m_):
        return float(w_occ) * ((o_ - tgt) / sd_occ) ** 2 + ((m_ - tgt_mean) / sd_mean) ** 2

    def _tail_pen(p95):
        # SOFT penalty, not a veto. A hard filter was tried and misfired: on the indiv-widened
        # J10 not one graded candidate met the guard, so a lexicographic rule handed the win to
        # a 'compliant' 1-D config with mean r/R 0.352 -- worse on the moments the classifier
        # actually reads than the config the guard rejected (p95 excess 0.0005, i.e. 0.05% of
        # bouts). Quadratic in the EXCESS over tail_max, weighted 4x the occupancy term: a
        # marginal leak costs ~0.01 (never decisive), a real 0.5-1% leak costs 1-10 (decisive).
        if tail_max is None:
            return 0.0
        return 4.0 * float(w_occ) * (max(0.0, p95 - tail_max) / sd_occ) ** 2

    if graded:
        # WIDE GRADED bias: bulk pinned at ZERO and the gate opened to mid-dish, so the bias is
        # exactly nothing across the inner ~60% and ramps from there to the wall.
        #
        # Both other forms are on one trade curve -- recovering the mean radius costs wall
        # occupancy almost proportionally -- because they are wrong in opposite directions. The
        # 1-D form carries a constant `bulk` that acts EVERYWHERE (measured: it drags mean r/R to
        # 0.387 against a real 0.456, i.e. -0.92 between-fish SD, which makes rad_mean/rad_median
        # the largest single tell in the trajectory C2ST -- a defect MANUFACTURED by the patch,
        # since the native model sits at 0.4525). The decoupled form lowers bulk but keeps the gate
        # at u0 = 0.75, and a bias that only fires once the fish is already in the shell is too
        # late: the wall excess is an ENTRY problem, not a dwell problem.
        #
        # A wide graded bias acts on APPROACH -- while the fish is still heading outward -- so it
        # prevents visits rather than shortening them, and is identically zero in the centre so the
        # mean is untouched. Measured on re_swap (u0 = 0.60, shell = 80): mean r/R 0.419 (-0.49 sd,
        # vs -0.92 shipped) AND occupancy 1.08x real (vs 1.28x). Better on BOTH objectives, which
        # no point on the old trade curve achieves.
        gcurve = []
        for u0g in u0_grid:
            for shg in graded_shell_grid:
                cb = {"bulk": 0.0, "shell": float(shg), "u0": float(u0g)}
                o_, m_, p_ = _roll_stats(cb)
                gcurve.append((u0g, shg, o_, m_, _obj(o_, m_), p_))
        gbest = min(gcurve, key=lambda c: c[4] + _tail_pen(c[5]))
        edge = (gbest[0] in (u0_grid[0], u0_grid[-1])
                or gbest[1] in (graded_shell_grid[0], graded_shell_grid[-1]))
        if edge:
            print(f"    WARNING: graded containment pinned at a grid edge "
                  f"(u0={gbest[0]}, shell={gbest[1]}); widen the grid before trusting this arm")
        return {"bulk": 0.0, "shell": float(gbest[1]), "u0": float(gbest[0]), "graded": True,
                "target_occ": tgt, "target_mean": tgt_mean,
                "got_occ": gbest[2], "got_mean": gbest[3], "got_p95": gbest[5],
                "sd_occ": sd_occ, "sd_mean": sd_mean}

    u0s = (u0,) if np.isscalar(u0) else tuple(u0)
    curve = []
    for u0v in u0s:
        for sc in grid:
            o_, m_, p_ = _roll_stats({"bulk": base[0] * sc, "shell": base[1] * sc,
                                      "u0": float(u0v)})
            curve.append((sc, float(u0v), o_, m_, p_))
    # TWO objectives. Targeting occupancy alone drives the bias up until the fish is pulled too far
    # toward the centre, which just moves the error into rad_mean (also a C2ST feature). Score both
    # on their own scales and take the best compromise.
    # Weight each objective by how much the C2ST actually cares. Measured permutation
    # importances on the current best model: rad_frac_wall = 0.304 of ALL remaining
    # detectability (the single largest feature, 3x the next one), while rad_mean + rad_median
    # together are 0.116. Equal weighting in SD units -- which is what I used first -- is
    # statistically tidy but de-prioritises the feature that dominates the metric, and the
    # 1.8x-real residual occupancy it leaves is worth ~30% of the total signal.
    W_OCC, W_MEAN = float(w_occ), 1.0

    def _score(q):
        return (W_OCC * ((q[2] - tgt) / sd_occ) ** 2
                + W_MEAN * ((q[3] - tgt_mean) / sd_mean) ** 2)
    hit = min(curve, key=lambda q: _score(q) + _tail_pen(q[4]))
    sc = float(np.clip(hit[0], 0.0, 3.0))
    if len(u0s) > 1 and (hit[1] in (min(u0s), max(u0s)) or hit[0] in (grid[0], grid[-1])):
        print(f"    WARNING: containment pinned at a grid edge (scale={hit[0]}, u0={hit[1]}); "
              "widen the grid before trusting this arm")
    out = {"bulk": base[0] * sc, "shell": base[1] * sc, "u0": float(hit[1]), "scale": sc,
           "target_occ": tgt, "target_mean": tgt_mean, "got_occ": hit[2], "got_mean": hit[3],
           "got_p95": hit[4], "sd_occ": sd_occ, "sd_mean": sd_mean}
    if not decouple:
        return out
    # 2-D refinement. Locking bulk and shell to a fixed ratio forces one compromise between two
    # near-INDEPENDENT effects: the bulk term sets the mean r/R (it acts everywhere) while the
    # shell term sets the tail (it only switches on inside u0). The 1-D fit nails occupancy at the
    # mean's expense -- measured mean r/R 0.385 vs real 0.449, which puts rad_median back in the
    # top C2ST features even though wall_frac_gap is ~0. Coordinate descent from the 1-D solution:
    # bulk against the mean, shell against the tail, one pass each (they barely interact).
    def _eval(bulk, shell):
        return _roll_stats({"bulk": bulk, "shell": shell, "u0": out["u0"]})
    bulk, shell = out["bulk"], out["shell"]
    bulk = min(bulk_grid, key=lambda b: abs(_eval(b, shell)[1] - tgt_mean) / sd_mean)
    shell = min(shell_grid, key=lambda sh: abs(_eval(bulk, sh)[0] - tgt) / sd_occ)
    go, gm, gp = _eval(bulk, shell)
    # got_p95 re-measured at the FINAL (bulk, shell) -- keeping the 1-D hit's value would let a
    # bounds check read PASS for a config that was never measured.
    out.update(bulk=float(bulk), shell=float(shell), got_occ=go, got_mean=gm, got_p95=gp,
               decoupled=True)
    return out


# --------------------------------------------------------------------------- #
#  Shallow-border REJECTION: the containment mechanism that acts on POSITION
# --------------------------------------------------------------------------- #
# d0 is a PHYSICAL parameter -- the width of the rim the dish is too shallow to swim in -- so
# unlike GAIN_GRID this one is deliberately NOT extended when it pins. The data bracket it: real
# occupancy is 0.021% inside 1 mm, EXACTLY ZERO between 1 and 2 mm, and 0.005% in 2-3 mm, and the
# empty band above occupied ground is a discontinuity a continuous trajectory cannot make, so the
# sub-millimetre mass is tracking glitch. Letting a two-moment statistical objective run below
# 1 mm would be fitting that glitch. J20 pins at 1.0 (2026-08-24); 1.5 gives the search resolution
# inside the supported band instead of range outside it.
REJECT_D0_GRID = (1.0, 1.5, 2.0, 3.0)     # hard floor, mm from the physical wall
REJECT_D1_GRID = (6.0, 10.0, 14.0)        # full acceptance beyond this, mm
REJECT_BULK_GRID = (0.0, 2.0, 4.0)        # residual constant inward turn bias, deg (0 first)


def _radial_targets(records):
    """``(target_occ, target_mean, sd_occ, sd_mean)`` -- real P(r/R > 0.85) and mean r/R, plus
    the BETWEEN-FISH sd of each, which is the unit both containment searches score errors in.

    Same computation ``_search_containment`` does inline (kept there verbatim so that arm stays
    bit-identical); this is the shared copy for the rejection search."""
    def _ru(r):
        return np.hypot(np.asarray(r.x_px, float) - r.dish_cx,
                        np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R
    _occ = [float(np.mean(_ru(r) > 0.85)) for r in records]
    _mn = [float(np.mean(_ru(r))) for r in records]
    return (float(np.mean(_occ)), float(np.mean(_mn)),
            max(float(np.std(_occ)), 1e-4), max(float(np.std(_mn)), 1e-4))


def _search_wall_reject(model, records, cfg_feat, seed=0, max_fish=12, w_occ=2.6,
                        d0_grid=REJECT_D0_GRID, d1_grid=REJECT_D1_GRID,
                        bulk_grid=REJECT_BULK_GRID, tail_max=None, max_draws=20,
                        decouple=False, graded=False, **gen_kw):
    """Fit ``(bulk, d0, d1)`` -- the shallow-border boundary condition -- returning
    ``(centre_bias, wall_reject)``.

    WHAT THIS REPLACES, AND WHAT IT KEEPS. ``centre_bias`` carries two terms doing two different
    jobs: ``bulk`` (a small constant inward turn, acting everywhere -> sets the MEAN radius) and
    ``shell`` (a ramp inside ``u0`` -> sets the TAIL). The shell ramp is the damaging half: on the
    shipped J23 calibration it reaches 39 deg at the wall, which is why shell E|dtheta| reads ~100
    deg against a real 66, and buying the tail with it drags the mean to 0.428 against a target
    0.456. Rejection does the shell's job as a constraint on POSITION, so the emission's own turn
    distribution survives inside the shell. ``shell`` is therefore pinned to 0 here and ``bulk``
    stays a FITTED knob with 0 first in its grid.

    That last point is deliberate insurance rather than a preference. Archived measurements say
    the joint family's free mean r/R is already right (0.450-0.456 vs real 0.456) and only the
    tail is wrong, in which case the search picks ``bulk = 0`` and the arm is pure rejection. If
    that is not true of a particular fit, it picks 2 or 4 and still gets the tail from rejection
    at no turn-marginal cost. Hard-wiring ``contain=False`` would have bet the arm on one outcome.

    Same objective as ``_search_containment`` (occupancy and mean r/R in between-fish sd units,
    ``w_occ`` on occupancy, soft quadratic tail penalty), so the two are directly comparable.
    Coordinate descent, <= 21 rollout sets: ``(d0, d1)`` at bulk 0, then ``bulk``, then ``(d0,d1)``
    again if bulk moved.

    NOTE the tail guard stops being informative here: with ``d0 >= 2`` mm the region r/R > 0.96 is
    forbidden by construction, so ``P(r/R > 0.95)`` is ~0 for every candidate. Judge these arms on
    the radial SHAPE (rad_q90 / rad_median) and the shell turn profile instead.
    """
    if decouple or graded:
        raise ValueError("contain_mode='reject' is incompatible with contain_decouple / "
                         "contain_graded: all three parameterise the same shell term.")
    tgt, tgt_mean, sd_occ, sd_mean = _radial_targets(records)
    recs = list(records)[:max_fish]
    cf = dict(cfg_feat)
    cache = {}

    def _eval(bulk, d0, d1):
        key = (round(float(bulk), 6), round(float(d0), 6), round(float(d1), 6))
        if key in cache:
            return cache[key]
        cf["centre_bias"] = {"bulk": float(bulk), "shell": 0.0, "u0": 0.75}
        cf["wall_reject"] = {"d0": float(d0), "d1": float(d1), "max_draws": int(max_draws)}
        o_, m_, p_, dr_, fb_ = [], [], [], [], []
        for r in recs:
            g = _gen_for(cf)(model, r, cf, np.random.default_rng([seed, int(r.fish_id)]), **gen_kw)
            u = np.hypot(np.asarray(g["x_px"], float) - r.dish_cx,
                         np.asarray(g["y_px"], float) - r.dish_cy) / r.dish_R
            nb = max(len(u), 1)
            o_.append(float(np.mean(u > 0.85))); m_.append(float(np.mean(u)))
            p_.append(float(np.mean(u > 0.95)))
            dr_.append(float(g.get("n_draws", nb)) / nb)
            fb_.append(float(g.get("n_reject_fallback", 0)) / nb)
        out = (float(np.mean(o_)), float(np.mean(m_)), float(np.mean(p_)),
               float(np.mean(dr_)), float(np.mean(fb_)))
        cache[key] = out
        return out

    def _score(q):
        s = float(w_occ) * ((q[0] - tgt) / sd_occ) ** 2 + ((q[1] - tgt_mean) / sd_mean) ** 2
        if tail_max is not None:
            s += 4.0 * float(w_occ) * (max(0.0, q[2] - tail_max) / sd_occ) ** 2
        return s

    b0 = float(bulk_grid[0])
    pairs = [(d0, d1) for d0 in d0_grid for d1 in d1_grid if d1 > d0]
    d0b, d1b = min(pairs, key=lambda p: _score(_eval(b0, p[0], p[1])))
    bulk_b = min(bulk_grid, key=lambda bk: _score(_eval(bk, d0b, d1b)))
    if bulk_b != b0:                                   # bulk moved -> the (d0,d1) optimum may too
        d0b, d1b = min(pairs, key=lambda p: _score(_eval(bulk_b, p[0], p[1])))
    o_, m_, p_, dr_, fb_ = _eval(bulk_b, d0b, d1b)
    if d0b in (d0_grid[0], d0_grid[-1]) or d1b in (d1_grid[0], d1_grid[-1]):
        print(f"    WARNING: shallow-border ramp pinned at a grid edge (d0={d0b}, d1={d1b}); "
              "widen the grid before trusting this arm")
    if fb_ > 0.005:
        print(f"    WARNING: rejection fell back to the guardrail on {100 * fb_:.2f}% of bouts "
              f"(mean {dr_:.2f} draws/bout) -- max_draws={max_draws} may be too small")
    cb = {"bulk": float(bulk_b), "shell": 0.0, "u0": 0.75, "reject": True,
          "target_occ": tgt, "target_mean": tgt_mean,
          "got_occ": o_, "got_mean": m_, "got_p95": p_,
          "sd_occ": sd_occ, "sd_mean": sd_mean}
    wr = {"d0": float(d0b), "d1": float(d1b), "max_draws": int(max_draws),
          "draws_per_bout": dr_, "fallback_frac": fb_}
    return cb, wr


def _search_wall(model, records, cfg_feat, *, mode="bias", **kw):
    """Fit whichever containment mechanism ``mode`` selects and return the cfg_feat UPDATES.

    One dispatcher because the four fit() call sites (joint x2, fishre x2) must stay in step;
    a mechanism wired into three of them and not the fourth is exactly the class of bug the
    duplicated cfg builds in this family keep producing."""
    if mode == "reject":
        cb, wr = _search_wall_reject(model, records, cfg_feat, **kw)
        return {"centre_bias": cb, "wall_reject": wr}
    return {"centre_bias": _search_containment(model, records, cfg_feat, **kw)}


def _fit_qmap(model, records, cfg_feat, n_grid=256, seed=0, max_fish=10, **gen_kw):
    """Monotone quantile maps taking the model's OWN emitted marginals onto the REAL ones.

    Motivation: an oracle that fixes the per-feature marginals of the 39 trajectory-C2ST window
    statistics drops the classifier from 0.756 to 0.478 -- the real-vs-real floor -- so essentially
    ALL remaining detectability is marginal mis-calibration rather than joint or temporal structure.

    A quantile map is the right shape of correction: it matches the marginal exactly while being
    monotone, so it cannot leave the support and cannot reorder samples. That is what distinguishes
    it from ``kin_var_scale``, which widened the log-normal SYMMETRICALLY and made things worse --
    right variance, wrong shape.

    Calibrated on TRAIN fish only (both the real target and the model's own marginal), so it is
    deployable rather than an oracle -- unlike ``calibrate.MagCalibrated``, which maps onto the
    HELD-OUT fish's marginal and is therefore a diagnostic upper bound only.
    """
    recs = list(records)[:max_fish]
    gen = [_gen_for(cfg_feat)(model, r, cfg_feat, np.random.default_rng([seed, int(r.fish_id)]),
                                 **gen_kw) for r in recs]
    q = np.linspace(0.0, 1.0, n_grid)
    out = {}
    for ch, gkey, rget in (("dtheta", "dtheta", lambda r: np.asarray(r.dtheta, float)),
                           ("dist", "distance_px", lambda r: np.asarray(r.distance_px, float)),
                           ("dt", "dt_s", lambda r: np.asarray(r.dt_s, float))):
        gv = np.concatenate([np.asarray(g[gkey], float) for g in gen])
        rv = np.concatenate([rget(r) for r in recs])
        gv = gv[np.isfinite(gv)]; rv = rv[np.isfinite(rv)]
        if len(gv) < 100 or len(rv) < 100:
            continue
        out[ch] = dict(src=np.quantile(gv, q).tolist(), dst=np.quantile(rv, q).tolist())
    return out


def _apply_qmap(qm, ch, v):
    """Interpolate v through the stored monotone map (identity outside the fitted range)."""
    t = qm.get(ch)
    if not t:
        return v
    return float(np.interp(v, t["src"], t["dst"]))

def _kin_acf_targets(records):
    """Real lag-1 autocorrelation of log-distance and log-dt (the calibration targets)."""
    def _a(x):
        v = np.log(np.clip(np.asarray(x, float), 1e-6, None))
        v = v[np.isfinite(v)]
        if len(v) < 5:
            return np.nan
        v = v - v.mean()
        d = float(np.sum(v * v))
        return float(np.sum(v[1:] * v[:-1]) / d) if d else np.nan
    def _v(x):
        v = np.log(np.clip(np.asarray(x, float), 1e-6, None)); v = v[np.isfinite(v)]
        return float(np.var(v)) if len(v) > 2 else np.nan
    return ((float(np.nanmean([_a(r.distance_px) for r in records])),
             float(np.nanmean([_a(r.dt_s) for r in records]))),
            (float(np.nanmean([_v(r.distance_px) for r in records])),
             float(np.nanmean([_v(r.dt_s) for r in records]))))


def _search_kin_feedback(model, records, cfg_feat, targets, var_targets, grid=(1.0, 0.75, 0.5, 0.3),
                         seed=0, max_fish=8, **gen_kw):
    """Pick the per-channel kinematic feedback shrinkage that reproduces the real lag-1 acf.

    Teacher-forced training fits the head against the REAL history; free-running feeds it its own
    noisy samples, so the loop re-amplifies and over-persists (measured gen log-distance acf +0.42
    vs real +0.23). Shrinking the z-scored history toward the mean damps that loop. The response is
    smooth and monotone in the shrinkage, so a small grid plus linear interpolation suffices --
    same approach as ``_search_artr_beta``. Calibrated on TRAIN fish only.
    """
    def _a(x):
        v = np.log(np.clip(np.asarray(x, float), 1e-6, None)); v = v[np.isfinite(v)]
        if len(v) < 5:
            return np.nan
        v = v - v.mean(); d = float(np.sum(v * v))
        return float(np.sum(v[1:] * v[:-1]) / d) if d else np.nan

    def _v(x):
        v = np.log(np.clip(np.asarray(x, float), 1e-6, None)); v = v[np.isfinite(v)]
        return float(np.var(v)) if len(v) > 2 else np.nan

    # The acf response is smooth and monotone in the shrinkage, so 4 grid points plus linear
    # interpolation is as good as 6, and the target is a POOLED mean over fish, so a subset
    # suffices. Both matter: this runs inside fit(), once per CV fold.
    records = list(records)[:max_fish]
    cf = dict(cfg_feat)
    curve = []
    for g in grid:
        cf["kin_feedback"] = g
        gen = [_gen_for(cf)(model, r, cf, np.random.default_rng([seed, int(r.fish_id)]),
                            **gen_kw) for r in records]
        curve.append((g, float(np.nanmean([_a(x["distance_px"]) for x in gen])),
                      float(np.nanmean([_a(x["dt_s"]) for x in gen])),
                      float(np.nanmean([_v(x["distance_px"]) for x in gen])),
                      float(np.nanmean([_v(x["dt_s"]) for x in gen]))))
    out = {}
    for ch, col, tgt in (("dist", 1, targets[0]), ("dt", 2, targets[1])):
        pts = [(c[0], c[col]) for c in curve if np.isfinite(c[col])]
        best = min(pts, key=lambda gv: abs(gv[1] - tgt))
        # linear interpolation between the two grid points that bracket the target
        lo = [q for q in pts if q[1] <= tgt]
        hi = [q for q in pts if q[1] >= tgt]
        if lo and hi:
            a = max(lo, key=lambda q: q[1]); b = min(hi, key=lambda q: q[1])
            out[ch] = (float(a[0]) if a[1] == b[1] else
                       float(a[0] + (b[0] - a[0]) * (tgt - a[1]) / (b[1] - a[1])))
        else:
            out[ch] = float(best[0])
        out[ch] = float(np.clip(out[ch], 0.1, 1.0))
    # VARIANCE COMPENSATION, free: the grid rollouts already measured the log-variance at each
    # shrinkage, so interpolate it at the chosen lambda and solve c analytically. Rescaling the
    # emitted series about its mean by c multiplies its variance by c^2 and leaves the
    # autocorrelation unchanged, so this does not disturb the lambda that was just fitted
    # (measured residual coupling: dist acf drifts ~0.03 over c = 1.0 -> 1.5, from the clip).
    var_scale = {}
    for ch, col, tv in (("dist", 3, var_targets[0]), ("dt", 4, var_targets[1])):
        pts = sorted([(c[0], c[col]) for c in curve if np.isfinite(c[col])])
        if not pts or not np.isfinite(tv) or tv <= 0:
            var_scale[ch] = 1.0
            continue
        xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
        at_lam = float(np.interp(out[ch], xs, ys))
        var_scale[ch] = float(np.clip(np.sqrt(tv / max(at_lam, 1e-9)), 0.5, 3.0))
    return out, var_scale

def _artr_targets(records):
    """Real turn-DIRECTION persistence targets: (rho, acf_lag1). rho from the mean same-direction run-length
    (switch prob ~ 1/runlen) -> the ARTR latent's AR(1) persistence; acf_lag1 = the calibration target."""
    def _runlen(s):
        runs, c = [], 1
        for i in range(1, len(s)):
            c = c + 1 if s[i] == s[i - 1] else (runs.append(c) or 1)
        runs.append(c)
        return float(np.mean(runs))
    acfs, rls = [], []
    for r in records:
        d = np.asarray(r.dtheta, float); s = np.sign(d[np.abs(d) > M.DTHETA0]); s = s[s != 0]
        if len(s) < 10:
            continue
        acfs.append(_turnsign_acf1(r.dtheta)); rls.append(_runlen(s))
    acf = float(np.nanmean(acfs)); rl = float(np.nanmean(rls))
    rho = float(np.clip(1.0 - 1.0 / max(rl, 1.05), 0.1, 0.95))
    return rho, acf


def _indiv_targets(records):
    """Real BETWEEN-FISH spreads for the two channels vigor does not cover.

    Measured on all 15: forward fraction mean 0.117 sd 0.054 (range 0.062-0.277), mean r/R
    sd 0.075. Generated: 0.005 and 0.016-0.044. Every rollout is the average fish.
    """
    fw, rad = [], []
    for r in records:
        a = np.abs(np.asarray(r.dtheta, float))
        fw.append(float(np.mean(a[np.isfinite(a)] < M.DTHETA0)))
        u = _radial_frac(r, r.x_px, r.y_px)
        if u is not None:
            rad.append(float(np.mean(u)))
    return (float(np.std(fw)) if len(fw) > 1 else 0.0,
            float(np.std(rad)) if len(rad) > 1 else 0.0)


def _radial_frac(rec, x, y):
    cx, cy, R = getattr(rec, "dish_cx", None), getattr(rec, "dish_cy", None), getattr(rec, "dish_R", None)
    if cx is None or R is None or x is None:
        return None
    u = np.hypot(np.asarray(x, float) - cx, np.asarray(y, float) - cy) / float(R)
    u = u[np.isfinite(u)]
    return u if len(u) else None


def _search_indiv(model, records, cfg_feat, targets, *, closed_loop, guardrail, eps, stim_loop,
                  seed=0, n_fish=10, reps=2, fw_grid=(0.0, 0.25, 0.45, 0.65),
                  rad_probe=0.6, w1_tol=1.15):
    """Calibrate the two per-trajectory latent SDs, CONSTRAINED so spread cannot be bought by
    wrecking the turn marginal.

    WHY THIS IS A GRID AND NOT A 2-POINT SOLVE. The first version extrapolated linearly from
    s=0 and s=1. On the jindiv run that chose fw=1.02, which STILL missed the forward-fraction
    target (0.027 achieved vs 0.054 wanted) while DOUBLING abs_dtheta_w1 (6.83 -> 12.59) -- the
    response is strongly sublinear, so extrapolating from two points demands a huge s and the
    marginal pays for it. A monotone walk with a marginal guard cannot do that: it stops at the
    last s whose abs_dtheta_w1 is still within ``w1_tol`` of the untreated model, even if the
    spread target is not reached. Under-shooting the spread is a much cheaper error than
    destroying the distribution it lives in.

    WHY THE RADIAL LATENT GETS ITS OWN PASS. Sharing one pass with the fw probe (an earlier
    speed optimisation) made its solve read noise: the same calibrator returned rad=0.0 on one
    rung and 1.2 -- the clip ceiling -- on another, from identical data. The fw probe perturbs
    the radial measurement it is supposed to isolate, so they must be measured separately, and
    the probe point is large enough (0.6) for the signal to clear the sampling noise.

    The quantity matched is a BETWEEN-TRAJECTORY spread, so it is measured ACROSS rollouts
    (n_fish x reps draws), never within one.
    """
    from scipy.stats import wasserstein_distance
    sub = list(records)[:n_fish]
    tgt_fw, tgt_rad = targets
    real_absdth = np.abs(np.concatenate([np.asarray(r.dtheta, float) for r in sub]))

    def measure(s_fw, s_rad):
        """(forward-fraction spread, mean-r/R spread, abs_dtheta_w1) across rollouts."""
        trial = dict(cfg_feat); trial["indiv"] = dict(fw=float(s_fw), rad=float(s_rad))
        fw, rad, ad = [], [], []
        for rep in range(reps):
            rng = np.random.default_rng([seed, 7717, rep])
            for r in sub:
                g = _gen_for(trial)(model, r, trial, rng, closed_loop=closed_loop,
                                    guardrail=guardrail, eps=eps, stim_loop=stim_loop)
                a = np.abs(np.asarray(g["dtheta"], float))
                fw.append(float(np.mean(a < M.DTHETA0))); ad.append(a)
                u = _radial_frac(r, g.get("x_px"), g.get("y_px"))
                if u is not None:
                    rad.append(float(np.mean(u)))
        return (float(np.std(fw)) if len(fw) > 1 else 0.0,
                float(np.std(rad)) if len(rad) > 1 else 0.0,
                float(wasserstein_distance(real_absdth, np.concatenate(ad))))

    base = measure(0.0, 0.0)
    s_fw, got_fw, w1 = 0.0, base[0], base[2]
    for s in fw_grid[1:]:                       # monotone walk upward, stop on the guard
        m = measure(s, 0.0)
        if m[2] > w1_tol * base[2]:             # marginal degraded past tolerance -> keep last good
            break
        s_fw, got_fw, w1 = s, m[0], m[2]
        if m[0] >= tgt_fw:                      # spread target reached
            break
    rp = measure(0.0, rad_probe)                # radial gets its OWN pass (see above)
    s_rad = (0.0 if rp[1] - base[1] <= 1e-6
             else rad_probe * (tgt_rad - base[1]) / (rp[1] - base[1]))
    # rad is clipped HARD at 0.5. Both solves so far have run to whatever ceiling they were
    # given (1.2 then 0.8), and at 1.2 the b_both rung produced radial dispersion 5.55 against a
    # target of 1.0 and wall_frac_gap 0.047 -- the containment re-fit corrects the MEAN occupancy
    # but cannot correct its spread. exp(+-0.5) is a 0.61x-1.65x swing in the bias, which is a
    # real spatial personality without letting any trajectory lose containment. Under-shooting
    # the spatial spread is the cheap error here, exactly as it is for fw.
    return dict(fw=float(np.clip(s_fw, 0.0, 0.8)), rad=float(np.clip(s_rad, 0.0, 0.5)),
                got_fw=got_fw, tgt_fw=float(tgt_fw), w1_base=base[2], w1_at_fw=w1)


def _search_rad_indiv(model, records, cfg_feat, tgt_rad, *, closed_loop, guardrail, eps,
                      stim_loop, seed=0, n_fish=10, reps=2, probe=0.6):
    """Calibrate the fishre radial latent SD (``fish_re['rad_sd']``) by method of moments.

    The latent multiplies the containment bias per trajectory (``cb *= exp(u_rad_re)`` in
    ``generate_mix_joint_loop``), so it is GENERATION-ONLY: ``score``/``_ghq`` read only
    ``fish_re['sd']`` and the likelihood cannot see it -- exactly like J10's ``indiv['rad']``,
    which is why moments (not likelihood) choose it. Matched moment = the across-rollout spread
    of per-rollout mean r/R; target = the real between-fish SD (~0.075). Two-point linear solve
    on its OWN pass (sharing a pass with other probes made the earlier radial solve read noise),
    clipped hard at 0.5: exp(+-0.5) is a 0.61-1.65x swing in the bias -- a real spatial
    personality without letting any trajectory lose containment. Measured across n_fish x reps
    rollouts, never within one; per-checkpoint radial VALIDATION still needs all 15 fish."""
    sub = list(records)[:n_fish]

    def measure(s):
        trial = dict(cfg_feat)
        trial["fish_re"] = dict(cfg_feat["fish_re"])
        trial["fish_re"]["rad_sd"] = float(s)
        rad = []
        for rep in range(reps):
            for r in sub:
                # rng keyed PER ROLLOUT (like the harness), never shared across fish: a shared
                # generator makes each fish's spawn children depend on how many spawns earlier
                # fish consumed, and the base-vs-probe contrast stops being CRN-paired.
                rng = np.random.default_rng([seed, 7719, rep, int(r.fish_id)])
                g = _gen_for(trial)(model, r, trial, rng, closed_loop=closed_loop,
                                    guardrail=guardrail, eps=eps, stim_loop=stim_loop)
                u = _radial_frac(r, g.get("x_px"), g.get("y_px"))
                if u is not None:
                    rad.append(float(np.mean(u)))
        return float(np.std(rad)) if len(rad) > 1 else 0.0

    base = measure(0.0)
    got = measure(probe)
    s_raw = (float("inf") if got - base <= 1e-6
             else probe * (float(tgt_rad) - base) / (got - base))
    if s_raw > 1.0:
        # The PAIRED contrast says the lever is INERT, not noisy: on a graded (bulk=0) bias the
        # measured slope was ~0.0003 spread per unit rad_sd -- the bias only acts in the outer
        # ring, so scaling it per trajectory cannot move per-fish MEAN r/R. Saturating the clip
        # instead was measured to pay ~0.035 of mean r/R (via the post-widening containment
        # re-fit) for ~+0.004 of spread. Return 0: no widening, no re-fit cost.
        print(f"    rad_indiv: lever inert (solve wants rad_sd {s_raw:.1f}, clip is 0.5; "
              f"base {base:.4f} -> probe {got:.4f}, target {float(tgt_rad):.4f}); returning 0")
        return dict(rad_sd=0.0, base_sd=base, probe_sd=got, tgt=float(tgt_rad), inert=True)
    return dict(rad_sd=float(np.clip(s_raw, 0.0, 0.5)), base_sd=base, probe_sd=got,
                tgt=float(tgt_rad))


def _search_rad_gate(model, records, cfg_feat, tgt_rad, *, closed_loop, guardrail, eps,
                     stim_loop, seed=0, n_fish=10, reps=2, probe=0.10, clip=0.20):
    """Per-trajectory containment GATE offset sd (cfg ``rad_gate={'sd': s}``) by moments.

    Radial-individuality mechanism v2. v1 (u_rad/rad_sd, multiplicative on the bias) is measured
    INERT on graded bulk=0 biases -- scaling a bias that only acts in the outer ring cannot move
    a trajectory's MEAN r/R. Shifting the GATE moves where avoidance engages: the recal sweeps
    measured u0 0.60 -> 0.45 moving mean r/R by ~0.03, so a +-1 sd swing of ~0.1 in u0 is a real
    lever. The generator clips each personal gate to [0.20, 0.90]. JOINT-family generator only
    (generate_mix_joint_loop); the kin generator ignores the key.

    Same discipline as _search_rad_indiv: rng keyed per rollout, spawn gated on KEY presence so
    base (sd=0.0) and probe are CRN-paired; two-point solve with an inertness guard; matched
    moment = across-rollout spread of per-rollout mean r/R (target = real between-fish SD
    ~0.075). Per-checkpoint radial VALIDATION still needs all 15 fish."""
    sub = list(records)[:n_fish]

    def measure(s):
        trial = dict(cfg_feat)
        trial["rad_gate"] = {"sd": float(s)}
        rad = []
        for rep in range(reps):
            for r in sub:
                rng = np.random.default_rng([seed, 7723, rep, int(r.fish_id)])
                g = _gen_for(trial)(model, r, trial, rng, closed_loop=closed_loop,
                                    guardrail=guardrail, eps=eps, stim_loop=stim_loop)
                u = _radial_frac(r, g.get("x_px"), g.get("y_px"))
                if u is not None:
                    rad.append(float(np.mean(u)))
        return float(np.std(rad)) if len(rad) > 1 else 0.0

    base = measure(0.0)
    got = measure(probe)
    s_raw = (float("inf") if got - base <= 1e-6
             else probe * (float(tgt_rad) - base) / (got - base))
    if s_raw > 2 * clip:
        print(f"    rad_gate: lever inert (solve wants sd {s_raw:.2f}, clip is {clip}; "
              f"base {base:.4f} -> probe {got:.4f}, target {float(tgt_rad):.4f}); returning 0")
        return dict(sd=0.0, base_sd=base, probe_sd=got, tgt=float(tgt_rad), inert=True)
    return dict(sd=float(np.clip(s_raw, 0.0, clip)), base_sd=base, probe_sd=got,
                tgt=float(tgt_rad))


def _search_innov_comp(model, records, cfg_feat, *, closed_loop, guardrail, eps, stim_loop,
                       seed=0, n_fish=10, iters=2):
    """Solve the innovation scale c so free-run sd(log distance)/sd(log dt) match the train fold.

    Not analytic, because the widened innovation feeds BACK through the (shrunken) history, so the
    map c -> sd is only approximately linear. Two fixed-point steps converge from ~0.83 to ~1.0 in
    practice and cost one rollout pass each -- the same price as _rescale_vigor_drift.
    """
    sub = records[:n_fish]
    ld = np.concatenate([np.log(np.clip(np.asarray(r.distance_px, float), 1e-6, None)) for r in sub])
    lt = np.concatenate([np.log(np.clip(np.asarray(r.dt_s, float), 1e-6, None)) for r in sub])
    tgt_d = float(np.std(ld[np.isfinite(ld)])); tgt_t = float(np.std(lt[np.isfinite(lt)]))
    c_d = c_t = 1.0
    for _ in range(iters):
        trial = dict(cfg_feat); trial["kin_innov"] = dict(dist=c_d, dt=c_t)
        rng = np.random.default_rng([seed, 5501])
        gd, gt = [], []
        for r in sub:
            g = _gen_for(trial)(model, r, trial, rng, closed_loop=closed_loop,
                                guardrail=guardrail, eps=eps, stim_loop=stim_loop)
            gd.append(np.log(np.clip(np.asarray(g["distance_px"], float), 1e-6, None)))
            gt.append(np.log(np.clip(np.asarray(g["dt_s"], float), 1e-6, None)))
        sd_d = float(np.std(np.concatenate(gd))); sd_t = float(np.std(np.concatenate(gt)))
        if sd_d > 1e-6:
            c_d = float(np.clip(c_d * tgt_d / sd_d, 0.5, 3.0))
        if sd_t > 1e-6:
            c_t = float(np.clip(c_t * tgt_t / sd_t, 0.5, 3.0))
    return dict(dist=c_d, dt=c_t)


def _search_artr_beta(model, records, cfg_feat, rho, target_acf, *, closed_loop, guardrail, eps, stim_loop,
                      seed=0, n_fish=1000):
    """Calibrate the L/R bias strength beta so the GENERATED turn-sign lag-1 acf matches real. The beta->acf map
    is smooth & ~linear near the target, so we 2-point interpolate: measure gen acf at beta=0 (base) and beta=1,
    then solve beta = (target - acf0)/(acf1 - acf0). Generation-only: the closed-loop + head-persistence feedback
    is absorbed into the measured slope. Uses ALL train fish (turn persistence is strongly fish-specific, so a
    small subset's base is not representative -> population a0/a1)."""
    sub = records[:n_fish]

    def gen_acf(beta):
        trial = dict(cfg_feat); trial["artr"] = dict(rho=rho, beta=float(beta))
        rng = np.random.default_rng(seed)
        return float(np.nanmean([_turnsign_acf1(generate_mix_kin_loop(model, r, trial, rng, closed_loop=closed_loop,
                     guardrail=guardrail, eps=eps, stim_loop=stim_loop)["dtheta"]) for r in sub]))

    a0, a1 = gen_acf(0.0), gen_acf(1.0)
    slope = a1 - a0
    beta = 1.0 if not np.isfinite(slope) or slope <= 1e-4 else (target_acf - a0) / slope
    beta = float(np.clip(beta, 0.0, 3.0))
    return beta, float(a0 + slope * beta)                # (beta, predicted gen acf at that beta)


class MLPMixKinDensity(MLPMixLoopDensity):
    """Honest closed-loop mix model with JOINT AR kinematics (distance log-normal + dt 2-comp log-normal,
    autoregressive on kin history) instead of i.i.d. pool draws -- targets the trajectory C2ST (the i.i.d.
    distance/dt were its dominant tell). ``fit`` trains the joint emission; ``score`` (Metric A) stays
    TURN-ONLY (inherited: ``MDNMixKin.forward`` returns the angle mix), so bits/bout is comparable."""

    def __init__(self, *args, K_kin=3, kin_wd=1e-4, kin_hidden=32, vigor=False, vigor_turn=False,
                 artr=False, vigor_drift_calib=False, kin_fb_calib=False,
                 kin_var_comp=False, contain=False, contain2d=False, n_dist=1, n_dt=2,
                 contain_fish=12, contain_w_occ=2.6, indiv=False, innov_comp=False,
                 epochs_override=None, patience_override=None, qmap=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_kin = True
        # per-trajectory slow 'swim-vigor' latent (individuality + tempo drift), injected at generation only
        # (no net retrain). Fixes the between-fish level [B] + within-fish drift [C2] that the pooled trunk misses.
        # vigor_turn also scales |dtheta| per trajectory (turn-magnitude individuality); it implies the vigor dict.
        # artr = a slow L/R turn-bias oscillator (ARTR) that adds the turn-DIRECTION persistence (acf/transitions)
        # the pooled emission under-produces in free-run; also generation-only, calibrated to the real sign-acf.
        self.vigor = vigor or vigor_turn
        self.vigor_turn = vigor_turn
        self.artr = artr
        # re-size the vigor DRIFT against the rollout's own window-to-window variance instead of
        # assuming it starts at zero (default False keeps every existing zoo key bit-identical)
        self.vigor_drift_calib = vigor_drift_calib
        # shrink the kinematic AR feedback so the free-run lag-1 acf matches real (exposure bias)
        self.kin_fb_calib = kin_fb_calib
        # restore the dispersion that kin_feedback's shrinkage removes (orthogonal knob)
        self.kin_var_comp = kin_var_comp
        # shell-gated inward bias + reflecting wall, calibrated to real wall occupancy
        self.contain = contain
        # fit bulk (sets mean r/R) and shell (sets the tail) separately instead of on a fixed ratio
        self.contain2d = contain2d
        # calibration settings, exposed so old vs new can be contrasted in one SELECT.
        # v1 was (6 fish, equal weight) and under-shot to 1.60x real occupancy.
        self.contain_fish, self.contain_w_occ = int(contain_fish), float(contain_w_occ)
        # indiv      = per-trajectory latents on turning style and spatial preference (the two
        #              channels vigor does not cover; between-fish share 0.074 vs real 0.255).
        # innov_comp = widen the kin innovation to restore the spread kin_feedback removes,
        #              without disturbing the acf the shrinkage exists to fix.
        self.indiv, self.innov_comp = bool(indiv), bool(innov_comp)
        # cfg['mlp_epochs'] normally WINS over the constructor, which makes it impossible to
        # put two epoch budgets in ONE SELECT. These win over cfg instead, so a convergence
        # contrast runs PAIRED rather than across runs (cross-run comparison is unreliable).
        self.epochs_override, self.patience_override = epochs_override, patience_override
        # monotone map from the model's own emitted marginals onto the REAL train-fold ones
        self.qmap = qmap
        # mixture components on the kinematic heads. The real upper tails are far heavier than
        # 1-comp distance / 2-comp dt can carry (gen dt q99.9 = 24.6 s vs real 103.6; only
        # 0.055% of generated bouts exceed 30 s vs 0.63% real), which collapses every
        # tail-sensitive window feature. (2, 3) gives each channel a rare slow mode.
        self.n_dist, self.n_dt = int(n_dist), int(n_dt)
        # SEPARATE kin trunk (isolates distance/dt from the position/turn autocorrelation) + a SHORT history:
        # on the separate trunk the AR self-feedback over-persists (acf +0.10 over real at every K_kin), and MORE
        # lags drift the rollout toward the q99.9 clip -> fatter marginal (dist_w1 1.2 @ K_kin=8 vs 0.32 @ K_kin=3).
        # K_kin=3 minimises the marginal (the direct c2st_traj feature) while keeping the lag-1 structure.
        self.K_kin = K_kin
        self.kin_wd = kin_wd
        self.kin_hidden = kin_hidden
        self.name = (f"{self.name}_kin" + ("_vigor" if self.vigor else "") + ("_turn" if vigor_turn else "")
                     + ("_artr" if artr else ""))

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
        adist = np.concatenate([r.distance_px for r in train_records]); adist = adist[np.isfinite(adist) & (adist > 0)]
        adt = np.concatenate([r.dt_s for r in train_records]); adt = adt[np.isfinite(adt) & (adt > 0)]
        self._kin_scales = (float(np.log(adist).mean()), float(max(np.log(adist).std(), 1e-3)),
                            float(np.log(adt).mean()), float(max(np.log(adt).std(), 1e-3)))
        self._kin_clip = (float(np.quantile(adist, 0.999)), float(np.quantile(adt, 0.999)))   # realistic caps
        # use_kin=False => the ANGLE feature matrix (build_feature_matrix) excludes the kin block; the kin
        # history is fed ONLY to the separate trunk (Xk below). kin_scales/K_kin/kin_clip stay for the trunk.
        self._cfg_feat = dict(K=self.K, stim_mode=self.stim_mode, include_symbols=self.include_symbols,
                              dth_scale=self._dth_scale, rich_scales=self._rich_scales,
                              use_border=self.use_border, border_mode=self.border_mode,
                              n_dist=self.n_dist, n_dt=self.n_dt,
                              wall_mode=self.wall_mode,
                              use_adapt=self.use_adapt,
                              adapt_lambda=self.adapt_lambda, use_kin=False, kin_scales=self._kin_scales,
                              K_kin=self.K_kin, kin_clip=self._kin_clip)
        if self.vigor:                                    # estimate the swim-vigor latent from the train fold
            vg = _calibrate_vigor(train_records)
            vg["apply_turn"] = bool(self.vigor_turn)      # dt/dist always applied; turn only when vigor_turn
            self._cfg_feat["vigor"] = vg
        X = np.concatenate([MB.build_feature_matrix(r, self._cfg_feat) for r in train_records], 0)
        Xk = np.concatenate([MB._kin_history_block(r, self._cfg_feat) for r in train_records], 0)
        ya = np.concatenate([r.dtheta.astype(np.float32) for r in train_records])
        yd = np.concatenate([np.asarray(r.distance_px, np.float32) for r in train_records])
        yt = np.concatenate([np.asarray(r.dt_s, np.float32) for r in train_records])
        self._model, self.history = fit_mix_kin(X, Xk, ya, yd, yt, self.hidden, kin_hidden=self.kin_hidden,
                                                epochs=self.epochs, lr=self.lr, patience=self.patience,
                                                batch_size=self.batch_size, seed=self.seed, kin_wd=self.kin_wd,
                                                n_dist=self.n_dist, n_dt=self.n_dt)
        if self.vigor and self.vigor_drift_calib:         # needs the FITTED net (rolls out to measure)
            self._rescale_vigor_drift(train_records)
        if self.kin_fb_calib:                             # match the free-run kinematic acf to real
            tg, vtg = _kin_acf_targets(train_records)
            kfb, kvs = _search_kin_feedback(
                self._model, train_records, self._cfg_feat, tg, vtg,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
            self._cfg_feat['kin_feedback'] = kfb
            if self.kin_var_comp:                         # restore the spread the shrinkage cost
                self._cfg_feat['kin_var_scale'] = kvs
        if self.qmap:                                     # BEFORE containment: the map rescales
            # step length, so the wall bias must be calibrated against the MAPPED rollout
            self._cfg_feat['qmap'] = _fit_qmap(
                self._model, train_records, self._cfg_feat,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
        if self.contain:                                  # match generated wall occupancy to real
            self._cfg_feat['wall_mode'] = 'reflect'
            self._cfg_feat['centre_bias'] = _search_containment(
                self._model, train_records, self._cfg_feat,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed, decouple=self.contain2d,
                max_fish=self.contain_fish, w_occ=self.contain_w_occ)
        # ORDER MATTERS. innov_comp is solved BEFORE indiv because it changes the per-bout spread
        # that the individuality search then measures on top of; and both come after contain,
        # because widening the model increases wall contact and the containment bias is what
        # absorbs that (measured corr(movement spread, wall_frac_gap) = +0.59).
        if self.innov_comp:                               # restore the spread kin_feedback removed
            self._cfg_feat["kin_innov"] = _search_innov_comp(
                self._model, train_records, self._cfg_feat,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
        if self.indiv:                                    # per-trajectory turning style + spatial preference
            self._cfg_feat["indiv"] = _search_indiv(
                self._model, train_records, self._cfg_feat, _indiv_targets(train_records),
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
        if self.contain and (self.indiv or self.innov_comp):
            # RE-FIT containment with the latents in place. The first pass calibrated a narrower
            # model; both new mechanisms widen movement, and movement spread drives wall contact
            # (measured corr +0.59) while rad_frac_wall is the single top C2ST feature. Skipping
            # this is how contain_vc lost: it widened the model and left the bias where it was.
            self._cfg_feat['centre_bias'] = _search_containment(
                self._model, train_records, self._cfg_feat,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed + 1, decouple=self.contain2d,
                max_fish=min(self.contain_fish, 8), w_occ=self.contain_w_occ)
        if self.artr:                                     # calibrate the ARTR L/R oscillator on the fitted model
            rho_ar, tgt = _artr_targets(train_records)
            beta, gacf = _search_artr_beta(self._model, train_records, self._cfg_feat, rho_ar, tgt,
                                           closed_loop=self.closed_loop, guardrail=self.guardrail,
                                           eps=self.eps, stim_loop=self.stim_loop, seed=self.seed)
            self._cfg_feat["artr"] = dict(rho=rho_ar, beta=beta)
        self.config = dict(mode="full", hidden=self.hidden, n_features=X.shape[1], K=self.K,
                           stim_mode=self.stim_mode, use_border=self.use_border, use_kin=True,
                           kin_features=Xk.shape[1], kin_hidden=self.kin_hidden,
                           n_dist=self.n_dist, n_dt=self.n_dt)   # load_mix rebuilds the heads from these
        return self

    def _rescale_vigor_drift(self, train_records, n_fish=8, W=60, seed=0):
        """Re-size the vigor DRIFT so gen window-to-window variance matches real, not exceeds it.

        ``_calibrate_vigor`` picks sd so the injected AR(1) drift alone reproduces the real
        window-median variance of log(value). But the rollout ALREADY varies window-to-window
        from its own AR kinematics and sampling noise, and the two variances add. Measured on a
        fitted model: log-distance real 0.0194, no-vigor baseline 0.0135, with-vigor 0.0349
        (= baseline + target, 1.80x real); log-dt real 0.0341, baseline 0.0216, with 0.0554.

        Fix: measure the baseline by rolling out with the drift switched off (intercept left on --
        it shifts each trajectory's level and cannot affect a WITHIN-trajectory centred series),
        then scale sd by sqrt(max(target - baseline, 0) / target). Train fold only, so leak-free.
        """
        vg = self._cfg_feat.get("vigor")
        if not vg or "tgt_var" not in vg:
            return
        recs = list(train_records)[:n_fish]
        saved = {c: float(vg[c][1]) for c in ("dt", "dist")}
        for c in ("dt", "dist"):                       # drift OFF -> measures the rollout's own spread
            vg[c][1] = 0.0
        rng = np.random.default_rng(seed)
        gen = [self.sample(r, rng) for r in recs]
        for c, key in (("dt", "dt_s"), ("dist", "distance_px")):
            vs = []
            for g in gen:
                lx = np.log(np.clip(np.asarray(g[key], float), 1e-6, None))
                m = [np.nanmedian(lx[a:a + W]) for a in range(0, max(1, len(lx) - W + 1), W)
                     if len(lx[a:a + W]) == W]
                if len(m) >= 3:
                    m = np.asarray(m, float); vs.append(float(np.var(m - m.mean())))
            base = float(np.mean(vs)) if vs else 0.0
            tgt = float(vg["tgt_var"][c])
            scale = float(np.sqrt(max(tgt - base, 0.0) / tgt)) if tgt > 0 else 0.0
            vg[c][1] = saved[c] * scale
            vg[c][3] = vg[c][0] ** 2 + vg[c][1] ** 2     # var drives the exp de-bias at generation
            vg.setdefault("drift_calib", {})[c] = dict(target=tgt, baseline=base, scale=scale)

    def sample(self, rec, rng):
        return generate_mix_kin_loop(self._model, rec, self._cfg_feat, rng,
                                     closed_loop=self.closed_loop, guardrail=self.guardrail,
                                     eps=self.eps, stim_loop=self.stim_loop)
