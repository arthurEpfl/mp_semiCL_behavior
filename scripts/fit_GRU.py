"""GRU mixture-density model (recurrent MDN) -- a drop-in generative model for the
same Metric-A/B/C2ST pipeline as ``fit_MLP``.

Motivation: the C2ST classifier keys on the TURN SEQUENCE's temporal structure
(turn-sign autocorrelation, run-lengths, transitions). A fixed-K feedforward MDN
cannot represent persistence beyond K bouts. This replaces the K-window body with a
**GRU that carries hidden state across bouts** (unbounded memory), keeping the exact
Normal+2Gamma mixture emission of ``fit_MLP.MDN`` (Graves-2013 LSTM-MDN recipe).

Per-bout input is small (the GRU holds the history): ``[prev_dtheta/dth_scale, stim]``
where stim is the ``simple`` scalar ``rec.b`` or the ``rich_csv`` block ``rec.stim_csv``.
Emission math (mixture_logdens, symbol PMFs, emission init) is reused from ``fit_MLP``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate as E
import fit_MLP as M
import fit_MLP_border as MB   # for the teacher-forced wall covariate (use_border=True)


# =========================================================================== #
#  Per-bout input:  [prev_dtheta/scale, stim block]   (GRU carries the history)
# =========================================================================== #
def build_inputs(rec, cfg_feat):
    """(n, 1+stim_w) autoregressive input: previous dtheta (teacher-forced, scaled)
    + the teacher-forceable stimulus block (reused from fit_MLP)."""
    n = rec.n
    prev = np.zeros(n, dtype=np.float32)
    prev[1:] = rec.dtheta[:-1]
    blocks = [(prev / cfg_feat["dth_scale"]).reshape(n, 1),
              M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                            phi_ref=cfg_feat.get("phi_ref", "ori"))]
    if cfg_feat.get("use_border"):
        blocks.append(MB._border_block(rec, cfg_feat))   # teacher-forced [dist_to_border, sin/cos psi]
    return np.concatenate(blocks, axis=1).astype(np.float32)


# =========================================================================== #
#  Recurrent MDN module (GRU body + the fit_MLP MDN emission head)
# =========================================================================== #
class GRUMDN(nn.Module):
    def __init__(self, input_dim, hidden, mode="full", num_layers=1, init_emis=None):
        super().__init__()
        self.mode = mode
        self.gru = nn.GRU(input_dim, hidden, num_layers, batch_first=True)
        if mode == "basic":
            self.head = nn.Linear(hidden, 3)                 # weight logits only
            s0, a0, t0 = init_emis or (30.0, 1.5, 25.0)
            self.raw_sigma = nn.Parameter(torch.tensor(M._inv_softplus(s0)))
            self.raw_alpha = nn.Parameter(torch.tensor(M._inv_softplus(max(a0 - M.ALPHA_MIN, 1e-3))))
            self.raw_theta = nn.Parameter(torch.tensor(M._inv_softplus(t0)))
        elif mode == "full":
            self.head = nn.Linear(hidden, 8)                 # 3 logits + sigmaF,aL,tL,aR,tR
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def _emit(self, out):
        """(N,3 or 8) head output -> the 6-tuple, identical to fit_MLP.MDN.forward."""
        if self.mode == "basic":
            logw = F.log_softmax(out, dim=1)
            sig = F.softplus(self.raw_sigma) + M.EPS
            aL = F.softplus(self.raw_alpha) + M.ALPHA_MIN
            tL = F.softplus(self.raw_theta) + M.EPS
            return logw, sig, aL, tL, aL, tL
        logw = F.log_softmax(out[:, :3], dim=1)
        sp = F.softplus
        return (logw, sp(out[:, 3]) + M.EPS,
                sp(out[:, 4]) + M.ALPHA_MIN, sp(out[:, 5]) + M.EPS,
                sp(out[:, 6]) + M.ALPHA_MIN, sp(out[:, 7]) + M.EPS)

    def forward(self, x, h0=None):
        """x: (B,T,input_dim) -> (6-tuple over B*T timesteps, final hidden state)."""
        z, hn = self.gru(x, h0)                 # z: (B,T,hidden)
        B, T, Hd = z.shape
        return self._emit(self.head(z.reshape(B * T, Hd))), hn


def _params_to_numpy(params, n):
    """torch 6-tuple -> numpy (w (n,3), sigmaF,aL,tL,aR,tR each (n,)); mirrors predict_params."""
    logw, sF, aL, tL, aR, tR = params

    def col(z):
        z = z.detach()
        return np.full(n, float(z)) if z.dim() == 0 else z.numpy().astype(float)

    return np.exp(logw.detach().numpy()), col(sF), col(aL), col(tL), col(aR), col(tR)


# =========================================================================== #
#  Training -- sequence-based, teacher-forced, truncated BPTT
# =========================================================================== #
def fit_gru(records, cfg_feat, mode, hidden, num_layers, epochs, lr, tbptt,
            init_emis, seed, patience=8, val_frac=0.15):
    """Train the GRU-MDN by one-step NLL over per-fish sequences. TBPTT bounds memory;
    a few held-out fish give a teacher-forced val for early stopping. Returns (model, hist)."""
    M._seed_everything(seed)
    device = M._get_device()                     # CUDA on Colab, MPS on mac, else CPU
    input_dim = build_inputs(records[0], cfg_feat).shape[1]
    model = GRUMDN(input_dim, hidden, mode, num_layers, init_emis).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    seqs = [(build_inputs(r, cfg_feat), r.dtheta.astype(np.float32)) for r in records]
    nval = max(1, int(val_frac * len(seqs))) if len(seqs) > 2 else 0
    tr, va = (seqs[:-nval], seqs[-nval:]) if nval else (seqs, seqs[-1:])

    best_val, best_state, no_imp = float("inf"), None, 0
    hist = {"train": [], "val": []}
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for i in rng.permutation(len(tr)):
            X, y = tr[i]
            h = None
            for s in range(0, len(y), tbptt):
                e = min(s + tbptt, len(y))
                params, h = model(torch.tensor(X[s:e], device=device).unsqueeze(0), h)
                h = h.detach()
                loss = -M.mixture_logdens(torch.tensor(y[s:e], device=device), params).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.item()); nb += 1
        model.eval(); vtot = 0.0; vn = 0
        with torch.no_grad():
            for X, y in va:
                params, _ = model(torch.tensor(X, device=device).unsqueeze(0))
                vtot += float(-M.mixture_logdens(torch.tensor(y, device=device), params).mean().item()); vn += 1
        vloss = vtot / max(vn, 1)
        hist["train"].append(tot / max(nb, 1)); hist["val"].append(vloss)
        if vloss < best_val - 1e-4:
            best_val, no_imp = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model = model.to("cpu")                       # store/sample on CPU (numpy sampler + score)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


# =========================================================================== #
#  Fast numpy GRU-cell sampler (avoids torch per-step overhead in rollouts)
# =========================================================================== #
def _numpy_gru(model):
    """Return (step, emit, H): a numpy GRU-cell step + emission, numerically identical
    to the torch module (single layer). Mirrors fit_MLP._numpy_predictor for the MLP."""
    sd = model.state_dict()
    Wih = sd["gru.weight_ih_l0"].numpy(); Whh = sd["gru.weight_hh_l0"].numpy()
    bih = sd["gru.bias_ih_l0"].numpy();   bhh = sd["gru.bias_hh_l0"].numpy()
    Wo = sd["head.weight"].numpy();       bo = sd["head.bias"].numpy()
    H = Whh.shape[1]

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    def softplus(z):
        return np.logaddexp(0.0, z)

    def step(x, h):                                # x:(in,) h:(H,) -> (out, h')
        gi = Wih @ x + bih; gh = Whh @ h + bhh
        r = sig(gi[:H] + gh[:H]); zg = sig(gi[H:2 * H] + gh[H:2 * H])
        ncell = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        h2 = (1.0 - zg) * ncell + zg * h
        return Wo @ h2 + bo, h2

    if model.mode == "basic":
        sig0 = float(softplus(sd["raw_sigma"].numpy())) + M.EPS
        aL0 = float(softplus(sd["raw_alpha"].numpy())) + M.ALPHA_MIN
        tL0 = float(softplus(sd["raw_theta"].numpy())) + M.EPS

        def emit(o):
            w = np.exp(o - o.max()); w /= w.sum()
            return w, sig0, aL0, tL0, aL0, tL0
    else:
        def emit(o):
            lw = o[:3]; w = np.exp(lw - lw.max()); w /= w.sum()
            return (w, float(softplus(o[3])) + M.EPS,
                    float(softplus(o[4])) + M.ALPHA_MIN, float(softplus(o[5])) + M.EPS,
                    float(softplus(o[6])) + M.ALPHA_MIN, float(softplus(o[7])) + M.EPS)
    return step, emit, H


def generate_gru(model, rec, cfg_feat, dt_pool, dist_pool, rng):
    """Autoregressive rollout: generated dtheta fed back, GRU hidden carried; stimulus
    teacher-forced. dtheta clipped to +/-180 (wrap-limited). Returns the usual dict."""
    n = rec.n
    dth_scale = cfg_feat["dth_scale"]
    stim = M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                         phi_ref=cfg_feat.get("phi_ref", "ori"))   # teacher-forced
    border = MB._border_block(rec, cfg_feat) if cfg_feat.get("use_border") else np.zeros((n, 0))
    step, emit, H = _numpy_gru(model)
    h = np.zeros(H); dth = np.zeros(n); prev = 0.0
    for t in range(n):
        x = np.concatenate([[prev / dth_scale], stim[t], border[t]]).astype(float)   # SAME order as build_inputs
        o, h = step(x, h)
        w, sF, aL, tL, aR, tR = emit(o)
        c = rng.choice(3, p=w)
        if c == 0:
            dth[t] = rng.normal(0.0, sF)
        elif c == 1:
            dth[t] = -rng.gamma(aL, tL)
        else:
            dth[t] = +rng.gamma(aR, tR)
        dth[t] = float(np.clip(dth[t], -180.0, 180.0))
        prev = dth[t]
    return dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                dt_s=rng.choice(dt_pool, size=n), distance_px=rng.choice(dist_pool, size=n))


# =========================================================================== #
#  Evaluation-pipeline adapter
# =========================================================================== #
class GRUDensity(E.GenerativeModel):
    """Recurrent MDN (GRU) as a drop-in generative model for Metric A / B / C2ST."""
    has_density = True
    has_symbol = True
    can_sample = True

    def __init__(self, mode="full", stim_mode="simple", hidden=64, num_layers=1,
                 epochs=40, lr=1e-3, tbptt=400, patience=8, seed=0, use_border=False):
        self.mode = mode
        self.name = f"gru_border_{mode}" if use_border else f"gru_{mode}"
        self.use_border = use_border
        self.stim_mode = stim_mode
        self.hidden = hidden
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.tbptt = tbptt
        self.patience = patience
        self.seed = seed
        self.requires_absdict = (stim_mode == "rich")   # 'rich_csv' is CSV-only, not gated

    def fit(self, train_records, ctx):
        c = ctx.cfg or {}
        self.hidden = int(c.get("gru_hidden", self.hidden))
        self.num_layers = int(c.get("gru_layers", self.num_layers))
        self.epochs = int(c.get("gru_epochs", self.epochs))
        self.lr = float(c.get("gru_lr", self.lr))
        self.tbptt = int(c.get("gru_tbptt", self.tbptt))
        self.stim_mode = c.get("gru_stim_mode", self.stim_mode)
        self._dt_pool, self._dist_pool = ctx.dt_pool, ctx.dist_pool
        self._rich_scales = ctx.rich_scales

        alld = np.concatenate([r.dtheta for r in train_records])
        self._dth_scale = float(max(np.std(alld), 1.0))
        fwd = alld[np.abs(alld) < M.DTHETA0]
        turn = np.abs(alld[np.abs(alld) >= M.DTHETA0])
        s0 = float(max(np.std(fwd) if len(fwd) > 5 else 8.0, 0.5))
        m_, v_ = (turn.mean(), turn.var()) if len(turn) > 5 else (40.0, 400.0)
        a0 = float(max(m_ * m_ / max(v_, 1e-12), M.ALPHA_MIN))
        t0 = float(max(v_ / max(m_, 1e-12), 1.0))

        self._cfg_feat = dict(stim_mode=self.stim_mode, dth_scale=self._dth_scale,
                              rich_scales=self._rich_scales, use_border=self.use_border)
        input_dim = build_inputs(train_records[0], self._cfg_feat).shape[1]
        self._model, self.history = fit_gru(
            train_records, self._cfg_feat, self.mode, self.hidden, self.num_layers,
            self.epochs, self.lr, self.tbptt, (s0, a0, t0), self.seed, self.patience)
        self.config = dict(mode=self.mode, hidden=self.hidden, num_layers=self.num_layers,
                           input_dim=input_dim, stim_mode=self.stim_mode, dth_scale=self._dth_scale)
        return self

    def score(self, rec):
        X = build_inputs(rec, self._cfg_feat)
        self._model.eval()
        with torch.no_grad():
            params, _ = self._model(torch.tensor(X).unsqueeze(0))
            logdens = M.mixture_logdens(torch.tensor(rec.dtheta.astype(np.float32)), params).numpy()
        w, sF, aL, tL, aR, tR = _params_to_numpy(params, rec.n)
        if self.mode == "basic":
            slq = E.symbol_logmass(w, sF[0], aL[0], tL[0], mu=0.0)
        else:
            slq = M.symbol_logq_full(w, sF, aL, tL, aR, tR)
        return dict(logdens=logdens, symbol_logq=slq)

    def sample(self, rec, rng):
        return generate_gru(self._model, rec, self._cfg_feat,
                            self._dt_pool, self._dist_pool, rng)


# =========================================================================== #
#  Smoke test:  uv run scripts/fit_GRU.py
# =========================================================================== #
if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    df = E.load_bouts(str(root / "data" / "bouts_GsizeMotionSingle.csv"))
    recs = list(E.build_records(df, min_len=10).values())
    ctx = E.make_context(recs, cfg=dict(gru_epochs=3))
    g = GRUDensity(mode="full", stim_mode="simple", hidden=32, tbptt=300).fit(recs, ctx)
    r0 = recs[0]
    sc = g.score(r0); gen = g.sample(r0, np.random.default_rng(0))
    print(f"gru name={g.name} input_dim={g.config['input_dim']} "
          f"score_finite={np.isfinite(sc['logdens']).all()} "
          f"mean_nats={np.nanmean(sc['logdens']):.3f} sample_finite={np.isfinite(gen['dtheta']).all()} "
          f"|dtheta|max={np.abs(gen['dtheta']).max():.1f}")
