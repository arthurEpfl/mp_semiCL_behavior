"""
Mixture Density Network (MDN / MLP) for zebrafish bout reorientation angles.

WHAT
----
A feed-forward MLP conditions on an autoregressive history window of previous
reorientation angles (dtheta) plus a stimulus feature, and outputs a 3-component
mixture over the SAME emission family as scripts/fit_hmm.py:

    F (forward) = Normal(0, sigma)
    L (left)    = Gamma(-dtheta; alpha_L, theta_L)   on dtheta < 0
    R (right)   = Gamma(+dtheta; alpha_R, theta_R)   on dtheta > 0
    p(dtheta) = wF*N + wL*Gamma(-dtheta) + wR*Gamma(+dtheta),   sum(w) = 1

Two variants (one `MDN` class, `mode`-parameterised):
  * 'basic' : MLP -> 3 mixture-weight logits; sigma/alpha/theta are GLOBAL
              learnable scalars, gamma SYMMETRIC (shared L/R). Homoscedastic.
  * 'full'  : MLP -> 8 per-bout outputs = 3 weight-logits + [sigma_F, alpha_L,
              theta_L, alpha_R, theta_R] (F mean fixed at 0; gammas ASYMMETRIC).
              Heteroscedastic.

This replaces the HMM's latent-state prior over {F,L,R} with a direct feedforward
function of (recent dtheta history + stimulus). It exposes the same per-bout
one-step predictive density `log p(dtheta_t | features_t)` as the HMM, so it plugs
straight into the evaluation pipeline (evaluate.cross_validate_likelihood) and is
ranked alongside `spontaneous` / `simple_stim` / `marginal_density` in Metric A.

dtheta is in DEGREES and the density is over degrees (matching fit_hmm) -- no
radian conversion anywhere.

INPUT  : data/bouts_GsizeMotionSingle.csv (via evaluate.build_records).
OUTPUT : a torch checkpoint (.pt) = {state_dict, config}; optional generated CSV.

Usage
-----
    uv run scripts/fit_MLP.py --bouts data/bouts_GsizeMotionSingle.csv \
        --output data/mlp_basic.pt --mode basic --K 5 --stim-mode simple \
        --hidden 64 --epochs 60 --lr 1e-3
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import fit_hmm as H          # emission constants + numpy reference emissions
import evaluate as E         # GenerativeModel, build_records, make_context, symbol_logmass

# ----------------------------- constants ------------------------------------ #
DTHETA0 = H.DTHETA0_DEG       # 10.0 deg forward/turn threshold
LOG_FLOOR = H.LOG_FLOOR       # -700.0
ALPHA_MIN = H.ALPHA_MIN       # 1.0  (Gamma shape floor, paper constraint)
EPS = H.EPS                   # 1e-12
LOG2PI = math.log(2 * math.pi)


# --------------------------- device / seeding ------------------------------- #
def _get_device():
    """CPU/CUDA/MPS pick (mirrors src/model_NN._get_device)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _inv_softplus(y):
    """raw such that softplus(raw) == y (so params start at data moments)."""
    return float(np.log(np.expm1(max(y, 1e-4))))


# =========================================================================== #
#  Feature builder -- the SINGLE source of truth (reused by fit/score/sample)
# =========================================================================== #
def _history_block(dtheta, K, dth_scale):
    """(n, K) AR block: column k-1 = dtheta[t-k], zero-padded for t < k."""
    n = len(dtheta)
    Hh = np.zeros((n, K))
    for k in range(1, K + 1):
        Hh[k:, k - 1] = dtheta[:-k]
    return Hh / dth_scale


def _symbol_block(symbol, K):
    """(n, 3K) one-hot of the previous K symbols (zeros for missing history)."""
    n = len(symbol)
    S = np.zeros((n, 3 * K))
    for k in range(1, K + 1):
        for t in range(k, n):
            S[t, 3 * (k - 1) + int(symbol[t - k])] = 1.0
    return S


def _stim_block(rec, stim_mode, rich_scales):
    """(n, s_dim) stimulus features from FishRecord fields only (teacher-forceable)."""
    n = rec.n
    if stim_mode == "none":
        return np.zeros((n, 0))
    if stim_mode == "simple":
        return rec.b.reshape(n, 1).astype(float)      # sin(phi) on moving bouts, else 0
    if stim_mode == "rich":
        if rec.u_raw is None:
            raise RuntimeError("stim_mode='rich' needs rec.u_raw "
                               "(build_records with abs_dict & roi).")
        scales = rich_scales if rich_scales is not None else 1.0
        return (np.asarray(rec.u_raw, float) / scales).astype(float)   # (n,5) standardised
    raise ValueError(f"unknown stim_mode {stim_mode!r}")


def build_feature_matrix(rec, cfg_feat):
    """(n, D) input matrix for one record from its OWN dtheta history + stimulus.

    cfg_feat = dict(K, stim_mode, include_symbols, dth_scale, rich_scales).
    """
    blocks = [_history_block(rec.dtheta, cfg_feat["K"], cfg_feat["dth_scale"])]
    if cfg_feat["include_symbols"]:
        blocks.append(_symbol_block(rec.symbol, cfg_feat["K"]))
    blocks.append(_stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"]))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_feature_sequences(records, cfg_feat):
    """Stack (X (N,D), y (N,)) over records. y = dtheta target (deg)."""
    Xs, ys = [], []
    for r in records:
        Xs.append(build_feature_matrix(r, cfg_feat))
        ys.append(r.dtheta.astype(np.float32))
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


# =========================================================================== #
#  torch MDN module (basic: 3 outputs + global scalars; full: 8 per-bout)
# =========================================================================== #
class MDN(nn.Module):
    def __init__(self, n_features, hidden, mode="basic", init_emis=None):
        super().__init__()
        self.mode = mode
        self.body = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        if mode == "basic":
            self.head = nn.Linear(hidden, 3)                 # weight logits only
            s0, a0, t0 = init_emis or (30.0, 1.5, 25.0)
            self.raw_sigma = nn.Parameter(torch.tensor(_inv_softplus(s0)))
            self.raw_alpha = nn.Parameter(torch.tensor(_inv_softplus(max(a0 - ALPHA_MIN, 1e-3))))
            self.raw_theta = nn.Parameter(torch.tensor(_inv_softplus(t0)))
        elif mode == "full":
            self.head = nn.Linear(hidden, 8)                 # 3 logits + sigmaF,aL,tL,aR,tR
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def forward(self, x):
        h = self.body(x)
        if self.mode == "basic":
            logw = F.log_softmax(self.head(h), dim=1)        # (n,3)
            sig = F.softplus(self.raw_sigma) + EPS           # scalar tensors (0-dim)
            aL = F.softplus(self.raw_alpha) + ALPHA_MIN
            tL = F.softplus(self.raw_theta) + EPS
            return logw, sig, aL, tL, aL, tL                 # symmetric: L == R
        out = self.head(h)
        logw = F.log_softmax(out[:, :3], dim=1)
        sp = F.softplus
        return (logw,
                sp(out[:, 3]) + EPS,                         # sigma_F  (n,)
                sp(out[:, 4]) + ALPHA_MIN, sp(out[:, 5]) + EPS,   # alpha_L, theta_L
                sp(out[:, 6]) + ALPHA_MIN, sp(out[:, 7]) + EPS)   # alpha_R, theta_R


# --------------------------- mixture log-density ---------------------------- #
def _gamma_logpdf_torch(x, alpha, theta):
    """Differentiable log Gamma(x; shape=alpha, scale=theta); LOG_FLOOR where x<=0.

    alpha/theta may be scalar or (n,) tensors and broadcast against x (n,).
    """
    pos = x > 0
    x_safe = torch.where(pos, x, torch.ones_like(x))         # avoid log(0)/neg -> NaN grad
    logp = ((alpha - 1.0) * torch.log(x_safe) - x_safe / theta
            - alpha * torch.log(theta) - torch.lgamma(alpha))
    return torch.where(pos, logp, torch.full_like(x, LOG_FLOOR))


def mixture_logdens(dtheta, params):
    """Per-bout log predictive density (nats over DEGREES). params = MDN.forward output."""
    logw, sigmaF, aL, tL, aR, tR = params
    logNF = -0.5 * (LOG2PI + 2.0 * torch.log(sigmaF)) - dtheta ** 2 / (2.0 * sigmaF ** 2)
    logGL = _gamma_logpdf_torch(-dtheta, aL, tL)             # Left: support dtheta < 0
    logGR = _gamma_logpdf_torch(+dtheta, aR, tR)             # Right: support dtheta > 0
    comp = torch.stack([logw[:, 0] + logNF,
                        logw[:, 1] + logGL,
                        logw[:, 2] + logGR], dim=1)           # (n,3)
    return torch.logsumexp(comp, dim=1)                      # (n,)


def mdn_nll(model, X, y):
    """Mean negative log-likelihood over a batch (the training objective)."""
    return -mixture_logdens(y, model(X)).mean()


# ------------------------------ training loop ------------------------------- #
def fit_mlp(X, y, mode, hidden, *, epochs=60, lr=1e-3, batch_size=256, patience=8,
            val_frac=0.15, seed=0, init_emis=None, device=None, verbose=False):
    """Train an MDN by NLL with a held-out validation split + early stopping.

    Returns (model_on_cpu, history dict with 'train'/'val' loss lists).
    """
    device = device or _get_device()
    _seed_everything(seed)
    N = len(X)
    idx = np.random.permutation(N)
    n_val = max(1, int(val_frac * N))
    vi, ti = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(X[ti], device=device); yt = torch.tensor(y[ti], device=device)
    Xv = torch.tensor(X[vi], device=device); yv = torch.tensor(y[vi], device=device)

    model = MDN(X.shape[1], hidden, mode=mode, init_emis=init_emis).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val, best_state, no_imp = float("inf"), None, 0
    hist = {"train": [], "val": []}
    nb = int(np.ceil(len(ti) / batch_size))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ti), device=device)
        tot = 0.0
        for b in range(nb):
            bi = perm[b * batch_size:(b + 1) * batch_size]
            loss = mdn_nll(model, Xt[bi], yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        model.eval()
        with torch.no_grad():
            vloss = float(mdn_nll(model, Xv, yv).item())
        hist["train"].append(tot / max(1, nb)); hist["val"].append(vloss)
        if verbose:
            print(f"  epoch {ep:3d}  train {hist['train'][-1]:.4f}  val {vloss:.4f}")
        if vloss < best_val - 1e-4:
            best_val, no_imp = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to("cpu"), hist


# =========================================================================== #
#  Scoring helpers (log predictive density + 3-symbol PMF)
# =========================================================================== #
@torch.no_grad()
def predict_params(model, X):
    """Numpy per-bout mixture params (w (n,3), sigmaF, aL, tL, aR, tR each (n,))."""
    model.eval()
    n = X.shape[0]
    logw, sigmaF, aL, tL, aR, tR = model(torch.tensor(X))

    def col(z):
        return np.full(n, float(z)) if z.dim() == 0 else z.numpy().astype(float)

    return (np.exp(logw.numpy()), col(sigmaF), col(aL), col(tL), col(aR), col(tR))


def symbol_logq_full(w, sigmaF, aL, tL, aR, tR, dtheta0=DTHETA0, eps=1e-300):
    """(n,3) predictive log-PMF over {F,L,R} for a PER-BOUT-parameter mixture.

    Integrates each component over the +/-dtheta0 bins:
      Forward Normal(0,sigmaF): masses nL/nF/nR via the erf CDF.
      Left  Gamma(alpha_L,theta_L) on dtheta<0: GcL into F (0<mag<dtheta0), 1-GcL into L.
      Right Gamma(alpha_R,theta_R) on dtheta>0: GcR into F, 1-GcR into R.
    (This is the asymmetric per-bout generalisation of evaluate.symbol_logmass.)
    """
    from scipy.special import erf, gammainc
    sqrt2 = math.sqrt(2.0)
    Phi_hi = 0.5 * (1.0 + erf((dtheta0 / sigmaF) / sqrt2))
    Phi_lo = 0.5 * (1.0 + erf((-dtheta0 / sigmaF) / sqrt2))
    nF = Phi_hi - Phi_lo
    nL = Phi_lo
    nR = 1.0 - Phi_hi
    GcL = gammainc(aL, dtheta0 / tL)        # Left  magnitude CDF at dtheta0 (per bout)
    GcR = gammainc(aR, dtheta0 / tR)        # Right magnitude CDF at dtheta0 (per bout)
    qF = w[:, 0] * nF + w[:, 1] * GcL + w[:, 2] * GcR
    qL = w[:, 0] * nL + w[:, 1] * (1.0 - GcL)
    qR = w[:, 0] * nR + w[:, 2] * (1.0 - GcR)
    q = np.maximum(np.stack([qF, qL, qR], axis=1), eps)
    return np.log(q / q.sum(axis=1, keepdims=True))


# =========================================================================== #
#  Autoregressive generation
# =========================================================================== #
def _numpy_predictor(model):
    """Return fn(x_vec) -> (w (3,), sigmaF, aL, tL, aR, tR) doing the MDN forward in
    NUMPY. The autoregressive sampler calls the net once per bout; a numpy forward on
    this tiny net avoids torch per-call overhead (~50x faster) and is numerically
    identical to MDN.forward."""
    sd = model.state_dict()
    W1 = sd["body.0.weight"].numpy(); b1 = sd["body.0.bias"].numpy()
    W2 = sd["head.weight"].numpy();   b2 = sd["head.bias"].numpy()

    def softplus(z):
        return np.logaddexp(0.0, z)

    if model.mode == "basic":
        sig = float(softplus(sd["raw_sigma"].numpy())) + EPS
        aL = float(softplus(sd["raw_alpha"].numpy())) + ALPHA_MIN
        tL = float(softplus(sd["raw_theta"].numpy())) + EPS

        def fwd(x):
            h = np.maximum(W1 @ x + b1, 0.0)
            o = W2 @ h + b2
            w = np.exp(o - o.max()); w /= w.sum()
            return w, sig, aL, tL, aL, tL
    else:
        def fwd(x):
            h = np.maximum(W1 @ x + b1, 0.0)
            o = W2 @ h + b2
            lw = o[:3]; w = np.exp(lw - lw.max()); w /= w.sum()
            return (w, float(softplus(o[3])) + EPS,
                    float(softplus(o[4])) + ALPHA_MIN, float(softplus(o[5])) + EPS,
                    float(softplus(o[6])) + ALPHA_MIN, float(softplus(o[7])) + EPS)
    return fwd


def generate_mlp(model, rec, cfg_feat, dt_pool, dist_pool, rng):
    """Free-run rollout: feed generated dtheta history back; stimulus teacher-forced
    from `rec`. Returns dict(dtheta, symbol, dt_s, distance_px) of length rec.n."""
    n = rec.n
    K = cfg_feat["K"]
    dth_scale = cfg_feat["dth_scale"]
    stim = _stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"])  # teacher-forced
    fwd = _numpy_predictor(model)
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
                    d = dth[t - k]
                    s = 0 if abs(d) < DTHETA0 else (1 if d < 0 else 2)
                    sb[3 * (k - 1) + s] = 1.0
            row.append(sb)
        row.append(stim[t])
        x = np.concatenate(row).astype(np.float32)
        w, sF, aL, tL, aR, tR = fwd(x)
        c = rng.choice(3, p=w)
        if c == 0:
            dth[t] = rng.normal(0.0, sF)
        elif c == 1:
            dth[t] = -rng.gamma(aL, tL)
        else:
            dth[t] = +rng.gamma(aR, tR)
    return dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                dt_s=rng.choice(dt_pool, size=n), distance_px=rng.choice(dist_pool, size=n))


# =========================================================================== #
#  Save / load
# =========================================================================== #
def save_mlp(path, model, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_mlp(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = MDN(cfg["n_features"], cfg["hidden"], mode=cfg["mode"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


# =========================================================================== #
#  Evaluation-pipeline adapter (evaluate.GenerativeModel)
# =========================================================================== #
class MLPDensity(E.GenerativeModel):
    """Mixture-density MLP as a drop-in generative model for Metric A / B."""
    has_density = True
    has_symbol = True
    can_sample = True

    def __init__(self, mode="basic", K=5, stim_mode="simple", include_symbols=False,
                 hidden=64, epochs=60, lr=1e-3, patience=8, batch_size=256, seed=0):
        self.mode = mode
        self.name = f"mlp_{mode}"
        self.K = K
        self.stim_mode = stim_mode
        self.include_symbols = include_symbols
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.requires_absdict = (stim_mode == "rich")

    def fit(self, train_records, ctx):
        c = ctx.cfg or {}
        self.K = int(c.get("mlp_K", self.K))
        self.stim_mode = c.get("mlp_stim_mode", self.stim_mode)
        self.include_symbols = bool(c.get("mlp_include_symbols", self.include_symbols))
        self.hidden = int(c.get("mlp_hidden", self.hidden))
        self.epochs = int(c.get("mlp_epochs", self.epochs))
        self.lr = float(c.get("mlp_lr", self.lr))
        self.requires_absdict = (self.stim_mode == "rich")
        self._dt_pool, self._dist_pool = ctx.dt_pool, ctx.dist_pool
        self._rich_scales = ctx.rich_scales

        # train-fold-only scale + emission init (leak-free)
        alld = np.concatenate([r.dtheta for r in train_records])
        self._dth_scale = float(max(np.std(alld), 1.0))
        fwd = alld[np.abs(alld) < DTHETA0]
        turn = np.abs(alld[np.abs(alld) >= DTHETA0])
        s0 = float(max(np.std(fwd) if len(fwd) > 5 else 8.0, 0.5))
        m_, v_ = (turn.mean(), turn.var()) if len(turn) > 5 else (40.0, 400.0)
        a0 = float(max(m_ * m_ / max(v_, 1e-12), ALPHA_MIN))
        t0 = float(max(v_ / max(m_, 1e-12), 1.0))

        self._cfg_feat = dict(K=self.K, stim_mode=self.stim_mode,
                              include_symbols=self.include_symbols,
                              dth_scale=self._dth_scale, rich_scales=self._rich_scales)
        X, y = build_feature_sequences(train_records, self._cfg_feat)
        self._model, self.history = fit_mlp(
            X, y, self.mode, self.hidden, epochs=self.epochs, lr=self.lr,
            patience=self.patience, batch_size=self.batch_size, seed=self.seed,
            init_emis=(s0, a0, t0))
        self.config = dict(mode=self.mode, hidden=self.hidden, n_features=X.shape[1],
                           K=self.K, stim_mode=self.stim_mode,
                           include_symbols=self.include_symbols, dth_scale=self._dth_scale,
                           rich_scales=(None if self._rich_scales is None
                                        else np.asarray(self._rich_scales).tolist()))
        return self

    def score(self, rec):
        X = build_feature_matrix(rec, self._cfg_feat)
        w, sigmaF, aL, tL, aR, tR = predict_params(self._model, X)
        with torch.no_grad():
            logdens = mixture_logdens(
                torch.tensor(rec.dtheta.astype(np.float32)),
                self._model(torch.tensor(X))).numpy()
        if self.mode == "basic":
            slq = E.symbol_logmass(w, sigmaF[0], aL[0], tL[0], mu=0.0)
        else:
            slq = symbol_logq_full(w, sigmaF, aL, tL, aR, tR)
        return dict(logdens=logdens, symbol_logq=slq)

    def sample(self, rec, rng):
        return generate_mlp(self._model, rec, self._cfg_feat,
                            self._dt_pool, self._dist_pool, rng)


# =========================================================================== #
#  CLI
# =========================================================================== #
def main():
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bouts", type=Path, default=Path("data/bouts_GsizeMotionSingle.csv"))
    p.add_argument("--output", type=Path, default=Path("data/mlp_basic.pt"))
    p.add_argument("--mode", choices=["basic", "full"], default="basic")
    p.add_argument("--K", type=int, default=5, help="AR history window (bouts)")
    p.add_argument("--stim-mode", choices=["none", "simple", "rich"], default="simple")
    p.add_argument("--include-symbols", action="store_true")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-len", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-output", type=Path, default=None)
    p.add_argument("--n-gen", type=int, default=2000)
    args = p.parse_args()

    df = pd.read_csv(args.bouts)
    records = E.build_records(df, min_len=args.min_len)   # rich mode needs abs_dict/roi
    recs = list(records.values())
    print(f"loaded {len(recs)} fish ({sum(r.n for r in recs)} bouts, min_len={args.min_len})")

    ctx = E.make_context(recs, cfg=dict(mlp_epochs=args.epochs, mlp_hidden=args.hidden,
                                        mlp_K=args.K, mlp_stim_mode=args.stim_mode,
                                        mlp_lr=args.lr,
                                        mlp_include_symbols=args.include_symbols))
    m = MLPDensity(mode=args.mode, K=args.K, stim_mode=args.stim_mode,
                   include_symbols=args.include_symbols, hidden=args.hidden,
                   epochs=args.epochs, lr=args.lr, seed=args.seed)
    m.fit(recs, ctx)
    print(f"trained {m.name}: features D={m.config['n_features']}, "
          f"final val NLL={m.history['val'][-1]:.4f} ({len(m.history['val'])} epochs)")

    # in-sample mean bits/bout as a quick sanity readout
    ld = np.concatenate([m.score(r)["logdens"] for r in recs])
    print(f"in-sample: {np.mean(ld):.4f} nats/bout | {-np.mean(ld) / np.log(2):.4f} bits/bout")

    save_mlp(args.output, m._model, m.config)
    print("saved ->", args.output)

    if args.gen_output is not None:
        rng = np.random.default_rng(args.seed)
        gen = m.sample(recs[0], rng)
        pd.DataFrame({"bout_angle": gen["dtheta"][:args.n_gen]}).to_csv(args.gen_output, index=False)
        print(f"generated {min(args.n_gen, len(gen['dtheta']))} bouts -> {args.gen_output}")


if __name__ == "__main__":
    main()
