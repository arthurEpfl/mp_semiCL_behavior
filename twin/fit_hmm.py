"""
Fit a 3-state Markov Chain and a 3-state Hidden Markov Model to the segmented
bouts (data/bouts_GsizeMotionSingle.csv), in the same style as Dommanget-Kott
et al. 2026 (PLoS Comput Biol; "Linking brain and behavior states ... using
hidden Markov models"), and generate synthetic swim trajectories from the HMM.

WHAT THE PAPER DOES (reproduced here, in Python)
------------------------------------------------
* Markov Chain (MC): threshold the reorientation angle dtheta at +/- dtheta0
  (=10 deg) into Forward / Left / Right, then estimate the 3x3 transition matrix
  by counting transitions (Eq 2). This mirrors utils/MarkovChains.py of the paper
  repo (threshold_classifier + bout_transitions + steady_state).

* Hidden Markov Model (HMM), model type `ZebrafishHMM_G3_Sym`:
    - 3 hidden states: 0=Forward (F), 1=Left (L), 2=Right (R).
    - Emissions E(dtheta | state):
        F : Normal(dtheta ; 0, sigma)                       (centered forward)
        L : Gamma(-dtheta ; alpha, theta)  on dtheta < 0    (turn magnitude)
        R : Gamma(+dtheta ; alpha, theta)  on dtheta > 0
      (single shared Gamma (alpha, theta) for both turns; alpha >= 1, theta > 0.)
      This is exactly the emission set stored in the paper's fitted models
      (`sforw` scalar + `turn` = [alpha, theta]); dtheta is in DEGREES.
    - Left-right SYMMETRY is enforced on the transition matrix and initial
      distribution: P(F->L)=P(F->R), P(L->L)=P(R->R), P(L->R)=P(R->L),
      P(L->F)=P(R->F), and one shared turn emission.
    - OPTIMISATION = Baum-Welch (Expectation-Maximisation). The paper's "custom
      Julia implementation" is the generic Baum-Welch of HiddenMarkovModels.jl
      (Dalle 2024) specialised to this emission type; there is no other optimizer.
      We implement Baum-Welch directly (log-space forward-backward + a symmetric,
      tied M-step with a weighted Gamma MLE for the turn emission).
      Training: up to 500 EM iterations, stop when the log-likelihood gain < 1e-6
      (paper's Sect 4.4); transition matrix initialised at random, respecting
      left-right symmetry; a few random restarts, keep the best log-likelihood.

* Generation (paper Sect 2.7.2): sample a state sequence from the HMM, sample a
  reorientation angle from each state's emission, resample the inter-bout interval
  dt and traveled distance d from the EMPIRICAL pools (ignoring bout-type
  dependence, as the paper does), then reconstruct x,y by cumulative sums (Eq 5).

INPUT  : data/bouts_GsizeMotionSingle.csv (from scripts/segment_bouts.py).
OUTPUT : an .npz HMM model + a few generated trajectories (.csv), plus a printed
         diagnostics summary.

Usage
-----
    uv run scripts/fit_hmm.py \
        --bouts   data/bouts_GsizeMotionSingle.csv \
        --output  data/hmm_GsizeMotionSingle.npz \
        --restarts 5 --min-len 10
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import digamma, gammaln, polygamma

# --------------------------- constants / conventions ------------------------ #
STATE_NAMES = ["forward", "left", "right"]   # 0=F, 1=L (dtheta<0), 2=R (dtheta>0)
DTHETA0_DEG = 10.0        # MC threshold (paper: +/- 10 deg)
LOG_FLOOR = -700.0        # log of ~1e-304, used where an emission has no support
ALPHA_MIN = 1.0           # paper constraint on the Gamma shape (alpha >= 1)
EPS = 1e-12


# =========================================================================== #
#  Sequence loading
# =========================================================================== #
def load_sequences_from_bouts(df, angle_col="dtheta_deg", min_len=10,
                              group_cols=("fish_id", "segment_id"),
                              order_col="bout_index"):
    """Split the tidy bout dataframe into per-trajectory dtheta sequences (deg).

    One sequence per continuous trajectory = one (fish_id, segment_id) group,
    ordered by bout_index. Sequences shorter than `min_len` bouts are dropped
    (too short to constrain an HMM). Returns a list of 1-D float arrays.
    """
    seqs = []
    cols = list(group_cols)
    for _, g in df.sort_values(cols + [order_col]).groupby(cols, sort=False):
        a = g[angle_col].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if len(a) >= min_len:
            seqs.append(a)
    return seqs


# =========================================================================== #
#  Markov Chain (thresholded) -- mirrors the paper's utils/MarkovChains.py
# =========================================================================== #
def threshold_classifier(seqs, dtheta0=DTHETA0_DEG):
    """Classify each dtheta into 0=Forward, 1=Left (dtheta<-dtheta0),
    2=Right (dtheta>+dtheta0). (Paper repo convention: left = negative.)"""
    out = []
    for s in seqs:
        b = np.zeros_like(s, dtype=np.int8)
        b[s < -dtheta0] = 1
        b[s > +dtheta0] = 2
        out.append(b)
    return out


def bout_proba(bout_seqs, n_state=3):
    """A-priori probability of each state (empirical fraction)."""
    allb = np.concatenate(bout_seqs)
    h, _ = np.histogram(allb, bins=np.arange(n_state + 1))
    return h / h.sum()


def bout_transitions(bout_seqs, n_state=3):
    """Row-stochastic transition matrix by counting transitions (paper Eq 2)."""
    P = np.zeros((n_state, n_state))
    for b in bout_seqs:
        for i in range(len(b) - 1):
            P[b[i], b[i + 1]] += 1
    return P / P.sum(axis=1, keepdims=True)


def steady_state(T):
    """Stationary distribution of a row-stochastic transition matrix T."""
    n = T.shape[0]
    q = np.c_[T - np.eye(n), np.ones(n)]
    return np.linalg.solve(q @ q.T, np.ones(n))


# =========================================================================== #
#  HMM (G3, symmetric) -- emissions, Baum-Welch, Viterbi
# =========================================================================== #
def _gamma_logpdf(x, alpha, theta):
    """log Gamma(x; shape=alpha, scale=theta) for x > 0 (else LOG_FLOOR)."""
    out = np.full_like(x, LOG_FLOOR, dtype=float)
    pos = x > 0
    xp = x[pos]
    out[pos] = ((alpha - 1.0) * np.log(xp) - xp / theta
                - alpha * np.log(theta) - gammaln(alpha))
    return out


def emission_logprob(dtheta, params):
    """(T, 3) matrix of log emission probabilities for one sequence.

    params: dict(sigma=..., alpha=..., theta=...). dtheta in degrees.
      col 0 (F): Normal(0, sigma); col 1 (L): Gamma(-dtheta); col 2 (R): Gamma(+dtheta).
    """
    sigma, alpha, theta = params["sigma"], params["alpha"], params["theta"]
    T = len(dtheta)
    logB = np.empty((T, 3))
    logB[:, 0] = -0.5 * np.log(2 * np.pi * sigma * sigma) - dtheta**2 / (2 * sigma * sigma)
    logB[:, 1] = _gamma_logpdf(-dtheta, alpha, theta)   # Left  -> negative angles
    logB[:, 2] = _gamma_logpdf(+dtheta, alpha, theta)   # Right -> positive angles
    return logB


def _forward_backward(pi, T, B):
    """Scaled (Rabiner) forward-backward for one sequence.

    `B` is the (n, 3) matrix of emission *probabilities* (not logs). Using the
    per-step scaling trick keeps everything numerically stable while avoiding the
    per-timestep logsumexp calls -- the recursions are plain 3x3 matmuls, which is
    ~10-30x faster on our long (thousands-of-bouts) sequences. The transition
    responsibilities xi are summed with a single vectorised einsum.

    Returns loglik, gamma (n,3), xi_sum (3,3), gamma0 (3,).
    """
    n = B.shape[0]
    alpha = np.empty((n, 3))
    c = np.empty(n)                                  # scaling factors
    a = pi * B[0]
    s = a.sum(); s = s if s > 0 else 1e-300
    c[0] = s; alpha[0] = a / s
    for t in range(1, n):
        a = (alpha[t - 1] @ T) * B[t]
        s = a.sum(); s = s if s > 0 else 1e-300
        c[t] = s; alpha[t] = a / s
    loglik = float(np.sum(np.log(c)))

    beta = np.empty((n, 3))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (T @ (B[t + 1] * beta[t + 1])) / c[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi_sum = np.zeros((3, 3))
    if n > 1:
        bb = (B[1:] * beta[1:]) / c[1:, None]        # (n-1, 3)
        # xi_sum[i,j] = sum_t alpha[t,i] * T[i,j] * bb[t,j]
        xi_sum = np.einsum("ti,ij,tj->ij", alpha[:-1], T, bb)
    return loglik, gamma, xi_sum, gamma[0]


def _weighted_gamma_mle(Wt, Sx, Slog, alpha_min=ALPHA_MIN, newton_iter=50):
    """MLE of Gamma(alpha, theta) from weighted sufficient statistics:
    Wt=sum w, Sx=sum w*x, Slog=sum w*log(x). Enforces alpha >= alpha_min."""
    if Wt <= EPS or Sx <= EPS:
        return alpha_min, 1.0
    mean = Sx / Wt
    meanlog = Slog / Wt
    s = np.log(mean) - meanlog                      # >= 0
    s = max(s, 1e-9)
    alpha = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)   # Minka init
    for _ in range(newton_iter):
        g = np.log(alpha) - digamma(alpha) - s
        gp = 1.0 / alpha - polygamma(1, alpha)
        step = g / gp
        alpha_new = alpha - step
        if not np.isfinite(alpha_new) or alpha_new <= 0:
            break
        if abs(alpha_new - alpha) < 1e-8:
            alpha = alpha_new
            break
        alpha = alpha_new
    alpha = max(alpha, alpha_min)
    theta = mean / alpha
    return float(alpha), float(theta)


def _symmetrise_transitions(A):
    """Tie an expected-transition-count matrix A (3x3) into a left-right
    symmetric row-stochastic transition matrix (states 0=F,1=L,2=R)."""
    ff = A[0, 0]
    f_turn = A[0, 1] + A[0, 2]                       # F->L + F->R
    same = A[1, 1] + A[2, 2]                         # L->L + R->R
    flip = A[1, 2] + A[2, 1]                         # L->R + R->L
    to_f = A[1, 0] + A[2, 0]                         # L->F + R->F
    T = np.array([
        [ff,        f_turn / 2, f_turn / 2],
        [to_f / 2,  same / 2,   flip / 2],
        [to_f / 2,  flip / 2,   same / 2],
    ])
    T = np.maximum(T, EPS)
    return T / T.sum(axis=1, keepdims=True)


def _init_params(seqs, rng):
    """Random-but-sensible symmetric initialisation (transitions random, emissions
    from a rough +/-10 deg split of the data)."""
    alld = np.concatenate(seqs)
    fwd = alld[np.abs(alld) < DTHETA0_DEG]
    turn = np.abs(alld[np.abs(alld) >= DTHETA0_DEG])
    sigma = max(np.std(fwd) if len(fwd) > 5 else 8.0, 1.0)
    if len(turn) > 5:
        m, v = turn.mean(), turn.var()
        alpha = max(m * m / max(v, EPS), ALPHA_MIN)
        theta = max(v / max(m, EPS), 1.0)
    else:
        alpha, theta = 2.0, 20.0
    params = dict(sigma=float(sigma), alpha=float(alpha), theta=float(theta))

    # random symmetric transition matrix
    a = rng.uniform(0.3, 0.8)                        # F->F
    row_f = np.array([a, (1 - a) / 2, (1 - a) / 2])
    d, e, f = rng.uniform(0.2, 0.5), rng.uniform(0.3, 0.6), rng.uniform(0.1, 0.4)
    row_l = np.array([d, e, f]); row_l /= row_l.sum()
    T = np.array([row_f, row_l, [row_l[0], row_l[2], row_l[1]]])
    p_turn = rng.uniform(0.4, 0.8)
    pi = np.array([1 - p_turn, p_turn / 2, p_turn / 2])
    return pi, T, params


def _emission_mstep(seqs, gammas):
    """Symmetric emission M-step from per-sequence responsibilities.
    Returns dict(sigma, alpha, theta)."""
    # forward sigma: centered normal
    num_f = 0.0
    den_f = 0.0
    # turn gamma sufficient stats (pool L on -dtheta<0 and R on +dtheta>0)
    Wt = Sx = Slog = 0.0
    for s, g in zip(seqs, gammas):
        gF, gL, gR = g[:, 0], g[:, 1], g[:, 2]
        num_f += np.sum(gF * s * s)
        den_f += np.sum(gF)
        neg = s < 0
        pos = s > 0
        xw_neg, w_neg = -s[neg], gL[neg]            # magnitude, weight (Left)
        xw_pos, w_pos = s[pos], gR[pos]             # magnitude, weight (Right)
        Wt += w_neg.sum() + w_pos.sum()
        Sx += np.sum(w_neg * xw_neg) + np.sum(w_pos * xw_pos)
        Slog += (np.sum(w_neg * np.log(xw_neg + EPS))
                 + np.sum(w_pos * np.log(xw_pos + EPS)))
    sigma = np.sqrt(max(num_f / max(den_f, EPS), EPS))
    alpha, theta = _weighted_gamma_mle(Wt, Sx, Slog)
    return dict(sigma=float(max(sigma, 0.5)), alpha=alpha, theta=theta)


def fit_hmm_g3_sym(seqs, n_restarts=5, max_iter=500, tol=1e-6, seed=0, verbose=True):
    """Fit the symmetric 3-state HMM by Baum-Welch with random restarts.

    Returns dict(pi, T, sigma, alpha, theta, loglik, history, n_iter).
    """
    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + r)
        pi, T, params = _init_params(seqs, rng)
        prev_ll = -np.inf
        history = []
        for it in range(max_iter):
            # ---- E-step ----
            total_ll = 0.0
            gammas, A, gamma0 = [], np.zeros((3, 3)), np.zeros(3)
            for s in seqs:
                B = np.exp(emission_logprob(s, params))
                ll, gamma, xi_sum, g0 = _forward_backward(pi, T, B)
                total_ll += ll
                gammas.append(gamma)
                A += xi_sum
                gamma0 += g0
            history.append(total_ll)
            # ---- M-step (symmetric / tied) ----
            T = _symmetrise_transitions(A)
            pF = gamma0[0]; pT = (gamma0[1] + gamma0[2]) / 2
            pi = np.array([pF, pT, pT]); pi = pi / pi.sum()
            params = _emission_mstep(seqs, gammas)
            # ---- convergence ----
            if total_ll - prev_ll < tol and it > 0:
                break
            prev_ll = total_ll
        if verbose:
            print(f"  restart {r}: loglik={total_ll:.1f}  iters={it + 1}  "
                  f"sigma={params['sigma']:.2f} alpha={params['alpha']:.2f} "
                  f"theta={params['theta']:.2f}")
        if best is None or total_ll > best["loglik"]:
            best = dict(pi=pi, T=T, loglik=total_ll, history=history,
                        n_iter=it + 1, **params)
    return best


def viterbi(seq, hmm):
    """Most likely state sequence (0=F,1=L,2=R) for one dtheta sequence (deg)."""
    log_pi = np.log(hmm["pi"] + EPS)
    log_T = np.log(hmm["T"] + EPS)
    logB = emission_logprob(seq, hmm)
    T = len(seq)
    delta = np.empty((T, 3)); psi = np.empty((T, 3), dtype=int)
    delta[0] = log_pi + logB[0]
    for t in range(1, T):
        m = delta[t - 1][:, None] + log_T
        psi[t] = np.argmax(m, axis=0)
        delta[t] = logB[t] + np.max(m, axis=0)
    states = np.empty(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


# =========================================================================== #
#  Generation
# =========================================================================== #
def sample_emission(state, hmm, rng):
    """Sample one dtheta (deg) from the emission of `state`."""
    if state == 0:
        return rng.normal(0.0, hmm["sigma"])
    mag = rng.gamma(hmm["alpha"], hmm["theta"])
    return -mag if state == 1 else +mag             # Left negative, Right positive


def generate_bouts(hmm, n, dt_pool, dist_pool, rng):
    """Generate n synthetic bouts. Returns dict(state, angle_deg, dt_s, dist_px).

    States are sampled from the HMM; angles from the emissions; dt and dist are
    resampled independently from the empirical pools (bout-type-independent, as
    in the paper Sect 2.7). Distances are in PIXELS (pipeline is pixel-native)."""
    pi, T = hmm["pi"], hmm["T"]
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(3, p=pi)
    for t in range(1, n):
        states[t] = rng.choice(3, p=T[states[t - 1]])
    angle = np.array([sample_emission(s, hmm, rng) for s in states])
    dt = rng.choice(dt_pool, size=n)
    dist = rng.choice(dist_pool, size=n)
    return dict(state=states, angle_deg=angle, dt_s=dt, dist_px=dist)


def reconstruct_xy(angle_deg, dist_px, theta0=0.0):
    """Reconstruct a 2-D path from per-bout reorientation angles and distances (px)
    (paper Eq 5): theta_n = theta0 + cumsum(dtheta); x,y = cumsum(d cos/sin)."""
    theta = theta0 + np.cumsum(np.radians(angle_deg))
    x = np.cumsum(dist_px * np.cos(theta))
    y = np.cumsum(dist_px * np.sin(theta))
    return x, y, theta


# =========================================================================== #
#  Save / load
# =========================================================================== #
def save_hmm(path, hmm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pi=hmm["pi"], T=hmm["T"], sigma=hmm["sigma"],
             alpha=hmm["alpha"], theta=hmm["theta"], loglik=hmm["loglik"])


def load_hmm(path):
    d = np.load(path)
    return dict(pi=d["pi"], T=d["T"], sigma=float(d["sigma"]),
                alpha=float(d["alpha"]), theta=float(d["theta"]),
                loglik=float(d["loglik"]))


# =========================================================================== #
#  CLI
# =========================================================================== #
def main():
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bouts", type=Path, default=Path("data/bouts_GsizeMotionSingle.csv"))
    p.add_argument("--output", type=Path, default=Path("data/hmm_GsizeMotionSingle.npz"))
    p.add_argument("--gen-output", type=Path, default=Path("data/hmm_generated_bouts.csv"))
    p.add_argument("--min-len", type=int, default=10, help="min bouts per trajectory")
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--n-gen", type=int, default=2000, help="bouts to generate")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.bouts)
    seqs = load_sequences_from_bouts(df, min_len=args.min_len)
    n_bouts = sum(len(s) for s in seqs)
    print(f"loaded {len(seqs)} trajectories ({n_bouts} bouts, min_len={args.min_len})")

    # ---- Markov Chain (thresholded) ----
    bout_seqs = threshold_classifier(seqs)
    P_mc = bout_transitions(bout_seqs)
    pi_mc = bout_proba(bout_seqs)
    print("\n=========== MARKOV CHAIN (threshold +/-10 deg) ===========")
    print("state fractions (F,L,R):", np.round(pi_mc, 3))
    print("transition matrix (rows=from F,L,R):\n", np.round(P_mc, 3))
    print("steady state:", np.round(steady_state(P_mc), 3))

    # ---- HMM (Baum-Welch) ----
    print(f"\n=========== HMM G3 symmetric (Baum-Welch, {args.restarts} restarts) ===========")
    hmm = fit_hmm_g3_sym(seqs, n_restarts=args.restarts, max_iter=args.max_iter,
                         seed=args.seed)
    print("\nbest model:")
    print("  loglik      :", round(hmm["loglik"], 1), f"(converged in {hmm['n_iter']} iters)")
    print("  sigma (fwd) :", round(hmm["sigma"], 3), "deg")
    print("  turn Gamma  : alpha =", round(hmm["alpha"], 3),
          " theta =", round(hmm["theta"], 3), "deg  (mean turn =",
          round(hmm["alpha"] * hmm["theta"], 1), "deg)")
    print("  pi (F,L,R)  :", np.round(hmm["pi"], 3))
    print("  T (rows=from F,L,R):\n", np.round(hmm["T"], 3))
    print("  HMM steady state:", np.round(steady_state(hmm["T"]), 3))
    save_hmm(args.output, hmm)
    print("saved model ->", args.output)

    # ---- Generate synthetic bouts + one example trajectory ----
    rng = np.random.default_rng(args.seed)
    dt_pool = df["dt_s"].dropna().to_numpy()
    dist_pool = df["distance_px"].dropna().to_numpy()
    gen = generate_bouts(hmm, args.n_gen, dt_pool, dist_pool, rng)
    out = pd.DataFrame({"bout_angle": gen["angle_deg"], "bout_time": gen["dt_s"],
                        "bout_dist": gen["dist_px"], "state": gen["state"]})
    args.gen_output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.gen_output, index=False)
    print(f"generated {args.n_gen} bouts -> {args.gen_output}")

    gfrac = np.bincount(gen["state"], minlength=3) / args.n_gen
    print("generated state fractions (F,L,R):", np.round(gfrac, 3))
    print("real |dtheta| median:", round(np.median(np.abs(np.concatenate(seqs))), 1),
          "deg  | generated:", round(np.median(np.abs(gen["angle_deg"])), 1), "deg")


if __name__ == "__main__":
    main()
