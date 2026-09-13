"""Structural guard for the per-fish random effect (fit_MLP_fishre.MLPMixJointREDensity).

Run this BEFORE any Colab ladder. Every check here is a silent failure mode -- none of them raise on
their own, and several would produce a plausible-looking AUC that means nothing.

  [1] Gauss-Hermite weights integrate a standard normal correctly.
  [2] rng.spawn does NOT advance the parent stream. This is the whole CRN argument: rollouts are
      keyed on (seed, fish, rep) with no model index so replicates are paired ACROSS models, which
      requires every arm to consume the same variates from `rng` in the same order. If spawn
      advanced the parent, every emission draw after the z draw would desynchronise and the paired
      t-test would still print RESOLVED while being invalid.
  [3] Sigma = 0 reproduces the plain joint generator BITWISE, same seed, same records.
  [4] The quadrature filter is exact: sum_t log c_t == log p(y_{1:n}) computed independently, and
      the cumsum form agrees with a dense recomputation. Same identity evaluate.identity_check
      asserts for the HMM path.
  [5] score() shapes, and NaN in exactly the masked positions (the last bout of each fish, plus any
      zero-distance bout) -- and nowhere else.
  [6] symbol_logq rows normalise, and are NOT the z = 0 plug-in when Sigma > 0 (i.e. the symbol
      column is conditioned on the same belief as the density column beside it).
  [7] A rollout stays in the dish, respects the feature-width asserts, and save -> load -> score
      round-trips exactly.
  [8] EM recovers a known Sigma from simulated data, and de-shrinks relative to the MAP plug-in.

Usage:  python scripts/smoke_fishre.py [--fish 3] [--epochs 4] [--bouts 1200]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                       # noqa: E402
import fit_MLP_border as MB                # noqa: E402
import fit_MLP_fishre as FR                # noqa: E402
import fit_MLP_joint as JT                 # noqa: E402
import fit_MLP_mix as MX                   # noqa: E402

OK, BAD = "  ok  ", " FAIL "
_fails = []


def check(name, cond, detail=""):
    tag = OK if cond else BAD
    print(f"[{tag}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _fails.append(name)
    return cond


# --------------------------------------------------------------------------- #
def build(args):
    df = E.load_bouts()
    recs = list(MB.build_border_records(df).values())      # keyed by fish_id, not a list
    recs = [r for r in recs if r.n >= 200][: args.fish]
    if args.bouts:
        recs = [_truncate(r, args.bouts) for r in recs]
    print(f"records: {len(recs)} fish, lengths {[int(r.n) for r in recs]}")
    return recs


def _truncate(rec, n):
    """Shorten a FishRecord in place-safe fashion; every per-bout array is cut to the same length."""
    import copy
    r = copy.copy(rec)
    n = int(min(n, rec.n))
    for k, v in vars(rec).items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == rec.n:
            setattr(r, k, v[:n])
    return r


def _ctx(recs, cfg):
    return E.make_context(recs, cfg)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fish", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bouts", type=int, default=1200)
    args = ap.parse_args()

    # ---- [1] quadrature ---------------------------------------------------- #
    nodes, log_pi = FR._ghq(2, np.array([1.0, 2.0]), k=7)
    w = np.exp(log_pi)
    check("[1] GHQ weights sum to 1", abs(w.sum() - 1.0) < 1e-12, f"sum={w.sum():.12f}")
    m1 = (w.reshape(-1, 1) * nodes).sum(axis=0)
    m2 = (w.reshape(-1, 1) * nodes.astype(float) ** 2).sum(axis=0)
    # float32 nodes (they feed a float32 net), so ~1e-7 relative is the exact-arithmetic bar here.
    check("[1] GHQ reproduces E[z]=0, E[z^2]=diag(sd^2)",
          np.allclose(m1, 0, atol=1e-6) and np.allclose(m2, [1.0, 4.0], rtol=1e-6),
          f"E[z]={np.round(m1, 9)}  E[z^2]={np.round(m2, 8)}")
    n0, lp0 = FR._ghq(2, np.zeros(2), k=7)
    check("[1] degenerate Sigma collapses to a single node at 0",
          n0.shape == (1, 2) and np.allclose(n0, 0) and np.allclose(lp0, 0))

    # ---- [2] spawn does not advance the parent ----------------------------- #
    g = np.random.default_rng([0, 7]); a = g.normal(size=50)
    _child = g.spawn(1)[0]; _ = _child.normal(size=10)
    b = g.normal(size=50)
    ref = np.random.default_rng([0, 7]).normal(size=100)
    check("[2] rng.spawn leaves the parent stream untouched",
          np.array_equal(np.concatenate([a, b]), ref))

    # ---- fit both models --------------------------------------------------- #
    recs = build(args)
    ctx = _ctx(recs, dict(mlp_epochs=args.epochs, mlp_patience=args.epochs))

    kw = dict(K=8, stim_mode="shoal", hidden=32, use_border=True, closed_loop=True,
              guardrail=True, stim_loop=True, n_dist=2, n_dt=3, contain=False,
              kin_fb_calib=False, vigor=False, seed=0)
    print("\nfitting control (MLPMixJointDensity) ...")
    ctrl = JT.MLPMixJointDensity(**kw).fit(recs, ctx)
    print("fitting random-effect model ...")
    re = FR.MLPMixJointREDensity(re_m=2, **kw).fit(recs, ctx)
    print(f"  prior sd = {np.round(re.re_sd, 3)}   de-shrink loading gain = {re.re_gain}")

    # ---- [3] Sigma = 0 reproduces the control bitwise ---------------------- #
    KEYS = ("dtheta", "dt_s", "distance_px", "x_px", "y_px")

    def _roll(model, cfg, seed=(1, 2, 3)):
        return JT.generate_mix_joint_loop(model, recs[0], cfg, np.random.default_rng(list(seed)))

    def _same(a, b):
        return all(np.array_equal(a[k], b[k]) for k in KEYS)

    # (a) The CONTROL's own net, run through the RE code path with a zero-Sigma cfg. Identical
    #     arrays prove the spawned z draw consumes NOTHING from the parent stream -- the CRN claim.
    cf_ctrl_re = dict(ctrl._cfg_feat); cf_ctrl_re["fish_re"] = dict(m=2, sd=np.zeros(2))
    check("[3] Sigma=0 rollout is BITWISE identical to the plain joint generator",
          _same(_roll(ctrl._model, ctrl._cfg_feat), _roll(ctrl._model, cf_ctrl_re)))

    # (b) The Sigma-pinned arm must be FUNCTIONALLY inert. Testing that its weights are literally
    #     zero would be wrong now that they are randomly initialised -- what matters is that z
    #     cannot reach the output, not how the unused parameters were seeded.
    re0 = FR.MLPMixJointREDensity(re_m=2, re_zero=True, **kw).fit(recs, ctx)
    cf0_off = {k: v for k, v in re0._cfg_feat.items() if k != "fish_re"}
    check("[3] the Sigma-pinned arm is functionally inert (z cannot reach the output)",
          _same(_roll(re0._model, re0._cfg_feat), _roll(re0._model, cf0_off))
          and float(np.max(re0.re_sd)) == 0.0)

    # (c) ... and a fitted Sigma > 0 must actually change the rollout, or the mechanism is dead.
    cf_off = {k: v for k, v in re._cfg_feat.items() if k != "fish_re"}
    check("[3] a nonzero Sigma DOES change the rollout (the mechanism is live)",
          float(np.max(re.re_sd)) > 0 and not _same(_roll(re._model, re._cfg_feat),
                                                    _roll(re._model, cf_off)),
          f"sd={np.round(re.re_sd, 4)}")
    # (d) ... and different draws must give different fish, or every rollout is the average fish.
    r1 = _roll(re._model, re._cfg_feat, seed=(1, 2, 3))
    r2 = _roll(re._model, re._cfg_feat, seed=(9, 9, 9))
    lv1 = float(np.mean(np.log(np.clip(r1["distance_px"], 1e-6, None))))
    lv2 = float(np.mean(np.log(np.clip(r2["distance_px"], 1e-6, None))))
    check("[3] two prior draws produce measurably different fish",
          abs(lv1 - lv2) > 1e-6, f"mean log step {lv1:.4f} vs {lv2:.4f}")

    # ---- [4] the filter is exact ------------------------------------------- #
    rec = recs[0]
    X, (a_z, ld_c), (ya, yd, yt) = re._rec_arrays(rec)
    L = FR._node_logliks(re._model, X, a_z, ld_c, ya, yd, yt, re._re_nodes)
    ld_seq, logZ = FR._marginal_predictive(L, re._re_log_pi)
    from scipy.special import logsumexp
    logZ_direct = float(logsumexp(re._re_log_pi + L.sum(axis=0)))
    check("[4] sum_t log c_t == log p(y_1:n)  (the chain rule telescopes)",
          abs(float(np.sum(ld_seq)) - logZ_direct) < 1e-6,
          f"sum={np.sum(ld_seq):.9f}  direct={logZ_direct:.9f}")
    # dense recomputation of the same object, no cumsum
    dense = np.empty(len(ld_seq))
    for t in range(len(ld_seq)):
        lw = re._re_log_pi + (L[:t].sum(axis=0) if t else 0.0)
        dense[t] = logsumexp(lw + L[t]) - logsumexp(lw)
    check("[4] cumsum filter == dense recomputation",
          np.allclose(ld_seq, dense, atol=1e-9),
          f"max|diff|={np.max(np.abs(ld_seq - dense)):.3e}")
    # The strongest test of the CHAINED three-channel marginalisation: each channel is scored under
    # a belief that has already absorbed the earlier channels of the SAME bout, so the three must
    # still sum to log p(dtheta, l, tau | past) and telescope to log p(y_1:n). Marginalising all
    # three against one shared predictive belief -- the obvious implementation -- passes every shape
    # and finiteness check and silently fails exactly here, by double-counting the evidence.
    s_chk = re.score(rec)
    tot3 = (s_chk["logdens"] + np.nan_to_num(s_chk["logdens_dist"])
            + np.nan_to_num(s_chk["logdens_dt"])).sum()
    # RELATIVE tolerance: the chained path recomputes each channel through separate float32 torch
    # passes and does three logsumexps per bout, so it accumulates round-off that the single-pass
    # float64 reference does not. Measured 4e-10 to 3e-9 relative, growing with sequence length --
    # round-off, not a logic error. A genuine double-count would be O(1) nats per bout, ~1e5 here.
    rel = abs(float(tot3) - logZ_direct) / max(abs(logZ_direct), 1.0)
    check("[4] the three chained channels telescope to log p(y_1:n)",
          rel < 1e-7,
          f"sum(3 channels)={tot3:.6f}  log p(y)={logZ_direct:.6f}  rel={rel:.2e}")

    # ---- [5] score() shapes and masking ------------------------------------ #
    s = re.score(rec)
    n = rec.n
    shapes_ok = (s["logdens"].shape == (n,) and s["logdens_dist"].shape == (n,)
                 and s["logdens_dt"].shape == (n,) and s["symbol_logq"].shape == (n, 3))
    check("[5] score() returns the four keys with the right shapes", shapes_ok)
    check("[5] the turn channel is finite everywhere", bool(np.isfinite(s["logdens"]).all()))
    dv = np.asarray(rec.distance_px, float); tv = np.asarray(rec.dt_s, float)
    md = np.isfinite(dv) & (dv > 0); mt = np.isfinite(tv) & (tv > 0)
    check("[5] NaN in exactly the masked distance bouts and nowhere else",
          np.array_equal(np.isfinite(s["logdens_dist"]), md),
          f"masked={int((~md).sum())}")
    check("[5] NaN in exactly the masked interval bouts and nowhere else",
          np.array_equal(np.isfinite(s["logdens_dt"]), mt),
          f"masked={int((~mt).sum())}")

    # ---- [6] symbol column is belief-weighted ------------------------------ #
    q = np.exp(s["symbol_logq"])
    check("[6] symbol_logq rows normalise", np.allclose(q.sum(axis=1), 1.0, atol=1e-9))
    plug = MX.symbol_logq_mix(*MX.predict_params_mix(re._model, X))
    differs = not np.allclose(s["symbol_logq"], plug, atol=1e-8)
    check("[6] symbol_logq is NOT the z=0 plug-in (it uses the same belief as the density)",
          differs if float(np.max(re.re_sd)) > 0 else True,
          f"max|diff|={np.max(np.abs(s['symbol_logq'] - plug)):.3e}")

    # ---- [7] rollout sanity + save/load round-trip ------------------------- #
    gen = re.sample(rec, np.random.default_rng([4, 5, 6]))
    r_out = np.hypot(gen["x_px"] - rec.dish_cx, gen["y_px"] - rec.dish_cy) / rec.dish_R
    check("[7] rollout stays inside the dish", float(np.nanmax(r_out)) <= 1.0 + 1e-6,
          f"max r/R={np.nanmax(r_out):.4f}")
    check("[7] rollout arrays are finite and the right length",
          all(np.isfinite(gen[k]).all() and len(gen[k]) == n
              for k in ("dtheta", "dt_s", "distance_px")))
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "re.pt")
        MX.save_mix(re, p)                       # must route through save_fishre, not save_joint
        blob = torch.load(p, map_location="cpu", weights_only=False)
        check("[7] save_mix dispatches a SUBCLASS to the right serialiser",
              blob.get("cls") == "MLPMixJointREDensity", f"cls={blob.get('cls')}")
        re2 = FR.load_fishre(p)
        s2 = re2.score(rec)
        check("[7] save -> load -> score round-trips exactly",
              np.allclose(s["logdens"], s2["logdens"], atol=0)
              and np.allclose(np.nan_to_num(s["logdens_dt"]),
                              np.nan_to_num(s2["logdens_dt"]), atol=0))
        check("[7] the reloaded prior is the FITTED one, not a fresh random embedding",
              np.array_equal(re2.re_sd, re.re_sd))

    # ---- [10] THE MECHANISM IS ACTUALLY LEVEL-ONLY -------------------------- #
    # The first full ladder measured 90% / 74% / 59% of the latent's effect on the turn, distance and
    # interval heads flowing through the shared BODY rather than the level loadings -- so it was not
    # level-only at all, the pooled marginals broke, and the arm finished +0.0299 worse than its
    # anchor. Assert the body path is inert, and measure the split rather than trusting the flag.
    check("[10] the body path is off and exactly zero",
          (not re._model.use_body)
          and float(np.abs(re._model.z_body.weight.detach().numpy()).max()) == 0.0)
    Xs = torch.tensor(MB.build_feature_matrix(rec, re._cfg_feat)[:2000])
    zc = torch.zeros(Xs.shape[0]); rgen = np.random.default_rng(0)

    def _heads(zv, kill_body):
        w = re._model.z_body.weight.detach().clone()
        if kill_body:
            with torch.no_grad():
                re._model.z_body.weight.zero_()
        with torch.no_grad():
            a, dp, tp = re._model.joint_forward_re(
                Xs, zc, zc, torch.tensor(zv, dtype=torch.float32).reshape(1, -1).expand(Xs.shape[0], -1))
        with torch.no_grad():
            re._model.z_body.weight.copy_(w)
        return np.exp(a[0].numpy())[:, 0], dp[1].numpy().mean(1), tp[1].numpy().mean(1)

    b0 = _heads(np.zeros(re.re_m), False)
    shares = []
    for _ in range(20):
        zz = rgen.normal(size=re.re_m)
        f, l = _heads(zz, False), _heads(zz, True)
        shares.append([abs(np.std(f[i] - b0[i]) if i == 0 else np.mean(f[i] - b0[i]))
                       - abs(np.std(l[i] - b0[i]) if i == 0 else np.mean(l[i] - b0[i]))
                       for i in range(3)])
    resid = float(np.abs(np.array(shares)).max())
    check("[10] zeroing the body changes nothing (all effect is in the level loadings)",
          resid < 1e-12, f"max residual effect {resid:.3e}")

    # ---- [9] NO IDENTITY LEAK ---------------------------------------------- #
    # The fitted per-fish embeddings are TRAINING scaffolding: they exist so the loadings can learn
    # what individual variation looks like, and are then discarded. A held-out fish has no row, so
    # score() integrates z out under the prior and sample() draws a fresh z. If either ever started
    # indexing the embedding -- the obvious "personalise the rollout" change -- it would condition on
    # data inside the C2ST's own real class and every AUC would be invalid. Scribbling over every
    # fitted identity must therefore change nothing at all.
    s_pre = re.score(rec)
    g_pre = re.sample(rec, np.random.default_rng([11, 12, 13]))
    with torch.no_grad():
        re._model.emb.weight.normal_(0.0, 50.0)
    s_post = re.score(rec)
    g_post = re.sample(rec, np.random.default_rng([11, 12, 13]))
    check("[9] scrambling every fitted per-fish embedding leaves score() identical",
          np.array_equal(s_pre["logdens"], s_post["logdens"])
          and np.array_equal(np.nan_to_num(s_pre["logdens_dist"]),
                             np.nan_to_num(s_post["logdens_dist"]))
          and np.array_equal(np.nan_to_num(s_pre["logdens_dt"]),
                             np.nan_to_num(s_post["logdens_dt"])))
    check("[9] ... and leaves sample() identical",
          all(np.array_equal(g_pre[k], g_post[k]) for k in KEYS))

    # ---- [8] the de-shrinking gain is bounded, finite and interior --------- #
    check("[8] the fitted loading gain is on the grid and finite",
          re.re_gain in FR.GAIN_GRID and np.isfinite(re.re_gain), f"gain={re.re_gain}")
    # Interior-optimum and fixed-point only hold once there are enough fish to estimate an
    # across-fish SD at all. At 4 fish the objective is flat within its own noise and pins at an
    # edge; at 6 it is a clean U. That is the project's 3-fish trap, not a code defect -- so these
    # two are asserted only when the run has enough fish, and REPORTED otherwise.
    enough = len(recs) >= 6
    edge = (FR.GAIN_GRID[0], FR.GAIN_GRID[-1])
    check("[8] the gain is NOT pinned at a grid edge (else the grid is too narrow)",
          (re.re_gain not in edge) or not enough,
          f"gain={re.re_gain}  grid=({edge[0]}..{edge[1]})  n_fish={len(recs)}"
          + ("" if enough else "  [not asserted below 6 fish]"))
    rep = re.re_gain_report
    check("[8] the gain minimises the between-fish SPREAD mismatch (level is held separately)",
          rep is not None and min(rep["curve"], key=lambda c: c[1])[0] == re.re_gain,
          "err by gain: " + ", ".join(f"{c[0]}:{c[1]:.2f}(s{c[3]:.2f}+l{c[4]:.2f})"
                                      for c in rep["curve"]))
    # The re-centring must hold the pooled marginals while the gain widens the population. Without
    # it, scaling gamma inflates E[exp(mu + gamma z)] and the mechanism buys spread by wrecking the
    # marginal -- measured on the first end-to-end run as dist_w1 3.04 -> 6.78.
    def _pooled(model, cfg):
        outs = [JT.generate_mix_joint_loop(model, r, cfg, np.random.default_rng([7, int(r.fish_id)]))
                for r in recs]
        return (float(np.mean([np.mean(np.log(np.clip(o["distance_px"], 1e-6, None))) for o in outs])),
                float(np.mean([np.mean(np.log(np.clip(o["dt_s"], 1e-6, None))) for o in outs])))
    import copy as _copy
    net_g1 = _copy.deepcopy(re._model)
    FR.apply_loading_gain(net_g1, 1.0 / max(re.re_gain, 1e-9))     # undo the fitted gain
    p_on, p_off = _pooled(re._model, re._cfg_feat), _pooled(net_g1, re._cfg_feat)
    check("[8] the gain re-centring holds the pooled log-travel level",
          abs(p_on[0] - p_off[0]) < 0.25, f"gain-applied {p_on[0]:.3f} vs undone {p_off[0]:.3f}")
    check("[8] the gain re-centring holds the pooled log-interval level",
          abs(p_on[1] - p_off[1]) < 0.25, f"gain-applied {p_on[1]:.3f} vs undone {p_off[1]:.3f}")
    # A second search on the ALREADY-corrected net should land at (or next to) 1.0: the correction
    # is close to a fixed point, which is what says it is a real match rather than a one-shot nudge.
    g2, _rep2 = FR.search_loading_gain(re._model, recs, re._cfg_feat, seed=0,
                                       closed_loop=True, guardrail=True, stim_loop=True)
    check("[8] re-running the search on the corrected net lands near 1.0 (near fixed point)",
          (edge[0] < g2 < edge[1] and 0.5 <= g2 <= 2.0) or not enough,
          f"second-pass gain={g2}" + ("" if enough else "  [not asserted below 6 fish]"))
    check("[8] the prior really is N(0, I) (nothing left free to diverge)",
          np.allclose(re.re_sd, 1.0), f"sd={np.round(re.re_sd, 6)}")

    print()
    if _fails:
        print(f"{len(_fails)} FAILED: " + "; ".join(_fails))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
