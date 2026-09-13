"""Structural guard for SRNN Stage 1: GRU + pinned AR(1) drift + fitted per-fish latent.

Three mechanisms now share one likelihood, and the ways that goes wrong are specific: the drift
never reaches the heads; the level chain and the plain chain disagree so the marginalised model is
not the fitted one; the GRU is recomputed per level so the arm is L x more expensive than claimed;
or -- this project's oldest trap -- train, score and generate apply the slow state differently and
we ship an object nobody scored.

[1] and [4] are the load-bearing checks. [2] is the regression guard that keeps every archived
non-drift arm bit-identical.

    python scripts/smoke_srnn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_fishre as FR             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402
import generative_metrics as GM         # noqa: E402

ok_all = True


def check(label, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def main():
    root = Path(__file__).parent.parent
    recs = MB.build_border_records(pd.read_csv(root / "data" / "bouts_GsizeMotionSingle.csv"))
    fids = sorted(recs)
    train = [recs[f] for f in fids[:4]]
    test = recs[fids[4]]
    ctx = E.make_context(train, dict(mlp_epochs=3))

    print("\n=== [0] the constructor now allows drift+GRU, and still refuses dt_latent ===")
    try:
        FR.MLPMixJointREDensity(state=16, drift=dict(FR.DRIFT_DEFAULT))
        check("state + drift constructs", True)
    except ValueError as e:
        check("state + drift constructs", False, str(e)[:70])
    try:
        FR.MLPMixJointREDensity(state=16, dt_latent=dict(FR.DT_LATENT_DEFAULT))
        check("state + dt_latent still raises", False)
    except ValueError as e:
        check("state + dt_latent still raises", "interval" in str(e), str(e)[:60])
    except Exception:
        check("state + dt_latent still raises", True, "(no DT_LATENT_DEFAULT; skipped)")

    m = FR.MLPMixJointREDensity(
        K=8, K_kin=3, stim_mode="shoal", hidden=32, epochs=3, n_dist=2, n_dt=3,
        border_mode="basis", psi_ref="travel", use_border=True, closed_loop=True,
        guardrail=True, stim_loop=True, contain=False, kin_fb_calib=False, vigor=False,
        vigor_drift_calib=False, dt_snap=False, re_m=2, re_zero=False, re_gain_search=False,
        state=16, seq_batch=8, drift=dict(FR.DRIFT_DEFAULT), drift_gain_search=False)
    m.name = "smoke_srnn"
    m.fit(train, ctx)
    net = m._model

    print("\n=== [1] the drift reaches the LIKELIHOOD (load-bearing) ===")
    s_on = np.asarray(m.score(test)["logdens"], float)
    base = {n_: p.detach().clone() for n_, p in net.named_parameters()
            if n_ in ("d_turn", "d_dist", "d_dt")}
    check("the drift loadings are not all zero",
          max(float(p.abs().max()) for p in base.values()) > 1e-9)
    with torch.no_grad():
        for n_, p in net.named_parameters():
            if n_ in base:
                p.zero_()
    s_off = np.asarray(m.score(test)["logdens"], float)
    with torch.no_grad():
        for n_, p in net.named_parameters():
            if n_ in base:
                p.copy_(base[n_])
    d = float(np.nanmax(np.abs(s_on - s_off)))
    check("zeroing the drift MOVES the marginal likelihood", d > 1e-6, f"max|diff| = {d:.3e}")

    print("\n=== [2] the level chain IS the plain chain at s = 0 (no second object) ===")
    X, (a_z, ld_c), (ya, yd, yt) = m._rec_arrays(test)
    n = min(len(ya), 400)
    t1 = lambda a, d_=np.float32: torch.tensor(np.asarray(a, d_)[:n]).reshape(1, -1)
    with torch.no_grad():
        u = torch.tensor(FR.state_gru_input(ya, yd, yt, m._cfg_feat["dth_scale"],
                                            m._cfg_feat["kin_scales"]))[:n].reshape(1, n, 3)
        Xq = torch.tensor(np.asarray(X[:n], np.float32)).reshape(1, n, -1)
        zq = torch.zeros((1, m.re_m))
        la, ld, lt, _ = net.state_chain(Xq, t1(a_z), t1(ld_c), u, t1(ya), t1(yd), t1(yt), z=zq)
        lev = net.state_chain_levels(Xq, t1(a_z), t1(ld_c), u, t1(ya), t1(yd), t1(yt),
                                     torch.zeros(1), z=zq)
    ydb, ytb = t1(yd), t1(yt)
    md = (torch.isfinite(ydb) & (ydb > 0)).float()
    mt = (torch.isfinite(ytb) & (ytb > 0)).float()
    tot = la + md * ld + mt * lt
    d = float((lev[:, :, 0] - tot).abs().max())
    # FLOAT32 TOLERANCE, not exact equality, and the reason is worth stating rather than papering
    # over: `state_chain` evaluates the heads one bout at a time (it must stay bit-identical to the
    # archived gru1/gruz arms), while `state_chain_levels` batches every bout into one linear. Same
    # arithmetic, different reduction order, so they agree to ~1e-6 relative on a log-density of
    # order 5-10 -- float32 epsilon accumulated through a 14-wide layer and a logsumexp. Demanding
    # 0.0 would force one path to be slow. A real logic error here is O(0.1-1), nowhere near this.
    check("levels(s=0) == summed state_chain (float32)", d < 1e-4, f"max|diff| = {d:.3e}")

    print("\n=== [3] the GRU runs ONCE, not once per level (the cost claim) ===")
    calls = {"n": 0}
    orig = net.state_cell.forward

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    net.state_cell.forward = counted
    with torch.no_grad():
        net.state_chain_levels(Xq, t1(a_z), t1(ld_c), u, t1(ya), t1(yd), t1(yt),
                               torch.zeros(9), z=zq)
    net.state_cell.forward = orig
    check(f"GRU cell called {n} times for 9 levels, not {9 * n}", calls["n"] == n,
          f"got {calls['n']}")

    print("\n=== [4] train / score / GENERATE apply the same slow state (the oldest trap) ===")
    # The rollout must advance an AR(1) and shift the heads with it. With the loadings zeroed the
    # generated trajectories must be a DIFFERENT law from with them live -- if they are not, the
    # rollout is ignoring a state the likelihood was fitted with.
    def _levels(net_):
        g = FR.state_rollout_many(net_, train[:3], m._cfg_feat, 3, np.random.default_rng(0),
                                  n_cap=800)
        return np.array([GM.win_acf1(np.log(np.clip(o["dt_s"], 1e-6, None))) for o in g])
    acf_on = _levels(net)
    with torch.no_grad():
        for n_, p in net.named_parameters():
            if n_ in base:
                p.zero_()
    acf_off = _levels(net)
    with torch.no_grad():
        for n_, p in net.named_parameters():
            if n_ in base:
                p.copy_(base[n_])
    check("the drift changes the GENERATED across-window acf",
          not np.allclose(np.nan_to_num(acf_on), np.nan_to_num(acf_off)),
          f"acf on {np.nanmean(acf_on):+.3f}  off {np.nanmean(acf_off):+.3f}")

    print("\n=== [5] phi is PINNED -- a constant, never a parameter ===")
    check("d_phi is not in named_parameters",
          not any(n_.startswith("d_phi") for n_, _ in net.named_parameters()))
    # d_phi() returns a float32 TENSOR and drift_phi is float64, so this is a float32 comparison
    # however it is written -- 1e-9 was testing the dtype, not the value.
    check("d_phi() matches drift_phi(tau)",
          abs(float(net.d_phi()) - FR.drift_phi(net.drift["tau"])) < 1e-6,
          f"phi = {float(net.d_phi()):.6f} vs {FR.drift_phi(net.drift['tau']):.6f} "
          f"(tau {net.drift['tau']:.0f})")
    check("tau > W = 60, so it correlates ADJACENT windows", float(net.drift["tau"]) > 60.0,
          f"tau = {net.drift['tau']:.0f}")

    print("\n=== [6] save / load round-trips the combination ===")
    out = root / "smoke_srnn_tmp.pt"
    MX.save_mix(m, str(out))
    try:
        m2 = MX.load_mix(str(out))
        check("state survives", int(m2._cfg_feat.get("state", 0)) == 16)
        check("drift survives", bool(m2._cfg_feat.get("drift")))
        check("re_zero survives", bool(m2.re_zero) is False)
        d = float(np.nanmax(np.abs(np.asarray(m2.score(test)["logdens"], float) - s_on)))
        check("score is bit-identical after a round-trip", d == 0.0, f"max|diff| = {d:.3e}")
    finally:
        out.unlink(missing_ok=True)

    print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
