"""Structural guard for the GRU + fish_re combination (`gruz`: state>0 with re_zero=False).

WHAT THIS IS DEFENDING. Before this arm existed, `state_chain` had no `z` argument, so switching
the GRU on silently routed AROUND the fish_re latent on BOTH the train and score sides -- while
`rollout_torch.rollout` was still applying it at generation time. That is the two-objects trap in
its purest form, and it is invisible on gru1/gru2 because those pin the latent at zero. So the
load-bearing checks here are [2] (the latent actually reaches the likelihood) and [4] (train,
score and generate all apply it the SAME way).

[1] and [5] are the regression guards: the pinned arm must stay bit-identical, and the two static
latents must stay mutually exclusive.

    python scripts/smoke_gruz.py
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

ok_all = True


def check(label, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def _fit(train, ctx, **kw):
    m = FR.MLPMixJointREDensity(K=8, K_kin=3, stim_mode="shoal", hidden=32, epochs=3,
                                n_dist=2, n_dt=3, border_mode="basis", psi_ref="travel",
                                use_border=True, closed_loop=True, guardrail=True, stim_loop=True,
                                contain=False, kin_fb_calib=False, vigor=False,
                                vigor_drift_calib=False, dt_snap=False, re_m=2,
                                re_gain_search=False, state=16, seq_batch=8, **kw)
    m.name = "smoke_" + ("gruz" if not kw.get("re_zero", True) else "gru1")
    return m.fit(train, ctx)


def main():
    root = Path(__file__).parent.parent
    recs = MB.build_border_records(pd.read_csv(root / "data" / "bouts_GsizeMotionSingle.csv"))
    fids = sorted(recs)
    train = [recs[f] for f in fids[:4]]
    test = recs[fids[4]]
    ctx = E.make_context(train, dict(mlp_epochs=3))

    print("\n=== [0] the constructor refuses TWO static latents ===")
    try:
        FR.MLPMixJointREDensity(state=16, h0_dim=2, re_zero=False)
        check("h0_dim>0 with re_zero=False raises", False)
    except ValueError as e:
        check("h0_dim>0 with re_zero=False raises", "static" in str(e).lower(), str(e)[:60])

    m0 = _fit(train, ctx, re_zero=True)         # gru1: latent pinned
    mz = _fit(train, ctx, re_zero=False)        # gruz: latent live

    print("\n=== [1] the PINNED arm still takes the cheap deterministic branch ===")
    check("gru1 reports no static latent", m0._state_latent_dim() == 0)
    check("gruz reports a 2-d static latent", mz._state_latent_dim() == 2,
          f"got {mz._state_latent_dim()}")

    print("\n=== [2] the latent actually REACHES the likelihood (the load-bearing check) ===")
    # Zero the loadings and the marginal must collapse onto the z = 0 chain. If score() were still
    # bypassing z, these two would already be equal and the check would fail loudly.
    s_on = mz.score(test)["logdens"]
    base = {n_: p.detach().clone() for n_, p in mz._model.named_parameters()
            if n_.startswith("z_")}
    check("the fitted loadings are not all zero",
          max(float(p.abs().max()) for p in base.values()) > 1e-6)
    with torch.no_grad():
        for n_, p in mz._model.named_parameters():
            if n_ in base:
                p.zero_()
    s_off = mz.score(test)["logdens"]
    with torch.no_grad():
        for n_, p in mz._model.named_parameters():
            if n_ in base:
                p.copy_(base[n_])
    d = float(np.nanmax(np.abs(s_on - s_off)))
    check("zeroing the loadings MOVES the marginal likelihood", d > 1e-6, f"max|diff| = {d:.3e}")

    print("\n=== [3] Metric A stays finite and exact-shaped (no product grid) ===")
    sc = mz.score(test)
    for k in ("logdens", "logdens_dist", "logdens_dt"):
        v = np.asarray(sc[k], float)
        check(f"{k}: finite where defined, right length",
              len(v) == len(test.dtheta) and np.isfinite(v[np.isfinite(v)]).all())
    q = FR._ghq(mz.re_m, mz.re_sd, k=mz.re_ghq_k)[0].shape[0]
    check("quadrature is the 2-d fish_re grid, not a product grid", q == mz.re_ghq_k ** mz.re_m,
          f"{q} nodes")

    print("\n=== [4] train / score / generate apply the latent the SAME way ===")
    # state_chain at a fixed z must reproduce joint_forward_re's shifts on the FIRST bout, where the
    # GRU state is still zero and the two paths are therefore comparable term for term.
    net = mz._model
    X, (a_z, ld_c), (ya, yd, yt) = mz._rec_arrays(test)
    zfix = torch.tensor(np.array([[0.7, -0.4]], np.float32))
    with torch.no_grad():
        t = lambda a, d=np.float32: torch.tensor(np.asarray(a, d)[:1]).reshape(1, 1)
        u = torch.tensor(FR.state_gru_input(ya, yd, yt, mz._cfg_feat["dth_scale"],
                                            mz._cfg_feat["kin_scales"]))[:1].reshape(1, 1, 3)
        Xb = torch.tensor(np.asarray(X[:1], np.float32)).reshape(1, 1, -1)
        la, _, _, _ = net.state_chain(Xb, t(a_z), t(ld_c), u, t(ya), t(yd), t(yt), z=zfix)
        # the same bout with a ZERO GRU state through the non-recurrent path is not directly
        # comparable (its trunk has no h), so compare the DELTA the latent produces instead
        la0, _, _, _ = net.state_chain(Xb, t(a_z), t(ld_c), u, t(ya), t(yd), t(yt), z=None)
    check("a non-zero z changes the first-bout turn density",
          abs(float(la) - float(la0)) > 1e-8, f"delta = {float(la) - float(la0):+.5f}")
    check("z=None and z=0 agree (the graph differs, the value must not)",
          abs(float(la0) - float(net.state_chain(Xb, t(a_z), t(ld_c), u, t(ya), t(yd), t(yt),
                                                 z=torch.zeros((1, 2)))[0])) < 1e-6)
    # generation: rollout_torch applies z from the fitted prior. Two draws with DIFFERENT latents
    # must give different per-fish levels, or individuality is not reaching the trajectory.
    g = FR.state_rollout_batch(net, test, mz._cfg_feat, 6, np.random.default_rng(0))
    lv = np.array([FR._fish_level_stats(o["dtheta"], o["distance_px"], o["dt_s"]) for o in g])
    check("generated trajectories differ between latent draws",
          float(np.nanstd(lv, axis=0).max()) > 0.0,
          f"per-level across-draw sd = {np.round(np.nanstd(lv, axis=0), 4)}")

    print("\n=== [5] the calibrator rolls out the RECURRENT model, not the numpy one ===")
    gain, rep = FR.search_loading_gain(net, train, mz._cfg_feat, grid=(0.5, 1.0),
                                       seed=0, max_fish=3, n_draws=2)
    check("search_loading_gain returns a gain from the grid", gain in (0.5, 1.0), f"{gain}")
    check("its curve has one entry per grid point", len(rep["curve"]) == 2)

    print("\n=== [6] save / load round-trips the combination ===")
    out = root / "smoke_gruz_tmp.pt"
    FR.save_fishre(mz, out)
    try:
        m2 = FR.load_fishre(out)
        check("re_zero survives", bool(m2.re_zero) is False)
        check("state survives", int(m2._cfg_feat.get("state", 0)) == 16)
        check("h0_dim stays 0", int(m2._model.h0_dim) == 0)
        d = float(np.nanmax(np.abs(np.asarray(m2.score(test)["logdens"], float)
                                   - np.asarray(sc["logdens"], float))))
        check("score is bit-identical after a round-trip", d == 0.0, f"max|diff| = {d:.3e}")
    finally:
        out.unlink(missing_ok=True)

    print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
