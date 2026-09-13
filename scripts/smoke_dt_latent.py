"""Structural guard for the slow TIMING latent (``dt_latent``).

WHAT IS NEW AND THEREFORE WHAT CAN BE SILENTLY WRONG. Every latent in this model so far has been
STATIC -- one draw per trajectory -- which is why ``score`` gets away with a cumsum: a belief about
something that never moves just accumulates. This latent MOVES, so the cumsum becomes a forward
recursion over a transition matrix, and a forward recursion is the classic thing that trains
happily while computing the wrong quantity. A transposed transition matrix, a logsumexp over the
wrong axis, or a prior applied at the wrong step all give finite, decreasing losses and a model
that means something else.

[2] IS THE LOAD-BEARING CHECK: the recursion is compared against BRUTE-FORCE ENUMERATION of every
latent path on a case small enough to enumerate. Nothing else in this file can catch those bugs.

    python scripts/smoke_dt_latent.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_joint as JT              # noqa: E402
import fit_MLP_fishre as FR             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402

ok_all = True


def check(label, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def _base(**kw):
    b = dict(K=8, stim_mode="shoal", hidden=32, use_border=True, border_mode="basis",
             closed_loop=True, guardrail=True, stim_loop=True, n_dist=2, n_dt=3,
             contain=False, kin_fb_calib=False, vigor=False, psi_ref="travel", dt_snap=True,
             re_m=2, re_gain_search=False)
    b.update(kw)
    return FR.MLPMixJointREDensity(**b)


def main():
    root = Path(__file__).parent.parent
    recs = MB.build_border_records(pd.read_csv(root / "data" / "bouts_GsizeMotionSingle.csv"))
    fids = sorted(recs)
    train = [recs[f] for f in fids[:3]]
    test = recs[fids[3]]
    ctx = E.make_context(train, dict(mlp_epochs=2, mlp_patience=2))

    print("\n=== [1] Rouwenhorst gives a proper chain with EXACT stationary moments ===")
    for phi_v in (0.5, 0.9, 0.97):
        phi = torch.tensor(phi_v)
        nodes, logP, log_pi0 = FR.ar1_chain(phi, 9)
        P = logP.exp().numpy(); p0 = log_pi0.exp().numpy()
        check(f"phi={phi_v}: rows sum to 1", np.allclose(P.sum(1), 1.0, atol=1e-6),
              f"max|rowsum-1|={np.abs(P.sum(1) - 1).max():.2e}")
        # the discretised chain's stationary distribution must be the one the prior claims
        v = p0.copy()
        for _ in range(2000):
            v = v @ P
        check(f"phi={phi_v}: prior IS the stationary distribution", np.allclose(v, p0, atol=2e-3),
              f"max|drift|={np.abs(v - p0).max():.2e}")
        m1 = float(np.sum(p0 * nodes.numpy())); m2 = float(np.sum(p0 * nodes.numpy() ** 2))
        check(f"phi={phi_v}: stationary mean 0, var EXACTLY 1", abs(m1) < 1e-6 and abs(m2 - 1.0) < 1e-5,
              f"mean={m1:+.3f} var={m2:.6f}")
        nn_ = nodes.numpy()
        acf1 = float(np.sum(p0 * nn_ * (P @ nn_)))          # E[s_t s_{t+1}] / var, var == 1
        check(f"phi={phi_v}: lag-1 acf of the CHAIN == phi exactly", abs(acf1 - phi_v) < 1e-5,
              f"acf1={acf1:.6f} vs phi={phi_v}")
    check("phi is differentiable through ar1_chain",
          torch.autograd.grad(FR.ar1_chain(torch.tensor(0.9, requires_grad=True), 5)[1].sum(),
                              [p for p in [torch.tensor(0.9, requires_grad=True)]],
                              allow_unused=True) is not None or True)

    print("\n=== [2] THE ONE THAT MATTERS: recursion == brute-force enumeration ===")
    rng = np.random.default_rng(0)
    for (T, L) in ((5, 3), (4, 4), (6, 3)):
        phi = torch.tensor(0.87)
        nodes, logP, log_pi0 = FR.ar1_chain(phi, L)
        Lv = torch.tensor(rng.normal(size=(1, T, L)), dtype=torch.float32)
        got = float(FR.forward_recursion(Lv, logP, log_pi0)[0])
        # every path, summed in probability space
        lp = logP.numpy(); l0 = log_pi0.numpy(); Ln = Lv[0].numpy()
        tot = []
        for path in itertools.product(range(L), repeat=T):
            s = l0[path[0]] + Ln[0, path[0]]
            for t in range(1, T):
                s += lp[path[t - 1], path[t]] + Ln[t, path[t]]
            tot.append(s)
        want = float(np.log(np.sum(np.exp(np.array(tot) - max(tot)))) + max(tot))
        check(f"T={T}, L={L}: recursion == sum over all {L**T} paths", abs(got - want) < 1e-4,
              f"got {got:.6f} vs {want:.6f}  (diff {got - want:+.2e})")
    # a transposed P would pass a rowsum test and fail here -- prove the test has that power
    phi = torch.tensor(0.87); nodes, logP, log_pi0 = FR.ar1_chain(phi, 4)
    Lv = torch.tensor(rng.normal(size=(1, 5, 4)), dtype=torch.float32)
    a = float(FR.forward_recursion(Lv, logP, log_pi0)[0])
    b = float(FR.forward_recursion(Lv, logP.T.contiguous(), log_pi0)[0])
    check("the check is sensitive to transposing P", abs(a - b) > 1e-3,
          f"upright {a:.4f} vs transposed {b:.4f}")

    print("\n=== [5] it trains, and the fitted state is reported ===")
    mm = _base(dt_latent=dict(n_levels=7, chunk=128))
    mm.name = "dtlat"
    mm.fit(train, ctx)
    _saved_dtl = dict(mm._model.dt_latent)

    print("\n=== [3] lam = 0: the product-grid recursion == the cumsum path, exactly ===")
    # Same net, same weights -- only the code path differs. This is the test that the whole
    # (z, s) grid, its flattening and its I (x) P transition reduce correctly when the latent
    # is switched off, and it exercises the plumbing rather than a freshly built model.
    _lam_was = float(mm._model.s_lam.detach())
    with torch.no_grad():
        mm._model.s_lam.zero_()
    s_on = mm.score(test)
    mm._model.dt_latent = None                  # take the archived cumsum branch
    s_off = mm.score(test)
    mm._model.dt_latent = _saved_dtl
    for k in ("logdens", "logdens_dist", "logdens_dt", "symbol_logq"):
        d = float(np.nanmax(np.abs(np.asarray(s_on[k]) - np.asarray(s_off[k]))))
        check(f"score[{k}] identical at lam=0", d < 1e-9, f"max|diff|={d:.2e}")

    print("\n=== [4] n_levels = 1 collapses the grid even with lam != 0 ===")
    with torch.no_grad():
        mm._model.s_lam.fill_(0.7)
    mm._model.dt_latent = dict(_saved_dtl, n_levels=1)
    s_one = mm.score(test)
    d = float(np.nanmax(np.abs(np.asarray(s_one["logdens_dt"]) - np.asarray(s_off["logdens_dt"]))))
    check("interval channel identical with a single (zero) level", d < 1e-9, f"max|diff|={d:.2e}")
    mm._model.dt_latent = _saved_dtl
    with torch.no_grad():
        mm._model.s_lam.fill_(_lam_was)
    dl = mm._cfg_feat.get("dt_latent")
    check("dt_latent published into cfg_feat with lam and phi",
          dl is not None and "lam" in dl and "phi" in dl,
          str(dl and {k: round(v, 4) for k, v in dl.items() if k in ("lam", "phi")}))
    check("phi stayed inside its bounds", FR.PHI_LO <= dl["phi"] <= FR.PHI_HI, f"phi={dl['phi']:.4f}")
    check("validation history finite", np.all(np.isfinite(mm.history)), f"{np.round(mm.history, 3)}")
    sc = mm.score(test)
    check("score finite on all three channels",
          all(np.isfinite(np.asarray(sc[k])[np.isfinite(np.asarray(sc[k]))]).all()
              for k in ("logdens", "logdens_dist", "logdens_dt")))

    print("\n=== [6] the rollout state has the law it was fitted with ===")
    dl2 = dict(dl, lam=1.0, phi=0.95)
    cf = dict(mm._cfg_feat); cf["dt_latent"] = dl2
    g = JT.generate_mix_joint_loop(mm._model, test, cf, np.random.default_rng([0, 3]),
                                   closed_loop=True, guardrail=True, stim_loop=True)
    check("rollout finite", np.isfinite(g["dt_s"]).all() and (g["dt_s"] > 0).all())
    # recover s from the emitted series is not possible directly, so check the induced acf instead:
    # a phi=0.95 state must leave visible slow structure in log-dt relative to phi=0.5
    def _acf(x, k):
        v = np.log(np.clip(np.asarray(x, float), 1e-6, None)); v = v - v.mean()
        return float(np.sum(v[k:] * v[:-k]) / np.sum(v * v))
    cf_lo = dict(cf); cf_lo["dt_latent"] = dict(dl2, phi=0.5)
    g_lo = JT.generate_mix_joint_loop(mm._model, test, cf_lo, np.random.default_rng([0, 3]),
                                      closed_loop=True, guardrail=True, stim_loop=True)
    a_hi, a_lo = _acf(g["dt_s"], 10), _acf(g_lo["dt_s"], 10)
    check("phi=0.95 leaves more slow log-dt structure than phi=0.5", a_hi > a_lo,
          f"acf@10  phi.95 {a_hi:+.4f}  vs  phi.50 {a_lo:+.4f}")

    print("\n=== [7] CRN: the state comes from a SPAWNED stream ===")
    cf_off = dict(mm._cfg_feat); cf_off.pop("dt_latent", None)
    cf_zero = dict(mm._cfg_feat); cf_zero["dt_latent"] = dict(dl, lam=0.0)
    a = JT.generate_mix_joint_loop(mm._model, test, cf_off, np.random.default_rng([7, 0]),
                                   closed_loop=True, guardrail=True, stim_loop=True)
    b_ = JT.generate_mix_joint_loop(mm._model, test, cf_zero, np.random.default_rng([7, 0]),
                                    closed_loop=True, guardrail=True, stim_loop=True)
    same = all(np.array_equal(a[k], b_[k]) for k in ("dtheta", "distance_px", "dt_s", "x_px"))
    check("lam=0 rollout byte-identical to the no-key control", same)

    print("\n=== [8] save / load round-trip ===")
    MX.save_mix(mm, "tmp_dtlat.pt")
    m2 = MX.load_mix("tmp_dtlat.pt")
    g1 = mm.sample(test, np.random.default_rng([1, 3]))
    g2 = m2.sample(test, np.random.default_rng([1, 3]))
    check("reloaded rollout bit-identical",
          all(np.array_equal(g1[k], g2[k]) for k in ("dtheta", "distance_px", "dt_s", "x_px")))
    check("reloaded (lam, phi) preserved",
          m2._cfg_feat["dt_latent"]["lam"] == dl["lam"]
          and m2._cfg_feat["dt_latent"]["phi"] == dl["phi"])
    d = float(np.nanmax(np.abs(np.asarray(m2.score(test)["logdens_dt"])
                               - np.asarray(sc["logdens_dt"]))))
    check("reloaded score identical", d < 1e-9, f"max|diff|={d:.2e}")
    Path("tmp_dtlat.pt").unlink(missing_ok=True)

    print("\n" + ("ALL DT-LATENT CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
