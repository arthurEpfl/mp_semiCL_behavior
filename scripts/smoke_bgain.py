"""Structural guard for re_bgain -- the likelihood-visible per-fish border-response gain.

Cheap fit on purpose (4 fish, 4 epochs, hidden 64, GHQ k=3): checks WIRING, not quality.
The two failure modes that matter:
  - the z=0 population path must be bit-identical to a net without the mechanism (archived
    consumers run joint_forward / predict_params_mix at z=0);
  - the numpy rollout predictor folds the gain into W1's border columns, and any slice or
    clamp mismatch with the torch path silently rolls out a different model than the one
    scored -- so torch joint_forward_re and the numpy closures must agree at the SAME z.

    python scripts/smoke_bgain.py
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
import fit_MLP_mix as MX                # noqa: E402
import fit_MLP_joint as JT              # noqa: E402
import fit_MLP_fishre as FR             # noqa: E402

ROOT = Path(__file__).parent.parent
ok_all = True


def check(label, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def main():
    recs = MB.build_border_records(pd.read_csv(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    fids = sorted(recs)
    train = [recs[f] for f in fids[:4]]
    test = recs[fids[4]]
    ctx = E.make_context(train, dict(mlp_epochs=4))

    m = FR.MLPMixJointREDensity(K=8, K_kin=3, stim_mode="shoal", hidden=64, epochs=4,
                                n_dist=2, n_dt=3, re_m=3, re_bgain=True, re_ghq_k=3,
                                re_gain_search=False)
    m.name = "smoke_bgain"
    m.fit(train, ctx)
    net = m._model

    print("\n=== [1] mechanism exists and left the saddle ===")
    check("net has z_bord and border_slice", hasattr(net, "z_bord") and net.border_slice is not None,
          f"slice={net.border_slice}")
    wnorm = float(net.z_bord.weight.norm())
    check("z_bord loading nonzero after fit", wnorm > 1e-6, f"|b|={wnorm:.4f}")
    s0, s1 = net.border_slice
    D = int(net.body[0].in_features)
    check("border slice inside the feature row and 3 wide (raw mode)",
          0 < s0 < s1 <= D and (s1 - s0) == 3, f"({s0},{s1}) of D={D}")
    check("validation NLL decreased", m.history[-1] < m.history[0],
          f"{m.history[0]:.4f} -> {m.history[-1]:.4f}")

    print("\n=== [2] z=0 population path identical to the plain forward ===")
    X = MB.build_feature_matrix(test, m._cfg_feat)[:64]
    Xt = torch.tensor(X)
    az = torch.zeros(len(X)); lc = torch.zeros(len(X))
    z0 = torch.zeros((len(X), net.m))
    with torch.no_grad():
        a_re = net.joint_forward_re(Xt, az, lc, z0)[0]
        a_pl = net.joint_forward(Xt, az, lc)[0]
    d = max(float((a_re[i] - a_pl[i]).abs().max()) for i in range(len(a_re)))
    check("joint_forward_re(z=0) == joint_forward (angle params)", d < 1e-6, f"max|diff|={d:.2e}")

    print("\n=== [3] numpy W1-fold agrees with the torch path at the same z ===")
    z = np.array([0.7, -0.4, 1.1], np.float32)[:net.m]
    body_angle, dist_params, dt_params = JT._numpy_predictor_joint(net, z=z)
    zt = torch.tensor(z).reshape(1, -1).expand(len(X), -1)
    with torch.no_grad():
        ang_t = net.joint_forward_re(Xt, az, lc, zt)[0]
    worst = 0.0
    for i in range(0, len(X), 16):
        _, ang_n = body_angle(X[i])
        worst = max(worst, float(np.max(np.abs(np.exp(ang_t[0][i].numpy()) - ang_n[0]))))
    check("mixture weights agree torch vs numpy at z != 0", worst < 1e-4, f"max|dw|={worst:.2e}")

    print("\n=== [4] rollout, score, save/load ===")
    g0 = m.sample(test, np.random.default_rng(3))
    g1 = m.sample(test, np.random.default_rng(3))
    check("rollout deterministic under the same rng", np.array_equal(g0["dtheta"], g1["dtheta"]))
    sc = m.score(test)
    fin = [int(np.isfinite(sc[k]).sum()) for k in ("logdens", "logdens_dist", "logdens_dt")]
    check("score() finite on 3 channels (GHQ with border gain per node)",
          fin[0] == test.n and fin[1] > 0 and fin[2] > 0, f"finite={fin} of {test.n}")
    p = ROOT / "_smoke_bgain.pt"
    MX.save_mix(m, str(p))
    m2 = MX.load_mix(str(p))
    check("round-trip keeps the mechanism",
          bool(m2.re_bgain) and getattr(m2._model, "border_gain", False)
          and tuple(m2._model.border_slice) == (s0, s1))
    g2 = m2.sample(test, np.random.default_rng(3))
    check("save/load round-trip rollout bit-identical", np.array_equal(g0["dtheta"], g2["dtheta"]))
    p.unlink(missing_ok=True)

    print("\n=== [5] archived checkpoints unaffected ===")
    ck = ROOT / "ladder_re_swap_gr.pt"
    if ck.exists():
        old = MX.load_mix(str(ck))
        check("old fishre checkpoint loads with border_gain off",
              not getattr(old._model, "border_gain", False) and not getattr(old, "re_bgain", False))
    else:
        print("  (skip: ladder_re_swap_gr.pt not present)")

    print("\n=== [6] BLOCK-STRUCTURED latent: routing enforced by the weight shapes ===")
    # THE LOAD-BEARING CHECK. Without the block, every loading reads every dimension, so
    # "re_m=3 gives re_bgain a spatial axis" was a hope and the extra capacity went to noise
    # (the m=3 arm fitted a gain of 0.28-0.40, the pre-registered fitting-noise band). A slice
    # that silently still reads the wrong columns trains fine and means the unconstrained thing
    # again, so assert the GRADIENTS are exactly zero rather than eyeballing the code.
    import torch as _t
    for blk in (False, True):
        net = FR.MDNMixJointRE(41, 32, n_dist=2, n_dt=3, n_fish=3, m=3,
                               border_gain=True, border_slice=(26, 35), block=blk)
        z = _t.randn(4, 3, requires_grad=True)
        def _g(out):
            return _t.autograd.grad(out.sum(), z, retain_graph=True)[0].abs().sum(0).detach().numpy()
        gt = _g(net.z_turn(net._z_core(z)))
        gd = _g(net.z_dist(net._z_core(z)))
        gx = _g(net.z_dt(net._z_core(z)))
        gb = _g(net.z_bord(net._z_spatial(z)))
        if blk:
            check("block: no level loading reads the spatial dim (exact 0)",
                  gt[2] == 0.0 and gd[2] == 0.0 and gx[2] == 0.0,
                  f"d/dz2 = turn {gt[2]:.3e}, dist {gd[2]:.3e}, dt {gx[2]:.3e}")
            check("block: the border gain reads ONLY the spatial dim (exact 0)",
                  gb[0] == 0.0 and gb[1] == 0.0, f"d/dz01 = {gb[0]:.3e}, {gb[1]:.3e}")
            check("block: the spatial dim is still LIVE on the border gain", gb[2] > 0)
            check("block: shapes reserve one dim", net.n_core == 2
                  and tuple(net.z_turn.weight.shape) == (1, 2)
                  and tuple(net.z_bord.weight.shape) == (1, 1))
        else:
            check("no block: every loading reads every dim (archived behaviour)",
                  min(gt[2], gd[2], gx[2], gb[0], gb[1]) > 0)
    # NOTE, and it is not a defect: with block=True the spatial dim still INFLUENCES the turn head,
    # because the border gain scales input columns that feed the shared body. That is exactly what
    # a per-fish wall response means. What the block forbids is the spatial dim entering any head's
    # LEVEL loading, which is the unconstrained pathway that spent m=3 on noise.
    for bad, why in ((dict(block=True, border_gain=False), "block without border_gain"),
                     (dict(block=True, border_gain=True, border_slice=(26, 35), m=1),
                      "block with m=1 (no core dims left)")):
        try:
            FR.MDNMixJointRE(41, 32, n_fish=3, **{**dict(m=3), **bad})
            check(f"{why} raises", False)
        except ValueError:
            check(f"{why} raises", True)

    print("\nALL BGAIN SMOKE CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
