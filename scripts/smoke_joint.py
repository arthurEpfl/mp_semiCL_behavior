"""Structural guard for the joint-likelihood model (scripts/fit_MLP_joint.py).

In-sample and cheap on purpose: this checks that the thing is WIRED correctly, not that it is
good. Local performance numbers in this project have overstated held-out ones by close to an
order of magnitude; the real A/B is the paired Colab ladder.

The load-bearing check is [2]. The feature row is assembled by hand inside the rollout and
independently by ``MB.build_feature_matrix`` at training time, and nothing but a runtime width
assert connects them. A silent column-order drift between the two would train on one layout and
generate from another, and every downstream number would look plausible and be wrong. That
failure mode is why ``smoke_shoal.py`` exists, and the joint model adds a fourth block (kinematic
history) to the same hand-assembled row.

    python scripts/smoke_joint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP as M                     # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402
import fit_MLP_joint as JT              # noqa: E402

EXPECT_D = {"dot": 23, "shoalr": 30, "shoal": 35}     # 8 turn lags + stim + 3 border + 6 kin
# K_stim > 1 inserts MB._stimhist_block BETWEEN stim and border: 1 + 4*(K_stim-1) columns.
EXPECT_D_KSTIM = {(s, 3): d + 9 for s, d in EXPECT_D.items()}
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
    ctx = E.make_context(train, dict(mlp_epochs=4))

    for stim in ("dot", "shoalr", "shoal"):
        print(f"\n=== stim_mode={stim} ===")
        m = JT.MLPMixJointDensity(K=8, K_kin=3, stim_mode=stim, hidden=64, epochs=4,
                                  n_dist=2, n_dt=3)
        m.name = f"joint_{stim}"
        m.fit(train, ctx)
        cf = m._cfg_feat
        D = int(m._model.body[0].in_features)

        # [1] width -- the joint x really is turn history + stim + border + kin history
        X = MB.build_feature_matrix(test, cf)
        check("feature width matches the expected joint layout",
              D == EXPECT_D[stim] == X.shape[1], f"D={D} expected {EXPECT_D[stim]}")

        # [2] THE ONE THAT MATTERS: the hand-assembled online row must equal the trained matrix.
        # Teacher-forced (closed_loop=False, stim_loop=False) so pose-driven blocks are identical
        # by construction and any mismatch is a LAYOUT bug, not a trajectory difference.
        g = JT.generate_mix_joint_loop(m._model, test, cf, np.random.default_rng(0),
                                       closed_loop=False, stim_loop=False, keep_features=True)
        Xr = g["X"]
        stim_blk = M._stim_block(test, cf["stim_mode"], cf["rich_scales"])
        shist_blk = MB._stimhist_block(test, cf)
        bord_blk = MB._border_block(test, cf)
        s0 = 8                                   # after the 8 turn-history lags
        s1 = s0 + stim_blk.shape[1]
        sh = s1 + shist_blk.shape[1]
        s2 = sh + bord_blk.shape[1]
        dstim = float(np.max(np.abs(Xr[:, s0:s1] - stim_blk)))
        dbord = float(np.max(np.abs(Xr[:, sh:s2] - bord_blk)))
        check("online stim slice == trained stim block", dstim < 1e-6, f"max|diff|={dstim:.2e}")
        if shist_blk.shape[1]:
            dsh = float(np.max(np.abs(Xr[:, s1:sh] - shist_blk)))
            check("online stimhist slice == trained K_stim block", dsh < 1e-6,
                  f"max|diff|={dsh:.2e}")
        check("online border slice == trained border block", dbord < 1e-6, f"max|diff|={dbord:.2e}")
        check("row 0 == trained X row 0 (histories empty at t=0)",
              float(np.max(np.abs(Xr[0] - X[0]))) < 1e-6)
        check("kin slice occupies the LAST 2*K_kin columns", s2 + 2 * cf["K_kin"] == D)

        # [3] containment: the guardrail is what keeps the fish in the dish
        gh = m.sample(test, np.random.default_rng(1))
        u = np.hypot(gh["x_px"] - test.dish_cx, gh["y_px"] - test.dish_cy) / test.dish_R
        check("0% out of dish", float(np.max(u)) <= 1.0 + 1e-6, f"max r/R={np.max(u):.4f}")

        # [4] three channels, right lengths, last bout masked
        sc = m.score(test)
        n = test.n
        okc = all(sc[k].shape == (n,) for k in ("logdens", "logdens_dist", "logdens_dt"))
        fin = [int(np.isfinite(sc[k]).sum()) for k in ("logdens", "logdens_dist", "logdens_dt")]
        check("score() returns 3 channels of length n", okc, f"finite = {fin} of {n}")
        # This dataset has NO NaN distance/dt (checked: 0 of 70590) but DOES have 10 bouts at
        # distance exactly 0. Training masks `yd > 0`; score() must mask identically or a
        # log-normal at the 1e-6 clamp contributes ~-850 nats and silently shifts bits/bout.
        nzero = int((np.asarray(test.distance_px, float) <= 0).sum())
        check("distance channel masks non-positive bouts exactly as training does",
              fin[1] == n - nzero, f"masked {n - fin[1]}, expected {nzero}")
        check("dt channel keeps every positive bout", fin[2] == n)
        bits = {k: float(-np.nanmean(sc[v]) / np.log(2))
                for k, v in (("ang", "logdens"), ("dist", "logdens_dist"), ("dt", "logdens_dt"))}
        print(f"       bits/bout  angle {bits['ang']:.4f}  distance {bits['dist']:.4f}  dt {bits['dt']:.4f}"
              "   (different base measures -- compare across MODELS, not to each other)")

        # [5] the joint loss actually went down
        check("joint validation NLL decreased", m.history[-1] < m.history[0],
              f"{m.history[0]:.4f} -> {m.history[-1]:.4f}")

        # [6] save/load round-trip must reproduce a rollout exactly
        p = root / f"_smoke_joint_{stim}.pt"
        MX.save_mix(m, str(p))                          # dispatches to JT.save_joint
        m2 = JT.load_joint(str(p))
        a = m.sample(test, np.random.default_rng(7))["dtheta"]
        b = m2.sample(test, np.random.default_rng(7))["dtheta"]
        check("save/load round-trip is bit-identical", bool(np.array_equal(a, b)))
        p.unlink(missing_ok=True)

    print("\nALL JOINT SMOKE CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
