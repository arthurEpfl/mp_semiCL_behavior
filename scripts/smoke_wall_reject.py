"""Structural guard for the shallow-border REJECTION sampler (contain_mode='reject').

The mechanism is a constraint on POSITION rather than a bias on turning, so the things that can
go wrong are: it does not actually enforce the floor, it costs far more draws than predicted, it
falls back to the guardrail too often, or it silently changes the LIKELIHOOD (it must not --
containment is generation-only, so any movement in ``score`` is a bug).

The archive once rejected "rollout rejection" on an ``acceptance -> 0`` argument. That objection
was against a hard rejection with no graded ramp; this file is the measurement that settles it
rather than the argument. [2] and [3] are the load-bearing checks.

    python scripts/smoke_wall_reject.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402
import fit_MLP_joint as JT              # noqa: E402

D0, D1 = 2.0, 10.0
MAX_DRAWS = 20
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

    print("\n=== [0] accept_prob is the ramp it claims to be ===")
    R = test.dish_R
    for d_mm, want in ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (6.0, 0.5), (10.0, 1.0), (30.0, 1.0)):
        u = 1.0 - d_mm / (0.5 * MB.DISH_MM)
        got = MB.accept_prob(u, D0, D1)
        check(f"accept_prob at {d_mm:4.1f} mm == {want}", abs(got - want) < 1e-9, f"got {got:.3f}")
    check("wall_mm inverts r/R exactly", abs(MB.wall_mm(0.85) - 7.5) < 1e-9)

    m = JT.MLPMixJointDensity(K=8, K_kin=3, stim_mode="shoal", hidden=64, epochs=4,
                              n_dist=2, n_dt=3, border_mode="basis", psi_ref="travel",
                              dt_snap=True, contain=False)
    m.name = "smoke_reject"
    m.fit(train, ctx)
    cf_base = dict(m._cfg_feat)
    cf_base["wall_mode"] = "reflect"
    cf = dict(cf_base)
    cf["centre_bias"] = {"bulk": 0.0, "shell": 0.0, "u0": 0.75}
    cf["wall_reject"] = {"d0": D0, "d1": D1, "max_draws": MAX_DRAWS}

    print("\n=== [1] the floor is actually enforced ===")
    gaps, draws, redraw, fallback, nb_tot = [], 0, 0, 0, 0
    for r in [recs[f] for f in fids[4:10]]:
        g = JT.generate_mix_joint_loop(m._model, r, cf, np.random.default_rng([1, int(r.fish_id)]),
                                       closed_loop=True, guardrail=True, stim_loop=True)
        u = np.hypot(g["x_px"] - r.dish_cx, g["y_px"] - r.dish_cy) / r.dish_R
        nb = len(u); nb_tot += nb
        gaps.append(float(MB.wall_mm(u.max())))
        draws += g["n_draws"]; redraw += g["n_redraw_bouts"]; fallback += g["n_reject_fallback"]
    worst = min(gaps)
    check(f"no bout lands inside d0 = {D0} mm", worst >= D0 - 1e-6,
          f"closest approach over 6 fish = {worst:.3f} mm")

    print("\n=== [2] the loop terminates cheaply (the acceptance->0 objection) ===")
    # Thresholds are for THIS deliberately under-trained 4-epoch model, which wanders far more
    # than a ladder model does. The archived J23 checkpoint measures 1.024 draws/bout with zero
    # fallbacks; if that number ever approaches these bounds, something has regressed.
    dpb = draws / nb_tot
    check("mean draws per bout < 1.5 (trained model: 1.02)", dpb < 1.5, f"{dpb:.4f}")
    check("bouts needing a redraw < 15%", redraw / nb_tot < 0.15,
          f"{100 * redraw / nb_tot:.2f}%")
    check("best-of-N fallback < 2% of bouts", fallback / nb_tot < 0.02,
          f"{100 * fallback / nb_tot:.4f}%")

    print("\n=== [3] STRESS: seeded ON the wall, it still escapes ===")
    import copy
    r = recs[fids[4]]
    stress = copy.copy(r)
    ang = np.arctan2(r.y_px[0] - r.dish_cy, r.x_px[0] - r.dish_cx)
    xs = np.array(r.x_px, float).copy(); ys = np.array(r.y_px, float).copy()
    xs[0] = r.dish_cx + 0.995 * r.dish_R * np.cos(ang)      # r/R 0.995 = 0.25 mm from the wall,
    ys[0] = r.dish_cy + 0.995 * r.dish_R * np.sin(ang)      # deeper than any real fish ever goes
    stress.x_px = xs; stress.y_px = ys
    g = JT.generate_mix_joint_loop(m._model, stress, cf, np.random.default_rng([2, 0]),
                                   closed_loop=True, guardrail=True, stim_loop=True)
    u = np.hypot(g["x_px"] - r.dish_cx, g["y_px"] - r.dish_cy) / r.dish_R
    check("escapes the forbidden ring by bout 1", MB.wall_mm(u[1]) >= D0 - 1e-6,
          f"bout 1 at {MB.wall_mm(u[1]):.2f} mm")
    check("the whole stressed rollout still respects the floor",
          float(MB.wall_mm(u.max())) >= D0 - 1e-6,
          f"closest = {MB.wall_mm(u.max()):.3f} mm")

    print("\n=== [4] generation-only: the LIKELIHOOD must not move ===")
    s_off = m.score(test)
    m._cfg_feat = cf
    s_on = m.score(test)
    d = max(float(np.nanmax(np.abs(s_on[k] - s_off[k])))
            for k in ("logdens", "logdens_dist", "logdens_dt"))
    check("score() bit-identical with the ramp on", d == 0.0, f"max|diff|={d:.3e}")

    print("\n=== [5] key ABSENT => byte-identical to the archived path ===")
    cf_none = dict(cf_base)
    cf_none["centre_bias"] = {"bulk": 4.0, "shell": 35.0, "u0": 0.75}
    a = JT.generate_mix_joint_loop(m._model, test, cf_none, np.random.default_rng([3, 0]),
                                   closed_loop=True, guardrail=True, stim_loop=True)
    b = JT.generate_mix_joint_loop(m._model, test, cf_none, np.random.default_rng([3, 0]),
                                   closed_loop=True, guardrail=True, stim_loop=True)
    same = all(np.array_equal(a[k], b[k]) for k in ("dtheta", "distance_px", "dt_s", "x_px"))
    check("no wall_reject key -> deterministic, unchanged rollout", same)
    check("no reject counters leak into the output dict", "n_draws" not in a)

    print("\n=== [6] the calibrator returns a usable pair ===")
    cb, wr = MX._search_wall_reject(
        m._model, train, cf_base, seed=0, max_fish=3, w_occ=2.6, tail_max=0.002,
        # d0 is PINNED, not swept (see REJECT_D0_MM) -- passing the old `d0_grid=` name here
        # silently fell through **gen_kw into the generator and raised a TypeError.
        d0=2.0, d1_grid=(10.0,), bulk_grid=(0.0, 2.0),
        closed_loop=True, guardrail=True, stim_loop=True)
    check("centre_bias has shell pinned to 0", cb["shell"] == 0.0)
    check("centre_bias flags reject", cb.get("reject") is True)
    check("wall_reject carries d0/d1/max_draws", {"d0", "d1", "max_draws"} <= set(wr))
    check("wall_reject reports its own cost", "draws_per_bout" in wr and "fallback_frac" in wr,
          f"draws/bout={wr.get('draws_per_bout', float('nan')):.3f}")
    try:
        MX._search_wall_reject(m._model, train, cf_base, graded=True)
        check("graded + reject raises", False)
    except ValueError:
        check("graded + reject raises (they parameterise the same term)", True)

    print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
