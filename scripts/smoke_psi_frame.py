"""Structural guard for the psi/phi travel-frame features, dt_snap and rad_indiv.

Cheap and checkpoint-based on purpose (no fitting): it checks WIRING, not quality.

The two failure modes that matter:
  - back-compat: every archived checkpoint's cfg lacks psi_ref/phi_ref/dt_snap, so its features
    and rollouts must be BIT-identical to pre-change behaviour;
  - CRN: dt_snap draws nothing and rad_indiv draws only from a SPAWNED stream, so paired arms
    must stay aligned bout for bout.

    python scripts/smoke_psi_frame.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP as M                     # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402
import fit_MLP_joint as JT              # noqa: E402

ROOT = Path(__file__).parent.parent
ok_all = True


def check(label, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def main():
    recs = MB.build_border_records(pd.read_csv(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    rec = min(recs.values(), key=lambda r: r.n)          # shortest fish -> fastest rollouts
    cf_base = dict(K=8, stim_mode="shoal", include_symbols=False, dth_scale=30.0,
                   rich_scales=None, use_border=True, border_mode="raw", use_kin=False)

    print("\n=== [1] back-compat: absent key == psi_ref='ori'/phi_ref='ori', bit-identical ===")
    for bm in ("raw", "basis"):
        X0 = MB.build_feature_matrix(rec, dict(cf_base, border_mode=bm))
        X1 = MB.build_feature_matrix(rec, dict(cf_base, border_mode=bm,
                                               psi_ref="ori", phi_ref="ori"))
        check(f"border_mode={bm}: feature matrices bit-identical", np.array_equal(X0, X1))

    print("\n=== [2] travel-frame border block matches an independent recompute ===")
    Xt = MB._border_block(rec, dict(cf_base, psi_ref="travel"))
    prev = MB._travel_ref(rec)
    psi = MB._wrap_pi(np.arctan2(rec.dish_cy - np.asarray(rec.y_px, float),
                                 rec.dish_cx - np.asarray(rec.x_px, float)) - prev)
    ref = np.stack([np.asarray(rec.dist_to_border_norm, float),
                    np.nan_to_num(np.sin(psi)), np.nan_to_num(np.cos(psi))], 1)
    check("psi columns equal wrap(bearing - prev_travel_heading)",
          float(np.max(np.abs(Xt - ref.astype(np.float32)))) < 1e-6)

    print("\n=== [3] _travel_ref identity: theta_{t-1} == heading shifted by one (in-segment) ===")
    hd = np.asarray(rec.heading_rad, float)
    agree = np.abs(MB._wrap_pi(prev[1:] - hd[:-1])) < 1e-9
    frac = float(np.mean(agree))
    check("prev[t] == heading[t-1] on >= 99% of rows (rest = segment boundaries)",
          frac >= 0.99, f"agree {frac:.4f}")

    print("\n=== [4] stim_angle_travel preserves the absolute bearing exactly ===")
    sa = np.asarray(rec.stim_angle, float)
    st = np.asarray(rec.stim_angle_travel, float)
    m_ = np.isfinite(sa)
    d = MB._wrap_pi((st[m_] + prev[m_]) - (sa[m_] + np.asarray(rec.f_ori_rad, float)[m_]))
    check("wrap(stim_travel + prev) == wrap(stim_ori + f_ori)", float(np.max(np.abs(d))) < 1e-9)
    check("NaN (no dot) propagates", int(np.isfinite(st).sum()) == int(m_.sum()))

    ckpt = ROOT / "ladder_b_indiv150.pt"
    if ckpt.exists():
        print("\n=== [5] dt_snap: lattice on dt, every other channel bit-identical ===")
        mj = MX.load_mix(str(ckpt))
        cf = mj._cfg_feat
        g0 = JT.generate_mix_joint_loop(mj._model, rec, dict(cf), np.random.default_rng(11))
        g1 = JT.generate_mix_joint_loop(mj._model, rec, dict(cf, dt_snap=True),
                                        np.random.default_rng(11))
        on_lat = np.allclose(np.asarray(g1["dt_s"]) * 30.0,
                             np.round(np.asarray(g1["dt_s"]) * 30.0), atol=1e-9)
        check("snapped dt is exactly on the 1/30 s lattice", on_lat)
        check("dt >= 1/30 everywhere (zero-rounding guard)",
              float(np.min(g1["dt_s"])) >= 1.0 / 30 - 1e-12)
        for k in ("dtheta", "distance_px", "x_px", "y_px"):
            check(f"{k} bit-identical to unsnapped control", np.array_equal(g0[k], g1[k]))
        check("snap_dt(1e-3) climbs to one frame", MB.snap_dt(1e-3) == 1.0 / 30)
    else:
        print("\n(skip [5]: ladder_b_indiv150.pt not present)")

    ckpt = ROOT / "ladder_re_swap_gr.pt"
    if ckpt.exists():
        print("\n=== [6] rad_indiv: spawned stream leaves the control aligned ===")
        mf = MX.load_mix(str(ckpt))
        cf = mf._cfg_feat

        def roll(rad_sd):
            c = dict(cf); c["fish_re"] = dict(cf["fish_re"])
            if rad_sd is not None:
                c["fish_re"]["rad_sd"] = rad_sd
            return JT.generate_mix_joint_loop(mf._model, rec, c, np.random.default_rng(13))

        a = roll(None)                    # archived cfg: no key
        b = roll(0.0)                     # explicit zero -> no spawn, must be identical
        check("rad_sd=0.0 rollout bit-identical to no-key rollout",
              np.array_equal(a["dtheta"], b["dtheta"]) and np.array_equal(a["x_px"], b["x_px"]))
        # Epsilon swing. The loop is CHAOTIC, so a 1e-9 bias perturbation legitimately diverges
        # the trajectory after enough bouts -- only bout 0 is a valid stream-alignment probe:
        # if u_rad_re had consumed a main-stream variate (or displaced the z spawn), dtheta[0]
        # would be a completely different draw (tens of degrees), not a ~1e-7 bias delta.
        c = roll(1e-9)
        d0 = abs(float(a["dtheta"][0]) - float(c["dtheta"][0]))
        check("rad_sd=1e-9: bout-0 draw aligned with control (z spawn + main stream intact)",
              d0 < 1e-3, f"|d dtheta[0]|={d0:.2e}")
        e = roll(0.4)                     # the mechanism actually does something
        check("rad_sd=0.4 changes the rollout", not np.array_equal(a["dtheta"], e["dtheta"]))

        print("\n=== [7] rad_gate: same CRN contract as rad_sd ===")

        def rollg(sd):
            c = dict(cf)
            if sd is not None:
                c["rad_gate"] = {"sd": sd}
            return JT.generate_mix_joint_loop(mf._model, rec, c, np.random.default_rng(17))

        ga = rollg(None)                  # archived cfg: no key -> no spawn
        gb = rollg(0.0)                   # key present, sd=0 -> spawn consumed, u_gate=0
        check("rad_gate sd=0.0 rollout bit-identical to no-key rollout",
              np.array_equal(ga["dtheta"], gb["dtheta"]) and np.array_equal(ga["x_px"], gb["x_px"]))
        gc_ = rollg(1e-9)                 # bout-0 probe (loop is chaotic beyond that)
        d0 = abs(float(ga["dtheta"][0]) - float(gc_["dtheta"][0]))
        check("rad_gate sd=1e-9: bout-0 draw aligned with control", d0 < 1e-3,
              f"|d dtheta[0]|={d0:.2e}")
        ge = rollg(0.3)
        check("rad_gate sd=0.3 changes the rollout", not np.array_equal(ga["dtheta"], ge["dtheta"]))
    else:
        print("\n(skip [6]/[7]: ladder_re_swap_gr.pt not present)")

    print("\nALL PSI-FRAME SMOKE CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
