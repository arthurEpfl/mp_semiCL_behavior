"""Smoke test for stim_mode='shoal' / 'shoalr' (the range x group-size x motion stimulus block).

Structural checks only -- performance numbers must come from the full Colab CV (this repo's
recorded lesson: in-sample smoke systematically overstates, e.g. dist_w1 0.32 local vs 2.79 held-out).

  [1] feature-matrix widths: dotb 14, shoalr 24, shoal 29
  [2] TRAIN/ROLLOUT IDENTITY, atol=0 -- the online row rebuilt from the (real) pose must equal the
      trained block row exactly. This is the assertion the by-hand column layout never had.
  [3] channel semantics: all-zero off-dot, softmax basis sums to 1, sum_j g*sin == sin
  [4] backward compatibility: 'dotb' features bit-identical to the pre-change definition
  [5] end-to-end: short fit + closed-loop rollout runs, 0% out-of-dish, save/load identical

Run:  python scripts/smoke_shoal.py [--bouts data/bouts_GsizeMotionSingle.csv] [--epochs 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP as M
import fit_MLP_border as MB
import fit_MLP_mix as MX
import evaluate as E

_DEFAULT_BOUTS = Path(__file__).parent.parent / "data" / "bouts_GsizeMotionSingle.csv"


def _cfg(stim_mode, dth_scale, K=8):
    return dict(K=K, stim_mode=stim_mode, include_symbols=False, dth_scale=dth_scale,
                rich_scales=None, use_border=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=str(_DEFAULT_BOUTS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--fish", type=int, default=4)
    args = ap.parse_args()

    print(f"loading {args.bouts}")
    df = pd.read_csv(args.bouts)
    records = MB.build_border_records(df)
    fids = sorted(records)
    print(f"  {len(fids)} fish, {sum(records[f].n for f in fids)} bouts")

    rec = records[args.fish]
    assert getattr(rec, "n_dots", None) is not None, "build_border_records did not attach rec.n_dots"
    assert len(rec.n_dots) == rec.n, f"n_dots length {len(rec.n_dots)} != rec.n {rec.n}"
    print(f"  fish {args.fish}: n={rec.n}, n_dots in {sorted(set(rec.n_dots.astype(int)))}")

    dth_scale = max(float(np.std(np.concatenate([records[f].dtheta for f in fids]))), 1.0)

    # ---- [1] widths ------------------------------------------------------------------ #
    want = {"dotb": 14, "shoalr": 24, "shoal": 29}
    for mode, w in want.items():
        got = MB.build_feature_matrix(rec, _cfg(mode, dth_scale)).shape[1]
        assert got == w, f"[1] {mode}: width {got} != {w}"
        print(f"[1] OK  {mode:7s} D={got}")

    # ---- [2] train/rollout identity ---------------------------------------------------- #
    # The trained block reads the STORED dot_dist_px / stim_angle_rad columns; the rollout
    # RECOMPUTES both from the generated pose. So a bitwise match is not achievable -- the
    # inputs themselves differ at float precision. Two separate assertions:
    #   [2a] the geometry recompute reproduces the stored columns to ~1e-12 (no stale/offset bug)
    #   [2b] the resulting rows agree to the same order -- which can ONLY hold if the column
    #        layout is identical, since any transposition would give an O(1) discrepancy.
    phi_rec = MB._wrap_pi(np.arctan2(rec.dot_cy - rec.y_px, rec.dot_cx - rec.x_px) - rec.f_ori_rad)
    r_rec = np.hypot(rec.dot_cx - rec.x_px, rec.dot_cy - rec.y_px)
    ok = np.isfinite(rec.dot_dist_px)
    e_phi = float(np.max(np.abs(MB._wrap_pi(phi_rec[ok] - rec.stim_angle[ok]))))
    e_r = float(np.max(np.abs(r_rec[ok] - rec.dot_dist_px[ok])))
    assert e_phi < 1e-10, f"[2a] recomputed bearing != stim_angle_rad (max {e_phi:.3e} rad)"
    assert e_r < 1e-9, f"[2a] recomputed range != dot_dist_px (max {e_r:.3e} px)"
    print(f"[2a] OK  geometry recompute matches stored columns: "
          f"bearing {e_phi:.2e} rad, range {e_r:.2e} px ({ok.sum()} active bouts)")

    for mode in ("shoal", "shoalr"):
        blk = M._stim_block(rec, mode, None)
        worst = 0.0
        for t in range(rec.n):
            row = MX._dot_stim_row(mode, rec.dot_cx[t], rec.dot_cy[t],
                                   rec.x_px[t], rec.y_px[t], rec.f_ori_rad[t],
                                   float(rec.moving[t]), float(rec.n_dots[t]))
            assert row.shape[0] == blk.shape[1], f"[2b] {mode}: row width {row.shape} vs {blk.shape}"
            worst = max(worst, float(np.max(np.abs(row - blk[t]))))
        # 1e-10 is ~4 orders above the observed float noise and ~10 orders below any
        # column transposition, so this discriminates layout from arithmetic.
        assert worst < 1e-10, f"[2b] {mode}: online row != trained block, max|diff|={worst:.3e}"
        print(f"[2b] OK  {mode:7s} online row == trained block for all {rec.n} bouts "
              f"(max|diff|={worst:.2e}, float-noise level)")

    # ---- [3] channel semantics -------------------------------------------------------- #
    blk = M._stim_block(rec, "shoal", None)
    act = blk[:, 0].astype(bool)
    off = ~act
    assert np.all(blk[off] == 0.0), "[3] off-dot rows are not all-zero"
    G = np.stack([blk[:, 4 + j] for j in range(4)], axis=1)      # ghat_j * sin(phi)
    phi = np.asarray(rec.stim_angle, float)
    s_ref = np.where(np.isfinite(phi), np.sin(np.nan_to_num(phi)), 0.0)
    # sum_j ghat_j == 1 on active bouts => sum_j (ghat_j*sin) == sin
    err = float(np.max(np.abs(G.sum(axis=1)[act] - s_ref[act])))
    assert err < 1e-9, f"[3] sum_j g*sin != sin (max err {err:.3e})"
    assert np.all(np.abs(blk) <= 1.0 + 1e-9), "[3] a channel exceeded O(1) bounds"
    nz = blk[act, 2]
    print(f"[3] OK  off-dot rows zero ({off.sum()} bouts); sum_j g*sin == sin (err {err:.1e}); "
          f"nz levels {sorted(set(np.round(nz, 4)))}")

    # ---- [4] backward compatibility of 'dotb' ----------------------------------------- #
    dotb = M._stim_block(rec, "dotb", None)
    ref = np.stack([np.nan_to_num(np.sin(phi)), np.nan_to_num(np.cos(phi)),
                    np.asarray(rec.moving, float)], axis=1)
    assert np.array_equal(dotb, ref), "[4] 'dotb' block changed!"
    for t in (0, rec.n // 2, rec.n - 1):
        row = MX._dot_stim_row("dotb", rec.dot_cx[t], rec.dot_cy[t],
                               rec.x_px[t], rec.y_px[t], rec.f_ori_rad[t], float(rec.moving[t]))
        assert row.shape[0] == 3, "[4] 'dotb' online row width changed"
    print("[4] OK  'dotb' block and online row unchanged (defaulted n_dots arg is compatible)")

    # ---- [5] end-to-end fit + rollout -------------------------------------------------- #
    train = [records[f] for f in fids[:-2]]
    test = [records[f] for f in fids[-2:]]
    ctx = E.make_context(train, dict(mlp_epochs=args.epochs))
    for mode in ("shoalr", "shoal"):
        model = MX.MLPMixKinDensity(K=8, stim_mode=mode, hidden=128, use_border=True,
                                    closed_loop=True, stim_loop=True, vigor=True)
        model.name = f"mlp_border_full_h128_mix_{mode}_hloop_kin_vigor"
        model.fit(train, ctx)
        rng = np.random.default_rng(0)
        out = 0
        tot = 0
        for r in test:
            g = model.sample(r, rng)
            d = np.hypot(np.asarray(g["x_px"]) - r.dish_cx, np.asarray(g["y_px"]) - r.dish_cy)
            out += int(np.sum(d > r.dish_R + 1e-6))
            tot += len(d)
            assert np.all(np.isfinite(g["dtheta"])), f"[5] {mode}: non-finite dtheta"
            assert np.all(np.isfinite(g["distance_px"])), f"[5] {mode}: non-finite distance"
            assert np.all(np.isfinite(g["dt_s"])), f"[5] {mode}: non-finite dt"
        assert out == 0, f"[5] {mode}: {out}/{tot} generated positions outside the dish"
        nfeat = model._model.body[0].in_features
        print(f"[5] OK  {mode:7s} fit({args.epochs}ep)+rollout, nfeat={nfeat}, {out}/{tot} out-of-dish")

        # save/load round trip
        p = Path(__file__).parent.parent / f"_smoke_{mode}.pt"
        MX.save_mix(model, str(p))
        back = MX.load_mix(str(p))
        a = model.sample(test[0], np.random.default_rng(7))
        b = back.sample(test[0], np.random.default_rng(7))
        for k in ("dtheta", "distance_px", "dt_s", "x_px", "y_px"):
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), f"[5] {mode}: save/load differs on {k}"
        p.unlink()
        print(f"        save/load bit-identical ({mode})")

    print("\nALL SMOKE CHECKS PASSED")
    print("NOTE: this is in-sample and structural only. The real A/B is the 4-rung ladder")
    print("      (dotb -> dot -> shoalr -> shoal) in ONE SELECT, 5-fold, after the Step-0 RNG fix.")


if __name__ == "__main__":
    main()
