"""Border/containment diagnostic -- the wall analogue of diag_range_profile.py.

Why this exists: with radial-occupancy features in the trajectory C2ST, FOUR numbers describing
where the fish sits in the dish carry AUC 0.797 on their own (vs 0.890 for all 39 features and
0.594 for the 20 turn features). Generated fish sit at r/R > 0.85 on ~8% of bouts versus 1.2% for
real fish, and real fish NEVER exceed r/R 0.95. That is the single largest remaining tell, and it
is not a stimulus problem -- the excess is as large during no-dot pauses as during dot blocks.

Only ~3% of generated steps are actually clipped by the _ray_to_wall guardrail, so the guardrail is
not mechanically parking the fish there. This script asks what is:

  [A] wall avoidance:  P(turn toward the dish centre) vs distance-to-border, real vs gen.
      The direct analogue of the shoal range profile. If gen is flat where real rises, the model
      has the same defect at the wall that it had at the shoal: a covariate it can see but whose
      response it has not learned to modulate.
  [B] turn magnitude vs border distance (does the model turn HARDER when cornered?)
  [C] escape dynamics: given a bout at r/R > 0.85, what is r/R after 1/2/5 bouts? Real fish leave
      quickly; if gen lingers, the state is absorbing.
  [D] step geometry at the wall: fraction of steps clipped, and the heading relative to the wall
      normal (is the model driving INTO the wall and being held there by the clip?)
  [E] occupancy distribution over r/R, real vs gen, split by stimulus present / pause.

Usage:
    python scripts/diag_border_profile.py --model ladder_out/ladder_shoal.pt
    python scripts/diag_border_profile.py --real-only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP_border as MB
import fit_MLP_mix as MX
import diag_range_profile as DR

_ROOT = Path(__file__).parent.parent
_DEFAULT_BOUTS = _ROOT / "clean_data" / "bouts_clean.csv"

# distance-to-border in units of dish radius: 0 = at the wall, 1 = at the centre
B_EDGES = np.array([0.0, 0.05, 0.10, 0.20, 0.35, 0.55, 1.01])
B_LABELS = ["<.05", ".05-.10", ".10-.20", ".20-.35", ".35-.55", ">.55"]


def _bin(bnorm):
    return np.clip(np.digitize(bnorm, B_EDGES[1:-1]), 0, len(B_LABELS) - 1)


def _profile(dtheta, x, y, ori, cx, cy, R):
    """P(turn toward centre), E|dtheta| and the radial restoring force E[dr] per border-distance
    bin. ``dr`` is the mean next-bout change of r/R given the current bin -- the radial analogue of
    diag_range_profile's E[dr|r], which is what located the 41 px shoal set point. ``rmid`` is the
    median r/R of the bin, so ``DR._zero_crossing`` can interpolate the radial set point."""
    x = np.asarray(x, float); y = np.asarray(y, float); ori = np.asarray(ori, float)
    r = np.hypot(x - cx, y - cy)
    bnorm = (R - r) / R
    rnorm = r / R
    nxt = np.full(len(rnorm), np.nan)
    nxt[:-1] = rnorm[1:]
    psi = MB._wrap_pi(np.arctan2(cy - y, cx - x) - ori)      # egocentric bearing to the centre
    b = _bin(bnorm)
    ok = np.isfinite(dtheta) & np.isfinite(psi)
    toward = np.full(len(B_LABELS), np.nan)
    mag = np.full(len(B_LABELS), np.nan)
    dr = np.full(len(B_LABELS), np.nan)
    rmid = np.full(len(B_LABELS), np.nan)
    n = np.zeros(len(B_LABELS), int)
    for j in range(len(B_LABELS)):
        m = ok & (b == j)
        n[j] = int(m.sum())
        if n[j] >= 20:
            toward[j] = float(np.mean(np.sign(dtheta[m]) == np.sign(psi[m])))
            mag[j] = float(np.mean(np.abs(dtheta[m])))
            rmid[j] = float(np.median(rnorm[m]))
        md = m & np.isfinite(nxt)
        if md.sum() >= 20:
            dr[j] = float(np.mean(nxt[md] - rnorm[md]))
    return dict(toward=toward, mag=mag, dr=dr, rmid=rmid, n=n, rnorm=rnorm, bnorm=bnorm)


def _set_point(parts):
    """Radial set point in r/R units: zero crossing of the pooled restoring force. Bins are
    ordered wall -> centre (descending r/R), and ``DR._zero_crossing`` expects ascending x with
    the force crossing + -> -, so both pooled arrays are reversed first."""
    dr = _pool(parts, "dr"); rmid = _pool(parts, "rmid")
    return DR._zero_crossing(np.asarray(dr)[::-1], np.asarray(rmid)[::-1])


def _escape(rnorm, thr=0.85):
    """Given a bout at r/R>thr, mean r/R after k bouts (does the fish leave the wall?)."""
    rn = np.asarray(rnorm, float)
    hit = np.flatnonzero(rn > thr)
    out = {}
    for k in (1, 2, 5, 10):
        idx = hit[hit + k < len(rn)]
        out[k] = float(np.mean(rn[idx + k])) if len(idx) else np.nan
    out["p_still"] = (float(np.mean(rn[hit[hit + 5 < len(rn)] + 5] > thr))
                      if len(hit[hit + 5 < len(rn)]) else np.nan)
    return out


def _pool(parts, key):
    V = np.stack([p[key] for p in parts]); N = np.stack([p["n"] for p in parts]).astype(float)
    w = np.where(np.isfinite(V), N, 0.0)
    return np.where(w.sum(0) > 0, np.nansum(np.nan_to_num(V) * w, 0) / np.maximum(w.sum(0), 1), np.nan)


def _row(v, p=3):
    return "  ".join("  nan" if not np.isfinite(x) else f"{x:6.{p}f}" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=str(_DEFAULT_BOUTS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--real-only", action="store_true")
    args = ap.parse_args()

    records = MB.build_border_records(pd.read_csv(args.bouts))
    # LIKE-FOR-LIKE frames: the gen side below reconstructs the previous TRAVEL heading from the
    # rollout positions, so the real side must use the same reference (MB._travel_ref), not the
    # raw body orientation -- f_ori carries ~40 deg of axis-flip noise and attenuates the profile.
    # The f_ori-framed profile is kept only for the wiring check printed below.
    rp, rp_ori, resc, rocc, rmn = [], [], [], [], []
    for r in records.values():
        p = _profile(np.asarray(r.dtheta, float), r.x_px, r.y_px, MB._travel_ref(r),
                     r.dish_cx, r.dish_cy, r.dish_R)
        po = _profile(np.asarray(r.dtheta, float), r.x_px, r.y_px, r.f_ori_rad,
                      r.dish_cx, r.dish_cy, r.dish_R)
        rp.append(p); rp_ori.append(po); resc.append(_escape(p["rnorm"]))
        rocc.append(p["rnorm"]); rmn.append(float(np.mean(p["rnorm"])))
    hdr = "         " + "  ".join(f"{s:>7s}" for s in B_LABELS)

    tw_t, tw_o = _pool(rp, "toward"), _pool(rp_ori, "toward")
    print(f"\n[frame check] real shell P(turn toward centre), bins <.05/.05-.10: "
          f"travel {tw_t[0]:.3f}/{tw_t[1]:.3f}  f_ori {tw_o[0]:.3f}/{tw_o[1]:.3f} "
          "(travel must be the larger; ~0.82 vs ~0.73)")
    print("\n=== [A] P(turn toward dish centre) by distance-to-border (0 = at wall, 1 = centre) ===")
    print(hdr); print("real    " + _row(tw_t))
    print("\n=== [B] E|dtheta| (deg) by distance-to-border ===")
    print(hdr); print("real    " + _row(_pool(rp, "mag"), 1))
    print("\n=== [F] radial restoring force E[d(r/R)] by distance-to-border ===")
    print(hdr); print("real    " + _row(_pool(rp, "dr"), 4))
    print(f"    real radial set point (zero crossing): r/R = {_set_point(rp):.3f}")
    ro = np.concatenate(rocc)
    print(f"\n=== [E] real occupancy: mean r/R {ro.mean():.3f}  "
          f"frac>0.85 {np.mean(ro > 0.85):.4f}  frac>0.95 {np.mean(ro > 0.95):.4f}  max {ro.max():.3f}")
    print(f"    between-fish SD of mean r/R: {np.std(rmn):.4f}  "
          f"(computed on {len(rmn)} fish; radial spread statistics are invalid below 15)")
    print("=== [C] real escape from r/R>0.85: " + "  ".join(
        f"+{k}b {np.nanmean([e[k] for e in resc]):.3f}" for k in (1, 2, 5, 10))
        + f"   still-at-wall after 5 bouts {np.nanmean([e['p_still'] for e in resc]):.3f}")

    if args.real_only or not args.model:
        return

    model = MX.load_mix(args.model)
    gp, gesc, gocc, gmn, clip_frac, into_wall = [], [], [], [], [], []
    for r in records.values():
        for rep in range(args.reps):
            g = model.sample(r, np.random.default_rng([7, int(r.fish_id), rep]))
            x = np.asarray(g["x_px"], float); y = np.asarray(g["y_px"], float)
            h = np.zeros(len(x)); h[0] = r.f_ori_rad[0]
            dx_, dy_ = np.diff(x), np.diff(y)
            h[1:] = np.arctan2(dy_, dx_)
            # zero-length steps (a fully blocked clip-mode bout) give atan2(0,0) = 0 -- an
            # absolute +x "heading" landing exactly in the shell bins this diagnostic exists to
            # measure. Carry the last real travel heading forward instead.
            for t in np.flatnonzero((dx_ == 0) & (dy_ == 0)) + 1:
                h[t] = h[t - 1]
            p = _profile(np.asarray(g["dtheta"], float), x, y, h, r.dish_cx, r.dish_cy, r.dish_R)
            gp.append(p); gesc.append(_escape(p["rnorm"])); gocc.append(p["rnorm"])
            gmn.append(float(np.mean(p["rnorm"])))
            # step t departs along h[t+1] (the reconstructed heading INTO x[t+1]), not h[t],
            # which is the arrival direction -- the generator clips along the post-turn heading.
            allow = np.array([MB._ray_to_wall(x[t], y[t], h[t + 1], r.dish_cx, r.dish_cy, r.dish_R)
                              for t in range(len(x) - 1)])
            dv = np.asarray(g["distance_px"], float)[:-1]
            clip_frac.append(float(np.mean(dv > allow - 1e-3)))
            # heading vs the outward wall normal, on bouts already near the wall
            near = p["rnorm"][:-1] > 0.85
            if near.any():
                nx = (x[:-1] - r.dish_cx); ny = (y[:-1] - r.dish_cy)
                nn = np.hypot(nx, ny) + 1e-9
                cosang = (np.cos(h[:-1]) * nx / nn + np.sin(h[:-1]) * ny / nn)
                into_wall.append(float(np.mean(cosang[near] > 0)))
    print("\ngen  (" + Path(args.model).stem + ")")
    print("\n=== [A] ==="); print(hdr)
    print("real    " + _row(_pool(rp, "toward"))); print("gen     " + _row(_pool(gp, "toward")))
    print("\n=== [B] ==="); print(hdr)
    print("real    " + _row(_pool(rp, "mag"), 1)); print("gen     " + _row(_pool(gp, "mag"), 1))
    print("\n=== [F] radial restoring force E[d(r/R)] ==="); print(hdr)
    print("real    " + _row(_pool(rp, "dr"), 4)); print("gen     " + _row(_pool(gp, "dr"), 4))
    print(f"    set point (zero crossing of E[dr]):  real r/R = {_set_point(rp):.3f}   "
          f"gen r/R = {_set_point(gp):.3f}")
    go = np.concatenate(gocc)
    print(f"\n=== [E] occupancy   real: mean {ro.mean():.3f} frac>0.85 {np.mean(ro > 0.85):.4f} "
          f"frac>0.95 {np.mean(ro > 0.95):.4f}")
    print(f"                    gen : mean {go.mean():.3f} frac>0.85 {np.mean(go > 0.85):.4f} "
          f"frac>0.95 {np.mean(go > 0.95):.4f}")
    print(f"    between-fish SD of mean r/R:  real {np.std(rmn):.4f}  gen {np.std(gmn):.4f}  "
          f"(gen over {len(gmn)} rollouts = {len(rmn)} fish x {args.reps} reps)")
    print("\n=== [C] escape from r/R>0.85 (mean r/R after k bouts) ===")
    for nm, E_ in (("real", resc), ("gen ", gesc)):
        print(f"  {nm}: " + "  ".join(f"+{k}b {np.nanmean([e[k] for e in E_]):.3f}" for k in (1, 2, 5, 10))
              + f"   still-at-wall after 5 {np.nanmean([e['p_still'] for e in E_]):.3f}")
    print(f"\n=== [D] gen steps clipped by _ray_to_wall: {np.mean(clip_frac):.4f}")
    if into_wall:
        print(f"    heading OUTWARD while at r/R>0.85: {np.mean(into_wall):.3f} "
              "(0.5 = no bias; >0.5 = driving into the wall)")


if __name__ == "__main__":
    main()
