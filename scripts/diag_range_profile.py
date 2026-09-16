"""Range-resolved acceptance test for the stimulus block -- the diagnostic that decides whether
a stimulus change worked. The trajectory C2ST cannot do this job: its dot-distance features are
~2/3 block-onset placement artefact, and it has no radial-occupancy channel at all.

Measured on the real data (15 fish) and on the shipped dotb checkpoint, the five signatures are:

  [A] P(turn toward the shoal) vs centroid RANGE -- real is an inverted U peaking at 70-90 px:
        real  .546 .604 .632 .663 .677 .620 .592   over <25/25-40/40-55/55-70/70-90/90-120/>120 px
        gen   .593 .612 .602 .608 .609 .608 .630   (FLAT -- dotb has no range input)
      curvature = peak bin minus the mean of the two end bins: real +0.094, gen -0.003.
  [B] restoring force E[dr | r] -- real crosses zero at ~43 px (a set point) and flattens far away
      (-0.53 px/bout beyond 120 px); gen keeps pulling hard when far (-4.09) and never backs off.
  [C] frontal turn suppression -- with |phi|<30 deg, real P(|dtheta|<10 deg) rises 0.116 -> 0.315
      from 17 px to 103 px; gen is flat at 0.14-0.17 (one phi->magnitude curve for all ranges).
  [D] the dotdist_q90 gen-real gap per group size N: +3.7 (N=1) -> +78.6 (N=8). A working range
      channel must FLATTEN this gradient. A model that only lowers the overall mean has instead
      fixed the occupancy term ([E]) and should not be credited for the stimulus.
  [E] radial occupancy -- generated fish sit at r/R > 0.85 on 7.9-9.2% of bouts vs real 1.2%, and
      the excess is LARGER during no-dot pauses. This is a containment defect, not a stimulus one;
      it is reported here so [D] is not misread.

Usage:
    python scripts/diag_range_profile.py --model mix_dotb_hloop_kin_vigor.pt
    python scripts/diag_range_profile.py --stim-mode shoal --epochs 40      # fit a fresh one
    python scripts/diag_range_profile.py --real-only                        # targets only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP_border as MB
import fit_MLP_mix as MX
import evaluate as E

_ROOT = Path(__file__).parent.parent
_DEFAULT_BOUTS = _ROOT / "clean_data" / "bouts_clean.csv"

R_EDGES = np.array([0.0, 25.0, 40.0, 55.0, 70.0, 90.0, 120.0, np.inf])
R_LABELS = ["<25", "25-40", "40-55", "55-70", "70-90", "90-120", ">120"]
PEAK_BIN = 4          # the 70-90 px bin, where real following peaks
FRONTAL_DEG = 30.0
FORWARD_DEG = 10.0


def _bin_idx(r):
    return np.clip(np.digitize(r, R_EDGES[1:-1]), 0, len(R_LABELS) - 1)


def _curvature(prof):
    """peak bin minus the mean of the two end bins; +ve = inverted-U (a range set point)."""
    ends = np.nanmean([prof[0], prof[-1]])
    return float(prof[PEAK_BIN] - ends)


def _profiles(dtheta, phi, r, moving, nxt_r=None):
    """[A] P(turn toward), [B] E[dr|r], [C] P(forward | frontal) per range bin, moving bouts only."""
    use = np.asarray(moving, bool) & np.isfinite(phi) & np.isfinite(r) & np.isfinite(dtheta)
    b = _bin_idx(np.where(np.isfinite(r), r, 0.0))
    toward = np.full(len(R_LABELS), np.nan)
    dr = np.full(len(R_LABELS), np.nan)
    fwd = np.full(len(R_LABELS), np.nan)
    rmid = np.full(len(R_LABELS), np.nan)
    n = np.zeros(len(R_LABELS), int)
    frontal = use & (np.abs(phi) < np.radians(FRONTAL_DEG))
    for j in range(len(R_LABELS)):
        m = use & (b == j)
        n[j] = int(m.sum())
        if n[j] >= 20:
            toward[j] = float(np.mean(np.sign(dtheta[m]) == np.sign(phi[m])))
            rmid[j] = float(np.median(r[m]))
        if nxt_r is not None:
            md = m & np.isfinite(nxt_r)
            if md.sum() >= 20:
                dr[j] = float(np.mean(nxt_r[md] - r[md]))
        mf = frontal & (b == j)
        if mf.sum() >= 20:
            fwd[j] = float(np.mean(np.abs(dtheta[mf]) < FORWARD_DEG))
    return dict(toward=toward, dr=dr, fwd=fwd, rmid=rmid, n=n)


def _zero_crossing(dr, rmid):
    """First range at which the restoring force changes sign (the set point), by linear interp."""
    ok = np.isfinite(dr) & np.isfinite(rmid)
    d, x = dr[ok], rmid[ok]
    for i in range(len(d) - 1):
        if d[i] > 0 >= d[i + 1]:
            f = d[i] / (d[i] - d[i + 1])
            return float(x[i] + f * (x[i + 1] - x[i]))
    return np.nan


def _fmt(row, prec=3):
    return "  ".join("  nan" if not np.isfinite(v) else f"{v:5.{prec}f}" for v in row)


def real_side(records):
    parts = []
    for r in records.values():
        rr = np.asarray(r.dot_dist_px, float)
        nxt = np.full(len(rr), np.nan)
        nxt[:-1] = rr[1:]
        parts.append(_profiles(np.asarray(r.dtheta, float), np.asarray(r.stim_angle, float),
                               rr, np.asarray(r.moving, bool), nxt))
    return _pool(parts, records, real=True)


def _pool(parts, records, real):
    # weight bins by their bout counts so the pooled profile is bout-weighted, like the metric
    N = np.stack([p["n"] for p in parts]).astype(float)
    out = {}
    for k in ("toward", "dr", "fwd", "rmid"):
        V = np.stack([p[k] for p in parts])
        w = np.where(np.isfinite(V), N, 0.0)
        out[k] = np.where(w.sum(0) > 0, np.nansum(np.nan_to_num(V) * w, 0) / np.maximum(w.sum(0), 1), np.nan)
    out["n"] = N.sum(0).astype(int)
    return out


def gen_side(model, records, reps, seed):
    parts, radial, q90 = [], [], {}
    for r in records.values():
        for rep in range(reps):
            g = model.sample(r, np.random.default_rng([seed, int(r.fish_id), rep]))
            if "x_px" not in g:
                raise SystemExit("model.sample returned no x_px -- need a closed-loop (loop) model")
            gx, gy = np.asarray(g["x_px"], float), np.asarray(g["y_px"], float)
            gr = np.hypot(gx - np.asarray(r.dot_cx, float), gy - np.asarray(r.dot_cy, float))
            nxt = np.full(len(gr), np.nan)
            nxt[:-1] = gr[1:]
            # the bearing the model ACTUALLY reacted to (honest), not the real fish's
            phi = np.asarray(g.get("stim_angle_gen", r.stim_angle), float)
            parts.append(_profiles(np.asarray(g["dtheta"], float), phi, gr,
                                   np.asarray(r.moving, bool), nxt))
            rad = np.hypot(gx - r.dish_cx, gy - r.dish_cy) / r.dish_R
            has = np.isfinite(np.asarray(r.dot_dist_px, float))
            radial.append((rad, has))
            for nn in (1, 2, 4, 6, 8):
                m = np.asarray(r.moving, bool) & (np.asarray(getattr(r, "n_dots", np.zeros(r.n))) == nn)
                if m.sum() >= 20:
                    q90.setdefault(nn, []).append(np.nanquantile(gr[m], 0.9))
    pooled = _pool(parts, records, real=False)
    rr = np.concatenate([a for a, _ in radial])
    hh = np.concatenate([b for _, b in radial])
    pooled["radial"] = dict(frac_wall=float(np.mean(rr > 0.85)),
                            frac_wall_dot=float(np.mean(rr[hh] > 0.85)),
                            frac_wall_pause=float(np.mean(rr[~hh] > 0.85)),
                            mean=float(np.mean(rr)))
    pooled["q90"] = {k: float(np.mean(v)) for k, v in sorted(q90.items())}
    return pooled


def real_extras(records):
    rr = np.concatenate([np.hypot(np.asarray(r.x_px, float) - r.dish_cx,
                                  np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R
                         for r in records.values()])
    hh = np.concatenate([np.isfinite(np.asarray(r.dot_dist_px, float)) for r in records.values()])
    q90 = {}
    for r in records.values():
        for nn in (1, 2, 4, 6, 8):
            m = np.asarray(r.moving, bool) & (np.asarray(getattr(r, "n_dots", np.zeros(r.n))) == nn)
            if m.sum() >= 20:
                q90.setdefault(nn, []).append(np.nanquantile(np.asarray(r.dot_dist_px, float)[m], 0.9))
    return dict(radial=dict(frac_wall=float(np.mean(rr > 0.85)),
                            frac_wall_dot=float(np.mean(rr[hh] > 0.85)),
                            frac_wall_pause=float(np.mean(rr[~hh] > 0.85)),
                            mean=float(np.mean(rr))),
                q90={k: float(np.mean(v)) for k, v in sorted(q90.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=str(_DEFAULT_BOUTS))
    ap.add_argument("--model", default=None, help="path to a .pt saved by fit_MLP_mix.save_mix")
    ap.add_argument("--stim-mode", default=None, help="fit a fresh model with this stim_mode instead")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-only", action="store_true")
    args = ap.parse_args()

    records = MB.build_border_records(pd.read_csv(args.bouts))
    R = real_side(records)
    RX = real_extras(records)

    hdr = "        " + "  ".join(f"{s:>6s}" for s in R_LABELS)
    print("\n=== [A] P(turn toward shoal) by centroid range, moving-dot bouts ===")
    print(hdr)
    print("real   " + "  ".join(f"{v:6.3f}" for v in R["toward"]))

    if args.real_only:
        print(f"\ncurvature (peak - mean of ends): real {_curvature(R['toward']):+.3f}")
        print("\n=== [B] real restoring force E[dr|r] px/bout ===")
        print(hdr); print("real   " + "  ".join(f"{v:6.2f}" for v in R["dr"]))
        print(f"zero crossing: {_zero_crossing(R['dr'], R['rmid']):.1f} px")
        print("\n=== [C] real P(|dtheta|<10 deg | |phi|<30 deg) ===")
        print(hdr); print("real   " + "  ".join(f"{v:6.3f}" for v in R["fwd"]))
        print(f"\n=== [E] real radial occupancy: frac r/R>0.85 = {RX['radial']['frac_wall']:.3f} "
              f"(dot {RX['radial']['frac_wall_dot']:.3f} / pause {RX['radial']['frac_wall_pause']:.3f})")
        return

    if args.model:
        model = MX.load_mix(args.model)
        label = Path(args.model).stem
    else:
        if not args.stim_mode:
            raise SystemExit("give --model or --stim-mode")
        fids = sorted(records)
        train = [records[f] for f in fids]
        ctx = E.make_context(train, dict(mlp_epochs=args.epochs))
        model = MX.MLPMixKinDensity(K=8, stim_mode=args.stim_mode, hidden=128, use_border=True,
                                    closed_loop=True, stim_loop=True, vigor=True)
        model.fit(train, ctx)
        label = f"fresh:{args.stim_mode}:{args.epochs}ep"
        print(f"(in-sample fit -- structural/sign use only, NOT a performance number)")

    G = gen_side(model, records, args.reps, args.seed)

    print("gen    " + "  ".join(f"{v:6.3f}" for v in G["toward"]) + f"   <- {label}")
    cr, cg = _curvature(R["toward"]), _curvature(G["toward"])
    print(f"\ncurvature (peak - mean of ends): real {cr:+.3f}   gen {cg:+.3f}   "
          f"[target: gen -> real; dotb baseline was -0.003]")

    print("\n=== [B] restoring force E[dr|r], px/bout (moving-dot bouts) ===")
    print(hdr)
    print("real   " + "  ".join(f"{v:6.2f}" for v in R["dr"]))
    print("gen    " + "  ".join(f"{v:6.2f}" for v in G["dr"]))
    print(f"zero crossing: real {_zero_crossing(R['dr'], R['rmid']):.1f} px   "
          f"gen {_zero_crossing(G['dr'], G['rmid']):.1f} px   [real set point ~43 px]")
    print(f"far-field (>120 px): real {R['dr'][-1]:+.2f}   gen {G['dr'][-1]:+.2f}   "
          f"[dotb baseline -4.09 = never backs off]")

    print("\n=== [C] P(|dtheta|<10 deg | shoal frontal, |phi|<30 deg) ===")
    print(hdr)
    print("real   " + "  ".join(f"{v:6.3f}" for v in R["fwd"]))
    print("gen    " + "  ".join(f"{v:6.3f}" for v in G["fwd"]))

    print("\n=== [D] dotdist_q90 gen-real gap by group size N (must FLATTEN) ===")
    ks = sorted(set(RX["q90"]) & set(G["q90"]))
    print("   N:   " + "  ".join(f"{k:6d}" for k in ks))
    print("   gap: " + "  ".join(f"{G['q90'][k] - RX['q90'][k]:+6.1f}" for k in ks)
          + "   [dotb baseline +3.7 -> +78.6]")

    print("\n=== [E] radial occupancy (a CONTAINMENT defect, not a stimulus one) ===")
    for nm, d in (("real", RX["radial"]), ("gen ", G["radial"])):
        print(f"  {nm}: frac r/R>0.85 = {d['frac_wall']:.3f}  "
              f"(dot blocks {d['frac_wall_dot']:.3f} / pauses {d['frac_wall_pause']:.3f})  "
              f"mean r/R = {d['mean']:.3f}")
    print("  [real 0.012 vs dotb gen 0.079-0.092; if gen stays high, [D] is measuring the wall,")
    print("   not the stimulus -- fix containment before crediting a stimulus change]")


if __name__ == "__main__":
    main()
