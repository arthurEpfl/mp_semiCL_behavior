"""A small, shippable summary of the REAL fish data, so the rollout tool runs without it.

``notebooks/stimulus_tool.ipynb`` uses the recorded experiment for two things that have nothing to
do with generating trajectories: drawing a plausible start pose, and drawing the real comparison
curves in the sanity panel. Both only need aggregates, but the file they came from
(``data/bouts_GsizeMotionSingle.csv``, 70 590 bouts with fish identity and timing) is real animal
data and is not published.

This module reduces that to ~100 kB of HISTOGRAMS -- no fish identity, no time ordering, no
trajectories, nothing per-individual:

  * ``pose_hist``  3-D count histogram over (u, v, heading) in NORMALISED dish coordinates.
    Joint rather than three marginals, because position and heading are coupled in the real data
    (fish near the wall face inward) and independent marginals would start fish facing the wall.
  * ``curves``     following index by group size, and binned densities for the three emission
    heads (|dtheta|, distance, dt) plus the r/R the trajectory ends up with.

``build_reference`` is run ONCE, locally, by whoever holds the bouts CSV; ``load_reference`` is what
the notebook calls. See the Data section of the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Binning of the shipped curves. Fixed here (not a parameter) so a reference file built on one
# machine is directly comparable with a panel drawn on another.
TURN_DEG = 10.0                       # |dtheta| below this is a forward bout, not a turn
HIST_BINS = 60
DTHETA_RANGE = (0.0, 180.0)
DT_RANGE = (0.0, 6.0)
RR_RANGE = (0.0, 1.0)
# Distance is the THIRD emission head and the only one the tool did not plot. Range stops at 60 px
# (real q99 = 44 px, q99.9 = 85, max 219): the head is heavy-tailed by design, so plotting to the
# max would put 99% of the mass in the first two bins and hide the bulk, which is where the
# kin_feedback shrinkage and the dispersion problems actually show.
DIST_RANGE = (0.0, 60.0)
POSE_BINS = (32, 32, 16)              # u, v, heading -> 16384 cells, overwhelmingly zero


def following_by_n(dtheta, phi, n_dots, moving, turn_deg=TURN_DEG, min_bouts=50):
    """P(turn toward the stimulus) per group size, on moving-stimulus turning bouts.

    ``dtheta`` (deg) and ``phi`` (rad) are both signed in the same transformed frame, so a sign
    match means the fish turned toward the shoal. Shared by the real and generated sides so the two
    cannot drift apart -- the notebook imports this rather than redefining it.
    """
    d = np.asarray(dtheta, float)
    p = np.asarray(phi, float)
    n = np.asarray(n_dots, float)
    ok = np.isfinite(p) & np.isfinite(d) & (np.abs(d) > turn_deg) & (np.asarray(moving, float) > 0)
    out = {}
    for k in sorted(set(n[ok].astype(int))):
        m = ok & (n == k)
        if m.sum() > min_bouts:
            out[int(k)] = float(np.mean(np.sign(d[m]) == np.sign(p[m])))
    return out


@dataclass
class ReferenceStats:
    """Loaded summary. ``curves`` holds everything the sanity panel draws."""
    pose_hist: np.ndarray             # (nu, nv, nh) counts
    u_edges: np.ndarray
    v_edges: np.ndarray
    h_edges: np.ndarray
    curves: dict
    n_bouts: int = 0

    # -- start poses -------------------------------------------------------- #
    def sample_start_pose(self, rng, dish):
        """Draw (x, y, ori) for a fresh fish: pick a histogram cell with probability proportional
        to its count, then jitter uniformly inside it, then map into ``dish``."""
        H = self.pose_hist
        flat = H.ravel().astype(float)
        idx = int(rng.choice(flat.size, p=flat / flat.sum()))
        i, j, k = np.unravel_index(idx, H.shape)
        u = float(rng.uniform(self.u_edges[i], self.u_edges[i + 1]))
        v = float(rng.uniform(self.v_edges[j], self.v_edges[j + 1]))
        ori = float(rng.uniform(self.h_edges[k], self.h_edges[k + 1]))
        cx, cy, R = dish
        # A corner cell can sit outside the unit disc; rejectionless fix = clamp the radius.
        r = np.hypot(u, v)
        if r > 1.0:
            u, v = u / r, v / r
        return cx + u * R, cy + v * R, ori

    # -- reporting ---------------------------------------------------------- #
    def summary(self) -> str:
        c = self.curves
        fi = c["following_by_n"]
        return (f"real reference: {self.n_bouts} bouts summarised, "
                f"{int(self.pose_hist.sum())} poses in {self.pose_hist.size} cells\n"
                f"  following index by N: {({k: round(v, 3) for k, v in fi.items()})}\n"
                f"  mean r/R {c['mean_rR']:.3f} | median |dtheta| {c['median_abs_dtheta']:.1f} deg "
                f"| median dt {c['median_dt']:.2f} s")


# --------------------------------------------------------------------------- #
# Build (run once, locally, by whoever holds the bouts CSV)
# --------------------------------------------------------------------------- #
def build_reference(records, out_path):
    """Reduce ``build_border_records`` output to the shippable summary and save it."""
    recs = list(records.values() if isinstance(records, dict) else records)

    u_all, v_all, h_all = [], [], []
    d_all, p_all, n_all, m_all, dt_all, rr_all = [], [], [], [], [], []
    di_all = []
    for r in recs:
        u = (np.asarray(r.x_px, float) - r.dish_cx) / r.dish_R
        v = (np.asarray(r.y_px, float) - r.dish_cy) / r.dish_R
        u_all.append(u); v_all.append(v)
        h_all.append(np.asarray(r.f_ori_rad, float))
        d_all.append(np.asarray(r.dtheta, float))
        p_all.append(np.asarray(r.stim_angle, float))
        n_all.append(np.asarray(r.n_dots, float))
        m_all.append(np.asarray(r.moving, float))
        dt_all.append(np.asarray(r.dt_s, float))
        di_all.append(np.asarray(r.distance_px, float))
        rr_all.append(np.hypot(u, v))
    u_all, v_all = np.concatenate(u_all), np.concatenate(v_all)
    h_all = np.concatenate(h_all)
    d_all, p_all = np.concatenate(d_all), np.concatenate(p_all)
    n_all, m_all = np.concatenate(n_all), np.concatenate(m_all)
    dt_all, rr_all = np.concatenate(dt_all), np.concatenate(rr_all)
    di_all = np.concatenate(di_all)

    ok = np.isfinite(u_all) & np.isfinite(v_all) & np.isfinite(h_all)
    hw = (np.asarray(h_all) + np.pi) % (2 * np.pi) - np.pi          # wrap to (-pi, pi]
    H, edges = np.histogramdd(
        np.stack([u_all[ok], v_all[ok], hw[ok]], axis=1), bins=POSE_BINS,
        range=[(-1.0, 1.0), (-1.0, 1.0), (-np.pi, np.pi)])

    def dens(x, rng_):
        c, e = np.histogram(x[np.isfinite(x)], bins=HIST_BINS, range=rng_, density=True)
        return c, e

    ad_c, ad_e = dens(np.abs(d_all), DTHETA_RANGE)
    dt_c, dt_e = dens(dt_all, DT_RANGE)
    rr_c, rr_e = dens(rr_all, RR_RANGE)
    di_c, di_e = dens(np.abs(di_all), DIST_RANGE)
    fi = following_by_n(d_all, p_all, n_all, m_all)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pose_hist=H.astype(np.uint32),
        u_edges=edges[0], v_edges=edges[1], h_edges=edges[2],
        fi_n=np.array(sorted(fi), dtype=np.int32),
        fi_v=np.array([fi[k] for k in sorted(fi)], dtype=np.float64),
        abs_dtheta_density=ad_c, abs_dtheta_edges=ad_e,
        dt_density=dt_c, dt_edges=dt_e,
        rr_density=rr_c, rr_edges=rr_e,
        dist_density=di_c, dist_edges=di_e,
        mean_rR=float(np.nanmean(rr_all)),
        median_abs_dtheta=float(np.nanmedian(np.abs(d_all))),
        median_dt=float(np.nanmedian(dt_all)),
        median_dist=float(np.nanmedian(di_all)),
        n_bouts=int(d_all.size))
    return out_path


def load_reference(path):
    z = np.load(Path(path), allow_pickle=False)
    curves = dict(
        following_by_n={int(k): float(v) for k, v in zip(z["fi_n"], z["fi_v"])},
        abs_dtheta=(z["abs_dtheta_density"], z["abs_dtheta_edges"]),
        dt=(z["dt_density"], z["dt_edges"]),
        rr=(z["rr_density"], z["rr_edges"]),
        # OPTIONAL, so a reference built before the distance head was plotted still loads; the
        # notebook skips the panel when it is absent rather than failing at import.
        **({"dist": (z["dist_density"], z["dist_edges"])} if "dist_density" in z.files else {}),
        **({"median_dist": float(z["median_dist"])} if "median_dist" in z.files else {}),
        mean_rR=float(z["mean_rR"]),
        median_abs_dtheta=float(z["median_abs_dtheta"]),
        median_dt=float(z["median_dt"]))
    return ReferenceStats(pose_hist=z["pose_hist"], u_edges=z["u_edges"], v_edges=z["v_edges"],
                          h_edges=z["h_edges"], curves=curves, n_bouts=int(z["n_bouts"]))


def reference_from_records(records):
    """The same summary computed in-memory, for contributors who DO have the bouts CSV -- so the
    notebook draws identical curves either way instead of having two code paths that can disagree."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = build_reference(records, Path(td) / "ref.npz")
        return load_reference(p)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evaluate as E
    import fit_MLP_border as MB

    _ROOT = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bouts", default=str(_ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--out", default=str(_ROOT / "assets" / "real_reference.npz"))
    a = ap.parse_args()

    recs = MB.build_border_records(E.load_bouts(a.bouts))
    p = build_reference(recs, a.out)
    ref = load_reference(p)
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} kB)")
    print(ref.summary())
