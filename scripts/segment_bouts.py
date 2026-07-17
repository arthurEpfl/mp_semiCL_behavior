"""
Bout segmentation for the GsizeMotionSingle (semi-closed-loop visual stimulus)
dataset, adapting the bout-detection method of Le Goc et al. 2021 (BMC Biology,
"Thermal modulation of Zebrafish exploratory statistics", doi:10.1186/s12915-021-01126-w)
to this experiment's per-frame continuous tracking.

INPUT  : the per-fish `*_transformed_abs.csv` files (the `abs_dict` / `df_abs_raw`
         representation) -- continuous tracking at 30 fps with columns
         `frame, stimulus, f{N}_x, f{N}_y, f{N}_ori, d0_x..d7_stim`.
         Positions are in pixels; orientation in radians (CCW, 0 = right).

OUTPUT : a tidy per-BOUT dataframe (one row per detected swim bout) carrying the
         core kinematics (reorientation angle dtheta, interbout interval dt,
         traveled distance) plus the visual-stimulus context of each bout.

Design decisions (confirmed):
  * Output           : tidy bout dataframe + stimulus context.
  * Heading / dtheta : DISPLACEMENT-derived from the discretized bout positions
                       (onset-to-onset net travel -- each bout's heading points to
                       the next bout, i.e. the direction the fish actually goes
                       until the next bout), robust to 180-degree body-orientation
                       flips and to where in the bout the speed peaks. Measured on
                       RAW positions: Gaussian smoothing is used ONLY to detect bout
                       timing, never to distort the direction/distance we report.
  * Trajectory split : ONLY at tracking gaps. A bout's reorientation may span a
                       pause->stimulus transition, preserving the onset
                       reorientation (the fish turning toward newly-appeared
                       dots) -- the signal of this assay. Each bout is still
                       tagged by the stimulus condition active at its onset.
                       (--split-at-blocks restores splitting at every block.)

Bout onset is placed at the local speed minimum preceding each peak (the
physical bout start; onset_mode="trough", default). Le Goc's literal "80 ms
before the peak" is available as onset_mode="lead". Heading/dtheta/distance are
now onset-to-onset differences, so onset placement modestly but consistently
sets them; "trough" (two rest points) gives the cleanest net-travel baseline and
a meaningful fish state for the per-bout stimulus angle.

Le Goc parameters, reconverted from 25 fps to this dataset's 30 fps:
  * bout lead ("lead" mode) = 80 ms before the speed peak -> round(0.08*fps) frames
  * refractory  = reject bouts < 0.4 s apart      -> round(0.40 * fps) frames
  * displacement window = +/- 0.5 s around onset  -> round(0.50 * fps) frames
  * displacement gate   = reject < 0.3 mm or > 18 mm (applied in PX via PX_PER_MM=3.67)

Usage
-----
    uv run scripts/segment_bouts.py \
        --input  data/data/df_transformed \
        --roi    data/data/ROIdef.csv \
        --output data/bouts_GsizeMotionSingle.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ----------------------------- constants ------------------------------------ #
FPS = 30.0
# PIXEL-NATIVE: fish distances/positions are reported in PIXELS (the tracking
# unit). We do NOT scale per fish -- the per-dish ROIdef radius is the 99.5th
# percentile of each fish's own trajectory (trajectory-derived, not a physical
# dish measurement), so a per-fish 50/radius scale has no independent basis.
# A SINGLE global nominal scale is used only to (a) convert Le Goc's physical
# thresholds to pixels and (b) emit a nominal-mm column for interpretability.
PX_PER_MM = 3.67           # repo nominal projector scale (transformed_data_pipeline.py)
POS_SMOOTH_SIGMA = 2.0     # Gaussian trajectory smoothing (frames) -- DETECTION only

BOUT_LEAD_FR = round(0.08 * FPS)   # 80 ms before peak  -> bout onset
REFRACTORY_FR = round(0.40 * FPS)  # min frames between bouts
DISP_WIN_FR = round(0.50 * FPS)    # +/- window for displacement gate
DISP_MIN_MM = 0.3                  # Le Goc: reject sub-threshold wobble (physical)
DISP_MAX_MM = 18.0                 # Le Goc: reject tracking jumps (physical)
DISP_MIN_PX = DISP_MIN_MM * PX_PER_MM   # = 1.101 px (gate applied in pixels)
DISP_MAX_PX = DISP_MAX_MM * PX_PER_MM   # = 66.06 px
WALL_SHELL_PX = 5.0 * PX_PER_MM         # = 18.35 px (Le Goc 5 mm near-wall shell)
SMOOTH_FR = 3                      # speed smoothing window (frames)
FORWARD_THRESH_DEG = 10.0          # |dtheta| < 10 deg -> forward (Le Goc / HMM)

STIM_NONE = {"none", "inter_stim_pause", "bout", "None", None}


# --------------------------- small utilities -------------------------------- #
def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def parse_stimulus(name):
    """Parse 'CLsemi1800_g{L}-{R}_s{sL}-{sR}' into context fields.

    Returns dict with n_dots_left, n_dots_right, speed_left, speed_right,
    side ('left'/'right'/'none'), moving (bool). Non-stimulus labels -> NaNs.
    """
    out = dict(n_dots_left=np.nan, n_dots_right=np.nan,
               speed_left=np.nan, speed_right=np.nan, side="none", moving=False)
    if not isinstance(name, str) or not name.startswith("CLsemi"):
        return out
    mg = re.search(r"_g(\d+)-(\d+)", name)
    ms = re.search(r"_s([\d.]+)-([\d.]+)", name)
    if mg:
        out["n_dots_left"], out["n_dots_right"] = int(mg.group(1)), int(mg.group(2))
        if out["n_dots_left"] > 0 and out["n_dots_right"] == 0:
            out["side"] = "left"
        elif out["n_dots_right"] > 0 and out["n_dots_left"] == 0:
            out["side"] = "right"
        elif out["n_dots_left"] > 0 and out["n_dots_right"] > 0:
            out["side"] = "both"
    if ms:
        out["speed_left"], out["speed_right"] = float(ms.group(1)), float(ms.group(2))
        out["moving"] = (out["speed_left"] > 0) or (out["speed_right"] > 0)
    return out


def fish_prefix(df):
    """Find the f{N} prefix used for position columns in a per-fish CSV."""
    for c in df.columns:
        m = re.match(r"(f\d+)_x$", c)
        if m:
            return m.group(1)
    raise ValueError("No f{N}_x column found.")


def stim_angle_at(row, fx, fy, f_ori, n_stimuli=8):
    """Angle (rad, relative to fish facing) of the active-dot centroid.

    Uses body orientation f_ori (CCW) as the fish facing reference. Returns
    NaN if no active dot this frame. Positive = to the fish's left.
    """
    xs, ys = [], []
    for k in range(n_stimuli):
        sc, xc, yc = f"d{k}_stim", f"d{k}_x", f"d{k}_y"
        if sc not in row or row[sc] in STIM_NONE or pd.isna(row.get(xc)):
            continue
        xs.append(row[xc])
        ys.append(row[yc])
    if not xs:
        return np.nan
    dx = np.mean(xs) - fx
    dy = np.mean(ys) - fy
    return wrap_pi(np.arctan2(dy, dx) - f_ori)


# --------------------------- core segmentation ------------------------------ #
def fish_xy_px(seg, prefix):
    """Sub-pixel-smoothed fish position in PIXELS.

    Integer-pixel quantisation spawns spurious speed peaks, so the trajectory is
    Gaussian-smoothed for BOUT DETECTION only. Heading/distance/position are
    measured from the raw pixel positions (see segment_fish), so smoothing never
    biases the reported direction.
    """
    x = seg[f"{prefix}_x"].to_numpy(dtype=float)
    y = seg[f"{prefix}_y"].to_numpy(dtype=float)
    if POS_SMOOTH_SIGMA > 0 and len(x) > 3:
        x = gaussian_filter1d(x, POS_SMOOTH_SIGMA)
        y = gaussian_filter1d(y, POS_SMOOTH_SIGMA)
    return x, y


def _snap_to_trough(v, p):
    """Walk back from speed peak `p` to the preceding local minimum (the rest
    point from which the fish accelerates). Returns that frame index."""
    i = p
    while i > 0 and v[i - 1] <= v[i]:
        i -= 1
    return i


def detect_bouts_in_segment(x, y, speed_thresh_k, onset_mode="trough"):
    """Detect bouts within one continuous segment.

    Parameters
    ----------
    x, y : smoothed position arrays (px).
    speed_thresh_k : peak height threshold = mean(v) + k * std(v) (px/s).
    onset_mode : "trough" -> bout onset at the local speed minimum before each
                 peak (the physical bout start; default);
                 "lead"   -> Le Goc literal: BOUT_LEAD_FR frames before the peak.

    Returns two aligned integer arrays (onsets, peaks):
      * onsets  -- the bout-onset frame (timing / position / stimulus context),
                   placed per onset_mode;
      * peaks   -- the speed-peak frame, kept for the body-orientation cross-check
                   and the per-bout speed readout. Heading/dtheta/distance are
                   computed onset-to-onset (see segment_fish): the net displacement
                   between two rest points is a robust travel direction, whereas the
                   instantaneous direction at a single rest frame would be noisy.
    """
    empty = (np.array([], dtype=int), np.array([], dtype=int))
    n = len(x)
    if n < 2 * DISP_WIN_FR + 2:
        return empty

    # instantaneous speed (px/s), lightly smoothed
    step = np.hypot(np.diff(x), np.diff(y))
    v = np.concatenate([[0.0], step]) * FPS
    if SMOOTH_FR > 1:
        v = pd.Series(v).rolling(SMOOTH_FR, center=True, min_periods=1).mean().to_numpy()

    thr = np.nanmean(v) + speed_thresh_k * np.nanstd(v)
    peaks, _ = find_peaks(v, height=thr, distance=REFRACTORY_FR)

    # bout onset placement
    if onset_mode == "trough":
        onsets = np.array([_snap_to_trough(v, p) for p in peaks], dtype=int)
    else:  # "lead": fixed offset before the peak (Le Goc, 80 ms)
        onsets = np.clip(peaks - BOUT_LEAD_FR, 0, n - 1)

    # displacement gate (+/- DISP_WIN_FR around onset), keeping (onset, peak) pairs
    keep = []
    for o, p in zip(onsets, peaks):
        a = max(o - DISP_WIN_FR, 0)
        b = min(o + DISP_WIN_FR, n - 1)
        disp = np.hypot(x[b] - x[a], y[b] - y[a])
        if DISP_MIN_PX <= disp <= DISP_MAX_PX:
            keep.append((o, p))
    if not keep:
        return empty
    keep.sort()
    on = np.array([k[0] for k in keep], dtype=int)
    pk = np.array([k[1] for k in keep], dtype=int)
    _, idx = np.unique(on, return_index=True)   # drop rare duplicate onsets
    return on[idx], pk[idx]


def segment_fish(df, prefix, fish_id, speed_thresh_k, split_at_blocks=False,
                 onset_mode="trough"):
    """Segment one fish into bouts; return a list of per-bout dict records.

    Positions/distances are reported in PIXELS (no per-fish scaling).
    `onset_mode` is passed to detect_bouts_in_segment ("trough" or "lead").

    By default, trajectories are split ONLY at tracking gaps, so a bout's
    reorientation may span a pause->stimulus transition. This keeps the onset
    reorientation (fish turning toward newly-appeared dots), which is the signal
    of this assay; each bout is still tagged by the condition active at its
    onset. Set split_at_blocks=True to also break at every stimulus-block
    boundary (cleaner attribution, but drops the first reorientation per block).
    """
    df = df.reset_index(drop=True).copy()
    # continuous segments: break at tracking gaps (and optionally stim blocks)
    has_pos = df[f"{prefix}_x"].notna() & df[f"{prefix}_y"].notna()
    edges = (~has_pos) & has_pos.shift(fill_value=False)
    if split_at_blocks:
        edges = edges | df["stimulus"].ne(df["stimulus"].shift())
    df = df.assign(_seg=edges.cumsum())

    records = []
    for sid, seg in df.groupby("_seg", sort=True):
        seg = seg[has_pos.loc[seg.index]].reset_index(drop=True)
        if len(seg) < 2 * DISP_WIN_FR + 2:
            continue
        # SMOOTHED positions are used ONLY to detect bout timing (the integer-pixel
        # jitter otherwise spawns spurious speed peaks). Everything measured -- heading,
        # distance, position -- is read from the RAW positions, so smoothing never
        # distorts the direction we report.
        x, y = fish_xy_px(seg, prefix)             # smoothed -> detection only
        onsets, peaks = detect_bouts_in_segment(x, y, speed_thresh_k, onset_mode=onset_mode)
        if len(onsets) < 3:
            continue  # need neighbours to define a reorientation

        frames = seg["frame"].to_numpy()
        ori_raw = seg[f"{prefix}_ori"].to_numpy(dtype=float)
        xr = seg[f"{prefix}_x"].to_numpy(dtype=float)   # RAW px -> everything measured
        yr = seg[f"{prefix}_y"].to_numpy(dtype=float)

        # heading & distance from the ONSETS (bout starts): theta_i =
        # dir(onset_{i+1} - onset_i) is the net travel from this bout to the next,
        # i.e. the direction the fish actually goes until the next bout. Measured on
        # RAW positions, so it is robust to where in the bout the speed peaks and to
        # mid-bout body-orientation flips. dt matches (onsets).
        ox, oy = xr[onsets], yr[onsets]
        seg_dx, seg_dy = np.diff(ox), np.diff(oy)
        theta = np.arctan2(seg_dy, seg_dx)          # len = len(onsets)-1
        dist = np.hypot(seg_dx, seg_dy)             # px, onset-to-onset leg
        dt = np.diff(frames[onsets]) / FPS          # s, onset-to-onset interval

        # cross-check: body orientation at each peak (active phase),
        # flip-corrected to the travel direction
        ori_pk = ori_raw[peaks]
        mv = np.concatenate([theta, theta[-1:]])    # travel dir aligned to peaks
        flip = np.cos(ori_pk - mv) < 0
        ori_c = np.where(flip, wrap_pi(ori_pk + np.pi), ori_pk)

        # reorientation at bout i (i=1..len(onsets)-2): dtheta_i = theta_i - theta_{i-1}
        for i in range(1, len(onsets) - 1):
            o = onsets[i]
            row = seg.iloc[o]
            f_ori = float(ori_raw[o])
            dtheta = wrap_pi(theta[i] - theta[i - 1])
            dtheta_ori = wrap_pi(ori_c[i] - ori_c[i - 1])   # orientation-based check
            stim = row["stimulus"]
            ctx = parse_stimulus(stim)
            rec = dict(
                fish_id=fish_id,
                segment_id=int(sid),
                bout_index=i,
                frame_onset=int(row["frame"]),
                t_onset_s=float(row["frame"]) / FPS,
                dtheta_rad=float(dtheta),
                dtheta_deg=float(np.degrees(dtheta)),
                dtheta_ori_deg=float(np.degrees(dtheta_ori)),
                dt_s=float(dt[i]),               # interval to NEXT bout
                distance_px=float(dist[i]),      # travel to NEXT bout (px)
                distance_nominal_mm=float(dist[i] / PX_PER_MM),  # nominal only
                heading_rad=float(theta[i]),
                f_ori_rad=f_ori,
                x_px=float(xr[o]),
                y_px=float(yr[o]),
                stimulus=stim,
                stim_angle_rad=float(stim_angle_at(row, xr[o], yr[o], f_ori)),
                **ctx,
            )
            adeg = abs(rec["dtheta_deg"])
            rec["bout_type"] = ("forward" if adeg < FORWARD_THRESH_DEG
                                else ("left" if dtheta > 0 else "right"))
            records.append(rec)
    return records


def flag_near_wall(bouts, frac=0.95):
    """Non-destructive border flag: empirical dish centre per fish (median
    position) and empirical radius (frac-quantile of radial distance);
    mark bouts in the outer 5 mm shell."""
    bouts = bouts.copy()
    bouts["near_wall"] = False
    for fid, g in bouts.groupby("fish_id"):
        cx, cy = g["x_px"].median(), g["y_px"].median()
        r = np.hypot(g["x_px"] - cx, g["y_px"] - cy)
        r_dish = np.quantile(r, frac)
        bouts.loc[g.index, "near_wall"] = (r > (r_dish - WALL_SHELL_PX)).to_numpy()
    return bouts


# -------------------------------- driver ------------------------------------ #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/data/df_transformed"))
    p.add_argument("--roi", type=Path, default=Path("data/data/ROIdef.csv"),
                   help="DEPRECATED / unused: pipeline is pixel-native (no per-fish scaling).")
    p.add_argument("--output", type=Path, default=Path("data/bouts_GsizeMotionSingle.csv"))
    p.add_argument("--speed-thresh-k", type=float, default=1.0,
                   help="Peak height = mean(speed) + k*std(speed). Tune for ~1-2 Hz bout rate.")
    p.add_argument("--split-at-blocks", action="store_true",
                   help="Also break trajectories at every stimulus-block boundary "
                        "(default: split only at tracking gaps; tag bouts by onset condition).")
    p.add_argument("--onset-mode", choices=["trough", "lead"], default="trough",
                   help="Bout onset placement: 'trough' = local speed minimum before "
                        "the peak (physical bout start, default); 'lead' = Le Goc literal "
                        "80 ms before the peak.")
    args = p.parse_args()

    csvs = sorted(args.input.glob("*_transformed_abs.csv"))
    if not csvs:
        raise SystemExit(f"No *_transformed_abs.csv in {args.input}")

    # PIXEL-NATIVE: no per-fish scaling. --roi is accepted but unused (kept for
    # backward-compatible invocation); distances are reported in pixels.

    all_records = []
    for path in csvs:
        m = re.search(r"_f(\d+)_transformed_abs", path.name)
        fish_id = int(m.group(1)) if m else -1
        df = pd.read_csv(path)
        prefix = fish_prefix(df)
        recs = segment_fish(df, prefix, fish_id, args.speed_thresh_k,
                            split_at_blocks=args.split_at_blocks,
                            onset_mode=args.onset_mode)
        all_records.extend(recs)
        print(f"  {path.name}: {len(recs)} bouts (fish {fish_id})")

    bouts = pd.DataFrame(all_records)
    if bouts.empty:
        raise SystemExit("No bouts detected -- lower --speed-thresh-k.")
    bouts = flag_near_wall(bouts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bouts.to_csv(args.output, index=False)

    # ---------------------------- diagnostics ------------------------------- #
    n = len(bouts)
    interior = bouts[~bouts["near_wall"]]
    rate = 1.0 / bouts["dt_s"].median()
    ftype = bouts["bout_type"].value_counts(normalize=True)
    print("\n================ SEGMENTATION SUMMARY ================")
    print(f"total bouts            : {n}  ({n - len(interior)} flagged near_wall)")
    print(f"fish                   : {bouts['fish_id'].nunique()}")
    print(f"median interbout dt    : {bouts['dt_s'].median():.3f} s  (~{rate:.2f} Hz)")
    print(f"median distance        : {bouts['distance_px'].median():.2f} px "
          f"({bouts['distance_nominal_mm'].median():.2f} nominal mm)")
    print(f"bout-type fractions    : " +
          ", ".join(f"{k}={v:.2f}" for k, v in ftype.items()))
    print(f"|dtheta| median        : {bouts['dtheta_deg'].abs().median():.1f} deg "
          f"(orientation-check: {bouts['dtheta_ori_deg'].abs().median():.1f} deg)")
    fwd_disp = (bouts['dtheta_deg'].abs() < FORWARD_THRESH_DEG).mean()
    fwd_ori = (bouts['dtheta_ori_deg'].abs() < FORWARD_THRESH_DEG).mean()
    print(f"forward fraction       : displacement={fwd_disp:.2f}  orientation={fwd_ori:.2f}")
    print(f"bouts during stimulus  : {(bouts['side'] != 'none').sum()}")
    print(f"saved -> {args.output}")
    print("Sanity targets (Le Goc): bout rate ~1-2 Hz; dtheta histogram trimodal,")
    print("forward/turn crossing near +/-10 deg. Tune --speed-thresh-k if rate is off.")


if __name__ == "__main__":
    main()
