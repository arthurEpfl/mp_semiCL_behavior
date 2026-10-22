"""From the 30 fps tables to the training table ``bouts_clean.csv`` (notebooks/01_bout_discretisation.ipynb).

1. ``repair_frames``: tracking excursions -- the tracker loses the fish for 1-5 frames and reports it elsewhere
   (generally at the dish edge), which the speed smoothing turns into impossibly fast bouts. A step over ``JUMP_PX``
   (100 mm/s in one frame) opens an excursion that closes at the first frame back within ``JUMP_PX`` of the position
   before the jump; only runs of at most ``MAX_RUN`` frames are repaired, by linear interpolation.
2. ``segment``: bout detection on the repaired trajectories (``segment_bouts.segment_fish``), the wall and dot geometry.
3. ``flag_gaps``: a bout whose interval exceeds 10 s spans a quiescent gap; its step and turn are measured across the
   gap, not swum. ``active`` = neither this leg nor the previous one spans a gap: the bouts every model is fitted on.
4. ``add_model_columns``: the block clocks, the run structure (runs = stretches between gaps), the lags within a run,
   the dot and wall bearings in the travel (chord) frame.
5. ``add_dot_positions``: the position of every dot at each bout onset, from the 30 fps tables.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from twin import paths
from twin import segment_bouts as SB
from twin.stim_program import STIM_NONE

PX_PER_MM_PHYS = 3.60               # physical scale from the raw ROIdef (100 mm dish)
FPS = SB.FPS
QUIET_S = 10.0                      # a "no-bout stretch" starts here
JUMP_PX = 100.0 * PX_PER_MM_PHYS / FPS      # 12 px/frame = 100 mm/s
MAX_RUN = 5                                  # longest excursion treated as a tracking loss
CYCLE_FR, PAUSE_FR, STIM_FR = 3300, 1500, 1800     # 110 s: 50 s pause + 60 s stimulus
K_DOTS = 8


def find_excursions(x, y, jump_px=JUMP_PX, max_run=MAX_RUN):
    """Boolean mask of the frames where the tracker left and came back (runs of 1-``max_run`` frames)."""
    n = len(x)
    step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    bad = np.zeros(n, bool)
    i = 1
    while i < n:
        if step[i] > jump_px:
            hx, hy = x[i - 1], y[i - 1]                 # where it was before the leap
            j, stop = i, min(i + max_run + 1, n)
            while j < stop and np.hypot(x[j] - hx, y[j] - hy) > jump_px:
                j += 1
            if j < stop:                                # came back -> tracking loss
                bad[i:j] = True
                i = j
                continue
        i += 1
    return bad


def repair(x, y, mask):
    """Linear interpolation across the flagged frames (a short bridge between two trusted positions)."""
    xs, ys = x.astype(float).copy(), y.astype(float).copy()
    xs[mask], ys[mask] = np.nan, np.nan
    xs = pd.Series(xs).interpolate(limit_direction="both").to_numpy()
    ys = pd.Series(ys).interpolate(limit_direction="both").to_numpy()
    return xs, ys


def repair_frames(frames_dir=paths.FRAMES):
    """[(fish id, repaired 30 fps table)], the per-fish report and the list of repaired frames."""
    report, frames_out, repaired_rows = [], [], []
    for path in sorted(Path(frames_dir).glob("*_transformed_abs.csv")):
        fid = int(path.stem.split("_f")[-1].split("_")[0])
        df = pd.read_csv(path)
        px, py = f"f{fid}_x", f"f{fid}_y"
        ok = df[px].notna() & df[py].notna()
        x = df.loc[ok, px].to_numpy(float)
        y = df.loc[ok, py].to_numpy(float)
        bad = find_excursions(x, y)
        s0 = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
        xs, ys = repair(x, y, bad)
        s1 = np.r_[0.0, np.hypot(np.diff(xs), np.diff(ys))]
        df.loc[ok, px], df.loc[ok, py] = xs, ys
        frames_out.append((fid, df))
        fr = df.loc[ok, "frame"].to_numpy()[bad]
        report.append(dict(fish=fid, frames=len(x), repaired=int(bad.sum()), pct=round(100 * bad.mean(), 4),
                           max_step_before=round(float(s0.max()), 1), max_step_after=round(float(s1.max()), 1),
                           over_100mm_s_before=int((s0 > JUMP_PX).sum()),
                           over_100mm_s_after=int((s1 > JUMP_PX).sum())))
        if len(fr):
            repaired_rows.append(pd.DataFrame(dict(fish_id=fid, frame=fr)))
    rep = pd.DataFrame(report).set_index("fish")
    rf = (pd.concat(repaired_rows, ignore_index=True) if repaired_rows
          else pd.DataFrame(columns=["fish_id", "frame"]))
    return frames_out, rep, rf


def segment(frames, frames_dir=paths.FRAMES):
    """Bouts of every fish from (repaired) 30 fps tables, with the wall and active-dot geometry."""
    recs = []
    for fid, df in frames:
        recs += SB.segment_fish(df, SB.fish_prefix(df), fid, speed_thresh_k=1.0, onset_mode="trough")
    bouts = SB.flag_near_wall(pd.DataFrame(recs))
    bouts = SB.add_dist_to_border(bouts, SB.load_physical_dish(paths.ROI_PHYS))
    bouts = SB.add_dist_to_dot(bouts, transformed_dir=frames_dir)
    bouts["v_mm_s"] = bouts["distance_px"] / PX_PER_MM_PHYS / bouts["dt_s"]     # bout mean speed (a check, unused)
    return bouts


def flag_gaps(bouts):
    bouts = bouts.sort_values(["fish_id", "bout_index"]).reset_index(drop=True)
    bouts["gap_after"] = bouts["dt_s"] > QUIET_S
    bouts["gap_before"] = bouts.groupby("fish_id")["gap_after"].shift(1).fillna(False)
    bouts["active"] = ~(bouts["gap_after"] | bouts["gap_before"])
    return bouts


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def add_model_columns(bouts):
    """The columns a bout-to-bout model needs that the segmenter does not produce."""
    pos = bouts["frame_onset"] % CYCLE_FR
    in_stim = pos >= PAUSE_FR
    st = bouts["stimulus"].astype(str).str.startswith("CLsemi")
    # block clocks: a pause block is 50 s and a stimulus block 60 s, so each has its own 0-1 clock
    bouts["blk"] = bouts["frame_onset"] // CYCLE_FR * 2 + in_stim.astype(int)
    bouts["t_blk"] = np.where(in_stim, pos - PAUSE_FR, pos) / FPS
    bouts["cond"] = np.where(~st, "none", np.where(bouts["moving"].astype(bool), "moving", "static"))
    bouts["t_pause_norm"] = np.where(in_stim, 0.0, pos / PAUSE_FR)
    bouts["t_stim_norm"] = np.where(in_stim, (pos - PAUSE_FR) / STIM_FR, 0.0)
    bouts["dots_on"] = st.to_numpy()
    bouts["r_norm"] = 1.0 - bouts["dist_to_border_norm"]            # radial position, 0 = centre, 1 = wall
    # runs: the stretches between quiescent gaps; lags never reach across one
    bouts["run_id"] = (bouts["gap_before"] | (bouts["fish_id"] != bouts["fish_id"].shift())).cumsum()
    bouts["run_n_active"] = bouts["run_id"].map(bouts.groupby("run_id")["active"].sum()).astype(int)
    bouts["run_pos"] = bouts.groupby("run_id").cumcount()
    g = bouts.groupby("run_id")
    for c in ("dtheta_deg", "dt_s", "distance_px"):
        bouts[f"{c}_prev"] = g[c].shift(1)
    # the dot and dish-centre bearings in the travel (chord) frame of the previous step, the frame the closed loop
    # presents at rollout: theta_{t-1} = heading_rad[t] - dtheta_t
    theta_prev = _wrap(bouts["heading_rad"] - np.radians(bouts["dtheta_deg"]))
    bouts["stim_angle_travel"] = _wrap(np.arctan2(bouts["dot_cy"] - bouts["y_px"],
                                                  bouts["dot_cx"] - bouts["x_px"]) - theta_prev)
    dish = SB.load_physical_dish(paths.ROI_PHYS)
    psi = _wrap(np.arctan2(bouts["fish_id"].map(dish["cy"]) - bouts["y_px"],
                           bouts["fish_id"].map(dish["cx"]) - bouts["x_px"]) - theta_prev)
    bouts["wall_psi_rad"] = psi
    bouts["wall_sin_psi"] = np.sin(psi)
    bouts["wall_cos_psi"] = np.cos(psi)
    bouts["crosses_block"] = (pos + bouts["dt_s"] * FPS) > np.where(in_stim, CYCLE_FR, PAUSE_FR)
    return bouts


def add_dot_positions(bouts, frames_dir=paths.FRAMES):
    """``d{k}_x, d{k}_y`` for the 8 dot slots at each onset (NaN where the slot is off), from the 30 fps tables; their
    centroid equals the stored ``dot_cx, dot_cy``."""
    df = bouts.copy()
    for k in range(K_DOTS):
        df[f"d{k}_x"] = np.nan
        df[f"d{k}_y"] = np.nan
    for fid, g in df.groupby("fish_id"):
        src = sorted(Path(frames_dir).glob(f"*_f{int(fid)}_transformed_abs.csv"))[0]
        use = [f"d{k}_{v}" for k in range(K_DOTS) for v in ("x", "y")] + [f"d{k}_stim" for k in range(K_DOTS)]
        t = pd.read_csv(src, usecols=use)
        fr = np.clip(g["frame_onset"].to_numpy(int), 0, len(t) - 1)
        for k in range(K_DOTS):
            on = (~t[f"d{k}_stim"].astype(str).isin(list(STIM_NONE)).to_numpy()
                  & np.isfinite(t[f"d{k}_x"].to_numpy(float)))
            df.loc[g.index, f"d{k}_x"] = np.where(on, t[f"d{k}_x"].to_numpy(float), np.nan)[fr]
            df.loc[g.index, f"d{k}_y"] = np.where(on, t[f"d{k}_y"].to_numpy(float), np.nan)[fr]
    return df


def build(frames_dir=paths.FRAMES):
    """The training table and the repaired-frame list, from the 30 fps tables."""
    frames, report, repaired = repair_frames(frames_dir)
    bouts = add_model_columns(flag_gaps(segment(frames, frames_dir)))
    return add_dot_positions(bouts, frames_dir), repaired, report
