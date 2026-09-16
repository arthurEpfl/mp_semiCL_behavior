"""Bout-level dataframes for real fish and simulated rollouts, in one shared schema,
kept as separate per-individual dicts so real and generated data never need to be
concatenated into a combined dataframe to be compared.

Used by ``notebooks/rollout_analyses.ipynb`` for population-level bout analyses
(occupancy, position-after-appearance, distance-to-dots, LR time ratio) run on
both real fish (segmented down from the raw per-frame tracking CSVs) and
simulated rollouts (already bout-native).

Real -> bouts reuses ``segment_bouts.segment_fish``/``add_dist_to_dot`` -- the
same pipeline ``scripts/tool_reference.py`` uses to build ``assets/real_reference.npz``,
just stopped one step earlier (per-bout records instead of a final histogram).
Generated -> bouts needs no segmentation at all: ``stim_rollout.rollout_many`` already
returns bout-onset arrays, so ``rollout_bout_dict`` only reshapes them.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from segment_bouts import add_dist_to_dot, fish_prefix, flag_near_wall, parse_stimulus, segment_fish


def real_bout_dict(data_dir, speed_thresh_k=1.0):
    """Build one bout dataframe per real fish.

    Reads every ``*_f{N}_transformed_abs.csv`` file in ``data_dir``, segments each
    fish's raw per-frame trajectory into bouts, and enriches each bout with its
    distance to the active dot and a near-wall flag. No cross-fish concatenation --
    each fish keeps its own dataframe.

    Args:
        data_dir (str or pathlib.Path): Directory containing the per-fish
            ``*_f{N}_transformed_abs.csv`` files.
        speed_thresh_k (float): Speed-threshold multiplier passed through to
            ``segment_bouts.segment_fish`` for bout-onset detection. Defaults to
            1.0.

    Returns:
        dict: Mapping from ``"f{fish_id}"`` (str) to that fish's bout dataframe
        (pandas.DataFrame).

    Raises:
        FileNotFoundError: If ``data_dir`` contains no matching CSV files.
    """
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("*_transformed_abs.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_transformed_abs.csv files in {data_dir}")

    real_bouts = {}
    for path in csvs:
        m = re.search(r"_f(\d+)_transformed_abs", path.name)
        if not m:
            continue
        fish_id = int(m.group(1))
        df = pd.read_csv(path)
        prefix = fish_prefix(df)
        recs = segment_fish(df, prefix, fish_id, speed_thresh_k)
        if not recs:
            continue
        fish_df = pd.DataFrame(recs)
        fish_df = add_dist_to_dot(fish_df, transformed_dir=data_dir)   # dot_cx/dot_cy/dot_dist_px
        fish_df = flag_near_wall(fish_df)                              # cheap, kept for parity
        real_bouts[f"f{fish_id}"] = fish_df
    return real_bouts


def rollout_bout_dict(rollouts):
    """Build one bout dataframe per simulated rollout.

    Rollouts are already bout-indexed (``stim_rollout.rollout_many`` output) -- this
    only reshapes them, no segmentation needed.

    Also carries a per-bout ``f_ori_rad``: the model's own latched CLsemi anchor
    heading for that bout's block (``g["anchors"][block_id][2]``) -- the TRUE fish
    heading at the instant the block was entered, independent of anything the fish
    did afterwards. ``enrich_bout_df`` uses this (not a heading derived from the
    rollout's own subsequent bout-to-bout displacement, which would bias the
    rotated frame toward whatever direction the fish happened to move next).

    Args:
        rollouts (list): Rollout records as produced by ``stim_rollout.rollout_many``.

    Returns:
        dict: Mapping from rollout index (int) to that rollout's bout dataframe
        (pandas.DataFrame).
    """
    gen_bouts = {}
    for k, g in enumerate(rollouts):
        block_id = np.asarray(g["block_id"], dtype=int)
        anchors = g.get("anchors", {})
        fallback_ori = float(g["start_pose"][2])
        f_ori_rad = np.array(
            [anchors[b][2] if b in anchors else fallback_ori for b in block_id], dtype=float)
        gen_bouts[k] = pd.DataFrame({
            "fish_id": k,
            "frame_onset": np.asarray(g["frame_onset"], dtype=int),
            "t_onset_s": np.asarray(g["t_onset_s"], dtype=float),
            "x_px": np.asarray(g["x_px"], dtype=float),
            "y_px": np.asarray(g["y_px"], dtype=float),
            "f_ori_rad": f_ori_rad,
            "dtheta_deg": np.asarray(g["dtheta"], dtype=float),
            "dt_s": np.asarray(g["dt_s"], dtype=float),
            "dot_cx": np.asarray(g["dot_cx"], dtype=float),
            "dot_cy": np.asarray(g["dot_cy"], dtype=float),
            "stim_angle_rad": np.asarray(g["stim_angle_gen"], dtype=float),
        })
    return gen_bouts


def _block_speeds(program):
    """Compute the actual stimulus speed per block.

    Read from each block's own raw label -- not a hardcoded number -- so labelling
    adapts to any programme (different dot counts, different speeds), not just the
    shipped 3.5-mm/s one.

    Args:
        program: A ``stim_program.StimulusProgram``.

    Returns:
        numpy.ndarray: Speed in mm/s per block, shape ``(program.n_blocks,)``.
    """
    speeds = np.zeros(program.n_blocks)
    for b, lbl in enumerate(program.labels):
        ctx = parse_stimulus(lbl)
        vals = [v for v in (ctx["speed_left"], ctx["speed_right"]) if np.isfinite(v) and v > 0]
        speeds[b] = max(vals) if vals else 0.0
    return speeds


def _rotate(px, py, ax, ay, aori, px_to_mm):
    """Translate to the anchor and rotate into the fish-relative frame.

    Left = +x, matching ``stim_program.anchor_offsets``'s convention exactly (not
    just its style) -- ``anchor_offsets`` maps ``(dx, dy) -> (ox, oy)`` via a matrix
    that is its own inverse, so this is that same matrix applied to
    ``(px - ax, py - ay)``.

    Args:
        px (array-like): X position(s) to rotate, in pixels.
        py (array-like): Y position(s) to rotate, in pixels.
        ax (array-like): Anchor X position(s), in pixels.
        ay (array-like): Anchor Y position(s), in pixels.
        aori (array-like): Anchor orientation(s), in radians.
        px_to_mm (float): Pixels-per-millimetre scale factor.

    Returns:
        tuple: ``(x_rot, y_rot)`` -- rotated position, in millimetres.
    """
    ox, oy = px - ax, py - ay
    s, c = np.sin(aori), np.cos(aori)
    return (-ox * s + oy * c) / px_to_mm, (ox * c + oy * s) / px_to_mm


def enrich_bout_df(fish_df, program, framerate=30.0, px_to_mm=3.67):
    """Add block/condition/trial-relative-time/rotated-position/distance columns.

    Enriches ONE individual's (real fish or simulated rollout) bout dataframe. Call
    once per dict value from ``real_bout_dict``/``rollout_bout_dict`` -- real and
    generated never need to occupy the same dataframe to go through this
    identically.

    Args:
        fish_df (pandas.DataFrame): One individual's bout dataframe, as produced by
            ``real_bout_dict`` or ``rollout_bout_dict``.
        program: A ``stim_program.StimulusProgram`` describing the stimulus
            programme this fish/rollout was run against.
        framerate (float): Frames per second. Defaults to 30.0.
        px_to_mm (float): Pixels-per-millimetre scale factor. Defaults to 3.67.

    Returns:
        pandas.DataFrame: ``fish_df``, sorted by bout onset and enriched with
        ``block_id``, ``n_dots``, ``moving``, ``is_stim``, ``stimulus``,
        ``t_since_onset_s``, ``anchor_x``/``anchor_y``/``anchor_ori``, ``x_rot``/
        ``y_rot``, ``dot_cx_rot``/``dot_cy_rot``, ``dist_mm``, and ``stim_side``.
    """
    df = fish_df.sort_values("t_onset_s").reset_index(drop=True)
    n_frames = program.n_frames

    # 1. block id -- recomputed uniformly here so real and generated use exactly the same
    #    lookup, rather than trusting two code paths (segment_fish's own `stimulus` string vs
    #    the rollout's own g["block_id"]) to agree.
    frame_onset = df["frame_onset"].to_numpy(int)
    block_id = program.block_id[np.clip(frame_onset, 0, n_frames - 1)]
    df["block_id"] = block_id

    # 2. condition from the block (vectorized lookups)
    n_dots = program.n_dots_of[block_id]
    moving = program.moving_of[block_id]
    is_stim = program.is_stim_of[block_id]
    df["n_dots"] = n_dots
    df["moving"] = moving
    df["is_stim"] = is_stim

    # 3. stimulus label: "d{n}s{speed}" (matches AllFish_GsizeMotionSingle.ipynb), or pause.
    #    Speed comes from _block_speeds (the block's own raw label), not a hardcoded "3.5".
    speed = _block_speeds(program)[block_id]
    speed_str = np.array([f"{s:g}" for s in speed])
    df["stimulus"] = np.where(
        is_stim, "d" + n_dots.astype(int).astype(str) + "s" + speed_str, "inter_stim_pause")

    # 4. time since this block started -- bout-level stand-in for frame_count_stimulus/framerate
    block_start_s = program.starts[block_id] / framerate
    df["t_since_onset_s"] = df["t_onset_s"].to_numpy(float) - block_start_s

    # 5. anchor pose per block_id: first bout's position, and its OWN independently-known
    #    heading (f_ori_rad -- real: the fish's measured body orientation at that bout's onset;
    #    generated: the model's latched CLsemi anchor heading for the block). Deliberately NOT
    #    derived from this bout's displacement to the next bout, which would bias the rotated
    #    frame toward whatever direction the fish moved next -- a mechanical "always forward"
    #    artifact, independent of any real stimulus response, that showed up as a same-direction
    #    offset in both L and R panels of analysis 2 before this fix.
    ax = np.full(len(df), np.nan)
    ay = np.full(len(df), np.nan)
    aori = np.zeros(len(df))
    for _, idx in df.groupby("block_id").groups.items():
        idx = np.asarray(idx)
        x0, y0 = df.loc[idx[0], "x_px"], df.loc[idx[0], "y_px"]
        heading = df.loc[idx[0], "f_ori_rad"]
        heading = float(heading) if np.isfinite(heading) else 0.0
        ax[idx], ay[idx], aori[idx] = x0, y0, heading
    df["anchor_x"], df["anchor_y"], df["anchor_ori"] = ax, ay, aori

    # 6. rotate fish position and dot centroid into trial-heading-aligned mm coordinates
    df["x_rot"], df["y_rot"] = _rotate(
        df["x_px"].to_numpy(float), df["y_px"].to_numpy(float), ax, ay, aori, px_to_mm)
    df["dot_cx_rot"], df["dot_cy_rot"] = _rotate(
        df["dot_cx"].to_numpy(float), df["dot_cy"].to_numpy(float), ax, ay, aori, px_to_mm)

    # 7. distance from the fish to the active-dot centroid (NaN where no dot is active)
    df["dist_mm"] = np.hypot(df["x_px"] - df["dot_cx"], df["y_px"] - df["dot_cy"]) / px_to_mm

    # 8. stimulus side, from the rotated dot centroid (only meaningful when is_stim/dot_cx finite)
    df["stim_side"] = np.where(df["dot_cx_rot"] < 0, "L", "R")

    return df
