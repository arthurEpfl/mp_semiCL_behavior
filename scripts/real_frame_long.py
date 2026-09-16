"""Frame-level ``fish_df_long`` for real fish, rebuilt from the per-fish CSVs
``AllFish_GsizeMotionSingle.ipynb``'s "export a df for each fish" cell writes.

Replays that notebook's post-export cells ("Divide dataframes" -> "Calculate dots
center"/"...out of rois" -> "Rotate fish and stimuli coordinates..." -> "Transform to
long format" -> "add stimuli centers columns" -> "Compute distance with dots center" ->
speed) so the result is the same ``fish_df_long`` used there -- validated column-for-column
against a ground-truth export (max diff ~1e-3, at the rounding boundary).

Only needs the columns every export version guarantees -- ``frame``/``stimulus``/``x``/
``y``/``theta``/``d{i}_x``/``d{i}_y``/``d{i}_stim`` (also what ``src/transformed_data_pipeline.py``
writes) -- plus an ROI file; everything else (trial/block ids, dots_center, dots_out) is
recomputed here rather than required already in the CSV.

Caveat: trial_id/stimulus_block_id are exact (they only depend on condition *transitions*,
unaffected by dropped rows), but if a fish's export ever dropped leading rows of its first
stimulus block, that block's frame_count_stimulus would be off by a constant offset.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import analyse as an

_NAME_RE = re.compile(r'^(?P<date>.+)_(?P<experiment>[^_]+)_f(?P<fish>\d+)_transformed_abs$')
_ID_COLS = ['frame', 'trial_id', 'condition', 'stimulus_block_id', 'frame_count_stimulus', 'trial_start']


def _load_fish_raw(data_dir):
    """Load every fish's raw per-frame CSV in a data directory.

    Args:
        data_dir (str or pathlib.Path): Directory containing the per-fish
            ``*_transformed_abs.csv`` files.

    Returns:
        tuple: ``(fish_raw, experiment)`` where ``fish_raw`` (dict[int,
        pandas.DataFrame]) maps fish id to its raw dataframe, and ``experiment``
        (str) is the experiment name parsed from the filenames.

    Raises:
        FileNotFoundError: If ``data_dir`` contains no matching CSV files.
        ValueError: If a file name doesn't match the expected
            ``{date}_{experiment}_f{fish}_transformed_abs.csv`` pattern.
    """
    csvs = sorted(Path(data_dir).glob('*_transformed_abs.csv'))
    if not csvs:
        raise FileNotFoundError(f'no *_transformed_abs.csv files in {data_dir}')
    fish_raw, experiment = {}, None
    for p in csvs:
        m = _NAME_RE.match(p.stem)
        if not m:
            raise ValueError(f'unexpected file name: {p.name}')
        experiment = experiment or m.group('experiment')
        fish_raw[int(m.group('fish'))] = pd.read_csv(p).sort_values('frame').reset_index(drop=True)
    return fish_raw, experiment


def _add_derived_columns(fdf, fish_id, n_stimuli, dish_for):
    """Add derived columns to one fish's raw dataframe, in place.

    Adds trial/block ids from condition transitions and ``dots_center``/
    ``dots_out`` from the shown dots -- replaying ``AllFish``'s own logic on the
    base columns.

    Args:
        fdf (pandas.DataFrame): One fish's raw per-frame dataframe. Modified in
            place.
        fish_id (int): The fish's id, passed through to ``dish_for``.
        n_stimuli (int): Number of dot slots (``d0``..``d{n_stimuli-1}``).
        dish_for (callable): ``dish_for(fish_id) -> (center_x, center_y, radius)``.
            Abstracts over a per-fish ROI lookup (real data) vs. a single shared
            dish (every generated rollout uses the same one).

    Returns:
        None
    """
    fdf['trial_start'] = (fdf['stimulus'] == 'inter_stim_pause') & (fdf['stimulus'].shift(1) != 'inter_stim_pause')
    fdf['trial_id'] = fdf['trial_start'].cumsum()
    fdf['stimulus_block_id'] = (fdf['stimulus'] != fdf['stimulus'].shift(1)).cumsum()
    fdf['frame_count_stimulus'] = fdf.groupby('stimulus_block_id').cumcount()

    dot_x, dot_y = [f'd{d}_x' for d in range(n_stimuli)], [f'd{d}_y' for d in range(n_stimuli)]
    shown = (fdf[[f'd{d}_stim' for d in range(n_stimuli)]] != 'none').to_numpy()
    fdf['dots_center_x_abs'] = fdf[dot_x].where(shown, np.nan).mean(axis=1, skipna=True)
    fdf['dots_center_y_abs'] = fdf[dot_y].where(shown, np.nan).mean(axis=1, skipna=True)

    cx, cy, r = dish_for(fish_id)
    fdf['dots_out'] = sum(((fdf[f'd{d}_x'] - cx) ** 2 + (fdf[f'd{d}_y'] - cy) ** 2 > r ** 2).astype(int)
                          for d in range(n_stimuli))


def _assemble_fish_df_long(fish_raw, experiment, n_stimuli, dish_for, framerate=30, px_to_mm=3.67):
    """Build the shared frame-level long-format dataframe from raw per-fish tables.

    Shared by real (``build_fish_df_long``) and generated
    (``gen_frame_long.build_gen_fish_df_long``) data: given a per-individual raw
    frame table (``frame``, ``stimulus``, ``x``, ``y``, ``theta``, ``d{i}_x/y/stim``),
    does everything downstream of that -- derived columns, rotate-to-egocentric,
    melt to long format, distances, labels, speed.

    Args:
        fish_raw (dict[int, pandas.DataFrame]): Mapping from fish id to that fish's
            raw per-frame dataframe.
        experiment (str): Experiment name, passed to
            ``src.analyse.rotate_trajectory``/``convert_label``/``group_condition_name``.
        n_stimuli (int): Number of dot slots (``d0``..``d{n_stimuli-1}``).
        dish_for (callable): ``dish_for(fish_id) -> (center_x, center_y, radius)``.
            Abstracts over a per-fish ROI lookup (real data) vs. a single shared
            dish (every generated rollout uses the same one).
        framerate (int): Frames per second, used for the speed columns. Defaults
            to 30.
        px_to_mm (float): Pixels-per-millimetre scale factor. Defaults to 3.67.

    Returns:
        pandas.DataFrame: The long-format dataframe, one row per (fish, frame).
        Callers attach their own ``meta``.
    """
    fish_ids = sorted(fish_raw)
    for f in fish_ids:
        _add_derived_columns(fish_raw[f], f, n_stimuli, dish_for)

    # Rotate fish + stimuli coordinates so fish orientation is at 0, per stimulus block
    fish_ego_wide, stim_ego_dict, shared_ids = {}, {}, None
    for f in fish_ids:
        fdf = fish_raw[f].rename(columns={'x': f'f{f}_x', 'y': f'f{f}_y', 'theta': f'f{f}_ori', 'stimulus': 'condition'})
        ego_fish_trials, ego_stim_trials = [], []
        for _, group in fdf.groupby('stimulus_block_id'):
            ego_fish, ego_stim = an.rotate_trajectory(f, group, group, n_stimuli, experiment)
            ego_fish_trials.append(ego_fish)
            ego_stim_trials.append(ego_stim)
        ego_fish_df = pd.concat(ego_fish_trials).reset_index(drop=True)
        fish_cols = fdf[[f'f{f}_x', f'f{f}_y', f'f{f}_ori']].reset_index(drop=True).round(3)
        fish_ego_wide[f] = pd.concat([ego_fish_df, fish_cols], axis=1)
        stim_ego_dict[f'f{f}'] = pd.concat([pd.concat(ego_stim_trials).reset_index(drop=True),
                                            fdf.reset_index(drop=True)], axis=1)
        if shared_ids is None:
            shared_ids = fdf[_ID_COLS].reset_index(drop=True)  # identical across fish (global timing/condition)
    ego_df = pd.concat([fish_ego_wide[f] for f in fish_ids] + [shared_ids], axis=1)

    # Long format: one row per (fish, frame)
    fish_value_cols = [c for c in ego_df.columns if any(c.startswith(f'f{i}_') for i in fish_ids)]
    long_df = pd.melt(ego_df, id_vars=_ID_COLS, value_vars=fish_value_cols, var_name='fish_number', value_name='value')
    long_df[['fish_number', 'type']] = long_df['fish_number'].str.extract(r'f(\d{1,2})_(\w+)')
    long_df['fish_number'] = long_df['fish_number'].astype(int)
    long_df = long_df.pivot(index=['fish_number'] + _ID_COLS, columns='type', values='value').reset_index()
    long_df = long_df.rename_axis(None, axis=1).sort_values(['fish_number', 'frame'])

    # Dots/square centers + distances
    dots_columns = ['dots_center_x_abs', 'dots_center_x_ego', 'dots_center_x_rot',
                    'dots_center_y_abs', 'dots_center_y_ego', 'dots_center_y_rot', 'dots_out']
    for col in dots_columns:
        long_df[col] = pd.concat([stim_ego_dict[f'f{f}'][col].reset_index(drop=True) for f in fish_ids], ignore_index=True)
    masks = {'left': long_df['dots_center_x_rot'] < 0, 'right': long_df['dots_center_x_rot'] > 0}
    for side in ['left', 'right']:
        for coord in ['x', 'y']:
            center = an.get_average_center(stim_ego_dict, n_stimuli, side=side, coord=coord)
            long_df.loc[masks[side], f'square_center_{coord}_rot'] = center

    long_df['dotsDist'] = an.euclidean_distance(long_df.x, long_df.y, long_df.dots_center_x_abs, long_df.dots_center_y_abs)
    long_df['dotsDist_ego'] = an.euclidean_distance(long_df.x_ego, long_df.y_ego, long_df.dots_center_x_ego, long_df.dots_center_y_ego)
    long_df['dotsDist_rot'] = an.euclidean_distance(long_df.x_rot, long_df.y_rot, long_df.dots_center_x_rot, long_df.dots_center_y_rot)
    long_df['squareDist_rot'] = an.euclidean_distance(long_df.x_rot, long_df.y_rot, long_df.square_center_x_rot, long_df.square_center_y_rot)

    long_df['stim_side'] = np.where(long_df['dots_center_x_rot'] < 0, 'L', 'R')
    long_df['stimulus'] = long_df['condition'].apply(an.convert_label, experiment_type=experiment)
    long_df = long_df.groupby(['fish_number', 'trial_id'], group_keys=False)[long_df.columns].apply(an.assign_first_choice2, experiment_type=experiment)
    long_df['condition_grouped'] = long_df['condition'].apply(an.group_condition_name)

    # Speed
    x_shift, y_shift = long_df.groupby('fish_number')['x'].diff(), long_df.groupby('fish_number')['y'].diff()
    long_df['speed'] = np.sqrt(x_shift ** 2 + y_shift ** 2)
    long_df['scaled_speed'] = long_df['speed'] * framerate / px_to_mm

    return long_df


def build_fish_df_long(data_dir, roi_file, framerate=30, px_to_mm=3.67):
    """Build the frame-level long-format dataframe for real fish.

    Args:
        data_dir (str or pathlib.Path): Directory containing the per-fish
            ``*_transformed_abs.csv`` files.
        roi_file (str or pathlib.Path): Path to the ROI definition CSV
            (``xoff, yoff, diameter, x, y, radius`` columns, no header, one row per
            fish).
        framerate (int): Frames per second, used for the speed columns. Defaults
            to 30.
        px_to_mm (float): Pixels-per-millimetre scale factor. Defaults to 3.67.

    Returns:
        tuple: ``(fish_df_long, meta)`` where ``fish_df_long`` (pandas.DataFrame)
        is the frame-level long-format dataframe and ``meta`` (dict) has keys
        ``experiment``, ``n_fish``, ``n_stimuli``.
    """
    fish_raw, experiment = _load_fish_raw(data_dir)
    fish_ids = sorted(fish_raw)
    n_stimuli = sum(1 for c in fish_raw[fish_ids[0]].columns if re.fullmatch(r'd\d+_x', c))

    roi_df = pd.read_csv(roi_file, header=None)
    roi_df.columns = ['xoff', 'yoff', 'diameter', 'x', 'y', 'radius']

    def dish_for(fish_id):
        row = roi_df.loc[fish_id]
        return row['x'] - row['xoff'], row['y'] - row['yoff'], row['radius']

    long_df = _assemble_fish_df_long(fish_raw, experiment, n_stimuli, dish_for, framerate, px_to_mm)
    return long_df, {'experiment': experiment, 'n_fish': len(fish_ids), 'n_stimuli': n_stimuli}
