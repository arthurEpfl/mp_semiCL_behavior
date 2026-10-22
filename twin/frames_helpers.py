import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def compute_stimuli_absolute(F_ID, stim_group, fish_row, n_stimuli):
    """
    Compute stimuli absolute positions relative to fish using original fish values.

    Parameters:
        stim_group : DataFrame with stimuli positions in pixels (already scaled to mm).
        fish_row   : A Series with fish's x, y, and ori at trial start.
        n_stimuli  : Number of stimuli/dots.

    Returns:
        DataFrame with absolute positions for each stimulus.
    """
    new_data = {}
    first_x, first_y, first_ori = fish_row[f'f{F_ID}_x'], fish_row[f'f{F_ID}_y'], fish_row[f'f{F_ID}_ori']
    # Heading adjustment: subtract pi/2 so that a fish ori of 90° (pi/2) gives heading zero.
    heading = first_ori - np.pi / 2

    for dot in range(n_stimuli):
        dot_x = stim_group[f'd{dot}_x']
        dot_y = stim_group[f'd{dot}_y']
        dot_theta, dot_r = cart2pol(dot_x, dot_y)
        # Rotate the dot’s displacement by the fish’s heading.
        dot_x_rot, dot_y_rot = pol2cart(dot_theta + heading, dot_r)
        # Compute absolute positions by offsetting with fish's starting x and y.
        new_data[f'd{dot}_x'] = dot_x_rot + first_x
        new_data[f'd{dot}_y'] = dot_y_rot + first_y
        new_data[f'd{dot}_stim'] = stim_group[f'd{dot}_stim']

    return pd.DataFrame(new_data, index=stim_group.index)


def transform_coordinates(fish_df, stimuli_allfish, roi_df, invert_ori=True, invert_y=True):
    """
    Apply the transformation: invert fish orientation (to CCW) and invert y positions.
    These transformations are applied before computing rotated trajectories.

    Parameters:
        fish_df   : DataFrame with fish coordinates.
        stimuli_df: Dictionary with stimuli DataFrames for all fish.
        diameter  : ROI diameter.

    Returns:
        fish_df, stimuli_df transformed.
    """
    fish_df_copy = fish_df.copy()

    if invert_ori:
        # Invert all fish orientation columns (columns that end with '_ori')
        ori_cols = [col for col in fish_df.columns if col.endswith('_ori')]
        fish_df_copy[ori_cols] = (2 * np.pi - fish_df_copy[ori_cols]) % (2 * np.pi)

    if invert_y:
        for row in roi_df.iterrows():
            fish_ID = row[0]
            diameter = row[1]["diameter"]
            fish_df_copy[f'f{fish_ID}_y'] = diameter - fish_df_copy[f'f{fish_ID}_y']

        def process_stimuli_df(idx, stimuli_df):
            """Helper function to process a stimulus DataFrame."""
            #mask = stimuli_df['condition'] != 'inter_stim_pause'
            y_cols_stim = [col for col in stimuli_df.columns if col.endswith('_y')]
            diameter = roi_df.loc[idx, "diameter"]
            #stimuli_df.loc[mask, y_cols_stim] = diameter - stimuli_df.loc[mask, y_cols_stim]
            stimuli_df[y_cols_stim] = diameter - stimuli_df[y_cols_stim]
            return stimuli_df

        # Check if stimuli_allfish is a dictionary or a single DataFrame
        if isinstance(stimuli_allfish, dict):
            stimuli_transformed = {items[0]: process_stimuli_df(i, items[1].copy()) for i, items in enumerate(stimuli_allfish.items())}
        elif isinstance(stimuli_allfish, pd.DataFrame):
            stimuli_transformed = process_stimuli_df(0, stimuli_allfish.copy())
        else:
            raise ValueError("stimuli_allfish must be either a dictionary of DataFrames or a single DataFrame.")

    return fish_df_copy, stimuli_transformed


def _dots_center(stimuli_copy, mask_stim, mask_left, mask_right, n_stimuli, experiment_type):
    """Helper function to compute stimuli centers based on experiment type."""
    if experiment_type == 'GsizeMotionChoice':
        mask_stim_left = mask_stim.to_numpy() & mask_left.to_numpy()
        mask_stim_right = mask_stim.to_numpy() & mask_right.to_numpy()

        for side in ['left', 'right']:
            mask_side = mask_stim_left if side == 'left' else mask_stim_right
            stim_x_values = stimuli_copy[[f"d{k}_x" for k in range(n_stimuli)]].where(mask_side, np.nan)
            stim_y_values = stimuli_copy[[f"d{k}_y" for k in range(n_stimuli)]].where(mask_side, np.nan)
            stimuli_copy[f"dots_center_x_abs_{side}"] = stim_x_values.mean(axis=1, skipna=True)
            stimuli_copy[f"dots_center_y_abs_{side}"] = stim_y_values.mean(axis=1, skipna=True)
    else:
        stim_x_values = stimuli_copy[[f"d{k}_x" for k in range(n_stimuli)]].where(mask_stim.to_numpy(), np.nan)
        stim_y_values = stimuli_copy[[f"d{k}_y" for k in range(n_stimuli)]].where(mask_stim.to_numpy(), np.nan)
        stimuli_copy["dots_center_x_abs"] = stim_x_values.mean(axis=1, skipna=True)
        stimuli_copy["dots_center_y_abs"] = stim_y_values.mean(axis=1, skipna=True)

    return stimuli_copy

def pause_copy_trajectory(group, dot_xy_cols, dot_stim_cols):
    pause_block = group[group['condition'] == 'inter_stim_pause']
    stim_block = group[group['condition'] != 'inter_stim_pause']
    mask_stim = np.array(stim_block[dot_stim_cols] != "none")

    if not pause_block.empty and not stim_block.empty:
        n = len(pause_block)
        # Assign values to the pause block
        group.loc[pause_block.index, dot_xy_cols] = stim_block.iloc[:n][dot_xy_cols].values
        group.loc[pause_block.index, dot_stim_cols] = pause_block[dot_stim_cols].mask(mask_stim[:n], 'inter_stim_pause')
    return group

def compute_stimuli_centers(stimuli_allfish_dict, n_stimuli, mask_left, mask_right, experiment_type):

    stimuli_dict = {}

    for fish, key in enumerate(stimuli_allfish_dict.keys()):
        stimuli_copy = stimuli_allfish_dict[key].copy()
        mask_stim = stimuli_copy[[f"d{k}_stim" for k in range(n_stimuli)]] != "none"
        stimuli_copy = _dots_center(stimuli_copy, mask_stim, mask_left, mask_right, n_stimuli, experiment_type)
        stimuli_dict[key] = stimuli_copy

    return stimuli_dict

def calculate_dots_out(stim_dict, n_stimuli, roi_df, experiment_type, mask_left, mask_right):
    """
    Calculate the number of dots outside the ROI for each frame.

    Parameters:
        stim_dict : Dictionary with stimuli DataFrames for all fish.
        n_stimuli : Number of stimulus dots.
        roi       : DataFrame containing ROI information.

    Returns:
        Dictionary with DataFrames containing the number of dots outside the ROI for each frame.
    """
    stim_dict_copy = stim_dict.copy()
    for fish, key in enumerate(stim_dict.keys()):
        center_x = roi_df.loc[fish, 'x'] - roi_df.loc[fish, 'xoff']
        center_y = roi_df.loc[fish, 'y'] - roi_df.loc[fish, 'yoff']
        radius = roi_df.loc[fish, 'radius']
        if experiment_type == 'GsizeMotionChoice':
            x_left = stim_dict_copy[key].filter(regex='x$').where(mask_left.to_numpy(), np.nan)
            y_left = stim_dict_copy[key].filter(regex='y$').where(mask_left.to_numpy(), np.nan)
            dots_out_left = sum(((x_left[f"d{d}_x"] - center_x) ** 2 + (y_left[f"d{d}_y"] - center_y) ** 2 > radius ** 2).astype(int)
                                for d in range(n_stimuli))

            x_right = stim_dict_copy[key].filter(regex='x$').where(mask_right.to_numpy(), np.nan)
            y_right = stim_dict_copy[key].filter(regex='y$').where(mask_right.to_numpy(), np.nan)
            dots_out_right = sum(((x_right[f"d{d}_x"] - center_x) ** 2 + (y_right[f"d{d}_y"] - center_y) ** 2 > radius ** 2).astype(int)
                                for d in range(n_stimuli))

            stim_dict_copy[key]['dots_out_left'] = dots_out_left
            stim_dict_copy[key]['dots_out_right'] = dots_out_right

        else:
            dots_out = sum(((stim_dict_copy[key][f"d{d}_x"] - center_x) ** 2 + (stim_dict_copy[key][f"d{d}_y"] - center_y)
                                 ** 2 > radius ** 2).astype(int) for d in range(n_stimuli))

            stim_dict_copy[key]['dots_out'] = dots_out

    return stim_dict_copy

def rotate_trajectory(F_ID, fish_group, stim_group, n_stimuli, experiment_type):
    """
    Compute the rotated (egocentric) trajectory for fish and stimuli.

    Assumes fish and stimuli data have been transformed.

    Parameters:
        fish_group: DataFrame for one trial of fish data.
        stim_group: DataFrame for corresponding stimuli absolute positions.
        n_stimuli : Number of stimuli/dots.

    Returns:
        Tuple of DataFrames: (egocentric fish trajectory, egocentric stimuli trajectory)
    """
    new_fish = {}
    new_stim = {}

    # Use the first frame of the trial for the transformation.
    first_x = fish_group.iloc[0][f'f{F_ID}_x']
    first_y = fish_group.iloc[0][f'f{F_ID}_y']
    first_ori = fish_group.iloc[0][f'f{F_ID}_ori']

    # Compute fish displacement (egocentric coordinates)
    x_ego = fish_group[f'f{F_ID}_x'] - first_x
    y_ego = fish_group[f'f{F_ID}_y'] - first_y
    theta, r = cart2pol(x_ego.values, y_ego.values)
    theta_adjusted = (theta - first_ori) + (np.pi / 2)
    x_rot, y_rot = pol2cart(theta_adjusted, r)

    new_fish[f'f{F_ID}_x_ego'] = np.round(x_ego, 3)
    new_fish[f'f{F_ID}_y_ego'] = np.round(y_ego,3)
    new_fish[f'f{F_ID}_x_rot'] = np.round(x_rot,3)
    new_fish[f'f{F_ID}_y_rot'] = np.round(y_rot, 3)
    new_fish[f'f{F_ID}_theta_adj'] = np.round(theta_adjusted, 3)

    # Compute stimuli displacement (egocentric) for each dot.
    for dot in range(n_stimuli):
        dot_x = stim_group[f'd{dot}_x'] - first_x
        dot_y = stim_group[f'd{dot}_y'] - first_y
        dot_theta, dot_r = cart2pol(dot_x, dot_y)
        dot_theta_adjusted = (dot_theta - first_ori) + (np.pi / 2)
        dot_x_rot, dot_y_rot = pol2cart(dot_theta_adjusted, dot_r)

        new_stim[f'd{dot}_x_ego'] = np.round(dot_x, 3)
        new_stim[f'd{dot}_y_ego'] = np.round(dot_y, 3)
        new_stim[f'd{dot}_x_rot'] = np.round(dot_x_rot, 3)
        new_stim[f'd{dot}_y_rot'] = np.round(dot_y_rot, 3)
        new_stim[f'd{dot}_r'] = np.round(dot_r, 3)
        new_stim[f'd{dot}_angle'] = np.round(dot_theta_adjusted, 3)

    # Compute the center of the dots in egocentric coordinates.
    if experiment_type == "GsizeMotionChoice":
        for side in ["left", "right"]:
            dots_center_x_ego = stim_group[f'dots_center_x_abs_{side}'] - first_x
            dots_center_y_ego = stim_group[f'dots_center_y_abs_{side}'] - first_y
            dots_center_theta, dots_center_r = cart2pol(dots_center_x_ego, dots_center_y_ego)
            dots_center_theta_adjusted = (dots_center_theta - first_ori) + (np.pi / 2)
            dots_center_x_rot, dots_center_y_rot = pol2cart(dots_center_theta_adjusted, dots_center_r)

            new_stim[f'dots_center_x_ego_{side}'] = np.round(dots_center_x_ego, 3)
            new_stim[f'dots_center_y_ego_{side}'] = np.round(dots_center_y_ego, 3)
            new_stim[f'dots_center_x_rot_{side}'] = np.round(dots_center_x_rot, 3)
            new_stim[f'dots_center_y_rot_{side}'] = np.round(dots_center_y_rot, 3)
    else:
        dots_center_x_ego = stim_group['dots_center_x_abs'] - first_x
        dots_center_y_ego = stim_group['dots_center_y_abs'] - first_y
        dots_center_theta, dots_center_r = cart2pol(dots_center_x_ego, dots_center_y_ego)
        dots_center_theta_adjusted = (dots_center_theta - first_ori) + (np.pi / 2)
        dots_center_x_rot, dots_center_y_rot = pol2cart(dots_center_theta_adjusted, dots_center_r)

        new_stim[f'dots_center_x_ego'] = np.round(dots_center_x_ego, 3)
        new_stim[f'dots_center_y_ego'] = np.round(dots_center_y_ego, 3)
        new_stim[f'dots_center_x_rot'] = np.round(dots_center_x_rot, 3)
        new_stim[f'dots_center_y_rot'] = np.round(dots_center_y_rot, 3)

    return pd.DataFrame(new_fish, index=fish_group.index), pd.DataFrame(new_stim, index=stim_group.index)

# Define mapping function
def convert_label(label, experiment_type):

    parts = label.split("_")  # Split by underscore

    if experiment_type == "GsizeMotionChoice":

        if label == "inter_stim_pause":
            return label, label

        g_values = parts[1].replace("g", "").split("-")
        s_values = parts[2].split("-")

        g_left = int(g_values[0])
        g_right = int(g_values[1])
        s_left = float(s_values[0].replace('s', ''))
        s_right = float(s_values[1])
        return f"d{g_left}s{s_left}", f"d{g_right}s{s_right}"

    else:
        if label == "inter_stim_pause":
            return label  # Keep this unchanged
        g_values = parts[1].replace("g", "").split("-")
        s_values = parts[2].split("-")

        abs_diff = abs(int(g_values[0]) - int(g_values[1]))
        return f"d{abs_diff}{s_values[0]}"  # Construct new label

def convert_label2(label):

    parts = label.split("_")  # Split by underscore

    if label == "inter_stim_pause":
        return label, label

    g_values = parts[1].replace("g", "").split("-")
    s_values = parts[2].split("-")

    g_left = int(g_values[0])
    g_right = int(g_values[1])
    s_left = float(s_values[0].replace('s', ''))
    s_right = float(s_values[1])
    return f"d{g_left}s{s_left}", f"d{g_right}s{s_right}"

def group_condition_name(cond):
    if cond == 'inter_stim_pause':
        return cond

    match = re.match(r'.*_(g(\d+)-(\d+))_s([\d\.]+)-([\d\.]+)', cond)
    if match:
        # Extract group sizes and speeds
        g1, g2 = int(match.group(2)), int(match.group(3))
        s1, s2 = float(match.group(4)), float(match.group(5))

        # Sort group sizes to make order irrelevant
        sorted_group = sorted([g1, g2])
        group = f"g{sorted_group[0]}-{sorted_group[1]}"

        # Define condition type
        # Define condition type
        if s1 == s2:
            suffix = "motion" if s1 > 0 else "static"
        else:
            suffix = "static-motion"

        return f"{group}_{suffix}"

    return cond  # fallback for unmatched strings


def reshape_stimulus_data(df, id_vars):
    """
    Reshapes the stimulus data by melting the DataFrame and then pivoting it.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the stimulus data.
    id_vars (list): List of columns to remain as identifiers (non-variable columns).
    value_vars (list): List of columns to reshape into long format.

    Returns:
    pd.DataFrame: A reshaped DataFrame with the desired structure.
    """
    value_vars = [
        'dots_center_x_abs_left', 'dots_center_x_abs_right',
        'dots_center_y_abs_left', 'dots_center_y_abs_right',
        'dots_center_x_ego_left', 'dots_center_x_ego_right',
        'dots_center_y_ego_left', 'dots_center_y_ego_right',
        'dots_center_x_rot_left', 'dots_center_x_rot_right',
        'dots_center_y_rot_left', 'dots_center_y_rot_right',
        'dots_out_left', 'dots_out_right']

    # Melt the DataFrame to long format
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')

    # Create a 'side' column based on the 'variable' column
    df_long['side'] = df_long['variable'].apply(lambda x: 'left' if 'left' in x else 'right')

    # Clean the 'variable' column by removing '_left' and '_right'
    df_long['variable'] = df_long['variable'].str.replace('_left', '').str.replace('_right', '')

    # Pivot the DataFrame to wide format based on the 'variable' and 'side'
    df_pivoted = df_long.pivot(index=id_vars + ['side'], columns='variable', values='value').reset_index()
    df_pivoted = df_pivoted.rename_axis(None, axis=1)
    return df_pivoted

def assign_first_choice(fish_long_df, experiment_type, n_frames=90):
    df = fish_long_df.copy()
    df["first_choice"] = np.nan

    for (fish_number, trial_id), group in df.groupby(["fish_number", "trial_id"]):
        # Find trial start
        trial_start_idx = group.index[group['trial_start'] == True]

        if not trial_start_idx.empty:
            start_idx = trial_start_idx[0]
            trial_start_frame = group.loc[start_idx, 'frame']
            trial_group = group[group['frame'] >= trial_start_frame].head(n_frames)

            avg_x_rot = trial_group['x_rot'].mean()
            went_left = avg_x_rot < 0

            if experiment_type == "GsizeMotionChoice":
                # Get indexes for where trial started and which side
                left_idx = group[(group['trial_start'] == True) & (group['side'] == 'left')].index
                right_idx = group[(group['trial_start'] == True) & (group['side'] == 'right')].index

                df.loc[left_idx, "first_choice"] = float(went_left)
                df.loc[right_idx, "first_choice"] = float(not went_left)
            else:
                df["first_choice"] = df["first_choice"].astype("string")
                # Get indexes for where trial started and which side
                idx_start = group[(group['trial_start'] == True)].index
                df.loc[idx_start, "first_choice"] = 'left' if went_left else 'right'

    return df

def assign_first_choice2(group, experiment_type, n_frames=90):
    trial_start_idx = group.index[group['trial_start'] == True]

    if trial_start_idx.empty:
        return group

    start_idx = trial_start_idx[0]
    trial_start_frame = group.loc[start_idx, 'frame']
    trial_group = group[group['frame'] >= trial_start_frame].head(n_frames)

    avg_x_rot = trial_group['x_rot'].mean()
    went_left = avg_x_rot < 0
    group["first_choice"] = np.nan

    if experiment_type == "GsizeMotionChoice":
        left_idx = group[(group['trial_start'] == True) & (group['side'] == 'left')].index
        right_idx = group[(group['trial_start'] == True) & (group['side'] == 'right')].index

        group.loc[left_idx, "first_choice"] = float(went_left)
        group.loc[right_idx, "first_choice"] = float(not went_left)
    else:
        group["first_choice"] = group["first_choice"].astype("string")
        first_choice = 'left' if went_left else 'right'
        group.loc[start_idx, "first_choice"] = first_choice

    return group

def melt_bout_df(bout_df, x_cols, y_cols, ori_cols):
    bout_df_long = pd.melt(bout_df, id_vars=['frame', 'condition'], value_vars=x_cols + y_cols + ori_cols,
                           var_name='measure', value_name='value')

    bout_df_long['fish_number'] = bout_df_long['measure'].str.extract(r'f(\d+)_')[0].astype(int)
    bout_df_long['variable'] = bout_df_long['measure'].str.extract(r'_(x|y|ori)')[0]

    # Pivot data
    bout_df_long = bout_df_long.pivot_table(index=['frame', 'condition', 'fish_number'],
                                            columns='variable', values='value').reset_index()

    bout_df_long.columns.name = None
    bout_df_long = bout_df_long[["fish_number", "frame", "condition", "x", "y", "ori"]]
    bout_df_long = bout_df_long.sort_values(by=["fish_number", "frame"]).reset_index(drop=True)

    return bout_df_long

def shift_coordinates(df, to_shift, framerate):
    # df = bout df
    # to_shift = number of frames to shift
    # framerate = framerate of the video

    # Calculate shifted values
    df['x_shift'] = df.groupby('fish_number')['x'].shift(to_shift)
    df['y_shift'] = df.groupby('fish_number')['y'].shift(to_shift)
    df['ori_shift'] = df.groupby('fish_number')['ori'].shift(to_shift)

    # Handle edge cases for shifts
    df.loc[df.groupby('fish_number')['x_shift'].head(to_shift).index, 'x_shift'] = df.groupby('fish_number')['x'].tail(to_shift).reset_index(drop=True)
    df.loc[df.groupby('fish_number')['y_shift'].head(to_shift).index, 'y_shift'] = df.groupby('fish_number')['y'].tail(to_shift).reset_index(drop=True)
    df.loc[df.groupby('fish_number')['ori_shift'].head(to_shift).index, 'ori_shift'] = df.groupby('fish_number')['ori'].tail(to_shift).reset_index(drop=True)

    return df

def shuffle_coordinates(df):
    "Shuffle coordinates within each fish group"
    df['x_shuffled'] = df.groupby('fish_number')['x'].transform(np.random.permutation)
    df['y_shuffled'] = df.groupby('fish_number')['y'].transform(np.random.permutation)
    df['ori_shuffled'] = df.groupby('fish_number')['ori'].transform(np.random.permutation)
    return df

def get_average_center(stim_ego_dict, n_stimuli, side='left', coord='x'):
    """
    Compute the average center coordinate (x or y) for a given side ('left' or 'right')
    across all fish in stim_ego_dict.

    Parameters:
    - stim_ego_dict (dict): Dictionary with data for each fish (e.g., 'f0', 'f1', ...).
    - side (str): 'left' or 'right'.
    - coord (str): 'x' or 'y'.
    - n_stimuli (int): Number of stimuli per trial (default 8).

    Returns:
    - float: The average center coordinate across all fish.
    """
    center_values = []

    for fish_id, df in stim_ego_dict.items():
        mask = df['dots_center_x_rot'] < 0 if side == 'left' else df['dots_center_x_rot'] > 0

        min_col = [df.loc[mask, f'd{d}_{coord}_rot'].min() for d in range(n_stimuli)]
        max_col = [df.loc[mask, f'd{d}_{coord}_rot'].max() for d in range(n_stimuli)]

        avg_center = np.nanmean([min_col, max_col])  # mean of min and max values
        center_values.append(avg_center)

    return np.nanmean(center_values)