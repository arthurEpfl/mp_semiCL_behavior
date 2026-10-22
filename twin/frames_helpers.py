import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re


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




