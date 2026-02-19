import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display
from matplotlib import animation
from IPython.display import Image
import warnings
from matplotlib.patches import Wedge
import re





def stack_dict_pipeline(fish_dict):
    """
    Stack all per-fish DataFrames from abs_dict or polar_dict (output of the pipeline)
    into a single long-format DataFrame.

    Parameters
    ----------
    fish_dict : dict
        Dictionary like {"f0": df_f0, ..., "f14": df_f14}.

    Returns
    -------
    pd.DataFrame
        Long DataFrame with one row per frame per fish, and columns:
        - frame, source_frame, stimulus, ...
        - fish_x, fish_y, fish_ori
        - d0_x/d0_y/d0_stim ... (for abs_dict)
        - or d0_angle/d0_radius/d0_stim ... (for polar_dict)
        - fish_id (int)
    """
    stacked = []

    for fish_key, df in fish_dict.items():
        tmp = df.copy()

        # Extract numeric fish index from key "f0", "f1", ...
        try:
            fish_id = int(fish_key.lstrip("f"))
        except ValueError:
            fish_id = fish_key

        # Rename fish-specific columns (f0_x, f0_y, f0_ori) -> generic names
        rename_map = {}
        for var in ["x", "y", "ori"]:
            col_name = f"{fish_key}_{var}"
            if col_name in tmp.columns:
                rename_map[col_name] = f"f_{var}"

        tmp = tmp.rename(columns=rename_map)

        # Add fish_id column
        tmp["fish_id"] = fish_id

        stacked.append(tmp)

    # Concatenate all fish
    stacked_df = pd.concat(stacked, ignore_index=True, sort=False)
    return stacked_df




def remove_nan_plus_constant_and_correlated_features(df, corr_threshold=0.95, remove_nan_threshold=0.2, verbose=True):
    """
    Remove constant and highly correlated features from a DataFrame. 
    If NaN values represent less than 2% of dataset, the corresponding rows are removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (numerical features only).
    corr_threshold : float, default=0.95
        Absolute correlation threshold above which one of the correlated
        features will be removed.
    verbose : bool, default=True
        If True, prints information about removed columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with constant and highly correlated features removed.
    list
        List of removed features.
    """


    df = df.copy()

    nan_percentage = df.isna().any(axis=1).mean() * 100
    print("The dataset contains : " + str(nan_percentage) + "% of NaN.")
    
    if nan_percentage < remove_nan_threshold:
        df = df.dropna()
        print("There are few NaN, therefore we make the choice to remove them.")
    
    removed_features = []

    # --- 1. Remove constant features --- I COMMENT THIS PART BECAUSE I WANT TO KEEP d0_stim
    '''
    selector = VarianceThreshold(threshold=0)
    selector.fit(df)
    constant_columns = [col for col, var in zip(df.columns, selector.variances_) if var == 0]

    if constant_columns:
        if verbose:
            print(f"🧩 Constant features removed: {constant_columns}")
        df.drop(columns=constant_columns, inplace=True)
        removed_features.extend(constant_columns)
    else:
        if verbose:
            print("✅ No constant features found.")
    '''

    # --- 2. Remove correlated features ---
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    correlated_to_remove = set()
    for col in upper.columns:
        correlated_cols = [idx for idx, val in upper[col].items() if val > corr_threshold and idx != col]
        correlated_to_remove.update(correlated_cols)

    if correlated_to_remove:
        if verbose:
            print(f"⚙️ {len(correlated_to_remove)} correlated features removed (|r| > {corr_threshold}):")
            print(list(correlated_to_remove))
        df.drop(columns=list(correlated_to_remove), inplace=True)
        removed_features.extend(correlated_to_remove)
    else:
        if verbose:
            print("✅ No highly correlated features found.")

    return df, removed_features




def compute_speed(df, x_col="f_x", y_col="f_y", name="f_speed"):
    """
    Compute the instantaneous speed (magnitude of displacement)
    based on x and y positions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing x and y position columns.
    x_col, y_col : str
        Names of the columns representing position coordinates.
    name : str
        Name of the output column for speed.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with a new column for speed.
    """
    df = df.copy()
    df["dx"] = df[x_col].diff()
    df["dy"] = df[y_col].diff()
    df[name] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df = df.drop(columns=["dx", "dy"])
    
    first = [name]
    others = [c for c in df.columns if c not in first]
    df = df[first + others]
    
    df = df.dropna()
    
    return df




def compute_d_orientation(df, ori_col="f_ori", name="d_ori"):
    """
    Compute the change in orientation (Δθ) between consecutive frames.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an orientation column.
    ori_col : str
        Name of the orientation column.
    name : str
        Name of the output column for Δθ.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with a new column for change in orientation.
    """
    df = df.copy()
    # 1. raw difference (might be around ±2π)
    raw = df[ori_col].diff()

    # 2. wrap into (-π, π] to remove big jumps
    dtheta = (raw + np.pi) % (2 * np.pi) - np.pi
    
    df[name] = dtheta
    
    df = df[df[name].abs() <= 1]
    print("Removed rows with |d_ori| > 1 radians. There are artefacts representing less than 4% of the data.")
    
    first = [name]
    others = [c for c in df.columns if c not in first]
    df = df[first + others]
    
    df = df.dropna()
    
    return df




def temporal_train_val_test_split(
    df,
    target_cols,
    train_size=0.7,
    val_size=0.15,
    test_size=None,
    verbose=True,
):
    """
    Split a DataFrame into train, validation, and test sets (temporal order preserved).

    Returns directly:
        X_train, X_val, X_test,
        y_train_targets..., y_val_targets..., y_test_targets...

    So one can unpack in one line:
        X_train, X_val, X_test, target_speed_train, target_speed_val, target_speed_test, target_ori_train, target_ori_val, target_ori_test = ...
    """

    if test_size is None:
        test_size = 1 - (train_size + val_size)

    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test sizes must sum to 1.")

    n = len(df)
    split_1 = int(n * train_size)
    split_2 = int(n * (train_size + val_size))

    # Split
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train, X_val, X_test = X.iloc[:split_1], X.iloc[split_1:split_2], X.iloc[split_2:]
    y_train, y_val, y_test = y.iloc[:split_1], y.iloc[split_1:split_2], y.iloc[split_2:]

    if verbose:
        print(f"✅ Temporal split complete:")
        print(f"  Train: {len(X_train)} rows ({train_size*100:.1f}%)")
        print(f"  Val:   {len(X_val)} rows ({val_size*100:.1f}%)")
        print(f"  Test:  {len(X_test)} rows ({test_size*100:.1f}%)")

    # Dynamically unpack target columns
    results = [X_train, X_val, X_test]
    for col in target_cols:
        results.extend([y_train[col], y_val[col], y_test[col]])

    return tuple(results)




def extract_polar_features(polar_grid):
    """
    Extracts 10 biologically and informationally meaningful features 
    from a polar pixel encoding grid.
    
    Parameters
    ----------
    polar_grid : np.ndarray
        2D array (r_bins × theta_bins) of 0/1 encoding stimulus positions.

    Returns
    -------
    dict
        Dictionary with 10 scalar features.
    """
    polar_grid = np.array(polar_grid)
    r_bins, theta_bins = polar_grid.shape
    total_bins = r_bins * theta_bins
    eps = 1e-9  # to avoid log(0)

    # 1. Number of active bins
    stim_count = polar_grid.sum()

    # 2. Average radial distance of active bins
    radii = np.repeat(np.arange(r_bins), theta_bins)
    avg_radius = (polar_grid.flatten() * radii).sum() / (stim_count + eps)

    # 3. Minimum radial distance (closest active stimulus)
    if stim_count > 0:
        min_radius = radii[polar_grid.flatten() == 1].min()
    else:
        min_radius = r_bins  # max possible

    # 4. Fraction of active bins in the front sector (e.g., -60° to +60°)
    front_bins = np.arange(theta_bins//3, 2*theta_bins//3)
    front_ratio = polar_grid[:, front_bins].sum() / (stim_count + eps)

    # 5. Left-right balance (difference between left/right hemisphere)
    left_half = polar_grid[:, :theta_bins//2].sum()
    right_half = polar_grid[:, theta_bins//2:].sum()
    left_right_balance = (right_half - left_half) / (stim_count + eps)

    # 6. Angular entropy (spread of active bins along angles)
    p_theta = polar_grid.sum(axis=0) / (stim_count + eps)
    angular_entropy = -np.sum(p_theta * np.log(p_theta + eps))

    # 7. Radial entropy (spread of active bins along radius)
    p_r = polar_grid.sum(axis=1) / (stim_count + eps)
    radial_entropy = -np.sum(p_r * np.log(p_r + eps))

    # 8. Center of mass in angular space (weighted mean angle)
    theta_vals = np.linspace(-np.pi, np.pi, theta_bins, endpoint=False)
    weights = polar_grid.sum(axis=0)
    if weights.sum() > 0:
        center_of_mass_theta = np.arctan2(
            (np.sin(theta_vals) * weights).sum(),
            (np.cos(theta_vals) * weights).sum()
        )
    else:
        center_of_mass_theta = 0.0

    # 9. Number of active rings (radii containing ≥1 active bin)
    n_rings_active = (polar_grid.sum(axis=1) > 0).sum()

    # 10. Number of active angular sectors (angles containing ≥1 active bin)
    n_angles_active = (polar_grid.sum(axis=0) > 0).sum()

    return {
        "stim_count": stim_count,
        "avg_radius": avg_radius,
        "min_radius": min_radius,
        "front_ratio": front_ratio,
        "left_right_balance": left_right_balance,
        "angular_entropy": angular_entropy,
        "radial_entropy": radial_entropy,
        "center_of_mass_theta": center_of_mass_theta,
        "n_rings_active": n_rings_active,
        "n_angles_active": n_angles_active,
    }
    
    
    
    
def compute_orientations_and_radii(df, x_cols, y_cols, ori_names=None, r_names=None):
    """
    Compute orientations (angles in radians) and radii for several (x, y) coordinate pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing x and y coordinates.
    x_cols : list of str
        List of column names for x coordinates (e.g., ['f0_x', 'd0_x', ...]).
    y_cols : list of str
        List of corresponding y coordinate column names (same length as x_cols).
    ori_names : list of str, optional
        List of names for orientation columns. If None, uses x_col.replace('_x', '_ori').
    r_names : list of str, optional
        List of names for radius columns. If None, uses x_col.replace('_x', '_r').

    Returns
    -------
    pd.DataFrame
        DataFrame with orientation and radius columns (same index as input df).
    """

    if len(x_cols) != len(y_cols):
        raise ValueError("x_cols and y_cols must have the same length.")

    if ori_names is None:
        ori_names = [c.replace("_x", "_ori") for c in x_cols]
    if r_names is None:
        r_names = [c.replace("_x", "_r") for c in x_cols]

    result = pd.DataFrame(index=df.index)

    for x_col, y_col, ori_col, r_col in zip(x_cols, y_cols, ori_names, r_names):
        # Orientation relative to horizontal
        result[ori_col] = np.arctan2(df[y_col], df[x_col])

        # Distance to origin (radius)
        result[r_col] = np.sqrt(df[x_col]**2 + df[y_col]**2)

    return result




def make_egocentric(df, x_col='f_x', y_col='f_y'):
    """
    Convert stimulus positions to egocentric coordinates relative to the fish.
    Only applies to active stimuli (d{k}_stim == 1).
    Inactive stimuli remain exactly at (0,0).
    """
    df = df.copy()

    n_stimuli = sum(1 for col in df.columns if col.startswith('d') and col.endswith('_x'))

    for k in range(n_stimuli):
        stim_x = f'd{k}_x'
        stim_y = f'd{k}_y'
        stim_flag = f'd{k}_stim'

        if stim_x in df.columns and stim_y in df.columns and stim_flag in df.columns:

            mask = df[stim_flag].astype(bool)

            # egocentrize only active stimuli
            df.loc[mask, stim_x] = df.loc[mask, stim_x] - df.loc[mask, x_col]
            df.loc[mask, stim_y] = df.loc[mask, stim_y] - df.loc[mask, y_col]

            # ensure inactive stimuli stay exactly zero
            df.loc[~mask, stim_x] = 0.0
            df.loc[~mask, stim_y] = 0.0

    return df




def make_absolute(df, x_col='f_x', y_col='f_y'):
    """
    Convert egocentric stimulus positions back to absolute positions,
    only when the stimulus is active (d{k}_stim == 1).
    Inactive stimuli remain at 0.
    """
    df = df.copy()

    n_stimuli = sum(
        1 for col in df.columns if col.startswith('d') and col.endswith('_x')
    )

    for k in range(n_stimuli):

        stim_x = f'd{k}_x'
        stim_y = f'd{k}_y'
        stim_flag = f'd{k}_stim'

        if stim_x in df.columns and stim_y in df.columns and stim_flag in df.columns:

            mask = df[stim_flag].astype(bool)

            # restore absolute coordinates ONLY where active
            df.loc[mask, stim_x] = df.loc[mask, stim_x] + df.loc[mask, x_col]
            df.loc[mask, stim_y] = df.loc[mask, stim_y] + df.loc[mask, y_col]

            # keep inactive stimuli at 0
            df.loc[~mask, stim_x] = 0.0
            df.loc[~mask, stim_y] = 0.0

    return df




def add_engineered_features(df, representation, n_stimuli=8):
    """
    Add a compact set of engineered features on top of the base dataframe.

    Inputs
    ------
    df : pd.DataFrame
        Must contain at least:
        - Fish state:
            * f_speed : scalar speed of the fish at each frame
            * d_ori   : orientation change (delta angle) per frame
        - Stimulus columns for each k in [0, n_stimuli):
            If representation == "absolute":
                * d{k}_x, d{k}_y : stimulus position in dish/screen coordinates (relative to fish)
                * d{k}_stim      : binary flag (0 = inactive, 1 = active)
            If representation == "polar":
                * d{k}_radius, d{k}_angle : egocentric polar coordinates
                * d{k}_stim               : binary flag (0 = inactive, 1 = active)

    Added features
    --------------
    Per-stimulus (when columns exist):
        - d{k}_dist   : distance of stimulus k from the fish (0 if inactive)
        - d{k}_side   : {-1, 0, +1}, right / none / left of the fish
        - d{k}_angle_rel (relative to fish orientation): in radians, 0 if stimuli in front of the fish, CCW increasing(0 if inactive)
    Aggregated stimulus features:
        - n_active_stimuli  : number of active stimuli at frame t
        - min_dist_active   : minimum distance among active stimuli (0 if none)
        - mean_dist_active  : mean distance among active stimuli (0 if none)
        - mean_angle_active : mean angle among active stimuli (0 if none)
        - lateral_balance   : sum of d{k}_side (left vs right imbalance)
    """

    df = df.copy()

    stim_flag_cols = []
    dist_pairs = []   # list of (dist_col, stim_flag)
    angle_pairs = []  # list of (angle_col, stim_flag)
    side_cols = []

    # ------------------------------
    # 1) Per-stimulus features
    # ------------------------------
    for k in range(n_stimuli):
        stim_flag = f"d{k}_stim"
        if stim_flag not in df.columns:
            continue

        stim_flag_cols.append(stim_flag)
        active = (df[stim_flag] == 1)

        if representation == "absolute":
            x_col = f"d{k}_x"
            y_col = f"d{k}_y"
            if x_col in df.columns and y_col in df.columns:
                # distance (0 if inactive)
                dist_col = f"d{k}_dist"
                dist_val = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2)
                df[dist_col] = np.where(active, dist_val, 0.0)
                dist_pairs.append((dist_col, stim_flag))

                # effective angle (0 if inactive)
                # effective angle RELATIVE to f_ori (0 if inactive)
                angle_col = f"d{k}_angle_rel"

                # angle absolu du stimulus dans le repère (0, 2π)
                angle_abs = (np.arctan2(df[y_col], df[x_col]) + 2 * np.pi) % (2 * np.pi)

                # s'assurer que f_ori est bien dans [0, 2π)
                f_ori = df["f_ori"] % (2 * np.pi)

                # angle relatif dans [0, 2π)
                angle_rel = (angle_abs - f_ori + 2 * np.pi) % (2 * np.pi)

                df[angle_col] = np.where(active, angle_rel, 0.0)
                angle_pairs.append((angle_col, stim_flag))

                # side: left/right in absolute frame (angle 0 = right)
                side_col = f"d{k}_side"
                df[side_col] = 0.0

                # left = angle_rel dans (0, π)
                df.loc[active & (angle_rel > 0) & (angle_rel < np.pi), side_col] = 1.0

                # right = angle_rel dans (π, 2π)
                df.loc[active & (angle_rel > np.pi), side_col] = -1.0

                side_cols.append(side_col)

        elif representation == "polar":
            r_col = f"d{k}_radius"
            a_col = f"d{k}_angle"
            if r_col in df.columns and a_col in df.columns:
                # radius as distance (0 if inactive)
                dist_col = f"d{k}_dist"
                df[dist_col] = np.where(active, df[r_col], 0.0)
                dist_pairs.append((dist_col, stim_flag))

                # angle (0 if inactive)
                angle_col = a_col
                df[angle_col] = np.where(active, df[a_col], 0.0)
                angle_pairs.append((angle_col, stim_flag))

                # side: left/right in polar frame (angle 0 = front)
                side_col = f"d{k}_side"
                angle_vals = df[a_col]
                left = active & (angle_vals > np.pi / 2) & (angle_vals < 3 * np.pi / 2)
                right = active & ~left

                df[side_col] = 0.0
                df.loc[left, side_col] = 1.0
                df.loc[right, side_col] = -1.0
                side_cols.append(side_col)

        else:
            raise ValueError("representation must be 'absolute' or 'polar'")

    # ------------------------------
    # 2) Aggregated stimulus features
    # ------------------------------
    if stim_flag_cols :
        df["n_active_stimuli"] = df[stim_flag_cols].sum(axis=1)
    else:
        df["n_active_stimuli"] = 0

    # distances Aggregated
    if dist_pairs:
        dist_masked_list = []
        for dist_col, stim_flag in dist_pairs:
            # keep distance only when active, else NaN (for aggregation)
            dist_masked = df[dist_col].where(df[stim_flag] == 1, np.nan)
            dist_masked_list.append(dist_masked)

        dist_mat = pd.concat(dist_masked_list, axis=1)
        df["min_dist_active"] = dist_mat.min(axis=1)
        df["mean_dist_active"] = dist_mat.mean(axis=1)
    else:
        df["min_dist_active"] = np.nan
        df["mean_dist_active"] = np.nan

    # angles Aggregated
    if angle_pairs:
        angle_masked_list = []
        for angle_col, stim_flag in angle_pairs:
            angle_masked = df[angle_col].where(df[stim_flag] == 1, np.nan)
            angle_masked_list.append(angle_masked)

        angle_mat = pd.concat(angle_masked_list, axis=1)
        df["mean_angle_active"] = angle_mat.mean(axis=1)
    else:
        df["mean_angle_active"] = np.nan

    # balance left/right
    if side_cols:
        df["lateral_balance"] = df[side_cols].sum(axis=1)
    else:
        df["lateral_balance"] = 0.0

    # Fill NaNs in aggregated features with neutral constants
    df["min_dist_active"] = df["min_dist_active"].fillna(0.0)
    df["mean_dist_active"] = df["mean_dist_active"].fillna(0.0)
    df["mean_angle_active"] = df["mean_angle_active"].fillna(0.0)


    # Optionally, you can also fill these NaNs if needed before modeling
    return df





def add_is_moving_feature(df, stim_col="stimulus"):
    """
    Extract a binary feature indicating whether dots are moving
    based on the encoded stimulus string.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    stim_col : str, default="stimulus"
        Column containing encoded stimulus strings.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with an added column:
        - "is_dots_moving" (0 if dots are static, 1 if moving)
    """
    df = df.copy()

    n_dots_list = []
    is_active_list = []

    for stim in df[stim_col]:

        # Cas pause ou valeur non string
        if not isinstance(stim, str) or stim == "inter_stim_pause":
            n_dots_list.append(0)
            is_active_list.append(0)
            continue

        # --- extraire group sizes ---
        m_g = re.search(r"_g(\d+)-(\d+)", stim)
        m_s = re.search(r"_s([0-9.]+)-([0-9.]+)", stim)

        if not m_g or not m_s:
            n_dots_list.append(0)
            is_active_list.append(0)
            continue

        g_left = int(m_g.group(1))
        g_right = int(m_g.group(2))
        n_dots = g_left + g_right

        s_left = float(m_s.group(1))
        s_right = float(m_s.group(2))
        moving = (s_left != 0.0) or (s_right != 0.0)

        is_active_list.append(1 if (n_dots > 0 and moving) else 0)

    df["is_dots_moving"] = is_active_list

    return df




def add_lag_features_for_all(
    df,
    max_lag_seconds,
    fps=30,
    subsample=1,
    exclude_cols=None,
    verbose=True,
):
    """
    Create lagged features for (almost) all columns in a time-series dataframe.

    Assumptions
    ----------
    - df should already contain ONLY numeric features (no categorical / string columns).
    - One row per frame, sorted in time.
    - All columns except `exclude_cols` will be lagged.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    max_lag_seconds : float
        How far back in time we go, in seconds (e.g. 2.0 = 2 seconds).
    fps : int, default=30
        Frames per second of the recording.
    subsample : int, default=1
        Lag step in frames.
        - 1  -> use every frame (t-1, t-2, t-3, ...)
        - 3  -> use one frame out of 3 (t-3, t-6, t-9, ...)
    exclude_cols : list or None
        Columns that should NOT be lagged (e.g. indices, IDs, *_next targets).

    Returns
    -------
    df_lag : pd.DataFrame
        Dataframe with original columns + lagged versions.
    created_lag_cols : list of str
        Names of all lag columns that were created.
    """

    df = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    # ---------- 0) Sanity check: all numeric ----------
    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "add_lag_features_for_all expects a dataframe with only numeric features.\n"
            f"Non-numeric columns found: {non_numeric}"
        )

    # ---------- 1) Convert seconds -> frames ----------
    max_lag_frames = int(max_lag_seconds * fps)
    if max_lag_frames < subsample:
        raise ValueError(
            f"max_lag_seconds={max_lag_seconds} with fps={fps} "
            f"gives max_lag_frames={max_lag_frames}, which is < subsample={subsample}."
        )

    # ---------- 2) Columns to lag ----------
    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    created_lag_cols = []

    # ---------- 3) Create lag features ----------
    for col in candidate_cols:
        for lag in range(subsample, max_lag_frames + 1, subsample):
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = df[col].shift(lag)
            created_lag_cols.append(lag_col)
    if verbose:
        print(
            f"Created {len(created_lag_cols)} lag features "
            f"for {len(candidate_cols)} base columns, "
            f"using lags from {subsample} to {max_lag_frames} frames "
            f"(step={subsample}, horizon ≈ {max_lag_seconds:.2f} s at {fps} fps)."
        )
    
    # ---------- 4) Drop warm-up rows introduced by shifts ----------
    df = df.iloc[max_lag_frames:].copy()
    df = df.reset_index(drop=True)

    return df, created_lag_cols




def polar_pixel_encoding_row(row, r_bins=20, theta_bins=36, max_radius=400.0, n_stimuli=8):
    """
    Encode a single row into a polar pixel grid where:
        - t0 corresponds to θ ≈ 0 (straight ahead)
        - t1, t2... correspond to increasing CCW angles (left)
        - t(theta_bins-1) corresponds to slightly negative θ (right)

    Inputs
    ------
    row : pandas.Series
        Must contain, for each k in [0, n_stimuli):
            - d{k}_stim   : 0/1 (1 = stimulus visible)
            - d{k}_radius : distance of dot k (same unit as max_radius, e.g. pixels)
            - d{k}_angle  : egocentric angle of dot k in radians (around [-pi, pi])

    Returns
    -------
    grid : np.ndarray of shape (r_bins, theta_bins)
        Binary grid: 1 if at least one stimulus falls in that (r, theta) bin, else 0.
    """
    # the grid to fill
    grid = np.zeros((r_bins, theta_bins), dtype=float)

    # Bin edges
    r_edges = np.linspace(0.0, max_radius, r_bins + 1)
    theta_edges = np.linspace(-np.pi, np.pi, theta_bins + 1)

    for k in range(n_stimuli):
        stim_flag = f"d{k}_stim"
        r_col = f"d{k}_radius"
        a_col = f"d{k}_angle"

        # Check that columns exist
        if stim_flag not in row or r_col not in row or a_col not in row:
            continue

        # Inactive stimulus → ignore
        if row[stim_flag] != 1:
            continue

        r = row[r_col]
        theta = row[a_col]

        # Ignore invalid values
        if pd.isna(r) or pd.isna(theta):
            continue

        # Wrap angle to [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Radial and angular binning
        r_idx = np.digitize(r, r_edges) - 1
        t_idx = np.digitize(theta, theta_edges) - 1

        # Clip to stay within grid
        if r_idx < 0 or r_idx >= r_bins:
            continue
        t_idx = min(max(t_idx, 0), theta_bins - 1)

        # Mark the cell as occupied
        grid[r_idx, t_idx] = 1.0

    return grid




def build_polar_pixel_encoding_df(df, r_bins=20, theta_bins=36, max_radius=None, n_stimuli=8):
    """
    Build polar pixel-encoding features for all rows of a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain d{k}_stim, d{k}_radius, d{k}_angle for k in [0, n_stimuli).
    r_bins : int
        Number of radial bins.
    theta_bins : int
        Number of angular bins.
    max_radius : float or None
        Maximum radius for the visual field. If None, it is inferred from the data
        as the max radius observed when a stimulus is active.
    n_stimuli : int
        Number of stimulus slots (default 8).

    Returns
    -------
    encoded_df : pandas.DataFrame
        One row per original frame, with r_bins * theta_bins features:
        columns are named "px_r{i}_t{j}".
    """
    df = df.copy()

    # Détermine max_radius automatiquement si non fourni
    if max_radius is None:
        all_radii = []
        for k in range(n_stimuli):
            stim_flag = f"d{k}_stim"
            r_col = f"d{k}_radius"
            if stim_flag in df.columns and r_col in df.columns:
                active = df[stim_flag] == 1
                all_radii.append(df.loc[active, r_col])
        if all_radii:
            max_radius = pd.concat(all_radii).max()
        else:
            max_radius = 1.0  # fallback arbitraire

    grids = []
    for _, row in df.iterrows():
        grid = polar_pixel_encoding_row(
            row,
            r_bins=r_bins,
            theta_bins=theta_bins,
            max_radius=max_radius,
            n_stimuli=n_stimuli,
        )
        grids.append(grid)

    grids = np.stack(grids, axis=0)           # shape: (n_rows, r_bins, theta_bins)
    flat = grids.reshape(len(df), -1)         # shape: (n_rows, r_bins * theta_bins)

    # Noms de colonnes explicites
    col_names = [
        f"px_r{ri}_t{ti}"
        for ri in range(r_bins)
        for ti in range(theta_bins)
    ]

    encoded_df = pd.DataFrame(flat, index=df.index, columns=col_names)
    return encoded_df




def compute_trial_onset_flag(df):
    """
    Return a binary Series marking trial onset frames (1 = onset).
    Uses the same logic as in the pipeline, but does NOT modify the dataframe.
    """

    # 1) Build the temporary "condition" exactly like in the pipeline
    condition = df["d0_stim"].apply(
        lambda s: "inter_stim_pause" if s in ["none", None, "None", 0] else s
    )

    # 2) Identify stimulus blocks by changes in condition
    stimulus_block_id = (condition != condition.shift(1)).cumsum()

    # 3) Compute index of each frame within its block
    frame_count_stimulus = stimulus_block_id.groupby(stimulus_block_id).cumcount()

    # 4) Trial onset = first frame of each block
    trial_onset_flag = (frame_count_stimulus == 0).astype(int)

    return trial_onset_flag




def plot_spiderweb_grid(grid, ax=None, max_radius=100, title=None):
    """
    Plot a polar pixel grid as a clean circular spider-web:
    - dark gray for 0 (no stimulus)
    - yellow for 1 (stimulus present)
    - arrow at center pointing to the right (fish heading).

    If ax is provided, draw into that axis.
    If ax is None, create a new figure.
    """

    grid = np.asarray(grid)
    r_bins, t_bins = grid.shape

    r_edges = np.linspace(0, max_radius, r_bins + 1)
    theta_edges = np.linspace(0, 2*np.pi, t_bins + 1)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True

    ax.set_aspect("equal")

    # Draw wedges
    for ri in range(r_bins):
        r_in  = r_edges[ri]
        r_out = r_edges[ri + 1]

        for ti in range(t_bins):
            th1 = theta_edges[ti]
            th2 = theta_edges[ti + 1]

            val = grid[ri, ti]

            # Dark gray for 0, yellow for >0
            if val <= 0:
                color = (0.3, 0.3, 0.3, 1.0)   # RGBA dark gray
            else:
                color = (1.0, 1.0, 0.0, 1.0)   # RGBA yellow

            wedge = Wedge(
                center=(0, 0),
                r=r_out,
                theta1=np.degrees(th1),
                theta2=np.degrees(th2),
                width=r_out - r_in,
                facecolor=color,
                edgecolor="black",
                linewidth=0.4
            )
            ax.add_patch(wedge)

    # Arrow showing fish looking to the right (positive x)
    arrow_len = 0.6 * max_radius
    ax.arrow(
        0, 0,
        arrow_len, 0,
        head_width=max_radius * 0.06,
        head_length=max_radius * 0.08,
        length_includes_head=True,
        linewidth=1.2,
        color="black"
    )

    # Limits
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)

    # Remove ticks / labels / frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_frame_on(False)

    if title is not None:
        ax.set_title(title)

    if created_fig:
        plt.show()




def plot_random_spiderweb_grids(px_features,
                                r_bins,
                                theta_bins,
                                n_examples=9,
                                n_cols=3,
                                max_radius=100,
                                random_state=None):
    """
    Randomly sample several rows from px_features and plot their polar pixel grids
    in a big figure with multiple subplots.

    Parameters
    ----------
    px_features : pandas.DataFrame
        Dataframe with pixel features, columns named like 'px_r{i}_t{j}'.
    r_bins : int
        Number of radial bins (used when reshaping).
    theta_bins : int
        Number of angular bins (used when reshaping).
    n_examples : int
        Number of random rows/plots to draw.
    n_cols : int
        Number of subplot columns (e.g. 3 → grids arranged in rows of 3).
    max_radius : float
        Outer radius used for plotting.
    random_state : int or None
        Seed for reproducibility.
    """

    n_rows_total = len(px_features)
    if n_rows_total == 0:
        print("No rows in px_features.")
        return

    n_examples = min(n_examples, n_rows_total)

    rng = np.random.default_rng(random_state)
    row_indices = rng.choice(n_rows_total, size=n_examples, replace=False)

    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows)
    )

    # axes can be 1D if n_rows == 1; normalize to 2D array-like
    if n_rows == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(row_indices):
        r = i // n_cols
        c = i % n_cols

        ax = axes[r, c]

        row = px_features.iloc[idx]
        grid = row.to_numpy().reshape(r_bins, theta_bins)

        title = f"row {idx}"
        plot_spiderweb_grid(grid, ax=ax, max_radius=max_radius, title=title)

    # Turn off unused subplots if any
    for j in range(i + 1, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()