from . import functions
import pandas as pd


def preprocessing_pipeline(
    df_raw: pd.DataFrame,
    *,
    subsample: int = 5,
    max_lag_seconds: int = 3,
    horizon_frames: int = 5,
    representation: str = "absolute",
    cols_to_lag: list[str] = None):
    """

    Input columns (df_raw)
    ------------------------------
    Required "fish state" + identifiers:
        - frame: int (monotonic within a trial)
        - source_frame: int (original frame index corresponding to frames before using transformes_data_pipeline.py)
        - fish_id: identifier (int/str)
        - f_x, f_y: float (fish position)
        - f_ori: float (fish orientation in radians)
        - stimulus: any (trial condition id)

    Required stimulus columns, for each k in {0,...,n_stimuli-1}:
        - d{k}_stim: {0,1}   (active flag)
        - d{k}_x, d{k}_y: float (stimulus position in the SAME coordinate system as f_x,f_y
          *as you are using it*, i.e. "absolute" in your code)

    Concretely, df_raw must contain:
        ['frame','source_frame','stimulus','f_x','f_y','f_ori',
         'd0_x','d0_y','d0_stim', ... , 'd7_x','d7_y','d7_stim',
         'fish_id']

    Parameters
    ----------
    subsample:
        Lag step in frames used in the model input (e.g. 5 means lag_5, lag_10, ...).
    max_lag_seconds:
        Max history length to encode via lags (seconds).
        max_lag_frames = int(max_lag_seconds * fps).
    horizon_frames:
        Prediction horizon in frames. Targets are defined at t+horizon_frames.
    cols_to_lag:
        List of base feature columns that will be lagged
    representation:
        "absolute" or "polar".

    What this pipeline does
    -----------------------
    1) Create base targets (teacher-forcing target at t+horizon_frames)
       - f_speed(t): computed from (f_x,f_y) difference (or if you already have it, keep it)
       - d_ori(t): computed as wrapped angle difference of f_ori (or if already present, keep it)
       - f_speed_next(t) = f_speed(t + horizon_frames)
       - d_ori_next(t)  = d_ori(t  + horizon_frames)
       - Rows near the end (where t+horizon_frames is out of bounds) are dropped.

    2) Add engineered features at time t (NO lagging yet)
 
    3) Add lag features for selected columns (model inputs)
       - Define cols_to_lag = [...] (e.g. ["f_speed", "d_ori", "is_dots_moving", ...]).
       - For each base in cols_to_lag, create:
            base_lag_5, base_lag_10, ..., base_lag_{max_lag_frames}
         stepping by `subsample`.
       - Rows that don’t have enough history to build lags are dropped (first max_lag_frames frames).

    4) Return model-ready dataframe
       Output columns include:
       - all current-time features (base + engineered, at time t)
       - all lagged features (at time t-5, t-10, ..., t-max_lag_frames)
       - targets f_speed_next, d_ori_next (at time t+horizon_frames)

    Returns
    -------
    df_model : pd.DataFrame
        A clean dataset ready for training
    df_base : before engineered features and lag, used for sampling
    """
    
    df_raw = functions.add_is_moving_feature(df_raw)
    
    #### There is a non sens to give conditional experiment information to the model. We want to give the present and past full state position of the fish and its visual field.
    df_raw = df_raw.drop(columns=['stimulus'])
    # transform dk_stim for each stimulus k into 0 if the feature is 'none' else 1
    for k in range(8):
        df_raw[f'd{k}_stim'] = df_raw[f'd{k}_stim'].apply(lambda x: 0 if x == 'none' else 1)
        df_raw[f'd{k}_x'] = df_raw[f'd{k}_x']*df_raw[f'd{k}_stim']
        df_raw[f'd{k}_y'] = df_raw[f'd{k}_y']*df_raw[f'd{k}_stim']
    df_raw = df_raw.drop(columns=['fish_id']) # Make no sens to keep fish_id for prediction
    df_raw = df_raw.drop(columns=['frame']) # Make no sens to keep frame for prediction
    df_raw = df_raw.drop(columns=['source_frame'])
    
    #### Remove NaN and constant features, and 100% correlated features 
    print("------------ Raw dataframe ------------")
    df_raw, _ = functions.remove_nan_plus_constant_and_correlated_features(df_raw, remove_nan_threshold=100, corr_threshold=1.0)

    df_raw = functions.compute_d_orientation(df_raw)
    df_raw = functions.compute_speed(df_raw)
    
    df_base = functions.make_egocentric(df_raw)
    
    df_feature_engineered = functions.add_engineered_features(df_base, representation=representation, n_stimuli=8)

    exclude_cols = [c for c in df_feature_engineered.columns if c not in cols_to_lag]

    df, _ = functions.add_lag_features_for_all(df_feature_engineered, max_lag_seconds=max_lag_seconds, subsample=subsample, exclude_cols=exclude_cols)

    df['f_speed_next'] = df['f_speed'].shift(-horizon_frames)
    df['d_ori_next'] = df['d_ori'].shift(-horizon_frames)

    # Drop only rows where future targets are missing
    df = df.dropna(subset=["f_speed_next", "d_ori_next"])


    return df, df_base