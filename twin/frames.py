import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
from pathlib import Path
import seaborn as sns
import matplotlib.gridspec as gridspec
plt.style.use('default')
import random

import sys
import os

repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(repo_root)
from src import analyse as an




def generate_transformed_datasets(stimuli_df, df, roi_df, output="both", experiment ="GsizeMotionSingle", verbose=True):
    """
    Build transformed datasets (absolute and/or polar) from the three raw experiment files.

    Parameters
    ----------
    stimuli_df : pd.DataFrame
        Trajectory file (e.g. `*_trajectory_*.csv`):
        dot positions in fish-centered millimeter coordinates at stimulus onset.
    df : pd.DataFrame
        PositionTxt file: raw fish positions and orientations (pixels, clockwise).
    roi_df : pd.DataFrame
        ROI file: dish geometry and screen offsets (pixels).
    output : {"absolute", "polar", "both"}, default "both"
        Which representation to return:
        - "absolute": fish + dots in absolute screen coordinates
        - "polar":   fish-centered polar coordinates of the dots
        - "both":    both, in a dict.
    experiment : {"GsizeMotionSingle", "GsizeMotionChoice"}, default "GsizeMotionSingle"
        Experiment configuration used to compute dot centers (left/right vs single cloud).
        Be carefull this pipeline is designed for GsizeMotionSingle experiment. for other experiments additional tests and modifications might be needed.

    Returns
    -------
    dict or dict[str, dict[str, pd.DataFrame]]
        If output == "absolute":
            {"f0": df_f0_abs, ..., "f14": df_f14_abs}
        If output == "polar":
            {"f0": df_f0_polar, ..., "f14": df_f14_polar}
        If output == "both":
            {
                "absolute": {"f0": df_f0_abs, ...},
                "polar":    {"f0": df_f0_polar, ...},
            }
        Each DataFrame combines the fish trajectory with the corresponding
        transformed stimulus positions for all frames of that fish.
    """
    
    stimuli_df, df, roi_df = stimuli_df.copy(), df.copy(), roi_df.copy()

    
    #----------------- Give proper column names to the dataframes -----------------#
    stimuli_df_col_names = ["name", "x", "y", "size", "color", "backg"]
    n_stimuli = len(stimuli_df.columns)//len(stimuli_df_col_names)
    stimuli_df.columns = stimuli_df_col_names * n_stimuli
    
    n_fish = len(df.columns)//3 - n_stimuli
    fish_columns = [f"f{i}_{var}" for i in range(n_fish) for var in ["x", "y", "ori"]]
    stimuli_columns = [f"d{i}_{var}" for i in range(n_stimuli) for var in ["x", "y", "stim"]]
    df.columns = fish_columns + stimuli_columns
    # Keep track of original PositionTxt row for debugging / checks
    df["source_frame"] = np.arange(len(df))
    
    roi_columns = ["xoff", "yoff", "diameter", "x", "y", "radius"]
    roi_df.columns = roi_columns
    
    
    
    
    
    
    #------------------ Sanity check that everything matches -------------------#
                   
    #Check that stimuli df and df_fish have the same number of rows
    #Check that stimuli df and df_fish[stimuli_columns] have the same values in name/stim

    # Select only columns containing x, y, and name
    stimuli_df = stimuli_df.drop(columns=['size', 'color', 'backg'])

    # Rename columns to match the desired format
    new_column_names = [col for i in range(n_stimuli) for col in (f'd{i}_stim', f'd{i}_x', f'd{i}_y')]
    stimuli_df.columns = new_column_names

    # Reorder columns to match the desired structure
    ordered_columns = [col for i in range(n_stimuli) for col in (f'd{i}_x', f'd{i}_y', f'd{i}_stim')]
    stimuli_df = stimuli_df[ordered_columns]    
    
    stim_columns = [f"d{i}_stim" for i in range(n_stimuli)]
    all_true = (df[stim_columns] == stimuli_df[stim_columns]).all().all()

    if verbose :
        if all_true:
            print("All stimuli values between PositionTxt and trajectory files match")
        else:
            print("Some stimuli values between PositionTxt and trajectory files do not match. \n"
                "ATTENTION!!!")
        
        
        
        
    #----------------------- Extract parameters of the experiment -----------------------#
    
    # 1. Calculate the duration of the initial pause
    framerate = 30
    MM_TO_PX = 3.67
    avgROIRadius = round(roi_df['radius'].mean())

    # Identify key indices
    start_interpause_idx = (df["d0_stim"] == "none").idxmax()
    CLstimuli_mask = df["d0_stim"].str.startswith("CLsemi", na=False)
    CLstimuli_start_idx = CLstimuli_mask.idxmax()
    CLstimuli_end_idx = CLstimuli_mask[CLstimuli_start_idx:].idxmin()

    # Compute durations
    CLstimuli_duration = (CLstimuli_end_idx - CLstimuli_start_idx) // framerate
    post_stim_pause_start = (df.loc[CLstimuli_end_idx:]["d0_stim"] == "none").idxmin()
    inter_pause_duration = (post_stim_pause_start - CLstimuli_end_idx) / framerate

    initial_pause = (df["d0_stim"] == "start_none").idxmin() / framerate
    bout_duration = len(df.loc[df["d0_stim"] == "bout"])//framerate

    n_trials_tot = len(df.index[(df['d0_stim'] != "none") & (df['d0_stim'].shift(1) == "none")])
    experimental_conditions = [val for val in df["d0_stim"].unique() if val.startswith("CLsemi")]
    n_trials_per_condition = n_trials_tot // len(experimental_conditions)

    if verbose :
        # Make a nice print that describe and summarize the experiment
        print(f"Experiment type: {experiment}")
        print(f"Initial pause: {initial_pause/60} min")
        print(f"Interstimulus pause: {inter_pause_duration} s")
        print(f"CLstimuli duration: {CLstimuli_duration} s")
        print(f"Bout duration: {bout_duration/60} min")
        print(f"Number of trials: {n_trials_tot}")
        print(f"Number of trials per condition: {n_trials_per_condition}")
        print(f"{len(experimental_conditions)} experimental conditions:{experimental_conditions}")
        
    
    
    
    #------------------------- extract bout and fish dataframes -------------------------#
    
    #remove initial pause
    df["condition"] = np.where(df["d0_stim"] == "none", "inter_stim_pause", df["d0_stim"])

    #save bout data separately and remove from main data
    bout_df = df.loc[df["d0_stim"] == "bout"].copy()
    bout_df = bout_df.drop(columns=stimuli_columns)
    stimuli_bout_df = stimuli_df.loc[stimuli_df["d0_stim"] == "bout"].copy()
    stimuli_bout_df = stimuli_bout_df[['d0_x', 'd0_y', 'd0_stim']]
    stimuli_bout_df = stimuli_bout_df.round(3)
    stimuli_bout_df["condition"] = bout_df["condition"]

    df_selected = df.loc[(CLstimuli_start_idx - int(inter_pause_duration*framerate)):].copy()
    fish_df = df_selected.loc[df["d0_stim"] != "bout"].copy()
    fish_df = fish_df.drop(columns=stimuli_columns)
    stimuli_fish_df = stimuli_df.loc[(CLstimuli_start_idx - int(inter_pause_duration*framerate)):].copy()
    stimuli_fish_df = stimuli_fish_df.loc[stimuli_fish_df["d0_stim"] != "bout"]
    stimuli_fish_df["condition"] = fish_df["condition"]
    
    
    #----------------- Extract information about trial and stimulus block start -----------------#
    
    # fish_df['trial_start'] = (fish_df['condition'] != "inter_stim_pause") & (fish_df['condition'].shift(1) == "inter_stim_pause")

    fish_df['trial_start'] = (fish_df['condition'] == "inter_stim_pause") & (fish_df['condition'].shift(1) != "inter_stim_pause")
    fish_df['trial_id'] = fish_df['trial_start'].cumsum()

    fish_df = fish_df.copy()
    fish_df['stimulus_block_id'] = (fish_df['condition'] != fish_df['condition'].shift(1)).cumsum()
    fish_df['frame_count_stimulus'] = fish_df.groupby('stimulus_block_id').cumcount()
    fish_df['frame'] = np.arange(0, len(fish_df))

    stimuli_fish_df['trial_id'] = fish_df['trial_id']
    stimuli_fish_df['stimulus_block_id'] = fish_df['stimulus_block_id']
    stimuli_fish_df['frame_count_stimulus'] = fish_df['frame_count_stimulus']
    stimuli_fish_df['trial_start'] = fish_df['trial_start']
    stimuli_fish_df['frame'] = np.arange(0, len(stimuli_fish_df))
    
    
    
    
    #----------------- Apply pause-copy logic to each trial independently -----------------#
    #   Replace dot coordinates during inter_stim_pause by copying the first frames of the corresponding stimulus block
    #   Ensures that pauses contain consistent dot positions before stimulus onset
    
    dot_xy_cols = [col for col in stimuli_fish_df.columns if col.startswith('d') and ('_x' in col or '_y' in col)]
    dot_stim_cols = [col for col in stimuli_fish_df.columns if col.startswith('d') and '_stim' in col]

    stimuli_fish_df  = stimuli_fish_df.groupby('trial_id', group_keys=False)[stimuli_fish_df.columns].apply(an.pause_copy_trajectory,
                                                                                        dot_xy_cols=dot_xy_cols,
                                                                                           dot_stim_cols=dot_stim_cols)
    
    
    
    
    #---------- Computes left/right dot counts, converts coordinates to millimeters, and generates absolute stimulus trajectories for each fish across all trials -----#
    
    mask_left = stimuli_fish_df.filter(like='x').gt(0)
    mask_right = stimuli_fish_df.filter(like='x').lt(0)
    stimuli_fish_df['n_dots_left'] = mask_left.sum(axis=1)
    stimuli_fish_df['n_dots_right'] = mask_right.sum(axis=1)

    stimuli_fish_df[stimuli_fish_df.filter(like='x').columns] *= MM_TO_PX
    stimuli_fish_df[stimuli_fish_df.filter(like='y').columns] *= MM_TO_PX

    stimuli_allfish_dict = {}
    for f in range(n_fish):
        abs_trials = []
        for block_id, stim_group in stimuli_fish_df.groupby('stimulus_block_id'):
            # Use the corresponding fish row at trial start (without transformation).
            # (Here, frames_before is 0 as in your code.)
            fish_row = fish_df.loc[stim_group.index[0]]
            trial_abs = an.compute_stimuli_absolute(f, stim_group, fish_row, n_stimuli)
            abs_trials.append(trial_abs)

        stimuli_abs_df = pd.concat(abs_trials)
        extra_cols = stimuli_fish_df.columns[n_stimuli*len(["x", "y", "stim"]):]
        stimuli_abs_df = pd.concat([stimuli_abs_df, stimuli_fish_df[extra_cols]], axis=1)

        stimuli_allfish_dict[f'f{f}'] = stimuli_abs_df
        
        
        
        
    # ---------------------- Invert y axis and fish orientation to ccw ----------------------------#
    fish_df, stimuli_allfish_dict = an.transform_coordinates(fish_df, stimuli_allfish_dict, roi_df)



    # -------------------------------- Calculate dots center --------------------------------------#
    stimuli_allfish_dict = an.compute_stimuli_centers(stimuli_allfish_dict, n_stimuli, mask_left, mask_right, experiment)
    
    
    
    # ------------------------ Calculate how many dots are out of rois -----------------------------#
    stimuli_allfish_dict = an.calculate_dots_out(stimuli_allfish_dict, n_stimuli, roi_df, experiment, mask_left, mask_right)


    # --------------------- Compute absolute positions of fish and stimuli --------------------------#
    # For each fish, create a datafame with fish x, y, ori and all dots x, y stim from stimuli_allfish_dict and frame number and condition. Save as a csv file each fish dataframe.
    abs_dict = {}
    
    for f in range(n_fish):
        fish_key = f'f{f}'
        fish_data = fish_df[[f'f{f}_x', f'f{f}_y', f'f{f}_ori', 'condition', 'frame',  'source_frame']].copy()
        stimuli_data = stimuli_allfish_dict[fish_key].copy()
        stimuli_data = stimuli_data.drop(columns=['trial_id', 'condition', 'stimulus_block_id', 'frame_count_stimulus', 'trial_start', 'frame', 'n_dots_left', 'n_dots_right'])
        combined_df = pd.concat([fish_data.reset_index(drop=True), stimuli_data.reset_index(drop=True)], axis=1)
        #remove rowa where dot0_stim is 'none'
        combined_df = combined_df[combined_df['d0_stim'] != 'none']
        combined_df = combined_df.rename(columns={'condition': 'stimulus'})
        #reorder columns by frame, condition, fish x, fish y, fish ori, dot1 x, dot1 y, dot1 stim, dot2 x, dot2 y, dot2 stim, ...
        ordered_columns = ['frame',  'source_frame', 'stimulus', f'f{f}_x', f'f{f}_y', f'f{f}_ori']

        for i in range(n_stimuli):
            ordered_columns.extend([f'd{i}_x', f'd{i}_y', f'd{i}_stim'])
        combined_df = combined_df[ordered_columns]
        abs_dict[fish_key] = combined_df
        
        
    # --------------------- Compute polar coordinates of stimuli for each fish --------------------------#
    # For each fish, create a datafame with all dots x, y stim from abs_dict and frame number and condition. Save as a csv file each fish dataframe.    
    polar_dict = {}
        
    n_stimuli = 8    
        
    for fish_key in abs_dict:
        # Copy of the absolute DF for this fish
        df_polar = abs_dict[fish_key].copy()

        # Columns of the fish corresponding to this fish_key (f0, f1, ...)
        fish_x_col = f"{fish_key}_x"
        fish_y_col = f"{fish_key}_y"
        fish_ori_col = f"{fish_key}_ori"

        # Initialize new columns angle / radius for each dot
        for k in range(n_stimuli):
            df_polar[f"d{k}_angle"] = np.nan
            df_polar[f"d{k}_radius"] = np.nan

        # loop on rows
        for idx, row in df_polar.iterrows():
            f_ori = row[fish_ori_col]
            fx = row[fish_x_col]
            fy = row[fish_y_col]

            for k in range(n_stimuli):
                stim_col = f"d{k}_stim"
                x_col = f"d{k}_x"
                y_col = f"d{k}_y"
                # If the columns do not exist (for safety)
                if x_col not in df_polar.columns or y_col not in df_polar.columns:
                    continue
                #Compute coordinates of stimulus in a system centered on the fish
                d_x = row[x_col] - fx
                d_y = row[y_col] - fy
                d_ori = (np.arctan2(d_y, d_x) + 2 * np.pi) % (2 * np.pi)
                if d_ori > f_ori:
                    angle_diff = d_ori - f_ori
                else:
                    angle_diff = 2 * np.pi - (f_ori - d_ori)
                radius = np.sqrt(d_x**2 + d_y**2)
                # Write in the new columns
                df_polar.at[idx, f"d{k}_angle"] = angle_diff
                df_polar.at[idx, f"d{k}_radius"] = radius

        # Drop columns d{k}_x and d{k}_y, but keep d{k}_stim
        drop_cols = []
        for k in range(n_stimuli):
            drop_cols.append(f"d{k}_x")
            drop_cols.append(f"d{k}_y")

        df_polar = df_polar.drop(columns=drop_cols, errors="ignore")
        
        polar_dict[fish_key] = df_polar
        
    
    if output == "absolute":
        return abs_dict
    elif output == "polar":
        return polar_dict          
    elif output == "both":
        return {
            "absolute": abs_dict,
            "polar": polar_dict,
        }
    else:
        raise ValueError("output must be 'absolute', 'polar', or 'both'")
    
    
    
    
    
    
    
    
    
    
    
    

    
def run_diagnostics(abs_dict=None, polar_dict=None, stimuli_df=None, fish_df=None, n_examples=5):
    """
    Run sanity checks on transformed datasets.

    This diagnostic routine can take:
        - abs_dict  : dictionary of absolute-coordinate DataFrames
        - polar_dict: dictionary of polar-coordinate DataFrames
        - stimuli_df : pd.DataFrame
            Trajectory file (e.g. `*_trajectory_*.csv`):
            dot positions in fish-centered millimeter coordinates at stimulus onset.
        - fish_df : pd.DataFrame
            PositionTxt file: raw fish positions and orientations (pixels, clockwise).
        - n_examples: number of random frames to visually compare

    The function performs three possible checks:
        1. If abs_dict is provided:
            - plots the density of dot angles at trial onset
              (should cluster around ±90° for lateral stimuli)

        2. If polar_dict is provided:
            - plots the density of polar angles at trial onset
              (same expected structure as absolute)

        3. If both are provided + stimuli_df and fish_df also provided:
            - randomly selects example frames and compares:
                * INITIAL view (before transformation) → fish plotted as an oriented arrow
                * ABSOLUTE view → fish plotted as an oriented arrow
                * POLAR view     → fish at center, facing 0 rad
              The three views must show consistent dot positions.

    Returns
    -------
    None
        The function only displays diagnostic plots.
    """
    
    if abs_dict is None and polar_dict is None:
        raise ValueError("run_diagnostics: you must provide at least abs_dict or polar_dict.")

    print("\n===== RUNNING DIAGNOSTICS =====\n")
    
    fish_key = f'f{random.randint(0, 14)}'
    
    
    if abs_dict is not None :
        print("[ABSOLUTE COORDINATES DIAGNOSTIC]")
        print(
            "- We look at the distribution of dot ANGLES in the fish-centered frame\n"
            "  at stimulus onset (frame_count_stimulus == 0).\n"
            "- What you should see:\n"
            "    * For lateral stimuli, the angle density should concentrate\n"
            "      around ±90° (≈ π/2 and 3π/2), i.e. on the sides of the fish.\n"
        )
        
        df_abs = abs_dict[fish_key]

        # Detect when a trial starts
        df_abs["condition"] = np.where(df_abs["d0_stim"] == "none",
                           "inter_stim_pause",
                           df_abs["d0_stim"])
        df_abs['trial_start'] = (df_abs['condition'] == "inter_stim_pause") & (df_abs['condition'].shift(1) != "inter_stim_pause")
        df_abs['trial_id'] = df_abs['trial_start'].cumsum()
        df_abs['stimulus_block_id'] = (df_abs['condition'] != df_abs['condition'].shift(1)).cumsum()
        df_abs['frame_count_stimulus'] = df_abs.groupby('stimulus_block_id').cumcount()
        df_abs['frame'] = np.arange(len(df_abs))
        
        angle_list = []
        for index, row in df_abs.iterrows():
            f_ori = row[f"{fish_key}_ori"]
            f_x = row[f"{fish_key}_x"]
            f_y = row[f"{fish_key}_y"]
            
            # We are interested only on the beginning of trials
            if row['frame_count_stimulus'] != 0:
                continue
            
            for n_stim in range(8) :
                if row[f"d{n_stim}_stim"] in ["none", None, np.nan, "bout", "inter_stim_pause"]:
                    continue
                d_x = row[f"d{n_stim}_x"]
                d_y = row[f"d{n_stim}_y"]

                d_x_centered = d_x - f_x
                d_y_centered = d_y - f_y

                d_ori = (np.arctan2(d_y_centered, d_x_centered) + 2 * np.pi) % (2 * np.pi)
                if d_ori>f_ori : angle_diff = d_ori - f_ori
                else: angle_diff = 2*np.pi - (f_ori - d_ori)

                angle_list.append(angle_diff)
              

        angle_series = pd.Series(angle_list, name="angle_diff")

        ax = angle_series.plot(kind="density")  # KDE density plot
        ax.set_xlabel("Angle difference (rad)")
        ax.set_ylabel("Density")
        ax.set_title("Circular density of angle between stimulus position and fish orientation for the ABSOLUTE dataframe")
        plt.show()
    
    
    
    
    
    
    if polar_dict is not None : 
        print("[POLAR COORDINATES DIAGNOSTIC]")
        print(
            "- We look at the distribution of dot ANGLES in the fish-centered frame\n"
            "  at stimulus onset (frame_count_stimulus == 0).\n"
            "- What you should see:\n"
            "    * For lateral stimuli, the angle density should concentrate\n"
            "      around ±90° (≈ π/2 and 3π/2), i.e. on the sides of the fish.\n"
        )
        
        df_polar = polar_dict[fish_key]

        # Detect when a trial starts
        df_polar["condition"] = np.where(df_polar["d0_stim"] == "none",
                           "inter_stim_pause",
                           df_polar["d0_stim"])
        df_polar['trial_start'] = (df_polar['condition'] == "inter_stim_pause") & (df_polar['condition'].shift(1) != "inter_stim_pause")
        df_polar['trial_id'] = df_polar['trial_start'].cumsum()
        df_polar['stimulus_block_id'] = (df_polar['condition'] != df_polar['condition'].shift(1)).cumsum()
        df_polar['frame_count_stimulus'] = df_polar.groupby('stimulus_block_id').cumcount()
        df_polar['frame'] = np.arange(len(df_polar))
        
        angle_list = []
        for index, row in df_polar.iterrows():
            # We are interested only on the beginning of trials
            if row['frame_count_stimulus'] != 0:
                continue
            
            for n_stim in range(8) :
                if row[f"d{n_stim}_stim"] in ["none", None, np.nan, "bout", "inter_stim_pause"]:
                    continue
                d_angle = row[f"d{n_stim}_angle"]
                angle_list.append(d_angle)
              
        angle_series = pd.Series(angle_list, name="angle_diff")

        ax = angle_series.plot(kind="density")  # KDE density plot
        ax.set_xlabel("Angle difference (rad)")
        ax.set_ylabel("Density")
        ax.set_title("Circular density of angle between stimulus position and fish orientation for the POLAR dataframe")
        plt.show()
        
        
        
        
    
    if abs_dict is not None and polar_dict is not None and stimuli_df is not None and fish_df is not None:
        compare_raw_abs_polar(abs_dict, polar_dict, stimuli_df, fish_df, fish_key="f0", n_examples=n_examples)
        
        
    
    
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       

def compare_raw_abs_polar(abs_dict,
                          polar_dict,
                          stimuli_df,
                          fish_df,
                          fish_key="f0",
                          n_examples=5):
    """
    Compare, for a few frames, three views of the SAME underlying data:
    1) A minimal "raw" reconstruction from Trajectory + PositionTxt
    2) The ABSOLUTE dictionary output of the pipeline
    3) The POLAR dictionary output of the pipeline

    This is meant as a sanity check: the three plots should tell the same story.
    """
    
    print("[RAW vs ABSOLUTE vs POLAR DIAGNOSTIC]")
    print(
        "- For a few randomly chosen frames, we plot three views of the SAME underlying data:\n"
        "    1) RECONSTRUCTION  → minimal reconstruction from PositionTxt + Trajectory\n"
        "    2) ABSOLUTE → pipeline output in screen coordinates (abs_dict)\n"
        "    3) POLAR    → pipeline output in fish-centered polar coordinates (polar_dict)\n"
        "\n"
        "- RECONSTRUCTION view:\n"
        "    * Uses only the essential geometry from the original files.\n"
        "    * Reconstructs where dots should be on the screen using the fish\n"
        "      position/orientation at trial onset and the mm trajectories.\n"
        "    * Does NOT reapply all pipeline transforms – it is a minimal\n"
        "      reconstruction of the experiment used to check the pipeline.\n"
        "\n"
        "- ABSOLUTE view (abs_dict):\n"
        "    * Shows the fish at its absolute position in the arena and the dots\n"
        "      in the absolute screen coordinate system.\n"
        "    * Geometry (left/right, front/behind, distances) should be consistent\n"
        "      with the minimal reconstruction view, even if the global origin is different.\n"
        "\n"
        "- POLAR view (polar_dict):\n"
        "    * Fish is always at the center and looks toward angle 0 (to the right).\n"
        "    * Dot angles/radii should match what you see in the ABSOLUTE / RAW views.\n"
        "      For example: if ABSOLUTE shows dots to the right of the fish, then in\n"
        "      POLAR these dots should cluster around angle 3*pi/2; if they are on the left,\n"
        "      angles should be around π/2.\n"
        "\n"
        "- In short:\n"
        "    >>> The three subplots must tell the SAME story.\n"
        "    RECONSTRUCTION is a simple ground-truth reconstruction; ABSOLUTE and POLAR are the\n"
        "    pipeline outputs. Any systematic mismatch (left/right flipped, wrong\n"
        "    orientation, wrong distances, unit errors) indicates a problem in the\n"
        "    transformation pipeline.\n"
    )

    # --------- 1. Prepare raw data (Trajectory + PositionTxt) in a consistent format --------- #
    stimuli_df = stimuli_df.copy()
    fish_df = fish_df.copy()

    # Column naming as in the pipeline
    stimuli_df_col_names = ["name", "x", "y", "size", "color", "backg"]
    n_stimuli = len(stimuli_df.columns) // len(stimuli_df_col_names)
    stimuli_df.columns = stimuli_df_col_names * n_stimuli

    n_fish = len(fish_df.columns) // 3 - n_stimuli
    fish_columns = [f"f{i}_{var}" for i in range(n_fish) for var in ["x", "y", "ori"]]
    stimuli_columns = [f"d{i}_{var}" for i in range(n_stimuli) for var in ["x", "y", "stim"]]
    fish_df.columns = fish_columns + stimuli_columns

    # Drop size/color/backg and reorder stimuli_df to d0_x, d0_y, d0_stim, d1_x, ...
    stimuli_df = stimuli_df.drop(columns=["size", "color", "backg"])
    new_column_names = [
        col for i in range(n_stimuli) for col in (f"d{i}_stim", f"d{i}_x", f"d{i}_y")
    ]
    stimuli_df.columns = new_column_names
    ordered_columns = [
        col for i in range(n_stimuli) for col in (f"d{i}_x", f"d{i}_y", f"d{i}_stim")
    ]
    stimuli_df = stimuli_df[ordered_columns]

    # Replace stimulus values in fish_df by the ones from stimuli_df (as in your plot() code)
    for k in range(n_stimuli):
        for col in (f"d{k}_x", f"d{k}_y", f"d{k}_stim"):
            fish_df[col] = stimuli_df[col].values

    # Add a source_frame index to match the pipeline outputs
    fish_df["source_frame"] = np.arange(len(fish_df))
    
    # --- Detect trials / stimulus blocks on RAW fish_df (same logic as pipeline) ---
    fish_df["condition"] = np.where(
        fish_df["d0_stim"] == "none",
        "inter_stim_pause",
        fish_df["d0_stim"],
    )

    fish_df["trial_start"] = (
        (fish_df["condition"] == "inter_stim_pause")
        & (fish_df["condition"].shift(1) != "inter_stim_pause")
    )
    fish_df["trial_id"] = fish_df["trial_start"].cumsum()

    fish_df["stimulus_block_id"] = (
        fish_df["condition"] != fish_df["condition"].shift(1)
    ).cumsum()
    fish_df["frame_count_stimulus"] = (
        fish_df.groupby("stimulus_block_id").cumcount()
    )

    # --------- 2. Select candidate frames from abs_dict / polar_dict using source_frame --------- #
    df_abs = abs_dict[fish_key]
    df_pol = polar_dict[fish_key]

    if "source_frame" not in df_abs.columns:
        raise ValueError(
            "compare_raw_abs_polar expects 'source_frame' in abs_dict[fish_key].\n"
            "Make sure your pipeline adds this column before building abs_dict."
        )

    # Keep frames with at least one active stimulus in abs_dict
    active_mask = np.zeros(len(df_abs), dtype=bool)
    for k in range(n_stimuli):
        stim_col = f"d{k}_stim"
        if stim_col not in df_abs.columns:
            continue
        vals = df_abs[stim_col]
        mask_k = ~vals.isin(["none", None, "None", "bout", "inter_stim_pause"]) & ~vals.isna()
        active_mask |= mask_k

    df_candidates = df_abs.loc[active_mask]

    if df_candidates.empty:
        print(f"No frames with active stimuli found for {fish_key}.")
        return

    source_frames = df_candidates["source_frame"].unique()
    n_examples = min(n_examples, len(source_frames))
    chosen_source_frames = random.sample(list(source_frames), n_examples)

    print(f"\nComparing RAW vs ABSOLUTE vs POLAR for {n_examples} frames of {fish_key}.\n")

    # --------- 3. For each chosen source_frame, plot the three views side by side --------- #
    raw_fish_col_prefix = fish_key

    for sf in chosen_source_frames:
        # RAW row from initial PositionTxt/Trajectory
        row_raw = fish_df.loc[fish_df["source_frame"] == sf]
        if row_raw.empty:
            continue
        row_raw = row_raw.iloc[0]

        # ABS row from abs_dict
        row_abs = df_abs.loc[df_abs["source_frame"] == sf]
        if row_abs.empty:
            continue
        row_abs = row_abs.iloc[0]

        # POLAR row from polar_dict
        row_pol = df_pol.loc[df_pol["source_frame"] == sf]
        if row_pol.empty:
            continue
        row_pol = row_pol.iloc[0]

        # --- Figure with 3 subplots: RAW | ABSOLUTE | POLAR --- #
        fig = plt.figure(figsize=(15, 4))
        ax_raw = fig.add_subplot(1, 3, 1)
        ax_abs = fig.add_subplot(1, 3, 2)
        ax_pol = fig.add_subplot(1, 3, 3, projection="polar")

        # ============== SUBPLOT 1 : RAW VIEW (minimal reconstruction) ============== #
        arrow_len = 12
        MM_TO_PX = 3.67  # 1 mm -> 3.67 px

        # 1) fish orientation to draw the arrow
        f_ori_draw = row_raw[f"{raw_fish_col_prefix}_ori"]  # clockwise, 0 = right

        forward_draw_x = np.cos(f_ori_draw)
        forward_draw_y = -np.sin(f_ori_draw) # in PositionTxt y goes down so we multiply by -1

        # --- 1) Compute fish position relative to block onset ---
        block_id   = row_raw["stimulus_block_id"]
        block_rows = fish_df[fish_df["stimulus_block_id"] == block_id]
        onset_rows = block_rows[block_rows["frame_count_stimulus"] == 0]

        if onset_rows.empty:
            onset_row = row_raw  
        else:
            onset_row = onset_rows.iloc[0]

        # fish position in this raw (repère raw, top-left)
        fx_curr = row_raw[f"{raw_fish_col_prefix}_x"]
        fy_curr = row_raw[f"{raw_fish_col_prefix}_y"]

        # fish position from beginning of the trial
        fx_onset = onset_row[f"{raw_fish_col_prefix}_x"]
        fy_onset = onset_row[f"{raw_fish_col_prefix}_y"]

        # compute position of fish from the position at the beginning of the trial --> have same origin as stimulus
        fx_plot = fx_curr - fx_onset
        fy_plot = -(fy_curr - fy_onset)

        # --- 2) Draw fish
        ax_raw.scatter(fx_plot, fy_plot, s=80, color="deepskyblue")
        ax_raw.arrow(
            fx_plot, fy_plot,
            arrow_len * forward_draw_x,
            arrow_len * forward_draw_y,
            head_width=2,
            head_length=3,
            length_includes_head=True,
            color="deepskyblue",
            linewidth=1.5,
        )

        # 3) Orientation at trial onset
        f_ori_ego = onset_row[f"{raw_fish_col_prefix}_ori"]  # clockwise, 0 = right

        forward_x = np.cos(f_ori_ego)
        forward_y = -np.sin(f_ori_ego)
        left_x    = np.sin(f_ori_ego)
        left_y    = np.cos(f_ori_ego)

        # stock in order to do zoom after
        xs_raw = [fx_plot]
        ys_raw = [fy_plot]

        colors = plt.cm.viridis(np.linspace(0, 1, n_stimuli))

        for k, c in zip(range(n_stimuli), colors):
            stim_col = f"d{k}_stim"
            x_col    = f"d{k}_x"
            y_col    = f"d{k}_y"

            # Check columns exist
            if any(col not in fish_df.columns for col in [stim_col, x_col, y_col]):
                continue

            stim_val = row_raw[stim_col]
            if stim_val in ["none", None, 0, "None", "bout", "inter_stim_pause"] or pd.isna(stim_val):
                continue

            x_ego_mm = row_raw[x_col]
            y_ego_mm = row_raw[y_col]
            if pd.isna(x_ego_mm) or pd.isna(y_ego_mm):
                continue

            # mm -> px
            x_ego = x_ego_mm * MM_TO_PX
            y_ego = y_ego_mm * MM_TO_PX

            # Ego -> screen (coordinates centered on fish position at onset)
            dx = x_ego * left_x + y_ego * forward_x
            dy = x_ego * left_y + y_ego * forward_y

            ax_raw.scatter(dx, dy, s=40, color=c)
            ax_raw.text(dx + 1, dy + 1, str(k), fontsize=7, ha="center", va="center")

            xs_raw.append(dx)
            ys_raw.append(dy)

        # ------------------ AUTO-ZOOM ------------------
        margin = 10
        xmin, xmax = min(xs_raw) - margin, max(xs_raw) + margin
        ymin, ymax = min(ys_raw) - margin, max(ys_raw) + margin

        ax_raw.set_xlim(xmin, xmax)
        ax_raw.set_ylim(ymin, ymax)
        ax_raw.set_aspect("equal", adjustable="box")
        ax_raw.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax_raw.set_xlabel("x (screen right)")
        ax_raw.set_ylabel("y (screen up)")
        ax_raw.set_title(f"RECONSTRUCTION – fish 0, source_frame {int(sf)}")

        # ============== SUBPLOT 2 : ABSOLUTE VIEW (pipeline abs_dict) ============== #
        fx_abs = row_abs[f"{fish_key}_x"]
        fy_abs = row_abs[f"{fish_key}_y"]
        f_ori_abs = row_abs[f"{fish_key}_ori"]  # CCW, 0 = right

        arrow_len_abs = 20
        ux = np.cos(f_ori_abs)
        uy = np.sin(f_ori_abs)

        ax_abs.scatter(fx_abs, fy_abs, s=50, color="tab:blue")
        ax_abs.arrow(
            fx_abs, fy_abs,
            arrow_len_abs * ux, arrow_len_abs * uy,
            head_width=5,
            head_length=8,
            length_includes_head=True,
            color="tab:blue",
            linewidth=1.5,
        )

        xs_abs = [fx_abs]
        ys_abs = [fy_abs]

        for k in range(n_stimuli):
            stim_col = f"d{k}_stim"
            x_col = f"d{k}_x"
            y_col = f"d{k}_y"

            if stim_col not in df_abs.columns:
                continue

            stim_val = row_abs[stim_col]
            if stim_val in ["none", None, "None", "bout", "inter_stim_pause"] or pd.isna(stim_val):
                continue

            if x_col not in df_abs.columns or y_col not in df_abs.columns:
                continue

            dx = row_abs[x_col]
            dy = row_abs[y_col]
            if np.isnan(dx) or np.isnan(dy):
                continue

            ax_abs.scatter(dx, dy, s=40, color="tab:orange")
            ax_abs.text(dx, dy, str(k), fontsize=7, ha="center", va="center")

            xs_abs.append(dx)
            ys_abs.append(dy)

        if len(xs_abs) > 1:
            margin = 20
            xmin, xmax = min(xs_abs) - margin, max(xs_abs) + margin
            ymin, ymax = min(ys_abs) - margin, max(ys_abs) + margin
            ax_abs.set_xlim(xmin, xmax)
            ax_abs.set_ylim(ymin, ymax)

        ax_abs.set_aspect("equal", adjustable="box")
        ax_abs.set_title(f"ABS – {fish_key}, source_frame {int(sf)}")
        ax_abs.grid(alpha=0.3, linestyle="--", linewidth=0.5)

        # ============== SUBPLOT 3 : POLAR VIEW (pipeline polar_dict) ============== #
        head_r = 3.0
        arrow_theta = 0  # fish looks to the right

        ax_pol.annotate(
            "",
            xy=(arrow_theta, head_r),
            xytext=(arrow_theta, 0),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2),
        )

        radii_pol = []
        for k in range(n_stimuli):
            stim_col = f"d{k}_stim"
            r_col = f"d{k}_radius"
            a_col = f"d{k}_angle"

            if stim_col not in df_pol.columns or r_col not in df_pol.columns or a_col not in df_pol.columns:
                continue

            stim_val = row_pol[stim_col]
            if stim_val in ["none", None, "None", "bout", "inter_stim_pause"] or pd.isna(stim_val):
                continue

            r = row_pol[r_col]
            a = row_pol[a_col]
            if np.isnan(r) or np.isnan(a):
                continue

            radii_pol.append(r)
            ax_pol.scatter(a, r, s=40, color="tab:orange")
            ax_pol.text(a, r, str(k), fontsize=7, ha="center", va="center")

        if radii_pol:
            R_pol = max(radii_pol) * 1.2
        else:
            R_pol = 1.0
        ax_pol.set_ylim(0, R_pol)

        ax_pol.set_title(f"POLAR – {fish_key}, source_frame {int(sf)}")
        ax_pol.set_theta_zero_location("E")  # 0 rad = right
        ax_pol.set_theta_direction(1)        # CCW
        ax_pol.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
            
