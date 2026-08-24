# <center> Zebrafish Behavior Modeling 

## Quick start — generate trajectories from a stimulus file

Give the tool a Bonsai **stimulus trajectory file** and a trained model; it simulates fish and
writes an mp4. No training, no real fish data, nothing to download separately.

```bash
git clone -b script-workflow https://github.com/arthurEpfl/mp_semiCL_behavior.git
cd mp_semiCL_behavior
uv sync                                   # or: pip install -e .
jupyter lab notebooks/stimulus_tool.ipynb # run top to bottom
```

Output lands in `tool_videos/`: a 2×3 mp4 (two stimulus blocks × three simulated fish) plus the
per-frame panel CSVs behind it.

**Cost:** rollouts are fast — about **3 s per simulated fish** over a 175-minute programme. The
**mp4 is the slow step, ~3.5 minutes**, because it renders 6 panels × 1800 frames; it runs in a
subprocess so it cannot take down the kernel.

### Using your own stimulus file

Point `TRAJ_FILE` at it in the config cell. It must keep the rig's format: headerless CSV, one row
per frame at 30 fps, **8 dot slots × 6 fields** in the order `name, x, y, size, color, backg`
(48 columns), with

- `x`, `y` in **millimetres relative to the fish**, left = positive `x`;
- `size` 4 when a dot is on, 0 when off, and unused slots written literally as `none,0,0,0,1,0`;
- block labels matching `CLsemi..._g{L}-{R}_s{sL}-{sR}` (e.g. `CLsemi1800_g4-0_s3.5-3.5`), and
  `none` for inter-stimulus pauses.

Anything not starting with `CLsemi` is treated as no stimulus. Because the paradigm is
semi-closed-loop, the dot track is anchored to the *simulated* fish's pose at each block onset — so
the same file produces a different absolute dot track on every rollout, exactly as it would for a
different real fish.

### What ships in `assets/` (≈2 MB)

| File | What it is |
|---|---|
| `stimulus/…_180min.csv.gz` | An example stimulus programme. This is rig **input** — the experimental design — and contains no animal data. |
| `ROIdef2025-02-20T12_09_16.csv` | Physical dish geometry. |
| `models/ladder_b_indiv.pt` | The **J10** model (`mlp_joint_h128_shoal_contain_kfb_vig_indiv`). |
| `real_reference.npz` | A ~13 kB aggregate summary of the recorded experiment: an occupancy histogram for start poses, plus the real comparison curves. |

The raw recordings (`PositionTxt`, `df_transformed/`, `bouts_GsizeMotionSingle.csv`) are **not**
published. `real_reference.npz` holds histograms only — no fish identity, no timing, no
trajectories — which is enough for the tool to place a plausible starting fish and to draw the real
baseline in the sanity panel. If you do have the full dataset locally under `data/`, the notebook
detects it and uses it automatically.

The sanity panel's real curves are a **population baseline, not a matched control**: they are
reusable for any programme whose conditions stay in the recorded vocabulary (N ∈ {1,2,4,6,8},
moving/static), but a programme with a different *mix* of conditions will shift the marginal
histograms for compositional reasons. The following-index-by-N panel is robust to that.

### Checking it still works

```bash
python scripts/smoke_stim_program.py    # 22 checks; needs the full local dataset
```

---

## The rest of this repository

Below is the original per-frame pipeline (`src/`), which the bout-level pipeline in `scripts/`
supersedes. It is kept for reference and provenance.

This repository contains pipelines and models used to learn and sample zebrafish trajectories from visual stimuli using a probabilistic neural network.

The code is organized as a modular pipeline covering:

1. Raw data transformation  
2. Feature engineering and preprocessing  
3. Model training  
4. Autoregressive sampling and trajectory visualization  

This repository is intended for further scaling, GPU training, and experimentation.

---

## Project Structure

All core code is located in:

```
src/
│
├── transformed_data_pipeline.py   # Raw experiment → transformed dataframe
├── preprocessing.py               # Feature engineering and preprocessing pipeline
├── model_NN.py                    # Neural network definition and training
├── sampling.py                    # Autoregressive sampling and trajectory generation
├── functions.py                   # Utility functions used by preprocessing pipeline
```

Notebooks used for experiments demonstrate the full pipeline from raw data to trajectory sampling and visualization.

---




## Pipeline Overview


### 1. Data Transformation

File:  
`src/transformed_data_pipeline.py`

Purpose:  
Transforms raw experiment files into analysis-ready dataframes with consistent coordinate systems.

Input:

- Stimuli trajectory file
- Fish position file
- ROI file

Output:

- Absolute coordinate dataframe
- Polar coordinate dataframe



### 2. Preprocessing Pipeline

File:  
`src/preprocessing.py`

Purpose:

- Feature engineering
- Lag feature creation
- Stimulus feature construction
- Dataset preparation for training and sampling

Input:

- Absolute coordinate dataframe
- Feature configuration parameters (lags, horizon_frames, subsample)

This pipeline relies heavily on utility functions defined in:

`src/functions.py`

Output:

- Preprocessed dataframe for model training
- Preprocessed dataframe for trajectory sampling



### 3. Model Training Pipeline

File:  
`src/model_NN.py`

Purpose:

- Split data into training, validation, and test sets
- Train probabilistic neural network
- Learn behavioral distribution parameters

Input:

- Preprocessed training dataframe

Model outputs 5 parameters per timestep:

- μ_st      → mean of Student-t distribution (orientation change)
- σ_st      → scale of Student-t distribution
- p_pos     → probability of positive speed
- μ_ln      → mean of lognormal speed distribution (log-space)
- σ_ln      → std of lognormal speed distribution

Output:

- Trained model
- Scalers
- Dataset splits



### 4. Sampling Pipeline

File:  
`src/sampling.py`

Purpose:

- Generate autoregressive trajectories
- Sample speed and orientation from learned distributions
- Generate trajectory videos

Input:

- Preprocessed sampling dataframe segment
- Trained model
- Scalers

Important:

- Stimulus positions are not generated by the model.
- The sampling pipeline uses exactly the stimulus positions provided in the input dataframe.
- To test specific stimulus conditions, modify the stimulus columns (`d{k}_x`, `d{k}_y`, `d{k}_stim`) in the sampling dataframe before running sampling.
- Autoregressive sampling requires `horizon_frames = 1`, since the model must generate trajectories frame-by-frame.

Output:

- Sampled trajectories
- Distribution parameters
- Visualization videos

---

## Typical Usage



### Recommended Data Loading Workflow

Because the raw pipeline transformation is computationally expensive, the recommended workflow is:

1. Run the transformation pipeline once
2. Save the transformed data locally as a .pkl file
3. Load the .pkl file for all subsequent experiments

This avoids recomputing transformations every time.


### Run transformation pipeline and save

```python
from src.transformed_data_pipeline import generate_transformed_datasets
import pickle

results = generate_transformed_datasets(stimuli_df, position_df, roi_df)

abs_dict = results["absolute"]

with open("abs_dict.pkl", "wb") as f:
    pickle.dump(abs_dict, f)
```

This saves the transformed absolute-coordinate dataset.



### Now to run experiments

Recommended workflow for experiments:

```python
# Load transformed dataset
with open("abs_dict.pkl", "rb") as f:
    abs_dict = pickle.load(f)

# Stack fish into single dataframe
df_abs_raw = functions.stack_dict_pipeline(abs_dict)

# Run preprocessing pipeline
df_for_training, df_for_sampling = preprocessing_pipeline(df_abs_raw, cols_to_lag=cols_to_lag,horizon_frames=horizon_frames, subsample=subsample) # data for training

# Run Model pipeline
model, scalers, splits = train_nn_pipeline(df_for_training, early_stopping=True)

# IMPORTANT:
# Do NOT pass the full df_for_sampling to sample_trajectory.
# Instead, select a short segment (e.g., one trial or ~1000–2000 frames),
# since sampling is autoregressive and starts from the beginning of the provided dataframe.

df_sampling_segment = df_for_sampling.iloc[:1800].copy()


# Run Sampling and video pipeline
window, prediction_sampling, params_sampling = sample_trajectory(df_sampling_segment, model, scalers, cols_to_lag=cols_to_lag, horizon_frames=horizon_frames, subsample=subsample)
anim = generate_trajectory_video(window, horizon_frames=horizon_frames)
HTML(anim.to_jshtml())



# Train model or run sampling
```

This avoids recomputing transformations and ensures consistent data usage.

--- 


## Key Parameters

Important parameters used throughout pipelines:

General:

- fps → frames per second  
- horizon_frames → prediction horizon  
- subsample → lag subsampling factor  
- max_lag_seconds → history window size 
- cols_to_lag → list of features for which lagged versions are created  

Sampling:

- N_sampling_seconds → duration of generated trajectory  

Important:

- Sampling and preprocessing before training the model must use consistent feature configurations.

- Autoregressive sampling requires horizon_frames = 1. Because the model predicts the next state autoregressively, it must generate frame t+1 from frame t. If horizon_frames > 1, intermediate frames are missing and autoregressive sampling cannot be performed correctly. Some assumptions can be done to perform autoregressive sampling with horizon_frame > 1 with interpolation or a second model that predicts the intermediate states.

---

## Model Description

The neural network predicts parameters of two probabilistic distributions:

Orientation change:

Student-t distribution  
Parameters: μ_st, σ_st  

Speed:

Zero-inflated lognormal distribution  
Parameters: p_pos, μ_ln, σ_ln  

This allows realistic probabilistic trajectory generation.

---

Tested with Python 3.9.13 on macOS.
