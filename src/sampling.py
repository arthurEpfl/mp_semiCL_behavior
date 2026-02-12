from . import functions
import pandas as pd
import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



#                     WARNING 
# !   subsample must be equal to horizon_frames !



def sample_trajectory(
    df_trial_base: pd.DataFrame,
    model: torch.nn.Module,
    scalers: dict,
    *,
    fps: int = 30,
    subsample: int = 5,
    horizon_frames: int = 5,
    max_lag_seconds: int = 3,
    cols_to_lag: list[str],
    N_sampling_seconds: int = 60
):
    """
    Autoregressive sampling on a SINGLE trial.

    Input
    -----
    df_trial_base : DataFrame created by preprocessing_pipeline but 
                    containing exactly ONE trial

    model : trained neural net that predicts distribution parameters for (speed, d_ori) at horizon=subsample frames.

    scalers : dict of fitted scalers returned by train pipeline:
              {"X": scaler_X, "speed": scaler_speed, "ori": scaler_ori}

    What it does (high-level)
    -------------------------
    1) Take the first (max_lag_seconds) of the trial to build the initial lag state (seed window).
    2) Build engineered features + lag features ONCE on the seed window (with subsample).
    3) Loop autoregressively:
        - scale last lag row with scalers["X"]
        - model -> params -> sample (speed, d_ori) in normalized space
        - inverse_transform speed/orientation with scalers["speed"], scalers["ori"]
        - update fish state (f_x,f_y,f_ori,f_speed,d_ori) at t+subsample
        - copy TRUE stimulus at aligned time (from df_trial_raw)
        - compute engineered features for that new row
        - manually shift lag columns by horizon_frames=subsample
        - update current (non-lag) cols from engineered/new row
    4) Returns both the generated trajectory and useful debug objects.

    Returns
    -------
        - "window": DataFrame of generated frames (same schema as df_trial_raw for fish cols + stimuli)
            the first max_lag_seconds contained in the window correspond to the real fish
        - "predictions": list of [pred_speed, pred_dori] (DENORMALIZED)
            the first max_lag_seconds  correspond to the real fish
        - "params": list of raw model outputs per step
    """
    
    max_lag_frames = int(max_lag_seconds * fps)
    N_total_prediction = int(N_sampling_seconds * fps // subsample)  # because of subsampling every 5 frames
    stim_cols = [f"d{i}_{s}" for i in range(8) for s in ["stim","x","y"]]
    

    # 100 following rows from random start index
    seed_len = int(max_lag_seconds * fps + 1)
    window = df_trial_base.iloc[:seed_len].copy().reset_index(drop=True)
    t_last = seed_len - 1

    window_with_engineered_features = functions.add_engineered_features(window, representation="absolute", n_stimuli=8)
    exclude_cols = [c for c in window_with_engineered_features.columns if c not in cols_to_lag]
    
    window_lag, _ = functions.add_lag_features_for_all(window_with_engineered_features, max_lag_seconds=max_lag_seconds, subsample=subsample, exclude_cols=exclude_cols)

    window_lag.head()

    feature_cols = window_lag.columns.tolist()
    param = []
    prediction = []


    for k in range(N_total_prediction):


        # ---- 1) One-step prediction from LAST row of window_lag ----
        model.eval()
        with torch.no_grad():
            x_np = scalers["X"].transform(window_lag.iloc[[-1]][feature_cols])

            x_in = torch.from_numpy(x_np.astype("float32"))

            out = model(x_in)
            param.append(out.cpu().numpy().flatten())
            mu_st = out[:, 0]
            sigma_st = torch.nn.functional.softplus(out[:, 1]) + 1e-6

            p_pos = torch.sigmoid(out[:, 2])
            mu_ln = out[:, 3]
            sigma_ln = 1e-3 + (1.0 - 1e-3) * torch.sigmoid(out[:, 4])

            # --- Sample orientation ---
            t_sample = torch.distributions.StudentT(df=5).sample(mu_st.shape).to(mu_st.device)
            pred_ori = mu_st + sigma_st * t_sample

            # --- Sample speed (Zero-Inflated LogNormal) ---
            u = torch.rand_like(p_pos)
            bern = (u < p_pos).float()
            pred_speed = bern * torch.exp(mu_ln + sigma_ln * torch.randn_like(mu_ln))

            pred_speed_val = float(pred_speed.item())
            pred_ori_val   = float(pred_ori.item())
            
        
        pred_speed_val = scalers["speed"].inverse_transform([[pred_speed_val]])[0, 0]   # to obtain the real speed value in original units 
        pred_ori_val   = scalers["ori"].inverse_transform([[pred_ori_val]])[0, 0]
        
        prediction.append([pred_speed_val, pred_ori_val])

        # ---- 2) Target time ----
        t_next = t_last + horizon_frames

        if t_next >= len(df_trial_base):
            break

        # ---- 3) Build new base row ---
        
        last_row = window.iloc[-1]
        new_row = last_row.copy()

        new_row["f_speed"] = pred_speed_val
        new_row["d_ori"]   = pred_ori_val

        new_row["f_ori"] = (last_row["f_ori"] + pred_ori_val) % (2 * np.pi)

        dt = horizon_frames / fps
        new_row["f_x"] = float(last_row["f_x"]) + pred_speed_val * np.cos(new_row["f_ori"]) * dt
        new_row["f_y"] = float(last_row["f_y"]) + pred_speed_val * np.sin(new_row["f_ori"]) * dt
        
        

        # ---- Copy TRUE stimulus aligned ----
        true_stim_row = df_trial_base.iloc[t_next]
        for c in stim_cols:
            new_row[c] = true_stim_row[c]

        # ---- Append to raw window (for next engineered computation only) ----
        window = pd.concat([window, new_row.to_frame().T], ignore_index=True)
        t_last = t_next

        # ---- 4) Compute engineered features ONLY for new row ----
        new_row_df = pd.DataFrame([new_row])
        new_row_eng = functions.add_engineered_features(
            new_row_df,
            representation="absolute",
            n_stimuli=8
        ).iloc[0]

        # ---- 5) Manual lag shift ----
        prev = window_lag.iloc[-1]
        next_feat = {}

        for base in cols_to_lag:

            # shift: 90 <- 85 <- ... <- 5 <- current
            for L in range(max_lag_frames, subsample - 1, -subsample):
                src_L = L - horizon_frames
                dst = f"{base}_lag_{L}"

                if src_L >= subsample:
                    src = f"{base}_lag_{src_L}"
                    next_feat[dst] = float(prev[src])
                else:
                    # lag_5 = previous current
                    next_feat[dst] = float(prev[base])

        # ---- 6) Update CURRENT columns ----
        for col in feature_cols:
            if "_lag_" in col:
                continue

            if col in new_row_eng.index:
                next_feat[col] = float(new_row_eng[col])
            elif col in new_row.index:
                next_feat[col] = float(new_row[col])
            else:
                next_feat[col] = float(prev[col])

        # ---- 7) Build next lag row ----
        window_lag = pd.DataFrame([next_feat]).reindex(columns=feature_cols)

        # Safety check
        if window_lag.isna().any(axis=None):
            print("PROBLEM: NaN detected in window_lag. Stopping sampling.")
            break
        
    window = window.iloc[seed_len:].reset_index(drop=True) # to keep only the generated part (the first max_lag_seconds correspond to the real fish)
    
    return window, prediction, param
        
        
        
        






def generate_trajectory_video(
    window,
    fps: int = 30,
    n_stimuli: int = 8,
    horizon_frames: int = 1
):
    """
    Generate an animation of a single trial trajectory.

    Parameters
    ----------
    window : pd.DataFrame
        DataFrame containing a single trial with absolute coordinates.
        Must include columns:
            - "f_x", "f_y", "f_ori"
            - "d{k}_x", "d{k}_y", "d{k}_stim" for k in [0, ..., n_stimuli-1]

    fps : int
        Frames per second of the animation.

    n_stimuli : int
        Number of stimuli (default: 8).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object.
    """

    traj_df = window.reset_index(drop=True)
    traj_df = functions.make_absolute(traj_df)

    # -------------------------
    # 1) Fixed frame limits
    # -------------------------
    xs = traj_df["f_x"].to_numpy()
    ys = traj_df["f_y"].to_numpy()

    # optional: include stimuli positions in bounds when active
    for k in range(n_stimuli):
        stim = traj_df[f"d{k}_stim"].to_numpy().astype(bool)
        xs = np.concatenate([xs, traj_df.loc[stim, f"d{k}_x"].to_numpy()])
        ys = np.concatenate([ys, traj_df.loc[stim, f"d{k}_y"].to_numpy()])

    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)

    pad = 0.05 * max(xmax - xmin, ymax - ymin)  # 5% margin
    xmin, xmax = xmin - pad, xmax + pad
    ymin, ymax = ymin - pad, ymax + pad

    # -------------------------
    # 2) Figure
    # -------------------------
    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    # fish arrow (constant length)
    arrow_len = 0.03 * max(xmax - xmin, ymax - ymin)

    # init artists
    fish_quiv = ax.quiver([traj_df.loc[0, "f_x"]], [traj_df.loc[0, "f_y"]],
                        [np.cos(traj_df.loc[0, "f_ori"])], [np.sin(traj_df.loc[0, "f_ori"])],
                        angles="xy", scale_units="xy", scale=1/arrow_len)

    stim_scatter = ax.scatter([], [], s=20)  # points

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    
    # -------------------------
    # 3) Update function
    # -------------------------
    def update(t):
        fx = traj_df.loc[t, "f_x"]
        fy = traj_df.loc[t, "f_y"]
        fo = traj_df.loc[t, "f_ori"]

        # update fish arrow (U,V are direction vectors scaled by arrow_len via quiver scale)
        fish_quiv.set_offsets(np.array([[fx, fy]]))
        fish_quiv.set_UVC(np.cos(fo), np.sin(fo))

        # gather active stimuli positions
        stim_x = []
        stim_y = []
        for k in range(n_stimuli):
            if int(traj_df.loc[t, f"d{k}_stim"]) == 1:
                stim_x.append(traj_df.loc[t, f"d{k}_x"])
                stim_y.append(traj_df.loc[t, f"d{k}_y"])

        if len(stim_x) == 0:
            stim_scatter.set_offsets(np.empty((0, 2)))
        else:
            stim_scatter.set_offsets(np.c_[stim_x, stim_y])

        t_sec = t / fps
        title.set_text(f"t={t_sec:.1f}s")
        return fish_quiv, stim_scatter, title


    # -------------------------
    # 4) Animate + save GIF
    # -------------------------
    interval = 1000 * horizon_frames / fps
    anim = FuncAnimation(fig, update, frames=len(traj_df), interval=interval, blit=False)

    return anim



