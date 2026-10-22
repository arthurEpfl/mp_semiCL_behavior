"""The rig's three raw files -> one table per fish at 30 frames per second (``data/frames``).

  trajectory   per frame and dot: the stimulus label and the dot's offset from the fish at stimulus onset (mm)
  positions    per frame: every fish's x, y and orientation (pixels, clockwise), then the dots' columns again
  roi          per dish: screen offsets, diameter, centre and radius (pixels)

Each fish's table starts one pause before the first stimulus and drops the rig's "bout" calibration frames. A dot's
offset is placed in the dish from the fish's pose at the first frame of its stimulus block; during a pause the dots
hold the positions of the block that follows. The y axis and the orientation are flipped (counter-clockwise angles).
"""
import numpy as np
import pandas as pd

from twin import paths

FPS = 30
MM_TO_PX = 3.67                     # the trajectory file's dot offsets are in mm
PAUSE = "inter_stim_pause"


def read_raw():
    """The three raw files as the rig wrote them."""
    return (pd.read_csv(paths.TRAJECTORY, header=None), pd.read_csv(paths.POSITIONS, header=None, sep=r"\s+"),
            pd.read_csv(paths.ROI_PHYS, header=None))


def _pause_copy(trial, xy, stim):
    """A trial = a pause then a stimulus block: the pause takes the block's first dot positions."""
    pause = trial[trial["condition"] == PAUSE]
    block = trial[trial["condition"] != PAUSE]
    if len(pause) and len(block):
        n = len(pause)
        trial.loc[pause.index, xy] = block.iloc[:n][xy].values
        trial.loc[pause.index, stim] = pause[stim].mask(np.array(block[stim] != "none")[:n], PAUSE)
    return trial


def build(trajectory, positions, roi):
    """{"f0": table, ..., "f14": table}: frame, stimulus, f<k>_x, f<k>_y, f<k>_ori, d<i>_x, d<i>_y, d<i>_stim."""
    n_dots = trajectory.shape[1] // 6
    n_fish = positions.shape[1] // 3 - n_dots
    dot_cols = [f"d{i}_{v}" for i in range(n_dots) for v in ("x", "y", "stim")]
    xy = [c for c in dot_cols if not c.endswith("_stim")]
    ys = [f"d{i}_y" for i in range(n_dots)]
    stim_cols = [f"d{i}_stim" for i in range(n_dots)]
    stim = trajectory.copy()
    stim.columns = [f"d{i}_{v}" for i in range(n_dots) for v in ("stim", "x", "y", "size", "color", "backg")]
    stim = stim[dot_cols]
    df = positions.copy()
    df.columns = [f"f{k}_{v}" for k in range(n_fish) for v in ("x", "y", "ori")] + dot_cols
    roi = roi.copy()
    roi.columns = ["xoff", "yoff", "diameter", "x", "y", "radius"]
    if not (df[stim_cols] == stim[stim_cols]).all().all():
        raise ValueError("the stimulus labels of the positions and trajectory files differ")

    # start one inter-stimulus pause before the first stimulus; drop the calibration ("bout") frames
    on = df["d0_stim"].str.startswith("CLsemi", na=False)
    first = on.idxmax()
    end = on[first:].idxmin()
    pause = ((df.loc[end:]["d0_stim"] == "none").idxmin() - end) / FPS
    start = first - int(pause * FPS)
    df["condition"] = np.where(df["d0_stim"] == "none", PAUSE, df["d0_stim"])
    fish = df.loc[start:].loc[df["d0_stim"] != "bout"].drop(columns=dot_cols)
    dots = stim.loc[start:]
    dots = dots.loc[dots["d0_stim"] != "bout"].copy()
    cond = fish["condition"]
    dots["condition"] = cond
    dots["trial_id"] = ((cond == PAUSE) & (cond.shift(1) != PAUSE)).cumsum()
    dots["block_id"] = (cond != cond.shift(1)).cumsum()
    dots = dots.groupby("trial_id", group_keys=False)[dots.columns].apply(_pause_copy, xy=xy, stim=stim_cols)
    dots[xy] *= MM_TO_PX

    out = {}
    for k in range(n_fish):
        # the dots of a block, placed from the fish's pose at the block's first frame
        parts = []
        for _, b in dots.groupby("block_id"):
            row = fish.loc[b.index[0]]
            heading = row[f"f{k}_ori"] - np.pi / 2
            part = {}
            for i in range(n_dots):
                theta = np.arctan2(b[f"d{i}_y"], b[f"d{i}_x"]) + heading
                r = np.hypot(b[f"d{i}_x"], b[f"d{i}_y"])
                part[f"d{i}_x"] = r * np.cos(theta) + row[f"f{k}_x"]
                part[f"d{i}_y"] = r * np.sin(theta) + row[f"f{k}_y"]
                part[f"d{i}_stim"] = b[f"d{i}_stim"]
            parts.append(pd.DataFrame(part, index=b.index))
        d = pd.concat(parts)
        # flip y (dish diameter - y) and turn the orientation counter-clockwise
        diameter = roi.loc[k, "diameter"]
        d[ys] = diameter - d[ys]
        f = fish[[f"f{k}_x", f"f{k}_y", f"f{k}_ori", "condition"]].copy()
        f[f"f{k}_ori"] = (2 * np.pi - f[f"f{k}_ori"]) % (2 * np.pi)
        f[f"f{k}_y"] = diameter - f[f"f{k}_y"]
        f.insert(0, "frame", np.arange(len(f)))
        t = pd.concat([f.reset_index(drop=True), d.reset_index(drop=True)], axis=1)
        t = t[t["d0_stim"] != "none"].rename(columns={"condition": "stimulus"})
        out[f"f{k}"] = t[["frame", "stimulus", f"f{k}_x", f"f{k}_y", f"f{k}_ori"] + dot_cols]
    return out


def write(tables, out_dir=paths.FRAMES, prefix="2025.02.20_GsizeMotionSingle"):
    """One csv per fish: ``<prefix>_f<k>_transformed_abs.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(out_dir / f"{prefix}_{k}_transformed_abs.csv", index=False)
