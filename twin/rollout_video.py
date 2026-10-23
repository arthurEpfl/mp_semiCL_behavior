"""Videos of programme rollouts (``stim_rollout.rollout_many``), drawn by ``twin.videos``: one row per stimulus window,
one column per simulated fish, and, on the rig's programme, real fish next to them in the same windows.

Each fish is shown in its whole dish with the dots it saw: a simulated fish's dots are the programme anchored to its
own pose at the block onset (the anchors ``rollout_program`` returns), a real fish's are the dots the rig displayed
for it (``data/frames``). The same programme therefore lands in a different place for every fish.

    windows = pick_windows(program)                       # the first moving block at the largest group, the static one
    write_video(rollouts, program, out_mp4, windows, real_fish=(0, 8, 14))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from twin import paths
from twin import videos as V
from twin.segment_bouts import PX_PER_MM, STIM_NONE, load_physical_dish

SIM_COL, REAL_COL = "#1F3B8C", "#3C3C3C"
DOT_COL = "#d62728"


def pick_windows(program, max_s=None):
    """Two contrasting stimulus blocks: the first MOVING block at the largest group size and the first STATIC block at
    the same size (else the largest static one; else the first two stimulus blocks). [(lo, hi, label)] in frames."""
    t = program.block_table()
    st = t[t["is_stim"]]
    if len(st) < 2:
        raise ValueError("the programme has fewer than 2 stimulus blocks")
    mv, sta = st[st["moving"]], st[~st["moving"]]
    if len(mv) and len(sta):
        n_top = int(mv["n_dots"].max())
        a = mv[mv["n_dots"] == n_top].iloc[0]
        same = sta[sta["n_dots"] == n_top]
        b = (same if len(same) else sta[sta["n_dots"] == sta["n_dots"].max()]).iloc[0]
    else:
        a, b = st.iloc[0], st.iloc[1]
    out = []
    for r in (a, b):
        s, e = int(r["start_frame"]), int(r["end_frame"])
        if max_s is not None:
            e = min(e, s + int(max_s * program.fps))
        out.append((s, e, f"{int(r['n_dots'])} dot{'s' if r['n_dots'] > 1 else ''}, "
                          f"{'moving' if r['moving'] else 'static'} (block {int(r['block'])})"))
    return out


def _dots_mm(X, Y, on, cx, cy):
    """(8, F, 2) dot positions in mm, dish-centred, NaN where a dot is off."""
    D = np.stack([(X - cx) / PX_PER_MM, (Y - cy) / PX_PER_MM], -1)            # (F, 8, 2)
    D[~on] = np.nan
    return np.transpose(D, (1, 0, 2))


def sim_panel(g, program, window, taus_f, label):
    """One simulated fish over the programme frames ``taus_f`` (float frames) of ``window``."""
    cx, cy, R = g["dish"]
    xy = V.frames(g["t_onset_s"], g["x_px"], g["y_px"], taus_f / program.fps)
    fr = np.clip(np.round(taus_f).astype(int), 0, program.n_frames - 1)
    X = np.full((len(fr), program.n_dots), np.nan); Y = X.copy()
    on = program.active[fr].copy()
    for i, f in enumerate(fr):
        a = g["anchors"].get(int(program.block_id[f]))
        if a is None:                                  # no bout in this block yet: the fish has not seen it
            on[i] = False
            continue
        X[i], Y[i] = program.dots_absolute_all(f, a)
    return dict(xy=np.stack([(xy[:, 0] - cx) / PX_PER_MM, (xy[:, 1] - cy) / PX_PER_MM], -1)[None],
                view=(-R / PX_PER_MM * 1.05, R / PX_PER_MM * 1.05, -R / PX_PER_MM * 1.05, R / PX_PER_MM * 1.05),
                dish=(0.0, 0.0, R / PX_PER_MM), label=label, colors=[SIM_COL],
                dots=[dict(xy=_dots_mm(X, Y, on, cx, cy), color=DOT_COL, size=14)])


def real_panel(fish, window, taus_f, label, frames_dir=paths.FRAMES):
    """One real fish over the same frames (the recording's frame index is the programme's)."""
    lo, hi = int(window[0]), int(window[1])
    src = sorted(Path(frames_dir).glob(f"*_f{int(fish)}_transformed_abs.csv"))
    if not src:
        raise FileNotFoundError(f"no frame table of fish {fish} in {frames_dir} (data/README.md)")
    use = [f"f{fish}_x", f"f{fish}_y"] + [f"d{k}_{v}" for k in range(8) for v in ("x", "y", "stim")]
    w = pd.read_csv(src[0], usecols=use, nrows=hi).iloc[lo:hi].reset_index(drop=True)
    d = load_physical_dish(paths.ROI_PHYS).loc[int(fish)]
    cx, cy, R = float(d["cx"]), float(d["cy"]), float(d["radius"])
    i = np.clip(np.round(taus_f).astype(int) - lo, 0, len(w) - 1)
    w = w.iloc[i]
    X = np.stack([w[f"d{k}_x"].to_numpy(float) for k in range(8)], 1)
    Y = np.stack([w[f"d{k}_y"].to_numpy(float) for k in range(8)], 1)
    on = np.stack([~w[f"d{k}_stim"].astype(str).isin({str(s) for s in STIM_NONE} | {"nan"}).to_numpy()
                   for k in range(8)], 1) & np.isfinite(X)
    fx = (w[f"f{fish}_x"].to_numpy(float) - cx) / PX_PER_MM
    fy = (w[f"f{fish}_y"].to_numpy(float) - cy) / PX_PER_MM
    return dict(xy=np.stack([fx, fy], -1)[None],
                view=(-R / PX_PER_MM * 1.05, R / PX_PER_MM * 1.05, -R / PX_PER_MM * 1.05, R / PX_PER_MM * 1.05),
                dish=(0.0, 0.0, R / PX_PER_MM), label=label, colors=[REAL_COL],
                dots=[dict(xy=_dots_mm(X, Y, on, cx, cy), color=DOT_COL, size=14)])


def write_video(rollouts, program, out_mp4, windows=None, n_sim=3, real_fish=(), speed=3.0, model_label="model"):
    """Rows = ``windows`` [(lo, hi, label)] (frames; default ``pick_windows``), columns = the first ``n_sim``
    rollouts, then the real fish ``real_fish`` (the rig's programme only). Every window is cut to the shortest."""
    windows = windows or pick_windows(program)
    n_f = min(hi - lo for lo, hi, _ in windows)
    step = speed                                            # programme frames per video frame
    panels = []
    for lo, hi, lab in windows:
        taus_f = lo + np.arange(0.0, n_f, step)
        for c in range(min(n_sim, len(rollouts))):
            panels.append(sim_panel(rollouts[c], program, (lo, hi), taus_f, f"{model_label}, simulated fish {c + 1}\n{lab}"))
        for f in real_fish:
            panels.append(real_panel(f, (lo, hi), taus_f, f"real fish {f}\n{lab}"))
    taus = np.arange(0.0, n_f, step) / program.fps
    footer = [f"t = {t:4.1f} s into the block (×{speed:g} speed, trails 4 s)" for t in taus]
    ncols = min(n_sim, len(rollouts)) + len(real_fish)
    title = (f"{Path(program.path).name or 'programme'}: dots red, simulated fish blue, real fish grey; "
             "each fish sees the programme anchored to its own pose at the block onset")
    return V.write(panels, taus, title, footer, out_mp4, speed=speed, ncols=ncols)
