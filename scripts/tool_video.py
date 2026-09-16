"""2x3 video of programme-driven rollouts: two stimulus windows (rows) x three synthetic fish
(columns).

Unlike ``scripts/visualize.py`` -- whose two rows are hard-wired to Real vs Synthetic -- there is
no real fish here, so both rows show generated trajectories: the SAME three rollouts during two
different parts of the programme. That makes the columns comparable (one fish across conditions)
and the rows contrastive (e.g. moving vs static shoal at matched group size).

The per-panel drawing helpers are imported from ``visualize`` unchanged (``_dish_limits``,
``_init_panel``, ``_update_panel``) -- they are panel-local and grid-agnostic. This module adds
the grid/labelling and a short motion trail, and leaves ``visualize.py`` untouched.

Rendering runs in a SUBPROCESS (``make_tool_video``): a multi-panel FuncAnimation over ~1800
frames with ``blit=False`` is heavy enough to take down a notebook kernel, and the subprocess also
picks up the bundled-ffmpeg shim so ``.mp4`` works without a system ffmpeg.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_TRANSFORMED = Path(__file__).resolve().parents[1] / "data" / "data" / "df_transformed"

N_STIMULI = 8
FPS = 30
TRAIL_S = 6.0                 # seconds of path kept behind the fish
ROW_COLORS = ("#1F3B8C", "#B3541E")     # window A / window B (lineage blue, warm contrast)


# --------------------------------------------------------------------------- #
# Choosing the two windows
# --------------------------------------------------------------------------- #
def pick_windows(program, n_frames=None):
    """Two contrasting stimulus blocks: the first MOVING block at the largest group size, and the
    first STATIC block at the same size. Matching N is what makes the rows a clean comparison --
    the only thing that differs is whether the shoal moves.

    Falls back to the largest-N static block if that exact size is never shown static, and to the
    first two stimulus blocks if the programme has no static condition at all.
    """
    t = program.block_table()
    st = t[t["is_stim"]]
    if len(st) < 2:
        raise ValueError("programme has fewer than 2 stimulus blocks")
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
        if n_frames is not None:
            e = min(e, s + int(n_frames))
        out.append((s, e, str(r["label"])))
    return out


# --------------------------------------------------------------------------- #
# Panel CSVs
# --------------------------------------------------------------------------- #
def _anchor_for(g, program, block, lo):
    """The anchor this rollout latched for ``block``.

    Missing only if the fish executed no bout inside that block -- essentially impossible for a
    60 s block at ~1.3 s/bout, but handled rather than crashed: fall back to the fish's own
    interpolated pose at the block-onset frame, which is what the latch would have produced.
    """
    if not getattr(program, "needs_anchor", True):
        return None            # PhysicalProgram: placement is already absolute
    a = g.get("anchors", {}).get(int(block))
    if a is not None:
        return a
    import bout_video as BV
    onset = np.asarray(g["frame_onset"], int)
    sx, sy, so = BV._synthetic_path(np.array([lo]), onset,
                                    np.asarray(g["x_px"], float), np.asarray(g["y_px"], float),
                                    3, h0=float(g["start_pose"][2]))
    return float(sx[0]), float(sy[0]), float(so[0])


def write_panel(g, program, window, out_path, bout_frames=3, n_stimuli=N_STIMULI):
    """Write one panel's per-frame CSV in the existing ``df_base`` schema
    (``f_x, f_y, f_ori, d{k}_x, d{k}_y, d{k}_stim``), so the rows drop straight into the
    ``visualize`` panel helpers.

    Fish poses come from ``bout_video._synthetic_path`` (the same dart-and-hold bout->frame
    reconstruction the comparison videos use). Dots are re-derived from the programme with the
    anchor THIS rollout latched, so each panel shows the stimulus that fish actually saw.
    """
    import bout_video as BV
    lo, hi, _label = window
    lo, hi = int(lo), min(int(hi), program.n_frames)
    frames = np.arange(lo, hi, dtype=int)

    sx, sy, so = BV._synthetic_path(frames, np.asarray(g["frame_onset"], int),
                                    np.asarray(g["x_px"], float), np.asarray(g["y_px"], float),
                                    bout_frames, h0=float(g["start_pose"][2]))

    cols = {"f_x": sx, "f_y": sy, "f_ori": so}
    DX = np.zeros((len(frames), n_stimuli))
    DY = np.zeros((len(frames), n_stimuli))
    AC = np.zeros((len(frames), n_stimuli), int)
    cache = {}
    for i, fr in enumerate(frames):
        b = int(program.block_id[fr])
        if b not in cache:
            cache[b] = _anchor_for(g, program, b, int(program.starts[b]))
        X, Y = program.dots_absolute_all(fr, cache[b])
        DX[i], DY[i] = X, Y
        AC[i] = program.active[fr].astype(int)
    for k in range(n_stimuli):
        cols[f"d{k}_x"] = DX[:, k]
        cols[f"d{k}_y"] = DY[:, k]
        cols[f"d{k}_stim"] = AC[:, k]
    df = pd.DataFrame(cols)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


def write_panels(rollouts, program, windows, out_dir, n_cols=3, bout_frames=3):
    """Write the 2 x ``n_cols`` panel CSVs plus a manifest describing the grid."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(rollouts) < n_cols:
        raise ValueError(f"need >= {n_cols} rollouts for a 2x{n_cols} grid, got {len(rollouts)}")
    lens = []
    for r, win in enumerate(windows):
        for c in range(n_cols):
            lens.append(write_panel(rollouts[c], program, win,
                                    out_dir / f"panel_r{r}c{c}.csv", bout_frames=bout_frames))
    dish = rollouts[0]["dish"]
    manifest = dict(
        n_rows=len(windows), n_cols=n_cols, fps=FPS,
        dish=[float(v) for v in dish],
        row_labels=[f"{w[2]}\n[frames {int(w[0])}-{int(w[1])}]" for w in windows],
        col_labels=[f"rollout #{c + 1}" for c in range(n_cols)],
        n_frames=int(min(lens)),
        program=Path(program.path).name)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# Rendering (runs in the child process)
# --------------------------------------------------------------------------- #
def _render_grid(panel_dir, out_path, speed=1.0, trail_s=TRAIL_S, dpi=120, verbose=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import visualize as V

    panel_dir = Path(panel_dir)
    man = json.loads((panel_dir / "manifest.json").read_text(encoding="utf-8"))
    R, C, fps = man["n_rows"], man["n_cols"], man["fps"]
    dx, dy, dr = man["dish"]

    segs = [[pd.read_csv(panel_dir / f"panel_r{r}c{c}.csv") for c in range(C)] for r in range(R)]
    n_frames = min(len(s) for row in segs for s in row)
    trail = max(1, int(trail_s * fps))
    if verbose:
        print(f"rendering {R}x{C} panels, {n_frames} frames ...", flush=True)

    fig, axes = plt.subplots(R, C, figsize=(3.5 * C, 3.75 * R + 0.4),
                             gridspec_kw={"hspace": 0.18, "wspace": 0.05})
    axes = np.atleast_2d(axes)
    fig.suptitle(f"{man['program']} - simulated fish", fontsize=11, y=0.985)

    artists = []
    for r in range(R):
        color = ROW_COLORS[r % len(ROW_COLORS)]
        for c in range(C):
            ax = axes[r, c]
            traj = segs[r][c]
            quiv, scat, txt = V._init_panel(ax, traj, dx, dy, dr,
                                            label=(man["col_labels"][c] if r == 0 else ""),
                                            color=color)
            # visualize.py draws no path history; a still frame of a 2x3 grid is much easier to
            # read with a short trail behind each fish.
            (line,) = ax.plot([], [], lw=0.9, color=color, alpha=0.55, zorder=2)
            artists.append((quiv, scat, txt, line, traj))
        axes[r, 0].set_ylabel(man["row_labels"][r], fontsize=8, labelpad=6, color=color)

    def update(t):
        for (quiv, scat, txt, line, traj) in artists:
            V._update_panel(t, traj, quiv, scat, txt, fps)
            lo = max(0, t - trail)
            line.set_data(traj["f_x"].values[lo:t + 1], traj["f_y"].values[lo:t + 1])
        return []

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / (fps * speed), blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = "pillow" if out_path.suffix.lower() == ".gif" else "ffmpeg"
    anim.save(str(out_path), writer=writer, fps=int(fps * speed), dpi=dpi)
    plt.close(fig)
    if verbose:
        print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out_path


# --------------------------------------------------------------------------- #
# Public entry point (runs in the parent, renders in a child)
# --------------------------------------------------------------------------- #
def make_tool_video(rollouts, program, out_mp4, panel_dir=None, windows=None,
                    n_cols=3, speed=1.0, bout_frames=3, verbose=True):
    """Rollouts -> panel CSVs -> mp4. Returns ``(out_mp4, manifest)``."""
    import bout_video as BV

    out_mp4 = Path(out_mp4)
    panel_dir = Path(panel_dir) if panel_dir is not None else out_mp4.parent / (out_mp4.stem + "_panels")
    windows = windows if windows is not None else pick_windows(program)
    man = write_panels(rollouts, program, windows, panel_dir, n_cols=n_cols, bout_frames=bout_frames)
    if verbose:
        for r, lab in enumerate(man["row_labels"]):
            print(f"  row {r}: {lab}")
        print(f"  {man['n_frames']} frames/panel -> {out_mp4}")

    render_panels(panel_dir, out_mp4, speed=speed)
    return out_mp4, man


def render_panels(panel_dir, out_mp4, speed=1.0):
    """Render an already-written panel directory to ``out_mp4`` in a SUBPROCESS.

    Split out of ``make_tool_video`` so real-fish panels (``write_real_panels``) render through
    exactly the same path as generated ones -- the two videos are then comparable frame for frame
    rather than being drawn by two code paths that can drift.
    """
    import bout_video as BV

    out_mp4 = Path(out_mp4)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--render", str(panel_dir),
           "--output", str(out_mp4), "--speed", str(speed)]
    env, rc_dir = BV._render_env(out_mp4)          # UTF-8 + bundled ffmpeg when none on PATH
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        if rc_dir is not None:
            import shutil
            shutil.rmtree(rc_dir, ignore_errors=True)
    return out_mp4


# --------------------------------------------------------------------------- #
# Real-fish panels (needs the recorded experiment; not shipped with the repo)
# --------------------------------------------------------------------------- #
def write_real_panels(fish_ids, windows, out_dir, transformed_dir=None, roi_path=None,
                      n_stimuli=N_STIMULI, fps=FPS):
    """Panel CSVs straight from the recording: rows = windows, columns = ``fish_ids``.

    Same grid semantics and same ``df_base`` schema as :func:`write_panels`, so the resulting mp4
    is directly comparable with the generated one. Each panel carries that fish's OWN dots -- which
    is the point: the same programme is anchored to each fish separately, so the three columns show
    the same stimulus sequence landing in three different places.
    """
    import pandas as pd
    from segment_bouts import STIM_NONE, load_physical_dish

    transformed_dir = Path(transformed_dir) if transformed_dir is not None else _DEFAULT_TRANSFORMED
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lens = []
    for r, (lo, hi, _lab) in enumerate(windows):
        lo, hi = int(lo), int(hi)
        for c, fid in enumerate(fish_ids):
            cands = sorted(transformed_dir.glob(f"*_f{int(fid)}_transformed_abs.csv"))
            if not cands:
                raise FileNotFoundError(f"no transformed CSV for fish {fid} in {transformed_dir}")
            use = ([f"f{fid}_x", f"f{fid}_y", f"f{fid}_ori"]
                   + [f"d{k}_{v}" for k in range(n_stimuli) for v in ("x", "y")]
                   + [f"d{k}_stim" for k in range(n_stimuli)])
            # `frame` equals the row index 1:1 in these files, so nrows+iloc is exact and avoids
            # reading a 115 MB CSV in full for a 60 s window.
            w = pd.read_csv(cands[0], usecols=use, nrows=hi).iloc[lo:hi].reset_index(drop=True)
            cols = {"f_x": w[f"f{fid}_x"].to_numpy(float),
                    "f_y": w[f"f{fid}_y"].to_numpy(float),
                    "f_ori": w[f"f{fid}_ori"].to_numpy(float)}
            for k in range(n_stimuli):
                x = w[f"d{k}_x"].to_numpy(float)
                cols[f"d{k}_x"] = x
                cols[f"d{k}_y"] = w[f"d{k}_y"].to_numpy(float)
                cols[f"d{k}_stim"] = ((~w[f"d{k}_stim"].astype(str).isin(STIM_NONE).to_numpy())
                                      & np.isfinite(x)).astype(int)
            df = pd.DataFrame(cols)
            df.to_csv(out_dir / f"panel_r{r}c{c}.csv", index=False)
            lens.append(len(df))

    dish_tab = load_physical_dish(roi_path) if roi_path is not None else None
    if dish_tab is not None:
        d = dish_tab.loc[list(fish_ids)]
        dish = [float(d["cx"].median()), float(d["cy"].median()), float(d["radius"].median())]
    else:
        dish = [180.0, 174.0, 183.5]
    manifest = dict(n_rows=len(windows), n_cols=len(fish_ids), fps=fps, dish=dish,
                    row_labels=[f"{w[2]}\n[frames {int(w[0])}-{int(w[1])}]" for w in windows],
                    col_labels=[f"REAL fish {int(f)}" for f in fish_ids],
                    n_frames=int(min(lens)), program="recorded experiment")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", required=True, help="panel directory (with manifest.json)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args()
    _render_grid(a.render, a.output, speed=a.speed, dpi=a.dpi)
