"""Video for Final_report_screendatatest.ipynb: real screen fish against F (seed 0) around one switch of the dot from
continuous to bout-like motion. Three real fish and three simulated fish drawn at random (fixed seed), each in its own
70 mm dish; the fish shown at its bout-onset positions (as the simulator moves it), a 5 s trail, the dot at its true
2 mm size. 60 s before the switch to 120 s after, at 4x speed.

    uv run --with pyarrow python screen_video.py      # -> Model/twin_test/videos/screen_real_vs_F.mp4 (+ .json)
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twin_test as TT  # noqa: E402
import screen_eval as SE  # noqa: E402

VID = TT.OUT / "videos"
PRE, POST, STEP_FR, SPEED, TRAIL_S = 60, 120, 6, 4, 5.0
N_EACH, SEED, SWITCH_INDEX, SIM_KEY = 3, 2026, 2, "14f"


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def pos_at(r, frames):
    """Fish position at each frame: its last bout-onset position (held between onsets)."""
    f = r["f"].astype(int)
    j = np.searchsorted(f, frames, side="right") - 1
    ok = j >= 0
    x = np.where(ok, r["x"][np.clip(j, 0, None)], np.nan)
    y = np.where(ok, r["y"][np.clip(j, 0, None)], np.nan)
    return x, y


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    st = TT.stimulus()
    kind = st.stim_kind.to_numpy().astype(str)
    b = (kind == "dot_bout").astype(int)
    starts = [s for s in np.flatnonzero(np.diff(b) == 1) + 1 if TT.WIN[0] + PRE * 30 <= s <= TT.WIN[1] - POST * 30]
    s0 = int(starts[min(SWITCH_INDEX, len(starts) - 1)])
    frames = np.arange(s0 - PRE * TT.FPS, s0 + POST * TT.FPS, STEP_FR)

    R = pickle.load(open(TT.OUT / "screen_raw_real.pkl", "rb"))["res"]
    G = pickle.load(open(TT.OUT / f"screen_raw_sim_{SIM_KEY}_sighted.pkl", "rb"))
    rng = np.random.default_rng(SEED)
    real_ids = [int(i) for i in rng.choice(sorted(R), N_EACH, replace=False)]
    sim_ids = [int(i) for i in rng.choice(sorted(G), N_EACH, replace=False)]
    ft = TT.fish_table().set_index("fish_id")
    panels = []
    for fid in real_ids:
        row = ft.loc[fid]
        dxa, dya = TT.dot_mm(st, row.centre_x_px, row.centre_y_px, row.px_per_mm)
        panels.append(("real", fid, R[fid], dxa, dya))
    S = TT._sim_setup()
    for i in sim_ids:
        panels.append(("F", i, G[i], S["dxa"], S["dya"]))

    VID.mkdir(parents=True, exist_ok=True)
    out = VID / "screen_real_vs_F.mp4"
    fig, axs = plt.subplots(2, N_EACH, figsize=(4.2 * N_EACH, 8.6), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03, wspace=0.05, hspace=0.12)
    arts = []
    R_MM = TT.R_MM
    for ax, (side, fid, r, dxa, dya) in zip(axs.ravel(), panels):
        ax.set_xlim(-R_MM - 1, R_MM + 1); ax.set_ylim(-R_MM - 1, R_MM + 1); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(plt.Circle((0, 0), R_MM, fill=False, color="0.4", lw=1.2))
        ax.set_title(f"{'real fish' if side == 'real' else 'F (seed 0)'} {fid}", fontsize=10,
                     color="k" if side == "real" else "#D55E00")
        px, py = pos_at(r, frames)
        trail, = ax.plot([], [], "-", color="k" if side == "real" else "#D55E00", lw=1.0, alpha=0.5)
        fish = ax.scatter([], [], s=40, color="k" if side == "real" else "#D55E00", zorder=3)
        dot = ax.scatter([], [], s=1, color="#7a3cff", zorder=4)
        arts.append((px, py, dxa, dya, trail, fish, dot, ax))
    txt = fig.suptitle("", fontsize=12)
    fig.canvas.draw()
    proc = subprocess.Popen([ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                             "-s", f"{fig.canvas.get_width_height()[0]}x{fig.canvas.get_width_height()[1]}",
                             "-r", str(int(TT.FPS / STEP_FR * SPEED)), "-i", "-", "-pix_fmt", "yuv420p",
                             "-vcodec", "libx264", str(out)], stdin=subprocess.PIPE)
    ntrail = int(TRAIL_S * TT.FPS / STEP_FR)
    for k, fr in enumerate(frames):
        t = (fr - s0) / TT.FPS
        motion = "continuous" if kind[fr] != "dot_bout" else "BOUT-LIKE"
        txt.set_text(f"t = {t:+6.1f} s from the switch    dot motion: {motion}    ({SPEED}x speed)")
        for px, py, dxa, dya, trail, fish, dot, ax in arts:
            a = max(0, k - ntrail)
            trail.set_data(px[a:k + 1], py[a:k + 1])
            fish.set_offsets([[px[k], py[k]]] if np.isfinite(px[k]) else np.empty((0, 2)))
            dot.set_offsets([[dxa[fr], dya[fr]]])
            ppm = ax.get_window_extent().width / (2 * R_MM + 2)
            dot.set_sizes([(2.0 * ppm * 72.0 / fig.dpi) ** 2])
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    proc.stdin.close(); proc.wait()
    plt.close(fig)
    meta = dict(mp4=str(out.relative_to(TT.ROOT / "Model")).replace("\\", "/"), switch_frame=s0,
                t_session_min=round(s0 / TT.FPS / 60, 1), real=real_ids, sim=sim_ids, sim_key=SIM_KEY,
                pre_s=PRE, post_s=POST, speed=SPEED)
    (VID / "screen_real_vs_F.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, meta)


if __name__ == "__main__":
    main()
