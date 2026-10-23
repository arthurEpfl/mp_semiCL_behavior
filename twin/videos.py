"""Every video of the repository, drawn by one renderer.

A video is a row (or grid) of PANELS, one dish each, played over the same frame times ``taus``. A panel holds:

    xy      (n_fish, F, 2) fish positions in mm, in the panel's view coordinates (image y down)
    view    (x0, x1, y0, y1) the visible window in mm
    dish    (cx, cy, R) the dish wall in view coordinates, mm
    label   the panel title
    colors  one colour per fish (default: tab10)
    dots    optional, [dict(xy=(k, F, 2) mm, color=, size=)]: dot groups that move with the frames
    lines   optional, [dict(x=, y=, color=, lw=, ls=, alpha=)]: static lines (tracks, midline)
    trace   optional, dict(t=(F,), y=(F,), lines=[(y, color)], ylim=, ylabel=, xlabel=, span=(a, b)): a time trace
            drawn under the dish, filled in as the video plays

and the video a ``title`` and one ``footer`` string per frame. Bouts become frame positions by one rule, ``frames``:
the fish holds its position between bouts and covers each bout in a 0.2 s straight line (the model's bouts are
instantaneous). Rendering runs in a subprocess (a long FuncAnimation is heavy for a notebook kernel) with the ffmpeg
that ``imageio-ffmpeg`` bundles when none is on the PATH.

    write(panels, taus, title, footer, out_mp4)          # the one entry point
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FPS = 30
BOUT_S = 0.2                       # a bout is drawn as a 0.2 s straight line


def frames(t, x, y, taus):
    """Positions at the frame times ``taus``: held between bouts, a linear ``BOUT_S`` glide through each bout. ``t``,
    ``x``, ``y`` are the bout onset times and the positions at those onsets (any unit)."""
    t = np.asarray(t, float)
    k = np.clip(np.searchsorted(t, taus, side="right") - 1, 0, len(t) - 2)    # last onset <= tau
    a = np.clip((taus - t[k]) / BOUT_S, 0.0, 1.0)
    x0, y0 = np.asarray(x, float)[k], np.asarray(y, float)[k]
    x1, y1 = np.asarray(x, float)[k + 1], np.asarray(y, float)[k + 1]
    return np.stack([x0 + a * (x1 - x0), y0 + a * (y1 - y0)], 1)


def clock_footer(taus, speed, trail_s=4.0):
    """The shoal videos' footer: session clock, speed and trail length."""
    return [f"t = {int(m):02d}:{s:04.1f}   (×{speed:g} speed, trails {trail_s:g} s)"
            for m, s in (divmod(float(t), 60.0) for t in taus)]


def render(spec_path, out_mp4, dpi=90):
    """Draw the video described by the pickled spec at ``spec_path`` into ``out_mp4`` (called in the subprocess)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    with open(spec_path, "rb") as fh:
        S = pickle.load(fh)
    panels, taus, trail_s, speed = S["panels"], S["taus"], S["trail_s"], S["speed"]
    P, F = len(panels), len(taus)
    traced = any(p.get("trace") for p in panels)
    if traced:                                            # one column per dish, its trace underneath
        fig, AX = plt.subplots(2, P, figsize=(5.0 * P, 7.4), squeeze=False,
                               gridspec_kw=dict(height_ratios=(3.2, 1.0)))
        axes, taxes = list(AX[0]), list(AX[1])
    else:
        nc = int(S.get("ncols") or min(P, 4)); nr = int(np.ceil(P / nc))
        fig, AX = plt.subplots(nr, nc, figsize=(4.3 * nc, 4.6 * nr + 0.5), squeeze=False)
        axes, taxes = list(AX.ravel()), [None] * (nr * nc)
    trail = max(1, int(trail_s * FPS / speed))
    arts = []
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= P:
            continue
        p = panels[i]
        x0, x1, y0, y1 = (float(v) for v in p["view"])
        cx, cy, R = (float(v) for v in p["dish"])
        ax.add_patch(plt.Circle((cx, cy), R, fill=False, lw=1.2, color="0.35"))
        for ln in p.get("lines", ()):
            ax.plot(ln["x"], ln["y"], color=ln.get("color", "0.5"), lw=ln.get("lw", 0.8), ls=ln.get("ls", "-"),
                    alpha=ln.get("alpha", 1.0), solid_capstyle="round")
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal")
        ax.set_title(str(p["label"]), fontsize=p.get("title_size", 9.5))
        w = x1 - x0
        bar = 10.0 if w < 120 else (20.0 if w < 250 else 50.0)
        ax.plot([x0 + 0.05 * w, x0 + 0.05 * w + bar], [y1 - 0.04 * w] * 2, color="k", lw=2)
        ax.text(x0 + 0.05 * w + bar / 2, y1 - 0.07 * w, f"{bar:.0f} mm", ha="center", fontsize=8)
        if w < 2 * R and p.get("zoom_note", True):
            ax.text(x1 - 0.02 * w, y0 + 0.03 * w, f"zoomed: {w:.0f} of {2 * R:.0f} mm", ha="right", va="top",
                    fontsize=7, color="0.4")
        xy = np.asarray(p["xy"], float)
        live = np.isfinite(xy[:, 0, 0])
        X, Y = xy[live, :, 0], xy[live, :, 1]
        cols = p.get("colors")
        cols = plt.get_cmap("tab10")(np.arange(xy.shape[0]) % 10)[live] if cols is None else np.asarray(cols)[live]
        lines = [ax.plot([], [], lw=0.9, color=cols[j], alpha=0.65)[0] for j in range(len(X))]
        fish = ax.scatter(X[:, 0], Y[:, 0], s=26, c=cols[:len(X)], zorder=3, edgecolors="k", linewidths=0.4)
        dots = [(np.asarray(d["xy"], float), ax.scatter(*np.asarray(d["xy"], float)[:, 0].T, s=d.get("size", 14),
                                                         c=d.get("color", "C3"), zorder=2))
                for d in p.get("dots", ())]
        tr = None
        if traced and p.get("trace"):
            q, tax = p["trace"], taxes[i]
            for yv, col in q.get("lines", ()):
                tax.axhline(yv, color=col, lw=1.5 if col != "0.5" else 0.8, alpha=0.6, ls="--" if col == "0.5" else "-")
            if q.get("span"):
                tax.axvspan(*q["span"], color="0.9", lw=0)
            tax.set_xlim(float(np.min(q["t"])), float(np.max(q["t"]))); tax.set_ylim(*q["ylim"])
            tax.set_xlabel(q.get("xlabel", "time (s)"), fontsize=8); tax.set_ylabel(q.get("ylabel", ""), fontsize=8)
            tax.tick_params(labelsize=7)
            tr = (np.asarray(q["t"], float), np.asarray(q["y"], float), tax.plot([], [], color="0.07", lw=1.0)[0])
        elif traced:
            taxes[i].axis("off")
        arts.append((X, Y, lines, fish, dots, tr))
    fig.suptitle(str(S["title"]), fontsize=12 if not traced else 11)
    ttl = fig.text(0.5, 0.01 if not traced else 0.005, "", ha="center", fontsize=10)
    if not traced:
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    footer = S["footer"]

    def update(fr):
        lo = max(0, fr - trail)
        for X, Y, lines, fish, dots, tr in arts:
            for j, ln in enumerate(lines):
                ln.set_data(X[j, lo:fr + 1], Y[j, lo:fr + 1])
            fish.set_offsets(np.c_[X[:, fr], Y[:, fr]])
            for D, sc in dots:
                sc.set_offsets(D[:, fr])
            if tr is not None:
                tr[2].set_data(tr[0][:fr + 1], tr[1][:fr + 1])
        ttl.set_text(footer[fr])
        return []

    FuncAnimation(fig, update, frames=F, interval=1000.0 / FPS, blit=False).save(str(out_mp4), writer="ffmpeg",
                                                                                   fps=FPS, dpi=dpi)
    plt.close(fig)


def _render_env(out_path):
    """Environment for the render subprocess (+ a temp dir to clean up, or None): UTF-8, and for ``.mp4`` with no
    ffmpeg on the PATH, matplotlib pointed at the one ``imageio-ffmpeg`` bundles."""
    import shutil
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if Path(out_path).suffix.lower() != ".mp4" or shutil.which("ffmpeg"):
        return env, None
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = None
    if not exe or not os.path.exists(exe):
        raise RuntimeError(f"Cannot write {Path(out_path).name}: no ffmpeg found (uv sync installs imageio-ffmpeg, "
                           "or put ffmpeg on the PATH, or write a .gif)")
    rc_dir = Path(tempfile.mkdtemp(prefix="mplffmpeg_"))
    (rc_dir / "matplotlibrc").write_text(f"animation.ffmpeg_path: {exe.replace(os.sep, '/')}\n", encoding="utf-8")
    env["MATPLOTLIBRC"] = str(rc_dir)
    return env, rc_dir


def write(panels, taus, title, footer, out_mp4, speed, trail_s=4.0, dpi=90, ncols=None):
    """Render ``panels`` over the frame times ``taus`` into ``out_mp4`` (see the module doc); without traces the panels
    fill a grid of ``ncols`` columns (default: up to 4). Returns the path."""
    import shutil
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if len(footer) != len(taus):
        raise ValueError("one footer string per frame")
    fd, spec = tempfile.mkstemp(suffix=".pkl", prefix="video_")
    os.close(fd)
    with open(spec, "wb") as fh:
        pickle.dump(dict(panels=panels, taus=np.asarray(taus, float), title=title, footer=list(footer),
                         speed=float(speed), trail_s=float(trail_s), ncols=ncols), fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    env, rc_dir = _render_env(out_mp4)
    try:
        subprocess.run([sys.executable, "-m", "twin.videos", "--render", spec, str(out_mp4), str(dpi)], check=True,
                       env=env, cwd=str(Path(__file__).resolve().parents[1]))
    finally:
        os.unlink(spec)
        if rc_dir is not None:
            shutil.rmtree(rc_dir, ignore_errors=True)
    return out_mp4


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--render"]:
        render(a[1], a[2], dpi=int(a[3]) if len(a) > 3 else 90)
    else:
        raise SystemExit(__doc__)
