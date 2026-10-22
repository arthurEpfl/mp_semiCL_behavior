"""Videos of every shoaling experiment in the thesis (Results, "Multiple fish"), for Model/Final_report_shoaling.ipynb.
Seed 0 only; the simulations already exist (scripts/multi_fish.py); nothing is re-simulated.

    F shoals         tab:shoal-grid  training dish and 5x dish, 2 / 5 / 10 fish, beside blind fish of the same dish
    Repulsion        tab:shoal-rep   training dish, pairs and 5 fish: F, x1, x2, x4, x8, x1 + collision, blind
    Screen pairs     tab:shoal-dm    training dish and 5x dish: F, D4 trained on the screen, D4 + direction

The window is NOT chosen to look good: a random group (the same group index in every panel of a video, i.e. the same
starts and random streams where the conditions share them) and a random 180 s stretch inside the readout period, from a
fixed seed. Each panel's title gives that window's median nearest-neighbour distance and the condition's median over
all groups and the whole readout, so a typical window can be told from a lucky one.

    uv run --no-sync python scripts/shoal_videos.py            # all videos (parallel), -> Model/multi_fish/videos/report/
    uv run --no-sync python scripts/shoal_videos.py --render NPZ MP4     # (internal) render one video
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multi_fish as MF  # noqa: E402

OUT = MF.VIDEO_DIR / "report"
FPS = MF.VIDEO_FPS
LENGTH_S, SPEED, SEED = 180.0, 4.0, 2026


def _A(d):
    return "training dish" if d == "A1" else "5× dish"


def video_specs():
    """[(name, title, [(cond, label, n_fish)])] in the order of the report's tables."""
    V = []
    for d in ("A1", "A5"):
        for N in (2, 5, 10):
            c = MF.grid_name(d, N, "F", 0)
            V.append((f"F_{d}_N{N}", f"F, {N} fish, {_A(d)}",
                      [(c, f"F, {N} fish: they see each other", N), (MF.blind_of(c), f"{N} blind fish", N)]))
    lab = {"F": "F", "grT": "repulsion ×1", "grT2": "×2", "grT4": "×4", "grT8": "×8", "grTC": "×1 + collision"}
    for N in (2, 5):
        panels = [(MF.grid_name("A1", N, v, 0), lab[v], N) for v in lab]
        panels.append((MF.blind_of(panels[0][0]), f"{N} blind fish", N))
        V.append((f"rep_A1_N{N}", f"Repulsion sweep, {N} fish, training dish", panels))
    for d in ("A1", "A5"):
        V.append((f"pairs_{d}", f"Pairs, {_A(d)}: F and the screen-trained D4",
                  [(MF.grid_name(d, 2, "F", 0), "F", 2), (f"pairdm_{d}_sD4_s0", "D4 trained on the screen", 2),
                   (f"pairdm_{d}_sD4dm_s0", "D4 + dot direction", 2)]))
    for d in ("A1", "A5"):
        conds = [(MF.grid_name(d, 2, "F", 0), "F"), (MF.grid_name(d, 2, "grT", 0), "F + repulsion"),
                 (f"pairdm_{d}_sD4_s0", "D4 trained on the screen"), (f"pairdm_{d}_sD4dm_s0", "D4 + dot direction")]
        V.append((f"pairsm_{d}", f"Speed-matched pairs, {_A(d)}",
                  [(c, f"{lab} (pair {g})", 2, g) for (c, lab), g in zip(conds, [matched_pair(c) for c, _ in conds])]))
    return V


def pair_speeds(cond):
    """(G, 2) distance swum per second (mm/s) of each fish of each pair over the readout period."""
    sp = MF.CONDS[cond]
    return np.array([[f["d"][(f["t"] >= sp["t_read"]) & (f["t"] < sp["T"])].sum() / MF.PX_MM / (sp["T"] - sp["t_read"])
                      for f in grp[:2]] for grp in MF.load(cond)["groups"]])


def matched_pair(cond):
    """The pair whose two fish swim most alike and closest to the condition's median speed (chosen on how each fish
    swims, never on how the pair shoals): argmin |log(v1/v2)| + max_i |log(v_i/median)|."""
    S = pair_speeds(cond)
    med = np.median(S)
    return int(np.argmin(np.abs(np.log(S[:, 0] / S[:, 1])) + np.abs(np.log(S / med)).max(1)))


def _nn(P):
    """Median nearest-neighbour distance (mm) of positions P (fish, times, 2) in mm."""
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    D[np.arange(len(P)), np.arange(len(P))] = np.inf
    return float(np.median(D.min(1)))


def _condition_nn(cond, n):
    """The condition's median nearest distance over all groups and its readout period (1 s grid)."""
    sp = MF.CONDS[cond]
    taus = np.arange(sp["t_read"], sp["T"], 1.0)
    vals = []
    for grp in MF.load(cond)["groups"]:
        P = np.array([MF.pos_at(f, taus) for f in grp[:n]]) / MF.PX_MM
        D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
        D[np.arange(n), np.arange(n)] = np.inf
        vals.append(D.min(1).ravel())
    return float(np.median(np.concatenate(vals)))


def _view(xy_mm, R, margin=15.0):
    """Square view (mm, dish-centred) holding the fish of one panel; the whole dish when that is smaller."""
    P = xy_mm.reshape(-1, 2)
    lo, hi = P.min(0) - margin, P.max(0) + margin
    c, h = (lo + hi) / 2, max(hi - lo) / 2
    h = max(h, 30.0)
    if h >= R * 1.03 or R < 100.0:                       # the training dish (R ~ 49 mm) is always shown whole
        return (-R * 1.03, R * 1.03, -R * 1.03, R * 1.03)
    return (float(c[0] - h), float(c[0] + h), float(c[1] - h), float(c[1] + h))


def _draw(panels, rng):
    """The video's random group and window start (same draws whether or not the video is rendered)."""
    G = min(MF.CONDS[p[0]]["G"] for p in panels)
    T0 = max(MF.CONDS[p[0]]["t_read"] for p in panels)
    T1 = min(MF.CONDS[p[0]]["T"] for p in panels)
    return int(rng.integers(G)), float(np.round(rng.uniform(T0, T1 - LENGTH_S)))


def build_npz(name, title, panels, rng):
    panels = [p if len(p) == 4 else p + (None,) for p in panels]          # (cond, label, n, fixed group or None)
    g, t0 = _draw(panels, rng)
    t1 = t0 + LENGTH_S
    taus = np.arange(t0, t1, SPEED / FPS)
    xy, views, labels, stats = [], [], [], []
    for cond, lab, n, gp in panels:
        grp = MF.load(cond)["groups"][g if gp is None else gp][:n]
        cx, cy, Rpx = MF.dish(MF.CONDS[cond]["arena"])
        P = np.array([MF.frames_xy(f, taus) for f in grp])
        P = np.stack([(P[..., 0] - cx) / MF.PX_MM, (P[..., 1] - cy) / MF.PX_MM], -1)      # mm, dish-centred
        R = Rpx / MF.PX_MM
        nn_w = _nn(np.array([MF.pos_at(f, np.arange(t0, t1, 1.0)) for f in grp]) / MF.PX_MM)
        nn_all = _condition_nn(cond, n)
        xy.append(P); views.append(_view(P, R) + (R,))
        labels.append(f"{lab}\nnearest {nn_w:.0f} mm here (all groups {nn_all:.0f} mm)")
        stats.append(dict(cond=cond, label=lab, n=n, group=int(g if gp is None else gp), nn_window_mm=nn_w,
                          nn_condition_mm=nn_all))
    nmax = max(p.shape[0] for p in xy)
    arr = np.full((len(xy), nmax, len(taus), 2), np.nan)
    for i, p in enumerate(xy):
        arr[i, :p.shape[0]] = p
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / f"{name}.npz"
    np.savez_compressed(npz, xy=arr, taus=taus, views=np.array(views), labels=np.array(labels), title=np.array(title),
                        speed=SPEED)
    return npz, dict(name=name, title=title, group=g, t0=t0, t1=t1, panels=stats, mp4=str(OUT / f"{name}.mp4"))


def render(npz, out_mp4, trail_s=4.0, dpi=90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    z = np.load(npz, allow_pickle=False)
    xy, taus, views, labels, speed = z["xy"], z["taus"], z["views"], z["labels"], float(z["speed"])
    P, N, F = xy.shape[0], xy.shape[1], xy.shape[2]
    nc = min(P, 4); nr = int(np.ceil(P / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 4.6 * nr + 0.5), squeeze=False)
    cols = plt.get_cmap("tab10")(np.arange(N) % 10)
    trail = max(1, int(trail_s * FPS / speed))
    arts = []
    for p, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if p >= P:
            continue
        x0, x1, y0, y1, R = (float(v) for v in views[p])
        ax.add_patch(plt.Circle((0, 0), R, fill=False, lw=1.2, color="0.35"))
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal")
        ax.set_title(str(labels[p]), fontsize=9.5)
        w = x1 - x0
        bar = 10.0 if w < 120 else (20.0 if w < 250 else 50.0)
        ax.plot([x0 + 0.05 * w, x0 + 0.05 * w + bar], [y1 - 0.04 * w] * 2, color="k", lw=2)
        ax.text(x0 + 0.05 * w + bar / 2, y1 - 0.07 * w, f"{bar:.0f} mm", ha="center", fontsize=8)
        if w < 2 * R:
            ax.text(x1 - 0.02 * w, y0 + 0.03 * w, f"zoomed: {w:.0f} of {2 * R:.0f} mm", ha="right", va="top",
                    fontsize=7, color="0.4")
        live = np.isfinite(xy[p, :, 0, 0])
        X, Y = xy[p, live, :, 0], xy[p, live, :, 1]
        lines = [ax.plot([], [], lw=0.9, color=cols[i], alpha=0.65)[0] for i in range(len(X))]
        dots = ax.scatter(X[:, 0], Y[:, 0], s=26, c=cols[:len(X)], zorder=3, edgecolors="k", linewidths=0.4)
        arts.append((X, Y, lines, dots))
    fig.suptitle(str(z["title"]), fontsize=12)
    ttl = fig.text(0.5, 0.01, "", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    def update(fr):
        lo = max(0, fr - trail)
        for X, Y, lines, dots in arts:
            for i, ln in enumerate(lines):
                ln.set_data(X[i, lo:fr + 1], Y[i, lo:fr + 1])
            dots.set_offsets(np.c_[X[:, fr], Y[:, fr]])
        m, s = divmod(float(taus[fr]), 60.0)
        ttl.set_text(f"t = {int(m):02d}:{s:04.1f}   (×{speed:g} speed, trails 4 s)")
        return []

    FuncAnimation(fig, update, frames=F, interval=1000.0 / FPS, blit=False).save(str(out_mp4), writer="ffmpeg",
                                                                                   fps=FPS, dpi=dpi)
    plt.close(fig)


def _render_one(npz):
    import bout_video as BV
    out = npz.with_suffix(".mp4")
    env, rc_dir = BV._render_env(out)
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--render", str(npz), str(out)], check=True,
                       env=env)
    finally:
        if rc_dir is not None:
            import shutil
            shutil.rmtree(rc_dir, ignore_errors=True)
    npz.unlink(missing_ok=True)
    return out


def main(workers=5, only=None):
    rng = np.random.default_rng(SEED)
    p_idx = OUT / "videos_report.json"
    old = {m["name"]: m for m in json.loads(p_idx.read_text(encoding="utf-8"))} if (only and p_idx.exists()) else {}
    jobs, index = [], []
    for name, title, panels in video_specs():
        if only and not name.startswith(only):
            _draw(panels, rng)                       # keep the random stream as in a full run
            if name in old:
                index.append(old[name])
            continue
        npz, meta = build_npz(name, title, panels, rng)
        jobs.append(npz); index.append(meta)
        print(f"frames ready: {name} (group {meta['group']}, {meta['t0']:.0f}-{meta['t1']:.0f} s)", flush=True)
    (OUT / "videos_report.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    with ThreadPoolExecutor(workers) as ex:
        for out in ex.map(_render_one, jobs):
            print(f"rendered {out.name}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--render":
        render(a[1], a[2])
    else:
        main(only=a[0] if a else None)
