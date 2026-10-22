"""Video of the two-target experiment (Final_report_extrapolations_2.ipynb, section 8): ONE simulated fish, the same
fish and the same block geometry at three separations side by side, so the switch from swimming between the two dot
groups (small L) to following one of them (large L) can be seen.

The fish and block are CHOSEN to show the pattern clearly (it is an illustration, not a random sample): among the 15
fish x 10 blocks of the 2x-dish 4 v 4 runs under the effects rule, the (fish, block) whose following bouts (20-60 s)
sit closest to the midline at the smallest L shown and closest to one track at the largest, both with enough following
bouts. The fish is then re-simulated with exactly the seeds of that run (same start pose, latent, drift, random streams
and tracks), and the rollout is checked against the stored bouts.

Drawn: the dish wall where in view, the two tracks (faint), the dots of each group (4 dots at fixed offsets around the
group centroid; the model sees each group only as its centroid, count and moving flag, so the offsets are for display),
the fish (position held between bouts, a 0.2 s glide through each bout) with a 10 s trail, the midline between the
tracks (dashed). Below each dish: the fish's lateral position (+ = toward A) over the block, with both tracks at +-L/2:
near 0 = between the groups, near +-L/2 = on one group. Time runs at x3.

With --anchored: the equidistant-start runs (two_targets.AnchoredParallelProvider, notebook section 2e): the pair is
placed at the fish's first bout of the block, the fish on the midline; before that bout the pair is drawn held at its
start position (the model sees the targets only at its bouts). The view is centred on the tracks.

    python two_targets_video.py [--tag 14f] [--Ls 20 50 80] [--anchored]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import two_targets as TT

VIDEO_FPS, SPEED, TRAIL_S, BOUT_S = 30, 3.0, 10.0, 0.2
OUT_DIR = TT.OUT / "videos"
DOT_OFFS_MM = np.array([[0.0, 3.5], [3.5, 0.0], [0.0, -3.5], [-3.5, 0.0]])   # display only (the model sees the centroid)
COL_A, COL_B, COL_FISH = "#d62728", "#1f77b4", "#111111"


def pick(tag, Ls, cond="effects", pair="4v4", min_follow=8, onset="free"):
    """(fish, block) showing the pattern best: at min(Ls) the mean |lateral offset| of following bouts small, at max(Ls)
    the mean |offset| close to the track (L/2) on ONE side; score = |side mean| / (L/2) at the largest L minus
    mean |offset| / (L/2) at the smallest. Returns (fish, block, table)."""
    lo, hi = min(Ls), max(Ls)
    per = {}
    for L in (lo, hi):
        f = TT.following(TT.load(tag, pair, L, cond, TT.ARENA_BIG, onset).query("t_blk >= 20"), L)
        per[L] = f.groupby(["fish", "blk"]).lat_mm.agg(n="size", absm=lambda v: np.abs(v).mean(), m="mean")
    T = per[lo].join(per[hi], lsuffix="_lo", rsuffix="_hi", how="inner")
    T = T[(T.n_lo >= min_follow) & (T.n_hi >= min_follow)]
    T["score"] = T.m_hi.abs() / (hi / 2) - T.absm_lo / (lo / 2)
    k, j = T.score.idxmax()
    return int(k), int(j), T.sort_values("score", ascending=False)


def full_run(tag, L, k, cond="effects", pair="4v4", onset="free"):
    """Re-simulate fish k of TT.simulate(tag, scene(pair, L), cond, arena=ARENA_BIG) with its own seeds; returns the
    generator output and the provider (tracks)."""
    models = TT._models(tag)
    per = TT.N_FISH // len(models)
    m = models[k // per]
    _, dish, ref = TT.MF._programme()
    dish = (float(dish[0]), float(dish[1]), float(dish[2]) * TT.ARENA_BIG)
    rng = np.random.default_rng([0, k])
    pose = ref.sample_start_pose(rng, dish)
    Prov = TT.AnchoredParallelProvider if onset == "anchored" else TT.ParallelProvider
    prov = Prov(TT.scene(pair, L), k, dish, TT.N_BLOCKS)
    rec = TT.SR.make_synthetic_record(int(prov.T / 0.2) + 100, pose, dish, fish_id=k)
    g = m.sample(rec, rng, stim_provider=prov, stop_time_s=prov.T, block_seed=(0, k), multi_rule=cond)
    # check against the stored run: the same fish's bouts in the stored pickle
    b = TT.load(tag, pair, L, cond, TT.ARENA_BIG, onset)
    keep = sorted(t for t in prov.rec if t < len(g["dtheta"]))
    if not np.allclose(np.asarray(g["dtheta"])[keep], b[b.fish == k].dth.to_numpy()):
        raise RuntimeError("re-simulation differs from the stored run")
    return g, prov, dish


def frames(g, prov, j, taus_blk):
    """Fish (x, y) px, both centroids (px) and the track geometry at block-j times ``taus_blk`` (s from block onset)."""
    t0 = (prov.lead_f + j * prov.cyc_f) / prov.fps
    taus = t0 + taus_blk
    t = np.asarray(g["t_onset_s"], float)
    x, y = np.asarray(g["x_px"], float), np.asarray(g["y_px"], float)
    k = np.clip(np.searchsorted(t, taus, side="right") - 1, 0, len(t) - 2)
    a = np.clip((taus - t[k]) / BOUT_S, 0.0, 1.0)
    fish = np.c_[x[k] + a * (x[k + 1] - x[k]), y[k] + a * (y[k + 1] - y[k])]
    A, B, lat = [], [], []
    t_anchor = prov.anchor[j][2] if hasattr(prov, "anchor") and j in prov.anchor else 0.0
    for i, tb in enumerate(taus_blk):
        # anchored onset: before the block's first bout the pair is held at its start position
        (pa, pb), _, mid, nA = prov._state(j, int(round(max(tb, t_anchor) * prov.fps)))
        A.append(pa); B.append(pb); lat.append(float(np.dot(fish[i] - mid, nA)) / TT.PX_MM)
    return fish, np.array(A), np.array(B), np.array(lat)


def render(panels, out_mp4, title):
    """``panels`` = [(L, fish px, A px, B px, prov, j, dish, taus_blk, lateral mm)]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    P = len(panels)
    fig, AX = plt.subplots(2, P, figsize=(5.0 * P, 7.4), gridspec_kw=dict(height_ratios=(3.2, 1.0)))
    axes, laxes = AX[0], AX[1]
    trail = int(TRAIL_S * VIDEO_FPS / SPEED)
    arts = []
    for ax, lax, (L, F, A, B, prov, j, dish, taus, lat) in zip(axes, laxes, panels):
        cx, cy, R = dish
        c0 = np.array([cx, cy])
        tc = prov.centre[j] if getattr(prov, "centre", None) is not None and prov.centre[j] is not None else c0
        mm = lambda p: (np.asarray(p, float) - tc) / TT.PX_MM          # view centred on the tracks
        half = 75.0
        dc = (c0 - tc) / TT.PX_MM                                      # the dish centre in view coordinates
        ax.add_patch(plt.Circle((dc[0], dc[1]), R / TT.PX_MM, fill=False, lw=1.2, color="0.35"))
        u, nA, _ = prov.geom[j]
        for side, col in ((1, COL_A), (-1, COL_B)):                # the two tracks
            p0 = side * prov.half * nA - prov.amp * u
            p1 = side * prov.half * nA + prov.amp * u
            ax.plot([p0[0] / TT.PX_MM, p1[0] / TT.PX_MM], [p0[1] / TT.PX_MM, p1[1] / TT.PX_MM], color=col, lw=6,
                    alpha=0.10, solid_capstyle="round")
        m0, m1 = (-prov.amp * u) / TT.PX_MM, (prov.amp * u) / TT.PX_MM
        ax.plot([m0[0], m1[0]], [m0[1], m1[1]], color="0.5", lw=0.8, ls="--")      # the midline
        lax.axhline(L / 2, color=COL_A, lw=1.5, alpha=0.6); lax.axhline(-L / 2, color=COL_B, lw=1.5, alpha=0.6)
        lax.axhline(0, color="0.5", lw=0.8, ls="--")
        lax.axvspan(0, 20, color="0.9", lw=0)
        lax.set_xlim(0, TT.STIM_S); lax.set_ylim(-58, 58)
        lax.set_xlabel("time in block (s)", fontsize=8); lax.tick_params(labelsize=7)
        lax.set_ylabel("lateral (mm)", fontsize=8)
        lln, = lax.plot([], [], color=COL_FISH, lw=1.0)
        ax.set_xlim(-half, half); ax.set_ylim(half, -half); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"L = {L:g} mm", fontsize=13)
        ax.plot([-half + 6, -half + 26], [half - 6] * 2, color="k", lw=2)
        ax.text(-half + 16, half - 9, "20 mm", ha="center", fontsize=8)
        Fm, Am, Bm = mm(F), mm(A), mm(B)
        ln, = ax.plot([], [], lw=1.0, color=COL_FISH, alpha=0.55)
        fish = ax.scatter([Fm[0, 0]], [Fm[0, 1]], s=36, c=COL_FISH, zorder=4)
        da = ax.scatter(*(Am[0] + DOT_OFFS_MM).T, s=14, c=COL_A, zorder=3)
        db = ax.scatter(*(Bm[0] + DOT_OFFS_MM).T, s=14, c=COL_B, zorder=3)
        arts.append((Fm, Am, Bm, ln, fish, da, db, lat, lln))
    taus = panels[0][7]
    fig.suptitle(title, fontsize=11)
    ttl = fig.text(0.5, 0.005, "", ha="center", fontsize=10)

    def update(fr):
        lo = max(0, fr - trail)
        for Fm, Am, Bm, ln, fish, da, db, lat, lln in arts:
            ln.set_data(Fm[lo:fr + 1, 0], Fm[lo:fr + 1, 1])
            lln.set_data(taus[:fr + 1], lat[:fr + 1])
            fish.set_offsets(Fm[fr:fr + 1])
            da.set_offsets(Am[fr] + DOT_OFFS_MM); db.set_offsets(Bm[fr] + DOT_OFFS_MM)
        tb = float(taus[fr])
        ttl.set_text(f"t = {tb:4.1f} s into the block (x{SPEED:g} speed)" + ("   [readout window 20-60 s]" if tb >= 20 else
                                                                              "   [onset phase, not read]"))
        return []

    anim = FuncAnimation(fig, update, frames=len(taus), interval=1000.0 / VIDEO_FPS, blit=False)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_mp4), writer="ffmpeg", fps=VIDEO_FPS, dpi=100)
    plt.close(fig)


def make(tag="14f", Ls=(20.0, 50.0, 80.0), cond="effects", onset="free"):
    k, j, T = pick(tag, Ls, cond, onset=onset)
    print(f"picked fish {k}, block {j}:\n{T.head(5).round(2).to_string()}", flush=True)
    taus = np.arange(0.0, TT.STIM_S, SPEED / VIDEO_FPS)
    panels = []
    for L in Ls:
        g, prov, dish = full_run(tag, L, k, cond, onset=onset)
        F, A, B, lat = frames(g, prov, j, taus)
        panels.append((L, F, A, B, prov, j, dish, taus, lat))
    lab = {"26e": "M6 seed 0", "30d": "M6 seed 101", "14f": "F seed 0", "32b": "F seed 101"}[tag]
    start = ", equidistant start (pair anchored on the fish)" if onset == "anchored" else ""
    title = (f"{lab}, rule '{cond}', 4 v 4, dish of twice the radius{start}: the same fish and block (fish {k}, block {j}) at "
             f"three separations\nred = group A, blue = group B, black = the fish (10 s trail); below: its lateral "
             f"position (lines: the two tracks, dashed: the midline; grey: 0-20 s, not read)")
    out = OUT_DIR / f"two_targets_4v4_{tag}_{cond}{'_anch' if onset == 'anchored' else ''}.mp4"
    render(panels, out, title)
    print("wrote", out, flush=True)
    return out, k, j


if __name__ == "__main__":
    a = sys.argv
    tag = a[a.index("--tag") + 1] if "--tag" in a else "14f"
    Ls = tuple(float(v) for v in a[a.index("--Ls") + 1:a.index("--Ls") + 4]) if "--Ls" in a else (20.0, 50.0, 80.0)
    make(tag, Ls, onset="anchored" if "--anchored" in a else "free")
