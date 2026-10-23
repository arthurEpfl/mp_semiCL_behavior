"""Videos of every shoaling experiment in the thesis (Results, "Multiple fish"), drawn by ``twin.videos``. Seed 0 only;
they are cut from the shoal simulations of ``twin.multi_fish`` (run those first), nothing is simulated here.

    F shoals         tab:shoal-grid  training dish and 5x dish, 2 / 5 / 10 fish, beside blind fish of the same dish
    Repulsion        tab:shoal-rep   training dish, pairs and 5 fish: F, x1, x2, x4, x8, x1 + collision, blind;
                                     5x dish, 5 fish: F, x1, x1 + collision, blind
    Screen pairs     tab:shoal-dm    training dish and 5x dish: F, D4 trained on the screen, D4 + direction

The window is NOT chosen to look good: a random group (the same group index in every panel of a video, i.e. the same
starts and random streams where the conditions share them) and a random 180 s stretch inside the readout period, from a
fixed seed. Each panel's title gives that window's median nearest-neighbour distance and the condition's median over
all groups and the whole readout, so a typical window can be told from a lucky one.

    uv run python -m twin.shoal_videos                  # every video -> output/multi_fish/videos/report/
    uv run python -m twin.shoal_videos F_A1_N5          # the videos whose name starts with F_A1_N5

Without the shoal simulations, ``video(name)`` simulates only the groups the video shows (the same groups as in the
full runs, so the same video; about half a minute per group) and renders into output/experiments/shoaling/; the panel
titles then give the window's nearest distance only (the condition's median needs every group).
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from twin import multi_fish as MF
from twin import paths
from twin import videos as V

OUT = MF.VIDEO_DIR / "report"
LITE = paths.EXPERIMENTS / "shoaling"           # videos and single groups simulated without the full shoal runs
LENGTH_S, SPEED, SEED = 180.0, 4.0, 2026


def _A(d):
    return "training dish" if d == "A1" else "5× dish"


def video_specs():
    """[(name, title, [(cond, label, n_fish)])] in the order of the report's tables (new videos go at the end, so the
    random group and window of every earlier one stay as they were)."""
    V_ = []
    for d in ("A1", "A5"):
        for N in (2, 5, 10):
            c = MF.grid_name(d, N, "F", 0)
            V_.append((f"F_{d}_N{N}", f"F, {N} fish, {_A(d)}",
                       [(c, f"F, {N} fish: they see each other", N), (MF.blind_of(c), f"{N} blind fish", N)]))
    lab = {"F": "F", "grT": "repulsion ×1", "grT2": "×2", "grT4": "×4", "grT8": "×8", "grTC": "×1 + collision"}
    for N in (2, 5):
        panels = [(MF.grid_name("A1", N, v, 0), lab[v], N) for v in lab]
        panels.append((MF.blind_of(panels[0][0]), f"{N} blind fish", N))
        V_.append((f"rep_A1_N{N}", f"Repulsion sweep, {N} fish, training dish", panels))
    for d in ("A1", "A5"):
        V_.append((f"pairs_{d}", f"Pairs, {_A(d)}: F and the screen-trained D4",
                   [(MF.grid_name(d, 2, "F", 0), "F", 2), (f"pairdm_{d}_sD4_s0", "D4 trained on the screen", 2),
                    (f"pairdm_{d}_sD4dm_s0", "D4 + dot direction", 2)]))
    for d in ("A1", "A5"):
        conds = [(MF.grid_name(d, 2, "F", 0), "F"), (MF.grid_name(d, 2, "grT", 0), "F + repulsion"),
                 (f"pairdm_{d}_sD4_s0", "D4 trained on the screen"), (f"pairdm_{d}_sD4dm_s0", "D4 + dot direction")]
        V_.append((f"pairsm_{d}", f"Speed-matched pairs, {_A(d)}",
                   [(c, lab_, 2, "matched") for c, lab_ in conds]))
    panels = [(MF.grid_name("A5", 5, v, 0), lab[v], 5) for v in ("F", "grT", "grTC")]
    panels.append((MF.blind_of(panels[0][0]), "5 blind fish", 5))
    V_.append(("rep_A5_N5", "Repulsion and collision, 5 fish, 5× dish", panels))
    return V_


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


def _full(cond):
    return (MF.OUT / f"{cond}.pkl").exists()


def _groups(cond, g, sim_dir=LITE):
    """The fish of group ``g`` of ``cond``: from the full run when it exists, else simulated once (the same group as
    in the full run) and kept in ``sim_dir``."""
    if _full(cond):
        return MF.load(cond)["groups"][g]
    p = Path(sim_dir) / "groups" / f"{cond}_g{g}.pkl"
    if not p.exists():
        res = MF.run_group(MF.CONDS[cond], g)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(p, "rb") as fh:
        return pickle.load(fh)


def build(name, title, panels, rng, sim_dir=LITE):
    """The panels of one video (``twin.videos`` format), its frame times and its index entry."""
    panels = [p if len(p) == 4 else p + (None,) for p in panels]          # (cond, label, n, fixed group or None)
    for i, (cond, lab, n, gp) in enumerate(panels):
        if gp == "matched":
            if not _full(cond):
                raise FileNotFoundError(f"{name}: the speed-matched pair is chosen over every group of {cond}, "
                                        "which needs its full run (python -m twin.multi_fish --run-grid)")
            gp = matched_pair(cond)
            panels[i] = (cond, f"{lab} (pair {gp})", n, gp)
    g, t0 = _draw(panels, rng)
    t1 = t0 + LENGTH_S
    taus = np.arange(t0, t1, SPEED / V.FPS)
    out, stats = [], []
    nmax = max(p[2] for p in panels)
    for cond, lab, n, gp in panels:
        grp = _groups(cond, g if gp is None else gp, sim_dir)[:n]
        cx, cy, Rpx = MF.dish(MF.CONDS[cond]["arena"])
        P = np.array([V.frames(f["t"], f["x"], f["y"], taus) for f in grp])
        P = np.stack([(P[..., 0] - cx) / MF.PX_MM, (P[..., 1] - cy) / MF.PX_MM], -1)      # mm, dish-centred
        R = Rpx / MF.PX_MM
        nn_w = _nn(np.array([MF.pos_at(f, np.arange(t0, t1, 1.0)) for f in grp]) / MF.PX_MM)
        nn_all = _condition_nn(cond, n) if _full(cond) else None
        xy = np.full((nmax, len(taus), 2), np.nan)
        xy[:len(P)] = P
        note = f" (all groups {nn_all:.0f} mm)" if nn_all is not None else ""
        out.append(dict(xy=xy, view=_view(P, R), dish=(0.0, 0.0, R), label=f"{lab}\nnearest {nn_w:.0f} mm here{note}"))
        stats.append(dict(cond=cond, label=lab, n=n, group=int(g if gp is None else gp), nn_window_mm=nn_w,
                          nn_condition_mm=nn_all))
    meta = dict(name=name, title=title, group=g, t0=t0, t1=t1, panels=stats)
    return out, taus, meta


def main(only=None, out_dir=OUT, workers=5, missing_only=False):
    """Render the videos (those whose name starts with ``only``; with ``missing_only``, only those not yet in
    ``out_dir``). The random group and window of each video are drawn in a fixed order, so a partial run gives every
    video the same window as a full one. Returns the paths written."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(SEED)
    p_idx = out_dir / "videos_report.json"
    old = {m["name"]: m for m in json.loads(p_idx.read_text(encoding="utf-8"))} if p_idx.exists() else {}
    jobs, index = [], []
    for name, title, panels in video_specs():
        mp4 = out_dir / f"{name}.mp4"
        if (only and not name.startswith(only)) or (missing_only and mp4.exists()):
            _draw(panels, rng)                       # keep the random stream as in a full run
            if name in old:
                index.append(old[name])
            continue
        pan, taus, meta = build(name, title, panels, rng)
        meta["mp4"] = str(mp4)
        jobs.append((pan, taus, title, mp4)); index.append(meta)
        print(f"frames ready: {name} (group {meta['group']}, {meta['t0']:.0f}-{meta['t1']:.0f} s)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_idx.write_text(json.dumps(index, indent=1), encoding="utf-8")

    def one(job):
        pan, taus, title, mp4 = job
        return V.write(pan, taus, title, V.clock_footer(taus, SPEED), mp4, speed=SPEED)

    with ThreadPoolExecutor(workers) as ex:
        done = list(ex.map(one, jobs))
    for p in done:
        print(f"rendered {p.name}", flush=True)
    return done


def video(name, out_dir=LITE):
    """The mp4 of one thesis video: the rendered one if it exists (``OUT``, then ``out_dir``), else rendered into
    ``out_dir`` from the full runs or, without them, from the groups it shows (simulated here)."""
    for d in (OUT, Path(out_dir)):
        if (d / f"{name}.mp4").exists():
            return d / f"{name}.mp4"
    rng = np.random.default_rng(SEED)
    for nm, title, panels in video_specs():
        if nm != name:
            _draw(panels, rng)                       # the same random stream as a full run
            continue
        pan, taus, meta = build(name, title, panels, rng, sim_dir=out_dir)
        mp4 = Path(out_dir) / f"{name}.mp4"
        V.write(pan, taus, title, V.clock_footer(taus, SPEED), mp4, speed=SPEED)
        (Path(out_dir) / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return mp4
    raise KeyError(f"no video {name!r}; the names: {[v[0] for v in video_specs()]}")


if __name__ == "__main__":
    main(only=sys.argv[1] if len(sys.argv) > 1 else None)
