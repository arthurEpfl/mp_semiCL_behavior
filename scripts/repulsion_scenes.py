"""How do F and its repulsion terms combine SEVERAL close dots? (2026-10-18)

The repulsion terms (turn, step) were calibrated on single-dot bouts (Larsch screen), where their sum over dots has one
term; in the shoal they sum over every neighbour within reach. This probe puts phantom moving dots at FIXED distances and
bearings from the fish, re-placed at every bout from its current pose, and records which way it turns. The network sees
the dots clustered into groups exactly as in the shoal runs (multi_fish.neighbour_groups, effects rule); the repulsion
terms see each dot individually (``dots``). Bearings are relative to the fish's travel heading, + = left (CCW); a turn
dtheta > 0 is CCW, i.e. toward a dot at a positive bearing.

    python repulsion_scenes.py            # every scene x model -> Model/repulsion_scenes/table.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multi_fish as MFi  # noqa: E402

OUT = HERE.parent / "Model" / "repulsion_scenes"
T_S, PER_FOLD = 600.0, 2
SCENES = {
    "1 dot beside (4 mm, +90)": [(4.0, 90.0)],
    "1 dot ahead (4 mm, 0)": [(4.0, 0.0)],
    "flanked (4 mm, +90 and -90)": [(4.0, 90.0), (4.0, -90.0)],
    "two on one side (4 mm, +60 and +120)": [(4.0, 60.0), (4.0, 120.0)],
    "close + far (4 mm +90, 15 mm -60)": [(4.0, 90.0), (15.0, -60.0)],
    "cluster on one side (4 +90, 6 +45, 6 +135)": [(4.0, 90.0), (6.0, 45.0), (6.0, 135.0)],
    "two ahead (4 mm, +20 and -20)": [(4.0, 20.0), (4.0, -20.0)],
    "1 dot far beside (15 mm, +90)": [(15.0, 90.0)],
    "blind": [],
}
TAGS = {"F": ("14f", "32b"), "F + turn": ("14f_rep", "32b_rep"), "F + turn + step": ("14f_reps", "32b_reps")}
# the group repulsion variants (scripts/grouprep_fit.py): `python repulsion_scenes.py gr`
TAGS_GR = {"F": ("14f", "32b"), "GR": ("14f_grT", "32b_grT"), "GR + step": ("14f_grTS", "32b_grTS"),
           "GR x2": ("14f_grT2", "32b_grT2"), "GR x4": ("14f_grT4", "32b_grT4"), "GR x8": ("14f_grT8", "32b_grT8"),
           "GR x n": ("14f_grTb1", "32b_grTb1"), "GR + collision": ("14f_grTC", "32b_grTC")}


class FixedScene:
    """Phantom moving dots at fixed (distance mm, bearing deg) from the fish's pre-bout pose, every bout."""

    def __init__(self, dots):
        self.dots_spec = dots
        self._pts = np.zeros((0, 2))

    def __call__(self, *a):
        raise RuntimeError

    def _place(self, x, y, h):
        if not self.dots_spec:
            return np.zeros((0, 2))
        r = np.array([d for d, _ in self.dots_spec]) * MFi.PX_MM
        a = h + np.radians([b for _, b in self.dots_spec])
        return np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1)

    def groups(self, t, time_s, x, y, h):
        self._pts = self._place(x, y, h)
        self._near = []
        if not len(self._pts):
            return []
        out, lg = MFi.neighbour_groups(x, y, self._pts)
        self._near = lg["near"]
        return out

    def group_nearest(self):
        return list(self._near)

    def neighbours_xy(self):
        return np.asarray(self._pts, float).reshape(-1, 2)

    def dots(self, t, time_s, x, y, h):
        p = self._pts
        return p[:, 0].copy(), p[:, 1].copy(), np.ones(len(p), bool), np.ones(len(p)), float(len(p))

    def block_time_s(self, t, t_now):
        return MFi.LATE_S


def run(tag, scene):
    _, SR = MFi._imports()
    ms = MFi.models(tag)
    spec = SCENES[scene]
    rows = []
    for fi, m in enumerate(ms):
        for r in range(PER_FOLD):
            seed = [11, fi, r]
            rng = np.random.default_rng(seed)
            rec = SR.make_synthetic_record(int(T_S / 0.05) + 100, MFi.start_pose(900 + fi, r), MFi.dish(1), fish_id=fi)
            prov = MFi.FishView(0, None, [], clock="late") if not spec else FixedScene(spec)
            o = m.sample(rec, rng, stim_provider=prov, stop_time_s=T_S, multi_rule="effects")
            dth = np.asarray(o["dtheta"], float)
            st = np.asarray(o["distance_px"], float) / MFi.PX_MM
            dt = np.asarray(o["dt_s"], float)
            rows.append(pd.DataFrame(dict(fish=f"{fi}_{r}", dth=dth, step=st, dt=dt)))
    return pd.concat(rows, ignore_index=True)


def summarise(b):
    big = np.abs(b.dth) > 10.0
    per = b.groupby("fish").apply(lambda g: g.step.sum() / g.dt.sum())     # travel speed per fish (mm/s)
    return dict(bouts=len(b), p_left=float(np.mean(b.dth[big] > 0)), mean_turn=float(b.dth.mean()),
                p_gt60=float(np.mean(np.abs(b.dth) > 60.0)), med_step=float(b.step.median()),
                med_abs_turn=float(np.median(np.abs(b.dth))), med_dt=float(b.dt.median()),
                speed=float(per.mean()))


def main(tagset="old"):
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    tags_ = TAGS_GR if tagset == "gr" else TAGS
    jobs = [(mod, seed, tag, sc) for mod, tags in tags_.items() for seed, tag in zip((0, 101), tags) for sc in SCENES]
    rows = []
    with ProcessPoolExecutor(8) as ex:
        futs = {ex.submit(run, tag, sc): (mod, seed, tag, sc) for mod, seed, tag, sc in jobs}
        for f, (mod, seed, tag, sc) in futs.items():
            rows.append(dict(model=mod, seed=seed, scene=sc, **summarise(f.result())))
            print(f"{mod:16s} s{seed:<3d} {sc:45s} done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / ("table_grouprep.csv" if tagset == "gr" else "table.csv"), index=False)
    return T


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "old")
