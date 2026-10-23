"""The C2ST protocol's windowing asymmetry, measured (2026-09-20).

The ladder's C2ST windows the REAL side inside runs (a 60-bout window never straddles a > 10 s pause;
``generative_metrics._window_starts`` with the record's ``run_id``) and the GENERATED side straight
through (a rollout never pauses). The evaluation rollouts are BOUT-indexed on the record's stimulus
(``stim_rollout.RecordAnchoredProvider``: generated bout t sees what real bout t saw), so the two sides
can be windowed on IDENTICAL bout indices -- which exposes three things the plain protocol folds into
every AUC: (1) the generated side hands the classifier ~1.6x more windows per replicate (a power
effect, not the model); (2) the bouts the run restriction drops on the real side are richer in
moving-dot bouts, so real windows carry ~5 points less moving-dot content; (3) real windows start
after a pause, generated ones anywhere.

    condition_share_table()                      real vs generated windowed bouts by stimulus condition
    matched_c2st_table(ckpt_dir, arm, rej)        the AUC under plain / count-balanced / matched windowing
    profile_pause_split()                         real blocks with vs without a pause, per condition

Nothing here changes the harness; it measures what the harness's numbers contain.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from twin import paths


W, REPS, SEED = 60, 8, 0
FPS = 30.0


def _ckpt_dir(d):
    d = Path(d)
    for q in (d, paths.CKPTS / d):
        if (q / "manifest_model1.json").exists():
            return q
    raise FileNotFoundError(f"no manifest_model1.json under {d}")


def _records():
    from twin import evaluate as E
    from twin import features as MB
    return MB.build_border_records(E.load_bouts())


def _cond(rec):
    mv = np.asarray(rec.moving, bool)
    nd = np.nan_to_num(np.asarray(rec.n_dots, float))
    return np.where(mv & (nd > 0), "moving", np.where(nd > 0, "static", "none"))


def condition_share_table(recs=None):
    """Share of bouts by stimulus condition inside the windows each side of the C2ST uses. The generated
    side is bout-indexed on the record, so its windowed bouts are the plain tiling of the record's own
    bouts; the real side is the run-restricted tiling (and its steady-state subset)."""
    from twin import generative_metrics as GM
    recs = recs or _records()
    pools = {"all active bouts": [], "real windows (inside runs)": [], "real steady-state windows": [],
             "generated windows (tiled straight through)": []}
    per = []
    for f, rec in sorted(recs.items()):
        c = _cond(rec); n = len(c); runs = rec.run_id
        sel = {}
        for k, kw in (("real windows (inside runs)", dict(runs=runs)),
                      ("real steady-state windows", dict(runs=runs, lead=GM.INTERIOR[0], tail=GM.INTERIOR[1])),
                      ("generated windows (tiled straight through)", dict())):
            m = np.zeros(n, bool)
            for a in GM._window_starts(n, W, W, **kw):
                m[a:a + W] = True
            sel[k] = m
            pools[k].append(c[m])
        pools["all active bouts"].append(c)
        per.append(dict(fish=f, bouts=n, real_window_coverage=sel["real windows (inside runs)"].mean(),
                        moving_all=(c == "moving").mean(),
                        moving_real_windows=(c[sel["real windows (inside runs)"]] == "moving").mean() if sel["real windows (inside runs)"].any() else np.nan,
                        moving_generated_windows=(c[sel["generated windows (tiled straight through)"]] == "moving").mean()))
    rows = []
    for k, v in pools.items():
        v = np.concatenate(v)
        rows.append(dict(windows=k, bouts=len(v), moving=np.mean(v == "moving"), static=np.mean(v == "static"), none=np.mean(v == "none")))
    tab = pd.DataFrame(rows).set_index("windows")
    pd.set_option("display.width", 200)
    print("share of windowed bouts by stimulus condition, pooled over the 15 fish:")
    print(tab.round(3).to_string())
    print("\nper fish (moving-dot share of the bouts inside the real vs the generated windows):")
    print(pd.DataFrame(per).round(3).to_string(index=False))
    return tab, pd.DataFrame(per)


def _job(job):
    """(ckpt, fish, rej, reps, seed) -> {kind: (real, gens_matched, gens_plain, gens_subsampled)}."""
    ckpt, fid, rej, reps, seed = job
    import torch
    torch.set_num_threads(1)
    from twin import emission as MX
    from twin import generative_metrics as GM
    recs = _records()
    rec = recs[fid]
    m = MX.load_mix(str(ckpt))
    if rej:
        m._cfg_feat = {**m._cfg_feat, "wall_reject": dict(rej)}
    runs = getattr(rec, "run_id", None)
    real_rad = GM._radial_of(rec, rec.x_px, rec.y_px)
    dd = getattr(rec, "dot_dist_px", None)
    out = {}
    for kind, (lead, tail) in (("all", (0, 0)), ("steady state", GM.INTERIOR)):
        real = GM.window_features_traj(rec.dtheta, rec.symbol, rec.distance_px, rec.dt_s, dd, W=W, runs=runs,
                                       lead=lead, tail=tail, radial=real_rad)
        matched, plain, sub = [], [], []
        for rep in range(int(reps)):
            g = m.sample(rec, np.random.default_rng([int(seed), int(fid), rep]), **GM._stim_kw(m, rec, None))
            gx, gy = np.asarray(g["x_px"], float), np.asarray(g["y_px"], float)
            gdd = np.hypot(gx - np.asarray(g.get("dot_cx", rec.dot_cx), float),
                           gy - np.asarray(g.get("dot_cy", rec.dot_cy), float))          # the re-anchored dots
            grad = GM._radial_of(rec, gx, gy)
            n = min(len(g["dtheta"]), len(rec.dtheta))
            matched.append(GM.window_features_traj(g["dtheta"][:n], g["symbol"][:n], g["distance_px"][:n], g["dt_s"][:n],
                                                   gdd[:n], W=W, runs=(None if runs is None else np.asarray(runs)[:n]),
                                                   lead=lead, tail=tail, radial=grad[:n]))
            gp = GM.window_features_traj(g["dtheta"], g["symbol"], g["distance_px"], g["dt_s"], gdd, W=W, lead=lead, radial=grad)
            plain.append(gp)
            nk = min(len(gp), len(real))
            sub.append(gp[np.sort(np.random.default_rng([int(seed), int(fid), rep, 7]).choice(len(gp), nk, replace=False))] if nk else gp[:0])
        out[kind] = (real, matched, plain, sub)
    return fid, out


def matched_c2st_table(ckpt_dir, arm, rej=None, reps=REPS, seed=SEED, jobs=10):
    """The held-out C2ST of ``arm`` under three windowings of the SAME rollouts (the ladder's seeds):
    plain tiling (the harness), plain tiling subsampled to the real window count (1 generated window
    per real one), and generated windows matched to the real run ids (identical bout indices, 1:1) --
    on all windows and on the steady-state windows, with feature blocks dropped. Cached next to the
    checkpoints as ``protocol_windows_<arm>.npz``, recomputed when a checkpoint is newer than the cache."""
    from twin import generative_metrics as GM
    from twin import registry
    d = _ckpt_dir(ckpt_dir)
    cache = d / f"protocol_windows_{arm}.npz"
    if cache.exists() and not registry.stale(cache, [d]):
        res = np.load(cache, allow_pickle=True)["res"].item()
    else:
        mani = json.loads((d / "manifest_model1.json").read_text())
        jl = [(d / e["file"], int(f), rej, reps, seed) for e in mani if e["arm"] == arm for f in e["heldout"]]
        with ProcessPoolExecutor(max_workers=min(jobs, len(jl))) as ex:
            res = dict(ex.map(_job, jl))
        np.savez(cache, res=np.array(res, dtype=object))
    dim = next(iter(res.values()))["all"][0].shape[1]
    names = GM.traj_names_for_dim(dim)
    sets = {"all features": list(range(dim)), "no radial": [i for i, n in enumerate(names) if not n.startswith("rad")],
            "three heads": [i for i, n in enumerate(names) if not n.startswith(("rad", "dotdist"))]}
    rows = []
    for kind in ("all", "steady state"):
        for side, gi in (("plain tiling (the harness)", 2), ("plain tiling, 1 generated window per real one", 3),
                         ("matched to the real runs (same bouts, 1:1)", 1)):
            r = dict(windows=kind, generated=side,
                     n_real=sum(len(res[f][kind][0]) for f in res), n_gen_per_rep=sum(len(res[f][kind][gi][0]) for f in res))
            for lab, cols in sets.items():
                wins = [{f: (res[f][kind][0][:, cols], res[f][kind][gi][k][:, cols]) for f in res
                         if len(res[f][kind][0]) and len(res[f][kind][gi][k])} for k in range(reps)]
                mean, sd, per = GM.c2st_auc_replicates(wins, seed=seed)
                r[lab] = mean; r[lab + " se"] = sd / np.sqrt(max(len(per), 1))
            rows.append(r)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    print(f"held-out C2ST of {arm} under three windowings of the same rollouts (floor: real halves, 0.516):")
    print(tab.round(3).to_string(index=False))
    print("(the harness's steady-state reading subsamples the generated side to the fish's standard ratio; the rows above do not)")
    return tab


def profile_pause_split(recs=None, within_s=(20.0, 60.0)):
    """Real stimulus blocks with vs without a pause (a > 10 s gap inside the block, or the fish inactive
    at its start or end), per condition: bouts and the mean distance to the dots early (0-20 s) and
    late (20-60 s). The dot-profile plot pools both; a never-pausing model is like-for-like with the
    fully active blocks only."""
    from twin import features as MB
    recs = recs or _records()
    rows = []
    for f, rec in recs.items():
        s0, bid, fr = MB.block_onsets(rec)
        tb = (fr - s0) / FPS
        c = _cond(rec); runs = np.asarray(rec.run_id)
        dd = np.asarray(rec.dot_dist_px, float) / 3.6
        for b in np.unique(bid):
            i = bid == b
            cond = "moving" if (c[i] == "moving").any() else ("static" if (c[i] == "static").any() else "none")
            if cond == "none":
                continue
            paused = len(np.unique(runs[i])) > 1 or tb[i].min() > 5 or tb[i].max() < 55
            early = dd[i & (tb < within_s[0])]; late = dd[i & (tb >= within_s[0]) & (tb < within_s[1])]
            rows.append(dict(fish=f, cond=cond, paused=paused, bouts=int(i.sum()),
                             early=np.nanmean(early) if len(early) else np.nan, late=np.nanmean(late) if len(late) else np.nan))
    d = pd.DataFrame(rows)
    tab = d.groupby(["cond", "paused"]).agg(blocks=("bouts", "size"), bouts=("bouts", "sum"),
                                            dist_0_20s_mm=("early", "mean"), dist_20_60s_mm=("late", "mean"))
    pd.set_option("display.width", 200)
    print("real stimulus blocks with vs without a pause (mean distance to the dots over the block's active bouts, mm):")
    print(tab.round(1).to_string())
    print("share of blocks with a pause:", d.groupby("cond").paused.mean().round(2).to_dict())
    return tab
