"""M6 and D4 trained FROM SCRATCH on the Larsch screen (no weights from the 15-fish models), evaluated like the zero-shot
models in Final_report_screendatatest.ipynb.

UNPUBLISHED DATA (Larsch lab): code only; the bout table and checkpoints stay in git-ignored folders.

1. TABLE (`build`). 100 sighted reference siblings, one per clutch (twin_test/screen_train_fish.json), segmented with the
   training data's own `segment_bouts.segment_fish` (repulsion_lik._segment_plate), 60-130 min in the dish. Kept: bouts
   with onset in a BOUT-LIKE dot episode, active-only like training (dt <= 10 s; the bout after a > 10 s rest dropped,
   its turn spans the rest). A new run at every tracking-segment break, rest and episode change, so no lag reaches
   across the continuous episodes that were dropped. Positions in model px (mm x 3.60) around the training dish centre,
   the 35 mm dish, the single dot as centroid and as the only individual dot, N = 1, moving.
2. TRAINING (`fold` / `run`). train_model1_colab.py's `rzlo` ladder (Model 42: the range input clipped below at 2 px =
   0.56 mm, so the dot's whole distance range is seen; the screen's 0.1 % quantile is 0.69 mm), both arms
   (M6 = S4xlo_adt_z2_cdt_tr_rc_al_w3ft, D4 = S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st), 5 folds over the 100 fish
   (= over clutches), seeds 0 and 101, with: the bout table above, the 35 mm dish (wall band in mm via DISH_MM = 70),
   the block clock HELD LATE (60 s into the block on every bout; the dot never appears or disappears here). Features
   that never vary here (dot count, static block) are constant and learn nothing.

    python screen_train.py build
    python screen_train.py fold SEED K ARM OUTDIR [--smoke]     # one fit (what `run` launches)
    python screen_train.py run [SLOTS]                          # 2 arms x 2 seeds x 5 folds, then merge
3. EVALUATION (`sim` / `analyse`), as screen_eval.py: 200 simulated fish per model and seed, sighted and DECOUPLED (the
   model sees the dot rotated by 90 / 180 / 270 deg; blind is an input these models never saw -- kept for kinematics only), fold
   model i mod 5, block clock late, the 35 mm dish (DISH_MM = 70 here; the zero-shot sims ran with 100, so their
   wall band sat at 1.4-7 mm instead of 2-10 mm). Real side: only fish from the 98 clutches NOT used in training
   (the pool); the zero-shot M6 and F are re-read against the same real samples, so every number in
   screen_trained_eval.json is paired. Readings: all 39 features, no radial, dot only, the 30 turn / step / interval
   features; floor from the pool's two clutch halves; individuality, kinematics, response by distance and the switch
   profile on the pool.

    python screen_train.py sim [WORKERS]                        # -> twin_test/screen_raw_sim_<sM6|sD4>_s<seed>_*.pkl
    python screen_train.py analyse [WORKERS]                    # -> twin_test/screen_trained_eval.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "Model")); sys.path.insert(0, str(ROOT))
import twin_test as TT  # noqa: E402

TABLE = TT.OUT / "screen_train_bouts.npz"      # plain arrays: built in one interpreter, trained in another
TABLE_META = TT.OUT / "screen_train_bouts.json"
FISH_JSON = TT.OUT / "screen_train_fish.json"
RUNS = ROOT / "Model" / "screen_trained_runs"
DST = {0: ROOT / "Model" / "screen_trained_s0_ckpts", 101: ROOT / "Model" / "screen_trained_s101_ckpts"}
ARMS = {"M6": "S4xlo_adt_z2_cdt_tr_rc_al_w3ft", "D4": "S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st",
        "M6dm": "S4xlo_adt_z2_cdt_tr_rc_al_w3ft_dm",     # + the dot motion direction input (Model 51)
        "D4dm": "S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st_dm"}
SEEDS = (0, 101)
CX, CY = 180.0, 174.0                         # the training dish centre (model px), as repulsion_lik
R_PX = TT.R_MM * TT.PX
LATE_S = 60.0
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SKEYS = {f"s{m}_s{s}": (m, s) for m in ARMS for s in SEEDS}       # sM6_s0, sM6_s101, sD4_s0, sD4_s101
ZERO = {"M6": ("26e", "30d"), "F": ("14f", "32b")}                    # the zero-shot units (screen_eval.py)


def _install_models():
    """twin_test._models also serves the screen-trained units (spawned workers import this module, so they get it)."""
    if getattr(TT, "_screen_trained", False):
        return
    orig = TT._models

    def _models(key):
        if key not in SKEYS:
            return orig(key)
        if key not in TT._SIM_CACHE:
            import contextlib
            import io
            import fit_MLP_border as MB
            import model_figures as MF
            MB.DISH_MM = 2.0 * TT.R_MM
            TT._sim_setup()
            m, s = SKEYS[key]
            with contextlib.redirect_stdout(io.StringIO()):
                ms, _ = MF.fold_models(str(DST[s]), ARMS[m])
            TT._SIM_CACHE[key] = [ms[k] for k in sorted(ms)]
        return TT._SIM_CACHE[key]
    TT._models = _models
    TT._screen_trained = True


_install_models()


def _wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def _table(fid_new, b):
    """One fish's training rows from its segment_fish bout table (+ dot_cx, dot_cy, kind)."""
    import segment_bouts as SB
    b = b.sort_values("frame_onset").reset_index(drop=True)
    b = b[np.isfinite(b.dtheta_deg.to_numpy(float))].reset_index(drop=True)
    seg = b["segment_id"].to_numpy()
    dt = b.dt_s.to_numpy(float)
    kind = b.kind.to_numpy().astype(str)
    same = np.r_[False, seg[1:] == seg[:-1]]
    rest_before = same & np.r_[False, dt[:-1] > TT.MAX_DT]       # the previous interval was a rest: turn spans it
    keep = (dt <= TT.MAX_DT) & ~rest_before & (kind == "dot_bout") & np.isfinite(b.dot_cx.to_numpy(float))
    prev_kept = np.r_[False, keep[:-1]]
    newrun = ~(same & prev_kept & np.r_[False, kind[1:] == kind[:-1]])
    run = np.cumsum(newrun)
    b = b.assign(run_id=run)[keep].reset_index(drop=True)
    rid = b.run_id.to_numpy()
    rp = np.zeros(len(b), int)
    for t in range(1, len(b)):
        rp[t] = rp[t - 1] + 1 if rid[t] == rid[t - 1] else 0
    x, y = b.x_px.to_numpy(float), b.y_px.to_numpy(float)
    dcx, dcy = b.dot_cx.to_numpy(float), b.dot_cy.to_numpy(float)
    f_ori = b.f_ori_rad.to_numpy(float)
    dd = np.hypot(dcx - x, dcy - y)
    n = len(b)
    out = pd.DataFrame(dict(
        fish_id=fid_new, segment_id=0, bout_index=np.arange(n), frame_onset=b.frame_onset.to_numpy(int),
        t_onset_s=b.frame_onset.to_numpy(float) / TT.FPS, dtheta_deg=b.dtheta_deg.to_numpy(float),
        dt_s=b.dt_s.to_numpy(float), distance_px=b.distance_px.to_numpy(float), heading_rad=b.heading_rad.to_numpy(float),
        f_ori_rad=f_ori, x_px=x, y_px=y, stimulus="screen_dot_bout",
        stim_angle_rad=_wrap(np.arctan2(dcy - y, dcx - x) - f_ori),
        n_dots_left=1.0, n_dots_right=0.0, speed_left=3.5, speed_right=3.5, moving=True, active=True,
        dist_to_border_norm=np.clip((R_PX - np.hypot(x - CX, y - CY)) / R_PX, 0.0, 1.0),
        dot_cx=dcx, dot_cy=dcy, dot_dist_px=dd, dot_prox=1.0 / (1.0 + (dd / SB.PX_PER_MM) / SB.R0_MM),
        run_id=rid, run_pos=rp, d0_x=dcx, d0_y=dcy))
    for k in range(1, 8):
        out[f"d{k}_x"] = np.nan; out[f"d{k}_y"] = np.nan
    return out


def build(workers=10):
    import repulsion_lik as RL
    ids = json.loads(FISH_JSON.read_text())
    ids = ids["fish"] if isinstance(ids, dict) else ids
    f = TT.fish_table().set_index("fish_id", drop=False).loc[[int(i) for i in ids]]
    jobs = [(int(p), g[["fish_id", "well", "centre_x_px", "centre_y_px", "px_per_mm"]].to_dict("records"))
            for p, g in f.groupby("plate")]
    bouts = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(RL._segment_plate, jobs):
            bouts.update(r)
    rows, idmap = [], {}
    for new, fid in enumerate(int(i) for i in ids):            # model ids 0..99 in the json's (random) order
        if fid not in bouts:
            continue
        idmap[new] = fid
        rows.append(_table(new, bouts[fid]))
    df = pd.concat(rows, ignore_index=True)
    clutch = dict(zip(f.fish_id.astype(int), f.clutch))
    np.savez_compressed(TABLE, **{c: (df[c].to_numpy().astype("U") if df[c].dtype == object else df[c].to_numpy())
                                  for c in df.columns})
    TABLE_META.write_text(json.dumps(dict(idmap=idmap, clutch={k: str(clutch[v]) for k, v in idmap.items()})))
    g = df.groupby("fish_id")
    print(f"wrote {TABLE.name}: {len(df):,} bouts, {df.fish_id.nunique()} fish (median {int(g.size().median())}), "
          f"{df.groupby(['fish_id', 'run_id']).ngroups:,} runs; dot distance min {df.dot_dist_px.min() / TT.PX:.2f} mm, "
          f"0.1 % {df.dot_dist_px.quantile(0.001) / TT.PX:.2f} mm; dnorm == 0 on {np.mean(df.dist_to_border_norm == 0):.4f}")


def fold(seed, k, arm, outdir, smoke=False):
    """One fit of train_model1_colab.py, with the screen patched in (see the module docstring)."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    import train_model1_colab as T
    import fit_MLP_border as MB
    z = np.load(TABLE, allow_pickle=False)
    df = pd.DataFrame({c: z[c] for c in z.files})
    fish = sorted(df.fish_id.unique())
    T.OUT = outdir
    T.E.load_bouts = lambda path=None: df
    MB.load_physical_dish = lambda path=None: pd.DataFrame(
        dict(cx=CX, cy=CY, radius=R_PX), index=pd.Index([int(f) for f in fish], name="fish_id"))
    MB.DISH_MM = 2.0 * TT.R_MM

    def late(rec, fps=MB._FPS):
        fr = np.asarray(getattr(rec, "frame_onset", np.arange(rec.n) * fps), float)
        return fr - LATE_S * fps, np.zeros(rec.n, np.int64), fr
    MB.block_onsets = late
    sys.argv = ["train_model1_colab.py", "--ladder", "rzlo_dm" if arm.endswith("_dm") else "rzlo", "--arms", arm, "--fold", str(k), "--seed", str(seed),
                "--threads", "1"] + (["--smoke"] if smoke else [])
    T.main()
    print("SCREEN FIT DONE", flush=True)


def run(slots=18, only=None):
    sys.path.insert(0, str(ROOT / "Model"))
    from rollout_model1_local import merge_ckpt_dirs
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE), str(ROOT), str(ROOT / "Model")]),
               PYTHONIOENCODING="utf-8", MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sem = threading.Semaphore(slots)
    arms = [a for m, a in ARMS.items() if only is None or m == only]
    jobs = [(s, a, k) for s in SEEDS for a in arms for k in range(5)]
    rc_of, t0 = {}, time.time()

    def one(s, a, k):
        d = RUNS / f"s{s}" / f"{a}_fold{k}"
        lg = d / "log.txt"
        if (d / f"{a}_fold{k}.pt").exists() and lg.exists() and "SCREEN FIT DONE" in lg.read_text(errors="replace"):
            rc_of[(s, a, k)] = 0
            return
        with sem:
            d.mkdir(parents=True, exist_ok=True)
            with open(lg, "w", encoding="utf-8") as fh:
                rc_of[(s, a, k)] = subprocess.Popen(
                    [str(PY), "-u", "-W", "ignore", str(HERE / "screen_train.py"), "fold", str(s), str(k), a, str(d)],
                    cwd=str(d), env=env, stdout=fh, stderr=subprocess.STDOUT).wait()
        print(f"[{time.time() - t0:6.0f}s] seed {s} {a} fold {k}: exit {rc_of[(s, a, k)]}", flush=True)
    ths = [threading.Thread(target=one, args=j) for j in jobs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for s in SEEDS:
        for a in arms:
            if all(rc_of.get((s, a, k)) == 0 for k in range(5)):
                for k in range(5):
                    merge_ckpt_dirs(DST[s], RUNS / f"s{s}" / f"{a}_fold{k}")
                print(f"merged seed {s} {a} -> {DST[s].name}", flush=True)
            else:
                print(f"seed {s} {a}: FAILED folds {[k for k in range(5) if rc_of.get((s, a, k))]}", flush=True)
    print(f"SCREEN TRAINING ALL DONE in {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
# The decoupled reference sees the dot rotated about the dish centre by 90, 180 or 270 deg (fish i: TT.ROTS[i % 3]), the
# rotations of the real fish's chance reference: a fish following a dot rotated by 180 deg alone sits on the far side of
# the true one (2.5-3 mm FARTHER than chance), the three together average to chance.


def _sim_decoupled(args):
    """screen_eval._sim, but the model SEES the dot rotated by TT.ROTS[i % 3] about the dish centre while the record keeps the
    true dot: the screen-trained models never saw a dish without a dot (blind = an input they were never trained on;
    blind M6 steps 1.4 mm against 4.2), so their chance reference keeps a dot in view, uncoupled from the measured one.
    Same random stream as the sighted fish (key, i)."""
    import contextlib
    import io
    import zlib
    import stim_rollout as SR
    key, i = args
    S = TT._sim_setup()
    m = TT._models(key)[i % 5]
    cx, cy, R = S["dish"]
    c, s = np.cos(TT.ROTS[i % 3]), np.sin(TT.ROTS[i % 3])
    rx, ry = c * S["dxa"] - s * S["dya"], s * S["dxa"] + c * S["dya"]
    pose = S["ref"].sample_start_pose(np.random.default_rng([TT.SIM_SEED, 1, i]), S["dish"])
    prov = TT.PathProvider(cx + rx * TT.PX, cy + ry * TT.PX, show=True)
    T = (TT.WIN[1] - TT.WIN[0]) / TT.FPS
    rec = SR.make_synthetic_record(int(T / 0.05) + 100, pose, S["dish"], fish_id=0)
    rng = np.random.default_rng([TT.SIM_SEED, zlib.crc32(key.encode()), i])
    with contextlib.redirect_stdout(io.StringIO()):
        out = m.sample(rec, rng, stim_provider=prov, stop_time_s=T)
    t = np.asarray(out["t_onset_s"], float)
    n = len(t)
    x = (np.asarray(out["x_px"], float)[:n] - cx) / TT.PX
    y = (np.asarray(out["y_px"], float)[:n] - cy) / TT.PX
    f = np.clip(TT.WIN[0] + np.round(t * TT.FPS).astype(int), 0, len(S["dxa"]) - 1)
    return i, dict(f=f.astype(np.int32), x=x.astype(np.float32), y=y.astype(np.float32), seg=np.zeros(n, np.int32),
                   dx=S["dxa"][f].astype(np.float32), dy=S["dya"][f].astype(np.float32))


def sim(workers=8, only=None):
    import pickle
    import screen_eval as SE
    for key in [k for k in SKEYS if only is None or SKEYS[k][0] == only]:
        for show in ("sighted", "blind", "decoupled"):
            nm = TT.OUT / f"screen_raw_sim_{key}_{show}.pkl"
            if nm.exists():
                continue
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=workers) as ex:
                if show == "decoupled":
                    res = dict(ex.map(_sim_decoupled, [(key, i) for i in range(SE.N_SIM)], chunksize=4))
                else:
                    res = {i: r for _, _, i, r in ex.map(SE._sim, [(key, i, show == "sighted") for i in range(SE.N_SIM)],
                                                         chunksize=4)}
            pickle.dump(res, open(nm, "wb"))
            print(f"wrote {nm.name} ({len(res)} fish, {time.time() - t0:.0f}s)", flush=True)


def _measure_from_raw(r):
    """twin_test.measure sums (response by distance) from a raw simulated fish, as twin_test._sim_one computes them."""
    S = TT._sim_setup()
    f = r["f"].astype(int); x = r["x"].astype(float); y = r["y"].astype(float)
    th = np.r_[np.arctan2(np.diff(y), np.diff(x)), np.nan]
    th_prev = np.r_[np.nan, th[:-1]]
    turn = np.degrees(TT.wrap(th - th_prev))
    x_next = np.r_[x[1:], np.nan]; y_next = np.r_[y[1:], np.nan]
    dvx, dvy = TT.dot_velocity(S["dxa"], S["dya"], f)
    dt = np.r_[np.diff(f) / TT.FPS, np.nan]
    return TT.measure(x, y, x_next, y_next, th_prev, turn, dt, S["kind"][f], r["dx"].astype(float),
                      r["dy"].astype(float), dvx, dvy)


def _mfr(args):
    i, r = args
    return i, _measure_from_raw(r)


def analyse(workers=10, reps=None):
    import pickle
    import repulsion_fit as RF
    import screen_eval as SE
    reps = reps or SE.REPS
    meta = json.loads(TABLE_META.read_text())
    used = set(meta["clutch"].values())
    R = pickle.load(open(TT.OUT / "screen_raw_real.pkl", "rb"))
    pool = {i: r for i, r in R["res"].items() if str(R["clutch"].get(int(i))) not in used}
    print(f"real pool: {len(pool)} fish from clutches not used in training ({len(used)} used)", flush=True)
    wr = SE.windows_of(pool, workers)
    ids = np.array(sorted(wr))
    halves = RF.clutch_split()
    idA = np.array([i for i in ids if R["clutch"][int(i)] in set(halves["fit"])])
    idB = np.array([i for i in ids if R["clutch"][int(i)] in set(halves["test"])])
    units = {}
    for m, keys in ZERO.items():
        for s, k in zip(SEEDS, keys):
            units[f"{m} zero-shot s{s} sighted"] = f"screen_raw_sim_{k}_sighted.pkl"
    for k, (m, s) in SKEYS.items():
        for show in ("sighted", "decoupled", "blind"):         # blind: kinematics only (an untrained input here)
            units[f"{m} screen s{s} {show}"] = f"screen_raw_sim_{k}_{show}.pkl"
    sims = {u: pickle.load(open(TT.OUT / fn, "rb")) for u, fn in units.items()}
    wins = {}
    for u, res in sims.items():
        wins[u] = SE.windows_of(res, workers)
        print(f"  {u}: {len(wins[u])} fish, {sum(len(v[0]) for v in wins[u].values())} windows", flush=True)
    readings = dict(SE.READINGS, **{"30 features": "k30"})

    def sel(X, rk):
        return X[:, :30] if rk == "k30" else SE.cols(X, rk)
    out = dict(n_pool_fish=int(len(wr)), n_pool_windows=int(sum(len(v[0]) for v in wr.values())),
               n_floor=[int(len(idA)), int(len(idB))], n_train_fish=len(meta["idmap"]),
               n_train_bouts=int(len(np.load(TABLE)["fish_id"])), reps=reps, n_real_sample=SE.N_REAL,
               auc={}, indiv={}, kin=dict(real=SE.kinematics(wr)), curves={}, prof={}, attr={})
    for reading, rk in readings.items():
        rows = {"real vs real": []}
        for rep in range(reps):
            rng = np.random.default_rng([2026, rep])
            a = rng.choice(idA, SE.N_REAL, replace=False); b = rng.choice(idB, SE.N_REAL, replace=False)
            XA, gA = SE.stack(wr, a); XB, gB = SE.stack(wr, b)
            rows["real vs real"].append(SE.c2st(sel(XA, rk), gA, sel(XB, rk), gB, seed=rep)[0])
        for u, ws in wins.items():
            if u.endswith(" blind"):
                continue
            XS, gS = SE.stack(ws, np.array(sorted(ws)), offset=10 ** 7)
            vals = []
            for rep in range(reps):
                rng = np.random.default_rng([2027, rep])
                XR, gR = SE.stack(wr, rng.choice(ids, SE.N_REAL, replace=False))
                vals.append(SE.c2st(sel(XR, rk), gR, sel(XS, rk), gS, seed=rep)[0])
            rows[u] = vals
            print(f"  C2ST [{reading}] {u}: {np.mean(vals):.3f} +/- {np.std(vals, ddof=1):.3f}", flush=True)
        print(f"  C2ST [{reading}] real vs real: {np.mean(rows['real vs real']):.3f}", flush=True)
        out["auc"][reading] = rows
    for u, ws in wins.items():
        if u.endswith(" sighted"):
            out["indiv"][u[:-len(" sighted")]] = SE.individuality(wr, ws)
        out["kin"][u] = SE.kinematics(ws)
    # response by distance (real: twin_test's real.pkl, pool fish only)
    T = pickle.load(open(TT.OUT / "real.pkl", "rb"))
    ref = [f for f, g, c in zip(T["meta"]["fish_id"], T["meta"]["group"], T["meta"]["clutch"])
           if g == "ref" and str(c) not in used]
    out["curves"]["real"] = SE.curves(T["res"], ref)
    for m, keys in ZERO.items():
        for s, k in zip(SEEDS, keys):
            r = pickle.load(open(TT.OUT / f"sim_{k}_sighted.pkl", "rb"))["res"]
            out["curves"][f"{m} zero-shot s{s}"] = SE.curves(r, list(r))
    for k, (m, s) in SKEYS.items():
        with ProcessPoolExecutor(max_workers=workers) as ex:
            r = dict(ex.map(_mfr, list(sims[f"{m} screen s{s} sighted"].items()), chunksize=8))
        out["curves"][f"{m} screen s{s}"] = SE.curves(r, list(r))
    out["dbins"] = TT.DBINS.tolist()
    switch_profiles(out, pool, sims)
    (TT.OUT / "screen_trained_eval.json").write_text(json.dumps(out, indent=1))
    print("wrote screen_trained_eval.json", flush=True)


def switch_profiles(out, pool, sims):
    """Distance to the dot across the switch to bout-like motion, and how much closer than chance. Chance = the mean
    distance to the dot rotated by 90 / 180 / 270 deg; for a DECOUPLED fish the rotation it saw is left out (it follows
    that dot, so including it would put the fish closer to 'chance' than to nothing)."""
    import screen_eval as SE
    clock = SE._episode_clock(SE._kind())

    def both(res, decoupled=False):
        d = np.array([SE.fish_profile(r, clock) for r in res.values()])
        d0 = np.array([np.nanmean([SE.fish_profile(r, clock, a) for j, a in enumerate(TT.ROTS)
                                   if not (decoupled and j == int(i) % 3)], 0) for i, r in res.items()])
        return d, d0
    d, d0 = both(pool)
    out["prof"]["real"] = SE._boot_profile(d); out["prof"]["real, rotated dot"] = SE._boot_profile(d0)
    out["attr"]["real"] = SE._boot_profile(d0 - d)
    for u, res in sims.items():
        if u.endswith(" blind"):
            continue
        d, d0 = both(res, decoupled=u.endswith(" decoupled"))
        out["prof"][u] = SE._boot_profile(d); out["attr"][u] = SE._boot_profile(d0 - d)
    out["pbins"] = SE.PBINS.tolist()


def profiles_only():
    """Recompute only the switch profiles of screen_trained_eval.json (the rest kept)."""
    import pickle
    meta = json.loads(TABLE_META.read_text())
    used = set(meta["clutch"].values())
    R = pickle.load(open(TT.OUT / "screen_raw_real.pkl", "rb"))
    pool = {i: r for i, r in R["res"].items() if str(R["clutch"].get(int(i))) not in used}
    p = TT.OUT / "screen_trained_eval.json"
    out = json.loads(p.read_text())
    sims = {}
    for m, keys in ZERO.items():
        for s, k in zip(SEEDS, keys):
            sims[f"{m} zero-shot s{s} sighted"] = pickle.load(open(TT.OUT / f"screen_raw_sim_{k}_sighted.pkl", "rb"))
    for k, (m, s) in SKEYS.items():
        for show in ("sighted", "decoupled"):
            sims[f"{m} screen s{s} {show}"] = pickle.load(open(TT.OUT / f"screen_raw_sim_{k}_{show}.pkl", "rb"))
    switch_profiles(out, pool, sims)
    p.write_text(json.dumps(out, indent=1))
    print("updated the switch profiles in screen_trained_eval.json", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "build":
        build()
    elif a[0] == "fold":
        fold(int(a[1]), int(a[2]), a[3], a[4], smoke="--smoke" in a)
    elif a[0] == "run":
        run(int(a[1]) if len(a) > 1 else 18, a[2] if len(a) > 2 else None)       # [SLOTS [M6|D4|M6dm]]
    elif a[0] == "sim":
        sim(int(a[1]) if len(a) > 1 else 8, a[2] if len(a) > 2 else None)       # [WORKERS [M6|D4]]
    elif a[0] == "analyse":
        analyse(int(a[1]) if len(a) > 1 else 10)
    elif a[0] == "profiles":
        profiles_only()
