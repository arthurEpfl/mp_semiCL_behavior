"""THE MODELS ON THE LARSCH SCREEN WITH THE THESIS'S OWN YARDSTICKS (2026-10-19; Final_report_screendatatest.ipynb).

The screen (twin_test.py): one fish per 70 mm dish, ONE dot always on screen, 5-min episodes alternating bout-like and
continuous motion, 60-130 min. Only the BOUT-LIKE episodes are used here. Models unchanged (M6 26e | 30d, F 14f | 32b;
D4 for the curves), block clock held late (F's correction off), 200 simulated fish per unit, plus each model's BLIND
twin (dot hidden). Real side: the 2819 sighted reference siblings, bouts segmented as for our data; for the classifier,
a RANDOM SAMPLE of fish (no age selection).

  1. matched C2ST (the thesis's 39 window features, generative_metrics.window_features_traj): 60-bout windows that lie
     inside one bout-like episode and never cross a > 10 s interval; per replicate a fresh random sample of 200 real
     fish, all 200 simulated fish, the larger side subsampled to equal window counts; random forest (200 trees),
     5-fold cross-validation grouped by fish; 8 replicates. Readings: all features, without the 4 radial features, and
     the 5 dot-distance features only. Floor: real vs real, two random 200-fish samples from the two clutch halves.
  2. individuality: between-fish SD of per-fish window means and mean within-fish SD across windows, simulated over
     real (turn magnitude, displacement, interval, radial position); all real fish; bootstrap over fish.
  3. response by distance (the screen's equivalent of the dot profiles): turn toward, distance gain per bout and bouts
     per distance bin, each against the rotated dot (twin_test caches), 95 % bootstrap over fish.

    uv run --with pyarrow python screen_eval.py real        # -> Model/twin_test/screen_raw_real.pkl
    python screen_eval.py sim                               # -> Model/twin_test/screen_raw_sim_<key>_<sighted|blind>.pkl
    python screen_eval.py analyse                           # -> Model/twin_test/screen_eval.json
"""
from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from twin import twin_test as TT  # noqa: E402

OUT = TT.OUT
PX = TT.PX
W = 60
KIND = "dot_bout"
N_SIM = 200
N_REAL = 200
REPS = 8
UNITS = {"M6": ("26e", "30d"), "F": ("14f", "32b"), "D4": ("11e", "32a")}
C2ST_UNITS = {"M6": ("26e", "30d"), "F": ("14f", "32b")}
SEEDS = (0, 101)
NAMES = None


# --------------------------------------------------------------------------- #
# Raw bouts
# --------------------------------------------------------------------------- #
def _real_plate(args):
    plate, rows = args
    import pyarrow.parquet as pq
    import loader
    from twin import segment_bouts as SB
    z = zipfile.ZipFile(TT.ZIP)
    cols = ["frame"] + [f"{k}_{int(r['well']):02d}" for r in rows for k in ("x", "y", "heading", "flag")]
    tab = pq.read_table(io.BytesIO(z.read(f"schizscreen_twin_v1/trajectories/plate_{plate:03d}.parquet")), columns=cols,
                        filters=[("frame", ">=", TT.WIN[0]), ("frame", "<", TT.WIN[1])]).to_pandas()
    st = TT.stimulus()
    frames = tab.frame.to_numpy()
    epi = st.episode.to_numpy().astype(str)
    res = {}
    for r in rows:
        w = int(r["well"])
        x, y, o = loader.to_mm(tab[f"x_{w:02d}"].to_numpy(float), tab[f"y_{w:02d}"].to_numpy(float),
                               tab[f"heading_{w:02d}"].to_numpy(float) / loader.HEADING_SCALE,
                               r["centre_x_px"], r["centre_y_px"], r["px_per_mm"])
        fl = tab[f"flag_{w:02d}"].fillna(1).to_numpy().astype(np.uint8)
        bad = ((fl & loader.DEFAULT_MASK) > 0) | ~np.isfinite(x)
        df = pd.DataFrame({"frame": frames, "stimulus": epi[frames],
                           "f_x": np.where(bad, np.nan, x * PX), "f_y": np.where(bad, np.nan, y * PX),
                           "f_ori": np.where(bad, np.nan, o)})
        with contextlib.redirect_stdout(io.StringIO()):
            b = pd.DataFrame(SB.segment_fish(df, "f", int(r["fish_id"]), 1.0))
        if b.empty:
            continue
        seg = b["segment_id"].to_numpy() if "segment_id" in b else np.zeros(len(b), int)
        dxa, dya = TT.dot_mm(st, r["centre_x_px"], r["centre_y_px"], r["px_per_mm"])
        f = b.frame_onset.to_numpy(int)
        res[int(r["fish_id"])] = dict(f=f.astype(np.int32), x=(b.x_px.to_numpy(float) / PX).astype(np.float32),
                                      y=(b.y_px.to_numpy(float) / PX).astype(np.float32), seg=seg.astype(np.int32),
                                      dx=dxa[f].astype(np.float32), dy=dya[f].astype(np.float32))
    return res


def run_real(workers=10):
    f = TT.fish_table()
    f = f[f.group == "ref"]
    jobs = [(int(p), g[["fish_id", "well", "centre_x_px", "centre_y_px", "px_per_mm"]].to_dict("records"))
            for p, g in f.groupby("plate")]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_real_plate, jobs)):
            res.update(r)
            if i % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} plates, {len(res)} fish", flush=True)
    clutch = dict(zip(f.fish_id.astype(int), f.clutch))
    pickle.dump(dict(res=res, clutch=clutch), open(OUT / "screen_raw_real.pkl", "wb"))
    print(f"wrote screen_raw_real.pkl: {len(res)} fish", flush=True)


def _sim(args):
    key, i, show = args
    from twin import stim_rollout as SR
    S = TT._sim_setup()
    m = TT._models(key)[i % 5]
    cx, cy, R = S["dish"]
    pose = S["ref"].sample_start_pose(np.random.default_rng([TT.SIM_SEED, 1, i]), S["dish"])
    prov = TT.PathProvider(cx + S["dxa"] * PX, cy + S["dya"] * PX, show=show)
    T = (TT.WIN[1] - TT.WIN[0]) / TT.FPS
    rec = SR.make_synthetic_record(int(T / 0.05) + 100, pose, S["dish"], fish_id=0)
    rng = np.random.default_rng([TT.SIM_SEED, zlib.crc32(key.encode()), i])
    with contextlib.redirect_stdout(io.StringIO()):
        out = m.sample(rec, rng, stim_provider=prov, stop_time_s=T)
    t = np.asarray(out["t_onset_s"], float)
    n = len(t)
    x = (np.asarray(out["x_px"], float)[:n] - cx) / PX
    y = (np.asarray(out["y_px"], float)[:n] - cy) / PX
    f = np.clip(TT.WIN[0] + np.round(t * TT.FPS).astype(int), 0, len(S["dxa"]) - 1)
    return key, show, i, dict(f=f.astype(np.int32), x=x.astype(np.float32), y=y.astype(np.float32),
                              seg=np.zeros(n, np.int32), dx=S["dxa"][f].astype(np.float32),
                              dy=S["dya"][f].astype(np.float32))


def run_sim(workers=10):
    keys = [k for pair in C2ST_UNITS.values() for k in pair]
    todo = [(k, s) for k in keys for s in (True, False)
            if not (OUT / f"screen_raw_sim_{k}_{'sighted' if s else 'blind'}.pkl").exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, s in todo:
            res = {i: r for _, _, i, r in ex.map(_sim, [(k, i, s) for i in range(N_SIM)], chunksize=4)}
            nm = f"screen_raw_sim_{k}_{'sighted' if s else 'blind'}.pkl"
            pickle.dump(res, open(OUT / nm, "wb"))
            print(f"wrote {nm} ({len(res)} fish)", flush=True)


# --------------------------------------------------------------------------- #
# Bouts -> windows
# --------------------------------------------------------------------------- #
def _kind():
    return TT.stimulus().stim_kind.to_numpy().astype(str)


def bout_table(r, kind_fr):
    """Per-bout channels from onset positions (onset-to-onset chords, as segment_bouts and the generator): turn
    (deg), displacement (px), interval (s), fish-to-dot distance (px), radial r/R; kept = bout-like episode, previous
    and next leg in the same segment, both intervals <= 10 s. Run id breaks wherever a bout is dropped."""
    f, x, y, seg = r["f"].astype(int), r["x"].astype(float), r["y"].astype(float), r["seg"]
    n = len(f)
    if n < 3:
        return None
    L = len(kind_fr)
    th = np.arctan2(np.diff(y), np.diff(x))                       # leg i: onset i -> i+1
    leg_ok = (seg[1:] == seg[:-1]) & (np.diff(f) > 0) & (np.diff(f) <= TT.MAX_DT * TT.FPS)
    idx = np.arange(1, n - 1)                                     # bout i needs legs i-1 and i
    ok = leg_ok[idx - 1] & leg_ok[idx]
    ok &= (kind_fr[np.clip(f[idx], 0, L - 1)] == KIND) & (kind_fr[np.clip(f[idx + 1], 0, L - 1)] == KIND)
    i = idx[ok]
    if len(i) < W:
        return None
    dth = np.degrees(TT.wrap(th[i] - th[i - 1]))
    dist = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) * PX
    dt = (f[i + 1] - f[i]) / TT.FPS
    dd = np.hypot(r["dx"][i] - x[i], r["dy"][i] - y[i]) * PX
    rad = np.hypot(x[i], y[i]) / TT.R_MM
    run = np.cumsum(np.r_[1, np.diff(i) != 1])
    return dict(dth=dth, dist=dist, dt=dt, dd=dd, rad=rad, run=run)


def fish_windows(r, kind_fr):
    from twin import generative_metrics as GM
    from twin import evaluate as E
    b = bout_table(r, kind_fr)
    if b is None:
        return None, None
    X = GM.window_features_traj(b["dth"], E.symbols_from_dtheta(b["dth"]), b["dist"], b["dt"], b["dd"], W=W,
                                radial=b["rad"], runs=b["run"])
    # per-window means for the individuality readout (same windows)
    starts = GM._window_starts(len(b["dth"]), W, W, b["run"])
    M = np.array([[np.mean(np.abs(b["dth"][a:a + W])), np.mean(b["dist"][a:a + W]) / PX, np.mean(b["dt"][a:a + W]),
                   np.mean(b["rad"][a:a + W])] for a in starts]) if starts else np.zeros((0, 4))
    return X, M


def _fw(args):
    fid, r = args
    if "k" not in _KIND_CACHE:
        _KIND_CACHE["k"] = _kind()
    X, M = fish_windows(r, _KIND_CACHE["k"])
    return fid, X, M


_KIND_CACHE = {}


def windows_of(res, workers=10):
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fid, X, M in ex.map(_fw, list(res.items()), chunksize=16):
            if X is not None and len(X):
                out[fid] = (X, M)
    return out


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #
def c2st(XA, gA, XB, gB, seed):
    """Real (A, label 1) vs other (B, label 0): equal window counts, 5-fold CV grouped by fish, RF 200 trees."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = min(len(XA), len(XB))
    ia = rng.choice(len(XA), n, replace=False); ib = rng.choice(len(XB), n, replace=False)
    X = np.vstack([XA[ia], XB[ib]]); y = np.r_[np.ones(n), np.zeros(n)]; g = np.r_[gA[ia], gB[ib]]
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), int(n)


def stack(wins, ids, offset=0):
    X = np.vstack([wins[i][0] for i in ids]); g = np.concatenate([np.full(len(wins[i][0]), int(i) + offset) for i in ids])
    return X, g


READINGS = {"all": None, "no radial": "norad", "dot only": "dot"}


def cols(X, reading):
    if reading == "norad":
        return X[:, :35]
    if reading == "dot":
        return X[:, 30:35]
    return X


def individuality(wr, ws, nb=300, seed=0):
    def spreads(w):
        means = np.array([w[i][1].mean(0) for i in w if len(w[i][1]) >= 2])
        sds = np.array([w[i][1].std(0, ddof=1) for i in w if len(w[i][1]) >= 2])
        return means, sds
    mr, sr = spreads(wr); ms, ss = spreads(ws)
    est = dict(between=(ms.std(0, ddof=1) / mr.std(0, ddof=1)).tolist(), within=(ss.mean(0) / sr.mean(0)).tolist())
    rng = np.random.default_rng(seed)
    bb, bw = [], []
    for _ in range(nb):
        a = rng.integers(0, len(mr), len(mr)); b = rng.integers(0, len(ms), len(ms))
        bb.append(ms[b].std(0, ddof=1) / mr[a].std(0, ddof=1)); bw.append(ss[b].mean(0) / sr[a].mean(0))
    ci = dict(between=np.percentile(bb, [2.5, 97.5], 0).tolist(), within=np.percentile(bw, [2.5, 97.5], 0).tolist())
    real = dict(between=mr.std(0, ddof=1).tolist(), within=sr.mean(0).tolist(), n_fish=len(mr))
    return dict(est=est, ci=ci, real=real, n_sim=len(ms))


def kinematics(w):
    m = np.array([w[i][1].mean(0) for i in w if len(w[i][1])])
    return np.median(m, 0).tolist()


def curves(res, ids, nb=300, seed=0):
    """Turn toward, distance gain and bout occupancy by distance, real dot minus rotated dot (twin_test sums),
    95 % bootstrap over fish."""
    ids = [i for i in ids if i in res]

    def pool(sel):
        S = {}
        for i in sel:
            for k, v in res[i][KIND].items():
                S[k] = S.get(k, 0.0) + np.asarray(v, float)
        with np.errstate(all="ignore"):
            return np.vstack([S["ntow"] / S["nturn"] - S["ntow_null"] / S["nturn_null"],
                              S["gain"] / S["ng"] - S["gain_null"] / S["ng_null"],
                              S["n"] / (S["n_null"] / 3.0)])
    est = pool(ids)
    rng = np.random.default_rng(seed)
    bs = np.array([pool([ids[j] for j in rng.integers(0, len(ids), len(ids))]) for _ in range(nb)])
    return dict(est=est.tolist(), lo=np.nanpercentile(bs, 2.5, 0).tolist(), hi=np.nanpercentile(bs, 97.5, 0).tolist())


def analyse(workers=10):
    from twin import repulsion_fit as RF
    R = pickle.load(open(OUT / "screen_raw_real.pkl", "rb"))
    print("windows: real ...", flush=True)
    wr = windows_of(R["res"], workers)
    real_ids = np.array(sorted(wr))
    print(f"  {len(wr)} real fish, {sum(len(v[0]) for v in wr.values())} windows", flush=True)
    halves = RF.clutch_split()
    hA = set(halves["fit"]); hB = set(halves["test"])
    idA = np.array([i for i in real_ids if R["clutch"].get(int(i)) in hA])
    idB = np.array([i for i in real_ids if R["clutch"].get(int(i)) in hB])
    out = dict(n_real_fish=int(len(wr)), n_real_windows=int(sum(len(v[0]) for v in wr.values())),
               reps=REPS, n_real_sample=N_REAL, auc={}, indiv={}, kin={}, curves={})
    out["kin"]["real"] = kinematics(wr)
    sims = {}
    for model, keys in C2ST_UNITS.items():
        for sd, k in zip(SEEDS, keys):
            for show in ("sighted", "blind"):
                res = pickle.load(open(OUT / f"screen_raw_sim_{k}_{show}.pkl", "rb"))
                ws = windows_of(res, workers)
                sims[(model, sd, show)] = ws
                print(f"  {model} s{sd} {show}: {len(ws)} fish, {sum(len(v[0]) for v in ws.values())} windows",
                      flush=True)
    # ---- C2ST
    for reading, rk in READINGS.items():
        rows = {}
        vals = []
        for rep in range(REPS):
            rng = np.random.default_rng([2026, rep])
            a = rng.choice(idA, N_REAL, replace=False); b = rng.choice(idB, N_REAL, replace=False)
            XA, gA = stack(wr, a); XB, gB = stack(wr, b)
            vals.append(c2st(cols(XA, rk), gA, cols(XB, rk), gB, seed=rep)[0])
        rows["real vs real"] = vals
        for (model, sd, show), ws in sims.items():
            vals = []
            sid = np.array(sorted(ws))
            XS, gS = stack(ws, sid, offset=10 ** 7)
            for rep in range(REPS):
                rng = np.random.default_rng([2027, rep])
                a = rng.choice(real_ids, N_REAL, replace=False)
                XR, gR = stack(wr, a)
                vals.append(c2st(cols(XR, rk), gR, cols(XS, rk), gS, seed=rep)[0])
            rows[f"{model} s{sd} {show}"] = vals
            print(f"  C2ST [{reading}] {model} s{sd} {show}: {np.mean(vals):.3f} +/- {np.std(vals, ddof=1):.3f}",
                  flush=True)
        print(f"  C2ST [{reading}] real vs real: {np.mean(rows['real vs real']):.3f}", flush=True)
        out["auc"][reading] = rows
    # ---- individuality + kinematics
    for (model, sd, show), ws in sims.items():
        if show == "sighted":
            out["indiv"][f"{model} s{sd}"] = individuality(wr, ws)
        out["kin"][f"{model} s{sd} {show}"] = kinematics(ws)
    # ---- response by distance (twin_test caches, incl. D4)
    T = pickle.load(open(OUT / "real.pkl", "rb"))
    ref = [f for f, g in zip(T["meta"]["fish_id"], T["meta"]["group"]) if g == "ref"]
    out["curves"]["real"] = curves(T["res"], ref)
    for model, keys in UNITS.items():
        for sd, k in zip(SEEDS, keys):
            s = pickle.load(open(OUT / f"sim_{k}_sighted.pkl", "rb"))["res"]
            out["curves"][f"{model} s{sd}"] = curves(s, list(s))
    out["dbins"] = TT.DBINS.tolist()
    (OUT / "screen_eval.json").write_text(json.dumps(out, indent=1))
    print("wrote screen_eval.json", flush=True)




PBINS = np.arange(-60, 301, 10)                  # s from the switch to bout-like motion


def _episode_clock(kind_fr):
    """Per frame: seconds since the start of the bout-like episode it lies in, or (negative) seconds to the start of
    the next bout-like episode for frames in the continuous episode just before; NaN elsewhere."""
    L = len(kind_fr)
    b = (kind_fr == KIND).astype(int)
    starts = np.flatnonzero(np.diff(b) == 1) + 1
    t = np.full(L, np.nan)
    for s in starts:
        e = s
        while e < L and kind_fr[e] == KIND:
            e += 1
        t[s:e] = (np.arange(s, e) - s) / TT.FPS
        p = s - 1
        while p >= 0 and kind_fr[p] != KIND and (s - p) <= 60 * TT.FPS:
            p -= 1
        t[p + 1:s] = (np.arange(p + 1, s) - s) / TT.FPS
    return t


def fish_profile(r, clock, rotate=None):
    f = r["f"].astype(int)
    x, y = r["x"].astype(float), r["y"].astype(float)
    dx, dy = r["dx"].astype(float), r["dy"].astype(float)
    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        dx, dy = c * dx - s * dy, s * dx + c * dy
    dt = np.r_[np.diff(f) / TT.FPS, np.nan]
    t = clock[np.clip(f, 0, len(clock) - 1)]
    ok = np.isfinite(t) & np.isfinite(dx) & (dt <= TT.MAX_DT)
    k = np.digitize(t[ok], PBINS) - 1
    d = np.hypot(dx[ok] - x[ok], dy[ok] - y[ok])
    m = (k >= 0) & (k < len(PBINS) - 1)
    s = np.bincount(k[m], weights=d[m], minlength=len(PBINS) - 1)
    n = np.bincount(k[m], minlength=len(PBINS) - 1)
    with np.errstate(all="ignore"):
        return s / n


def _boot_profile(P, nb=300, seed=0):
    P = np.asarray(P, float)
    est = np.nanmean(P, 0)
    rng = np.random.default_rng(seed)
    bs = np.array([np.nanmean(P[rng.integers(0, len(P), len(P))], 0) for _ in range(nb)])
    return dict(est=est.tolist(), lo=np.nanpercentile(bs, 2.5, 0).tolist(), hi=np.nanpercentile(bs, 97.5, 0).tolist(),
                n=int(len(P)))


def profiles(workers=10):
    """Distance to the dot over the bout-like episode (from 60 s before the switch to bout-like motion), per fish then
    over fish, 95 % bootstrap over fish: real (and real against the rotated dot), M6 and F sighted and blind."""
    clock = _episode_clock(_kind())
    out = dict(bins=PBINS.tolist(), prof={})
    R = pickle.load(open(OUT / "screen_raw_real.pkl", "rb"))["res"]
    out["attr"] = {}

    def both(res):
        d = np.array([fish_profile(r, clock) for r in res.values()])
        d0 = np.array([np.nanmean([fish_profile(r, clock, a) for a in TT.ROTS], 0) for r in res.values()])
        return d, d0
    d, d0 = both(R)
    out["prof"]["real"] = _boot_profile(d)
    out["prof"]["real, rotated dot"] = _boot_profile(d0)
    out["attr"]["real"] = _boot_profile(d0 - d)            # how much closer than chance (mm)
    for model, keys in C2ST_UNITS.items():
        for sd, k in zip(SEEDS, keys):
            for show in ("sighted", "blind"):
                res = pickle.load(open(OUT / f"screen_raw_sim_{k}_{show}.pkl", "rb"))
                d, d0 = both(res)
                out["prof"][f"{model} s{sd} {show}"] = _boot_profile(d)
                out["attr"][f"{model} s{sd} {show}"] = _boot_profile(d0 - d)
    (OUT / "screen_profiles.json").write_text(json.dumps(out, indent=1))
    print("wrote screen_profiles.json", {k: round(float(np.nanmean(v["est"])), 2) for k, v in out["prof"].items()},
          flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    {"real": run_real, "sim": run_sim, "analyse": analyse, "profiles": profiles}[cmd](w)
