"""The final test on the Larsch lab screen data (schizscreen_twin_v1): do the models, unchanged, behave like the screen's
fish in the screen's experiment?

UNPUBLISHED DATA (Larsch lab, see schizscreen_twin_v1/DATA_USE.md). This file holds code only. The package (metadata
extracted to <repo>/schizscreen_twin_v1/, trajectories read from the zip) and every cache (Model/twin_test/) are
git-ignored; never commit them.

THE EXPERIMENT. One fish per 70 mm dish, one ~2 mm dot on a fixed open-loop path, the same for every fish; 5-min
episodes alternate a bout-like dot (a ~2.1 mm jump every 20 frames, close to the training's moving dots) and a
continuous one (glides; never in training). Window used here: 60-130 min in the dish (frames 90000-215999, episodes
10-23: 7 bout-like + 7 continuous; the loom block starts at 130 min).

TWO SIDES, ONE MEASUREMENT.
  real   the sighted reference siblings (role 'reference sibling', sighted, in the screen's contrasts, no exclusion, no
         identity doubt) and the blind mutants; positions -> dish-centred mm (loader.to_mm) -> model px (x 3.60) ->
         segment_bouts.segment_fish, the training data's own bout segmentation (relative speed threshold, k = 1)
  model  each model unchanged, simulated in the same window: the dot path of a median well in mm -> model px around the
         training dish centre; dish radius 35 mm (the model's wall terms are relative to R, so its 2-10 mm rejection
         band becomes 1.4-7 mm); block clock held 'late' (60 s: the dot never appears, it is always there); fresh latent
         per simulated fish; fold i % 5. The 'blind model' is the same model with the dot hidden.
Bouts with an interval > 10 s are dropped on the real side (the models are trained on active-only bouts).

MEASURES (per fish, per dot kind; distance = fish-dot distance at the bout onset, mm):
  occupancy      bouts per distance bin, real dot vs the NULL dot (the dot rotated about the dish centre by 90 / 180 /
                 270 deg: same radius, same speed, no coupling to the fish)
  turn toward    P(the bout turns toward the dot | |turn| > 5 deg), bearing from the previous travel heading; vs null
  gain           mm closer to where the dot was at onset after the bout; vs null
  alignment      continuous dot: cos(travel heading - dot direction of motion), by the dot's bearing sector
  scalars        shoaling index (null - real mean distance) / null mean distance; mean distance; turn toward and gain at
                 5-20 mm (attraction) and 0-5 mm (repulsion), each real minus null; kinematics (median interval, median
                 step, mean distance from the centre, mean |turn|)

    python twin_test.py extract-meta          # metadata out of the zip into <repo>/schizscreen_twin_v1/
    python twin_test.py real                  # needs pyarrow: uv run --with pyarrow python twin_test.py real
    python twin_test.py sim KEY [KEY ...]     # report keys (26e, 30d, 11e, 32a, ...) -> sighted + blind runs
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

import numpy as np
import pandas as pd

from twin import paths

ZIP = paths.SCREEN_ZIP
PKG = paths.SCREEN
OUT = paths.SCREEN_OUT

PX = 3.60                                   # model px per mm (the physical scale of the training rig)
FPS = 30
WIN = (90000, 216000)                       # 60-130 min in the dish
R_MM = 35.0                                 # dish radius
MAX_DT = 10.0                               # active-only, as training
DBINS = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 70], float)
NB = len(DBINS) - 1
KINDS = ("dot_bout", "dot_continuous")
ROTS = np.radians([90.0, 180.0, 270.0])
SECTORS = ((0, 45), (45, 135), (135, 181))   # |bearing| of the dot from the fish: ahead, side, behind
N_SIM = 200                                 # simulated fish per model unit
SIM_SEED = 8101


SPLIT_SEED = 2026                           # the clutch split into a calibration and a test half
TRAIN_CX, TRAIN_CY = 180.0, 174.0           # the training dish centre (model px)


def load_loader():
    """The screen package's own reader (``loader.py`` in the package folder)."""
    if str(PKG) not in sys.path:
        sys.path.insert(0, str(PKG))
    import loader
    return loader


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# One measurement for both sides
# --------------------------------------------------------------------------- #
def measure(x, y, x_next, y_next, th_prev, turn_deg, dt, kind, dx, dy, dvx, dvy, kind_all=None, dt_all=None):
    """Per-fish sums. All positions dish-centred mm; th_prev = previous travel heading (rad); turn_deg CCW positive;
    kind = dot kind at onset; (dx, dy) = dot at onset (NaN = none); (dvx, dvy) = dot displacement over +-0.5 s."""
    out = {}
    turn = np.radians(np.asarray(turn_deg, float))
    big = np.abs(turn) > np.radians(5.0)
    okn = np.isfinite(x_next) & np.isfinite(y_next)
    r_c = np.hypot(x, y)
    st_mm = np.hypot(x_next - x, y_next - y)
    at_deg = np.abs(np.degrees(turn))
    kind_all = kind if kind_all is None else kind_all       # bouts before the > 10 s pause cut (pause diagnosis)
    dt_all = dt if dt_all is None else dt_all
    for kd in KINDS:
        sel = (kind == kd) & np.isfinite(dx) & np.isfinite(th_prev) & np.isfinite(turn)
        o = {}
        dots = [("", dx, dy)] + [("_null", np.cos(a) * dx - np.sin(a) * dy, np.sin(a) * dx + np.cos(a) * dy) for a in ROTS]
        vels = [dvx, dvy] if dvx is not None else [None, None]
        for j, (tag, DX, DY) in enumerate(dots):
            d0 = np.hypot(DX - x, DY - y)
            d1 = np.hypot(DX - x_next, DY - y_next)
            phi = wrap(np.arctan2(DY - y, DX - x) - th_prev)
            tow = (np.sign(turn) == np.sign(phi)) & big
            k = np.digitize(d0, DBINS) - 1
            ok = sel & (k >= 0) & (k < NB)
            def add(key, msk, w=None):
                v = np.bincount(k[msk], weights=None if w is None else w[msk], minlength=NB).astype(float)
                o[key + tag] = o.get(key + tag, 0.0) + v
            add("n", ok); add("nturn", ok & big); add("ntow", ok & tow)
            add("ng", ok & okn); add("gain", ok & okn, d0 - d1)
            # DIAGNOSIS of the backing off: turn size, large turns, away turns, step and interval (component sums)
            away = big & ~tow
            add("sabs", ok, at_deg); add("n90", ok & (at_deg > 90.0))
            add("naway", ok & away); add("sabs_away", ok & away, at_deg)
            add("slst", ok & okn, np.log(np.maximum(st_mm, 1e-3)))
            okdt = np.isfinite(dt)
            add("ndt", ok & okdt); add("sldt", ok & okdt, np.log(np.maximum(dt, 1e-3)))
            ok_all = (kind_all == kd) & np.isfinite(dx) & np.isfinite(th_prev) & np.isfinite(turn) & (k >= 0) & (k < NB)
            add("nall", ok_all & np.isfinite(dt_all)); add("nlong", ok_all & (dt_all > MAX_DT))
            o["sum_d" + tag] = o.get("sum_d" + tag, 0.0) + float(np.nansum(d0[sel]))
            o["cnt_d" + tag] = o.get("cnt_d" + tag, 0.0) + float(np.sum(sel & np.isfinite(d0)))
            if kd == "dot_continuous" and vels[0] is not None:
                a = 0.0 if j == 0 else ROTS[j - 1]
                VX = np.cos(a) * dvx - np.sin(a) * dvy; VY = np.sin(a) * dvx + np.cos(a) * dvy
                mv = np.hypot(VX, VY) > 1.0                        # the dot moved > 1 mm in 1 s
                th_now = np.arctan2(y_next - y, x_next - x)       # this bout's travel heading
                c = np.cos(th_now - np.arctan2(VY, VX))
                for s, (lo, hi) in enumerate(SECTORS):
                    sec = np.abs(np.degrees(phi)) >= lo
                    sec &= np.abs(np.degrees(phi)) < hi
                    m2 = ok & okn & mv & sec
                    add(f"al_n{s}", m2); add(f"al_c{s}", m2, c)
        out[kd] = o
    m = np.isfinite(dt) & np.isfinite(turn)
    st = np.hypot(x_next - x, y_next - y)
    out["kin"] = dict(n=float(m.sum()), med_dt=float(np.nanmedian(dt[m])) if m.any() else np.nan,
                      med_step=float(np.nanmedian(st[m & okn])) if (m & okn).any() else np.nan,
                      mean_r=float(np.nanmean(r_c[m])) if m.any() else np.nan,
                      mean_absturn=float(np.nanmean(np.abs(turn_deg[m]))) if m.any() else np.nan)
    return out


# --------------------------------------------------------------------------- #
# Package access
# --------------------------------------------------------------------------- #
def extract_meta():
    z = zipfile.ZipFile(ZIP)
    names = [n for n in z.namelist() if not n.startswith("schizscreen_twin_v1/trajectories/")
             and not n.startswith("schizscreen_twin_v1/pilot/")]
    for n in names:
        z.extract(n, PKG.parent)
    print(f"extracted {len(names)} files to {PKG}")


def fish_table():
    f = pd.read_csv(PKG / "fish.csv")
    base = (f.exclusion_class == "none") & (f.identity_uncertain != True)          # noqa: E712
    ref = base & (f.role == "reference sibling") & (f.sighted == True) & (f.in_screen_contrasts == True)  # noqa: E712
    blind = base & (f.sighted == False)                                             # noqa: E712
    f = f[ref | blind].copy()
    f["group"] = np.where(blind[f.index], "blind", "ref")
    return f


def stimulus():
    st = pd.read_parquet(PKG / "stimulus.parquet")
    return st.set_index("frame").sort_index()


def dot_mm(st, cx, cy, ppm):
    """The dot path in a well's dish-centred mm (y up), NaN while not shown."""
    loader = load_loader()
    dx, dy, _ = loader.to_mm(st.stim_x.to_numpy(float), st.stim_y.to_numpy(float), None, cx, cy, ppm)
    shown = st.dot_shown.to_numpy()
    return np.where(shown, dx, np.nan), np.where(shown, dy, np.nan)


def dot_velocity(dx, dy, f):
    """Dot displacement over +-15 frames around frame indices f (into the full-length arrays)."""
    a = np.clip(f - 15, 0, len(dx) - 1); b = np.clip(f + 15, 0, len(dx) - 1)
    return dx[b] - dx[a], dy[b] - dy[a]


# --------------------------------------------------------------------------- #
# Real side
# --------------------------------------------------------------------------- #
def _real_plate(args):
    plate, rows = args
    import pyarrow.parquet as pq
    loader = load_loader()
    from twin import segment_bouts as SB
    z = zipfile.ZipFile(ZIP)
    cols = ["frame"] + [f"{k}_{int(r['well']):02d}" for r in rows for k in ("x", "y", "heading", "flag")]
    tab = pq.read_table(io.BytesIO(z.read(f"schizscreen_twin_v1/trajectories/plate_{plate:03d}.parquet")), columns=cols,
                        filters=[("frame", ">=", WIN[0]), ("frame", "<", WIN[1])]).to_pandas()
    st = stimulus()
    frames = tab.frame.to_numpy()
    kind_all = st.stim_kind.to_numpy().astype(str)
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
            recs = SB.segment_fish(df, "f", int(r["fish_id"]), 1.0)
        b = pd.DataFrame(recs)
        if b.empty:
            continue
        res[int(r["fish_id"])] = _bouts_to_measure(b, st, r["centre_x_px"], r["centre_y_px"], r["px_per_mm"], kind_all)
    return res


def _bouts_to_measure(b, st, cx, cy, ppm, kind_all):
    seg = b["segment_id"].to_numpy() if "segment_id" in b else np.zeros(len(b), int)
    x = b.x_px.to_numpy(float) / PX; y = b.y_px.to_numpy(float) / PX
    f = b.frame_onset.to_numpy(int)
    same_prev = np.r_[False, seg[1:] == seg[:-1]]
    same_next = np.r_[seg[1:] == seg[:-1], False]
    th = b.heading_rad.to_numpy(float)
    th_prev = np.where(same_prev, np.r_[np.nan, th[:-1]], np.nan)
    x_next = np.where(same_next, np.r_[x[1:], np.nan], np.nan)
    y_next = np.where(same_next, np.r_[y[1:], np.nan], np.nan)
    dt = b.dt_s.to_numpy(float)
    keep = dt <= MAX_DT
    dxa, dya = dot_mm(st, cx, cy, ppm)
    dx, dy = dxa[f], dya[f]
    dvx, dvy = dot_velocity(dxa, dya, f)
    k = np.where(keep, kind_all[f], "drop")
    return measure(x, y, x_next, y_next, th_prev, b.dtheta_deg.to_numpy(float), np.where(keep, dt, np.nan), k,
                   dx, dy, dvx, dvy, kind_all=kind_all[f], dt_all=dt)


def run_real(workers=10):
    OUT.mkdir(parents=True, exist_ok=True)
    f = fish_table()
    jobs = [(int(p), g[["fish_id", "well", "centre_x_px", "centre_y_px", "px_per_mm"]].to_dict("records"))
            for p, g in f.groupby("plate")]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_real_plate, jobs)):
            res.update(r)
            if i % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} plates, {len(res)} fish", flush=True)
    meta = {c: f[c].astype(object).tolist() for c in ("fish_id", "group", "clutch", "plate", "age_dpf", "line_code")}
    pickle.dump(dict(res=res, meta=meta), open(OUT / "real.pkl", "wb"))
    print(f"wrote {OUT / 'real.pkl'}: {len(res)} fish")


# --------------------------------------------------------------------------- #
# Model side
# --------------------------------------------------------------------------- #
class PathProvider:
    """The screen's dot, open loop: position at the bout onset from the recorded path (model px); 'late' block clock."""

    def __init__(self, px_x, px_y, show=True):
        self.px_x, self.px_y, self.show = px_x, px_y, show

    def __call__(self, t, time_s, x, y, h):
        f = WIN[0] + int(round(float(time_s) * FPS))
        if not self.show or f >= WIN[1] or not np.isfinite(self.px_x[f]):
            return (np.nan, np.nan, 0.0, 0.0)
        return (float(self.px_x[f]), float(self.px_y[f]), 1.0, 1.0)

    def block_time_s(self, t, t_now):
        return 60.0

    def dots(self, t, time_s, x, y, h):
        """The individual dots (one), for a per-dot term (the repulsion term): (xs, ys, on, moving_k, n)."""
        cx_, cy_, mv, nd = self(t, time_s, x, y, h)
        return np.array([cx_]), np.array([cy_]), np.array([np.isfinite(cx_)]), np.array([mv]), nd


_SIM_CACHE = {}


def _sim_setup():
    if "st" not in _SIM_CACHE:
        from twin import model_figures as MF
        st = stimulus()
        fish = pd.read_csv(PKG / "fish.csv")
        ref = fish[fish.exclusion_class == "none"]
        cx, cy, ppm = (float(ref.centre_x_px.median()), float(ref.centre_y_px.median()), float(ref.px_per_mm.median()))
        dxa, dya = dot_mm(st, cx, cy, ppm)
        _, d, refp = MF._programme()
        dish = (float(d[0]), float(d[1]), R_MM * PX)
        _SIM_CACHE.update(st=st, dxa=dxa, dya=dya, kind=st.stim_kind.to_numpy().astype(str), dish=dish, ref=refp, MF=MF)
    return _SIM_CACHE


def _models(key):
    if key not in _SIM_CACHE:
        from twin import registry
        S = _sim_setup()
        with contextlib.redirect_stdout(io.StringIO()):
            ms, _ = S["MF"].fold_models(registry.ckpt_dir(key), registry.arm(key))
        _SIM_CACHE[key] = [ms[k] for k in sorted(ms)]
    return _SIM_CACHE[key]


def _sim_one(args):
    key, i, show = args
    from twin import stim_rollout as SR
    S = _sim_setup()
    m = _models(key)[i % 5]
    cx, cy, R = S["dish"]
    pose = S["ref"].sample_start_pose(np.random.default_rng([SIM_SEED, 1, i]), S["dish"])
    prov = PathProvider(cx + S["dxa"] * PX, cy + S["dya"] * PX, show=show)
    T = (WIN[1] - WIN[0]) / FPS
    rec = SR.make_synthetic_record(int(T / 0.05) + 100, pose, S["dish"], fish_id=0)
    rng = np.random.default_rng([SIM_SEED, zlib.crc32(key.encode()), i])
    with contextlib.redirect_stdout(io.StringIO()):
        out = m.sample(rec, rng, stim_provider=prov, stop_time_s=T)
    x = (np.asarray(out["x_px"], float) - cx) / PX; y = (np.asarray(out["y_px"], float) - cy) / PX
    t = np.asarray(out["t_onset_s"], float)
    n = len(t)
    x, y = x[:n], y[:n]
    # bout i goes from onset i to onset i+1: its travel heading is that chord (as segment_bouts on the real side),
    # its turn the change from the previous bout's heading, CCW positive
    th = np.r_[np.arctan2(np.diff(y), np.diff(x)), np.nan]
    th_prev = np.r_[np.nan, th[:-1]]
    turn = np.degrees(wrap(th - th_prev))
    x_next = np.r_[x[1:], np.nan]; y_next = np.r_[y[1:], np.nan]
    f = np.clip(WIN[0] + np.round(t * FPS).astype(int), 0, len(S["dxa"]) - 1)
    dx, dy = S["dxa"][f], S["dya"][f]                           # blind model: where the dot would have been
    dvx, dvy = dot_velocity(S["dxa"], S["dya"], f)
    dt = np.asarray(out["dt_s"], float)[:n]
    return key, show, i, measure(x, y, x_next, y_next, th_prev, turn, dt, S["kind"][f], dx, dy, dvx, dvy)


def run_sim(keys, workers=10, sighted_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(k, i, show) for k in keys for show in ((True,) if sighted_only else (True, False)) for i in range(N_SIM)]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for j, (k, show, i, r) in enumerate(ex.map(_sim_one, jobs, chunksize=4)):
            res.setdefault((k, show), {})[i] = r
            if j % 200 == 0:
                print(f"  {j + 1}/{len(jobs)} simulated fish", flush=True)
    for (k, show), r in res.items():
        pickle.dump(dict(res=r, key=k, sighted=show), open(OUT / f"sim_{k}_{'sighted' if show else 'blind'}.pkl", "wb"))
    print("wrote", sorted(f"sim_{k}_{'sighted' if s else 'blind'}.pkl" for k, s in res))




def clutch_split():
    p = OUT / "clutch_split.json"
    if p.exists():
        return json.loads(p.read_text())
    real = pickle.load(open(OUT / "real.pkl", "rb"))
    m = real["meta"]
    cl = sorted({c for c, g in zip(m["clutch"], m["group"]) if g == "ref"})
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(cl))
    half = len(cl) // 2
    out = {"fit": sorted(cl[i] for i in perm[:half]), "test": sorted(cl[i] for i in perm[half:])}
    p.write_text(json.dumps(out, indent=1))
    return out


def pool(res, ids, kind="dot_bout"):
    S = {}
    for i in ids:
        for k, v in res[i][kind].items():
            S[k] = S.get(k, 0.0) + np.asarray(v, float)
    return S


def curves(S):
    with np.errstate(all="ignore"):
        return (S["ntow"] / S["nturn"] - S["ntow_null"] / S["nturn_null"],
                S["gain"] / S["ng"] - S["gain_null"] / S["ng_null"])


def _segment_plate(args):
    """Real side as _real_plate, but per-fish bout tables with the dot (model px around the training dish centre)."""
    plate, rows = args
    import pyarrow.parquet as pq
    loader = load_loader()
    from twin import segment_bouts as SB
    z = zipfile.ZipFile(ZIP)
    cols = ["frame"] + [f"{k}_{int(r['well']):02d}" for r in rows for k in ("x", "y", "heading", "flag")]
    tab = pq.read_table(io.BytesIO(z.read(f"schizscreen_twin_v1/trajectories/plate_{plate:03d}.parquet")), columns=cols,
                        filters=[("frame", ">=", WIN[0]), ("frame", "<", WIN[1])]).to_pandas()
    st = stimulus()
    frames = tab.frame.to_numpy()
    epi = st.episode.to_numpy().astype(str)
    kind_all = st.stim_kind.to_numpy().astype(str)
    out = {}
    for r in rows:
        w = int(r["well"])
        x, y, o = loader.to_mm(tab[f"x_{w:02d}"].to_numpy(float), tab[f"y_{w:02d}"].to_numpy(float),
                               tab[f"heading_{w:02d}"].to_numpy(float) / loader.HEADING_SCALE,
                               r["centre_x_px"], r["centre_y_px"], r["px_per_mm"])
        fl = tab[f"flag_{w:02d}"].fillna(1).to_numpy().astype(np.uint8)
        bad = ((fl & loader.DEFAULT_MASK) > 0) | ~np.isfinite(x)
        df = pd.DataFrame({"frame": frames, "stimulus": epi[frames],
                           "f_x": np.where(bad, np.nan, TRAIN_CX + x * PX), "f_y": np.where(bad, np.nan, TRAIN_CY + y * PX),
                           "f_ori": np.where(bad, np.nan, o)})
        with contextlib.redirect_stdout(io.StringIO()):
            b = pd.DataFrame(SB.segment_fish(df, "f", int(r["fish_id"]), 1.0))
        if b.empty:
            continue
        dxa, dya = dot_mm(st, r["centre_x_px"], r["centre_y_px"], r["px_per_mm"])
        f = b.frame_onset.to_numpy(int)
        b["dot_cx"] = TRAIN_CX + dxa[f] * PX
        b["dot_cy"] = TRAIN_CY + dya[f] * PX
        b["kind"] = kind_all[f]
        out[int(r["fish_id"])] = b
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract-meta":
        extract_meta()
    elif cmd == "real":                       # [workers]
        run_real(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif cmd == "sim":                        # KEY [KEY ...] [--workers N]; each worker loads torch (~2 GB committed)
        args = sys.argv[2:]
        w = int(args[args.index("--workers") + 1]) if "--workers" in args else 5
        so = "--sighted-only" in args
        args = [a for a in args if a != "--sighted-only"]
        run_sim([k for j, k in enumerate(args) if k != "--workers" and (j == 0 or args[j - 1] != "--workers")], w, so)
    else:
        raise SystemExit(__doc__)
