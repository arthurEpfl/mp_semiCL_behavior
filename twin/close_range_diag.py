"""WHAT MAKES THE CLOSE-RANGE HOLE? (2026-10-18, step 1 of closing the shoal's close-range gap; no training)

Real screen fish hold off the last millimetres of the dot (bouts at 0-2 / 2-4 mm ~0.84 / 0.92 of the 4-8 mm level, dot
vs rotated dot); F + the calibrated group-repulsion turn term matches the real turn-away yet not that hole (0.94-0.95).
This splits the hole into its parts, measured identically on both sides from the bout onsets (bout-like dot episodes,
60-130 min; real = the reference siblings of the HELD-OUT half of the clutches, model = the held-out test's 200 simulated
fish per variant, same streams as grouprep_fit.test):

  per onset distance to the dot (DBINS), real dot and rotated (null) dot:
    n          bouts                                  -> bout occupancy (the calibration's shape)
    log dt     interval to the next bout              -> does a near dot shorten / lengthen the wait?
    log step   bout displacement
    apart      the bout lands farther from where the dot was at its onset (the fish's own move)
    apart_d    farther from the dot at the NEXT onset (own move + dot motion)
    in4        the next onset is within 4 mm
  time series (0.5 s grid, the fish held at its landing point between onsets, as the simulator; the dot moves):
    time occupancy per bin; per close threshold (2, 4 mm): entries per hour and mean stay (s), real and null.

    uv run --with pyarrow python close_range_diag.py real      # -> Model/twin_test/crd_real.pkl
    python close_range_diag.py sim                             # -> Model/twin_test/crd_sim_<key>_<var>.pkl
    python close_range_diag.py report                          # -> Model/grouprep/close_range_diag.json + tables
"""
from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from twin import twin_test as TT  # noqa: E402

OUT = TT.OUT
GOUT = HERE.parent / "Model" / "grouprep"
DBINS = TT.DBINS
NB = len(DBINS) - 1
KIND = "dot_bout"
GRID = 15                                     # time grid, frames (0.5 s)
CLOSE = (2.0, 4.0)
KEYS = ("14f", "32b")
VARS = ("F", "grT", "grT2", "grT4", "grT8")
N_SIM, WORKERS = 200, 12


# --------------------------------------------------------------------------- #
# One measurement for both sides
# --------------------------------------------------------------------------- #
def diag(x, y, f, seg, dxa, dya, kind_fr):
    """x, y: onset positions (dish-centred mm); f: onset frames (absolute); seg: segment ids; dxa/dya: the dot per
    absolute frame (mm, NaN = hidden); kind_fr: the stimulus kind per absolute frame. Per-fish sums."""
    x, y, f = np.asarray(x, float), np.asarray(y, float), np.asarray(f, int)
    nxt = np.r_[seg[1:] == seg[:-1], False]
    f1 = np.r_[f[1:], -1]
    dtf = f1 - f
    ok = nxt & (dtf > 0) & (dtf <= TT.MAX_DT * TT.FPS)
    ok &= kind_fr[np.clip(f, 0, len(kind_fr) - 1)] == KIND
    ok &= kind_fr[np.clip(f1, 0, len(kind_fr) - 1)] == KIND
    x1, y1 = np.r_[x[1:], np.nan], np.r_[y[1:], np.nan]
    fc, f1c = np.clip(f, 0, len(dxa) - 1), np.clip(f1, 0, len(dxa) - 1)
    out = {}
    for j, a in enumerate((0.0,) + tuple(TT.ROTS)):
        tag = "" if j == 0 else "_null"
        ca, sa = np.cos(a), np.sin(a)
        DX, DY = ca * dxa - sa * dya, sa * dxa + ca * dya
        d0 = np.hypot(DX[fc] - x, DY[fc] - y)
        d_own = np.hypot(DX[fc] - x1, DY[fc] - y1)
        d1 = np.hypot(DX[f1c] - x1, DY[f1c] - y1)
        good = ok & np.isfinite(d0) & np.isfinite(d1)
        k = np.digitize(d0, DBINS) - 1
        good &= (k >= 0) & (k < NB)
        st = np.hypot(x1 - x, y1 - y)
        dt = dtf / TT.FPS

        def add(key, m, w=None):
            v = np.bincount(k[m], weights=None if w is None else w[m], minlength=NB).astype(float)
            out[key + tag] = out.get(key + tag, 0.0) + v
        add("n", good)
        add("ldt", good, np.log(np.maximum(dt, 1e-3)))
        add("lst", good, np.log(np.maximum(st, 1e-3)))
        add("apart", good & (d_own > d0))
        add("apart_d", good & (d1 > d0))
        add("in4", good & (d1 < 4.0))
        # time series: the fish sits at its landing point (x1, y1) over (f, f1], sampled on the global 0.5 s grid
        idx = np.flatnonzero(good)
        if len(idx):
            s0 = (f[idx] // GRID + 1) * GRID
            cnt = np.maximum((f1[idx] - s0) // GRID + 1, 0)
            rep = np.repeat(np.arange(len(idx)), cnt)
            off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            fr = s0[rep] + off * GRID
            fr = np.clip(fr, 0, len(dxa) - 1)
            ds = np.hypot(DX[fr] - x1[idx][rep], DY[fr] - y1[idx][rep])
            vk = np.isfinite(ds) & (kind_fr[fr] == KIND)
            kk = np.digitize(ds, DBINS) - 1
            m = vk & (kk >= 0) & (kk < NB)
            out["tocc" + tag] = out.get("tocc" + tag, 0.0) + np.bincount(kk[m], minlength=NB).astype(float)
            # contiguous runs: consecutive grid samples (gap > one grid step = a break)
            fr_v, ds_v = fr[vk], ds[vk]
            brk = np.r_[True, np.diff(fr_v) != GRID]
            out["t_samp" + tag] = out.get("t_samp" + tag, 0.0) + float(len(fr_v))
            for thr in CLOSE:
                inn = ds_v < thr
                start = inn & (brk | np.r_[True, ~inn[:-1]])
                out[f"entries{thr:g}" + tag] = out.get(f"entries{thr:g}" + tag, 0.0) + float(start.sum())
                out[f"inside{thr:g}" + tag] = out.get(f"inside{thr:g}" + tag, 0.0) + float(inn.sum())
    return out


# --------------------------------------------------------------------------- #
# Check 2: chasing the position vs moving with the dot; and the bout timing relative to the dot's jumps
# --------------------------------------------------------------------------- #
PH_BINS = np.array([0, 5, 10, 15, 20])          # frames since the dot's last jump (it jumps every 20 frames)
NPH = len(PH_BINS) - 1
PAST = 30                                       # the dot's recent motion: its displacement over the last 1 s
REG = ("toward", "along", "heading")            # the bout's displacement = a u_toward + b u_along + c u_heading


def _last_jump(dxa, dya):
    mv = np.r_[False, np.hypot(np.diff(dxa), np.diff(dya)) > 0.5]
    return np.maximum.accumulate(np.where(mv, np.arange(len(dxa)), -10 ** 9))


def diag_p(x, y, f, seg, dxa, dya, kind_fr):
    """Per distance bin x jump-phase bin: the normal equations of the bout displacement (onset -> next onset, mm) on
    three unit vectors (toward the dot at onset, along the dot's motion over the last 1 s, along the fish's previous
    travel heading), the alignment cos(displacement, dot motion), and 'apart_d' (farther from the dot at the next
    onset); real dot and rotated (null) dots. Per-fish sums, arrays (NB * NPH)."""
    x, y, f = np.asarray(x, float), np.asarray(y, float), np.asarray(f, int)
    L = len(dxa)
    prv = np.r_[False, seg[1:] == seg[:-1]]
    nxt = np.r_[seg[1:] == seg[:-1], False]
    f0, f1 = np.r_[-1, f[:-1]], np.r_[f[1:], -1]
    mx = TT.MAX_DT * TT.FPS
    ok = prv & nxt & (f1 - f > 0) & (f1 - f <= mx) & (f - f0 > 0) & (f - f0 <= mx) & (f - PAST >= 0) & (f1 < L)
    fc, f1c, fpc = np.clip(f, 0, L - 1), np.clip(f1, 0, L - 1), np.clip(f - PAST, 0, L - 1)
    ok &= (kind_fr[fc] == KIND) & (kind_fr[f1c] == KIND) & (kind_fr[fpc] == KIND)
    x0, y0 = np.r_[np.nan, x[:-1]], np.r_[np.nan, y[:-1]]
    x1, y1 = np.r_[x[1:], np.nan], np.r_[y[1:], np.nan]
    mvx, mvy = x1 - x, y1 - y
    hx, hy = x - x0, y - y0
    hn = np.hypot(hx, hy)
    ok &= hn > 0.5
    with np.errstate(all="ignore"):
        uh = np.stack([hx / hn, hy / hn], -1)
    ph = fc - _last_jump(dxa, dya)[fc]
    ok &= (ph >= 0) & (ph < PH_BINS[-1])
    kph = np.clip(np.digitize(ph, PH_BINS) - 1, 0, NPH - 1)
    mvec = np.stack([mvx, mvy], -1)
    mn = np.hypot(mvx, mvy)
    out = {}
    for j, a in enumerate((0.0,) + tuple(TT.ROTS)):
        tag = "" if j == 0 else "_null"
        ca, sa = np.cos(a), np.sin(a)
        DX, DY = ca * dxa - sa * dya, sa * dxa + ca * dya
        tx, ty = DX[fc] - x, DY[fc] - y
        d0 = np.hypot(tx, ty)
        vx, vy = DX[fc] - DX[fpc], DY[fc] - DY[fpc]
        vn = np.hypot(vx, vy)
        d1 = np.hypot(DX[f1c] - x1, DY[f1c] - y1)
        k = np.digitize(d0, DBINS) - 1
        good = ok & np.isfinite(d0) & np.isfinite(d1) & np.isfinite(mn) & (vn > 1.0) & (d0 > 0) & (k >= 0) & (k < NB)
        idx = np.flatnonzero(good)
        cell = k[idx] * NPH + kph[idx]
        with np.errstate(all="ignore"):
            U = np.stack([np.stack([tx / d0, ty / d0], -1), np.stack([vx / vn, vy / vn], -1), uh], 1)[idx]  # (n,3,2)
        M = mvec[idx]

        def add(key, w):
            v = np.bincount(cell, weights=w, minlength=NB * NPH).astype(float)
            out[key + tag] = out.get(key + tag, 0.0) + v
        add("n", np.ones(len(idx)))
        for a_ in range(3):
            add(f"r{a_}", np.einsum("nd,nd->n", M, U[:, a_]))
            for b_ in range(a_, 3):
                add(f"G{a_}{b_}", np.einsum("nd,nd->n", U[:, a_], U[:, b_]))
        big = mn[idx] > 0.5
        with np.errstate(all="ignore"):
            cs = np.einsum("nd,nd->n", M, U[:, 1]) / mn[idx]
        add("nc", big.astype(float))
        add("cos", np.where(big, cs, 0.0))
        add("apd", (d1[idx] > d0[idx]).astype(float))
    return out


def readouts_p(S, phase=None):
    """Per distance bin (pooled over the jump phase, or one phase bin): the three coefficients (mm per bout), the
    alignment and apart_d, real dot and null."""
    def cut(v):
        v = np.asarray(v, float).reshape(NB, NPH)
        return v.sum(1) if phase is None else v[:, phase]
    r = {}
    for tag in ("", "_null"):
        G = np.zeros((NB, 3, 3))
        rr = np.zeros((NB, 3))
        for a_ in range(3):
            rr[:, a_] = cut(S[f"r{a_}{tag}"])
            for b_ in range(a_, 3):
                G[:, a_, b_] = G[:, b_, a_] = cut(S[f"G{a_}{b_}{tag}"])
        beta = np.full((NB, 3), np.nan)
        nn = cut(S["n" + tag])
        for i in range(NB):
            if nn[i] > 30:
                beta[i] = np.linalg.solve(G[i], rr[i])
        for a_, nm in enumerate(REG):
            r[nm + tag] = beta[:, a_]
        with np.errstate(all="ignore"):
            r["align" + tag] = cut(S["cos" + tag]) / cut(S["nc" + tag])
            r["apart_d" + tag] = cut(S["apd" + tag]) / nn
        r["n" + tag] = nn
    for nm in REG + ("align", "apart_d"):
        r[nm + "_x"] = r[nm] - r[nm + "_null"]                 # the dot's effect: real dot minus rotated dot
    return r




# --------------------------------------------------------------------------- #
# Real side (the reference siblings; segmentation exactly as twin_test)
# --------------------------------------------------------------------------- #
def _real_plate(args):
    plate, rows, fn = args
    import pyarrow.parquet as pq
    import loader
    from twin import segment_bouts as SB
    z = zipfile.ZipFile(TT.ZIP)
    cols = ["frame"] + [f"{k}_{int(r['well']):02d}" for r in rows for k in ("x", "y", "heading", "flag")]
    tab = pq.read_table(io.BytesIO(z.read(f"schizscreen_twin_v1/trajectories/plate_{plate:03d}.parquet")), columns=cols,
                        filters=[("frame", ">=", TT.WIN[0]), ("frame", "<", TT.WIN[1])]).to_pandas()
    st = TT.stimulus()
    frames = tab.frame.to_numpy()
    kind_fr = st.stim_kind.to_numpy().astype(str)
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
                           "f_x": np.where(bad, np.nan, x * TT.PX), "f_y": np.where(bad, np.nan, y * TT.PX),
                           "f_ori": np.where(bad, np.nan, o)})
        with contextlib.redirect_stdout(io.StringIO()):
            b = pd.DataFrame(SB.segment_fish(df, "f", int(r["fish_id"]), 1.0))
        if b.empty:
            continue
        seg = b["segment_id"].to_numpy() if "segment_id" in b else np.zeros(len(b), int)
        dxa, dya = TT.dot_mm(st, r["centre_x_px"], r["centre_y_px"], r["px_per_mm"])
        res[int(r["fish_id"])] = FNS[fn](b.x_px.to_numpy(float) / TT.PX, b.y_px.to_numpy(float) / TT.PX,
                                      b.frame_onset.to_numpy(int), seg, dxa, dya, kind_fr)
    return res


def run_real(workers=10, fn="diag"):
    f = TT.fish_table()
    f = f[f.group == "ref"]
    jobs = [(int(p), g[["fish_id", "well", "centre_x_px", "centre_y_px", "px_per_mm"]].to_dict("records"), fn)
            for p, g in f.groupby("plate")]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_real_plate, jobs)):
            res.update(r)
            if i % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} plates, {len(res)} fish", flush=True)
    clutch = dict(zip(f.fish_id.astype(int), f.clutch))
    pickle.dump(dict(res=res, clutch=clutch), open(OUT / f"{PFX[fn]}_real.pkl", "wb"))
    print(f"wrote {OUT / (PFX[fn] + '_real.pkl')}: {len(res)} fish", flush=True)


# --------------------------------------------------------------------------- #
# Model side (the held-out test's simulations, same streams)
# --------------------------------------------------------------------------- #
def _gp(key, var):
    from twin import grouprep_fit as GF
    return None if var == "F" else GF.variants(key)[var][0]


def _sim(args):
    key, var, i, fn = args
    from twin import stim_rollout as SR
    S = TT._sim_setup()
    m = TT._models(key)[i % 5]
    base = m.__dict__.setdefault("_cfg_base", dict(m._cfg_feat))
    cfg = dict(base)
    cfg.pop("repulsion", None)
    gp = _gp(key, var)
    if gp:
        cfg["grouprep"] = dict(gp)
    m._cfg_feat = cfg
    try:
        cx, cy, R = S["dish"]
        pose = S["ref"].sample_start_pose(np.random.default_rng([TT.SIM_SEED, 1, i]), S["dish"])
        prov = TT.PathProvider(cx + S["dxa"] * TT.PX, cy + S["dya"] * TT.PX, show=True)
        T = (TT.WIN[1] - TT.WIN[0]) / TT.FPS
        rec = SR.make_synthetic_record(int(T / 0.05) + 100, pose, S["dish"], fish_id=0)
        import zlib
        rng = np.random.default_rng([TT.SIM_SEED, zlib.crc32(key.encode()), i])
        with contextlib.redirect_stdout(io.StringIO()):
            out = m.sample(rec, rng, stim_provider=prov, stop_time_s=T)
    finally:
        m._cfg_feat = dict(base)
    t = np.asarray(out["t_onset_s"], float)
    n = len(t)
    x = (np.asarray(out["x_px"], float)[:n] - cx) / TT.PX
    y = (np.asarray(out["y_px"], float)[:n] - cy) / TT.PX
    f = TT.WIN[0] + np.round(t * TT.FPS).astype(int)
    return key, var, i, FNS[fn](x, y, f, np.zeros(n, int), S["dxa"], S["dya"], S["kind"])


def run_sim(workers=WORKERS, fn="diag", variants=VARS):
    todo = [(k, v) for k in KEYS for v in variants if not (OUT / f"{PFX[fn]}_sim_{k}_{v}.pkl").exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, v in todo:
            res = {i: r for _, _, i, r in ex.map(_sim, [(k, v, i, fn) for i in range(N_SIM)], chunksize=4)}
            pickle.dump(res, open(OUT / f"{PFX[fn]}_sim_{k}_{v}.pkl", "wb"))
            print(f"wrote {PFX[fn]}_sim_{k}_{v}.pkl ({len(res)} fish)", flush=True)


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #
def pool(res, ids):
    S = {}
    for i in ids:
        for k, v in res[i].items():
            S[k] = S.get(k, 0.0) + np.asarray(v, float)
    return S


def readouts(S):
    """Per-bin curves (dot vs rotated dot) and the stay/entry decomposition."""
    with np.errstate(all="ignore"):
        r = {}
        occ = S["n"] / (S["n_null"] / 3.0)
        tocc = (S["tocc"] / S["tocc"].sum()) / (S["tocc_null"] / S["tocc_null"].sum())
        for nm, v in (("occ", occ), ("tocc", tocc)):
            ref = np.nanmean(v[2:4])
            r[nm] = v
            r[nm + "_shape"] = np.array([v[0] / ref, v[1] / ref])
        r["dt"] = np.exp(S["ldt"] / S["n"])                       # geometric-mean interval (s), real dot
        r["dt_null"] = np.exp(S["ldt_null"] / S["n_null"])
        r["step"] = np.exp(S["lst"] / S["n"])
        r["step_null"] = np.exp(S["lst_null"] / S["n_null"])
        for q in ("apart", "apart_d", "in4"):
            r[q] = S[q] / S["n"]
            r[q + "_null"] = S[q + "_null"] / S["n_null"]
        hrs, hrs0 = S["t_samp"] * GRID / TT.FPS / 3600, S["t_samp_null"] * GRID / TT.FPS / 3600
        for thr in CLOSE:
            e, e0 = S[f"entries{thr:g}"], S[f"entries{thr:g}_null"]
            ins, ins0 = S[f"inside{thr:g}"], S[f"inside{thr:g}_null"]
            r[f"entries{thr:g}_per_h"] = e / hrs
            r[f"entries{thr:g}_per_h_null"] = e0 / hrs0
            r[f"stay{thr:g}_s"] = ins * GRID / TT.FPS / e
            r[f"stay{thr:g}_s_null"] = ins0 * GRID / TT.FPS / e0
            r[f"share{thr:g}"] = ins / S["t_samp"]
            r[f"share{thr:g}_null"] = ins0 / S["t_samp_null"]
    return r


def _boot(res, groups, nb, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nb):
        ids = [i for j in rng.integers(0, len(groups), len(groups)) for i in groups[j]]
        out.append(readouts(pool(res, ids)))
    return out


def summarise(res, groups, nb=300, seed=0):
    est = readouts(pool(res, [i for g in groups for i in g]))
    bs = _boot(res, groups, nb, seed)
    ci = {k: np.nanpercentile([b[k] for b in bs], [2.5, 97.5], axis=0) for k in est}
    return est, ci


def report(nb=300):
    from twin import repulsion_fit as RF
    R = pickle.load(open(OUT / "crd_real.pkl", "rb"))
    keep = set(RF.clutch_split()["test"])
    by = {}
    for fid, c in R["clutch"].items():
        if c in keep and fid in R["res"]:
            by.setdefault(c, []).append(fid)
    allc = {}
    for fid, c in R["clutch"].items():
        if fid in R["res"]:
            allc.setdefault(c, []).append(fid)
    rows = {"real (test half)": summarise(R["res"], list(by.values()), nb),
            "real (fit half)": summarise(R["res"], [g for c, g in allc.items() if c not in keep], nb)}
    for k in KEYS:
        for v in VARS:
            p = OUT / f"crd_sim_{k}_{v}.pkl"
            if p.exists():
                res = pickle.load(open(p, "rb"))
                rows[f"{v} s{0 if k == '14f' else 101}"] = summarise(res, [[i] for i in res], nb, 1)
    js = {nm: {k: dict(est=np.asarray(e[k]).tolist(), lo=np.asarray(ci[k][0]).tolist(), hi=np.asarray(ci[k][1]).tolist())
               for k in e} for nm, (e, ci) in rows.items()}
    GOUT.mkdir(parents=True, exist_ok=True)
    (GOUT / "close_range_diag.json").write_text(json.dumps(dict(bins=DBINS.tolist(), rows=js), indent=1))
    lab = [f"{DBINS[i]:g}-{DBINS[i + 1]:g}" for i in range(7)]

    def tab(key, fmt="{:.2f}", n=7):
        print(f"\n## {key}")
        print(pd.DataFrame({nm: [fmt.format(v) for v in np.asarray(e[key])[:n]] for nm, (e, _) in rows.items()},
                           index=lab[:n]).T.to_string())
    for key in ("occ", "tocc", "dt", "dt_null", "step", "step_null", "apart", "apart_null", "apart_d", "apart_d_null",
                "in4", "in4_null"):
        tab(key)
    print("\n## scalars (estimate [95 % CI])")
    sc = ["occ_shape", "tocc_shape"] + [f"{q}{t:g}{u}{s}" for t in CLOSE
                                        for q, u in (("share", ""), ("entries", "_per_h"), ("stay", "_s"))
                                        for s in ("", "_null")]
    out = {nm: {q: " / ".join(f"{a:.3g} [{b:.3g}, {c:.3g}]" for a, b, c in
                              zip(np.atleast_1d(e[q]), np.atleast_1d(ci[q][0]), np.atleast_1d(ci[q][1])))
                for q in sc} for nm, (e, ci) in rows.items()}
    print(pd.DataFrame(out).to_string())


FNS = {"diag": diag, "pursuit": diag_p}
PFX = {"diag": "crd", "pursuit": "crp"}
P_VARS = ("F", "grT", "grT2")




if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "real":
        run_real(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "sim":
        run_sim(int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS)
    elif cmd == "real-pursuit":
        run_real(int(sys.argv[2]) if len(sys.argv) > 2 else 10, fn="pursuit")
    elif cmd == "sim-pursuit":
        run_sim(int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS, fn="pursuit", variants=P_VARS)
    else:
        raise SystemExit(__doc__)
