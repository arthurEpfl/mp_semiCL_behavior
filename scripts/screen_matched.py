"""MATCHED window classifier on the screen (2026-10-20; tab:screen-auc, Final_report_screendatatest.ipynb sec. 10).

UNPUBLISHED DATA (Larsch lab): code only; output Model/twin_test/screen_matched.json (git-ignored).

As the thesis's matched AUC on our own data, cut to the screen: every simulated fish is paired with real fish
(real fish of rank j -> simulated fish j mod 200), and for every window of a real fish (60 bouts inside one run of a
bout-like episode, screen_eval.bout_table) the simulated window is the one that starts at the SAME FRAME of the
session in its paired simulated fish: the first kept bout at or after that frame, moved back if the run ends
within 60 bouts. Both sides then cover the same moments of the dot's fixed path, 1:1, and the classifier cannot use
where in the session the windows lie (real fish rest and lose tracking, simulated ones do not). Window features
exactly as screen_eval (39 features). Readings, replicates, real samples, folds and forests as screen_eval /
screen_train (8 replicates of 200 real fish, GroupKFold 5 by fish, 200 trees), plus the 15-fish-a-side column
(20 replicates, stratified grouped folds, as screen_auc15.py) and permutation importances on replicate 0.

Blocks: zero-shot (all screen fish; zero-shot M6, F sighted and blind) and screen-trained (the fish of the 98
clutches not used in training; zero-shot M6, F re-read, screen-trained M6, D4 and the same + dot direction,
sighted and decoupled). The real-vs-real floors are unmatched by nature (both sides real) and recomputed as before.

    uv run --no-sync --with pyarrow python screen_matched.py [WORKERS]
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen_eval as SE  # noqa: E402
import twin_test as TT  # noqa: E402

OUT = TT.OUT
W = SE.W
SEEDS = (0, 101)
_K = {}


def _kind():
    if "k" not in _K:
        _K["k"] = SE._kind()
    return _K["k"]


def kept(r):
    """screen_eval.bout_table + the onset frame of every kept bout."""
    kind_fr = _kind()
    b = SE.bout_table(r, kind_fr)
    if b is None:
        return None
    f, x, y, seg = r["f"].astype(int), r["x"].astype(float), r["y"].astype(float), r["seg"]
    n = len(f)
    L = len(kind_fr)
    leg_ok = (seg[1:] == seg[:-1]) & (np.diff(f) > 0) & (np.diff(f) <= TT.MAX_DT * TT.FPS)
    idx = np.arange(1, n - 1)
    ok = leg_ok[idx - 1] & leg_ok[idx]
    ok &= (kind_fr[np.clip(f[idx], 0, L - 1)] == SE.KIND) & (kind_fr[np.clip(f[idx + 1], 0, L - 1)] == SE.KIND)
    b["fr"] = f[idx[ok]]
    return b


def feats(b, a):
    import evaluate as E
    import generative_metrics as GM
    s = slice(a, a + W)
    X = GM.window_features_traj(b["dth"][s], E.symbols_from_dtheta(b["dth"][s]), b["dist"][s], b["dt"][s], b["dd"][s],
                                W=W, radial=b["rad"][s])
    return X[0]


def real_windows(args):
    """(features (k, 39), start frames (k,)) of one real fish, exactly screen_eval's windows."""
    import generative_metrics as GM
    fid, r = args
    b = kept(r)
    if b is None:
        return fid, None, None
    st = GM._window_starts(len(b["dth"]), W, W, b["run"])
    if not st:
        return fid, None, None
    return fid, np.array([feats(b, a) for a in st]), b["fr"][np.array(st)]


def matched_windows(args):
    """Simulated windows of one simulated fish starting at the given session frames (one per frame, or none)."""
    r, frames = args
    b = kept(r)
    out = np.full((len(frames), 39), np.nan)
    if b is None:
        return out
    fr, run = b["fr"], b["run"]
    n = len(fr)
    first = {}
    last = {}
    for i, q in enumerate(run):
        first.setdefault(q, i); last[q] = i
    for j, f0 in enumerate(frames):
        a = int(np.searchsorted(fr, f0))
        if a >= n:
            continue
        q = run[a]
        a = min(a, last[q] - W + 1)
        if a < first[q]:
            continue
        out[j] = feats(b, a)
    return out


def real_side(res, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        R = {fid: (X, F) for fid, X, F in ex.map(real_windows, list(res.items()), chunksize=16) if X is not None}
    return R


def sim_side(R, sims, workers):
    """{real fish id: (simulated windows matched to its windows, simulated fish id)}, pairing rank j -> sim j mod n."""
    sid = sorted(sims)
    rid = sorted(R)
    pair = {f: sid[j % len(sid)] for j, f in enumerate(rid)}
    jobs = [(sims[pair[f]], R[f][1]) for f in rid]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(matched_windows, jobs, chunksize=8))
    return {f: (Xs, pair[f]) for f, Xs in zip(rid, out)}


def stack_pair(R, S, ids, offset=10 ** 7):
    XR, gR, XS, gS = [], [], [], []
    for f in ids:
        Xr = R[f][0]; Xs, s = S[f]
        ok = np.isfinite(Xs).all(1)                      # a window kept on both sides only
        XR.append(Xr[ok]); gR.append(np.full(ok.sum(), int(f)))
        XS.append(Xs[ok]); gS.append(np.full(ok.sum(), int(s) + offset))
    return np.vstack(XR), np.concatenate(gR), np.vstack(XS), np.concatenate(gS)


def sel(X, rk):
    return X[:, :30] if rk == "k30" else SE.cols(X, rk)


def c2st_strat(XA, gA, XB, gB, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.vstack([XA, XB]); y = np.r_[np.ones(len(XA)), np.zeros(len(XB))]; g = np.r_[gA, gB]
    aucs = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        clf = RandomForestClassifier(200, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def importances(XR, gR, XS, gS, n_repeats=5):
    import generative_metrics as GM
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X = np.vstack([XR, XS]); y = np.r_[np.ones(len(XR)), np.zeros(len(XS))]; g = np.r_[gR, gS]
    names = GM.traj_names_for_dim(X.shape[1])
    rp = np.random.default_rng(0)
    drops, aucs = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
        base = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]); aucs.append(base)
        d = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            for _ in range(n_repeats):
                Xp = X[te].copy(); Xp[:, j] = Xp[rp.permutation(len(Xp)), j]
                d[j] += (base - roc_auc_score(y[te], clf.predict_proba(Xp)[:, 1])) / n_repeats
        drops.append(d)
    imp = np.mean(drops, 0)
    grp = GM.traj_groups(names)
    pos = np.clip(imp, 0, None)
    share = {k: float(sum(pos[names.index(f)] for f in fl) / pos.sum()) for k, fl in grp.items()}
    return dict(names=list(names), imp=imp.tolist(), auc=float(np.mean(aucs)), share=share)


def run_block(tag, R, ids_all, idA, idB, units, workers, out, imp_units=()):
    readings = dict(SE.READINGS, **{"30 features": "k30"})
    blk = dict(auc={r: {} for r in readings}, auc15={}, imp={}, n_real=int(len(ids_all)))
    # floors: real against real, the two clutch halves (unmatched by nature), as before
    for reading, rk in readings.items():
        v = []
        for rep in range(SE.REPS):
            g = np.random.default_rng([2026, rep])
            a = g.choice(idA, SE.N_REAL, replace=False); b = g.choice(idB, SE.N_REAL, replace=False)
            XA = np.vstack([R[f][0] for f in a]); gA = np.concatenate([np.full(len(R[f][0]), int(f)) for f in a])
            XB = np.vstack([R[f][0] for f in b]); gB = np.concatenate([np.full(len(R[f][0]), int(f)) for f in b])
            v.append(SE.c2st(sel(XA, rk), gA, sel(XB, rk), gB, seed=rep)[0])
        blk["auc"][reading]["real vs real"] = v
    v = []
    for rep in range(20):
        g = np.random.default_rng([31, rep])
        a = g.choice(idA, 15, replace=False); b = g.choice(idB, 15, replace=False)
        XA = np.vstack([R[f][0] for f in a]); gA = np.concatenate([np.full(len(R[f][0]), int(f)) for f in a])
        XB = np.vstack([R[f][0] for f in b]); gB = np.concatenate([np.full(len(R[f][0]), int(f)) for f in b])
        v.append(c2st_strat(XA, gA, XB, gB, rep))
    blk["auc15"]["real vs real"] = v
    print(f"[{tag}] floors: " + ", ".join(f"{r} {np.mean(blk['auc'][r]['real vs real']):.3f}" for r in readings)
          + f", 15 fish {np.mean(v):.3f}", flush=True)
    for unit, fn in units.items():
        S = sim_side(R, pickle.load(open(OUT / fn, "rb")), workers)
        for reading, rk in readings.items():
            v = []
            for rep in range(SE.REPS):
                ids = np.random.default_rng([2027, rep]).choice(ids_all, SE.N_REAL, replace=False)
                XR, gR, XS, gS = stack_pair(R, S, ids)
                v.append(SE.c2st(sel(XR, rk), gR, sel(XS, rk), gS, seed=rep)[0])
            blk["auc"][reading][unit] = v
        v = []
        for rep in range(20):
            ids = np.random.default_rng([32, rep]).choice(ids_all, 15, replace=False)
            XR, gR, XS, gS = stack_pair(R, S, ids)
            v.append(c2st_strat(XR, gR, XS, gS, rep))
        blk["auc15"][unit] = v
        if unit in imp_units:
            ids = np.random.default_rng([2027, 0]).choice(ids_all, SE.N_REAL, replace=False)
            blk["imp"][unit] = importances(*stack_pair(R, S, ids))
        print(f"[{tag}] {unit}: " + ", ".join(f"{r} {np.mean(blk['auc'][r][unit]):.3f}" for r in readings)
              + f", 15 fish {np.mean(blk['auc15'][unit]):.3f}", flush=True)
        out[tag] = blk
        (OUT / "screen_matched.json").write_text(json.dumps(out, indent=1))
    return blk


def single_table(workers=10):
    """ONE real reference for the whole table (2026-10-21): everything on the fish of the 98 clutches not used in
    training. Adds to screen_matched.json under 'pool': the real-vs-real floor with the cross-validation grouped by
    CLUTCH (no sibling of a test fish in training; grouped by fish, ~4 siblings per clutch let the classifier recognise
    clutches, 0.577), the zero-shot BLIND models matched on the pool, and our 15 real fish against the pool (30
    features; unmatched by nature, no shared session)."""
    import repulsion_fit as RF
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    Rr = pickle.load(open(OUT / "screen_raw_real.pkl", "rb"))
    halves = RF.clutch_split()
    hA, hB = set(halves["fit"]), set(halves["test"])
    used = set(json.loads((OUT / "screen_train_bouts.json").read_text())["clutch"].values())
    pool_res = {f: r for f, r in Rr["res"].items() if str(Rr["clutch"].get(int(f))) not in used}
    R = real_side(pool_res, workers)
    ids = np.array(sorted(R))
    cl = {int(f): str(Rr["clutch"].get(int(f))) for f in ids}
    code = {c: k for k, c in enumerate(sorted(set(cl.values())))}
    idA = np.array([f for f in ids if cl[f] in hA]); idB = np.array([f for f in ids if cl[f] in hB])
    out = json.loads((OUT / "screen_matched.json").read_text())
    P = dict(auc={}, auc15={}, n_real=int(len(ids)))
    readings = dict(SE.READINGS, **{"30 features": "k30"})

    def by_clutch(XA, cA, XB, cB, seed, strat=False, equal=True):
        rng = np.random.default_rng(seed)
        if equal:
            n = min(len(XA), len(XB))
            ia = rng.choice(len(XA), n, replace=False); ib = rng.choice(len(XB), n, replace=False)
            XA, cA, XB, cB = XA[ia], cA[ia], XB[ib], cB[ib]
        X = np.vstack([XA, XB]); y = np.r_[np.ones(len(XA)), np.zeros(len(XB))]; g = np.r_[cA, cB]
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed) if strat else GroupKFold(n_splits=5)
        a = []
        for tr, te in cv.split(X, y, g):
            clf = RandomForestClassifier(200, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
            a.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        return float(np.mean(a))

    def side(fs):
        return (np.vstack([R[f][0] for f in fs]),
                np.concatenate([np.full(len(R[f][0]), code[cl[f]]) for f in fs]))
    for reading, rk in readings.items():
        v = []
        for rep in range(SE.REPS):
            g = np.random.default_rng([2026, rep])
            XA, cA = side(g.choice(idA, SE.N_REAL, replace=False)); XB, cB = side(g.choice(idB, SE.N_REAL, replace=False))
            v.append(by_clutch(sel(XA, rk), cA, sel(XB, rk), cB, rep))
        P["auc"].setdefault(reading, {})["real vs real"] = v
    v = []
    for rep in range(20):
        g = np.random.default_rng([31, rep])
        XA, cA = side(g.choice(idA, 15, replace=False)); XB, cB = side(g.choice(idB, 15, replace=False))
        v.append(by_clutch(XA, cA, XB, cB, rep, strat=True, equal=False))
    P["auc15"]["real vs real"] = v
    print("floor (grouped by clutch): " + ", ".join(f"{r} {np.mean(P['auc'][r]['real vs real']):.3f}" for r in readings)
          + f", 15 fish {np.mean(v):.3f}", flush=True)
    # zero-shot blind, matched on the pool (the sighted ones are in the 'screen-trained' block)
    for m, keys in (("M6", ("26e", "30d")), ("F", ("14f", "32b"))):
        for s, k in zip(SEEDS, keys):
            u = f"{m} zero-shot s{s} blind"
            S = sim_side(R, pickle.load(open(OUT / f"screen_raw_sim_{k}_blind.pkl", "rb")), workers)
            for reading, rk in readings.items():
                vv = []
                for rep in range(SE.REPS):
                    ii = np.random.default_rng([2027, rep]).choice(ids, SE.N_REAL, replace=False)
                    XR, gR, XS, gS = stack_pair(R, S, ii)
                    vv.append(SE.c2st(sel(XR, rk), gR, sel(XS, rk), gS, seed=rep)[0])
                P["auc"][reading][u] = vv
            vv = []
            for rep in range(20):
                ii = np.random.default_rng([32, rep]).choice(ids, 15, replace=False)
                vv.append(c2st_strat(*stack_pair(R, S, ii), rep))
            P["auc15"][u] = vv
            print(f"{u}: " + ", ".join(f"{r} {np.mean(P['auc'][r][u]):.3f}" for r in readings)
                  + f", 15 fish {np.mean(vv):.3f}", flush=True)
    # our 15 real fish against the pool, 30 features
    XO, gO = SE.our_windows()
    vv = []
    for rep in range(SE.REPS):
        ii = np.random.default_rng([2027, rep]).choice(ids, SE.N_REAL, replace=False)
        XR = np.vstack([R[f][0] for f in ii]); gR = np.concatenate([np.full(len(R[f][0]), int(f)) for f in ii])
        vv.append(SE.c2st(XR[:, :30], gR, XO, gO, seed=rep)[0])
    P["our15_30features"] = vv
    print(f"our 15 fish vs pool, 30 features: {np.mean(vv):.3f}", flush=True)
    out["pool"] = P
    (OUT / "screen_matched.json").write_text(json.dumps(out, indent=1))
    print("updated screen_matched.json (pool)", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "single":
    single_table(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    sys.exit(0)

if __name__ == "__main__":
    import repulsion_fit as RF
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    Rr = pickle.load(open(OUT / "screen_raw_real.pkl", "rb"))
    halves = RF.clutch_split()
    hA, hB = set(halves["fit"]), set(halves["test"])
    meta = json.loads((OUT / "screen_train_bouts.json").read_text())
    used = set(meta["clutch"].values())
    print("real windows ...", flush=True)
    R = real_side(Rr["res"], workers)
    cl = {int(f): Rr["clutch"].get(int(f)) for f in R}
    ids = np.array(sorted(R))
    out = {}
    zero = {f"{m} s{s} {v}": f"screen_raw_sim_{k}_{v}.pkl" for m, keys in (("M6", ("26e", "30d")), ("F", ("14f", "32b")))
            for s, k in zip(SEEDS, keys) for v in ("sighted", "blind")}
    run_block("zero-shot", R, ids, np.array([f for f in ids if cl[f] in hA]), np.array([f for f in ids if cl[f] in hB]),
              zero, workers, out)
    pool = np.array([f for f in ids if str(cl[f]) not in used])
    Rp = {f: R[f] for f in pool}
    tr = {f"{m} zero-shot s{s} sighted": f"screen_raw_sim_{k}_sighted.pkl"
          for m, keys in (("M6", ("26e", "30d")), ("F", ("14f", "32b"))) for s, k in zip(SEEDS, keys)}
    for m in ("M6", "D4", "M6dm", "D4dm"):
        for s in SEEDS:
            for v in ("sighted", "decoupled"):
                fn = f"screen_raw_sim_s{m}_s{s}_{v}.pkl"
                if (OUT / fn).exists():
                    tr[f"{m} screen s{s} {v}"] = fn
    imp_units = [u for u in tr if u.endswith("sighted")]
    run_block("screen-trained", Rp, pool, np.array([f for f in pool if cl[f] in hA]),
              np.array([f for f in pool if cl[f] in hB]), tr, workers, out, imp_units=imp_units)
    print("wrote screen_matched.json", flush=True)
