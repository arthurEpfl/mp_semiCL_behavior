"""The screen classifier at the size of our own data: 15 fish a side (2026-10-20, tab:screen-auc).

UNPUBLISHED DATA (Larsch lab): code only; output Model/twin_test/screen_auc15.json (git-ignored).

Same windows and readings as screen_eval / screen_train (all 39 features), but 15 real against 15 simulated fish per
replicate, 20 replicates, equal window counts, 5-fold cross-validation grouped by fish AND stratified by class
(StratifiedGroupKFold: with 30 fish plain GroupKFold can leave a fold with one class), random forest of 200 trees.
Rows as the table: zero-shot block against all screen fish (real vs real: 15 + 15 fish from the two clutch halves),
screen-trained block against the fish of the 98 clutches not used in training (real vs real from the halves of these).

    uv run --no-sync --with pyarrow python screen_auc15.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen_eval as SE  # noqa: E402
import twin_test as TT  # noqa: E402

NF, REPS = 15, 20


def c2st(XA, gA, XB, gB, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    rng = np.random.default_rng(seed)
    n = min(len(XA), len(XB))
    ia = rng.choice(len(XA), n, replace=False); ib = rng.choice(len(XB), n, replace=False)
    X = np.vstack([XA[ia], XB[ib]]); y = np.r_[np.ones(n), np.zeros(n)]; g = np.r_[gA[ia], gB[ib]]
    aucs = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        clf = RandomForestClassifier(200, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def block(wr, ids_a, ids_b, sims, tag):
    out = {}
    v = []
    for rep in range(REPS):
        r = np.random.default_rng([31, rep])
        XA, gA = SE.stack(wr, r.choice(ids_a, NF, replace=False))
        XB, gB = SE.stack(wr, r.choice(ids_b, NF, replace=False))
        v.append(c2st(XA, gA, XB, gB, rep))
    out["real vs real"] = v
    print(f"[{tag}] real vs real: {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f}", flush=True)
    ids = np.r_[ids_a, ids_b]
    for nm, fn in sims.items():
        ws = SE.windows_of(pickle.load(open(TT.OUT / fn, "rb")), 10)
        sid = np.array(sorted(ws))
        v = []
        for rep in range(REPS):
            r = np.random.default_rng([32, rep])
            XA, gA = SE.stack(wr, r.choice(ids, NF, replace=False))
            XB, gB = SE.stack(ws, r.choice(sid, NF, replace=False), offset=10 ** 7)
            v.append(c2st(XA, gA, XB, gB, rep))
        out[nm] = v
        print(f"[{tag}] {nm}: {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f}", flush=True)
    return out


if __name__ == "__main__":
    import repulsion_fit as RF
    R = pickle.load(open(TT.OUT / "screen_raw_real.pkl", "rb"))
    halves = RF.clutch_split()
    hA, hB = set(halves["fit"]), set(halves["test"])
    wr = SE.windows_of(R["res"], 10)
    ids = np.array(sorted(wr))
    cl = {int(i): R["clutch"].get(int(i)) for i in ids}
    meta = json.loads((TT.OUT / "screen_train_bouts.json").read_text())
    used = set(meta["clutch"].values())
    zero = {f"{m}{', blind' if v == 'blind' else ''} s{s}": f"screen_raw_sim_{k}_{v}.pkl"
            for m, keys in (("M6", ("26e", "30d")), ("F", ("14f", "32b"))) for s, k in zip((0, 101), keys)
            for v in ("sighted", "blind")}
    out = {"zero-shot": block(wr, np.array([i for i in ids if cl[i] in hA]), np.array([i for i in ids if cl[i] in hB]),
                              zero, "zero-shot")}
    pool = np.array([i for i in ids if str(cl[i]) not in used])
    tr = {f"{m} zero-shot s{s}": f"screen_raw_sim_{k}_sighted.pkl"
          for m, keys in (("M6", ("26e", "30d")), ("F", ("14f", "32b"))) for s, k in zip((0, 101), keys)}
    tr.update({f"{m}{', decoupled' if v == 'decoupled' else ''} s{s}": f"screen_raw_sim_s{m}_s{s}_{v}.pkl"
               for m in ("M6", "D4") for s in (0, 101) for v in ("sighted", "decoupled")})
    out["screen-trained"] = block(wr, np.array([i for i in pool if cl[i] in hA]),
                                  np.array([i for i in pool if cl[i] in hB]), tr, "screen-trained")
    (TT.OUT / "screen_auc15.json").write_text(json.dumps(dict(nf=NF, reps=REPS, auc=out), indent=1))
    print("wrote screen_auc15.json", flush=True)
