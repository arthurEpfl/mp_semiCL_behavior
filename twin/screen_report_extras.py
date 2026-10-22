"""Two readouts for the thesis's screen section, zero-shot and screen-trained side by side (2026-10-19).

UNPUBLISHED DATA (Larsch lab): code only; outputs in Model/twin_test/ (git-ignored).

1. `importances`: what the window classifier uses -- permutation importance, as the thesis's appendix table for our own
   data (GM.c2st_importances): replicate 0 of the screen classifier (screen_eval.c2st: 200 real fish of the pool vs the
   200 simulated fish, equal window counts, 5-fold cross-validation grouped by fish, 200 trees), each feature's column
   permuted 5 times in the held-out fold, importance = mean drop of the held-out AUC over the folds.
   Units: zero-shot M6 (26e | 30d) and F (14f | 32b), screen-trained M6 and D4 (sM6_s*, sD4_s*), sighted.
   -> Model/twin_test/screen_importances.json
2. `closerange`: close_range_diag's two measurements (time-weighted occupancy per distance; the bout's displacement
   split onto toward-the-dot / along-the-dot's-motion / along-the-previous-heading, and the heading alignment) for the
   real fish of the pool (the 98 clutches not used in training; 95 % bootstrap over clutches), zero-shot F (its
   close_range_diag simulations) and the screen-trained M6 and D4 (computed from their raw simulations; bootstrap over
   fish). -> Model/twin_test/screen_closerange.json

    uv run --no-sync --with pyarrow python screen_report_extras.py importances
    uv run --no-sync --with pyarrow python screen_report_extras.py closerange
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from twin import twin_test as TT  # noqa: E402

OUT = TT.OUT
UNITS = {"M6 zero-shot s0": "26e", "M6 zero-shot s101": "30d", "F zero-shot s0": "14f", "F zero-shot s101": "32b",
         "M6 screen s0": "sM6_s0", "M6 screen s101": "sM6_s101", "D4 screen s0": "sD4_s0", "D4 screen s101": "sD4_s101",
         "M6dm screen s0": "sM6dm_s0", "M6dm screen s101": "sM6dm_s101",
         "D4dm screen s0": "sD4dm_s0", "D4dm screen s101": "sD4dm_s101"}     # M6dm: + dot motion direction (Model 51)
SCREEN = {k: v for k, v in UNITS.items() if "screen" in k}


def _pool():
    meta = json.loads((OUT / "screen_train_bouts.json").read_text())
    used = set(meta["clutch"].values())
    R = pickle.load(open(OUT / "screen_raw_real.pkl", "rb"))
    return {i: r for i, r in R["res"].items() if str(R["clutch"].get(int(i))) not in used}, R["clutch"], used


def importances(workers=10, n_repeats=5, only=None):
    from twin import generative_metrics as GM
    from twin import screen_eval as SE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    pool, _, _ = _pool()
    wr = SE.windows_of(pool, workers)
    ids = np.array(sorted(wr))
    p_out = OUT / "screen_importances.json"
    out = json.loads(p_out.read_text()) if (only and p_out.exists()) else {}
    for unit, key in UNITS.items():
        if only and only not in unit:
            continue
        ws = SE.windows_of(pickle.load(open(OUT / f"screen_raw_sim_{key}_sighted.pkl", "rb")), workers)
        rng = np.random.default_rng([2027, 0])                     # replicate 0 of the screen classifier
        XR, gR = SE.stack(wr, rng.choice(ids, SE.N_REAL, replace=False))
        XS, gS = SE.stack(ws, np.array(sorted(ws)), offset=10 ** 7)
        r2 = np.random.default_rng(0)
        n = min(len(XR), len(XS))
        ia = r2.choice(len(XR), n, replace=False); ib = r2.choice(len(XS), n, replace=False)
        X = np.vstack([XR[ia], XS[ib]]); y = np.r_[np.ones(n), np.zeros(n)]; g = np.r_[gR[ia], gS[ib]]
        names = GM.traj_names_for_dim(X.shape[1])
        rp = np.random.default_rng(0)
        drops, aucs = [], []
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
            base = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
            aucs.append(base)
            d = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                for _ in range(n_repeats):
                    Xp = X[te].copy()
                    Xp[:, j] = Xp[rp.permutation(len(Xp)), j]
                    d[j] += (base - roc_auc_score(y[te], clf.predict_proba(Xp)[:, 1])) / n_repeats
            drops.append(d)
        imp = np.mean(drops, 0)
        grp = GM.traj_groups(names)
        pos = np.clip(imp, 0, None)
        share = {gname: float(sum(pos[names.index(f)] for f in fl) / pos.sum()) for gname, fl in grp.items()}
        out[unit] = dict(names=list(names), imp=imp.tolist(), auc=float(np.mean(aucs)), share=share)
        top = sorted(zip(imp, names), reverse=True)[:4]
        print(f"{unit:18s} AUC {np.mean(aucs):.3f}  shares " + " ".join(f"{k} {v:.0%}" for k, v in share.items())
              + "  top: " + ", ".join(f"{nm} {v * 1e3:.1f}" for v, nm in top), flush=True)
    (OUT / "screen_importances.json").write_text(json.dumps(out, indent=1))
    print("wrote screen_importances.json", flush=True)


def _cr_one(args):
    i, r = args
    from twin import close_range_diag as CR
    S = TT._sim_setup()
    f = r["f"].astype(int); x = r["x"].astype(float); y = r["y"].astype(float); seg = r["seg"].astype(int)
    return i, CR.diag(x, y, f, seg, S["dxa"], S["dya"], S["kind"]), CR.diag_p(x, y, f, seg, S["dxa"], S["dya"], S["kind"])


def closerange(workers=10, nb=200, only=None):
    from twin import close_range_diag as CR
    pool, clutch, used = _pool()
    Rd = pickle.load(open(OUT / "crd_real.pkl", "rb"))["res"]
    Rp = pickle.load(open(OUT / "crp_real.pkl", "rb"))["res"]
    by = {}
    for fid in pool:
        if fid in Rd and fid in Rp:
            by.setdefault(clutch[int(fid)], []).append(fid)
    groups_real = list(by.values())
    sides = {"real (pool)": (Rd, Rp, groups_real)}
    for s, k in ((0, "14f"), (101, "32b")):
        d = pickle.load(open(OUT / f"crd_sim_{k}_F.pkl", "rb")); p = pickle.load(open(OUT / f"crp_sim_{k}_F.pkl", "rb"))
        sides[f"F zero-shot s{s}"] = (d, p, [[i] for i in d])
    for unit, key in SCREEN.items():
        if only and only not in unit:
            continue
        raw = pickle.load(open(OUT / f"screen_raw_sim_{key}_sighted.pkl", "rb"))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_cr_one, list(raw.items()), chunksize=8))
        d = {i: a for i, a, _ in res}; p = {i: b for i, _, b in res}
        sides[unit] = (d, p, [[i] for i in d])
        print(f"  {unit}: {len(d)} fish", flush=True)

    def scope(Sd, Sp):
        a = CR.readouts(Sd); b = CR.readouts_p(Sp)
        return dict(tocc=a["tocc"], occ=a["occ"], toward_x=b["toward_x"], along_x=b["along_x"], align_x=b["align_x"],
                    align=b["align"], align_null=b["align_null"])
    out = {}
    rng = np.random.default_rng(0)
    for nm, (rd, rp, groups) in sides.items():
        ids = [i for g in groups for i in g]
        est = scope(CR.pool(rd, ids), CR.pool(rp, ids))
        bs = []
        for _ in range(nb):
            pick = [i for j in rng.integers(0, len(groups), len(groups)) for i in groups[j]]
            bs.append(scope(CR.pool(rd, pick), CR.pool(rp, pick)))
        out[nm] = {q: dict(est=np.asarray(est[q], float).tolist(),
                           lo=np.nanpercentile([b[q] for b in bs], 2.5, 0).tolist(),
                           hi=np.nanpercentile([b[q] for b in bs], 97.5, 0).tolist()) for q in est}
        print(f"{nm:18s} n={len(ids)}  along_x 2-10 mm " + " ".join(f"{v:+.2f}" for v in out[nm]['along_x']['est'][1:5])
              + " | toward_x " + " ".join(f"{v:+.2f}" for v in out[nm]['toward_x']['est'][1:5])
              + " | align " + " ".join(f"{v:.2f}" for v in out[nm]['align']['est'][:5])
              + " | tocc " + " ".join(f"{v:.2f}" for v in out[nm]['tocc']['est'][:5]), flush=True)
    p_cr = OUT / "screen_closerange.json"
    if only and p_cr.exists():                       # keep the units not recomputed
        old = json.loads(p_cr.read_text())["sides"]
        out = {**old, **out}
    p_cr.write_text(json.dumps(dict(bins=TT.DBINS.tolist(), sides=out), indent=1))
    print("wrote screen_closerange.json", flush=True)


def tow_clutch(nb=300, only=None):
    """The real fish's turn toward the dot by distance (screen_eval.curves, row 0), for the pool's reference fish (the
    98 clutches not used in training), with the 95 % bootstrap over CLUTCHES instead of fish, so that it shares its
    resampling unit with the close-range readouts. -> Model/twin_test/screen_tow_clutch.json"""
    from twin import screen_eval as SE
    used = set(json.loads((OUT / "screen_train_bouts.json").read_text())["clutch"].values())
    T = pickle.load(open(OUT / "real.pkl", "rb"))
    by = {}
    for f, g, c in zip(T["meta"]["fish_id"], T["meta"]["group"], T["meta"]["clutch"]):
        if g == "ref" and str(c) not in used and f in T["res"]:
            by.setdefault(str(c), []).append(f)
    groups = list(by.values())

    def tow(sel):
        S = {}
        for i in sel:
            for k, v in T["res"][i][SE.KIND].items():
                S[k] = S.get(k, 0.0) + np.asarray(v, float)
        with np.errstate(all="ignore"):
            return S["ntow"] / S["nturn"] - S["ntow_null"] / S["nturn_null"]
    ids = [i for g in groups for i in g]
    est = tow(ids)
    rng = np.random.default_rng(0)
    bs = np.array([tow([i for j in rng.integers(0, len(groups), len(groups)) for i in groups[j]]) for _ in range(nb)])
    out = dict(bins=TT.DBINS.tolist(), n_fish=len(ids), n_clutches=len(groups),
               real=dict(est=est.tolist(), lo=np.nanpercentile(bs, 2.5, 0).tolist(),
                         hi=np.nanpercentile(bs, 97.5, 0).tolist()))
    (OUT / "screen_tow_clutch.json").write_text(json.dumps(out, indent=1))
    print(f"wrote screen_tow_clutch.json ({len(ids)} fish, {len(groups)} clutches)", flush=True)


if __name__ == "__main__":
    {"importances": importances, "closerange": closerange, "tow_clutch": tow_clutch}[sys.argv[1]](
        only=sys.argv[2] if len(sys.argv) > 2 else None)
