"""Fit the repulsion term (fit_MLP_joint.repulsion_shifts) on F to the Larsch screen data, then build the corrected
checkpoints (report Model 43, Final_report_extra_data.ipynb section 4).

UNPUBLISHED DATA (Larsch lab): code only; caches in Model/twin_test/ (git-ignored).

SPLIT. The sighted siblings' clutches are split once in two halves (fixed seed): the FIT half gives the targets, the
TEST half is only used afterwards in the notebook.

TARGETS (bout-like dot, fit half): turn toward and gain (each real - rotated null) by distance bin, with the clutch
bootstrap's standard error.

SIMULATION. twin_test's protocol (60-130 min, 35 mm dish, late clock), 200 simulated fish of F (fold i % 5), the SAME
seeds and start poses for every candidate (common random numbers), only the repulsion parameters changing.

LOSS. Sum over the fitted bins of ((sim - real) / sqrt(se_real^2 + se_sim^2))^2, turn toward and gain; se_sim from a
bootstrap over F's simulated fish without the term.
  turn-only ('T'):     a x r_c grid (w = 1 mm), bins 0-10 mm
  turn + brake ('TB'): a in {a_T, 0.7 a_T} x c x r_b grid (r_c from 'T', w = 1 mm, w_b = 1.5 mm), bins 0-15 mm

    python repulsion_fit.py fit KEY [KEY ...]      # F at seed 0 / 101: 14f 32b  (needs pyarrow: uv run --with pyarrow)
    python repulsion_fit.py extend KEY [KEY ...]   # if the best T radius is on the grid's edge: r_c 8-10 mm, TB redone
    python repulsion_fit.py select KEY [KEY ...]   # final choice of T and TB: one loss over 0-20 mm (see select)
    python repulsion_fit.py build KEY [KEY ...]    # the corrected checkpoints + held-out likelihood tables
"""
from __future__ import annotations

import contextlib
import io
import json
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from twin import twin_test as TT  # noqa: E402

SPLIT_SEED = 2026
N_FISH = 200
GRID_T = dict(a=(0.05, 0.1, 0.15, 0.2, 0.3), r_c=(4.0, 5.0, 6.0, 7.0), w=1.0)
GRID_B = dict(c=(0.1, 0.25, 0.4), r_b=(8.0, 11.0), w_b=1.5)
BINS_T = slice(0, 5)        # 0-2 ... 8-10 mm
BINS_TB = slice(0, 7)       # 0-2 ... 12-15 mm
DEST = {"14f": "rep43_s0", "32b": "rep43_s101"}


def clutch_split():
    p = TT.OUT / "clutch_split.json"
    if p.exists():
        return json.loads(p.read_text())
    real = pickle.load(open(TT.OUT / "real.pkl", "rb"))
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


def real_targets(half="fit", nb=300):
    real = pickle.load(open(TT.OUT / "real.pkl", "rb"))
    m, res = real["meta"], real["res"]
    keep = set(clutch_split()[half])
    groups = {}
    for fid, g, c in zip(m["fish_id"], m["group"], m["clutch"]):
        if g == "ref" and c in keep and fid in res:
            groups.setdefault(c, []).append(fid)
    gl = list(groups.values())
    tow, gain = curves(pool(res, [i for g in gl for i in g]))
    rng = np.random.default_rng(0)
    bs = [curves(pool(res, [i for j in rng.integers(0, len(gl), len(gl)) for i in gl[j]])) for _ in range(nb)]
    return dict(tow=tow, gain=gain, se_tow=np.nanstd([b[0] for b in bs], 0), se_gain=np.nanstd([b[1] for b in bs], 0),
                n_clutches=len(gl), n_fish=sum(len(g) for g in gl))


def _sim(args):
    key, i, rep = args
    S = TT._sim_setup()
    m = TT._models(key)[i % 5]
    base = m.__dict__.setdefault("_cfg_base", dict(m._cfg_feat))
    m._cfg_feat = dict(base, repulsion=rep) if rep else dict(base)
    _, _, _, meas = TT._sim_one((key, i, True))
    m._cfg_feat = dict(base)
    return i, meas


def simulate(ex, key, rep):
    res = dict(ex.map(_sim, [(key, i, rep) for i in range(N_FISH)], chunksize=4))
    return res


def loss(res, tgt, se_sim, bins):
    tow, gain = curves(pool(res, list(res)))
    z1 = (tow - tgt["tow"]) / np.sqrt(tgt["se_tow"] ** 2 + se_sim[0] ** 2)
    z2 = (gain - tgt["gain"]) / np.sqrt(tgt["se_gain"] ** 2 + se_sim[1] ** 2)
    return float(np.nansum(z1[bins] ** 2) + np.nansum(z2[bins] ** 2)), tow, gain


def fit(key, workers=7):
    tgt = real_targets("fit")
    out = dict(key=key, targets={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in tgt.items()}, grid=[])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        base = simulate(ex, key, None)
        rng = np.random.default_rng(1); ids = list(base)
        bs = [curves(pool(base, [ids[j] for j in rng.integers(0, len(ids), len(ids))])) for _ in range(200)]
        se_sim = (np.nanstd([b[0] for b in bs], 0), np.nanstd([b[1] for b in bs], 0))
        L, tw, gn = loss(base, tgt, se_sim, BINS_T)
        LB, _, _ = loss(base, tgt, se_sim, BINS_TB)
        out["base"] = dict(loss_T=L, loss_TB=LB, tow=tw.tolist(), gain=gn.tolist())
        print(f"[{key}] F without the term: loss T {L:.1f}, TB {LB:.1f}", flush=True)
        best_T = None
        for a in GRID_T["a"]:
            for rc in GRID_T["r_c"]:
                rep = dict(a=a, r_c=rc, w=GRID_T["w"])
                L, tw, gn = loss(simulate(ex, key, rep), tgt, se_sim, BINS_T)
                out["grid"].append(dict(variant="T", rep=rep, loss=L, tow=tw.tolist(), gain=gn.tolist()))
                print(f"[{key}] T  {rep}: loss {L:.1f}", flush=True)
                if best_T is None or L < best_T[0]:
                    best_T = (L, rep)
        out["best_T"] = dict(loss=best_T[0], rep=best_T[1])
        best_TB = None
        for a in (best_T[1]["a"], round(0.7 * best_T[1]["a"], 4)):
            for c in GRID_B["c"]:
                for rb in GRID_B["r_b"]:
                    rep = dict(best_T[1], a=a, c=c, r_b=rb, w_b=GRID_B["w_b"])
                    L, tw, gn = loss(simulate(ex, key, rep), tgt, se_sim, BINS_TB)
                    out["grid"].append(dict(variant="TB", rep=rep, loss=L, tow=tw.tolist(), gain=gn.tolist()))
                    print(f"[{key}] TB {rep}: loss {L:.1f}", flush=True)
                    if best_TB is None or L < best_TB[0]:
                        best_TB = (L, rep)
        out["best_TB"] = dict(loss=best_TB[0], rep=best_TB[1])
    (TT.OUT / f"repfit_{key}.json").write_text(json.dumps(out, indent=1))
    print(f"[{key}] best T {out['best_T']}  best TB {out['best_TB']}", flush=True)


def extend(key, workers=7):
    """When the best T radius sits on the grid's edge: T at r_c 8-10 mm around the best strength, then the TB grid
    again from the new best T. Appends to repfit_<key>.json (the main grid is kept)."""
    p = TT.OUT / f"repfit_{key}.json"
    out = json.loads(p.read_text())
    tgt = {k: (np.array(v) if isinstance(v, list) else v) for k, v in out["targets"].items()}
    rT = out["best_T"]["rep"]
    edge_r, edge_a = rT["r_c"] >= max(GRID_T["r_c"]), rT["a"] >= max(GRID_T["a"])
    edge_c = out["best_TB"]["rep"]["c"] <= min(GRID_B["c"])
    if not (edge_r or edge_a or edge_c):
        print(f"[{key}] best T / TB inside the grid: no extension", flush=True)
        return
    a0 = rT["a"]
    a_set = sorted({a for a in GRID_T["a"] if abs(GRID_T["a"].index(a) - GRID_T["a"].index(a0)) <= 1}
                   | ({round(1.5 * a0, 4)} if edge_a else set()))
    r_set = (rT["r_c"], 8.0, 9.0, 10.0) if edge_r else (rT["r_c"],)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        base = simulate(ex, key, None)
        rng = np.random.default_rng(1); ids = list(base)
        bs = [curves(pool(base, [ids[j] for j in rng.integers(0, len(ids), len(ids))])) for _ in range(200)]
        se_sim = (np.nanstd([b[0] for b in bs], 0), np.nanstd([b[1] for b in bs], 0))
        best_T = (out["best_T"]["loss"], rT)
        done = {(e["rep"]["a"], e["rep"]["r_c"]) for e in out["grid"] if e["variant"] == "T"}
        for a in a_set:
            for rc in r_set:
                if (a, rc) in done:
                    continue
                rep = dict(a=a, r_c=rc, w=GRID_T["w"])
                L, tw, gn = loss(simulate(ex, key, rep), tgt, se_sim, BINS_T)
                out["grid"].append(dict(variant="T", rep=rep, loss=L, tow=tw.tolist(), gain=gn.tolist(), ext=True))
                print(f"[{key}] T+ {rep}: loss {L:.1f}", flush=True)
                if L < best_T[0]:
                    best_T = (L, rep)
        out["best_T"] = dict(loss=best_T[0], rep=best_T[1])
        if best_T[1] != rT or edge_c:
            out["grid"] = [e for e in out["grid"] if e["variant"] != "TB"]
            best_TB = None
            rc = best_T[1]["r_c"]
            for a in (best_T[1]["a"], round(0.7 * best_T[1]["a"], 4)):
                for c in (0.05, 0.1, 0.2):
                    for rb in (rc + 1.0, rc + 4.0):
                        rep = dict(best_T[1], a=a, c=c, r_b=rb, w_b=GRID_B["w_b"])
                        L, tw, gn = loss(simulate(ex, key, rep), tgt, se_sim, BINS_TB)
                        out["grid"].append(dict(variant="TB", rep=rep, loss=L, tow=tw.tolist(), gain=gn.tolist(), ext=True))
                        print(f"[{key}] TB+ {rep}: loss {L:.1f}", flush=True)
                        if best_TB is None or L < best_TB[0]:
                            best_TB = (L, rep)
            out["best_TB"] = dict(loss=best_TB[0], rep=best_TB[1])
    p.write_text(json.dumps(out, indent=1))
    print(f"[{key}] after the extension: best T {out['best_T']}  best TB {out['best_TB']}", flush=True)


BINS_SEL = slice(0, 8)      # 0-2 ... 15-20 mm: from contact to the attraction peak


def select(key, workers=7):
    """FINAL SELECTION (revised objective, decided before any test-half result was looked at). The grid losses use
    0-10 mm (T) / 0-15 mm (TB); a wide term (r_c ~ 10 mm) is then rewarded for cancelling the models' excess pull at
    6-10 mm while the damage it does to the real attraction at 10-20 mm is unseen. So the best T and TB are chosen
    again, among every evaluated grid point (curves stored), with one loss over 0-20 mm for both variants."""
    p = TT.OUT / f"repfit_{key}.json"
    out = json.loads(p.read_text())
    tgt = {k: (np.array(v) if isinstance(v, list) else v) for k, v in out["targets"].items()}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        base = simulate(ex, key, None)
    rng = np.random.default_rng(1); ids = list(base)
    bs = [curves(pool(base, [ids[j] for j in rng.integers(0, len(ids), len(ids))])) for _ in range(200)]
    se = (np.nanstd([b[0] for b in bs], 0), np.nanstd([b[1] for b in bs], 0))

    def L(tow, gain):
        tow, gain = np.asarray(tow), np.asarray(gain)
        z1 = (tow - tgt["tow"]) / np.sqrt(tgt["se_tow"] ** 2 + se[0] ** 2)
        z2 = (gain - tgt["gain"]) / np.sqrt(tgt["se_gain"] ** 2 + se[1] ** 2)
        return float(np.nansum(z1[BINS_SEL] ** 2) + np.nansum(z2[BINS_SEL] ** 2))
    out["base"]["loss_sel"] = L(out["base"]["tow"], out["base"]["gain"])
    for e in out["grid"]:
        e["loss_sel"] = L(e["tow"], e["gain"])
    for var in ("T", "TB"):
        g = [e for e in out["grid"] if e["variant"] == var]
        b = min(g, key=lambda e: e["loss_sel"])
        out[f"best_{var}_gridloss"] = out[f"best_{var}"]
        out[f"best_{var}"] = dict(loss=b["loss_sel"], rep=b["rep"], objective="0-20 mm")
    p.write_text(json.dumps(out, indent=1))
    print(f"[{key}] selection over 0-20 mm: F {out['base']['loss_sel']:.1f}; best T {out['best_T']}; "
          f"best TB {out['best_TB']}", flush=True)


def build(key):
    """Copy F's fold checkpoints with cfg_feat['repulsion'] set to the fitted values (T and TB), new arm names, the
    manifest, and the held-out likelihood table (profile_finetune.write_likA, which scores with the term)."""
    import torch
    from twin import report_eval as RM
    from twin import profile_finetune as PF
    u = RM._unit(key)
    src = TT.ROOT / "Model" / u["src"]
    fitres = json.loads((TT.OUT / f"repfit_{key}.json").read_text())
    for var in ("T", "TB"):
        rep = fitres[f"best_{var}"]["rep"]
        arm = f"{u['arm']}_rep{var}"
        dst = TT.ROOT / "Model" / DEST[key] / arm
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in sorted(src.glob(f"{u['arm']}_fold[0-9].pt")):
            blob = torch.load(f, map_location="cpu", weights_only=False)
            blob["cfg_feat"] = dict(blob["cfg_feat"], repulsion=dict(rep))
            blob["repulsion_fit"] = dict(variant=var, key=key, rep=rep, loss=fitres[f"best_{var}"]["loss"],
                                         data="schizscreen_twin_v1, fit half of the clutches (twin_test/clutch_split.json)")
            torch.save(blob, dst / f.name.replace(u["arm"], arm))
        man = json.loads((src / "manifest_model1.json").read_text())
        man = [dict(e, file=e["file"].replace(u["arm"], arm), arm=arm) for e in man if e.get("arm") == u["arm"]]
        (dst / "manifest_model1.json").write_text(json.dumps(man, indent=1))
        with contextlib.redirect_stdout(io.StringIO()):
            PF.write_likA(dst, arm)
        print(f"built {dst} ({len(list(dst.glob('*_fold*.pt')))} folds) with {rep}", flush=True)


if __name__ == "__main__":
    cmd, keys = sys.argv[1], sys.argv[2:]
    for k in keys:
        {"fit": fit, "extend": extend, "select": select, "build": build}[cmd](k)
