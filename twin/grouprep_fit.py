"""The GROUP REPULSION term (fit_MLP_joint.grouprep_shifts) on F: calibration on the Larsch screen, the variants, the
held-out screen test and the held-out likelihood on our own recordings (2026-10-18, Final_report_shoaling.ipynb).

One representation for attraction and repulsion: the dot groups the network sees. Each visible moving group's term is
the single-dot form applied to its NEAREST member; groups add (the effects rule on the term). For one dot (the screen)
the nearest member is the dot, so the screen calibrates it directly.

CALIBRATION (fit half of the screen's clutches, each seed separately; closed-loop simulations as twin_test, 100 fish per
grid point): loss = sum of squared z over 0-20 mm of the turn-toward and distance-gain curves (as repulsion_fit's final
selection) + the NEW close-range occupancy shape: bouts per distance bin, dot vs rotated dot, at 0-2 and 2-4 mm relative
to its 4-8 mm level (real fish ~0.84 / 0.92: they hold off the last millimetres). Turn grid a x r_c (w = 1 mm), extended
once if the best point is on an edge; then a step grid at the best turn point.

VARIANTS (built as checkpoint copies of F with cfg_feat['grouprep'], Model/gr_s0 | gr_s101):
    grT    calibrated turn             grTS   calibrated turn + step
    grT2 / grT4 / grT8   turn strength 2 / 4 / 8 x a*
    grTb1  calibrated turn, strength x group count (beta = 1)
    grTC   calibrated turn + the collision rule (2 mm; only acts in the shoal)

    python grouprep_fit.py check            # a = 0 is bit-identical to F; the term acts
    python grouprep_fit.py fit 14f|32b      # calibration -> Model/twin_test/grfit_<key>.json
    python grouprep_fit.py build 14f|32b    # the variants' checkpoints
    python grouprep_fit.py test 14f|32b     # held-out screen test of every variant -> grtest_<key>.json
    python grouprep_fit.py lik              # held-out likelihood on our recordings -> Model/grouprep/lik.csv
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
import pandas as pd

HERE = Path(__file__).resolve().parent
from twin import twin_test as TT  # noqa: E402
from twin import repulsion_fit as RF  # noqa: E402

OUT = TT.OUT
GOUT = HERE.parent / "Model" / "grouprep"
N_GRID, N_TEST, WORKERS = 100, 200, 12
BINS_SEL = slice(0, 8)                           # 0-2 ... 15-20 mm
GRID_A = (0.1, 0.2, 0.3, 0.45, 0.6, 0.9)
GRID_RC = (4.0, 6.0, 8.0, 10.0, 12.0)
GRID_SAMP = (-0.02, -0.035, -0.05, -0.08)
GRID_SR = (8.0, 13.5)
DEST = {"14f": "gr_s0", "32b": "gr_s101"}
SEED_OF = {"14f": 0, "32b": 101}


# --------------------------------------------------------------------------- #
# Simulation with a term
# --------------------------------------------------------------------------- #
def _sim(args):
    key, i, gp, extra = args
    TT._sim_setup()
    m = TT._models(key)[i % 5]
    base = m.__dict__.setdefault("_cfg_base", dict(m._cfg_feat))
    cfg = dict(base)
    cfg.pop("repulsion", None)
    if gp:
        cfg["grouprep"] = dict(gp)
    cfg.update(extra or {})
    m._cfg_feat = cfg
    try:
        _, _, _, meas = TT._sim_one((key, i, True))
    finally:
        m._cfg_feat = dict(base)
    return i, meas


def simulate(ex, key, gp, n=N_GRID, extra=None):
    return dict(ex.map(_sim, [(key, i, gp, extra) for i in range(n)], chunksize=4))


def occ_shape(S):
    """Bouts per distance bin, dot vs rotated dot (three rotations), and its 0-2 / 2-4 mm levels relative to 4-8 mm."""
    with np.errstate(all="ignore"):
        occ = S["n"] / (S["n_null"] / 3.0)
    ref = np.nanmean(occ[2:4])
    return occ, np.array([occ[0] / ref, occ[1] / ref])


def all_curves(S):
    tow, gain = RF.curves(S)
    occ, shape = occ_shape(S)
    return tow, gain, occ, shape


def real_targets(half, nb=300):
    real = pickle.load(open(OUT / "real.pkl", "rb"))
    m, res = real["meta"], real["res"]
    keep = set(RF.clutch_split()[half])
    groups = {}
    for fid, g, c in zip(m["fish_id"], m["group"], m["clutch"]):
        if g == "ref" and c in keep and fid in res:
            groups.setdefault(c, []).append(fid)
    gl = list(groups.values())
    tow, gain, occ, shape = all_curves(RF.pool(res, [i for g in gl for i in g]))
    rng = np.random.default_rng(0)
    bs = [all_curves(RF.pool(res, [i for j in rng.integers(0, len(gl), len(gl)) for i in gl[j]])) for _ in range(nb)]
    sd = lambda k: np.nanstd([b[k] for b in bs], 0)
    return dict(tow=tow, gain=gain, occ=occ, shape=shape, se_tow=sd(0), se_gain=sd(1), se_occ=sd(2), se_shape=sd(3),
                n_clutches=len(gl), n_fish=sum(len(g) for g in gl))


def sim_se(res, nb=200):
    rng = np.random.default_rng(1)
    ids = list(res)
    bs = [all_curves(RF.pool(res, [ids[j] for j in rng.integers(0, len(ids), len(ids))])) for _ in range(nb)]
    return tuple(np.nanstd([b[k] for b in bs], 0) for k in range(4))


def loss(res, tgt, se):
    tow, gain, occ, shape = all_curves(RF.pool(res, list(res)))
    z1 = (tow - tgt["tow"]) / np.sqrt(tgt["se_tow"] ** 2 + se[0] ** 2)
    z2 = (gain - tgt["gain"]) / np.sqrt(tgt["se_gain"] ** 2 + se[1] ** 2)
    z3 = (shape - tgt["shape"]) / np.sqrt(tgt["se_shape"] ** 2 + se[3] ** 2)
    L = float(np.nansum(z1[BINS_SEL] ** 2) + np.nansum(z2[BINS_SEL] ** 2) + np.nansum(z3 ** 2))
    return L, dict(tow=tow.tolist(), gain=gain.tolist(), occ=occ.tolist(), shape=shape.tolist(),
                   L_curves=float(np.nansum(z1[BINS_SEL] ** 2) + np.nansum(z2[BINS_SEL] ** 2)),
                   L_shape=float(np.nansum(z3 ** 2)))


def _jsonable(d):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def fit(key, workers=WORKERS):
    p = OUT / f"grfit_{key}.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    tgt = real_targets("fit")
    out.setdefault("key", key)
    out["targets"] = _jsonable(tgt)
    out.setdefault("grid", [])
    done = {json.dumps(e["gp"], sort_keys=True) for e in out["grid"]}

    def save():
        p.write_text(json.dumps(out, indent=1))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        if "base" not in out:
            base = simulate(ex, key, None)
            se = sim_se(base)
            out["se_sim"] = [s.tolist() for s in se]
            L, cur = loss(base, tgt, se)
            out["base"] = dict(loss=L, **cur)
            save()
            print(f"[{key}] F without the term: loss {L:.1f} (curves {cur['L_curves']:.1f}, shape {cur['L_shape']:.1f})",
                  flush=True)
        se = tuple(np.array(s) for s in out["se_sim"])

        def point(gp, stage):
            k = json.dumps(gp, sort_keys=True)
            if k in done:
                return next(e for e in out["grid"] if json.dumps(e["gp"], sort_keys=True) == k)
            L, cur = loss(simulate(ex, key, gp), tgt, se)
            e = dict(gp=gp, stage=stage, loss=L, **cur)
            out["grid"].append(e); done.add(k); save()
            print(f"[{key}] {stage} {gp}: loss {L:.1f} (curves {cur['L_curves']:.1f}, shape {cur['L_shape']:.1f}; "
                  f"0-2 mm {cur['shape'][0]:.2f})", flush=True)
            return e

        for a in GRID_A:
            for rc in GRID_RC:
                point(dict(a=a, r_c=rc, w=1.0), "turn")
        best = min((e for e in out["grid"] if e["stage"] in ("turn", "turn+")), key=lambda e: e["loss"])
        a0, r0 = best["gp"]["a"], best["gp"]["r_c"]
        ext_a = [round(1.5 * a0, 3)] if a0 >= max(GRID_A) else []
        ext_r = ([r0 + 2.0, r0 + 4.0] if r0 >= max(GRID_RC) else []) + ([r0 - 1.0, r0 - 2.0] if r0 <= min(GRID_RC) else [])
        for a in [a0] + ext_a:
            for rc in [r0] + ext_r:
                point(dict(a=a, r_c=rc, w=1.0), "turn+")
        for a in ext_a:
            point(dict(a=a, r_c=r0, w=1.0), "turn+")
        best = min((e for e in out["grid"] if e["stage"] in ("turn", "turn+")), key=lambda e: e["loss"])
        out["best_turn"] = best["gp"]; save()
        for sa in GRID_SAMP:
            for sr in GRID_SR:
                point(dict(best["gp"], s_amp=sa, s_r=sr, s_w=1.0), "step")
        bs = min((e for e in out["grid"] if e["stage"] == "step"), key=lambda e: e["loss"])
        out["best_step"] = bs["gp"]; save()
    print(f"[{key}] best turn {out['best_turn']} (loss {best['loss']:.1f}); best turn+step {out['best_step']} "
          f"(loss {bs['loss']:.1f}); F alone {out['base']['loss']:.1f}", flush=True)


# --------------------------------------------------------------------------- #
# The variants
# --------------------------------------------------------------------------- #
def variants(key):
    fr = json.loads((OUT / f"grfit_{key}.json").read_text())
    t, ts = dict(fr["best_turn"], beta=0.0), dict(fr["best_step"], beta=0.0)
    return {"grT": (t, {}), "grTS": (ts, {}), "grT2": (dict(t, a=2 * t["a"]), {}),
            "grT4": (dict(t, a=4 * t["a"]), {}), "grT8": (dict(t, a=8 * t["a"]), {}),
            "grTb1": (dict(t, beta=1.0), {}), "grTC": (t, {"collision_mm": 2.0})}


def build(key):
    import torch
    from twin import report_eval as RM
    from twin import profile_finetune as PF
    u = RM._unit(key)
    src = TT.ROOT / "Model" / u["src"]
    for name, (gp, extra) in variants(key).items():
        arm = f"{u['arm']}_{name}"
        dst = TT.ROOT / "Model" / DEST[key] / arm
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in sorted(src.glob(f"{u['arm']}_fold[0-9].pt")):
            blob = torch.load(f, map_location="cpu", weights_only=False)
            cf = dict(blob["cfg_feat"])
            cf.pop("repulsion", None)
            cf["grouprep"] = dict(gp)
            cf.update(extra)
            blob["cfg_feat"] = cf
            blob["grouprep_fit"] = dict(variant=name, key=key, grouprep=gp, extra=extra,
                                        data="schizscreen_twin_v1, fit half (twin_test/clutch_split.json)")
            torch.save(blob, dst / f.name.replace(u["arm"], arm))
        man = json.loads((src / "manifest_model1.json").read_text())
        man = [dict(e, file=e["file"].replace(u["arm"], arm), arm=arm) for e in man if e.get("arm") == u["arm"]]
        (dst / "manifest_model1.json").write_text(json.dumps(man, indent=1))
        with contextlib.redirect_stdout(io.StringIO()):
            PF.write_likA(dst, arm)              # the pipeline's table (torch path: WITHOUT the term -- see lik())
        print(f"built {dst} with {gp} {extra}", flush=True)


# --------------------------------------------------------------------------- #
# Held-out screen test
# --------------------------------------------------------------------------- #
def shares(cur, tgt):
    """Model curves as a share of the real effect: turn toward 0-4 mm, distance gain 0-6 mm (backing off), gain 6-20 mm
    (attraction); and the occupancy shape (0-2 / 2-4 mm relative to 4-8 mm)."""
    m = lambda v, b: float(np.nanmean(np.asarray(v)[b]))
    return dict(turn_away_0_4=m(cur["tow"], slice(0, 2)) / m(tgt["tow"], slice(0, 2)),
                backing_off_0_6=m(cur["gain"], slice(0, 3)) / m(tgt["gain"], slice(0, 3)),
                attraction_6_20=m(cur["gain"], slice(3, 8)) / m(tgt["gain"], slice(3, 8)),
                shape_0_2=cur["shape"][0], shape_2_4=cur["shape"][1],
                real_shape_0_2=float(tgt["shape"][0]), real_shape_2_4=float(tgt["shape"][1]),
                occ_4_8=float(np.nanmean(np.asarray(cur["occ"])[2:4])), real_occ_4_8=float(np.nanmean(tgt["occ"][2:4])))


def test(key, workers=WORKERS):
    p = OUT / f"grtest_{key}.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    tgt = real_targets("test")
    out["targets"] = _jsonable(tgt)
    fr = json.loads((OUT / f"grfit_{key}.json").read_text())
    se = tuple(np.array(s) for s in fr["se_sim"])
    todo = [("F", (None, {}))] + [(k, v) for k, v in variants(key).items() if k != "grTC"]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, (gp, extra) in todo:
            if name in out.get("variants", {}):
                continue
            L, cur = loss(simulate(ex, key, gp, n=N_TEST, extra=extra), tgt, se)
            out.setdefault("variants", {})[name] = dict(gp=gp, loss_test=L, **cur, shares=shares(cur, tgt))
            p.write_text(json.dumps(out, indent=1))
            sh = out["variants"][name]["shares"]
            print(f"[{key}] test {name}: turn-away {sh['turn_away_0_4']:.2f}, backing off {sh['backing_off_0_6']:.2f}, "
                  f"attraction {sh['attraction_6_20']:.2f}, 0-2 mm shape {sh['shape_0_2']:.2f} "
                  f"(real {sh['real_shape_0_2']:.2f})", flush=True)


# --------------------------------------------------------------------------- #
# Held-out likelihood on our own recordings (the torch paths do not see the term)
# --------------------------------------------------------------------------- #
def _base_params(model, rec):
    import torch
    from twin import fit_MLP_border as MB
    from twin import fit_MLP_joint as JT
    cf = model._cfg_feat
    X = MB.build_feature_matrix(rec, cf)
    az, lc, lt = JT._cond_cols(rec.dtheta, rec.distance_px, cf["dth_scale"], cf["kin_scales"], y_dt=rec.dt_s)
    with torch.no_grad():
        ang, dpar, _ = model._model.joint_forward(torch.tensor(X), torch.tensor(az), torch.tensor(lc), torch.tensor(lt))
    return ang, dpar


def _rec_shifts(rec, gp):
    from twin import fit_MLP_border as MB
    from twin import fit_MLP_joint as JT
    h = np.asarray(MB._travel_ref(rec), float)
    dx, dy = np.asarray(rec.dot_xs, float), np.asarray(rec.dot_ys, float)
    s = np.zeros(len(h)); u = np.zeros(len(h)); dmin = np.full(len(h), np.inf)
    for t in range(len(h)):
        q = JT.nearest_dot(dx[t], dy[t], None, rec.x_px[t], rec.y_px[t])
        if q is None or not np.isfinite(h[t]):
            continue
        dmin[t] = np.hypot(q[0] - rec.x_px[t], q[1] - rec.y_px[t]) / JT.REP_PX_PER_MM
        s[t], u[t] = JT.grouprep_shifts([(q[0], q[1], rec.n_dots[t], rec.moving[t])], rec.x_px[t], rec.y_px[t], h[t], gp)
    return s, u, dmin


def lik(keys=("14f", "32b"), n_boot=2000):
    import torch
    from twin import fit_MLP_mix as MX
    from twin import model_figures as MF
    from twin import report_eval as RM
    GOUT.mkdir(parents=True, exist_ok=True)
    rows = []
    recs = MF.records()
    for key in keys:
        u = RM._unit(key)
        with contextlib.redirect_stdout(io.StringIO()):
            models, held = MF.fold_models(u["src"], u["arm"])
        vs = {k: v for k, v in variants(key).items() if k != "grTC"}
        for f, ck in sorted(held.items()):
            m, rec = models[ck], recs[f]
            ang, dpar = _base_params(m, rec)
            dth = torch.tensor(np.asarray(rec.dtheta, np.float64), dtype=ang[0].dtype)
            dist = torch.clamp(torch.tensor(np.asarray(rec.distance_px, np.float64), dtype=ang[0].dtype), min=1e-6)
            ll = {}
            for name, (gp, _) in [("F", (dict(a=0.0), {}))] + list(vs.items()):
                s, uu, dmin = _rec_shifts(rec, gp)
                st = torch.tensor(s, dtype=ang[0].dtype); z = torch.zeros_like(st)
                lw = torch.log_softmax(ang[0] + torch.stack([z, -st, -st, st, st], dim=1), dim=1)
                la = MX.mix_logdens(dth, (lw,) + tuple(ang[1:])).numpy()
                ld = MX._lnk_logpdf(dist, dpar[0], dpar[1] + torch.tensor(uu, dtype=ang[0].dtype).reshape(-1, 1),
                                    dpar[2]).numpy()
                ll[name] = (la, ld)
            nd = np.asarray(rec.n_dots, float); mv = np.asarray(rec.moving, float)
            for name in ll:
                rows.append(pd.DataFrame(dict(seed=SEED_OF[key], fish=f, variant=name, dmin=dmin, n_dots=nd, moving=mv,
                                              ll_turn=ll[name][0], ll_step=ll[name][1])))
    D = pd.concat(rows, ignore_index=True)
    D.to_pickle(GOUT / "lik_bouts.pkl")
    out = []
    rng = np.random.default_rng(0)
    for (sd, var), g in D.groupby(["seed", "variant"]):
        if var == "F":
            continue
        F = D[(D.seed == sd) & (D.variant == "F")]
        for lab, sel in (("all bouts", lambda d: d.dmin > -1), ("moving, nearest dot < 10 mm",
                                                                lambda d: (d.moving > 0) & (d.dmin < 10)),
                         ("moving 1 dot < 10 mm", lambda d: (d.moving > 0) & (d.n_dots == 1) & (d.dmin < 10)),
                         ("moving 2-8 dots < 10 mm", lambda d: (d.moving > 0) & (d.n_dots >= 2) & (d.dmin < 10))):
            a, b = g[sel(g)], F[sel(F)]
            for head in ("turn", "step"):
                diff = (a.groupby("fish")[f"ll_{head}"].sum() - b.groupby("fish")[f"ll_{head}"].sum()) / np.log(2)
                n = a.groupby("fish").size()
                est = diff.sum() / n.sum()
                fish = diff.index.to_numpy()
                bs = [diff.loc[k].sum() / n.loc[k].sum() for k in (rng.choice(fish, len(fish)) for _ in range(n_boot))]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                out.append(dict(seed=sd, variant=var, bouts=lab, head=head, n=int(n.sum()), bits_vs_F=est, lo=lo, hi=hi))
    T = pd.DataFrame(out)
    T.to_csv(GOUT / "lik.csv", index=False)
    print(T.round(4).to_string(index=False))
    return T


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check():
    """a = 0 reproduces F bit for bit on the screen; a > 0 changes the run."""
    TT._sim_setup()
    _, f0 = _sim(("14f", 0, None, None))
    _, g0 = _sim(("14f", 0, dict(a=0.0, r_c=8.0, w=1.0), None))
    _, g1 = _sim(("14f", 0, dict(a=0.6, r_c=8.0, w=1.0), None))
    same = all(np.array_equal(np.asarray(f0["dot_bout"][k]), np.asarray(g0["dot_bout"][k])) for k in f0["dot_bout"])
    diff = any(not np.array_equal(np.asarray(f0["dot_bout"][k]), np.asarray(g1["dot_bout"][k])) for k in f0["dot_bout"])
    print(f"check: a = 0 identical to F: {same}; a = 0.6 differs: {diff}", flush=True)
    if not (same and diff):
        raise SystemExit("grouprep check FAILED")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "lik":
        lik()
    else:
        for k in sys.argv[2:]:
            {"fit": fit, "build": build, "test": test}[cmd](k)
