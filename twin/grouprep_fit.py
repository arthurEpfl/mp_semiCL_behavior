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
    python grouprep_fit.py test 14f|32b     # held-out screen test of every variant -> grtest_<key>.json
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from twin import paths
from twin import twin_test as TT  # noqa: E402

OUT = TT.OUT
GOUT = paths.GROUPREP
N_GRID, N_TEST, WORKERS = 100, 200, 12
BINS_SEL = slice(0, 8)                           # 0-2 ... 15-20 mm
GRID_A = (0.1, 0.2, 0.3, 0.45, 0.6, 0.9)
GRID_RC = (4.0, 6.0, 8.0, 10.0, 12.0)
GRID_SAMP = (-0.02, -0.035, -0.05, -0.08)
GRID_SR = (8.0, 13.5)


# --------------------------------------------------------------------------- #
# Simulation with a term
# --------------------------------------------------------------------------- #
def _sim(args):
    key, i, gp, extra = args
    TT._sim_setup()
    m = TT._models(key)[i % 5]
    base = m.__dict__.setdefault("_cfg_base", dict(m._cfg_feat))
    cfg = dict(base)
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
    tow, gain = TT.curves(S)
    occ, shape = occ_shape(S)
    return tow, gain, occ, shape


def real_targets(half, nb=300):
    real = pickle.load(open(OUT / "real.pkl", "rb"))
    m, res = real["meta"], real["res"]
    keep = set(TT.clutch_split()[half])
    groups = {}
    for fid, g, c in zip(m["fish_id"], m["group"], m["clutch"]):
        if g == "ref" and c in keep and fid in res:
            groups.setdefault(c, []).append(fid)
    gl = list(groups.values())
    tow, gain, occ, shape = all_curves(TT.pool(res, [i for g in gl for i in g]))
    rng = np.random.default_rng(0)
    bs = [all_curves(TT.pool(res, [i for j in rng.integers(0, len(gl), len(gl)) for i in gl[j]])) for _ in range(nb)]
    sd = lambda k: np.nanstd([b[k] for b in bs], 0)
    return dict(tow=tow, gain=gain, occ=occ, shape=shape, se_tow=sd(0), se_gain=sd(1), se_occ=sd(2), se_shape=sd(3),
                n_clutches=len(gl), n_fish=sum(len(g) for g in gl))


def sim_se(res, nb=200):
    rng = np.random.default_rng(1)
    ids = list(res)
    bs = [all_curves(TT.pool(res, [ids[j] for j in rng.integers(0, len(ids), len(ids))])) for _ in range(nb)]
    return tuple(np.nanstd([b[k] for b in bs], 0) for k in range(4))


def loss(res, tgt, se):
    tow, gain, occ, shape = all_curves(TT.pool(res, list(res)))
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
    else:
        for k in sys.argv[2:]:
            {"fit": fit, "test": test}[cmd](k)
