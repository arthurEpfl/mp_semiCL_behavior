"""Likelihood fit of short-range terms on top of F + turn (report Model 43), on the Larsch screen data.

UNPUBLISHED DATA (Larsch lab): code only; caches in Model/twin_test/ (git-ignored).

1. RECORDS. The screen's sighted siblings (fit half of the clutches, twin_test/clutch_split.json), segmented with the
   training data's own `segment_bouts.segment_fish`, become model records: positions in model px (mm x 3.60) around the
   training dish centre, the 35 mm dish, the single dot as centroid AND as the only individual dot, N = 1, moving;
   active-only like training (bouts followed by > 10 s of rest dropped, a new run after each); block clock held late.
2. BASE. The model's teacher-forced per-bout predictions at the population fish (z = 0, no drift): the turn mixture, the
   step and interval log-normal mixtures (the interval truncated at 10 s), conditioned on the real turn / step. F + turn's
   repulsion term is included (side input), so a new term is fitted ON TOP of it.
3. TERMS (each a few numbers, gated by g(d) = sigma((r - d) / w) on the dot's distance d, mm):
     size   the away-side LARGE-angle component's logit + v * g(d)   (turning away harder; the mixture's larger-mean
            Gamma on the side away from the dot)
     step   the step's log-means + u * g(d)                           (u < 0: shorter steps near the dot)
     pause  the interval's log-means + p * g(d)                       (p > 0: longer pauses near the dot)
   fitted by maximising the exact log-likelihood of the near-dot bouts (d < 20 mm) of the fit half, by gradient.

    python repulsion_lik.py records KEY          # KEY = 43a (seed 0) / 43b (seed 101): build + cache the base
    python repulsion_lik.py fit KEY TERM ...     # TERM in size / step / pause
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
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "Model"))
import twin_test as TT  # noqa: E402

NEAR_MM = 20.0
CX, CY = 180.0, 174.0               # the training dish centre (model px); the dish radius is 35 mm here
MAX_FISH = 400                      # fit-half fish used (a random subset, fixed seed): plenty for a few numbers


def _segment_plate(args):
    """Real side as twin_test._real_plate, but returning per-fish bout tables with the dot (model px) instead of sums."""
    plate, rows = args
    import pyarrow.parquet as pq
    sys.path.insert(0, str(TT.PKG)); sys.path.insert(0, str(HERE))
    import loader
    import segment_bouts as SB
    z = zipfile.ZipFile(TT.ZIP)
    cols = ["frame"] + [f"{k}_{int(r['well']):02d}" for r in rows for k in ("x", "y", "heading", "flag")]
    tab = pq.read_table(io.BytesIO(z.read(f"schizscreen_twin_v1/trajectories/plate_{plate:03d}.parquet")), columns=cols,
                        filters=[("frame", ">=", TT.WIN[0]), ("frame", "<", TT.WIN[1])]).to_pandas()
    st = TT.stimulus()
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
                           "f_x": np.where(bad, np.nan, CX + x * TT.PX), "f_y": np.where(bad, np.nan, CY + y * TT.PX),
                           "f_ori": np.where(bad, np.nan, o)})
        with contextlib.redirect_stdout(io.StringIO()):
            b = pd.DataFrame(SB.segment_fish(df, "f", int(r["fish_id"]), 1.0))
        if b.empty:
            continue
        dxa, dya = TT.dot_mm(st, r["centre_x_px"], r["centre_y_px"], r["px_per_mm"])
        f = b.frame_onset.to_numpy(int)
        b["dot_cx"] = CX + dxa[f] * TT.PX
        b["dot_cy"] = CY + dya[f] * TT.PX
        b["kind"] = kind_all[f]
        out[int(r["fish_id"])] = b
    return out


def _record(fid, b):
    """One model record per fish from its bout table (several segments / runs concatenated, run_pos restarting)."""
    import evaluate as E
    import fit_MLP_border as MB
    import fit_hmm_stim as S
    b = b[np.isfinite(b.dtheta_deg.to_numpy(float))].copy()
    if len(b) < 10:
        return None
    seg = b["segment_id"].to_numpy() if "segment_id" in b else np.zeros(len(b), int)
    keep = b.dt_s.to_numpy(float) <= TT.MAX_DT                      # active-only: drop the bout before a > 10 s rest
    newrun = np.r_[True, (~keep[:-1]) | (seg[1:] != seg[:-1])]
    b = b.assign(_run=np.cumsum(newrun))[keep]
    if len(b) < 10:
        return None
    rid = b._run.to_numpy()
    rp = np.zeros(len(b), int)
    for t in range(1, len(b)):
        rp[t] = rp[t - 1] + 1 if rid[t] == rid[t - 1] else 0
    x = b.x_px.to_numpy(float); y = b.y_px.to_numpy(float)
    dth = b.dtheta_deg.to_numpy(float)
    head = b.heading_rad.to_numpy(float)
    f_ori = b.f_ori_rad.to_numpy(float) if "f_ori_rad" in b else head
    dcx = b.dot_cx.to_numpy(float); dcy = b.dot_cy.to_numpy(float)
    prev = MB._wrap_pi(head - np.radians(dth))
    phi_ori = MB._wrap_pi(np.arctan2(dcy - y, dcx - x) - f_ori)
    moving = np.isfinite(dcx)
    rec = E.FishRecord(fish_id=int(fid), dtheta=dth, symbol=E.symbols_from_dtheta(dth),
                       b=S.bias_from_phi(phi_ori, moving), stim_angle=phi_ori, moving=moving,
                       dt_s=b.dt_s.to_numpy(float), distance_px=b.distance_px.to_numpy(float))
    rec.x_px, rec.y_px, rec.f_ori_rad, rec.heading_rad = x, y, f_ori, head
    rec.stim_angle_travel = MB._wrap_pi(np.arctan2(dcy - y, dcx - x) - prev)
    R = TT.R_MM * TT.PX
    rec.dish_cx, rec.dish_cy, rec.dish_R = CX, CY, R
    rec.dist_to_border_norm = (R - np.hypot(x - CX, y - CY)) / R
    rec.run_id, rec.run_pos, rec.active = rid, rp, np.ones(len(b), bool)
    rec.frame_onset = b.frame_onset.to_numpy(int)
    rec.dot_cx, rec.dot_cy = dcx, dcy
    rec.dot_dist_px = np.hypot(dcx - x, dcy - y)
    rec.dot_prox = np.zeros(len(b))
    rec.dot_xs = np.full((len(b), 8), np.nan); rec.dot_ys = np.full((len(b), 8), np.nan)
    rec.dot_xs[:, 0], rec.dot_ys[:, 0] = dcx, dcy
    rec.n_dots = np.where(moving, 1.0, 0.0)
    rec.kind = b.kind.to_numpy()
    return rec


def build_records(half="fit", workers=6):
    import repulsion_fit as RF
    keep = set(RF.clutch_split()[half])
    f = TT.fish_table()
    f = f[(f.group == "ref") & f.clutch.isin(keep)]
    f = f.sample(n=min(MAX_FISH, len(f)), random_state=0)
    jobs = [(int(p), g[["fish_id", "well", "centre_x_px", "centre_y_px", "px_per_mm"]].to_dict("records"))
            for p, g in f.groupby("plate")]
    bouts = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_segment_plate, jobs):
            bouts.update(r)
    recs = {fid: _record(fid, b) for fid, b in bouts.items()}
    return {k: v for k, v in recs.items() if v is not None}


def base_params(model, rec):
    """Teacher-forced per-bout predictions of ``model`` on ``rec`` at z = 0, no drift, block clock held late, with the
    model's own repulsion term (side input). Returns numpy arrays (angle params, dist params, dt params)."""
    import torch
    import fit_MLP_border as MB
    import fit_MLP_joint as JT
    cf = model._cfg_feat
    X = MB.build_feature_matrix(rec, cf)
    st = getattr(model._model, "stim_turn", None)
    ck = None
    if st is not None and hasattr(st, "spec"):
        ck = (st.spec().get("idx") or {}).get("clock")
    if ck is None:
        ck = X.shape[1] - MB.tail_width(cf) - (1 if cf.get("run_clock_tau") else 0) - 1
    X[:, ck] = 0.0                                                      # late: the dot is always there
    az, lc, lt = JT._cond_cols(rec.dtheta, rec.distance_px, cf["dth_scale"], cf["kin_scales"], y_dt=rec.dt_s)
    with torch.no_grad(), JT.repulsion_side(model, rec):
        ang, dpar, tpar = model._model.joint_forward(torch.tensor(X), torch.tensor(az), torch.tensor(lc), torch.tensor(lt))
    return ([a.numpy().astype(np.float64) for a in ang], [a.numpy().astype(np.float64) for a in dpar],
            [a.numpy().astype(np.float64) for a in tpar])


def cache_base(key):
    """Records of the fit half + the base predictions of model ``key`` (F + turn, 43a / 43b) -> twin_test/replik_<key>.pkl."""
    import report_models as RM
    sys.path.insert(0, str(HERE.parent / "Model"))
    import model_figures as MF
    u = RM._unit(key)
    with contextlib.redirect_stdout(io.StringIO()):
        ms, ho = MF.fold_models(u["src"], u["arm"])
    models = [ms[k] for k in sorted(ms)]
    recs = build_records("fit")
    rows = []
    for i, (fid, rec) in enumerate(sorted(recs.items())):
        m = models[i % len(models)]                                     # screen fish are unseen by every fold
        ang, dpar, tpar = base_params(m, rec)
        d = rec.dot_dist_px / TT.PX
        phi = rec.stim_angle_travel
        near = np.isfinite(d) & (d < NEAR_MM) & (rec.kind == "dot_bout")
        if not near.any():
            continue
        sl = lambda arr: np.asarray(arr)[near]
        rows.append(dict(fid=fid, d=sl(d), phi=sl(phi), dth=sl(rec.dtheta), dist=sl(rec.distance_px), dt=sl(rec.dt_s),
                         ang=[sl(a) for a in ang], dpar=[sl(a) for a in dpar], tpar=[sl(a) for a in tpar]))
    meta = dict(key=key, n_fish=len(rows), n_bouts=int(sum(len(r["d"]) for r in rows)),
                dt_trunc=models[0]._cfg_feat.get("dt_trunc"), repulsion=models[0]._cfg_feat.get("repulsion"))
    pickle.dump(dict(rows=rows, meta=meta), open(TT.OUT / f"replik_{key}.pkl", "wb"))
    print(f"[{key}] cached {meta['n_bouts']} near-dot bouts of {meta['n_fish']} fish", flush=True)


# --------------------------------------------------------------------------- #
# Likelihood of the near-dot bouts with a candidate term
# --------------------------------------------------------------------------- #
def _stack(rows):
    cat = lambda k: np.concatenate([r[k] for r in rows])
    cat_l = lambda k, j: np.concatenate([r[k][j] for r in rows])
    return dict(d=cat("d"), phi=cat("phi"), dth=cat("dth"), dist=cat("dist"), dt=cat("dt"),
                ang=[cat_l("ang", j) for j in range(len(rows[0]["ang"]))],
                dpar=[cat_l("dpar", j) for j in range(3)], tpar=[cat_l("tpar", j) for j in range(3)])


def loglik(D, term, theta, dt_hi):
    """Per-bout log-likelihood (turn, step, interval) with ``term`` at parameters ``theta`` (torch, differentiable)."""
    import torch
    import fit_MLP_mix as MX
    T = lambda a: torch.as_tensor(a, dtype=torch.float64)
    d, phi = T(D["d"]), T(D["phi"])
    logw, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = [T(a) for a in D["ang"]]
    dlw, dmu, dsg = [T(a) for a in D["dpar"]]
    tlw, tmu, tsg = [T(a) for a in D["tpar"]]
    g = None
    off = 0.0
    if term is not None:
        # a LOCAL gate (reach 3-14 mm, width 0.3-2 mm) + a global offset over every cached bout (0-20 mm): the offset
        # absorbs fish-level differences (the screen's older fish bout ~18 % faster everywhere), so ``amp`` is the
        # near-dot correction relative to the 15-20 mm bouts
        amp, off = theta[0], theta[3]
        r = 3.0 + 11.0 * torch.sigmoid(theta[1]); w = 0.3 + 1.7 * torch.sigmoid(theta[2])
        g = torch.sigmoid((r - d) / w)
    if term == "size":
        # the larger-mean Gamma on the side AWAY from the dot (dot at phi > 0 -> away = dtheta < 0 = components 1, 2)
        away_left = phi > 0
        mL1, mL2, mR1, mR2 = aL1 * tL1, aL2 * tL2, aR1 * tR1, aR2 * tR2
        big_L = torch.where(mL2 > mL1, 2, 1); big_R = torch.where(mR2 > mR1, 4, 3)
        comp = torch.where(away_left, big_L, big_R)
        shift = torch.zeros_like(logw).scatter(1, comp.reshape(-1, 1), (off + amp * g).reshape(-1, 1))
        logw = torch.log_softmax(logw + shift, dim=1)
    if term == "step":
        dmu = dmu + (off + amp * g).reshape(-1, 1)
    if term == "pause":
        tmu = tmu + (off + amp * g).reshape(-1, 1)
    la = MX.mix_logdens(T(D["dth"]), (logw, sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2))
    ld = MX._lnk_logpdf(torch.clamp(T(D["dist"]), min=1e-6), dlw, dmu, dsg)
    lt = MX._dt_logpdf(torch.clamp(T(D["dt"]), min=1e-6), tlw, tmu, tsg, hi=dt_hi)
    return la, ld, lt


def fit(key, term, steps=400, lr=0.05):
    import torch
    blob = pickle.load(open(TT.OUT / f"replik_{key}.pkl", "rb"))
    D = _stack(blob["rows"])
    hi = blob["meta"]["dt_trunc"]
    ok = np.isfinite(D["dt"]) & np.isfinite(D["dist"]) & (D["dist"] > 0)
    D = {k: (v[ok] if not isinstance(v, list) else [a[ok] for a in v]) for k, v in D.items()}
    with torch.no_grad():
        base = [x.sum().item() for x in loglik(D, None, None, hi)]
    theta = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        la, ld, lt = loglik(D, term, theta, hi)
        loss = -(la.sum() + ld.sum() + lt.sum()) / len(D["d"])
        loss.backward()
        opt.step()
    with torch.no_grad():
        la, ld, lt = loglik(D, term, theta, hi)
    new = [la.sum().item(), ld.sum().item(), lt.sum().item()]
    n = len(D["d"])
    w = float(0.3 + 1.7 * torch.sigmoid(theta[2]).item()); rr = float(3.0 + 11.0 * torch.sigmoid(theta[1]).item())
    with torch.no_grad():                                   # the offset alone: what the fish-level difference is worth
        th0 = theta.clone(); th0[0] = 0.0
        la0, ld0, lt0 = loglik(D, term, th0, hi)
    off_only = [la0.sum().item(), ld0.sum().item(), lt0.sum().item()]
    res = dict(key=key, term=term, n_bouts=n, amp=float(theta[0]), r=rr, w=w, offset=float(theta[3]),
               gain_bits_near_term=(sum(new) - sum(off_only)) / n / np.log(2),
               gain_bits_per_bout={c: (new[i] - base[i]) / n / np.log(2) for i, c in enumerate(("turn", "step", "interval"))},
               gain_bits_total_per_bout=(sum(new) - sum(base)) / n / np.log(2))
    print(json.dumps(res, indent=1), flush=True)
    return res




# --------------------------------------------------------------------------- #
# Diagnosis: which component of the backing off do the models miss?
# --------------------------------------------------------------------------- #
COMPONENTS = (("mean |turn| (deg)", "sabs", "n"), ("share of turns > 90 deg", "n90", "n"),
              ("mean |turn| of turns away (deg)", "sabs_away", "naway"), ("mean log step", "slst", "ng"),
              ("mean log interval (<= 10 s)", "sldt", "ndt"), ("share of pauses > 10 s", "nlong", "nall"),
              ("P(turn toward)", "ntow", "nturn"), ("gain (mm/bout)", "gain", "ng"))


def _comp(S, num, den, b):
    """Real-dot minus null-dot value of one component in bin(s) ``b``."""
    with np.errstate(all="ignore"):
        return S[num][b].sum() / S[den][b].sum() - S[num + "_null"][b].sum() / S[den + "_null"][b].sum()


def diagnose(half="fit", bins=(0, 1, 2, 3, 4), nb=200):
    """Table: per component and distance bin, real - null for the real fish (``half`` of the clutches; 95 % over
    clutches) and for F / F + turn at both seeds (simulated, screen protocol)."""
    import repulsion_fit as RF
    real = pickle.load(open(TT.OUT / "real.pkl", "rb"))
    m, res = real["meta"], real["res"]
    keep = set(RF.clutch_split()[half])
    groups = {}
    for fid, g, c in zip(m["fish_id"], m["group"], m["clutch"]):
        if g == "ref" and c in keep and fid in res:
            groups.setdefault(c, []).append(fid)
    gl = list(groups.values())

    def pool(rr, ids):
        S = {}
        for i in ids:
            for k, v in rr[i]["dot_bout"].items():
                S[k] = S.get(k, 0.0) + np.asarray(v, float)
        return S
    sims = {lab: pickle.load(open(TT.OUT / f"sim_{k}_sighted.pkl", "rb"))["res"]
            for lab, k in (("F s0", "14f"), ("F s101", "32b"), ("F+turn s0", "43a"), ("F+turn s101", "43b"))}
    S_real = pool(res, [i for g in gl for i in g])
    rng = np.random.default_rng(0)
    boots = [pool(res, [i for j in rng.integers(0, len(gl), len(gl)) for i in gl[j]]) for _ in range(nb)]
    S_sim = {lab: pool(r, list(r)) for lab, r in sims.items()}
    rows = []
    for name, num, den in COMPONENTS:
        for b in bins:
            sl = slice(b, b + 1)
            v = _comp(S_real, num, den, sl)
            lo, hi = np.nanpercentile([_comp(B, num, den, sl) for B in boots], [2.5, 97.5])
            r = {"component": name, "bin (mm)": f"{TT.DBINS[b]:g}-{TT.DBINS[b + 1]:g}", "real": v, "real lo": lo, "real hi": hi}
            for lab, S in S_sim.items():
                r[lab] = _comp(S, num, den, sl)
            rows.append(r)
    return pd.DataFrame(rows)


def build_step(key):
    """F + turn (43a / 43b) + the likelihood-fitted STEP term -> report Model 50 (50a / 50b): the same checkpoints with
    cfg_feat['repulsion'] extended by s_amp / s_r / s_w, new arm name, manifest, held-out likelihood table."""
    import shutil
    import torch
    import report_models as RM
    import profile_finetune as PF
    u = RM._unit(key)
    src = TT.ROOT / "Model" / u["src"]
    fr = next(r for r in json.loads((TT.OUT / f"replik_fit_{key}.json").read_text()) if r["term"] == "step")
    step = dict(s_amp=round(fr["amp"], 5), s_r=round(fr["r"], 3), s_w=round(fr["w"], 3))
    arm = u["arm"] + "S"
    dst = TT.ROOT / "Model" / {"43a": "rep50_s0", "43b": "rep50_s101"}[key] / arm
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in sorted(src.glob(f"{u['arm']}_fold[0-9].pt")):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        blob["cfg_feat"] = dict(blob["cfg_feat"], repulsion=dict(blob["cfg_feat"]["repulsion"], **step))
        blob["repulsion_step_fit"] = dict(fr, data="schizscreen_twin_v1, fit half (twin_test/clutch_split.json)")
        torch.save(blob, dst / f.name.replace(u["arm"], arm))
    man = json.loads((src / "manifest_model1.json").read_text())
    man = [dict(e, file=e["file"].replace(u["arm"], arm), arm=arm) for e in man if e.get("arm") == u["arm"]]
    (dst / "manifest_model1.json").write_text(json.dumps(man, indent=1))
    with contextlib.redirect_stdout(io.StringIO()):
        PF.write_likA(dst, arm)
    print(f"built {dst} with {dict(json.loads(json.dumps(blob['cfg_feat']['repulsion'])))}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "records":
        for k in sys.argv[2:]:
            cache_base(k)
    elif cmd == "build":
        for k in sys.argv[2:]:
            build_step(k)
    elif cmd == "fit":
        k, terms = sys.argv[2], sys.argv[3:]
        out = [fit(k, t) for t in terms]
        (TT.OUT / f"replik_fit_{k}.json").write_text(json.dumps(out, indent=1))
