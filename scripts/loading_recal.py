"""Model 23 -- PER-HEAD loading gains recalibrated POST HOC on fitted checkpoints, no retraining.

The fish_re loading gain is a rollout-calibrated constant folded into the fitted loadings after training
(``fishre.apply_loading_gain``; the fold is exactly invertible, re-centring included). The archived
ladders use ONE global gain for the three heads, matched to a sum of three level spreads -- and on
Model 17 it leaves the population of sampled individuals 1.59x too spread in interval level, 1.25x in
step, 0.70x in turn (the harness's per-replicate ``bf_*``). Models 12-13 searched three gains inside the
fit, but against the mean-log-level moments of 8 train fish (a noisy target: bf_dist landed at 0.78)
and were judged on the POOLED dispersion ratios, which mostly measure the mixture of individuals.

Here each fold checkpoint is unfolded to its raw fitted loadings and refitted with three gains matched
to the HARNESS'S OWN between-individual moments: for each head, the SD across sampled individuals of
the per-individual mean of the harness window feature (``absdth_mean`` / ``dist_mean`` / ``dt_mean``,
per replicate, as ``between_fish_metrics`` reads it) against the SD across the fold's 12 TRAIN fish of
the same feature (held-out fish never enter the calibration). For a level-only loading the induced
spread adds in quadrature, sd^2 = base^2 + (g c)^2, so two rollout sets per head (gain 0 and the current
gain) give the gain in closed form; heads are swept in chain order (turn -> step -> interval, the
downstream heads see the settled upstream), twice, with a verification set after each sweep and a
secant correction from it. ~7 rollout sets x 12 fish x 2 draws per fold.

    recalibrate_loading("gain1_ladder_checkpoints", "dip4_ladder_checkpoints",
                        "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", "S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_g3h",
                        rej={"d0": 2.0, "d1": 10.0, "max_draws": 20})

Metric A / C of the new arm are recomputed on the held-out fish (the gain is folded into the scored
weights, so it changes the likelihood, unlike the drift gains).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

HEADS = (("turn", "absdth_mean"), ("dist", "dist_mean"), ("dt", "dt_mean"))
W = 60
G_MIN, G_MAX = 0.1, 6.0


def _owner(pname):
    for k, _ in HEADS:
        if pname.startswith("z_" + k):
            return k
    return "turn"                                  # z_body follows the turn channel (as the fit does)


def _feat_idx():
    import generative_metrics as GM
    nm = GM.traj_names_for_dim(30)                 # no dot / radial block: turn + kin only
    return {k: nm.index(f) for k, f in HEADS}


def _bf(windows, idx):
    """SD across individuals of the per-individual window mean, per head; ``windows`` = list of (n_w, d)."""
    means = np.array([[np.nanmean(w[:, idx[k]]) for k, _ in HEADS] for w in windows if len(w)])
    return {k: float(np.nanstd(means[:, i], ddof=1)) for i, (k, _) in enumerate(HEADS)}


def recal_one(job):
    """(src, dst, rej, draws, seed) -> report. Runs in a worker process (1 torch thread)."""
    src, dst, rej, draws, seed = job
    import torch
    torch.set_num_threads(1)
    import evaluate as E
    import fit_MLP_border as MB
    import fit_MLP_mix as MX
    import fit_MLP_fishre as FR
    import generative_metrics as GM
    from train_model1_colab import metricA, metricC
    t0 = time.time()
    recs = MB.build_border_records(E.load_bouts())
    blob = torch.load(src, map_location="cpu", weights_only=False)
    m = MX.load_mix(str(src))
    net = m._model
    if rej:
        m._cfg_feat = {**m._cfg_feat, "wall_reject": dict(rej)}
    idx = _feat_idx()
    train = [recs[int(f)] for f in blob["train_fish"]]
    held = [int(f) for f in blob["heldout_fish"]]
    # real target: SD across the fold's TRAIN fish of the per-fish window mean (windows inside runs)
    real_w = [GM.window_features_traj(r.dtheta, r.symbol, r.distance_px, r.dt_s, None, W=W,
                                      runs=getattr(r, "run_id", None)) for r in train]
    tgt = _bf(real_w, idx)
    # unfold the archived gain (exact inverse, re-centring included)
    g_old = m.re_gain
    inv = ({k: 1.0 / float(g_old[k]) for k, _ in HEADS} if isinstance(g_old, dict)
           else {k: 1.0 / float(g_old) for k, _ in HEADS})
    FR.apply_loading_gain(net, inv, which="all")
    base = {n_: p.detach().clone() for n_, p in net.named_parameters() if n_.startswith("z_")}
    gen_kw = dict(closed_loop=m.closed_loop, guardrail=m.guardrail, eps=m.eps, stim_loop=m.stim_loop)
    gen = MX._gen_for(m._cfg_feat)
    cache = {}

    def measure(g):
        """Harness bf per head at RAW gains ``g`` (no re-centring: a level shift cannot move a spread)."""
        key = tuple(round(float(g[k]), 5) for k, _ in HEADS)
        if key in cache:
            return cache[key]
        with torch.no_grad():
            for n_, p in net.named_parameters():
                if n_ in base:
                    p.copy_(base[n_] * float(g[_owner(n_)]))
        per_rep = []
        for d in range(int(draws)):
            ws = []
            for r in train:
                o = gen(net, r, m._cfg_feat, np.random.default_rng([int(seed), int(r.fish_id), d, 23]), **gen_kw)
                ws.append(GM.window_features_traj(o["dtheta"], o["symbol"], o["distance_px"], o["dt_s"], None, W=W))
            per_rep.append(_bf(ws, idx))
        out = {k: float(np.mean([p[k] for p in per_rep])) for k, _ in HEADS}
        cache[key] = out
        return out

    g = {k: 1.0 for k, _ in HEADS}
    trace = []
    for sweep in range(2):
        for k, _ in HEADS:
            s0 = measure(dict(g, **{k: 0.0}))[k]
            g1 = g[k] if sweep else 1.0
            s1 = measure(dict(g, **{k: g1}))[k]
            c2 = (s1 ** 2 - s0 ** 2) / (g1 ** 2)
            if tgt[k] ** 2 <= s0 ** 2:
                g_new = G_MIN          # already wider than the target with the latent OFF; the loading cannot help
            elif c2 <= 1e-12:
                # the two measurements are inconsistent (rollout noise), so the loading's effect is not
                # resolvable here: KEEP the current gain. Running to the ceiling instead is what broke
                # fold 2 of the population-code arm -- g = 6.0 drove the rollouts into the wall and
                # COLLAPSED the measured step spread to 0.22 of target, the opposite of the intent.
                g_new = g1
            else:
                g_new = float(np.sqrt((tgt[k] ** 2 - s0 ** 2) / c2))
            g[k] = float(np.clip(g_new, G_MIN, G_MAX))
            trace.append(dict(sweep=sweep, head=k, base_sd=s0, sd_at_g1=s1, g1=g1, gain=g[k], target=tgt[k]))
        got = measure(g)
        trace.append(dict(sweep=sweep, head="verify", **{f"ratio_{k}": got[k] / tgt[k] for k, _ in HEADS}))
    # SAFETY NET. If the verified spread is still off by more than 2x in either direction the solve did not
    # converge -- noise, or a gain large enough to push the rollouts into a degenerate regime. Fall back to
    # the archived gain for that head rather than shipping a worse calibration than the one we started from.
    fallback = []
    for k, _ in HEADS:
        if not (0.5 <= got[k] / max(tgt[k], 1e-9) <= 2.0):
            g[k] = float(g_old[k]) if isinstance(g_old, dict) else float(g_old)
            fallback.append(k)
    if fallback:
        got = measure(g)
        trace.append(dict(sweep=-1, head="fallback", heads=fallback,
                          **{f"ratio_{k}": got[k] / tgt[k] for k, _ in HEADS}))

    # fold the final gains in properly (re-centred) on the RAW loadings
    with torch.no_grad():
        for n_, p in net.named_parameters():
            if n_ in base:
                p.copy_(base[n_])
    FR.apply_loading_gain(net, dict(g), which="all")
    m.re_gain = dict(g)
    m.re_gain_mode = "per_channel"
    m.re_gain_report = dict(kind="harness_bf_recal", target=tgt, got=got, old_gain=g_old, trace=trace, draws=int(draws))
    MX.save_mix(m, str(dst))
    nb = torch.load(dst, map_location="cpu", weights_only=False)
    for k in ("heldout_fish", "train_fish", "train_seed", "rollout_seed", "phi_ref", "psi_ref", "cfg_feat_extra"):
        if k in blob:
            nb[k] = blob[k]
    torch.save(nb, dst)
    likA = [dict(fish=f, **metricA(m, recs[f]), **metricC(m, recs[f])) for f in held]
    return dict(file=Path(src).name, old=g_old, new=g, target=tgt, got=got, fallback=fallback,
                ratio={k: got[k] / tgt[k] for k, _ in HEADS}, trace=trace, likA=likA, secs=time.time() - t0)


def recalibrate_loading(dst, src_dir, arm, new_arm, rej=None, draws=2, seed=0, jobs=5):
    """Write ``new_arm`` = ``arm``'s fold checkpoints with per-head gains recalibrated (see module doc)
    into ``dst`` (which must already hold the control, e.g. via merge_ckpt_dirs). Returns the table."""
    dst, src_dir = Path(dst), Path(src_dir)
    mani_src = json.loads((src_dir / "manifest_model1.json").read_text())
    ents = sorted((e for e in mani_src if e["arm"] == arm), key=lambda e: e["fold"])
    jl = [(src_dir / e["file"], dst / e["file"].replace(arm + "_fold", new_arm + "_fold"), rej, draws, seed) for e in ents]
    print(f"recalibrating {len(jl)} checkpoints of {arm} -> {new_arm} (draws {draws}, {min(jobs, len(jl))} processes)", flush=True)
    with ProcessPoolExecutor(max_workers=min(jobs, len(jl))) as ex:
        reps = list(ex.map(recal_one, jl))
    mani = json.loads((dst / "manifest_model1.json").read_text()) if (dst / "manifest_model1.json").exists() else []
    lik_rows = []
    for e, r in zip(ents, reps):
        mani.append(dict(e, file=e["file"].replace(arm + "_fold", new_arm + "_fold"), arm=new_arm,
                         gain="per_channel_harness_bf", recal=dict(old=r["old"], new=r["new"], ratio=r["ratio"])))
        for row in r["likA"]:
            lik_rows.append(dict(arm=new_arm, rep=e["rep"], order=e["order"], latent=e["latent"], fold=e["fold"], **row))
    (dst / "manifest_model1.json").write_text(json.dumps(mani, indent=1))
    lik = pd.DataFrame(lik_rows)
    if (dst / "likA_model1.csv").exists():
        lik = pd.concat([pd.read_csv(dst / "likA_model1.csv"), lik], ignore_index=True)
    lik.to_csv(dst / "likA_model1.csv", index=False)
    rows = []
    for r in reps:
        old = r["old"] if isinstance(r["old"], dict) else {k: r["old"] for k, _ in HEADS}
        rows.append(dict(file=r["file"], **{f"g_{k}_old": round(float(old[k]), 3) for k, _ in HEADS},
                         **{f"g_{k}": round(r["new"][k], 3) for k, _ in HEADS},
                         **{f"bf_{k}": f"{r['got'][k]:.4f}/{r['target'][k]:.4f} ({r['ratio'][k]:.2f})" for k, _ in HEADS},
                         secs=int(r["secs"])))
    tab = pd.DataFrame(rows).set_index("file")
    pd.set_option("display.width", 250)
    print("\nper-head loading gains, old (global) -> new, and the between-individual spread reached on the TRAIN fish, "
          "gen/real (ratio):")
    print(tab.to_string())
    tab.to_csv(dst / "recalibration_loading.csv")
    return tab
