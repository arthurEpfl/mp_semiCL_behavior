"""PROFILE FINE-TUNE: match the dot-distance profile directly, on a handful of readable parameters.

WHY. Per-bout likelihood cannot see block-level behaviour: the group-size contrast is worth ~0.002 bits/bout, so
every stimulus rung since Model 24 moved a PER-BOUT number and left the approach curve where it was (Models 30
and 31 fired / confirmed the pre-registered stopping rule that names the training objective as the next lever).
So the curve itself becomes a training target -- but only through ``JT.StimTurnTerm``: 6 numbers, a log-odds
shift between turning TOWARD and AWAY from each MOVING dot group, as a function of its log dot count, mirror-shared
across left/right. Everything else in the checkpoint stays frozen.

THE OBJECTIVE, per fold, on the fold's TRAINING fish only (the verdict comes from its held-out fish):

    L(theta) = NLL(theta)  +  lambda * sum_conditions sum_bins (xbar_c(t) - ybar_c(t))^2 / SEM_c(t)^2

* ``NLL``: the usual joint likelihood -- turn + step + interval -- on the real training bouts at each fish's
  fitted z, with the per-head regime weights the model was trained with (``w3ft``: x3 on moving turn bouts
  >= 60 px). v1 levers (6 numbers) reach the turn head only; the v2 levers (``--v2``, 12 numbers: + range and
  group-ahead straightness / step) also reach the step head. The interval head reads the REAL turn and step at
  training time and no lever, so its term is a constant: computed and logged (per step and on all data before /
  after), not trained.
* the PROFILE term: distance to the dot centroid in 2-s bins over 0-60 s after stimulus onset, per moving
  condition. ``xbar`` = the model's mean over BLOCK-START ROLLOUTS -- each starts at a real training block's
  onset with the real fish's pose, the last <= 64 real bouts replayed as history, its fitted z (twin mode),
  and that block's real dots replayed bout by bout -- ``ybar`` = the same blocks' real curve (paired, so the
  block composition is identical). Bins where the dot centroid is outside the dish are masked on BOTH sides:
  the dots are replayed, so the mask is the same frames on both sides.
* the GRADIENT of the profile term by the score function: rollouts are random draws, so each rollout gets an
  advantage -- how far its own curve, against the other M-1 rollouts of the same block (leave-one-out
  baseline), pushes its condition's mean toward the real curve -- and the gradient raises the log-probability
  of the turns taken in rollouts that helped. Matching CONDITION MEANS pooled over many blocks, not individual
  trajectories, is what keeps it resolvable (``rollout_aware.py`` matched window features trajectory by
  trajectory and was null).

KNOWN BIAS, documented not fixed: with ``rej`` on, turns near the wall are drawn from the rejection-truncated
distribution, not the one whose log-probability the estimator uses. Masking outside-dish frames keeps most
wall-bound bouts out of the scored bins.

Entry points:
    gradient_gate(src_ckpt, ...)          -- is the gradient resolvable? run BEFORE any fine-tune
    finetune_fold(job)                    -- one fold (worker process)
    finetune_arm(src_dir, arm, dst_dir, new_arm, ...)   -- all folds -> ``<arm>_pf`` checkpoints + report
"""
from __future__ import annotations

import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


FPS = 30.0
PX_TO_MM = 3.67
BIN_S = 2.0
T_MAX_S = 60.0
P_MAX = 64
GAP_S = 10.0
EDGES = np.arange(0.0, T_MAX_S + 1e-9, BIN_S)
MOVING_CONDS = ("d1s3.5", "d2s3.5", "d4s3.5", "d6s3.5", "d8s3.5")
STATIC_CONDS = ("d1s0", "d4s0", "d8s0")                 # in the objective only when asked (v5 'Bst')


# ----------------------------------------------------------------------------- setup
def attach_stim_turn(model, v2=False, v3=False, v4=False, v5=None):
    """Attach a zero-initialised profile term to a loaded fishre model (no-op if one is attached).
    ``v2`` adds the range and group-ahead levers (12 numbers; Model 33); ``v3`` is the saturating-count term
    (10 numbers; Model 34), which also works on the centroid (S4) layout; ``v4`` is one shift per dot count plus a
    late-block factor (15 numbers; Model 40), centroid layout only; ``v5`` = 'S' / 'B' / 'BS' / 'Bst', the
    count-gated terms (report Model 11; ``JT.StimTurnTermV5``), centroid layout only."""
    import torch
    from twin import fit_MLP_joint as JT
    net = model._model
    if int(getattr(net, "state", 0) or 0):
        raise ValueError("the profile fine-tune is built for non-recurrent nets (state = 0)")
    if getattr(net, "stim_turn", None) is None:
        if v5:
            n_in = next(mm for mm in net.body.modules() if isinstance(mm, torch.nn.Linear)).in_features
            net.stim_turn = JT.stim_turn_v5(model._cfg_feat, str(v5), n_features=n_in)
        elif v4:
            n_in = next(mm for mm in net.body.modules() if isinstance(mm, torch.nn.Linear)).in_features
            net.stim_turn = JT.stim_turn_v4(model._cfg_feat, n_in)
        elif v3:
            net.stim_turn = JT.stim_turn_v3(model._cfg_feat)
        else:
            mv, nz = JT.stim_turn_indices(model._cfg_feat)
            net.stim_turn = JT.StimTurnTerm(mv, nz, rho_idx=JT.stim_rho_indices(model._cfg_feat) if v2 else None)
    return net.stim_turn


def _theta(st):
    """The trainable numbers, in a fixed order (coef, then coef_rng and ahead when v2)."""
    return [p for _, p in st.named_parameters()]


def _flat(st):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in _theta(st)])


def _unfold_gain(m):
    """Unfold the post-hoc per-head loading gains (exact inverse, re-centring included) and return them for
    ``_refold``. They were calibrated for FRESH prior draws; the fine-tune rolls the TRAINING fish out at their
    FITTED z, which belongs with the loadings as fitted -- the g3h step gain of ~0.5 would otherwise halve each
    training fish's own step level."""
    from twin import fit_MLP_fishre as FR
    g = getattr(m, "re_gain", None)
    if g is None:
        return None
    gd = dict(g) if isinstance(g, dict) else {k: float(g) for k in ("turn", "dist", "dt")}
    FR.apply_loading_gain(m._model, {k: 1.0 / float(v) for k, v in gd.items()}, which="all")
    return gd


def _refold(m, gd):
    from twin import fit_MLP_fishre as FR
    if gd is not None:
        FR.apply_loading_gain(m._model, dict(gd), which="all")


def fitted_z(model, blob):
    """{fish_id: fitted embedding} for the checkpoint's TRAINING fish (row i of ``emb`` is train_fish[i],
    by construction of the fit's fish index)."""
    W = model._model.emb.weight.detach().cpu().numpy()
    tr = [int(f) for f in blob["train_fish"]]
    assert W.shape[0] == len(tr), (W.shape, len(tr))
    return {f: W[i].astype(float) for i, f in enumerate(tr)}


def _slice_rec(rec, idx):
    n0 = int(rec.n)
    r = copy.copy(rec)
    for attr, val in list(vars(rec).items()):
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n0:
            setattr(r, attr, val[idx])
    return r


def block_starts(model, rec, conds=MOVING_CONDS, t_max=T_MAX_S, p_max=P_MAX, min_bouts=6, warm=0, reanchor=False):
    """Block-start records for one fish: each is the real block (bouts up to ``t_max`` s after onset,
    truncated at the first > 10 s gap) preceded by up to ``p_max`` real bouts of the same run, replayed.

    ``warm`` > 0 (Model 34) adds, for every block whose fish was already swimming at least ``warm`` bouts
    before onset, a FREE-WARM-UP variant ``rs_w``: the real bouts are replayed only up to ``warm`` bouts
    before onset and the model swims the last ``warm`` itself (no dots on screen then: the inter-block
    pause), so it meets the block in ITS OWN pose and history -- the state a free-running evaluation fish
    arrives in, which the real-start rollouts never show the objective. The dots are replayed bout-indexed
    from onset (bout P) on in both variants, so the real curve and the scoring are unchanged.

    ⚠ ``reanchor`` (report Model 12 of the profile fine-tunes; False = the archived behaviour): the replayed dots were
    placed around the REAL fish's onset pose, so a warm-up rollout -- which reaches bout P wherever its own warm-up took
    it -- met the block 32 mm from the dots against 17 for real starts (measured on Model 17's folds), could never match
    the early curve, and pushed every v5 fine-tune toward more pull (the lost late-block rebound). With ``reanchor`` the
    warm-up record carries ``_reanchor_at = P``: both engines move the track rigidly onto the simulated fish's pose at
    bout P (``MB.reanchor_track``, the rig's rule) and score the rollout against those dots."""
    from twin import fit_MLP_border as MB
    cf = model._cfg_feat
    lev = MB.kin_dt_level(rec.dt_s, cf["kin_scales"][2])
    s0, bid, fr = MB.block_onsets(rec)
    tb = (fr - s0) / FPS
    mv = np.asarray(rec.moving, bool)
    nd = np.nan_to_num(np.asarray(getattr(rec, "n_dots", np.zeros(rec.n)), float)).astype(int)
    dcx = np.asarray(rec.dot_cx, float); dcy = np.asarray(rec.dot_cy, float)
    x = np.asarray(rec.x_px, float); y = np.asarray(rec.y_px, float)
    rp = np.asarray(rec.run_pos).astype(int)
    out = []
    for b in np.unique(bid):
        ii = np.where(bid == b)[0]
        i0 = ii[0]
        if nd[i0] <= 0:
            continue
        cond = f"d{nd[i0]}s{'3.5' if mv[i0] else '0'}"
        if cond not in conds:
            continue
        ii = ii[tb[ii] < t_max]
        gaps = np.flatnonzero(np.diff(fr[ii]) / FPS > GAP_S)
        if gaps.size:
            ii = ii[:gaps[0] + 1]
        if len(ii) < min_bouts:
            continue
        P = int(min(p_max, rp[i0]))
        idx = np.arange(i0 - P, ii[-1] + 1)
        rs = _slice_rec(rec, idx)
        rs._force_P = P
        rs._rp_off = int(rp[i0] - P)
        rs._kdt_lev0 = float(lev[i0 - P])
        cen_in = np.hypot(dcx[ii] - rec.dish_cx, dcy[ii] - rec.dish_cy) <= rec.dish_R
        d_real = np.hypot(x[ii] - dcx[ii], y[ii] - dcy[ii]) / PX_TO_MM
        s = dict(fish=int(rec.fish_id), block=int(b), cond=cond, rs=rs, P=P, tb=tb[ii],
                 dcx=dcx[ii], dcy=dcy[ii], mask=cen_in, y=_binned(tb[ii], d_real, cen_in))
        if warm and P >= int(warm):
            rw = copy.copy(rs)
            rw._force_P = P - int(warm)
            if reanchor:
                rw._reanchor_at = P
            s["rs_w"] = rw
        out.append(s)
    return out


def _binned(tb, d, mask):
    """(n_bins,) per-bin mean of ``d`` over the bouts in ``mask``; NaN where a bin has none."""
    k = np.clip(np.floor(np.asarray(tb) / BIN_S).astype(int), 0, len(EDGES) - 2)
    out = np.full(len(EDGES) - 1, np.nan)
    for j in np.unique(k[mask]):
        out[j] = float(np.mean(d[mask & (k == j)]))
    return out


def real_weights(starts, conds=MOVING_CONDS):
    """1 / SEM^2 per (condition, bin) of the real curve, SEM across FISH (per-fish mean over blocks first,
    the natural replicate unit), from ALL the fold's training blocks. Bins with < 3 fish get weight 0."""
    w = {}
    for c in conds:
        per = {}
        for s in starts:
            if s["cond"] == c:
                per.setdefault(s["fish"], []).append(s["y"])
        if not per:
            continue
        fm = np.array([np.nanmean(np.vstack(v), 0) for v in per.values()])        # (fish, bins)
        n = np.sum(np.isfinite(fm), 0)
        sem = np.nanstd(fm, 0, ddof=1) / np.sqrt(np.maximum(n, 1))
        w[c] = np.where((n >= 3) & np.isfinite(sem) & (sem > 0), 1.0 / np.maximum(sem, 0.05) ** 2, 0.0)
    return w


# ----------------------------------------------------------------------------- rollouts and log-probs
def rollout_starts(model, batch, M, seed, step, crn=1):
    """M rollouts of every block start in ``batch``: (per-rollout binned curve, free-run feature rows, the
    free-run turn and step DRAWS and the step head's turn conditioner) -- the forced prefix is excluded from
    everything scored. ``crn`` > 1 = COMMON RANDOM NUMBERS: the same draws for ``crn`` consecutive steps, so
    step-to-step changes in the gradient come from theta, not from fresh noise."""
    from twin import fit_MLP_joint as JT
    cf = model._cfg_feat
    res = []
    for s in batch:
        for m in range(M):
            # odd rollouts of a block with a free-warm-up variant start from it: half real starts, half own state
            v = int("rs_w" in s and m % 2 == 1)
            g = model.sample(s["rs_w"] if v else s["rs"],
                             np.random.default_rng([int(seed), int(step) // max(int(crn), 1),
                                                    s["fish"], s["block"], m]), keep_features=True)
            P = s["P"]
            gx = np.asarray(g["x_px"], float)[P:]; gy = np.asarray(g["y_px"], float)[P:]
            dcx, dcy, mask = s["dcx"], s["dcy"], s["mask"]
            r0 = s["rs_w"] if v else s["rs"]
            if getattr(r0, "_reanchor_at", None) is not None:
                # a re-anchored warm-up start is scored against the dots IT saw, masked by where THEY sit
                dcx = np.asarray(g["dot_cx"], float)[P:]; dcy = np.asarray(g["dot_cy"], float)[P:]
                mask = np.hypot(dcx - r0.dish_cx, dcy - r0.dish_cy) <= r0.dish_R
            d = np.hypot(gx - dcx, gy - dcy) / PX_TO_MM
            az, _ = JT._cond_cols(np.asarray(g["dtheta"], float)[P:], np.asarray(g["distance_px"], float)[P:],
                                  cf["dth_scale"], cf["kin_scales"])
            # the log-density is scored at the value the mixture DREW (pre-containment-bias / pre-vigor)
            res.append(dict(s=s, m=m, v=v, x=_binned(s["tb"], d, mask),
                            X=np.asarray(g["X"], np.float32)[P:], az=az,
                            dth=np.asarray(g["dth_base"], float)[P:], dist=np.asarray(g["dist_base"], float)[P:],
                            dturn=_drift_tilt(cf, g, P)))
    return res


def _drift_tilt(cf, g, P):
    """(n,) the drift's tilt of the FORWARD component's logit on the scored bouts (turn loading x drift state), or
    None without a drift: the generator draws each turn from the mixture WITH it, so the score-function log-density
    must include it too."""
    dr = cf.get("drift") or {}
    d_turn = float(dr.get("turn", 0.0))
    if not d_turn or g.get("d_state") is None:
        return None
    return (d_turn * np.asarray(g["d_state"], float)[P:]).astype(np.float32)


def rollouts(model, batch, M, seed, step, zmap, crn=1, engine="numpy"):
    """The block-start rollouts with either engine: 'numpy' = one ``model.sample`` per trajectory (the reference),
    'batch' = ``rollout_batch`` (all trajectories of a chunk at once; verified equal in features, equal in
    distribution). Same per-rollout dicts, same order."""
    if engine == "batch":
        from twin import rollout_batch as RB
        return RB.rollout_starts(model, batch, M, seed, step, zmap, crn=crn)
    return rollout_starts(model, batch, M, seed, step, crn=crn)


def _frozen_logits(net, X, z, az=None):
    """(turn head's 14 raw outputs, step head's raw outputs), both WITHOUT the profile term, at latent ``z``
    (no grad). ``az`` = the step head's turn conditioner (only the step outputs depend on it)."""
    import torch
    with torch.no_grad():
        Xt = torch.as_tensor(X, dtype=torch.float32)
        z = np.asarray(z, np.float32)
        zt = torch.as_tensor(z if z.ndim == 2 else np.repeat(z[None], len(X), 0))   # one z, or one per row
        zero = torch.zeros(len(X))
        a = zero if az is None else torch.as_tensor(az, dtype=torch.float32)
        out, do, _ = net._trunk(Xt, a, zero, zt)
        k = net.n_dist
        st = net.step_term(Xt)
        if torch.is_tensor(st):
            do = torch.cat([do[:, :k], do[:, k:2 * k] - st, do[:, 2 * k:]], dim=1)
        return out - net.stim_turn(Xt), do


def _logp(net, base, X, dth, base_do=None, dist=None):
    """(n,) log p(turn | row; theta) [+ log p(step | row, turn; theta) when the v2 step lever is attached and
    ``base_do`` / ``dist`` are given], with the frozen parts ``base`` / ``base_do`` -- differentiable in theta."""
    import torch
    from twin import fit_MLP_mix as MX
    from twin import fit_MLP as M
    Xt = torch.as_tensor(X, dtype=torch.float32)
    lp = MX.mix_logdens(torch.as_tensor(dth, dtype=torch.float32), net._angle_params(base + net.stim_turn(Xt)))
    if base_do is not None and getattr(net.stim_turn, "v2", False):
        k = net.n_dist
        yd = torch.as_tensor(dist, dtype=torch.float32)
        ok = (torch.isfinite(yd) & (yd > 0)).float()
        dpar = (torch.log_softmax(base_do[:, :k], dim=1), base_do[:, k:2 * k] + net.step_term(Xt),
                torch.nn.functional.softplus(base_do[:, 2 * k:3 * k]) + M.EPS)
        lp = lp + ok * MX._lnk_logpdf(torch.clamp(torch.nan_to_num(yd, nan=1.0), min=1e-6), *dpar)
    return lp


def profile_surrogate(net, rolls, w, zmap, chunk=65536):
    """(surrogate whose gradient estimates d(profile loss)/d(theta), profile loss value, per-cond gaps).
    Paired: xbar and ybar over the SAME blocks of the batch."""
    import torch
    scored = []
    by = {}
    for r in rolls:
        by.setdefault(r["s"]["cond"], []).append(r)
    surrogate = torch.zeros(())
    loss = 0.0; gaps = {}
    for c, rs in by.items():
        if c not in w:
            continue
        Xc = np.vstack([r["x"] for r in rs])                                   # (n_roll, bins)
        blocks = {}                       # leave-one-out baseline groups: one block x one start variant
        for r in rs:
            blocks.setdefault((r["s"]["block"], r["s"]["fish"], r.get("v", 0)), []).append(r)
        Yc = np.vstack([s_["y"] for s_ in {(k[0], k[1]): b[0]["s"] for k, b in blocks.items()}.values()])
        xbar = np.nanmean(Xc, 0); ybar = np.nanmean(Yc, 0)
        n_x = np.sum(np.isfinite(Xc), 0)
        ok = np.isfinite(xbar) & np.isfinite(ybar) & (w[c] > 0)
        gap = np.where(ok, xbar - ybar, 0.0)
        loss += float(np.sum(w[c][ok] * gap[ok] ** 2)); gaps[c] = gap
        coef = np.where(ok, 2.0 * w[c] * gap / np.maximum(n_x, 1), 0.0)       # d loss / d x_bm(t)
        for b in blocks.values():
            xs = np.vstack([r["x"] for r in b])
            for i, r in enumerate(b):
                others = np.delete(xs, i, 0)
                base_line = np.nanmean(others, 0) if len(others) else np.full_like(r["x"], np.nan)
                dev = r["x"] - base_line
                adv = float(np.nansum(np.where(np.isfinite(dev), coef * dev, 0.0)))
                if adv == 0.0 or len(r["X"]) == 0:
                    continue
                scored.append((adv, r))
    # ONE batched pass over every scored bout (sum_r adv_r sum_t log p_rt == sum over rows of adv x log p), chunked;
    # identical to scoring rollout by rollout, without ~10^4 small forward calls
    rows = [(np.full(len(r["X"]), a, np.float32), r) for a, r in scored]
    if rows:
        cat = lambda k: np.concatenate([r[k] for _, r in rows])  # noqa: E731
        X = np.concatenate([r["X"] for _, r in rows]).astype(np.float32)
        A = np.concatenate([a for a, _ in rows])
        Zr = np.concatenate([np.repeat(np.asarray(zmap[r["s"]["fish"]], np.float32)[None], len(r["X"]), 0)
                             for _, r in rows])
        az, dth, dist = cat("az"), cat("dth"), cat("dist")
        # the drift's forward-logit tilt, per scored bout (0 where a rollout carries none)
        tilt = np.concatenate([r["dturn"] if r.get("dturn") is not None else np.zeros(len(r["X"]), np.float32)
                               for _, r in rows]).astype(np.float32)
        for c0 in range(0, len(X), chunk):
            sl = slice(c0, c0 + chunk)
            base, base_do = _frozen_logits(net, X[sl], Zr[sl], az[sl])
            if np.any(tilt[sl]):
                base = base.clone(); base[:, 0] = base[:, 0] + torch.as_tensor(tilt[sl])
            lp = _logp(net, base, X[sl], dth[sl], base_do, dist[sl])
            surrogate = surrogate + (torch.as_tensor(A[sl]) * lp).sum()
    return surrogate, loss, gaps


# ----------------------------------------------------------------------------- the likelihood anchor
def _head_weights(r, rw):
    """{turn, dist, dt} per-bout training weights, exactly as the trainer applies ``regime_w``."""
    from twin import fit_MLP_border as MB
    ones = np.ones(r.n)
    if not rw:
        return dict(turn=ones, dist=ones, dt=ones)
    wv, info = MB.regime_weights(r, **rw)
    if "per_head" in info:
        return {h: np.asarray(info["per_head"][h], float) for h in ("turn", "dist", "dt")}
    heads = set(rw.get("heads") or ("turn", "dist", "dt"))
    return {h: (np.asarray(wv, float) if h in heads else ones) for h in ("turn", "dist", "dt")}


def nll_cache(model, recs, train_fish, zmap):
    """Everything the JOINT likelihood needs, built once per fold: feature rows, the three real channels,
    the within-bout conditioners, each fish's fitted z per row, and the training regime weights per head."""
    from twin import fit_MLP_border as MB
    from twin import fit_MLP_joint as JT
    cf = model._cfg_feat
    rw = getattr(model, "regime_w", None)
    parts = []
    for f in train_fish:
        r = recs[f]
        X = MB.build_feature_matrix(r, cf).astype(np.float32)
        az, lc = JT._cond_cols(r.dtheta, r.distance_px, cf["dth_scale"], cf["kin_scales"])
        hw = _head_weights(r, rw)
        parts.append(dict(X=X, az=az, lc=lc, ya=np.asarray(r.dtheta, np.float32),
                          yd=np.asarray(r.distance_px, np.float32), yt=np.asarray(r.dt_s, np.float32),
                          z=np.repeat(np.asarray(zmap[f], np.float32)[None], r.n, 0),
                          wa=hw["turn"], wd=hw["dist"], wt=hw["dt"]))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def nll_joint(net, cache, idx):
    """(turn, step, interval) weighted-mean NLL on rows ``idx`` -- the trainer's per-head form. Only the turn
    and step terms depend on theta; the interval head reads the REAL turn and step at training time and no
    lever, so its NLL is a constant here: computed and logged, not trained."""
    import torch
    from twin import fit_MLP_fishre as FR
    t = lambda k, dt=torch.float32: torch.as_tensor(cache[k][idx], dtype=dt)
    la, ld, lt, md, mt = FR._chan_logliks(net, t("X"), t("az"), t("lc"), t("ya"), t("yd"), t("yt"), t("z"))
    wa, wd, wt = t("wa"), t("wd"), t("wt")
    zero = torch.zeros(())
    n_a = -(wa * la).sum() / wa.sum()
    n_d = -(wd[md] * ld[md]).sum() / wd[md].sum() if md.any() else zero
    n_t = -(wt[mt] * lt[mt]).sum() / wt[mt].sum() if mt.any() else zero
    return n_a, n_d, n_t


def nll_full(net, cache, chunk=16384):
    """The three NLLs on EVERY training bout (no grad), row-weighted over chunks."""
    import torch
    n = len(cache["ya"])
    acc = np.zeros(3); wsum = np.zeros(3)
    with torch.no_grad():
        for i in range(0, n, chunk):
            idx = np.arange(i, min(n, i + chunk))
            vals = nll_joint(net, cache, idx)
            yd, yt = cache["yd"][idx], cache["yt"][idx]
            ws = np.array([cache["wa"][idx].sum(),
                           cache["wd"][idx][np.isfinite(yd) & (yd > 0)].sum(),
                           cache["wt"][idx][np.isfinite(yt) & (yt > 0)].sum()])
            acc += np.array([float(v) for v in vals]) * ws; wsum += ws
    return dict(zip(("turn", "step", "interval"), (acc / np.maximum(wsum, 1e-12)).tolist()))


# ----------------------------------------------------------------------------- the gate
def _grad(net, value):
    """d value / d theta, flattened in ``_theta`` order (zeros for a number the value does not reach)."""
    ps = _theta(net.stim_turn)
    for p in ps:
        p.grad = None
    if value.requires_grad:
        value.backward()
    return np.concatenate([(p.grad if p.grad is not None else torch_zeros_like(p)).detach().cpu().numpy().ravel()
                           for p in ps]).copy()


def torch_zeros_like(p):
    import torch
    return torch.zeros_like(p)


def _freeze_base(net):
    """Only the profile term trains: everything else stops accumulating gradients (and graph)."""
    for p in net.parameters():
        p.requires_grad_(False)
    for p in _theta(net.stim_turn):
        p.requires_grad_(True)


def _load(src, rej):
    from twin import fit_MLP_mix as MX
    import torch
    m = MX.load_mix(str(src))
    if rej:
        m._cfg_feat = {**m._cfg_feat, "wall_reject": dict(rej)}
    blob = torch.load(src, map_location="cpu", weights_only=False)
    if getattr(m, "regime_w", None) is None:
        m.regime_w = blob.get("regime_w")
    return m, blob


def gradient_gate(src, rej=None, B=40, M=8, pairs=4, seed=0, v2=False, v3=False, warm=0, engine="numpy", v4=False):
    """At theta = 0: is the profile gradient RESOLVABLE? Draws ``pairs`` pairs of independent batches (each
    pair split by FISH, so the halves share no animal) and reports the cosine between their gradients. The
    pre-registered bar is a mean cosine >= 0.4 at the planned batch size; below it the fine-tune would be
    pushing on noise, as ``rollout_aware.py`` did."""
    import torch
    from twin import model_figures as MF
    torch.set_num_threads(1)
    m, blob = _load(src, rej)
    attach_stim_turn(m, v2=v2, v3=v3, v4=v4)
    _unfold_gain(m)
    _freeze_base(m._model)
    zmap = fitted_z(m, blob)
    m.z_twin = zmap
    recs = MF.records()
    tr = [int(f) for f in blob["train_fish"]]
    starts = [s for f in tr for s in block_starts(m, recs[f], warm=warm)]
    w = real_weights(starts)
    rng = np.random.default_rng(seed)
    halves = (tr[0::2], tr[1::2])
    cos, gA_all, t0 = [], [], time.time()
    for k in range(pairs):
        gs = []
        for h in halves:
            pool = [s for s in starts if s["fish"] in h]
            batch = pool if B is None else _stratified(pool, B, rng)    # None = every block of the half
            rolls = rollouts(m, batch, M, seed + 1000 + k, 0, zmap, engine=engine)
            sur, _, _ = profile_surrogate(m._model, rolls, w, zmap)
            gs.append(_grad(m._model, sur))
        gA_all.append(gs)
        cos.append(float(np.dot(gs[0], gs[1]) / (np.linalg.norm(gs[0]) * np.linalg.norm(gs[1]) + 1e-12)))
    return dict(cos=cos, mean_cos=float(np.mean(cos)), passes=bool(np.mean(cos) >= 0.4),
                grads=gA_all, secs=time.time() - t0, n_starts=len(starts), B=B, M=M)


def _stratified(pool, B, rng):
    by = {}
    for s in pool:
        by.setdefault(s["cond"], []).append(s)
    per = max(1, B // max(len(by), 1))
    out = []
    for c, ss in by.items():
        out += [ss[i] for i in rng.choice(len(ss), size=min(per, len(ss)), replace=False)]
    return out


# ----------------------------------------------------------------------------- the fine-tune
def finetune_fold(job):
    """(src, dst, cfg) -> report. cfg: rej, steps, B (None = every training block), M, lr, lam_ratio,
    nll_batch, seed."""
    src, dst, cfg = job
    import torch
    from twin import fit_MLP_mix as MX
    from twin import model_figures as MF
    torch.set_num_threads(1)
    t0 = time.time()
    m, blob = _load(src, cfg.get("rej"))
    st = attach_stim_turn(m, v2=bool(cfg.get("v2")), v3=bool(cfg.get("v3")), v4=bool(cfg.get("v4")),
                          v5=cfg.get("v5"))
    gains = _unfold_gain(m)
    _freeze_base(m._model)
    zmap = fitted_z(m, blob)
    m.z_twin = zmap
    recs = MF.records()
    tr = [int(f) for f in blob["train_fish"]]
    conds = tuple(cfg.get("conds") or MOVING_CONDS)
    starts = [s for f in tr for s in block_starts(m, recs[f], conds=conds, warm=int(cfg.get("warm", 0)),
                                                  reanchor=bool(cfg.get("reanchor")))]
    w = real_weights(starts, conds)
    cache = nll_cache(m, recs, tr, zmap)
    nll0 = nll_full(m._model, cache)       # the whole joint likelihood before (interval must not move)
    rng = np.random.default_rng(cfg.get("seed", 0))
    ps = _theta(st)
    opt = torch.optim.Adam(ps, lr=float(cfg.get("lr", 0.05)))
    # LEARNING-RATE DECAY (Model 34): Model 33's profile loss plateaued by step ~20 while the coefficients kept
    # wandering ~0.02 per step along flat directions (a/b/c, h/k trade-offs) -- Adam's own step size. Halving the
    # rate at the given steps lets them settle once the loss has stopped improving. None = constant (as before).
    halve = [int(s) for s in (cfg.get("lr_halve") or [])]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=halve, gamma=0.5) if halve else None
    nb = int(cfg.get("nll_batch", 8192))
    lam, trace, g0, snaps = None, [], None, []
    B = cfg.get("B")
    for step in range(int(cfg["steps"])):
        # B = None -> EVERY training block each step: the gate showed the profile gradient is resolvable only at
        # full data (fish-split cosine 0.97 with all ~300 blocks per half, 0.05 with 40)
        batch = starts if B is None else _stratified(starts, int(B), rng)
        rolls = rollouts(m, batch, int(cfg["M"]), cfg.get("seed", 0), step, zmap, crn=int(cfg.get("crn", 1)),
                         engine=cfg.get("engine", "numpy"))
        sur, ploss, gaps = profile_surrogate(m._model, rolls, w, zmap)
        idx = rng.choice(len(cache["ya"]), size=min(nb, len(cache["ya"])), replace=False)
        n_a, n_d, n_t = nll_joint(m._model, cache, idx)
        nll = n_a + n_d + n_t                                          # the usual likelihood, all three heads
        g_p = _grad(m._model, sur)                                     # the profile's pull, logged per step
        g_n = _grad(m._model, nll)                                     # the likelihood anchor's pull
        if lam is None:                    # set ONCE so both terms pull with comparable strength at the start
            lam = float(cfg.get("lam_ratio", 1.0)) * np.linalg.norm(g_n) / max(np.linalg.norm(g_p), 1e-12)
        opt.zero_grad()
        g = g_n + lam * g_p
        # WEAK REGULARISERS (v3 only, Model 36). Model 34's traces: log kappa kept drifting where the data do not care
        # (at a step every count >= 2 is saturated whatever kappa; on a ramp only b / kappa matters; with b ~ 0 kappa
        # does nothing), and ahead-k kept growing while the profile loss barely registered it. A pull toward 0 on those
        # three numbers pins them where the data are indifferent. Scale: at |theta| = 1 the pull is `prior_logk` /
        # `pen_k` x the typical per-parameter pull at step 0 (G0), so the data win wherever they actually care.
        # Which numbers they act on comes from the term (`reg_index`): v3 = log kappa [2, 6] and ahead k [9], v4 = ahead
        # k [13] (v4 has no kappa); v1 / v2 have none.
        # PULL ON EVERY NUMBER (`pen_all`, Model 41): Model 40's fold 4 ran away along the range slopes -- a genuine but
        # weak signal that nothing bounded -- until its rollouts changed so much the gradient estimate turned to noise.
        # The same G0-scaled pull toward 0, on all levers, stops a lever where its profile signal is weak.
        reg = 0.0
        w_lk, w_k = float(cfg.get("prior_logk", 0.0) or 0.0), float(cfg.get("pen_k", 0.0) or 0.0)
        w_all = float(cfg.get("pen_all", 0.0) or 0.0)
        ri = getattr(st, "reg_index", None)
        if (ri and (w_lk or w_k)) or w_all:
            if g0 is None:
                g0 = float(np.mean(np.abs(g)))
            th = _flat(st)
            rg = np.zeros_like(g)
            ilk, ik = (list(ri.get("logk", [])), list(ri.get("k", []))) if ri else ([], [])
            if ilk and w_lk:
                rg[ilk] += w_lk * g0 * th[ilk]
            if ik and w_k:
                rg[ik] += w_k * g0 * th[ik]
            if w_all:
                rg += w_all * g0 * th
            reg = 0.5 * g0 * (w_lk * float(np.sum(th[ilk] ** 2)) + w_k * float(np.sum(th[ik] ** 2))
                              + w_all * float(np.sum(th ** 2)))
            g = g + rg
        o = 0
        for p in ps:
            p.grad = torch.as_tensor(g[o:o + p.numel()].reshape(p.shape), dtype=p.dtype); o += p.numel()
        lr_now = float(opt.param_groups[0]["lr"])
        snaps.append(_flat(st))                                        # the numbers this step's loss was measured at
        opt.step()
        if sched is not None:
            sched.step()
        trace.append(dict(step=step, lr=lr_now, profile=ploss, nll=float(nll.detach()), nll_turn=float(n_a.detach()),
                          nll_step=float(n_d.detach()), nll_interval=float(n_t.detach()), lam=lam,
                          loss=float(nll.detach()) + lam * ploss + reg, reg=reg, n_blocks=len(batch),
                          coef=_flat(st).tolist(),
                          grad_profile=(lam * g_p).tolist(), grad_nll=g_n.tolist(),
                          gap2=float(np.mean(gaps.get("d2s3.5", [np.nan]))), secs=time.time() - t0))
        print(f"  [{Path(src).name}] step {step}: profile {ploss:.1f}  nll turn {float(n_a.detach()):.4f} step "
              f"{float(n_d.detach()):.4f} int {float(n_t.detach()):.4f}  coef {np.round(_flat(st), 3).tolist()}  "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    # KEEP THE BEST STEP (`keep_best` = W, Model 41), not the last: the numbers at the step whose W-step running mean of
    # the training objective is lowest. Training fish only -- ordinary early stopping; a run that diverges late (Model
    # 40's fold 4: loss 558 at step ~50, then 1100-2500 after step 100) keeps its good stretch.
    best = None
    keep = int(cfg.get("keep_best", 0) or 0)
    if keep and trace:
        L = np.array([t["loss"] for t in trace], float)
        W = min(keep, len(L))
        roll = np.convolve(L, np.ones(W) / W, mode="valid")              # roll[i] = mean of steps i .. i + W - 1
        i = int(np.argmin(roll)); s_best = i + W - 1
        th, o = snaps[s_best], 0
        with torch.no_grad():
            for p in ps:
                p.copy_(torch.as_tensor(th[o:o + p.numel()].reshape(p.shape), dtype=p.dtype)); o += p.numel()
        best = dict(step=s_best, window=W, window_mean_loss=float(roll[i]), last_window_mean_loss=float(roll[-1]))
        print(f"  [{Path(src).name}] keep_best: step {s_best} (running mean {roll[i]:.1f} vs last {roll[-1]:.1f})",
              flush=True)
    nll1 = nll_full(m._model, cache)
    print(f"  [{Path(src).name}] full-data NLL before {nll0}  after {nll1}", flush=True)
    m.z_twin = None                        # a fold checkpoint must roll held-out fish on prior draws
    _refold(m, gains)                      # back to the calibrated gains: only the profile term changed
    for p in m._model.parameters():
        p.requires_grad_(True)
    MX.save_mix(m, str(dst))
    nb_ = torch.load(dst, map_location="cpu", weights_only=False)
    for k in ("heldout_fish", "train_fish", "train_seed", "rollout_seed", "phi_ref", "psi_ref", "cfg_feat_extra"):
        if k in blob:
            nb_[k] = blob[k]
    nb_["profile_finetune"] = dict(cfg={k: v for k, v in cfg.items() if k != "rej"}, trace=trace,
                                   nll_full_before=nll0, nll_full_after=nll1, best=best)
    torch.save(nb_, dst)
    return dict(file=Path(src).name, coef=_flat(st).tolist(), trace=trace, secs=time.time() - t0,
                n_starts=len(starts), nll_before=nll0, nll_after=nll1, best=best)


def finetune_arm(src_dir, arm, dst_dir, new_arm=None, rej=None, steps=25, B=None, M=8, lr=0.1, lam_ratio=1.0,
                 seed=0, jobs=5, v2=False, crn=1, v3=False, warm=0, lr_halve=None, engine="numpy",
                 prior_logk=0.0, pen_k=0.0, v4=False, pen_all=0.0, keep_best=0, v5=None, static=False,
                 reanchor=False):
    """Fine-tune every fold of ``arm`` into ``dst_dir`` as ``new_arm`` (default ``<arm>_pf``; ``_pf2`` with the
    v2 levers, ``_pf3`` with v3, ``_pf4`` with v4, ``_pf5<arm>`` with v5); manifest, report CSV (one row per fold per
    step) and the coefficients per fold. ``static`` adds the static conditions to the profile objective."""
    import pandas as pd
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    new_arm = new_arm or arm + (f"_pf5{v5}" if v5 else "_pf4" if v4 else "_pf3" if v3 else "_pf2" if v2 else "_pf")
    ents = sorted((e for e in json.loads((src_dir / "manifest_model1.json").read_text()) if e["arm"] == arm),
                  key=lambda e: e["fold"])
    cfg = dict(rej=rej, steps=steps, B=B, M=M, lr=lr, lam_ratio=lam_ratio, seed=seed, v2=bool(v2), crn=int(crn),
               v3=bool(v3), warm=int(warm), lr_halve=list(lr_halve or []), engine=str(engine),
               prior_logk=float(prior_logk), pen_k=float(pen_k), v4=bool(v4), pen_all=float(pen_all),
               keep_best=int(keep_best), v5=(str(v5) if v5 else None),
               conds=list(MOVING_CONDS + (STATIC_CONDS if static else ())), reanchor=bool(reanchor))
    jl = [(src_dir / e["file"], dst_dir / e["file"].replace(arm + "_fold", new_arm + "_fold"), cfg) for e in ents]
    with ProcessPoolExecutor(max_workers=min(jobs, len(jl))) as ex:
        reps = list(ex.map(finetune_fold, jl))
    mani = json.loads((dst_dir / "manifest_model1.json").read_text()) if (dst_dir / "manifest_model1.json").exists() else []
    for e, r in zip(ents, reps):
        mani.append(dict(e, file=e["file"].replace(arm + "_fold", new_arm + "_fold"), arm=new_arm,
                         profile_finetune=dict(coef=r["coef"], steps=steps, B=B, M=M, seed=seed, v2=bool(v2),
                                               v3=bool(v3), v4=bool(v4), v5=cfg["v5"], conds=cfg["conds"],
                                               warm=int(warm), reanchor=bool(reanchor), crn=int(crn),
                                               lr=lr, lr_halve=list(lr_halve or []), engine=str(engine),
                                               lam_ratio=float(lam_ratio), prior_logk=float(prior_logk),
                                               pen_k=float(pen_k), pen_all=float(pen_all), keep_best=int(keep_best),
                                               best=r.get("best"),
                                               nll_before=r["nll_before"], nll_after=r["nll_after"])))
    (dst_dir / "manifest_model1.json").write_text(json.dumps(mani, indent=1))
    rows = [dict(fold=e["fold"], **t) for e, r in zip(ents, reps) for t in r["trace"]]
    pd.DataFrame(rows).to_csv(dst_dir / f"profile_finetune_{new_arm}.csv", index=False)
    write_likA(dst_dir, new_arm)           # the per-fish held-out table merge_ckpt_dirs requires
    return reps


def write_likA(dst_dir, new_arm):
    """Held-out Metric A / C per fish for the ``new_arm`` checkpoints in ``dst_dir`` -> ``likA_model1.csv`` (appended),
    the per-fish table every ladder folder carries and ``merge_ckpt_dirs`` requires. Teacher-forced, no rollouts."""
    import pandas as pd
    from twin import fit_MLP_mix as MX
    from twin import model_figures as MF
    from twin.training import metricA
    dst_dir = Path(dst_dir)
    recs = MF.records()
    rows = []
    for e in json.loads((dst_dir / "manifest_model1.json").read_text()):
        if e["arm"] != new_arm:
            continue
        m = MX.load_mix(str(dst_dir / e["file"]))
        for f in e.get("heldout", []):
            rows.append(dict(arm=new_arm, rep=e.get("rep"), order=e.get("order"), latent=e.get("latent"),
                             fold=e["fold"], fish=int(f), **metricA(m, recs[int(f)])))
    lik = pd.DataFrame(rows)
    if (dst_dir / "likA_model1.csv").exists():
        lik = pd.concat([pd.read_csv(dst_dir / "likA_model1.csv"), lik], ignore_index=True)
    lik.to_csv(dst_dir / "likA_model1.csv", index=False)
    return lik
