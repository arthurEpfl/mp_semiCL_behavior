"""BATCHED block-start rollouts for the profile fine-tune -- the numpy generator's semantics, many trajectories at once.

WHY. The profile fine-tune (scripts/profile_finetune.py) rolls out every training block x M from its real onset
at every step: ~10-16 k trajectories of ~60-160 bouts. ``generate_mix_joint_loop`` does them one at a time in a
per-bout Python loop, which made one step of Model 34's fine-tune cost ~40 min per fold. Here all trajectories of
a chunk advance TOGETHER, bout by bout, so the per-bout Python cost is paid once per chunk, not once per
trajectory.

THE ONE THING THAT MUST BE TRUE: it is the SAME object as ``generate_mix_joint_loop`` on the block-start path
(record-indexed stimulus, ``_force_P`` forcing, twin z). Two design choices make that checkable rather than hoped:

  * FEATURES come from the generator's own vectorised feature functions (``M.shoal_channels`` -- whose one-group path
    is vectorised and verified equal to its loop --, ``M.shoal_channels``, ``M.static_channels``,
    ``MB.border_channels``, ``MB.align_feature``), called on all rows at once. Verified exact against
    ``MB.build_feature_matrix`` under full forcing (``verify_features``).
  * HEADS go through the model's OWN torch modules (``net._trunk``), so the skip term, the profile term (v1/v2/v3),
    the per-fish loadings and the step lever are the trained object, not a re-port.

What is NOT bit-identical: the random stream (vectorised numpy draws, not the per-bout rng.choice sequence), so the
two agree in DISTRIBUTION, checked by ``verify_distribution`` on the objective's own curves.

Scope, and it refuses the rest: the record-indexed block-start path only (no stimulus provider, no time-driven
programme); no drift / dt_latent / state / vigor / indiv / centre_bias / home pull / kin_innov / dt_snap / onset /
static-clock / static-bump / K_stim > 1 / adapt. Those raise, so a config outside the scope cannot silently run.
"""
from __future__ import annotations


import numpy as np

from twin import stimulus as M                      # noqa: E402
from twin import features as MB              # noqa: E402

_UNSUPPORTED = ("dt_latent", "vigor", "indiv", "kin_innov", "kin_feedback", "dt_snap", "stim_onset",
                "stim_static_clock", "stim_static_bump", "stim_lags", "rad_gate", "use_adapt")
# THE DRIFT (report Model 11, fine-tunes on Model 17) IS SUPPORTED, as the generator implements it: one AR(1) state
# per trajectory, stationary N(0, 1) at the start, advanced ONCE PER BOUT (forced bouts included, never per reject
# draw); it tilts the forward component's logit by turn * d and shifts the step / interval log-locations by
# dist * g_dist * d - dist_c and dt * g_dt * d - dt_c (inside the truncation). Only its containment-bias loading
# (`bord`) is out of scope.


def _check_scope(model, cfg):
    bad = [k for k in _UNSUPPORTED if cfg.get(k)]
    if cfg.get("drift") and abs(float((cfg.get("drift") or {}).get("bord", 0.0))) > 0:
        bad.append("drift bord loading")
    if cfg.get("centre_bias"):
        bad.append("centre_bias")
    if int(cfg.get("K_stim", 1)) > 1:
        bad.append("K_stim>1")
    if (cfg.get("wall_reject") or {}).get("pull") is not None:
        bad.append("wall_reject.pull")
    if str(cfg.get("wall_mode", "clip")) != "clip":
        bad.append(f"wall_mode={cfg.get('wall_mode')}")
    net = model._model
    if int(getattr(net, "state", 0) or 0) or getattr(net, "dt_latent", None) or getattr(net, "use_body", False) \
            or getattr(net, "border_gain", False):
        bad.append("net state / dt_latent / body path / border gain")
    if not model.closed_loop or not model.guardrail or not model.stim_loop:
        bad.append("open-loop / no guardrail")
    from twin import joint as JT
    if tuple(JT.parse_order(getattr(net, "order", None))) != tuple(JT.parse_order("adt")):
        bad.append(f"head order {getattr(net, 'order', None)}")
    if bad:
        raise NotImplementedError(f"rollout_batch does not implement: {bad}")


# ----------------------------------------------------------------------------- per-trajectory static arrays
_CLOCK_CACHE = {}


def _clock_track(rec, cfg):
    """(n, n_taus) the block clock the generator computes on the record-indexed path (``_clock_of`` with
    ``_clock_src`` = the record's frames): reset at the first bout of a new (count, moving) condition, 0 on
    no-dot bouts. A pure function of the record -- vectorised, and CACHED per record object (the fine-tune rolls
    the same block-start records out M times at every step; the per-bout loop was half the rollout time)."""
    mode = str(cfg.get("clock_mode", "") or "").lower()
    taus = tuple(MB.clock_taus(cfg)) if mode in ("linear", "exp") else ()
    key = (id(rec), mode, taus)
    hit = _CLOCK_CACHE.get(key)
    if hit is not None and hit[0] is rec:
        return hit[1]
    if mode not in ("linear", "exp"):
        out = np.zeros((rec.n, 0), np.float32)
    else:
        n = rec.n
        fo = np.asarray(rec.frame_onset, float) / MB._FPS
        nd = np.nan_to_num(np.asarray(getattr(rec, "n_dots", np.zeros(n)), float))
        cond = nd * 10.0 + np.nan_to_num(np.asarray(rec.moving, float))
        new = np.r_[True, cond[1:] != cond[:-1]]
        start = np.maximum.accumulate(np.where(new, np.arange(n), 0))
        tb = np.maximum(fo - fo[start], 0.0)
        if mode == "linear":
            out = np.minimum(tb / MB.CLOCK_BLOCK_S, 1.0)[:, None]
        else:
            out = np.exp(-tb[:, None] / np.asarray(taus, float)[None, :])
        out = np.where((nd > 0.0)[:, None], out, 0.0).astype(np.float32)
    _CLOCK_CACHE[key] = (rec, out)
    return out




class _Batch:
    """Padded (B, T, ...) arrays for a list of (record slice, z) trajectories."""

    def __init__(self, recs, zs, cfg):
        self.B = B = len(recs)
        self.n = np.array([int(r.n) for r in recs])
        self.T = T = int(self.n.max())
        f = lambda: np.zeros((B, T))  # noqa: E731
        self.dth, self.dist, self.dt, self.x, self.y, self.h = f(), f(), f(), f(), f(), f()
        self.dcx = np.full((B, T), np.nan); self.dcy = np.full((B, T), np.nan)
        self.mv, self.nd = f(), f()
        ck = _clock_track(recs[0], cfg)
        self.clock = np.zeros((B, T, ck.shape[1]), np.float32)
        self.force = np.array([int(getattr(r, "_force_P", 0) or 0) for r in recs])
        # warm-up re-anchoring (the generator's `_reanchor_at`): -1 = none
        self.ra = np.array([-1 if getattr(r, "_reanchor_at", None) is None else int(r._reanchor_at) for r in recs])
        self.rp_off = np.array([int(getattr(r, "_rp_off", 0) or 0) for r in recs])
        dm, ds, tm, ts = cfg["kin_scales"]
        self.lev0 = np.array([float(getattr(r, "_kdt_lev0", tm)) for r in recs])
        self.cx = np.array([r.dish_cx for r in recs], float); self.cy = np.array([r.dish_cy for r in recs], float)
        self.R = np.array([r.dish_R for r in recs], float)
        self.z = np.asarray(zs, np.float32)
        # dot-jump phase (MB._jump_block, Model 41): the record-indexed path reads each bout's own frame, as the clock
        self.jump = np.zeros((B, T, 2 if cfg.get("stim_jump") else 0), np.float32)
        for i, r in enumerate(recs):
            k = int(r.n)
            self.dth[i, :k] = r.dtheta; self.dist[i, :k] = r.distance_px; self.dt[i, :k] = r.dt_s
            self.x[i, :k] = r.x_px; self.y[i, :k] = r.y_px
            self.h[i, :k] = (MB._travel_ref(r) if cfg.get("psi_ref") == "travel" else np.asarray(r.f_ori_rad, float))
            self.dcx[i, :k] = np.asarray(getattr(r, "dot_cx", np.full(k, np.nan)), float)
            self.dcy[i, :k] = np.asarray(getattr(r, "dot_cy", np.full(k, np.nan)), float)
            self.mv[i, :k] = np.asarray(r.moving, float)
            self.nd[i, :k] = np.asarray(getattr(r, "n_dots", np.zeros(k)), float)
            self.clock[i, :k] = _clock_track(r, cfg)
            if self.jump.shape[2]:
                self.jump[i, :k] = MB._jump_block(r, cfg)


# ----------------------------------------------------------------------------- the rollout
def rollout(model, recs, zs, rng, keep_X_from=None, sample_forced=False):
    """Advance every trajectory of ``recs`` (sliced block-start records with ``_force_P`` etc.) through its full
    length, all at once. ``zs`` (B, m) = each trajectory's latent (twin z for training fish).
    Returns dict of (B, T) arrays -- x_px, y_px (pose at the START of each bout, as the generator's ``x_px``),
    dtheta, dist_base, dt, and X (B, T, D) feature rows -- plus ``n`` (per-trajectory length).
    ``keep_X_from`` (B,) optional: rows before it are not stored (memory; the fine-tune scores t >= P only).
    ``sample_forced``: draw even on forced bouts (only to mimic the generator's RNG consumption; never needed)."""
    import torch
    cfg = model._cfg_feat
    _check_scope(model, cfg)
    net = model._model
    b = _Batch(recs, zs, cfg)
    B, T = b.B, b.T
    K = int(cfg["K"]); Kk = int(cfg.get("K_kin", 3))
    dth_scale = float(cfg["dth_scale"])
    dm, ds, tm, ts = cfg["kin_scales"]
    dist_hi, dt_hi = cfg.get("kin_clip", (400.0, 600.0))
    log_tr = float(np.log(cfg["dt_trunc"])) if cfg.get("dt_trunc") else None
    if log_tr is not None:
        from scipy.special import ndtr, ndtri
        dt_hi = float(cfg["dt_trunc"])
    kin_dt = bool(cfg.get("kin_dt_lags", True)); kin_dt_c = cfg.get("kin_dt_lags", True) == "centred"
    kdt_a = 1.0 / max(float(cfg.get("kin_dt_ema", MB.KIN_DT_EMA)), 1.0)
    mode = cfg["stim_mode"]
    skw = M.shoal_kw(mode)
    if mode not in M.SHOAL_MODES:
        raise NotImplementedError(f"stim_mode {mode!r}")
    use_border = bool(cfg.get("use_border")); bmode = cfg.get("border_mode", "raw")
    lv = str(cfg.get("lag_valid", "") or "").lower()
    rc_tau = float(cfg.get("run_clock_tau") or 0.0)
    al_on, st_on = bool(cfg.get("stim_align")), bool(cfg.get("stim_static"))
    if cfg.get("stim_dmove"):
        raise NotImplementedError("stim_dmove (Model 51): numpy generator only")
    wr = cfg.get("wall_reject")
    rd0 = rd1 = None; rmax = 1
    R_g = b.R.copy()
    if wr:
        rd0 = float(wr.get("d0", MB.WALL_REJECT_DEFAULT["d0"])); rd1 = float(wr.get("d1", MB.WALL_REJECT_DEFAULT["d1"]))
        rmax = max(int(wr.get("max_draws", MB.WALL_REJECT_DEFAULT["max_draws"])), 1)
        R_g = b.R * (1.0 - rd0 / (0.5 * MB.DISH_MM))
    eps = float(getattr(model, "eps", 1e-3))
    any_ra = bool((b.ra >= 0).any())
    D = int(net.body[0].in_features)
    Z = torch.as_tensor(b.z)
    zeros_t = torch.zeros(B)
    dr = cfg.get("drift") or {}
    dstate = None
    if dr:
        dr_turn = float(dr.get("turn", 0.0))
        dr_dist = float(dr.get("dist", 0.0)) * float(dr.get("g_dist", 1.0))
        dr_dt = float(dr.get("dt", 0.0)) * float(dr.get("g_dt", 1.0))
        dr_dist_c, dr_dt_c = float(dr.get("dist_c", 0.0)), float(dr.get("dt_c", 0.0))
        dr_phi = float(dr["phi"])
        dstate = rng.normal(size=B)                          # stationary N(0, 1), as the generator's chain

    # state
    x, y, h = b.x[:, 0].copy(), b.y[:, 0].copy(), b.h[:, 0].copy()
    if wr:
        u0 = np.hypot(x - b.cx, y - b.cy) / R_g
        o = u0 > 1.0
        x[o] = b.cx[o] + (x[o] - b.cx[o]) / u0[o] * (1.0 - 1e-9); y[o] = b.cy[o] + (y[o] - b.cy[o]) / u0[o] * (1.0 - 1e-9)
    dth_h = np.zeros((B, K)); ld_h = np.zeros((B, Kk)); lt_h = np.zeros((B, Kk))
    lev = b.lev0.copy()
    out = {k: np.zeros((B, T)) for k in ("x_px", "y_px", "dtheta", "dist_base", "dt")}
    if dstate is not None:
        out["dturn"] = np.zeros((B, T), np.float32)          # the forward-logit tilt each bout was drawn with
    Xk = np.zeros((B, T, D), np.float32)
    keep = np.zeros(B, int) if keep_X_from is None else np.asarray(keep_X_from, int)

    # The heads in STAGES (net._trunk_turn / _trunk_dist / _trunk_dt, whose composition IS net._trunk): the body, the
    # skip / profile term and the angle parameters once per bout; the step and interval heads on their conditioners.
    k_d, k_t = int(net.n_dist), int(net.n_dt)
    sp_ = torch.nn.functional.softplus

    def turn_stage(X_t, idx):
        with torch.no_grad():
            b_, zc_, out_, xx_ = net._trunk_turn(torch.as_tensor(X_t), Z[torch.as_tensor(idx)])
            return (b_, zc_, xx_), net._angle_params(out_)

    def dist_par(st_, a_z):
        with torch.no_grad():
            do_ = net._trunk_dist(*st_, torch.as_tensor(a_z, dtype=torch.float32))
            return (torch.log_softmax(do_[:, :k_d], 1), do_[:, k_d:2 * k_d], sp_(do_[:, 2 * k_d:3 * k_d]) + M.EPS)

    def dt_par(st_, a_z, ld_c):
        with torch.no_grad():
            to_ = net._trunk_dt(*st_, torch.as_tensor(a_z, dtype=torch.float32), torch.as_tensor(ld_c, dtype=torch.float32))
            return (torch.log_softmax(to_[:, :k_t], 1), to_[:, k_t:2 * k_t], sp_(to_[:, 2 * k_t:3 * k_t]) + M.EPS)

    def draw_angle(ang, n_, tilt=None):
        logw = ang[0].numpy().astype(float)
        if tilt is not None:                             # the drift's forward-weight tilt, then renormalise
            logw = logw.copy(); logw[:, 0] = logw[:, 0] + tilt
            logw = logw - np.logaddexp.reduce(logw, axis=1, keepdims=True)
        p = np.exp(logw)
        c = (rng.random(n_)[:, None] > np.cumsum(p, 1)[:, :-1]).sum(1)       # inverse-CDF component draw
        sF = ang[1].numpy().astype(float)
        a = np.stack([ang[i].numpy() for i in (2, 4, 6, 8)], 1).astype(float)
        th = np.stack([ang[i].numpy() for i in (3, 5, 7, 9)], 1).astype(float)
        v = np.empty(n_)
        m0 = c == 0
        v[m0] = rng.normal(0.0, sF[m0])
        for j, sgn in ((1, -1.0), (2, -1.0), (3, 1.0), (4, 1.0)):
            mj = c == j
            if mj.any():
                v[mj] = sgn * rng.gamma(a[mj, j - 1], th[mj, j - 1])
        return np.clip(v, -180.0, 180.0)

    def draw_lognorm(par, n_, hi, trunc, shift=None):
        w = np.exp(par[0].numpy().astype(float)); mu = par[1].numpy().astype(float); sg = par[2].numpy().astype(float)
        if shift is not None:                            # the drift's log-location shift (whole mixture)
            mu = mu + np.asarray(shift, float)[:, None]
        if trunc is None:
            j = (rng.random(n_)[:, None] > np.cumsum(w, 1)[:, :-1]).sum(1)
            ld = rng.normal(mu[np.arange(n_), j], sg[np.arange(n_), j])
        else:
            F = ndtr((trunc - mu) / sg)
            p = w * F; p = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
            j = (rng.random(n_)[:, None] > np.cumsum(p, 1)[:, :-1]).sum(1)
            ii = np.arange(n_)
            ld = mu[ii, j] + sg[ii, j] * ndtri(np.clip(rng.random(n_) * F[ii, j], 1e-12, 1.0 - 1e-12))
        return np.clip(np.exp(ld), 1e-3, hi)

    def emit(st_, ang, sh=None):
        """One draw of the chained triple for the rows of the turn stage ``st_`` (ang evaluated on them); ``sh`` =
        the drift's (forward tilt, step shift, interval shift) for those rows, or None."""
        n_ = st_[0].shape[0]
        tl, s_d, s_t = sh if sh is not None else (None, None, None)
        d_th = draw_angle(ang, n_, tl)
        az = np.abs(d_th) / dth_scale
        d_di = draw_lognorm(dist_par(st_, az), n_, dist_hi, None, s_d)
        ldc = (np.log(np.maximum(d_di, 1e-6)) - dm) / ds
        d_dt = draw_lognorm(dt_par(st_, az, ldc), n_, dt_hi, log_tr, s_t)
        return d_th, d_di, d_dt

    for t in range(T):
        live = t < b.n
        out["x_px"][:, t], out["y_px"][:, t] = x, y
        if any_ra:                               # block onset of a warm-up start: latch the simulated pose (generator's rule)
            ra_t = np.flatnonzero(b.ra == t)
            if ra_t.size:
                b.dcx[ra_t, t:], b.dcy[ra_t, t:] = MB.reanchor_track(
                    b.dcx[ra_t, t:], b.dcy[ra_t, t:], (b.x[ra_t, t], b.y[ra_t, t], b.h[ra_t, t]), (x[ra_t], y[ra_t], h[ra_t]))
        # ---- feature row, from the generated pose, with the generator's own feature functions
        blocks = [dth_h / dth_scale]
        dcx, dcy = b.dcx[:, t], b.dcy[:, t]
        fin = np.isfinite(dcx) & np.isfinite(dcy)
        phi_g = np.where(fin, MB._wrap_pi(np.arctan2(np.where(fin, dcy, 0) - y, np.where(fin, dcx, 0) - x) - h), np.nan)
        r_t = np.where(fin, np.hypot(np.where(fin, dcx, 0) - x, np.where(fin, dcy, 0) - y), np.nan)
        blocks.append(M.shoal_channels(phi_g, r_t, b.nd[:, t], b.mv[:, t], **skw))
        if use_border:
            rr = np.hypot(x - b.cx, y - b.cy); psi = MB._wrap_pi(np.arctan2(b.cy - y, b.cx - x) - h)
            blocks.append(MB.border_channels((b.R - rr) / b.R, psi, bmode))
        kin = np.zeros((B, 2 * Kk if kin_dt else Kk))
        for k in range(1, Kk + 1):
            if t - k >= 0:
                kin[:, k - 1] = (ld_h[:, k - 1] - dm) / ds
                if kin_dt_c:
                    kin[:, Kk + k - 1] = (lt_h[:, k - 1] - lev) / ts
                elif kin_dt:
                    kin[:, Kk + k - 1] = (lt_h[:, k - 1] - tm) / ts
        blocks.append(kin)
        if lv == "scalar":
            blocks.append((np.minimum(t + b.rp_off, K) / max(K, 1)).reshape(B, 1))
        elif lv == "mask":
            blocks.append(np.repeat((t >= np.arange(1, K + 1))[None, :], B, 0).astype(float))
        if b.clock.shape[2]:
            blocks.append(b.clock[:, t])
        if rc_tau > 0:
            blocks.append(np.exp(-(t + b.rp_off) / rc_tau).reshape(B, 1))
        if al_on:
            blocks.append(MB.align_feature(phi_g, r_t, b.mv[:, t], b.nd[:, t]).reshape(B, 1))
        if st_on:
            blocks.append(M.static_channels(phi_g, r_t, b.nd[:, t], b.mv[:, t]))
        if b.jump.shape[2]:                       # the dot-jump phase, LAST as in the builder
            blocks.append(b.jump[:, t])
        X_all = np.concatenate(blocks, axis=1).astype(np.float32)
        assert X_all.shape[1] == D, (X_all.shape, D)
        st_ = t >= keep
        Xk[st_, t] = X_all[st_]

        # ---- emission on the free rows (forced rows replay the real bout, as the generator does)
        forced = t < b.force
        free = np.flatnonzero(live & (~forced | sample_forced))
        d_th = b.dth[:, t].copy(); d_di = b.dist[:, t].copy(); d_dt = b.dt[:, t].copy()
        if free.size:
            stg, ang = turn_stage(X_all[free], free)
            sh = None
            if dstate is not None:
                dsf = dstate[free]
                sh = (dr_turn * dsf if dr_turn else None, dr_dist * dsf - dr_dist_c, dr_dt * dsf - dr_dt_c)
            th1, di1, dt1 = emit(stg, ang, sh)
            if rd0 is not None:
                best = [th1.copy(), di1.copy(), dt1.copy()]; best_u = np.full(free.size, np.inf)
                pending = np.ones(free.size, bool)
                cur = [th1, di1, dt1]
                for k in range(rmax):
                    idx = np.flatnonzero(pending)
                    hh = h[free[idx]] + np.radians(cur[0][idx])
                    ul = np.hypot(x[free[idx]] + cur[1][idx] * np.cos(hh) - b.cx[free[idx]],
                                  y[free[idx]] + cur[1][idx] * np.sin(hh) - b.cy[free[idx]]) / b.R[free[idx]]
                    dmm = 0.5 * MB.DISH_MM * (1.0 - ul)
                    acc_p = np.clip((dmm - rd0) / (rd1 - rd0), 0.0, 1.0) if rd1 > rd0 else (dmm >= rd0).astype(float)
                    ok = (acc_p >= 1.0) | (rng.random(idx.size) < acc_p)
                    bad = idx[~ok]
                    better = ul[~ok] < best_u[bad]
                    for q in range(3):
                        best[q][bad[better]] = cur[q][bad[better]]
                    best_u[bad[better]] = ul[~ok][better]
                    for q in range(3):                   # accepted rows keep this draw
                        best[q][idx[ok]] = cur[q][idx[ok]]
                    pending[idx[ok]] = False
                    if not pending.any() or k == rmax - 1:
                        break
                    idx2 = np.flatnonzero(pending)       # a fresh draw of the whole chain for the rejected rows
                    it2 = torch.as_tensor(idx2)
                    sh2 = None if sh is None else tuple(None if q_ is None else q_[idx2] for q_ in sh)
                    n_th, n_di, n_dt = emit(tuple(s_[it2] for s_ in stg), tuple(a_[it2] for a_ in ang), sh2)
                    cur[0][idx2], cur[1][idx2], cur[2][idx2] = n_th, n_di, n_dt
                th1, di1, dt1 = best                     # exhausted rows: the least-bad draw (best_u)
            d_th[free], d_di[free], d_dt[free] = th1, di1, dt1
        out["dtheta"][:, t] = d_th; out["dist_base"][:, t] = d_di; out["dt"][:, t] = d_dt
        if dstate is not None:                   # record this bout's tilt, THEN advance the chain once per bout
            out["dturn"][:, t] = dr_turn * dstate
            dstate = dr_phi * dstate + np.sqrt(max(1.0 - dr_phi * dr_phi, 0.0)) * rng.normal(size=B)

        # ---- history, level, pose
        dth_h = np.concatenate([d_th[:, None], dth_h[:, :K - 1]], 1)
        ldv = np.log(np.maximum(d_di, 1e-6)); ltv = np.log(np.maximum(d_dt, 1e-6))
        ld_h = np.concatenate([ldv[:, None], ld_h[:, :Kk - 1]], 1)
        lt_h = np.concatenate([ltv[:, None], lt_h[:, :Kk - 1]], 1)
        if kin_dt_c:
            lev = lev + kdt_a * (ltv - lev)
        h = h + np.radians(d_th)
        dx, dy = x - b.cx, y - b.cy
        ux, uy = np.cos(h), np.sin(h)
        dd = dx * ux + dy * uy
        ray = -dd + np.sqrt(np.maximum(dd * dd - (dx * dx + dy * dy - R_g * R_g), 0.0))
        L = np.minimum(d_di, np.maximum(ray - eps, 0.0))
        x = x + L * ux; y = y + L * uy
        fz = forced & (t + 1 < b.n)
        if fz.any():
            x[fz], y[fz], h[fz] = b.x[fz, t + 1], b.y[fz, t + 1], b.h[fz, t + 1]
    out["X"] = Xk
    out["n"] = b.n
    if any_ra:                                   # the (re-anchored) dots each trajectory actually saw
        out["dot_cx"], out["dot_cy"] = b.dcx, b.dcy
    return out


# ----------------------------------------------------------------------------- the fine-tune's entry point
def rollout_starts(model, batch, M_, seed, step, zmap, crn=1, chunk=2048):
    """``profile_finetune.rollout_starts``, batched: the same trajectories (M per block start, odd ones from the
    warm-up variant when it exists), the same per-rollout dicts, in the same order."""
    from twin.profile_finetune import PX_TO_MM, _binned
    from twin import joint as JT
    cf = model._cfg_feat
    jobs = []
    for s in batch:
        for m in range(M_):
            v = int("rs_w" in s and m % 2 == 1)
            jobs.append((s, m, v))
    res = []
    for c0 in range(0, len(jobs), int(chunk)):
        part = jobs[c0:c0 + int(chunk)]
        recs = [s["rs_w"] if v else s["rs"] for s, m, v in part]
        zs = np.stack([np.asarray(zmap[s["fish"]], np.float32) for s, m, v in part])
        rng = np.random.default_rng([int(seed), int(step) // max(int(crn), 1), c0 // int(chunk), 7919])
        g = rollout(model, recs, zs, rng, keep_X_from=np.array([s["P"] for s, m, v in part]))
        for i, (s, m, v) in enumerate(part):
            P, n = s["P"], int(g["n"][i])
            gx, gy = g["x_px"][i, P:n], g["y_px"][i, P:n]
            dcx, dcy, mask = s["dcx"], s["dcy"], s["mask"]
            if getattr(recs[i], "_reanchor_at", None) is not None:
                # a re-anchored warm-up start is scored against the dots IT saw, masked by where THEY sit
                r0 = recs[i]
                dcx, dcy = g["dot_cx"][i, P:n], g["dot_cy"][i, P:n]
                mask = np.hypot(dcx - r0.dish_cx, dcy - r0.dish_cy) <= r0.dish_R
            d = np.hypot(gx - dcx, gy - dcy) / PX_TO_MM
            dth = g["dtheta"][i, P:n]; dist = g["dist_base"][i, P:n]
            az, _ = JT._cond_cols(dth, dist, cf["dth_scale"], cf["kin_scales"])
            res.append(dict(s=s, m=m, v=v, x=_binned(s["tb"], d, mask), X=g["X"][i, P:n], az=az,
                            dth=dth, dist=dist, dturn=(g["dturn"][i, P:n] if "dturn" in g else None)))
    return res
