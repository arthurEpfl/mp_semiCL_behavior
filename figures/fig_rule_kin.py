"""Check figure for the Discussion's claim that the approach is carried by the turn, not by how far or how often the
fish swims: Figure 5.3 (fig_rule --onset) with two more panels, displacement and interval, all against the time since
the dots appeared at the same matched geometry (moving dots, |bearing| 45-135 deg, 0-60 mm, 3 x 6 cells, every time
bin reweighted to one common cell mix). Real fish: the observed bout (geometric mean over a fish's bouts, mean over
fish, 95 % bootstrap band over fish). M6 and F (both seeds): the teacher-forced expectation on the same held-out bouts
(population latent; model_figures.conditional_kin for step and interval, given the real turn and step), mean over
fish with a 95 % bootstrap interval over fish -- narrower than the real band by construction (an expectation, no
single-bout scatter). Writes figures/out/fig_rule_3panel.pdf.

    uv run python -m figures.fig_rule_kin            # build the kinematics cache if missing, draw
"""

from figures.style import *  # noqa: F401,F403,E402

import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from figures import fig_rule as FR  # noqa: E402
from twin import missing_pieces as P  # noqa: E402


N_DRAW = P.N_DRAW


def _kin_draws(MF, model, rec, rng, n_draw):
    """n_draw single (log step, log interval) per bout DRAWN from the model's teacher-forced step and interval
    mixtures (the same conditional as MF.conditional_kin: given the real feature row, the real turn and, for the
    interval, the real step; population latent). Log-normal mixtures in log space; the interval truncated at dt_trunc."""
    from scipy.stats import norm, truncnorm
    import torch
    cf = model._cfg_feat
    X = MF.MB.build_feature_matrix(rec, cf)
    az, lc, lt = MF.JT._cond_cols(rec.dtheta, rec.distance_px, cf["dth_scale"], cf["kin_scales"], y_dt=rec.dt_s)
    with torch.no_grad(), MF.JT.repulsion_side(model, rec):
        _, dpar, tpar = model._model.joint_forward(torch.tensor(X), torch.tensor(az), torch.tensor(lc), torch.tensor(lt))

    def draw(par, hi):
        lw, mu, sg = [q.numpy().astype(float) for q in par]
        w = np.exp(lw)
        if hi:
            b = (np.log(float(hi)) - mu) / sg
            w = w * norm.cdf(b)                    # truncation renormalises each component's weight
        w = w / w.sum(1, keepdims=True)
        cw = np.cumsum(w, 1)
        k = (rng.random((len(w), n_draw))[:, :, None] > cw[:, None, :-1]).sum(-1)
        m_, s_ = np.take_along_axis(mu, k, 1), np.take_along_axis(sg, k, 1)
        if hi:
            ub = (np.log(float(hi)) - m_) / s_
            return truncnorm.rvs(-np.inf, ub, loc=m_, scale=s_, random_state=rng)
        return m_ + s_ * rng.standard_normal(m_.shape)
    return draw(dpar, None), draw(tpar, cf.get("dt_trunc"))


def _kin_job(job):
    lab, d, file, fids = job
    P._threads1()
    from twin import model_figures as MF
    m = P.load_model(d, file)
    recs = P.records()
    parts = []
    for f in fids:
        rec = recs[f]
        e_ld, e_lt = MF.conditional_kin(m, rec)
        rng = np.random.default_rng([13, list(P.HEAD).index(lab), int(f)])
        dl, tl = _kin_draws(MF, m, rec, rng, N_DRAW)
        phi = np.asarray(getattr(rec, "stim_angle_travel", rec.stim_angle), float)
        fo = np.asarray(rec.frame_onset, float); pos = fo % 3300
        t_blk = np.where(pos >= 1500, pos - 1500, pos) / 30.0
        dist = np.asarray(rec.distance_px, float); dt = np.asarray(rec.dt_s, float)
        parts.append(pd.DataFrame(dict(
            model=lab, fish=f, phi=phi, rng_mm=np.asarray(rec.dot_dist_px, float) / P.PX_MM,
            n=np.nan_to_num(np.asarray(rec.n_dots, float)).astype(int), moving=np.asarray(rec.moving, float) > 0,
            t_blk=t_blk, ld_real=np.log(np.clip(dist, 1e-6, None) / P.PX_MM), lt_real=np.log(np.clip(dt, 1e-6, None)),
            ld_model=np.asarray(e_ld, float) - np.log(P.PX_MM), lt_model=np.asarray(e_lt, float),
            ok=np.isfinite(dist) & (dist > 0) & np.isfinite(dt) & (dt > 0))))
        for j in range(N_DRAW):
            parts[-1][f"ld_d{j}"] = dl[:, j] - np.log(P.PX_MM)
            parts[-1][f"lt_d{j}"] = tl[:, j]
    return pd.concat(parts, ignore_index=True)


def kin_data(jobs=6):
    if P.have("laws_kin_draw"):
        return P.load("laws_kin_draw")
    jl = [(lab, d, file, ho) for lab, (d, arm, key) in P.HEAD.items() for file, ho in P.folds_of(d, arm)]
    with ProcessPoolExecutor(jobs) as ex:
        tab = pd.concat(list(ex.map(_kin_job, jl)), ignore_index=True)
    P.save("laws_kin_draw", tab)
    return tab


def main():
    xc = np.arange(len(FR.T_BINS), dtype=float)
    offs = {"M6 s0": -0.21, "M6 s101": -0.07, "F s0": 0.07, "F s101": 0.21}
    turn, _ = FR.onset_data()                                            # Figure 5.3's own numbers (turn toward)
    kt = FR._geo(kin_data())
    kt = kt[kt.ok & (kt.n > 0) & np.isfinite(kt.phi) & np.isfinite(kt.rng_mm)]
    w = kt[kt.model == "M6 s0"].groupby("cell").size(); w = w / w.sum()
    for a, lab in (("ld", "log step"), ("lt", "log interval")):        # the draws must look like real single bouts
        q = kt[kt.model == "F s0"]
        print(f"  {lab}: mean draw - expected {np.mean(q[a + '_d0'] - q[a + '_model']):+.3f}, SD one draw "
              f"{q[a + '_d0'].std():.3f} vs real {q[a + '_real'].std():.3f}")
    kin = {}
    for col, real_col, model_col in (("step", "ld_real", "ld_model"), ("interval", "lt_real", "lt_model")):
        kin[col] = {"real": FR._std(kt[kt.model == "M6 s0"], real_col, w)}
        pre = real_col[:2]
        for lab in P.HEAD:
            tt = kt[kt.model == lab]
            kin[col][lab] = FR._std(tt, model_col, w)
            kin[col][lab + "|draws"] = [FR._std(tt, f"{pre}_d{j}", w) for j in range(N_DRAW)]

    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.45))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.27, wspace=0.42)
    panels = [(axs[0], turn, False, "turn toward the dots (deg)", "turn"),
              (axs[1], kin["step"], True, "displacement (mm)", "step"),
              (axs[2], kin["interval"], True, "interval (s)", "interval")]
    nums = {}
    for ax, D, logged, ylab, key in panels:
        tr = (lambda v: np.exp(v)) if logged else (lambda v: v)
        r = D["real"]
        m = r.mean(axis=0).to_numpy()
        ci = np.array([P._boot_ci(r[c].to_numpy()) for c in r.columns])
        ax.fill_between(xc, tr(ci[:, 0]), tr(ci[:, 1]), color=FAINT, alpha=0.7, lw=0, zorder=1)
        ax.plot(xc, tr(m), "-o", color=REAL_C, ms=2.8, lw=1.3, zorder=3, label="real fish")
        nums[key] = {"real": tr(m).round(3).tolist()}
        for src, name, c, seed in FR.SOURCES[1:]:
            e = D[src].mean(axis=0).to_numpy()
            reps = D[src + "|draws"]                     # interval built as the real one, on single drawn bouts
            off = np.array([[(lambda x: np.subtract(P._boot_ci(x), np.nanmean(x)))(rp[cc].to_numpy())
                             for cc in rp.columns] for rp in reps]).mean(0)
            eci = e[:, None] + off
            ax.errorbar(xc + offs[src], tr(e), yerr=[tr(e) - tr(eci[:, 0]), tr(eci[:, 1]) - tr(e)], fmt="o", ls="-",
                        lw=0.8, color=c, ms=2.6, mfc=c if seed == 0 else "white", mec=c, mew=0.7, elinewidth=0.6,
                        capsize=0, zorder=4, label=f"{name} s{seed}")
            nums[key][src] = tr(e).round(3).tolist()
        ax.axvspan(-0.5, 1.5, color="#f2f2f2", lw=0, zorder=0)
        ax.set_xlim(-0.5, len(FR.T_BINS) - 0.5)
        ax.set_xticks(xc); ax.set_xticklabels([f"{a}–{b}" for a, b in FR.T_BINS], fontsize=6, rotation=45)
        ax.set_xlabel("time since the dots appeared (s)", fontsize=7)
        ax.set_ylabel(ylab, fontsize=7.5); ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)
    axs[0].set_ylim(0, None)
    axs[0].legend(loc="upper right", fontsize=5.5, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)
    for ax, lt in zip(axs, "abc"):
        panel(ax, lt)
    save(fig, "fig_rule_3panel")
    out = FR.RF / "fig_rule_3panel_numbers.json"
    out.write_text(json.dumps(dict(nums, t_bins=FR.T_BINS), indent=1))
    for key in ("step", "interval"):
        print(f"\n{key} by time bin {[f'{a}-{b}' for a, b in FR.T_BINS]}")
        for src, v in nums[key].items():
            print(f"  {src:8s} " + " ".join(f"{x:6.3f}" for x in v))


if __name__ == "__main__":
    main()
