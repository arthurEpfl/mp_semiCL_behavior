"""The free-run counterpart of Figure 5.3 (fig_rule_kin): turn toward the dots, displacement and interval against the
time since the dots appeared, at the same matched geometry (moving dots, |bearing| 45-135 deg, 0-60 mm, 3 x 6 cells,
every time bin reweighted to the real fish's common cell mix), but for the models from FREE simulated runs instead of
the teacher-forced expectation on the real bouts.

Real fish: exactly as in Figure 5.3 (the observed bouts; geometric mean over a fish's bouts for displacement and
interval, mean over fish, 95 % bootstrap band over fish). M6 and F (both seeds): 50 simulated fish per unit (10 per
fold model), each swimming the whole recorded programme on its own
clock from a sampled start pose, the dots re-anchored to its pose at every block onset, wall rejection on, per-block
random streams -- the rollouts of the units' dot profiles (model_figures.programme_bouts). A bout's bearing and
range are the generated ones; its displacement and interval are the generated step and interval. Mean over the
simulated fish and 95 % bootstrap interval over them. Writes figures/out/fig_rule_3panel_sim.pdf.

    uv run python -m figures.fig_rule_kin_sim            # simulate if the cache is missing (20 fold jobs), draw
"""

from figures.style import *  # noqa: F401,F403,E402

import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from figures import fig_rule as FR  # noqa: E402
from figures import fig_rule_kin as FK  # noqa: E402
from twin import missing_pieces as P  # noqa: E402

R_PER_MODEL = 10                                  # simulated fish per fold model: 5 x 10 = 50 per unit
SIM_CACHE = FR.CACHE / f"fig_rule_3panel_sim_r{R_PER_MODEL}.pkl"


def _sim_job(job):
    """Free programme rollouts of one fold model -> one row per generated stimulus bout."""
    lab, d, file, k = job
    P._threads1()
    import contextlib
    import io
    from twin import model_figures as MF
    from twin import stim_rollout as SR
    m = P.load_model(d, file)
    program, dish_sim, ref = MF._programme()
    with contextlib.redirect_stdout(io.StringIO()):
        rolls = SR.rollout_many(m, program, R_PER_MODEL, seed=k, dish=dish_sim, pose_sampler=ref.sample_start_pose,
                                on_budget="truncate", block_seed=True)
    rows = []
    for r, g in enumerate(rolls):
        if g.get("runaway"):
            continue
        b = np.asarray(g["block_id"])
        fo = np.asarray(g["frame_onset"])
        dist = np.asarray(g["distance_px"], float); dt = np.asarray(g["dt_s"], float)
        rng = np.hypot(np.asarray(g["x_px"]) - np.asarray(g["dot_cx"]), np.asarray(g["y_px"]) - np.asarray(g["dot_cy"]))
        rows.append(pd.DataFrame(dict(
            model=lab, fish=k * R_PER_MODEL + r, t_blk=(fo - program.starts[b]) / program.fps,
            n=program.n_dots_of[b].astype(int), moving=program.moving_of[b] > 0,
            phi=np.asarray(g["stim_angle_gen"], float), rng_mm=rng / P.PX_MM,
            tow=np.asarray(g["dtheta"], float) * np.sign(np.asarray(g["stim_angle_gen"], float)),
            ld=np.log(np.clip(dist, 1e-6, None) / P.PX_MM), lt=np.log(np.clip(dt, 1e-6, None)),
            ok=np.isfinite(dist) & (dist > 0) & np.isfinite(dt) & (dt > 0)))[program.is_stim_of[b]])
    return pd.concat(rows, ignore_index=True)


def sim_data(jobs=10):
    from twin import registry
    if SIM_CACHE.exists() and not registry.stale(SIM_CACHE, [d for d, _, _ in P.HEAD.values()]):
        return pd.read_pickle(SIM_CACHE)
    jl = [(lab, d, file, k) for lab, (d, arm, key) in P.HEAD.items() for k, (file, _) in enumerate(P.folds_of(d, arm))]
    with ProcessPoolExecutor(jobs) as ex:
        tab = pd.concat(list(ex.map(_sim_job, jl)), ignore_index=True)
    SIM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tab.to_pickle(SIM_CACHE)
    return tab


def main():
    xc = np.arange(len(FR.T_BINS), dtype=float)
    offs = {"M6 s0": -0.21, "M6 s101": -0.07, "F s0": 0.07, "F s101": 0.21}
    turn, w = FR.onset_data()                                            # real turn toward and the common cell mix
    kt = FR._geo(FK.kin_data())
    kt = kt[kt.ok & (kt.n > 0) & np.isfinite(kt.phi) & np.isfinite(kt.rng_mm)]
    wk = kt[kt.model == "M6 s0"].groupby("cell").size(); wk = wk / wk.sum()
    real = {"turn": turn["real"], "step": FR._std(kt[kt.model == "M6 s0"], "ld_real", wk),
            "interval": FR._std(kt[kt.model == "M6 s0"], "lt_real", wk)}
    st = FR._geo(sim_data())
    st = st[st.ok & (st.n > 0) & np.isfinite(st.phi) & np.isfinite(st.rng_mm)]
    sim = {lab: {"turn": FR._std(st[st.model == lab], "tow", w), "step": FR._std(st[st.model == lab], "ld", wk),
                 "interval": FR._std(st[st.model == lab], "lt", wk)} for lab in P.HEAD}

    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.45))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.27, wspace=0.42)
    panels = [(axs[0], "turn", False, "turn toward the dots (deg)"),
              (axs[1], "step", True, "displacement (mm)"),
              (axs[2], "interval", True, "interval (s)")]
    nums = {}
    for ax, key, logged, ylab in panels:
        tr = (lambda v: np.exp(v)) if logged else (lambda v: v)
        r = real[key]
        m = r.mean(axis=0).to_numpy()
        ci = np.array([P._boot_ci(r[c].to_numpy()) for c in r.columns])
        ax.fill_between(xc, tr(ci[:, 0]), tr(ci[:, 1]), color=FAINT, alpha=0.7, lw=0, zorder=1)
        ax.plot(xc, tr(m), "-o", color=REAL_C, ms=2.8, lw=1.3, zorder=3, label="real fish")
        nums[key] = {"real": tr(m).round(3).tolist()}
        for src, name, c, seed in FR.SOURCES[1:]:
            D = sim[src][key].reindex(columns=r.columns)
            e = D.mean(axis=0).to_numpy()
            eci = np.array([P._boot_ci(D[cc].to_numpy()) for cc in D.columns])
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
    save(fig, "fig_rule_3panel_sim")
    out = FR.RF / "fig_rule_3panel_sim_numbers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(nums, t_bins=FR.T_BINS), indent=1))
    n_sim = {lab: int(st[st.model == lab].fish.nunique()) for lab in P.HEAD}
    print(f"simulated fish per unit: {n_sim}")
    for key in ("turn", "step", "interval"):
        print(f"\n{key} by time bin {[f'{a}-{b}' for a, b in FR.T_BINS]}")
        for src, v in nums[key].items():
            print(f"  {src:8s} " + " ".join(f"{x:6.3f}" for x in v))


if __name__ == "__main__":
    main()
