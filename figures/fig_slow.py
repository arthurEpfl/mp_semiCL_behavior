"""fig_slow (Chapter 5, sec:r-final, label fig:slow): slow window-to-window structure, real fish vs models.

Three panels, one per channel (log interval, log displacement, radial position r/R): the lag-1
autocorrelation of successive 60-bout window means within one fish (``generative_metrics.win_acf1``),
the statistic in the "acf" column of tab:drift.

  * Real fish: one grey point per fish, black diamond = mean over fish with a t-based 95 % CI.
    Real windows lie inside one quiescence-free run and only consecutive windows of one run are
    paired (``win_acf1(..., runs=rec.run_id)``); a rollout never pauses, so its series is cut flat.
  * Models M6, D1, D3 (no drift, blue) and D2, D4, F (AR(1) drift, vermillion), each at seed 0
    (filled) and seed 101 (open, dashed bar): mean over held-out fish +/- t-based 95 % CI over fish.
    A fish's simulated value is already the mean over the panel's R = 20 rollouts of that fish
    (``panel_for_fish``, rollout_model1_local.run(R=20)). Seeds are never pooled.
  * Each panel carries a small tag: a, b = calibration target of the drift gains (the rollout
    search in fit_MLP_mix._search_drift_gain matches exactly these two statistics, flat windows on
    training fish), c = not a target. Legend seed handles are errorbar containers so they show the
    solid / dashed bar.

Source (read only): output/checkpoints/M<key>/panelB_*.csv, one row per held-out fish:
    real   = real_wacf_<ch>
    model  = real_wacf_<ch> + wacf_<ch>_gap      (the panel stores the gap = mean over R of gen - real)
for ch in dt / dist / rad. Three fish (4, 6, 10) have fewer than six pairs of consecutive within-run
windows, so their real value -- and therefore the stored gap -- is NaN; every mean is over the
remaining 12 fish, on both sides (the same 12 for every model and seed).

TURN MAGNITUDE (panel d, 2026-10-15): not in the panel CSVs, so read from $RF/cache/fig_slow_turn.csv (key, fish,
real_turn, gen_turn): the same record-indexed rollouts as the panel (rollout_model1_local._one: rej override,
rng = default_rng([rollout_seed, fish]), 20 per held-out fish; the build checked that their interval value gives back
the archived panel to 1e-16), |dtheta| with the same win_acf1 (real within runs, rollout flat). Not a drift target.

    uv run python -m figures.fig_slow
"""
import glob

from figures.style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import t as _t  # noqa: E402

# public name -> (seed-0 key, seed-101 key, has the AR(1) drift)
MODELS = [("M6", "26e", "30d", False), ("D1", "18a", "24a", False), ("D3", "16e", "30e", False),
          ("D2", "24b", "24c", True), ("D4", "11e", "32a", True), ("F", "14f", "32b", True)]
CHANS = [("dt", "Log interval"), ("dist", "Log displacement"), ("rad", "Radial position r/R"),
         ("turn", "Turn magnitude")]
TURN_CSV = paths.FIG_CACHE / "cache" / "fig_slow_turn.csv"
# The drift's interval and displacement gains were picked by a rollout search that matches exactly
# this statistic (flat windows, training fish; fit_MLP_mix._search_drift_gain, Methods sec:m-drift);
# the radial channel is reported there, never optimised. Panels a, b are therefore calibration
# targets carried over to held-out fish, panel c is not -- the figure says so in each panel.
TARGET = {"dt": "calibration target\nof the drift gains", "dist": "calibration target\nof the drift gains",
          "rad": "not a target of\nthe drift gains", "turn": "not a target of\nthe drift gains"}
FAM_C = {False: BLUE, True: VERM}          # no drift = M6's blue, drift = F's vermillion
X_REAL = 0.0
X_MOD = {"M6": 1.25, "D1": 2.1, "D3": 2.95, "D2": 4.2, "D4": 5.05, "F": 5.9}
OFF = {0: -0.17, 101: 0.17}


def panel_b(key):
    f = glob.glob(str(paths.CKPTS / f"M{key}" / "panelB_*.csv"))
    assert len(f) == 1, (key, f)
    return pd.read_csv(f[0]).set_index("fish").sort_index()


def mean_ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    h = _t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(h), len(v)


def load():
    """real[ch] -> per-fish Series; gen[(name, seed, ch)] -> per-fish Series."""
    real, gen, ref = {}, {}, None
    T = pd.read_csv(TURN_CSV)
    for name, k0, k1, _ in MODELS:
        for seed, key in ((0, k0), (101, k1)):
            d = panel_b(key)
            t = T[T.key == key].set_index("fish").sort_index()
            assert t.index.equals(d.index), key
            d = d.assign(real_wacf_turn=t.real_turn, wacf_turn_gap=t.gen_turn - t.real_turn)
            r = d[[f"real_wacf_{c}" for c, _ in CHANS]]
            if ref is None:
                ref = r
            # the real reference must be identical in every unit (same records, same statistic)
            assert r.index.equals(ref.index) and np.allclose(r.to_numpy(), ref.to_numpy(), equal_nan=True), key
            for c, _ in CHANS:
                gen[(name, seed, c)] = d[f"real_wacf_{c}"] + d[f"wacf_{c}_gap"]
    for c, _ in CHANS:
        real[c] = ref[f"real_wacf_{c}"]
    return real, gen


def main():
    real, gen = load()
    rng = np.random.default_rng(0)          # jitter only
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.7), sharey=True,
                             gridspec_kw=dict(wspace=0.10, hspace=0.55))
    axes = axes.ravel()
    for k, (ax, (c, title)) in enumerate(zip(axes, CHANS)):
        rv = real[c].dropna()
        m, h, n = mean_ci(rv)
        ax.axhspan(m - h, m + h, color=FAINT, alpha=0.45, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.axvline((X_MOD["D3"] + X_MOD["D2"]) / 2, color=FAINT, lw=0.6, zorder=1)
        jx = X_REAL - 0.16 + rng.uniform(-0.09, 0.09, len(rv))
        ax.plot(jx, rv.to_numpy(), "o", ms=2.3, mfc=MUTED, mec="none", alpha=0.75, zorder=2)
        ax.errorbar(X_REAL + 0.16, m, yerr=h, fmt="D", ms=3.6, color=INK, mfc=INK,
                    elinewidth=0.9, capsize=1.6, zorder=4)
        for name, _, _, drift in MODELS:
            col = FAM_C[drift]
            for seed in (0, 101):
                gm, gh, gn = mean_ci(gen[(name, seed, c)])
                assert gn == n, (name, seed, c, gn, n)
                eb = ax.errorbar(X_MOD[name] + OFF[seed], gm, yerr=gh, fmt="o", ms=3.8, color=col,
                                 mec=col, mew=0.9, mfc=col if seed == 0 else "white",
                                 elinewidth=0.9, capsize=1.6, zorder=3)
                if seed == 101:
                    eb[2][0].set_linestyle((0, (2, 1.2)))
        ax.text((X_MOD["D2"] + X_MOD["F"]) / 2, 0.80, TARGET[c], ha="center", va="top", fontsize=6.5,
                style="italic", color=VERM if c in ("dt", "dist") else MUTED, linespacing=1.1, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        ax.set_xticks([X_REAL] + [X_MOD[nm] for nm, *_ in MODELS])
        ax.set_xticklabels(["Real"] + [nm for nm, *_ in MODELS])
        for tl, (nm, *_r, drift) in zip(ax.get_xticklabels()[1:], MODELS):
            tl.set_color(FAM_C[drift])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(-0.55, X_MOD["F"] + 0.5)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        panel(ax, "abcd"[k], title)
    axes[0].set_ylim(-0.32, 0.84)
    axes[0].set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    for a in (axes[0], axes[2]):
        a.set_ylabel("lag-1 corr. of successive\n60-bout window means")

    # seed handles as real errorbar containers (drawn outside the x range, so clipped away) so the
    # legend shows the solid (seed 0) and dashed (seed 101) bars, not just the marker fill
    seed_h = {}
    for seed in (0, 101):
        eb = axes[0].errorbar([-50.0], [0.0], yerr=[0.1], fmt="o", ms=3.8, color=MUTED, mec=MUTED, mew=0.9,
                              mfc=MUTED if seed == 0 else "white", elinewidth=0.9, capsize=1.6,
                              label=f"seed {seed} (" + ("solid" if seed == 0 else "dashed") + ")")
        if seed == 101:
            eb[2][0].set_linestyle((0, (2, 1.2)))
        seed_h[seed] = eb
    handles = [Line2D([], [], ls="none", marker="o", ms=2.3, mfc=MUTED, mec="none", label="real fish"),
               Line2D([], [], ls="none", marker="D", ms=3.6, mfc=INK, mec=INK, label="real mean ± 95 % CI"),
               Line2D([], [], ls="none", marker="o", ms=3.8, mfc=BLUE, mec=BLUE, label="no drift"),
               Line2D([], [], ls="none", marker="o", ms=3.8, mfc=VERM, mec=VERM, label="AR(1) drift"),
               seed_h[0], seed_h[101]]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.49, -0.06),
               handletextpad=0.2, columnspacing=0.9)
    save(fig, "fig_slow")

    # numbers for the notes / prose (printed, never hand-typed)
    print(f"real fish with a defined value: {list(real['dt'].dropna().index)} "
          f"(NaN: {list(real['dt'][real['dt'].isna()].index)})")
    for c, title in CHANS:
        m, h, n = mean_ci(real[c])
        rv = real[c].dropna()
        print(f"{title:17s} real  mean {m:+.4f}  95% CI [{m - h:+.4f}, {m + h:+.4f}]  n={n}  "
              f"per-fish range [{rv.min():+.4f}, {rv.max():+.4f}]  fish > 0: {(rv > 0).sum()}/{n}")
        for name, *_ in MODELS:
            s = []
            for seed in (0, 101):
                gm, gh, gn = mean_ci(gen[(name, seed, c)])
                s.append(f"{gm:+.4f} [{gm - gh:+.4f}, {gm + gh:+.4f}]  -> {gm:+.2f}")
            print(f"{'':17s} {name:4s}  s0 {s[0]}  |  s101 {s[1]}")


if __name__ == "__main__":
    main()
