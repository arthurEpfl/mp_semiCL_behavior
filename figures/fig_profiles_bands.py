"""fig_profiles_bands_<model> -- Appendix app:profile-spread (fig:app-profile-bands-*): the distance profiles of
fig_profiles with a between-fish band for the model as well, one full-page figure per model (M6, D4, F), seed 0 on
top (a-h) and seed 101 below (i-p), same layout as fig_profiles.

Real fish: as fig_profiles (mean over fish of each fish's 2 s bin means, 95 % percentile bootstrap over the 15 fish,
4000 resamples, seed 0). Model: the archived block-matched curve of fig_profiles, and around it the 95 % percentile
bootstrap band of the mean over 15 simulated fish resampled from the 50 (4000 resamples, seed 0), so both bands rest
on the same number of fish. Every simulated fish draws its own latent z, so the 50 are 50 individuals. Before the
bootstrap the simulated bouts are thinned at random to the real number per fish and bin (per condition: real median
over simulated median), because simulated fish never become quiescent and bout ~1.5x more per bin, which on its own
makes each fish's bin mean less noisy and the band narrower (by 12-14 %). The per-fish curves of the thinned bouts
are block-matched exactly as the archived ones (model_figures.weighted_curves with real_block_weights).

Also writes $RF/cache/fig_profiles_bands_ratio.csv: model band width over real band width (mean over the 0-60 s
bins), averaged over THIN_REPS thinnings -- the numbers of tab:profile-spread.

Inputs: the fig_profiles cache (curves + fig_profiles_gen_<key>.pkl); run fig_profiles.py first.

    uv run python -m figures.fig_profiles_bands
"""

import sys
from pathlib import Path

from figures.style import *  # noqa: F401,F403,E402

import contextlib  # noqa: E402
import io  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from figures import fig_profiles as FP  # noqa: E402

N_FISH, N_BOOT, BOOT_SEED, THIN_REPS = 15, 4000, 0, 5
OUT_RATIO = FP.CACHE / "fig_profiles_bands_ratio.csv"


def _real_side():
    from twin import model_figures as MF
    MF.BLOCK_SEED = True
    MF.EXCLUDE_CENTROID_OUT = True
    MF._CACHE.pop("st", None)
    with contextlib.redirect_stdout(io.StringIO()):
        program, _, _ = MF._programme()
        st = MF.real_dot_profile()
    w = MF.real_block_weights(st, program)
    n_real = st[st.t_blk < 60].groupby(["cond_label", "fish_id", "tbin"]).size().groupby("cond_label").median()
    return MF, w, n_real


def _thin(gen, n_real, rng):
    """Keep each simulated bout with probability (real median bouts per fish and bin) / (simulated median), per
    condition."""
    g = gen[gen.t_blk < 60]
    n_sim = g.groupby(["cond_label", "fish_id", "tbin"]).size().groupby("cond_label").median()
    p = (n_real / n_sim).clip(upper=1.0)
    keep = rng.random(len(g)) < g.cond_label.map(p).to_numpy(float)
    return g[keep]


def _boot(pf, cond, n):
    """Mean and 95 % percentile bootstrap of the mean over ``n`` fish resampled from the per-fish curves."""
    M = pf.loc[cond].unstack("tbin").sort_index(axis=1)
    M = M.loc[:, M.columns < 60]
    A = M.to_numpy(float)
    idx = np.random.default_rng(BOOT_SEED).integers(0, A.shape[0], size=(N_BOOT, n))
    bs = np.array([np.nanmean(A[i], axis=0) for i in idx])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return pd.DataFrame({"lo": lo, "hi": hi}, index=M.columns.to_numpy(float))


def bands(c):
    """{(name, seed): {cond: band of the first thinning}} and the width-ratio table (mean over THIN_REPS)."""
    MF, w, n_real = _real_side()
    real_w = {cond: float((b.hi - b.lo)[b.index < 60].mean()) for cond, b in c["real"]["band"].items()}
    out, rows = {}, []
    for (name, seed), m in c["models"].items():
        gen = pd.read_pickle(FP.CACHE / f"fig_profiles_gen_{m['key']}.pkl")
        widths = {cond: [] for cond in FP.CONDS}
        for rep in range(THIN_REPS):
            pf, _, _ = MF.weighted_curves(_thin(gen, n_real, np.random.default_rng(rep)), w)
            for cond in FP.CONDS:
                b = _boot(pf, cond, N_FISH)
                widths[cond].append(float((b.hi - b.lo).mean()))
                if rep == 0:
                    out.setdefault((name, seed), {})[cond] = b
        for cond in FP.CONDS:
            rows.append(dict(model=name, seed=seed, cond=cond, ratio=np.mean(widths[cond]) / real_w[cond],
                             ratio_sd_over_thinnings=np.std(widths[cond]) / real_w[cond]))
        print(f"  {name} s{seed} ({m['key']}): bands done", flush=True)
    return out, pd.DataFrame(rows)


def _block(sf, c, B, name, seed, letters):
    f = MM_3P67_TO_3P60
    col = FP.UNIT_C[name]
    gs = sf.add_gridspec(2, 4, left=0.075, right=0.995, bottom=0.10, top=0.88, wspace=0.34, hspace=0.30)
    axes = [sf.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    for i, ax in enumerate(axes):
        if i:
            ax.sharex(axes[0])
        if 0 < i < 5:
            ax.sharey(axes[0])
        elif i > 5:
            ax.sharey(axes[5])
    for i, (ax, cond) in enumerate(zip(axes, FP.CONDS)):
        lim = FP.Y_MOVING if i < 5 else FP.Y_STATIC
        b = c["real"]["band"][cond]
        b = b[b.index < 60]
        FP._inside(np.r_[b.lo, b.hi, b.mu] * f, lim, f"real {cond}")
        ax.fill_between(b.index, b.lo * f, b.hi * f, color=REAL_C, alpha=0.16, lw=0)
        ax.plot(b.index, b.mu * f, color=REAL_C, lw=1.6, zorder=4)
        mb = B[(name, seed)][cond]
        y = c["models"][(name, seed)]["matched_mean"][cond]
        y = y[y.index < 60]
        FP._inside(np.r_[mb.lo, mb.hi, y.values] * f, lim, f"{name} s{seed} {cond}")
        ax.fill_between(mb.index, mb.lo * f, mb.hi * f, color=col, alpha=0.20, lw=0)
        ax.plot(y.index, y.values * f, color=col, lw=1.3, zorder=3)
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 20, 40, 60])
        ax.grid(axis="y"); ax.set_axisbelow(True)
        panel(ax, letters[i], FP.TITLES[cond])
        ax.texts[-1].set_x(-0.06)
        if i < 4:
            ax.tick_params(labelbottom=False)
        if i not in (0, 4, 5):
            ax.tick_params(labelleft=False)
        if i == 5:
            ax.set_ylabel("distance (mm)", labelpad=1.5)
    axes[0].set_ylim(*FP.Y_MOVING); axes[0].set_yticks([10, 15, 20, 25, 30])
    axes[5].set_ylim(*FP.Y_STATIC); axes[5].set_yticks([20, 25, 30, 35, 40])
    p0, p4 = axes[0].get_position(), axes[4].get_position()
    sf.text(0.008, (p0.y1 + p4.y0) / 2, "distance to dots (mm)", rotation=90, ha="left", va="center", fontsize=8)
    pe, pf = axes[4].get_position(), axes[5].get_position()
    xr = pe.x1 + 0.10 * (pf.x0 - pe.x1)
    sf.add_artist(Line2D([xr, xr], [pe.y0 - 0.02, pe.y1 + 0.06], transform=sf.transSubfigure, color=FAINT, lw=0.8))
    sf.text(0.54, 0.0, "time since block onset (s)", ha="center", va="bottom", fontsize=8)
    sf.text(0.0, 0.995, f"Seed {seed}", ha="left", va="top", fontsize=9.5, fontweight="bold")


def draw(c, B, name):
    from matplotlib.legend_handler import HandlerTuple
    lead = 0.25
    fig = plt.figure(figsize=(TEXTWIDTH_IN, FP.BLOCK_H * 2 + lead))
    top = fig.subfigures(2, 1, height_ratios=[lead, FP.BLOCK_H * 2], hspace=0)
    col = FP.UNIT_C[name]
    handles = [(Patch(facecolor=REAL_C, alpha=0.16, lw=0), Line2D([], [], color=REAL_C, lw=1.6)),
               (Patch(facecolor=col, alpha=0.20, lw=0), Line2D([], [], color=col, lw=1.3))]
    labels = ["real fish (mean, 95 % CI over 15 fish)", f"{name} (mean, 95 % CI over 15 simulated fish)"]
    top[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=2, handlelength=2.4,
                  columnspacing=2.0, frameon=False, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})
    for j, (sf, seed) in enumerate(zip(top[1].subfigures(2, 1, hspace=0), FP.SEEDS)):
        _block(sf, c, B, name, seed, "abcdefghijklmnop"[8 * j: 8 * j + 8])
    return fig


def main():
    with open(FP.CURVES, "rb") as fh:
        c = pickle.load(fh)
    assert all(u in c["models"] for u in FP.HEAD), "run fig_profiles.py first (D4 missing from the cache)"
    B, R = bands(c)
    R.to_csv(OUT_RATIO, index=False)
    t = R.assign(unit=R.model + " s" + R.seed.astype(str)).pivot(index="cond", columns="unit", values="ratio")
    t = t.reindex(FP.CONDS)[[f"{n} s{s}" for n in FP.UNITS for s in FP.SEEDS]]
    pd.set_option("display.width", 200)
    print("model / real band width (bouts thinned, 15 fish each, mean of", THIN_REPS, "thinnings):")
    print(t.round(2).to_string())
    print("largest SD over thinnings:", round(float(R.ratio_sd_over_thinnings.max()), 3))
    for name in FP.UNITS:
        save(draw(c, B, name), f"fig_profiles_bands_{name}")


if __name__ == "__main__":
    main()
