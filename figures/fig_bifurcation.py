"""The two-group bifurcation maps (Results, fig:bifurcation; Appendix, fig:app-bif-*), from the two-target runs
(twin/two_targets.py, cached in output/two_targets/).

Rows: M6 and F at both seeds. Columns: the four combination rules, then the two references (only group A shown, no
group shown). Each column of a map is the distribution of the fish's lateral position at one separation L of the two
groups (dish of twice the radius, equidistant start), counting the bouts where the fish is behind the pair (the rule
of Sridhar et al. 2021), normalised to its maximum; dashed: the two group tracks.

    uv run python -m figures.fig_bifurcation          # -> figures/out/{1v1,4v4,1v4,2v8}_{2060,060}.png (+ .pdf)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from figures import show_png  # noqa: E402
from twin import paths  # noqa: E402
from twin import two_targets as TT  # noqa: E402

UNITS = [("26e", "M6 s0"), ("30d", "M6 s101"), ("14f", "F s0"), ("32b", "F s101")]
PARULA = LinearSegmentedColormap.from_list("parula", ["#352a87", "#0363e1", "#1485d4", "#06a7c6", "#38b99e",
                                                      "#92bf73", "#d9ba56", "#fcce2e", "#f9fb0e"])
COL_LAB = {"effects": "effects", "add": "add", "avg": "avg", "nearest": "nearest", "blind": "blind"}


def bif_map(ax, tag, pair, cond, Ls, ymax, binw, arena, onset, t_min, cmap, track_col="w"):
    """One map: lateral position of the bouts behind the pair, per separation, as a proportion of each column's max."""
    edges = np.arange(-ymax, ymax + binw, binw)
    xs = np.array(Ls, float)
    xe = np.r_[xs[0] - (xs[1] - xs[0]) / 2, (xs[1:] + xs[:-1]) / 2, xs[-1] + (xs[-1] - xs[-2]) / 2]
    Z = np.zeros((len(edges) - 1, len(xs)))
    for k, L in enumerate(Ls):
        b = TT.load(tag, pair, L, cond, arena, onset)
        if onset == "anchored":                      # the anchoring bout itself sits at lateral 0 by construction
            b = b[b.groupby(["fish", "blk"]).cumcount() > 0]
        w = b[b.t_blk >= t_min]
        h, _ = np.histogram(w[w.along_mm <= 0].lat_mm, bins=edges)
        Z[:, k] = h / h.max() if h.max() else 0
    mesh = ax.pcolormesh(xe, edges, Z, cmap=cmap, vmin=0, vmax=1, shading="flat")
    ax.plot(xs, xs / 2, "--", color=track_col, lw=0.8)
    ax.plot(xs, -xs / 2, "--", color=track_col, lw=0.8)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(xe[0], xe[-1])
    return mesh


def report_map(pair, t_min):
    a, b = TT.PAIRS[pair]
    lab_a = "only A" + ("" if a == b else f" ({a} dot{'s' if a > 1 else ''})")
    cols = TT.REPORT_CONDS
    fig, axes = plt.subplots(len(UNITS), len(cols), figsize=(2.55 * len(cols) + 0.9, 2.45 * len(UNITS)),
                             sharex=True, sharey=True)
    for i, (tag, lab) in enumerate(UNITS):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            src = "4v4" if (c == "blind" and a != b) else pair
            mesh = bif_map(ax, tag, src, c, Ls=TT.BIG_SEPS_MM, ymax=70, binw=4.0, arena=TT.ARENA_BIG,
                           onset="anchored", t_min=t_min, cmap=PARULA)
            if i == 0:
                ax.set_title(lab_a if c == "ref-A" else COL_LAB[c], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{lab}\nlateral position (mm)", fontsize=9)
            if i == len(UNITS) - 1:
                ax.set_xlabel("separation L (mm)", fontsize=9)
            ax.tick_params(labelsize=8)
    win = "20-60 s" if t_min == 20 else "0-60 s"
    pair_txt = f"{pair}" + ("" if a == b else f" (A = {a}, B = {b} dots; A on the + side)")
    fig.suptitle(f"{pair_txt}, equidistant start, {win}, bouts behind the pair", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    cax = fig.add_axes([0.952, 0.25, 0.012, 0.5])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("proportion of maximum", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    name = f"{pair}_{'2060' if t_min == 20 else '060'}"
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / f"{name}.png", dpi=100, bbox_inches="tight")
    fig.savefig(paths.FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    show_png(paths.FIGURES / f"{name}.png")


def main():
    for pair in TT.REPORT_PAIRS:
        for t_min in (20, 0):
            report_map(pair, t_min)


if __name__ == "__main__":
    main()
