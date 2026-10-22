"""fig_bout_trajectory (Data, fig:bouttraj): from the tracked trajectory to bouts.

The first stimulus block of fish 0: its raw 30 fps track (grey) and each bout as an arrow from its onset to the next
onset, so an arrow's direction is the bout's travel heading and its length its displacement. Millimetres at the
physical 3.60 px/mm, origin at the first onset. Bouts detected by ``segment_bouts.detect_bouts_in_segment`` on the
smoothed track, as in the bout table.

    uv run python -m figures.fig_bout_trajectory      # -> figures/out/fig_bout_trajectory.png (+ .pdf)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from figures import show_png  # noqa: E402
from twin import paths  # noqa: E402
from twin import segment_bouts as SB  # noqa: E402

MM = 3.60
BLUE, INK, MUTED = "#0072B2", "#1a1a1a", "#666666"
RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
      "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
      "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
      "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
      "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42}


def first_block(fish=0, min_frames=600):
    """The first stimulus block (longer than ``min_frames``) of one fish's 30 fps table, and the fish's prefix."""
    df = pd.read_csv(sorted(paths.FRAMES.glob(f"*_f{fish}_transformed_abs.csv"))[0])
    prefix = SB.fish_prefix(df)
    blk = (df["stimulus"] != df["stimulus"].shift()).cumsum()
    cand = df.assign(_b=blk).groupby("_b").filter(
        lambda g: g["stimulus"].iloc[0].startswith("CLsemi") and len(g) > min_frames)
    return cand[cand["_b"] == cand["_b"].iloc[0]].reset_index(drop=True), prefix


def main():
    seg, prefix = first_block()
    x, y = SB.fish_xy_px(seg, prefix)
    onsets, _ = SB.detect_bouts_in_segment(x, y, 1.0)
    xr, yr = seg[f"{prefix}_x"].to_numpy(float), seg[f"{prefix}_y"].to_numpy(float)
    X0, Y0 = xr[onsets[0]], yr[onsets[0]]
    rx, ry = (xr - X0) / MM, (yr - Y0) / MM               # raw positions (mm)
    px, py = rx[onsets], ry[onsets]                       # bout onsets (mm)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        ax.plot(rx, ry, color="#b0b0b0", lw=0.7, zorder=1, label="raw trajectory")
        for k in range(len(px) - 1):
            ax.annotate("", xy=(px[k + 1], py[k + 1]), xytext=(px[k], py[k]),
                        arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=7, color=BLUE, lw=0.8))
        ax.plot(px, py, "o", ms=2.2, color=INK, zorder=3, label="bout onset")
        ax.plot([], [], color=BLUE, lw=0.8, label="bout (onset to next onset)")   # legend entry for the arrows
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"One 60 s stimulus block: {len(px)} bouts", loc="left", pad=6)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.6)
        fig.tight_layout()
        paths.FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.FIGURES / "fig_bout_trajectory.png", dpi=100)
        fig.savefig(paths.FIGURES / "fig_bout_trajectory.pdf")
    plt.close(fig)
    show_png(paths.FIGURES / "fig_bout_trajectory.png", width=500)


if __name__ == "__main__":
    main()
