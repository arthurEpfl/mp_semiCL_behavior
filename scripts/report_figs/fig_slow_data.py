"""fig_slow_data -- Chapter 3, sec:data-sequential: real fish go through slow phases, with no trend over the session.

Windows: 60 consecutive active bouts inside one quiescence-free run (generative_metrics._window_starts, the windows of
win_acf1 and of fig_slow), so window means here give back the lag-1 values of fig_slow exactly at lag 1.
  (a) One fish (the one with the most windows): its window means over session time, for the log interval, log
      displacement and turn magnitude, against the same fish's bouts in random order (grey: 95 % range of the window
      means over SHUFFLES shuffles of its bouts across the session, which keeps every bout and destroys any slow
      structure). Interval and displacement are drawn as geometric means (exp of the window mean of the log).
  (b) Window-to-window correlation at lags 1-6 windows (60-360 bouts), within fish and within runs, then mean over the
      fish with at least six pairs at that lag, with a t-based 95 % CI over fish; grey: the same statistic on the
      shuffled bouts (mean of the per-fish shuffle means, and the 2.5-97.5 % range of their across-fish mean).
  (c) Trend: each fish's window means standardised within the fish, averaged in 15-min bins of session time, then mean
      over fish with a t-based 95 % CI.
Run:  uv run python scripts/report_figs/fig_slow_data.py      (writes Final_Report/figures/fig_slow_data.pdf + PNG)
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as _t  # noqa: E402

import generative_metrics as GM  # noqa: E402
import model_figures as MF  # noqa: E402

W, LAGS, SHUFFLES, FPS, PX_MM = 60, range(1, 7), 200, 30.0, 3.60
CH = [("dt", "interval (s)", BLUE), ("dist", "displacement (mm)", VERM), ("turn", "turn magnitude (deg)", GREEN),
      ("rad", "radial position r/R", PURPLE)]


def series(rec):
    return {"dt": np.log(np.clip(rec.dt_s, 1e-6, None)), "dist": np.log(np.clip(rec.distance_px / PX_MM, 1e-6, None)),
            "turn": np.abs(np.asarray(rec.dtheta, float)), "rad": np.asarray(GM._radial_of(rec, rec.x_px, rec.y_px))}


def windows(n, runs):
    return np.asarray(GM._window_starts(n, W, W, runs), int)


def wmeans(x, starts):
    return np.array([x[a:a + W].mean() if np.isfinite(x[a:a + W]).all() else np.nan for a in starts])


def lag_acf(m, starts, runs, k):
    """win_acf1's estimator at lag k: global demeaning within the fish, pairs k windows apart in the same run."""
    ok_w = np.isfinite(m)
    m, a = m[ok_w], starts[ok_w]
    if len(m) < 12:
        return np.nan
    x = m - m.mean()
    r = np.asarray(runs)
    ok = (r[a[k:]] == r[a[:-k]]) & (a[k:] - a[:-k] == k * W)
    d = float((x * x).sum())
    if d <= 0 or ok.sum() < 6:
        return np.nan
    return float((x[k:][ok] * x[:-k][ok]).sum() * (len(x) - k) / ok.sum() / d)


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    h = _t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return v.mean(), h, len(v)


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        recs = MF.records()
    rng = np.random.default_rng(0)
    real = {c: {k: [] for k in LAGS} for c, *_ in CH}
    null = {c: {k: [] for k in LAGS} for c, *_ in CH}              # per fish: (SHUFFLES,) values
    trend, ex = [], None
    for f, rec in recs.items():
        runs = np.asarray(rec.run_id)
        st = windows(len(rec.dt_s), runs)
        tmin = np.asarray(rec.frame_onset, float)[st + W // 2] / FPS / 60.0
        S = series(rec)
        for c, *_ in CH:
            x = S[c]
            m = wmeans(x, st)
            for k in LAGS:
                real[c][k].append(lag_acf(m, st, runs, k))
            sh = []
            for _ in range(SHUFFLES):
                xs = x.copy()
                fin = np.isfinite(xs)
                xs[fin] = rng.permutation(xs[fin])
                sh.append(wmeans(xs, st))
            sh = np.array(sh)
            for k in LAGS:
                null[c][k].append(np.array([lag_acf(s, st, runs, k) for s in sh]))
            if c != "rad":
                z = (m - np.nanmean(m)) / np.nanstd(m)
                trend.append(pd.DataFrame({"fish": f, "ch": c, "t": tmin, "z": z}))
            if ex is None or len(st) > ex["n"]:
                pass
        if ex is None or len(st) > ex["n"]:
            ex = {"fish": f, "n": len(st), "t": tmin, "runs": runs[st],
                  "m": {c: wmeans(S[c], st) for c, *_ in CH},
                  "band": {c: None for c, *_ in CH}, "S": S, "st": st}
    # shuffle band for the example fish
    for c, *_ in CH[:3]:
        x = ex["S"][c]
        sh = []
        for _ in range(SHUFFLES):
            xs = x.copy(); fin = np.isfinite(xs); xs[fin] = rng.permutation(xs[fin])
            sh.append(wmeans(xs, ex["st"]))
        ex["band"][c] = np.nanpercentile(np.array(sh), [2.5, 97.5])
    T = pd.concat(trend)
    T["bin"] = (np.floor(T.t / 15) + 0.5) * 15

    print(f"example fish {ex['fish']}: {ex['n']} windows")
    for c, lab, _ in CH:
        print(f"{c:5s} real " + "  ".join(f"lag{k}: {ci(real[c][k])[0]:+.3f}±{ci(real[c][k])[1]:.3f} (n={ci(real[c][k])[2]})"
                                        for k in LAGS))
        print(f"{'':5s} null " + "  ".join(f"lag{k}: {np.nanmean([np.nanmean(v) for v in null[c][k]]):+.3f}" for k in LAGS))

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 5.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], left=0.09, right=0.985, top=0.94, bottom=0.09,
                          wspace=0.32, hspace=0.62)
    tr = lambda c, v: np.exp(v) if c in ("dt", "dist") else v
    axa = []
    for i, (c, lab, col) in enumerate(CH[:3]):
        ax = fig.add_subplot(gs[i, 0], sharex=axa[0] if axa else None)
        axa.append(ax)
        lo, hi = ex["band"][c]
        ax.axhspan(tr(c, lo), tr(c, hi), color=FAINT, alpha=0.6, lw=0, zorder=0)
        t, m, r = ex["t"], ex["m"][c], ex["runs"]
        for run in np.unique(r):                      # connect windows only within one run
            s = r == run
            ax.plot(t[s], tr(c, m[s]), color=col, lw=1.0, marker="o", ms=2.2, zorder=3)
        if c in ("dt", "dist"):
            from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator, NullLocator
            ax.set_yscale("log")
            v = tr(c, m[np.isfinite(m)])
            ax.yaxis.set_major_locator(FixedLocator(MaxNLocator(5).tick_values(v.min(), v.max())))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_ylabel(lab, fontsize=7)
        ax.grid(axis="y"); ax.set_axisbelow(True)
        if i < 2:
            ax.tick_params(labelbottom=False)
        if i == 0:
            panel(ax, "a", f"One fish over the session (fish {ex['fish']})")
    axa[-1].set_xlabel("session time (min)")
    axa[-1].set_xlim(0, 175)

    axb = fig.add_subplot(gs[0:2, 1])
    ks = np.array(list(LAGS))
    for j, (c, lab, col) in enumerate(CH):
        mu = np.array([ci(real[c][k])[0] for k in ks]); h = np.array([ci(real[c][k])[1] for k in ks])
        off = (j - 1.5) * 0.06
        axb.errorbar(ks + off, mu, yerr=h, color=col, marker="o", ms=3, lw=1.1, capsize=0,
                     label=lab.split(" (")[0].replace(" r/R", ""))
    nl = []
    for k in ks:
        per = np.array([np.nanmean(np.concatenate([v[None] for v in null[c][k]]), axis=1).tolist()
                        for c, *_ in CH], dtype=object)
    allnull = np.concatenate([np.concatenate([v for v in null[c][k]]) for c, *_ in CH for k in ks])
    lo, hi = np.nanpercentile([np.nanmean([np.nanmean(v) for v in null[c][k]]) for c, *_ in CH for k in ks], [0, 100])
    band_lo = min(lo, -0.03); band_hi = max(hi, 0.03)
    axb.axhspan(band_lo, band_hi, color=FAINT, alpha=0.6, lw=0, zorder=0)
    axb.axhline(0, color=MUTED, lw=0.6)
    axb.set_xticks(ks); axb.set_xticklabels([f"{k * W}" for k in ks], fontsize=6.5)
    axb.set_xlabel("lag (bouts; 60 per window)", labelpad=1)
    axb.set_ylabel("window-to-window correlation")
    axb.grid(axis="y"); axb.set_axisbelow(True)
    axb.legend(fontsize=6.5, frameon=False, loc="upper right")
    panel(axb, "b", "Memory across windows")

    axc = fig.add_subplot(gs[2, 1])
    for c, lab, col in CH[:3]:
        g = T[T.ch == c].groupby(["fish", "bin"]).z.mean().unstack("bin")
        x = g.columns.to_numpy(float)
        st_ = np.array([ci(g[b])[:2] for b in g.columns])
        axc.plot(x, st_[:, 0], color=col, lw=1.1, marker="o", ms=2.6)
        axc.fill_between(x, st_[:, 0] - st_[:, 1], st_[:, 0] + st_[:, 1], color=col, alpha=0.15, lw=0)
    axc.axhline(0, color=MUTED, lw=0.6)
    axc.set_xlim(0, 175); axc.set_xlabel("session time (min)")
    axc.set_ylabel("window mean (z, within fish)", fontsize=7)
    axc.grid(axis="y"); axc.set_axisbelow(True)
    panel(axc, "c", "Over the session")
    save(fig, "fig_slow_data_full")                # the three-panel version, kept for reference
    print("shuffle level of (b), range over channels and lags:", round(lo, 3), round(hi, 3))

    # the report's figure (2026-10-15): panel (c) alone
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.4))
    fig.subplots_adjust(left=0.09, right=0.80, top=0.95, bottom=0.19)
    for c, lab, col in CH[:3]:
        g = T[T.ch == c].groupby(["fish", "bin"]).z.mean().unstack("bin")
        x = g.columns.to_numpy(float)
        st_ = np.array([ci(g[b])[:2] for b in g.columns])
        ax.plot(x, st_[:, 0], color=col, lw=1.3, marker="o", ms=3, label=lab.split(" (")[0])
        ax.fill_between(x, st_[:, 0] - st_[:, 1], st_[:, 0] + st_[:, 1], color=col, alpha=0.15, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(0, 175); ax.set_xlabel("session time (min)")
    ax.set_ylabel("window mean (z, within fish)")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7)
    save(fig, "fig_slow_data")


if __name__ == "__main__":
    main()
