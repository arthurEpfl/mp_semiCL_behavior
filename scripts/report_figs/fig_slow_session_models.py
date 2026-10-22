"""fig_slow_session_models -- the session panel of fig_slow_data (window means standardised within each fish, 15-min
bins of session time) for real fish against M6 (no slow state) and D4 (AR(1) slow state), at both seeds.

Rollouts: record-indexed, as the Metric-B panel (rollout_model1_local._one): the fold model that held each fish out,
the rej cfg_override, rng = default_rng([checkpoint rollout_seed, fish]), GM.sample_rollouts with R rollouts. A
record-indexed rollout has as many bouts as the real record and bout t is shown the record's bout-t stimulus, so the
same window starts (60 bouts inside one of the RECORD's runs) and the record's session time at each window's middle
bout are used on both sides. Per rollout the window means are standardised within the rollout, binned, and averaged
over the R rollouts of a fish; then mean over fish with a t-based 95 % CI (as for the real fish).
Writes Final_Report/figures/fig_slow_session_models.pdf (+ PNG); caches the per-fish curves.

    uv run python scripts/report_figs/fig_slow_session_models.py
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UNITS = [("M6", 0, "26e"), ("M6", 101, "30d"), ("D4", 0, "11e"), ("D4", 101, "32a")]
REJ = {"wall_reject": {"d0": 2.0, "d1": 10.0, "max_draws": 20}}
R, W, FPS, PX_MM, BIN = 10, 60, 30.0, 3.60, 15.0
CH = [("dt", "interval"), ("dist", "displacement"), ("turn", "turn magnitude")]
CACHE = REPORT_FINAL_DIR / "cache" / "fig_slow_session_models.pkl"


def _series(dt, dist, dth):
    return {"dt": np.log(np.clip(np.asarray(dt, float), 1e-6, None)),
            "dist": np.log(np.clip(np.asarray(dist, float) / PX_MM, 1e-6, None)),
            "turn": np.abs(np.asarray(dth, float))}


def _binned_z(x, starts, tmin):
    m = np.array([x[a:a + W].mean() if a + W <= len(x) and np.isfinite(x[a:a + W]).all() else np.nan
                  for a in starts])
    z = (m - np.nanmean(m)) / np.nanstd(m)
    b = (np.floor(tmin / BIN) + 0.5) * BIN
    return pd.Series(z).groupby(b).mean()


def job(args):
    key, ckpt = args
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "Model")]
    import json
    import torch
    torch.set_num_threads(1)
    import evaluate as E
    import fit_MLP_border as MB
    import fit_MLP_mix as MX
    import generative_metrics as GM
    recs = MB.build_border_records(E.load_bouts())
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    seed = int(blob.get("rollout_seed", 0))
    man = json.loads((Path(ckpt).parent / "manifest_model1.json").read_text())
    held = next(e["heldout"] for e in man if e["file"] == Path(ckpt).name)
    m = MX.load_mix(str(ckpt))
    m._cfg_feat = {**m._cfg_feat, **REJ}
    out = []
    for fid in held:
        rec = recs[fid]
        runs = np.asarray(rec.run_id)
        st = np.asarray(GM._window_starts(len(rec.dt_s), W, W, runs), int)
        tmin = np.asarray(rec.frame_onset, float)[st + W // 2] / FPS / 60.0
        real = _series(rec.dt_s, rec.distance_px, rec.dtheta)
        row = {"key": key, "fish": int(fid), "real": {c: _binned_z(real[c], st, tmin) for c, _ in CH}}
        rolls = GM.sample_rollouts(m, rec, np.random.default_rng([seed, int(fid)]), R=R)
        gen = {c: [] for c, _ in CH}
        for g in rolls:
            s = _series(g["dt_s"], g["distance_px"], g["dtheta"])
            for c, _ in CH:
                gen[c].append(_binned_z(s[c], st, tmin))
        row["gen"] = {c: pd.concat(v, axis=1).mean(axis=1) for c, v in gen.items()}
        out.append(row)
    return out


def _ci(df):
    from scipy.stats import t
    mu = df.mean(axis=0)
    n = df.notna().sum(axis=0)
    h = t.ppf(0.975, n - 1) * df.std(axis=0, ddof=1) / np.sqrt(n)
    return mu.index.to_numpy(float), mu.to_numpy(float), h.to_numpy(float)


def main():
    import glob
    if CACHE.exists():
        rows = pickle.load(open(CACHE, "rb"))
    else:
        jobs = [(k, f) for _, _, k in UNITS
                for f in sorted(glob.glob(str(ROOT / "Model" / "report_ckpts" / f"M{k}" / "*_fold[0-9].pt")))]
        rows = []
        with ProcessPoolExecutor(10) as ex:
            for r in ex.map(job, jobs):
                rows += r
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(rows, open(CACHE, "wb"))

    fig, axes = plt.subplots(3, 2, figsize=(TEXTWIDTH_IN, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.09, wspace=0.08, hspace=0.30)
    for j, model in enumerate(("M6", "D4")):
        col_m = MODEL_C["M6"] if model == "M6" else GREEN
        for i, (c, lab) in enumerate(CH):
            ax = axes[i, j]
            real = pd.DataFrame({r["fish"]: r["real"][c] for r in rows if r["key"] == "26e"}).T
            x, mu, h = _ci(real)
            ax.fill_between(x, mu - h, mu + h, color=REAL_C, alpha=0.12, lw=0)
            ax.plot(x, mu, color=REAL_C, lw=1.4, marker="o", ms=2.6, label="real fish")
            for seed, key in [(s, k) for n, s, k in UNITS if n == model]:
                g = pd.DataFrame({r["fish"]: r["gen"][c] for r in rows if r["key"] == key}).T
                x, mu_g, h_g = _ci(g)
                ax.plot(x, mu_g, color=col_m, lw=1.1, marker="o", ms=2.6, mfc=col_m if seed == 0 else "white",
                        ls="-" if seed == 0 else (0, (3, 1.6)), label=f"{model} seed {seed}")
                sl = np.polyfit(x, mu_g, 1)[0] * 100
                print(f"{model} s{seed} {c:5s}: slope of the fish-mean curve {sl:+.2f} z per 100 min "
                      f"(real {np.polyfit(x, mu, 1)[0] * 100:+.2f})")
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.grid(axis="y"); ax.set_axisbelow(True)
            ax.set_xlim(0, 175)
            if j == 0:
                ax.set_ylabel(f"{lab}\n(z, within fish)", fontsize=7)
            if i == 0:
                ax.set_title(f"{model} ({'no slow state' if model == 'M6' else 'AR(1) slow state'})", fontsize=8.5)
            if i == 0 and j == 1:
                pass
        axes[2, j].set_xlabel("session time (min)")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    fig.legend(h1 + h2[1:], l1 + l2[1:], loc="upper center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.54, 1.0))
    save(fig, "fig_slow_session_models")


if __name__ == "__main__":
    main()
