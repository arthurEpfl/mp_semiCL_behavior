"""Model 22 -- a per-fish HOME RADIUS inside the shallow-border rejection (generation time, no retraining).

WHY. With rej on, the pooled r/R histogram of generated bouts matches the real one, yet radial
occupancy is the classifier's largest tell (42 % of the signal on Model 17). The tell is window-level:
(1) each real fish has its own home radius (per-fish median r/R 0.33-0.68) while every generated fish
sits near the same 0.45 -- the 2-d latent has no spatial axis; (2) a real fish lingers in an annulus
for minutes and then moves (window-mean radius acf1 +0.39) while a generated fish re-draws its radius
every window (+0.03). Every lever on the wall response inside the turn law (rad_indiv, rad_gate, the
m=3 spatial axis of Model 19) was inert, because once rej is on the REJECTION BAND sets occupancy.
So the lever goes on the band: ``JT.generate_mix_joint_loop`` reads four extra keys inside
``wall_reject`` -- ``home_mu``, ``home_sd`` (one home draw per trajectory: a NEW fish's home),
``home_band`` (no-pull zone), ``home_tau`` (bouts; AR(1) drift of the home, None = static) and
``pull`` (redraw probability for a rim-accepted bout that moves away from home and lands outside the
band). A config without ``pull`` is bit-identical to rej.

WHAT IS CALIBRATED, AND AGAINST WHAT. Nothing is fitted by likelihood. home_mu / home_sd are pinned to
the real population of per-fish median r/R. ``pull`` and ``home_tau`` are chosen by rollouts on three
numbers, each with its real target: (a) the spread of INDIVIDUAL mean radii -- the SD over rollouts of
the per-rollout mean window radius against the SD over the 15 real fish of the per-fish mean window
radius (a rollout is one individual, so this is the between-fish spread of a population of sampled
individuals; the harness's ``bf_rad`` is the same reading per replicate, averaged over replicates);
(b) the per-individual window-mean radius acf1 (the harness's ``wacf_rad``); (c) the
pooled r/R histogram, total-variation distance to the real one, which rej already matches and must
not lose. The pick is the config with the smallest spread + acf error among those whose histogram
distance stays within 0.02 of the control's.

    calibrate_home(ckpt_dir, arm, rej)      -> the sweep table + home_calibration.json (cached)
    radial_figure(ckpt_dir, arm, variants)  -> the three radial readings, real vs each rule
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import generative_metrics as GM  # noqa: E402

W = 60
BINS = np.linspace(0.0, 1.0, 26)
PULLS = (0.3, 0.5, 0.7, 0.9)
TAUS = (None, 300, 1000)
SD_SCALES = (1.0, 1.5)


def _ckpt_dir(d):
    d = Path(d)
    for q in (d, Path("Model") / d, Path(__file__).parent.parent / "Model" / d):
        if (q / "manifest_model1.json").exists():
            return q
    raise FileNotFoundError(f"no manifest_model1.json under {d}")


def _records():
    import evaluate as E
    import fit_MLP_border as MB
    return MB.build_border_records(E.load_bouts())


def _win_means(r, runs=None):
    r = np.asarray(r, float)
    out = [np.nanmean(r[a:a + W]) for a in GM._window_starts(len(r), W, W, runs) if a + W <= len(r)]
    return np.array(out, float)


def _acf1(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / max(np.dot(x, x), 1e-12))


def _hist(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    h, _ = np.histogram(np.clip(r, 0, 1), bins=BINS)
    return h / max(h.sum(), 1)


def real_targets(recs=None):
    """The real population the rule is pinned to and scored against."""
    recs = recs or _records()
    med, wmean, acf, pooled = {}, {}, {}, []
    for f, rec in recs.items():
        r = GM._radial_of(rec, rec.x_px, rec.y_px)
        wm = _win_means(r, getattr(rec, "run_id", None))
        med[f] = float(np.nanmedian(r)); wmean[f] = float(wm.mean()); acf[f] = _acf1(wm)
        pooled.append(r[np.isfinite(r)])
    med, wmean, acf = pd.Series(med), pd.Series(wmean), pd.Series(acf)
    return dict(home_mu=float(med.mean()), home_sd=float(med.std(ddof=1)), sd_indiv=float(wmean.std(ddof=1)),
                level=float(wmean.mean()), wacf=float(acf.mean()), hist=_hist(np.concatenate(pooled)),
                per_fish=pd.DataFrame({"median_rR": med, "mean_window_rR": wmean, "wacf": acf}))


def _roll_job(job):
    """(ckpt, fish, wall_reject dict, seed, reps) -> (fish, [per-rollout radial series + counters])."""
    ckpt, fid, wr, seed, reps = job
    import torch
    torch.set_num_threads(1)
    import fit_MLP_mix as MX
    recs = _records()
    rec = recs[fid]
    m = MX.load_mix(str(ckpt))
    m._cfg_feat = {**m._cfg_feat, "wall_reject": dict(wr)}
    out = []
    for rep in range(reps):
        g = m.sample(rec, np.random.default_rng([int(seed), int(fid), rep, 22]), **GM._stim_kw(m, rec, None))
        r = GM._radial_of(rec, g["x_px"], g["y_px"])
        out.append(dict(r=np.asarray(r, np.float32), n=len(r), n_draws=int(g.get("n_draws", 0)),
                        n_home=int(g.get("n_home_reject", 0)), n_fb=int(g.get("n_home_fallback", 0))))
    return fid, out


def rollouts(ckpt_dir, arm, wr, reps=2, seed=0, jobs=10):
    """Every held-out fish of ``arm`` rolled out ``reps`` times under the rejection config ``wr``."""
    d = _ckpt_dir(ckpt_dir)
    mani = json.loads((d / "manifest_model1.json").read_text())
    jl = [(d / e["file"], int(f), dict(wr), seed, reps) for e in mani if e["arm"] == arm for f in e["heldout"]]
    with ProcessPoolExecutor(max_workers=min(jobs, len(jl))) as ex:
        return dict(ex.map(_roll_job, jl))


def gen_metrics(rolls, tgt):
    """The three calibration readings of one rule, from ``rollouts`` output."""
    ind_mean, ind_acf, pooled, draws, home, fb, nb = [], [], [], 0, 0, 0, 0
    for f, reps in rolls.items():
        for g in reps:
            wm = _win_means(g["r"])
            ind_mean.append(wm.mean()); ind_acf.append(_acf1(wm)); pooled.append(g["r"])
            draws += g["n_draws"]; home += g["n_home"]; fb += g["n_fb"]; nb += g["n"]
    ind_mean = np.array(ind_mean)
    return dict(sd_indiv=float(ind_mean.std(ddof=1)), sd_ratio=float(ind_mean.std(ddof=1) / tgt["sd_indiv"]),
                level=float(ind_mean.mean()), wacf=float(np.nanmean(ind_acf)),
                hist_tv=float(0.5 * np.abs(_hist(np.concatenate(pooled)) - tgt["hist"]).sum()),
                draws_per_bout=draws / max(nb, 1), home_rejects_per_bout=home / max(nb, 1),
                home_fallback_per_bout=fb / max(nb, 1), n_indiv=len(ind_mean))


def calibrate_home(ckpt_dir, arm, rej, pulls=PULLS, taus=TAUS, sd_scales=SD_SCALES, reps=2, seed=0,
                   jobs=10, force=False, tol_tv=0.02):
    """The sweep. Returns (table, chosen) and writes ``home_calibration.json`` next to the checkpoints
    (read back on the next call unless ``force``). ``chosen`` = {"drift": wall_reject dict of the best
    drifting home, "static": the best static home (home_tau None)}."""
    d = _ckpt_dir(ckpt_dir)
    cache = d / "home_calibration.json"
    if cache.exists() and not force:
        blob = json.loads(cache.read_text())
        table = pd.DataFrame(blob["table"])
        chosen = _choose(table, blob["targets"], dict(rej), tol_tv)   # re-chosen from the cached sweep
        _print(table, blob["targets"], chosen)
        return table, chosen
    t0 = time.time()
    recs = _records()
    tgt = real_targets(recs)
    base = dict(rej)
    rows = []
    grid = [("control", base)]
    for s in sd_scales:
        for tau in taus:
            for p in pulls:
                name = f"sd{s:g}_tau{tau if tau else 'static'}_pull{p:g}"
                grid.append((name, dict(base, home_mu=tgt["home_mu"], home_sd=s * tgt["home_sd"],
                                        home_band=0.10, home_tau=tau, pull=float(p))))
    for name, wr in grid:
        m = gen_metrics(rollouts(d, arm, wr, reps, seed, jobs), tgt)
        rows.append(dict(rule=name, home_mu=wr.get("home_mu"), home_sd=wr.get("home_sd"), home_band=wr.get("home_band"),
                         home_tau=wr.get("home_tau"), pull=wr.get("pull"), **m))
        print(f"  {name:28s} sd_ratio {m['sd_ratio']:.2f}  wacf {m['wacf']:+.3f}  hist_tv {m['hist_tv']:.3f}  "
              f"level {m['level']:.3f}  home rejects/bout {m['home_rejects_per_bout']:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    table = pd.DataFrame(rows)
    targets = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tgt.items() if k != "per_fish"}
    chosen = _choose(table, targets, base, tol_tv)
    cache.write_text(json.dumps(dict(table=json.loads(table.to_json(orient="records")), targets=targets, chosen=chosen), indent=1))
    _print(table, targets, chosen)
    return table, chosen


def _choose(table, targets, base, tol_tv):
    """The pick, per form (drifting / static home). Within the histogram tolerance when any rule is;
    otherwise the best-scoring rule of the whole sweep, flagged ``within_tol = False`` so the block reads it
    as a compromise. Drifting rules are scored on spread AND persistence, static ones on spread alone
    (a static home cannot move the acf, by construction)."""
    ctrl_tv = float(table.loc[table.rule == "control", "hist_tv"].iloc[0])
    cand = table[table.rule != "control"].copy()
    if "home_mu" not in cand.columns:            # sweep tables written before the keys were stored
        cand["home_mu"] = targets["home_mu"]
        cand["home_sd"] = cand.sd_scale * targets["home_sd"]
        cand["home_band"] = 0.10
    cand["within_tol"] = cand.hist_tv <= ctrl_tv + tol_tv
    chosen = {}
    for key, sub in (("drift", cand[cand.home_tau.notna()]), ("static", cand[cand.home_tau.isna()])):
        if not len(sub):
            continue
        sub = sub.copy()
        sub["score"] = (sub.sd_ratio - 1.0).abs()
        if key == "drift":
            sub["score"] += (sub.wacf - targets["wacf"]).abs() / abs(targets["wacf"])
        pool = sub[sub.within_tol] if sub.within_tol.any() else sub
        best = pool.sort_values("score").iloc[0]
        wr = dict(base, home_mu=float(best.home_mu), home_sd=float(best.home_sd), home_band=float(best.home_band),
                  home_tau=(None if pd.isna(best.home_tau) else float(best.home_tau)), pull=float(best.pull))
        chosen[key] = wr
        chosen[key + "_rule"] = str(best.rule)
        chosen[key + "_within_tol"] = bool(best.within_tol)
        chosen[key + "_readings"] = dict(sd_ratio=float(best.sd_ratio), wacf=float(best.wacf), hist_tv=float(best.hist_tv),
                                         control_hist_tv=ctrl_tv)
    return chosen


def _print(table, targets, chosen):
    pd.set_option("display.width", 220)
    print("HOME-RADIUS CALIBRATION (rollouts of the held-out fold models; 2 per fish; real targets from the 15 fish):")
    print(f"  real: per-fish median r/R mean {targets['home_mu']:.3f} sd {targets['home_sd']:.3f}; spread of per-fish "
          f"mean window radius {targets['sd_indiv']:.3f}; level {targets['level']:.3f}; window acf1 {targets['wacf']:+.3f}")
    print("  sd_ratio = spread of per-INDIVIDUAL mean radius / real (1 = matched); wacf = per-individual window acf1 "
          f"(real {targets['wacf']:+.2f}); hist_tv = total-variation distance of the pooled r/R histogram to the real one")
    cols = ["rule", "sd_ratio", "wacf", "hist_tv", "level", "draws_per_bout", "home_rejects_per_bout", "home_fallback_per_bout"]
    print(table[cols].round(3).to_string(index=False))
    for k in ("drift", "static"):
        if k in chosen:
            c, r = chosen[k], chosen[k + "_readings"]
            print(f"  chosen {k:6s}: {chosen[k + '_rule']}  (home_mu {c['home_mu']:.3f}, home_sd {c['home_sd']:.3f}, "
                  f"band {c['home_band']:.2f}, tau {c['home_tau']}, pull {c['pull']})  sd_ratio {r['sd_ratio']:.2f}  "
                  f"wacf {r['wacf']:+.2f}  hist_tv {r['hist_tv']:.3f} vs control {r['control_hist_tv']:.3f}"
                  + ("" if chosen[k + "_within_tol"] else "  -- OUTSIDE the histogram tolerance: a compromise, the block decides"))
        else:
            print(f"  chosen {k:6s}: none")


def radial_figure(ckpt_dir, arm, variants, reps=4, seed=1, jobs=10):
    """The three radial readings, real vs each rule in ``variants`` = {label: wall_reject dict}:
    the pooled r/R histogram, the per-individual mean radius (one dot per real fish / per rollout),
    and the per-individual window acf1. Returns the summary table."""
    import matplotlib.pyplot as plt
    d = _ckpt_dir(ckpt_dir)
    recs = _records()
    tgt = real_targets(recs)
    per = tgt["per_fish"]
    rows = [dict(rule="real", sd_indiv=tgt["sd_indiv"], sd_ratio=1.0, level=tgt["level"], wacf=tgt["wacf"], hist_tv=0.0)]
    data = {}
    for lab, wr in variants.items():
        rolls = rollouts(d, arm, wr, reps, seed, jobs)
        m = gen_metrics(rolls, tgt)
        rows.append(dict(rule=lab, **{k: m[k] for k in ("sd_indiv", "sd_ratio", "level", "wacf", "hist_tv")}))
        data[lab] = rolls
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    c = BINS[:-1] + np.diff(BINS) / 2
    ax[0].plot(c, tgt["hist"], "k-", lw=2, label="real")
    for lab, rolls in data.items():
        ax[0].plot(c, _hist(np.concatenate([g["r"] for reps_ in rolls.values() for g in reps_])), lw=1.5, label=lab)
    ax[0].set_xlabel("r / R"); ax[0].set_ylabel("fraction of bouts"); ax[0].set_title("pooled radial occupancy")
    ax[0].legend(frameon=False, fontsize=8)
    xs = ["real"] + list(data)
    ax[1].plot(np.zeros(len(per)) + np.random.default_rng(0).uniform(-0.12, 0.12, len(per)), per.mean_window_rR, "ko", ms=5)
    for i, (lab, rolls) in enumerate(data.items(), 1):
        v = np.array([_win_means(g["r"]).mean() for reps_ in rolls.values() for g in reps_])
        ax[1].plot(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, "o", ms=3.5, alpha=0.7)
    ax[1].set_xticks(range(len(xs))); ax[1].set_xticklabels(xs); ax[1].set_ylabel("mean window r/R of one individual")
    ax[1].set_title("one dot = one individual (a real fish / one rollout)\n" + "  ".join(f"{r.rule}: sd {r.sd_indiv:.3f}" for r in tab.itertuples()), fontsize=9)
    ax[2].bar(range(len(xs)), [tab.loc[tab.rule == x, "wacf"].iloc[0] for x in xs], color=["k"] + ["C%d" % i for i in range(len(data))])
    ax[2].set_xticks(range(len(xs))); ax[2].set_xticklabels(xs); ax[2].set_ylabel("window-mean radius acf1")
    ax[2].set_title("persistence of the radius across 60-bout windows")
    ax[2].axhline(0, color="0.7", lw=0.8)
    fig.tight_layout()
    print(tab.round(3).to_string(index=False))
    return tab
