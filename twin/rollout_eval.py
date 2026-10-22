"""LOCAL CELL — rollouts, Metric B and the C2ST AUCs from train_model1_colab.py's checkpoints.

No GPU. Each fold model rolls out ONLY the fish it never saw, so the union over folds gives a
full 15-fish panel with honest held-out status.

    python rollout_model1_local.py --ckpt-dir . --R 20
    python rollout_model1_local.py --R 4 --jobs 4 --smoke
    python rollout_model1_local.py --c2st-reps 0          # panel only, skip the AUCs
    python rollout_model1_local.py --importances          # + which channel gives the model away

TWO OBJECTS, TWO GRANULARITIES.
  METRIC B PANEL is per (model, fish): 17 named distances between the real and generated
  marginals/temporal statistics, averaged over ``--R`` rollouts. It says WHAT is off and by
  how much, in the observable's own units.
  C2ST is per ARM: pool every fish's windows and ask a RandomForest, GroupKFold BY FISH,
  whether it can tell real windows from generated ones in a way that generalises to a fish it
  never saw. 0.5 = indistinguishable, 1.0 = trivially detectable. It is an omnibus -- it can
  catch a defect no panel entry names, which is exactly why a panel alone is not enough. It
  has to be assembled here, across the arm's fold checkpoints, because one classifier needs
  all 15 fish.
  Three variants are reported: ``c2st`` (turn/symbol windows only), ``c2st_traj`` (+ step,
  interval, fish-to-dot distance and radial occupancy) and ``c2st_traj_eq``, the same rollout
  with each behavioural channel rank-mapped onto that fish's real marginal. eq << traj means
  the model has the right temporal structure and the wrong LEVELS; eq ~ traj means the
  structure itself is wrong and fixing the marginals would not help.
  ``c2st_neg`` is the protocol's own floor: real windows split into two random halves. It must
  come out near 0.5, or fish idiosyncrasy is leaking across the fold split and inflating
  everything above it.

SPEED. Rollouts are one-bout-at-a-time on a 128-unit net, so they are latency-bound, not
FLOP-bound -- a GPU would not help and threads fight each other. The knobs that work:
  --jobs   fish are independent, so fan them across processes (torch pinned to 1 thread each,
           otherwise N processes x N threads thrashes the cache and gets SLOWER)
  --R      panel replicates per fish. Use R=20 for a result; --smoke drops it for wiring.
  --c2st-reps  INDEPENDENT rollouts for the AUCs, on their own RNG stream so that changing
           this cannot move the panel. Rollout noise alone puts SD 0.0173 on c2st_traj at
           reps=1, i.e. SE of a two-arm difference 0.025; reps=8 gives ~0.012 and reps=24
           resolves 0.01. Total rollouts per fish are R + c2st_reps.

CRN. The rollout seed comes from the checkpoint and is combined per (fish, replicate), so two
models trained in separate runs still see matched random streams -- which is what makes the
arm-to-arm difference paired rather than seed noise.
"""
import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# One source of truth for every knob, shared by the CLI and the notebook entry point, so the
# two can never drift into different defaults.
DEFAULTS = dict(
    ckpt_dir="", R=20, jobs=min(15, max(1, (os.cpu_count() or 4) - 1)),
    out="panelB_model1.csv", c2st_out="c2st_model1.csv",
    c2st_reps=8, c2st_seed=0,
    importances=False, imp_repeats=3, imp_arms="",
    arms="", skip_panel=False, smoke=False, progress=True,
    # GENERATION-TIME overrides merged into every loaded model's cfg_feat, e.g. rej's
    # {"wall_reject": {...}}. Nothing is retrained: same weights, different rollout rule.
    # `tag` then suffixes every arm name and both output CSVs, so an overridden run can never
    # be mistaken for -- or overwrite -- the plain one.
    cfg_override=None, tag="",
    # DISTANCE-TO-STIM PROFILE: the Ablation_study plot in numbers, per arm -- programme rollouts
    # (own clock, dots re-anchored at every block onset), `profile_r` simulated fish per fold model.
    # Off with profile=False; skipped anyway when the panel is (skip_panel).
    profile=True, profile_r=3,
)

# Window features the INDIVIDUALITY metrics are reported on: one per behavioural channel, the
# window MEAN, so bf_sd reads directly as "how different are the fish" in that channel's units.
BF_CHANNELS = (("absdth_mean", "turn"), ("dist_mean", "dist"), ("dt_mean", "dt"),
               ("rad_mean", "rad"))


def merge_ckpt_dirs(dst, *srcs, arms=None):
    """Merge ladder checkpoint folders trained in SEPARATE Colab sessions (``--arms`` splits) into
    one folder ``dst`` that ``run()`` / ``model_figures`` read as a single ladder: copies every .pt
    and curve PNG, concatenates ``manifest_model1.json`` and ``likA_model1.csv``. ``arms`` keeps
    only those arm names (e.g. to pull one control arm out of another ladder).

        merge_ckpt_dirs("slow2_ladder_checkpoints",
                        "slow2_ladder_checkpoints_S3_adt_z2_drift_g3",
                        "slow2_ladder_checkpoints_S3_adt_z2_drift2_g3_S4_adt_z2_drift2_g3")
    """
    import shutil
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    keep = set(arms) if arms else None
    mani, lik = [], []
    for s in srcs:
        s = Path(s)
        ents = [e for e in json.loads((s / "manifest_model1.json").read_text())
                if keep is None or e["arm"] in keep]
        mani += ents
        lk = pd.read_csv(s / "likA_model1.csv")
        lik.append(lk if keep is None else lk[lk.arm.isin(keep)])
        for e in ents:
            shutil.copy2(s / e["file"], dst / e["file"])
            cv = s / f"curve_{e['arm']}.png"
            if cv.exists():
                shutil.copy2(cv, dst / cv.name)
    old = (json.loads((dst / "manifest_model1.json").read_text())
           if (dst / "manifest_model1.json").exists() else [])
    seen = {}
    for e in old + mani:                 # last one wins if an arm was trained twice
        seen[e["file"]] = e
    (dst / "manifest_model1.json").write_text(json.dumps(list(seen.values()), indent=1))
    if (dst / "likA_model1.csv").exists():
        lik.insert(0, pd.read_csv(dst / "likA_model1.csv"))
    pd.concat(lik, ignore_index=True).drop_duplicates().to_csv(dst / "likA_model1.csv", index=False)
    print(f"{dst}: {len(seen)} checkpoints, arms {sorted({e['arm'] for e in seen.values()})}")
    return dst




def _bar(done, total, t0, label=""):
    """One-line progress, overwritten in place. Works in a terminal and in a notebook cell.

    PADDED TO A FIXED WIDTH on purpose: '\\r' returns the cursor but erases nothing, so a short
    line drawn over a longer one leaves the tail behind and you get 'fish 5' + a stale '1'
    reading as 'fish 51'."""
    w = 24
    k = int(w * done / max(total, 1))
    el = time.time() - t0
    eta = el * (total - done) / done if done else 0.0
    line = (f"  [{'#' * k}{'.' * (w - k)}] {done}/{total}  {el:5.0f}s elapsed"
            f"  ~{eta:4.0f}s left   {label}")
    print("\r" + line.ljust(110)[:110], end="", flush=True)
    if done >= total:
        print(flush=True)


def _one(args_):
    """(ckpt, fish_id, R, seed, c2st_reps, skip_panel, cfg_override) -> (row, c2st windows)."""
    ckpt, fid, R, seed, creps, skip_panel, cfg_override = args_
    import torch
    torch.set_num_threads(1)                     # see SPEED above
    from twin import evaluate as E
    from twin import fit_MLP_border as MB
    from twin import fit_MLP_mix as MX
    from twin import generative_metrics as GM
    recs = _RECS if (_RECS := globals().get("_RECS")) else None
    if recs is None:
        recs = MB.build_border_records(E.load_bouts())
        globals()["_RECS"] = recs
    m = MX.load_mix(str(ckpt))
    if cfg_override:
        # MERGED, not replaced: every trained key (scales, stim_mode, feature layout) must stay
        # exactly as the checkpoint has it, or the feature rows stop matching the net.
        m._cfg_feat = {**m._cfg_feat, **cfg_override}
    if skip_panel:
        row = {}
    else:
        rng = np.random.default_rng([seed, int(fid)])   # CRN: same stream for every model
        row = GM.panel_for_fish(m, recs[fid], rng, R=R)
        row = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in row.items()} if isinstance(row, dict) else dict(row)
    row.update(fish=int(fid), ckpt=Path(ckpt).stem)
    # C2ST windows draw on their OWN stream ([seed, fish, rep], the ladder's convention), so
    # --c2st-reps can be changed without moving a single panel number.
    wins = (GM.c2st_windows_for_fish(m, recs[fid], fid, seed=seed, reps=creps)
            if creps else None)
    return row, wins


def _profile_one(args_):
    """(ckpt, fold index k, rollouts per model, cfg_override) -> raw bout table of k's programme
    rollouts (model_figures.programme_bouts); the parent assembles the arm and scores it."""
    ckpt, k, r, cfg_override = args_
    import torch
    torch.set_num_threads(1)
    from twin import fit_MLP_mix as MX
    from twin import model_figures as MF
    m = MX.load_mix(str(ckpt))
    if cfg_override:
        m._cfg_feat = {**m._cfg_feat, **cfg_override}
    b = MF.programme_bouts(m, k, r)
    return b, int(b.attrs.get("runaway", 0))      # (bouts, dropped runaway rollouts)


def _pool(fn, jobs, args, label_of):
    """Run fn over jobs (process pool when jobs > 1), results in job order, one progress bar."""
    t0 = time.time()
    out = [None] * len(jobs)
    if args.jobs > 1:
        # submit + as_completed rather than ex.map, ONLY so the bar can tick as each job lands
        # (map yields in submission order, so a slow first job hides all the finished ones).
        # The index carried in `futs` puts the results back in job order, which the C2ST
        # assembly relies on to pair each row with its windows.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
            for k, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                out[i] = fut.result()
                if args.progress:
                    _bar(k, len(jobs), t0, label_of(jobs[i]))
    else:
        for i, j in enumerate(jobs):
            out[i] = fn(j)
            if args.progress:
                _bar(i + 1, len(jobs), t0, label_of(j))
    return out


def run(**kw):
    """NOTEBOOK ENTRY POINT -- run in a normal cell, no ``!``, output lands in the cell.

        import sys; sys.path.insert(0, "Model")          # or wherever this file lives
        from twin.rollout_eval import run
        panel, c2st = run(ckpt_dir="model1_checkpoints", R=20)

        # the same checkpoints with a generation-time rule added (e.g. rej's wall rejection);
        # the tag keeps its arms and CSVs separate from the plain run
        panel, c2st = run(ckpt_dir="model1_checkpoints", cfg_override=REJ, tag="rej")

    Returns ``(panel_df, c2st_df)`` -- the same two frames written to disk, so the arm ranking
    can be re-sliced in the notebook without re-reading the CSVs. ``c2st_df`` is None when
    ``c2st_reps=0``. Keyword names are the CLI flags with dashes as underscores; see DEFAULTS.

    WHY THIS EXISTS rather than ``!python``: a ``!`` cell spawns a child whose stdout is a pipe,
    so Python block-buffers it and NOTHING appears until the run ends -- 20 minutes of blank
    cell, and the progress bar would be invisible. In-process, prints reach the cell live.
    """
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown option(s) {sorted(unknown)}; valid: {sorted(DEFAULTS)}")
    return _run(argparse.Namespace(**{**DEFAULTS, **kw}))




def _run(args):
    if args.smoke:
        args.R = 2
        args.c2st_reps = min(args.c2st_reps, 2)
    if args.skip_panel and not args.c2st_reps:
        raise ValueError("skip_panel with c2st_reps=0 would compute nothing at all")

    import torch
    from twin import fit_MLP_mix as MX
    # Colab pickles under numpy 2 ("numpy._core"); a local numpy 1 cannot find that module and
    # torch.load dies. The WORKERS get the shim for free via load_mix, but this parent-side read
    # of `rollout_seed` does not -- so the run failed before a single rollout started, purely on
    # which interpreter happened to be first on PATH.
    MX._numpy2_unpickle_compat()
    d = Path(args.ckpt_dir)
    mani = json.loads((d / "manifest_model1.json").read_text())
    # fold checkpoints only -- the all-data model has no held-out fish, so it is a generation
    # artifact and scoring it would be an in-sample number masquerading as a panel row.
    # Restrict to named arms. The C2ST still pools that arm's 15 fish across its folds, so a
    # single-arm run is a complete, honest AUC for that arm -- just 1/12 of the rollouts. What
    # it CANNOT give you is the cross-arm ranking, which needs them all.
    ov = args.cfg_override or {}
    if isinstance(ov, str):
        ov = json.loads(ov)
    tag = str(args.tag or "").strip("_")
    if ov and not tag:
        # Same arm names and same CSV paths as the plain run: the override would silently
        # overwrite it, and nothing in the files would say which rule produced them.
        raise ValueError(f"cfg_override {sorted(ov)} needs a tag (e.g. tag='rej') so its arms "
                         "and CSVs cannot be confused with, or overwrite, the plain run's")
    # accept an arm either bare ("S3_dta") or already tagged ("S3_dta_rej")
    want = {(re.sub(rf"_{re.escape(tag)}$", "", x) if tag else x)
            for x in (y.strip() for y in args.arms.split(",")) if x}
    jobs = []
    for e in mani:
        if e.get("insample_only") or not e["heldout"]:
            continue
        if want and re.sub(r"_fold\d+$", "", Path(e["file"]).stem) not in want:
            continue
        blob = torch.load(d / e["file"], map_location="cpu", weights_only=False)
        seed = int(blob.get("rollout_seed", 0))
        jobs += [(d / e["file"], f, args.R, seed, args.c2st_reps, args.skip_panel, ov)
                 for f in e["heldout"]]
    if not jobs:
        raise ValueError(f"no checkpoints matched arms={args.arms!r}; the manifest has "
                         + ", ".join(sorted({re.sub(r'_fold\d+$', '', Path(e['file']).stem)
                                             for e in mani if not e.get('insample_only')})))
    if tag:
        args.out = Path(args.out).stem + f"_{tag}.csv"
        args.c2st_out = Path(args.c2st_out).stem + f"_{tag}.csv"
        print(f"generation-time override {sorted(ov)} -> arms tagged '_{tag}', "
              f"writing {args.out} / {args.c2st_out}")
    if want:
        # A subset run must NOT clobber the full-sweep CSVs with a partial table -- that file is
        # the cross-arm comparison, and silently shrinking it to one arm is how a later read
        # concludes the sweep was never run.
        sfx = "_" + "_".join(sorted(want))
        args.out = Path(args.out).stem + sfx + ".csv"
        args.c2st_out = Path(args.c2st_out).stem + sfx + ".csv"
        print(f"arms filter {sorted(want)} -> writing {args.out} / {args.c2st_out}")
    per = (0 if args.skip_panel else args.R) + args.c2st_reps
    print(f"{len(jobs)} (checkpoint, fish) rollout jobs, "
          f"R={'skipped' if args.skip_panel else args.R}, "
          f"c2st_reps={args.c2st_reps}, jobs={args.jobs}  "
          f"({len(jobs) * per:,} rollouts total)")
    from twin import generative_metrics as GM
    print("stimulus: " + ("RE-ANCHORED to each simulated fish at block onset (the rig's rule)"
                          if GM.REANCHOR_EVAL and GM.eval_program() is not None else
                          "REPLAYED from the real fish -- block-onset placement artefact PRESENT"))

    t0 = time.time()
    out = _pool(_one, jobs, args, lambda j: f"rollouts · {j[0].stem} fish {j[1]}")
    rows = [r for r, _ in out]
    p = pd.DataFrame(rows)
    # `arm` is the whole checkpoint stem minus the fold, e.g. "S1_adt". Split it so the ordering
    # sweep can be read down a column; a checkpoint from before the sweep has no order suffix and
    # keeps arm == rep with order NaN, so the old artifacts still summarise.
    p["arm"] = p.ckpt.str.replace(r"_fold\d+$", "", regex=True) + (f"_{tag}" if tag else "")
    _sp = p.arm.str.extract(r"^(?P<rep>.+?)_(?P<order>[adt]{3})(?:_.+)?$")
    p["rep"] = _sp["rep"].fillna(p.arm)
    p["order"] = _sp["order"]
    from twin import generative_metrics as GM
    if args.skip_panel:
        print(f"panel skipped ({time.time()-t0:.0f}s of C2ST rollouts); "
              f"{args.out} left as it was")
    else:
        p.to_csv(d / args.out, index=False)
        print(f"wrote {args.out}  ({len(p)} rows) in {time.time()-t0:.0f}s")
        keep = [c for c in GM.PANEL_METRICS if c in p.columns]
        print("\nMetric B, mean over held-out fish (lower = closer to real):")
        print(p.groupby("arm")[keep].mean().round(4).T.to_string())
        print("\narms ranked by the load-bearing stimulus metrics:")
        lb = [c for c in ("following_gap_h", "following_gap_static_h", "tuning_mse",
                          "approach_gap") if c in p.columns]
        if lb:
            print(p.groupby("arm")[lb].mean().round(4).sort_values(lb[0]).to_string())
            if p.order.notna().any():
                # The sweep's own question: does the emission ordering move Metric B at all, once
                # averaged over the representations? Metric A can prefer an ordering that
                # generates no better -- likelihood and rollout panel are different objectives.
                print("\nordering, averaged over representations:")
                print(p.groupby("order")[lb].mean().round(4).sort_values(lb[0]).to_string())
        if args.profile:
            p.attrs["profile"] = profile(sorted({j[0] for j in jobs}), ov, tag, d, args)

    c = c2st(p, [w for _, w in out], d, args) if args.c2st_reps else None
    return p, c


def profile(ckpts, ov, tag, d, args):
    """DISTANCE-TO-STIM PROFILE, one row per arm: the Ablation_study distance-to-dots plot in
    numbers (model_figures.profile_metrics) on programme rollouts of the arm's fold models --
    the rig's rule, own clock -- against the 15 real fish. Written next to the panel as
    profile_model1[_tag].csv (per-condition rows); returns the per-arm summary table.

    Why it is not derived from the panel's rollouts: those follow each real fish's record and
    are averaged per fish over the whole block and every dot condition, which is exactly what
    hides the transient (dip amplitude / latency), the static drift-away rate and the dot-count
    grading this section reports."""
    from twin import model_figures as MF
    by_arm = {}
    for ck in ckpts:
        by_arm.setdefault(re.sub(r"_fold\d+$", "", ck.stem), []).append(ck)
    jobs = [(ck, k, args.profile_r, ov) for cks in by_arm.values() for k, ck in enumerate(cks)]
    n_fish = args.profile_r * max(len(v) for v in by_arm.values())
    print(f"\ndistance-to-stim profile: {len(jobs)} fold models x {args.profile_r} programme rollouts "
          f"({n_fish} simulated fish per arm)")
    t0 = time.time()
    parts = _pool(_profile_one, jobs, args, lambda j: f"profile · {j[0].stem}")
    real = MF.real_dot_profile()
    rows, full = [], []
    for arm, cks in by_arm.items():
        name = arm + (f"_{tag}" if tag else "")
        sel = [b for (b, _), j in zip(parts, jobs) if j[0] in cks]
        n_run = sum(nr for (_, nr), j in zip(parts, jobs) if j[0] in cks)
        gen = MF.finish_profile(pd.concat(sel, ignore_index=True), len(cks), args.profile_r,
                                progress=False)
        m = MF.profile_metrics(real, gen)
        full.append(m.drop(columns=["_pr", "_pg"]).reset_index().assign(arm=name))
        rows.append({"arm": name, **MF.profile_summary(m), "runaway": int(n_run)})
    s = pd.DataFrame(rows).set_index("arm")[MF.PROFILE_SUMMARY + ["runaway"]]
    if s.runaway.any():
        print("  RUNAWAY programme rollouts (intervals collapsing; dropped from the profile, a model "
              "defect of that fold): " + ", ".join(f"{a} {int(v)}" for a, v in s.runaway.items() if v))
    out = Path(args.out).with_name(Path(args.out).name.replace("panelB", "profile"))
    pd.concat(full, ignore_index=True).to_csv(d / out, index=False)
    print(f"wrote {out}  ({len(s)} arms) in {time.time()-t0:.0f}s")

    r = s.iloc[0]
    print("\nDISTANCE-TO-STIM PROFILE, per arm (mm; the Ablation_study plot in numbers, programme rollouts):"
          "\n  moving 2-8 dots: err = mean |gen - real| over 0-60 s, also in real-SEM units "
          "(noise floor: 8 real fish vs 15 = 1.3 mm / 0.8 SEM);"
          f"\n  dip = approach amplitude within 20 s (real {r.dip_real:.1f} mm at {r.lat_real:.0f} s); "
          f"plateau = 20-60 s level (real {r.plateau_real:.1f}); 1-dot plateau (real {r.d1_plateau_real:.1f});"
          f"\n  grading = plateau(1 dot) - plateau(8 dots) (real {r.grading_real:+.1f}); "
          f"static: err and drift-away 0-20 s in mm/s (real {r.drift_real:.2f})")
    show = ["mv_err_mm", "mv_err_sem", "dip_gen", "lat_gen", "plateau_gen", "d1_plateau_gen",
            "grading_gen", "st_err_mm", "st_err_sem", "drift_gen"]
    print(s[show].rename(columns=lambda c: c.replace("_gen", "")).round(2)
           .sort_values("mv_err_sem").to_string())
    order = s.index.to_series().str.extract(r"^.+?_(?P<order>[adt]{3})(?:_.+)?$")["order"]
    if order.notna().sum() > 1 and order.nunique() > 1:
        print("\nordering, averaged over representations:")
        print(s[show].groupby(order.values).mean().rename(columns=lambda c: c.replace("_gen", ""))
               .round(2).sort_values("mv_err_sem").to_string())
    return s


def c2st(p, wins, d, args):
    """Assemble the per-fish windows into one classifier PER ARM and report the AUCs.

    Per arm, not per checkpoint: the C2ST asks whether the difference generalises to an unseen
    FISH, so its GroupKFold needs every fish. Each fish's windows come from the fold model that
    held it out, so the union over an arm's folds is a full 15-fish, fully held-out panel --
    the same construction the Metric B table above uses, just pooled instead of averaged.
    """
    from twin import generative_metrics as GM
    t0 = time.time()
    _tg = str(getattr(args, "tag", "") or "").strip("_")
    imp_arms = {(x if not _tg or x.endswith(f"_{_tg}") else f"{x}_{_tg}")
                for x in (y.strip() for y in args.imp_arms.split(",")) if x}
    # arm -> kind -> [ {fish: (real, gen)} ] * reps
    by_arm = {}
    for (_, row), w in zip(p.iterrows(), wins):
        if w is None:
            continue
        a = by_arm.setdefault(row["arm"], {k: [{} for _ in range(args.c2st_reps)] for k in w})
        for kind, pairs in w.items():
            for rep, pair in enumerate(pairs):
                a[kind][rep][int(row["fish"])] = pair

    rows, imps, raw_imps = [], [], {}
    for ai, (arm, kinds) in enumerate(sorted(by_arm.items()), 1):
        if args.progress:
            _bar(ai - 1, len(by_arm), t0, f"c2st · {arm}")
        r = dict(arm=arm)
        # `traj_int` = the STEADY-STATE twin of `traj` (GM.INTERIOR: real windows well inside a
        # run against generated windows after the warm-up), from the same rollouts; its columns
        # carry the `_int` suffix throughout.
        for kind, name in (("turn", "c2st"), ("traj", "c2st_traj"), ("traj_eq", "c2st_traj_eq"),
                           ("traj_int", "c2st_traj_int")):
            if kind not in kinds:
                continue
            mean, sd, per = GM.c2st_auc_replicates(kinds[kind], seed=args.c2st_seed)
            r[name], r[name + "_sd"] = mean, sd
            # SE of the MEAN over replicates -- what an arm-to-arm difference must clear.
            r[name + "_se"] = sd / np.sqrt(len(per)) if len(per) > 1 else np.nan
        for kind, sfx in (("traj", ""), ("traj_int", "_int")):
            if kind not in kinds:
                continue
            # The protocol floor, on this arm's real windows. Identical across arms by
            # construction (the real side does not depend on the model), so it is a check, not a
            # comparison.
            r["c2st_neg" + sfx] = GM.c2st_negative_control(kinds[kind], seed=args.c2st_seed)
            # DISPERSION, next to the AUC and never without it. A model can lower its AUC by
            # generating LESS variety -- the kin ladder's winning rung once bought -0.0407 by
            # halving the distance dispersion, and nothing reported at the time could see it.
            # 1.0 = matched, <1 = too typical, >1 = too wild. Free: same windows, no rollouts.
            disp = GM.dispersion_report(kinds[kind])
            for g, v in (disp.get("var_ratio") or {}).items():
                r[f"var_{g.replace('-', '_')}{sfx}"] = float(v)
            # INDIVIDUALITY, in the feature's own units (GM.between_fish_metrics). The Metric B
            # panel is computed per fish and then averaged, so it is BLIND to whether the 15
            # generated fish differ from each other at all: a model that makes every fish the
            # same fish can score well on every panel row. These cannot be fooled that way:
            #   bf_sd  SD across fish of the per-fish window mean  ("how different are the fish")
            #   wf_sd  mean within-fish SD across windows           ("how variable is one fish")
            #   bf_w1  W1 between real and gen sets of per-fish means (shape -- e.g. an outlier)
            # Gen side averaged over replicates: a per-fish latent redraws z every replicate, so
            # a single replicate is noisy. The real side is replicate-invariant.
            _dim = next(iter(kinds[kind][0].values()))[0].shape[1]
            _nm = GM.traj_names_for_dim(_dim)
            bfs = [b for b in (GM.between_fish_metrics(wr, _nm) for wr in kinds[kind]) if b]
            if bfs:
                for col, ch in BF_CHANNELS:
                    if col not in _nm:
                        continue
                    j = _nm.index(col)
                    r[f"bf_sd_real_{ch}{sfx}"] = float(bfs[0]["bf_sd"]["real"][j])
                    r[f"bf_sd_gen_{ch}{sfx}"] = float(np.mean([b["bf_sd"]["gen"][j] for b in bfs]))
                    r[f"wf_sd_real_{ch}{sfx}"] = float(bfs[0]["wf_sd"]["real"][j])
                    r[f"wf_sd_gen_{ch}{sfx}"] = float(np.mean([b["wf_sd"]["gen"][j] for b in bfs]))
                    r[f"bf_w1_{ch}{sfx}"] = float(np.mean([b["bf_w1"][j] for b in bfs]))
            if sfx == "_int":
                # how much of the real side survives the selection (replicate 0): windows in
                # total and fish with at least one -- a pause-heavy fish can have none, and the
                # classifier then simply has no windows of that fish on either side.
                r["n_win_real_int"] = float(sum(len(a) for a, _ in kinds[kind][0].values()))
                r["n_win_real"] = float(sum(len(a) for a, _ in kinds["traj"][0].values()))
                r["n_win_gen_int"] = float(sum(len(g) for _, g in kinds[kind][0].values()))
                r["n_win_gen"] = float(sum(len(g) for _, g in kinds["traj"][0].values()))
                r["n_fish_int"] = float(sum(len(a) > 0 for a, _ in kinds[kind][0].values()))
        rows.append(r)
        if args.importances and (not imp_arms or arm in imp_arms):
            # WHICH channel gives the model away. Permutation importance = the drop in held-out
            # AUC when that feature is shuffled, so it is an effect size on the reported metric;
            # grouped because the members of a channel are correlated moments of one signal.
            w0 = kinds["traj"][0]
            dim = next(iter(w0.values()))[0].shape[1]
            names = GM.traj_names_for_dim(dim)
            if args.progress:
                print("\r" + f"  permutation importances for {arm} (slow) ...".ljust(110),
                      end="", flush=True)
            with warnings.catch_warnings():
                # sklearn's permutation_importance hands `delayed` to a bare joblib.Parallel and
                # warns once PER FEATURE PER FOLD about it -- hundreds of identical lines that
                # bury the actual report. Nothing here reads sklearn thread-local config, so the
                # warning has no consequence; silence it for this call only.
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils")
                imp = GM.c2st_importances(w0, names, seed=args.c2st_seed,
                                          n_repeats=args.imp_repeats).clip(lower=0)
            tot = float(imp.sum()) or 1.0
            raw_imps[arm] = imp            # the PER-FEATURE series, before any grouping
            # reindex, not [], so a width fallback inside c2st_importances (which renames the
            # index to f0..fN) degrades to zeros instead of a KeyError three hours in.
            imps.append(dict(arm=arm, **{g: float(imp.reindex(m).fillna(0.0).sum()) / tot
                                         for g, m in GM.traj_groups(names).items() if m}))

    if args.progress:
        _bar(len(by_arm), len(by_arm), t0, "c2st done")
    c = pd.DataFrame(rows)
    _sp = c.arm.str.extract(r"^(?P<rep>.+?)_(?P<order>[adt]{3})(?:_.+)?$")
    c["rep"] = _sp["rep"].fillna(c.arm); c["order"] = _sp["order"]
    c.to_csv(d / args.c2st_out, index=False)
    print(f"wrote {args.c2st_out} in {time.time()-t0:.0f}s")

    show = ["c2st", "c2st_traj", "c2st_traj_se", "c2st_traj_eq", "c2st_neg"]
    print("\nC2ST AUC, GroupKFold by fish (0.5 = indistinguishable, lower = better):")
    print(c.set_index("arm")[show].round(4).sort_values("c2st_traj").to_string())
    neg = c.c2st_neg.mean()
    print(f"\nnegative control (real vs real): {neg:.4f}"
          + ("   -- at the 0.5 floor, so the AUCs above are model signal"
             if neg < 0.60 else
             "   -- WELL ABOVE 0.5: fish idiosyncrasy is leaking across the fold split and "
             "every AUC above is inflated. Treat the ranking, not the level."))
    vc = [x for x in c.columns if x.startswith("var_")]
    if vc:
        print("\ndispersion, gen/real (1.0 = matched, <1 too typical, >1 too wild) -- READ WITH "
              "THE AUC:\n  a lower AUC bought by a lower var_ratio is the model generating less "
              "variety, not more realism.")
        print(c.set_index("arm")[vc].round(3).sort_values(vc[0]).to_string())
    chs = [x for _, x in BF_CHANNELS if f"bf_sd_gen_{x}" in c.columns]
    if chs:
        t = pd.DataFrame({"arm": c.arm})
        for x in chs:
            t[f"bf_{x}"] = c[f"bf_sd_gen_{x}"] / c[f"bf_sd_real_{x}"]
        for x in chs:
            t[f"wf_{x}"] = c[f"wf_sd_gen_{x}"] / c[f"wf_sd_real_{x}"]
        print("\nINDIVIDUALITY, gen/real per channel window mean (1.0 = matched). bf = how different "
              "the fish are\n  from each other, wf = how variable one fish is. The panel above cannot "
              "see bf at all: it is averaged per fish.")
        print(t.set_index("arm").round(3).sort_values(f"bf_{chs[0]}", ascending=False).to_string())
        print("  real, in units:  " + "   ".join(
            f"{x} bf_sd {c[f'bf_sd_real_{x}'].iloc[0]:.3g} wf_sd {c[f'wf_sd_real_{x}'].iloc[0]:.3g}"
            for x in chs))
        print("bf_w1 (real vs gen per-fish means, channel units, lower = better):")
        print(c.set_index("arm")[[f"bf_w1_{x}" for x in chs]].round(3)
              .sort_values(f"bf_w1_{chs[0]}").to_string())
    if "c2st_traj_int" in c.columns:
        # STEADY STATE: the same rollouts, real windows >= 120 bouts into a run and >= 60 before
        # it ends against generated windows from t >= 120 (GM.INTERIOR). The difference between
        # the two AUC columns is what the post-pause transient -- which a rollout never produces
        # but a third of the real windows sit in -- costs the model on the standard reading.
        ic = [x for x in ("c2st_traj", "c2st_traj_int", "c2st_traj_int_se", "c2st_neg_int",
                          "var_dt_int", "var_distance_int", "var_turn_int") if x in c.columns]
        print(f"\nSTEADY STATE (interior windows: real >= {GM.INTERIOR[0]} bouts into a run and "
              f">= {GM.INTERIOR[1]} before it ends, generated from t >= {GM.INTERIOR[0]} and "
              f"subsampled to the same gen:real ratio; real windows {c.n_win_real_int.iloc[0]:.0f} of "
              f"{c.n_win_real.iloc[0]:.0f}, generated {c.n_win_gen_int.iloc[0]:.0f} of "
              f"{c.n_win_gen.iloc[0]:.0f}, {c.n_fish_int.iloc[0]:.0f} of 15 fish have one):")
        print(c.set_index("arm")[ic].round(4).sort_values("c2st_traj_int").to_string())
        chs_i = [x for _, x in BF_CHANNELS if f"bf_sd_gen_{x}_int" in c.columns]
        if chs_i:
            t = pd.DataFrame({"arm": c.arm})
            for x in chs_i:
                t[f"bf_{x}"] = c[f"bf_sd_gen_{x}_int"] / c[f"bf_sd_real_{x}_int"]
            for x in chs_i:
                t[f"wf_{x}"] = c[f"wf_sd_gen_{x}_int"] / c[f"wf_sd_real_{x}_int"]
            print("  individuality on the interior windows, gen/real (bf / wf as above):")
            print(t.set_index("arm").round(3).sort_values(f"bf_{chs_i[0]}", ascending=False).to_string())
            print("  real, in units:  " + "   ".join(
                f"{x} bf_sd {c[f'bf_sd_real_{x}_int'].iloc[0]:.3g} wf_sd {c[f'wf_sd_real_{x}_int'].iloc[0]:.3g}"
                for x in chs_i))
    if c.order.notna().any():
        print("\nordering, averaged over representations:")
        print(c.groupby("order")[["c2st", "c2st_traj", "c2st_traj_eq"]]
              .mean().round(4).sort_values("c2st_traj").to_string())
    # An ordering is only worth calling better if it clears the replicate noise. Meaningless
    # with a single arm (best == worst), so only print it when there is a comparison to make.
    if len(c) > 1:
        best, worst = c.c2st_traj.idxmin(), c.c2st_traj.idxmax()
        gap = c.c2st_traj[worst] - c.c2st_traj[best]
        se = float(np.hypot(c.c2st_traj_se[best], c.c2st_traj_se[worst]))
        print(f"\nbest {c.arm[best]} {c.c2st_traj[best]:.4f} vs worst {c.arm[worst]} "
              f"{c.c2st_traj[worst]:.4f}: gap {gap:.4f} +/- {se:.4f} "
              f"({'resolved' if gap > 2 * se else 'NOT resolved at this --c2st-reps'})")
    if imps:
        print("\nshare of the traj-C2ST signal by channel (permutation importance, replicate 0):")
        print(pd.DataFrame(imps).set_index("arm").round(3).to_string())
    # The ungrouped per-feature series ride on .attrs so plot_importances can use them without
    # a second rollout pass. .attrs does not survive to_csv, which is why the CSV keeps the
    # grouped view and the notebook gets the raw one.
    c.attrs["importances"] = raw_imps
    return c


