"""The measurements every thesis model gets, one function each, and ``evaluate`` to run them all on a unit.

Each function takes a unit key (``registry``) and prints / draws what the models notebooks show:

    lik          held-out likelihood per head, bits per bout (lower = better), mean over the 15 held-out fish
    auc          the window classifier's AUC (standard and steady-state) and the spread of simulated individuals
                 relative to real fish, 8 rollouts per held-out fish
    auc2         the same classifier under two fairer windowings of the same rollouts: one generated window per real
                 one, and matched to the real runs (the generated windows cut at the real fish's own window
                 positions, 1:1), with all 39 features and without the four radial ones
    indv         every window statistic of radial position, interval, |turn| and displacement, real against simulated
    prof         distance to the dots across the block, real fish next to programme rollouts
    matched      the same profile with 50 simulated fish matched to the real block composition: dip, latency,
                 plateau, grading, moving and static errors
    attr         attraction (stimulus-blind null minus the 20-60 s plateau) per dot count
    tow          turn toward the moving dots by dot count, teacher-forced on the held-out fish
    hgap         each head against real fish, per stimulus condition and block phase

``evaluate(key)`` (pipelines/evaluate.py) saves every output in ``paths.EVAL / f"M{key}.json"`` and the headline
numbers, which every model table reads, in ``summary_M{key}.json``. Evaluation is deterministic: re-running a unit
reproduces its summary exactly.
"""
import base64
import contextlib
import io
import json
import time
import traceback

import numpy as np
import pandas as pd

from twin import paths, registry

pd.set_option("display.width", 250)
REJ = {"wall_reject": dict(registry.REJ)}       # generation-time wall containment, as a config override


def _u(key):
    u = registry.unit(str(key))
    return u, str(registry.ckpt_dir(u)), registry.arm(u), (REJ["wall_reject"] if u["rej"] else None)


@contextlib.contextmanager
def _streams(u):
    """Programme rollouts on per-block random streams (every unit but the seed-0 M1-M3, on the earlier stream)."""
    from twin.model_figures import block_seeding
    with block_seeding(bool(u["block_seed"])):
        yield


def lik(key):
    u, d, arm, rej = _u(key)
    t = pd.read_csv(f"{d}/likA_model1.csv").drop_duplicates(["arm", "fish"])
    s = t[t.arm == arm][["bits_per_bout", "bits_per_bout_dist", "bits_per_bout_dt", "bits_per_bout_joint"]].mean()
    s.index = ["turn", "displacement", "interval", "joint"]
    print(s.round(3).to_string())
    return s


def auc(key):
    from twin.rollout_eval import run
    u, d, arm, rej = _u(key)
    kw = dict(cfg_override=REJ, tag="rej") if rej else {}
    return run(ckpt_dir=d, arms=arm, R=20, jobs=4, progress=False, **kw)


def auc2(key):
    from twin.c2st_protocol import matched_c2st_table
    u, d, arm, rej = _u(key)
    with contextlib.redirect_stdout(io.StringIO()):
        t = matched_c2st_table(d, arm, rej=rej, jobs=4)
    t = t[(t.windows == "all") & t.generated.isin(["plain tiling, 1 generated window per real one",
                                                   "matched to the real runs (same bouts, 1:1)"])]
    t = t.set_index("generated")[["all features", "all features se", "no radial", "no radial se"]]
    print(t.round(3).to_string())
    return t


def indv(key):
    from twin.model_figures import window_stats_grid
    u, d, arm, rej = _u(key)
    return window_stats_grid(d, arm, rej=rej, reps=4)


def prof(key):
    from twin.model_figures import dot_profile
    u, d, arm, rej = _u(key)
    with _streams(u):
        return dot_profile(d, arm, rej=rej)


def matched(key):
    from twin.input_ablation import profile_variants_matched
    u, d, arm, rej = _u(key)
    with _streams(u):
        v, _ = profile_variants_matched(d, arm, {f"Model {key}": {"arm": arm}}, rej=rej, r_per_model=10,
                                        show_raw=False)
    return v


def attr(key):
    from twin.model_figures import attraction_table
    u, d, arm, rej = _u(key)
    with _streams(u):
        a, _ = attraction_table(d, [arm], rej=rej, r_per_model=6)
    return a


def tow(key):
    from twin.model_figures import toward_by_n
    u, d, arm, rej = _u(key)
    with contextlib.redirect_stdout(io.StringIO()):
        t = toward_by_n(d, [arm], rej=rej)
    print(t.round(2).to_string())
    return t


def hgap(key):
    from twin.model_figures import head_gaps
    u, d, arm, rej = _u(key)
    return head_gaps({f"Model {key}": (d, arm)}, rej=rej)[1]


CELLS = ("lik", "auc", "auc2", "indv", "prof", "matched", "attr", "tow", "hgap")


def summary(key, res):
    """The headline numbers of unit ``key`` from its measurements (None where one failed)."""
    u, d, arm, rej = _u(key)
    s = {"model": f"Model {key}"}
    with contextlib.suppress(Exception):
        s.update({f"lik_{c}": round(float(v), 4) for c, v in res["lik"].items()})
    with contextlib.suppress(Exception):
        c = res["auc"][1]
        row = c[c.arm.isin([arm, arm + "_rej"])].iloc[0]
        s.update(auc_std=round(float(row.c2st_traj), 4), auc_steady=round(float(row.c2st_traj_int), 4))
        for ch in ("turn", "dist", "dt", "rad"):
            s[f"bf_{ch}"] = round(float(row[f"bf_sd_gen_{ch}"] / row[f"bf_sd_real_{ch}"]), 3)
            s[f"wf_{ch}"] = round(float(row[f"wf_sd_gen_{ch}"] / row[f"wf_sd_real_{ch}"]), 3)
    with contextlib.suppress(Exception):
        t = res["auc2"]
        for row, tag in (("plain tiling, 1 generated window per real one", "1per"),
                         ("matched to the real runs (same bouts, 1:1)", "matched")):
            s[f"auc_{tag}"] = round(float(t.loc[row, "all features"]), 4)
            s[f"auc_{tag}_norad"] = round(float(t.loc[row, "no radial"]), 4)
    with contextlib.suppress(Exception):
        v = res["matched"]
        r = v.loc[[i for i in v.index if "[matched]" in str(i)][0]]
        s.update({c: round(float(r[c]), 3) for c in ("dip", "lat", "plateau", "d1_plateau", "grading", "mv_err_sem",
                                                     "st_err_sem", "static_drift")})
    with contextlib.suppress(Exception):
        a = res["attr"]
        col = [c for c in a.columns if c[0] != "real" and c[1] == "attraction"][0]
        for cond, n in (("d1s3.5", 1), ("d2s3.5", 2), ("d4s3.5", 4), ("d6s3.5", 6), ("d8s3.5", 8)):
            s[f"attr_N{n}"] = round(float(a.loc[cond, col]), 2)
    with contextlib.suppress(Exception):                        # the head analysis' three key gaps (model - real)
        h = res["hgap"]
        for k, cond, ph, q in (("gap_ahead_N8", "d8s3.5", "20-60 s", "ahead"),
                               ("gap_toward_N2", "d2s3.5", "20-60 s", "toward"),
                               ("gap_onset_N8", "d8s3.5", "0-4 s", "toward")):
            r = h[(h.cond == cond) & (h.phase == ph) & (h.qty == q)]
            s[k] = round(float(r.gap.iloc[0]), 2) if len(r) else None
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in s.items()}


def _capture(fn, key):
    """Run one measurement; its printed text and figures as notebook outputs."""
    import matplotlib.pyplot as plt
    buf, outs, res = io.StringIO(), [], None
    try:
        with contextlib.redirect_stdout(buf):
            res = fn(key)
    except Exception:
        buf.write("\n" + traceback.format_exc())
    if buf.getvalue():
        outs.append(dict(output_type="stream", name="stdout", text=buf.getvalue().splitlines(True)))
    for num in plt.get_fignums():
        png = io.BytesIO()
        plt.figure(num).savefig(png, format="png", dpi=100, bbox_inches="tight")
        outs.append(dict(output_type="display_data", metadata={},
                         data={"image/png": base64.b64encode(png.getvalue()).decode(), "text/plain": ["<Figure>"]}))
    plt.close("all")
    return res, outs


def evaluate(key, out_dir=None):
    """Run every measurement on unit ``key``; save the outputs (M<key>.json) and the summary (summary_M<key>.json)."""
    import matplotlib
    matplotlib.use("Agg")
    key = str(key)
    out = paths.EVAL if out_dir is None else out_dir
    out.mkdir(parents=True, exist_ok=True)
    res, cells = {}, []
    for cid in CELLS:
        t0 = time.time()
        res[cid], outs = _capture(globals()[cid], key)
        cells.append(dict(cid=cid, source=f"{cid}({key!r})", outputs=outs))
        print(f"  {key} {cid}: {time.time() - t0:.0f} s", flush=True)
    (out / f"M{key}.json").write_text(json.dumps(dict(k=key, arm=registry.arm(key), cells=cells)), encoding="utf-8")
    s = summary(key, res)
    (out / f"summary_M{key}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s
