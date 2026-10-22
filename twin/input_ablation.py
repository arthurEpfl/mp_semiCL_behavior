"""Rollout-time INPUT ABLATION for the fitted models: switch stimulus input groups off in the
generator's own feature row, same weights, no retraining.

The closed-loop generator (``fit_MLP_joint.generate_mix_joint_loop``) rebuilds the stimulus columns
per bout through three module-level functions -- ``fit_MLP.shoal_channels`` (the 18 shoal columns),
``fit_MLP.static_channels`` (the 8 static-dot columns) and ``fit_MLP_border.align_feature`` (the
aligned-and-far column) -- looked up by module attribute at call time. ``zero_inputs`` rebinds them
for the duration of a ``with`` block, zeroing the named columns, so every rollout inside the block
sees the model as if those inputs did not exist. The record-side builders call the same functions,
so a teacher-forced score inside the block is ablated identically.

    with zero_inputs(shoal_cols=("gns1", "gns2", "gns3", "gns4"), align=True):
        gen = generated_dot_profile(ckpt_dir, arm, rej, r_per_model)

    profile_variants(ckpt_dir, arm, {"standard": {}, "no align": {"align": True}, ...}, rej, r_per_model)

Zeroing is an IN-DISTRIBUTION operation for a stimulus column (every one of them is exactly 0 on
no-dot bouts, and the bearing products are 0 whenever the bearing is), which is why it is the right
"remove" for a column the model never learned to use; for a column it does use it is the ablation.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fit_MLP as M  # noqa: E402
import fit_MLP_border as MB  # noqa: E402


@contextlib.contextmanager
def zero_inputs(shoal_cols=(), align=False, static=False, stim_mode="shoal"):
    """Zero the named shoal columns (names from ``fit_MLP.shoal_cols(stim_mode)``), the align column
    and/or the whole static block for every feature row built inside the block."""
    names = M.shoal_cols(stim_mode)
    unknown = [c for c in shoal_cols if c not in names]
    if unknown:
        raise ValueError(f"unknown shoal columns {unknown} for {stim_mode}; known: {names}")
    idx = [names.index(c) for c in shoal_cols]
    _shoal, _static, _align = M.shoal_channels, M.static_channels, MB.align_feature

    zero_static, zero_align = bool(static), bool(align)      # read the flags BEFORE any local shadows them

    def shoal_zeroed(*a, **k):
        out = _shoal(*a, **k)
        if idx:
            out[:, idx] = 0.0
        return out

    def static_zeroed(*a, **k):
        return np.zeros_like(_static(*a, **k))

    def align_zeroed(phi, r_px, moving, n_dots):
        return np.zeros_like(np.atleast_1d(np.asarray(phi, float)))

    try:
        if idx:
            M.shoal_channels = shoal_zeroed
        if zero_static:
            M.static_channels = static_zeroed
        if zero_align:
            MB.align_feature = align_zeroed
        yield
    finally:
        M.shoal_channels, M.static_channels, MB.align_feature = _shoal, _static, _align


GNS = ("gns1", "gns2", "gns3", "gns4")


def profile_variants(ckpt_dir, arm, variants, rej=None, r_per_model=10, progress=False):
    """The dot-distance profile of ``arm`` under several rollout-time input ablations, on ONE figure
    (one panel per stimulus condition: real mean +/- SEM across fish in black, one line per variant)
    plus the headline numbers per variant. ``variants`` = {label: spec} with spec keys ``shoal_cols``,
    ``align``, ``static``, ``stim_mode`` (as ``zero_inputs``), ``seed`` (rollout-stream offset; a second
    seed of the unablated model is the profile's rollout-noise floor) and optionally ``arm`` /
    ``ckpt_dir`` to put ANOTHER arm on the same figure. Returns (summary table, {label: bouts})."""
    import matplotlib.pyplot as plt
    from model_figures import (STIMULI_ORDER, _curves, generated_dot_profile, profile_metrics,
                               profile_summary, real_dot_profile, stim_label)
    st = real_dot_profile()
    gens, rows = {}, []
    for lab, spec in variants.items():
        spec = dict(spec)
        seed = int(spec.pop("seed", 0))
        a, d = spec.pop("arm", arm), spec.pop("ckpt_dir", ckpt_dir)
        with zero_inputs(**spec):
            gen = generated_dot_profile(d, a, rej, r_per_model, progress=progress, seed=seed)
        gens[lab] = gen
        s = profile_summary(profile_metrics(st, gen))
        rows.append(dict(variant=lab, n_sim_fish=gen.fish_id.nunique(), runaway=int(gen.attrs.get("runaway", 0)),
                         mv_err_mm=s["mv_err_mm"], mv_err_sem=s["mv_err_sem"], dip=s["dip_gen"], lat=s["lat_gen"],
                         plateau=s["plateau_gen"], d1_plateau=s["d1_plateau_gen"], grading=s["grading_gen"],
                         st_err_mm=s["st_err_mm"], st_err_sem=s["st_err_sem"], static_drift=s["drift_gen"]))
    real = dict(variant="REAL", n_sim_fish=st.fish_id.nunique(), runaway=0, mv_err_mm=0.0, mv_err_sem=0.0,
                dip=s["dip_real"], lat=s["lat_real"], plateau=s["plateau_real"], d1_plateau=s["d1_plateau_real"],
                grading=s["grading_real"], st_err_mm=0.0, st_err_sem=0.0, static_drift=s["drift_real"])
    tab = pd.DataFrame([real] + rows).set_index("variant")
    # the figure: one panel per condition
    _, mr, sr = _curves(st)
    curves = {lab: _curves(g)[1] for lab, g in gens.items()}
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    colors = [f"C{i}" for i in range(len(gens))]
    for ax, c in zip(axes.ravel(), STIMULI_ORDER):
        r, e = mr[c], sr[c]
        ax.fill_between(r.index, r - e, r + e, color="0.75", alpha=0.6, lw=0)
        ax.plot(r.index, r, color="k", lw=2, label="real (15 fish)")
        for (lab, cv), col in zip(curves.items(), colors):
            g = cv[c].reindex(r.index)
            ax.plot(g.index, g, color=col, lw=1.5, label=lab)
        ax.set_title(stim_label(c), fontsize=10)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("time since block onset (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("distance to the dots (mm)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{arm}: distance-to-dots profile under rollout-time input ablations "
                 f"({rows[0]['n_sim_fish']} simulated fish per variant; band = real SEM across fish)", fontsize=11)
    fig.tight_layout()
    pd.set_option("display.width", 220)
    print("the plot in numbers: mv_err = mean |gen - real| over 0-60 s on the 2-8 moving-dot conditions (mm and real-SEM units); "
          "dip = approach amplitude within 20 s (mm) and its latency (s); plateau = 20-60 s level; grading = plateau(1 dot) - "
          "plateau(8 dots); static: err and drift-away over 0-20 s (mm/s)")
    print(tab.round(2).to_string())
    return tab, gens


def profile_variants_matched(ckpt_dir, arm, variants, rej=None, r_per_model=10, progress=False,
                             show_raw=True, gens=None):
    """``profile_variants`` with the BLOCK-SAMPLING artefact removed: identical rollouts, but each
    generated bin mean is taken over the same programme blocks, in the same proportions, as the real
    curve at that bin (``model_figures.real_block_weights`` / ``weighted_curves``).

    WHY. A real fish only contributes a bout to a bin when it bouts then, and it pauses; a rollout
    bouts in nearly every block of every replicate. Pooled over the 15 fish every block is still
    represented, but with very unequal weight, so the two sides average over the same blocks in
    DIFFERENT PROPORTIONS -- and because the
    dots' placement distance differs from block to block, part of any gap is the STIMULUS rather than
    the behaviour -- measured on Model 17, 0.77 mm of the 1.37 mm first-bin gap at 2 moving dots, with
    the remaining 0.64 +/- 0.19 mm a real difference in how fast the fish closes. The rollout is NOT
    changed: the simulated fish still swims the programme on its own clock, so its own intervals still
    decide where in a block it is. Only the averaging changes, which is why the reweighted curve reads
    "what this model does when sampled the way the real fish was sampled".

    What it does not fix: inside one 2 s bin the two sides' bouts still fall on different frames of the
    same dot track. Reweighting to a thin real sample also costs precision -- the effective number of
    blocks behind each generated point is printed per condition for the first bins.

    Returns (summary table, {label: bouts}); the table carries both readings per variant, `as plotted`
    (the unmatched curve, i.e. what ``profile_variants`` draws) and `matched`.
    """
    import matplotlib.pyplot as plt
    from model_figures import (STIMULI_ORDER, _curves, _programme, generated_dot_profile,
                               metrics_from_curves, parse_stim, profile_summary, real_block_weights,
                               real_dot_profile, stim_label, weighted_curves)
    program, _, _ = _programme()
    st = real_dot_profile()
    # the block mapping is by FRAME; refuse to go on if a real bout lands in a block of another condition
    bid = program.block_id[np.clip(np.asarray(st.frame_onset, int), 0, program.n_frames - 1)]
    nd, sp = zip(*[parse_stim(c) for c in st.cond_label])
    good = (program.n_dots_of[bid] == np.array(nd)) & (program.moving_of[bid].astype(bool) == (np.array(sp) != 0))
    if good.mean() < 0.999:
        raise RuntimeError(f"real bouts do not map onto the programme by frame ({100 * good.mean():.1f}% consistent)")
    w = real_block_weights(st, program)
    pr, mr, sr = _curves(st)

    # ``gens`` = the tables ``profile_variants`` already returned: reuse them so the two figures are the SAME
    # rollouts and the only difference is the averaging (and the rollouts are not paid for twice).
    gens, curves, rows = (dict(gens) if gens is not None else {}), {}, []
    for lab, spec in variants.items():
        spec = dict(spec)
        seed = int(spec.pop("seed", 0))
        a, d = spec.pop("arm", arm), spec.pop("ckpt_dir", ckpt_dir)
        if lab in gens:
            gen = gens[lab]
        else:
            with zero_inputs(**spec):
                gen = generated_dot_profile(d, a, rej, r_per_model, progress=progress, seed=seed)
        if "block_id" not in gen.columns:
            raise RuntimeError("the rollout table has no block_id (model_figures.programme_bouts must keep it)")
        gens[lab] = gen
        pg0, mg0, _ = _curves(gen)                       # as plotted
        pg1, mg1, _ = weighted_curves(gen, w)            # matched to the real block composition
        curves[lab] = (mg0, mg1)
        for tag, (pg, mg) in (("as plotted", (pg0, mg0)), ("matched", (pg1, mg1))):
            s = profile_summary(metrics_from_curves(pr, mr, sr, pg, mg))
            rows.append(dict(variant=f"{lab} [{tag}]", mv_err_mm=s["mv_err_mm"], mv_err_sem=s["mv_err_sem"],
                             dip=s["dip_gen"], lat=s["lat_gen"], plateau=s["plateau_gen"],
                             d1_plateau=s["d1_plateau_gen"], grading=s["grading_gen"],
                             st_err_mm=s["st_err_mm"], st_err_sem=s["st_err_sem"], static_drift=s["drift_gen"]))
    real = dict(variant="REAL", mv_err_mm=0.0, mv_err_sem=0.0, dip=s["dip_real"], lat=s["lat_real"],
                plateau=s["plateau_real"], d1_plateau=s["d1_plateau_real"], grading=s["grading_real"],
                st_err_mm=0.0, st_err_sem=0.0, static_drift=s["drift_real"])
    tab = pd.DataFrame([real] + rows).set_index("variant")

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    colors = [f"C{i}" for i in range(len(gens))]
    for ax, c in zip(axes.ravel(), STIMULI_ORDER):
        r, e = mr[c], sr[c]
        ax.fill_between(r.index, r - e, r + e, color="0.75", alpha=0.6, lw=0)
        ax.plot(r.index, r, color="k", lw=2, label="real (15 fish)")
        for (lab, (mg0, mg1)), col in zip(curves.items(), colors):
            if show_raw:
                ax.plot(mg0.index, mg0[c].reindex(mg0.index), color=col, lw=1.0, ls=":", alpha=0.65,
                        label=f"{lab} (as plotted)")
            ax.plot(mg1.index, mg1[c].reindex(mg1.index), color=col, lw=1.8, label=f"{lab} (matched)")
        ax.set_title(stim_label(c), fontsize=10); ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("time since block onset (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("distance to the dots (mm)")
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"{arm}: same rollouts, generated bins averaged over the SAME programme blocks as the real curve "
                 f"({r_per_model * 5} simulated fish per variant; band = real SEM across fish)", fontsize=11)
    fig.tight_layout()

    cov = (st.assign(block_id=bid).groupby(["cond_label", "tbin"], observed=True).block_id.nunique()
           .unstack("cond_label").reindex(columns=STIMULI_ORDER))
    pd.set_option("display.width", 220)
    print("distinct programme blocks behind each REAL point (95 exist; the generated side is reweighted onto these):")
    print(cov.head(6).to_string())
    print("\nthe plot in numbers, each variant as plotted and matched to the real block composition:")
    print(tab.round(2).to_string())
    return tab, gens
