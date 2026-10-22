"""COLAB CELL — train Model 1 over stimulus representation x joint-likelihood ORDERING.

Two representations (S1, S3) x the six exact factorisations of the bout joint = 12 arms, each
5 GroupKFold folds, plus Metric A on held-out fish, the train/val curves as PNG, and a
manifest. Rollouts and Metric B are NOT run here -- they are CPU work and belong on the local
machine (see rollout_model1_local.py).

    !python train_model1_colab.py                     # full: 12 arms x 5 folds = 60 fits
    !python train_model1_colab.py --smoke             # ~5 min wiring check
    !python train_model1_colab.py --orders adt,tad    # a subset of the sweep
    !python train_model1_colab.py --all-data          # + one in-sample model per arm (video only)
    !python train_model1_colab.py --latent 2          # Model 2 + per-fish latent z (m=2), S1_adt + S3_adt
    !python train_model1_colab.py --ladder stim       # Model 4: regime weights (+ second clock), S3
    !python train_model1_colab.py --ladder slow       # Model 3 + AR(1) drift / + scheduled sampling
    !python train_model1_colab.py --ladder s4         # S4 (dot-count grading): plain / w8 / p3 / z2 / z2+p3
    !python train_model1_colab.py --ladder slow2      # drift + per-head gains / + a second (interval) state / on S4

SLOW LADDER (--ladder slow). Model 3 (S3, adt, per-fish z m=2, rej baked in) and the two
TRAINING-TIME routes to the autocorrelation defect (the latent fixes between-fish identity but not
the within-window over-persistence of the lag heads, nor the missing window-to-window acf):
  S3_adt_z2_drift  an AR(1) slow state loading on all three heads (fit_MLP_fishre DRIFT_DEFAULT,
                   tau 70 bouts PINNED), fitted by likelihood and marginalised exactly at score
                   time -- Metric A stays exact and comparable.
  S3_adt_z2_ss     scheduled sampling on the lag inputs (Bengio et al. 2015, fit_MLP_fishre
                   SS_DEFAULT): the model's own one-step draws replace the recorded lags with
                   probability 1 - p, p decaying 1 -> 0.5 over epochs 5-45; validation and Metric A
                   stay teacher-forced, but the fitted objective is no longer an exact likelihood.
Both are paired fold for fold with the S3_adt_z2 control in the same run.

LATENT (--latent M). Model 2 with a per-fish random effect z ~ N(0, I_M) loading on all three
heads (fit_MLP_fishre.MLPMixJointREDensity): fitted by likelihood, integrated out EXACTLY at score
time (so Metric A stays comparable), drawn fresh per rollout at generation. Its loading gain is set
inside every fold's fit by method of moments against the TRAIN fish's between-fish spread -- that
step rolls out (~5 grid points x 8 fish x 2 draws) on the CPU, so a latent fold takes a few
minutes longer than a plain one. rej's wall band is baked into these checkpoints (cfg_feat_extra)
so the gain search rolls out under the same wall rule the local evaluation uses. Runs S1_adt and
S3_adt: the latent's heads are hard-wired in the adt order (the ordering sweep showed order is
worth 0.015 AUC end to end), and S1 vs S3 is a stimulus-response TRADE worth keeping both sides of
(S3 follows static dots better, S1 approaches better; tied on the AUC).

WHAT THE ORDERING SWEEP TESTS. The chain rule is exact in all six orderings, so every arm
targets the SAME joint density and `bits_per_bout_joint` is comparable straight across the
table. What differs is capacity: whichever channel is emitted first must be predicted from the
feature vector alone, and whichever is last also sees the other two outcomes of its own bout.
The EDA points at dt first -- |dtheta| scales with the time a bout spans (rho = 0.17) while
distance barely does (rho = -0.01) -- so "draw the interval, then turn and travel inside it"
may be the cheaper description. `adt` is the archived chain and is the control.

The three per-channel columns are NOT comparable across orderings on their own (a channel
gets cheaper simply by being later); only the joint sum is.

WHY THE SPLIT. Every metric change this month (dropping _late, the _h pair, run-aware
helpers, replacing between_fish_share) would have cost a full retrain under the coupled
design. Decoupled, a metric change is a CPU re-run over saved checkpoints.

SEEDS. Both the training seed and the rollout seed go into the blob. Arm-to-arm differences
here are small -- rollout_aware came back null because they sat under within-arm noise -- so
the local rollouts must be able to reproduce common random numbers across models trained in
separate runs. Without the recorded seed that is impossible. Every arm shares the per-fold
seed, so the 12 arms are PAIRED fold for fold.
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent          # artifacts land next to this script
ROOT = OUT if (OUT / "scripts").exists() else OUT.parent   # repo root, for imports only
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as E                     # noqa: E402
import fit_MLP_border as MB              # noqa: E402
import fit_MLP_joint as JT               # noqa: E402
import fit_MLP_mix as MX                 # noqa: E402
import fit_MLP_fishre as FR              # noqa: E402

# The two representations this sweep crosses with the orderings. S2 and S4 are dropped: on the
# R=20 Metric B panel S2 tracked S1 and S4 tracked S3 closely, so carrying all four would triple
# the GPU cost to re-measure a difference that is already known to be small.
REPS = {"S1": "s1",          # 6  [act, sin, cos, g, moving, nz]      monotone proximity
        "S3": "shoalr",      # 13 range BASIS instead of the scalar g; no group size
        "S4": "shoal",       # 18 S3 + the same bearing terms x dot count (the grading S3 lacks)
        "S4n2": "shoal_n2",  # 23 S4 + a group-size STEP n2 = 1[N >= 2] and its bearing products (Model 24)
        "S4s": "shoal_step", # 18 S4 with the step in place of the log2 ramp
        "S4l": "shoal_lean", # 16 S4s without the redundant `act` / `moving` flags
        "S4p": "shoal_pop",  # 55 PER-DOT population code (6 bearings x 4 ranges, all / static dots) + nz, n2, rz (Model 26)
        "S4pd": "shoal_popd",  # 70 the same with the motion fraction, nz and n2 PER DIRECTION (6 each) instead of global
        "S4pn": "shoal_popn",  # 72 Model 30: the two bump maps NORMALISED (a distribution each) + `pmag` / `smag`
        "S4pl": "shoal_popl",  # 86 Model 31: every channel LOCAL to a dot group (per-group maps, per-bearing masses, per-direction bearing)
        "S4pld": "shoal_popld",  # 92 the same + per-direction DOT DRIFT (the dots' velocity across the line of sight)
        "S4pdL": "shoal_popd_lab",  # 70 Model 34: popd with groups = stimulus groups (labels), never split by viewing angle
        "S4plL": "shoal_popl_lab",  # 86 Model 34: popl, the same
        "S0": "shoal_simple",   # 6  report sequence: presence, motion, log count, log range, bearing sin / cos
        "S4i": "shoal_int",     # 9  report sequence: + nz sin, moving sin, moving cos (S4 without the range bumps)
        "S4x": "shoal_rx",      # 14 report sequence: S4i + range x bearing per condition, linear in log range
        "S4inn": "shoal_int_nn",  # 7  report Model 22 ablation of D: no dot count
        "S4inm": "shoal_int_nm",  # 6  report Model 22 ablation of D: no moving indicator
        "Snull": "shoal_null",    # 0  report Model 22 ablation of D: no stimulus columns
        "S4xnn": "shoal_rx_nn",   # 11 report Model 29 ablation of 26e: no dot count
        "S4xnm": "shoal_rx_nm",   # 7  report Model 29 ablation of 26e: no moving indicator
        "S4xnr": "shoal_rx_nr",   # 8  ablation study (Model 33): 26e without the range
        "Scond": "shoal_cond",    # 2  ablation study (Model 33): moving vs static only (presence + moving flags)
        "S4c": "shoal_cnc",     # 22 S4 + count x range x cos (gnc1-4, the cos mirror of gns)
        "S4xlo": "shoal_rx_rzlo",  # 14 Model 42: S4x with rz clipped below at the 2 px floor (-2.56) instead of -1
        "S4lo": "shoal_rzlo"}      # 18 Model 42: S4 (bumps) with the same lower clip
SWEEP_REPS = ("S1", "S3")    # the reps x orders sweep stays 12 arms; S4 enters through ladders

# The six exact factorisations of p(dtheta, distance, dt | x). 'a' = dtheta, 'd' = distance,
# 't' = dt; the string is the EMISSION order, so 'tad' means dt is drawn first from x alone and
# dtheta last, conditioned on both. See fit_MLP_joint's module docstring.
ORDERS = ["adt", "atd", "dat", "dta", "tad", "tda"]

# use_kin is forced True by the joint class -- one x carrying everything -- so it is not a
# constructor arg. clock_mode / clock_tau_s / lag_valid are cfg_feat keys with no arg of their
# own, so they ride in via cfg_feat_extra and are saved with the checkpoint.
BASE = dict(K=8, K_kin=3, hidden=128, use_border=True, border_mode="raw",
            n_dist=2, n_dt=3,
            phi_ref="travel", psi_ref="travel",          # defaults now, set anyway for the record
            closed_loop=True, guardrail=True, stim_loop=True,
            contain=False, vigor=False, indiv=False, K_stim=1, noise_aug=None,
            cfg_feat_extra=dict(clock_mode="exp", clock_tau_s=4.0, lag_valid="scalar"))

# rej's generation-time wall rejection (redraw any bout landing 2-10 mm from the wall). Applied at
# rollout for the plain arms; BAKED IN for latent arms so their in-fit gain search sees it.
REJ_BAND = {"d0": 2.0, "d1": 10.0, "max_draws": 20}


def make_model(args, mode, order, seed, extra=None, regime_w=None, latent=None, drift=None, ss=None,
               dt_state=None, gain=None, dt_trunc=None, disp=None, lag=None, tail=None, skip=None,
               depth=None, block=None, bgain=None, drift_search=None, hist=None, hidden=None, tsize=None):
    """Model 2 (MLPMixJointDensity), or Model 2 + per-fish latent (= Model 3) when the latent
    dimension is > 0 (``latent`` per arm, else --latent).
    ``extra`` = additional cfg_feat keys for this arm (e.g. a second clock); ``regime_w`` = the
    per-bout weight recipe for the two under-fit regimes; ``drift`` = the AR(1) slow state (dict of
    DRIFT_DEFAULT overrides, e.g. tau; ``dt_off=True`` takes it off the interval head); ``dt_state``
    = the interval's own chain (dict, ``tau`` pins its timescale); ``ss`` = scheduled sampling;
    ``gain`` = 'global' (one loading gain) or 'per_channel' (three); ``disp`` / ``lag`` = z on the
    interval head's widths / lag slopes (re_disp / re_lag) -- all latent-trainer knobs."""
    latent = int(args.latent if latent is None else latent)
    kw = dict(BASE, stim_mode=mode, head_order=order, epochs=args.epochs, patience=args.patience,
              batch_size=args.batch_size, lr=args.lr, seed=seed, dt_trunc=dt_trunc,
              cfg_feat_extra=dict(BASE["cfg_feat_extra"], **(extra or {})))
    if hist is not None:                              # the history ablations: (turn lags K, kinematic lags K_kin)
        kw["K"], kw["K_kin"] = int(hist[0]), int(hist[1])
    if hidden is not None:                            # the capacity check: hidden width of the trunk (BASE: 128)
        kw["hidden"] = int(hidden)
    if not latent:
        if drift or ss or dt_state or gain or disp or lag or tail or skip or depth or block or bgain or tsize:
            raise SystemExit("drift / ss / dt_state / gain / disp / lag / tail / skip / depth / block "
                             "/ bgain live in the random-effect trainer: give the arm latent=2")
        return JT.MLPMixJointDensity(regime_w=regime_w, **kw)
    kw["cfg_feat_extra"] = dict(kw["cfg_feat_extra"], wall_reject=dict(REJ_BAND))
    # SPATIAL AXIS (Model 19): `block` reserves the LAST latent dimension for `bgain`, the per-fish
    # gain on the wall inputs, so the core loadings read m - 1 dims (m = 3 -> the usual two).
    # drift_search=False: no generation-time drift-gain search -- the drift runs at its likelihood-fitted size (gains 1),
    # identical in every arm (the search picks 0.15-5.0 per fold and hits both bounds; report Model 9).
    extra_re = {} if drift_search is None else dict(drift_gain_search=bool(drift_search))
    return FR.MLPMixJointREDensity(re_m=latent, re_gain_grid=args.gain_grid,
                                   re_gain_mode=(gain or "global"), regime_w=regime_w,
                                   drift=drift, dt_latent=dt_state, ss=ss,
                                   re_disp=bool(disp), re_lag=bool(lag), re_tail=bool(tail),
                                   turn_skip=bool(skip), depth=int(depth or 1),
                                   re_block=bool(block), re_bgain=bool(bgain), re_tsize=bool(tsize),
                                   **extra_re, **kw)


# ---- named ladders: arms that differ by more than (representation, order) ------------------ #
# 'stim' = the three arms Diagnostics.ipynb asks for, built on S3 (its stimulus block carries the
# range x bearing and moving x bearing products explicitly, and it is the better base on every
# diagnostic). Chained one change at a time so each contrast is attributable:
#   S3_adt          control: plain likelihood, one 4 s clock
#   S3_adt_w8       + the two under-fit regimes up-weighted x8 in the NLL (onset bouts, pursuit)
#   S3_adt_w8_ck15  + a second, slower block clock (tau 15 s) so the 2-8 s onset tail is expressible
# Read them on METRIC C (the teacher-forced turn law in exactly those regimes, no rollouts), then
# on the local rollouts. Metric A is not comparable across the weighted arms.
#
# 'slow' = Model 3 (S3, adt, per-fish z m=2, rej baked in) + the two training-time routes to the
# AUTOCORRELATION defect (window-to-window acf ~0 vs real +0.33; within-window acf of the lag heads
# too high). One change per arm, paired with the control fold for fold:
#   S3_adt_z2        control: Model 3 exactly as in model2z_checkpoints (same seeds)
#   S3_adt_z2_drift  + AR(1) slow state on all three heads, tau 70 bouts PINNED (DRIFT_DEFAULT),
#                    fitted by likelihood, marginalised exactly at score time -> Metric A exact
#   S3_adt_z2_ss     + scheduled sampling on the lag inputs (SS_DEFAULT: p 1 -> 0.5 over epochs
#                    5-45) -> Metric A is still the teacher-forced density but not the objective
# Read them on Metric A (control vs drift), then locally on the C2ST acf tells (dist_acf1 /
# dt_acf1, the individuality plot) and the across-window acf.
def _m17_recipe(al=True, st=True, rc=True, cdt=True, w3=True):
    """Model 17's recipe (dip4's S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st) with components switched off; the tag lists the
    ones kept, in the ladder's naming order. All flags on == dip4's entry exactly."""
    tag = "S4_adt_z2_drift" + ("_cdt_tr" if cdt else "") + ("_rc" if rc else "") + ("_al" if al else "") \
        + ("_w3ft" if w3 else "") + ("_st" if st else "")
    extra = {**({"kin_dt_lags": "centred"} if cdt else {}), **({"run_clock_tau": 30.0} if rc else {}),
             **({"stim_align": True} if al else {}), **({"stim_static": True} if st else {})}
    d = dict(tag=tag, rep="S4", order="adt", latent=2, drift=dict(tau=70.0), dt_trunc=(10.0 if cdt else None),
             extra=extra)
    if w3:
        d["regime_w"] = dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                             heads=("turn",))
    return d


def _m27_recipe(rep="S4", al=True, st=True, rc=True, cdt=True, tr=True, w3=True, lv=True, drift=False, clock=True,
                K=8, Kk=3):
    """Model 27's recipe (nodrift1's S4_adt_z2_cdt_tr_rc_al_w3ft_st: Model 17 without the drift) with one stimulus code
    and components switched off. `lv` = the lag-availability flag (BASE's lag_valid='scalar'; off -> lag_valid='').
    All defaults == nodrift1's entry exactly."""
    tag = f"{rep}_adt_z2" + ("_drift" if drift else "") + ("_cdt" if cdt else "") + ("_tr" if tr else "") \
        + ("_rc" if rc else "") + ("_al" if al else "") + ("_w3ft" if w3 else "") + ("_st" if st else "") \
        + ("" if lv else "_nlv") + ("" if clock else "_nck") + ("" if K == 8 else f"_h{K}") + ("" if Kk == 3 else f"_kk{Kk}")
    extra = {**({"kin_dt_lags": "centred"} if cdt else {}), **({"run_clock_tau": 30.0} if rc else {}),
             **({"stim_align": True} if al else {}), **({"stim_static": True} if st else {}),
             **({} if lv else {"lag_valid": ""}), **({} if clock else {"clock_mode": ""})}
    d = dict(tag=tag, rep=rep, order="adt", latent=2, dt_trunc=(10.0 if tr else None), extra=extra)
    if (K, Kk) != (8, 3):                             # the history ablations: shorter turn / kinematic history
        d["hist"] = (int(K), int(Kk))
    if drift:                                         # Model 17's AR(1) slow state (drift-gain search on)
        d["drift"] = dict(tau=70.0)
    if w3:
        d["regime_w"] = dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                             heads=("turn",))
    return d


def _d_recipe(rep="S4i", lv=False, st=True, rc=True, w3=True, drift=False, extra=None, al=False):
    """THE REPORT CHAIN (Models 20-23): Model 4 (`report1`'s S4i_adt_z2: S4i code + per-fish latent, global gain, BASE
    defaults = plain interval lags, no truncation, 4 s block clock, lag-availability flag on) and the steps built on it.
    Defaults = step D, the ablation model: Model 4 - lag flag + static block + run clock + x3 turn weight. `drift` adds
    Model 17's AR(1) slow state (tau 70, drift-gain search on). `extra` = further cfg_feat keys (the ablations)."""
    tag = f"{rep}_adt_z2" + ("_drift" if drift else "") + ("_rc" if rc else "") + ("_al" if al else "") \
        + ("_w3ft" if w3 else "") + ("_st" if st else "") + ("" if lv else "_nlv")
    ex = {**({"run_clock_tau": 30.0} if rc else {}), **({"stim_static": True} if st else {}),
          **({} if lv else {"lag_valid": ""}), **({"stim_align": True} if al else {}), **(extra or {})}
    d = dict(tag=tag, rep=rep, order="adt", latent=2, extra=ex)
    if drift:
        d["drift"] = dict(tau=70.0)
    if w3:
        d["regime_w"] = dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                             heads=("turn",))
    return d


LADDERS = {
    "stim": [dict(tag="S3_adt",         rep="S3", order="adt"),
             dict(tag="S3_adt_w8",      rep="S3", order="adt", regime_w=dict(w_onset=8.0, w_pursuit=8.0)),
             dict(tag="S3_adt_w8_ck15", rep="S3", order="adt", regime_w=dict(w_onset=8.0, w_pursuit=8.0),
                  extra=dict(clock2_tau_s=15.0))],
    "slow": [dict(tag="S3_adt_z2",       rep="S3", order="adt", latent=2),
             dict(tag="S3_adt_z2_drift", rep="S3", order="adt", latent=2, drift=dict(tau=70.0)),
             dict(tag="S3_adt_z2_ss",    rep="S3", order="adt", latent=2,
                  ss=dict(p_min=0.5, warmup=5, decay=40))],
    # 's4' = the stimulus-following round on S4 (`shoal`: S3 + bearing x dot-count products -- the
    # profile section showed every S3 arm blind to group size: grading +1-2 mm vs real +6). Paired
    # with the S3 arms already on disk (same seeds): S3_adt_rej (Model 2), S3_adt_w8 (Model 4),
    # S3_adt_z2 (Model 3).
    #   S4_adt        plain control (no S4 checkpoint exists in the current pipeline)
    #   S4_adt_w8     Model 4's recipe: onset + pursuit x8 -- does grading remove its 1-dot/static leak
    #   S4_adt_p3     pursuit-only x3: the part of the weights that worked, at a dose that should not leak
    #   S4_adt_z2     Model 3 on S4
    #   S4_adt_z2_p3  latent + pursuit-only x3 (weights now wired into the latent trainer)
    # Read on the DISTANCE-TO-STIM PROFILE section of the local run (grading, dip, static drift),
    # then the C2ST. Metric A is comparable only among the unweighted arms.
    "s4": [dict(tag="S4_adt",       rep="S4", order="adt"),
           dict(tag="S4_adt_w8",    rep="S4", order="adt", regime_w=dict(w_onset=8.0, w_pursuit=8.0)),
           dict(tag="S4_adt_p3",    rep="S4", order="adt", regime_w=dict(w_onset=1.0, w_pursuit=3.0)),
           dict(tag="S4_adt_z2",    rep="S4", order="adt", latent=2),
           dict(tag="S4_adt_z2_p3", rep="S4", order="adt", latent=2,
                regime_w=dict(w_onset=1.0, w_pursuit=3.0))],
    # 'slow2' = the drift arm of `slow` (the best AUC so far, 0.700) with the two fixes its
    # individuality plot asked for. The interval head was the remaining tell: one shared state at
    # tau 70 is the step's timescale, not the interval's (real window-acf decays fast for dt, slowly
    # for step), and one loading gain compromised three heads. One change per arm, paired by seed
    # with S3_adt_z2_drift (identical Stage A weights):
    #   S3_adt_z2_drift_g3   + THREE loading gains, one per head (re_gain_mode='per_channel')
    #   S3_adt_z2_drift2_g3  + TWO states: the slow chain (tau 70) on turn + step only, a faster
    #                        chain on the interval (dt_latent, tau 35 PINNED), both rollout-gained
    #   S4_adt_z2_drift2_g3  the same on S4 (dot-count grading)
    # Metric A is exact on all three (both chains marginalised at score time). Read the acf tells
    # (dt_acf1 level / spread), var_dt, bf_dt, then the AUC; the profile must not move.
    "slow2": [dict(tag="S3_adt_z2_drift_g3",  rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0), gain="per_channel"),
              dict(tag="S3_adt_z2_drift2_g3", rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0, dt_off=True), dt_state=dict(tau=35.0), gain="per_channel"),
              dict(tag="S4_adt_z2_drift2_g3", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0, dt_off=True), dt_state=dict(tau=35.0), gain="per_channel")],
    # 'slow3' = the INTERVAL channel, which Models 5-8 left as the top tell whatever gain or state
    # was tried: per-window interval spread 20% wide (0.97 vs 0.80), between-fish spread of it 0.37
    # vs 0.22, within-window lag-1 acf 0.12 vs 0.07. Two causes, two changes, chained so each is
    # attributable, on Model 8's recalibrated S3_adt_z2_drift as the control:
    #   S3_adt_z2_drift_nodt     interval lags OFF (kin_dt_lags=False): the lags carried the fish's
    #                            tempo level and turned one long interval into a run of them
    #   S3_adt_z2_drift_nodt_tr  + the interval head TRUNCATED at 10 s (dt_trunc, the active-only
    #                            data's own cut): the untruncated fit put too much density at 6-10 s
    #                            and the sampler piled the excess at the clip
    #   S4_adt_z2_drift_nodt_tr  the same on S4
    # Per-head gains dropped (Model 8: no effect); one state, tau 70. Run locally with --fold /
    # --threads 1 as parallel processes, merged with rollout_model1_local.merge_ckpt_dirs.
    "slow3": [dict(tag="S3_adt_z2_drift_nodt",    rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags=False)),
              dict(tag="S3_adt_z2_drift_nodt_tr", rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags=False), dt_trunc=10.0),
              dict(tag="S4_adt_z2_drift_nodt_tr", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags=False), dt_trunc=10.0)],
    # 'slow4' = Model 9's two follow-ups, on its nodt_tr arms as controls (already on disk):
    #   cdt  = CENTRED interval lags (MB.kin_dt_level: deviations from the fish's own running
    #          level) -- Model 9 dropped the lags and the lag-1 acf fell to 0.015 (real 0.073): the
    #          lags carried the real fast dependence too; this keeps it without the level
    #   nt   = the slow state's TURN loading off (drift turn_off) -- Model 9's state became a large
    #          vigour state whose swings the step lags amplified (var_distance 3x on the S3 arms)
    "slow4": [dict(tag="S3_adt_z2_drift_cdt_tr",     rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred"), dt_trunc=10.0),
              dict(tag="S3_adt_z2_drift_nodt_tr_nt", rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0, turn_off=True), extra=dict(kin_dt_lags=False), dt_trunc=10.0),
              dict(tag="S3_adt_z2_drift_cdt_tr_nt",  rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0, turn_off=True), extra=dict(kin_dt_lags="centred"), dt_trunc=10.0),
              dict(tag="S4_adt_z2_drift_cdt_tr_nt",  rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0, turn_off=True), extra=dict(kin_dt_lags="centred"), dt_trunc=10.0)],
    # 'slow5' = the RUN CLOCK (MB._runclock_block: exp(-run_pos / 30), 1 on the first bout after a
    # > 10 s pause). Measured on the clean data (Data_cleanup_final.ipynb step 5): the first ~60
    # bouts of a run are slower, shorter-stepping and more central, all of them active bouts the
    # model was trained on with nothing to say where they are; and a third of the real C2ST
    # windows are the first window of a run (dt_std 1.09 s vs 0.74 interior) while a rollout has no
    # pauses -- the interval-spread tell no gain or state could move (Models 5-10). One added input,
    # on the two best recipes and the S4 candidate; controls = Model 8's S3_adt_z2_drift and
    # Model 10's S3_adt_z2_drift_cdt_tr / S4_adt_z2_drift_cdt_tr_nt. Evaluated twice: on every
    # run-aware window as before, and on the steady-state windows (generative_metrics.INTERIOR).
    "slow5": [dict(tag="S3_adt_z2_drift_rc",        rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(run_clock_tau=30.0)),
              dict(tag="S3_adt_z2_drift_cdt_tr_rc", rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0),
              dict(tag="S4_adt_z2_drift_cdt_tr_rc", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0)],
    # 'slow6' = the top of the between-fish range, on Model 11's S4 run-clock arm (the best standard
    # AUC, 0.698, and the S4 profile). Measured on the individuality plots: the generated side
    # follows the real per-fish trend but (a) one generated fish blows up the step spread (the one
    # global loading gain over-widening step to buy interval spread) and (b) the top real fish on
    # interval spread (fish 0: slowest AND 3x the typical rate of > 5 s intervals) and on interval
    # acf (fish 6, bursty) are unreachable by a location-only latent. One change per arm:
    #   _g3  per-head loading gains (re_gain_mode='per_channel'; Model 8: step bf 1.05 -> 1.00)
    #   _ds  a DISPERSION axis: z loads on the interval head's widths (re_disp)
    #   _ls  a LAG-SLOPE axis: z x centred interval lags on the interval location (re_lag)
    # Control = Model 11's S4_adt_z2_drift_cdt_tr_rc, merged in by the local driver.
    "slow6": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_g3", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0, gain="per_channel"),
              dict(tag="S4_adt_z2_drift_cdt_tr_rc_ds", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0, disp=True),
              dict(tag="S4_adt_z2_drift_cdt_tr_rc_ls", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0, lag=True)],
    # 'slow7' (Model 13) = the two individuality fixes Model 12 pointed at, on BOTH Model 11
    # run-clock arms (S3: best steady-state AUC 0.616; S4: best standard AUC 0.698 + grading):
    #   _g3i     per-head loading gains picked by the monotone-smoothed, log-INTERPOLATED rule
    #            (MX._pick_gain) instead of the 5-point grid whose factor-1.8 spacing sent Model
    #            12's g3 from 1.5x over to 0.77x under on the step spread
    #   _g3i_tl  + the TAIL loading (re_tail: z on the interval component weights, its own gain
    #            matched to the between-fish spread of the window interval spread)
    # Controls = Model 11's S3_adt_z2_drift_cdt_tr_rc / S4_adt_z2_drift_cdt_tr_rc, merged in.
    "slow7": [dict(tag="S3_adt_z2_drift_cdt_tr_rc_g3i", rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0, gain="per_channel"),
              dict(tag="S4_adt_z2_drift_cdt_tr_rc_g3i", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0, gain="per_channel"),
              dict(tag="S3_adt_z2_drift_cdt_tr_rc_g3i_tl", rep="S3", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0, gain="per_channel", tail=True),
              dict(tag="S4_adt_z2_drift_cdt_tr_rc_g3i_tl", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   dt_trunc=10.0, gain="per_channel", tail=True)],
    # 'dip1' (Model 14) = the approach dip. MEASURED teacher-forced (Model 11): with the dots
    # AHEAD the real |dtheta| falls 36.7 -> 31.8 -> 23.8 deg from 40-60 to 80-120 px while the fit
    # stays at 39 -> 36 -> 34 -- the shoal block's range x bearing products are built on sin(bearing),
    # ~0 exactly there. `_al` adds MB.align_feature = range x cos(bearing) for moving dots (one
    # column, appended last), on both run-clock arms. Read the dip_levers table first, the dip second.
    "dip1": [dict(tag="S3_adt_z2_drift_cdt_tr_rc_al", rep="S3", order="adt", latent=2,
                  drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                  dt_trunc=10.0),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                  dt_trunc=10.0)],
    # 'dip2' (Model 15) = the far band up-weighted, on top of the aligned-and-far input. Model 14's
    # table: the input fixed the ahead law at 40-80 px and closed a quarter at 80-120 px, but the
    # off-axis toward-turn at 60-120 px got worse (side 80-120: 17.3 vs 24.9 deg real) and the dip
    # did not move -- the approach is decided in the whole far band, not the aligned cell. `w3f` =
    # every moving bout at >= 60 px (all bearings, 17 % of bouts -> ~38 % of the loss) weighted x3
    # in the NLL, now supported on the drift sequence loss (tempered emissions, see nll_seq_drift).
    # Controls = Model 14's `al` arms and Model 11's, merged in. Read the dip_levers table first.
    "dip2": [dict(tag="S3_adt_z2_drift_cdt_tr_rc_al_w3f", rep="S3", order="adt", latent=2,
                  drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                  dt_trunc=10.0, regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0,
                                               pursuit_deg=180.0, pursuit_px=60.0)),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3f", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                  dt_trunc=10.0, regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0,
                                               pursuit_deg=180.0, pursuit_px=60.0))],
    # 'dip3' (Model 16) = Model 15's far-band weight on the TURN HEAD ONLY (heads=('turn',): the
    # step and interval terms keep weight 1 in the row loss and in the drift recursion). Model 15's
    # per-bout weight bought the closest profile so far on S4 (dip 3.4 mm, plateau 16.8) but the
    # cost fell on the kinematic heads (Metric A +0.15-0.19, steps -3 px in the band) and the
    # approach law leaked into static blocks (drift-away 0.35 vs 0.58). Two arms on the S4 `al` base:
    #   _w3ft   turn-only x3 on the moving far band (>= 60 px, all bearings)
    #   _w3fta  turn-only x3 on the far band of moving AND static dots (include_static): does
    #           weighting the static far field equally stop the leak?
    # Controls = Model 11's S4, Model 14's S4 `al`, Model 15's S4 `al_w3f`, merged in.
    "dip3": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                  dt_trunc=10.0, regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0,
                                               pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3fta", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                  dt_trunc=10.0, regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0,
                                               pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",),
                                               include_static=True))],
    # 'dip4' (Model 17) = on Model 16's S4 `al_w3ft` (turn-only far-band weight), the static curves
    # and the rest of the dip. MEASURED teacher-forced on static-dot bouts: real fish turn AWAY from
    # static dots within ~60 px (toward -8.8 deg ahead, |dth| 53) and are neutral beyond; every arm
    # keeps +2 deg / 42 -- the shoal block has no static-specific bearing term. And the moving far
    # field is still head-limited (ahead 80-120 px 29.4 vs 23.8 deg; off-axis toward 20 vs 25).
    #   _st     static-dot bearing block (M.static_channels: range basis x sin / cos, 8 columns)
    #   _st_sk  + a SKIP PATH from the stimulus block, align and static columns straight to the
    #           turn head's outputs (turn_skip)
    #   _sk     the skip path alone (attribution)
    #   _d2     two hidden layers instead of one (depth=2): is the shared body the bottleneck?
    # Control = Model 16's S4_adt_z2_drift_cdt_tr_rc_al_w3ft, merged in.
    "dip4": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_sk", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0, skip=True,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_sk", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0, skip=True,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_d2", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0, depth=2,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",)))],
    # 'dip5' (Model 18) = Model 17's two wins combined, and the static near field pushed. On the
    # `al_w3ft_st` base (static block: best standard AUC 0.672, dip 3.3, plateau 16.8, static leak
    # fixed in the loop): `_d2` = + two hidden layers (Model 17's `d2` alone gave the best steady-state
    # AUC, 0.614); `_wsn` = + the STATIC NEAR band (static dots < 60 px, the evasive-turn cell the fit
    # reaches a third of) weighted x3 on the turn head, next to the moving far band; `_d2_wsn` = both.
    # Controls = Model 17's `_st` and `_d2`, merged in.
    "dip5": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d2", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0, depth=2,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_wsn", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",),
                                w_static=3.0, static_near_px=60.0)),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d2_wsn", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0, depth=2,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",),
                                w_static=3.0, static_near_px=60.0))],
    # 'dip6' (Model 19) = on Model 17's `al_w3ft_st` (the best profile+AUC model), the two tells
    # that are left. (1) The static early phase is a BLOCK-PHASE response (teacher-forced within
    # 80 px of static dots: real toward +2.8 deg at 0-4 s, -3.0 at 12-30 s; the fit flat) that the
    # 4 s clock cannot place: `_ck15sc` adds a second clock (tau 15 s) and the static block x that
    # clock (MB._static_clock_block). (2) Radial is 42 % of the C2ST signal, all between-fish (per-
    # fish mean r/R 0.40-0.66 real vs 0.36-0.52 generated): `_sp` gives the latent a THIRD, block-
    # structured dimension feeding a per-fish gain on the wall inputs (re_m=3, re_block, re_bgain;
    # core loadings still read two dims; spatial gain matched to sd(mean r/R), now interpolated).
    # `_ck15sc_sp` = both. Control = Model 17's `_st`, merged in. Global loading gain (per-channel
    # is refused with re_block).
    "dip6": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_ck15sc", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True,
                             clock2_tau_s=15.0, stim_static_clock=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_sp", rep="S4", order="adt", latent=3, block=True, bgain=True,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_ck15sc_sp", rep="S4", order="adt", latent=3, block=True, bgain=True,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True,
                             clock2_tau_s=15.0, stim_static_clock=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",)))],
    # 'dip7' (Model 20) = the static departure as a NEAR x MID-BLOCK term. MEASURED (Model 19, teacher-
    # forced): within 80 px of static dots real fish turn toward at 0-4 s (+2.8), away at 12-30 s
    # (-3.0), neutral after; beyond 80 px mildly toward at every phase (+3..+4, matched). Intervals and
    # steps match in every phase, so the departure IS the turn sign flip. A monotone clock x static
    # (Model 19) learned the onset, not the flip. `_sb` = the static block x a mid-block BUMP clock
    # (a Gaussian window centred at 18 s, width 8 s; MB.static_bump); `_sb_wsw` = + the static near band (< 80 px)
    # weighted x3 on the turn head ONLY in the 8-30 s window. Control = Model 17's `_st`, merged in.
    "dip7": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_sb", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True,
                             stim_static_bump=(18.0, 8.0)),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_sb_wsw", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True,
                             stim_static_bump=(18.0, 8.0)),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",),
                                w_static=3.0, static_near_px=80.0, static_window=(8.0, 30.0)))],
    # Model 24: the group-size STEP (real turn-toward is a step at 2 dots, the fitted law a log2 ramp -> the 2-dot
    # plateau 2.7 mm too far) on Model 17's recipe, with the inert `align` column out (measured: 0.001 bits/bout
    # teacher-forced, inside the rollout noise floor in the 50-fish profile). Control = Model 17's st, merged in.
    #   S4n2  ramp + step (51 inputs)     S4s  step in place of the ramp (46)     S4l  + `act`/`moving` out (44)
    "step1": [dict(tag=f"{rep}_adt_z2_drift_cdt_tr_rc_w3ft_st", rep=rep, order="adt", latent=2,
                   drift=dict(tau=70.0), dt_trunc=10.0,
                   extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_static=True),
                   regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",)))
              for rep in ("S4n2", "S4s", "S4l")],
    # Model 25: the ONSET x BEARING columns (the first turn toward moving dots does not scale with N in real
    # fish: 27-33 deg at onset for every N; the fit 21-31) and a SHARP pursuit weight (x5 on the turn and step
    # heads for moving dots ahead < 30 deg and >= 80 px, 4 % of bouts, on top of the broad x3), on Model 17's
    # FULL recipe. `_refit` = Model 17's recipe retrained with this ladder's seeds: the paired control and the
    # first measurement of the recipe's own fit-to-fit draw against the dip4 checkpoints (NEVER overwrites them:
    # a distinct arm name, and the driver merges dip4's arm INTO the new folder).
    "onset1": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_refit", rep="S4", order="adt", latent=2,
                    drift=dict(tau=70.0), dt_trunc=10.0,
                    extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                    regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
               dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_on", rep="S4", order="adt", latent=2,
                    drift=dict(tau=70.0), dt_trunc=10.0,
                    extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True, stim_onset=True),
                    regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))),
               dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_on_wp", rep="S4", order="adt", latent=2,
                    drift=dict(tau=70.0), dt_trunc=10.0,
                    extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True, stim_onset=True),
                    regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",),
                                  w_pursuit2=5.0, pursuit2_deg=30.0, pursuit2_px=80.0, heads2=("turn", "dist")))],
    # Model 26: the PER-DOT population stimulus code (fit_MLP.POP_MODE) on Model 17's recipe. The static law and the
    # moving-only bearing live inside the block, so stim_static / stim_align are off; 55 stimulus + block clock = 56
    # stimulus columns, 75 inputs. Success = a TIE with Model 17 on the single-group profile, static curves and AUC:
    # the code exists to make a two-group stimulus representable, not to improve anything.
    # Model 27: Model 17 with the AR(1) SLOW STATE REMOVED and nothing else changed (same seeds, so it is
    # paired with the dip4 checkpoints exactly). The drift loads on all three heads with tau 70 bouts and is
    # the model's only source of within-trajectory slow fluctuation; this measures what it is worth.
    "nodrift1": [dict(tag="S4_adt_z2_cdt_tr_rc_al_w3ft_st", rep="S4", order="adt", latent=2, dt_trunc=10.0,
                      extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                      regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                    heads=("turn",)))],
    # Model 28: the two stimulus-side treatments that fixed the per-bout law but not the block-level curve,
    # now WITHOUT the slow state -- Model 27 showed the drift erodes the multi-bout chase (dip 2.67 -> 3.83
    # matched), so the hypothesis is that the step and the population code had nothing to act on while it was
    # scrambling the sequence. Controls (Model 17 and its no-drift twin) are merged in from dip4 / nodrift1.
    #   n2    no-drift Model 17 + the group-size STEP columns, `align` kept -> the step is the ONLY change (52)
    #   popd  no-drift, the per-dot population code with per-direction size / motion (90; it subsumes the
    #         static block, `align` and the step, so it is "everything at once" against the same control)
    "nodrift2": [dict(tag="S4n2_adt_z2_cdt_tr_rc_al_w3ft_st", rep="S4n2", order="adt", latent=2, dt_trunc=10.0,
                      extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                      regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                    heads=("turn",))),
                 dict(tag="S4pd_adt_z2_cdt_tr_rc_w3ft_popd", rep="S4pd", order="adt", latent=2, dt_trunc=10.0,
                      extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                      regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                    heads=("turn",)))],
    # Model 31: the LOCAL population code, alone / + slow second block clock / + dot drift / + both. No drift
    # (AR slow state), same recipe and seeds as popn1, so every arm pairs with popn and with each other.
    "local1": [dict(tag=tag, rep=rep, order="adt", latent=2, dt_trunc=10.0,
                    extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, **({"clock2_tau_s": 15.0} if ck else {})),
                    regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                 heads=("turn",)))
               for tag, rep, ck in (("S4pl_adt_z2_cdt_tr_rc_w3ft_popl", "S4pl", False),
                                    ("S4pl_adt_z2_cdt_tr_rc_w3ft_popl_ck15", "S4pl", True),
                                    ("S4pld_adt_z2_cdt_tr_rc_w3ft_popld", "S4pld", False),
                                    ("S4pld_adt_z2_cdt_tr_rc_w3ft_popld_ck15", "S4pld", True))],
    # Model 38: SHORT-TERM STIMULUS LAGS (`stim_lags=3`: the dot centroid at the previous 3 bouts seen from the current
    # pose, [sin, cos, rz] each, 9 columns appended last, zero across block boundaries and pauses) on the two Model 29
    # bases' exact recipes (no drift). Recalibrated (_g3h) and compared with those bases.
    "lag1": [dict(tag="S4_adt_z2_cdt_tr_rc_al_w3ft_st_sl3", rep="S4", order="adt", latent=2, dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True,
                             stim_lags=3),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                heads=("turn",))),
             dict(tag="S4pd_adt_z2_cdt_tr_rc_w3ft_popd_sl3", rep="S4pd", order="adt", latent=2, dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_lags=3),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                heads=("turn",)))],
    # Model 34: GROUP LABELS (no angular split of a stimulus group: 26 % of real 2-dot bouts were read as two 1-dot
    # groups) x NO RUN STATE (`_nc`: the run clock and the lag-validity flag removed -- the only two inputs that tell
    # a fresh run from a long one, which a never-pausing rollout always gets wrong at block onset; the 8 lags stay).
    # popd_lab keeps both = the split fix alone; S4 = Model 29's centroid arm (+ static block, align) minus run state.
    "lab1": [dict(tag=tag, rep=rep, order="adt", latent=2, dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", **ex),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                heads=("turn",)))
             for tag, rep, ex in (("S4pdL_adt_z2_cdt_tr_rc_w3ft_popd_lab", "S4pdL", dict(run_clock_tau=30.0)),
                                  ("S4pdL_adt_z2_cdt_tr_w3ft_popd_lab_nc", "S4pdL", dict(lag_valid="")),
                                  ("S4_adt_z2_cdt_tr_al_w3ft_st_nc", "S4",
                                   dict(lag_valid="", stim_align=True, stim_static=True)),
                                  ("S4plL_adt_z2_cdt_tr_w3ft_popl_lab_nc", "S4plL", dict(lag_valid="")))],
    # THE REPORT SEQUENCE (Model/Report_models.ipynb): the centroid code built up from its simplest form, current
    # pipeline, same seeds as every other arm, no slow state. Model 5 of that notebook (full S4) = base1's S4_adt_z2.
    #   S0_adt      Models 1 (no wall rejection) and 2 (with it): simple code (6), no latent
    #   S0_adt_z2   Model 3: + per-fish latent, one shared loading gain, rej baked in
    #   S4i_adt_z2  Model 4: + the products nz sin, moving sin, moving cos (9), no range bumps
    "report1": [dict(tag="S0_adt",     rep="S0",  order="adt"),
                dict(tag="S0_adt_z2",  rep="S0",  order="adt", latent=2),
                dict(tag="S4i_adt_z2", rep="S4i", order="adt", latent=2)],
    # Model 7 of the report sequence: static and moving dots get their OWN range law, one arm per range shape (both
    # then recalibrated per head, `_g3h`, as Model 6 does to Models 4 and 5):
    #   S4x_adt_z2     no bumps: Model 4's code + moving / static x log range x sin / cos + count x log range x sin (14)
    #   S4_adt_z2_st   bumps: Model 5's code (S4) + the static bearing block (the four bumps x sin / cos, static only)
    "report2": [dict(tag="S4x_adt_z2",   rep="S4x", order="adt", latent=2),
                dict(tag="S4_adt_z2_st", rep="S4",  order="adt", latent=2, extra=dict(stim_static=True))],
    # Model 8 of the report sequence, on Model 6a's code (S4i, no bumps; every arm then recalibrated per head, `_g3h`):
    # a 2 x 2 on the run clock x the interval treatment, plus the far-band turn weight on the full combination.
    #   S4i_adt_z2_rc               8a  + run clock (exp(-k / 30), k = bouts since the fish resumed swimming)
    #   S4i_adt_z2_cdt_tr           8b  + centred interval lags (deviation from the fish's running level) + 10 s truncation
    #   S4i_adt_z2_cdt_tr_rc        8c  all three
    #   S4i_adt_z2_cdt_tr_rc_w3ft   8d  8c + turn loss x3 on moving bouts with the dots >= 60 px away (any bearing)
    "report3": [dict(tag="S4i_adt_z2_rc", rep="S4i", order="adt", latent=2, extra=dict(run_clock_tau=30.0)),
                dict(tag="S4i_adt_z2_cdt_tr", rep="S4i", order="adt", latent=2, dt_trunc=10.0,
                     extra=dict(kin_dt_lags="centred")),
                dict(tag="S4i_adt_z2_cdt_tr_rc", rep="S4i", order="adt", latent=2, dt_trunc=10.0,
                     extra=dict(kin_dt_lags="centred", run_clock_tau=30.0)),
                dict(tag="S4i_adt_z2_cdt_tr_rc_w3ft", rep="S4i", order="adt", latent=2, dt_trunc=10.0,
                     extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                     regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                   heads=("turn",)))],
    # Model 9 of the report sequence (what makes Model 17 work; 9a = Model 17 and 9d = Model 23's `_st_g3h` are
    # evaluated from their existing checkpoints):
    #   S4x_adt_z2_drift_cdt_tr_rc_al_w3ft  9b  Model 17's recipe WITHOUT the bumps: 7a's code (range x bearing per
    #                                           condition, linear in log range; its static terms replace the static block)
    #   S4i_adt_z2_drift                    9c  Model 6a + the AR(1) drift alone (tau 70 bouts, all three heads), -> g3h
    "report4": [dict(tag="S4x_adt_z2_drift_cdt_tr_rc_al_w3ft", rep="S4x", order="adt", latent=2,
                     drift=dict(tau=70.0), dt_trunc=10.0,
                     extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True),
                     regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                   heads=("turn",))),
                dict(tag="S4i_adt_z2_drift", rep="S4i", order="adt", latent=2, drift=dict(tau=70.0))],
    # Model 10 of the report sequence: SIMPLIFYING Model 17 (dip4's `S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st`). Every arm
    # drops `align` (measured inert) and runs the drift at its likelihood-fitted size (`_g1`: no drift-gain search, so
    # no per-fold gain lottery); S0 is the control the other three change ONE thing against:
    #   S0  S4_adt_z2_drift_cdt_tr_rc_w3ft_st_g1    Model 17 - aligned-and-far
    #   S1  S4_adt_z2_drift_cdt_tr_w3ft_st_g1       S0 - run clock
    #   S2  S4_adt_z2_drift_rc_w3ft_st_g1           S0 - centred interval lags - 10 s truncation
    #   S3  S4c_adt_z2_drift_cdt_tr_rc_w3ft_st_g1   S0 + count x range x cos (gnc1-4)
    "report5": [dict(tag=tag, rep=rep, order="adt", latent=2, drift=dict(tau=70.0), drift_search=False,
                     dt_trunc=(10.0 if cdt else None),
                     extra=dict(stim_static=True, **({"kin_dt_lags": "centred"} if cdt else {}),
                                **({"run_clock_tau": 30.0} if rc else {})),
                     regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                   heads=("turn",)))
                for tag, rep, cdt, rc in (("S4_adt_z2_drift_cdt_tr_rc_w3ft_st_g1", "S4", True, True),
                                          ("S4_adt_z2_drift_cdt_tr_w3ft_st_g1", "S4", True, False),
                                          ("S4_adt_z2_drift_rc_w3ft_st_g1", "S4", False, True),
                                          ("S4c_adt_z2_drift_cdt_tr_rc_w3ft_st_g1", "S4c", True, True))],
    # Report Model 15: SIMPLIFYING MODEL 17 -- its exact recipe (dip4's S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st, drift-gain
    # search ON, seed 0 = paired with Model 17) minus ONE component per arm. Model 10 removed components with the drift
    # gains fixed at 1 as well, so none of its costs could be attributed. `_m17_recipe` with every flag on == dip4's dict.
    #   15a  - aligned-and-far flag          S4_adt_z2_drift_cdt_tr_rc_w3ft_st
    #   15b  - static block                  S4_adt_z2_drift_cdt_tr_rc_al_w3ft
    #   15c  - run clock                     S4_adt_z2_drift_cdt_tr_al_w3ft_st
    #   15d  - centred interval lags + 10 s truncation   S4_adt_z2_drift_rc_al_w3ft_st
    #   15e  - x3 turn weight on far moving bouts        S4_adt_z2_drift_cdt_tr_rc_al_st
    "simp17": [_m17_recipe(al=al, st=st, rc=rc, cdt=cdt, w3=w3)
               for al, st, rc, cdt, w3 in ((False, True, True, True, True), (True, False, True, True, True),
                                           (True, True, False, True, True), (True, True, True, False, True),
                                           (True, True, True, True, False))],
    # Report Models 18 / 19: SIMPLIFYING MODEL 27 (Model 17 without the drift; report R5), one change per arm, seed 0
    # (paired with nodrift1). `_m27_recipe()` with defaults == nodrift1's entry.
    #   simp27s (stimulus representation): 18a S4x (range bumps -> linear range x bearing), 18b S4i (no range in the
    #     bearing terms), 18c S0 (simplest code, no count x bearing), 18d - static block, 18e - aligned-and-far flag
    #   simp27o (other components): 19a - run clock, 19b plain interval lags (truncation kept), 19c - 10 s truncation
    #     (centred lags kept), 19d - x3 turn weight, 19e - lag-availability flag
    "simp27s": [_m27_recipe(rep="S4x"), _m27_recipe(rep="S4i"), _m27_recipe(rep="S0"), _m27_recipe(st=False),
                _m27_recipe(al=False)],
    "simp27o": [_m27_recipe(rc=False), _m27_recipe(cdt=False), _m27_recipe(tr=False), _m27_recipe(w3=False),
                _m27_recipe(lv=False)],
    # Report Model 20: THE CHAIN from Model 4 to the ablation model D, cumulative, seed 0 (paired with report1's Model 4):
    #   A = Model 4 - lag-availability flag; B = A + static block; C = B + run clock; D = C + x3 turn weight
    "chainD": [_d_recipe(st=False, rc=False, w3=False), _d_recipe(rc=False, w3=False), _d_recipe(w3=False),
               _d_recipe()],
    # Report Model 21: D + Model 17's drift (run at two seeds by the driver)
    "chainDdrift": [_d_recipe(drift=True)],
    # Report Model 24: 18a (= Model 27 with the S4x code) + drift, run at seeds 0 / 101 (18a's own seed 101 = simp27s)
    "s4xdrift": [_m27_recipe(rep="S4x", drift=True)],
    # Report Model 25: Model 4 (report1's S4i_adt_z2) + ONE component each; + run clock and + centred lags / truncation
    # are report3's raw fits, + range products is report2's S4x_adt_z2 -- these are the three missing ones
    "m4plus": [dict(tag="S4i_adt_z2_al", rep="S4i", order="adt", latent=2, extra=dict(stim_align=True)),
               dict(tag="S4i_adt_z2_w3ft", rep="S4i", order="adt", latent=2,
                    regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                  heads=("turn",))),
               dict(tag="S4i_adt_z2_st", rep="S4i", order="adt", latent=2, extra=dict(stim_static=True))],
    # Report Model 26: 18a minus ONE component each (18a minus the range products = 18b, simp27s)
    "simp18a": [_m27_recipe(rep="S4x", cdt=False, tr=False), _m27_recipe(rep="S4x", rc=False),
                _m27_recipe(rep="S4x", al=False), _m27_recipe(rep="S4x", w3=False), _m27_recipe(rep="S4x", st=False)],
    # The ablation study (Model 33, Final_report_ablations.ipynb): 26e (= _m27_recipe(rep="S4x", st=False)) minus one
    # ingredient. Stimulus: moving vs static only (presence + moving flags) / range removed (both without the
    # aligned-and-far flag, which reads bearing and range). History: the turn lags 8 -> 4 / 1 / 0, the kinematic lags
    # 3 -> 1 / 0 (they feed the shared hidden layer, so every head sees the shorter history). Onset: no block clock.
    "abl26e": [_m27_recipe(rep="Scond", st=False, al=False), _m27_recipe(rep="S4xnr", st=False, al=False),
               _m27_recipe(rep="S4x", st=False, K=4), _m27_recipe(rep="S4x", st=False, K=1),
               _m27_recipe(rep="S4x", st=False, K=0), _m27_recipe(rep="S4x", st=False, Kk=1),
               _m27_recipe(rep="S4x", st=False, Kk=0), _m27_recipe(rep="S4x", st=False, clock=False)],
    # The ablation study, kinematic history LONGER than 26e's 3 lags (Model 44, Final_report_ablations.ipynb): the step and
    # interval lags 3 -> 4 / 5 / 6, everything else exactly 26e (the shorter histories 1 / 0 are in abl26e).
    "abl26e3": [_m27_recipe(rep="S4x", st=False, Kk=4), _m27_recipe(rep="S4x", st=False, Kk=5),
                _m27_recipe(rep="S4x", st=False, Kk=6)],
    # The ablation study, the latent REMOVED (Model 45, Final_report_ablations.ipynb): 26e retrained without the per-fish
    # random effect (latent = 0: the plain joint model, no z), everything else exactly 26e. The first round froze the
    # trained 26e's z at 0 at simulation time (33q / 33r); retraining asks whether the model NEEDS the latent.
    "abl26e4": [dict(_m27_recipe(rep="S4x", st=False), tag="S4x_adt_cdt_tr_rc_al_w3ft", latent=0)],
    # The ablation study, the remaining history lengths (Model 49, Final_report_ablations.ipynb): step and interval lags
    # 3 -> 2, turn lags 8 -> 3, everything else exactly 26e.
    "abl26e5": [_m27_recipe(rep="S4x", st=False, Kk=2), _m27_recipe(rep="S4x", st=False, K=3)],
    # The ablation study, second round (Model 35, Final_report_ablations.ipynb): 26e minus the components the first round
    # never removed from 26e itself -- the run clock, the centred interval lags with the 10 s truncation (removed
    # together, as in simp18a), the aligned-and-far flag on its own, the x3 turn weight.
    "abl26e2": [_m27_recipe(rep="S4x", st=False, rc=False), _m27_recipe(rep="S4x", st=False, cdt=False, tr=False),
                _m27_recipe(rep="S4x", st=False, al=False), _m27_recipe(rep="S4x", st=False, w3=False)],
    # Scaling up (Model 36, Final_report_models.ipynb): the headline recipes with THREE hidden layers of 128 instead of
    # one (depth=3; re_body is off, so the whole stack is the shared body every head reads). Everything else exactly as
    #   M6 (26e, simp18a)  -> S4x_adt_z2_cdt_tr_rc_al_w3ft_d3
    #   D4 (Model 17, dip4's control entry) -> S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d3
    # F = D4_d3 + the Model 14 profile correction (run_pf_one.py ... Bc, final step), outside this ladder.
    # The slow state's time constant (Model 37, Final_report_models.ipynb): D4 (dip4's control entry) with the AR(1)
    # drift's tau = 200 bouts instead of 70 -- the real window-to-window correlation decays with tau ~160-220 bouts
    # (displacement, radial), so the archived 70 forgets within about two windows. F = this + the Model 14 correction.
    # Turn-size individuality (Model 38, Final_report_models.ipynb): M6 and D4 with the per-fish latent ALSO loading on
    # the turning components' scales (re_tsize: one shared loading on the four Gamma scales), so how far a fish turns
    # is a trait alongside how often it swims straight. F = D4_tz + the Model 14 correction.
    "tsize": [dict(_m27_recipe(rep="S4x", st=False), tag="S4x_adt_z2_cdt_tr_rc_al_w3ft_tz", tsize=True),
              dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_tz", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), dt_trunc=10.0, tsize=True,
                   extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                   regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                 heads=("turn",)))],
    # Model 46: the turn-size latent (Model 38) on the tau = 200 slow state (Model 37) together, D4 only (M6 has no drift)
    "tsize200": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200_tz", rep="S4", order="adt", latent=2,
                      drift=dict(tau=200.0), dt_trunc=10.0, tsize=True,
                      extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                      regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                    heads=("turn",)))],
    "tau200": [dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_t200", rep="S4", order="adt", latent=2,
                    drift=dict(tau=200.0), dt_trunc=10.0,
                    extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                    regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                  heads=("turn",)))],
    "deep3": [dict(_m27_recipe(rep="S4x", st=False), tag="S4x_adt_z2_cdt_tr_rc_al_w3ft_d3", depth=3),
              dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_d3", rep="S4", order="adt", latent=2,
                   drift=dict(tau=70.0), dt_trunc=10.0, depth=3,
                   extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                   regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                 heads=("turn",)))],
    # Final_report_missingPieces (Part B), on 26e's recipe (S4x, no static block) exactly:
    #   _lo6m   leave-one-condition-out: bouts shown 6 MOVING dots get weight 0 on every head (they stay in the records);
    #           the count code interpolates (log2 6 / 3 between 4 and 8), so the held-out response is a pure prediction
    #   _w64 / _w256   the capacity check: hidden width 64 / 256 instead of 128
    # Model 41 (Model_design_and_training.ipynb): the DOT-JUMP PHASE input (`stim_jump`, MB._jump_block: [sin, cos] of
    # the frames since the last dot jump on the 18-frame jump clock, moving dots only), appended to two recipes exactly:
    #   M6 (= 26e, S4x, no static block)  -> S4x_adt_z2_cdt_tr_rc_al_w3ft_jp
    #   D4 (= dip4's Model 17, S4 bumps + static block + AR(1) drift) -> S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_jp
    # F + jump phase = D4 + jump phase with Model 14's profile correction (run_pf_one.py ... Bc), outside this ladder.
    "jump": [dict(_m27_recipe(rep="S4x", st=False), tag="S4x_adt_z2_cdt_tr_rc_al_w3ft_jp",
                  extra=dict(_m27_recipe(rep="S4x", st=False)["extra"], stim_jump=True)),
             dict(tag="S4_adt_z2_drift_cdt_tr_rc_al_w3ft_st_jp", rep="S4", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True,
                             stim_jump=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                heads=("turn",)))],
    # Model 42 (Final_report_models.ipynb): the LOWER-CLIP CHECK -- the log range rz clipped below at the 2 px floor
    # (log4(2 / 70) = -2.56) instead of -1 (17.5 px), on two recipes exactly (only the stimulus mode changes):
    #   M6 (= 26e, S4x, no static block)  -> S4xlo_adt_z2_cdt_tr_rc_al_w3ft
    #   D4 (= dip4's Model 17, S4 bumps + static block + AR(1) drift) -> S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st
    # F = D4 lower clip + Model 14's profile correction (run_pf_one.py ... Bc), outside this ladder.
    "rzlo": [_m27_recipe(rep="S4xlo", st=False),
             dict(tag="S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st", rep="S4lo", order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                heads=("turn",)))],
    # Model 51 (Final_report_screendatatest.ipynb, 2026-10-20): the DOT MOTION DIRECTION input (`stim_dmove`,
    # MB._dmove_block: [sin, cos] of the centroid's displacement from the previous bout to this one, in the bout's
    # reference frame) on the screen-trained M6 (= the rzlo M6 arm exactly, + the input). Trained by screen_train.py.
    "rzlo_dm": [dict(_m27_recipe(rep="S4xlo", st=False), tag="S4xlo_adt_z2_cdt_tr_rc_al_w3ft_dm",
                     extra=dict(_m27_recipe(rep="S4xlo", st=False)["extra"], stim_dmove=True)),
                # D4 + the input: the rzlo D4 arm exactly (drift-gain search on, as there), + stim_dmove
                dict(tag="S4lo_adt_z2_drift_cdt_tr_rc_al_w3ft_st_dm", rep="S4lo", order="adt", latent=2,
                     drift=dict(tau=70.0), dt_trunc=10.0,
                     extra=dict(kin_dt_lags="centred", run_clock_tau=30.0, stim_align=True, stim_static=True,
                                stim_dmove=True),
                     regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                   heads=("turn",)))],
    "missing": [dict(_m27_recipe(rep="S4x", st=False), tag="S4x_adt_z2_cdt_tr_rc_al_w3ft_lo6m",
                     regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                   heads=("turn",), drop_cond=[(6, 1)])),
                dict(_m27_recipe(rep="S4x", st=False), tag="S4x_adt_z2_cdt_tr_rc_al_w3ft_w64", hidden=64),
                dict(_m27_recipe(rep="S4x", st=False), tag="S4x_adt_z2_cdt_tr_rc_al_w3ft_w256", hidden=256)],
    # Report Model 29 (Run 1a): the missing main-line steps and the ablations of 26e (= 18a without the static block)
    #   step 6: 25d + the range products (Model 4 + x3 turn weight on the S4x code)
    #   reverse order: 25d + the temporal set (centred lags + truncation, run clock, aligned-and-far flag) = 18b - static
    #   ablations of 26e: no dot count / no moving indicator (also no aligned-and-far flag: it reads the moving flag) /
    #   no stimulus (no columns, no aligned-and-far flag, no block clock)
    "run1a": [dict(tag="S4x_adt_z2_w3ft", rep="S4x", order="adt", latent=2,
                   regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                 heads=("turn",))),
              _m27_recipe(rep="S4i", st=False),
              _m27_recipe(rep="S4xnn", st=False), _m27_recipe(rep="S4xnm", st=False, al=False),
              _m27_recipe(rep="Snull", st=False, al=False, clock=False)],
    # Report Model 20g: D + the aligned-and-far flag (is it worth adding to D before everything is built on D?)
    "chainDal": [_d_recipe(al=True)],
    # Report Model 22: ablations of D (seed 0, pairs with 20d): no dot count / no moving indicator (static block out
    # too: it is defined by "shown and not moving") / no stimulus at all (no static block, no block clock)
    "ablD": [_d_recipe(rep="S4inn"), _d_recipe(rep="S4inm", st=False),
             _d_recipe(rep="Snull", st=False, extra={"clock_mode": ""})],
    # Model 39: the report's R1-R4 (Chapter 5, Table 5.1) RETRAINED on the current pipeline (active-only records with
    # the run_pos fix) with the same seeds as every later arm, so the whole reported sequence sits on one harness:
    #   S3_adt      R1 (evaluated WITHOUT wall rejection) and R2 (the same checkpoints WITH it): no latent
    #   S3_adt_z2   R3: + per-fish latent, one shared loading gain (Stage B), rej baked in
    #   S4_adt_z2   R4: + the dot-count layer of the centroid code
    # No slow state anywhere; R5 = nodrift1's `S4_adt_z2_cdt_tr_rc_al_w3ft_st` adds the rest.
    "base1": [dict(tag="S3_adt",    rep="S3", order="adt"),
              dict(tag="S3_adt_z2", rep="S3", order="adt", latent=2),
              dict(tag="S4_adt_z2", rep="S4", order="adt", latent=2)],
    # Model 30: the normalised population code, no drift, paired with `popd` (the Model 28/29 control).
    "popn1": [dict(tag="S4pn_adt_z2_cdt_tr_rc_w3ft_popn", rep="S4pn", order="adt", latent=2, dt_trunc=10.0,
                   extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                   regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0,
                                 heads=("turn",)))],
    "pop1": [dict(tag=f"{rep}_adt_z2_drift_cdt_tr_rc_w3ft_{sfx}", rep=rep, order="adt", latent=2,
                  drift=dict(tau=70.0), dt_trunc=10.0,
                  extra=dict(kin_dt_lags="centred", run_clock_tau=30.0),
                  regime_w=dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",)))
             for rep, sfx in (("S4p", "pop"), ("S4pd", "popd"))],
}


def metricC(model, rec):
    """The two teacher-forced numbers the stimulus-response fixes are judged on, on the held-out
    fish, no rollout (Diagnostics.ipynb D2 / D5). Real values on the SAME bouts are reported next
    to them so each row is self-contained.

      C_onset_toward    mean E[dtheta] * sign(bearing) on the FIRST bout of a moving block at
                        matched geometry (range 40-85 px, |bearing| 45-135 deg); real ~42 deg
      C_late_toward     the same on bouts >= 10 s into the block; real ~18 deg
      C_pursuit_absdth  mean E|dtheta| on moving bouts with the dots ahead (|bearing| < 30 deg)
                        and >= 80 px away; real ~26 deg (the models were at 30-39)
    """
    e_dth, e_abs, _ = model.conditional_turn(rec)
    phi = getattr(rec, "stim_angle_travel", None)
    phi = np.asarray(phi if phi is not None else rec.stim_angle, float)
    sgn = np.sign(phi); absphi = np.degrees(np.abs(phi))
    rng = np.asarray(rec.dot_dist_px, float); mv = np.asarray(rec.moving, bool)
    nd = np.nan_to_num(np.asarray(getattr(rec, "n_dots", np.zeros(rec.n)), float))
    start_fr, bid, fr = MB.block_onsets(rec)
    ok = mv & np.isfinite(phi) & np.isfinite(rng) & (sgn != 0)
    matched = ok & (rng >= 40) & (rng <= 85) & (absphi >= 45) & (absphi <= 135)
    first = matched & (nd > 0) & np.r_[True, bid[1:] != bid[:-1]]
    late = matched & ((fr - start_fr) / 30.0 >= 10.0)
    pursuit = ok & (absphi < 30.0) & (rng >= 80.0)
    tw_c, tw_r = e_dth * sgn, np.asarray(rec.dtheta, float) * sgn
    def mean(v, m):
        return float(np.nanmean(v[m])) if m.any() else np.nan
    return dict(C_onset_toward=mean(tw_c, first), C_onset_toward_real=mean(tw_r, first),
                C_late_toward=mean(tw_c, late), C_late_toward_real=mean(tw_r, late),
                C_pursuit_absdth=mean(e_abs, pursuit),
                C_pursuit_absdth_real=mean(np.abs(np.asarray(rec.dtheta, float)), pursuit),
                C_n_onset=int(first.sum()), C_n_pursuit=int(pursuit.sum()))


def metricA(model, rec):
    """Held-out cross-entropy in BITS PER BOUT, per channel and jointly (lower = better).

    `score` returns PER-BOUT log densities, not scalars -- aggregate here with the same
    convention as evaluate (`-mean(logdens) / ln 2`) so these numbers sit on the same scale
    as every archived bits_per_bout. `joint` is the sum of the three channels, which is what
    the joint model actually optimises; `bits_per_bout` alone is the turn term only.
    """
    sc = model.score(rec)
    out = {}
    for key, name in (("logdens", "bits_per_bout"), ("logdens_dist", "bits_per_bout_dist"),
                      ("logdens_dt", "bits_per_bout_dt")):
        v = np.asarray(sc.get(key, np.nan), float)
        out[name] = float(-np.nanmean(v) / np.log(2.0)) if v.size else np.nan
    out["bits_per_bout_joint"] = float(np.nansum([out["bits_per_bout"],
                                                  out["bits_per_bout_dist"],
                                                  out["bits_per_bout_dt"]]))
    out["n_bouts"] = int(rec.n)
    return out


def make_folds(fish, n_folds=5, seed=0):
    """Deterministic GroupKFold over fish ids -> [(train_fish, heldout_fish), ...]."""
    f = np.array(sorted(fish))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(f))
    # plain ints, not np.int64 -- the fold membership is written to JSON in the manifest
    return [([int(x) for x in f[np.setdiff1d(order, order[i::n_folds])]],
             [int(x) for x in f[order[i::n_folds]]]) for i in range(n_folds)]


def curve_png(hists, out, title):
    """Train/val NLL per epoch, one panel per fold. `hists` = [(train, val, label), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(hists)
    fig, ax = plt.subplots(1, n, figsize=(3.1 * n, 3.0), sharey=True)
    ax = np.atleast_1d(ax)
    for a, (tr, va, lab) in zip(ax, hists):
        if tr:
            a.plot(range(1, len(tr) + 1), tr, color="#2a78d6", lw=1.5, label="train")
        a.plot(range(1, len(va) + 1), va, color="#eb6834", lw=1.5, label="val")
        if va:
            k = int(np.argmin(va))
            a.axvline(k + 1, color="#52514e", ls="--", lw=1)
            a.plot([k + 1], [va[k]], "o", color="#52514e", ms=5)
        a.set_xlabel("epoch"); a.set_title(lab, fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#e1e0d9", lw=.6); a.set_axisbelow(True)
    ax[0].set_ylabel("joint NLL  (nats / bout)")
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)   # GPU: big batches, see note below
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--out", type=str, default=None,
                    help="zip name (default model1_checkpoints.zip, model2z_checkpoints.zip "
                         "with --latent)")
    ap.add_argument("--orders", type=str, default=None,
                    help="comma-separated emission orders (default: all six; 'adt' with --latent)")
    ap.add_argument("--reps", type=str, default=None,
                    help="comma-separated representation tags (default: S1,S3)")
    ap.add_argument("--latent", type=int, default=0,
                    help="per-fish latent dimension m (0 = off, the plain Model 2; 2 = rej's)")
    ap.add_argument("--gain-grid", type=str, default="wide_up", choices=("std", "wide", "wide_up"),
                    help="candidate latent loading gains searched per fold (--latent only): std = "
                         "0.4..3.0 (5), wide = 0.1..3.0 (8), wide_up = 0.1..6.5 (10, default)")
    # OFF by default for this sweep. The all-data model has no held-out fish, so it is scored by
    # nothing and skipped by rollout_model1_local.py -- it exists only to render video. Training
    # one per arm would add 12 full fits to a run whose whole purpose is the 12-way comparison.
    ap.add_argument("--all-data", action="store_true",
                    help="also train one in-sample model per arm (generation artifact only)")
    ap.add_argument("--all-data-only", action="store_true",
                    help="train ONLY the in-sample model per arm, no folds (implies --all-data); "
                         "the zip name gets _alldata appended")
    ap.add_argument("--ladder", type=str, default=None, choices=sorted(LADDERS),
                    help="a named set of arms (see LADDERS) instead of the reps x orders sweep")
    ap.add_argument("--arms", type=str, default=None,
                    help="comma-separated arm names to train (a subset of the ladder / sweep), so "
                         "one ladder can be split over two Colab sessions; the zip name gets the "
                         "arms appended unless --out is given. Merge the downloads locally with "
                         "rollout_model1_local.merge_ckpt_dirs before evaluating them as one ladder.")
    ap.add_argument("--fold", type=int, default=None,
                    help="train ONE fold (0-based) of the split, for running the folds as parallel "
                         "processes on a many-core CPU; the zip name gets _fold<k> appended. Merge "
                         "the per-fold outputs with merge_ckpt_dirs.")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads for this process (1 when running folds in parallel)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.patience, args.folds = 3, 2, 2
    if args.orders is None:
        args.orders = "adt" if args.latent else ",".join(ORDERS)
    if args.reps is None:
        args.reps = ",".join(SWEEP_REPS)
    if args.out is None:
        args.out = (f"{args.ladder}_ladder_checkpoints.zip" if args.ladder else
                    "model2z_checkpoints.zip" if args.latent else "model1_checkpoints.zip")
    orders = [JT.order_tag(o) for o in args.orders.split(",") if o.strip()]
    reps = {t: REPS[t] for t in (s.strip() for s in args.reps.split(",")) if t}
    if args.latent and any(o != "adt" for o in orders):
        # fail before any GPU time is spent, not at the first fit
        raise SystemExit(f"--latent supports head order 'adt' only (the random-effect heads are "
                         f"hard-wired); got {orders}")
    if args.ladder and args.latent:
        raise SystemExit("a ladder sets the latent per arm (see LADDERS); drop --latent")

    if args.threads:
        torch.set_num_threads(int(args.threads))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}"
          + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else
             f"  -- NO GPU ({torch.get_num_threads()} threads)"))
    if dev == "cuda":
        # Big batches + TF32 are the two free wins here: the net is tiny (128 hidden) so the
        # bottleneck is kernel-launch overhead, not FLOPs. A 1024 batch cuts the number of
        # launches 4x against the 256 default at no cost to the fit -- the gradient is an
        # average either way, and lr is scaled with it. TF32 matmuls are ~free on Ampere+.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    df = E.load_bouts()                                   # clean_data/bouts_clean.csv
    recs = MB.build_border_records(df, min_len=args.min_len)   # active_only=True by default
    fish = sorted(recs)
    print(f"{len(df):,} rows -> {len(fish)} fish, "
          f"{sum(r.n for r in recs.values()):,} active bouts")
    folds = make_folds(fish, args.folds, seed=args.seed)
    fold_items = list(enumerate(folds))
    if args.fold is not None:
        if not 0 <= args.fold < len(folds):
            raise SystemExit(f"--fold {args.fold} out of range for {len(folds)} folds")
        fold_items = [fold_items[args.fold]]
        args.out = Path(args.out).stem + f"_fold{args.fold}.zip"
    if args.all_data_only:
        fold_items, args.all_data = [], True
        args.out = Path(args.out).stem + "_alldata.zip"

    if args.ladder:
        arms = [dict(a, arm=a["tag"]) for a in LADDERS[args.ladder]]
    else:
        arms = [dict(tag=t, rep=t, order=o, arm=f"{t}_{o}" + (f"_z{args.latent}" if args.latent else ""))
                for t in reps for o in orders]
    if args.arms:
        want = [s.strip() for s in args.arms.split(",") if s.strip()]
        missing = [w for w in want if w not in {a["arm"] for a in arms}]
        if missing:
            raise SystemExit(f"--arms {missing} not in this run's arms: {[a['arm'] for a in arms]}")
        arms = [a for a in arms if a["arm"] in want]
        if args.out == (f"{args.ladder}_ladder_checkpoints.zip" if args.ladder else
                        "model2z_checkpoints.zip" if args.latent else "model1_checkpoints.zip"):
            args.out = Path(args.out).stem + "_" + "_".join(want) + ".zip"
    n_fit = len(arms) * (len(fold_items) + int(args.all_data))
    print(f"{len(arms)} arms ({', '.join(a['arm'] for a in arms)}) x "
          f"{len(fold_items)} fold(s){' + all-data' if args.all_data else ''} = {n_fit} fits")

    likA, manifest, curves, t0 = [], [], {}, time.time()
    for ai, a in enumerate(arms):
        tag, order, arm, mode = a["rep"], a["order"], a["arm"], REPS[a["rep"]]
        extra, regime_w = a.get("extra"), a.get("regime_w")
        latent = int(a.get("latent", args.latent))
        drift, ss = a.get("drift"), a.get("ss")
        dt_state, gain, dt_trunc = a.get("dt_state"), a.get("gain"), a.get("dt_trunc")
        disp, lag, tail = a.get("disp"), a.get("lag"), a.get("tail")
        skip, depth = a.get("skip"), a.get("depth")
        block, bgain = a.get("block"), a.get("bgain")
        drift_search = a.get("drift_search")
        hist = a.get("hist")
        hidden = a.get("hidden")
        tsize = a.get("tsize")
        if args.smoke and ss:
            ss = dict(ss, warmup=0, decay=2)       # 3-epoch smoke: make the mixing actually run
        hists = []
        for k, (tr_f, ho_f) in fold_items:
            # SAME seed per fold across every arm, so the arms are paired: an arm-to-arm
            # difference is then the treatment, not a different initialisation.
            m = make_model(args, mode, order, seed=args.seed + k, extra=extra, regime_w=regime_w,
                           latent=latent, drift=drift, ss=ss, dt_state=dt_state, gain=gain,
                           dt_trunc=dt_trunc, disp=disp, lag=lag, tail=tail, skip=skip, depth=depth,
                           block=block, bgain=bgain, drift_search=drift_search, hist=hist, hidden=hidden,
                           tsize=tsize)
            m.name = f"{arm}_fold{k}"
            train = [recs[f] for f in tr_f]
            ctx = E.make_context(train, dict(mlp_epochs=args.epochs,
                                             mlp_patience=args.patience))
            m.fit(train, ctx)
            # Metric A (per-bout density) and Metric C (the teacher-forced turn law in the two
            # under-fit regimes) on the fish this fold never saw -- no rollouts for either
            for f in ho_f:
                likA.append(dict(arm=arm, rep=tag, order=order, latent=latent, fold=k,
                                 fish=f, **metricA(m, recs[f]), **metricC(m, recs[f])))
            p = OUT / f"{arm}_fold{k}.pt"
            MX.save_mix(m, str(p))
            # everything the local half needs to reproduce the run
            blob = torch.load(p, map_location="cpu", weights_only=False)
            blob["heldout_fish"] = ho_f
            blob["train_fish"] = tr_f
            blob["train_seed"] = args.seed + k
            blob["rollout_seed"] = args.rollout_seed
            blob["phi_ref"], blob["psi_ref"] = "travel", "travel"
            blob["cfg_feat_extra"] = dict(getattr(m, "cfg_feat_extra", BASE["cfg_feat_extra"]))
            torch.save(blob, p)
            hists.append((list(getattr(m._model, "_hist_train", [])),
                          list(m.history), f"fold {k} (held out {ho_f})"))
            manifest.append(dict(file=p.name, arm=arm, rep=tag, order=order, stim_mode=mode,
                                 fold=k, latent=latent, heldout=ho_f,
                                 epochs_run=len(m.history), regime_w=regime_w, extra=extra,
                                 drift=drift, ss=ss, dt_state=dt_state, gain=gain,
                                 dt_trunc=dt_trunc, disp=disp, lag=lag, tail=tail, skip=skip,
                                 depth=depth, block=block, bgain=bgain, hist=hist, hidden=hidden, tsize=tsize,
                                 spatial_gain=getattr(m, "spatial_gain", None)))
            if latent:
                print(f"    {arm} fold {k}: latent loading gain {getattr(m, 're_gain', None)} "
                      + (f"(per head, grid {FR.GAIN_GRID_PC})" if gain == "per_channel" else
                         f"(grid {args.gain_grid}: {FR.gain_grid_for(args.gain_grid)})"))
            print(f"  {arm} fold {k}: {len(m.history)} epochs, "
                  f"val {min(m.history):.4f}, held out {ho_f}  [{time.time()-t0:.0f}s]")

        if args.all_data:
            # generation artifact ONLY -- no held-out fish, so never a score
            m = make_model(args, mode, order, seed=args.seed, extra=extra, regime_w=regime_w,
                           latent=latent, drift=drift, ss=ss, dt_state=dt_state, gain=gain,
                           dt_trunc=dt_trunc, disp=disp, lag=lag, tail=tail, skip=skip, depth=depth,
                           block=block, bgain=bgain)
            m.name = f"{arm}_alldata"
            train = [recs[f] for f in fish]
            m.fit(train, E.make_context(train, dict(mlp_epochs=args.epochs,
                                                    mlp_patience=args.patience)))
            p = OUT / f"{arm}_alldata.pt"
            MX.save_mix(m, str(p))
            blob = torch.load(p, map_location="cpu", weights_only=False)
            blob.update(heldout_fish=[], train_fish=fish, train_seed=args.seed,
                        rollout_seed=args.rollout_seed, insample_only=True,
                        phi_ref="travel", psi_ref="travel")
            torch.save(blob, p)
            manifest.append(dict(file=p.name, arm=arm, rep=tag, order=order, stim_mode=mode,
                                 fold=-1, heldout=[], epochs_run=len(m.history),
                                 insample_only=True, regime_w=regime_w, extra=extra))
            hists.append((list(getattr(m._model, "_hist_train", [])), list(m.history),
                          "all data (in-sample)"))
        curves[arm] = f"curve_{arm}.png"
        curve_png(hists, OUT / curves[arm], f"Model 2 · {arm} ({mode})")
        done = (ai + 1) / len(arms)
        print(f"  [{ai+1}/{len(arms)} arms] {arm} done -> {curves[arm]}   "
              f"eta {(time.time()-t0)*(1-done)/done:.0f}s")

    lik = pd.DataFrame(likA)
    lik.to_csv(OUT / "likA_model1.csv", index=False)
    (OUT / "manifest_model1.json").write_text(json.dumps(manifest, indent=1))
    with zipfile.ZipFile(OUT / args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for e in manifest:
            z.write(OUT / e["file"], e["file"])
        for f in curves.values():
            z.write(OUT / f, f)
        z.write(OUT / "likA_model1.csv", "likA_model1.csv")
        z.write(OUT / "manifest_model1.json", "manifest_model1.json")
    mb = (OUT / args.out).stat().st_size / 1e6

    # The comparison the sweep exists for, printed so a killed download is not a lost run.
    # Only the JOINT column is comparable across orderings: a channel emitted late is cheaper
    # simply because it sees more, so the per-channel columns move against each other by
    # construction and only their sum is the same quantity in every arm.
    if not lik.empty and args.ladder:
        # METRIC C is the scoreboard for a ladder: the teacher-forced turn law in the two regimes
        # the diagnostics found short, next to the real value on the same held-out bouts.
        g = lik.groupby("arm")
        tab = pd.DataFrame({
            "A_joint": g.bits_per_bout_joint.mean(),
            "onset_toward": g.C_onset_toward.mean(), "onset_real": g.C_onset_toward_real.mean(),
            "late_toward": g.C_late_toward.mean(), "late_real": g.C_late_toward_real.mean(),
            "pursuit_absdth": g.C_pursuit_absdth.mean(), "pursuit_real": g.C_pursuit_absdth_real.mean(),
        }).reindex([a["arm"] for a in arms])
        print("\nMETRIC C -- teacher-forced turn law on held-out fish (deg), no rollouts:")
        print("  onset  = first bout of a moving block, matched geometry: real ~42 (models were 19-30)")
        print("  late   = same geometry >= 10 s into the block: real ~18")
        print("  pursuit= mean |dtheta| with the dots ahead and >= 80 px: real ~26 (models were 30-39)")
        print(tab.round(2).to_string())
        if args.ladder == "slow":
            print("\nMetric A (joint bits/bout): EXACT for the control and the drift arm (the slow state "
                  "is marginalised at score time), so control vs drift reads straight off A_joint. "
                  "The ss arm scores the same teacher-forced density but did not optimise it -- "
                  "judge it on the local rollouts (dist_acf1 / dt_acf1, the individuality plot).")
        else:
            print("\nMetric A (joint bits/bout) is NOT comparable across the weighted arms -- the "
                  "weighted arms optimise a different objective; read it within the control only.")
    elif not lik.empty:
        piv = lik.pivot_table(index="order", columns="rep", values="bits_per_bout_joint")
        print("\nMetric A, bits per bout, JOINT (all three channels; lower = better):")
        print(piv.round(4).to_string())
        print("\nper channel (turn / dist / dt) -- NOT comparable across orders, only within:")
        print(lik.groupby(["rep", "order"])[["bits_per_bout", "bits_per_bout_dist",
                                             "bits_per_bout_dt"]].mean().round(4).to_string())
        best = piv.mean(axis=1).idxmin()
        print(f"\nbest ordering on the mean of the {piv.shape[1]} representations: {best}"
              f"   (archived chain 'adt' = {piv.mean(axis=1).get('adt', float('nan')):.4f})")
        print("\nMETRIC C (teacher-forced turn law in the two under-fit regimes), mean over held-out fish:")
        print(lik.groupby("arm")[["C_onset_toward", "C_onset_toward_real", "C_pursuit_absdth",
                                  "C_pursuit_absdth_real"]].mean().round(2).to_string())
    print(f"\nwrote {args.out} ({mb:.1f} MB) in {time.time()-t0:.0f}s -- DOWNLOAD THIS.")
    print("Metric B is NOT in here: run rollout_model1_local.py against these checkpoints.")


if __name__ == "__main__":
    main()
