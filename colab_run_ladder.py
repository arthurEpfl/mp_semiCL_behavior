"""Colab entry point: run the 4-rung stimulus ladder and emit everything needed for offline analysis.

    dotb  ->  dot  ->  shoalr  ->  shoal

Three single-factor contrasts: (1) ANY range information at all, (2) NON-MONOTONE range vs the
old monotone 1/(1+r/20), (3) group size N on top of range. All four rungs MUST run in one call:
the per-unit RNG streams now give common random numbers across models, which makes the
comparison paired, and cross-run comparison of C2ST AUCs was never valid.

Outputs (zipped as ladder_results.zip):
  results.csv           the table, incl. c2st_*_sd replicate spreads
  panel.csv             per-fish Metric-B panel (for paired tests)
  likA.csv              per-fish Metric A
  artifacts.pkl         wins_traj window matrices + per-replicate AUC vectors + neg control
                        -> lets importances / other classifiers be recomputed WITHOUT re-running
  <rung>.pt             one all-fish checkpoint per rung, for local diag_range_profile.py

Usage in Colab:
    !python colab_run_ladder.py                       # full run, ~1h45m
    !python colab_run_ladder.py --folds 2 --epochs 5 --reps 2 --smoke   # ~10 min wiring check
"""
import argparse
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as E                      # noqa: E402
import generative_metrics as GM           # noqa: E402
import fit_MLP_border as MB               # noqa: E402
import fit_MLP_mix as MX                  # noqa: E402

LADDERS = {
    # step 1 (DONE): does the model need RANGE information, and in what functional form?
    "stim": ["mlp_border_full_h128_mix_dotb_hloop_kin_vigor",     # control: bearing only
             "mlp_border_full_h128_mix_dot_hloop_kin_vigor",      # + monotone proximity
             "mlp_border_full_h128_mix_shoalr_hloop_kin_vigor",   # + non-monotone range basis
             "mlp_border_full_h128_mix_shoal_hloop_kin_vigor"],   # + group size
    # step 2: the WALL. Radial occupancy is ~0.5 of permutation importance and 4 radial
    # features alone give C2ST AUC 0.797 (vs 0.890 for all 39). dotb_wall isolates the border
    # basis from the stimulus fix; shoal_wall tests that the two compose.
    "wall": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor",       # control = step-1 winner
             "mlp_border_full_h128_mix_dotb_wall_hloop_kin_vigor",   # border basis alone
             "mlp_border_full_h128_mix_shoal_wall_hloop_kin_vigor"], # both
    # step 3: the KINEMATIC AR. Adding acf features to the C2ST reopened a lever this project had
    # retired (it was dropped because the 32-feature metric could not see autocorrelation).
    # Measured: gen log-distance lag-1 acf +0.42 vs real +0.23, and the vigor drift double-counted
    # the rollout's own window-to-window variance (1.75x real). '_drift' fixes the second alone;
    # '_kfix' adds the exposure-bias shrinkage that fixes the first.
    "kin": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor",         # control = step-1 winner
            "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_drift",   # drift re-sized only
            "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_kfix"],   # + kin feedback shrinkage
    # step 4: DISPERSION. kin_feedback fixed the acf but halved the distance dispersion (1.04 -> 0.52
    # against a real target of 1.00) because shrinking an AR history damps persistence and spread
    # together. 'kfixvc' rescales the emitted series to restore the spread -- an orthogonal knob,
    # since scaling about the mean leaves the autocorrelation unchanged. '_turn' additionally gives
    # the turn channel a per-trajectory latent, targeting the between-fish share (gen 0.078 vs real
    # 0.255). Judge these on the DISPERSION line as well as the AUC.
    "disp": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor_kfix",          # control = step-3 winner
             "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_kfixvc",        # + variance compensation
             "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_kfixvc_turn"],  # + turn individuality
    # step 5: CONTAINMENT IS THE GATE. Across 5 models corr(wall_frac_gap, c2st_traj) = +0.869, and
    # kfix -> kfix_vc (a pure generation-time change on the same net) showed CAUSALLY that more
    # movement spread -> more wall contact -> worse AUC. So while the wall is broken the metric
    # rewards an unrealistically NARROW model, and dispersion cannot be fixed first. 'contain'
    # fixes the wall (occupancy 0.062 -> 0.026 locally, no cost to the stimulus response);
    # 'contain_vc' then re-tries the variance compensation that lost in step 4. If contain_vc now
    # BEATS contain, the gate hypothesis is confirmed and dispersion is unblocked.
    "contain": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor_kfix",        # control = best so far
                "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain",     # + containment
                "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_vc"], # + variance comp
    # step 6: the KINEMATIC TAIL. With the wall fixed, dt is the largest remaining group (0.33) and
    # dt_std the top single feature. Diagnosis: the BULK is right (gen dt_median 1.292 vs real 1.281)
    # but the TAIL is truncated -- gen dt q99.9 = 24.6 s vs real 103.6, only 0.055% of bouts over 30 s
    # vs 0.63% real -- which collapses every tail-sensitive window feature (across-window variance
    # ratios dt_std 0.04, dt_mean 0.08, dt_q90 0.12, while dt_median sits at 0.46). The clip is not
    # the cause (it binds on 0.001%). Fix is representational: 1-comp distance -> 2, 2-comp dt -> 3,
    # trained by likelihood. NOTE this is why kin_var_comp failed -- it widened the log-normal
    # SYMMETRICALLY, when what is missing is a heavy UPPER tail only.
    "tail": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain",       # control = current best
             "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail"], # + heavy-tailed kin heads
    # step 7: the FAR FIELD. The shoal basis saturates past its outermost knot (weights at
    # 160/280/400 px are 0.730/0.954/0.987 on knot 4), so the model cannot tell 160 px from 400 px
    # and the pull it learned there persists forever: generated restoring force is -3.4 px/bout
    # beyond 120 px where real fish sit at -0.53. 'shoalf' adds a 5th knot at 280 px (real range
    # reaches 280 px at q99), which hands over 0.215/0.717/0.901 across the same distances -- so the
    # head can LEARN the back-off rather than have a taper imposed. Targets the dot-dist group (0.17).
    "far": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail",    # control = current best
            "mlp_border_full_h128_mix_shoalf_hloop_kin_vigor_contain_tail"],  # + far-field knot
    # step 8: RE-CALIBRATE the containment. rad_frac_wall is ~30% of ALL remaining detectability --
    # 3x the next feature -- and the shipped calibration under-shoots it at 1.60x real occupancy.
    # An all-15-fish sweep on the shipped checkpoint shows occupancy falls MONOTONICALLY through
    # the real value (scale 2.0 -> 1.12x, 2.5 -> 1.02x), so the bias has NOT hit a ceiling; the
    # calibration was estimating a ~1% probability from 6 fish (per-fish spread is 7-12x) and
    # weighting occupancy equally with the mean when importances say it deserves 2.6x.
    # cal1 = the old settings (6 fish, equal weight) so the contrast is PAIRED in one SELECT.
    "cal": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_cal1",  # legacy calibration
            "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail"],      # 12 fish + 2.6:1
    # step 9: CONVERGENCE. A control on the rollout-aware fine-tune (anchor term only, i.e. plain
    # extra MLE steps) cut wall occupancy 0.0216 -> 0.0170 AND improved held-out bits 7.6575 ->
    # 7.6434 -- a bigger occupancy gain than the containment re-calibration achieved, for free.
    # The policy-gradient term added ~nothing on top (0.0166) and cost likelihood. So the 80-epoch
    # fits are NOT converged, and every ladder so far may have been comparing under-trained models.
    # Paired via epochs_override so both budgets run in ONE SELECT.
    "conv": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail",       # 80 epochs (current)
             "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_e250"], # 250, patience 30
    # step 10: MEMORY HORIZON, with an attribution control. The kinematics trunk sees K_kin=3
    # lags and nothing else, so generated log-dt autocorrelation matches real at lags 1-3
    # (.228/.204/.180 vs .234/.191/.167) and dies from lag 5 (.086/.038 vs .140/.111). That is
    # the AR(3) signature, and it is why kin_feedback exists: with 3 lags you cannot produce a
    # slow tail without inflating the head, so the lag-1 overshoot it was invented to fix is an
    # ARTEFACT OF K_kin=3. It paid for the fix with the distance dispersion -- dist_std 5.84 vs
    # real 8.39 -- which is both the model's dominant tell and the defect that MASKS the others
    # (an identical injected defect is worth +0.042 AUC alone, +0.001 alongside a 20% marginal
    # error, +0.049 once marginals are repaired). Several past nulls may be masked, not refuted.
    # _nofb is the ATTRIBUTION CONTROL: shrinkage off, K_kin still 3. Without it a win by
    # _k8_nofb (which changes both) would be miscredited to the memory.
    # Both rungs are pre-registered on scripts/smoke_kin_memory.py (kin trunk alone, 12 train
    # fish, free-run 20-40k bouts, 3-5 seeds). Measured, against real sd(log dist) 0.450 and
    # real dt acf@10 +0.111 / acf@20 +0.084:
    #   K=3  kfb=.52/.82 (SHIPPED)  sd 0.373   acf@10 +0.012  acf@20 -0.003   3/3 stable
    #   K=3  kfb=1                  sd 0.454   acf@10 +0.035  acf@20 -0.004   3/3 stable
    #   K=8  kfb=1                  sd 0.451   acf@10 +0.127  acf@20 +0.055   3/3 stable
    #   K=16 kfb=1                  sd 0.460   acf@10 +0.168  acf@20 +0.118   4/5 stable
    #   K=32 kfb=1                  sd 0.491   acf@10 +0.320  acf@20 +0.288   3/5 stable
    # So the shrinkage ALONE accounts for the whole distance-dispersion deficit (0.373 -> 0.454
    # at unchanged K=3, i.e. real), and the memory horizon is what buys the acf tail. NO K>=16
    # rung: K=16 hard-COLLAPSED on 1 of 5 seeds (acf 0.998 at every lag) and K=32 wandered on 2
    # of 5 (acf@50 +0.57 against a real +0.042) -- and K=32 has the BEST held-out NLL of any
    # arm, so validation loss does not see it and fit() has no detector. A 3-seed check passed
    # K=16 cleanly; the failure is intermittent, so it needs >=5 seeds.
    # Control is e250, the current best, so every rung inherits the 250-epoch budget.
    "mem": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_e250",     # control
            "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_nofb",     # K3, kfb off
            "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8_nofb",  # K8, kfb off
            "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8"],      # K8, kfb refitted
    # step 11: INDIVIDUALITY + the dispersion the shrinkage costs. The mem ladder settled that
    # kin_feedback is NOT removable -- 'nofb' fixed dist_std_ratio 0.79 -> 1.09 and distance
    # dispersion 0.51 -> 0.98, both bang on target, and LOST 0.068 because free-run distance
    # acf@1 blew to 0.394 against a real 0.218. So the spread has to come back some other way:
    # 'innov' widens the innovation about the per-bout conditional mean, which restores variance
    # while leaving the acf (phi^k, scale-free) alone. Separately 'indiv' gives the two channels
    # vigor never covered a per-trajectory latent -- between-fish share 0.074 vs real 0.255, the
    # one deficit that is ABSENT by construction rather than mistuned. Both are pre-registered on
    # scripts/smoke_indiv.py against ladder_k8.pt.
    # step 12: CLEAN-SLATE JOINT MODEL, run as two cells. One feature vector (turn history +
    # stimulus + border + kinematic history), one shared body, heads chained dtheta -> l -> tau,
    # trained on the full joint likelihood and REPORTING all three channels -- the kin family
    # optimises l and tau against a likelihood it never reports. No lineage calibrators.
    # 'joint' is the unconstrained arm; 'jointc' adds the shell-gated containment calibration.
    # The two cells stay comparable: cross_validate_generative keys its RNG on (seed, fish, rep)
    # with NO model index and `contain` draws no variates, so at the same --seed both cells see
    # identical streams and the contain contrast is paired per (fish, rep) offline.
    # step 13: WHICH LINEAGE TREATMENTS TRANSFER TO THE JOINT MODEL? Base = joint+contain on
    # shoal. The config diff against the lineage best (k8, 0.7340) is exactly four things:
    # vigor, kin_feedback, K_kin 8-vs-3, and 250-vs-80 epochs. These two cells test the first
    # three at a fixed 80 epochs. Control FIRST in each list so the printed paired contrasts are
    # against the untreated base.
    "jkin3": ["mlp_joint_h128_shoal_contain",
              "mlp_joint_h128_shoal_contain_kfb",
              "mlp_joint_h128_shoal_contain_vig",
              "mlp_joint_h128_shoal_contain_kfb_vig"],
    # jkin3 result: kfb alone -0.0067 (noise), vigor alone -0.0413, BOTH -0.1114 (t=-14.2) ->
    # 0.7414, level with the lineage best. Strongly SUPER-additive: vigor buys individuality but
    # inflates the distance marginal (dist_w1 2.60 -> 3.38), and kfb pulls that back (-> 3.16)
    # while keeping most of the between-fish spread (0.072 -> 0.061). Neither alone does both.
    # Single rung: only K_kin=8 on the winning combination. Its control is k3_both from the jkin3
    # run -- a valid PAIRED comparison offline, because both configs draw the same number of RNG
    # variates per bout (identical vigor + kfb), so the (seed, fish, rep) streams stay aligned.
    # step 14: the two mechanisms vigor+kfb cannot reach, on top of the jkin3 winner (0.7414).
    # Control first so the printed contrasts are against the untreated base. All four rungs share
    # identical weights (both mechanisms are generation-only), so Metric A must be identical
    # across the table -- any movement there would be a bug, not a finding.
    # step 15: indiv RE-CALIBRATED. The first attempt won -0.0375 (resolved) and reached
    # between-fish 0.187 (from 0.061, real 0.255) and turn-only C2ST 0.5061 (floor 0.476) --
    # the mechanism works. But the 2-point solve chose fw=1.02, still missing the spread target
    # while DOUBLING abs_dtheta_w1 to 12.59, and its radial solve returned 0.0 on one rung and
    # the clip ceiling on another. _search_indiv is now a monotone grid with an abs_dtheta_w1
    # guard, and the radial latent gets its own probe pass. Two rungs: control + fixed indiv.
    # single rung: indiv alone, for an epochs sweep. NOTE there is no control in this SELECT, so
    # no paired contrast is printed -- compare offline against a control trained at the SAME
    # epoch budget, or the epochs and the mechanism are confounded.
    "jindiv1": ["mlp_joint_h128_shoal_contain_kfb_vig_indiv"],
    "jindiv2": ["mlp_joint_h128_shoal_contain_kfb_vig",
                "mlp_joint_h128_shoal_contain_kfb_vig_indiv"],
    "jindiv": ["mlp_joint_h128_shoal_contain_kfb_vig",
               "mlp_joint_h128_shoal_contain_kfb_vig_indiv",
               "mlp_joint_h128_shoal_contain_kfb_vig_innov",
               "mlp_joint_h128_shoal_contain_kfb_vig_indinn"],
    # step 15: INDIVIDUALITY AS A FITTED VARIANCE COMPONENT, not a sampler constant.
    # A per-fish latent z ~ N(0, I) enters the shared body and loads LEVEL-ONLY on all three heads
    # (forward-mixture logit; log-normal mu for travel and interval). Fitted by joint MAP with the
    # prior in the loss, de-shrunk by one bounded global gain on the loadings matched to the real
    # between-fish spread on TRAIN fish, and marginalised EXACTLY at score time by Gauss-Hermite
    # quadrature -- so unlike vigor/indiv it is both FITTED and SCORED.
    #
    # Targets deficit 1 (between_fish_share 0.061 gen vs 0.255 real; across-fish sd of forward
    # fraction 0.0061 vs 0.0537) and, through it, most of deficit 2: ~65% of the missing turn
    # variance and ~87% of the missing distance variance is BETWEEN-fish. A per-trajectory constant
    # adds window-MEAN variance and contributes exactly zero to dist_acf1/dt_acf1 (_acf1 subtracts
    # the window mean), so it cannot fall into the sigma^2/(1-phi^2) trap that cost nofb 0.068.
    #
    # READ THE PANEL FIRST, NOT THE AUC. Predicted d(auc) is -0.005..-0.020, i.e. AT the resolution
    # floor -- run this at --reps 24. Budget: the default 80 epochs, which is what every archived
    # joint rung ran at, so the control is comparable. Note the known cost -- 'conv' measured that 80
    # is NOT converged and 250 bought -0.0113 -- and note that unlike the generation-time constants
    # it replaces, this mechanism is budget-sensitive: the loadings must organise before the latent
    # carries individuality rather than noise. If the loading-gain warning fires on re_add, the fit
    # was too short; rerun that arm at --epochs 250 --patience 30 before concluding anything. The pre-registered PRIMARY endpoints are between_fish_share
    # (0.061 -> 0.15-0.25) and the across-fish sds of forward fraction (0.0061 -> 0.03-0.05) and
    # median |dtheta| (1.30 -> 4-7 deg), all read on ALL 15 fish pooled across folds, never on a
    # 3-fish subset. Expect the per-fish panel W1s to DEGRADE by construction: panel_for_fish
    # compares each rollout to THAT fish's real marginals and an honest "new fish" is not
    # personality-matched. Metric A will move (<=0.03 bits): unlike jindiv, the weights differ
    # across these rungs, so the "Metric A identical across the table" invariant does not hold here.
    #
    # Two controls, answering different questions. re0 pins the prior at 0 -- the mechanism is inert,
    # so it isolates the trainer rewrite (reshuffle fix + contiguous per-fish validation split) from
    # the mechanism. re_shuf permutes fish labels row-wise -- identical capacity and gradient flow,
    # no fish signal -- so it isolates "used the between-fish structure" from "added parameters".
    # re_swap deletes vigor: run it only if re_add wins, and never fold the two into one arm, since
    # jkin3 measured ~0.063 AUC in the vigor x kfb interaction.
    # THREE arms, all sharing one trainer and one validation rule, so every contrast inside the
    # ladder is single-factor. re0 is the anchor: its latent is inert (zero-initialised loadings get
    # no gradient at z = 0), so it IS the population model, but fitted by the same code and the same
    # contiguous per-fish validation split as the other two. That makes it a BETTER-matched baseline
    # than the shipped k3_both, whose contrast would confound the mechanism with the change of
    # validation rule. Cost matters too: a 4th arm is +25% on a run that already takes ~4-5 h and
    # died once at 19/20 of Metric B with nothing written.
    #   re0     -> re_shuf   what the extra capacity alone buys (labels permuted = no fish signal)
    #   re_shuf -> re_add    what USING FISH IDENTITY buys -- the actual scientific claim
    #   re0     -> re_add    the mechanism end to end
    "fishre": ["mlp_joint_re0",                          # anchor + training-loop control
               "mlp_joint_re_shuf",                      # capacity control
               "mlp_joint_re_add"],                      # the mechanism
    # ANCHOR BACK TO THE LINEAGE. Run this only if fishre is positive and you need to state the
    # result against the shipped model rather than against re0. It must be its own run: the RNG
    # streams are keyed on (seed, fish, rep) with no model index, so arms are paired only WITHIN a
    # call, and the archived 0.7414 is not a valid anchor any more (the reshuffle bug fix at
    # fit_MLP_joint.py:128 changes the control's weights). Two arms, ~2.5 h -- deliberately a
    # separate, survivable session rather than a fourth arm on an already-fragile 5 h run.
    "fishre_anchor": ["mlp_joint_h128_shoal_contain_kfb_vig",   # the shipped model, REFIT
                      "mlp_joint_re_add"],
    # SINGLE ARM: the latent is the ONLY individuality mechanism in the model. vigor off (its
    # per-trajectory intercept on log-distance and log-interval is exactly what the latent is meant
    # to replace, and stacking them double-counts -- measured: between_fish_share overshot to 0.332
    # against a real 0.255 and the loading gain pinned at the grid FLOOR trying to compensate).
    # indiv/innov off. contain, kin_feedback and the guardrail stay: contain targets radial
    # occupancy which the latent does not touch and is the largest win in the lineage, removing
    # kin_feedback is the measured 'nofb' -0.068 disaster, and the guardrail is physics.
    #
    # READ THIS AS AN ABSOLUTE PANEL RESULT, NOT A CONTRAST. One arm means no paired comparison, so
    # the c2st_traj_auc is not adjudicable here (absolute AUCs are not comparable across runs). What
    # IS adjudicable is the pre-registered individuality panel, which is absolute: across-fish sd of
    # forward fraction (real 0.0537) and of median |dtheta| (real 6.735), and between_fish_share
    # (real 0.255). Those answer "can the latent ALONE generate a realistic population of fish?"
    "fishre_solo": ["mlp_joint_re_swap"],
    # TWO arms so the contrast is paired inside one call. Both share the same fitted net --
    # containment is generation-time and is fitted last -- so Metric A must be identical across
    # the two rows and any movement there is a bug. Endpoint: rad_mean / rad_median moving toward
    # the real 0.4556 WITHOUT wall_frac_gap getting worse, and then the paired AUC delta.
    "fishre_gr": ["mlp_joint_re_swap",          # control: shipped 1-D containment
                  "mlp_joint_re_swap_gr"],      # wide graded containment (bulk 0)
    # Solo variant. Valid because containment is generation-time and is fitted LAST, so the net
    # is bit-identical to the fishre_solo run and the (seed, fish, rep) streams stay aligned --
    # the same condition that made the re_swap_dc contrast pairable offline (t = +7.4 on 8
    # replicates). VALIDITY CHECK: Metric A must read 7.6791 / 4.5608 / 1.4921, exactly as in
    # ladder_reswap_results.zip. If it does not, the nets differ and the offline pairing is void.
    "fishre_gr_solo": ["mlp_joint_re_swap_gr"],
    # Ablation: the latent with NO generation-time corrections but the guardrail. Same net as
    # re_swap, so pair it offline. Expect the native radial profile (mean r/R already ~correct,
    # wall entry ~3x real) and the nofb failure mode on the kinematics (spread restored, free-run
    # autocorrelation blown out). It answers "how much of the model IS the corrections?".
    "fishre_bare": ["mlp_joint_re_swap_bare"],
    # Same model, 2-D containment search. Single arm again: the endpoint is the RADIAL profile,
    # which is absolute -- rad_mean / rad_median should move from 0.392 / 0.385 toward the real
    # 0.449 / 0.456 WITHOUT rad_frac_wall (gen 0.021 vs real 0.012) getting worse. If the mean
    # comes up and the wall holds, the decoupled search is the right parameterisation.
    "fishre_dc": ["mlp_joint_re_swap_dc"],
    # Run only after fishre: the swap claim, plus the latent-dimension check.
    "fishre2": ["mlp_joint_re_add",
                "mlp_joint_re_swap",
                "mlp_joint_re_m3"],
    "jk8both": ["mlp_joint_h128_shoal_contain_k8_kfb_vig"],
    "jkin8": ["mlp_joint_h128_shoal_contain_k8",
              "mlp_joint_h128_shoal_contain_k8_kfb",
              "mlp_joint_h128_shoal_contain_k8_vig",
              "mlp_joint_h128_shoal_contain_k8_kfb_vig"],
    "joint": ["mlp_joint_h128_shoal",
              "mlp_joint_h128_dot",
              "mlp_joint_h128_shoalr"],
    "jointc": ["mlp_joint_h128_shoal_contain",
               "mlp_joint_h128_dot_contain",
               "mlp_joint_h128_shoalr_contain"],
    "indiv": ["mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8",         # control
              "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8_indiv",   # + individuality
              "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8_innov",   # + innovation comp
              "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8_both"],   # both
    # PSI-FRAME LADDER (J10 recipe). Control FIRST and refit in the same call -- archived AUCs are
    # never anchors. One change per rung: psi_ref='travel'+dt_snap, then +border basis, then
    # +phi_ref='travel'. Mechanism endpoint (absolute, before any AUC): the RAW net's shell
    # turn-toward-centre profile (diag_border_profile with centre_bias zeroed) must rise from the
    # flat ~0.61 toward the real 0.72+, and the containment search inside fit() should pick a
    # SMALLER scale than the control's -- the patch shrinking is the root cause being removed.
    "psij": ["mlp_joint_h128_shoal_contain_kfb_vig_indiv",           # control, refit
             "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit",      # + travel-frame psi + dt_snap
             "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit_basis",  # + 9-ch border basis
             "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit_phit"],  # + travel-frame phi
    # 2-ARM SPLITS of psij, for parallel sessions (a 4-arm run is ~6 h and the 5 h fishre run
    # died once at 19/20). Every contrast keeps its control IN-CALL -- absolute AUCs are never
    # comparable across runs, so psij2/psij3 refit b_indiv_psi rather than importing psij1's.
    # The chain of one-change contrasts is: b_indiv -> psi (psij1), psi -> +basis (psij2),
    # psi -> +phi (psij3).
    "psij1": ["mlp_joint_h128_shoal_contain_kfb_vig_indiv",
              "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit"],
    "psij2": ["mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit",
              "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit_basis"],
    "psij3": ["mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit",
              "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit_phit"],
    # Same contrast on the TRUE J17 recipe (re_swap, shipped 1-D containment -- the better
    # control: 0.7285 vs re_swap_gr's 0.7321). The control is refit in-call, which also finally
    # produces a local ladder_re_swap.pt (the original was never retrieved from Colab). The
    # gr_psit zoo entries remain for offline pairing against the local graded checkpoints.
    "psire": ["mlp_joint_re_swap",                                   # control, refit
              "mlp_joint_re_swap_psit"],
    # Phase-5 rung: radial individuality on top of the psi-fixed fishre model. CRN-valid pair --
    # rad_indiv draws only from a SPAWNED stream (z stays child 0), so the control arm's rollouts
    # are byte-identical bout for bout. (Superseded by psire3: every generation-time radial lever
    # is measured dead on the psi/basis models.)
    "psire_urad": ["mlp_joint_re_swap_psit",
                   "mlp_joint_re_swap_psit_urad"],
    # The basis transfer to fishre. psire measured psi-alone at -0.0015 (nothing) with radial
    # still the top importance group (0.116, top-3 features all radial) -- exactly J10's
    # pre-basis signature, which psij2 resolved at -0.0385 (t=-3.5). Control refit in-call.
    "psire2": ["mlp_joint_re_swap_psit",
               "mlp_joint_re_swap_psit_basis"],
    # The re_bgain mechanism claim: control = the m=3 capacity control (same entry points, no
    # spatial pathway), treatment = + the fitted per-fish border-response gain. Isolates the
    # SPATIAL PATHWAY from latent capacity. (The m2->m3 question is deliberately not asked here
    # -- the user chose the 2-arm budget; basis(m2) numbers come from psire2.)
    "psire3": ["mlp_joint_re_swap_psit_basis_m3",
               "mlp_joint_re_swap_psit_basis_bgain"],
    # ===================================================================================== #
    #  2026-08-24 BLOCK: four new mechanisms on the J23 base (mlp_joint_re_swap_psit_basis,
    #  c2st_traj 0.6739). Run as 4 cells of 2 arms: wave 1 = wall1 + gain1 (nothing depends on
    #  anything), wave 2 = kstim1 + noise1.
    #
    #  NO SHARED CONTROL ARM, by the user's choice: all 8 slots go to new mechanisms and the
    #  reference is the archived J23. What replaces the control is an offline BIT-IDENTITY check
    #  (scripts/smoke_bit_identity.py) -- every new flag is key-gated, so with all of them absent
    #  a refit must reproduce psire2_out/ladder_re_swap_psib.pt exactly. Read the IN-CELL
    #  contrasts as evidence (paired, CRN, floor ~0.01) and the cross-cell ones as direction only
    #  (floor ~0.05 at reps=8).
    #
    #  Together with the archived J23 this is a 6-of-8-cell fractional factorial on
    #  {reject, noisy, kstim}: every main effect and every 2-way interaction is estimable, and
    #  each factor gets BOTH one single-factor arm and one exact in-cell paired contrast.
    # ===================================================================================== #
    # THE BOUNDARY. Replaces the shell ramp of centre_bias with rejection of samples that would
    # land in the shallow rim; `bulk` stays fitted with 0 first in its grid, so the search decides
    # whether any turn bias survives. PRE-REGISTERED, and already measured offline on the archived
    # J23 with no retrain (scripts/recalibrate_boundary.py): the search picks bulk = 0, i.e. NO
    # turn bias at all, and lands occ 0.0123 against a real 0.0121 with P(r/R>0.95) EXACTLY zero,
    # while the shipped bias reaches occ 0.0126 at the cost of mean r/R 0.4276 vs real 0.4556.
    # So the endpoint here is NOT "does the radial profile improve" -- that is already known -- it
    # is whether removing a 39 deg turn bias from the wall shell shows up in the C2ST.
    # In-cell contrast: does noise augmentation add anything ON TOP of the boundary.
    "wall1": ["mlp_joint_re_swap_psib_rej",
              "mlp_joint_re_swap_psib_rej_noise"],
    # THE LATENT. psire3 ran [m3, m3+bgain] and BOTH arms were m=3, so its in-cell contrast
    # (bgain -0.0935, the largest single-arm move in the lineage) was measured on a base that had
    # already failed: m3 came in at 0.8355 against psire2's m2 basis at 0.6739, and folds were
    # pinning the loading gain at the grid FLOOR (0.4 -- too WIDE a spread, not too narrow).
    # re_bgain is therefore UNTESTED, not refuted. This re-runs exactly that contrast with a grid
    # that can converge (re_gain_grid='wide', 8 points down to 0.10) and with the fitted gain
    # PRINTED every fold. Reading rule, pre-registered: a gain landing at 0.10-0.28 means the
    # third latent dimension is fitting noise across 12 training fish and m=2 is right.
    "gain1": ["mlp_joint_re_swap_psib_m3w",
              "mlp_joint_re_swap_psib_m3w_bgain"],
    # STIMULUS MEMORY. The net has never seen block phase or the previous presentations: the stim
    # block is a per-bout snapshot and every shoal channel is exactly 0 during a pause, while the
    # metrics have known since LATE_BLOCK_S that block phase matters. K_stim=3 adds 9 columns
    # (D 41 -> 50). READ THIS ON METRIC A FIRST: the joint ladder measured stimulus REPRESENTATION
    # changes (dot/shoalr/shoal) to be invisible to the trajectory C2ST, so a null there is not
    # evidence against the channel. In-cell contrast: does the boundary still pay once the model
    # has block context.
    "kstim1": ["mlp_joint_re_swap_psib_ks3",
               "mlp_joint_re_swap_psib_ks3_rej"],
    # NOISE AUGMENTATION alone (Sanchez-Gonzalez 2020 / DART / Data-as-Demonstrator): perturb the
    # POSE, re-derive every input from it, and correct the target back to the true next onset.
    # sigma_pos 4 px is calibrated, not guessed -- it puts the corrective turn at ~0.54x the real
    # turn sd while still straddling the innermost border-basis knot; at 8 px the correction is
    # LARGER than the behaviour (see MB.NOISE_AUG_DEFAULT for the measured table).
    # Expect Metric A on clean data to move: that is the cost of the trade, not a bug.
    # In-cell contrast: does stimulus memory pay on top of noise.
    "noise1": ["mlp_joint_re_swap_psib_noise",
               "mlp_joint_re_swap_psib_noise_ks3"],
    # ===================================================================================== #
    #  ROUND 2a (2026-08-24): the TIMING channel. Round 1 shipped `rej` (tied J23 at 0.6821
    #  while fixing the radial level exactly, zeroing the tail leak and removing the 39 deg
    #  shell turn bias) and refuted noise / K_stim / m=3. dt is now the largest tell for the
    #  best model -- importance 0.44 against radial 0.27 -- with dispersion 0.55,
    #  clock_ratio 0.888, quiesc20 -0.112, rate_acf_1min -0.268.
    #
    #  SINGLE-ARM ON PURPOSE, and read accordingly. There is no in-cell pairing, so
    #  c2st_traj_auc is NOT adjudicable here (cross-run floor ~0.05) -- the same rule
    #  fishre_solo carries. What IS adjudicable: `bits_per_bout_dt` paired per fish against
    #  ladder_psib_rej (fits and folds are deterministic, so Metric A is cross-run comparable
    #  -- that is how K_kin 3->8 was resolved at p=0.005), and the absolute guards
    #  clock_ratio -> 1.0, quiesc20 -> 0, rate_acf_1min -> 0, dt dispersion -> 1.0.
    # ===================================================================================== #
    # The cheap lever, and a KNOWN TRADE: dt -0.0218 bits (13/15 fish) against turn +0.0169
    # (0/15). Never carried into this lineage -- J19-J23 are all K_kin=3.
    "dt_k8": ["mlp_joint_re_swap_psib_rej_k8"],
    # The architectural answer: a slow AR(1) state on the interval head's location only, fitted
    # and marginalised exactly. It cannot reach the angle head, which is the whole point.
    "dt_lat": ["mlp_joint_re_swap_psib_rej_dtlat"],
    # ===================================================================================== #
    #  ROUND 3 (2026-08-25). Round 2a: both timing mechanisms did exactly what they were
    #  designed to do on the timing channel and NEITHER moved the AUC (K_kin=8 bought dt
    #  -0.0192 bits and paid turn +0.0108, RESOLVED worse; the latent bought dt -0.0229 at a
    #  turn cost of +0.0024, not resolved -- the architectural prediction confirmed).
    #
    #  The per-head individuality table says where the residual is. Across-fish SD of the
    #  per-fish mean, as % of real, for `rej`:
    #      fwd frac 80% | med|dth| 96% | dist 121% | dt 113% | RADIAL 32%
    #  Mean |log(gen/real)| over the four non-radial heads is 0.144; with radial it is 0.343.
    #  One channel more than doubles the individuality error.
    #
    #  SINGLE ARM EACH -- one model per session. The AUC is therefore NOT adjudicable
    #  (cross-run floor ~0.05); read the per-head individuality table, which is absolute, plus
    #  paired per-fish bits vs ladder_psib_rej, which is cross-run valid.
    # ===================================================================================== #
    # Radial individuality, likelihood-visible: a latent dimension that cannot be spent on
    # anything else. Endpoint: radial -> 100% WHILE THE OTHER FOUR HEADS STAY PUT (that second
    # clause is what separates a spatial channel from a global widening).
    "rad1": ["mlp_joint_re_swap_psib_rej_bgblk"],
    # The timing state with its timescale pinned at the measured value. Endpoint: the timing
    # gains SURVIVE (rate_acf_1min ~ -0.14, dt dispersion ~0.87) while the over-reach RETREATS
    # (dist 143% -> ~110%, dt 158% -> ~115%, dist_w1 4.70 -> ~3.3).
    "dt2": ["mlp_joint_re_swap_psib_rej_dtpin"],
    # DRIFT -- the one kind of object never tried: a WITHIN-trajectory slow state.
    #
    # THE MEASUREMENT THAT MOTIVATES IT (15 fish, W=60 bouts, GEN over 3 replicates). `acf1` is the
    # lag-1 autocorrelation of the WINDOW-MEAN series, computed within a fish, so every static
    # per-trajectory latent is removed by the demeaning and scores exactly zero by construction:
    #
    #     channel     REAL acf1        GEN acf1        gap
    #     dt        +0.332 +- 0.050  -0.001 +- 0.016   6.3 sigma
    #     distance  +0.337 +- 0.046  -0.002 +- 0.018   6.9 sigma
    #     radial    +0.377 +- 0.050  +0.043 +- 0.016   6.4 sigma
    #
    # fish_re z, rad_indiv, rad_gate and re_bgain are ALL static, which is why six mechanisms in a
    # row hit their own target and left the C2ST where it was -- none of them can move this number
    # at all. Only dt_latent is dynamic, and it loads on the interval head alone.
    #
    # ENDPOINT: the three acf1 values move from ~0 to 0.25-0.40 WHILE occupancy and the pooled
    # marginals hold (containment is re-fitted with the drift live, and the kinematic loadings are
    # re-centred so a scaled gain cannot inflate dist_w1 -- the failure mode the fishre gain hit).
    # If the acf moves and c2st_traj does not, that is the masking effect, not a dead mechanism:
    # read the acf table and var_ratio next to the AUC, never the AUC alone.
    "drift1": ["mlp_joint_re_swap_psib_rej_drift"],
    # ===================================================================================== #
    #  STAGE 2a of rollout-aware training: the RECURRENT model, MLE-pretrained.
    #
    #  A GRU state carried through the rollout replaces the entire hand-built latent zoo --
    #  drift's pinned AR(1), fish_re's static z, kin_feedback's shrinkage -- with ONE object
    #  whose timescales are LEARNED. Generation goes through the torch rollout
    #  (scripts/rollout_torch.py), verified against the numpy generator to within 0.5 sigma
    #  on nine statistics.
    #
    #  GATE: Metric A must BEAT the MLP baseline (7.5751 bits turn on psib_rej). A recurrent
    #  model with real memory that LOSES to a fixed-K window is mis-wired, and that is worth
    #  catching before spending a session on the adversarial stage.
    #
    #  Containment is deliberately NOT calibrated here (contain=False, no rejection, no
    #  kin_feedback): stage 2b is supposed to LEARN it. Expect wall occupancy to look bad at
    #  2a -- that is the baseline the GAN has to fix, not a defect of this rung.
    # ===================================================================================== #
    "gru1": ["mlp_joint_re_swap_psib_gru"],
    # gru1 + an explicit per-fish SEED for the recurrent state. gru1 measured:
    #   var_ratio distance 0.63 -> 1.04, dt 0.44 -> 0.91, radial 0.56 -> 0.84  (the WIN)
    #   between_fish_share 0.071 vs real 0.255, MLP 0.215                      (the GAP)
    # h_0 = 0 gave excellent WITHIN-fish slow structure and no BETWEEN-fish spread, so the
    # seed is given explicitly rather than hoped for. ENDPOINT: keep the var_ratio gains
    # while between_fish_share climbs toward 0.255. Metric A stays EXACT (the 2-d seed is a
    # GHQ grid whose nodes batch through state_chain, so it costs ~one pass, not Q).
    "gru2": ["mlp_joint_re_swap_psib_gruh"],
    # gruz: the GRU with the FITTED fish_re latent instead of a free h_0. gru1 and gru2 bracket
    # the answer -- gru1 has NO individuality (between_fish_share 0.071 vs real 0.255) and gru2's
    # free h_0 overshoots it (0.353, abs_dtheta_w1 blown out 7.78 -> 14.08). The two moved
    # var_ratio and between_fish_share INDEPENDENTLY, which is the measurement saying memory and
    # identity are orthogonal, so they get one mechanism each: GRU for within-fish slow structure,
    # z for between-fish identity. z is preferred over h_0 because it can only translate the three
    # head locations (so it cannot buy spread by distorting the marginals) and because its
    # magnitude is already validated at 0.215 on the MLP -- under, never over.
    # ENDPOINT: keep gru1's var_ratio (distance ~1.04, dt ~0.91) with between_fish_share climbing
    # from 0.071 toward 0.255. Metric A stays EXACT -- z is static, so it is the same 49-node GHQ
    # grid batched through state_chain that gru2 uses, not a product grid.
    # WATCH: if the fitted z loadings come out much SMALLER than J23's, the GRU is inferring
    # identity from its own history and the two ARE double-counting after all.
    "gruz": ["mlp_joint_re_swap_psib_gruz"],
    # ===================================================================================== #
    #  SRNN STAGE 1 -- one FITTED mechanism per variance component, all under one likelihood.
    #
    #  The metric reads a three-way decomposition (between-fish / within-fish window-to-window /
    #  within-window) and until now no arm had one mechanism per term: gruz stacked a static
    #  latent on top of a state whose timescale nothing pinned.
    #    z    static per fish        -> between-fish
    #    s_t  pinned AR(1) tau = 70  -> within-fish slow. tau > W = 60 ON PURPOSE, so it
    #         correlates ADJACENT windows rather than merely randomising each one.
    #    h_t  GRU, state 64 -> 32    -> fast context, deliberately bounded
    #
    #  AFFORDABLE BECAUSE h_t DOES NOT DEPEND ON s. The GRU is deterministic given the observed
    #  y, so state_chain_levels runs it ONCE and evaluates the 9 drift levels on the three small
    #  heads only -- 49 z-nodes x 9 levels = 441 states/bout, the figure the dt_latent arm
    #  already paid, not the 3969 that got two AR(1) states refused.
    #
    #  CONTAINMENT AND kin_feedback ARE BACK ON. Every gru arm ran with contain=False while
    #  radial carries ~half the C2ST permutation importance, so none of them was comparable to
    #  rej. This rung is the first GRU-family arm that is.
    #
    #  ENDPOINT: within_ratio and between_ratio (now printed) both -> 1.00 per group, with
    #  win_acf1 climbing toward +0.33/+0.34/+0.38. Metric A stays EXACT -- the recursion is a
    #  marginalisation, not a bound -- but it is the JOINT per-bout density, so read
    #  bits_per_bout against nll_seq_drift arms, not against the plain state arms.
    # ===================================================================================== #
    #  PAIRED, and that is not optional. The first srnn1 run shipped a ONE-ARM ladder, whose AUC
    #  (0.7230) cannot be read against rej's 0.6821 at all: "Absolute AUC values are not comparable
    #  across experiments -- the identical model reads 0.7714/0.7681/0.7712/0.7712 in four
    #  different ladders. Only the within-ladder paired delta is a measurement."
    #
    #  rej is the control: same containment (reject), same kin_feedback, same latent -- it differs
    #  from the srnn arm ONLY by the GRU state and the pinned AR(1) drift. So the delta is
    #  attributable to those two and nothing else.
    #
    #  WHAT THE FIRST RUN ESTABLISHED, and what still needs a control:
    #    won  -- win_acf1 dt and distance ON TARGET every fold (+0.32..+0.43 vs real +0.29..+0.40)
    #            against rej's +0.021/+0.008. The 6-sigma defect, closed on two of three channels.
    #    won  -- gruz's dt blow-up fixed: var_ratio 5.31 -> 0.90, within-fish 4.09 -> 0.93.
    #    open -- radial untouched (acf +0.02 vs real +0.37, between 0.26) and it is 41% of the
    #            permutation importance. One shared 1-D latent cannot serve kinematics AND radial.
    #    NEW DEFECT -- cpl_dt_dist gap +0.98 (gruz -0.08). Real corr(log dt, log dist) = -0.386,
    #            but the fitted loadings are both POSITIVE (dist +0.15, dt +0.09), so the shared
    #            latent forces positive co-variation. Fixing the acf broke the coupling.
    "srnn1": ["mlp_joint_re_swap_psib_rej",              # control = the current best
              "mlp_joint_re_swap_psib_srnn"],
}
LABEL = {"mlp_border_full_h128_mix_dotb_hloop_kin_vigor": "dotb",
         "mlp_border_full_h128_mix_dot_hloop_kin_vigor": "dot",
         "mlp_border_full_h128_mix_shoalr_hloop_kin_vigor": "shoalr",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor": "shoal",
         "mlp_border_full_h128_mix_dotb_wall_hloop_kin_vigor": "dotb_wall",
         "mlp_border_full_h128_mix_shoal_wall_hloop_kin_vigor": "shoal_wall",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_drift": "shoal_drift",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_kfix": "shoal_kfix",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_kfixvc": "kfix_vc",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_kfixvc_turn": "kfix_vc_turn",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain": "contain",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_vc": "contain_vc",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail": "contain_tail",
         "mlp_border_full_h128_mix_shoalf_hloop_kin_vigor_contain_tail": "shoalf_tail",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_cal1": "cal_v1",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_e250": "e250",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_nofb": "nofb",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8_nofb": "k8_nofb",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8": "k8",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8_indiv": "k8_indiv",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8_innov": "k8_innov",
         "mlp_border_full_h128_mix_shoal_hloop_kin_vigor_contain_tail_k8_both": "k8_both",
         "mlp_joint_h128_shoal": "j_shoal", "mlp_joint_h128_dot": "j_dot",
         "mlp_joint_h128_shoalr": "j_shoalr",
         "mlp_joint_h128_shoal_contain": "jc_shoal", "mlp_joint_h128_dot_contain": "jc_dot",
         "mlp_joint_h128_shoalr_contain": "jc_shoalr",
         "mlp_joint_h128_shoal_contain_kfb": "k3_kfb",
         "mlp_joint_h128_shoal_contain_vig": "k3_vig",
         "mlp_joint_h128_shoal_contain_kfb_vig": "k3_both",
         "mlp_joint_h128_shoal_contain_k8": "k8_base",
         "mlp_joint_h128_shoal_contain_k8_kfb": "k8_kfb",
         "mlp_joint_h128_shoal_contain_k8_vig": "k8_vig",
         "mlp_joint_h128_shoal_contain_k8_kfb_vig": "k8_both",
         "mlp_joint_re_add": "re_add", "mlp_joint_re0": "re0",
         "mlp_joint_re_shuf": "re_shuf", "mlp_joint_re_swap": "re_swap",
         "mlp_joint_re_m3": "re_m3", "mlp_joint_re_swap_dc": "re_swap_dc", "mlp_joint_re_swap_gr": "re_swap_gr", "mlp_joint_re_swap_bare": "re_swap_bare",
         "mlp_joint_h128_shoal_contain_kfb_vig_indiv": "b_indiv",
         "mlp_joint_h128_shoal_contain_kfb_vig_innov": "b_innov",
         "mlp_joint_h128_shoal_contain_kfb_vig_indinn": "b_both",
         "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit": "b_indiv_psi",
         "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit_basis": "b_indiv_psib",
         "mlp_joint_h128_shoal_contain_kfb_vig_indiv_psit_phit": "b_indiv_psiphi",
         "mlp_joint_re_swap_gr_psit": "re_gr_psi",
         "mlp_joint_re_swap_gr_psit_urad": "re_gr_psiur",
         "mlp_joint_re_swap_psit": "re_swap_psi",
         "mlp_joint_re_swap_psit_urad": "re_swap_psiur",
         "mlp_joint_re_swap_psit_basis": "re_swap_psib",
         "mlp_joint_re_swap_psit_basis_m3": "re_swap_psib3",
         "mlp_joint_re_swap_psit_basis_bgain": "re_swap_psibg",
         # 2026-08-24 block (wall1 / gain1 / kstim1 / noise1)
         "mlp_joint_re_swap_psib_rej": "psib_rej",
         "mlp_joint_re_swap_psib_rej_noise": "psib_rej_noise",
         "mlp_joint_re_swap_psib_m3w": "psib_m3w",
         "mlp_joint_re_swap_psib_m3w_bgain": "psib_m3w_bgain",
         "mlp_joint_re_swap_psib_ks3": "psib_ks3",
         "mlp_joint_re_swap_psib_ks3_rej": "psib_ks3_rej",
         "mlp_joint_re_swap_psib_noise": "psib_noise",
         "mlp_joint_re_swap_psib_noise_ks3": "psib_noise_ks3",
         # round 2a
         "mlp_joint_re_swap_psib_rej_k8": "psib_rej_k8",
         "mlp_joint_re_swap_psib_rej_dtlat": "psib_rej_dtlat",
         # round 3
         "mlp_joint_re_swap_psib_rej_bgblk": "psib_rej_bgblk",
         "mlp_joint_re_swap_psib_rej_dtpin": "psib_rej_dtpin",
         "mlp_joint_re_swap_psib_rej_drift": "psib_rej_drift",
         "mlp_joint_re_swap_psib_gru": "psib_gru",
         "mlp_joint_re_swap_psib_gruh": "psib_gruh",
         "mlp_joint_re_swap_psib_gruz": "psib_gruz",
         "mlp_joint_re_swap_psib_srnn": "psib_srnn"}


def load_quick_results():
    """Exec the notebook cell that defines MODEL_ZOO + quick_results (single source of truth)."""
    nb = json.load(open(ROOT / "notebooks" / "evaluate_models.ipynb", encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "def quick_results" in src and "MODEL_ZOO" in src:
            g = {"pd": pd, "np": np, "E": E, "GM": GM, "MB": MB, "time": time, "__name__": "__main__"}
            exec(compile(src, "quick_results_cell", "exec"), g)
            return g["quick_results"], g["MODEL_ZOO"]
    raise SystemExit("could not find the quick_results cell in evaluate_models.ipynb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--R", type=int, default=12, help="panel rollouts per held-out fish")
    ap.add_argument("--reps", type=int, default=8, help="independent C2ST rollout replicates")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stopping patience; 8 may be cutting fits off short")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ladder", default="indiv", choices=sorted(LADDERS),
                    help="which comparison to run (step 1 = stim, step 2 = wall)")
    ap.add_argument("--smoke", action="store_true", help="wiring check: skip the standalone fits")
    ap.add_argument("--force-epochs", type=int, default=None,
                    help="override the rungs' OWN epochs_override too. Needed for a fast wiring "
                         "check of any ladder whose rungs set epochs_override (mem, conv): those "
                         "beat ctx.cfg by design, so plain --epochs leaves them at 250 and a "
                         "'smoke' run takes hours. Never use this for a real result -- it would "
                         "silently retrain the control at a different budget than it was tuned at.")
    args = ap.parse_args()

    LADDER = LADDERS[args.ladder]
    STIM_OF = LABEL
    print(f"ladder '{args.ladder}': " + " -> ".join(LABEL[m] for m in LADDER))
    if args.out is None:
        args.out = str(ROOT / f"ladder_{args.ladder}_results.zip")
    quick_results, zoo = load_quick_results()
    missing = [m for m in LADDER if m not in zoo]
    if missing:
        raise SystemExit(f"zoo is missing {missing} -- your bundle predates the shoal block; re-upload it")
    if args.force_epochs:
        # `zoo` IS the dict quick_results closes over, so rebinding entries here reaches the fit.
        print(f"** --force-epochs {args.force_epochs}: WIRING CHECK ONLY, results are not comparable **")
        for _n in LADDER:
            def _mk(_base=zoo[_n], _ep=args.force_epochs, _pa=args.patience):
                m = _base(); m.epochs_override = _ep; m.patience_override = _pa; return m
            zoo[_n] = _mk

    print(f"records <- {args.bouts}")
    records = MB.build_border_records(pd.read_csv(args.bouts))
    r0 = records[sorted(records)[0]]
    if getattr(r0, "n_dots", None) is None:
        raise SystemExit("records have no rec.n_dots -- the bouts CSV lacks n_dots_left/right columns")
    if getattr(r0, "dot_dist_px", None) is None:
        raise SystemExit("records have no dot geometry -- run scripts/add_dot_geometry.py first")
    print(f"  {len(records)} fish, {sum(records[f].n for f in records)} bouts, dot geometry OK")

    t0 = time.time()
    res, likA, panel, wins = quick_results(
        LADDER, records, n_folds=args.folds, R=args.R, seed=args.seed,
        cfg=dict(mlp_epochs=args.epochs, mlp_patience=args.patience), c2st_reps=args.reps)
    print(f"\nladder finished in {(time.time() - t0) / 60:.1f} min")

    # wacf_*_gap were COMPUTED into panel.csv all along (generative_metrics.py:400-404) but never
    # printed, so the one measurement the drift/GRU line exists to move was sitting unread in every
    # results zip. They are gen-minus-real, so 0 is the target and the sign says which way it misses.
    cols = [c for c in ["bits_per_bout", "bits_dist", "bits_dt",
                        "c2st_auc", "c2st_auc_sd", "c2st_traj_auc", "c2st_traj_auc_sd",
                        "wacf_dt_gap", "wacf_dist_gap", "wacf_rad_gap",
                        "abs_dtheta_w1", "dist_w1", "dt_w1", "acf_l2",
                        "following_gap", "following_gap_h",
                        "following_gap_static", "following_gap_static_h",
                        "dot_dist_bias", "dot_dist_bias_late", "wall_frac_gap"] if c in res.columns]
    pd.set_option("display.width", 250)
    print("\n" + res[cols].round(4).to_string())

    nc = res.attrs.get("neg_control_traj", float("nan"))
    print(f"\ntraj C2ST floor (real-vs-real negative control): {nc:.3f}"
          "   <- read every AUC against THIS, not 0.5")
    base = LADDER[0]
    tr = res.attrs.get("auc_traj_reps", {})
    if base in tr and len(tr[base]) > 1:
        # PAIRED on replicate index. Replicate r of every model is generated from the same
        # (seed, fish, rep) stream, so the per-replicate DIFFERENCE cancels the shared rollout
        # noise -- which is the entire point of common random numbers. Combining the two models'
        # marginal SDs instead would throw that away and overstate the SE.
        print(f"\npaired contrasts vs {base} (c2st_traj_auc; negative = better):")
        b = np.asarray(tr[base], float)
        for m in LADDER[1:]:
            if m not in tr or len(tr[m]) != len(b):
                continue
            d = np.asarray(tr[m], float) - b
            n = len(d)
            se = float(np.std(d, ddof=1) / np.sqrt(n))
            t = float(np.mean(d)) / max(se, 1e-12)
            # n is small, so use a t critical value rather than 1.96
            from scipy import stats as _st
            crit = float(_st.t.ppf(0.975, n - 1))
            flag = "RESOLVED" if abs(t) > crit else "within noise"
            print(f"  {STIM_OF[m]:7s} {np.mean(d):+.4f}  (paired SE {se:.4f}, t={t:+.1f}, "
                  f"n={n}, crit {crit:.1f})  {flag}")
        if len(b) < 4:
            print("  NOTE: with <4 replicates the SE estimate is itself very noisy -- these")
            print("        RESOLVED/within-noise labels are only trustworthy at --reps 8+.")

    # ---- TIMING / REALISM guards, printed next to the AUC ---------------------------------- #
    # Two rungs have now won the AUC while degrading an uncounted property: kin_feedback took
    # -0.0407 while halving the distance dispersion, and e250 took -0.0113 while cutting
    # quiescent time from 13.0% to 6.9% of the session (real 24.4%) and finishing the 174-min
    # stimulus programme in 142. dispersion_report caught the first; this catches the second.
    # A rung that wins while pushing any of these AWAY from target is that trade again.
    print("\ntiming / realism guards   (all *_gap -> 0, ratios -> 1, sd_gen -> sd_real)")
    keys = ["quiesc20_gap", "quiesc60_gap", "clock_ratio", "dt_tail20_gap",
            "rate_acf_1min_gap", "cpl_dt_dist_gap", "cpl_dt_turn_gap", "dist_std_ratio",
            "bounce_rate", "fwd_frac_sd_gen", "absdth_median_sd_gen"]
    hdr = "  " + f"{'model':10s}" + "".join(f"{k.replace('_gap','').replace('_sd_gen','_sd'):>15s}" for k in keys)
    print(hdr)
    for name in LADDER:
        t = GM.timing_report(panel, name)
        if not t:
            continue
        print("  " + f"{LABEL[name]:10s}" + "".join(f"{t.get(k, float('nan')):15.4f}" for k in keys))
    t0r = GM.timing_report(panel, LADDER[0])
    print("  " + f"{'REAL':10s}" + "".join(
        f"{(0.0 if k.endswith('_gap') else 1.0 if k.endswith('ratio') else t0r.get('fwd_frac_sd_real', float('nan')) if k.startswith('fwd') else t0r.get('absdth_median_sd_real', float('nan')) if k.startswith('absdth') else 0.0):15.4f}"
        for k in keys))

    # ---- MARGINAL-EQUALISED C2ST ------------------------------------------------------------ #
    # Same rollouts with each behavioural channel rank-mapped onto that fish's real marginal.
    # The raw AUC confounds "wrong levels" with "wrong temporal structure": raw, the shipped
    # model looks like a surrogate preserving ~1 bout of real structure; equalised, ~2.8. Read
    # the pair -- a rung that improves raw but not equalised fixed marginals, and vice versa.
    # This is NOT a model score on its own; a real model must get its marginals right too.
    weq = panel.attrs.get("windows_traj_eq") or {}
    if weq:
        print("\nmarginal-equalised traj C2ST (raw -> equalised; the drop is the marginal's share)")
        for name in LADDER:
            if name not in weq:
                continue
            m_eq, sd_eq, _ = GM.c2st_auc_replicates(weq[name])
            raw = float(res.loc[name, "c2st_traj_auc"]) if name in res.index else float("nan")
            print(f"  {LABEL[name]:10s} raw {raw:.4f}   equalised {m_eq:.4f} (sd {sd_eq:.4f})"
                  f"   marginal share {raw - m_eq:+.4f}")

    # ---- WRITE THE SCIENCE FIRST ----------------------------------------------------------- #
    # The standalone checkpoints below cost another 4 full fits (~40-60 min at 250 epochs) and are
    # only a local-diagnostics convenience -- they can be refit any time. The CV result cannot.
    # An 8-hour run once lost everything because the zip was written after them and the session
    # dropped in between, so everything recoverable is now on disk BEFORE that.
    res.to_csv(ROOT / "results.csv")
    panel.to_csv(ROOT / "panel.csv", index=False)
    likA.to_csv(ROOT / "likA.csv", index=False)
    with open(ROOT / "artifacts.pkl", "wb") as fh:
        pickle.dump(dict(wins_traj=res.attrs.get("wins_traj"),
                         wins_traj_eq=weq,
                         timing=({m: GM.timing_report(panel, m) for m in LADDER}),
                         auc_reps=res.attrs.get("auc_reps"),
                         auc_traj_reps=res.attrs.get("auc_traj_reps"),
                         neg_control_traj=nc,
                         traj_feature_names=GM.C2ST_TRAJ_FEATURE_NAMES,
                         config=vars(args)), fh)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["results.csv", "panel.csv", "likA.csv", "artifacts.pkl"]:
            z.write(ROOT / f, f)
    print(f"\nwrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) -- DOWNLOAD THIS NOW.")
    print("The comparison is complete and safe. Everything below is optional and refittable.")

    # ---- standalone all-fish checkpoints, so the range-resolved acceptance test can run locally --
    saved = []
    if not args.smoke:
        print("\nfitting standalone all-fish models for local diagnostics ...")
        train = [records[f] for f in sorted(records)]
        ctx = E.make_context(train, dict(mlp_epochs=args.epochs, mlp_patience=args.patience))
        for name in LADDER:
            m = zoo[name]()
            m.name = name
            m.fit(train, ctx)
            p = ROOT / f"ladder_{LABEL[name]}.pt"
            MX.save_mix(m, str(p))
            saved.append(p)
            print(f"  saved {p.name}")
            with zipfile.ZipFile(args.out, "a", zipfile.ZIP_DEFLATED) as z:
                z.write(p, p.name)          # append as each one lands, not all at the end
    print(f"\nfinal {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
