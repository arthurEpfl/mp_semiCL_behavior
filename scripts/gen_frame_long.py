"""Frame-level ``gen_fish_df_long`` for simulated rollouts, synthesized from their bout
sequences via the codebase's existing "dart-and-hold" reconstruction -- the same one
``scripts/bout_video.py``/``scripts/tool_video.py`` use to render rollout comparison videos
-- then run through the same downstream pipeline ``real_frame_long.py`` uses for real fish.

Generated rollouts are bout-native, not frame-native: the model predicts ``(dtheta,
distance_px, dt_s)`` per bout and the position update is an instantaneous straight-line jump
(``fit_MLP_joint.generate_mix_joint_loop``); there is no simulated information about what
happens between two bout onsets. This module's fish-pose reconstruction is therefore a
*synthesis* layered on top of the bout data (linear ramp over ``bout_frames`` frames after
each onset, then hold), not a recovery of anything the model computed -- see
``bout_video._synthetic_path``, reused here unchanged.

Dot positions come from the SAME stimulus programme real fish were shown
(``stim_program.StimulusProgram``), projected through the anchor pose this particular
rollout latched for each stimulus block (``g["anchors"]``, set by ``ProgramStimProvider``
during generation) -- vectorized per block via ``stim_program.anchor_offsets`` rather than
per frame (a full rollout is ~315k frames; ``tool_video.write_panel``'s per-frame loop is
fine for a short video window but would not scale here).

Pause-copy parity: real per-fish CSVs don't leave dot columns at ``'none'`` during
``inter_stim_pause`` -- ``src.analyse.pause_copy_trajectory`` copies the *upcoming* stimulus
block's dot trajectory into the pause so ``dots_center``/``squareDist_rot`` etc. stay defined
through pauses too. Reproduced here once at the programme level (shared by every rollout,
since the programme itself is exogenous/fish-agnostic) so the ``'inter_stim_pause'`` control
curve isn't spuriously empty for generated data while real data shows a real one.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import bout_video as BV
import tool_video as TV
from stim_program import StimulusProgram, anchor_offsets
from real_frame_long import _assemble_fish_df_long

_NAME_RE = re.compile(r'trajectory_(?P<experiment>[^_]+)_')


def _pause_corrected_dots(program: StimulusProgram):
    """``(dx_eff, dy_eff, label_col, dstim_cols)`` -- ``program.dx``/``dy``/``active`` with
    each pause block's dots overwritten by a copy of the following stimulus block's first
    frames (matching ``src.analyse.pause_copy_trajectory``), plus the per-frame raw condition
    label and per-dot ``d{i}_stim`` string columns real per-fish CSVs carry. The trailing
    pause block (if any -- the programme need not end mid-stimulus) has no following block to
    copy from and is left inactive, same as it would be for a real fish with no next trial."""
    n_frames, n_dots = program.dx.shape
    dx_eff, dy_eff, active_eff = program.dx.copy(), program.dy.copy(), program.active.copy()
    label_col = np.empty(n_frames, dtype=object)
    for b in range(program.n_blocks):
        s, e = int(program.starts[b]), int(program.ends[b])
        is_pause = program.labels[b] == 'none'
        label_col[s:e] = 'inter_stim_pause' if is_pause else program.labels[b]
        if is_pause and b + 1 < program.n_blocks and program.is_stim_of[b + 1]:
            s2, e2 = int(program.starts[b + 1]), int(program.ends[b + 1])
            n_copy = min(e - s, e2 - s2)
            dx_eff[s:s + n_copy] = program.dx[s2:s2 + n_copy]
            dy_eff[s:s + n_copy] = program.dy[s2:s2 + n_copy]
            active_eff[s:s + n_copy] = program.active[s2:s2 + n_copy]

    # Active + pause-block frame -> 'inter_stim_pause' (matches pause_copy_trajectory's own
    # `.mask(mask_stim, 'inter_stim_pause')`, not the borrowed block's own label); active +
    # stim-block frame -> that block's raw label; inactive -> 'none'.
    dstim_cols = {k: np.where(active_eff[:, k], label_col, 'none') for k in range(n_dots)}
    return dx_eff, dy_eff, label_col, dstim_cols


def build_gen_fish_df_long(rollouts, program: StimulusProgram, bout_frames: int = 3,
                           framerate: int = 30, px_to_mm: float = 3.67):
    """Return ``(gen_fish_df_long, meta)``, shaped like
    ``real_frame_long.build_fish_df_long``'s return, for the ``rollouts`` list
    ``stim_rollout.rollout_many`` produced against ``program``. See module docstring for the
    bout -> frame synthesis this relies on."""
    t0 = time.time()
    m = _NAME_RE.search(Path(program.path).name)
    experiment = m.group('experiment') if m else 'unknown'
    n_stimuli = program.n_dots

    dx_eff, dy_eff, label_col, dstim_cols = _pause_corrected_dots(program)
    frames = np.arange(program.n_frames)

    fish_raw = {}
    for k, g in enumerate(rollouts):
        onset = np.asarray(g['frame_onset'], int)
        bx, by = np.asarray(g['x_px'], float), np.asarray(g['y_px'], float)
        sx, sy, so = BV._synthetic_path(frames, onset, bx, by, bout_frames,
                                        h0=float(g['start_pose'][2]))

        X = np.empty((program.n_frames, n_stimuli))
        Y = np.empty((program.n_frames, n_stimuli))
        for b in range(program.n_blocks):
            s, e = int(program.starts[b]), int(program.ends[b])
            ax, ay, aori = TV._anchor_for(g, program, b, s)
            ox, oy = anchor_offsets(dx_eff[s:e], dy_eff[s:e], aori)
            X[s:e], Y[s:e] = ax + ox, ay + oy

        cols = {'frame': frames, 'stimulus': label_col, 'x': sx, 'y': sy, 'theta': so}
        for i in range(n_stimuli):
            cols[f'd{i}_x'] = X[:, i]
            cols[f'd{i}_y'] = Y[:, i]
            cols[f'd{i}_stim'] = dstim_cols[i]
        fish_raw[k] = pd.DataFrame(cols)

        held_s = program.duration_s - g['last_onset_s']
        print(f'  rollout {k}: synthesized, held for the final {held_s:.1f}s '
             f'(ran out of bouts before the programme ended)')

    dish_for = lambda k: rollouts[k]['dish']
    long_df = _assemble_fish_df_long(fish_raw, experiment, n_stimuli, dish_for, framerate, px_to_mm)

    print(f'{len(rollouts)} rollouts synthesized to frames in {time.time() - t0:.1f}s')
    return long_df, {'experiment': experiment, 'n_fish': len(rollouts), 'n_stimuli': n_stimuli}
