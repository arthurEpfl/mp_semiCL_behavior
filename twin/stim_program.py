"""Parse a Bonsai stimulus *trajectory* file into a replayable stimulus program.

A trajectory file (``*_trajectory_GsizeMotionSingle_*.csv``) is the thing the rig actually
plays: one row per frame at 30 fps, ``6 x n_dots`` headerless columns in the repeating order
``name, x, y, size, color, backg`` (the layout ``src/transformed_data_pipeline.py`` assumes).
``x``/``y`` are **millimetres relative to the fish centroid**, left = +x.

It is FISH-AGNOSTIC: the same program is played to every dish, and ``CLsemi`` means the dot
track is anchored to each fish's own pose at block onset and then replayed open-loop in that
latched frame. So a trajectory file alone does not determine where the dots are -- it needs an
anchor pose per block. That is exactly what makes it reusable for a *simulated* fish.

THE ANCHORING TRANSFORM (``anchor_offsets``) is the one piece of new maths here. Composing
``analyse.compute_stimuli_absolute`` (rotate by ``f_ori - pi/2``, offset by the onset position)
with the later y-flip / CW->CCW step in ``analyse.transform_coordinates`` collapses to a single
rotation in the transformed analysis frame. Verified against the shipped
``data/data/df_transformed/*_transformed_abs.csv`` to **2.8e-14 px** (``--verify``).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from twin.segment_bouts import PX_PER_MM, STIM_NONE, parse_stimulus

FPS = 30.0
N_DOTS = 8
FIELDS_PER_DOT = 6          # name, x, y, size, color, backg
BOUT_LABEL = "bout"         # trailing calibration block, not part of the assay


# --------------------------------------------------------------------------- #
# The anchoring transform
# --------------------------------------------------------------------------- #
def anchor_offsets(dx, dy, ori):
    """Fish-relative (dx, dy) [px] -> offsets in the transformed absolute frame, for a fish
    whose transformed pose has orientation ``ori`` (rad, CCW, y already flipped).

    Derivation: the raw pipeline rotates by ``heading = f_ori_raw - pi/2`` and offsets by the
    raw onset position, then flips ``y -> diameter - y`` and ``f_ori_raw -> 2*pi - f_ori``.
    Writing the transformed orientation as ``ori``, the raw heading is ``-ori - pi/2``, so
    ``cos h = -sin(ori)``, ``sin h = -cos(ori)``; the y-flip negates the rotated y-offset.

    Sanity: a fish at ``ori = 0`` (facing +x) with a dot at ``(dx > 0, dy = 0)`` gets
    ``(0, +dx)`` -- i.e. to its left, matching the file's "left = positive x" convention.
    """
    s, c = np.sin(ori), np.cos(ori)
    return (-dx * s + dy * c,
            +dx * c + dy * s)


# --------------------------------------------------------------------------- #
# The program
# --------------------------------------------------------------------------- #
@dataclass
class StimulusProgram:
    """A frame-indexed stimulus program plus its block structure.

    Per frame: ``dx``/``dy`` (n_frames, n_dots) fish-relative offsets in PIXELS, ``active``
    (n_frames, n_dots) bool, ``block_id`` (n_frames,).
    Per block: ``labels``, ``n_dots_of``, ``moving_of``, ``side_of``, ``is_stim_of``,
    ``starts``, ``ends`` -- all indexed by block id, so a provider lookup is O(1).
    """
    dx: np.ndarray
    dy: np.ndarray
    active: np.ndarray
    block_id: np.ndarray
    labels: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    n_dots_of: np.ndarray
    moving_of: np.ndarray
    side_of: np.ndarray
    is_stim_of: np.ndarray
    fps: float = FPS
    path: str = ""

    # -- basics ------------------------------------------------------------- #
    @property
    def n_frames(self) -> int:
        return self.dx.shape[0]

    @property
    def n_dots(self) -> int:
        return self.dx.shape[1]

    @property
    def n_blocks(self) -> int:
        return len(self.labels)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def frame_of(self, time_s) -> int:
        """Wall-clock seconds -> frame index (may be >= n_frames; callers must range-check)."""
        return int(round(float(time_s) * self.fps))

    # -- geometry ----------------------------------------------------------- #
    def dots_absolute_all(self, frame, anchor):
        """(X, Y) absolute positions [px] of ALL dot slots at ``frame`` (length ``n_dots``), for a
        block anchored at ``anchor = (ax, ay, aori)``. Inactive slots carry ``(0, 0)`` mm and so
        land exactly on the anchor -- the same convention the shipped transformed CSVs use, which
        is why they show inactive dots sitting on the fish rather than as NaN."""
        ax, ay, aori = anchor
        ox, oy = anchor_offsets(self.dx[frame], self.dy[frame], aori)
        return ax + ox, ay + oy

    def dots_absolute(self, frame, anchor):
        """(X, Y) absolute positions [px] of the ACTIVE dots at ``frame``. Empty when none."""
        m = self.active[frame]
        if not m.any():
            return np.empty(0), np.empty(0)
        X, Y = self.dots_absolute_all(frame, anchor)
        return X[m], Y[m]

    def centroid(self, frame, anchor):
        """Active-dot centroid [px]. Matches ``segment_bouts.add_dist_to_dot``'s ``dot_cx/dot_cy``
        (unweighted mean over active dots only). ``(nan, nan)`` when nothing is active."""
        X, Y = self.dots_absolute(frame, anchor)
        if X.size == 0:
            return np.nan, np.nan
        return float(X.mean()), float(Y.mean())

    # -- reporting ---------------------------------------------------------- #
    def block_table(self) -> pd.DataFrame:
        return pd.DataFrame(dict(
            block=np.arange(self.n_blocks), label=self.labels,
            start_frame=self.starts, end_frame=self.ends,
            dur_s=(self.ends - self.starts) / self.fps,
            n_dots=self.n_dots_of, moving=self.moving_of,
            side=self.side_of, is_stim=self.is_stim_of))

    def summary(self) -> str:
        t = self.block_table()
        st = t[t["is_stim"]]
        return (f"{Path(self.path).name or '<program>'}: {self.n_frames} frames "
                f"({self.duration_s / 60:.1f} min @ {self.fps:g} fps), {self.n_dots} dot slots\n"
                f"  {self.n_blocks} blocks -> {len(st)} stimulus / {self.n_blocks - len(st)} pause\n"
                f"  {st['label'].nunique()} conditions; N in "
                f"{sorted(set(st['n_dots'].astype(int)))}; "
                f"moving {int(st['moving'].sum())} / static {int((~st['moving']).sum())}")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_program(path, n_dots=N_DOTS, px_per_mm=PX_PER_MM, fps=FPS, drop_bout=True):
    """Read a trajectory CSV into a :class:`StimulusProgram`.

    ``drop_bout`` truncates the trailing ``bout`` calibration block (9000 frames in the shipped
    file), which is not part of the assay and which the transform pipeline also splits off.
    """
    path = str(path)
    raw = pd.read_csv(path, header=None)
    ncol = raw.shape[1]
    if ncol % FIELDS_PER_DOT:
        raise ValueError(f"{path}: {ncol} columns is not a multiple of {FIELDS_PER_DOT} "
                         f"(expected '{'name,x,y,size,color,backg'}' per dot)")
    found = ncol // FIELDS_PER_DOT
    if found != n_dots:
        raise ValueError(f"{path}: {found} dot slots, expected {n_dots}. The downstream feature "
                         "code hard-codes 8 slots -- pad the file to 8 'none,0,0,0,1,0' groups.")

    name_c = [FIELDS_PER_DOT * k + 0 for k in range(n_dots)]
    x_c = [FIELDS_PER_DOT * k + 1 for k in range(n_dots)]
    y_c = [FIELDS_PER_DOT * k + 2 for k in range(n_dots)]
    size_c = [FIELDS_PER_DOT * k + 3 for k in range(n_dots)]

    names = raw[name_c].astype(str).to_numpy()                     # (F, K)
    active = ~np.isin(names, list(STIM_NONE))
    size_pos = raw[size_c].to_numpy(float) > 0
    dx = raw[x_c].to_numpy(float) * px_per_mm
    dy = raw[y_c].to_numpy(float) * px_per_mm
    # The block label lives in the ACTIVE slots only; slot 0 is active in every stimulus block
    # (every CLsemi condition has N >= 1), so row 0's name is the label. Assert rather than assume.
    label_per_frame = names[:, 0]
    stim_rows = active.any(axis=1)
    if not np.all(active[stim_rows, 0]):
        raise ValueError(f"{path}: some frames have active dots but slot 0 inactive; "
                         "the block label cannot be read from slot 0.")
    del raw, names

    # ---- run-length encode into blocks ---------------------------------- #
    starts, ends, labels = [], [], []
    pos = 0
    for lab, grp in itertools.groupby(label_per_frame):
        ln = len(list(grp))
        starts.append(pos); ends.append(pos + ln); labels.append(lab)
        pos += ln

    if drop_bout and BOUT_LABEL in labels:
        cut = starts[labels.index(BOUT_LABEL)]
        keep = [i for i, s in enumerate(starts) if s < cut]
        starts, ends, labels = ([starts[i] for i in keep],
                                [min(ends[i], cut) for i in keep],
                                [labels[i] for i in keep])
        dx, dy, active = dx[:cut], dy[:cut], active[:cut]
        size_pos = size_pos[:cut]

    # Cross-check dot activity against the independent `size` encoding (4 active / 0 inactive),
    # rather than trusting the name field alone -- a hand-built file could disagree. Restricted to
    # rows we keep: `bout` is a RESERVED label whose dots ARE drawn (size 4) while the pipeline
    # counts it as non-stimulus, so a disagreement there is by design, not corruption.
    check = ~np.isin(label_per_frame[:len(active)], [BOUT_LABEL])
    if not np.array_equal(active[check], size_pos[check]):
        bad = int((active[check] != size_pos[check]).sum())
        raise ValueError(f"{path}: dot activity disagrees between the name field and the size "
                         f"field on {bad} (frame, dot) cells -- the file is malformed.")
    del size_pos

    n_f = dx.shape[0]
    block_id = np.zeros(n_f, dtype=np.int32)
    for b, (s, e) in enumerate(zip(starts, ends)):
        block_id[s:e] = b

    ctx = [parse_stimulus(l) for l in labels]
    n_dots_of = np.array([np.nan_to_num(c["n_dots_left"]) + np.nan_to_num(c["n_dots_right"])
                          for c in ctx], float)
    moving_of = np.array([bool(c["moving"]) for c in ctx], bool)
    side_of = np.array([c["side"] for c in ctx], object)
    is_stim_of = np.array([isinstance(l, str) and l.startswith("CLsemi") for l in labels], bool)

    return StimulusProgram(
        dx=dx, dy=dy, active=active, block_id=block_id,
        labels=np.array(labels, object),
        starts=np.array(starts, int), ends=np.array(ends, int),
        n_dots_of=n_dots_of, moving_of=moving_of, side_of=side_of, is_stim_of=is_stim_of,
        fps=float(fps), path=path)




