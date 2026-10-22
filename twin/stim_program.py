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

from twin import paths
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


# --------------------------------------------------------------------------- #
# A programme resolved into ONE real fish's dish coordinates
# --------------------------------------------------------------------------- #
@dataclass
class PhysicalProgram:
    """The stimulus programme as **real fish N actually saw it**, in absolute dish pixels.

    A :class:`StimulusProgram` is fish-relative, so it only becomes physical once anchored; with a
    real fish, that anchoring already happened in the rig and is recorded in the transformed CSVs.
    This wraps that resolved track so a rollout can be driven by a FIXED stimulus instead of
    re-anchoring to each simulated fish.

    Why that matters: with the same physical programme, every rollout sees dots in the *same
    absolute places*, so differences between rollouts are behaviour rather than stimulus placement
    -- which is what isolates individuality.

    The trade: the onset-distance control is gone. The real fish was ~70 px from the shoal at each
    block onset by construction; a rollout that has wandered elsewhere is wherever it is. These
    rollouts are therefore NOT comparable to re-anchored ones on any fish-to-shoal distance metric.

    Exposes the same surface ``tool_video`` and ``pick_windows`` use, with ``needs_anchor = False``
    so the video path skips anchoring entirely.
    """
    fish_id: int
    base: "StimulusProgram"
    X: np.ndarray                 # (F, K) absolute px
    Y: np.ndarray
    active: np.ndarray            # (F, K) bool
    cx: np.ndarray                # (F,) active-dot centroid, NaN where nothing is on
    cy: np.ndarray
    start_pose: tuple             # (x, y, ori) of the real fish at its first tracked frame
    dish: tuple                   # (cx, cy, R) physical dish of THIS fish
    needs_anchor: bool = False

    # -- the StimulusProgram surface, delegated / truncated to F ------------ #
    @property
    def n_frames(self):
        return self.X.shape[0]

    @property
    def n_dots(self):
        return self.X.shape[1]

    @property
    def fps(self):
        return self.base.fps

    @property
    def duration_s(self):
        return self.n_frames / self.fps

    @property
    def block_id(self):
        return self.base.block_id[:self.n_frames]

    @property
    def path(self):
        return f"{Path(self.base.path).name} @ fish {self.fish_id}"

    def __getattr__(self, name):
        # starts / ends / labels / is_stim_of / moving_of / n_dots_of come from the base programme
        if name in ("starts", "ends", "labels", "is_stim_of", "moving_of", "side_of", "n_dots_of"):
            return getattr(self.base, name)
        raise AttributeError(name)

    def frame_of(self, time_s):
        return int(round(float(time_s) * self.fps))

    def dots_absolute_all(self, frame, anchor=None):
        """Absolute dot positions. ``anchor`` is accepted and IGNORED -- the placement is already
        physical, which is the whole point of this class."""
        return self.X[frame], self.Y[frame]

    def dots_absolute(self, frame, anchor=None):
        m = self.active[frame]
        if not m.any():
            return np.empty(0), np.empty(0)
        return self.X[frame][m], self.Y[frame][m]

    def centroid(self, frame, anchor=None):
        return float(self.cx[frame]), float(self.cy[frame])

    def block_table(self):
        t = self.base.block_table()
        return t[t["start_frame"] < self.n_frames]

    def summary(self):
        on = np.isfinite(self.cx)
        return (f"physical programme {self.fish_id}: {self.path}\n"
                f"  {self.n_frames} frames ({self.duration_s / 60:.1f} min); "
                f"dots active on {100 * on.mean():.1f}% of frames\n"
                f"  fish start pose ({self.start_pose[0]:.0f}, {self.start_pose[1]:.0f}, "
                f"{self.start_pose[2]:.2f} rad); dish {tuple(round(v, 1) for v in self.dish)}")


def build_physical_program(fish_id, program, transformed_dir=paths.FRAMES, roi_path=paths.ROI_PHYS):
    """Resolve ``program`` through real fish ``fish_id``'s recording -> :class:`PhysicalProgram`.

    Reads the absolute dot columns the transform pipeline already produced, rather than re-deriving
    them: those are the ground truth for what that dish displayed (and this module's ``--verify``
    proves our own anchoring reproduces them to 1e-13 px).
    """
    from twin.segment_bouts import load_physical_dish

    cands = sorted(Path(transformed_dir).glob(f"*_f{int(fish_id)}_transformed_abs.csv"))
    if not cands:
        raise FileNotFoundError(f"no transformed CSV for fish {fish_id} in {transformed_dir} "
                                "(the recorded experiment is not shipped with the repo)")
    K = program.n_dots
    use = ([f"f{fish_id}_x", f"f{fish_id}_y", f"f{fish_id}_ori"]
           + [f"d{k}_{v}" for k in range(K) for v in ("x", "y")]
           + [f"d{k}_stim" for k in range(K)])
    df = pd.read_csv(cands[0], usecols=use)

    X = np.stack([df[f"d{k}_x"].to_numpy(np.float64) for k in range(K)], axis=1)
    Y = np.stack([df[f"d{k}_y"].to_numpy(np.float64) for k in range(K)], axis=1)
    A = np.stack([~df[f"d{k}_stim"].astype(str).isin(list(STIM_NONE)).to_numpy()
                  for k in range(K)], axis=1) & np.isfinite(X)
    F = min(len(df), program.n_frames)
    X, Y, A = X[:F], Y[:F], A[:F]

    # Mean over ACTIVE dots only, done as sum/count rather than nanmean: the all-inactive rows
    # (every pause frame -- 45% of them) would otherwise each raise "Mean of empty slice".
    cnt = A.sum(axis=1)
    ok = cnt > 0
    den = np.maximum(cnt, 1)
    cx = np.where(ok, np.where(A, X, 0.0).sum(axis=1) / den, np.nan)
    cy = np.where(ok, np.where(A, Y, 0.0).sum(axis=1) / den, np.nan)

    fx = df[f"f{fish_id}_x"].to_numpy(float)[:F]
    fy = df[f"f{fish_id}_y"].to_numpy(float)[:F]
    fo = df[f"f{fish_id}_ori"].to_numpy(float)[:F]
    i0 = int(np.flatnonzero(np.isfinite(fx) & np.isfinite(fy) & np.isfinite(fo))[0])

    d = load_physical_dish(roi_path)
    return PhysicalProgram(
        fish_id=int(fish_id), base=program, X=X, Y=Y, active=A, cx=cx, cy=cy,
        start_pose=(float(fx[i0]), float(fy[i0]), float(fo[i0])),
        dish=(float(d.loc[fish_id, "cx"]), float(d.loc[fish_id, "cy"]),
              float(d.loc[fish_id, "radius"])))


# --------------------------------------------------------------------------- #
# Verification against the shipped transformed CSVs
# --------------------------------------------------------------------------- #
def verify_against_transformed(program, fish_id=0,
                               transformed_dir=paths.FRAMES,
                               max_blocks=6, tol=1e-9, verbose=True):
    """Re-derive the REAL absolute dot tracks from this program + the real fish pose at each
    block onset, and compare with the shipped ``*_f{N}_transformed_abs.csv``.

    This is the acceptance test for :func:`anchor_offsets`: the reference side was produced by a
    completely different code path (``src/transformed_data_pipeline`` + ``src/analyse``), so
    agreement to float precision means the collapsed transform is exact, not merely plausible.

    Returns ``(max_abs_err_px, n_compared)``.
    """
    cands = sorted(Path(transformed_dir).glob(f"*_f{int(fish_id)}_transformed_abs.csv"))
    if not cands:
        raise FileNotFoundError(f"no transformed CSV for fish {fish_id} in {transformed_dir}")
    K = program.n_dots
    use = (["frame", f"f{fish_id}_x", f"f{fish_id}_y", f"f{fish_id}_ori"]
           + [f"d{k}_{v}" for k in range(K) for v in ("x", "y")])
    tdf = pd.read_csv(cands[0], usecols=use).sort_values("frame").reset_index(drop=True)
    # The raw trajectory row index equals `frame` in the transformed CSV 1:1 (both start at the
    # same first pause); assert it instead of assuming.
    if int(tdf["frame"].iloc[0]) != 0:
        raise RuntimeError("transformed CSV does not start at frame 0; alignment assumption broken")

    errs, n_cmp, done = [], 0, 0
    for b in range(program.n_blocks):
        if not program.is_stim_of[b] or done >= max_blocks:
            continue
        s, e = int(program.starts[b]), int(program.ends[b])
        if e > len(tdf):
            break
        anchor = (float(tdf[f"f{fish_id}_x"].iloc[s]), float(tdf[f"f{fish_id}_y"].iloc[s]),
                  float(tdf[f"f{fish_id}_ori"].iloc[s]))
        if not np.isfinite(anchor).all():
            continue
        for fr in (s, s + 1, (s + e) // 2, e - 1):
            m = program.active[fr]
            X, Y = program.dots_absolute(fr, anchor)
            gx = tdf.loc[fr, [f"d{k}_x" for k in range(K)]].to_numpy(float)[m]
            gy = tdf.loc[fr, [f"d{k}_y" for k in range(K)]].to_numpy(float)[m]
            ok = np.isfinite(gx) & np.isfinite(gy)
            if ok.any():
                errs.append(np.abs(X[ok] - gx[ok]).max())
                errs.append(np.abs(Y[ok] - gy[ok]).max())
                n_cmp += int(ok.sum())
        done += 1

    if not errs:
        raise RuntimeError("nothing compared -- no finite reference dots found")
    mx = float(max(errs))
    if verbose:
        print(f"anchoring vs df_transformed (fish {fish_id}): "
              f"max abs err {mx:.3e} px over {n_cmp} (frame, dot) pairs "
              f"-> {'PASS' if mx < tol else 'FAIL'}")
    return mx, n_cmp


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="data/data/Original_data/"
                                      "2025.02.20-0957_trajectory_GsizeMotionSingle_180min.csv")
    ap.add_argument("--verify", action="store_true", help="check anchoring vs df_transformed")
    ap.add_argument("--fish", type=int, default=0)
    a = ap.parse_args()

    p = load_program(a.traj)
    print(p.summary())
    print()
    print(p.block_table().head(8).to_string(index=False))
    if a.verify:
        print()
        mx, _ = verify_against_transformed(p, fish_id=a.fish)
        sys.exit(0 if mx < 1e-9 else 1)
