"""Bout-model -> mp4 comparison bridge.

Turns any trained ``evaluate.GenerativeModel`` (mix / loop / gru / ...) into the
paired per-frame ``trial_*.csv`` / ``trial_*_real.csv`` files that
``scripts/visualize.py`` renders as a 2xN Real(top)-vs-Synthetic(bottom) video --
so you can *watch* how a bout-level model behaves against the real fish for the
SAME stimulus sequence, and check it stays inside the dish.

It mirrors ``scripts/fit_hmm_stim.py`` (``rollout_segment`` / ``generate_trials`` /
``_stim01_frame``) but is driven by a bout-level ``model.sample(rec, rng)`` instead
of an on-the-fly HMM rollout:

  * The model is sampled ONCE per fish -> a full bout sequence. Closed-loop models
    (``MLPMixLoopDensity``, ``MLPBorderLoopDensity``) return the tracked, dish-contained
    ``x_px, y_px``; other models are reconstructed via ``fit_hmm.reconstruct_xy`` offset
    by the real start pose (NOT contained -- informative for uncontained models).
  * Each bout is placed on the real frame timeline at its ``frame_onset`` (so synthetic
    and real are frame-aligned). Between bouts the fish darts-and-holds (mirrors
    ``rollout_segment``), so a sparse ~1 Hz bout sequence becomes a 30 fps trajectory.
  * The real per-frame trajectory and the actual dot positions come from
    ``data/data/df_transformed/*_f{N}_transformed_abs.csv`` (linked by ``frame_onset``);
    the SAME real dots are copied into both panels so the synthetic fish is shown
    reacting to exactly the stimuli the real one saw.

The model is used IN MEMORY (train + render in the same session -- e.g. a Colab
notebook after ``quick_results``); no checkpoint is needed. Records must be built with
``fit_MLP_border.build_border_records`` (which now attaches per-bout ``frame_onset``,
plus ``x_px/y_px/f_ori_rad`` and the physical dish).

Usage (from a notebook, after fitting ``model`` on ``records``):

    from twin import bout_video as BV
    BV.make_comparison_video(
        model, records,
        transformed_dir="data/data/df_transformed",
        out_dir="runs/mix_loop/samples",
        out_mp4="runs/mix_loop/comparison.mp4",
        fish_ids=[0, 3, 6, 9], n=4)
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from twin import paths
from twin.fit_hmm_stim import _stim01_frame                 # verified real-dots -> 0/1 converter
from twin.segment_bouts import parse_stimulus, fish_prefix  # exact HMM stimulus-block parsing

_DEFAULT_TRANSFORMED = paths.FRAMES
FPS = 30
N_STIMULI = 8


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_transformed(fid, transformed_dir):
    """(per-frame real df, fish column prefix ``f{fid}``) for one fish. The ``_f{fid}_``
    token (with both underscores) prevents f1 from matching f11 etc."""
    cands = sorted(Path(transformed_dir).glob(f"*_f{fid}_transformed_abs.csv"))
    if not cands:
        raise FileNotFoundError(f"no transformed CSV for fish {fid} in {transformed_dir}")
    return pd.read_csv(cands[0]), f"f{fid}"


def _bout_positions(model, rec, rng):
    """Sample the model once; return per-bout absolute (bx, by). Loop models supply
    the tracked, contained ``x_px/y_px``; otherwise reconstruct from turns+steps offset
    by the real start pose (uncontained)."""
    g = model.sample(rec, rng)
    if "x_px" in g and "y_px" in g:
        return np.asarray(g["x_px"], float), np.asarray(g["y_px"], float)
    from twin import fit_hmm
    rx, ry, _ = fit_hmm.reconstruct_xy(np.asarray(g["dtheta"], float),
                                       np.asarray(g["distance_px"], float),
                                       theta0=float(rec.f_ori_rad[0]))
    return float(rec.x_px[0]) + rx, float(rec.y_px[0]) + ry


def _synthetic_path(frames, onset, bx, by, bout_frames, h0):
    """Per-frame synthetic pose (sx, sy, so) by dart-and-hold between consecutive bout
    positions (mirrors ``fit_hmm_stim.rollout_segment``): within each inter-bout interval
    the fish ramps from bout t's position to bout t+1's over ``bout_frames`` frames, then
    holds. Heading = the bout's movement direction (carried forward when the step is ~0)."""
    F = len(frames); B = len(onset)
    sx = np.empty(F); sy = np.empty(F); so = np.empty(F)
    prev_h = float(h0)
    for i in range(F):
        fr = int(frames[i])
        t = int(np.searchsorted(onset, fr, side="right")) - 1
        t = min(max(t, 0), B - 1)
        if t >= B - 1:                                   # past the last bout onset -> hold
            sx[i], sy[i], h = bx[-1], by[-1], prev_h
        else:
            f_a, f_b = int(onset[t]), int(onset[t + 1])
            move = max(1, min(bout_frames, f_b - f_a))
            frac = min(1.0, max(0.0, (fr - f_a + 1) / move))   # clamp [0,1]: no extrapolation before onset[0]
            sx[i] = bx[t] + (bx[t + 1] - bx[t]) * frac
            sy[i] = by[t] + (by[t + 1] - by[t]) * frac
            dx, dy = bx[t + 1] - bx[t], by[t + 1] - by[t]
            h = float(np.arctan2(dy, dx)) if (dx * dx + dy * dy) > 1e-9 else prev_h
        so[i] = h; prev_h = h
    return sx, sy, so


# --------------------------------------------------------------------------- #
# Trial CSVs
# --------------------------------------------------------------------------- #
def make_trials(model, records, out_dir, *, transformed_dir=_DEFAULT_TRANSFORMED,
                fish_ids=None, rng=None, trial_length_s=60.0, bout_frames=3,
                fps=FPS, n_stimuli=N_STIMULI, verbose=True):
    """Write paired ``trial_{i}.csv`` / ``trial_{i}_real.csv`` (df_base schema:
    ``f_x,f_y,f_ori, d{0..7}_x/_y/_stim``, per frame) for each fish in ``fish_ids``
    (default: all, sorted). Returns the number of trials written.

    ``records`` is the dict from ``fit_MLP_border.build_border_records`` (keyed by
    fish_id; carries per-bout ``frame_onset``, ``x_px/y_px/f_ori_rad`` and the dish)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = rng if rng is not None else np.random.default_rng(0)
    ids = list(fish_ids) if fish_ids is not None else sorted(records)
    trial_frames = int(trial_length_s * fps)

    made = 0
    for fid in ids:
        rec = records.get(fid)
        if rec is None or not hasattr(rec, "frame_onset"):
            if verbose:
                print(f"  fish {fid}: no record / no frame_onset -> skip")
            continue
        onset = np.asarray(rec.frame_onset, int)
        bx, by = _bout_positions(model, rec, rng)

        tdf, prefix = _load_transformed(fid, transformed_dir)
        tdf = tdf.sort_values("frame").reset_index(drop=True)
        frame_col = tdf["frame"].to_numpy(int)
        f0 = int(onset[0])
        f_end = min(f0 + trial_frames, int(frame_col.max()) + 1)
        win = tdf[(frame_col >= f0) & (frame_col < f_end)].reset_index(drop=True)
        frames = win["frame"].to_numpy(int)
        if len(frames) < 2:
            if verbose:
                print(f"  fish {fid}: <2 frames in window -> skip")
            continue

        sx, sy, so = _synthetic_path(frames, onset, bx, by, bout_frames,
                                     h0=float(rec.f_ori_rad[0]))
        rx = win[f"{prefix}_x"].to_numpy(float)
        ry = win[f"{prefix}_y"].to_numpy(float)
        ro = win[f"{prefix}_ori"].to_numpy(float)

        syn_rows, real_rows = [], []
        for i in range(len(frames)):
            dots = _stim01_frame(win.iloc[i], n_stimuli)     # real dots, same for both panels
            syn = {"f_x": float(sx[i]), "f_y": float(sy[i]), "f_ori": float(so[i])}
            rea = {"f_x": float(rx[i]), "f_y": float(ry[i]), "f_ori": float(ro[i])}
            syn.update(dots); rea.update(dots)
            syn_rows.append(syn); real_rows.append(rea)
        pd.DataFrame(syn_rows).to_csv(out_dir / f"trial_{made:03d}.csv", index=False)
        pd.DataFrame(real_rows).to_csv(out_dir / f"trial_{made:03d}_real.csv", index=False)
        if verbose:
            print(f"  fish {fid}: frames {f0}:{f_end} ({len(frames)}) -> trial_{made:03d}")
        made += 1
    return made


# --------------------------------------------------------------------------- #
# Render (invokes scripts/visualize.py unchanged)
# --------------------------------------------------------------------------- #
def _dish_for(records, ids, made):
    """Common dish circle for the video = median physical (cx, cy, R) over the rendered
    fish (the per-fish physical dishes are ~identical). Falls back to visualize defaults."""
    dd = [(records[f].dish_cx, records[f].dish_cy, records[f].dish_R)
          for f in ids[:made] if f in records and hasattr(records[f], "dish_cx")]
    if not dd:
        return (180.0, 174.0, 183.5)
    return tuple(float(v) for v in np.median(np.asarray(dd, float), axis=0))


def _render_env(out_path):
    """Env for the visualize subprocess (+ a temp dir to clean up, or None).

    Always forces UTF-8 (so visualize.py's unicode prints don't crash a cp1252 console).
    For ``.mp4`` output with no system ffmpeg on PATH, transparently points matplotlib at the
    ffmpeg binary bundled by ``imageio-ffmpeg`` (``pip install imageio-ffmpeg``) via a throwaway
    matplotlibrc -- so .mp4 just works without a system ffmpeg install. Raises a clear,
    actionable error if .mp4 is requested and no ffmpeg is available anywhere."""
    import shutil
    import tempfile
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if Path(out_path).suffix.lower() != ".mp4" or shutil.which("ffmpeg"):
        return env, None
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = None
    if not exe or not os.path.exists(exe):
        raise RuntimeError(
            f"Cannot write {Path(out_path).name}: no ffmpeg found for .mp4 output.\n"
            "  Fix (pick one):\n"
            "    * pip install imageio-ffmpeg      (bundles ffmpeg, no admin) then re-run, or\n"
            "    * install ffmpeg on your PATH, or\n"
            "    * render a .gif instead (just change the output extension to .gif).")
    rc_dir = Path(tempfile.mkdtemp(prefix="mplffmpeg_"))
    (rc_dir / "matplotlibrc").write_text(
        f"animation.ffmpeg_path: {exe.replace(os.sep, '/')}\n", encoding="utf-8")
    env["MATPLOTLIBRC"] = str(rc_dir)                    # matplotlib in the child reads this
    return env, rc_dir


def render(samples_dir, out_path, n, *, dish=None, fps=FPS, speed=1.0, verbose=True):
    """Invoke ``scripts/visualize.py`` on already-written trial CSVs. ``out_path`` extension
    picks the writer (.mp4 -> ffmpeg, .gif -> pillow). .mp4 uses a bundled ffmpeg
    (imageio-ffmpeg) when no system ffmpeg is on PATH."""
    import shutil
    dx, dy, dr = dish if dish is not None else (180.0, 174.0, 183.5)
    script = str(Path(__file__).resolve().parent / "visualize.py")
    cmd = [sys.executable, script, "--samples", str(samples_dir), "--output", str(out_path),
           "--n", str(n), "--fps", str(fps), "--speed", str(speed),
           "--dish_x", f"{dx:.3f}", "--dish_y", f"{dy:.3f}", "--dish_radius", f"{dr:.3f}"]
    env, rc_dir = _render_env(out_path)
    if verbose:
        print("  render:", " ".join(cmd))
        if Path(out_path).suffix.lower() == ".mp4":
            print("  ffmpeg:", shutil.which("ffmpeg") or (env.get("MATPLOTLIBRC") and "imageio-ffmpeg (bundled)"))
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        if rc_dir is not None:
            import shutil as _sh
            _sh.rmtree(rc_dir, ignore_errors=True)
    return out_path


def make_comparison_video(model, records, out_dir, out_mp4, *,
                          transformed_dir=_DEFAULT_TRANSFORMED, fish_ids=None, n=None,
                          rng=None, trial_length_s=60.0, bout_frames=3, fps=FPS,
                          speed=1.0, dish=None, verbose=True):
    """End-to-end: sample ``model`` per fish -> trial CSVs -> ``visualize.py`` video.
    ``n`` caps the number of columns (default: all requested fish). Returns
    ``(out_mp4, n_trials)``."""
    ids = list(fish_ids) if fish_ids is not None else sorted(records)
    if n is not None:
        ids = ids[:n]
    made = make_trials(model, records, out_dir, transformed_dir=transformed_dir,
                       fish_ids=ids, rng=rng, trial_length_s=trial_length_s,
                       bout_frames=bout_frames, fps=fps, verbose=verbose)
    if made == 0:
        raise RuntimeError("no trials written (check fish_ids / records / transformed_dir)")
    if dish is None:
        dish = _dish_for(records, ids, made)
    render(out_dir, out_mp4, made, dish=dish, fps=fps, speed=speed, verbose=verbose)
    if verbose:
        print(f"wrote {made} trial(s) + {out_mp4}  (dish={tuple(round(v,1) for v in dish)})")
    return out_mp4, made


# --------------------------------------------------------------------------- #
# Match the HMM comparison.mp4 sample selection (same fish + same stimulus windows)
# --------------------------------------------------------------------------- #
def _slice_record(rec, lo, hi):
    """Shallow-copy ``rec`` and slice every per-bout array (``arr.shape[0] == rec.n``, incl. the
    2-D ``stim_csv``) to the bouts with ``frame_onset in [lo, hi)``; reset ``rec.n``. Dish scalars
    are kept. The sliced record's ``x_px[0]/y_px[0]/f_ori_rad[0]`` are the REAL pose at the first
    in-window bout, so a loop model seeded from them starts co-located with the real fish there
    (mirroring the HMM's 3-s real-seed)."""
    onset = np.asarray(rec.frame_onset, int)
    mask = (onset >= lo) & (onset < hi)
    n0 = int(rec.n)
    r = copy.copy(rec)
    for attr, val in list(vars(rec).items()):
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n0:
            setattr(r, attr, val[mask])
    return r                                              # rec.n is a property = len(dtheta) (auto-updates)


def _select_like_hmm(records, transformed_dir, n=4, trial_length_s=60.0, seed_seconds=3.0):
    """Replay ``fit_hmm_stim.generate_trials``'s trial selection on the transformed CSVs so the
    video uses the SAME fish + windows as ``runs/hmm_stim/comparison.mp4``. Per fish (sorted id):
    first lateralized moving-dot block (``parse_stimulus`` side in {left,right} AND moving), window
    ``[max(0, s0-2*FPS), +trial_length_s*FPS)``, skip overrun / non-finite; one per fish, up to n.
    Returns ``(windows, seed)`` where ``windows = [(fid, beg, end)]`` and ``seed = int(seed_seconds*FPS)+1``
    (frames dropped before display, exactly like the HMM)."""
    trial_frames = int(trial_length_s * FPS)
    seed = int(seed_seconds * FPS) + 1
    windows = []
    for fid in sorted(records):
        if len(windows) >= n:
            break
        try:
            tdf, prefix = _load_transformed(fid, transformed_dir)
        except FileNotFoundError:
            continue
        tdf = tdf.sort_values("frame").reset_index(drop=True)      # row == frame (contiguous)
        lab = tdf["stimulus"].astype(str)
        sides = lab.map(lambda s: parse_stimulus(s).get("side"))
        mov = lab.map(lambda s: parse_stimulus(s).get("moving"))
        is_target = sides.isin(["left", "right"]) & mov.astype(bool)
        starts = np.where(is_target.to_numpy() & ~is_target.shift(fill_value=False).to_numpy())[0]
        xcol = tdf[f"{prefix}_x"].to_numpy(float)
        for s0 in starts:
            beg = max(0, int(s0) - int(2 * FPS))
            end = beg + trial_frames
            if end > len(tdf):
                continue
            if not np.isfinite(xcol[beg:end]).all():
                continue
            windows.append((fid, beg, end))
            break
    return windows, seed


def make_trials_like_hmm(model, records, out_dir, *, transformed_dir=_DEFAULT_TRANSFORMED, n=4,
                         trial_length_s=60.0, seed_seconds=3.0, bout_frames=3,
                         dish=(180.0, 174.0, 183.5), rng=None, verbose=True):
    """Write paired trial CSVs using the SAME fish + stimulus windows as the HMM
    ``runs/hmm_stim/comparison.mp4``. Selection replays ``fit_hmm_stim.generate_trials``; the
    synthetic rollout is re-seeded at the real pose at each window start via ``_slice_record``; the
    containment dish is overridden to ``dish`` so the fish stays inside the same drawn circle the HMM
    used. Both panels are raw 30 fps per-frame over the same displayed frames ``[beg+seed, end)`` as
    the HMM. Returns the number of trials written."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = rng if rng is not None else np.random.default_rng(0)
    windows, seed = _select_like_hmm(records, transformed_dir, n, trial_length_s, seed_seconds)
    if not windows:
        raise RuntimeError("no HMM-style windows found (check transformed_dir / stimulus labels)")
    two_pi = 2.0 * np.pi
    made = 0
    for fid, beg, end in windows:
        emit_start = beg + seed
        rec2 = _slice_record(records[fid], emit_start, end)
        if rec2.n < 2:
            if verbose:
                print(f"  fish {fid}: <2 bouts in window -> skip")
            continue
        rec2.dish_cx, rec2.dish_cy, rec2.dish_R = float(dish[0]), float(dish[1]), float(dish[2])
        bx, by = _bout_positions(model, rec2, rng)

        tdf, prefix = _load_transformed(fid, transformed_dir)
        tdf = tdf.sort_values("frame").reset_index(drop=True)
        frame_col = tdf["frame"].to_numpy(int)
        win = tdf[(frame_col >= emit_start) & (frame_col < end)].reset_index(drop=True)
        frames = win["frame"].to_numpy(int)
        sx, sy, so = _synthetic_path(frames, np.asarray(rec2.frame_onset, int), bx, by, bout_frames,
                                     h0=float(rec2.f_ori_rad[0]))
        rx = win[f"{prefix}_x"].to_numpy(float)
        ry = win[f"{prefix}_y"].to_numpy(float)
        ro = win[f"{prefix}_ori"].to_numpy(float)

        syn_rows, real_rows = [], []
        for i in range(len(frames)):
            dots = _stim01_frame(win.iloc[i], N_STIMULI)             # real dots (same for both panels)
            syn = {"f_x": float(sx[i]), "f_y": float(sy[i]), "f_ori": float(so[i] % two_pi)}
            rea = {"f_x": float(rx[i]), "f_y": float(ry[i]), "f_ori": float(ro[i] % two_pi)}
            syn.update(dots); rea.update(dots)
            syn_rows.append(syn); real_rows.append(rea)
        pd.DataFrame(syn_rows).to_csv(out_dir / f"trial_{made:03d}.csv", index=False)
        pd.DataFrame(real_rows).to_csv(out_dir / f"trial_{made:03d}_real.csv", index=False)
        if verbose:
            print(f"  fish {fid}: window {beg}:{end} -> display {emit_start}:{end} "
                  f"({len(frames)} frames) -> trial_{made:03d}")
        made += 1
    if made == 0:
        raise RuntimeError("no trials written (windows had <2 bouts)")
    return made


def make_comparison_video_like_hmm(model, records, out_dir, out_mp4, *,
                                   transformed_dir=_DEFAULT_TRANSFORMED, n=4,
                                   trial_length_s=60.0, seed_seconds=3.0, bout_frames=3,
                                   speed=3.0, dish=(180.0, 174.0, 183.5), rng=None, verbose=True):
    """End-to-end HMM-matched comparison video: same fish + stimulus windows as
    ``runs/hmm_stim/comparison.mp4`` (see ``make_trials_like_hmm``) → ``visualize.py`` at 30 fps,
    speed 3, dish ``(180,174,183.5)`` — directly comparable to the HMM video."""
    made = make_trials_like_hmm(model, records, out_dir, transformed_dir=transformed_dir, n=n,
                                trial_length_s=trial_length_s, seed_seconds=seed_seconds,
                                bout_frames=bout_frames, dish=dish, rng=rng, verbose=verbose)
    render(out_dir, out_mp4, made, dish=dish, fps=FPS, speed=speed, verbose=verbose)
    if verbose:
        print(f"wrote {made} HMM-matched trial(s) + {out_mp4}  (dish={tuple(round(v,1) for v in dish)})")
    return out_mp4, made


# --------------------------------------------------------------------------- #
# Smoke test:  python scripts/bout_video.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from twin import evaluate as E
    from twin import fit_MLP_border as MB
    from twin import fit_MLP_mix as MX

    df = E.load_bouts(str(_ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    records = MB.build_border_records(df)
    ctx = E.make_context(list(records.values()), cfg={})
    model = MX.MLPMixLoopDensity(K=8, hidden=32, use_border=True, epochs=6, closed_loop=True)
    model.fit(list(records.values()), ctx)

    out_dir = _ROOT / "runs" / "bout_video_smoke" / "samples"
    out_gif = _ROOT / "runs" / "bout_video_smoke" / "comparison.gif"
    mp4, made = make_comparison_video(
        model, records, out_dir, out_gif, fish_ids=[0, 3], n=2,
        trial_length_s=20.0, rng=np.random.default_rng(0))
    print(f"OK: {made} trials, video -> {mp4}")
