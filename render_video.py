"""Render the real-vs-synthetic comparison video LOCALLY from a saved model.

Run from the repo root (c:\\Users\\arthu\\mp_semiCL_behavior) AFTER downloading the
.pt file you made in Colab with ``MX.save_mix(...)`` into this folder:

    python render_video.py
    python render_video.py --model mix_loop_h128.pt --out runs/mix_loop/comparison.gif --fish 0 3 6 9
    python render_video.py --seconds 90 --speed 2 --fish 0 5 10

Default output is .mp4. If you have no system ffmpeg, run once:  pip install imageio-ffmpeg
(a pip-only package that bundles ffmpeg, no admin) -- the renderer picks it up automatically.
Prefer a .gif with no install at all? pass  --out runs/mix_loop/comparison.gif .

It handles the import path for you, so you do NOT need to touch sys.path or run it from
inside notebooks/ -- just run it from the repo root where data/ and scripts/ live.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))          # make evaluate / fit_* / bout_video importable

import numpy as np                                 # noqa: E402
import evaluate as E                               # noqa: E402
import fit_MLP_border as MB                         # noqa: E402
import fit_MLP_mix as MX                            # noqa: E402
import bout_video as BV                             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mix_loop_h128.pt",
                    help="saved model file from MX.save_mix (default: mix_loop_h128.pt)")
    ap.add_argument("--bouts", default="data/bouts_GsizeMotionSingle.csv")
    ap.add_argument("--out", default="runs/mix_loop/comparison.mp4",
                    help=".mp4 (needs ffmpeg or `pip install imageio-ffmpeg`) or .gif (no ffmpeg). Default: .mp4")
    ap.add_argument("--match-hmm", action="store_true",
                    help="use the SAME fish + stimulus windows as runs/hmm_stim/comparison.mp4 "
                         "(so mix-MLP vs HMM is directly comparable). Ignores --fish/--seconds.")
    ap.add_argument("--fish", type=int, nargs="+", default=[0, 3, 6, 9],
                    help="fish ids -> one column each (default: 0 3 6 9). Ignored with --match-hmm.")
    ap.add_argument("--n", type=int, default=4, help="--match-hmm: number of columns (default 4 = fish 0-3)")
    ap.add_argument("--seconds", type=float, default=60.0, help="clip length per fish")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"model file not found: {args.model}\n"
                 f"  -> download the .pt you saved in Colab and put it in {ROOT}")

    print(f"loading bouts + building records ...")
    df = E.load_bouts(args.bouts)
    records = MB.build_border_records(df)
    print(f"loading model {args.model} ...")
    model = MX.load_mix(args.model)
    out_dir = Path(args.out).parent / "samples"
    if args.match_hmm:
        BV.make_comparison_video_like_hmm(
            model, records, out_dir, args.out, n=args.n,
            speed=(3.0 if args.speed == 1.0 else args.speed),   # HMM used --speed 3
            rng=np.random.default_rng(args.seed))
    else:
        BV.make_comparison_video(
            model, records, out_dir, args.out,
            fish_ids=args.fish, n=len(args.fish),
            rng=np.random.default_rng(args.seed),
            trial_length_s=args.seconds, speed=args.speed)
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
