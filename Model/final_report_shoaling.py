"""Build Model/Final_report_shoaling.ipynb (2026-10-21 rewrite): one video per shoaling experiment of the thesis
(Results, "Multiple fish"), seed 0, so that whether the simulated fish shoal can be SEEN rather than read off metrics.
The previous, metric-heavy version (group repulsion design, calibration, sweeps) is kept in the session scratchpad
(Final_report_shoaling.before_videos.ipynb) and its numbers live in the thesis tables.

Reads (nothing is re-run here):
    Model/multi_fish/videos/report/*.mp4, videos_report.json      scripts/shoal_videos.py
The numbers quoted per section are the thesis tables' (tab:shoal-grid, tab:shoal-rep, tab:shoal-dm), seed 0.

    python final_report_shoaling.py
"""
import base64
import json

import final_report_common as C
from final_report_common import md, code

NB = C.HERE / "Final_report_shoaling.ipynb"
VD = C.HERE / "multi_fish" / "videos" / "report"

INTRO = """# Final report shoaling — what the shoals look like

One video per shoaling experiment in the thesis (Results, *Multiple fish*), **seed 0**. Every simulated fish sees
the others as moving dots and runs the unchanged generator; the numbers under each heading are the thesis tables'.

**How the windows are chosen.** Not for looks: a random group and a random 180 s stretch inside the readout period
(fixed seed), the same group index in every panel of a video. Each panel's title gives the median nearest-neighbour
distance *in that window* and *over all groups and the whole readout*, so a typical window can be told from a lucky one.
Played at 4× speed, each fish with a 4 s trail; the training dish is shown whole, the 5× dish zoomed onto the fish
when they stay together (the corner note says so).

**What to look for.** *Shoaling*: the fish stay closer together than the blind fish beside them, for the whole clip.
*Alignment*: the trails run in parallel. *Spacing*: how far apart neighbours stay (real pairs: 5–20 mm).

1. [F: pairs and groups, both dishes](#f)
2. [Adding repulsion](#rep)
3. [Pairs of the screen-trained models](#pairs)
4. [Speed-matched pairs](#matched)"""

REF = """Real references (thesis `tab:shoal-refs`): attraction index of pairs that see each other as dots 0.56 ± 0.08;
nearest distance of real pairs 5–20 mm; real groups of five in a 12.6 cm arena: dispersion D = −0.48, alignment 0.46
(random headings ≈ 0.40). Nothing real exists for groups of 10."""

SECTIONS = [
    ("f", "1. F: pairs and groups, both dishes", REF, [
        ("F_A1_N2", "Pairs, training dish",
         "Attraction 0.35 (real 0.56); nearest distance 14.9 mm (real 5–20, blind 29); alignment 0.65 (blind 0.64)."),
        ("F_A1_N5", "5 fish, training dish",
         "D −0.30 (real −0.48); nearest 8.5 mm (blind 15); alignment 0.42 (blind 0.40; real 0.46 against 0.40)."),
        ("F_A1_N10", "10 fish, training dish",
         "D −0.19; nearest 6.0 mm (blind 10); alignment 0.30 (blind 0.28). No real reference; 9 neighbours is more "
         "than the 8 dots seen in training."),
        ("F_A5_N2", "Pairs, 5× dish (the fish start together)",
         "Attraction 0.35; nearest 24.5 mm (blind 67); alignment 0.65 (blind 0.64)."),
        ("F_A5_N5", "5 fish, 5× dish",
         "D −0.68 (real −0.48 in an arena between our two dishes); nearest 11.1 mm (blind 34); alignment 0.42 (0.40)."),
        ("F_A5_N10", "10 fish, 5× dish",
         "D −0.74; nearest 7.9 mm (blind 23); alignment 0.31 (0.28)."),
    ]),
    ("rep", "2. Adding repulsion", """F with the group repulsion term at 1, 2, 4 and 8 times the calibrated strength,
and at the calibrated strength with the collision rule, training dish (thesis `tab:shoal-rep`). The panels share the
group index, so their fish start alike; the last panel is blind fish.""", [
        ("rep_A1_N2", "Pairs",
         "Attraction 0.35 (F) → 0.33 (×1) → 0.32 (×8); nearest 14.9 → 15.2 → 16.1 mm; real 0.56 and 5–20 mm."),
        ("rep_A1_N5", "5 fish",
         "D −0.30 (F) → −0.29 (×1) → −0.22 (×8); nearest 8.5 → 8.9 → 10.4 mm; alignment 0.42 in every variant "
         "(real 0.46)."),
    ]),
    ("pairs", "3. Pairs of the screen-trained models", """F beside D4 trained on the screen, without and with the
dot's direction of motion as an input (thesis `tab:shoal-dm`). Each fish sees its partner as one moving dot, always in
view; the screen-trained models never ran blind, so there is no blind panel.""", [
        ("pairs_A1", "Training dish",
         "Attraction F 0.35, D4 0.21, + direction 0.26 (real 0.56); distance 14.9, 17.6, 17.0 mm (real 5–20); "
         "alignment above pseudo-pairs +0.016, +0.009, +0.036."),
        ("pairs_A5", "5× dish (the fish start together)",
         "Attraction F 0.35, D4 0.13, + direction 0.18; distance 24.5, 65.0, 61.3 mm; alignment above pseudo-pairs "
         "+0.016, +0.001, +0.027."),
    ]),
    ("matched", "4. Speed-matched pairs", """Every simulated fish has its own latent, so the two fish of a pair often
swim at different tempos: in a typical pair one swims 1.3–1.5 times as far per second as the other (median over the 50
pairs), and in 32–52 % of pairs more than 1.5 times. Here each panel shows instead the pair whose two fish swim most
alike and closest to the condition's median speed, **chosen on how each fish swims, never on how the pair shoals**
(`shoal_videos.matched_pair`); the window is random as before, and the pair number is in each panel's title.

How much this flatters the pairs: over all 50 pairs, the most speed-matched third has an attraction index 0.02–0.05
higher than all pairs in most conditions (F training dish 0.37 against 0.35; up to 0.09 for D4 in the 5× dish), and
the correlation between mismatch and attraction is weak (Spearman −0.03 to −0.28). The thesis tables use all pairs.""", [
        ("pairsm_A1", "Training dish: F, F + repulsion, D4, D4 + direction",
         "Attraction over all pairs F 0.35, + repulsion 0.33, D4 0.21, + direction 0.26."),
        ("pairsm_A5", "5× dish (the fish start together): F, F + repulsion, D4, D4 + direction",
         "Attraction over all pairs F 0.35, + repulsion 0.36, D4 0.13, + direction 0.18."),
    ]),
]


def video_output(mp4):
    b64 = base64.b64encode(mp4.read_bytes()).decode("ascii")
    html = (f'<video controls loop muted width="100%"><source src="data:video/mp4;base64,{b64}" '
            f'type="video/mp4"></video>')
    return dict(output_type="display_data", metadata={},
                data={"text/html": [html], "text/plain": ["<IPython.core.display.Video object>"]})


def stats_output(meta):
    rows = [f"{'panel':28s} {'nearest, this window':>22s} {'nearest, all groups':>21s}"]
    for p in meta["panels"]:
        rows.append(f"{p['label']:28s} {p['nn_window_mm']:20.1f} mm {p['nn_condition_mm']:19.1f} mm")
    head = f"group {meta['group']}, {meta['t0']:.0f}–{meta['t1']:.0f} s of the session\n"
    return dict(output_type="stream", name="stdout", text=(head + "\n".join(rows) + "\n").splitlines(True))


def build():
    index = {m["name"]: m for m in json.loads((VD / "videos_report.json").read_text(encoding="utf-8"))}
    cells = [md(INTRO, "intro")]
    n_vid, missing = 0, []
    for sid, title, text, vids in SECTIONS:
        cells.append(md(f'<a id="{sid}"></a>\n## {title}\n\n{text}', f"{sid}-md"))
        for name, sub, numbers in vids:
            cells.append(md(f"### {sub}\n\n{numbers}", f"{name}-md"))
            mp4 = VD / f"{name}.mp4"
            src = (f'from IPython.display import Video\n'
                   f'Video("multi_fish/videos/report/{name}.mp4", embed=True, html_attributes="controls loop muted")')
            if not mp4.exists():
                cells.append(code(src, f"{name}-v"))
                missing.append(name)
                continue
            cells.append(code(src, f"{name}-v", [video_output(mp4), stats_output(index[name])]))
            n_vid += 1
    C.write_nb(NB, cells, n_vid, missing)


if __name__ == "__main__":
    build()
