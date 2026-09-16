"""Build model_lineage.pptx -- the slide deck of the behaviour-model lineage.

    python slides/make_deck.py

Reads the figures written by slides/make_figures.py and rebuilds every number in the
master table directly from the ladder archives, so the deck cannot drift from the data.

SOURCING RULES (violating these has cost real time before):
  * NEVER read the repo-root results.csv / panel.csv / artifacts.pkl -- that is a smoke
    run (2 folds, 3 epochs) whose rung names match the first ladder but whose AUCs do not.
  * Absolute AUCs are NOT comparable across ladders (each refits its own control); only
    the within-ladder paired delta is a measurement.
  * The deck stops at the `conv` ladder. No mem/indiv result may appear in any table.
"""
import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

if not hasattr(np, "_core"):                      # Colab numpy2 -> local numpy1
    import numpy.core as _nc
    sys.modules["numpy._core"] = _nc
    for _s in ("multiarray", "umath", "numeric", "_multiarray_umath", "numerictypes"):
        try:
            sys.modules[f"numpy._core.{_s}"] = __import__(f"numpy.core.{_s}", fromlist=[_s])
        except Exception:
            pass

import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402
from pptx import Presentation                                    # noqa: E402
from pptx.dml.color import RGBColor                              # noqa: E402
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR                  # noqa: E402
from pptx.util import Inches, Pt, Emu                            # noqa: E402

FIGS = Path(__file__).resolve().parent / "figs"
OUT = ROOT / "model_lineage.pptx"
VIDEO = ROOT / "runs" / "mix_loop" / "ladder_e250.mp4"

# ---- palette -------------------------------------------------------------- #
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
FAINT = RGBColor(0x8C, 0x8C, 0x8C)
ACCENT = RGBColor(0x1F, 0x3B, 0x8C)
WIN = RGBColor(0x00, 0x6E, 0x3C)
LOSE = RGBColor(0xAA, 0x1E, 0x1E)
RULE = RGBColor(0xD8, 0xD8, 0xD8)
BAND = RGBColor(0xF2, 0xF4, 0xF8)

SW, SH = Inches(13.333), Inches(7.5)
M = Inches(0.62)
CONTENT_W = SW - 2 * M
TITLE_Y, TITLE_H = Inches(0.34), Inches(0.72)
BODY_Y = Inches(1.30)
BODY_H = Inches(5.45)
FOOT_Y = Inches(6.95)

FLOOR = 0.47610883928568803
CRIT = float(stats.t.ppf(0.975, 7))

LADDER_SRC = {
    "stim": ROOT / "ladder_out", "wall": ROOT / "wall_out", "kin": ROOT / "kin_out",
    "disp": ROOT / "disp_out", "contain": ROOT / "contain_out",
    "tail": ROOT / "ladder_tail_results.zip", "far": ROOT / "far_out",
    "cal": ROOT / "ladder_cal_results.zip", "conv": ROOT / "conv_out",
}
ORDER = ["stim", "wall", "kin", "disp", "contain", "tail", "far", "cal", "conv"]

_SHORT = {
    "mlp_border_full_h128_mix_": "", "_hloop_kin_vigor": "", "shoal_": "", "ladder_": "",
}


def short(name):
    s = name.replace("mlp_border_full_h128_mix_", "")
    s = s.replace("_hloop_kin_vigor", "")
    return s or "base"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_ladder(src):
    if src.suffix == ".zip":
        z = zipfile.ZipFile(src)
        r = [n for n in z.namelist() if n.endswith("results.csv")][0]
        a = [n for n in z.namelist() if n.endswith("artifacts.pkl")][0]
        return pd.read_csv(io.BytesIO(z.read(r)), index_col=0), pickle.loads(z.read(a))
    with open(src / "artifacts.pkl", "rb") as fh:
        art = pickle.load(fh)
    return pd.read_csv(src / "results.csv", index_col=0), art


def master_rows():
    """[(ladder, rung, auc, delta|None, t|None, verdict)] rebuilt from the archives."""
    rows, floors, bits = [], set(), []
    for lad in ORDER:
        res, art = load_ladder(LADDER_SRC[lad])
        reps = art["auc_traj_reps"]
        floors.add(round(float(art["neg_control_traj"]), 10))
        models = list(res.index)
        ctrl = models[0]
        base = np.asarray(reps[ctrl], float)
        for m in models:
            auc = float(res.loc[m, "c2st_traj_auc"])
            bits.append(float(res.loc[m, "bits_per_bout"]))
            if m == ctrl:
                rows.append((lad, short(m), auc, None, None, "control"))
                continue
            d = np.asarray(reps[m], float) - base
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            v = ("worse" if t > 0 else "resolved") if abs(t) > CRIT else "noise"
            rows.append((lad, short(m), auc, float(d.mean()), float(t), v))
    assert len(floors) == 1 and abs(list(floors)[0] - FLOOR) < 1e-9, floors
    return rows, (min(bits), max(bits))


# --------------------------------------------------------------------------- #
# slide primitives
# --------------------------------------------------------------------------- #
def _tf(shape, size, color=INK, bold=False, align=PP_ALIGN.LEFT, space=6):
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(space)
    r = p.add_run()
    r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color
    r.font.name = "Calibri"
    return tf, r


_re = __import__("re")
_TOK = _re.compile(r"(\*\*.+?\*\*|\*[^*\n]+?\*|`[^`\n]+?`)")
_CODE = _re.compile(r"(`[^`\n]+?`)")


def _emit(par, text, size, color, bold, ital):
    """Emit runs for one span, still honouring `code` NESTED inside it."""
    for part in _CODE.split(text):
        if not part:
            continue
        mono = part.startswith("`") and part.endswith("`") and len(part) > 2
        if mono:
            part = part[1:-1]
        r = par.add_run()
        r.text = part
        r.font.size = Pt(size * 0.93 if mono else size)
        r.font.name = "Consolas" if mono else "Calibri"
        r.font.bold = bold
        r.font.italic = ital
        r.font.color.rgb = color


def _rich(par, text, size, color=INK, bold=False):
    """Render **bold**, *italic* and `code` spans as separate runs.

    Without this the markers leak onto the slide as literal characters, which is
    exactly what they did the first time this deck was rendered.
    """
    for chunk in _TOK.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            _emit(par, chunk[2:-2], size, color, True, False)
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            _emit(par, chunk[1:-1], size, color, bold, True)
        else:
            _emit(par, chunk, size, color, bold, False)


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = SW, SH
        self.n = 0

    # -- chrome ------------------------------------------------------------- #
    def _blank(self):
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def _chrome(self, s, title, kicker=None):
        self.n += 1
        if kicker:
            k = s.shapes.add_textbox(M, Inches(0.20), CONTENT_W, Inches(0.26))
            _tf(k, 11, ACCENT, True)[1].text = kicker.upper()
        tb = s.shapes.add_textbox(M, TITLE_Y if not kicker else Inches(0.46),
                                  CONTENT_W, TITLE_H)
        tb.text_frame.word_wrap = True
        # via _rich, so `code` in a title renders as monospace rather than backticks
        _rich(tb.text_frame.paragraphs[0], title, 26, INK, bold=True)
        ln = s.shapes.add_shape(1, M, Inches(1.16), CONTENT_W, Emu(9525))
        ln.fill.solid(); ln.fill.fore_color.rgb = RULE
        ln.line.fill.background(); ln.shadow.inherit = False
        ft = s.shapes.add_textbox(M, FOOT_Y, CONTENT_W, Inches(0.28))
        tf, r = _tf(ft, 9, FAINT)
        r.text = f"Model lineage · zebrafish virtual-shoal assay          {self.n}"
        tf.paragraphs[0].alignment = PP_ALIGN.RIGHT

    def notes(self, s, text):
        s.notes_slide.notes_text_frame.text = text.strip()

    # -- slide kinds --------------------------------------------------------- #
    def title_slide(self, title, subtitle, meta, notes):
        s = self._blank()
        self.n += 1
        band = s.shapes.add_shape(1, Emu(0), Emu(0), SW, Inches(2.92))
        band.fill.solid(); band.fill.fore_color.rgb = BAND
        band.line.fill.background(); band.shadow.inherit = False
        # title is two lines at this width -- give it the room, then place the
        # subtitle BELOW it rather than at a guessed offset.
        tb = s.shapes.add_textbox(M, Inches(0.62), CONTENT_W, Inches(1.55))
        _tf(tb, 34, INK, True)[1].text = title
        sb = s.shapes.add_textbox(M, Inches(2.18), CONTENT_W, Inches(0.55))
        _tf(sb, 17, ACCENT)[1].text = subtitle
        mb = s.shapes.add_textbox(M, Inches(3.35), CONTENT_W, Inches(2.4))
        tf = mb.text_frame; tf.word_wrap = True
        for i, line in enumerate(meta):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(9)
            _rich(p, line, 14, MUTED)
        self.notes(s, notes)
        return s

    def bullets(self, title, items, notes, kicker=None, width=None, top=None, size=None):
        s = self._blank(); self._chrome(s, title, kicker)
        norm = []
        for it in items:
            if isinstance(it, tuple):
                norm.append(it if len(it) == 3 else (it[0], it[1], INK))
            else:
                norm.append((0, it, INK))
        # auto-fit: PowerPoint will not shrink text for us, so pick a size that fits.
        chars = sum(len(t) for _, t, _ in norm) + 26 * sum(1 for _, t, _ in norm if t == "")
        if size is None:
            size = 18 if chars < 620 else 17 if chars < 800 else 16 if chars < 980 else 15
        gap = 11 if size >= 17 else 8
        tb = s.shapes.add_textbox(M, top or BODY_Y, width or CONTENT_W, BODY_H)
        tf = tb.text_frame; tf.word_wrap = True
        for i, (lvl, txt, col) in enumerate(norm):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.level = lvl
            p.space_after = Pt(gap if lvl == 0 else gap - 4)
            if txt == "":
                continue
            _rich(p, ("" if lvl == 0 else "– ") + txt, size - 2.5 * lvl, col)
        self.notes(s, notes)
        return s

    def figure(self, title, png, caption, notes, kicker=None, max_h=5.05):
        s = self._blank(); self._chrome(s, title, kicker)
        from PIL import Image
        w, h = Image.open(png).size
        ar = w / h
        ph = min(max_h, 9.6 / ar)
        pw = ph * ar
        if pw > 11.9:
            pw, ph = 11.9, 11.9 / ar
        s.shapes.add_picture(str(png), Inches((13.333 - pw) / 2), BODY_Y,
                             Inches(pw), Inches(ph))
        cb = s.shapes.add_textbox(M, BODY_Y + Inches(ph) + Inches(0.10),
                                  CONTENT_W, Inches(0.75))
        tf = cb.text_frame; tf.word_wrap = True
        _rich(tf.paragraphs[0], caption, 13, MUTED)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        self.notes(s, notes)
        return s

    def _grid(self, s, headers, rows, left, top, width, colw, fsize, rowh):
        """Draw one table; return its bottom edge. headers=None -> no header row."""
        hdr = 1 if headers else 0
        nc = len(headers) if headers else len(rows[0])
        gt = s.shapes.add_table(len(rows) + hdr, nc, left, top, width,
                                Inches(rowh) * (len(rows) + hdr)).table
        gt.first_row = bool(headers)
        if colw:
            tot = sum(colw)
            for i, cw in enumerate(colw):
                gt.columns[i].width = Emu(int(width * cw / tot))
        for r in gt.rows:
            r.height = Inches(rowh)
        if headers:
            for j, htxt in enumerate(headers):
                c = gt.cell(0, j)
                c.text = ""
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                c.fill.solid(); c.fill.fore_color.rgb = BAND
                _rich(c.text_frame.paragraphs[0], htxt, fsize, INK, bold=True)
        for i, row in enumerate(rows, start=hdr):
            for j, cell in enumerate(row):
                txt, col, bold = cell, INK, False
                if isinstance(cell, tuple):
                    txt, col = cell[0], cell[1]
                    bold = cell[2] if len(cell) > 2 else False
                c = gt.cell(i, j)
                c.text = ""
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                c.fill.solid()
                c.fill.fore_color.rgb = (RGBColor(0xFF, 0xFF, 0xFF) if (i - hdr) % 2 == 0
                                         else RGBColor(0xFA, 0xFB, 0xFD))
                _rich(c.text_frame.paragraphs[0], str(txt), fsize, col, bold)
        return top + Inches(rowh) * (len(rows) + hdr)

    def table(self, title, headers, rows, notes, kicker=None, colw=None,
              fsize=11.5, note=None, top=None, rowh=0.30):
        s = self._blank(); self._chrome(s, title, kicker)
        top = top or BODY_Y
        bot = self._grid(s, headers, rows, M, top, CONTENT_W, colw, fsize, rowh)
        if note:
            nb = s.shapes.add_textbox(M, bot + Inches(0.14), CONTENT_W, Inches(0.8))
            tf = nb.text_frame; tf.word_wrap = True
            _rich(tf.paragraphs[0], note, 12, MUTED)
        self.notes(s, notes)
        return s

    def split_table(self, title, headers, rows, notes, kicker=None, colw=None,
                    fsize=10, note=None, top=None, rowh=0.245, gap=0.30):
        """One long table as two side-by-side halves, so it fits on one slide."""
        s = self._blank(); self._chrome(s, title, kicker)
        top = top or BODY_Y
        half = (len(rows) + 1) // 2
        w = Emu(int((CONTENT_W - Inches(gap)) / 2))
        b1 = self._grid(s, headers, rows[:half], M, top, w, colw, fsize, rowh)
        b2 = self._grid(s, headers, rows[half:], M + w + Inches(gap), top, w,
                        colw, fsize, rowh)
        if note:
            nb = s.shapes.add_textbox(M, max(b1, b2) + Inches(0.16), CONTENT_W, Inches(0.9))
            tf = nb.text_frame; tf.word_wrap = True
            _rich(tf.paragraphs[0], note, 11.5, MUTED)
        self.notes(s, notes)
        return s

    def statement(self, kicker, big, sub, notes):
        """A single load-bearing claim, centred."""
        s = self._blank()
        self.n += 1
        band = s.shapes.add_shape(1, Emu(0), Inches(1.55), SW, Inches(3.4))
        band.fill.solid(); band.fill.fore_color.rgb = BAND
        band.line.fill.background(); band.shadow.inherit = False
        kb = s.shapes.add_textbox(M, Inches(1.85), CONTENT_W, Inches(0.4))
        tf, r = _tf(kb, 13, ACCENT, True, PP_ALIGN.CENTER); r.text = kicker.upper()
        bb = s.shapes.add_textbox(Inches(1.1), Inches(2.35), SW - Inches(2.2), Inches(1.5))
        tf = bb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        _rich(p, big, 30, INK, bold=True)
        sb = s.shapes.add_textbox(Inches(1.3), Inches(3.95), SW - Inches(2.6), Inches(1.0))
        tf = sb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        _rich(p, sub, 16, MUTED)
        ft = s.shapes.add_textbox(M, FOOT_Y, CONTENT_W, Inches(0.28))
        tf, r = _tf(ft, 9, FAINT)
        r.text = f"Model lineage · zebrafish virtual-shoal assay          {self.n}"
        tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
        self.notes(s, notes)
        return s

    def video(self, title, path, poster, caption, notes, kicker=None):
        s = self._blank(); self._chrome(s, title, kicker)
        w, h = Inches(8.4), Inches(4.72)
        s.shapes.add_movie(str(path), Inches((13.333 - 8.4) / 2), BODY_Y, w, h,
                           poster_frame_image=str(poster), mime_type="video/mp4")
        cb = s.shapes.add_textbox(M, BODY_Y + h + Inches(0.12), CONTENT_W, Inches(0.6))
        tf = cb.text_frame; tf.word_wrap = True
        _rich(tf.paragraphs[0], caption, 13, MUTED)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        self.notes(s, notes)
        return s

    def save(self, path):
        self.prs.save(str(path))
        return path


# --------------------------------------------------------------------------- #
# the deck
# --------------------------------------------------------------------------- #
V = {"resolved": ("resolved", WIN, True), "worse": ("WORSE", LOSE, False),
     "noise": ("within noise", FAINT, False), "control": ("control", MUTED, False)}


def build():
    rows, (bit_lo, bit_hi) = master_rows()
    d = Deck()

    # -- 1 ------------------------------------------------------------------ #
    d.title_slide(
        "From an 8-head mixture density to a contained trajectory simulator",
        "Every mechanism added to the zebrafish bout model, its motivation, and what it measured",
        ["**Nine controlled experiments.** Trajectory-C2ST AUC **0.890 → 0.760** against a "
         "real-vs-real floor of **0.476** — about 31% of the closable gap.",
         "**Five changes adopted, three clean nulls, four resolvedly worse.**",
         "The predictive likelihood moved **0.008 bits/bout** across all 24 evaluated rows, because "
         "almost every accepted change lives outside it."],
        """Framing for the talk. Three things to land in the first minute:
(1) this is a generative model of behaviour, judged by whether a classifier can tell its
    trajectories from real ones;
(2) the number went from 0.89 to 0.76 against a floor of 0.476, not 0.5 — that floor is the
    single most misread number in the whole project;
(3) the honest caveat comes at the end and I am not hiding it: most of what worked is not in
    the likelihood.
If asked how long this took: nine Colab runs, roughly 1-6 h each.""")

    # -- 2 ------------------------------------------------------------------ #
    d.bullets("The problem", [
        "Build a generative model of zebrafish swimming that responds correctly to a virtual shoal — "
        "and whose **whole trajectories**, not just its next-turn predictions, are indistinguishable "
        "from real fish.",
        "",
        (0, "Two things a model must get right, and they are not the same:", MUTED),
        (1, "**Conditional accuracy** — given where the fish is and what it sees, what does it do next?"),
        (1, "**Rollout realism** — run it free for an hour; does the resulting path look like a fish?"),
        "",
        (0, "A model can be excellent at the first and useless at the second. Small per-bout biases "
            "integrate: the fish drifts into the wall, stops pausing, or loses its individuality."),
        "",
        (0, "**Everything in this deck is the second problem.**", ACCENT),
    ], """The distinction on this slide is the reason the project exists. One-step likelihood is
the standard metric in this literature and it is nearly blind to rollout drift.
Concrete example to give if asked: the model can predict each next turn well and still spend
8% of its bouts pressed against the dish wall where a real fish spends 1.2%.""",
        kicker="why this is hard")

    # -- 3 ------------------------------------------------------------------ #
    d.figure("Where it got to", FIGS / "fig2_progress.png",
             "Each point is the winning rung of one experiment. Drops between points are **not** "
             "paired measurements — controls are refit per ladder — but every step was adopted on a "
             "paired within-ladder contrast.",
             """This is the spine of the talk; everything else explains one step on this line.
Point at the floor first. 0.4761 is a real-vs-real negative control: relabel one fish's own
real windows into two random halves and run the identical classifier. If that is not ~0.5 the
protocol, not the model, is producing signal.
So 0.760 is not '26% above chance', it is 31% of the way from where we started to the floor.
The two big drops are `contain` (-0.049) and `kfix` (-0.041).""",
             kicker="headline result")

    # -- 4 ------------------------------------------------------------------ #
    d.bullets("The assay", [
        "**Lateralised virtual shoal.** 15 juvenile zebrafish, each in its own 10 cm dish, filmed "
        "from above at 30 fps in a 3×5 grid — all 15 recorded simultaneously.",
        "**Stimulus** = groups of 2 mm black dots projected from below, simulating conspecifics.",
        "",
        (0, "**Semi-closed loop** (`CLsemi`): at each block onset the group is placed left or right "
            "*relative to the fish's own centroid*, then replays a pre-recorded track.", ACCENT),
        "",
        (0, "Two consequences that shape everything downstream:", MUTED),
        (1, "Dot tracks are uncorrelated with behaviour across fish → **teacher-forcing the stimulus "
            "at evaluation is legitimate**."),
        (1, "Fish-to-shoal distance early in a block is an **initial condition, not behaviour** — "
            "this invalidates the naive version of five metrics."),
        "",
        (0, "Blocks: 60 s stimulus / 50 s pause, after 20 min habituation. 16 conditions = "
            "2 sides × {N ∈ 1,2,4,6,8 moving; N ∈ 1,4,8 static}."),
    ], """The 'placed relative to the fish' detail is the one people miss. At block onset the shoal
is ~75 px from the REAL fish. A free-running generated fish is ~134 px away at that instant —
that is just the swapped-track null, not a behavioural failure, and a 60 s block cannot close
it at 2-4 px per bout. Hence every dot-distance metric has a `_late` variant conditioned on
being >=40 s into a block. Raw gap +23.4 px vs late +8.8 px: two thirds of the raw number is
artefact.""", kicker="the data")

    # -- 5 ------------------------------------------------------------------ #
    d.table("The data, and what a 'bout' is", None,
            [[("bouts", MUTED), ("70,590", INK, True), ("median inter-bout interval", MUTED), "1.233 s"],
             [("fish", MUTED), ("15  (3,129–6,686 each)", INK, True), ("median travel", MUTED), "15.03 px = 4.10 mm"],
             [("session", MUTED), ("174 min of bout onsets", INK, True), ("median |Δθ|", MUTED), "39.0°"],
             [("with a stimulus", MUTED), ("40,048  (56.7%)", INK, True), ("forward / left / right", MUTED), "0.111 / 0.440 / 0.449"]],
            """All of these were recomputed from the bout CSV rather than trusted from notes.
The 11% forward fraction matters: the F/L/R alphabet is strongly turn-dominated, so a model
that gets the forward rate wrong is immediately visible.
Segmentation follows Le Goc et al. 2021 with one deliberate deviation — bout onset is the
speed TROUGH before the peak, not a fixed 80 ms lead. Two rest points give a cleaner
net-travel baseline and a meaningful pose at which to define the stimulus bearing.
Trajectories are split only at tracking gaps, NOT at block boundaries: a bout whose
reorientation spans a pause->stimulus transition is the fish turning toward newly appeared
dots, which is the signal of the assay.""",
            kicker="the data", colw=[1.05, 1.35, 1.35, 1.05], fsize=13, top=Inches(1.85),
            note="Segmentation after Le Goc et al. 2021, adapted 25→30 fps. Onset = the speed "
                 "**trough** before the peak. Kinematics onset-to-onset on raw, unsmoothed positions.")

    # -- 6 ------------------------------------------------------------------ #
    d.bullets("Metric A — predictive log-density", [
        "**bits/bout = −mean log₂ p(Δθ | history, stimulus)**, density in degrees.",
        (0, "Equals the held-out joint log-likelihood per bout by the chain rule, so HMMs and "
            "autoregressive networks are directly comparable on it.", MUTED),
        "",
        (0, "**Only the turn channel is reported.** Travel distance and inter-bout interval are "
            "not scored by Metric A — every adapter's `score()` returns a density over Δθ alone.", ACCENT),
        "",
        (0, "This is an **adapter convention, not a harness guarantee**: the harness averages "
            "whatever array it finds under the `logdens` key without checking its channel.", LOSE),
        "",
        (0, "Cross-validation: **leave-3-fish-out, 5 folds**, every fish tested exactly once. "
            "Uncertainty always by fish-level bootstrap — the fish is the exchangeable unit."),
        (0, "Everything that could leak — pools, scales, and every calibration constant later in "
            "this deck — is computed on the **training fold only**."),
    ], """Someone will ask 'why not just report the likelihood?'. Two answers.
First, it is nearly flat: 0.0084 bits across all 24 rows while the AUC moves 0.16.
Second, and more honest: most of what worked is not in it at all. That is slide 29.
If pressed on the convention-vs-structure point: I actually tested this. Feeding a joint
three-channel density through the real scoring path is accepted silently and reported as
bits/bout = 9.881. Nothing checks.""", kicker="how it is judged")

    # -- 7 ------------------------------------------------------------------ #
    d.table("Metric B — the trajectory classifier two-sample test",
            ["test", "dims", "what the classifier sees"],
            [[("turn C2ST", INK, True), "20",
              "symbol frequencies (3) · 3×3 transition matrix (9) · |Δθ| mean/sd/median/q90 (4) · "
              "turn-sign autocorrelation lags 1–3 (3) · mean run length (1)"],
             [("trajectory C2ST", ACCENT, True), "39",
              "the 20 above  +  travel distance mean/sd/median/q90/acf₁ (5)  +  interval, same 5  "
              "+  fish-to-shoal distance, same 5  +  radial occupancy r/R mean/median/q90/P(>0.85) (4)"]],
            """Random forest, 200 trees, GroupKFold BY FISH — so the classifier is always tested on
a fish it has never seen. Real windows are label 1. Windows are 60 bouts, non-overlapping.
The trajectory test is the headline because it is the only one that sees step sizes, timing,
proximity and occupancy. The turn test is kept because it isolates exactly what Metric A scores.
Note the radial block is reported as its own group. Without it, a model that hugs the wall is
only detectable through the dot-distance features — i.e. a CONTAINMENT defect gets
misattributed to the STIMULUS model. That mistake is why the group exists.""",
            kicker="how it is judged", colw=[1.0, 0.45, 5.4], fsize=12.5, top=Inches(1.6))

    # -- 8 ------------------------------------------------------------------ #
    d.statement("read every AUC against this",
                "The floor is 0.476, not 0.5",
                "Relabel one fish's **real** windows into two random halves and run the identical "
                "classifier. Both classes are real, from the same distribution — so the AUC should be "
                "chance. It measures **0.4761**, and is identical across all nine experiments.",
                """This is the slide to slow down on. Two real fish are not exchangeable draws, and
the classifier can exploit that even under GroupKFold, so 'chance' for this protocol is not 0.5.
Consequence for the headline: 0.760 has closed (0.890-0.760)/(0.890-0.4761) = 31% of the
closable gap. Reading it against 0.5 would say 33% of the way to an unreachable target.
The control being IDENTICAL across nine independent runs is also the strongest evidence that
the evaluation protocol itself is stable.""")

    # -- 9 ------------------------------------------------------------------ #
    d.bullets("Making a difference measurable", [
        "**Rollout noise alone gives the AUC an SD of 0.0173** at one rollout — so the SE of a "
        "two-model difference is 0.0245, and a single rollout cannot resolve anything below ~0.035.",
        "",
        (0, "**Fix 1 — average 8 balanced replicates.** Cuts the SE as 1/√R without moving the level. "
            "Pooling replicates into one oversized generated class instead *shifts* the level "
            "(measured 0.790 → 0.822 at R=4)."),
        (0, "**Fix 2 — common random numbers.** Each rollout draws from a stream keyed by "
            "(seed, fish, replicate) and deliberately **not** by model, so replicate r of model A and "
            "of model B see the same draws and the difference is paired."),
        "",
        (0, "Every contrast in this deck is that paired difference:  "
            "**t = mean(d) / (sd(d)/√n),  n = 8,  t_crit = 2.3646**.", ACCENT),
        "",
        (0, "Excluding the model from the stream key is not a detail — with one shared generator, "
            "the same configuration scored **0.760 in one run and 0.7946 in another**, purely from "
            "which other models were in the list.", LOSE),
    ], """The 0.760-vs-0.7946 story is worth telling; it is the most embarrassing bug in the
project and the cheapest to explain. A single RNG threaded through a nested loop over models,
fish and replicates means adding a model changes every other model's draws.
Also worth saying: because the contrast treats the fitted network as FIXED, it cannot see
fit-to-fit variation. On an A100 the same config gave 7.7388 vs 7.7365 bits/bout — pure weight
nondeterminism from cuBLAS atomics. Deterministic flags were added; without them nothing below
~0.01 AUC is resolvable.""", kicker="how it is judged")

    # -- 10 ----------------------------------------------------------------- #
    d.bullets("The guards — and why they exist", [
        (0, "**Twice**, a change improved the headline AUC while making the model materially less "
            "realistic in a dimension nothing was measuring. Both instruments were built afterwards.", LOSE),
        "",
        (0, "**Dispersion.** Per-group gen/real variance ratio (target 1.0) and the between-fish "
            "variance share (real 0.255). The AUC says *how* detectable, never *too narrow or too "
            "wide* — and those need opposite fixes."),
        (1, "Written because `kfix` won −0.0407 while **halving** distance dispersion, 1.04 → 0.52.", LOSE),
        "",
        (0, "**Timing.** Quiescent budget, clock ratio, bout-rate autocorrelation, cross-channel "
            "coupling, distance sd ratio, between-fish turn spread."),
        (1, "Written because `e250` won −0.0113 while cutting quiescent time from 13.0% to **6.9%** "
            "of the session against a real **24.4%**.", LOSE),
        "",
        (0, "**The reading protocol is three numbers on one line: AUC against the floor, the "
            "dispersion table, the timing table.**", ACCENT),
    ], """This is the methodological heart of the talk and the part most transferable to other
projects. A scalar metric will happily sell you a trade against anything it does not measure.
Real reference values to have ready: fish spend 24.4% of the session in gaps >20 s, from only
~0.6% of their bouts — so a bout-weighted version of that statistic reads as a rounding error
and you must time-weight it. Bout-rate autocorrelation at 1/2/3 min is 0.51/0.33/0.22 real.
Across-window corr(log dt, log dist) = -0.386 and corr(log dt, |dtheta|) = +0.308.""",
        kicker="how it is judged")

    # -- 11 ----------------------------------------------------------------- #
    d.bullets("The base model — an 8-head mixture density network", [
        "One hidden ReLU layer. Per bout the network emits **8 numbers** parameterising a "
        "3-component turn distribution:",
        "",
        (1, "**p(Δθ) = w_F · Normal(0, σ_F)  +  w_L · Γ(−Δθ; α_L, θ_L)  +  w_R · Γ(+Δθ; α_R, θ_R)**", ACCENT),
        "",
        (1, "3 mixture-weight logits (softmax)"),
        (1, "σ_F — forward width;  (α_L, θ_L) and (α_R, θ_R) — left and right turn shape/scale"),
        "",
        (0, "All eight vary per bout → heteroscedastic, and the two Gammas are **asymmetric**. "
            "The Normal mean is fixed at exactly 0 and is never an output."),
        "",
        (0, "Inputs are assembled block-wise — turn history · stimulus · wall · (adaptation). "
            "**The column order is a contract**: the closed-loop generator rebuilds single rows of "
            "it from the generated pose, so any drift silently corrupts the rollout.", MUTED),
    ], """The emission family is inherited unchanged from the project's HMMs — that is deliberate,
it is what makes the two families comparable on Metric A. The network replaces the HMM's latent
state prior over {F,L,R} with a direct feedforward function of history and stimulus.
On the column-order contract: two feature blocks were later rewritten to share one vectorised,
scalar-safe function between training and generation precisely so the hazard cannot recur. It
had already bitten more than once.""", kicker="the model")

    # -- 12 ----------------------------------------------------------------- #
    d.bullets("First change — a five-component emission", [
        (0, "**Motivation.** Permutation importances put the residual detectability on turn "
            "*magnitude* — |Δθ| median, sd, q90, mean. One Gamma per side cannot fit the body and "
            "the tail of a turn-magnitude distribution at once."),
        "",
        (0, "**Mechanism.** Split each side into a short-turn and a long-turn Gamma. The head grows "
            "from 8 to **14 outputs**: 5 weight logits, σ_F, and four (α, θ) pairs."),
        "",
        (0, "Only the emission changes — the input features are reused verbatim — so this is a clean "
            "emission-only comparison.", MUTED),
        "",
        (0, "**This is the last change to the turn emission family itself.** Everything later enters "
            "as input features, as separate kinematic heads, or as training budget.", ACCENT),
    ], """Short slide, mostly a bridge. The reason to keep it is that it establishes the pattern
used for the rest of the project: a diagnostic identifies WHICH feature is carrying the
detectability, and the fix is aimed at that feature specifically rather than at capacity in
general.""", kicker="the model")

    # -- 13 ----------------------------------------------------------------- #
    d.bullets("What the baseline already contains — including a joint likelihood", [
        (0, "Between the 8-head network and the first experiment, the model acquired four things "
            "**not attributable to any ladder below**. Stating them stops the lineage taking credit:"),
        (1, "the 5-component emission"),
        (1, "an **honest closed loop** — wall covariate *and* stimulus bearing recomputed from the "
            "fish's own generated pose each bout"),
        (1, "a **separate kinematic trunk** feeding log-normal travel and interval heads"),
        (1, "the per-trajectory `vigor` latent"),
        "",
        (0, "So the network is trained by minimising a **joint three-channel** negative log-likelihood:", ACCENT),
        (1, "**ℒ = −[ log p(Δθ | x_ang)  +  log p(ℓ | x_kin)  +  log p(τ | x_kin, log ℓ) ]**", ACCENT),
        "",
        (0, "The interval head takes log ℓ as an input, so the kinematic pair is a genuine joint "
            "p(ℓ)·p(τ|ℓ), not two marginals. **Metric A evaluates only the first term.**", LOSE),
    ], """This slide exists because I got it wrong first time and wrote a report claiming the `kin`
ladder introduced the trunk. It did not — the trunk is in the very first control, and that
ladder added only two generation-time calibrations.
x_ang and x_kin are DISJOINT: x_ang = 8 turn lags + stimulus + wall (29 dims for shoal);
x_kin = 3 lags of z-scored log-distance then 3 of log-interval (6 dims), with no stimulus,
position, wall or turn in it.
The angle path is still coupled to the kinematic channels through exactly one route: early
stopping is scored on the JOINT validation loss and the saved checkpoint is the full network.

IF ASKED "what ARE those three densities" — each is a FIXED family whose parameters are
per-bout network outputs; that is where the conditioning lives, and the density function
literally takes no feature argument.
- turn: 5-component mixture, w_F*Normal(0, sigma_F) + 2 Gammas per side, 14 outputs from one
  head on x_ang. Links: log_softmax on the 5 weights, softplus+eps on sigma_F and the four
  scales, softplus+1 on the four Gamma shapes.
- travel: mixture of n_dist log-normals on x_kin. interval: mixture of n_dt log-normals on
  (x_kin, log distance). n_dist=2, n_dt=3 after `tail`; 1 and 2 before it.
- The alpha >= 1 floor matters: Gamma ~ u^(alpha-1) as u -> 0, so alpha < 1 would DIVERGE at
  zero turn angle. With alpha > 1 all four Gammas vanish there, so all mass near dtheta = 0
  comes from the forward Normal alone — that is what makes the forward/turn split identifiable.
- The log-normal log-densities lead with -log(x), the change-of-variables Jacobian, so these
  are densities on PIXELS and SECONDS, not Gaussians on log-pixels and log-seconds. Verified
  by integration: 1.000 over the raw variable, 0.089 read as log-space. That is what lets the
  three terms compose into a joint density over deg x px x s. The terms are dimensional, so
  their magnitudes are NOT comparable with each other.""",
        kicker="the model")

    # ======================================================================= #
    # the nine experiments
    # ======================================================================= #
    def res(lad, rung):
        for l, r, auc, dl, t, v in rows:
            if l == lad and r == rung:
                return auc, dl, t, v
        raise KeyError((lad, rung))

    def verdict_line(lad, rung, label=None):
        auc, dl, t, v = res(lad, rung)
        col = {"resolved": WIN, "worse": LOSE, "noise": FAINT}[v]
        tag = {"resolved": "RESOLVED — adopted", "worse": "RESOLVED WORSE",
               "noise": "within noise"}[v]
        return (0, f"**{label or rung}**:  ΔAUC **{dl:+.4f}**   t = {t:+.2f}   →  {tag}", col)

    # -- 14 ----------------------------------------------------------------- #
    d.figure("① `stim` — the model was blind to range", FIGS / "fig4_range.png",
             "Knots at 25/50/90/160 px, softmax-normalised over log r so the gain **flat-extrapolates** "
             "instead of decaying to zero at both ends.   " +
             f"**shoal: ΔAUC {res('stim','shoal')[1]:+.4f}, t = {res('stim','shoal')[2]:+.2f} — adopted.**",
             """Motivation: the baseline stimulus was [sin phi, cos phi, moving] — a pure bearing.
Real following is an inverted-U peaking 55-90 px, weak beyond 120, and SIGN-REVERSED below
25 px where fish turn away. A monotone proximity weight 1/(1+r/r0) can represent neither a
peak nor a sign change.
Mechanism: a softmax basis over log r crossed with the bearing and group size. Softmax rather
than raw Gaussians is the load-bearing choice — it makes the profile flat-extrapolate, whereas
raw Gaussians force the gain to zero at the near end, which is exactly where a sustained
NEGATIVE gain is required.
Note what the figure also shows honestly: `shoal` captures the peak but overshoots at 90-120
and does not roll off enough beyond 120. That residual is what the `far` experiment later
targeted — and failed to resolve.
The control here is worth stating: `dot`, the monotone version, measured RESOLVEDLY WORSE
(+0.0112). The defect was the SHAPE of the range response, not its absence.""",
             kicker="experiment 1 of 9")

    # -- 15 ----------------------------------------------------------------- #
    d.bullets("② `wall` — a well-diagnosed defect that would not move", [
        (0, "**The best-diagnosed defect in the project.** Real fish turn toward the dish centre on "
            "**71–73%** of bouts inside the wall shell, falling to 57% mid-dish. The network learns a "
            "nearly flat 0.61 → 0.56 — a 10–12 point deficit exactly in the shell."),
        "",
        (0, "Measured consequences:", MUTED),
        (1, "heads outward on **64%** of at-wall bouts"),
        (1, "**2.4×** more likely than a real fish to still be at the wall 5 bouts later"),
        (1, "occupies r/R > 0.85 on **7.8%** of bouts vs **1.2%** real; r/R > 0.95 on 2.8%, where "
            "real fish *never* go (real max 0.990)"),
        "",
        (0, "**Mechanism.** The same remedy that worked for range: a softmax basis over log(distance "
            "to wall) crossed with the inward bearing, 3 channels → 9. Plus specular reflection "
            "instead of clipping."),
        "",
        verdict_line("wall", "shoal_wall", "shoal_wall"),
        verdict_line("wall", "dotb_wall", "dotb_wall"),
        "",
        (0, "**A per-bout response fix cannot repair a stationary-distribution defect.** The "
            "conditional turn probability at the wall improved; where the fish spends its time "
            "did not.", ACCENT),
    ], """This is the most important negative result in the deck and it is what eventually pointed
at experiment 5. Do not rush it.
The guardrail was itself implicated: 2.5% of steps are clipped at the wall, 59% of those land
beyond r/R 0.95, and 28% of all at-wall bouts directly follow a clip. Clipping PLACES the fish
on the wall and leaves its heading pointing outward, so the next bout starts blocked too.
Reflection fixes that mechanically and is still metric-invisible on its own — it was retained
inside the containment fix later.""", kicker="experiment 2 of 9")

    # -- 16 ----------------------------------------------------------------- #
    d.bullets("③ `kin` — repairing the autoregressive kinematics", [
        (0, "The kinematic trunk is **not** introduced here — it is already in the baseline. Both "
            "rungs are generation-time calibrations on the same fitted network, which is why "
            "bits/bout is identical to seven digits across all three rungs.", MUTED),
        "",
        (0, "**(i) Drift re-sizing.** The vigor drift was sized so the *injected* drift alone carried "
            "the full real window-to-window variance — but the rollout already varies and the two "
            "**add**, so the injected drift landed at **1.80×** the real spread. Baselines are refit per "
            "model and span 0.0147-0.0157 (log-distance) against a real target of 0.0194."),
        (1, "Fix: measure the baseline with the drift off, size the drift to the *difference*."),
        "",
        (0, "**(ii) Exposure-bias shrinkage.** Teacher-forced training fits the head against the "
            "*real* history; at generation it is fed its own noisy samples, so the loop re-amplifies. "
            "Free-run log-distance lag-1 autocorrelation **+0.42 vs +0.23 real**."),
        (1, "Fix: scale the z-scored kinematic history by λ < 1, found by grid + linear "
            "interpolation against the real acf. Refit per model: λ ∈ [0.455, 0.540] (distance), "
            "[0.813, 0.828] (interval)."),
        "",
        verdict_line("kin", "shoal_drift", "drift"),
        verdict_line("kin", "shoal_kfix", "kfix"),
    ], """This was the largest win at the time (-0.0407) and it is also the one that taught the
project its most transferable lesson — next slide.
On the drift bug: this is a 'calibrating against a moving baseline' error, and it recurred
later with the quantile map. The general fix is always the same: measure the baseline with the
mechanism switched OFF and size the mechanism to the difference.""",
        kicker="experiment 3 of 9")

    # -- 17 ----------------------------------------------------------------- #
    d.table("③ …and what it quietly cost",
            ["", "turn", "distance", "interval", "dot-dist", "radial"],
            [[("shoal", MUTED), "0.71", "1.23", "0.55", "1.18", "2.17"],
             [("+ drift", MUTED), "0.71", ("1.04", WIN, True), "0.47", "1.14", "2.17"],
             [("+ kfix", MUTED), "0.69", ("0.52", LOSE, True), ("0.34", LOSE), "1.16", "1.98"],
             [("real target", ACCENT, True), ("1.00", ACCENT), ("1.00", ACCENT), ("1.00", ACCENT),
              ("1.00", ACCENT), ("1.00", ACCENT)]],
            """The drift re-size did exactly what it was designed to do: 1.23 -> 1.04.
The shrinkage then HALVED it. That is mechanical, not a bug: for an AR(1) the stationary
variance is sigma^2/(1-phi^2), so shrinking the feedback damps autocorrelation AND spread
together. dt_w1 regressing 0.871 -> 0.933 in the same run is the visible symptom.
The metric rewarded the trade by -0.0407 and nothing reported at the time could see it. This
single result is why the dispersion report exists, and it is the template for the two failed
repairs on the next slide.""",
            kicker="experiment 3 of 9", fsize=13, top=Inches(1.75),
            colw=[1.3, 0.8, 0.9, 0.9, 0.9, 0.8],
            note="Gen ÷ real variance ratio per feature group. **`kfix` won −0.0407 AUC while halving "
                 "distance dispersion.** The AUC cannot tell *too narrow* from *too wide*, and those "
                 "need opposite fixes.")

    # -- 18 ----------------------------------------------------------------- #
    d.bullets("④ `disp` — two attempts to buy the dispersion back, both lost", [
        (0, "**The idea was sound.** Rescaling a series about its mean by c leaves its "
            "autocorrelation *exactly* unchanged while scaling its variance by c² — so λ sets the "
            "autocorrelation and c restores the spread. Orthogonal knobs."),
        "",
        verdict_line("disp", "shoal_kfixvc", "kin_var_comp"),
        verdict_line("disp", "shoal_kfixvc_turn", "+ vigor_turn"),
        "",
        (0, "**The largest effects in the entire project — in the wrong direction.**", LOSE),
        "",
        (0, "Diagnosis:", MUTED),
        (1, "`kin_var_comp` widens a log-normal **symmetrically about a fixed centre**: right "
            "variance, wrong shape. It pushes mass where the real distribution has none."),
        (1, "`vigor_turn` scales |Δθ| itself, which wrecked the pooled turn marginal — "
            "`abs_dtheta_w1` went **6.67 → 8.43**."),
        "",
        (0, "**Right variance with the wrong shape is worse than too little variance.**", ACCENT),
    ], """Worth flagging: these deltas were measured while the WALL was still broken, i.e. in an
unusually sensitive regime. See the masking slide — that matters for how much weight to put on
the magnitude, though not on the sign.
The eventual correct version of this idea is to rescale the AR's driving INNOVATION rather than
the emitted value, which preserves the normalised autocorrelation because acf(k) = phi^k is
scale-free. That is built but sits after this deck's cutoff.""",
        kicker="experiment 4 of 9")

    # -- 19 ----------------------------------------------------------------- #
    d.figure("⑤ `contain` — the largest single win", FIGS / "fig5_radial.png",
             "Wall excess factorises: **entry rate 2.7× × dwell 1.8× = 5.0× occupancy.** A constant "
             "inward bias can only attack entry, so the fix gates part of the bias on being inside "
             "the shell.   " +
             f"**ΔAUC {res('contain','shoal_contain')[1]:+.4f}, t = {res('contain','shoal_contain')[2]:+.2f}.**",
             """Four per-bout fixes had already failed on this defect. What cracked it was the
DECOMPOSITION: entry rate 0.0145 vs 0.0053 per bout (2.7x) times dwell 4.21 vs 2.32 bouts
(1.8x) = the 5.0x occupancy excess, and the identity closes to two digits. A constant bias matches the mean r/R while leaving
P(r/R>0.85) at 0.033 against a real 0.012, because the tail is DWELL, not entry.
Mechanism: cb = bulk + shell * max(0, (u-u0)/(1-u0)) with u0 = 0.75, added as cb*sin(psi_c) to
the turn. Bulk acts everywhere and sets the mean radius; shell switches on only inside the
shell and attacks dwell.
Supporting numbers (held out, 15 fish): occupancy 0.0610 -> 0.0164 (real 0.0123), mean r/R
0.482 -> 0.396 (real 0.449) -- it overshoots slightly INWARD,
wall_frac_gap 0.0499 -> 0.0042, and dot_dist_bias_late +3.20 -> -2.64 — the shoal-distance
error largely WAS the containment error. Radial dispersion collapsed 1.84 -> 0.72.
Crucially NO cost to the stimulus response: following curvature +0.065 -> +0.064, restoring-
force zero-crossing 40.8 -> 42.0 px against a real 41.4.
If asked about the calibration: two objectives (occupancy and mean radius) scored in
between-fish SD units and weighted 2.6:1 by measured permutation importance. An earlier version
summed RELATIVE errors, occupancy dominated 115:1, the search ran to the grid ceiling and
produced a WORSE absolute error than doing nothing.""",
             kicker="experiment 5 of 9")

    # -- 20 ----------------------------------------------------------------- #
    d.bullets("⑥ `tail` — the timing distribution was truncated, not mis-centred", [
        (0, "The real upper tails are far heavier than a 1-component distance / 2-component interval "
            "head can carry:"),
        (1, "**intervals**: median 1.23 s but q99 / q99.9 / max = **18.7 / 103.6 / 426.5 s**"),
        (1, "**distances**: median 15 px but 44 / 85 / 218 px"),
        "",
        (0, "The heads got the **bulk right and the tail wrong**: generated median interval 1.292 s "
            "against a real 1.281 s, but generated q99.9 was **24.6 s against 103.6 s**, and only "
            "**0.055%** of generated bouts exceeded 30 s against **0.63%** real."),
        "",
        (0, "That collapsed every tail-sensitive window feature — across-window variance ratios of "
            "0.04 (dt sd), 0.08 (dt mean), 0.12 (dt q90) — while dt *median*, a bulk statistic, "
            "sat at 0.46.", LOSE),
        "",
        (0, "**Mechanism.** Generalise both kinematic heads to K-component log-normal mixtures: "
            "n_dist = 2, n_dt = 3. Each channel gets a rare slow mode to put the long pauses in."),
        "",
        verdict_line("tail", "shoal_contain_tail", "tail"),
    ], """The clip was explicitly ruled out as the cause — it binds on 0.001% of generated bouts.
The win landed where the mechanism predicted: dt_w1 0.926 -> 0.870 and dist_w1 3.36 -> 3.19.
General lesson: a distribution can have the right centre and still be badly wrong. Three of the
four worst window features here were tail statistics of a channel whose median was essentially
perfect.""", kicker="experiment 6 of 9")

    # -- 21 ----------------------------------------------------------------- #
    d.bullets("⑦ ⑧ `far` and `cal` — two clean nulls", [
        (0, "**`far` — an extra far-field knot.** With knots at 25/50/90/160 the basis saturates "
            "beyond the outermost one, so the model cannot tell 160 px from 400 px. Measured: the "
            "generated restoring force is **−3.4 px/bout beyond 120 px where real fish are at −0.53** "
            "— it keeps hauling itself toward the shoal from any distance. Real range reaches 280 px "
            "at q99, so a fifth knot gives the head resolution to *learn* the back-off."),
        verdict_line("far", "shoalf_contain_tail", "shoalf"),
        "",
        (0, "**`cal` — the containment recalibration.** Corrected two-objective estimator (12 fish, "
            "weight 2.6) versus the original (6 fish, equal weight), run paired in one ladder."),
        verdict_line("cal", "shoal_contain_tail", "corrected calibration"),
        "",
        (0, "Both mechanisms are defensible on their own diagnostics. Neither is detectable on the "
            "headline metric.", MUTED),
        "",
        (0, "**Do not read either as a refutation yet — see the masking slide.**", LOSE),
    ], """The `far` null is visible in the range figure earlier: `shoal` really does fail to roll off
beyond 120 px, so the mechanism was aimed at a real defect. It just did not move the AUC.
The `cal` control is the OLD setting, so a negative delta would have favoured the legacy
calibration. It measured +0.0009, t = +0.34 — genuinely nothing.
Both of these are exactly the cases the masking finding reopens.""",
        kicker="experiments 7–8 of 9")

    # -- 22 ----------------------------------------------------------------- #
    d.bullets("⑨ `conv` — the fits were not converged", [
        (0, "An 80-epoch budget with patience 8 was stopping early: patience 8 halted at epoch 61 "
            "with the best at 52, while patience 25 ran to 108 with the best at 82."),
        (0, "**Every previous experiment had therefore been comparing under-trained models.**", LOSE),
        "",
        (0, "**Mechanism.** 250 epochs, patience 30 — made a *constructor* override that wins over "
            "the run config, so both budgets sit in one ladder and the contrast runs **paired** "
            "rather than across runs."),
        "",
        verdict_line("conv", "shoal_contain_tail_e250", "e250"),
        "",
        (0, "**But the guards object.** Second time an accepted change degraded an uninstrumented "
            "property — and this time the instrument caught it:", LOSE),
        (1, "time in gaps > 20 s:  13.0%  →  **6.9%**   (real **24.4%**)"),
        (1, "clock ratio: **0.870** — finishes the 174-minute programme in 142 minutes"),
        (1, "`abs_dtheta_w1`:  6.47 → 6.75"),
        "",
        (0, "Adopted, with the caveat recorded rather than buried.", ACCENT),
    ], """Because the shoal schedule is indexed by BOUT NUMBER rather than wall-clock, a model that
under-produces long pauses runs through the stimulus programme too fast — and
dot_dist_bias_late, which conditions on real time-in-block, partially masks that.
This is the cheapest win in the project (just train longer) and also the clearest illustration
that a single scalar metric will sell you a trade against anything it does not measure.""",
        kicker="experiment 9 of 9")

    # -- 23 ----------------------------------------------------------------- #
    trows = []
    for lad, rung, auc, dl, t, v in rows:
        col = {"resolved": WIN, "worse": LOSE, "noise": FAINT, "control": MUTED}[v]
        tag = {"resolved": "resolved", "worse": "WORSE", "noise": "noise", "control": "control"}[v]
        trows.append([(lad, MUTED), rung, f"{auc:.4f}",
                      "—" if dl is None else f"{dl:+.4f}",
                      "—" if t is None else f"{t:+.2f}",
                      (tag, col, v in ("resolved", "worse"))])
    d.split_table(f"All {len(rows)} evaluated rows — 16 distinct models",
                  ["ladder", "rung", "AUC", "Δ", "t", "verdict"], trows,
            """Every number here is rebuilt from the ladder archives at build time, so the deck cannot
drift from the data.
The one comparison rule to state out loud: absolute AUCs are NOT comparable across ladders,
because each ladder refits its own control. The identical model `contain_tail` reads 0.7714,
0.7681, 0.7712 and 0.7712 in four different ladders. Only the within-ladder paired delta is a
measurement; the absolute column is context.
If someone asks about the likelihood column (not shown): it spans 0.0084 bits across all 24
rows while the trajectory AUC moves 0.16.""",
                  kicker="results", fsize=9.5, top=Inches(1.42),
                  colw=[0.62, 1.85, 0.72, 0.78, 0.58, 0.92],
                  note=f"n = 8 replicates, t_crit = {CRIT:.4f}. Negative Δ is better. "
                 f"**Absolute AUC is not comparable across ladders** — each refits its own control. "
                       f"`bits_per_bout` across all {len(rows)} rows spans only "
                       f"**{bit_hi - bit_lo:.4f}** bits ({bit_lo:.4f}–{bit_hi:.4f}).")

    # -- 24 ----------------------------------------------------------------- #
    d.figure("What it actually produces", FIGS / "fig1_trajectories.png",
             "60 s of free-running behaviour, stimulus teacher-forced. Orange = active dot positions. "
             "The generated fish stays in the dish, engages the shoal, and produces plausible "
             "excursion structure — the residual detectability is **statistical, not visual**.",
             """Use this to make the point that we are well past 'the model looks broken'. Nothing
here is obviously wrong to the eye, which is exactly why a classifier is needed to adjudicate.
One honest visible difference if someone spots it: real paths are more angular — real bouts are
discrete darts with sharp direction changes — while generated paths look slightly smoother and
more looping. That is the dart-and-hold interpolation in the renderer as much as the model.
These trajectories were regenerated from ladder_e250.pt specifically for this deck. The samples
sitting in runs/mix_loop/ are from a later model and would have mixed eras.""",
             kicker="results", max_h=4.55)

    # -- 25 ----------------------------------------------------------------- #
    if VIDEO.exists():
        d.video("The same thing, moving", VIDEO, FIGS / "fig1_trajectories.png",
                "Real (top row) vs generated (bottom row), four fish, same stimulus. "
                "Click to play in slideshow mode.",
                """Let it run 15-20 seconds. What to point at:
- the generated fish darts and holds like a real one rather than gliding;
- it stays off the wall — that is the containment bias doing its job, and before experiment 5
  the bottom row visibly hugged the boundary;
- it turns toward the dot cloud when the cloud is at mid-range.
This is ladder_e250.mp4, the same model as the static figure.""",
                kicker="results")

    # -- 26 ----------------------------------------------------------------- #
    d.table("What did not work — mechanisms",
            ["mechanism", "Δ AUC", "diagnosis"],
            [["monotone proximity (`dot`)", ("+0.0112", LOSE, True),
              "Cannot represent an inverted-U or a sign change. Resolvedly worse than the "
              "bearing-only ablation it was meant to improve."],
             ["border basis (`_wall`)", ("−0.0065", FAINT), "null"],
             ["reflecting wall", ("—", FAINT),
              "Mechanically correct, metric-invisible alone. Retained inside `contain`."],
             ["constant centre bias", ("—", FAINT),
              "Attacks entry rate only; leaves the tail, which is dwell."],
             ["2-D containment", ("—", FAINT), "6× worse on the calibration objective than the 1-D scale."],
             ["`kin_var_comp`", ("+0.0570 / +0.0324", LOSE, True),
              "**Lost twice.** Symmetric log widening about a fixed centre: right variance, wrong shape."],
             ["`vigor_turn`", ("+0.0480", LOSE, True),
              "Scaling |Δθ| per trajectory wrecks the pooled turn marginal, 6.67 → 8.43."],
             ["far-field knot (`shoalf`)", ("−0.0058", FAINT),
              "Mechanism-positive on the restoring-force profile, undetectable on the AUC."],
             ["quantile map", ("—", FAINT),
              "Right *shape* of correction — monotone, support-preserving — but context-blind, and "
              "calibrated against a rollout it then changes."],
             ["rollout-aware policy gradient", ("—", FAINT),
              "A control with the policy-gradient weight set to zero reproduced most of the gain: "
              "the improvement was **under-training**, not the reinforcement signal."]],
            """There are more entries here than successes, and that is the honest shape of the project.
The policy-gradient control is the one I would highlight to a methods-minded audience: it would
have been very easy to publish that as 'RL fine-tuning improves rollout realism' without ever
running the zero-weight arm.""",
            kicker="negative results", fsize=10.5, top=Inches(1.36),
            colw=[1.5, 0.95, 4.0])

    # -- 27 ----------------------------------------------------------------- #
    d.bullets("What did not work — method", [
        (0, "**Three of the four most expensive mistakes were methodological, not architectural.**", LOSE),
        "",
        (0, "**Oracle support violation (twice).** An affine rescale, then a Gaussian offset, applied "
            "to *bounded* features that are frequently exactly 0. Both pushed values negative, outside "
            "the real support, and the classifier detected the impossible values rather than the "
            "intended effect: **AUC 0.997 and 0.968**. Fix: rank/quantile maps, support-preserving "
            "by construction."),
        "",
        (0, "**The 3-fish evaluation trap (three times).** Radial and dispersion statistics vary "
            "**7–12×** across fish. A 3-fish test made a working fix look like a regression and nearly "
            "caused a good one to be discarded. Between-fish spread is the quantity being estimated — "
            "a small subset's estimate of it is worthless."),
        "",
        (0, "**Calibrating against a moving baseline (recurring).** The mechanism is sized against a "
            "rollout that the mechanism then changes. Fix: measure with it off, size to the difference."),
        "",
        (0, "**Unnormalised multi-objective calibration.** Summing *relative* errors let a ~1% "
            "occupancy dominate a ~0.5 mean by **115:1**; the guard was inert and the search ran to "
            "the grid ceiling — a worse absolute error than doing nothing."),
    ], """These generalise beyond this project, which is why they get a slide.
The 3-fish trap has a written rule now: never evaluate a dispersion, radial or individuality
quantity on a subset. It fired three separate times.
The support-violation one is subtle and worth spelling out if there is time: many window
features are fractions that are exactly 0 in most windows. ANY additive or multiplicative
perturbation takes them negative, and 'impossible value' is the easiest signal a classifier
will ever find.""", kicker="negative results")

    # -- 28 ----------------------------------------------------------------- #
    d.statement("the finding that reframes every null",
                "AUC defects mask each other — 40× ",
                "An identical injected defect is worth **+0.042 alone**, **+0.001** once a single 20% "
                "marginal error is also present, and **+0.049 again** after the marginals are "
                "repaired. Baseline 0.740 — nowhere near a ceiling.",
                """This is the most consequential methodological finding in the project and the one I
would defend hardest.
The AUC is a RANK statistic, so defects combine sub-additively: a large unrelated defect
absorbs the classifier's power and a genuine fix then measures as a clean null.
Direct consequence: `far`, `cal` and the turn-persistence mechanism may be MASKED, not refuted.
They cannot be re-read until the distance marginal is repaired — currently dist_std 5.84
against a real 8.39, i.e. 26-32% low.
The rule that comes out of it: never record a null as a refutation without checking what else
is broken.""")

    # -- 29 ----------------------------------------------------------------- #
    d.table("Two objects, two likelihoods",
            ["mechanism", "fitted by MLE?", "scored by Metric A?", "where it lives"],
            [["5-component emission", ("yes", WIN, True), ("yes", WIN, True), "network head"],
             ["`shoal` range basis", ("yes", WIN, True), ("yes", WIN, True), "input features"],
             ["border basis · adaptation", ("yes", WIN, True), ("yes", WIN, True), "input features"],
             ["**kinematic trunk + heads**", ("yes", WIN, True), ("no", LOSE, True), "separate trunk"],
             ["**tail components**", ("yes", WIN, True), ("no", LOSE, True), "network heads"],
             ["convergence budget", ("yes", WIN, True), ("yes", WIN, True), "training"],
             ["`vigor` · drift re-size", ("no", LOSE), ("no", LOSE), "generation, post-fit"],
             ["`kin_feedback` · `kin_var_scale`", ("no", LOSE), ("no", LOSE), "generation, post-fit"],
             ["containment bias · wall mode", ("no", LOSE), ("no", LOSE), "generation, post-fit"],
             ["turn-persistence · quantile map", ("no", LOSE), ("no", LOSE), "generation, post-fit"]],
            """This is the honest caveat and it should be read before anyone quotes the results.
score() never reads vigor, kin_feedback, centre_bias, qmap, artr or kin_var_scale — those six
live only in the generator. It does not read the distance or interval heads either, even
though those ARE fitted.
Concretely: the shipped sampler discards 48% of its own distance history (lambda = 0.521) at
zero cost to the reported likelihood.
The two mismatched rows are the sharpest statement of the problem: roughly two thirds of the
fitted emission is optimised against a likelihood that is never reported.
In favour, and I do defend this: every generation-only constant is calibrated on TRAINING FISH
ONLY, so nothing leaks and they generalise to held-out fish exactly as a fitted parameter would.
Several are provably corrective — the exposure-bias shrinkage repairs a mismatch that exists
BECAUSE training is teacher-forced and generation is not.
Against: they are invisible to any likelihood, they are fitted SEQUENTIALLY so each absorbs the
previous ones' residuals, and experiments 4 and 5 prove they can buy a metric win by degrading
an unmeasured property.""",
            kicker="the honest caveat", fsize=11.5, top=Inches(1.42),
            colw=[2.1, 0.95, 1.15, 1.5],
            note="Metric A evaluates only the first term of the joint objective. **Two mechanisms are "
                 "fitted by maximum likelihood and invisible to it** — neither can move "
                 "`bits_per_bout` by a single digit.")

    # -- 30 ----------------------------------------------------------------- #
    d.bullets("Calibration order — two constraints that are load-bearing", [
        (0, "Eleven steps run inside `fit()`. Three come before the network is trained, the rest "
            "after, each measured on rollouts of the model as it stands at that moment.", MUTED),
        "",
        (0, "**1–2** train-fold scales and clips  →  vigor calibration *(needs no network)*"),
        (0, "**3  FIT THE NETWORK**  — joint angle + distance + interval NLL", ACCENT),
        (0, "**4–5** re-size the vigor drift *(needs the fitted net)*  →  `kin_feedback` shrinkage"),
        (0, "**6–11** quantile map → containment scale → innovation compensation → individuality "
            "latents → **re-fit containment** → turn-persistence β"),
        "",
        (0, "**The quantile map must precede containment** — it rescales step length, so the wall "
            "bias has to be calibrated against the *mapped* rollout.", LOSE),
        (0, "**Containment must be re-fitted after anything that widens movement** — movement spread "
            "drives wall contact (measured correlation **+0.59**) and wall occupancy is the single "
            "largest C2ST feature. Omitting that re-fit is exactly how `contain_vc` lost.", LOSE),
        "",
        (0, "One asymmetry remains unresolved: turn-persistence is calibrated **last**, after three "
            "mechanisms have already changed the turn stream, and the quantile map is never re-fitted "
            "after a later mechanism widens the marginals it was matched to.", MUTED),
    ], """The sequential-fitting critique from the previous slide is concrete here: each calibrator
absorbs the residuals of the ones before it, so the order is not cosmetic and the whole stack
is only locally optimal.
If asked what the principled alternative is: fit them jointly against the rollout statistics,
which is essentially the rollout-aware training direction — and the one attempt at that turned
out to be measuring under-training instead.""", kicker="the honest caveat")

    # -- 31 ----------------------------------------------------------------- #
    d.figure("Where it stands — dispersion", FIGS / "fig3_dispersion.png",
             "Radial over-dispersion was **eliminated** (2.17 → 0.65, slightly overshooting) and "
             "dot-distance brought to target. The behavioural channels remain "
             "**under**-dispersed — the model is still too typical.",
             """The two channels that improved dramatically — shoal distance and wall occupancy — are
the two that were structurally broken. The three per-bout marginal distances were already close
and remain so.
That is the correct reading of what nine experiments bought: the TRAJECTORY-LEVEL GEOMETRY was
repaired; the per-bout marginals were never the problem.
The between-fish share in the title is the headline deficit: 0.081 generated against 0.255 real,
and it has barely moved in nine experiments.""",
             kicker="where it stands", max_h=4.5)

    # -- 32 ----------------------------------------------------------------- #
    if (FIGS / "fig6_importances.png").exists():
        d.figure("Where it stands — what still gives it away", FIGS / "fig6_importances.png",
                 "Permutation importances: the drop in held-out AUC when one feature is shuffled in "
                 "the test fold. Measured on the reported metric, not on where the forest looked.",
                 """Impurity importance is measured on the training split and biased toward
high-variance features — it tells you where the forest looked, not what made the samples
detectable. The two disagree whenever features are correlated, and these are: they are moments
of the same channels.
Whatever group tops this chart is the next thing to fix. That is how every experiment in this
deck was chosen.""", kicker="where it stands", max_h=4.5)

    # -- 33 ----------------------------------------------------------------- #
    d.bullets("The three honest deficits", [
        (0, "**1 · Individuality.** Between-fish variance share **0.081 against a real 0.255**, and it "
            "has barely moved in nine experiments — not because the mechanism is mistuned but "
            "because **there is no mechanism.** The vigor latent covers interval and distance *level* "
            "only, and its turn term is hard-set to zero.", LOSE),
        (1, "Between-fish spread of forward fraction: **0.005 generated vs 0.054 real**. "
            "Of median |Δθ|: **0.6° vs 6.7°**. Every generated fish turns like the average fish."),
        "",
        (0, "**2 · Under-dispersion of the behavioural channels.** Distance 0.51, interval 0.40 "
            "against a target of 1.0. Precisely located: a scalar shrinkage on the kinematic history "
            "couples autocorrelation and spread through σ²/(1−φ²), so you get the correct lag-1 "
            "**or** the correct spread, not both.", LOSE),
        "",
        (0, "**3 · Timing.** Quiescent time **6.9% vs 24.4% real**; clock ratio 0.87; bout-rate "
            "autocorrelation 0.24/0.05/0.00 against 0.51/0.33/0.22. The cause is a missing "
            "**right tail** in the interval distribution, not missing slow correlation.", LOSE),
    ], """Be direct about these; an audience trusts a talk that names its own failures precisely.
Deficit 1 is the largest single identifiable gap and the most tractable, because it needs a
mechanism that does not exist rather than a better tuning of one that does.
A subtlety worth having ready for deficits 1 and 3: the missing structure is SLOW. The
per-bout independence the model assumes is empirically almost right — the within-bout partial
correlation between interval and turn magnitude, conditioned on what the model already
conditions on, is only +0.026, against +0.308 across 60-bout windows. So a richer per-bout
factorisation will not fix it; a shared slow arousal state would.""",
        kicker="where it stands")

    # -- 34 ----------------------------------------------------------------- #
    d.bullets("Next", [
        (0, "**Built and pre-registered, results after this deck's cut-off:**", ACCENT),
        (1, "`mem` — a four-arm memory and attribution ladder. The shrinkage may itself be an "
            "artefact of a 3-lag history: with three lags you cannot get a slow tail without "
            "inflating the head. Includes a no-shrinkage control so a longer-history win cannot be "
            "miscredited to the memory."),
        (1, "`indiv` — per-trajectory latents for the two channels vigor never covered: turning "
            "style, by tilting the forward mixture weight rather than scaling |Δθ| (that is exactly "
            "how `vigor_turn` failed), and spatial preference, by *multiplying* rather than "
            "offsetting the containment bias."),
        "",
        (0, "**The two well-posed open problems:**", ACCENT),
        (1, "A **variance-compensated shrinkage** that rescales the autoregressive *innovation* "
            "rather than the emitted value — decoupling autocorrelation from spread, since "
            "acf(k) = φᵏ is scale-free."),
        (1, "A **slow latent state scoped to the timing channel only** — measurement shows distance "
            "needs no latent (three lags absorb it) while the interval has residual slow structure "
            "no practical lag count absorbs."),
        "",
        (0, "And before trusting any past null again: **repair the distance marginal, then re-read "
            "`far`, `cal` and the turn-persistence result.**", LOSE),
    ], """Close on the masking point — it is the item with the highest expected value, because it
potentially converts three recorded nulls back into open questions at no experimental cost
beyond one repair.
On K_kin: free-run stability sets a hard ceiling at 8. Longer histories diverge in a
substantial fraction of initialisations and the validation loss cannot see it — one arm had the
BEST held-out NLL while wandering to an acf of 0.99 everywhere. fit() has no detector for this.""",
        kicker="next")

    return d


def main():
    missing = [f for f in ("fig1_trajectories", "fig2_progress", "fig3_dispersion",
                           "fig4_range", "fig5_radial") if not (FIGS / f"{f}.png").exists()]
    if missing:
        raise SystemExit(f"missing figures: {missing}\n  -> run: python slides/make_figures.py")
    d = build()
    p = d.save(OUT)
    print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1e6:.2f} MB, {d.n} slides)")


if __name__ == "__main__":
    main()
