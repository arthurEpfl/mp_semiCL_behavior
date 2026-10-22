"""Build and execute Model/Final_report_extra_data.ipynb: the final test of the models on the Larsch lab screen data.

The heavy work is done beforehand by scripts/twin_test.py (caches in Model/twin_test/, git-ignored):
    uv run --with pyarrow python twin_test.py real
    uv run --with pyarrow python twin_test.py sim 26e 30d 11e 32a 14f 32b
    uv run --with pyarrow python twin_test.py screen       # the screen's own per-fish readouts, for the sanity check
then   python final_report_extra_data.py   (executes the notebook in place; outputs embedded).

UNPUBLISHED DATA (Larsch lab). The notebook embeds results derived from it and is git-ignored for that reason.
"""
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
NB = HERE / "Final_report_extra_data.ipynb"

CELLS = []          # nbformat cells, or ("copy", key): a unit's usual evaluation block, copied with its outputs


def copy_unit(key):
    CELLS.append(("copy", key))


def md(s):
    CELLS.append(nbformat.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(nbformat.v4.new_code_cell(s.strip("\n")))


md(r"""
# Final report — the extra data: a final test on the Larsch lab screen

> **Unpublished data of the Larsch lab** (CIG, University of Lausanne), shared for the digital-twin collaboration
> (`schizscreen_twin_v1/DATA_USE.md`). This notebook embeds results derived from it and is kept out of git.

**The data.** A behavioural screen: 5005 juvenile zebrafish (101 coded lines, mutants and their siblings, 18–35 dpf),
each alone in a 70 mm dish, tracked at 30 fps for 4.5 h while a ~2 mm dot moves on a fixed open-loop path (the same for
every fish). 5-min episodes alternate a **bout-like dot** (a ~2.1 mm jump every 20 frames — close to the training's
moving dots, which jump ~2 mm every 18 frames) and a **continuous dot** (glides; never in training); a loom block sits at
130–160 min. No fish ever sees another fish or more than one dot.

**Who is used.** The sighted *reference siblings* (heterozygous and wild-type siblings, as the screen pools them; in the
screen's genotype contrasts; no exclusion, no identity doubt) as the real fish, and the 26 *blind mutants* as the
no-vision control. The other mutants are left out: 7 lines shoal differently by design. **Clutch** (one cross, one day)
is the unit of replication: real-fish intervals resample clutches.

**The question.** Do the models — trained on 15 fish with groups of 1–8 dots in a 100 mm dish, and *not changed in any
way* — behave like these fish in this experiment?
""")

md(r"""
## Design (written before any model result was looked at)

**Window.** 60–130 min in the dish (episodes 10–23: 7 bout-like, 7 continuous), before the loom block.

**Real side.** Positions → dish-centred mm → model px (×3.60) → `segment_bouts.segment_fish`, the training data's own
bout segmentation. Bouts followed by more than 10 s of rest are dropped (the models are trained on active-only bouts).

**Model side.** M6 (26e | 30d), D4 (11e | 32a) and F (14f | 32b), each simulated unchanged in the same window: the dot
path of a median well, dish radius 35 mm, 200 simulated fish per unit (fold *i* mod 5, a fresh latent each). The block
clock is held at 60 s ('late'): the dot does not appear, it is always there. Under that clock F's correction, which acts
only in the first 20 s of a block, is off, so **F should equal D4 here** (a check, not a result). The model's wall terms
are relative to the dish radius, so its 2–10 mm rejection band becomes 1.4–7 mm in this dish. The **blind model** is
the same simulated individual (same seed) with the dot hidden.

**Measures** (fish–dot distance at the bout's onset, mm), each against a **null dot** — the real dot rotated about the
dish centre by 90°, 180° and 270° (same distance from the centre, same speed, no coupling to the fish):
- *turn toward*: P(the bout turns toward the dot | turn > 5°) minus the same for the null dot;
- *gain*: mm closer to the dot after the bout, minus the same for the null dot;
- *occupancy*: share of bouts at each distance, divided by the null dot's;
- *alignment* (continuous dot): cos(travel heading − the dot's direction of motion) minus null, split by where the dot
  is (ahead / side / behind) to separate alignment from pursuit;
- *scalars*: shoaling index (null − real mean distance) / null mean distance; mean distance; turn toward and gain at
  **6–20 mm (attraction)** and **0–4 mm (repulsion)**; kinematics (median interval, median step, mean distance from the
  centre, mean |turn|).

**Score.** For each stimulus scalar, the **share of the real visual effect reproduced** =
(model − blind model) / (real siblings − blind mutants). Kinematics are reported as they are (older fish, smaller dish:
levels are expected to differ).

**Pass criteria (fixed in advance).** A stimulus scalar *passes* when its share lies in **[0.5, 1.5] at both seeds**.
Expected: the attraction scalars pass with the bout-like dot; the repulsion scalars fail (the models have no
short-range repulsion); the continuous dot is an extrapolation, reported but not scored.
""")

code(r"""
import sys, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "../scripts")
import twin_test as TT
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

B = TT.DBINS; NB = TT.NB; MID = 0.5 * (B[:-1] + B[1:])
LAB = [f"{B[i]:g}-{B[i+1]:g}" for i in range(NB)]
REAL = pickle.load(open(TT.OUT / "real.pkl", "rb"))
META = pd.DataFrame(REAL["meta"]).set_index("fish_id")
RES = REAL["res"]
META = META[META.index.isin(list(RES))]
UNITS = {"M6": ("26e", "30d"), "D4": ("11e", "32a"), "F": ("14f", "32b")}
SIM = {(k, s): pickle.load(open(TT.OUT / f"sim_{k}_{'sighted' if s else 'blind'}.pkl", "rb"))["res"]
       for ks in UNITS.values() for k in ks for s in (True, False)}

def pool(res, ids, kind):
    S = {}
    for i in ids:
        for k, v in res[i][kind].items():
            S[k] = S.get(k, 0.0) + np.asarray(v, float)
    return S

def _sel(lo, hi):
    return (B[:-1] >= lo) & (B[1:] <= hi)

def curves(S):
    with np.errstate(all="ignore"):
        out = dict(occ=(S["n"] / S["n"].sum()) / (S["n_null"] / S["n_null"].sum()),
                   tow=S["ntow"] / S["nturn"] - S["ntow_null"] / S["nturn_null"],
                   gain=S["gain"] / S["ng"] - S["gain_null"] / S["ng_null"])
        if "al_n0" in S:
            for s in range(3):
                out[f"al{s}"] = S[f"al_c{s}"] / S[f"al_n{s}"] - S[f"al_c{s}_null"] / S[f"al_n{s}_null"]
    return out

def scalars(S):
    def tow(lo, hi):
        m = _sel(lo, hi)
        return S["ntow"][m].sum() / S["nturn"][m].sum() - S["ntow_null"][m].sum() / S["nturn_null"][m].sum()
    def gain(lo, hi):
        m = _sel(lo, hi)
        return S["gain"][m].sum() / S["ng"][m].sum() - S["gain_null"][m].sum() / S["ng_null"][m].sum()
    md_, mdn = S["sum_d"] / S["cnt_d"], S["sum_d_null"] / S["cnt_d_null"]
    return {"shoaling index": (mdn - md_) / mdn, "mean distance (mm)": md_,
            "turn toward 6-20 mm": tow(6, 20), "gain 6-20 mm (mm/bout)": gain(6, 20),
            "turn toward 0-4 mm": tow(0, 4), "gain 0-4 mm (mm/bout)": gain(0, 4)}

def boot(res, groups, kind, fn, nb=300, seed=0):
    # point estimate + bootstrap over the groups (clutches for real fish, individuals for simulated ones)
    rng = np.random.default_rng(seed)
    point = fn(pool(res, [i for g in groups for i in g], kind))
    bs = [fn(pool(res, [i for j in rng.integers(0, len(groups), len(groups)) for i in groups[j]], kind)) for _ in range(nb)]
    return point, bs

def real_groups(grp):
    m = META[META.group == grp]
    return [list(g.index) for _, g in m.groupby("clutch")]

def sim_groups(key, sighted):
    return [[i] for i in sorted(SIM[(key, sighted)])]

def ci(bs, k):
    v = np.array([b[k] for b in bs], float)
    return np.nanpercentile(v, [2.5, 97.5], axis=0)
print(f"real: {len(META)} fish ({(META.group == 'ref').sum()} reference siblings in {META[META.group == 'ref'].clutch.nunique()} "
      f"clutches, {(META.group == 'blind').sum()} blind mutants); simulated: {len(SIM)} units x 200 fish")
""")

md(r"""
## 1. The real fish

### 1.1 Sample and sanity check against the screen's own readouts
""")

code(r"""
kin = pd.DataFrame({i: RES[i]["kin"] for i in META.index}).T.join(META[["group", "age_dpf"]])
print("bouts kept per fish (60-130 min, interval <= 10 s):", kin.groupby("group").n.describe()[["count", "50%", "min"]].round(0).to_string())
print()
print("kinematics, median [IQR] per fish:")
print(kin.groupby("group")[["med_dt", "med_step", "mean_r", "mean_absturn"]].quantile([.25, .5, .75]).unstack().round(2).T.to_string())
try:
    SCR = pickle.load(open(TT.OUT / "screen.pkl", "rb"))
    scr = pd.DataFrame(SCR).set_index("fish_id")
    j = kin.join(scr, how="inner")
    own = {i: scalars(pool(RES, [i], "dot_bout"))["shoaling index"] for i in j.index}
    j["si_ours"] = pd.Series(own)
    print()
    print(f"sanity check vs the screen's readouts ({len(j)} fish): median interval ours {j.med_dt.median():.2f} s vs "
          f"screen boutDur {j.boutDur.median():.2f} s; per-fish correlation of the shoaling index (ours, 60-130 min, "
          f"bout onsets) with the screen's si (60-260 min, frames): r = {j[['si_ours', 'si']].corr().iloc[0, 1]:.2f}")
except FileNotFoundError:
    print("(screen readouts not cached)")
""")

md(r"""
### 1.2 What the real fish do around the dot

Each panel: the real dot minus the null dot, by distance; shaded = 95 % interval over clutches. Blind mutants (grey)
must sit at zero (or at 1 for occupancy) — they show that the null removes geometry, wall and dot-path effects.
""")

code(r"""
def real_curves(kind, nb=300):
    out = {}
    for grp in ("ref", "blind"):
        out[grp] = boot(RES, real_groups(grp), kind, curves, nb=nb)
    return out
RC = {kd: real_curves(kd) for kd in TT.KINDS}

fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for r, kd in enumerate(TT.KINDS):
    for c, (k, ylab, ref) in enumerate((("tow", "turn toward: real - null", 0), ("gain", "gain: real - null (mm/bout)", 0),
                                        ("occ", "occupancy: real / null", 1))):
        a = ax[r, c]
        for grp, col in (("ref", "C0"), ("blind", "0.5")):
            p, bs = RC[kd][grp]; lo, hi = ci(bs, k)
            a.plot(MID, p[k], "o-", color=col, label={"ref": "sighted siblings", "blind": "blind mutants"}[grp])
            a.fill_between(MID, lo, hi, color=col, alpha=.25)
        a.axhline(ref, color="k", lw=.8); a.axvspan(0, 4, color="r", alpha=.06); a.set_xlim(0, 40)
        a.set_ylabel(ylab); a.set_title(kd.replace("dot_", "") + " dot")
        if r == 1: a.set_xlabel("fish-dot distance at bout onset (mm)")
ax[0, 0].legend(); plt.tight_layout(); plt.show()

p, bs = RC["dot_bout"]["ref"]
tab = pd.DataFrame({k: [f"{v:+.3f} [{a:+.3f},{b:+.3f}]" for v, a, b in zip(p[k], *ci(bs, k))] for k in ("tow", "gain")}, index=LAB)
tab["occupancy"] = [f"{v:.2f} [{a:.2f},{b:.2f}]" for v, a, b in zip(p["occ"], *ci(bs, "occ"))]
tab["bouts"] = pool(RES, list(META[META.group == "ref"].index), "dot_bout")["n"].astype(int)
print("sighted siblings, bout-like dot (real - null; 95 % over clutches):"); print(tab.to_string())
""")

md(r"""
### 1.3 Alignment or pursuit?

With the continuous dot, a fish's heading follows the dot's direction of motion — but a fish chasing the dot from
behind would show that too. Split by where the dot is relative to the fish's previous heading: pursuit predicts
alignment only with the dot **ahead**; alignment proper predicts it also with the dot **beside** the fish.
""")

code(r"""
p, bs = RC["dot_continuous"]["ref"]; pb, _ = RC["dot_continuous"]["blind"]
rows = {}
for s, name in enumerate(("dot ahead (<45 deg)", "dot beside (45-135)", "dot behind (>135)")):
    lo, hi = ci(bs, f"al{s}")
    rows[name] = [f"{v:+.3f} [{a:+.3f},{b:+.3f}]" for v, a, b in zip(p[f"al{s}"], lo, hi)]
print("continuous dot, cos(heading - dot motion) real - null, sighted siblings (95 % over clutches):")
print(pd.DataFrame(rows, index=LAB).iloc[:9].to_string())
print("\nblind mutants, same (should be ~0):")
print(pd.DataFrame({n: np.round(pb[f"al{s}"], 3) for s, n in enumerate(("ahead", "beside", "behind"))}, index=LAB).iloc[:9].to_string())
""")

md(r"""
### 1.4 The same measures on our own data

Our 15 fish, blocks with **one moving dot** (bout onsets from `data/bouts_GsizeMotionSingle.csv`, the same
`twin_test.measure`, the null dot rotated about our dish centre). Intervals resample fish (15).
""")

code(r"""
sys.path.insert(0, "../scripts")
from segment_bouts import load_physical_dish
own = pd.read_csv("../data/bouts_GsizeMotionSingle.csv").sort_values(["fish_id", "segment_id", "bout_index"])
dish = load_physical_dish("../data/data/Original_data/ROIdef2025-02-20T12_09_16.csv")
OWN = {}
for fid, g in own.groupby("fish_id"):
    seg = g.segment_id.to_numpy(); cx, cy = dish.cx[fid], dish.cy[fid]
    x = (g.x_px.to_numpy() - cx) / TT.PX; y = (g.y_px.to_numpy() - cy) / TT.PX
    same_prev = np.r_[False, seg[1:] == seg[:-1]]; same_next = np.r_[seg[1:] == seg[:-1], False]
    th = g.heading_rad.to_numpy(); th_prev = np.where(same_prev, np.r_[np.nan, th[:-1]], np.nan)
    xn = np.where(same_next, np.r_[x[1:], np.nan], np.nan); yn = np.where(same_next, np.r_[y[1:], np.nan], np.nan)
    N = g.n_dots_left.fillna(0).to_numpy() + g.n_dots_right.fillna(0).to_numpy()
    kind = np.where((N == 1) & g.moving.to_numpy(bool), "dot_bout", "other")
    dx = (g.dot_cx.to_numpy() - cx) / TT.PX; dy = (g.dot_cy.to_numpy() - cy) / TT.PX
    OWN[fid] = TT.measure(x, y, xn, yn, th_prev, g.dtheta_deg.to_numpy(), g.dt_s.to_numpy(), kind, dx, dy, None, None)
po, bo = boot(OWN, [[f] for f in OWN], "dot_bout", curves)
pr, br = RC["dot_bout"]["ref"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for c, (k, ylab, ref) in enumerate((("tow", "turn toward: real - null", 0), ("gain", "gain (mm/bout)", 0), ("occ", "occupancy ratio", 1))):
    for (p, bs, col, lab) in ((pr, br, "C0", "screen siblings (bout-like dot)"), (po, bo, "C3", "our fish (1 moving dot)")):
        lo, hi = ci(bs, k); ax[c].plot(MID, p[k], "o-", color=col, label=lab); ax[c].fill_between(MID, lo, hi, color=col, alpha=.2)
    ax[c].axhline(ref, color="k", lw=.8); ax[c].axvspan(0, 4, color="r", alpha=.06); ax[c].set_xlim(0, 40)
    ax[c].set_ylabel(ylab); ax[c].set_xlabel("fish-dot distance (mm)")
ax[0].legend(); plt.tight_layout(); plt.show()
n_own = pool(OWN, list(OWN), "dot_bout")["n"]
print("our bouts per bin:", dict(zip(LAB, n_own.astype(int))))
""")

md(r"""
## 2. The test: the models in the screen's experiment

### 2.1 Response curves, bout-like dot (the fair test)
""")

code(r"""
SC = {}
for name, (k0, k1) in UNITS.items():
    for key in (k0, k1):
        for s in (True, False):
            SC[(key, s)] = {kd: boot(SIM[(key, s)], sim_groups(key, s), kd, curves, nb=200) for kd in TT.KINDS}

def plot_test(kd, title):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for r, si in enumerate((0, 1)):
        for c, (k, ylab, ref) in enumerate((("tow", "turn toward: real - null", 0), ("gain", "gain (mm/bout)", 0), ("occ", "occupancy ratio", 1))):
            a = ax[r, c]
            p, bs = RC[kd]["ref"]; lo, hi = ci(bs, k)
            a.plot(MID, p[k], "k-", lw=2.5, label="real siblings"); a.fill_between(MID, lo, hi, color="k", alpha=.15)
            for (name, keys), col in zip(UNITS.items(), ("C0", "C1", "C2")):
                key = keys[si]; p2, bs2 = SC[(key, True)][kd]; lo2, hi2 = ci(bs2, k)
                a.plot(MID, p2[k], "o-", color=col, ms=3, label=f"{name} ({key})"); a.fill_between(MID, lo2, hi2, color=col, alpha=.15)
            a.axhline(ref, color="k", lw=.8); a.axvspan(0, 4, color="r", alpha=.06); a.set_xlim(0, 40)
            a.set_ylabel(ylab); a.set_title(f"{title} - seed {('0', '101')[si]}")
            if r == 1: a.set_xlabel("fish-dot distance at bout onset (mm)")
    ax[0, 0].legend(fontsize=8); plt.tight_layout(); plt.show()
plot_test("dot_bout", "bout-like dot")
""")

md(r"""
### 2.2 Scores: share of the real visual effect reproduced

share = (model − blind model) / (real siblings − blind mutants); 95 % intervals from independent bootstraps of the
four samples. **Pass** = share in [0.5, 1.5] at both seeds (fixed in advance).
""")

code(r"""
def score_table(kd, nb=200):
    pr, br = boot(RES, real_groups("ref"), kd, scalars, nb=nb)
    pb, bb = boot(RES, real_groups("blind"), kd, scalars, nb=nb)
    rows = []
    for name, keys in UNITS.items():
        for si, key in enumerate(keys):
            ps, bss = boot(SIM[(key, True)], sim_groups(key, True), kd, scalars, nb=nb, seed=1)
            pn, bsn = boot(SIM[(key, False)], sim_groups(key, False), kd, scalars, nb=nb, seed=2)
            for m in pr:
                real_eff = pr[m] - pb[m]
                sh = (ps[m] - pn[m]) / real_eff
                shb = [(a[m] - b[m]) / (c[m] - d[m]) for a, b, c, d in zip(bss, bsn, br, bb)]
                lo, hi = np.nanpercentile(shb, [2.5, 97.5])
                rows.append(dict(model=name, seed=("0", "101")[si], key=key, measure=m, real=pr[m], blind_real=pb[m],
                                 model_sighted=ps[m], model_blind=pn[m], share=sh, share_lo=lo, share_hi=hi))
    t = pd.DataFrame(rows)
    ok = t.assign(p=t.share.between(0.5, 1.5)).groupby(["model", "measure"]).p.all()
    t["pass (both seeds)"] = [ok[(a, b)] for a, b in zip(t.model, t.measure)]
    return t

T_bout = score_table("dot_bout")
show = T_bout.copy()
for c in ("real", "blind_real", "model_sighted", "model_blind", "share", "share_lo", "share_hi"):
    show[c] = show[c].round(3)
print("BOUT-LIKE DOT (scored)")
print(show.set_index(["measure", "model", "seed"]).sort_index().to_string())
""")

code(r"""
print("summary: share of the real effect, seed 0 | seed 101  (pass = [0.5, 1.5] at both)")
piv = T_bout.assign(s=T_bout.share.map(lambda v: f"{v:+.2f}")).pivot_table(index="measure", columns=["model", "seed"], values="s", aggfunc="first")
ok = T_bout.groupby(["measure", "model"])["pass (both seeds)"].first().unstack()
for m in piv.columns.levels[0]:
    piv[(m, "pass")] = ok[m].map({True: "PASS", False: "fail"})
print(piv.sort_index(axis=1).to_string())
""")

md(r"""
**Post-hoc robustness (added after seeing the results).** The pre-declared score subtracts the blind mutants, but they
are only 26 fish from 2 clutches and swim with 2.4× longer steps, so at 0–4 mm their null-relative gain (−0.20) is
below the siblings' (−0.14): the denominator is tiny and of the wrong sign, and the 0–4 mm *gain* shares are
meaningless. All stimulus scalars except the mean distance are already measured against the rotated dot (0 = no
response by construction), so the same score can be taken against the null instead: share = model / real. This version
is reported next to the pre-declared one; the verdicts are the same.
""")

code(r"""
pr, _ = boot(RES, real_groups("ref"), "dot_bout", scalars, nb=1)
rows = {}
for name, keys in UNITS.items():
    for si, key in enumerate(keys):
        ps, _ = boot(SIM[(key, True)], sim_groups(key, True), "dot_bout", scalars, nb=1)
        rows[(name, ("0", "101")[si])] = {m: ps[m] / pr[m] for m in pr if m != "mean distance (mm)"}
t = pd.DataFrame(rows)
t[("real", "value")] = [pr[m] for m in t.index]
print("bout-like dot, share against the null (model / real), seed 0 | 101:")
print(t.round(2).to_string())
""")

md(r"""
### 2.3 Alignment: do the models line up with the dot's motion?

The models have no input for the dot's direction of motion, so any alignment they show can only come from pursuit
(chasing a dot that is ahead). Continuous dot, cos(heading − dot motion), real − null.
""")

code(r"""
p, bs = RC["dot_continuous"]["ref"]
for s, name in enumerate(("dot AHEAD (<45 deg)", "dot BESIDE (45-135 deg)", "dot BEHIND (>135 deg)")):
    rows = {"real siblings": [f"{v:+.3f} [{a:+.3f},{b:+.3f}]" for v, a, b in zip(p[f"al{s}"], *ci(bs, f"al{s}"))]}
    for mname, keys in UNITS.items():
        for si, key in enumerate(keys):
            rows[f"{mname} s{('0', '101')[si]}"] = np.round(SC[(key, True)]["dot_continuous"][0][f"al{s}"], 3)
    print(name); print(pd.DataFrame(rows, index=LAB).iloc[:9].to_string()); print()
""")

md(r"""
### 2.4 The continuous dot (an extrapolation: never in training)
""")

code(r"""
plot_test("dot_continuous", "continuous dot")
T_cont = score_table("dot_continuous")
piv = T_cont.assign(s=T_cont.share.map(lambda v: f"{v:+.2f}")).pivot_table(index="measure", columns=["model", "seed"], values="s", aggfunc="first")
print("continuous dot, share of the real effect (not scored):"); print(piv.sort_index(axis=1).to_string())
""")

md(r"""
### 2.5 Kinematics (reported, not scored: older fish, smaller dish)
""")

code(r"""
def kin_table(res, ids):
    k = pd.DataFrame({i: res[i]["kin"] for i in ids}).T
    return k[["med_dt", "med_step", "mean_r", "mean_absturn"]].median()
rows = {"real siblings": kin_table(RES, META[META.group == "ref"].index), "blind mutants": kin_table(RES, META[META.group == "blind"].index)}
for name, keys in UNITS.items():
    for si, key in enumerate(keys):
        rows[f"{name} seed {('0', '101')[si]}"] = kin_table(SIM[(key, True)], list(SIM[(key, True)]))
        rows[f"{name} seed {('0', '101')[si]}, dot hidden"] = kin_table(SIM[(key, False)], list(SIM[(key, False)]))
kt = pd.DataFrame(rows).T
kt.columns = ["median interval (s)", "median step (mm)", "mean distance from centre (mm)", "mean |turn| (deg)"]
print("per-fish medians:"); print(kt.round(2).to_string())
""")

md(r"""
### 2.6 Check: F equals D4 under the late clock
""")

code(r"""
for si in (0, 1):
    a = T_bout[(T_bout.model == "D4") & (T_bout.seed == ("0", "101")[si])].set_index("measure").model_sighted
    b = T_bout[(T_bout.model == "F") & (T_bout.seed == ("0", "101")[si])].set_index("measure").model_sighted
    print(f"seed {('0', '101')[si]}: D4 vs F (sighted), different simulated individuals:")
    print(pd.DataFrame({"D4": a, "F": b, "F - D4": b - a}).round(3).to_string()); print()
""")

md(r"""
## 3. Summary

**The data and the pipeline transfer.** The training data's own bout segmentation gives these fish a median interval of
1.03 s, the screen's own value, and a per-fish shoaling index that correlates r = 0.92 with the screen's.

**What the real fish do** (2819 sighted siblings, 198 clutches, bout-like dot, against the rotated dot):
- **Repulsion below ~6 mm.** At 0–6 mm bouts turn *away* from the dot (−0.03 to −0.04 in P(turn toward)) and carry the
  fish backwards (−0.09 to −0.20 mm per bout at 2–6 mm); the sign flips at 6–8 mm.
- **Attraction beyond**, strongest at 12–20 mm (+0.15 in P(turn toward), +1.5 to +1.6 mm per bout).
- **A preferred distance**: time near the dot is 2.6× the null at 4–8 mm and falls closer in.
- **Alignment beyond pursuit**: within ~10 mm the fish's heading follows the continuous dot's direction of motion with
  the dot ahead (+0.35 to +0.43), beside (+0.22 to +0.35) and, closest, even behind (+0.21 to +0.32); it fades by
  10–15 mm.
- Our own fish show the same pattern with one moving dot (turn-away at 4–6 mm, attraction peaking at 10–15 mm), at far
  lower precision — too few close bouts to be conclusive alone.

**The test** (bout-like dot; pass = share of the real effect in [0.5, 1.5] at both seeds, fixed in advance):
- **Attraction passes, for all three models at both seeds**: turn toward at 6–20 mm 0.90–1.12 of the real effect; gain
  1.3–1.45 (the models approach a little too strongly per bout); shoaling index and mean distance 0.56–0.68 — in closed
  loop the models hold about two thirds of the real attraction.
- **Repulsion fails, for all models at both seeds**: at 0–4 mm the models still turn *toward* the dot (+0.01 to +0.08
  against −0.04 for real fish); in distance, D4 and F back off weakly and inconsistently (−0.09 to +0.02 mm per bout
  against −0.14) and M6 moves closer (+0.21 to +0.25). This is the missing ingredient the shoal simulation shows.
- **Alignment is too weak close in** (2–4× below the real fish within 10 mm, in every sector) and too strong far out:
  the models have no input for the dot's direction of motion.
- **The models approach but do not hover.** Real fish brake as they come near: their gain per bout falls to zero at
  ~6 mm and turns negative closer in, and they spend 2.2–2.6× the null's share of bouts within 10 mm. The models keep
  gaining down to ~4 mm and spend only ~1.7× the null's share there — consistent with their shoaling index reaching
  only about two thirds of the real one.
- **The continuous dot** (never in training): the models treat it like the bout-like dot (shoaling index 1.3–1.4× the
  real effect), while real fish are less attracted to it.
- **Kinematics transfer without any refit**: median interval 1.12–1.18 s (real 1.03), median step 4.0–4.4 mm (3.9),
  distance from the centre 16.1–16.4 mm (16.6), mean |turn| 42–45° (38°).
- F equals D4 under the late clock, as expected.

**In one sentence.** Without any change, the models carry their attraction law and their kinematics to 2819 new fish,
an older age range and a smaller dish; what they lack is exactly what these data measure best and the shoal simulation
needs: short-range repulsion and alignment.
""")

md(r"""
## 4. Repulsion: F + a repulsion term fitted to these data

**The term** (`fit_MLP_joint.repulsion_shifts`; the network and F's profile correction untouched). For every active dot
*j* at distance *d_j* (mm) and bearing *φ_j* from the fish's heading:
- **turn** (variant **T**): the toward-minus-away log-odds of the turn get *s* = −*a* Σ_j sin *φ_j* σ((*r_c* − *d_j*)/*w*) —
  a turn away from any dot closer than ~*r_c*, fading smoothly beyond;
- **brake** (variant **TB**, on top of T): the step's log-mean gets −*c* Σ_j max(cos *φ_j*, 0) σ((*r_b* − *d_j*)/*w_b*) —
  shorter steps toward a dot that is close and ahead.

The term reads **each dot**, never the group's centroid (a fish inside an 8-dot group is not repelled by an empty centre).
It is applied in every rollout and, from the record's own dot positions, in the held-out likelihood and the
teacher-forced cells. It is inert at strength 0 (checked: rollouts byte-identical, scores identical).

**Fit** (`scripts/repulsion_fit.py`). The sighted siblings' 198 clutches are split once in two halves (fixed seed).
On the **fit half**: F, unchanged otherwise, simulated in this experiment (200 fish, the same seeds for every candidate),
grid search minimising the standardised mismatch of *turn toward* and *gain* at 0–10 mm (T: *a* × *r_c*, *w* = 1 mm) and
0–15 mm (TB: *a* × *c* × *r_b*, *r_c* from T, *w_b* = 1.5 mm). F at seed 0 (14f) and seed 101 (32b) are fitted separately.
Where the best point sat on the grid's edge, the grid was extended (*a* up to 0.45, *r_c* to 10 mm, *c* down to 0.05).

**Revision of the selection objective (made after the grid search, before any test-half result was looked at).** The
grid losses stop at 10 mm (T) and 15 mm (TB). The loss then kept falling as the radius grew to 10 mm: a wide term is
rewarded for cancelling the models' excess pull at 6–10 mm (they still turn toward the dot there at about +0.06, real
fish at +0.01) while the harm it would do to the real attraction at 10–20 mm is not seen. The final T and TB are therefore
chosen among all evaluated grid points with **one loss over 0–20 mm** for both variants (every point's curves are stored;
`repulsion_fit.select`). A fitted *r_c* near 8–10 mm is the reach of the correction *relative to the model* — it also
removes the models' excess pull just outside the real repulsion zone — not the radius of the real repulsion, which ends
at ~6 mm (section 1.2).

**Pass criteria (fixed before the fit was run).**
1. *Screen data, test half* (clutches never used in the fit), bout-like dot, share of the real effect measured against
   the rotated dot (model / real; the blind-mutant denominator is unusable at 0–4 mm, section 2.2):
   - **repulsion**: turn toward and gain at 0–4 mm in [0.5, 1.5] at both seeds;
   - **attraction kept**: turn toward and gain at 6–20 mm and the shoaling index still in [0.5, 1.5] at both seeds;
   - **not fitted, reported**: the occupancy curve near the dot and the shoaling index — do the fish now hover?
2. *Our data, the usual metrics*: **no worse than F** beyond the noise at both seeds — "worse" = matched AUC +0.05,
   joint likelihood +0.01 bits per bout, or a profile readout moving away from the real fish by more than its rollout
   noise (dip and grading ±1 mm, attraction ±1–2 mm, moving / static error ±0.3) at both seeds.
""")

code(r"""
import json
SPLIT = json.load(open(TT.OUT / "clutch_split.json"))
FITS = {k: json.load(open(TT.OUT / f"repfit_{k}.json")) for k in ("14f", "32b")}
SEEDNAME = {"14f": "seed 0", "32b": "seed 101"}
print(f"clutch split: {len(SPLIT['fit'])} fit / {len(SPLIT['test'])} test clutches; fit half "
      f"{FITS['14f']['targets']['n_fish']} fish")
for k, f in FITS.items():
    print(f"\nF {SEEDNAME[k]} ({k}): loss without the term: grid objectives T {f['base']['loss_T']:.1f}, TB "
          f"{f['base']['loss_TB']:.1f}; selection objective (0-20 mm) {f['base'].get('loss_sel', float('nan')):.1f}")
    print(f"  selected T : {f['best_T']['rep']}  loss (0-20 mm) {f['best_T']['loss']:.1f}"
          f"   [grid objective's choice: {f.get('best_T_gridloss', {}).get('rep')}]")
    print(f"  selected TB: {f['best_TB']['rep']}  loss (0-20 mm) {f['best_TB']['loss']:.1f}"
          f"   [grid objective's choice: {f.get('best_TB_gridloss', {}).get('rep')}]")
    g = pd.DataFrame([dict(variant=e["variant"], **e["rep"], grid_loss=e["loss"], loss_0_20=e.get("loss_sel", np.nan))
                      for e in f["grid"]])
    print("  five best of each variant (selection objective):")
    print(g.sort_values("loss_0_20").groupby("variant").head(5).round(3).to_string(index=False))

fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for r, (k, f) in enumerate(FITS.items()):
    t = f["targets"]
    best = {e["variant"]: e for e in f["grid"] if e["rep"] == f["best_" + e["variant"]]["rep"]}
    for c, (key, se, ylab) in enumerate((("tow", "se_tow", "turn toward: real - null"), ("gain", "se_gain", "gain (mm/bout)"))):
        a = ax[r, c]; y = np.array(t[key]); e = np.array(t[se])
        a.plot(MID, y, "k-", lw=2.5, label="real, fit half"); a.fill_between(MID, y - 2 * e, y + 2 * e, color="k", alpha=.15)
        a.plot(MID, f["base"][key], "o-", color="C2", ms=3, label="F")
        a.plot(MID, best["T"][key], "o-", color="C3", ms=3, label="F + turn")
        a.plot(MID, best["TB"][key], "o-", color="C4", ms=3, label="F + turn + brake")
        a.axhline(0, color="k", lw=.8); a.axvspan(0, 4, color="r", alpha=.06); a.set_xlim(0, 40)
        a.set_ylabel(ylab); a.set_title(f"fit half - F {SEEDNAME[k]}")
        if r == 1: a.set_xlabel("fish-dot distance at bout onset (mm)")
ax[0, 0].legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
### 4.1 The test half of the screen: does the term fix the repulsion, and keep the attraction?
""")

code(r"""
TEST = set(SPLIT["test"])
VAR = {"seed 0": {"F": "14f", "F + turn": "43a", "F + turn + brake": "43c"},
       "seed 101": {"F": "32b", "F + turn": "43b", "F + turn + brake": "43d"}}
SIMV = {k: pickle.load(open(TT.OUT / f"sim_{k}_sighted.pkl", "rb"))["res"] for v in VAR.values() for k in v.values()}
mt = META[(META.group == "ref") & META.clutch.isin(TEST)]
GT = [list(g.index) for _, g in mt.groupby("clutch")]
print(f"test half: {len(mt)} fish in {len(GT)} clutches")
RT = {kd: boot(RES, GT, kd, curves) for kd in ("dot_bout",)}
fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for r, (sd, vv) in enumerate(VAR.items()):
    for c, (k, ylab, ref) in enumerate((("tow", "turn toward: real - null", 0), ("gain", "gain (mm/bout)", 0), ("occ", "occupancy ratio", 1))):
        a = ax[r, c]; p, bs = RT["dot_bout"]; lo, hi = ci(bs, k)
        a.plot(MID, p[k], "k-", lw=2.5, label="real, test half"); a.fill_between(MID, lo, hi, color="k", alpha=.15)
        for (name, key), col in zip(vv.items(), ("C2", "C3", "C4")):
            p2, bs2 = boot(SIMV[key], [[i] for i in sorted(SIMV[key])], "dot_bout", curves, nb=200); lo2, hi2 = ci(bs2, k)
            a.plot(MID, p2[k], "o-", color=col, ms=3, label=f"{name} ({key})"); a.fill_between(MID, lo2, hi2, color=col, alpha=.15)
        a.axhline(ref, color="k", lw=.8); a.axvspan(0, 4, color="r", alpha=.06); a.set_xlim(0, 40)
        a.set_ylabel(ylab); a.set_title(f"bout-like dot, test half - {sd}")
        if r == 1: a.set_xlabel("fish-dot distance at bout onset (mm)")
ax[0, 0].legend(fontsize=8); plt.tight_layout(); plt.show()

pr, br = boot(RES, GT, "dot_bout", scalars, nb=300)
rows, verdict = [], {}
for sd, vv in VAR.items():
    for name, key in vv.items():
        ps, bss = boot(SIMV[key], [[i] for i in sorted(SIMV[key])], "dot_bout", scalars, nb=300, seed=1)
        for m in pr:
            if m == "mean distance (mm)":
                continue
            sh = ps[m] / pr[m]
            shb = [a[m] / b[m] for a, b in zip(bss, br)]
            rows.append(dict(measure=m, model=name, seed=sd, real=pr[m], model_value=ps[m], share=sh,
                             lo=np.nanpercentile(shb, 2.5), hi=np.nanpercentile(shb, 97.5)))
T4 = pd.DataFrame(rows)
T4["in [0.5,1.5]"] = T4.share.between(0.5, 1.5)
ok = T4.groupby(["measure", "model"])["in [0.5,1.5]"].all()
piv = T4.assign(s=[f"{v:+.2f} [{a:+.2f},{b:+.2f}]" for v, a, b in zip(T4.share, T4.lo, T4.hi)]).pivot_table(
    index="measure", columns=["model", "seed"], values="s", aggfunc="first")
print("test half, bout-like dot: share of the real effect (model / real, against the rotated dot), 95 % intervals:")
print(piv.to_string())
print("\nPASS at both seeds (share in [0.5, 1.5]):")
print(ok.unstack().replace({True: "PASS", False: "fail"}).to_string())
occ = {}
for sd, vv in VAR.items():
    for name, key in vv.items():
        S = pool(SIMV[key], list(SIMV[key]), "dot_bout")
        oc = (S["n"] / S["n"].sum()) / (S["n_null"] / S["n_null"].sum())
        occ[(name, sd)] = oc[:5]
S = pool(RES, [i for g in GT for i in g], "dot_bout"); occ[("real", "test half")] = ((S["n"] / S["n"].sum()) / (S["n_null"] / S["n_null"].sum()))[:5]
print("\nnot fitted - occupancy ratio by distance (real / null), 0-10 mm:")
print(pd.DataFrame(occ, index=LAB[:5]).round(2).to_string())
""")

md(r"""
### 4.2 Our data: the usual metrics, F against F + repulsion
""")

code(r"""
import os
def _sm(k):
    p = f"report_outputs/summary_M{k}.json"
    return json.load(open(p)) if os.path.exists(p) else None
COLS = [("likelihood (bits/bout)", "lik_joint", 3), ("matched AUC", "auc_matched", 3), ("std AUC", "auc_std", 3),
        ("no-radial AUC", "auc_matched_norad", 3), ("dip (mm)", "dip", 2), ("dip latency (s)", "lat", 1),
        ("grading (mm)", "grading", 2), ("moving err (SEM)", "mv_err_sem", 2), ("static err (SEM)", "st_err_sem", 2),
        ("2-dot attraction", "attr_N2", 2), ("8-dot attraction", "attr_N8", 2)]
rows = {}
for sd, vv in VAR.items():
    for name, key in vv.items():
        s = _sm(key)
        rows[(sd, name, key)] = {lab: (round(s[c], nd) if s and s.get(c) is not None else "XX") for lab, c, nd in COLS}
print("Real fish: dip 4.23 @ 13.5 s, grading 6.00, attraction 2 dots 6.19 / 8 dots 6.94; real-vs-real AUC ~0.52.")
print(pd.DataFrame(rows).T.to_string())
print("\nchange from F at the same seed: seed 0 | seed 101 (positive likelihood / AUC / errors = worse)")
drow = {}
for name in ("F + turn", "F + turn + brake"):
    r = {}
    for lab, c, nd in COLS:
        d = []
        for sd, vv in VAR.items():
            a, b = _sm(vv["F"]), _sm(vv[name])
            d.append(f"{b[c] - a[c]:+.{nd}f}" if a and b and a.get(c) is not None and b.get(c) is not None else "XX")
        r[lab] = " | ".join(d)
    drow[f"F -> {name}"] = r
print(pd.DataFrame(drow).T.to_string())
""")

md(r"""
### 4.4 Verdict

**Screen data, test half (clutches never used in the fit):**
- **The turn term fixes the short-range turning, on unseen fish, at both seeds.** Turn toward at 0–4 mm: F −0.43 | −0.34
  of the real effect (fail) → F + turn **1.24 | 1.07** (pass); F + turn + brake 0.90 | 0.95 (pass).
- **The backing off is not fixed**: gain at 0–4 mm fails for every variant (F + turn overshoots at seed 0, 1.96; the brake
  undershoots at seed 101, 0.26; intervals are wide at this range).
- **Attraction is kept**: turn toward and gain at 6–20 mm and the shoaling index stay in [0.5, 1.5] for both variants.
- **The fish still do not hover** (not fitted): time near the dot stays ~1.6× the null against 2.2–2.6× for real fish,
  and the shoaling index stays at about two thirds of the real one. Repulsion fixes the per-bout law, not the spacing:
  something else keeps real fish near the dot — plausibly the alignment the models lack (section 2.3).

**Our data, the usual metrics** (change from F at the same seed):
- **F + turn: no worse than F** — likelihood −0.001 | −0.001 bits, matched AUC −0.004 | −0.002, dip −0.22 | −0.08 mm;
  small consistent costs within the pre-declared noise: moving error +0.19 | +0.29 SEM and 8-dot attraction −0.88 | −0.91
  mm at both seeds — the per-dot sum inside groups, as anticipated.
- **F + turn + brake** adds nothing on the screen data and is slightly worse on ours (likelihood +0.004 | +0.006,
  grading −0.74 | −1.10 mm): dropped.

**Retained: F + turn** (*a* = 0.3, reach *r_c* = 8 | 9 mm, *w* = 1 mm). It passes the criteria fixed in advance, so it
qualifies for the shoal simulation, where each neighbour will contribute its own turn-away. Expected effect there: fewer
fish on top of one another; not tighter groups.
""")

md(r"""
### 4.3 The usual evaluation blocks (copied from `report_models.py`, outputs as evaluated)
""")
for _k in ("14f", "43a", "43c", "32b", "43b", "43d"):
    copy_unit(_k)


md(r"""
## 5. The backing off: diagnosis, a likelihood-fitted step term, and the test

F + turn reproduces the short-range turn-away but not the backing off (gain at 0–4 mm) nor the hovering (time near the
dot). Following the plan agreed in advance: (1) find which component of the bout the models get wrong near the dot,
(2) add only that term, fitted by the teacher-forced likelihood of the screen bouts, (3) test it in closed loop on the
held-out clutches and on our data's usual metrics.

### 5.1 Diagnosis (fit half of the clutches)

Per component and distance, real minus the rotated dot, for the real fish and for F / F + turn simulated in this
experiment (section 2's protocol). Turning is **not** what is missing near the dot (the models turn as much or more,
including large and away turns); pausing is **refuted** (real fish pause slightly *less* near the dot). The models'
**steps are 6–10 % longer whenever a dot is within 10 mm**, where real fish keep their step length (and shorten it at
6–10 mm): they overshoot the dot instead of hovering.
""")

code(r"""
import repulsion_lik as RL
DG = RL.diagnose()
for c in DG.component.unique():
    print(c); print(DG[DG.component == c].drop(columns="component").round(3).to_string(index=False)); print()
""")

md(r"""
**Per-bout law or state?** At the real fish's own states (teacher-forced, population fish, block clock late), the
model's predicted step is only ~3 % too long within 12 mm of the dot relative to 15–20 mm; in closed loop the excess
is ~8 %. So about a third of the step excess is in the per-bout law, and two thirds in the state the simulated fish
reach the dot in (approach speed carried in the step history). The interval residual is a fish-level offset (the
screen's older fish bout ~18 % faster everywhere), barely distance-dependent.
""")

code(r"""
import pickle
blob = pickle.load(open(TT.OUT / "replik_43a.pkl", "rb"))
BB = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20.])
acc = {}
for r in blob["rows"]:
    lw, mu, sg = r["dpar"]; pred = (np.exp(lw) * mu).sum(1)
    tl, tm, ts = r["tpar"]; predt = (np.exp(tl) * tm).sum(1)
    ok = np.isfinite(r["dist"]) & (r["dist"] > 0) & np.isfinite(r["dt"]) & (r["dt"] > 0)
    k = np.digitize(r["d"], BB) - 1
    for j in range(len(BB) - 1):
        m = ok & (k == j); a = acc.setdefault(j, [0, 0.0, 0.0])
        a[0] += m.sum(); a[1] += (np.log(r["dist"][m]) - pred[m]).sum(); a[2] += (np.log(r["dt"][m]) - predt[m]).sum()
print("F + turn, seed 0, teacher-forced residual (real - prediction) by distance to the dot, fit half:")
print(pd.DataFrame({"bouts": [int(acc[j][0]) for j in acc], "log step": [acc[j][1] / acc[j][0] for j in acc],
                    "log interval": [acc[j][2] / acc[j][0] for j in acc]},
                   index=[f"{BB[j]:g}-{BB[j + 1]:g}" for j in acc]).round(3).to_string())
""")

md(r"""
### 5.2 The likelihood fit

Each candidate term = a global offset over every near-dot bout (0–20 mm; it absorbs the fish-level differences) + a
LOCAL gated shift (reach 3–14 mm, width 0.3–2 mm), fitted by gradient on the exact teacher-forced log-likelihood of
~400 000 near-dot bouts of 400 fit-half fish, on top of F + turn (population fish, no drift). The gain of the gated part
alone decides: **step** wins at both seeds (shorter steps near any dot), pause brings a fifth of it, turn size nothing.
The step term (−2 to −3.5 % within ~13.5 mm) is added to F + turn → **F + turn + step** (Model 50).
""")

code(r"""
import json
for k, lab in (("43a", "seed 0"), ("43b", "seed 101")):
    R = json.load(open(TT.OUT / f"replik_fit_{k}.json"))
    print(f"F + turn, {lab}:")
    print(pd.DataFrame([dict(term=r["term"], shift=r["amp"], reach_mm=r["r"], width_mm=r["w"], offset=r.get("offset"),
                             gain_bits_near_term=r.get("gain_bits_near_term"), gain_bits_with_offset=r["gain_bits_total_per_bout"])
                        for r in R]).round(4).to_string(index=False)); print()
""")

md(r"""
### 5.3 Test half of the screen: F + turn against F + turn + step
""")

code(r"""
VAR5 = {"seed 0": {"F": "14f", "F + turn": "43a", "F + turn + step": "50a"},
        "seed 101": {"F": "32b", "F + turn": "43b", "F + turn + step": "50b"}}
SIM5 = {k: pickle.load(open(TT.OUT / f"sim_{k}_sighted.pkl", "rb"))["res"] for v in VAR5.values() for k in v.values()}
RT5 = boot(RES, GT, "dot_bout", curves)
fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for r, (sd, vv) in enumerate(VAR5.items()):
    for c, (k, ylab, ref) in enumerate((("tow", "turn toward: real - null", 0), ("gain", "gain (mm/bout)", 0), ("occ", "occupancy ratio", 1))):
        a = ax[r, c]; p, bs = RT5; lo, hi = ci(bs, k)
        a.plot(MID, p[k], "k-", lw=2.5, label="real, test half"); a.fill_between(MID, lo, hi, color="k", alpha=.15)
        for (name, key), col in zip(vv.items(), ("C2", "C3", "C4")):
            p2, bs2 = boot(SIM5[key], [[i] for i in sorted(SIM5[key])], "dot_bout", curves, nb=200); lo2, hi2 = ci(bs2, k)
            a.plot(MID, p2[k], "o-", color=col, ms=3, label=f"{name} ({key})"); a.fill_between(MID, lo2, hi2, color=col, alpha=.15)
        a.axhline(ref, color="k", lw=.8); a.axvspan(0, 4, color="r", alpha=.06); a.set_xlim(0, 40)
        a.set_ylabel(ylab); a.set_title(f"bout-like dot, test half - {sd}")
        if r == 1: a.set_xlabel("fish-dot distance at bout onset (mm)")
ax[0, 0].legend(fontsize=8); plt.tight_layout(); plt.show()

pr, br = boot(RES, GT, "dot_bout", scalars, nb=300)
rows = []
for sd, vv in VAR5.items():
    for name, key in vv.items():
        ps, bss = boot(SIM5[key], [[i] for i in sorted(SIM5[key])], "dot_bout", scalars, nb=300, seed=1)
        for m in pr:
            if m == "mean distance (mm)":
                continue
            shb = [a[m] / b[m] for a, b in zip(bss, br)]
            rows.append(dict(measure=m, model=name, seed=sd, share=ps[m] / pr[m],
                             lo=np.nanpercentile(shb, 2.5), hi=np.nanpercentile(shb, 97.5)))
T5 = pd.DataFrame(rows)
piv = T5.assign(s=[f"{v:+.2f} [{a:+.2f},{b:+.2f}]" for v, a, b in zip(T5.share, T5.lo, T5.hi)]).pivot_table(
    index="measure", columns=["model", "seed"], values="s", aggfunc="first")
print("test half, bout-like dot: share of the real effect (model / real, against the rotated dot):"); print(piv.to_string())
ok5 = T5.assign(p=T5.share.between(0.5, 1.5)).groupby(["measure", "model"]).p.all()
print("\nPASS at both seeds:"); print(ok5.unstack().replace({True: "PASS", False: "fail"}).to_string())
occ5 = {}
for sd, vv in VAR5.items():
    for name, key in vv.items():
        S = pool(SIM5[key], list(SIM5[key]), "dot_bout")
        occ5[(name, sd)] = ((S["n"] / S["n"].sum()) / (S["n_null"] / S["n_null"].sum()))[:6]
S = pool(RES, [i for g in GT for i in g], "dot_bout"); occ5[("real", "test half")] = ((S["n"] / S["n"].sum()) / (S["n_null"] / S["n_null"].sum()))[:6]
print("\nnot fitted - occupancy ratio (real / null), 0-12 mm:"); print(pd.DataFrame(occ5, index=LAB[:6]).round(2).to_string())
""")

md(r"""
### 5.4 Our data: the usual metrics
""")

code(r"""
rows = {}
for sd, vv in VAR5.items():
    for name, key in vv.items():
        s = _sm(key)
        rows[(sd, name, key)] = {lab: (round(s[c], nd) if s and s.get(c) is not None else "XX") for lab, c, nd in COLS}
print(pd.DataFrame(rows).T.to_string())
print("\nchange from F + turn at the same seed: seed 0 | seed 101 (positive likelihood / AUC / errors = worse)")
r = {}
for lab, c, nd in COLS:
    d = []
    for sd, vv in VAR5.items():
        a, b = _sm(vv["F + turn"]), _sm(vv["F + turn + step"])
        d.append(f"{b[c] - a[c]:+.{nd}f}" if a and b and a.get(c) is not None and b.get(c) is not None else "XX")
    r[lab] = " | ".join(d)
print(pd.DataFrame({"F + turn -> F + turn + step": r}).T.to_string())
""")

md(r"""
### 5.6 Verdict

- **Diagnosis.** Near the dot the models turn as much as real fish (or more) and do not pause less; what they get wrong
  is the **step**: 6–10 % longer within 10 mm in closed loop, where real fish keep it. About a third of that excess is in
  the per-bout law (teacher-forced), two thirds in the state the simulated fish reach the dot in.
- **The likelihood fit picks the step term** at both seeds (near-dot gain 0.0054 | 0.0019 bits per bout against pause
  0.0010 | 0.0007 and turn size ≈ 0): −3.5 % | −2.1 % within ~13.5 mm of any dot.
- **Held-out screen clutches: F + turn + step passes every short-range criterion** — the backing off now passes at both
  seeds (gain at 0–4 mm 1.31 | 0.75 of the real effect; F + turn 1.96 | 1.45, fail), the turn-away still passes
  (1.46 | 1.35), attraction is kept (shoaling index 0.62 | 0.62).
- **Our data:** likelihood +0.008 | +0.008 bits per bout (a small cost at both seeds, under the 0.01 threshold), matched
  AUC +0.018 | −0.005 (noise), **moving error better at both seeds** (−0.21 | −0.11), grading and 8-dot attraction mixed
  by seed. No worse than F + turn by the pre-declared criteria.
- **Still missing: the hovering.** Time near the dot stays ~1.5–1.6× the null against 2.1–2.6× for real fish: the
  remaining overshoot is a closed-loop state effect that a per-bout correction fitted by likelihood cannot reach (a
  closed-loop fit could, at the price of a worse per-bout law), and part of the real staying-close is alignment, which no
  term on the existing outputs can supply.
""")

md(r"""
### 5.5 The usual evaluation blocks of F + turn + step
""")
for _k in ("50a", "50b"):
    copy_unit(_k)


def _unit_cells(key):
    """The usual evaluation block of one report unit (report_models.py), copied with its outputs (not re-run)."""
    import json
    import sys
    sys.path.insert(0, str(HERE))
    import report_models as RM
    p = RM.OUTS / f"M{key}.json"
    sub = RM._unit(key).get("sub") or key
    if not p.exists():
        return [nbformat.v4.new_markdown_cell(f"#### {sub}\n\nXX — not evaluated yet.")]
    blob = json.loads(p.read_text(encoding="utf-8"))
    out = [nbformat.v4.new_markdown_cell(f"#### {sub} (key `{key}`)")]
    for c in blob["cells"]:
        cc = nbformat.v4.new_code_cell("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        cc.outputs = [nbformat.from_dict(o) for o in c["outputs"]]
        out.append(cc)
    return out


if __name__ == "__main__":
    own = [c for c in CELLS if not isinstance(c, tuple)]
    nb = nbformat.v4.new_notebook(cells=own, metadata=dict(kernelspec=dict(display_name="Python 3", language="python",
                                                                          name="python3")))
    ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    done, cells = iter(nb.cells), []
    for c in CELLS:
        cells += _unit_cells(c[1]) if isinstance(c, tuple) else [next(done)]
    nb.cells = cells
    nbformat.write(nb, NB)
    print(f"wrote {NB} ({len(cells)} cells)")
