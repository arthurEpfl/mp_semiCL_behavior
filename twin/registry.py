"""Every model of the thesis: its name, how it is built, and where its checkpoints are.

A model is evaluated as a UNIT: five fold checkpoints at one cross-validation seed (fold k holds out three fish and is
initialised from seed + k), in ``paths.CKPTS / f"M{key}"``. The key is the unit's internal identifier (the thesis uses
the names); ``unit("M6", 0)`` gives seed 0 of M6 (key "26e"), ``unit("M6", 101)`` seed 101 ("30d").

How a unit's checkpoints are made (``make``):
  train      trained from scratch with ``recipe`` (pipelines/train.py)
  same       a copy of unit ``base``'s checkpoints, generated differently (M2 = M1 with wall rejection)
  correct    the profile correction fitted on unit ``base``, parameters at the last step (twin/correction.py)
  driftfix   unit ``base`` with the drift's generation strength fixed (twin/correction.py, ``fix_drift``)

Every unit generates with wall rejection except M1, and with per-block random streams (``block_seed``) except the
seed-0 units of M1-M3, scored on the earlier single stream.
"""
REJ = {"d0": 2.0, "d1": 10.0, "max_draws": 20}         # wall rejection at generation (mm from the wall)
W3FT = dict(w_onset=1.0, w_pursuit=3.0, onset_bouts=0, pursuit_deg=180.0, pursuit_px=60.0, heads=("turn",))


def recipe(rep, latent=2, drift=None, cdt=False, tr=False, rc=False, al=False, w3=False, st=False, lv=True,
           clock=True, K=8, Kk=3, depth=None, tsize=False, jump=False, dmove=False):
    """A training recipe (the arguments of ``training.make_model``) and its arm name, which lists the ingredients:
    stimulus code ``rep``, head order adt, z2 = two-dimensional per-fish latent, drift = AR(1) slow state (tau in
    bouts; _t200 when 200), cdt = centred interval lags, tr = 10 s truncation, rc = run clock, al = aligned-and-far
    input, w3ft = x3 turn weight on far moving-dot bouts, st = static block, nlv = no lag flag, nck = no block clock,
    hK / kkK = turn / step-and-interval history lengths, d3 = three hidden layers, tz = turn-size latent,
    jp = dot-jump phase input, dm = dot motion direction input."""
    tag = (f"{rep}_adt" + ("_z2" if latent else "") + ("_drift" if drift else "") + ("_cdt" if cdt else "")
           + ("_tr" if tr else "") + ("_rc" if rc else "") + ("_al" if al else "") + ("_w3ft" if w3 else "")
           + ("_st" if st else "") + ("" if lv else "_nlv") + ("" if clock else "_nck") + ("" if K == 8 else f"_h{K}")
           + ("" if Kk == 3 else f"_kk{Kk}") + ("_t200" if drift == 200 else "") + (f"_d{depth}" if depth else "")
           + ("_tz" if tsize else "") + ("_jp" if jump else "") + ("_dm" if dmove else ""))
    extra = {**({"kin_dt_lags": "centred"} if cdt else {}), **({"run_clock_tau": 30.0} if rc else {}),
             **({"stim_align": True} if al else {}), **({"stim_static": True} if st else {}),
             **({} if lv else {"lag_valid": ""}), **({} if clock else {"clock_mode": ""}),
             **({"stim_jump": True} if jump else {}), **({"stim_dmove": True} if dmove else {})}
    r = dict(arm=tag, rep=rep, latent=latent, extra=extra, dt_trunc=10.0 if tr else None,
             regime_w=dict(W3FT) if w3 else None, drift=dict(tau=float(drift)) if drift else None,
             hist=(K, Kk) if (K, Kk) != (8, 3) else None, depth=depth, tsize=tsize)
    return r


# the temporal set of M6 and the drift branch: centred lags + truncation, run clock, aligned-and-far, x3 turn weight
_T = dict(cdt=True, tr=True, rc=True, al=True, w3=True)
M6 = dict(rep="S4x", **_T)
D4 = dict(rep="S4", drift=70, st=True, **_T)

# the correction of F: pf-B (count-gated turn shifts), 20 s cut-off, warm starts re-anchored; 100 steps, last step
CORRECTION = dict(steps=100, B=None, M=16, lr=0.1, lam_ratio=3.0, seed=0, crn=5, warm=16, lr_halve=[50],
                  engine="batch", keep_best=20, v5="Bc", reanchor=True)

# name -> recipe (trained units) ; a dict of variants keeps the thesis names next to their ingredients
RECIPES = {
    # the main line
    "M1": recipe("S0", latent=0),
    "M3": recipe("S0"),
    "M4": recipe("S4i"),
    "M5": recipe("S4i", w3=True),
    "M5+R": recipe("S4x", w3=True),
    "M5+T": recipe("S4i", **_T),
    "M6": recipe(**M6),
    # the drift branch
    "D1": recipe("S4x", st=True, **_T),
    "D2": recipe("S4x", drift=70, st=True, **_T),
    "D3": recipe("S4", st=True, **_T),
    "D4": recipe(**D4),
    # ablations of M6
    "M6-stim": recipe("Snull", cdt=True, tr=True, rc=True, w3=True, clock=False),
    "M6-pos": recipe("Scond", cdt=True, tr=True, rc=True, w3=True),
    "M6-count": recipe("S4xnn", **_T),
    "M6-range": recipe("S4xnr", cdt=True, tr=True, rc=True, w3=True),
    "M6-motion": recipe("S4xnm", cdt=True, tr=True, rc=True, w3=True),
    "M6-turn4": recipe(**M6, K=4),
    "M6-turn3": recipe(**M6, K=3),
    "M6-turn1": recipe(**M6, K=1),
    "M6-turn0": recipe(**M6, K=0),
    "M6-kin5": recipe(**M6, Kk=5),
    "M6-kin4": recipe(**M6, Kk=4),
    "M6-kin2": recipe(**M6, Kk=2),
    "M6-kin1": recipe(**M6, Kk=1),
    "M6-kin0": recipe(**M6, Kk=0),
    "M6-block clock": recipe(**M6, clock=False),
    "M6-run clock": recipe("S4x", cdt=True, tr=True, al=True, w3=True),
    "M6-x3": recipe("S4x", cdt=True, tr=True, rc=True, al=True),
    "M6-z": recipe(**M6, latent=0),
    # negative results
    "M6, 3 layers": recipe(**M6, depth=3),
    "D4, 3 layers": recipe(**D4, depth=3),
    "M6, turn size": recipe(**M6, tsize=True),
    "D4, turn size": recipe(**D4, tsize=True),
    "D4, tau 200": recipe(**dict(D4, drift=200)),
    "D4, tau 200 + turn size": recipe(**dict(D4, drift=200), tsize=True),
    "M6, jump phase": recipe(**M6, jump=True),
    "D4, jump phase": recipe(**D4, jump=True),
    "M6, floor 2 px": recipe(**dict(M6, rep="S4xlo")),
    "D4, floor 2 px": recipe(**dict(D4, rep="S4lo")),
    # trained on the screen data set (twin/screen_train.py): the floor-2-px recipes, + the dot's direction of motion
    "M6, screen": recipe(**dict(M6, rep="S4xlo")),
    "M6 + direction, screen": recipe(**dict(M6, rep="S4xlo"), dmove=True),
    "D4, screen": recipe(**dict(D4, rep="S4lo")),
    "D4 + direction, screen": recipe(**dict(D4, rep="S4lo"), dmove=True),
}


def _u(name, s0, s101, make="train", base=None, rej=True, block_seed=(True, True), **kw):
    return {key: dict(name=name, seed=seed, key=key, make=make,
                      base=(base[i] if isinstance(base, tuple) else base), rej=rej, block_seed=block_seed[i], **kw)
            for i, (seed, key) in enumerate(((0, s0), (101, s101))) if key}


UNITS = {}
for args in [
    # the main line (M1-M3 at seed 0: the earlier single rollout stream)
    ("M1", "1", "34a", "train", None, False, (False, True)),
    ("M2", "2", "34b", "same", ("1", "34a"), True, (False, True)),
    ("M3", "3", "34c", "train", None, True, (False, True)),
    ("M4", "20f", "30a"), ("M5", "25d", "30b"), ("M5+R", "29a", "30c"), ("M5+T", "29b", "34d"), ("M6", "26e", "30d"),
    # the drift branch and the final model
    ("D1", "18a", "24a"), ("D2", "24b", "24c"), ("D3", "16e", "30e"), ("D4", "11e", "32a"),
    ("F", "14f", "32b", "correct", ("11e", "32a")),
    # ablations of M6
    ("M6-stim", "29e", "33u"), ("M6-pos", "33a", "33b"), ("M6-count", "29c", "33s"), ("M6-range", "33c", "33d"), ("M6-motion", "29d", "33t"),
    ("M6-turn4", "33e", "33f"), ("M6-turn3", "49c", "49d"), ("M6-turn1", "33g", "33h"), ("M6-turn0", "33i", "33j"),
    ("M6-kin5", "44c", "44d"), ("M6-kin4", "44a", "44b"), ("M6-kin2", "49a", "49b"), ("M6-kin1", "33k", "33l"),
    ("M6-kin0", "33m", "33n"), ("M6-block clock", "33o", "33p"), ("M6-run clock", "35a", "35b"),
    ("M6-x3", "35g", "35h"), ("M6-z", "45a", "45b"),
    # negative results (appendix)
    ("M6, 3 layers", "36a", "36b"), ("D4, 3 layers", "36c", "36d"),
    ("F, 3 layers", "36e", "36f", "correct", ("36c", "36d")),
    ("M6, turn size", "38a", "38b"), ("D4, turn size", "38c", "38d"),
    ("F, turn size", "38e", "38f", "correct", ("38c", "38d")),
    ("D4, tau 200", "37a", "37b"), ("F, tau 200", "37c", "37d", "correct", ("37a", "37b")),
    ("D4, tau 200 + turn size", "46a", "46b"),
    ("F, tau 200 + turn size", "46c", "46d", "correct", ("46a", "46b")),
    ("M6, jump phase", "41a", "41b"), ("D4, jump phase", "41c", "41d"),
    ("F, jump phase", "41e", "41f", "correct", ("41c", "41d")),
    ("M6, floor 2 px", "42a", "42b"), ("D4, floor 2 px", "42c", "42d"),
    ("F, floor 2 px", "42e", "42f", "correct", ("42c", "42d")),
]:
    UNITS.update(_u(*args))
# D4 with the drift's generation strength fixed: (strength factor of the full strength, tau 70 or 200)
DRIFT_FULL = {"dist": 0.05, "dt": 0.13}
for name, s0, s101, factor, base in [
        ("D4, strength 0", "47a", "47f", 0.0, ("11e", "32a")),
        ("D4, tau 70, half", "47b", "47g", 0.5, ("11e", "32a")),
        ("D4, tau 70, full", "47c", "47h", 1.0, ("11e", "32a")),
        ("D4, tau 200, half", "47d", "47i", 0.5, ("37a", "37b")),
        ("D4, tau 200, full", "47e", "47j", 1.0, ("37a", "37b"))]:
    UNITS.update(_u(name, s0, s101, "driftfix", base, factor=factor))
UNITS.update(_u("F on D4 at strength 0", "48a", "48b", "correct", ("47a", "47f")))

BY_NAME = {(u["name"], u["seed"]): k for k, u in UNITS.items()}


def unit(name_or_key, seed=None):
    """The unit dict of ``unit("M6", 0)`` or ``unit("26e")``."""
    key = BY_NAME[(name_or_key, seed)] if seed is not None else name_or_key
    return UNITS[str(key)]


def arm(u):
    """The arm name of a unit: the recipe's, with the correction's suffix for a corrected unit."""
    u = unit(u) if isinstance(u, str) else u
    if u["make"] == "correct":
        return arm(UNITS[u["base"]]) + "_pf5Bcr_last"
    if u["make"] in ("same", "driftfix"):
        return arm(UNITS[u["base"]])
    return RECIPES[u["name"]]["arm"]


def ckpt_dir(u):
    """The folder holding a unit's five fold checkpoints and its evaluation tables."""
    from twin import paths
    u = unit(u) if isinstance(u, str) else u
    return paths.CKPTS / f"M{u['key']}"
