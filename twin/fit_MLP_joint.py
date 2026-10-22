"""Joint-likelihood bout model: one feature vector, one body, chained heads.

WHAT IS DIFFERENT FROM ``fit_MLP_mix.MLPMixKinDensity``, and why it is worth measuring.

1. THE FACTORISATION MATCHES THE DATA. The kin model trains
   ``p(dtheta|x_ang) . p(l|x_kin) . p(tau|x_kin, l)`` -- the distance head sees no turn angle at
   all, and neither does the dt head. Here the emission is the full chain

       L = -[ log p(dtheta | x) + log p(l | x, dtheta) + log p(tau | x, dtheta, l) ]

   so each channel is conditioned on everything sampled before it within the same bout.

2. ONE FEATURE VECTOR. The kin model hard-codes ``use_kin=False`` and builds a SEPARATE 6-column
   kin trunk, so the angle body never sees kinematic history and the kin trunk never sees the
   stimulus, the border or the turn history. Here ``use_kin=True`` puts everything in one ``x``
   that a single shared body consumes -- ``MB.build_feature_matrix`` already supports this, the
   switch was simply never flipped.

3. METRIC A COVERS ALL THREE CHANNELS. ``MLPMixKinDensity`` inherits a ``score`` that evaluates
   only the angle mixture, so about two thirds of the fitted emission is optimised against a
   likelihood that is never reported. ``score`` here returns ``logdens`` (angle, same key as
   before) plus ``logdens_dist`` and ``logdens_dt``.

DELIBERATELY ABSENT: vigor, artr, kin_feedback, kin_var_comp, qmap, indiv, innov_comp. This is a
clean baseline, not the accumulated lineage. ``contain`` is available because it corrects the
radial OCCUPANCY distribution; the thing that actually keeps the fish in the dish is the
``guardrail``, which is on regardless and is not a "fix".

CONDITIONER CONVENTION. The heads receive ``|dtheta|``, not signed dtheta: left and right should
not change step size or timing by symmetry, and feeding the sign would break a symmetry the angle
mixture otherwise enforces. Every conditioner is scaled the same way the corresponding history
block is (``/dth_scale``, and centred by ``kin_scales``) so the heads see inputs on the scale they
were trained at.

HEAD ORDER (``head_order``, default ``"a,d,t"`` = the chain written above). The chain rule is exact
in all SIX orderings of (dtheta, distance, dt), so they all target the same joint density and their
summed bits/bout ARE comparable. What differs is CAPACITY: a channel emitted first is predicted
from ``x`` alone, a channel emitted last sees the other two outcomes of the same bout. Which
ordering wins is therefore an empirical question, and the EDA points at ``t`` first -- |dtheta|
scales with the time the bout spans (rho = 0.17) while distance barely does (rho = -0.01), so
"draw the interval, then turn and travel within it" may be the cheaper description.

``"a,d,t"`` is bit-identical to the pre-ordering code, RNG draw for RNG draw, and is what every
archived checkpoint reloads as. A head at position ``i`` is a ``Linear(hidden + i, ...)``; the
conditioners are appended in the ordering's own order, so position is the only thing that changes.

COMPARABILITY WARNING: because the angle body now also reads kinematic history, this model's
angle ``bits_per_bout`` is NOT comparable to the historical turn-only series from the kin zoo.
Within this family it is fine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from twin import evaluate as E                    # noqa: E402
from twin import fit_MLP as M                     # noqa: E402
from twin import fit_MLP_border as MB             # noqa: E402
from twin import fit_MLP_mix as MX                # noqa: E402

_EMPTY = np.zeros(0)                    # allocation-free stand-in for an absent feature block


# =========================================================================== #
#  Net
# =========================================================================== #
#: The three emission channels, by their one-letter code. 'a' = dtheta (angle), 'd' = distance,
#: 't' = dt (interval). An ORDER is a permutation of these three.
CHANNELS = ("a", "d", "t")
DEFAULT_ORDER = ("a", "d", "t")


def parse_order(order):
    """``"t,a,d"`` / ``"tad"`` / ``('t','a','d')`` -> ``('t','a','d')``. ``None`` -> the default.

    Rejects anything that is not a permutation of the three channels, LOUDLY: a typo that silently
    fell back to the default would run a whole sweep arm under the wrong name and the CSV would be
    indistinguishable from a real result."""
    if order is None:
        return DEFAULT_ORDER
    if not isinstance(order, str):
        order = ",".join(str(s) for s in order)
    ks = tuple(s for s in order.replace(",", " ").replace("-", " ").replace(">", " ").split() if s)
    if len(ks) == 1 and len(ks[0]) == 3:        # compact form: "tad"
        ks = tuple(ks[0])
    ks = tuple(s.strip().lower()[:1] for s in ks)
    if sorted(ks) != sorted(CHANNELS):
        raise ValueError(f"head_order must be a permutation of {CHANNELS}; got {order!r} -> {ks}")
    return ks


def order_tag(order):
    """``('t','a','d')`` -> ``'tad'``. The short label the sweep uses in file names and the CSV."""
    return "".join(parse_order(order))


class MDNMixJoint(MX.MDNMix):
    """Shared body -> three chained heads, emitted in ``order`` (default angle, distance, dt).

    ``dt_trunc`` (seconds, None = off) truncates the interval head's density at that bound -- see
    ``MX._dt_logpdf``. A plain attribute set by the trainers / loaders from cfg_feat, not a
    parameter, so archived state_dicts load unchanged.

    Each head is a ``Linear(hidden + i, ...)`` where ``i`` is its position in the chain: it reads
    the shared body plus the conditioner of every channel emitted BEFORE it, appended in the
    ordering's own order. So the first head sees the body alone and the last sees the body plus
    two within-bout outcomes.

    ``forward`` is inherited from ``MDNMix`` and returns the ANGLE parameters only, so
    ``predict_params_mix`` / ``symbol_logq_mix`` / the symbol track keep working unchanged -- but
    ONLY when the angle head is first, because otherwise those parameters are not defined without
    conditioners. It raises rather than returning a quietly-wrong plug-in. The chained heads are
    reached through ``joint_forward``.
    """

    def __init__(self, n_features, hidden, n_dist=2, n_dt=3, order=None, skip_slices=None, depth=1):
        super().__init__(n_features, hidden)
        # DEPTH (Model 17): ``depth`` hidden layers of ``hidden`` relu units. 1 = the archived
        # single layer, bit-identical (the body is not rebuilt, so the init stream is untouched).
        self.depth = int(depth)
        if self.depth > 1:
            layers = [nn.Linear(n_features, hidden), nn.ReLU()]
            for _ in range(self.depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            self.body = nn.Sequential(*layers)
        # SKIP PATH (Model 17): a linear term from the stimulus inputs (column slices of x: the
        # stimulus block and the appended align / static blocks) straight onto the turn head's
        # 14 outputs, bypassing the shared body. Zero-initialised (not a saddle: its gradient is
        # x times the output gradient), so the fit starts as the plain model.
        self.skip_slices = tuple((int(a), int(b)) for a, b in skip_slices) if skip_slices else None
        if self.skip_slices:
            self.skip = nn.Linear(sum(b - a for a, b in self.skip_slices), 14, bias=False)
            nn.init.zeros_(self.skip.weight)
        self.n_dist, self.n_dt = int(n_dist), int(n_dt)
        self.order = parse_order(order)
        self.head_pos = {c: i for i, c in enumerate(self.order)}
        # Only REPLACE the inherited angle head when it actually needs to widen. Rebuilding it
        # unconditionally would draw a fresh init from the same generator and shift every
        # subsequent parameter's random numbers, so the default order would no longer reproduce
        # the archived fits bit for bit.
        if self.head_pos["a"]:
            self.head = nn.Linear(hidden + self.head_pos["a"], 14)
        self.dist_head = nn.Linear(hidden + self.head_pos["d"], 3 * self.n_dist)
        self.dt_head = nn.Linear(hidden + self.head_pos["t"], 3 * self.n_dt)

    def forward(self, x):
        if self.head_pos["a"]:
            raise RuntimeError(
                f"forward() is the unconditional angle head, but this net emits {self.order}, so "
                "the angle parameters depend on conditioners drawn earlier in the bout. Use "
                "joint_forward(x, a_z, ld_c, lt_c) and take its first return value.")
        return super().forward(x)

    def skip_term(self, x):
        """The skip path's contribution to the turn head's outputs, plus the stimulus turn term when one is
        attached (``StimTurnTerm``); 0.0 when there is neither."""
        out = 0.0
        if getattr(self, "skip_slices", None):
            out = self.skip(torch.cat([x[:, a:b] for a, b in self.skip_slices], dim=1))
        if getattr(self, "stim_turn", None) is not None:
            out = out + self.stim_turn(x)
        rs = getattr(self, "_rep_side", None)
        if rs is not None:                                 # repulsion term (side input of the current record)
            s = rs[0]
            if s.shape[0] != x.shape[0]:
                raise RuntimeError(f"repulsion side input has {s.shape[0]} rows, the feature matrix {x.shape[0]}")
            z = torch.zeros_like(s)
            out = out + torch.stack([z, -s, -s, s, s] + [z] * 9, dim=1)
        return out

    def step_term(self, x):
        """(n, 1) shift of the step head's log-location from the profile term's v2 levers, or 0.0."""
        st = getattr(self, "stim_turn", None)
        out = 0.0 if (st is None or not getattr(st, "v2", False)) else st.step_shift(x).reshape(-1, 1)
        rs = getattr(self, "_rep_side", None)
        if rs is not None:                                 # repulsion brake (side input of the current record)
            if rs[1].shape[0] != x.shape[0]:
                raise RuntimeError(f"repulsion side input has {rs[1].shape[0]} rows, the feature matrix {x.shape[0]}")
            out = out + rs[1].reshape(-1, 1)
        return out

    def joint_forward(self, x, a_z, ld_c, lt_c=None):
        """(angle params, (logw,mu,sigma) for distance, same for dt) -- ALWAYS in that return
        order, whatever the emission order is.

        ``a_z`` = |dtheta|/dth_scale, ``ld_c`` = (log distance - dm)/ds, ``lt_c`` = (log dt - tm)/ts,
        all teacher-forced at train time. ``lt_c`` may be None only when dt is emitted LAST (the
        default), because then nothing conditions on it."""
        b = self.body(x)
        cond = {"a": a_z, "d": ld_c, "t": lt_c}
        out, prev = {}, []
        for i, ch in enumerate(self.order):
            inp = torch.cat([b] + prev, dim=1) if prev else b
            if ch == "a":
                out["a"] = self._angle_params(self.head(inp) + self.skip_term(x))
            else:
                k = self.n_dist if ch == "d" else self.n_dt
                o = (self.dist_head if ch == "d" else self.dt_head)(inp)
                if ch == "d":
                    o = torch.cat([o[:, :k], o[:, k:2 * k] + self.step_term(x), o[:, 2 * k:]], dim=1)
                out[ch] = (F.log_softmax(o[:, :k], dim=1), o[:, k:2 * k],
                           F.softplus(o[:, 2 * k:3 * k]) + M.EPS)
            if i < len(self.order) - 1:          # the last channel conditions nothing
                c = cond[ch]
                if c is None:
                    raise ValueError(f"emission order {self.order} needs the {ch!r} conditioner "
                                     f"at position {i}, but it was not supplied")
                prev.append(c.reshape(-1, 1))
        return out["a"], out["d"], out["t"]



# --------------------------------------------------------------------------------------------------------
#  STIMULUS TURN TERM (the profile fine-tune, scripts/profile_finetune.py)
# --------------------------------------------------------------------------------------------------------
# The turn head is a 5-component mixture: 0 = forward (Normal at 0), 1-2 = Gammas on dtheta < 0, 3-4 = Gammas
# on dtheta > 0. Turning TOWARD a dot group at bearing phi > 0 is dtheta > 0 (the verified toward-turn sign,
# dtheta * sign(phi) > 0), so the term is a LOG-ODDS SHIFT between turning toward and away from each group:
#     s = sum_d sin(psi_d) * mv_d * (a_g + b_g nz_d + c_g nz_d^2),   logits[3,4] += s,  logits[1,2] -= s
# psi_d = the six bearing centres (0, +-60, +-120, 180 deg); sin(psi_d) is 0 straight ahead and behind, where
# left/right is meaningless. mv_d (moving fraction) confines it to MOVING groups, so static behaviour cannot
# change. The coefficients are MIRROR-SHARED (+60 with -60, +120 with -120: fish have no handedness), so the
# whole term is 6 numbers, each readable: extra log-odds of turning toward a moving group of log-count nz at
# the side (g = 0) or behind the side (g = 1). Zero-initialised, so attaching it leaves the model unchanged.
_STIM_TURN_GROUP = {60.0: 0, -60.0: 0, 120.0: 1, -120.0: 1, 0.0: 0, 180.0: 0}


# V2 LEVERS (Model 33), present only when ``rho_idx`` is given, so Model 32's checkpoints load unchanged. They add
# the group's RANGE and the case the side term cannot reach -- a group straight AHEAD (sin 0), which is the pursuit
# regime where D4/D5 and dip_levers located the lost approach (real fish swim straighter and step further there):
#   rho_d = the group's log range, from the bump map's range profile in its bearing column: sum_j l_j P_dj / sum_j P_dj,
#           l_j = log2(knot_j / 70 px) -- negative close, positive far; 0 when the column is empty
#   turn   toward/away log-odds    += sin(psi_d) mv_d e_g rho_d                   (coef_rng: 2, mirror-shared)
#   turn   FORWARD-component logit += mv_0 (h + k rho_0)  -- straighter bouts at a group ahead   (ahead[0:2])
#   step   log-step LOCATION       += mv_0 (m + n rho_0)  -- longer steps at a group ahead       (ahead[2:4])
# 12 numbers in all, each with a plain meaning; all zero-initialised.
class StimTurnTerm(nn.Module):
    def __init__(self, mv_idx, nz_idx, rho_idx=None):
        super().__init__()
        self.mv_idx = [int(i) for i in mv_idx]; self.nz_idx = [int(i) for i in nz_idx]
        ang = np.asarray(M.POP_ANG_DEG, float)
        assert len(self.mv_idx) == len(self.nz_idx) == len(ang)
        self.register_buffer("sin", torch.tensor(np.sin(np.radians(ang)), dtype=torch.float32))
        self.register_buffer("grp", torch.tensor([_STIM_TURN_GROUP[float(a)] for a in ang], dtype=torch.long))
        self.coef = nn.Parameter(torch.zeros(2, 3))
        self.v2 = rho_idx is not None
        if self.v2:
            self.rho_idx = [[int(i) for i in row] for row in rho_idx]                   # (6, n_range)
            knots = np.asarray(M.SHOAL_RBF_PX, float)
            self.register_buffer("lk", torch.tensor(np.log2(knots / 70.0), dtype=torch.float32))
            self.coef_rng = nn.Parameter(torch.zeros(2))
            self.ahead = nn.Parameter(torch.zeros(4))

    def rho(self, x):
        """(n, 6) each direction's log range (0 where the column holds no mass)."""
        P = torch.clamp(x[:, torch.as_tensor(self.rho_idx)], min=0.0)               # (n, 6, R)
        tot = P.sum(-1)
        return torch.where(tot > 0, (P * self.lk).sum(-1) / torch.clamp(tot, min=1e-12), torch.zeros_like(tot))

    def shift(self, x):
        """(n,) the toward-minus-away log-odds shift for each row of the feature matrix ``x``."""
        mv = x[:, self.mv_idx]; nz = x[:, self.nz_idx]
        basis = torch.stack([torch.ones_like(nz), nz, nz * nz], dim=-1)            # (n, 6, 3)
        c = self.coef[self.grp]                                                    # (6, 3)
        s = (basis * c).sum(-1)
        if self.v2:
            s = s + self.coef_rng[self.grp] * self.rho(x)
        return (self.sin * mv * s).sum(1)

    def forward(self, x):
        s = self.shift(x)
        z = torch.zeros_like(s)
        f = z
        if self.v2:
            f = x[:, self.mv_idx[0]] * (self.ahead[0] + self.ahead[1] * self.rho(x)[:, 0])
        return torch.stack([f, -s, -s, s, s] + [z] * 9, dim=1)                     # (n, 14)

    def step_shift(self, x):
        """(n,) the shift of the step head's log-location (0 unless the v2 levers are attached)."""
        if not self.v2:
            return torch.zeros(x.shape[0], dtype=x.dtype)
        return x[:, self.mv_idx[0]] * (self.ahead[2] + self.ahead[3] * self.rho(x)[:, 0])

    def spec(self):
        d = dict(mv_idx=list(self.mv_idx), nz_idx=list(self.nz_idx), coef=self.coef.detach().cpu().numpy().tolist())
        if self.v2:
            d.update(rho_idx=self.rho_idx, coef_rng=self.coef_rng.detach().cpu().numpy().tolist(),
                     ahead=self.ahead.detach().cpu().numpy().tolist())
        return d


def stim_turn_np(coef, sin, grp, mv_idx, nz_idx, x, v2=None):
    """The numpy twin of ``StimTurnTerm`` for ONE feature row (the generator's per-bout path). ``v2`` =
    (rho_idx, lk, coef_rng per slot, ahead) when the v2 levers are attached. Returns (turn 14-vector, step shift)."""
    mv = x[mv_idx]; nz = x[nz_idx]
    c = coef[grp]
    s_d = c[:, 0] + c[:, 1] * nz + c[:, 2] * nz * nz
    f = 0.0; st = 0.0
    if v2 is not None:
        rho_idx, lk, rng_slot, ahead = v2
        P = np.clip(x[rho_idx], 0.0, None); tot = P.sum(-1)
        rho = np.where(tot > 0, (P * lk).sum(-1) / np.maximum(tot, 1e-12), 0.0)
        s_d = s_d + rng_slot * rho
        f = float(mv[0] * (ahead[0] + ahead[1] * rho[0])); st = float(mv[0] * (ahead[2] + ahead[3] * rho[0]))
    s = float(np.sum(sin * mv * s_d))
    o = np.zeros(14, dtype=np.float32); o[0] = f; o[1] = o[2] = -s; o[3] = o[4] = s
    return o, st


def stim_rho_indices(cfg_feat):
    """(6, n_range) feature-row indices of the bump map's cells p{d}_{j} (the v2 range levers read them)."""
    cols = M.pop_cols(cfg_feat["stim_mode"])
    k_s = int(cfg_feat["K"]) + (3 * int(cfg_feat["K"]) if cfg_feat.get("include_symbols") else 0)
    return [[k_s + cols.index(f"p{a}_{r}") for r in range(len(M.SHOAL_RBF_PX))] for a in range(len(M.POP_ANG_DEG))]


# V3 (Model 34): the count enters through a SATURATING curve with a fitted saturation point,
#   g(n) = 1 - exp(-(n - 1) / kappa),  kappa = exp(log_kappa), started at 1,
# so 1 -> 2 dots can be a STEP (small kappa) or a ramp (large kappa) -- the data decides. MEASURED: real attraction
# is ~1 mm at 1 dot and ~6 mm flat from 2 up; the smooth a + b nz + c nz^2 of v1/v2 could not bend that sharply
# (c came out negative in 10/10 folds, trying to). Per bearing group (side, rear-side): a, b, log kappa, e (range);
# group AHEAD: forward logit += w_ahead (h + k rho). No step levers (they stayed ~0 in Model 33). 10 numbers.
# Two layouts, same levers:
#   'pop'       per-direction slots (popd / popl, incl. '_lab'): n_d = 8 ** nz_d, sin of the slot centre, mv_d,
#               rho_d from the bump map; ahead = slot 0 (|bearing| < 30 deg).
#   'centroid'  S4 (one group, its centroid): n = 8 ** nz, sin = mv_sin, x `moving`; side coefficients when the
#               centroid is in front (mv_cos >= 0), rear-side behind; rho = 2 rz (log2 of range / 70 px, within
#               +-2 octaves); ahead = moving x 1[mv_cos > cos 30 deg].
# b starts at 0, so the term is exactly 0 at start (bit-identical to the base) while d/db != 0 -- kappa gets a
# gradient once b moves. log_kappa is clamped to [-3, 3] (kappa 0.05 .. 20) in use.
_V3_LK_CLAMP = 3.0


class StimTurnTermV3(nn.Module):
    version = 3
    v2 = False
    reg_index = {"logk": [2, 6], "k": [9]}                  # flat indices of the weak pulls (profile_finetune)

    def __init__(self, layout, idx):
        super().__init__()
        self.layout = str(layout)
        # plain ints / lists of ints / lists of lists (JSON-safe, what ``spec`` saves)
        self.idx = {k: np.asarray(v, dtype=int).tolist() for k, v in idx.items()}
        if self.layout == "pop":
            ang = np.asarray(M.POP_ANG_DEG, float)
            self.register_buffer("sin", torch.tensor(np.sin(np.radians(ang)), dtype=torch.float32))
            self.register_buffer("grp", torch.tensor([_STIM_TURN_GROUP[float(a)] for a in ang], dtype=torch.long))
            self.register_buffer("lk", torch.tensor(np.log2(np.asarray(M.SHOAL_RBF_PX, float) / 70.0),
                                                    dtype=torch.float32))
        elif self.layout != "centroid":
            raise ValueError(self.layout)
        self.coef3 = nn.Parameter(torch.zeros(2, 4))        # [a, b, log kappa, e] x (side, rear-side)
        self.ahead = nn.Parameter(torch.zeros(2))           # [h, k]

    # ---- shared pieces
    def _rho_pop(self, x):
        P = torch.clamp(x[:, torch.as_tensor(self.idx["rho"])], min=0.0)
        tot = P.sum(-1)
        return torch.where(tot > 0, (P * self.lk).sum(-1) / torch.clamp(tot, min=1e-12), torch.zeros_like(tot))

    def _g(self, n, grp):
        kap = torch.exp(torch.clamp(self.coef3[grp, 2], -_V3_LK_CLAMP, _V3_LK_CLAMP))
        return 1.0 - torch.exp(-torch.clamp(n - 1.0, min=0.0) / kap)

    def forward(self, x):
        if self.layout == "pop":
            mv = x[:, self.idx["mv"]]; nz = x[:, self.idx["nz"]]
            n = torch.pow(8.0, nz); rho = self._rho_pop(x)                          # (n, 6)
            c = self.coef3[self.grp]                                                # (6, 4)
            s_d = c[:, 0] + c[:, 1] * self._g(n, self.grp) + c[:, 3] * rho
            s = (self.sin * mv * s_d).sum(1)
            f = mv[:, 0] * (self.ahead[0] + self.ahead[1] * rho[:, 0])
        else:
            mvf = x[:, self.idx["moving"]]; sn = x[:, self.idx["mv_sin"]]; cs = x[:, self.idx["mv_cos"]]
            n = torch.pow(8.0, x[:, self.idx["nz"]]); rho = 2.0 * x[:, self.idx["rz"]]
            grp = (cs < 0).long()
            c = self.coef3[grp]                                                     # (n, 4)
            kap = torch.exp(torch.clamp(c[:, 2], -_V3_LK_CLAMP, _V3_LK_CLAMP))
            g = 1.0 - torch.exp(-torch.clamp(n - 1.0, min=0.0) / kap)
            s = sn * mvf * (c[:, 0] + c[:, 1] * g + c[:, 3] * rho)
            ah = mvf * (cs > float(np.cos(np.radians(30.0)))).float()
            f = ah * (self.ahead[0] + self.ahead[1] * rho)
        z = torch.zeros_like(s)
        return torch.stack([f, -s, -s, s, s] + [z] * 9, dim=1)

    def step_shift(self, x):
        return torch.zeros(x.shape[0], dtype=x.dtype)

    def numpy_fn(self):
        """x (one feature row) -> (14-vector, 0.0): the generator's per-bout twin of ``forward``."""
        c3 = self.coef3.detach().cpu().numpy().astype(np.float64)
        ah = self.ahead.detach().cpu().numpy().astype(np.float64)
        kap = np.exp(np.clip(c3[:, 2], -_V3_LK_CLAMP, _V3_LK_CLAMP))
        ix = {k: np.asarray(v) for k, v in self.idx.items()}
        if self.layout == "pop":
            sin = self.sin.cpu().numpy().astype(np.float64); grp = self.grp.cpu().numpy()
            lk = self.lk.cpu().numpy().astype(np.float64)

            def fn(x):
                mv = x[ix["mv"]]; n = 8.0 ** x[ix["nz"]]
                P = np.clip(x[ix["rho"]], 0.0, None); tot = P.sum(-1)
                rho = np.where(tot > 0, (P * lk).sum(-1) / np.maximum(tot, 1e-12), 0.0)
                g = 1.0 - np.exp(-np.clip(n - 1.0, 0.0, None) / kap[grp])
                c = c3[grp]
                s = float(np.sum(sin * mv * (c[:, 0] + c[:, 1] * g + c[:, 3] * rho)))
                f = float(mv[0] * (ah[0] + ah[1] * rho[0]))
                o = np.zeros(14, dtype=np.float32); o[0] = f; o[1] = o[2] = -s; o[3] = o[4] = s
                return o, 0.0
        else:
            ca = np.cos(np.radians(30.0))

            def fn(x):
                mvf = float(x[ix["moving"]]); sn = float(x[ix["mv_sin"]]); cs = float(x[ix["mv_cos"]])
                n = 8.0 ** float(x[ix["nz"]]); rho = 2.0 * float(x[ix["rz"]])
                gi = 1 if cs < 0 else 0
                g = 1.0 - np.exp(-max(n - 1.0, 0.0) / kap[gi])
                s = sn * mvf * (c3[gi, 0] + c3[gi, 1] * g + c3[gi, 3] * rho)
                f = mvf * float(cs > ca) * (ah[0] + ah[1] * rho)
                o = np.zeros(14, dtype=np.float32); o[0] = f; o[1] = o[2] = -s; o[3] = o[4] = s
                return o, 0.0
        return fn

    def spec(self):
        return dict(version=3, layout=self.layout, idx=self.idx,
                    coef3=self.coef3.detach().cpu().numpy().tolist(), ahead=self.ahead.detach().cpu().numpy().tolist())


def stim_turn_v3(cfg_feat):
    """A zero-started ``StimTurnTermV3`` for the checkpoint's stimulus layout."""
    mode = cfg_feat["stim_mode"]
    k_s = int(cfg_feat["K"]) + (3 * int(cfg_feat["K"]) if cfg_feat.get("include_symbols") else 0)
    if mode in M.POP_MODES:
        mv, nz = stim_turn_indices(cfg_feat)
        return StimTurnTermV3("pop", dict(mv=mv, nz=nz, rho=stim_rho_indices(cfg_feat)))
    cols = M.shoal_cols(mode)
    need = ("moving", "nz", "rz", "mv_sin", "mv_cos")
    if not all(c in cols for c in need):
        raise ValueError(f"the v3 term needs {need} in the stimulus block of {mode!r}")
    return StimTurnTermV3("centroid", {c: k_s + cols.index(c) for c in need})


# V4 (Model 40): ONE SHIFT PER DOT COUNT in place of a + b (1 - exp(-(n - 1) / kappa)). The programme has only five
# counts, so a table is as readable as the curve, and it removes the kappa-b trade-off Models 36/37 exposed (kappa ~1
# with the log-kappa prior, 0.15-0.34 without it, at similar loss). It can also give 4-6 dots LESS than 2 dots, which a
# saturating curve cannot -- needed because the base network already ramps with count (turn toward 11.9 -> 15.9 deg
# from 2 to 8 dots), so v3's shared step over-attracted 4-6 dots (attraction 7.0 / 7.6 vs real 5.2 / 6.4).
# Plus a LATE-BLOCK factor on the side shift: real fish relax back out after ~25 s, while a time-invariant turn term
# can only deepen the dip by pulling harder all block (Model 36: plateau 16.3 vs real 17.3 mm). The side shift is
# multiplied by 1 + tanh(gamma) (1 - exp(-t / 20 s)), t = time since the condition's first bout, read off the 4 s
# block clock as exp(-t / 20) = clock ** (4 / 20); tanh keeps the factor in (0, 2), so the pull never flips sign.
# Centroid (S4) layout only. Numbers: per bearing group (side, rear-side) b_1, b_2, b_4, b_6, b_8 and e (range);
# group ahead h, k; late gamma -- 15, all zero at the start, so the term is exactly 0 (bit-identical to the base).
V4_COUNTS = (1.0, 2.0, 4.0, 6.0, 8.0)
V4_LATE_TAU_S = 20.0


class StimTurnTermV4(nn.Module):
    version = 4
    v2 = False
    reg_index = {"logk": [], "k": [13]}                     # flat order: bN (10), e (2), ahead h, k (2), late (1)

    def __init__(self, idx, clock_tau_s=4.0, late_tau_s=V4_LATE_TAU_S):
        super().__init__()
        self.layout = "centroid"
        self.idx = {k: int(v) for k, v in idx.items()}
        self.clock_tau_s, self.late_tau_s = float(clock_tau_s), float(late_tau_s)
        self.register_buffer("counts", torch.tensor(V4_COUNTS, dtype=torch.float32))
        self.bN = nn.Parameter(torch.zeros(2, len(V4_COUNTS)))  # (side, rear-side) x (1, 2, 4, 6, 8 dots)
        self.e = nn.Parameter(torch.zeros(2))                    # range, per group
        self.ahead = nn.Parameter(torch.zeros(2))                # [h, k]
        self.late = nn.Parameter(torch.zeros(1))                 # gamma

    def _late_t(self, clock):
        """1 - exp(-t / late_tau) from the block clock exp(-t / clock_tau)."""
        return 1.0 - torch.pow(torch.clamp(clock, min=1e-30, max=1.0), self.clock_tau_s / self.late_tau_s)

    def forward(self, x):
        mvf = x[:, self.idx["moving"]]; sn = x[:, self.idx["mv_sin"]]; cs = x[:, self.idx["mv_cos"]]
        n = torch.pow(8.0, x[:, self.idx["nz"]]); rho = 2.0 * x[:, self.idx["rz"]]
        grp = (cs < 0).long()
        w = (torch.abs(n[:, None] - self.counts[None, :]) < 0.5).to(x.dtype)       # (n, 5): which count
        b = (self.bN[grp] * w).sum(1)
        fac = 1.0 + torch.tanh(self.late[0]) * self._late_t(x[:, self.idx["clock"]])
        s = sn * mvf * (b + self.e[grp] * rho) * fac
        ah = mvf * (cs > float(np.cos(np.radians(30.0)))).to(x.dtype)
        f = ah * (self.ahead[0] + self.ahead[1] * rho)
        z = torch.zeros_like(s)
        return torch.stack([f, -s, -s, s, s] + [z] * 9, dim=1)

    def step_shift(self, x):
        return torch.zeros(x.shape[0], dtype=x.dtype)

    def numpy_fn(self):
        """x (one feature row) -> (14-vector, 0.0): the generator's per-bout twin of ``forward``."""
        bN = self.bN.detach().cpu().numpy().astype(np.float64)
        e = self.e.detach().cpu().numpy().astype(np.float64)
        ah = self.ahead.detach().cpu().numpy().astype(np.float64)
        g = float(np.tanh(float(self.late.detach().cpu().numpy()[0])))
        counts = np.asarray(V4_COUNTS, float)
        ix, ca, p = dict(self.idx), float(np.cos(np.radians(30.0))), self.clock_tau_s / self.late_tau_s

        def fn(x):
            mvf = float(x[ix["moving"]]); sn = float(x[ix["mv_sin"]]); cs = float(x[ix["mv_cos"]])
            n = 8.0 ** float(x[ix["nz"]]); rho = 2.0 * float(x[ix["rz"]])
            gi = 1 if cs < 0 else 0
            b = float(bN[gi][np.abs(n - counts) < 0.5].sum())
            clk = min(max(float(x[ix["clock"]]), 1e-30), 1.0)
            s = sn * mvf * (b + e[gi] * rho) * (1.0 + g * (1.0 - clk ** p))
            f = mvf * float(cs > ca) * (ah[0] + ah[1] * rho)
            o = np.zeros(14, dtype=np.float32); o[0] = f; o[1] = o[2] = -s; o[3] = o[4] = s
            return o, 0.0
        return fn

    def spec(self):
        return dict(version=4, layout=self.layout, idx=self.idx, clock_tau_s=self.clock_tau_s,
                    late_tau_s=self.late_tau_s, bN=self.bN.detach().cpu().numpy().tolist(),
                    e=self.e.detach().cpu().numpy().tolist(), ahead=self.ahead.detach().cpu().numpy().tolist(),
                    late=self.late.detach().cpu().numpy().tolist())


def stim_turn_v4(cfg_feat, n_features):
    """A zero-started ``StimTurnTermV4`` for a centroid (S4) checkpoint with ``n_features`` inputs. The block clock's
    column is located from the feature layout: [..., lag flag][block clock][run clock][tail (align, static, ...)]."""
    mode = cfg_feat["stim_mode"]
    if mode in M.POP_MODES:
        raise ValueError("the v4 term is built for the centroid (S4) layout")
    if str(cfg_feat.get("clock_mode", "") or "").lower() != "exp":
        raise ValueError("the v4 late factor reads the exponential block clock (clock_mode='exp')")
    if int(cfg_feat.get("K_stim", 1) or 1) > 1 or cfg_feat.get("clock2_tau_s"):
        raise ValueError("the v4 clock index assumes one stimulus-block and one block-clock column")
    k_s = int(cfg_feat["K"]) + (3 * int(cfg_feat["K"]) if cfg_feat.get("include_symbols") else 0)
    cols = M.shoal_cols(mode)
    need = ("moving", "nz", "rz", "mv_sin", "mv_cos")
    if not all(c in cols for c in need):
        raise ValueError(f"the v4 term needs {need} in the stimulus block of {mode!r}")
    idx = {c: k_s + cols.index(c) for c in need}
    idx["clock"] = int(n_features) - MB.tail_width(cfg_feat) - (1 if cfg_feat.get("run_clock_tau") else 0) - 1
    return StimTurnTermV4(idx, clock_tau_s=float(cfg_feat.get("clock_tau_s", 4.0)))


# V5 (report notebook, Model 11): COUNT-GATED terms -- every piece is multiplied by a count curve that is 0 at ONE
# dot, so the 1-dot behaviour (Model 17's best moving curve) is untouched by construction; no level that acts at every
# count. Centroid (S4) layout. With n = 8 ** nz (the dot count), rho = 2 rz = log2(range / 70 px) and
# g_k(n) = 1 - exp(-(n - 1) / kappa):
#   straight (dots within +-30 deg ahead, moving):  f  = ahead * C(n) * (h + q rho),   C = nz ('nz') or g_ka(n) ('g')
#   sideways (moving, front / behind groups):       s  = mv_sin * g_kg(n) * (b_g + e_g rho)                ['side']
#   static (dots shown, not moving):                s += S sin * g_ks(n) * (b_st + e_st rho)
#                                                   f += S ahead * nz * (h_st + q_st rho)                   ['static']
# f goes to the forward component's logit, s is the toward-minus-away log-odds shift, exactly as in v1-v4. For static
# dots sin / cos of the bearing are sum_j gs_j / sum_j gc_j (the bump weights sum to one on dot bouts). All numbers
# start at 0 (log kappas at 0 = kappa 1), so attaching the term leaves the model bit-identical.
# Arms: 'S' (straight only, 2 numbers), 'B' (+ sideways, 8), 'BS' (straight count curve g_ka instead of nz, 9),
# 'Bst' (B + static, 13).
# 'BSl' (the re-anchored pf run, 10): BS with a LATE-BLOCK factor on the side shift, v4's form -- s x (1 + tanh(gamma)
# (1 - exp(-t / 20 s))), t = time since the condition's first bout read off the 4 s block clock (clock ** (4 / 20)).
# Measured on report Model 11: the time-invariant side pull that deepens the dip keeps the fish ~1.5 mm too close for
# the rest of the block; switching the term off after 20 s restored the late level and rebound, switching it off below
# 8-12 mm did not. The factor lies in (0, 2): the pull can relax (or grow) late in the block, never flip.
V5_ARMS = {"S": dict(side=False, static=False, fcount="nz"), "B": dict(side=True, static=False, fcount="nz"),
           "BS": dict(side=True, static=False, fcount="g"), "Bst": dict(side=True, static=True, fcount="nz"),
           "BSl": dict(side=True, static=False, fcount="g", late=True),
           "Bc": dict(side=True, static=False, fcount="nz", cut_s=20.0),
           "Bstc": dict(side=True, static=True, fcount="nz", cut_s=20.0)}
# 'Bc' / 'Bstc' (report Model 14): B / Bst with a HARD CUT-OFF -- the whole term (straight, side and static pieces)
# acts only while t < 20 s, t = time since the condition's first bout read off the 4 s block clock (on while
# clock > exp(-20 / 4)); after it the model is the base model again. Measured on Model 11: switching pf-B's term off
# after 20 s restored the late-block level and rebound while keeping the dip; 13d's fitted late factor was not
# identified (folds 0.08-1.12 at 60 s). The cut is FIXED, no number is added. Needs the block clock (idx['clock']).
V5_LATE_TAU_S = 20.0


class StimTurnTermV5(nn.Module):
    version = 5
    v2 = False
    reg_index = {"logk": [], "k": [1]}                           # q (the straight term's range slope): weak pull

    def __init__(self, idx, side=False, static=False, fcount="nz", late=False, clock_tau_s=4.0,
                 late_tau_s=V5_LATE_TAU_S, cut_s=None):
        super().__init__()
        self.idx = {k: np.asarray(v, dtype=int).tolist() for k, v in idx.items()}
        self.side, self.static, self.fcount = bool(side), bool(static), str(fcount)
        self.late_on = bool(late)
        self.clock_tau_s, self.late_tau_s = float(clock_tau_s), float(late_tau_s)
        if self.fcount not in ("nz", "g"):
            raise ValueError(self.fcount)
        if self.late_on and (not self.side or "clock" not in self.idx):
            raise ValueError("the late factor acts on the side shift and reads the block clock (idx['clock'])")
        self.cut_s = None if cut_s is None else float(cut_s)
        if self.cut_s is not None and "clock" not in self.idx:
            raise ValueError("the cut-off reads the block clock (idx['clock'])")
        self.ahead = nn.Parameter(torch.zeros(2))                  # [h, q]
        if self.fcount == "g":
            self.logk_a = nn.Parameter(torch.zeros(1))
        if self.side:
            self.coef_s = nn.Parameter(torch.zeros(2, 3))          # [b, e, log kappa] x (front, behind)
        if self.static:
            self.coef_st = nn.Parameter(torch.zeros(5))            # [b_st, e_st, log kappa_st, h_st, q_st]
        if self.late_on:
            self.late = nn.Parameter(torch.zeros(1))               # gamma (registered LAST: flat order unchanged)

    def _late_fac(self, clock):
        """1 + tanh(gamma) (1 - exp(-t / late_tau)), from the block clock exp(-t / clock_tau)."""
        lt = 1.0 - torch.pow(torch.clamp(clock, min=1e-30, max=1.0), self.clock_tau_s / self.late_tau_s)
        return 1.0 + torch.tanh(self.late[0]) * lt

    @staticmethod
    def _g(n, logk):
        kap = torch.exp(torch.clamp(logk, -_V3_LK_CLAMP, _V3_LK_CLAMP))
        return 1.0 - torch.exp(-torch.clamp(n - 1.0, min=0.0) / kap)

    def forward(self, x):
        ix = self.idx
        mvf = x[:, ix["moving"]]; act = x[:, ix["act"]]; nz = x[:, ix["nz"]]
        n = torch.pow(8.0, nz); rho = 2.0 * x[:, ix["rz"]]
        sn = x[:, ix["mv_sin"]]; cs = x[:, ix["mv_cos"]]
        ca = float(np.cos(np.radians(30.0)))
        cnt = nz if self.fcount == "nz" else self._g(n, self.logk_a[0])
        f = mvf * (cs > ca).float() * cnt * (self.ahead[0] + self.ahead[1] * rho)
        s = torch.zeros_like(f)
        if self.side:
            grp = (cs < 0).long()
            c = self.coef_s[grp]                                                   # (n, 3)
            side = sn * self._g(n, c[:, 2]) * (c[:, 0] + c[:, 1] * rho)
            if self.late_on:
                side = side * self._late_fac(x[:, ix["clock"]])
            s = s + side
        if self.static:
            S = act * (1.0 - mvf)
            sin_ = x[:, torch.as_tensor(ix["gs"])].sum(1); cos_ = x[:, torch.as_tensor(ix["gc"])].sum(1)
            cst = self.coef_st
            s = s + S * sin_ * self._g(n, cst[2]) * (cst[0] + cst[1] * rho)
            f = f + S * (cos_ > ca).float() * nz * (cst[3] + cst[4] * rho)
        if self.cut_s is not None:                       # on only while t < cut_s (clock > exp(-cut_s / tau))
            on = (x[:, ix["clock"]] > float(np.exp(-self.cut_s / self.clock_tau_s))).to(x.dtype)
            f = f * on; s = s * on
        z = torch.zeros_like(s)
        return torch.stack([f, -s, -s, s, s] + [z] * 9, dim=1)

    def step_shift(self, x):
        return torch.zeros(x.shape[0], dtype=x.dtype)

    def numpy_fn(self):
        """x (one feature row) -> (14-vector, 0.0): the generator's per-bout twin of ``forward``."""
        ah = self.ahead.detach().cpu().numpy().astype(np.float64)
        lka = float(self.logk_a.detach().cpu().numpy()[0]) if self.fcount == "g" else 0.0
        cs_ = self.coef_s.detach().cpu().numpy().astype(np.float64) if self.side else None
        cst = self.coef_st.detach().cpu().numpy().astype(np.float64) if self.static else None
        gl = float(np.tanh(float(self.late.detach().cpu().numpy()[0]))) if self.late_on else 0.0
        pl = self.clock_tau_s / self.late_tau_s
        # the same float32 threshold the torch path compares against
        cut_thr = None if self.cut_s is None else float(np.float32(np.exp(-self.cut_s / self.clock_tau_s)))
        ix = {k: np.asarray(v) for k, v in self.idx.items()}
        ca = np.cos(np.radians(30.0))
        g_ = lambda n, lk: 1.0 - np.exp(-max(n - 1.0, 0.0) / np.exp(np.clip(lk, -_V3_LK_CLAMP, _V3_LK_CLAMP)))  # noqa: E731

        def fn(x):
            mvf = float(x[ix["moving"]]); act = float(x[ix["act"]]); nz = float(x[ix["nz"]])
            n = 8.0 ** nz; rho = 2.0 * float(x[ix["rz"]])
            sn = float(x[ix["mv_sin"]]); cs = float(x[ix["mv_cos"]])
            cnt = nz if self.fcount == "nz" else g_(n, lka)
            f = mvf * float(cs > ca) * cnt * (ah[0] + ah[1] * rho)
            s = 0.0
            if cs_ is not None:
                gi = 1 if cs < 0 else 0
                side = sn * g_(n, cs_[gi, 2]) * (cs_[gi, 0] + cs_[gi, 1] * rho)
                if self.late_on:
                    clk = min(max(float(x[ix["clock"]]), 1e-30), 1.0)
                    side *= 1.0 + gl * (1.0 - clk ** pl)
                s += side
            if cst is not None:
                S = act * (1.0 - mvf)
                sin_ = float(np.sum(x[ix["gs"]])); cos_ = float(np.sum(x[ix["gc"]]))
                s += S * sin_ * g_(n, cst[2]) * (cst[0] + cst[1] * rho)
                f += S * float(cos_ > ca) * nz * (cst[3] + cst[4] * rho)
            if cut_thr is not None and not float(np.float32(x[ix["clock"]])) > cut_thr:
                f = 0.0; s = 0.0
            o = np.zeros(14, dtype=np.float32); o[0] = f; o[1] = o[2] = -s; o[3] = o[4] = s
            return o, 0.0
        return fn

    def spec(self):
        d = dict(version=5, idx=self.idx, side=self.side, static=self.static, fcount=self.fcount)
        if self.late_on:
            d.update(late=True, clock_tau_s=self.clock_tau_s, late_tau_s=self.late_tau_s)
        if self.cut_s is not None:
            d.update(cut_s=self.cut_s, clock_tau_s=self.clock_tau_s)
        return d


def stim_turn_v5(cfg_feat, arm, n_features=None):
    """A zero-started ``StimTurnTermV5`` of arm 'S' / 'B' / 'BS' / 'Bst' / 'BSl' for the checkpoint's centroid layout.
    'BSl' also needs ``n_features`` (the net's input width) to locate the block clock, as ``stim_turn_v4``."""
    mode = cfg_feat["stim_mode"]
    if mode in M.POP_MODES:
        raise ValueError("the v5 term is built for the centroid (S4) layout")
    k_s = int(cfg_feat["K"]) + (3 * int(cfg_feat["K"]) if cfg_feat.get("include_symbols") else 0)
    cols = M.shoal_cols(mode)
    kw = dict(V5_ARMS[arm])
    base = ("act", "moving", "nz", "rz", "mv_sin", "mv_cos")
    bumps = tuple(f"gs{j}" for j in range(1, 5)) + tuple(f"gc{j}" for j in range(1, 5))
    # the bump columns are read only by the static pieces: a layout without them (6a's 'shoal_int') runs S / B / BS
    need = base + (bumps if kw.get("static") else ())
    if not all(c in cols for c in need):
        raise ValueError(f"the v5 term needs {need} in the stimulus block of {mode!r}")
    idx = {c: k_s + cols.index(c) for c in base}
    if all(c in cols for c in bumps):                   # kept whenever present: archived specs are unchanged
        idx["gs"] = [k_s + cols.index(f"gs{j}") for j in range(1, 5)]
        idx["gc"] = [k_s + cols.index(f"gc{j}") for j in range(1, 5)]
    if kw.get("late") or kw.get("cut_s"):
        if n_features is None:
            raise ValueError(f"arm {arm!r} needs n_features to locate the block clock")
        if str(cfg_feat.get("clock_mode", "") or "").lower() != "exp":
            raise ValueError("the late factor reads the exponential block clock (clock_mode='exp')")
        if int(cfg_feat.get("K_stim", 1) or 1) > 1 or cfg_feat.get("clock2_tau_s"):
            raise ValueError("the clock index assumes one stimulus-block and one block-clock column")
        idx["clock"] = int(n_features) - MB.tail_width(cfg_feat) - (1 if cfg_feat.get("run_clock_tau") else 0) - 1
        kw["clock_tau_s"] = float(cfg_feat.get("clock_tau_s", 4.0))
    return StimTurnTermV5(idx, **kw)




def stim_turn_indices(cfg_feat):
    """Column indices of the per-direction moving fraction and log count in the FEATURE ROW for a population
    stimulus mode: [K history][(3K symbols)][stim block ...]."""
    mode = cfg_feat["stim_mode"]
    if mode not in M.POP_MODES:
        raise ValueError(f"the stimulus turn term needs a population stimulus mode, not {mode!r}")
    cols = M.pop_cols(mode)
    k_s = int(cfg_feat["K"]) + (3 * int(cfg_feat["K"]) if cfg_feat.get("include_symbols") else 0)
    return ([k_s + cols.index(f"mv{a}") for a in range(len(M.POP_ANG_DEG))],
            [k_s + cols.index(f"nz{a}") for a in range(len(M.POP_ANG_DEG))])

# --------------------------------------------------------------------------- #
#  REPULSION TERM (report Model 43, Final_report_extra_data.ipynb). A fixed-form term on a trained model, fitted
#  to the Larsch lab screen data (short-range turn-away and backing off from a single dot, twin_test.py), stored
#  as ``cfg_feat["repulsion"]`` = {"a", "r_c", "w"} (+ {"c", "r_b", "w_b"} for the brake). It reads EACH active
#  dot j (not the centroid), at distance d_j (mm) and bearing phi_j from the pre-bout heading (travel frame):
#      turn:  s = -a * sum_j sin(phi_j) * sigma((r_c - d_j) / w)      toward-minus-away log-odds: +s on the
#             right-turn (dtheta > 0) components, -s on the left -- the V5 term's convention (positive = toward)
#      brake: u = -c * sum_j max(cos(phi_j), 0) * sigma((r_b - d_j) / w_b)   added to the step head's log-means
#      step:  u += s_amp * sum_j sigma((s_r - d_j) / s_w)   the step's log-means near ANY dot (Model 50; s_amp < 0 =
#             shorter steps), fitted by the teacher-forced likelihood of the screen bouts (scripts/repulsion_lik.py)
#  The network and any profile correction are untouched. Applied in the numpy generator (every rollout) and, as a
#  per-bout side input computed from the record's own dot positions, in the torch teacher-forced paths (score /
#  likelihood, conditional_turn, conditional_kin). The batch engine refuses it.
# --------------------------------------------------------------------------- #
REP_PX_PER_MM = 3.60


def repulsion_shifts(dxs, dys, x, y, h, rp):
    """(turn shift s, step log-mean shift u) of the repulsion term. ``dxs``/``dys``: dot positions in px, (k,) for one
    bout or (n, k) for n bouts, NaN = inactive; ``x``/``y``/``h``: pre-bout pose (px, rad), scalars or (n,)."""
    dxs = np.atleast_2d(np.asarray(dxs, float)); dys = np.atleast_2d(np.asarray(dys, float))
    x = np.asarray(x, float).reshape(-1, 1); y = np.asarray(y, float).reshape(-1, 1)
    h = np.asarray(h, float).reshape(-1, 1)
    on = np.isfinite(dxs) & np.isfinite(dys) & np.isfinite(x) & np.isfinite(h)
    d = np.hypot(dxs - x, dys - y) / REP_PX_PER_MM
    phi = MB._wrap_pi(np.arctan2(dys - y, dxs - x) - h)
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    s = -float(rp.get("a", 0.0)) * np.where(on, np.sin(phi) * sig((float(rp.get("r_c", 0.0)) - d)
                                                                   / float(rp.get("w", 1.0))), 0.0).sum(1)
    u = np.zeros(s.shape)
    if rp.get("s_amp"):                                    # STEP term (Model 50): the step's log-mean near any dot,
        u = u + float(rp["s_amp"]) * np.where(on, sig((float(rp["s_r"]) - d)  # whatever its bearing (likelihood-fitted)
                                                       / float(rp.get("s_w", 1.0))), 0.0).sum(1)
    if rp.get("c"):
        u = -float(rp["c"]) * np.where(on, np.maximum(np.cos(phi), 0.0) * sig((float(rp["r_b"]) - d)
                                                                             / float(rp.get("w_b", 1.0))), 0.0).sum(1)
    return np.nan_to_num(s), np.nan_to_num(u)


# --------------------------------------------------------------------------- #
#  GROUP REPULSION TERM (2026-10-18, Final_report_shoaling.ipynb). ONE representation for attraction and repulsion:
#  the dot groups the network already sees. Each visible MOVING group g carries its NEAREST member (distance d*_g mm,
#  bearing phi*_g from the travel heading) and its count n_g; its term is the single-dot form of the Larsch-screen fit
#  applied to that nearest member, and the groups' terms ADD, which is exactly the effects rule applied to the term
#  (the effects rule adds every group's output; the term is additive on the turn log-odds and the step log-mean):
#      turn:  s_g = -a n_g^beta sin(phi*_g) sigma((r_c - d*_g) / w)       (toward-minus-away log-odds, as repulsion)
#      step:  u_g = s_amp n_g^beta sigma((s_r - d*_g) / s_w)               (the step head's log-mean)
#  stored as ``cfg_feat["grouprep"]`` = {"a", "r_c", "w", "beta", "s_amp", "s_r", "s_w"}. For one dot (the screen) d*
#  is the dot itself, so the screen fit needs no synthetic groups. Numpy generator only (every rollout); the torch
#  teacher-forced paths do NOT see it (score it with scripts/grouprep_fit.py instead). Exclusive with "repulsion".
#  ``cfg_feat["collision_mm"]`` (optional, the shoal reference): a bout whose landing point is within that distance of
#  a neighbour (provider.neighbours_xy()) is redrawn inside the wall-rejection loop.
# --------------------------------------------------------------------------- #
def nearest_dot(xs, ys, on, x, y):
    """(nx, ny) of the nearest active finite dot to (x, y), or None."""
    xs = np.atleast_1d(np.asarray(xs, float)); ys = np.atleast_1d(np.asarray(ys, float))
    ok = np.isfinite(xs) & np.isfinite(ys)
    if on is not None:
        ok &= np.atleast_1d(np.asarray(on, bool))
    if not ok.any():
        return None
    d = np.where(ok, np.hypot(xs - x, ys - y), np.inf)
    j = int(np.argmin(d))
    return float(xs[j]), float(ys[j])


def grouprep_shifts(near, x, y, h, gp):
    """(turn shift s, step log-mean shift u) of the group repulsion term. ``near`` = [(nx, ny, n, moving)] per visible
    group: its nearest member (px), count and moving flag; ``x``/``y``/``h`` = pre-bout pose (px, travel heading rad)."""
    a = float(gp.get("a", 0.0)); rc = float(gp.get("r_c", 0.0)); w = float(gp.get("w", 1.0))
    beta = float(gp.get("beta", 0.0))
    sa = float(gp.get("s_amp", 0.0)); sr = float(gp.get("s_r", 0.0)); sw = float(gp.get("s_w", 1.0))
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
    s = u = 0.0
    for nx, ny, n, mv in near:
        if not (np.isfinite(nx) and np.isfinite(ny)) or not mv or mv <= 0:
            continue
        d = float(np.hypot(nx - x, ny - y)) / REP_PX_PER_MM
        g = max(float(n), 1.0) ** beta
        if a:
            phi = float(MB._wrap_pi(np.arctan2(ny - y, nx - x) - h))
            s += -a * g * np.sin(phi) * float(sig((rc - d) / w))
        if sa:
            u += sa * g * float(sig((sr - d) / sw))
    return float(s), float(u)


class repulsion_side:
    """Context: attach the repulsion term's per-bout shifts, computed from ``rec``'s own dot positions and real
    pre-bout poses, to the net for one teacher-forced pass (``skip_term`` / ``step_term`` add them). No-op when the
    model has no repulsion term."""

    def __init__(self, model, rec):
        self.net = getattr(model, "_model", None)
        self.rp = (getattr(model, "_cfg_feat", None) or {}).get("repulsion")
        self.rec = rec

    def __enter__(self):
        if self.rp:
            rec = self.rec
            dxs, dys = getattr(rec, "dot_xs", None), getattr(rec, "dot_ys", None)
            if dxs is None:
                raise RuntimeError("repulsion term: the record carries no per-dot positions (rec.dot_xs)")
            s, u = repulsion_shifts(dxs, dys, rec.x_px, rec.y_px, MB._travel_ref(rec), self.rp)
            self.net._rep_side = (torch.tensor(s, dtype=torch.float32), torch.tensor(u, dtype=torch.float32))
        return self

    def __exit__(self, *exc):
        if self.rp:
            self.net._rep_side = None
        return False


def _cond_cols(y_ang, y_dist, dth_scale, kin_scales, y_dt=None):
    """The within-bout conditioners, on the scale the history blocks use.

    Two by default (|dtheta|, centred log distance) so every pre-ordering caller is unchanged;
    pass ``y_dt`` to also get the centred log interval, which the non-default orders need."""
    dm, ds, tm, ts = kin_scales
    a_z = np.abs(np.asarray(y_ang, float)) / float(dth_scale)
    ld = np.log(np.clip(np.asarray(y_dist, float), 1e-6, None))
    ld_c = (ld - dm) / ds
    out = (np.nan_to_num(a_z).astype(np.float32),
           np.nan_to_num(ld_c).astype(np.float32))
    if y_dt is None:
        return out
    lt = np.log(np.clip(np.asarray(y_dt, float), 1e-6, None))
    return out + (np.nan_to_num((lt - tm) / ts).astype(np.float32),)


def _noise_tensors(noise_builder, ep, dth_scale, kin_scales, dev):
    """Build one epoch's NOISY copy as device tensors, in the same row order as the clean one."""
    Xn, yan, ydn, ytn = noise_builder(ep)
    azn, lcn, ltn = _cond_cols(yan, ydn, dth_scale, kin_scales, y_dt=ytn)
    return (torch.tensor(np.asarray(Xn, np.float32), device=dev),
            torch.tensor(azn, device=dev), torch.tensor(lcn, device=dev),
            torch.tensor(ltn, device=dev),
            torch.tensor(np.asarray(yan, np.float32), device=dev),
            torch.tensor(np.asarray(ydn, np.float32), device=dev),
            torch.tensor(np.asarray(ytn, np.float32), device=dev))


# Measured, not guessed. Gradient-norm distribution over 402 steps of a 6-epoch fit (identical
# for plain m=2 and block-structured m=3): median 4.5, p90 28, p99 47, max 102. The first value
# tried, 10.0, sat between the median and p90 and clipped ~25% of steps -- that is not a backstop,
# it is a change of optimiser, and it would have made every future arm incomparable to the archived
# lineage while claiming to be a safety net. 150 is above every healthy gradient seen, so it binds
# only on a genuine excursion.
#
# NOTE the hard guarantee against NaN is the isfinite check in guarded_step, NOT this number: a
# non-finite loss or gradient is skipped outright and never written into the weights. The clip is
# a softener for large-but-finite steps, and _warn_unstable makes whatever is left visible.
GRAD_CLIP = 150.0


def guarded_step(opt, loss, params, clip=GRAD_CLIP, stats=None):
    """One optimiser step that refuses to poison the weights with a non-finite update.

    WHY THIS EXISTS. Neither joint trainer clipped gradients, and on an L4 the bgain smoke went
    ``26.86 -> nan`` in four epochs while the IDENTICAL code on CPU went ``26.82 -> 10.92``. Same
    config, same data -- CUDA's accumulation order is enough to trip an emission with lgamma and
    log terms in it. (rollout_finetune.py has clipped at 5.0 since it was written; the trainers
    that do all the actual work never did.)

    The silent-failure mode is what makes this urgent rather than cosmetic. Early stopping tests
    ``v < best - 1e-5``; with ``v = nan`` that is False, so a NaN counts as "no improvement",
    patience expires, and the fit returns ``best_state`` from whatever epoch it last improved on.
    A run that diverged at epoch 2 of 80 therefore finishes, saves, and reports as a completed
    5-hour arm while containing a 2-epoch model.

    Threshold 10.0 is deliberately loose: on a healthy fit it should never bind, so this is a
    backstop rather than a change of optimiser. It IS counted, and ``fit`` warns if it fires --
    an uninstrumented guard that quietly reshapes training would be its own trap.
    """
    if stats is not None:
        stats["n_steps"] = stats.get("n_steps", 0) + 1
    if not torch.isfinite(loss):
        if stats is not None:
            stats["n_bad_loss"] = stats.get("n_bad_loss", 0) + 1
        opt.zero_grad(set_to_none=True)
        return False
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, clip)
    if not torch.isfinite(gn):
        if stats is not None:
            stats["n_bad_grad"] = stats.get("n_bad_grad", 0) + 1
        opt.zero_grad(set_to_none=True)
        return False
    if stats is not None and float(gn) > clip:
        stats["n_clipped"] = stats.get("n_clipped", 0) + 1
    opt.step()
    return True


def _warn_unstable(stats, hist):
    """Say out loud that a fit was unstable, and how far it actually got.

    Without this a diverged fit is INVISIBLE: a non-finite validation loss fails
    ``v < best - 1e-5``, so it is counted as "no improvement", patience runs out, and the run
    returns the best state from before the blow-up. The arm completes, saves a checkpoint and
    prints a normal-looking results row while containing a model trained for a fraction of the
    requested budget. Measured on an L4: the bgain smoke went 26.86 -> nan in four epochs.
    """
    n_bad = int(stats.get("n_bad_loss", 0)) + int(stats.get("n_bad_grad", 0))
    n_clip = int(stats.get("n_clipped", 0))
    n_steps = max(int(stats.get("n_steps", 0)), 1)
    n_nan_val = int(np.sum(~np.isfinite(np.asarray(hist, float)))) if len(hist) else 0
    # TWO DIFFERENT SITUATIONS, and conflating them is how an alarm gets ignored. A non-finite
    # VALIDATION epoch means the fit actually broke and best_state is truncated -- that is the
    # five-hours-wasted case. A few skipped MINIBATCH steps with every validation epoch finite
    # means the guard did its job and the fit is healthy: measured on an L4, one skipped step out
    # of ~268 was the whole difference between 26.86 -> nan and 26.86 -> 10.94.
    if n_nan_val:
        print(f"    WARNING: UNSTABLE FIT -- {n_nan_val} of {len(hist)} epochs had a non-finite "
              f"validation loss ({n_bad} of {n_steps} minibatch steps also skipped). The returned "
              f"model is the best state from BEFORE the blow-up, so this arm is trained on far "
              f"less than the requested budget. Do not read it as a completed run.")
    elif n_bad:
        # SPLIT the count, and escalate above a few percent. A handful of skips is the guard doing
        # its job; HALF the updates being discarded is a different animal -- measured 192 of 376 on
        # a GRU fit at seq_batch=16, which is a large silent handicap that the old wording called
        # "healthy". Naming loss-vs-gradient is what makes it diagnosable without another run.
        frac = n_bad / n_steps
        lead = "note" if frac < 0.05 else "WARNING"
        print(f"    {lead}: the NaN guard skipped {n_bad} of {n_steps} minibatch step(s) "
              f"({100 * frac:.0f}%) -- {int(stats.get('n_bad_loss', 0))} non-finite LOSS, "
              f"{int(stats.get('n_bad_grad', 0))} non-finite GRADIENT. Every validation epoch was "
              f"finite, so the returned model is complete"
              + ("." if frac < 0.05 else
                 ", but it was trained on far fewer updates than the budget implies -- a larger "
                 "batch dilutes the offending bout and lowers this rate."))
    elif n_clip:
        print(f"    note: gradient clipping bound on {n_clip} step(s) (threshold {GRAD_CLIP}) -- "
              "the fit was steered but did not diverge")


def _noise_kwargs(model, train_records):
    """Translate a model's ``noise_aug`` dict into the trainer's noise kwargs ({} when off).

    ONE rng for the whole fit, advanced by every call, so each epoch sees a FRESH perturbation of
    the same records rather than a fixed augmented dataset -- which would merely be a bigger
    training set and would let the net memorise the particular jitter. Shared by the joint and
    fishre trainers so the two cannot drift."""
    na = getattr(model, "noise_aug", None)
    if not na:
        return {}
    rng = np.random.default_rng([int(model.seed), 7919])
    cfg, recs = model._cfg_feat, list(train_records)
    return dict(noise_builder=lambda _ep: MB.build_noisy_sequences(recs, cfg, rng),
                noise_mode=str(na.get("mode", "concat")),
                noise_frac=float(na.get("frac", 0.5)))


def fit_mix_joint(X, y_ang, y_dist, y_dt, hidden, dth_scale, kin_scales, *, epochs=80, lr=1e-3,
                  batch_size=256, patience=8, val_frac=0.15, seed=0, device=None,
                  n_dist=2, n_dt=3, weight_decay=1e-5, order=None,
                  noise_builder=None, noise_mode="concat", noise_frac=0.5, weights=None,
                  dt_trunc=None):
    """Train ``MDNMixJoint`` on the exact chained joint NLL. Distance/dt NaN-masked per channel
    (the last bout of every fish has no forward-looking dt or distance); early stopping is on the
    JOINT validation loss, so all three channels share one stopping criterion.

    ``noise_builder(epoch) -> (X~, dtheta~, distance~, dt)`` enables NOISE AUGMENTATION: a fresh
    pose-perturbed copy of the whole training set each epoch, with the targets corrected back to
    the true next onset (``MB.build_noisy_sequences``). ``noise_mode='concat'`` trains on clean
    and noisy rows both (~2x epoch cost, the faithful version); ``'mask'`` replaces a
    Bernoulli(``noise_frac``) subset of each clean batch instead (same cost, half the clean data).
    THE VALIDATION LOSS IS ALWAYS CLEAN -- otherwise early stopping optimises a different
    objective from the one reported, and the loss is not comparable to any archived arm."""
    M._seed_everything(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a_z, ld_c, lt_c = _cond_cols(y_ang, y_dist, dth_scale, kin_scales, y_dt=y_dt)
    Xt = torch.tensor(np.asarray(X, np.float32), device=dev)
    ya = torch.tensor(np.asarray(y_ang, np.float32), device=dev)
    yd = torch.tensor(np.asarray(y_dist, np.float32), device=dev)
    yt = torch.tensor(np.asarray(y_dt, np.float32), device=dev)
    az = torch.tensor(a_z, device=dev); lc = torch.tensor(ld_c, device=dev)
    lt_t = torch.tensor(lt_c, device=dev)
    # PER-BOUT WEIGHTS (MB.regime_weights): a weighted MEAN per channel, so the loss stays on
    # the same scale as the unweighted one and `weights=None` is bit-identical to before. Early
    # stopping is on the WEIGHTED validation loss -- the objective actually being fitted; Metric A
    # (score()) stays the plain per-bout density and is reported separately.
    wt = None if weights is None else torch.tensor(np.asarray(weights, np.float32), device=dev)
    if wt is not None and noise_builder is not None:
        raise ValueError("regime weights with noise augmentation are not wired up: the noisy "
                         "copy's rows would need the same weights")
    net = MDNMixJoint(Xt.shape[1], hidden, n_dist=n_dist, n_dt=n_dt, order=order).to(dev)
    net.dt_trunc = (float(dt_trunc) if dt_trunc else None)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    def nll_on(Xb, azb, lcb, ltb, yab, ydb, ytb, wb=None):
        ang, dpar, tpar = net.joint_forward(Xb, azb, lcb, ltb)
        la_i = MX.mix_logdens(yab, ang)
        md = torch.isfinite(ydb) & (ydb > 0); mt = torch.isfinite(ytb) & (ytb > 0)
        ld = MX._lnk_logpdf(torch.clamp(ydb, min=1e-6), *dpar)
        lt = MX._dt_logpdf(torch.clamp(ytb, min=1e-6), *tpar, hi=net.dt_trunc)
        z = torch.zeros((), device=dev)
        if wb is None:
            return -(la_i.mean() + (ld[md].mean() if md.any() else z)
                     + (lt[mt].mean() if mt.any() else z))
        la = (wb * la_i).sum() / wb.sum()
        ldw = (wb[md] * ld[md]).sum() / wb[md].sum() if md.any() else z
        ltw = (wb[mt] * lt[mt]).sum() / wb[mt].sum() if mt.any() else z
        return -(la + ldw + ltw)

    def nll(ix):
        return nll_on(Xt[ix], az[ix], lc[ix], lt_t[ix], ya[ix], yd[ix], yt[ix],
                      None if wt is None else wt[ix])

    n = Xt.shape[0]; nval = max(int(val_frac * n), 1)
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g).to(dev)
    tr, va = perm[nval:], perm[:nval]
    best, best_state, bad, hist, hist_tr = np.inf, None, 0, [], []
    _gstats = {}
    for _ep in range(epochs):
        net.train()
        # RESHUFFLE EVERY EPOCH. `perm` above fixes the train/val SPLIT (which must stay fixed);
        # the batch COMPOSITION must not. Slicing `tr` directly, as this loop used to, meant every
        # minibatch contained the same 256 bouts for all 80/250 epochs. `fit_mix` (:118) and
        # `fit_mix_kin` both reshuffle; this was the only trainer in the repo that did not.
        NT = (_noise_tensors(noise_builder, _ep, dth_scale, kin_scales, dev)
              if noise_builder is not None else None)
        tr_ep = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(tr_ep), batch_size):
            ix = tr_ep[i:i + batch_size]
            if NT is not None and noise_mode == "mask":
                m = torch.rand(len(ix), device=dev) < float(noise_frac)
                Xb = torch.where(m.unsqueeze(1), NT[0][ix], Xt[ix])
                azb = torch.where(m, NT[1][ix], az[ix]); lcb = torch.where(m, NT[2][ix], lc[ix])
                ltb = torch.where(m, NT[3][ix], lt_t[ix])
                yab = torch.where(m, NT[4][ix], ya[ix]); ydb = torch.where(m, NT[5][ix], yd[ix])
                ytb = torch.where(m, NT[6][ix], yt[ix])
                opt.zero_grad(); loss = nll_on(Xb, azb, lcb, ltb, yab, ydb, ytb)
            else:
                opt.zero_grad(); loss = nll(ix)
            guarded_step(opt, loss, net.parameters(), stats=_gstats)
        if NT is not None and noise_mode != "mask":       # 'concat': a second, noisy pass
            tr_ep2 = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(tr_ep2), batch_size):
                ix = tr_ep2[i:i + batch_size]
                opt.zero_grad()
                loss = nll_on(*(t[ix] for t in NT))
                guarded_step(opt, loss, net.parameters(), stats=_gstats)
        net.eval()
        with torch.no_grad():
            v = float(nll(va))
            # Training NLL on the SAME criterion, for the train/val curve. One extra forward
            # pass per epoch over the train split -- negligible on GPU, and it is the only way
            # to see the generalisation gap rather than just the early-stopping trace.
            t_ = float(nll(tr))
        hist.append(v)
        hist_tr.append(t_)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    _warn_unstable(_gstats, hist)
    if best_state is not None:
        net.load_state_dict(best_state)
    net = net.to("cpu").eval()
    net._hist_train = list(hist_tr)      # `hist` stays val-only so nothing downstream changes
    return net, hist


# =========================================================================== #
#  numpy rollout
# =========================================================================== #
def _numpy_predictor_joint(model, z=None):
    """Hand-rolled forward pass for the free-run loop (the torch graph is far too slow per bout).

    ``z`` is the per-trajectory random effect, or None. It loads LEVEL-ONLY and enters every head
    additively, so the whole mechanism folds into four bias vectors ONCE here and costs nothing per
    bout. Level-only means: the forward-mixture weight logit for the turn head, and the location
    parameters (not the scales, not the mixture shapes) for the two kinematic heads.
    """
    sd = model.state_dict()
    W1, b1 = sd["body.0.weight"].numpy(), sd["body.0.bias"].numpy()
    W2, b2 = sd["head.weight"].numpy(), sd["head.bias"].numpy()
    Wd, bd = sd["dist_head.weight"].numpy(), sd["dist_head.bias"].numpy()
    Wt, bt = sd["dt_head.weight"].numpy(), sd["dt_head.bias"].numpy()
    nd, ndt = int(model.n_dist), int(model.n_dt)
    _lag_w = None                                    # re_lag: set below when the loading exists
    # depth > 1 (Model 17): the extra relu layers, applied in order after the first
    _deep = [(sd[f"body.{2 * i}.weight"].numpy(), sd[f"body.{2 * i}.bias"].numpy())
             for i in range(1, int(getattr(model, "depth", 1)))]
    # skip path (Model 17): W_skip @ x[slices] added to the turn head's outputs; stashed per bout
    # by `body` (which has x) for `angle_params` (which does not)
    _skip_w = sd["skip.weight"].numpy() if "skip.weight" in sd else None
    _skip_sl = tuple(getattr(model, "skip_slices", None) or ())
    _skip_term = [np.zeros(14, dtype=np.float32)]
    _st = getattr(model, "stim_turn", None)
    _st_args = None
    _st_v2 = None
    _st_step = [0.0]
    _st_call = None                                  # x -> (14-vector turn shift, step shift)
    if _st is not None and getattr(_st, "version", 1) >= 3:
        _st_call = _st.numpy_fn()
    elif _st is not None:
        _st_args = (_st.coef.detach().cpu().numpy().astype(np.float64), _st.sin.cpu().numpy().astype(np.float64),
                    _st.grp.cpu().numpy(), np.asarray(_st.mv_idx), np.asarray(_st.nz_idx))
        if getattr(_st, "v2", False):
            _st_v2 = (np.asarray(_st.rho_idx), _st.lk.cpu().numpy().astype(np.float64),
                      _st.coef_rng.detach().cpu().numpy().astype(np.float64)[_st.grp.cpu().numpy()],
                      _st.ahead.detach().cpu().numpy().astype(np.float64))
        _st_call = lambda x: stim_turn_np(*_st_args, x, v2=_st_v2)  # noqa: E731
    if z is not None and "z_body.weight" in sd:
        # KEEP THE DTYPE. `b1 + (W_z @ z)` promotes float32 -> float64 the moment z is float64, and
        # every subsequent bout is then computed in a different precision from the control's. The
        # divergence is ~1e-8 per bout, which is invisible until it flips an rng.choice and the two
        # trajectories separate completely -- so a z = 0 arm would NOT reproduce the control and the
        # paired ladder would be silently unpaired.
        z = np.asarray(z, np.float32).reshape(-1)
        # BLOCK-STRUCTURED LATENT: turn/dist/dt read the leading `n_core` dimensions, the border
        # gain reads the trailing ones. Without this the dot products below would shape-mismatch
        # the moment block=True, because those loadings are (1, n_core) while z is length m.
        _ncore = int(getattr(model, "n_core", z.shape[0]))

        def _load1(key, core=True):
            """Scalar offset from a loading. The RESHAPE IS NOT COSMETIC: `W @ z` on a
            (1, m) weight returns a length-1 ARRAY, and converting that to a scalar is a
            deprecation under numpy 1.x but a hard TypeError under numpy 2 -- so the same code
            ran on one interpreter and crashed on another. Reduce to a float explicitly."""
            w = sd[key].numpy().reshape(-1)
            zz = z[:_ncore] if core else z[_ncore:]
            if w.shape[0] != zz.shape[0]:          # not block-structured: the loading reads all of z
                zz = z
            return float(np.dot(w, zz))

        b1 = (b1 + sd["z_body.weight"].numpy().reshape(len(b1), -1) @ z).astype(b1.dtype)
        b2 = b2.copy(); b2[0] = b2.dtype.type(b2[0] + _load1("z_turn.weight"))
        if "z_tsize.weight" in sd:
            # re_tsize: the turn-size loading folds into the four turning Gamma SCALE biases (pre-softplus), exactly
            # as the torch path adds it (MDNMixJointRE._turn_with_z)
            _ts = _load1("z_tsize.weight")
            for _k in (7, 9, 11, 13):
                b2[_k] = b2.dtype.type(b2[_k] + _ts)
        bd = bd.copy()
        bd[nd:2 * nd] = (bd[nd:2 * nd] + _load1("z_dist.weight")).astype(bd.dtype)
        bt = bt.copy()
        bt[ndt:2 * ndt] = (bt[ndt:2 * ndt] + _load1("z_dt.weight")).astype(bt.dtype)

        def _loadv(key):
            """Vector offset from a (k, n_core) loading, same block rule as _load1."""
            w = sd[key].numpy()
            zz = z[:_ncore]
            if w.shape[1] != zz.shape[0]:
                zz = z
            return w @ zz

        if "z_dt_sig.weight" in sd:
            # re_disp: the dispersion loading folds into the interval head's WIDTH biases, exactly
            # as the torch path adds it before the softplus (MDNMixJointRE._dt_with_z).
            bt[2 * ndt:3 * ndt] = (bt[2 * ndt:3 * ndt] + _loadv("z_dt_sig.weight")).astype(bt.dtype)
        if "z_dt_w.weight" in sd:
            # re_tail: the component-weight loading folds into the interval head's LOGIT biases
            # (softmax follows), as the torch path adds it.
            bt[:ndt] = (bt[:ndt] + _loadv("z_dt_w.weight")).astype(bt.dtype)
        if "z_lag.weight" in sd:
            # re_lag: mu_dt += (L z) . x[lag cols] is x-dependent, so it cannot fold into a bias;
            # `body` (called once per bout, before the heads) stashes the term for `dt_params`.
            _lag_w = _loadv("z_lag.weight").astype(np.float32)
            _lag_s0, _lag_s1 = model.lag_slice
        if "z_bord.weight" in sd:
            # re_bgain: the per-fish gain on the border INPUT columns folds into W1's border
            # columns once (W1 @ diag(g on slice) == scale those columns), zero per-bout cost.
            # Same clamp as the torch path; dtype preserved for the CRN reasons above.
            _g = float(np.exp(np.clip(_load1("z_bord.weight", core=False), -1.5, 1.5)))
            _s0, _s1 = model.border_slice
            W1 = W1.copy()
            W1[:, _s0:_s1] = (W1[:, _s0:_s1] * _g).astype(W1.dtype)

    def sp(z):
        return np.logaddexp(0.0, z)

    def _sm(lw):
        w = np.exp(lw - lw.max()); return w / w.sum()

    # Head positions. Each closure below takes the conditioners of the channels emitted before it
    # as trailing scalars -- so on the default order the calls stay exactly `body_angle(x)`,
    # `dist_params(b, a_z)`, `dt_params(b, a_z, ld_c)` and every existing caller is unchanged.
    pos = getattr(model, "head_pos", None) or {c: i for i, c in enumerate(DEFAULT_ORDER)}

    def _inp(b, prev, ch):
        if len(prev) != pos[ch]:
            raise ValueError(f"head {ch!r} sits at position {pos[ch]} of {getattr(model, 'order', DEFAULT_ORDER)} "
                             f"and needs that many conditioners; got {len(prev)}")
        # float64 on purpose: the old code wrote `np.concatenate([b, [a_z]])`, and a Python-float
        # list promotes the float32 body to float64 for the whole matmul. Forcing b.dtype here
        # would be "cleaner" and would silently change every archived rollout -- see the CRN note
        # on dtype above. Keep the promotion.
        return np.concatenate([b, np.asarray(prev, float)]) if prev else b

    _lag_term = [0.0]

    def body(x):
        if _lag_w is not None:                       # re_lag: this bout's z x interval-lag term
            _lag_term[0] = float(np.dot(_lag_w, x[_lag_s0:_lag_s1]))
        if _skip_w is not None:
            _skip_term[0] = _skip_w @ np.concatenate([x[a:b] for a, b in _skip_sl])
        if _st_call is not None:
            _o, _st_step[0] = _st_call(x)
            _skip_term[0] = (_skip_term[0] if _skip_w is not None else 0.0) + _o
        h = np.maximum(W1 @ x + b1, 0.0)
        for Wk, bk in _deep:
            h = np.maximum(Wk @ h + bk, 0.0)
        return h

    def angle_params(b, *prev):
        o = W2 @ _inp(b, prev, "a") + b2
        if _skip_w is not None or _st_call is not None:
            o = o + _skip_term[0]
        return (_sm(o[:5]), sp(o[5]) + M.EPS,
                sp(o[6]) + M.ALPHA_MIN, sp(o[7]) + M.EPS, sp(o[8]) + M.ALPHA_MIN, sp(o[9]) + M.EPS,
                sp(o[10]) + M.ALPHA_MIN, sp(o[11]) + M.EPS,
                sp(o[12]) + M.ALPHA_MIN, sp(o[13]) + M.EPS)

    def body_angle(x, *prev):
        """(body activation, angle params). Only meaningful in one call when the angle head is
        first; otherwise pass its conditioners, or call ``body`` and ``angle_params`` separately."""
        b = body(x)
        return b, angle_params(b, *prev)

    def dist_params(b, *prev):
        do = Wd @ _inp(b, prev, "d") + bd
        if _st_v2 is not None:                       # the profile term's step lever (stashed by ``body``)
            do = do.copy(); do[nd:2 * nd] = do[nd:2 * nd] + _st_step[0]
        return _sm(do[:nd]), do[nd:2 * nd], sp(do[2 * nd:3 * nd]) + M.EPS

    def dt_params(b, *prev):
        to = Wt @ _inp(b, prev, "t") + bt
        if _lag_w is not None:
            to[ndt:2 * ndt] += _lag_term[0]
        return _sm(to[:ndt]), to[ndt:2 * ndt], sp(to[2 * ndt:3 * ndt]) + M.EPS

    # `body` and `angle_params` ride along as attributes rather than extra return values, so the
    # arity stays 3 for smoke_bgain / smoke_rollout_aware and the fishre family.
    body_angle.body = body
    body_angle.angle_params = angle_params

    # RAW PER-ROW HEADS (multi-group scenes, ``generate_mix_joint_loop(multi_rule=...)``, report Model 31). The same
    # arithmetic as body / angle_params / dist_params / dt_params, but (1) nothing is stashed, because several rows are
    # evaluated per bout, and (2) each head's output is returned BEFORE its link function, so the outputs of several rows
    # can be combined and linked once. Pure additions: the closures above are untouched.
    def _hid(x):
        h = np.maximum(W1 @ x + b1, 0.0)
        for Wk, bk in _deep:
            h = np.maximum(Wk @ h + bk, 0.0)
        return h

    def _skip_of(x):
        return None if _skip_w is None else _skip_w @ np.concatenate([x[a:b] for a, b in _skip_sl])

    def _lag_of(x):
        return None if _lag_w is None else float(np.dot(_lag_w, x[_lag_s0:_lag_s1]))

    def _a_raw(h, add, *prev):
        o = W2 @ _inp(h, prev, "a") + b2
        return o if add is None else o + add

    def _a_link(o):
        return (_sm(o[:5]), sp(o[5]) + M.EPS,
                sp(o[6]) + M.ALPHA_MIN, sp(o[7]) + M.EPS, sp(o[8]) + M.ALPHA_MIN, sp(o[9]) + M.EPS,
                sp(o[10]) + M.ALPHA_MIN, sp(o[11]) + M.EPS,
                sp(o[12]) + M.ALPHA_MIN, sp(o[13]) + M.EPS)

    def _d_raw(h, *prev):
        return Wd @ _inp(h, prev, "d") + bd

    def _d_link(do):
        return _sm(do[:nd]), do[nd:2 * nd], sp(do[2 * nd:3 * nd]) + M.EPS

    def _t_raw(h, lag, *prev):
        to = Wt @ _inp(h, prev, "t") + bt
        if lag is not None:
            to[ndt:2 * ndt] += lag
        return to

    def _t_link(to):
        return _sm(to[:ndt]), to[ndt:2 * ndt], sp(to[2 * ndt:3 * ndt]) + M.EPS

    body_angle.raw = dict(hid=_hid, skip=_skip_of, lag=_lag_of, st=_st_call, st_step=_st_v2 is not None,
                          a=_a_raw, a_link=_a_link, d=_d_raw, d_link=_d_link, t=_t_raw, t_link=_t_link)
    return body_angle, dist_params, dt_params


# --------------------------------------------------------------------------- #
# Multi-group scenes (report Model 31)
# --------------------------------------------------------------------------- #
# The recorded programme never shows two dot groups at once (0 % of 39,542 stimulus bouts), so how a trained network
# responds to two is NOT learned -- it is fixed by the rule that turns two groups into the network's input. Four rules,
# all on the same trained weights (scripts/multi_stim.py):
#   'effects'  each group alone through the whole network, plus a no-dots baseline; every head's output (before its link
#              function) = baseline + sum of the groups' effects. Every row is a valid single-group (or no-dots) input.
#   'add'      one row whose stimulus columns are the SUM of the groups' columns (first-layer drives add; cos may be > 1).
#   'avg'      one row whose stimulus columns are the MEAN of the groups' columns.
#   'nearest'  one row from the nearest group only (winner-take-all).
# In 'add' / 'avg' the aligned-and-far flag of the combined row is the MAX over groups ('nearest' uses the nearest group's
# own row unchanged; 'effects' each group's own row); the block clock is shared (time since
# the scene's onset; 0 in the no-dots baseline row, as on every no-dot bout in training), and the profile term (pf) is
# evaluated on each group's own row: summed for 'effects' / 'add', averaged for 'avg', the nearest group's for
# 'nearest'. With ONE group every rule gives exactly the standard row, so the standard simulator is reproduced bit for
# bit (scripts/multi_stim.py --validate).
MULTI_RULES = ("effects", "add", "avg", "nearest")
# HEADING-WEIGHTED rule (2026-10-15, Final_report_extrapolations_2): 'heading<kappa>', e.g. 'heading2'. As 'effects', each
# group alone through the network against the no-dots baseline, but the group effects are combined with weights that sum
# to one, w_i = softmax_i(kappa * cos(phi_i)), phi_i the group's bearing from the fish's heading: groups nearly
# straight ahead weigh about equally (the effects are averaged), and of two groups far apart the one more ahead dominates,
# which turns the fish further toward it -- the positive feedback behind a switch from averaging to choosing
# (Sridhar et al. 2021). kappa = 0 is the plain average of the effects; with ONE group w = 1 and the row is the standard one.


def heading_kappa(rule):
    """kappa of a 'heading<kappa>' rule, else None."""
    if isinstance(rule, str) and rule.startswith("heading"):
        return float(rule[len("heading"):])
    return None


def heading_weights(phis, kappa):
    c = kappa * np.cos(np.asarray(phis, float))
    c = c - np.max(c)
    w = np.exp(c)
    return w / w.sum()


class MultiBout:
    """One bout of a multi-group scene: the hidden activations of its rows and how to combine their head outputs.
    ``base`` True = rows[0] is the no-dots baseline and rows[1:] the groups ('effects'); else one combined row.
    ``w`` (heading rule) = the groups' weights: baseline + sum_i w_i (group_i - baseline) instead of the plain sum."""

    __slots__ = ("raw", "hs", "adds", "lags", "base", "w")

    def __init__(self, raw, rows, adds, base, w=None):
        self.raw = raw
        self.hs = [raw["hid"](r) for r in rows]
        self.adds = adds
        self.lags = [raw["lag"](r) for r in rows]
        self.base = bool(base)
        self.w = None if w is None else np.asarray(w, float)

    def _comb(self, outs):
        if not self.base:
            return outs[0]
        if self.w is not None:                        # heading rule: weighted effects, weights summing to one
            if len(self.w) == 1:
                return outs[1]                        # one group: exactly its own row
            o = outs[0].copy()
            for wi, q in zip(self.w, outs[1:]):
                o = o + wi * (q - outs[0])
            return o
        o = outs[1]                                   # sum of the groups' outputs minus (G - 1) baselines: the
        for q in outs[2:]:                            # baseline plus every group's effect. G = 1 -> outs[1] exactly.
            o = o + q
        G = len(outs) - 1
        if G > 1:
            o = o - (G - 1) * outs[0]
        return o

    def angle(self, *prev):
        return self.raw["a_link"](self._comb([self.raw["a"](h, a, *prev) for h, a in zip(self.hs, self.adds)]))

    def dist(self, *prev):
        return self.raw["d_link"](self._comb([self.raw["d"](h, *prev) for h in self.hs]))

    def dt(self, *prev):
        return self.raw["t_link"](self._comb([self.raw["t"](h, g, *prev) for h, g in zip(self.hs, self.lags)]))


def multi_bout(raw, rule, group_rows, base_row=None, comb_row=None, weights=None):
    """The ``MultiBout`` of one bout. ``group_rows`` = each group's own full input row (block clock on, its own
    aligned-and-far flag); ``base_row`` = the no-dots row ('effects'); ``comb_row`` = the combined row ('add' / 'avg' /
    'nearest'). The turn head's additive term (skip path + pf) follows the rule -- see MULTI_RULES."""
    def _add(row_net, pf_rows, mean=False):
        sk = raw["skip"](row_net)
        if raw["st"] is None:
            return sk
        p = raw["st"](pf_rows[0])[0]
        for r in pf_rows[1:]:
            p = p + raw["st"](r)[0]
        if mean:
            p = p / len(pf_rows)
        return (sk if sk is not None else 0.0) + p    # same order as the standard path's stash: skip + pf

    if rule == "effects":
        rows = [base_row] + list(group_rows)
        return MultiBout(raw, rows, [_add(r, [r]) for r in rows], base=True)
    if heading_kappa(rule) is not None:
        rows = [base_row] + list(group_rows)
        return MultiBout(raw, rows, [_add(r, [r]) for r in rows], base=True, w=weights)
    if rule == "nearest":
        return MultiBout(raw, [comb_row], [_add(comb_row, [comb_row])], base=False)
    if rule in ("add", "avg"):
        return MultiBout(raw, [comb_row], [_add(comb_row, list(group_rows), mean=(rule == "avg"))], base=False)
    raise ValueError(f"multi_rule {rule!r} not in {MULTI_RULES}")


def generate_mix_joint_loop(model, rec, cfg_feat, rng, closed_loop=True, guardrail=True, eps=1e-3,
                            stim_loop=True, keep_features=False,
                            stim_provider=None, stop_time_s=None, block_seed=None, multi_rule=None):
    """Free-run one trajectory. Same contract as ``MX.generate_mix_kin_loop`` so the calibrators
    and the harness can use either.

    Closed-loop from the model's own pose: the border covariate and (when ``stim_loop``) the
    shoal bearing/range are recomputed each bout from the generated position. The turn history and
    BOTH kinematic history lags live in the same feature row, matching training.

    ROLLOUT ON A *NEW* STIMULUS PROGRAM (``stim_provider`` / ``stop_time_s``, both default off):
    ``rec.dot_cx[t]`` is indexed by BOUT, and is materialised as an array before this loop starts,
    so a fixed stimulus timeline cannot be pre-baked into it -- the bout times are themselves
    generated here. ``stim_provider(t, time_s, x, y, h) -> (dot_cx, dot_cy, moving, n_dots)`` is
    called instead, with the live clock and pose, so the program can be resolved online (see
    ``scripts/stim_rollout.py``). ``stop_time_s`` ends the run at the last bout whose ONSET is
    still inside the programme and truncates every returned array to that length.

    Both are pure additions: with ``stim_provider=None`` and ``stop_time_s=None`` the reads and
    the RNG draw order are exactly as before, so output stays bit-identical (asserted by
    ``scripts/smoke_stim_program.py``).

    MULTI-GROUP SCENES (``multi_rule`` in MULTI_RULES, default None = off): the provider's ``groups(t, time_s, x, y, h)``
    returns one ``(centroid x, centroid y, moving, n_dots)`` per dot group on screen, and the network's input is built
    from them by the rule (see MULTI_RULES). Extra outputs ``grp_cx`` / ``grp_cy`` / ``grp_phi`` (n, 4) and ``grp_n``.
    """
    n = rec.n
    # BLOCK-START FORCING (the profile fine-tune, scripts/profile_finetune.py): a sliced record may carry
    # `_force_P` = how many of its leading bouts to REPLAY (the real turn, step and interval, and the real
    # pose after each) before free-running, `_rp_off` = its first bout's real run position (run clock and
    # lag-valid columns), `_kdt_lev0` = the real running interval level there. All absent -> 0 / None and
    # every existing path is byte-identical.
    _force_P = int(getattr(rec, "_force_P", 0) or 0)
    _rp_off = int(getattr(rec, "_rp_off", 0) or 0)
    _trv_f = MB._travel_ref(rec) if _force_P else None
    K = cfg_feat["K"]; Kk = int(cfg_feat.get("K_kin", 3))
    # kin_dt_lags=False: the interval lags are NOT fed back (MB._kin_history_block) -- the
    # interval's persistence then comes from the latent and the state only. See the border
    # module's block for the measurement that motivates it.
    _kin_dt = bool(cfg_feat.get("kin_dt_lags", True))
    # 'centred': interval lags as deviations from the fish's OWN running level (MB.kin_dt_level),
    # the EMA maintained here on the EMITTED log intervals exactly as the builder does on the real.
    _kin_dt_c = (cfg_feat.get("kin_dt_lags", True) == "centred")
    _kdt_a = 1.0 / max(float(cfg_feat.get("kin_dt_ema", MB.KIN_DT_EMA)), 1.0)
    dth_scale = cfg_feat["dth_scale"]
    dm, ds, tm, ts = cfg_feat["kin_scales"]
    dist_hi, dt_hi = cfg_feat.get("kin_clip", (400.0, 600.0))
    # TRUNCATED interval head (MX._dt_logpdf): draws are made INSIDE the bound -- component weights
    # conditioned on it, then an inverse-CDF draw of the truncated normal in log space -- so there
    # is no mass beyond the cap and no pile-up at the clip. Consumes one extra uniform per bout,
    # so a truncated arm is not CRN-paired bout-for-bout with an untruncated one.
    _dt_tr = cfg_feat.get("dt_trunc")
    _log_tr = float(np.log(_dt_tr)) if _dt_tr else None
    if _log_tr is not None:
        from scipy.special import ndtr as _ndtr, ndtri as _ndtri
        dt_hi = float(_dt_tr)          # the bound IS the cap; the q99.9 clip would pile up below it
    stim = M._stim_block(rec, cfg_feat["stim_mode"], cfg_feat["rich_scales"],
                         phi_ref=cfg_feat.get("phi_ref", "ori"))
    use_border = cfg_feat.get("use_border")
    tf_border = MB._border_block(rec, cfg_feat) if use_border else np.zeros((n, 0))
    # K_stim block: pure function of the record's own stimulus programme, so it is indexed by
    # bout exactly as the teacher-forced stim block is. (The whole rollout is indexed by bout
    # number rather than wall-clock -- the known clock_ratio caveat -- so this is consistent with
    # how `moving` and `n_dots` are already fed.)
    tf_shist = (MB._stimhist_block(rec, cfg_feat) if int(cfg_feat.get("K_stim", 1)) > 1
                else np.zeros((n, 0)))
    recompute_stim = (stim_loop and closed_loop
                      and cfg_feat["stim_mode"] in MX.ONLINE_STIM_MODES)
    if stim_provider is not None and not recompute_stim:
        # Fail loudly: silently teacher-forcing the record's stimulus while the caller believes a
        # new programme is driving the fish would look like a working tool and be a replay.
        raise ValueError(
            "stim_provider needs the online stimulus path: require stim_loop=True, "
            f"closed_loop=True and stim_mode in {MX.ONLINE_STIM_MODES}; got "
            f"stim_loop={stim_loop}, closed_loop={closed_loop}, "
            f"stim_mode={cfg_feat['stim_mode']!r}.")
    if stim_provider is not None and tf_shist.shape[1]:
        # Same reasoning as above: the K_stim channels describe the RECORD's block programme, so
        # replaying them under a new programme would be a silent replay of the old one.
        raise ValueError("stim_provider is incompatible with K_stim > 1: the stimulus-memory "
                         "block would still describe the record's own programme. Resolve the "
                         "previous onsets from the provider's programme before enabling both.")
    if recompute_stim:
        dot_cx_a = np.asarray(getattr(rec, "dot_cx", np.full(n, np.nan)), float)
        dot_cy_a = np.asarray(getattr(rec, "dot_cy", np.full(n, np.nan)), float)
        mv_a = np.asarray(rec.moving, float); phi_gen = np.full(n, np.nan)
        nd_a = np.asarray(getattr(rec, "n_dots", np.zeros(n)), float)
        dcx_used = np.full(n, np.nan); dcy_used = np.full(n, np.nan)
    # PER-DOT population code (Model 26): the row needs every active dot's position, from the
    # provider's `dots()` when there is one and from the record's per-dot arrays otherwise.
    _pop = cfg_feat["stim_mode"] in M.POP_MODES
    _pop_kw = M.pop_kw(cfg_feat["stim_mode"]) if cfg_feat["stim_mode"] in M.POP_MODES else {}
    # DOT DRIFT (Model 31, 'shoal_popld'): the previous bout's dot slots and the time they were read, in
    # the clock the positions come from -- the record's frames on the replay path and under the
    # record-anchored provider, the generated clock under the programme provider (``last_time_s``).
    _pop_drift = bool(_pop_kw.get("drift"))
    _prev_dots = None
    _fo_rec = getattr(rec, "frame_onset", None)
    _dxs = _dys = None
    if _pop and recompute_stim:
        if stim_provider is not None and not callable(getattr(stim_provider, "dots", None)):
            raise ValueError("stim_mode='shoal_pop' needs a provider with a dots() method")
        if stim_provider is None:
            _dxs = getattr(rec, "dot_xs", None); _dys = getattr(rec, "dot_ys", None)
            if _dxs is None:
                raise RuntimeError("stim_mode='shoal_pop' needs rec.dot_xs / rec.dot_ys (add_dot_positions.py)")
            _dxs = np.asarray(_dxs, float); _dys = np.asarray(_dys, float)
    # WARM-UP RE-ANCHORING (profile fine-tune, ``PF.block_starts(reanchor=True)``): a sliced record may carry
    # `_reanchor_at` = P, the block-onset bout of a FREE-WARM-UP start. Its replayed dot track was placed around the REAL
    # fish's onset pose, but the simulated fish reaches bout P wherever its own warm-up took it (measured: 32 mm from the
    # dots against 17 for real starts), so from bout P on the track is moved rigidly onto the simulated pose
    # (MB.reanchor_track) -- the rig's rule. The real pose is the one a forced replay would place the fish on at P.
    # Absent -> byte-identical.
    _ra = getattr(rec, "_reanchor_at", None)
    _ra = None if _ra is None else int(_ra)
    # REPULSION TERM (see ``repulsion_shifts``): per-dot positions every bout -- the provider's dots() or the record's
    _rep = cfg_feat.get("repulsion") or None
    _rep_cur = [0.0, 0.0]                                    # this bout's (turn shift, step log-mean shift)
    _rdx = _rdy = None
    if _rep:
        _no_dots = multi_rule is not None and not callable(getattr(stim_provider, "dots", None))
        if _no_dots or _ra is not None or _pop or not recompute_stim:
            raise NotImplementedError("repulsion term: closed-loop centroid codes only (multi-group scenes only with a "
                                      "provider giving the individual neighbours via dots(); no warm-up re-anchoring, "
                                      "per-dot codes or replayed stimulus rows)")
        if stim_provider is not None and not callable(getattr(stim_provider, "dots", None)):
            raise ValueError("repulsion term: the stimulus provider needs a dots() method (individual dots)")
        if stim_provider is None:
            _rdx, _rdy = getattr(rec, "dot_xs", None), getattr(rec, "dot_ys", None)
            if _rdx is None:
                raise RuntimeError("repulsion term: the record carries no per-dot positions (rec.dot_xs)")
            _rdx = np.asarray(_rdx, float); _rdy = np.asarray(_rdy, float)
    # GROUP REPULSION TERM (see ``grouprep_shifts``): each visible group's nearest member every bout -- the multi-group
    # provider's group_nearest(), the single-group provider's dots(), or the record's own dot positions
    _grp_rep = cfg_feat.get("grouprep") or None
    _gdx = _gdy = None
    if _grp_rep:
        if _rep:
            raise ValueError("cfg_feat: 'grouprep' and 'repulsion' are exclusive")
        if _ra is not None or _pop or not recompute_stim:
            raise NotImplementedError("grouprep term: closed-loop centroid codes only (no warm-up re-anchoring, "
                                      "per-dot codes or replayed stimulus rows)")
        if multi_rule is not None and not callable(getattr(stim_provider, "group_nearest", None)):
            raise ValueError("grouprep term: a multi-group provider needs group_nearest()")
        if multi_rule is None and stim_provider is not None and not callable(getattr(stim_provider, "dots", None)):
            raise ValueError("grouprep term: the stimulus provider needs a dots() method (individual dots)")
        if stim_provider is None:
            _gdx, _gdy = getattr(rec, "dot_xs", None), getattr(rec, "dot_ys", None)
            if _gdx is None:
                raise RuntimeError("grouprep term: the record carries no per-dot positions (rec.dot_xs)")
            _gdx = np.asarray(_gdx, float); _gdy = np.asarray(_gdy, float)
    _coll_px = float(cfg_feat.get("collision_mm") or 0.0) * REP_PX_PER_MM
    if _coll_px and not callable(getattr(stim_provider, "neighbours_xy", None)):
        raise ValueError("collision_mm: the stimulus provider needs neighbours_xy()")
    if _ra is not None:
        if not recompute_stim or stim_provider is not None or _pop or int(cfg_feat.get("stim_lags") or 0):
            raise NotImplementedError("_reanchor_at: centroid stimulus replayed from the record only "
                                      "(no provider, no per-dot code, no stimulus lags)")
        dot_cx_a = dot_cx_a.copy(); dot_cy_a = dot_cy_a.copy()        # never write into the record
        _trv_ra = MB._travel_ref(rec)
    # PER-FISH RANDOM EFFECT. One draw per TRAJECTORY from the fitted prior -- the honest "new fish"
    # claim, and the only option that cannot leak: the evaluated fish is held out, so it has no
    # fitted embedding. Sigma is diagonal by construction, so a scaled standard normal is the draw.
    #
    # SPAWNED STREAM, and this is load-bearing. generative_metrics keys rollouts on (seed, fish, rep)
    # with NO model index, deliberately, so replicate r of model A and model B are paired -- which
    # requires every arm to consume the same variates from `rng` in the same order. Drawing z from
    # `rng` itself would desynchronise every emission draw after it and the paired t-test would
    # still print RESOLVED while being invalid. SeedSequence.spawn bumps the parent's child counter
    # but does not advance its stream, so the emission draws stay byte-identical to the control.
    # Asserted in smoke_fishre.py, not assumed.
    _re = cfg_feat.get("fish_re")
    z_re = None
    u_rad_re = 0.0
    if _re is not None:
        _sd_re = np.asarray(_re["sd"], float).reshape(-1)
        z_re = rng.spawn(1)[0].normal(size=_sd_re.shape[0]) * _sd_re
        if _re.get("z_fixed") is not None:
            # TWIN MODE (in-sample checkpoints only, see fishre.z_twin): a KNOWN fish's fitted
            # embedding replaces the prior draw. The spawn above is still consumed, so every
            # other stream is byte-identical to the prior-draw rollout of the same (seed, fish, rep).
            z_re = np.asarray(_re["z_fixed"], float).reshape(_sd_re.shape[0])
        if "rad_sd" in _re:
            # RADIAL individuality for the fishre family (rad_indiv): a per-trajectory
            # multiplicative swing on the containment bias, mirroring indiv's u_rad -- an
            # additive offset leaks into the turn marginal and can flip a fish outward.
            # SPAWN ORDER IS LOAD-BEARING: z stays child 0 (a cfg without the key draws exactly
            # as before), this is child 1, and the main emission stream is never touched, so the
            # CRN pairing against the no-rad control arm holds bout for bout.
            # Gate on KEY PRESENCE, not truthiness: _search_rad_indiv's base pass sets
            # rad_sd=0.0 and shares one rng across its fish, so both passes must consume the
            # SAME spawn count per rollout -- a truthiness gate shifts every later fish's z_re
            # to a different spawn child and the two-point solve reads z-resampling noise.
            # rad_sd=0.0 still yields u_rad_re == 0.0, i.e. a bit-identical rollout.
            u_rad_re = float(rng.spawn(1)[0].normal() * float(_re["rad_sd"]))
    # PER-FISH GATE OFFSET (rad_gate): each trajectory shifts the containment gate u0 by a fixed
    # draw -- a personal HOME RADIUS, moving where wall avoidance engages rather than how hard.
    # Built because the multiplicative swing (u_rad/rad_sd) is measured INERT on graded bulk=0
    # biases: scaling a bias that only acts in the outer ring cannot move per-fish MEAN r/R,
    # while moving the gate 0.60->0.45 shifts it by ~0.03. Same CRN contract as rad_sd: spawned
    # stream (never the main one), key-presence gated so the calibrator's base pass stays paired.
    u_gate = 0.0
    _rg = cfg_feat.get("rad_gate")
    if _rg is not None and "sd" in _rg:
        u_gate = float(rng.spawn(1)[0].normal() * float(_rg["sd"]))
    # SLOW TIMING STATE. Unlike every latent above it, this one EVOLVES: s_t = phi s_{t-1} +
    # sqrt(1-phi^2) eps, entering the interval head's location as mu_tau + lam*s. That is the
    # whole point -- a per-trajectory constant shifts every window equally and so contributes
    # nothing to rate_acf_1min / quiesc20 / dt dispersion, which is exactly what the fishre latent
    # cannot fix. Drawn from a SPAWNED stream (same contract as rad_indiv/rad_gate) so a control
    # arm without the key consumes the identical main-stream variates and stays CRN-paired.
    _dtl = cfg_feat.get("dt_latent")
    s_lam = s_phi = 0.0
    s_lam_c = 0.0
    s_state = 0.0
    s_rng = None
    if _dtl:
        # `g_lam` / `lam_c`: the rollout-fitted amplitude gain and its log-normal re-centring, set by
        # MX._search_drift_gain when dt_latent is the interval chain of the two-state design (absent
        # -> 1 / 0, bit-identical to every archived dt_latent arm).
        s_lam = float(_dtl["lam"]) * float(_dtl.get("g_lam", 1.0)); s_phi = float(_dtl["phi"])
        s_lam_c = float(_dtl.get("lam_c", 0.0))
        s_rng = rng.spawn(1)[0]
        s_state = float(s_rng.normal())          # stationary N(0,1), matching tauchen's prior
    # DRIFT. The same AR(1) machinery as dt_latent, but loading on ALL FOUR channels -- turn logit,
    # distance location, interval location, and the containment-bias scale. The last one is why this
    # exists: every previous individuality mechanism is static per trajectory and therefore cannot
    # move the across-window autocorrelation at all (real radial +0.377, generated -0.035). Drawn
    # from a SPAWNED stream, same contract as rad_indiv/rad_gate/dt_latent, so a control arm without
    # the key consumes identical main-stream variates and stays CRN-paired.
    _dr = cfg_feat.get("drift")
    d_turn = d_dist = d_dt = d_bord = 0.0
    d_dist_c = d_dt_c = 0.0
    d_phi = 0.0
    d_state = 0.0
    d_rng = None
    if _dr:
        if _dtl and abs(float(_dr.get("dt", 0.0))) > 1e-12:
            raise ValueError("drift and dt_latent both set: two AR(1) states competing for the "
                             "same variance. drift already loads on the interval head. (The "
                             "two-state design sets drift['dt_off'], so its dt loading is 0.)")
        # `gain` scales the LIKELIHOOD-FITTED loadings so the rollout search can put each channel's
        # across-window acf on target; `dist_c`/`dt_c` are the matching re-centring constants.
        # THE RE-CENTRING IS NOT OPTIONAL, for exactly the reason apply_loading_gain documents: the
        # kinematic heads are log-normal, so a zero-mean state of unit variance carries into the
        # ARITHMETIC mean as E[exp(d s)] = exp(d^2/2). Training absorbs that at the fitted d; scaling
        # d -> g d afterwards inflates every generated travel and interval by exp((g^2-1) d^2/2)
        # with nothing to cancel it -- which is how the fishre gain destroyed dist_w1 (3.04 -> 6.78)
        # the first time. The turn loading needs no analogue: it tilts a logit, so the softmax bounds
        # it and it cannot inflate a marginal without bound.
        # PER-CHANNEL gains. A single shared gain cannot serve both kinematic channels: measured on
        # rej, dt and distance reach their targets at different loadings, and the one gain that put
        # dt on target left distance at +0.118 against +0.253. They are near-independent (each
        # channel's across-window acf responds to its own loading), so they are searched separately.
        d_turn = float(_dr.get("turn", 0.0))
        d_dist = float(_dr.get("dist", 0.0)) * float(_dr.get("g_dist", 1.0))
        d_dt = float(_dr.get("dt", 0.0)) * float(_dr.get("g_dt", 1.0))
        d_bord = float(_dr.get("bord", 0.0))
        d_dist_c = float(_dr.get("dist_c", 0.0)); d_dt_c = float(_dr.get("dt_c", 0.0))
        d_phi = float(_dr["phi"])
        d_rng = rng.spawn(1)[0]
        d_state = float(d_rng.normal())          # stationary N(0,1), matching the chain's prior
    body_angle, dist_params, dt_params = _numpy_predictor_joint(model, z=z_re)
    cx, cy, R = rec.dish_cx, rec.dish_cy, rec.dish_R
    # Initial heading: in psi_ref='travel' the whole rollout runs in the travel frame, so seed it
    # with theta_{-1} (exact from the first bout's dtheta), NOT f_ori -- 10.3% of bouts carry an
    # uncorrected 180-deg body-axis flip, which would start one rollout in ten facing backwards.
    h0 = (float(MB._travel_ref(rec)[0]) if cfg_feat.get("psi_ref") == "travel"
          else float(rec.f_ori_rad[0]))
    x, y, h = float(rec.x_px[0]), float(rec.y_px[0]), h0
    dth = np.zeros(n); dist = np.zeros(n); dtg = np.zeros(n)
    xs = np.empty(n); ys = np.empty(n)
    ld_hist = np.zeros(n); lt_hist = np.zeros(n)
    _kdt_lev = float(getattr(rec, "_kdt_lev0", tm))                       # the running log-interval level (centred lags)
    # Running wall-clock. t_now is ALWAYS the onset time of the bout about to be generated, so it
    # is what a stimulus programme must be sampled at. Costs one float add per bout and consumes
    # no RNG, so the default path is unaffected.
    t_now = 0.0; t_onset = np.zeros(n); n_done = n
    wall_stats = {}
    Xrec = (np.zeros((n, int(model.body[0].in_features)), dtype=np.float32)
            if keep_features else None)
    # POLICY-GRADIENT HANDLES (keep_features only; pure recording, no RNG consumed, so the default
    # path stays bit-identical). A rollout-aware objective must differentiate the log-density of the
    # value the emission ACTUALLY drew, and that is NOT what the sample dict reports: ``dtheta`` is
    # post-containment-bias, ``distance_px`` is post-vigor, ``dt_s`` is post-vigor AND post-dt_snap.
    # Scoring those evaluates the mixture at a point it never sampled -- a silently wrong score
    # function, which is one of the defects that made rollout_finetune.py unusable.
    # See scripts/rollout_aware.py.
    dth_base = np.zeros(n) if keep_features else None
    dist_base = np.zeros(n) if keep_features else None
    dt_base = np.zeros(n) if keep_features else None
    s_rec = np.zeros(n) if keep_features else None
    d_rec = np.zeros(n) if keep_features else None
    cb_cfg = cfg_feat.get("centre_bias", 0.0)
    # SHALLOW-BORDER REJECTION (key absent => rd0 is None => the loop below is byte-identical to
    # every archived arm, which is the only bit-identity guarantee offered: rejection consumes a
    # VARIABLE number of variates, so a reject arm can never be CRN-paired bout-for-bout against
    # a non-reject one. The ladder pairs at replicate level, which is unaffected.)
    _wr = cfg_feat.get("wall_reject")
    rd0 = rd1 = None
    rmax = 0
    R_g = R                    # radius the GUARDRAIL acts at (the border FEATURES always use R)
    if _wr:
        rd0 = float(_wr.get("d0", MB.WALL_REJECT_DEFAULT["d0"]))
        rd1 = float(_wr.get("d1", MB.WALL_REJECT_DEFAULT["d1"]))
        rmax = max(int(_wr.get("max_draws", MB.WALL_REJECT_DEFAULT["max_draws"])), 1)
        # THE HARD FLOOR. Rejection SHAPES the distribution across the ramp band, but on its own
        # it can only make d < d0 improbable, not impossible -- exhaust the draw budget and
        # something has to be emitted. So the guardrail moves inward to the forbidden-ring
        # boundary and does the same job there it already does at the physical wall: 'reflect'
        # preserves the step-length marginal and leaves the heading inward, unlike a clip.
        # Note the FEATURES still see the true R: the fish's distance to the real wall is real,
        # and only its reachable set is smaller.
        R_g = R * (1.0 - rd0 / (0.5 * MB.DISH_MM))
        _u0 = np.hypot(x - cx, y - cy) / R_g
        if _u0 > 1.0:
            # Seeded inside the ring (real fish do reach 0.5 mm on ~0.02% of bouts, and those
            # are the tracking-glitch bouts anyway). Pull the seed onto the boundary, or
            # _ray_to_wall returns 0 and the trajectory freezes in place for the whole rollout.
            x = cx + (x - cx) / _u0 * (1.0 - 1e-9)
            y = cy + (y - cy) / _u0 * (1.0 - 1e-9)
    # HOME RADIUS (Model 22). A per-trajectory preferred radius r/R that the shallow-border rejection
    # ALSO serves: a rim-accepted bout that lands farther from the home radius than the fish is now,
    # and outside a tolerance band around it, is redrawn with probability `pull`. The rim ramp is
    # unchanged and tested first, so the wall band this rule sits inside is exactly rej's. Keys live
    # INSIDE wall_reject so every consumer that merges the rej band (model_figures.fold_models, the
    # harness's cfg_override) carries them: home_mu / home_sd = the real population of per-fish
    # median r/R (one draw per trajectory -- a NEW fish's home), home_band = the no-pull zone,
    # home_tau (bouts) drifts the home as an AR(1) so the fish lingers in an annulus and then moves
    # (None = static home). Built because occupancy is set by the rejection band once rej is on, so
    # every lever on the wall response inside the turn law (rad_indiv, rad_gate, re_bgain) was inert.
    # Spawned stream, key-presence gated: a config without `pull` consumes identical main-stream
    # variates and is bit-identical to rej; pull = 0.0 draws no extra variate either.
    hm_pull = 0.0; hm_mu = 0.0; hm_sd = 0.0; hm_band = 0.0; hm_phi = 1.0
    hm_state = 0.0; hm_rng = None
    if _wr and _wr.get("pull") is not None:
        hm_pull = float(_wr["pull"])
        hm_mu = float(_wr.get("home_mu", 0.45)); hm_sd = float(_wr.get("home_sd", 0.0))
        hm_band = float(_wr.get("home_band", 0.10))
        hm_rng = rng.spawn(1)[0]
        hm_state = hm_mu + hm_sd * float(hm_rng.normal())
        _htau = _wr.get("home_tau")
        hm_phi = float(np.exp(-1.0 / float(_htau))) if _htau else 1.0
    # KIN_FEEDBACK: shrink the kinematic history ON READ. The block is z-scored (mean 0), so a
    # factor < 1 pulls every lag toward the population mean, which rescales the effective AR
    # coefficient of the free-running loop and damps the exposure-bias over-persistence.
    # NOTE this model differs from the kin family: those columns feed the SHARED body, so the
    # shrinkage reaches the turn head too. Watch the turn panel metrics, not just the kinematic ones.
    # LAG-VALIDITY and BLOCK-CLOCK columns, appended by MB.build_feature_matrix AFTER the kin
    # block. Both are pure functions of the bout index and the clock -- not of the generated
    # behaviour -- so they are precomputed here rather than rebuilt per bout.
    #   lag_valid: a rollout is ONE run, so run_pos == t and the scalar is min(t, K)/K.
    #   clock:     time into the current stimulus block on the fixed 110 s grid, 0 during a
    #              pause, matching MB._clock_block. exp(-t/tau) or the 0->1 ramp.
    _lv_mode = str(cfg_feat.get("lag_valid", "") or "").lower()
    _clk_mode = str(cfg_feat.get("clock_mode", "") or "").lower()
    _lagv_a = None
    if _lv_mode == "scalar":
        _lagv_a = (np.minimum(np.arange(n) + _rp_off, K) / max(K, 1)).astype(np.float32).reshape(n, 1)
    elif _lv_mode == "mask":
        _lagv_a = (np.arange(n)[:, None] >= np.arange(1, K + 1)[None, :]).astype(np.float32)
    _clk_taus = MB.clock_taus(cfg_feat)                   # one exp column per timescale
    # RUN CLOCK (MB._runclock_block): a rollout is one run, so run_pos == t -> exp(-t / tau).
    _rc_tau = float(cfg_feat.get("run_clock_tau") or 0.0)
    _rc_a = (np.exp(-(np.arange(n) + _rp_off) / _rc_tau).astype(np.float32).reshape(n, 1)
             if _rc_tau > 0 else None)
    # ALIGNED-AND-FAR feature (MB._align_block): range x cos(bearing) for moving dots. From the
    # generated pose and the dots the bout is shown (the same phi / range `_dot_stim_row` uses) on
    # the closed-loop path, from the record on the teacher-forced-stimulus path.
    _al_on = bool(cfg_feat.get("stim_align"))
    _al_rec = MB._align_block(rec, cfg_feat)[:, 0] if _al_on else None
    # STATIC-dot bearing block (MB._static_block / M.static_channels), same two paths as align.
    _st_on = bool(cfg_feat.get("stim_static"))
    _st_rec = MB._static_block(rec, cfg_feat) if _st_on else None
    # static block x the slowest block clock (MB._static_clock_block): the clock value comes from
    # `_clock_of` on this bout, so the product is exact on both stimulus paths.
    _sc_on = bool(cfg_feat.get("stim_static_clock"))
    if _sc_on and (not _st_on or _clk_mode not in ("linear", "exp")):
        raise ValueError("stim_static_clock needs stim_static and a block clock")
    # static block x the MID-BLOCK BUMP (MB._static_bump_block): the bump reads the block time the
    # clock closure computes for this bout (`_clk_state['tb']`, 0 on no-dot bouts).
    _sb_taus = cfg_feat.get("stim_static_bump")
    _sb_on = bool(_sb_taus)
    if _sb_on and (not _st_on or _clk_mode not in ("linear", "exp")):
        raise ValueError("stim_static_bump needs stim_static and a block clock")
    # ONSET x BEARING (MB._onset_block, Model 25): the first clock column of THIS bout (from the
    # clock closure, so it is exact on both stimulus paths) x [sin, cos] of the bearing -- the
    # generated bearing on the closed-loop path, the record's on the teacher-forced one.
    _on_on = bool(cfg_feat.get("stim_onset"))
    if _on_on and _clk_mode not in ("linear", "exp"):
        raise ValueError("stim_onset needs a block clock (clock_mode)")
    # SHORT-TERM STIMULUS LAGS (MB.stim_lag_rows, Model 38): the centroid each earlier bout was SHOWN (the provider's
    # re-anchored dots on the evaluation path, the record's on the replay path), seen from the CURRENT pose; segments
    # break at condition changes (a rollout is one run, so no run breaks). Teacher-forced path: the record's block.
    _sl_K = int(cfg_feat.get("stim_lags") or 0)
    _sl_rec = (MB._stimlag_block(rec, cfg_feat) if (_sl_K and not recompute_stim) else None)
    if _sl_K:
        _sl_dx = np.full(n, np.nan); _sl_dy = np.full(n, np.nan); _sl_seg = np.full(n, -1, dtype=np.int64)
        _sl_state = {"cond": None, "seg": -1}
    # DOT-JUMP PHASE (MB._jump_block, Model 41): a function of the programme frame the bout's stimulus is read on (the
    # clock source below, as the block clock) and of its condition; the teacher-forced path reads the record's block.
    _jp_on = bool(cfg_feat.get("stim_jump"))
    _jp_rec = MB._jump_block(rec, cfg_feat) if (_jp_on and not recompute_stim) else None
    # DOT MOTION DIRECTION (MB.dmove_rows, Model 51): the centroid shown at the previous bout and at this one, seen in
    # this bout's reference frame; resets at every change of condition (a rollout is one run). Teacher-forced path:
    # the record's block.
    _dm_on = bool(cfg_feat.get("stim_dmove"))
    _dm_rec = MB._dmove_block(rec, cfg_feat) if (_dm_on and not recompute_stim) else None
    _dm_state = {"cond": None, "prev": (np.nan, np.nan)}
    _phi_rec = None
    if _on_on:
        _phi_rec = getattr(rec, "stim_angle_travel", None) if cfg_feat.get("phi_ref", "ori") == "travel" else None
        _phi_rec = np.asarray(_phi_rec if _phi_rec is not None else rec.stim_angle, float)

    # WHICH CLOCK. The block phase must come from the same clock the STIMULUS comes from, or the
    # net is told "a stimulus just started" at moments unrelated to when it actually started.
    # The tool's programme provider resolves the stimulus at t_now, so t_now is right there. A
    # RECORD-INDEXED provider (stim_rollout.RecordAnchoredProvider, the evaluation path) resolves
    # bout t at the record's own frame -- and a model trained on active bouts never pauses, so
    # t_now drifts off that grid within minutes. Such a provider exposes ``clock_s(t, t_now)``:
    # the wall-clock its stimulus for bout t was read at. Measured before this existed: the
    # onset response present in the teacher-forced law (S3 31 deg vs 22 later) was ABSENT in
    # evaluation rollouts (21 vs 21) and present in own-clock ones (31 vs 22) -- i.e. it was
    # this desynchronisation, not the model.
    if stim_provider is not None:
        _clock_src = (stim_provider.clock_s
                      if callable(getattr(stim_provider, "clock_s", None)) else None)
    else:
        # No provider = the RECORD's stimulus replayed by bout index (the calibrators' path and
        # `reanchor=False`), so the clock must be read on the record's frames as well; t_now is
        # only the right clock when the stimulus itself is resolved at t_now.
        _fo_rec = getattr(rec, "frame_onset", None)
        _clock_src = ((lambda t, t_s, _fo=np.asarray(_fo_rec, float): float(_fo[t]) / MB._FPS
                       if t < len(_fo) else float(t_s))
                      if _fo_rec is not None else None)
    # WHICH ORIGIN. MB._clock_block, the TRAINING feature, anchors t = 0 at the FIRST BOUT of a
    # block: block_onsets() puts a boundary wherever the (dot count, moving) condition changes
    # between consecutive bouts. So every first bout after the dots appear is trained with clock
    # == 1.0 exactly -- the onset-bout indicator the turn head learns from (real fish turn 42 deg
    # toward the dots on that bout against 18 later). This generator used to anchor on the
    # stimulus ONSET FRAME (t mod the 110 s cycle) instead, which hands that same first bout
    # exp(-latency / tau) = 0.3-0.9, so the learned onset response was never fully triggered in
    # any rollout, the tool path included. Same rule as training here: reset at the first bout of
    # a new condition, 0 on every no-dot bout. The condition comes from the provider when there is
    # one (it IS the stimulus the net is shown) and from the record otherwise.
    _mv_all = np.asarray(rec.moving, float)
    _nd_all = np.asarray(getattr(rec, "n_dots", np.zeros(n)), float)
    _clk_state = {"cond": None, "t0": 0.0}
    # PROVIDER-SET BLOCK TIME (scripts/multi_fish.py): a provider exposing ``block_time_s(t, t_now)`` states the time
    # since its scene's onset itself. Needed where the condition key below is no onset marker: in a shoal the visible
    # neighbours regroup every few bouts, and each regrouping would restart the clock and replay the onset response.
    # Absent (every other provider) -> the key rule below, unchanged.
    _bt_src = (stim_provider.block_time_s
               if stim_provider is not None and callable(getattr(stim_provider, "block_time_s", None)) else None)

    def _clock_of(t_s, mv_t, nd_t, tb_ext=None):
        """Block clock for the bout about to be generated: ``t_s`` is the wall-clock its stimulus
        was resolved on (t_now, or the record's frame under a record-indexed provider), ``mv_t`` /
        ``nd_t`` its stimulus condition. NOT precomputable on the t_now path: the generated dt
        decides where a bout falls. ``tb_ext`` = a provider-set block time (see ``_bt_src``)."""
        if _clk_mode not in ("linear", "exp"):
            return None
        nd_ = float(np.nan_to_num(nd_t))
        cond = nd_ * 10.0 + float(np.nan_to_num(mv_t))       # block_onsets' condition key
        if _clk_state["cond"] is None or cond != _clk_state["cond"]:
            _clk_state["t0"] = float(t_s)                    # first bout of a new block
        _clk_state["cond"] = cond
        _w = 1 if _clk_mode == "linear" else len(_clk_taus)
        if not nd_ > 0.0:                                    # gate: 0 on every no-dot bout
            _clk_state["tb"] = 0.0
            return np.zeros((_w,), dtype=np.float32)
        tb = max(float(t_s) - _clk_state["t0"], 0.0) if tb_ext is None else max(float(tb_ext), 0.0)
        _clk_state["tb"] = tb                                # read by the static bump block
        if _clk_mode == "linear":
            return np.array([min(tb / MB.CLOCK_BLOCK_S, 1.0)], dtype=np.float32)
        return np.array([np.exp(-tb / tau) for tau in _clk_taus], dtype=np.float32)
    _kfb = cfg_feat.get("kin_feedback", 1.0)
    kfb_d, kfb_t = ((float(_kfb["dist"]), float(_kfb["dt"])) if isinstance(_kfb, dict)
                    else (float(_kfb), float(_kfb)))
    # VIGOR: per-trajectory log-modulation of the emitted step / interval. u = a fixed intercept
    # (individuality), d = a slow stationary AR(1) (tempo drift). Applied OUTSIDE the loop -- the
    # history keeps the unmodulated value, or the mean-reverting body would push back against z
    # and it would compound. -var/2 keeps E[exp(z)] = 1 so the population mean is unchanged.
    # INDIV: per-trajectory individuality for the two channels vigor does not cover.
    # u_fw tilts the FORWARD mixture weight -- changing how OFTEN the fish turns, not how hard,
    # which is the axis real fish actually vary on (forward fraction spans 4.5x across the 15,
    # median |dtheta| only 2x) and which leaves the pooled turn marginal alone.
    # u_rad SCALES the containment bias. An additive offset was tried and is wrong twice over:
    # centre_bias steers by adding cb*sin(psi) to dtheta so an offset leaks into the turn
    # marginal, and a negative draw flips a fish OUTWARD and triples the wall fraction.
    _iv = cfg_feat.get("indiv"); u_fw = u_rad = 0.0
    if _iv:
        u_fw = float(rng.normal(0.0, float(_iv.get("fw", 0.0))))
        u_rad = float(rng.normal(0.0, float(_iv.get("rad", 0.0))))
    # INNOV_COMP: widen the innovation about THIS bout's conditional mean, restoring the spread
    # kin_feedback removes. For a linear AR, scaling the innovation multiplies the stationary
    # variance while leaving acf(k)=phi^k untouched -- so lambda sets the acf and c the spread.
    # NOT kin_var_comp, which rescaled about the fixed TRAINING log-mean with the history left
    # unscaled: a pure output transform that widened the marginal without restoring the loop.
    _kin = cfg_feat.get("kin_innov")
    ci_d = float(_kin.get("dist", 1.0)) if _kin else 1.0
    ci_t = float(_kin.get("dt", 1.0)) if _kin else 1.0
    vg = cfg_feat.get("vigor"); z_di = z_dt = 0.0
    if vg:
        su_di, sd_di, ph_di, var_di = vg["dist"]; su_dt, sd_dt, ph_dt, var_dt = vg["dt"]
        c_di = float(np.sqrt(1.0 - ph_di * ph_di)); c_dt = float(np.sqrt(1.0 - ph_dt * ph_dt))
        u_di = rng.normal(0.0, su_di); u_dt = rng.normal(0.0, su_dt)
        d_di = rng.normal(0.0, sd_di); d_dt = rng.normal(0.0, sd_dt)

    # EMISSION ORDER. Read off the net, so a checkpoint generates in the order it was trained in
    # and an old one (no attribute) keeps the archived chain. When the angle head is first it takes
    # no conditioners, so the body+angle forward pass stays OUTSIDE the redraw loop exactly as it
    # was -- the shallow-border rejection then still costs no matmul per redraw.
    _ORDER = parse_order(getattr(model, "order", None))
    _a_first = _ORDER[0] == "a"
    _body_fn = getattr(body_angle, "body", None)
    _angle_fn = getattr(body_angle, "angle_params", None)
    if not _a_first and (_body_fn is None or _angle_fn is None):
        raise RuntimeError(f"emission order {_ORDER} needs a conditioned angle head, but this "
                           "numpy predictor only exposes the unconditional one")

    def _adj_w(w):
        """The mixture weights after the two forward-weight tilts. Both multiply the FIRST weight
        and renormalise, which is exactly the torch path's `out[:, :1] + shift` then softmax --
        doing it on the weights keeps the numpy predictor's single forward pass intact."""
        if u_fw:
            w = np.array([w[0] * float(np.exp(u_fw)), w[1], w[2], w[3], w[4]]); w = w / w.sum()
        if d_turn:
            w = np.array([w[0] * float(np.exp(d_turn * d_state)), w[1], w[2], w[3], w[4]])
            w = w / w.sum()
        if _rep_cur[0]:                                   # repulsion: logits -s (left) / +s (right), renormalised
            e = float(np.exp(_rep_cur[0]))
            w = np.array([w[0], w[1] / e, w[2] / e, w[3] * e, w[4] * e]); w = w / w.sum()
        return w

    # MULTI-GROUP SCENES (see MULTI_RULES): supported on the centroid codes with the align / static blocks -- the
    # features every report model uses. Anything that reads the stimulus another way fails loudly.
    _mg = multi_rule is not None
    _groups = ()
    if _mg:
        if multi_rule not in MULTI_RULES and heading_kappa(multi_rule) is None:
            raise ValueError(f"multi_rule {multi_rule!r} not in {MULTI_RULES} (or 'heading<kappa>')")
        if stim_provider is None or not callable(getattr(stim_provider, "groups", None)) or not recompute_stim:
            raise ValueError("multi_rule needs the online stimulus path and a provider with groups()")
        if (_pop or tf_shist.shape[1] or _sl_K or _on_on or _sc_on or _sb_on or _ra is not None or keep_features
                or _force_P or body_angle.raw["st_step"]):
            raise NotImplementedError("multi_rule: centroid code (+ align / static blocks) only -- no per-dot code, "
                                      "stimulus memory or lags, onset / static-clock / bump blocks, re-anchoring, "
                                      "feature recording, forced starts or a step-lever profile term")
        _MG_MAX = 4
        grp_cx = np.full((n, _MG_MAX), np.nan); grp_cy = np.full((n, _MG_MAX), np.nan)
        grp_phi = np.full((n, _MG_MAX), np.nan); grp_n = np.zeros(n, dtype=np.int8)
        _raw = body_angle.raw
        _dist_std, _dt_std, _ang_std = dist_params, dt_params, _angle_fn

        def dist_params(b, *prev):                   # noqa: F811 -- dispatch: a MultiBout combines its rows
            return b.dist(*prev) if isinstance(b, MultiBout) else _dist_std(b, *prev)

        def dt_params(b, *prev):                     # noqa: F811
            return b.dt(*prev) if isinstance(b, MultiBout) else _dt_std(b, *prev)

        def _angle_fn(b, *prev):                     # noqa: F811
            return b.angle(*prev) if isinstance(b, MultiBout) else _ang_std(b, *prev)

    _bs_last = None                                  # the programme block the per-block streams were last seeded at
    if _coll_px and rd0 is None:
        raise ValueError("collision_mm needs the wall-rejection loop (cfg_feat['wall_reject'])")
    for t in range(n):
        xs[t], ys[t] = x, y
        if _ra is not None and t == _ra:                 # block onset of a warm-up start: latch the simulated pose
            dot_cx_a[t:], dot_cy_a[t:] = MB.reanchor_track(dot_cx_a[t:], dot_cy_a[t:],
                                                           (rec.x_px[t], rec.y_px[t], _trv_ra[t]), (x, y, h))
        if use_border and closed_loop:
            r = np.hypot(x - cx, y - cy); psi = MB._wrap_pi(np.arctan2(cy - y, cx - x) - h)
            border_t = MB.border_channels((R - r) / R, psi, cfg_feat.get("border_mode", "raw"))[0]
            assert border_t.shape[0] == tf_border.shape[1], "border row width != block width"
        else:
            border_t = tf_border[t]
        if recompute_stim:
            if stim_provider is None:
                dcx, dcy, mv_t, nd_t = dot_cx_a[t], dot_cy_a[t], mv_a[t], nd_a[t]
            else:                            # resolve the programme from the live clock + pose
                if _mg:                      # multi-group scene: group A stands in for the single-group records
                    _groups = [tuple(float(v) for v in g_) for g_ in stim_provider.groups(t, t_now, x, y, h)]
                    dcx, dcy, mv_t, nd_t = _groups[0] if _groups else (np.nan, np.nan, 0.0, 0.0)
                else:
                    dcx, dcy, mv_t, nd_t = stim_provider(t, t_now, x, y, h)
                dcx = float(dcx); dcy = float(dcy)
                mv_t = float(mv_t); nd_t = float(nd_t)
                if block_seed is not None:
                    # PER-BLOCK RANDOM STREAMS (report Model 11): on entering a programme block, the main
                    # stream and the drift's innovations restart from (block_seed, block), so two models
                    # that differ only in one condition draw IDENTICAL randomness in every other block --
                    # a static curve then differs only through the state the fish carries into the block.
                    _b = getattr(stim_provider, "cur_block", None)
                    if _b is not None and _b != _bs_last:
                        _bs_last = _b
                        rng = np.random.default_rng([*[int(v) for v in block_seed], int(_b)])
                        if d_rng is not None:
                            d_rng = np.random.default_rng([*[int(v) for v in block_seed], int(_b), 1])
            dcx_used[t] = dcx; dcy_used[t] = dcy
            if np.isfinite(dcx):
                phi_gen[t] = MB._wrap_pi(np.arctan2(dcy - y, dcx - x) - h)
            if _pop:
                _grp = None                  # group labels ('_lab' modes): None = one group
                if stim_provider is not None:
                    _xs, _ys, _on, _mvk, _ = stim_provider.dots(t, t_now, x, y, h)
                    _ts = float(getattr(stim_provider, "last_time_s", t_now))
                    _grp = getattr(stim_provider, "last_groups", None)
                else:
                    _xs, _ys = _dxs[t], _dys[t]
                    _on = np.isfinite(_xs) & np.isfinite(_ys); _mvk = np.full(len(_xs), mv_t)
                    _ts = float(_fo_rec[t]) / MB._FPS if _fo_rec is not None else float(t_now)
                    _dg = getattr(rec, "dot_group", None)
                    _grp = None if _dg is None else np.asarray(_dg)[t]
                _vt = None
                if _pop_drift:
                    _xa = np.where(_on, np.asarray(_xs, float), np.nan)
                    _ya = np.where(_on, np.asarray(_ys, float), np.nan)
                    if _prev_dots is not None and len(_prev_dots[0]) == len(_xa):
                        _vt = M.dot_tangential(_prev_dots[0], _prev_dots[1], _xa, _ya, x, y,
                                               _ts - _prev_dots[2])[0]
                    else:
                        _vt = np.full(len(_xa), np.nan)
                    _prev_dots = (_xa, _ya, _ts)
                stim_t = M.pop_row(_xs, _ys, _on, _mvk, x, y, h, vt_k=_vt,
                                   **(dict(_pop_kw, grp_k=_grp) if _pop_kw.get("by_label") else _pop_kw))
            else:
                stim_t = MX._dot_stim_row(cfg_feat["stim_mode"], dcx, dcy,
                                          x, y, h, mv_t, nd_t, dish=(cx, cy, R))
            if _rep:
                if stim_provider is not None:
                    _rx, _ry, _ron, _, _ = stim_provider.dots(t, t_now, x, y, h)
                    _rx = np.where(np.asarray(_ron, bool), np.asarray(_rx, float), np.nan)
                    _ry = np.where(np.asarray(_ron, bool), np.asarray(_ry, float), np.nan)
                else:
                    _rx, _ry = _rdx[t], _rdy[t]
                _s, _u = repulsion_shifts(_rx, _ry, x, y, h, _rep)
                _rep_cur[0], _rep_cur[1] = float(_s[0]), float(_u[0])
            if _grp_rep:
                if _mg:
                    _near = ([(q_[0], q_[1], g_[3], g_[2]) for q_, g_ in zip(stim_provider.group_nearest(), _groups)]
                             if _groups else [])
                else:
                    if stim_provider is not None:
                        _qx, _qy, _qon, _, _ = stim_provider.dots(t, t_now, x, y, h)
                        _q = nearest_dot(_qx, _qy, _qon, x, y)
                    else:
                        _q = nearest_dot(_gdx[t], _gdy[t], None, x, y)
                    _near = [] if _q is None else [(_q[0], _q[1], nd_t, mv_t)]
                _s, _u = grouprep_shifts(_near, x, y, h, _grp_rep)
                _rep_cur[0], _rep_cur[1] = _s, _u
            assert stim_t.shape[0] == stim.shape[1], "closed-loop stim row width != block width"
            if _dm_on:
                _c = float(np.nan_to_num(nd_t)) * 10.0 + float(np.nan_to_num(mv_t))
                _same = _dm_state["cond"] is not None and _c == _dm_state["cond"]
                _dm_t = MB.dmove_rows(_dm_state["prev"][0], _dm_state["prev"][1], dcx, dcy, h,
                                      _same and float(np.nan_to_num(mv_t)) > 0 and float(np.nan_to_num(nd_t)) > 0)[0]
                _dm_state["cond"] = _c; _dm_state["prev"] = (dcx, dcy)
            if _al_on or _st_on:
                _r_t = float(np.hypot(dcx - x, dcy - y)) if np.isfinite(dcx) else np.nan
            if _al_on:
                _al_t = float(MB.align_feature(phi_gen[t], _r_t, mv_t, nd_t)[0])
            if _st_on:
                _st_t = M.static_channels([phi_gen[t]], [_r_t], [nd_t], [mv_t])[0].astype(np.float32)
            if _mg and _groups:              # each group's own stimulus columns, flag and static block, from this pose
                _gp = []
                for _gi, (_gx, _gy, _gm, _gn) in enumerate(_groups):
                    _gphi = MB._wrap_pi(np.arctan2(_gy - y, _gx - x) - h) if np.isfinite(_gx) else np.nan
                    _gr = float(np.hypot(_gx - x, _gy - y)) if np.isfinite(_gx) else np.nan
                    _gp.append((MX._dot_stim_row(cfg_feat["stim_mode"], _gx, _gy, x, y, h, _gm, _gn,
                                                 dish=(cx, cy, R)),
                                float(MB.align_feature(_gphi, _gr, _gm, _gn)[0]) if _al_on else 0.0,
                                (M.static_channels([_gphi], [_gr], [_gn], [_gm])[0].astype(np.float32)
                                 if _st_on else None),
                                _gr))
                    if _gi < _MG_MAX:
                        grp_cx[t, _gi] = _gx; grp_cy[t, _gi] = _gy; grp_phi[t, _gi] = _gphi
                grp_n[t] = len(_groups)
            if _sl_K:
                _c = float(np.nan_to_num(nd_t)) * 10.0 + float(np.nan_to_num(mv_t))
                if _sl_state["cond"] is None or _c != _sl_state["cond"]:
                    _sl_state["seg"] += 1
                _sl_state["cond"] = _c
                _sl_dx[t] = dcx; _sl_dy[t] = dcy; _sl_seg[t] = _sl_state["seg"]
                _ks = np.arange(t - 1, t - 1 - _sl_K, -1)                  # bouts t-1 .. t-K
                _okk = _ks >= 0
                _sl_t = MB.stim_lag_rows(np.where(_okk, _sl_dx[np.maximum(_ks, 0)], np.nan)[None],
                                         np.where(_okk, _sl_dy[np.maximum(_ks, 0)], np.nan)[None],
                                         np.where(_okk, _sl_seg[np.maximum(_ks, 0)], -1)[None],
                                         [x], [y], [h], [_sl_state["seg"]])[0]
        else:
            stim_t = stim[t]
            if _al_on:
                _al_t = float(_al_rec[t]) if t < len(_al_rec) else 0.0
            if _st_on:
                _st_t = _st_rec[t] if t < len(_st_rec) else np.zeros(8, dtype=np.float32)
            if _sl_K:
                _sl_t = _sl_rec[t] if t < len(_sl_rec) else np.zeros(3 * _sl_K, dtype=np.float32)
            if _dm_on:
                _dm_t = _dm_rec[t] if t < len(_dm_rec) else np.zeros(2, dtype=np.float32)
        hist = np.zeros(K)
        for k in range(1, K + 1):
            if t - k >= 0:
                hist[k - 1] = dth[t - k]
        kin = np.zeros(2 * Kk if _kin_dt else Kk)
        for k in range(1, Kk + 1):
            if t - k >= 0:
                kin[k - 1] = kfb_d * (ld_hist[t - k] - dm) / ds
                if _kin_dt_c:
                    kin[Kk + k - 1] = kfb_t * (lt_hist[t - k] - _kdt_lev) / ts
                elif _kin_dt:
                    kin[Kk + k - 1] = kfb_t * (lt_hist[t - k] - tm) / ts
        # block order MUST match MB.build_feature_matrix:
        #   history, stim, stimhist, border, kin, lag_valid, clock, run_clock, align
        _tail = [kin]
        if _lagv_a is not None:
            _tail.append(_lagv_a[t])
        _ck = _clock_of(t_now if _clock_src is None else float(_clock_src(t, t_now)),
                        # a multi-group scene's condition key: total count, moving pattern (one group -> its own)
                        *(((sum(g_[2] * 2.0 ** i_ for i_, g_ in enumerate(_groups)), sum(g_[3] for g_ in _groups))
                           if (_mg and _groups) else (mv_t, nd_t))
                          if recompute_stim else (_mv_all[t], _nd_all[t])),
                        tb_ext=None if _bt_src is None else _bt_src(t, t_now))
        if _ck is not None:
            _tail.append(_ck)
        if _rc_a is not None:
            _tail.append(_rc_a[t])
        if _al_on:
            _tail.append(np.array([_al_t], dtype=np.float32))
        if _st_on:
            _tail.append(_st_t)
        if _sc_on:
            _tail.append((_st_t * np.float32(_ck[-1])).astype(np.float32))
        if _sb_on:
            _tail.append((_st_t * np.float32(MB.static_bump(_clk_state.get("tb", 0.0), _sb_taus))).astype(np.float32))
        if _on_on:
            _phi_t = phi_gen[t] if recompute_stim else (_phi_rec[t] if t < len(_phi_rec) else np.nan)
            _nd_t = nd_t if recompute_stim else _nd_all[t]
            _tail.append(MB.onset_channels(_ck[0], _phi_t, _nd_t)[0].astype(np.float32))
        if _sl_K:
            _tail.append(np.asarray(_sl_t, dtype=np.float32))              # as the record builder
        if _jp_on:                                                          # LAST, as the record builder
            if _mg and _groups:
                raise NotImplementedError("stim_jump: one dot group only (no multi-group rules)")
            if recompute_stim:
                _ts_j = t_now if _clock_src is None else float(_clock_src(t, t_now))
                _jp_t = MB.jump_channels(MB.jump_age_at(_ts_j * MB._FPS), mv_t, nd_t)[0]
            else:
                _jp_t = _jp_rec[t] if t < len(_jp_rec) else np.zeros(2, dtype=np.float32)
            _tail.append(np.asarray(_jp_t, dtype=np.float32))
        if _dm_on:                                                          # LAST, as the record builder
            if _mg and _groups:
                raise NotImplementedError("stim_dmove: one dot group only (no multi-group rules)")
            _tail.append(np.asarray(_dm_t, dtype=np.float32))
        row = np.concatenate([hist / dth_scale, stim_t, tf_shist[t] if tf_shist.shape[1] else _EMPTY,
                              border_t] + _tail).astype(np.float32)
        assert row.shape[0] == int(model.body[0].in_features), "joint row width != net input dim"
        if keep_features:
            Xrec[t] = row
        if _mg and _groups:
            def _mrow(sv, alv, stv, ckv):            # the standard row's block order with these stimulus parts
                parts = [hist / dth_scale, sv, _EMPTY, border_t, kin]
                if _lagv_a is not None:
                    parts.append(_lagv_a[t])
                if ckv is not None:
                    parts.append(ckv)
                if _rc_a is not None:
                    parts.append(_rc_a[t])
                if _al_on:
                    parts.append(np.array([alv], dtype=np.float32))
                if _st_on:
                    parts.append(stv)
                return np.concatenate(parts).astype(np.float32)

            _grows = [_mrow(sv_, al_, st_, _ck) for sv_, al_, st_, _ in _gp]
            if multi_rule == "effects" or heading_kappa(multi_rule) is not None:   # + the no-dots row (clock 0)
                _b0 = _mrow(MX._dot_stim_row(cfg_feat["stim_mode"], np.nan, np.nan, x, y, h, 0.0, 0.0,
                                             dish=(cx, cy, R)), 0.0,
                            (M.static_channels([np.nan], [np.nan], [0.0], [0.0])[0].astype(np.float32)
                             if _st_on else None),
                            None if _ck is None else np.zeros_like(_ck))
                if multi_rule == "effects":
                    b = multi_bout(_raw, "effects", _grows, base_row=_b0)
                else:                                # heading rule: weights from each group's bearing off the heading
                    _ph = [MB._wrap_pi(np.arctan2(g_[1] - y, g_[0] - x) - h) for g_ in _groups]
                    b = multi_bout(_raw, multi_rule, _grows, base_row=_b0,
                                   weights=heading_weights(_ph, heading_kappa(multi_rule)))
            elif multi_rule == "nearest":
                _rg = np.array([g_[3] for g_ in _gp], float)
                b = multi_bout(_raw, "nearest", _grows,
                               comb_row=_grows[int(np.argmin(np.where(np.isfinite(_rg), _rg, np.inf)))])
            else:                                    # 'add' / 'avg': one row from the summed / averaged columns
                _sv = _gp[0][0]; _stv = _gp[0][2]
                for g_ in _gp[1:]:
                    _sv = _sv + g_[0]
                    if _st_on:
                        _stv = _stv + g_[2]
                if multi_rule == "avg":
                    _sv = _sv / len(_gp)
                    if _st_on:
                        _stv = _stv / np.float32(len(_gp))
                b = multi_bout(_raw, multi_rule, _grows,
                               comb_row=_mrow(_sv, max(g_[1] for g_ in _gp), _stv, _ck))
            assert _grows[0].shape[0] == int(model.body[0].in_features), "multi-group row width != net input dim"
            ang = b.angle() if _a_first else None
        elif _a_first:
            b, ang = body_angle(row)
        else:
            b, ang = _body_fn(row), None
        _w0 = _adj_w(ang[0]) if _a_first else None

        def _draw_a(prev):
            """dtheta | x, (whatever was emitted before it). Returns (post-bias, mixture draw)."""
            if _a_first:
                w = _w0
                sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = ang[1:]
            else:
                _a = _angle_fn(b, *prev)
                w = _adj_w(_a[0])
                sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2 = _a[1:]
            c = rng.choice(5, p=w)
            if c == 0:
                v = rng.normal(0.0, sF)
            elif c == 1:
                v = -rng.gamma(aL1, tL1)
            elif c == 2:
                v = -rng.gamma(aL2, tL2)
            elif c == 3:
                v = +rng.gamma(aR1, tR1)
            else:
                v = +rng.gamma(aR2, tR2)
            dth_t = float(np.clip(v, -180.0, 180.0))
            # The MIXTURE draw, before the deterministic containment shift below. This is the
            # value ``MX.mix_logdens`` is the density of; the shift is a function of the state,
            # not of the sample, so it carries no score-function term.
            dth_raw = dth_t
            cb = cb_cfg
            if isinstance(cb, dict):                      # shell-gated containment bias
                _u = np.hypot(x - cx, y - cy) / R
                _u0 = float(cb.get("u0", 0.75))
                if u_gate:
                    # clip keeps a personal gate physically sane: never inside r/R 0.20 (a bias
                    # acting over the whole dish) and never beyond 0.90 (no containment at all).
                    _u0 = float(np.clip(_u0 + u_gate, 0.20, 0.90))
                cb = (float(cb.get("bulk", 0.0))
                      + float(cb.get("shell", 0.0)) * max(0.0, (_u - _u0) / max(1.0 - _u0, 1e-6)))
            if u_rad:
                cb = float(cb) * float(np.exp(u_rad))
            if u_rad_re:
                cb = float(cb) * float(np.exp(u_rad_re))
            if d_bord:
                # THE LOAD-BEARING SHIFT: a slowly wandering radial SET POINT, so the fish's
                # preferred radius moves over minutes instead of being pinned. Real radial window
                # means autocorrelate at +0.377, generated at -0.035, and nothing static can change
                # that.
                #
                # ADDITIVE, NOT MULTIPLICATIVE, and that is measured rather than stylistic. The
                # multiplicative form (`cb * exp(d_bord * s)`, which is what u_rad/u_rad_re do) is
                # exactly INERT here: the rej family calibrates to bulk = 0 and shell = 0, carrying
                # its containment entirely in the rejection sampler, so scaling the bias scales
                # zero. Measured on rej with the multiplicative form -- radial acf1 went 0.117 ->
                # 0.036, i.e. it did nothing but add noise. Same trap already documented for
                # rad_indiv ("inert on graded bulk=0 biases").
                #
                # An additive zero-mean term is also the right shape for this job: it cancels in the
                # POOLED turn marginal (E[s] = 0) while leaving the WINDOW-level statistics moved,
                # which is precisely the split this mechanism needs. Negative values push outward;
                # the rejection sampler still enforces the physical boundary, so that is safe.
                cb = float(cb) + d_bord * d_state
            if cb:
                psi_c = MB._wrap_pi(np.arctan2(cy - y, cx - x) - h)
                dth_t = float(np.clip(dth_t + cb * np.sin(psi_c), -180.0, 180.0))
            return dth_t, dth_raw

        def _draw_d(prev):
            """distance | x, (whatever was emitted before it)."""
            dw, dmu, dsig = dist_params(b, *prev)
            j = rng.choice(len(dw), p=dw)
            # clip the 'average fish' step FIRST, then scale by the vigor -- otherwise a genuinely
            # long-stepping trajectory is squashed back into the pooled q99.9 cap and loses exactly
            # the large steps that make it long-stepping.
            _ld = rng.normal(dmu[j] + d_dist * d_state - d_dist_c + _rep_cur[1], dsig[j])
            if ci_d != 1.0:
                # Re-centre on the conditional mean INCLUDING the drift shift, for the same reason
                # the interval channel does below: innov_comp widens about THIS bout's mean, and
                # with the drift on, that mean has moved.
                _m = float(np.dot(dw, dmu)) + d_dist * d_state - d_dist_c + _rep_cur[1]
                _ld = _m + ci_d * (_ld - _m)
            return float(np.clip(np.exp(_ld), 1e-3, dist_hi))

        def _draw_t(prev):
            """dt | x, (whatever was emitted before it)."""
            tw, tmu, tsig = dt_params(b, *prev)
            # The timing latent shifts the LOCATION of the whole interval mixture, exactly as it
            # does in dt_level_logliks at train and score time.
            _shift = s_lam * s_state - s_lam_c + d_dt * d_state - d_dt_c
            if _log_tr is None:
                jt = rng.choice(len(tw), p=tw)
                _lt = rng.normal(tmu[jt] + _shift, tsig[jt])
            else:
                _F = _ndtr((_log_tr - (np.asarray(tmu) + _shift)) / np.asarray(tsig))
                _p = np.asarray(tw) * _F
                _p = _p / max(float(_p.sum()), 1e-300)
                jt = rng.choice(len(_p), p=_p)
                _lt = tmu[jt] + _shift + tsig[jt] * float(_ndtri(min(max(rng.random() * _F[jt], 1e-12),
                                                                       1.0 - 1e-12)))
            if ci_t != 1.0:
                # Re-centre on the conditional mean INCLUDING the latent shift -- innov_comp widens
                # about this bout's mean, and with the latent on that mean has moved.
                _m = float(np.dot(tw, tmu)) + _shift
                _lt = _m + ci_t * (_lt - _m)
            return float(np.clip(np.exp(_lt), 1e-3, dt_hi))

        def _emit():
            """One draw of the chained bout triple ``(dtheta, base distance, base dt)``, emitted in
            the net's own ``order`` and each channel conditioned on the ones before it.

            Factored out ONLY so the shallow-border rejection can redraw it. On the default order
            the body forward pass has already happened above, so a redraw is a handful of rng calls
            and no matmul -- which is why a max_draws of 20 costs nothing. Nothing here depends on
            the draw index, so calling it once reproduces the original code exactly, variate for
            variate.

            The conditioners are the SAME quantities the trainer teacher-forces (``_cond_cols``):
            |dtheta| after the containment shift, and the BASE distance / interval before vigor --
            so distance individuality cannot leak into the interval channel (real between-fish corr
            of the two levels ~ 0)."""
            got, prev = {}, []
            for _i, _ch in enumerate(_ORDER):
                if _ch == "a":
                    dth_t, dth_raw = _draw_a(prev)
                    got["a"] = (dth_t, dth_raw)
                    _c = abs(dth_t) / dth_scale
                elif _ch == "d":
                    got["d"] = _draw_d(prev)
                    _c = (np.log(max(got["d"], 1e-6)) - dm) / ds
                else:
                    got["t"] = _draw_t(prev)
                    _c = (np.log(max(got["t"], 1e-6)) - tm) / ts
                if _i < len(_ORDER) - 1:
                    prev.append(_c)
            return got["a"][0], got["d"], got["t"], got["a"][1]

        def _vigor_step():
            nonlocal d_di, d_dt, z_di, z_dt
            d_di = ph_di * d_di + sd_di * c_di * rng.normal()
            d_dt = ph_dt * d_dt + sd_dt * c_dt * rng.normal()
            z_di = u_di + d_di - 0.5 * var_di
            z_dt = u_dt + d_dt - 0.5 * var_dt

        if rd0 is None:
            dth_t, base_di, base_dt, dth_raw = _emit()
            if vg:
                _vigor_step()
        else:
            # SHALLOW-BORDER REJECTION. Redraw the whole triple (the heads are chained, so
            # keeping the turn and redrawing only the step would sample from a different law)
            # until the LANDING POINT is accepted by MB.accept_prob.
            #
            # This terminates. The dish is convex, so heading straight inward always gives the
            # largest reachable wall distance d + L, and a draw therefore succeeds whenever
            # L >= d0 - d; with d0 = 2 mm = 7.2 px and a median real step of 15.0 px, 95% of
            # steps clear that even from ON the wall, where roughly half the heading circle is
            # still available. There is no corner to be trapped in. The ramp is also inactive for
            # ~95% of bouts (only 0.07% of real bouts are inside 4 mm), so the expected cost is
            # ~1 draw. All of which is INSTRUMENTED below rather than assumed.
            #
            # The vigor AR state advances ONCE per bout, not once per draw -- otherwise a rejected
            # bout would silently age the tempo latent.
            if vg:
                _vigor_step()
            _exp_di = float(np.exp(z_di))
            _best = None; _best_u = np.inf
            _best_ok = None                       # first rim-ACCEPTED draw the home rule turned down
            _u_now = np.hypot(x - cx, y - cy) / R if hm_pull > 0.0 else 0.0
            for _k in range(rmax):
                dth_t, base_di, base_dt, dth_raw = _emit()
                _hh = h + np.radians(dth_t)
                _dv = base_di * _exp_di
                _ul = np.hypot(x + _dv * np.cos(_hh) - cx, y + _dv * np.sin(_hh) - cy) / R
                _a = MB.accept_prob(_ul, rd0, rd1)
                if _a >= 1.0 or rng.random() < _a:
                    # HOME RADIUS: a rim-accepted bout that moves AWAY from the home radius and
                    # lands outside its band is redrawn with probability `pull` (pull 0 -> no draw).
                    if (hm_pull > 0.0 and abs(_ul - hm_state) > max(abs(_u_now - hm_state), hm_band)
                            and rng.random() < hm_pull):
                        if _best_ok is None:
                            _best_ok = (dth_t, base_di, base_dt, dth_raw)
                        wall_stats["n_home_reject"] = wall_stats.get("n_home_reject", 0) + 1
                        continue
                    if _coll_px:                      # COLLISION: no landing within collision_mm of a neighbour
                        _nb = np.asarray(stim_provider.neighbours_xy(), float).reshape(-1, 2)
                        if len(_nb) and np.any(np.hypot(_nb[:, 0] - (x + _dv * np.cos(_hh)),
                                                        _nb[:, 1] - (y + _dv * np.sin(_hh))) < _coll_px):
                            if _best_ok is None:
                                _best_ok = (dth_t, base_di, base_dt, dth_raw)
                            wall_stats["n_collide"] = wall_stats.get("n_collide", 0) + 1
                            continue
                    break
                if _best is None or _ul < _best_u:     # remember the least-bad candidate
                    _best_u = _ul; _best = (dth_t, base_di, base_dt, dth_raw)
            else:
                # BUDGET EXHAUSTED -> take the BEST of the draws, not the last one. Keeping the
                # last was the first implementation and it is strictly worse: it throws away up
                # to 19 better candidates for no reason, and it was measured putting bouts
                # 0.011 mm from the wall on a deliberately under-trained model. The hard floor is
                # then still guaranteed by the guardrail, which runs at R_g (the forbidden-ring
                # boundary) rather than the physical wall -- see below.
                # With the home rule, a rim-accepted candidate it turned down beats the least-bad
                # rim-rejected one: the home pull may never push a fish INTO the ring.
                if _best_ok is not None:
                    dth_t, base_di, base_dt, dth_raw = _best_ok
                    wall_stats["n_home_fallback"] = wall_stats.get("n_home_fallback", 0) + 1
                else:
                    dth_t, base_di, base_dt, dth_raw = _best
                    wall_stats["n_reject_fallback"] = wall_stats.get("n_reject_fallback", 0) + 1
            wall_stats["n_draws"] = wall_stats.get("n_draws", 0) + _k + 1
            if _k:
                wall_stats["n_redraw_bouts"] = wall_stats.get("n_redraw_bouts", 0) + 1
        if t < _force_P:                          # BLOCK-START FORCING: replay the real bout
            dth_t = float(rec.dtheta[t]); dth_raw = dth_t
            base_di = float(rec.distance_px[t]) / float(np.exp(z_di))
            base_dt = float(rec.dt_s[t]) / float(np.exp(z_dt))
        dth[t] = dth_t
        if keep_features:
            # s_rec[t] is the state that shifted THIS bout's interval mixture, recorded before the
            # AR step below advances it -- scoring against the advanced value would put every dt
            # log-density one bout out of phase.
            dth_base[t] = dth_raw; dist_base[t] = base_di
            dt_base[t] = base_dt; s_rec[t] = s_state; d_rec[t] = d_state
        dval = base_di * float(np.exp(z_di)); tval = base_dt * float(np.exp(z_dt))
        if cfg_feat.get("dt_snap"):
            # EMITTED value only; ld_hist/lt_hist keep the continuous base below, so the loop
            # dynamics, the RNG stream and every other channel stay byte-identical to the
            # unsnapped control -- the contrast is attributable to the measurement operator alone.
            tval = MB.snap_dt(tval)
        dist[t] = dval; dtg[t] = tval
        t_onset[t] = t_now; t_now += tval          # t_now is now bout t+1's onset time
        # history keeps the BASE: the body must stay in average-fish world (see the vigor note above)
        # -- note base_dt ALREADY carries the timing latent's shift, and that is deliberate. vigor
        # was a post-hoc multiplier the body was never trained under, so its history had to stay
        # unmodulated; lam*s is inside the fitted likelihood, i.e. the model explains the REAL
        # interval as mu + lam*s, so the emitted interval is what _kin_history_block saw at train
        # time. Stripping it here would train one object and roll out another.
        ld_hist[t] = np.log(max(base_di, 1e-6)); lt_hist[t] = np.log(max(base_dt, 1e-6))
        if _kin_dt_c:                             # level update AFTER the bout, as the builder's EMA
            _kdt_lev += _kdt_a * (lt_hist[t] - _kdt_lev)
        if _dtl:                                  # advance ONCE per bout, never per reject-redraw
            s_state = s_phi * s_state + np.sqrt(max(1.0 - s_phi * s_phi, 0.0)) * s_rng.normal()
        if _dr:                                   # same contract: one step per BOUT, not per draw
            d_state = d_phi * d_state + np.sqrt(max(1.0 - d_phi * d_phi, 0.0)) * d_rng.normal()
        h = h + np.radians(dth[t])
        if guardrail:
            x, y, h = MB.advance(x, y, h, dval, cx, cy, R_g, eps,
                                 mode=cfg_feat.get("wall_mode", "clip"), stats=wall_stats)
        else:
            x = x + dval * np.cos(h); y = y + dval * np.sin(h)
        if t < _force_P and t + 1 < n:            # ... and put the fish on the real pose it had next
            x, y, h = float(rec.x_px[t + 1]), float(rec.y_px[t + 1]), float(_trv_f[t + 1])
        if hm_rng is not None and hm_phi < 1.0:      # the home drifts ONCE per bout, never per draw
            hm_state = hm_mu + hm_phi * (hm_state - hm_mu) + hm_sd * np.sqrt(max(1.0 - hm_phi * hm_phi, 0.0)) * float(hm_rng.normal())
        if stop_time_s is not None and t_now >= stop_time_s:
            # The test is on the NEXT onset, so the last kept bout is the last one starting
            # strictly inside the programme; the bout that would start past the end (and whose
            # stimulus lookup would be out of range) is never generated. The realised end
            # therefore undershoots stop_time_s by up to one inter-bout interval.
            n_done = t + 1
            break
    if n_done < n:                                  # truncate every per-bout array together
        dth, dist, dtg = dth[:n_done], dist[:n_done], dtg[:n_done]
        xs, ys, t_onset = xs[:n_done], ys[:n_done], t_onset[:n_done]
        if recompute_stim:
            phi_gen = phi_gen[:n_done]; dcx_used = dcx_used[:n_done]; dcy_used = dcy_used[:n_done]
        if keep_features:
            Xrec = Xrec[:n_done]
            dth_base = dth_base[:n_done]; dist_base = dist_base[:n_done]
            dt_base = dt_base[:n_done]; s_rec = s_rec[:n_done]; d_rec = d_rec[:n_done]
        if _mg:
            grp_cx, grp_cy, grp_phi, grp_n = grp_cx[:n_done], grp_cy[:n_done], grp_phi[:n_done], grp_n[:n_done]
    out = dict(dtheta=dth, symbol=E.symbols_from_dtheta(dth), dt_s=dtg, distance_px=dist,
               x_px=xs, y_px=ys,
               n_hits=wall_stats.get("n_hit", 0), n_bounces=wall_stats.get("n_bounce", 0))
    if rd0 is not None:
        # Instrumented, not assumed. An uninstrumented quantity a downstream calibrator can
        # absorb is exactly the shape of the kin_feedback trap, so count the draws.
        out["n_draws"] = wall_stats.get("n_draws", 0)
        out["n_redraw_bouts"] = wall_stats.get("n_redraw_bouts", 0)
        out["n_reject_fallback"] = wall_stats.get("n_reject_fallback", 0)
        if hm_rng is not None:
            out["n_home_reject"] = wall_stats.get("n_home_reject", 0)
            out["n_home_fallback"] = wall_stats.get("n_home_fallback", 0)
        if _coll_px:
            out["n_collide"] = wall_stats.get("n_collide", 0)
            out["n_home_fallback"] = wall_stats.get("n_home_fallback", 0)
    if recompute_stim:
        out["stim_angle_gen"] = phi_gen
    if stim_provider is not None or stop_time_s is not None:
        # Tool-mode extras only -- the default sample dict keeps exactly the keys it always had.
        out["t_onset_s"] = t_onset
        out["end_time_s"] = float(t_now)
        if recompute_stim:
            out["dot_cx"] = dcx_used; out["dot_cy"] = dcy_used
    if _ra is not None:                                # the re-anchored dots this rollout actually saw
        out["dot_cx"] = dcx_used; out["dot_cy"] = dcy_used
    if _mg:                                            # every group's centroid and bearing, per bout
        out.update(grp_cx=grp_cx, grp_cy=grp_cy, grp_phi=grp_phi, grp_n=grp_n)
    if keep_features:
        out["X"] = Xrec
        out["dth_base"] = dth_base            # the mixture draw, pre-containment-bias
        out["dist_base"] = dist_base          # pre-vigor
        out["dt_base"] = dt_base              # pre-vigor, pre-dt_snap
        out["s_state"] = s_rec                # timing-latent value that shifted each dt mixture
        out["d_state"] = d_rec                # drift value that shifted all four channels
        # THE LATENT THAT ACTED. joint_forward_re(..., z) has to be evaluated here, not at z=0:
        # the old fine-tuner scored at z=0 and therefore computed a valid-looking gradient for a
        # DIFFERENT policy than the one that produced the actions.
        out["z_re"] = (np.zeros(0) if z_re is None else np.asarray(z_re, float))
    return out


# =========================================================================== #
#  Adapter
# =========================================================================== #
class MLPMixJointDensity(MX.MLPMixLoopDensity):
    """Joint-likelihood model with a single feature vector and chained heads.

    ``score`` reports THREE channels, which is the point: the kin family optimises distance and dt
    against a likelihood it never reports.
    """

    def __init__(self, K=8, K_kin=3, stim_mode="shoal", include_symbols=False, hidden=128,
                 epochs=80, lr=1e-3, patience=8, batch_size=256, seed=0,
                 use_border=True, border_mode="raw", wall_mode="clip",
                 closed_loop=True, guardrail=True, eps=1e-3, stim_loop=True,
                 n_dist=2, n_dt=3, contain=False, contain_fish=12, contain_w_occ=2.6,
                 vigor=False, vigor_drift_calib=False, kin_fb_calib=False,
                 indiv=False, innov_comp=False,
                 # travel frame by DEFAULT for new models: the body axis sits a median
                 # 40.6 deg from the travel heading and a rollout has no body axis to
                 # query. The load path keeps its own "ori" fallback, so a checkpoint
                 # saved without these keys still reloads in the frame it was trained in.
                 dt_snap=False, psi_ref="travel", phi_ref="travel",
                 # Which of the six exact factorisations of the joint to fit. See the module
                 # docstring: same density, different capacity per channel.
                 head_order="a,d,t",
                 # Per-bout training weights for the two under-fit regimes (MB.regime_weights):
                 # None = plain likelihood; a dict of its keyword arguments otherwise.
                 regime_w=None,
                 # Truncate the interval head at this many seconds (the active-only data's own cut
                 # is 10 s): renormalised likelihood, bounded sampler. None = the archived head.
                 dt_trunc=None,
                 K_stim=1, noise_aug=None, contain_mode="bias",
                 # Escape hatch for cfg_feat keys that have no constructor arg of their own
                 # (clock_mode, clock_tau_s, lag_valid, use_adapt, ...). Merged LAST in fit,
                 # so it can also override a computed default -- and it is saved with the
                 # checkpoint, so a reloaded model rebuilds the same feature matrix.
                 cfg_feat_extra=None, **kwargs):
        super().__init__(K=K, stim_mode=stim_mode, include_symbols=include_symbols, hidden=hidden,
                         epochs=epochs, lr=lr, patience=patience, batch_size=batch_size, seed=seed,
                         use_border=use_border, border_mode=border_mode, wall_mode=wall_mode,
                         closed_loop=closed_loop, guardrail=guardrail, eps=eps,
                         stim_loop=stim_loop, **kwargs)
        self.K_kin = int(K_kin)
        self.n_dist, self.n_dt = int(n_dist), int(n_dt)
        self.contain = bool(contain)
        self.contain_fish, self.contain_w_occ = int(contain_fish), float(contain_w_occ)
        # Ported from the kin lineage so the treatments can be compared like-for-like.
        # vigor      = per-trajectory intercept (+ optional slow AR drift) on the EMITTED step/interval.
        # kin_fb_calib = shrink the kinematic history on read, damping the free-run over-persistence.
        self.vigor, self.vigor_drift_calib = bool(vigor), bool(vigor_drift_calib)
        self.kin_fb_calib = bool(kin_fb_calib)
        # indiv      = per-trajectory turning style + spatial preference (the channels vigor misses)
        # innov_comp = restore the spread kin_feedback removes, without disturbing the acf
        self.indiv, self.innov_comp = bool(indiv), bool(innov_comp)
        # dt_snap  = quantise the EMITTED interval to the 1/30 s camera lattice (MB.snap_dt)
        # psi_ref  = angular reference of the border bearing: 'ori' (archived behaviour, raw body
        #            orientation) or 'travel' (previous travel heading -- the frame the closed
        #            loop actually queries). phi_ref = the same choice for the dot bearing.
        self.dt_snap = bool(dt_snap)
        self.psi_ref, self.phi_ref = str(psi_ref), str(phi_ref)
        self.head_order = parse_order(head_order)
        self.regime_w = dict(regime_w) if regime_w else None
        self.dt_trunc = float(dt_trunc) if dt_trunc else None
        self.cfg_feat_extra = dict(cfg_feat_extra or {})
        # K_stim = how many stimulus ONSETS the net sees (1 = the current snapshot only, i.e.
        #          every archived arm; 3 = plus the two previous presentations). MB._stimhist_block.
        # noise_aug = pose-perturbed training pairs with the target corrected back to the truth
        #          (dict: sigma_pos px, sigma_ori deg, mode 'concat'|'mask', frac). None = off.
        # contain_mode = 'bias' (the archived shell-gated centre_bias) or 'reject' (shallow-border
        #          rejection replaces the SHELL ramp; the bulk term stays a fitted knob with 0 in
        #          its grid, so the search decides whether any turn bias is needed at all).
        self.K_stim = int(K_stim)
        self.noise_aug = dict(noise_aug) if noise_aug else None
        self.contain_mode = str(contain_mode)
        self.use_kin = True

    def fit(self, train_records, ctx):
        c = ctx.cfg
        self.K = int(c.get("mlp_K", self.K))
        self.stim_mode = c.get("mlp_stim_mode", self.stim_mode)
        self.hidden = int(c.get("mlp_hidden", self.hidden))
        # epochs_override / patience_override are how colab_run_ladder's --force-epochs runs a PAIRED
        # budget contrast inside one select. They were read only from ctx.cfg here, so they never
        # reached any joint rung and a budget ladder silently ran both arms at the same budget.
        # Same precedence as MLPMixKinDensity.fit (fit_MLP_mix.py:1552).
        self.epochs = int(getattr(self, "epochs_override", None)
                          if getattr(self, "epochs_override", None) is not None
                          else c.get("mlp_epochs", self.epochs))
        self.patience = int(getattr(self, "patience_override", None)
                            if getattr(self, "patience_override", None) is not None
                            else c.get("mlp_patience", self.patience))
        self.lr = float(c.get("mlp_lr", self.lr))
        self._dt_pool, self._dist_pool = ctx.dt_pool, ctx.dist_pool
        self._rich_scales = ctx.rich_scales
        alld = np.concatenate([r.dtheta for r in train_records])
        self._dth_scale = float(max(np.std(alld), 1.0))
        adist = np.concatenate([r.distance_px for r in train_records])
        adist = adist[np.isfinite(adist) & (adist > 0)]
        adt = np.concatenate([r.dt_s for r in train_records]); adt = adt[np.isfinite(adt) & (adt > 0)]
        self._kin_scales = (float(np.log(adist).mean()), float(max(np.log(adist).std(), 1e-3)),
                            float(np.log(adt).mean()), float(max(np.log(adt).std(), 1e-3)))
        self._kin_clip = (float(np.quantile(adist, 0.999)), float(np.quantile(adt, 0.999)))
        # use_kin=True is the whole point: ONE x carrying turn history, stimulus, border AND
        # kinematic history. joint=True tells the shared calibrators which generator to use.
        self._cfg_feat = dict(K=self.K, stim_mode=self.stim_mode, include_symbols=self.include_symbols,
                              dth_scale=self._dth_scale, rich_scales=self._rich_scales,
                              use_border=self.use_border, border_mode=self.border_mode,
                              wall_mode=self.wall_mode, use_adapt=False, adapt_lambda=0.6,
                              use_kin=True, joint=True, kin_scales=self._kin_scales,
                              K_kin=self.K_kin, kin_clip=self._kin_clip,
                              n_dist=self.n_dist, n_dt=self.n_dt,
                              dt_snap=self.dt_snap, psi_ref=self.psi_ref, phi_ref=self.phi_ref,
                              K_stim=self.K_stim)
        if self.noise_aug:
            self._cfg_feat["noise_aug"] = dict(self.noise_aug)
        if self.dt_trunc:
            self._cfg_feat["dt_trunc"] = float(self.dt_trunc)
        self._cfg_feat.update(self.cfg_feat_extra)      # last word, and saved with the blob
        X = np.concatenate([MB.build_feature_matrix(r, self._cfg_feat) for r in train_records], 0)
        y_ang = np.concatenate([r.dtheta for r in train_records])
        y_dist = np.concatenate([np.asarray(r.distance_px, float) for r in train_records])
        y_dt = np.concatenate([np.asarray(r.dt_s, float) for r in train_records])
        weights = None
        if self.regime_w:
            ws, infos = zip(*[MB.regime_weights(r, **self.regime_w) for r in train_records])
            weights = np.concatenate(ws)
            self.regime_info = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]
                                if k != "heads"}
            print(f"    [{self.name}] regime weights {self.regime_w}: onset {100 * self.regime_info['frac_onset']:.1f}% "
                  f"of bouts -> {100 * self.regime_info['share_onset']:.0f}% of the loss, pursuit "
                  f"{100 * self.regime_info['frac_pursuit']:.1f}% -> {100 * self.regime_info['share_pursuit']:.0f}%")
        self._model, self.history = fit_mix_joint(
            X, y_ang, y_dist, y_dt, self.hidden, self._dth_scale, self._kin_scales,
            epochs=self.epochs, lr=self.lr, patience=self.patience,
            batch_size=self.batch_size, seed=self.seed, n_dist=self.n_dist, n_dt=self.n_dt,
            order=self.head_order, weights=weights, dt_trunc=self.dt_trunc,
            **_noise_kwargs(self, train_records))
        # SAME ORDER as MLPMixKinDensity.fit, so the treatments are comparable to the lineage:
        # vigor -> drift rescale -> kin_feedback -> containment. (The known ordering quirk that the
        # drift is sized before the shrinkage is preserved deliberately; its impact here is small
        # because these models have almost no window-level drift of their own to subtract.)
        if self.vigor:
            vg = MX._calibrate_vigor(train_records)
            vg["apply_turn"] = False                     # turn individuality is a separate mechanism
            self._cfg_feat["vigor"] = vg
            if self.vigor_drift_calib:
                MX.MLPMixKinDensity._rescale_vigor_drift(self, train_records)
        if self.kin_fb_calib:
            tg, vtg = MX._kin_acf_targets(train_records)
            kfb, _kvs = MX._search_kin_feedback(
                self._model, train_records, self._cfg_feat, tg, vtg,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
            self._cfg_feat["kin_feedback"] = kfb
        if self.contain:
            self._cfg_feat["wall_mode"] = "reflect"
            self._cfg_feat.update(MX._search_wall(
                self._model, train_records, self._cfg_feat, mode=self.contain_mode,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed,
                max_fish=self.contain_fish, w_occ=self.contain_w_occ))
        # innov BEFORE indiv: it changes the per-bout spread that the individuality search then
        # measures on top of. Both AFTER contain, then contain is RE-FIT, because both widen the
        # model and movement spread drives wall contact (corr +0.59) while rad_frac_wall is the
        # top C2ST feature. Not re-fitting is how contain_vc lost.
        if self.innov_comp:
            self._cfg_feat["kin_innov"] = MX._search_innov_comp(
                self._model, train_records, self._cfg_feat,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
        if self.indiv:
            self._cfg_feat["indiv"] = MX._search_indiv(
                self._model, train_records, self._cfg_feat, MX._indiv_targets(train_records),
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed)
        if self.contain and (self.indiv or self.innov_comp):
            self._cfg_feat.update(MX._search_wall(
                self._model, train_records, self._cfg_feat, mode=self.contain_mode,
                closed_loop=self.closed_loop, guardrail=self.guardrail, eps=self.eps,
                stim_loop=self.stim_loop, seed=self.seed + 1,
                max_fish=min(self.contain_fish, 8), w_occ=self.contain_w_occ))
        self.config = dict(mode="joint", hidden=self.hidden, n_features=X.shape[1], K=self.K,
                           K_kin=self.K_kin, stim_mode=self.stim_mode, use_border=self.use_border,
                           use_kin=True, n_dist=self.n_dist, n_dt=self.n_dt,
                           head_order=order_tag(self.head_order))
        return self

    def score(self, rec):
        """THREE channels. ``logdens`` keeps the angle meaning so every existing consumer is
        unaffected; the harness picks up the two extra keys and reports them separately.

        Units differ per channel (nats per degree / pixel / second), so the three numbers are not
        comparable to EACH OTHER in absolute level -- only across models within a channel.
        """
        X = MB.build_feature_matrix(rec, self._cfg_feat)
        a_z, ld_c, lt_c = _cond_cols(rec.dtheta, rec.distance_px, self._cfg_feat["dth_scale"],
                                     self._cfg_feat["kin_scales"], y_dt=rec.dt_s)
        with torch.no_grad():
            Xt = torch.tensor(X)
            ang, dpar, tpar = self._model.joint_forward(Xt, torch.tensor(a_z), torch.tensor(ld_c),
                                                        torch.tensor(lt_c))
            logdens = MX.mix_logdens(torch.tensor(rec.dtheta.astype(np.float32)), ang).numpy()
            yd = torch.tensor(np.asarray(rec.distance_px, np.float32))
            yt = torch.tensor(np.asarray(rec.dt_s, np.float32))
            ld = MX._lnk_logpdf(torch.clamp(yd, min=1e-6), *dpar).numpy()
            lt = MX._dt_logpdf(torch.clamp(yt, min=1e-6), *tpar,
                               hi=self._cfg_feat.get("dt_trunc")).numpy()
        # Mask EXACTLY the bouts the training loss masks (`isfinite & > 0`), so the reported
        # likelihood is over the same support the model was fitted on. This is not cosmetic: 10
        # of the 70590 bouts have distance exactly 0, and a log-normal evaluated at the 1e-6 clamp
        # returns about -850 nats. Ten such bouts would move a fish's bits/bout by ~0.26 bits --
        # small enough to look like a real effect and entirely an artefact.
        dv = np.asarray(rec.distance_px, float); tv = np.asarray(rec.dt_s, float)
        ld = np.where(np.isfinite(dv) & (dv > 0), ld, np.nan)
        lt = np.where(np.isfinite(tv) & (tv > 0), lt, np.nan)
        # Symbol PMF from the angle parameters that were just computed, rather than a second
        # unconditional forward pass. On the default order those are the SAME numbers (the angle
        # head reads nothing but the body), so this is bit-identical to the old
        # `predict_params_mix` call -- but on an order where dtheta is emitted last, the
        # unconditional head does not exist and the plug-in would be silently wrong.
        params = (np.exp(ang[0].numpy()),) + tuple(z.numpy().astype(float) for z in ang[1:])
        return dict(logdens=logdens, symbol_logq=MX.symbol_logq_mix(*params),
                    logdens_dist=ld, logdens_dt=lt)

    def sample(self, rec, rng, **kw):
        # **kw forwards the tool-mode knobs (stim_provider / stop_time_s / keep_features); the
        # harness only ever calls sample(rec, rng), so the adapter contract is unchanged.
        return generate_mix_joint_loop(self._model, rec, self._cfg_feat, rng,
                                       closed_loop=self.closed_loop, guardrail=self.guardrail,
                                       eps=self.eps, stim_loop=self.stim_loop, **kw)

    def conditional_turn(self, rec):
        """The TEACHER-FORCED turn law on a real record, per bout: ``(E[dtheta], E|dtheta|,
        P(dtheta > 0))`` from the angle mixture given the real feature row (and, when the angle
        head is not first, the real step and interval). No rollout: this is the fit itself, which
        is what Diagnostics.ipynb D1/D2/D5 showed to be where the stimulus response is short. The
        random-effect subclass inherits it at z = 0, the population fish."""
        w, (sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2) = self._turn_mixture(rec)
        m_neg = w[:, 1] * aL1 * tL1 + w[:, 2] * aL2 * tL2      # components 1, 2 emit NEGATIVE dtheta
        m_pos = w[:, 3] * aR1 * tR1 + w[:, 4] * aR2 * tR2      # components 3, 4 emit POSITIVE dtheta
        e_abs = w[:, 0] * sF * np.sqrt(2.0 / np.pi) + m_neg + m_pos
        return m_pos - m_neg, e_abs, w[:, 3] + w[:, 4] + 0.5 * w[:, 0]

    def _turn_mixture(self, rec):
        """The teacher-forced angle mixture per bout: weights (n, 5) and the nine component
        parameters (sigma_F, then shape/scale of L1, L2, R1, R2), as numpy."""
        cf = self._cfg_feat
        X = MB.build_feature_matrix(rec, cf)
        az, lc, lt = _cond_cols(rec.dtheta, rec.distance_px, cf["dth_scale"], cf["kin_scales"],
                                y_dt=rec.dt_s)
        with torch.no_grad(), repulsion_side(self, rec):
            ang = self._model.joint_forward(torch.tensor(X), torch.tensor(az), torch.tensor(lc),
                                            torch.tensor(lt))[0]
        return np.exp(ang[0].numpy()), [z.numpy().astype(float) for z in ang[1:]]

    def conditional_turn_draws(self, rec, rng, n_draw=1):
        """``n_draw`` turns per bout DRAWN from the same teacher-forced angle mixture as
        ``conditional_turn`` (component by its weight, then Normal(0, sigma_F) or a signed Gamma),
        shape (n, n_draw). A single draw per bout carries the scatter of a single real turn, so
        statistics built on it are read the same way as on the real turns."""
        w, (sF, aL1, tL1, aL2, tL2, aR1, tR1, aR2, tR2) = self._turn_mixture(rec)
        fl = MX.SCALE_FLOOR
        n = len(w)
        cw = np.cumsum(w / w.sum(1, keepdims=True), axis=1)
        k = (rng.random((n, n_draw))[:, :, None] > cw[:, None, :-1]).sum(-1)       # component index
        shape = np.stack([np.ones(n), aL1, aL2, aR1, aR2], 1)
        scale = np.maximum(np.stack([sF, tL1, tL2, tR1, tR2], 1), fl)
        sign = np.array([0.0, -1.0, -1.0, 1.0, 1.0])
        a, s = np.take_along_axis(shape, k, 1), np.take_along_axis(scale, k, 1)
        g = rng.gamma(a, s)
        return np.where(k == 0, rng.normal(0.0, 1.0, k.shape) * s, sign[k] * g)


def save_joint(model, path):
    """Persist a fitted joint model. Mirrors ``MX.save_mix``; kept separate because the net has a
    different head layout and ``load_mix`` rebuilds from ``config``."""
    if not hasattr(model, "_model"):
        raise RuntimeError("model is not fitted; call .fit(...) before save_joint")
    torch.save(dict(cls="MLPMixJointDensity", state_dict=model._model.state_dict(),
                    n_features=int(model.config["n_features"]), hidden=int(model.config["hidden"]),
                    cfg_feat=model._cfg_feat, config=model.config, name=model.name,
                    dt_pool=np.asarray(model._dt_pool), dist_pool=np.asarray(model._dist_pool),
                    history=getattr(model, "history", None),
                    K=model.K, K_kin=model.K_kin, stim_mode=model.stim_mode,
                    include_symbols=model.include_symbols, use_border=model.use_border,
                    border_mode=model.border_mode, wall_mode=model._cfg_feat.get("wall_mode", "clip"),
                    n_dist=model.n_dist, n_dt=model.n_dt,
                    head_order=order_tag(getattr(model, "head_order", DEFAULT_ORDER)),
                    regime_w=getattr(model, "regime_w", None),      # training-only; for the record
                    vigor=model.vigor, vigor_drift_calib=model.vigor_drift_calib,
                    kin_fb_calib=model.kin_fb_calib,
                    indiv=model.indiv, innov_comp=model.innov_comp,
                    K_stim=int(getattr(model, "K_stim", 1)),
                    noise_aug=getattr(model, "noise_aug", None),
                    contain_mode=str(getattr(model, "contain_mode", "bias")),
                    loop=dict(closed_loop=model.closed_loop, guardrail=model.guardrail,
                              eps=model.eps, stim_loop=model.stim_loop)), path)
    return path


def load_joint(path, map_location="cpu"):
    """Reload a joint model saved by ``save_joint``, ready to ``.sample`` / ``.score``."""
    MX._numpy2_unpickle_compat()
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    lp = blob["loop"]
    m = MLPMixJointDensity(K=blob["K"], K_kin=blob["K_kin"], stim_mode=blob["stim_mode"],
                           include_symbols=blob["include_symbols"], hidden=blob["hidden"],
                           use_border=blob["use_border"], border_mode=blob["border_mode"],
                           wall_mode=blob.get("wall_mode", "clip"),
                           closed_loop=lp["closed_loop"], guardrail=lp["guardrail"],
                           eps=lp["eps"], stim_loop=lp["stim_loop"],
                           n_dist=blob["n_dist"], n_dt=blob["n_dt"],
                           # absent on every pre-ordering checkpoint, which all used the default
                           head_order=blob.get("head_order", DEFAULT_ORDER))
    net = MDNMixJoint(blob["n_features"], blob["hidden"], n_dist=blob["n_dist"], n_dt=blob["n_dt"],
                      order=m.head_order)
    net.load_state_dict(blob["state_dict"]); net.eval()
    net.dt_trunc = blob["cfg_feat"].get("dt_trunc")
    m.dt_trunc = net.dt_trunc
    m._model = net
    m._cfg_feat = blob["cfg_feat"]
    m._rich_scales = blob["cfg_feat"].get("rich_scales")
    m._dt_pool = blob["dt_pool"]; m._dist_pool = blob["dist_pool"]
    m.config = blob["config"]; m.name = blob["name"]; m.history = blob.get("history")
    m._dth_scale = blob["cfg_feat"]["dth_scale"]; m._kin_scales = blob["cfg_feat"]["kin_scales"]
    # Restore the treatment flags (mirrors load_fishre). Generation/scoring read _cfg_feat and
    # are correct either way, but a re-save or a refit of the loaded object reads THESE -- a
    # psit/dt_snap checkpoint refit without them would silently train an 'ori'-frame, unsnapped
    # model under the same name.
    for _f in ("vigor", "vigor_drift_calib", "kin_fb_calib", "indiv", "innov_comp"):
        if _f in blob:
            setattr(m, _f, bool(blob[_f]))
    m.contain = m._cfg_feat.get("centre_bias") is not None
    m.dt_snap = bool(m._cfg_feat.get("dt_snap", False))
    m.psi_ref = str(m._cfg_feat.get("psi_ref", "ori"))
    m.phi_ref = str(m._cfg_feat.get("phi_ref", "ori"))
    # Read the new treatments from cfg_feat first (it is the object generation actually uses) and
    # fall back to the top-level keys, so a checkpoint written before these existed still loads.
    m.K_stim = int(m._cfg_feat.get("K_stim", blob.get("K_stim", 1)))
    m.noise_aug = m._cfg_feat.get("noise_aug", blob.get("noise_aug"))
    m.contain_mode = str(blob.get("contain_mode",
                                  "reject" if m._cfg_feat.get("wall_reject") else "bias"))
    m.regime_w = blob.get("regime_w")
    return m


def stim_turn_from_spec(spec):
    """Rebuild a saved profile term (v1 / v2 / v3 / v4 / v5) -- parameters are then filled by ``load_state_dict``."""
    v = int(spec.get("version", 1))
    if v == 5:
        return StimTurnTermV5(spec["idx"], side=spec.get("side", False), static=spec.get("static", False),
                              fcount=spec.get("fcount", "nz"), late=spec.get("late", False),
                              clock_tau_s=spec.get("clock_tau_s", 4.0), late_tau_s=spec.get("late_tau_s", V5_LATE_TAU_S),
                              cut_s=spec.get("cut_s"))
    if v == 4:
        return StimTurnTermV4(spec["idx"], clock_tau_s=spec.get("clock_tau_s", 4.0),
                              late_tau_s=spec.get("late_tau_s", V4_LATE_TAU_S))
    if v == 3:
        return StimTurnTermV3(spec["layout"], spec["idx"])
    return StimTurnTerm(spec["mv_idx"], spec["nz_idx"], rho_idx=spec.get("rho_idx"))
