"""Figure fig_scene (Chapter 4, sec:m-twogroups and sec:m-multifish): how the in-silico scenes are built.

(a) A two-group scene exactly as the two-group simulator builds it (``scripts/multi_stim.py``): scene ``ang90``
    (4 v 4 moving dots, 2 theta = 90 deg, both at 75 px = 20.8 mm), simulated fish 0 of M6+C (tag 28b, seed 0), its first
    block. The fish's onset pose is NOT drawn by hand: the fish's 100 s lead-in is re-simulated with the archived model
    and seeds (``multi_stim.simulate``'s per-fish loop, one block), and the provider's latched anchor is used. The check
    is exact: every bout of that block equals the archived run ``Model/multi_stim/28b_effects_ang90.pkl``. The dots are
    the provider's own placed templates (``SceneProvider.tracks``: ``TemplateBank.place`` with ``template_index``).
    Drawn in the fish's frame at onset (heading up, the fish's left on the left) at Chapter 3's physical 3.60 px/mm:
    the two-group code divides px by 3.67 (multi_stim.PX_PER_MM, asserted), so the drawing uses its mm x
    style.MM_3P67_TO_3P60 (= px / 3.60); the archive checks run in the code's own 3.67 mm before that.

(b) What one fish sees in the multi-fish experiment (``scripts/multi_fish.py``), one real frame of the archived run
    ``Model/multi_fish/A5_N10_effects.pkl`` (F seed 0, 10 fish, 5x dish, 'effects'): at the focal fish's bout onset,
    the other fish where the scheduler shows them (``pos_at``), the view horizon (``RADIUS_PX``), the dot groups
    (``neighbour_groups`` / ``cluster``) and their centroids; checked against the run's own per-bout log (number seen,
    number of groups, largest group, nearest-group range). Drawn in the focal fish's frame (heading up); 3.60 px/mm,
    the multi-fish code's scale. The frame was chosen by the criteria in ``pick_frames`` (documented there).

    uv run python scripts/report_figs/fig_scene.py            # the figure
    uv run python scripts/report_figs/fig_scene.py --facts    # + the numbers and checks, as JSON on stdout
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403,E402  (sets rcParams, gives save(), panel(), colours)

import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, Circle  # noqa: E402

import model_figures as MF  # noqa: E402
import multi_fish as MFi  # noqa: E402
import multi_stim as MS  # noqa: E402
import stim_program as SPm  # noqa: E402
import stim_rollout as SR  # noqa: E402

# ---- panel (a): which scene, model, fish and block ------------------------------------------------
SCENE_A, TAG_A, FISH_A, BLOCK_A, SEED_A = "ang90", "28b", 0, 0, 0
C_A, C_B = BLUE, SKY                     # both groups are moving dots (COND_C moving = blue), told apart by lightness
FISH_C = VERM                            # the simulated fish whose scene / view is drawn, in both panels

# ---- panel (b): which run and frame (see pick_frames) ---------------------------------------------
COND_B = "A5_N10_effects"
FRAME_B = (4, 1, 1076)                   # (dish g, focal fish i, its bout index k), picked from pick_frames()
VIEW_MM = 92.0                           # half-width of panel (b) around the focal fish
OUTLINE_PAD_MM = 4.5                     # a group outline = its spread (farthest member from the centroid) + this


# =================================================================================================
# helpers
# =================================================================================================
_DART = np.array([[0.0, 1.0], [0.42, -0.62], [0.0, -0.32], [-0.42, -0.62], [0.0, 1.0]])


def fish_marker(angle_rad):
    """A dart pointing along ``angle_rad`` (display frame, 0 = up, + = counter-clockwise), as marker vertices."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return [tuple(v) for v in _DART @ np.array([[c, s], [-s, c]])]


def scale_bar(ax, x0, y0, length_mm, label=None, color=INK):
    ax.plot([x0, x0 + length_mm], [y0, y0], color=color, lw=1.4, solid_capstyle="butt", zorder=6)
    ax.annotate(label or f"{length_mm:g} mm", (x0 + length_mm / 2, y0), xytext=(0, -2.5), textcoords="offset points",
                ha="center", va="top", fontsize=7, color=color, zorder=6)


def _seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(p - a - t * (b - a))))


def label_spot(q, V, pad, gap=4.5):
    """Where to write a group's count: of 16 spots around its outline, the one farthest from everything drawn
    (fish, other outlines, the bearing lines, the view circle)."""
    rad = V["facts"]["radius_mm"]
    best = None
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        rr = q["spread_mm"] + pad + gap
        p = np.array([q["x"] + rr * np.cos(ang), q["y"] + rr * np.sin(ang)])
        d = [np.hypot(*(p - P)) for P in V["P"]] + [np.hypot(*p)]
        d += [np.hypot(p[0] - o["x"], p[1] - o["y"]) - o["spread_mm"] - pad for o in V["groups"]]
        d += [_seg_dist(p, (0, 0), (o["x"], o["y"])) for o in V["groups"]]
        d += [abs(np.hypot(*p) - rad)]
        m = min(d)
        if best is None or m > best[0] + 1e-9:
            best = (m, p)
    return best[1]


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# =================================================================================================
# (a) the two-group scene
# =================================================================================================
def scene_data():
    """The block-``BLOCK_A`` scene of fish ``FISH_A``: placed tracks in the fish's onset frame (mm), the dish, checks."""
    scene = MS.SCENES[SCENE_A]
    models = _quiet(MS._models, TAG_A)
    _, dish, ref = MF._programme()
    bank = MS.template_bank()
    per = MS.N_FISH // len(models)
    m = models[FISH_A // per]                                      # fish k runs on fold model k // 10, as simulate()
    rng = np.random.default_rng([SEED_A, FISH_A])
    pose = ref.sample_start_pose(rng, dish)
    prov = MS.SceneProvider(scene, bank, FISH_A, BLOCK_A + 1, visible=None, dish=dish)
    rec = SR.make_synthetic_record(int(prov.T / 0.2) + 100, pose, dish, fish_id=FISH_A)
    g = m.sample(rec, rng, stim_provider=prov, stop_time_s=prov.T, block_seed=(SEED_A, FISH_A), multi_rule="effects")

    # --- check 1: the re-simulated block equals the archived run, bout for bout
    fo = np.round(np.asarray(g["t_onset_s"]) * prov.fps).astype(int)
    loc = np.array([prov.locate(f) for f in fo]).reshape(-1, 3)
    keep = np.flatnonzero(loc[:, 1] == BLOCK_A)
    arch = MS.load(TAG_A, "effects", SCENE_A)
    ab = arch["bouts"]
    ab = ab[(ab.fish == FISH_A) & (ab.blk == BLOCK_A)].sort_values("t_blk")
    xs, ys = np.asarray(g["x_px"])[keep], np.asarray(g["y_px"])[keep]
    assert len(keep) == len(ab), "re-simulated block has a different number of bouts than the archive"
    dev = float(max(np.abs(xs - ab.x.to_numpy()).max(), np.abs(ys - ab.y.to_numpy()).max()))
    assert dev < 1e-3, f"re-simulated block differs from the archive by {dev} px"

    ax_, ay_, ao = prov.anchors[1 + 2 * BLOCK_A]                   # the pose latched at the block's first bout
    k0 = int(loc[keep[0], 2])                                      # template frame at the latch
    tracks = prov.tracks[BLOCK_A]                                  # [(frames, n, 2) left, forward px] per group
    assert_code_scale(MS.PX_PER_MM, "multi_stim.PX_PER_MM")
    px = MS.PX_PER_MM                                              # the code's 3.67 px/mm: archive checks only
    px_d = px / MM_3P67_TO_3P60                                    # = 3.60 px/mm: everything drawn or reported

    # --- check 2: the provider's own world centroids at the latch == my transform of the placed tracks
    prov.anchor = (ax_, ay_, ao)
    cw = prov._centroids(BLOCK_A, k0)
    for gi, tr in enumerate(tracks):
        ox, oy = SPm.anchor_offsets(tr[k0, :, 0], tr[k0, :, 1], ao)
        assert np.allclose(cw[gi], ((ax_ + ox).mean(), (ay_ + oy).mean()))
    # --- check 3: the archived first-bout distance to each group == the drawn onset distance
    d_onset = [float(np.hypot(*tr[k0].mean(axis=0))) / px for tr in tracks]
    d_arch = [float(ab.dA.iloc[0]), float(ab.dB.iloc[0])]
    assert np.allclose(d_onset, d_arch, atol=1e-3), (d_onset, d_arch)

    # --- check 4: two groups of one condition are exact mirror images (multi_stim: same template, bearing -theta)
    mirror = bool(scene.groups[0][:2] == scene.groups[1][:2]
                  and np.allclose(tracks[1][..., 0], -tracks[0][..., 0]) and np.allclose(tracks[1][..., 1], tracks[0][..., 1]))

    # fish frame -> display (heading up, the fish's left on the left), mm at 3.60 px/mm
    def disp(L, F):
        return -np.asarray(L) / px_d, np.asarray(F) / px_d

    cx, cy, R = dish
    ox, oy = cx - ax_, cy - ay_                                    # dish centre relative to the fish, world
    Lc = -ox * np.sin(ao) + oy * np.cos(ao)                        # inverse of stim_program.anchor_offsets
    Fc = ox * np.cos(ao) + oy * np.sin(ao)
    assert np.allclose(SPm.anchor_offsets(Lc, Fc, ao), (ox, oy))     # the inverse is exact
    trial = arch["trials"]
    trial = trial[(trial.fish == FISH_A) & (trial.blk == BLOCK_A)].iloc[0]
    out = dict(
        tracks=[disp(tr[..., 0], tr[..., 1]) for tr in tracks],
        onset=[disp(tr[k0, :, 0], tr[k0, :, 1]) for tr in tracks],
        centroid=[disp(tr[k0, :, 0].mean(), tr[k0, :, 1].mean()) for tr in tracks],
        dish=(float(disp(Lc, Fc)[0]), float(disp(Lc, Fc)[1]), R / px_d),
        groups=scene.groups, desc=scene.desc, k0=k0, n_frames=int(tracks[0].shape[0]), fps=prov.fps,
        facts=dict(scene=SCENE_A, model_tag=TAG_A, fish=FISH_A, block=BLOCK_A, fold_model=FISH_A // per,
                   n_bouts_block=int(len(keep)), max_dev_vs_archive_px=dev, latch_frame=k0,
                   onset_dist_px=[round(v * px, 3) for v in d_onset],
                   onset_dist_mm=[round(v * MM_3P67_TO_3P60, 3) for v in d_onset],           # 3.60 px/mm, as drawn
                   check_code_mm_3p67=dict(onset=[round(v, 3) for v in d_onset],             # checked vs archive
                                           archived_first_bout=[round(v, 3) for v in d_arch]),
                   onset_bearing_deg=[round(float(np.degrees(np.arctan2(tr[k0, :, 0].mean(), tr[k0, :, 1].mean()))), 3)
                                      for tr in tracks],
                   dish_radius_mm=round(R / px_d, 2), fish_r_over_R=round(float(np.hypot(ox, oy) / R), 3),
                   px_per_mm=round(px_d, 6), code_px_per_mm=px,
                   n_templates=[bank.size(*grp[:2]) for grp in scene.groups],
                   template_draw_A=MS.template_index(FISH_A, BLOCK_A, 0) % bank.size(*scene.groups[0][:2]),
                   b_is_mirror_of_a=mirror,
                   archived_trial=dict(kept=bool(trial.ok), choice=trial.choice, si_late=round(float(trial.si_late), 3),
                                       late_inside=round(float(trial.late_in), 3))),
    )
    return out


def draw_scene(ax, S):
    dcx, dcy, R = S["dish"]
    cols = (C_A, C_B)
    allX = np.concatenate([X.ravel() for X, _ in S["tracks"]])
    allY = np.concatenate([Y.ravel() for _, Y in S["tracks"]])
    xw = float(np.abs(allX).max()) + 6.0                          # window: the tracks, symmetric about the heading
    ylo, yhi = -9.0, float(allY.max()) + 7.0
    ax.add_patch(Circle((dcx, dcy), R, fill=False, ec=MUTED, lw=0.9, zorder=1))     # clipped to the window
    for gi, ((X, Y), (x0, y0), (cxg, cyg)) in enumerate(zip(S["tracks"], S["onset"], S["centroid"])):
        for n in range(X.shape[1]):
            ax.plot(X[:, n], Y[:, n], color=cols[gi], lw=0.45, alpha=0.28, zorder=2, solid_capstyle="round")
        ax.plot([0, cxg], [0, cyg], color=INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
        ax.plot(x0, y0, "o", color=cols[gi], ms=4.4, mec="white", mew=0.5, zorder=4)
        ax.plot(cxg, cyg, marker="+", color=INK, ms=7, mew=1.2, zorder=5)
    # heading axis and the two bearings
    d = S["facts"]["onset_dist_mm"]
    th = S["facts"]["onset_bearing_deg"]
    ax.plot([0, 0], [0, yhi], color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=2)
    r_arc = 0.42 * min(d)
    ax.add_patch(Arc((0, 0), 2 * r_arc, 2 * r_arc, theta1=90, theta2=90 + th[0], color=INK, lw=0.7, zorder=3))
    ax.add_patch(Arc((0, 0), 2 * r_arc, 2 * r_arc, theta1=90 + th[1], theta2=90, color=INK, lw=0.7, zorder=3))
    for t_, lab in ((th[0], r"$+\theta$"), (th[1], r"$-\theta$")):
        a_mid = np.radians(90 + t_ / 2)
        ax.text((r_arc + 3.6) * np.cos(a_mid), (r_arc + 3.6) * np.sin(a_mid), lab, ha="center", va="center",
                fontsize=7.5, zorder=6)
    for gi, (cxg, cyg) in enumerate(S["centroid"]):
        lab = "AB"[gi]
        u = np.array([cxg, cyg]) / np.hypot(cxg, cyg)
        nrm = np.array([u[1], -u[0]]) * (1 if cxg > 0 else -1)        # perpendicular, pointing away from the axis
        nrm = nrm if nrm[1] <= 0 else -nrm
        m = 0.62 * np.array([cxg, cyg]) + 3.2 * nrm
        ax.text(m[0], m[1], rf"$d_{lab}$", ha="center", va="center", fontsize=7.5, zorder=6)
        ax.text(cxg + 5.5 * u[0] + (-4.0 if cxg < 0 else 4.0), cyg - 1.0, lab, ha="center", va="center",
                fontsize=9, fontweight="bold", color=cols[gi], zorder=6)
    ax.plot(0, 0, marker=fish_marker(0.0), ms=12, color=FISH_C, mec="white", mew=0.3, ls="none", zorder=7)
    ax.set_xlim(-xw, xw)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    scale_bar(ax, xw - 12.0, ylo + 3.0, 10.0)


def legend_scene(ax, S):
    f = S["facts"]
    n, mv, _, _ = S["groups"][0]
    th = abs(f["onset_bearing_deg"][0])
    h = [Line2D([], [], marker=fish_marker(0.0), ms=9, color=FISH_C, ls="none"),
         Line2D([], [], marker="o", ms=4.4, color=C_A, mec="white", mew=0.5, ls="none"),
         Line2D([], [], color=C_A, lw=0.8, alpha=0.6),
         Line2D([], [], marker="+", ms=7, mew=1.2, color=INK, ls="none"),
         Line2D([], [], color=MUTED, lw=0.9)]
    lab = ["fish at onset", "dot at onset", f"dot track, {S['n_frames'] / S['fps']:.0f} s", "group centroid",
           "dish wall"]
    leg = ax.legend(h, lab, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.5, handletextpad=0.5,
                    borderaxespad=0, labelspacing=0.5)
    leg._legend_box.align = "left"
    note = (f"{n} v {n} {'moving' if mv else 'static'} dots,\n"
            + rf"$\theta$ = {th:.0f}°," + "\n"
            + rf"$d_A = d_B = {f['onset_dist_mm'][0]:.0f}$ mm." + "\n"
            + "A and B swap sides\nevery block.\n"
            + "References: only A,\nonly B or neither\nshown, same tracks.")
    ax.text(0.0, 0.53, note, transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK, linespacing=1.35)
    ax.axis("off")


# =================================================================================================
# (b) how a fish sees the other fish
# =================================================================================================
def view_data(cond=COND_B, frame=FRAME_B):
    """One bout onset of one focal fish: positions (focal frame, mm), groups, members; checked against the run's log."""
    blob = MFi.load(cond)
    spec = blob["spec"]
    g, i, k = frame
    grp = blob["groups"][g]
    f = grp[i]
    t = float(f["t"][k])
    x, y = float(f["x"][k]), float(f["y"][k])                     # pre-bout pose = what the provider was called with
    h = float(MFi.heading_at(f, np.array([np.nextafter(t, -np.inf)]))[0])     # heading before this bout
    js = [j for j in range(len(grp)) if j != i]
    P = np.array([MFi.pos_at(grp[j], np.array([t]))[0] for j in js])           # where the scheduler shows them
    H = np.array([MFi.heading_at(grp[j], np.array([t]))[0] for j in js])
    groups, lg = MFi.neighbour_groups(x, y, P, spec.get("mapping", "cluster"))
    L = f["log"][k]                                                # (t, n_vis, G, n_raw_max, spread_ratio, r_near, n_out)
    assert abs(L[0] - t) < 1e-3 and int(L[1]) == lg["n_vis"] and int(L[2]) == lg["G"] \
        and int(L[3]) == lg["n_raw_max"] and abs(L[5] - lg["r_near"]) < 1e-2, (L, lg)
    d = np.hypot(P[:, 0] - x, P[:, 1] - y)
    vis = d <= MFi.RADIUS_PX
    cl = MFi.cluster(P[vis])                                       # the same call neighbour_groups makes
    members = [[int(np.flatnonzero(vis)[c]) for c in cc] for cc in cl]
    for (gx, gy, _, gn), mem in zip(groups, members):              # membership reproduces the groups exactly
        assert np.allclose((gx, gy), P[mem].mean(0)) and gn == min(len(mem), MFi.N_CAP)
    a = np.pi / 2 - h                                              # rotate: focal heading -> up

    def disp(px_, py_):
        rx, ry = np.asarray(px_) - x, np.asarray(py_) - y
        return (rx * np.cos(a) - ry * np.sin(a)) / MFi.PX_MM, (rx * np.sin(a) + ry * np.cos(a)) / MFi.PX_MM

    PX, PY = disp(P[:, 0], P[:, 1])
    cx, cy, R = MFi.dish(spec["arena"])
    dcx, dcy = disp(cx, cy)
    G = []
    for (gx, gy, _, gn), mem in zip(groups, members):
        X, Y = disp(gx, gy)
        G.append(dict(x=float(X), y=float(Y), n=int(gn), members=mem,
                      r_mm=float(np.hypot(gx - x, gy - y) / MFi.PX_MM),
                      bearing_deg=float(np.degrees(np.arctan2(-X, Y))),          # + = left of the heading
                      spread_mm=float(np.max(np.hypot(PX[mem] - X, PY[mem] - Y)))))
    G.sort(key=lambda q: q["r_mm"])
    return dict(P=np.stack([PX, PY], 1), H=H - h, vis=vis, groups=G, dish=(float(dcx), float(dcy), R / MFi.PX_MM),
                facts=dict(cond=cond, label=MFi.LABEL[cond], model_tag=spec["tag"], N=spec["N"], arena=spec["arena"],
                           rule=spec["rule"], dish_g=g, focal_i=i, bout_k=k, t_s=round(t, 2),
                           readout_from_s=spec["t_read"], n_others=len(js), n_seen=int(vis.sum()),
                           n_hidden=int((~vis).sum()), hidden_dist_mm=[round(float(v), 1) for v in d[~vis] / MFi.PX_MM],
                           groups=[dict(n=q["n"], r_mm=round(q["r_mm"], 1), bearing_deg=round(q["bearing_deg"], 1),
                                        spread_mm=round(q["spread_mm"], 1)) for q in G],
                           log_row=dict(n_vis=int(L[1]), G=int(L[2]), n_raw_max=int(L[3]),
                                        r_near_mm=round(float(L[5]) / MFi.PX_MM, 2)),
                           radius_px=MFi.RADIUS_PX, radius_mm=round(MFi.RADIUS_PX / MFi.PX_MM, 1),
                           px_per_mm=MFi.PX_MM, dish_radius_mm=round(R / MFi.PX_MM, 1),
                           focal_r_over_R=round(float(np.hypot(x - cx, y - cy) / R), 3),
                           envelope_mm={n: round(MFi.envelope_px(n) / MFi.PX_MM, 1) for n in (2, 4, 8)}))


GREY_FILL, GREY_EDGE = "#d9d9d9", "#8c8c8c"                     # a fish the focal fish does not see
VIEW_FILL = "#f2f8fb"                    # BLUE at 5 % on white, opaque (no transparency in the PDF)


def view_stats(cond):
    """How often the view horizon and the pooling matter, from the run's own per-bout log (readout window only):
    share of bouts with at least one other fish unseen, and the distribution of the number of groups seen."""
    blob = MFi.load(cond)
    spec = blob["spec"]
    L = np.concatenate([f["log"][(f["t"] >= spec["t_read"]) & (f["t"] <= spec["T"])]
                        for grp in blob["groups"] for f in grp])
    G = L[:, 2].astype(int)
    return dict(cond=cond, n_bouts=int(len(L)), share_some_unseen=round(float(np.mean(L[:, 1] < spec["N"] - 1)), 4),
                share_groups={int(k): round(float(np.mean(G == k)), 3) for k in np.unique(G)},
                share_largest_group_ge_9=round(float(np.mean(L[:, 3] >= 9)), 3))


def draw_view(ax, V, view=VIEW_MM):
    rad = V["facts"]["radius_mm"]
    dcx, dcy, R = V["dish"]
    if np.hypot(dcx, dcy) - R < view * 1.45:                       # the dish wall, where it enters the window
        ax.add_patch(Circle((dcx, dcy), R, fill=False, ec=MUTED, lw=0.9, zorder=1))
    ax.add_patch(Circle((0, 0), rad, fill=True, fc=VIEW_FILL, ec="none", zorder=0))
    ax.add_patch(Circle((0, 0), rad, fill=False, ec=BLUE, lw=0.8, ls=(0, (4, 2.5)), zorder=1))
    pad = OUTLINE_PAD_MM
    for q in V["groups"]:
        ax.plot([0, q["x"]], [0, q["y"]], color=BLUE, lw=0.8, zorder=2)
        ax.add_patch(Circle((q["x"], q["y"]), q["spread_mm"] + pad, fill=True, fc="white", ec=BLUE, lw=0.9,
                            zorder=3))
        if q["n"] > 1:                                             # a one-fish group's centroid is the fish
            ax.plot(q["x"], q["y"], marker="+", color=BLUE, ms=7, mew=1.2, zorder=6)
        lx, ly = label_spot(q, V, pad)
        ax.text(lx, ly, str(q["n"]), ha="center", va="center", fontsize=8, fontweight="bold", color=BLUE, zorder=6)
    for (px_, py_), hh, v in zip(V["P"], V["H"], V["vis"]):
        ax.plot(px_, py_, marker=fish_marker(hh), ms=8, mfc=INK if v else GREY_FILL, mec="white" if v else GREY_EDGE,
                mew=0.3 if v else 0.5, ls="none", zorder=4 if v else 2)
    ax.plot(0, 0, marker=fish_marker(0.0), ms=12, color=FISH_C, mec="white", mew=0.3, ls="none", zorder=7)
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    scale_bar(ax, view - 22.0, -view + 3.0, 20.0)


def legend_view(ax, V):
    f = V["facts"]
    h = [Line2D([], [], marker=fish_marker(0.0), ms=9, color=FISH_C, ls="none"),
         Line2D([], [], marker=fish_marker(0.0), ms=7, color=INK, ls="none"),
         Line2D([], [], marker=fish_marker(0.0), ms=7, mfc=GREY_FILL, mec=GREY_EDGE, mew=0.5, ls="none"),
         Line2D([], [], color=BLUE, lw=0.8, ls=(0, (4, 2.5))),
         Line2D([], [], marker="o", ms=7, mfc="white", mec=BLUE, mew=0.9, ls="none"),
         Line2D([], [], color=BLUE, lw=0.8, marker="+", ms=7, mew=1.2)]
    lab = ["focal fish", "fish seen", "fish not seen", f"view, {f['radius_mm']:.0f} mm", "dot group",
           "bearing to\ncentroid"]
    leg = ax.legend(h, lab, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.5, handletextpad=0.5,
                    borderaxespad=0, labelspacing=0.5)
    leg._legend_box.align = "left"
    note = ("Neighbours are\nmoving dots;\nclose ones pool\ninto one group\n(number = count,\n"
            f"at most {MFi.N_CAP}).")
    ax.text(0.0, 0.45, note, transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK, linespacing=1.35)
    ax.axis("off")


# =================================================================================================
# frame choice for (b) (run once; the chosen frame is FRAME_B)
# =================================================================================================
def pick_frames(cond=COND_B, step=5):
    """Candidate frames for (b), every ``step``-th qualifying bout after the readout start: 5-8 of the 9 other fish
    seen; three groups of three different sizes, the largest >= 3; every unseen fish inside the window and >= 8 mm
    clear of every group outline; outlines (spread + OUTLINE_PAD_MM) >= 8 mm apart and >= 5 mm clear of the focal
    fish. The figure's frame was picked by eye among these for legibility; it is a selection, not a typical frame."""
    blob = MFi.load(cond)
    spec = blob["spec"]
    out = []
    for g, grp in enumerate(blob["groups"]):
        for i, f in enumerate(grp):
            L = f["log"]
            ok = np.flatnonzero((f["t"] >= spec["t_read"]) & (f["t"] <= spec["T"]) & (L[:, 1] >= 5) & (L[:, 1] <= 8)
                                & (L[:, 2] == 3) & (L[:, 3] >= 3))
            for k in ok[::step]:
                V = view_data(cond, (g, i, int(k)))
                G = V["groups"]
                if len({q["n"] for q in G}) != 3:
                    continue
                H = V["P"][~V["vis"]]
                if not len(H) or np.abs(H).max() > VIEW_MM - 6:
                    continue
                clear_h = min(np.hypot(hx - q["x"], hy - q["y"]) - q["spread_mm"] - OUTLINE_PAD_MM
                              for hx, hy in H for q in G)
                gaps = [np.hypot(a["x"] - b["x"], a["y"] - b["y"]) - a["spread_mm"] - b["spread_mm"]
                        - 2 * OUTLINE_PAD_MM for ai, a in enumerate(G) for b in G[ai + 1:]]
                foc = min(q["r_mm"] - q["spread_mm"] - OUTLINE_PAD_MM for q in G)
                if clear_h >= 8 and min(gaps) >= 8 and foc >= 5:
                    out.append(((g, i, int(k)), V["facts"]["n_seen"], [q["n"] for q in G], round(min(gaps), 1),
                                round(clear_h, 1)))
    return out


# =================================================================================================
def main(facts=False):
    S = scene_data()
    V = view_data()
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.45))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.15, 0.98, 2.2, 0.97], wspace=0.04,
                          left=0.012, right=0.985, top=0.9, bottom=0.02)
    axa, la, axb, lb = (fig.add_subplot(gs[0, j]) for j in range(4))
    draw_scene(axa, S); legend_scene(la, S)
    draw_view(axb, V); legend_view(lb, V)
    panel(axa, "a", "Two dot groups")
    panel(axb, "b", "Several fish")
    # the two axes differ in height (equal aspect): put both letters at the same height on the page
    fig.canvas.draw()
    yb = axb.texts[-1].get_window_extent().y1
    ta = axa.texts[-1]
    ta.set_y(axa.transAxes.inverted().transform((0, yb))[1])
    save(fig, "fig_scene")
    if facts:
        V["facts"]["envelope_mm"][3] = round(MFi.envelope_px(3) / MFi.PX_MM, 1)
        stats = [view_stats(c) for c in (COND_B, "A1_N10_effects")]
        print(json.dumps(dict(a=S["facts"], b=V["facts"], b_context=stats), indent=1, default=str))


if __name__ == "__main__":
    main(facts="--facts" in sys.argv)
