"""Structural guard for rollout-aware fine-tuning (scripts/rollout_aware.py).

Cheap fit on purpose (4 fish, 4 epochs, hidden 64): checks WIRING, not quality. The failure mode
that matters is silent -- a policy gradient computed for a DIFFERENT policy than the one that
generated the actions trains happily, reports a falling loss, and optimises nothing. That is
exactly the defect in the superseded rollout_finetune.py (it evaluated the head at z=0 while the
rollout ran at z ~ prior), and no amount of watching the loss curve would reveal it.

So section [2] tests the REINFORCE consistency condition directly rather than eyeballing the code:

    E_{a ~ p(.|x,z)} [ grad_theta log p(a|x,z) ] = 0

Under the policy that actually acted this expectation is zero and the sample mean is pure Monte
Carlo noise. Under any mismatched policy -- wrong latent, wrong action -- it is a real gradient
and stands far above that noise floor. The noise floor is MEASURED by split-half rather than
assumed, so the check does not depend on a magic threshold.

    python scripts/smoke_rollout_aware.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E                    # noqa: E402
import fit_MLP_border as MB             # noqa: E402
import fit_MLP_mix as MX                # noqa: E402
import fit_MLP_joint as JT              # noqa: E402
import fit_MLP_fishre as FR             # noqa: E402
import generative_metrics as GM         # noqa: E402
import rollout_aware as RA              # noqa: E402

ROOT = Path(__file__).parent.parent
ok_all = True


def check(label, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def _mean_grad(net, params, logp_fn):
    """(flat mean gradient over bouts, n) for a scalar-sum log-prob builder."""
    for p in params:
        if p.grad is not None:
            p.grad = None
    logp, n = logp_fn()
    logp.sum().backward()
    g = torch.cat([(p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel()))
                   for p in params]).detach().numpy() / max(n, 1)
    for p in params:
        p.grad = None
    return g


def main():
    recs = MB.build_border_records(pd.read_csv(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    fids = sorted(recs)
    train = [recs[f] for f in fids[:4]]
    ctx = E.make_context(train, dict(mlp_epochs=4))

    m = FR.MLPMixJointREDensity(K=8, K_kin=3, stim_mode="shoal", hidden=64, epochs=4,
                                n_dist=2, n_dt=3, re_m=2, re_ghq_k=3, re_gain_search=False,
                                dt_snap=True)
    m.name = "smoke_ra"
    m.fit(train, ctx)
    net = m._model
    cfg = m._cfg_feat
    # Match the `rej` generation-time configuration: physical shallow-border rejection with a
    # zero turn bias. Injected rather than searched -- the calibrators are not what is under test.
    cfg["centre_bias"] = dict(bulk=0.0, shell=0.0, u0=0.75, reject=True)
    cfg["wall_reject"] = dict(d0=2.0, d1=10.0, max_draws=20)
    cfg["wall_mode"] = "reflect"
    rec = train[0]
    gen_kw = dict(closed_loop=m.closed_loop, guardrail=m.guardrail, eps=m.eps,
                  stim_loop=m.stim_loop)

    print("\n=== [1] the generator exposes what the emission actually drew ===")
    a = JT.generate_mix_joint_loop(net, rec, cfg, np.random.default_rng(5), **gen_kw)
    g = JT.generate_mix_joint_loop(net, rec, cfg, np.random.default_rng(5),
                                   keep_features=True, **gen_kw)
    check("keep_features does not perturb the rollout",
          all(np.array_equal(a[k], g[k], equal_nan=True)
              for k in ("dtheta", "distance_px", "dt_s", "x_px", "y_px")))
    check("policy handles present",
          all(k in g for k in ("X", "dth_base", "dist_base", "dt_base", "s_state", "z_re")))
    check("z_re is the m-vector that acted", g["z_re"].shape == (net.m,) and np.any(g["z_re"] != 0),
          f"z={np.round(g['z_re'], 4)}")
    # THE TRAP THIS EXISTS TO CATCH: dt_snap makes the reported interval a quantised version of
    # the drawn one, so scoring `dt_s` evaluates the mixture at a point it never sampled.
    dsnap = float(np.max(np.abs(g["dt_base"] - g["dt_s"])))
    check("dt_snap really does move the emitted interval (so dt_base != dt_s matters)",
          dsnap > 1e-4, f"max|dt_base - dt_s| = {dsnap:.4f} s")
    check("dth_base == dtheta when the containment bias is zero",
          np.allclose(g["dth_base"], g["dtheta"]))

    print("\n=== [2] THE LOAD-BEARING CHECK: the score function matches the policy that acted ===")
    # Probe at the REAL operating point, not this cheap fit's. Two knobs matter and both are set
    # to what a shipped model carries, because the test is a power calculation as much as a
    # correctness one -- a trap you cannot resolve above the noise floor is a trap you have not
    # tested for:
    #   * fish_re sd -> 1.0 (the shipped rej value). The 4-epoch fit above lands sigma at ~0.16,
    #     where z=0 and z=z_re are nearly the SAME policy and the check has no power.
    #   * centre_bias bulk -> 4.0, so dtheta and dth_base genuinely differ. With rej's bulk=0 they
    #     are equal and "scored the post-bias array" is untestable on that config alone.
    # dt_snap stays on, so the emitted interval is quantised as well.
    cfg2 = dict(cfg)
    cfg2["fish_re"] = dict(cfg["fish_re"], sd=np.ones(net.m))
    cfg2["centre_bias"] = dict(bulk=4.0, shell=0.0, u0=0.75)
    cfg2.pop("wall_reject", None)          # isolate the score function from the rejection bias
    params, _ = RA.trainable(net)
    for p in net.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    def _variant(gd, kind):
        gg = dict(gd)
        if kind == "emitted":              # WRONG action: post-bias / post-vigor / post-snap
            gg.update(dth_base=gd["dtheta"], dist_base=gd["distance_px"], dt_base=gd["dt_s"])
        elif kind == "az_pre":             # WRONG conditioner: a_z from the PRE-bias turn
            gg["dtheta"] = gd["dth_base"]
        return gg

    def _sum_logp(gd, kind="correct", chan="all"):
        la, ld, lt, msk = RA.logp_actions(net, _variant(gd, kind), cfg2)
        return {"all": la + ld + lt, "turn": la, "dist": ld}[chan][msk], int(msk.sum())

    # Independent REPLICATES, not a split half: E[grad log p] = 0 holds per bout, but consecutive
    # bouts of one closed-loop rollout share a latent and a state, so two halves of the same
    # trajectory are not independent draws of the same zero.
    n_rep = 8
    G = {}
    clipf = []
    for k in range(n_rep):
        gk = JT.generate_mix_joint_loop(net, train[k % len(train)], cfg2,
                                        np.random.default_rng([77, k]),
                                        keep_features=True, **gen_kw)
        clipf.append(RA.clip_count(gk, cfg2) / len(gk["dtheta"]))
        for kind, chan in (("correct", "turn"), ("correct", "dist"),
                           ("emitted", "turn"), ("az_pre", "dist")):
            G.setdefault((kind, chan), []).append(
                _mean_grad(net, params, lambda gk=gk, kind=kind, chan=chan: _sum_logp(gk, kind, chan)))

    def _ratio(A, ref=None):
        """||mean gradient|| in units of its own standard error; paired against ``ref`` if given.

        Under the null (the score function IS the policy that acted) the mean is zero and
        E||gbar||^2 = sum_c var_c / n_rep = SE^2, so the ratio concentrates at 1.0 -- tightly,
        since it sums over ~2k parameter dimensions. MEASURED on the shipped rej checkpoint over
        8 rollouts, every channel: 0.95-1.36 at bulk 0 and bulk 4. That is the yardstick.
        """
        D = np.stack(A) - (np.stack(ref) if ref is not None else 0.0)
        se = float(np.sqrt(np.sum(D.var(0, ddof=1)) / n_rep))
        return float(np.linalg.norm(D.mean(0))) / max(se, 1e-12)

    # The TURN channel carries the strict identity test. The distance channel cannot: the
    # generator clips it at the real q99.9 (``kin_clip``), which folds the tail onto an atom that
    # ``_lnk_logpdf`` is not the density of. MEASURED -- shipped rej clips 1.9% of bouts and holds
    # at ratio 1.17; this deliberately under-trained 4-epoch fixture clips ~12% and sits at 3.2.
    # Note that dropping the clipped bouts does NOT repair it and is not a valid fix:
    # E[grad log p(a) | a in A] = grad log P(A), which is a different non-zero quantity.
    r_turn = _ratio(G[("correct", "turn")])
    print(f"       clip {np.mean(clipf):.2%} of bouts (shipped rej: 1.9%)")
    print(f"       ||mean grad||/SE, null == 1.0:  turn identity {r_turn:.2f}   "
          f"emitted-action {_ratio(G[('emitted', 'turn')], G[('correct', 'turn')]):.2f} (paired)  "
          f"az-pre-bias {_ratio(G[('az_pre', 'dist')], G[('correct', 'dist')]):.2f} (paired)")
    check("correct policy: E[grad log p] is consistent with zero on the turn channel",
          r_turn <= 3.0, f"ratio {r_turn:.2f}")
    # PAIRED for the two small perturbations: both variants share the same rollouts, so the
    # rollout-to-rollout spread of the gradient -- which dwarfs a 2.6 deg bias on a 70 deg
    # spread -- cancels. Same common-random-numbers logic the ladder uses to compare arms.
    check("scoring the EMITTED values is detectably wrong",
          _ratio(G[("emitted", "turn")], G[("correct", "turn")]) > 3.0,
          "dtheta is post-bias, dt_s is post-snap")
    check("conditioning a_z on the PRE-bias turn is detectably wrong",
          _ratio(G[("az_pre", "dist")], G[("correct", "dist")]) > 3.0,
          "the chained heads saw abs(dth_t) AFTER the containment shift")

    print("\n=== [2b] the latent that ACTED is the one scored (exact, not statistical) ===")
    # A statistical probe cannot test this on a 4-epoch fixture: its z_* loadings are near zero,
    # so z=0 and z=z_re are genuinely the same policy and there is nothing to detect (measured
    # ratio 0.75). Scaling the loadings up to create an effect drives the fixture into a 74%-clip
    # degenerate regime. So test it EXACTLY instead -- the parameters themselves must differ, and
    # must equal what the numpy predictor that generated the rollout produced.
    gz = JT.generate_mix_joint_loop(net, rec, cfg2, np.random.default_rng([77, 0]),
                                    keep_features=True, **gen_kw)
    with torch.no_grad():
        (a_re, d_re, t_re), _ = RA.policy_params(net, gz, cfg2)
        (a_z0, d_z0, t_z0), _ = RA.policy_params(net, gz, cfg2, z_override=np.zeros(net.m))
    dz = max(float((a_re[0] - a_z0[0]).abs().max()), float((d_re[1] - d_z0[1]).abs().max()),
             float((t_re[1] - t_z0[1]).abs().max()))
    check("scoring at z=0 gives DIFFERENT parameters (so the latent is live in the score)",
          dz > 1e-6, f"max|param diff| = {dz:.2e}  (z = {np.round(gz['z_re'], 4)})")

    print("\n=== [3] the SCORED policy is bit-for-bit the one the generator sampled from ===")
    # Routed through RA.policy_params -- the production path -- against the numpy predictor the
    # rollout actually drew from, at the SAME latent. Run on the bulk=4.0 config, so the
    # conditioner ``a_z`` is genuinely ambiguous (pre- vs post-bias turn differ by ~2.6 deg) and
    # this is an exact test of that choice as well as of the latent.
    body_angle, dist_params, dt_params = JT._numpy_predictor_joint(net, z=gz["z_re"])
    az_gen = np.abs(np.asarray(gz["dtheta"], float)) / cfg2["dth_scale"]      # what _emit used
    dm, ds, _, _ = cfg2["kin_scales"]
    lc_gen = (np.log(np.clip(gz["dist_base"], 1e-6, None)) - dm) / ds
    with torch.no_grad():
        (ang_t, dp_t, tp_t), (ya_t, _, _) = RA.policy_params(net, gz, cfg2)
    w_err = dw_err = dt_err = 0.0
    for i in range(0, len(gz["X"]), 97):
        b, ang_n = body_angle(gz["X"][i])
        w_err = max(w_err, float(np.max(np.abs(np.exp(ang_t[0][i].numpy()) - ang_n[0]))))
        _, dmu_n, _ = dist_params(b, float(az_gen[i]))
        dw_err = max(dw_err, float(np.max(np.abs(dmu_n - dp_t[1][i].numpy()))))
        _, tmu_n, _ = dt_params(b, float(az_gen[i]), float(lc_gen[i]))
        dt_err = max(dt_err, float(np.max(np.abs(tmu_n - tp_t[1][i].numpy()))))
    check("angle mixture weights agree", w_err < 1e-4, f"max|dw| = {w_err:.2e}")
    check("distance mixture locations agree", dw_err < 1e-4, f"max|dmu| = {dw_err:.2e}")
    check("dt mixture locations agree", dt_err < 1e-4, f"max|dmu| = {dt_err:.2e}")
    check("the scored ACTION is the pre-bias draw, not the emitted turn",
          np.array_equal(ya_t.numpy(), np.asarray(gz["dth_base"], np.float32))
          and not np.allclose(gz["dth_base"], gz["dtheta"]),
          f"mean|dtheta - dth_base| = {np.mean(np.abs(gz['dtheta'] - gz['dth_base'])):.3f} deg")
    # Teeth: the same comparison with the PRE-bias conditioner must fail, or [3] proves nothing.
    with torch.no_grad():
        (_, dp_w, _), _ = RA.policy_params(net, dict(gz, dtheta=gz["dth_base"]), cfg2)
    w_bad = max(float(np.max(np.abs(dist_params(body_angle(gz["X"][i])[0], float(az_gen[i]))[1]
                                    - dp_w[1][i].numpy()))) for i in range(0, len(gz["X"]), 97))
    check("...and this comparison HAS teeth (pre-bias a_z disagrees)", w_bad > 1e-4,
          f"max|dmu| = {w_bad:.2e}")

    print("\n=== [4] energy distance and its witness ===")
    rr = np.random.default_rng(0).normal(size=(200, 6))
    gg2 = np.random.default_rng(1).normal(size=(150, 6)) + 0.8
    h, D2 = RA.energy_witness(gg2, rr)
    # identity: D^2 = mean(h) + gg_mean - rr_mean
    gg_mean = RA._pdist(gg2, gg2).sum() / (150 * 149)
    check("D^2 reconstructs from the witness", abs(D2 - (h.mean() + gg_mean - RA.real_real_mean(rr))) < 1e-9)
    check("D^2 > 0 for shifted clouds", D2 > 0.1, f"D2 = {D2:.4f}")
    _, D2_same = RA.energy_witness(rr[:100], rr[100:])
    check("D^2 ~ 0 for two halves of the same sample", abs(D2_same) < 0.05, f"D2 = {D2_same:.4f}")
    check("witness ranks a far point above a near one",
          h[np.argmax(np.linalg.norm(gg2 - rr.mean(0), axis=1))] > h.mean())

    print("\n=== [5] window features are the metric's own, W=60 non-overlapping ===")
    wr = RA.real_window_feats(rec)
    wg = RA.gen_window_feats(rec, g)
    ref = GM.window_features_traj(rec.dtheta, rec.symbol, rec.distance_px, rec.dt_s,
                                  rec.dot_dist_px, W=60,
                                  radial=GM._radial_of(rec, rec.x_px, rec.y_px))
    check("real window features identical to generative_metrics", np.array_equal(wr, ref))
    check("non-overlapping: n_windows == floor(n / 60)", len(wr) == rec.n // 60,
          f"{len(wr)} windows from {rec.n} bouts")
    check("39 traj features (turn 20 + kin 10 + dot 5 + radial 4)", wr.shape[1] == 39,
          f"d = {wr.shape[1]}")
    check("generated side has the same width", wg.shape[1] == wr.shape[1])

    print("\n=== [6] the fine-tune runs, moves weights, and stays finite ===")
    before = torch.cat([p.detach().reshape(-1).clone() for p in params])
    m2 = RA.finetune(m, train[:3], steps=2, batch=2, lr=1e-4, anchor_batch=256,
                     seed=1, verbose=True, log_every=1, eval_every=2, holdout=train[3:])
    after = torch.cat([p.detach().reshape(-1) for p in params])
    hist = m2.history_ra
    check("history recorded", len(hist) == 2)
    ev = m2.evals_ra
    check("fixed-seed eval ran, in-sample and HELD-OUT reported separately",
          len(ev) == 2 and all(np.isfinite([e["D2_pool"], e["D2_pool_ho"]]).all() for e in ev),
          f"held-out D2_pool {ev[-1]['D2_pool_ho']:.3f} vs in-sample {ev[-1]['D2_pool']:.3f}")
    # batch=2 of the 3 TRAINING records, so a step can never produce more windows than the two
    # longest of them -- if the held-out fish were being rolled out too, this bound would break.
    wmax = sum(sorted((r.n // 60) for r in train[:3])[-2:])
    check("the held-out fish were NOT rolled out for training",
          all(h["n_win"] <= wmax for h in hist),
          f"windows/step {hist[0]['n_win']} <= {wmax}")
    check("gradient balance instrumented", np.isfinite(hist[0]["grad_bal"]),
          f"|g_pg|/|g_anchor| = {hist[0]['grad_bal']:.3f}")
    check("all reported quantities finite",
          all(np.isfinite([h["D2_pool"], h["D2_bf"], h["anchor_nll"], h["pg"]]).all() for h in hist))
    check("weights actually moved", float((after - before).abs().max()) > 0,
          f"max|dw| = {float((after - before).abs().max()):.2e}")
    check("redraw fraction instrumented and small",
          0.0 <= hist[0]["redraw_frac"] < 0.10, f"{hist[0]['redraw_frac']:.4f}")

    print("\n=== [7] pg_weight=0 is a real control (no policy-gradient contribution) ===")
    b2 = torch.cat([p.detach().reshape(-1).clone() for p in params])
    RA.finetune(m, train[:3], steps=1, batch=2, lr=0.0, anchor_batch=256, pg_weight=0.0,
                seed=1, verbose=False)
    check("lr=0 leaves weights untouched (harness sanity)",
          float((torch.cat([p.detach().reshape(-1) for p in params]) - b2).abs().max()) == 0.0)
    check("trainable scope excludes the shared body by default",
          "body" not in RA.trainable(net)[1] and "body" in RA.trainable(net, True)[1])

    print("\n=== [8] the objective refuses a model it cannot score ===")
    # The superseded rollout_finetune.py silently trained the WRONG thing on this family (kin
    # generator, policy evaluated at z=0, shared body moved under a single-channel reward). It has
    # been deleted; what replaces that check is a guard that the new code fails LOUDLY rather than
    # silently on a net without the latent-aware forward pass.
    class _NoRE:
        pass
    fake = type("M", (), {"_model": _NoRE(), "_cfg_feat": cfg})()
    try:
        RA.finetune(fake, train[:1], steps=1, verbose=False)
        check("a net without joint_forward_re is rejected", False, "it ran")
    except TypeError as exc:
        check("a net without joint_forward_re is rejected", "joint_forward_re" in str(exc),
              f"{str(exc)[:60]}")

    print("\nALL ROLLOUT-AWARE SMOKE CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
