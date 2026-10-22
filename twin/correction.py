"""The units that are not trained from scratch (``registry.UNITS`` with make = correct / driftfix / same).

correct    F and its variants: the profile correction (``profile_finetune``) fitted on the base unit's five fold
           models, then the parameters taken at the LAST of its 100 steps (the fit itself keeps the step with the
           best running mean of its objective; the thesis uses the last step).
driftfix   D4 with the drift's generation strength fixed. The per-fold gain search gives the interval the same
           effective strength g x |lambda| in every fold, but in some folds the likelihood learns lambda_dist ~ 0 and
           the search's cap leaves them without displacement drift. Here every fold gets the same effective strength,
           g = a / |lambda| per fold and head, with the generator's log-normal re-centring c = (g^2 - 1) lambda^2 / 2;
           a is a factor (0, 1/2, 1) of the median the tau-70 calibration reached (``registry.DRIFT_FULL``). The
           likelihood is unchanged: the gains act at generation only.
same       a copy of the base unit (M2 = M1 generated with wall rejection).
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from twin import paths, profile_finetune as PF, registry

CORRECTION_ARM = "_pf5Bcr"          # pf-B (count-gated), 20 s cut-off, re-anchored warm starts


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def last_step(src_dir, arm, dst_dir):
    """Write the final-step parameter set of a fitted correction: each fold of ``arm`` in ``src_dir`` with its
    correction parameters replaced by those after the last step of its stored trace, as ``<arm>_last`` in
    ``dst_dir``, with the manifest and the held-out likelihood table."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    new = arm + "_last"
    mani = []
    for e in sorted((e for e in json.loads((src_dir / "manifest_model1.json").read_text()) if e["arm"] == arm),
                    key=lambda e: e["fold"]):
        b = _load(src_dir / e["file"])
        coef = b["profile_finetune"]["trace"][-1]["coef"]
        names = [n for n in b["state_dict"] if n.startswith("stim_turn.")]   # named_parameters order (PF._theta)
        o = 0
        for n in names:
            t = b["state_dict"][n]
            b["state_dict"][n] = torch.as_tensor(np.asarray(coef[o:o + t.numel()]).reshape(t.shape), dtype=t.dtype)
            o += t.numel()
        assert o == len(coef), (e["file"], o, len(coef))
        b["profile_finetune"] = dict(b["profile_finetune"], kept="last step", kept_coef=list(coef))
        f = e["file"].replace(arm + "_fold", new + "_fold")
        torch.save(b, dst_dir / f)
        mani.append(dict(e, file=f, arm=new))
    (dst_dir / "manifest_model1.json").write_text(json.dumps(mani, indent=1))
    PF.write_likA(dst_dir, new)
    return dst_dir


def correct_unit(key, jobs=5):
    """Fit the correction on the base unit of ``key`` (into paths.TRAIN_RUNS) and write the final-step set into
    the unit's folder."""
    u = registry.unit(key)
    base = registry.UNITS[u["base"]]
    arm = registry.arm(base)
    fit_dir = paths.TRAIN_RUNS / f"correction_M{u['key']}"
    PF.finetune_arm(registry.ckpt_dir(base), arm, fit_dir, new_arm=arm + CORRECTION_ARM, rej=registry.REJ,
                    jobs=jobs, **registry.CORRECTION)
    return last_step(fit_dir, arm + CORRECTION_ARM, registry.ckpt_dir(u))


def fix_drift(key):
    """Write unit ``key``: its base unit with the drift's generation gains fixed at ``factor`` x DRIFT_FULL."""
    u = registry.unit(key)
    base = registry.UNITS[u["base"]]
    arm, src, dst = registry.arm(base), registry.ckpt_dir(base), registry.ckpt_dir(u)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name in ("manifest_model1.json", "likA_model1.csv") or (p.suffix == ".png" and arm in p.name):
            shutil.copy2(p, dst / p.name)
    tau = base["name"].endswith("tau 200") and 200.0 or 70.0
    for p in sorted(src.glob(f"{arm}_fold[0-9].pt")):
        b = _load(p)
        d = b["cfg_feat"]["drift"]
        lam = {"dist": float(d["dist"]), "dt": float(d["dt"])}
        g = {h: 0.0 if u["factor"] == 0 else u["factor"] * registry.DRIFT_FULL[h] / max(abs(lam[h]), 1e-6)
             for h in lam}
        upd = dict(g_dist=g["dist"], g_dt=g["dt"], dist_c=0.5 * (g["dist"] * g["dist"] - 1.0) * lam["dist"] * lam["dist"],
                   dt_c=0.5 * (g["dt"] * g["dt"] - 1.0) * lam["dt"] * lam["dt"])
        d.update(upd)
        if isinstance(b.get("drift"), dict):
            b["drift"].update(upd)
        b["drift_fixed"] = dict(variant=u["name"], tau=tau, factor=u["factor"],
                                a_dist=u["factor"] * registry.DRIFT_FULL["dist"],
                                a_dt=u["factor"] * registry.DRIFT_FULL["dt"],
                                calibrated=dict(g_dist=float(b.get("drift_report", {}).get("g_dist", float("nan"))),
                                                g_dt=float(b.get("drift_report", {}).get("g_dt", float("nan")))))
        torch.save(b, dst / p.name)
    return dst


def copy_unit(key):
    """Unit ``key`` = a copy of its base unit's checkpoints (generated differently, see the registry)."""
    u = registry.unit(key)
    src, dst = registry.ckpt_dir(registry.UNITS[u["base"]]), registry.ckpt_dir(u)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.suffix in (".pt", ".json") or p.name == "likA_model1.csv":
            shutil.copy2(p, dst / p.name)
    return dst
