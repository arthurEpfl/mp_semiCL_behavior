"""Estimate the swim-vigor latent's parameters from REAL data, so the generation-time AR(1) latent
reproduces (a) between-fish level heterogeneity and (b) within-fish slow tempo drift.

Model per channel (c = log-dt, log-dist):  m_c[t] = a_c * z_c[t]  added to the head's log-mean, with
  z_c[t] = intercept_c + drift_c[t],  drift_c[t] = phi_c*drift_c[t-1] + eps  (AR1, stationary).
We report, per channel: between-fish level std (the [B] target), the window(W=60)-median series lag-1 acf
(the [C2] drift target), and a slow/fast variance split. Also the between-channel correlation (=> shared
latent vs one-per-channel). Prints suggested (a_dt, a_dist, phi) + whether to share the latent.

    python scripts/calibrate_vigor.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import evaluate as E
import fit_MLP_border as MB

W = 60


def _logser(x):
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    return np.log(x)


def _winmed(x):
    x = np.asarray(x, float)
    return np.array([np.nanmedian(x[a:a + W]) for a in range(0, max(1, len(x) - W + 1), W)
                     if len(x[a:a + W]) == W])


def _acf1_pooled(series_list):
    num = den = 0.0
    for s in series_list:
        s = np.asarray(s, float); s = s[np.isfinite(s)]
        if len(s) < 3:
            continue
        x = s - s.mean(); num += float(np.sum(x[1:] * x[:-1])); den += float(np.sum(x * x))
    return num / den if den else float("nan")


def main():
    df = E.load_bouts(str(ROOT / "data" / "bouts_GsizeMotionSingle.csv"))
    recs = list(MB.build_border_records(df).values())

    fmean_dt, fmean_di = [], []
    win_dt_series, win_di_series = [], []       # window-median of log-value, per fish (centred within fish)
    within_dt_var, within_di_var = [], []
    for r in recs:
        ldt, ldi = _logser(r.dt_s), _logser(r.distance_px)
        fmean_dt.append(ldt.mean()); fmean_di.append(ldi.mean())
        wdt = _winmed(np.log(np.clip(np.asarray(r.dt_s, float), 1e-6, None)))
        wdi = _winmed(np.log(np.clip(np.asarray(r.distance_px, float), 1e-6, None)))
        if len(wdt) >= 3:
            win_dt_series.append(wdt - wdt.mean()); within_dt_var.append(np.var(wdt))
        if len(wdi) >= 3:
            win_di_series.append(wdi - wdi.mean()); within_di_var.append(np.var(wdi))

    fmean_dt, fmean_di = np.array(fmean_dt), np.array(fmean_di)
    lvl_dt, lvl_di = fmean_dt.std(), fmean_di.std()          # [B] between-fish level std
    acf_dt = _acf1_pooled(win_dt_series); acf_di = _acf1_pooled(win_di_series)   # [C2] drift acf (within-fish)
    corr = np.corrcoef(fmean_dt, fmean_di)[0, 1]
    win_var_dt = float(np.mean(within_dt_var)); win_var_di = float(np.mean(within_di_var))

    print(f"n_fish={len(recs)}")
    print("\n[B] between-fish LEVEL std of mean(log .):")
    print(f"    dt   = {lvl_dt:.3f}      dist = {lvl_di:.3f}")
    print(f"    corr(mean log dt, mean log dist) across fish = {corr:+.2f}  "
          f"-> {'SHARE one latent' if abs(corr) > 0.6 else 'one latent PER channel'}")
    print("\n[C2] within-fish window-median drift lag-1 acf:")
    print(f"    dt   = {acf_dt:+.3f}     dist = {acf_di:+.3f}")
    print(f"    within-fish var of window-median(log .): dt={win_var_dt:.4f}  dist={win_var_di:.4f}")

    # ---- map the OBSERVED window-median acf to a per-bout AR persistence phi ----
    # window-median samples are W bouts apart; a per-bout AR1 with persistence phi has lag-W corr phi**W,
    # but the window-median also carries fast noise that DILUTES the observed acf. Solve for the SLOW acf
    # assuming a slow/fast split: acf_obs ~ (Vslow/(Vslow+Vfast)) * rho_win, with rho_win the slow block-acf.
    # We take a pragmatic upper estimate: rho_win ~ acf_obs / f_slow, then phi = rho_win**(1/W).
    def phi_from(acf_obs, f_slow=0.7):
        rho_win = min(0.98, max(1e-3, acf_obs / f_slow))
        return float(rho_win ** (1.0 / W))
    phi_dt, phi_di = phi_from(acf_dt), phi_from(acf_di)
    print("\nsuggested per-bout AR persistence phi (drift), f_slow=0.7:")
    print(f"    phi_dt={phi_dt:.4f}   phi_dist={phi_di:.4f}   (block-acf {phi_dt**W:.2f}/{phi_di**W:.2f})")
    print("\nsuggested loadings (latent z stationary std=1): a_dt≈{:.3f}  a_dist≈{:.3f}"
          .format(lvl_dt, lvl_di), "(pure-level start; verify+bump on the full generator)")


if __name__ == "__main__":
    main()
