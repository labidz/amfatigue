"""Evaluation metrics for censored fatigue lives (all on the ln N scale)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def cindex(lt, d, risk_lnmed):
    """Harrell's concordance: shorter predicted life should fail first. risk_lnmed = predicted median ln N."""
    lt = np.asarray(lt); d = np.asarray(d); pr = np.asarray(risk_lnmed, float)
    pr = np.where(np.isfinite(pr), pr, 1e6)
    num = den = 0.0
    for i in np.flatnonzero(d > 0):
        comp = lt > lt[i]  # j outlived i, i failed
        if not comp.any():
            continue
        den += comp.sum()
        num += (pr[comp] > pr[i]).sum() + 0.5 * (pr[comp] == pr[i]).sum()
    return num / den if den else np.nan


def dcal(F_obs, d, bins=10):
    """D-calibration (Haider et al. 2020). F_obs = predicted CDF at observed ln time.

    Failures contribute their PIT value; censored observations spread uniform mass over [F(c), 1].
    Returns (chi2 statistic, p-value, max abs deviation of bin mass from 1/bins).
    """
    F = np.clip(np.asarray(F_obs, float), 0, 1); d = np.asarray(d)
    edges = np.linspace(0, 1, bins + 1)
    mass = np.zeros(bins)
    for f, di in zip(F, d):
        if di > 0:
            mass[min(int(f * bins), bins - 1)] += 1
        else:
            if f >= 1 - 1e-12:
                mass[-1] += 1
                continue
            lo = np.maximum(edges[:-1], f); hi = edges[1:]
            mass += np.clip(hi - lo, 0, None) / (1 - f)
    n = len(F)
    exp = n / bins
    chi2 = ((mass - exp) ** 2 / exp).sum()
    return chi2, 1 - stats.chi2.cdf(chi2, bins - 1), np.max(np.abs(mass / n - 1 / bins))


def gaussian_censored_ll(lt, d, mu, sd):
    z = (np.asarray(lt) - mu) / sd
    return np.where(np.asarray(d) > 0, stats.norm.logpdf(z) - np.log(sd), stats.norm.logsf(z))
