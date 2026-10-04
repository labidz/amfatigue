"""Study-level conformal lower predictive bounds (LPBs), PAC (tolerance-type) bounds, and forensic p-values.

Score for a calibration specimen with covariates X and life T (possibly censored at C >= c0):
    E = ln(min(Q(X), c0)) - ln(min(T, c0))
where Q is any model's alpha-quantile of life. T ^ c0 is observed whenever C >= c0, so censoring
does not bias the score (Candes, Lei & Ren 2023, Type I censoring argument). Exchangeability holds at
the study level, so one specimen is drawn per calibration study (Dunn, Wasserman & Ramdas 2023).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def scores(lnQ, lt, d, lc0):
    """lnQ, lt: ln quantile and ln observed life; d: 1 failure, 0 run-out. Requires C >= c0 for run-outs."""
    lnQ = np.minimum(np.asarray(lnQ, float), lc0)
    lT = np.where(np.asarray(d) > 0, np.minimum(lt, lc0), lc0)
    return lnQ - lT


def observable(lt, d, lc0):
    """Is T ^ c0 observed? (failure, or run-out at/after c0)."""
    return (np.asarray(d) > 0) | (np.asarray(lt) >= lc0 - 1e-9)


def split_eta(E, alpha):
    E = np.sort(np.asarray(E))
    m = len(E)
    k = int(np.ceil((1 - alpha) * (m + 1)))
    return np.inf if k > m else E[k - 1]


def pac_eta(E, alpha, delta):
    """Order statistic giving coverage >= 1-alpha with probability >= 1-delta over the calibration draw.

    Coverage of the k-th smallest score is Beta(k, m-k+1) distributed for continuous scores
    (Wilks 1941 / Vovk 2012), so pick the smallest k with P(Beta(k, m+1-k) < 1-alpha) <= delta.
    """
    E = np.sort(np.asarray(E))
    m = len(E)
    for k in range(1, m + 1):
        if stats.beta.cdf(1 - alpha, k, m + 1 - k) <= delta:
            return E[k - 1]
    return np.inf


def lpb(lnQ, eta, lc0):
    """Conformal lower predictive bound on ln T."""
    return np.minimum(np.minimum(np.asarray(lnQ, float), lc0) - eta, lc0)


def one_per_study(study_ids, rng):
    """Index of one randomly chosen eligible specimen per study."""
    study_ids = np.asarray(study_ids)
    out = []
    for s in np.unique(study_ids):
        idx = np.flatnonzero(study_ids == s)
        out.append(rng.choice(idx))
    return np.array(out, int)


def pvalues(E_cal, E_test):
    """Conformal p-values for 'life is shorter than expected' (large score)."""
    E_cal = np.sort(np.asarray(E_cal))
    m = len(E_cal)
    ge = m - np.searchsorted(E_cal, np.asarray(E_test), side="left")
    return (1 + ge) / (m + 1)


def simes(p):
    p = np.sort(np.asarray(p))
    n = len(p)
    return float(np.min(n * p / np.arange(1, n + 1))) if n else 1.0


def bh(pvals, q):
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    ok = p[order] <= thresh
    rej = np.zeros(n, bool)
    if ok.any():
        kmax = np.max(np.flatnonzero(ok))
        rej[order[: kmax + 1]] = True
    return rej


def normal_tolerance_k(n, p=0.90, conf=0.95):
    """One-sided normal tolerance factor (MMPDS-style B-basis when p=0.90, conf=0.95)."""
    zp = stats.norm.ppf(p)
    return stats.nct.ppf(conf, df=n - 1, nc=zp * np.sqrt(n)) / np.sqrt(n)
