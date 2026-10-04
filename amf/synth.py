"""Semi-synthetic multi-study AM fatigue benchmark ("SynAM") with known ground truth.

Each simulated study mimics one S-N dataset in a literature database: an alloy, a process route,
post-processing, a stress ratio, a test plan with 3-6 stress levels, a Type I run-out limit, and
lab-to-lab variation. The generator is deliberately NOT identical to the fitted model:
  * the within-study fatigue limit follows a Gumbel (largest-defect) law, not log-normal;
  * the S-N slope m varies study-to-study but is not a random effect in the model;
  * stresses and lives carry digitisation noise; stresses are rounded to 5 MPa;
  * the true HIP benefit depends on surface state (HIP x machined interaction).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

ALLOYS = ["Ti64", "AlSi10Mg", "316L", "IN718"]
ALLOY_P = [0.40, 0.25, 0.20, 0.15]
UTS_MEAN = {"Ti64": 1050.0, "AlSi10Mg": 420.0, "316L": 620.0, "IN718": 1300.0}
UTS_SD = {"Ti64": 60.0, "AlSi10Mg": 35.0, "316L": 45.0, "IN718": 80.0}
BASE_MU = {"Ti64": np.log(0.24), "AlSi10Mg": np.log(0.20), "316L": np.log(0.30), "IN718": np.log(0.25)}
INC_MACH = {"Ti64": 0.45, "AlSi10Mg": 0.35, "316L": 0.25, "IN718": 0.40}
INC_HIP_M = {"Ti64": 0.35, "AlSi10Mg": 0.25, "316L": 0.15, "IN718": 0.30}
INC_HIP_AB = 0.05
B0_OFF = {"Ti64": 0.0, "AlSi10Mg": -0.3, "316L": 0.4, "IN718": 0.2}
RUNOUT = ([1e6, 2e6, 5e6, 1e7, 2e7, 1e8], [0.10, 0.15, 0.15, 0.45, 0.10, 0.05])
W_TRUE = 0.6
RE_SD = (0.35, 0.10)
RE_RHO = 0.4
GUMBEL_SD = np.pi / np.sqrt(6)
EULER = 0.5772156649


def true_mu(alloy, mach, hip, vert, ht, ebm, uts):
    mu = BASE_MU[alloy] + INC_MACH[alloy] * mach
    mu += hip * (INC_HIP_M[alloy] * mach + INC_HIP_AB * (1 - mach))
    mu += -0.06 * vert + 0.03 * ht - 0.05 * ebm - 0.15 * (uts / UTS_MEAN[alloy] - 1.0)
    return mu


def feature_vector(st):
    a = st["alloy"]
    oh = [1.0 if a == x else 0.0 for x in ALLOYS]
    uts_rep = st["uts_reported"]
    miss = 1.0 if uts_rep is None else 0.0
    utsn = 0.0 if uts_rep is None else uts_rep / UTS_MEAN[a] - 1.0
    R = st["R_study"]
    return np.array(oh + [st["hip"], st["mach"], st["vert"], st["ht"], st["ebm"], R, utsn, miss], float)


FEATURE_NAMES = ALLOYS + ["HIP", "machined", "vertical", "heat_treated", "EBM", "R", "UTS_norm", "UTS_missing"]


def sample_gamma_log(mu, tau, size, rng):
    """ln gamma = mu - tau * standardized Gumbel(max): fatigue limit set by the largest defect."""
    g = rng.gumbel(size=size)
    return mu - tau * (g - EULER) / GUMBEL_SD


def generate(n_studies, seed=0, anomaly_frac=0.0, anomaly_kind="surface", misspec=False):
    """misspec=True adds four departures the model does not encode (stress test): duplex initiation
    (30 % of as-built specimens fail from a surface mode at 1/5 of the life), Student-t(4) life scatter,
    run-out limits that depend on the surface state (as-built tests stopped earlier), and a 2.7x larger
    study-to-study spread of the S-N slope. With misspec=False the random stream is unchanged."""
    rng = np.random.default_rng(seed)
    cov = np.array([[RE_SD[0] ** 2, RE_RHO * RE_SD[0] * RE_SD[1]], [RE_RHO * RE_SD[0] * RE_SD[1], RE_SD[1] ** 2]])
    studies = []
    for s in range(n_studies):
        alloy = rng.choice(ALLOYS, p=ALLOY_P)
        uts = rng.normal(UTS_MEAN[alloy], UTS_SD[alloy])
        mach = int(rng.random() < 0.55); hip = int(rng.random() < 0.30); vert = int(rng.random() < 0.5)
        ht = int(rng.random() < 0.6) if not hip else 1
        ebm = int(rng.random() < 0.15) if alloy == "Ti64" else 0
        R = -1.0 if rng.random() < 0.6 else 0.1
        b = rng.multivariate_normal([0, 0], cov)
        anomalous = rng.random() < anomaly_frac
        mach_true = mach
        if anomalous and anomaly_kind == "surface":
            mach_true = 0  # reported machined/HIP data but the true surface is as-built-like
        mu = true_mu(alloy, mach_true, hip, vert, ht, ebm, uts) + b[1]
        if anomalous and anomaly_kind == "surface" and mach == 0:
            mu -= 0.35  # unreported large defect population (e.g. contaminated powder lot)
        m = 4.5 + 1.2 * mach_true + 0.5 * hip + rng.normal(0, 0.8 if misspec else 0.3)
        beta0 = 5.0 + 0.5 * mach_true + B0_OFF[alloy] + b[0]
        s_life = max(0.2, 0.55 - 0.12 * mach_true - 0.12 * hip + rng.normal(0, 0.03))
        tau = max(0.03, 0.10 - 0.03 * hip - 0.02 * mach_true + rng.normal(0, 0.01))
        n = int(rng.integers(8, 25))
        L = int(rng.integers(3, 7))
        gam_med = np.exp(mu)
        u_lo, u_hi = gam_med * 1.15, min(gam_med * 2.8, 0.95)
        levels = np.exp(np.linspace(np.log(u_lo), np.log(u_hi), L))
        u = rng.choice(levels, size=n)
        ratio = (2.0 / (1.0 - R)) ** (1.0 - W_TRUE)
        sa_true = u * uts / ratio
        lg = sample_gamma_log(mu, tau, n, rng)
        gam = np.exp(lg)
        eps = rng.normal(size=n) if not misspec else rng.standard_t(4, size=n) / np.sqrt(2.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lnN = np.where(u > gam, beta0 - m * np.log(np.maximum(u - gam, 1e-300)) + s_life * eps, np.inf)
        duplex_p = 0.3 if (misspec and mach_true == 0) else 0.0
        if duplex_p > 0:
            lnN = lnN - np.log(5.0) * (rng.random(n) < duplex_p)
        if misspec:
            C = (rng.choice([1e6, 2e6, 5e6, 1e7], p=[0.3, 0.3, 0.2, 0.2]) if mach == 0 else
                 rng.choice([5e6, 1e7, 2e7, 1e8], p=[0.1, 0.5, 0.25, 0.15]))
        else:
            C = rng.choice(RUNOUT[0], p=RUNOUT[1])
        T = np.exp(np.minimum(lnN, 700))
        T[~np.isfinite(lnN)] = np.inf
        d = (T <= C).astype(float)
        N_obs = np.where(d > 0, T, C)
        # digitisation noise: lives +-3 % (failures only), stresses rounded to 5 MPa with +-1 % error
        N_obs = np.where(d > 0, N_obs * np.exp(rng.normal(0, 0.03, n)), N_obs)
        sa_rep = np.round(sa_true * np.exp(rng.normal(0, 0.01, n)) / 5.0) * 5.0
        sa_rep = np.maximum(sa_rep, 5.0)
        uts_rep = None if rng.random() < 0.3 else float(np.round(uts, -1))
        ref = uts_rep if uts_rep is not None else UTS_MEAN[alloy]
        st = dict(alloy=alloy, hip=hip, mach=mach, vert=vert, ht=ht, ebm=ebm, R_study=R, uts_reported=uts_rep,
                  sa=sa_rep, R=np.full(n, R), N=N_obs, d=d, ref=ref, C=C, T_true=T,
                  truth=dict(beta0=beta0, m=m, s=s_life, mu=mu, tau=tau, uts=uts, b=b, anomalous=anomalous,
                             sa_true=sa_true, mach_true=mach_true, t_df=4 if misspec else None,
                             duplex_p=duplex_p))
        st["z"] = feature_vector(st)
        st["id"] = s
        studies.append(st)
    return studies


# ----------------------------------------------------------------------------- ground-truth functionals
_QG = (np.arange(4000) + 0.5) / 4000


def true_logsf(st, sa, lnN, nq=4000):
    """True log P(N > exp(lnN)) at amplitude sa (arrays broadcast) for study st (nq quadrature nodes)."""
    t = st["truth"]
    ratio = (2.0 / (1.0 - st["R_study"])) ** (1.0 - W_TRUE)
    u = np.asarray(sa, float) * ratio / t["uts"]
    # Gumbel(max) quantiles for g -> ln gamma = mu - tau*(g-EULER)/GUMBEL_SD (decreasing in g)
    qg = _QG if nq == 4000 else (np.arange(nq) + 0.5) / nq
    g = -np.log(-np.log(qg))
    lg = t["mu"] - t["tau"] * (g - EULER) / GUMBEL_SD  # (4000,)
    gam = np.exp(lg)
    u = np.asarray(u)[..., None]
    lnN = np.asarray(lnN, float)[..., None]
    can = u > gam
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (lnN - t["beta0"] + t["m"] * np.log(np.where(can, u - gam, 1.0))) / t["s"]
    df_ = t.get("t_df")

    def sf(zz):
        return stats.t.sf(zz * np.sqrt(2.0), df_) if df_ else stats.norm.sf(zz)

    surv = np.where(can, sf(z), 1.0).mean(-1)
    p = t.get("duplex_p", 0.0)
    if p:  # T = T_main / 5 with probability p  ->  P(T > t) = P(T_main > 5 t)
        z5 = z + np.log(5.0) / t["s"]
        surv = (1 - p) * surv + p * np.where(can, sf(z5), 1.0).mean(-1)
    return np.log(np.clip(surv, 1e-300, 1.0))


def true_allowable(st, Nd, rel, lo=1.0, hi=None, iters=60):
    """Amplitude sigma* with true P(N > Nd) = rel (oracle allowable)."""
    hi = hi or st["truth"]["uts"]
    a, b = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if np.exp(true_logsf(st, np.exp(mid), np.log(Nd))) > rel:
            a = mid
        else:
            b = mid
    return float(np.exp(0.5 * (a + b)))


def true_quantile_lnN(st, sa, p, lo=0.0, hi=60.0, iters=60):
    """True p-quantile of ln N at amplitude sa (inf if P(fail ever) < p)."""
    sa = np.atleast_1d(np.asarray(sa, float))
    a = np.full(sa.shape, lo); b = np.full(sa.shape, hi)
    target = np.log1p(-p)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        up = true_logsf(st, sa, mid) > target
        a = np.where(up, mid, a); b = np.where(up, b, mid)
    out = 0.5 * (a + b)
    out[true_logsf(st, sa, np.full(sa.shape, hi)) > target] = np.inf
    return out
