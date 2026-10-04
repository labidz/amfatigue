"""Stage 2 (synthetic): turn saved predictions into the E1-E8 tables and figures.

Usage: python experiments/analyze_synthetic.py --config fast
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import conformal as cf  # noqa: E402
from amf import metrics, synth  # noqa: E402
from experiments.config import ALPHA, C0, C0_FORENSIC, CONFIGS, DELTA  # noqa: E402

LC0, LC0F, LN10 = np.log(C0), np.log(C0_FORENSIC), np.log(10)
ROOT = os.path.join(os.path.dirname(__file__), "..")
OURS = "H-NRFL (ours)"


# ----------------------------------------------------------------------------- helpers
def study_mean(vals, sid):
    """Average within study, then across studies (the population the guarantee refers to)."""
    df = pd.DataFrame(dict(v=vals.astype(float), s=sid))
    return float(df.groupby("s").v.mean().mean())


def per_study(vals, sid, min_n=5):
    df = pd.DataFrame(dict(v=vals.astype(float), s=sid))
    g = df.groupby("s").v
    m = g.mean()[g.size() >= min_n]
    return m.values


def covered(lT, L):
    return np.asarray(lT) >= np.asarray(L) - 1e-12


def draw_eta(pr_cal, base_cal, shot, rng, lc0=LC0):
    ok = (shot == 0) & (base_cal["C"] >= np.exp(lc0) * (1 - 1e-9))
    idx = np.flatnonzero(ok)
    pick = idx[cf.one_per_study(base_cal["sid"][idx], rng)]
    return cf.scores(pr_cal["lnQ"][pick], base_cal["lt"][pick], base_cal["d"][pick], lc0)


def mean_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n/a"
    return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


LT_GRID = np.linspace(np.log(1e2), LC0, 90)


def true_sf_table(test_studies, nq=4000):
    """True log S(ln t) on LT_GRID for every test specimen (exact, from the generator)."""
    rows = []
    for st in test_studies:
        sa = np.asarray(st["truth"]["sa_true"])[:, None]
        rows.append(synth.true_logsf(st, sa, LT_GRID[None, :], nq))
    return np.vstack(rows)  # (n_spec, G)


def pac_fresh_pool(run, config, seed, n_pool=1000, draws=None):
    """PAC check for H-NRFL on a large FRESH pool of new studies (removes test-set sampling noise).

    Rebuilds the saved model, predicts the alpha-quantile on n_pool new studies, and evaluates the true
    super-population coverage for every calibration draw's offset."""
    import jax.numpy as jnp
    from amf import evaluate as ev
    from amf import rfl
    mp = run["model_params"][OURS]
    cfgd = {k: v for k, v in mp["cfg"].items()}
    M = rfl.FittedRFL({k: jnp.asarray(v) for k, v in mp["P"].items()}, rfl.RFLConfig(**cfgd), mp["norm_stats"], [], 0)
    pool = synth.generate(n_pool, seed=7000 + seed)
    pr = ev.RFLPredictor(M, chunk=50).predict(pool, None, ALPHA)
    tab = true_sf_table(pool, nq=1000)
    sid = np.concatenate([np.full(len(st["N"]), i) for i, st in enumerate(pool)])
    rng = np.random.default_rng(seed + 23)
    prc, bc = run["preds"][(OURS, 0, "cal")], run["base"]["cal"]
    sh0 = run["shots_flat"]["cal"][0]
    res = {"Split conformal (1-α)": [], "PAC conformal (B-basis 90/95)": []}
    for _ in range(draws or 200):
        E = draw_eta(prc, bc, sh0, rng)
        for lab, eta in (("Split conformal (1-α)", cf.split_eta(E, ALPHA)),
                         ("PAC conformal (B-basis 90/95)", cf.pac_eta(E, ALPHA, DELTA))):
            res[lab].append(true_cov(tab, cf.lpb(pr["lnQ"], eta, LC0), sid))
    raw = true_cov(tab, np.minimum(pr["lnQ"], LC0), sid)
    # conditional coverage on the fresh pool, at the median split offset
    eta_med = float(np.median([cf.split_eta(draw_eta(prc, bc, sh0, rng), ALPHA) for _ in range(50)]))
    L = cf.lpb(pr["lnQ"], eta_med, LC0)
    ueq = np.concatenate([np.asarray(st["sa"]) * (2 / (1 - st["R_study"])) ** 0.4 / st["ref"] for st in pool])
    qb = pd.qcut(ueq, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    alloy = np.concatenate([np.full(len(st["N"]), st["alloy"]) for st in pool])
    surf = np.concatenate([np.full(len(st["N"]), "machined" if st["mach"] else "as-built") for st in pool])
    cond = [dict(group="Walker-equivalent stress " + str(b), coverage=true_cov(tab, L, sid, np.asarray(qb == b)))
            for b in qb.categories]
    cond += [dict(group="alloy " + a, coverage=true_cov(tab, L, sid, alloy == a)) for a in synth.ALLOYS]
    cond += [dict(group="surface " + v, coverage=true_cov(tab, L, sid, surf == v)) for v in ("as-built", "machined")]
    pac_fresh_pool.cond = pd.DataFrame(cond)
    rows = [dict(model=OURS + ", 1,000 fresh test studies", method="Model quantile, uncalibrated", mean_coverage=raw,
                 p_cov_ge_90=float(raw >= 0.9), mean_lpb_log10=np.nan, n_draws=1)]
    for lab, v in res.items():
        v = np.array(v)
        rows.append(dict(model=OURS + ", 1,000 fresh test studies", method=lab, mean_coverage=float(v.mean()),
                         p_cov_ge_90=float((v >= 0.9).mean()), mean_lpb_log10=np.nan, n_draws=len(v)))
    return pd.DataFrame(rows)


def true_cov(logS_tab, L, sid, sel=None):
    """Expected coverage P(T >= L) per specimen from the true survival function, averaged per study."""
    L = np.clip(np.asarray(L, float), LT_GRID[0], LT_GRID[-1])
    pos = (L - LT_GRID[0]) / (LT_GRID[1] - LT_GRID[0])
    i0 = np.clip(np.floor(pos).astype(int), 0, len(LT_GRID) - 2)
    w = pos - i0
    r = np.arange(len(L))
    lS = (1 - w) * logS_tab[r, i0] + w * logS_tab[r, i0 + 1]
    v = np.exp(lS)
    if sel is not None:
        return study_mean(v[sel], sid[sel])
    return study_mean(v, sid)


def per_study_true(logS_tab, L, sid, min_n=5):
    L = np.clip(np.asarray(L, float), LT_GRID[0], LT_GRID[-1])
    pos = (L - LT_GRID[0]) / (LT_GRID[1] - LT_GRID[0])
    i0 = np.clip(np.floor(pos).astype(int), 0, len(LT_GRID) - 2); w = pos - i0
    r = np.arange(len(L))
    v = np.exp((1 - w) * logS_tab[r, i0] + w * logS_tab[r, i0 + 1])
    return per_study(v, sid, min_n)


def mondrian_eta(pr_cal, base_cal, shot, rng, edges, alpha=ALPHA, lc0=LC0):
    """Split-conformal offset per bin of predicted median life (one specimen per study within each bin)."""
    ok = (shot == 0) & (base_cal["C"] >= np.exp(lc0) * (1 - 1e-9))
    b = np.digitize(pr_cal["lnmed"], edges)
    etas = {}
    for k in range(len(edges) + 1):
        idx = np.flatnonzero(ok & (b == k))
        if len(idx) < 10:
            etas[k] = np.nan
            continue
        pick = idx[cf.one_per_study(base_cal["sid"][idx], rng)]
        E = cf.scores(pr_cal["lnQ"][pick], base_cal["lt"][pick], base_cal["d"][pick], lc0)
        etas[k] = cf.split_eta(E, alpha)
    return etas


def local_eta(pr_cal, base_cal, shot, rng, center, width=1.5, alpha=ALPHA, delta=DELTA, lc0=LC0):
    """Offsets calibrated on specimens whose predicted median life is within +-width (ln units) of the design life."""
    ok = (shot == 0) & (base_cal["C"] >= np.exp(lc0) * (1 - 1e-9)) & (np.abs(pr_cal["lnmed"] - center) <= width)
    idx = np.flatnonzero(ok)
    pick = idx[cf.one_per_study(base_cal["sid"][idx], rng)]
    E = cf.scores(pr_cal["lnQ"][pick], base_cal["lt"][pick], base_cal["d"][pick], lc0)
    return cf.split_eta(E, alpha), cf.pac_eta(E, alpha, delta), len(E)


def interp_grid(grid_vals, eta_grid, eta):
    """Interpolate per-study allowable (or reliability) stored on the eta grid at a given eta."""
    if not np.isfinite(eta):
        return np.zeros(grid_vals.shape[0])
    eta = float(np.clip(eta, eta_grid[0], eta_grid[-1]))
    j = int(np.clip(np.searchsorted(eta_grid, eta) - 1, 0, len(eta_grid) - 2))
    w = (eta - eta_grid[j]) / (eta_grid[j + 1] - eta_grid[j])
    return (1 - w) * grid_vals[:, j] + w * grid_vals[:, j + 1]


def regen_test(config, seed):
    C = CONFIGS[config]
    allst = synth.generate(C["n_train"] + C["n_cal"] + C["n_test"], seed=1000 + seed)
    return allst[C["n_train"] + C["n_cal"]:]


# ----------------------------------------------------------------------------- E2/E3 zero-shot
def zero_shot(run, tab):
    rows = []
    te = run["base"]["test"]
    for (name, k, split), pr in run["preds"].items():
        if split != "test" or k != 0 or pr is None:
            continue
        lp = pr["lp"]
        nll = -np.mean(lp) if np.all(np.isfinite(lp)) else np.nan
        c = metrics.cindex(te["lt"], te["d"], pr["lnmed"])
        chi2, pv, maxdev = metrics.dcal(pr["F"], te["d"])
        eta = run["eta"][(name, 0)]["split"]
        L = cf.lpb(pr["lnQ"], eta, LC0)
        rows.append(dict(model=name, nll=nll, cindex=c, dcal_maxdev=maxdev, dcal_p=pv,
                         raw_q10_cov=true_cov(tab, np.minimum(pr["lnQ"], LC0), te["sid"]),
                         conf_cov=true_cov(tab, L, te["sid"]), mean_lpb_log10=float(np.mean(L) / LN10)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E1 fatigue strength bias
def fs_bias(run):
    rows = []
    tr = run["truth"]
    for (name, k), v in run["fs"].items():
        for q in ("fs50", "fs10"):
            r = np.log(v[q] / tr[q])
            rows.append(dict(model=name, k=k, quantity=q, bias_pct=float((np.exp(np.mean(r)) - 1) * 100),
                             rmse_log=float(np.sqrt(np.mean(r ** 2)))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E4 conformal under study shift
def conformal_shift(run, naive, test_studies, reps, seed, tab):
    rows, cond = [], []
    te = run["base"]["test"]
    rng = np.random.default_rng(seed + 7)
    sh0 = run["shots_flat"]["cal"][0]
    # reported normalised stress and alloy per test specimen, for conditional coverage
    u = np.concatenate([np.asarray(st["sa"]) / st["ref"] for st in test_studies])
    alloy = np.concatenate([np.full(len(st["N"]), st["alloy"]) for st in test_studies])
    qbin = pd.qcut(u, 4, labels=["Q1 (lowest stress)", "Q2", "Q3", "Q4 (highest stress)"])
    for name in (OURS, "GBM (asfail)", [m for m in run["eta"] if str(m[0]).startswith("Deep ensemble")][0][0]):
        prc, prt = run["preds"][(name, 0, "cal")], run["preds"][(name, 0, "test")]
        bc = run["base"]["cal"]
        variants = {}
        if naive is not None and name in naive:
            eta = cf.split_eta(naive[name]["E_naive"], ALPHA)
            Ln = cf.lpb(naive[name]["lnQ_test"], eta, LC0)
            rows.append(dict(model=name, calibration="Naive point split (seen studies)",
                             coverage=true_cov(tab, Ln, te["sid"]), coverage_sd=0.0,
                             p10_study_coverage=float(np.percentile(per_study_true(tab, Ln, te["sid"]), 10)),
                             mean_lpb_log10=float(np.mean(Ln) / LN10)))
        ok = bc["C"] >= C0 * (1 - 1e-9)
        Eall = cf.scores(prc["lnQ"][ok], bc["lt"][ok], bc["d"][ok], LC0)
        variants["Study split, specimens pooled"] = [cf.split_eta(Eall, ALPHA)]
        draws = [draw_eta(prc, bc, sh0, rng) for _ in range(reps)]
        variants["Study split, one per study (exact)"] = [cf.split_eta(E, ALPHA) for E in draws]
        variants["Repeated one-per-study (median)"] = [float(np.median([cf.split_eta(E, ALPHA) for E in draws]))]
        for vname, etas in variants.items():
            covs, lows, lpbs = [], [], []
            for eta in etas:
                L = cf.lpb(prt["lnQ"], eta, LC0)
                covs.append(true_cov(tab, L, te["sid"]))
                lows.append(np.percentile(per_study_true(tab, L, te["sid"]), 10))
                lpbs.append(np.mean(L) / LN10)
            rows.append(dict(model=name, calibration=vname, coverage=float(np.mean(covs)),
                             coverage_sd=float(np.std(covs)), p10_study_coverage=float(np.mean(lows)),
                             mean_lpb_log10=float(np.mean(lpbs))))
            if vname == "Study split, one per study (exact)" and name == OURS:
                eta_med = float(np.median(etas))
                L = cf.lpb(prt["lnQ"], eta_med, LC0)
                # Mondrian: offsets per quartile of predicted median life (bins fixed on calibration data)
                okc = bc["C"] >= C0 * (1 - 1e-9)
                edges = np.quantile(prc["lnmed"][okc & np.isfinite(prc["lnmed"])], [0.25, 0.5, 0.75])
                Lm_list = []
                for _ in range(reps):
                    et = mondrian_eta(prc, bc, sh0, rng, edges)
                    bt = np.digitize(prt["lnmed"], edges)
                    eta_vec = np.array([et.get(b, np.nan) for b in bt])
                    eta_vec = np.where(np.isfinite(eta_vec), eta_vec, eta_med)
                    Lm_list.append(cf.lpb(prt["lnQ"], eta_vec, LC0))
                for b in qbin.categories:
                    sel = np.asarray(qbin == b)
                    cond.append(dict(model=name, group="stress " + b, coverage=true_cov(tab, L, te["sid"], sel),
                                     coverage_mondrian=float(np.mean([true_cov(tab, Lm, te["sid"], sel)
                                                                      for Lm in Lm_list]))))
                for a in synth.ALLOYS:
                    sel = alloy == a
                    cond.append(dict(model=name, group="alloy " + a, coverage=true_cov(tab, L, te["sid"], sel),
                                     coverage_mondrian=float(np.mean([true_cov(tab, Lm, te["sid"], sel)
                                                                      for Lm in Lm_list]))))
                covm = [true_cov(tab, Lm, te["sid"]) for Lm in Lm_list]
                rows.append(dict(model=name, calibration="Mondrian by predicted life (one per study per bin)",
                                 coverage=float(np.mean(covm)), coverage_sd=float(np.std(covm)),
                                 p10_study_coverage=float(np.mean([np.percentile(per_study_true(tab, Lm, te["sid"]), 10)
                                                                   for Lm in Lm_list])),
                                 mean_lpb_log10=float(np.mean([np.mean(Lm) for Lm in Lm_list]) / LN10)))
    return pd.DataFrame(rows), pd.DataFrame(cond)


# ----------------------------------------------------------------------------- E6 PAC / B-basis
def pac(run, draws, seed, tab):
    rows = []
    te = run["base"]["test"]; bc = run["base"]["cal"]
    rng = np.random.default_rng(seed + 11)
    sh0 = run["shots_flat"]["cal"][0]
    names = [n for (n, k, s) in run["preds"] if k == 0 and s == "cal"]
    for name in names:
        prc, prt = run["preds"][(name, 0, "cal")], run["preds"][(name, 0, "test")]
        res = {"Model quantile, uncalibrated": [], "Split conformal (1-α)": [], "PAC conformal (B-basis 90/95)": []}
        for _ in range(draws):
            E = draw_eta(prc, bc, sh0, rng)
            for lab, eta in (("Split conformal (1-α)", cf.split_eta(E, ALPHA)),
                             ("PAC conformal (B-basis 90/95)", cf.pac_eta(E, ALPHA, DELTA))):
                L = cf.lpb(prt["lnQ"], eta, LC0)
                res[lab].append((true_cov(tab, L, te["sid"]), np.mean(L) / LN10))
        L = np.minimum(prt["lnQ"], LC0)
        res["Model quantile, uncalibrated"].append((true_cov(tab, L, te["sid"]), np.mean(L) / LN10))
        for lab, v in res.items():
            v = np.array(v)
            rows.append(dict(model=name, method=lab, mean_coverage=float(v[:, 0].mean()),
                             p_cov_ge_90=float((v[:, 0] >= 0.9).mean()), mean_lpb_log10=float(v[:, 1].mean()),
                             n_draws=len(v)))
        info = run["info"].get(name, {})
        if "scale" in info and np.isfinite(info.get("scale", np.nan)):
            kf = cf.normal_tolerance_k(info["n_fail"], 1 - ALPHA, 1 - DELTA)
            L = np.minimum(prt["lnmed"] - kf * info["scale"], LC0)
            cvn = true_cov(tab, L, te["sid"])
            rows.append(dict(model=name, method="Normal tolerance bound, pooled i.i.d. (MMPDS-style)",
                             mean_coverage=cvn, p_cov_ge_90=float(cvn >= 0.9),
                             mean_lpb_log10=float(np.mean(L) / LN10), n_draws=1))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E5 few-shot qualification
def _allow_stats(a, rel, oracle):
    a = np.asarray(a, float); rel = np.asarray(rel, float)
    pos = a > 0
    return dict(frac_zero=float(np.mean(~pos)),
                mean_rel=float(np.mean(rel[pos])) if pos.any() else np.nan,
                frac_rel_ge_90=float(np.mean(rel[pos] >= 0.9)) if pos.any() else np.nan,
                median_eff=float(np.median(np.where(pos, a, 0.0) / oracle)))


def few_shot(run, ks, tab, seed, draws=100):
    """E5. Allowables are averaged over `draws` calibration draws (offsets interpolated on the stored eta grid);
    a zero allowable (no stress can be certified) is reported as frac_zero, never scored as 'safe'."""
    rows = []
    rng = np.random.default_rng(seed + 17)
    from experiments.config import ND
    te = run["base"]["test"]; tr = run["truth"]
    for name in (OURS, "w/o covariates (RE only)"):
        for k in ks:
            if (name, k, "test") not in run["preds"]:
                continue
            pr = run["preds"][(name, k, "test")]; prc = run["preds"][(name, k, "cal")]
            ev_ = run["shots_flat"]["test"][k] == 0
            shc = run["shots_flat"]["cal"][k]
            etas = [cf.split_eta(draw_eta(prc, run["base"]["cal"], shc, rng), ALPHA) for _ in range(draws)]
            covs = [true_cov(tab, cf.lpb(pr["lnQ"], e, LC0), te["sid"], ev_) for e in etas]
            row = dict(model=name, k=k, nll=float(-np.mean(pr["lp"][ev_])), lpb_cov=float(np.mean(covs)),
                       lpb_cov_sd=float(np.std(covs)))
            a0 = run["allow"][(name, k, "model")]; r0 = run["allow_rel"][(name, k, "model")]
            for kk, v in _allow_stats(a0, r0, tr["allowB"]).items():
                row[f"model_{kk}"] = v
            if (name, k) in run.get("allow_grid", {}):
                G, Gr, eg = run["allow_grid"][(name, k)], run["allow_grid_rel"][(name, k)], run["eta_grid"]
                acc = {}
                for _ in range(draws):
                    E = draw_eta(prc, run["base"]["cal"], shc, rng)
                    e_ls, e_lp, _m = local_eta(prc, run["base"]["cal"], shc, rng, np.log(ND))
                    for lab, eta in (("split", cf.split_eta(E, ALPHA)), ("pac", cf.pac_eta(E, ALPHA, DELTA)),
                                     ("local_split", e_ls), ("local_pac", e_lp)):
                        if np.isfinite(eta) and eta <= eg[-1]:
                            a = interp_grid(G, eg, eta); rel = interp_grid(Gr, eg, eta)
                        else:
                            a = np.zeros(G.shape[0]); rel = np.ones(G.shape[0])
                        for kk, v in _allow_stats(a, rel, tr["allowB"]).items():
                            acc.setdefault(f"{lab}_{kk}", []).append(v)
                for kk, v in acc.items():
                    row[kk] = float(np.nanmean(v))
            rows.append(row)
    for k in ks:
        key = ("Per-study E739 + tolerance bound", k, "classical")
        if key not in run["allow"]:
            continue
        a = run["allow"][key]; rel = run["allow_rel"][key]
        comp = np.isfinite(a)
        st_ = _allow_stats(np.where(comp, a, 0.0)[comp], rel[comp], tr["allowB"][comp]) if comp.any() else {}
        rows.append(dict(model="Per-study ASTM E739 + tolerance bound", k=k, computable=float(comp.mean()),
                         **{f"classical_{kk}": v for kk, v in st_.items()}))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E7 forensic screening
def _study_scores(pr, b, n_st):
    F = np.clip(pr["F"], 1e-12, 1 - 1e-12)
    z = stats.norm.ppf(F)
    r = np.where(b["d"] > 0, -z, -stats.norm.pdf(z) / (1 - F))
    return np.array([r[b["sid"] == s_].mean() for s_ in range(n_st)])


def forensic(run, seed, q=0.10, R=200, n_sub=150):
    """E7. Study-level conformal screening. FDR and power are averaged over R resamplings of the calibration
    studies and of the screened pool (marginal FDR, the quantity BH controls), and separately over resamplings
    of the screened pool with the full calibration set fixed (calibration-conditional FDR)."""
    rows = []
    ba, bc = run["base"]["anom"], run["base"]["cal"]
    anomalous = run["truth"]["anomalous"]
    rng = np.random.default_rng(seed + 13)
    n_cal = int(bc["sid"].max()) + 1; n_a = len(anomalous)
    for name in (OURS, "GBM (asfail)", "ASTM E739 pooled (drop)"):
        if (name, 0, "anom") not in run["preds"]:
            continue
        prc, pra = run["preds"][(name, 0, "cal")], run["preds"][(name, 0, "anom")]
        Sc, Sa = _study_scores(prc, bc, n_cal), _study_scores(pra, ba, n_a)
        p_model = np.where(ba["d"] > 0, pra["F"], 1.0)
        ps_model = np.array([cf.simes(p_model[ba["sid"] == s_]) for s_ in range(n_a)])
        res = {k: [] for k in ("marg", "cond", "model")}
        for _ in range(R):
            ti = rng.choice(n_a, n_sub, replace=False)
            ci = rng.choice(n_cal, n_sub, replace=False)
            for lab, p in (("marg", cf.pvalues(Sc[ci], Sa[ti])), ("cond", cf.pvalues(Sc, Sa[ti])),
                           ("model", ps_model[ti])):
                rej = cf.bh(p, q); an = anomalous[ti]
                res[lab].append((float((rej & ~an).sum() / max(rej.sum(), 1)), float((rej & an).sum() / max(an.sum(), 1))))
        for lab, test in (("marg", "conformal, study-level (marginal FDR)"),
                          ("cond", "conformal, study-level (conditional on calibration set)"),
                          ("model", "model-based specimen p-values + Simes, no calibration")):
            v = np.array(res[lab])
            rows.append(dict(model=name, test=test, fdr=float(v[:, 0].mean()), fdp_sd=float(v[:, 0].std()),
                             power=float(v[:, 1].mean()), resamples=R))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E8 root cause
def root_cause(run):
    df = pd.DataFrame(run["rootcause"])
    rc = run["info"][OURS]["re_cov"]
    extra = pd.DataFrame([dict(alloy="all", surface="RE sd(beta0)", est=float(np.sqrt(rc[0, 0])), true=synth.RE_SD[0]),
                          dict(alloy="all", surface="RE sd(mu)", est=float(np.sqrt(rc[1, 1])), true=synth.RE_SD[1]),
                          dict(alloy="all", surface="Walker exponent w", est=run["info"][OURS]["walker"],
                               true=synth.W_TRUE)])
    return pd.concat([df, extra], ignore_index=True)


def _cf_study(st, hip, mach, ht):
    """Counterfactual copy of a synthetic study: reported AND true process changed, everything else kept."""
    t = dict(st["truth"]); a = st["alloy"]
    mt, h0 = t["mach_true"], st["hip"]
    m_noise = t["m"] - (4.5 + 1.2 * mt + 0.5 * h0)
    s_noise = t["s"] - (0.55 - 0.12 * mt - 0.12 * h0)
    tau_noise = t["tau"] - (0.10 - 0.03 * h0 - 0.02 * mt)
    t["mu"] = synth.true_mu(a, mach, hip, st["vert"], ht, st["ebm"], t["uts"]) + t["b"][1]
    t["m"] = 4.5 + 1.2 * mach + 0.5 * hip + m_noise
    t["beta0"] = 5.0 + 0.5 * mach + synth.B0_OFF[a] + t["b"][0]
    t["s"] = max(0.2, 0.55 - 0.12 * mach - 0.12 * hip + s_noise)
    t["tau"] = max(0.03, 0.10 - 0.03 * hip - 0.02 * mach + tau_noise)
    t["mach_true"] = mach
    c = dict(st); c.update(hip=hip, mach=mach, ht=ht, truth=t)
    c["z"] = synth.feature_vector(c)
    return c


def root_cause_cf(run, config, seed, n_per_alloy=40):
    """E8 (corrected). Effects on ln fatigue strength at 1e7 cycles (median), HIP always with heat treatment
    (on-support, as in the generator), compared with the generator's counterfactual truth."""
    import jax.numpy as jnp
    from amf import evaluate as ev
    from amf import rfl
    from experiments.config import N_FS
    mp = run["model_params"][OURS]
    M = rfl.FittedRFL({k: jnp.asarray(v) for k, v in mp["P"].items()}, rfl.RFLConfig(**mp["cfg"]), mp["norm_stats"], [], 0)
    P = ev.RFLPredictor(M, chunk=50)
    test = regen_test(config, seed)
    rows = []
    for alloy in synth.ALLOYS:
        sub = [s for s in test if s["alloy"] == alloy and not s["truth"]["anomalous"]][:n_per_alloy]
        if not sub:
            continue
        cfs = {(h, m): [_cf_study(s, h, m, 1) for s in sub] for h in (0, 1) for m in (0, 1)}
        est = {key: np.log(P.allowable(v, None, np.log(N_FS), 0.5)) for key, v in cfs.items()}
        tru = {key: np.log([synth.true_allowable(c, N_FS, 0.5) for c in v]) for key, v in cfs.items()}
        for lab, (k1, k0) in (("HIP effect, machined", ((1, 1), (0, 1))), ("HIP effect, as-built", ((1, 0), (0, 0))),
                              ("machining effect (no HIP)", ((0, 1), (0, 0)))):
            rows.append(dict(alloy=alloy, effect=lab, est=float(np.mean(est[k1] - est[k0])),
                             true=float(np.mean(tru[k1] - tru[k0])), n=len(sub)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver
def main(config):
    C = CONFIGS[config]
    files = sorted(glob.glob(os.path.join(ROOT, "results", "raw", f"syn_{config}_seed*.pkl")))
    out = {k: [] for k in ("zero", "fs", "shift", "cond", "pac", "few", "for", "rc", "rc_cf", "cond_fresh")}
    for f in files:
        run = pickle.load(open(f, "rb"))
        seed = run["seed"]
        nf = os.path.join(ROOT, "results", "raw", f"naive_{config}_seed{seed}.pkl")
        naive = None
        if os.path.exists(nf):
            nv = pickle.load(open(nf, "rb"))
            naive = {k: v for k, v in nv.items() if isinstance(v, dict)}
        test_st = regen_test(config, seed)
        tab = true_sf_table(test_st)
        pac_df = pd.concat([pac(run, C["pac_draws"], seed, tab),
                            pac_fresh_pool(run, config, seed, draws=C["pac_draws"])], ignore_index=True)
        for key, df in (("zero", zero_shot(run, tab)), ("fs", fs_bias(run)), ("pac", pac_df),
                        ("few", few_shot(run, C["ks"], tab, seed)), ("for", forensic(run, seed)),
                        ("rc", root_cause(run)), ("rc_cf", root_cause_cf(run, config, seed)),
                        ("cond_fresh", pac_fresh_pool.cond)):
            df["seed"] = seed
            out[key].append(df)
        sh, cond = conformal_shift(run, naive, test_st, C["conf_reps"], seed, tab)
        sh["seed"] = seed; cond["seed"] = seed
        out["shift"].append(sh); out["cond"].append(cond)
        print("analysed", os.path.basename(f), flush=True)
    tabdir = os.path.join(ROOT, "results", "tables"); os.makedirs(tabdir, exist_ok=True)
    agg = {}
    for key, parts in out.items():
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(tabdir, f"syn_{config}_{key}_perseed.csv"), index=False)
        agg[key] = df
    with open(os.path.join(tabdir, f"syn_{config}_all.pkl"), "wb") as f:
        pickle.dump(agg, f)
    print("seeds:", len(files))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    main(ap.parse_args().config)
