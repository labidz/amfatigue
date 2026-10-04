"""Checks added after the internal pre-submission review (fast configuration, saved runs).

1. init      Walker exponent and random-effect SDs are refitted on seed-0 training data from initial values away
             from the generator's truth (w0 = 0.3 and 0.9; RE sd0 = (0.6, 0.3)).
2. monotone  Predicted failure probability P(N <= t | sigma) is checked to be non-decreasing in stress for the
             saved seed-0 model (200 test studies x 60 stresses x 3 lives), i.e. Proposition 1 numerically.
3. fdr       Marginal FDR of the study-level conformal screen: calibration studies AND screened studies are drawn
             afresh in every replicate from large independent pools (1,500 + 1,500 studies per seed), which is
             the quantity Benjamini-Hochberg with conformal p-values controls (Bates et al. 2023).
4. allow     Confidence of the allowable: over calibration draws, the share of draws whose allowables have mean
             true reliability >= 0.90 across test studies, and the per-study shortfall, for k = 0..8.

Usage: python experiments/run_review_checks.py [init monotone fdr allow]
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import conformal as cf  # noqa: E402
from amf import evaluate as ev  # noqa: E402
from amf import rfl, synth  # noqa: E402
from experiments.analyze_synthetic import (LC0, OURS, _allow_stats, _study_scores, draw_eta,  # noqa: E402
                                           interp_grid)
from experiments.config import ALPHA, CONFIGS, DELTA  # noqa: E402
from experiments.run_synthetic import base_arrays  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
C = CONFIGS["fast"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(seed):
    return pickle.load(open(os.path.join(ROOT, "results", "raw", f"syn_fast_seed{seed}.pkl"), "rb"))


def saved_model(run):
    mp = run["model_params"][OURS]
    return rfl.FittedRFL({k: jnp.asarray(v) for k, v in mp["P"].items()}, rfl.RFLConfig(**mp["cfg"]),
                         mp["norm_stats"], [], 0), mp


def check_init():
    run = load(0)
    _, mp = saved_model(run)
    allst = synth.generate(C["n_train"] + C["n_cal"] + C["n_test"], seed=1000)
    train, test = allst[:C["n_train"]], allst[C["n_train"] + C["n_cal"]:]
    rows = []
    for w0, re0 in ((0.3, (0.3, 0.1)), (0.9, (0.3, 0.1)), (0.6, (0.6, 0.3)), (0.6, (0.3, 0.1))):
        cfgd = dict(mp["cfg"]); cfgd.update(init_walker=float(np.log(w0 / (1 - w0))), init_re_sd=re0)
        M = rfl.fit(train, rfl.RFLConfig(**cfgd))
        pr = ev.RFLPredictor(M, chunk=50).predict(test, None, ALPHA)
        rows.append(dict(init_w=w0, init_sd_beta0=re0[0], init_sd_mu=re0[1], walker=M.walker,
                         sd_beta0=float(np.sqrt(M.re_cov[0, 0])), sd_mu=float(np.sqrt(M.re_cov[1, 1])),
                         test_nll_k0=float(-np.mean(pr["lp"])), true_w=synth.W_TRUE,
                         true_sd_beta0=synth.RE_SD[0], true_sd_mu=synth.RE_SD[1]))
        log("init", rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "review_init_sensitivity.csv"), index=False)


def check_monotone():
    run = load(0)
    M, _ = saved_model(run)
    P = ev.RFLPredictor(M, chunk=20)
    test = synth.generate(C["n_train"] + C["n_cal"] + C["n_test"], seed=1000)[C["n_train"] + C["n_cal"]:]
    grid = np.exp(np.linspace(np.log(0.05), np.log(1.2), 60))
    rows = []
    for life in (1e5, 2e6, 1e7):
        viol, worst = 0, 0.0
        tmp = []
        for st in test:
            s = dict(st); s["sa"] = st["ref"] * grid; s["R"] = np.full(len(grid), st["R"][0])
            s["N"] = np.full(len(grid), life); s["d"] = np.ones(len(grid))
            tmp.append(s)
        F = P.predict(tmp, None, ALPHA)["F"].reshape(len(test), len(grid))
        dF = np.diff(F, axis=1)
        viol = int((dF < -1e-6).sum()); worst = float(dF.min())
        rows.append(dict(life=life, studies=len(test), stresses=len(grid), decreasing_steps=viol,
                         most_negative_step=worst))
        log("monotone", rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "review_monotonicity.csv"), index=False)


def check_fdr(R=200, n_sub=150, q=0.10, n_pool=1500):
    rows = []
    for seed in (0, 1, 2):
        run = load(seed)
        M, _ = saved_model(run)
        P = ev.RFLPredictor(M, chunk=50)
        calp = synth.generate(n_pool, seed=8000 + seed)
        scr = synth.generate(n_pool, seed=8500 + seed, anomaly_frac=0.2)
        bc, bs = base_arrays(calp), base_arrays(scr)
        Sc = _study_scores(P.predict(calp, None, ALPHA), bc, n_pool)
        Ss = _study_scores(P.predict(scr, None, ALPHA), bs, n_pool)
        anom = np.array([st["truth"]["anomalous"] for st in scr])
        rng = np.random.default_rng(seed + 31)
        fdp, pw = [], []
        for _ in range(R):
            ci = rng.choice(n_pool, n_sub, replace=False)
            ti = rng.choice(n_pool, n_sub, replace=False)
            rej = cf.bh(cf.pvalues(Sc[ci], Ss[ti]), q); an = anom[ti]
            fdp.append((rej & ~an).sum() / max(rej.sum(), 1)); pw.append((rej & an).sum() / max(an.sum(), 1))
        rows.append(dict(seed=seed, fdr=float(np.mean(fdp)), fdr_se=float(np.std(fdp) / np.sqrt(R)),
                         power=float(np.mean(pw)), replicates=R, n_cal=n_sub, n_screened=n_sub,
                         pool_size=n_pool, anomaly_share=float(anom.mean())))
        log("fdr", rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "review_marginal_fdr.csv"), index=False)


def check_allow(draws=200):
    rows = []
    for seed in (0, 1, 2):
        run = load(seed)
        rng = np.random.default_rng(seed + 41)
        tr = run["truth"]
        for k in C["ks"]:
            if (OURS, k) not in run["allow_grid"]:
                continue
            G, Gr, eg = run["allow_grid"][(OURS, k)], run["allow_grid_rel"][(OURS, k)], run["eta_grid"]
            prc = run["preds"][(OURS, k, "cal")]; shc = run["shots_flat"]["cal"][k]
            acc = {"split": [], "pac": []}
            for _ in range(draws):
                E = draw_eta(prc, run["base"]["cal"], shc, rng)
                for lab, eta in (("split", cf.split_eta(E, ALPHA)), ("pac", cf.pac_eta(E, ALPHA, DELTA))):
                    if np.isfinite(eta) and eta <= eg[-1]:
                        a, rel = interp_grid(G, eg, eta), interp_grid(Gr, eg, eta)
                    else:
                        a, rel = np.zeros(G.shape[0]), np.ones(G.shape[0])
                    s = _allow_stats(a, rel, tr["allowB"])
                    acc[lab].append((s["mean_rel"], s["frac_rel_ge_90"], s["median_eff"], s["frac_zero"]))
            for lab, v in acc.items():
                v = np.array(v, float)
                rows.append(dict(seed=seed, k=k, method=lab, mean_rel=float(np.nanmean(v[:, 0])),
                                 p_mean_rel_ge_90=float(np.nanmean(v[:, 0] >= 0.9)),
                                 share_studies_ge_90=float(np.nanmean(v[:, 1])),
                                 median_eff=float(np.nanmean(v[:, 2])), frac_zero=float(np.nanmean(v[:, 3])),
                                 draws=draws))
        log("allow seed", seed)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "review_allowable_confidence.csv"), index=False)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["allow", "monotone", "fdr", "init"]
    for t in todo:
        {"init": check_init, "monotone": check_monotone, "fdr": check_fdr, "allow": check_allow}[t]()
