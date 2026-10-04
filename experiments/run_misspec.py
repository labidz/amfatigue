"""A2 stress test: the benchmark with four departures the model does not encode (amf.synth misspec=True).

Departures: duplex initiation (30 % of as-built specimens at 1/5 of the life), Student-t(4) scatter, run-out
limits that depend on surface state (violates C independent of z), 2.7x larger slope spread.
Reports zero-shot NLL and exact true coverage of the conformal lower bound (overall and by surface state),
unweighted vs inverse-propensity-weighted calibration (weights 1/P(C_s >= c0 | surface)), 3-coupon NLL and
coverage, and the true reliability of PAC allowables at k = 0 and 3.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import baselines as bl  # noqa: E402
from amf import conformal as cf  # noqa: E402
from amf import evaluate as ev  # noqa: E402
from amf import rfl, synth  # noqa: E402
from experiments.analyze_synthetic import true_cov, true_sf_table  # noqa: E402
from experiments.config import ALPHA, C0, CONFIGS, DELTA, ND  # noqa: E402

LC0, LND = np.log(C0), np.log(ND)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def weighted_eta(E, w_cal, w_test, alpha):
    """Weighted split-conformal quantile (Tibshirani et al. 2019) for a test point of weight w_test."""
    o = np.argsort(E); E, w = np.asarray(E)[o], np.asarray(w_cal)[o]
    p = np.cumsum(w) / (w.sum() + w_test)
    k = np.searchsorted(p, 1 - alpha)
    return E[k] if k < len(E) else np.inf


def main(config, seed):
    C = CONFIGS[config]
    t0 = time.time()
    st = synth.generate(C["n_train"] + C["n_cal"] + C["n_test"], seed=9000 + seed, misspec=True)
    train, cal, test = st[:C["n_train"]], st[C["n_train"]:C["n_train"] + C["n_cal"]], st[C["n_train"] + C["n_cal"]:]
    tab = true_sf_table(test, nq=1000)
    sid_t = np.concatenate([np.full(len(s["N"]), i) for i, s in enumerate(test)])
    mach_t = np.concatenate([np.full(len(s["N"]), s["mach"]) for s in test])
    X_c, lt_c, d_c, sid_c = bl.specimen_table(cal)
    C_c = np.concatenate([np.full(len(s["N"]), s["C"]) for s in cal])
    mach_c = np.concatenate([np.full(len(s["N"]), s["mach"]) for s in cal])
    # propensity of being an eligible calibration study (C >= c0), by surface state, estimated on cal studies
    elig_st = np.array([s["C"] >= C0 * 0.999 for s in cal]); mach_st = np.array([s["mach"] for s in cal])
    pi = {m: max(elig_st[mach_st == m].mean(), 1e-3) for m in (0, 1)}
    shots_c, shots_t = ev.shot_masks(cal, [3], seed + 11), ev.shot_masks(test, [3], seed + 12)
    fs_c, fs_t = ev.flat_shot(shots_c[3], cal), ev.flat_shot(shots_t[3], test)
    common = dict(steps=C["steps"], batch_studies=C["batch"], n_gh_train=C["gh_train"], n_gh_pred=C["gh_pred"],
                  K=C["K"], seed=seed)
    models = {"H-NRFL (ours)": rfl.RFLConfig(**common), "linear g": rfl.RFLConfig(covnet="linear", **common),
              "Pascual-Meeker RFL, pooled": rfl.RFLConfig(covnet="linear", re=0, **common)}
    rows, rng = [], np.random.default_rng(seed + 3)
    Xtr, ltr, dtr, gtr = bl.specimen_table(train)
    preds = {}
    for name, cfg in models.items():
        log("fit", name)
        P = ev.RFLPredictor(rfl.fit(train, cfg), chunk=50)
        for k in ((0, 3) if cfg.re else (0,)):
            preds[(name, k)] = (P.predict(cal, shots_c[k] if k else None, ALPHA),
                                P.predict(test, shots_t[k] if k else None, ALPHA), P)
    for name, mdl in (("GBM (asfail)", bl.GaussRegressor("GBM", "asfail", seed)),
                      (f"Deep ensemble x{C['ens']}", bl.CensoredLogNormalMLP(C["ens"], steps=C["aft_steps"], seed=seed))):
        log("fit", name)
        mdl.fit(Xtr, ltr, dtr, gtr)
        P = ev.BaselinePredictor(mdl)
        preds[(name, 0)] = (P.predict(cal, None, ALPHA), P.predict(test, None, ALPHA), P)
    for (name, k), (pc, pt, P) in preds.items():
        shot_c = fs_c if k else np.zeros_like(fs_c); shot_t = fs_t if k else np.zeros_like(fs_t)
        ev_t = shot_t == 0
        ok = (shot_c == 0) & (C_c >= C0 * 0.999)
        res = {"unweighted": [], "weighted": [], "pac": []}
        cov_surf = {"unweighted": {0: [], 1: []}, "weighted": {0: [], 1: []}}
        for _ in range(50):
            idx = np.flatnonzero(ok)
            pick = idx[cf.one_per_study(sid_c[idx], rng)]
            E = cf.scores(pc["lnQ"][pick], lt_c[pick], d_c[pick], LC0)
            eta = cf.split_eta(E, ALPHA)
            L = cf.lpb(pt["lnQ"], eta, LC0)
            wc = np.array([1 / pi[m] for m in mach_c[pick]])
            eta_w = np.array([weighted_eta(E, wc, 1 / pi[m], ALPHA) for m in (0, 1)])
            Lw = cf.lpb(pt["lnQ"], eta_w[mach_t], LC0)
            Lp = cf.lpb(pt["lnQ"], cf.pac_eta(E, ALPHA, DELTA), LC0)
            for lab, LL in (("unweighted", L), ("weighted", Lw), ("pac", Lp)):
                res[lab].append(true_cov(tab, LL, sid_t, ev_t))
            for lab, LL in (("unweighted", L), ("weighted", Lw)):
                for m in (0, 1):
                    cov_surf[lab][m].append(true_cov(tab, LL, sid_t, ev_t & (mach_t == m)))
        row = dict(model=name, k=k, nll=float(-np.mean(pt["lp"][ev_t])) if np.all(np.isfinite(pt["lp"])) else np.nan,
                   cov_unweighted=float(np.mean(res["unweighted"])), cov_weighted=float(np.mean(res["weighted"])),
                   cov_pac=float(np.mean(res["pac"])), p_pac_ge_90=float(np.mean(np.array(res["pac"]) >= 0.9)),
                   cov_asbuilt_unweighted=float(np.mean(cov_surf["unweighted"][0])),
                   cov_machined_unweighted=float(np.mean(cov_surf["unweighted"][1])),
                   cov_asbuilt_weighted=float(np.mean(cov_surf["weighted"][0])),
                   cov_machined_weighted=float(np.mean(cov_surf["weighted"][1])),
                   share_eligible_asbuilt=float(pi[0]), share_eligible_machined=float(pi[1]))
        if name == "H-NRFL (ours)":
            idx = np.flatnonzero(ok); pick = idx[cf.one_per_study(sid_c[idx], rng)]
            eta_p = cf.pac_eta(cf.scores(pc["lnQ"][pick], lt_c[pick], d_c[pick], LC0), ALPHA, DELTA)
            if np.isfinite(eta_p) and LND + eta_p <= LC0:
                a = P.allowable(test, shots_t[k] if k else None, LND + eta_p, ALPHA)
                rel = np.array([np.exp(synth.true_logsf(s, x, LND, 1000)) for s, x in zip(test, a)])
                row.update(allow_mean_rel=float(rel.mean()), allow_frac_rel_ge_90=float((rel >= 0.9).mean()))
        rows.append(row)
        log(name, k, {kk: round(v, 3) for kk, v in row.items() if isinstance(v, float)})
    df = pd.DataFrame(rows); df["seed"] = seed
    out = os.path.join(os.path.dirname(__file__), "..", "results")
    df.to_csv(os.path.join(out, "tables", f"misspec_{config}_seed{seed}.csv"), index=False)
    log("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.config, a.seed)
