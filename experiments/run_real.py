"""Real-data experiments on six censored fatigue datasets reachable from this workspace (SMRD, pylife).

R1  Single-study datasets: run-out strategy (censored / dropped / as failures) and model form
    (RFL vs Basquin log-normal vs GBM) -> cross-validated censored NLL, fatigue-strength shifts,
    conformal LPB coverage (specimen-level; one study, so specimens are exchangeable).
R2  Multi-lot HCF data (Ti alloy, R = -1/0.1/0.5): leave-one-lot-out, zero-shot vs 2-shot,
    hierarchical (random effects + Walker) vs pooled Pascual-Meeker RFL.

Usage: python experiments/run_real.py --config fast
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import pyreadr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import baselines as bl  # noqa: E402
from amf import conformal as cf  # noqa: E402
from amf import evaluate as ev  # noqa: E402
from amf import rfl  # noqa: E402
from experiments.config import ALPHA, CONFIGS  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _rd(name):
    return pyreadr.read_r(os.path.join(RAW, "smrd", f"{name}.RData"))[name]


def load_single():
    """Each dataset -> dict(name, x (stress or strain amplitude), N, d, unit, runout_limit, source)."""
    out = []
    d = _rd("laminatepanel")
    out.append(dict(name="Laminate panel (CFRP, notched)", x=d.mpa.values.astype(float), N=d.kilocycles.values * 1e3,
                    d=(d.event.astype(str) == "Failed").values.astype(float), unit="MPa",
                    source="Shimokawa & Hamaguchi (1987) via SMRD"))
    d = _rd("inconel")
    out.append(dict(name="Inconel 718 (strain-life)", x=d.strain.values.astype(float), N=d.cycles.values.astype(float),
                    d=(d.event.astype(str) == "Failed").values.astype(float), unit="strain",
                    source="Shen (1994) via SMRD"))
    d = _rd("titanium")
    out.append(dict(name="Ti alloy (strain-life)", x=d.strain.values.astype(float), N=d.kilocycles.values * 1e3,
                    d=(d.event.astype(str) == "Failed").values.astype(float), unit="strain (scaled)",
                    source="Meeker & Escobar (1998) via SMRD"))
    d = _rd("superalloy")
    d = d.loc[d.index.repeat(d["count"].astype(int))]
    out.append(dict(name="Ni superalloy (LCF, pseudo-stress)", x=d.pstress.values.astype(float),
                    N=d.kilocycles.values * 1e3, d=(d.event.astype(str) == "Failure").values.astype(float),
                    unit="ksi (pseudo)", source="Nelson (1990) via SMRD"))
    d = pd.read_csv(os.path.join(RAW, "fatigue-data-fractures.csv"))
    out.append(dict(name="Steel S-N (pylife demo)", x=d.iloc[:, 0].values.astype(float), N=d.iloc[:, 1].values.astype(float),
                    d=(d.iloc[:, 2].astype(str).str.lower() == "failure").values.astype(float), unit="MPa",
                    source="pylife demo data (Bosch Research)"))
    for o in out:
        o["runout_limit"] = float(np.min(o["N"][o["d"] == 0])) if (o["d"] == 0).any() else float(o["N"].max())
    return out


def as_study(x, N, d, ref):
    return dict(z=np.zeros(1), sa=x, R=np.full(len(x), -1.0), N=N, d=d, ref=ref)


def ls_init(x, N, d, fl=True):
    """Least-squares start on failures: ln N = b0 - m ln(u - g0) (or ln u for Basquin)."""
    ref = float(x.max())
    u = x / ref
    g0 = 0.9 * u.min()
    xf = np.log(u[d > 0] - g0) if fl else np.log(u[d > 0])
    yf = np.log(N[d > 0])
    slope, b0 = np.polyfit(xf, yf, 1)
    m0 = float(np.clip(-slope, 1.0, 30.0))
    s0 = float(np.clip(np.std(yf - (b0 + slope * xf)), 0.1, 3.0))
    return ref, (float(b0), rfl.softplus_inv(m0 - 0.5), rfl.softplus_inv(max(s0 - 0.05, 0.05)), float(np.log(g0)),
                 rfl.softplus_inv(0.05))


def single_cfg(x, N, d, steps, seed, fl=True):
    ref, init = ls_init(x, N, d, fl)
    return rfl.RFLConfig(covnet="none", re=0, walker=False, fatigue_limit=fl, steps=steps, lr=2e-2, seed=seed,
                         init_theta=init, n_gh_pred=1, n_gh_train=1, K=24), ref


def r1(C, seed=0, repeats=3):
    res = []
    fs_rows = []
    for ds in load_single():
      for rep in range(repeats):
        rng = np.random.default_rng(seed + 1000 * rep)
        x, N, d = ds["x"], ds["N"], ds["d"]
        n = len(x)
        lc0 = np.log(ds["runout_limit"])
        folds = np.array_split(rng.permutation(n), C["real_folds"])
        models = {"RFL, censored (proper)": ("rfl", "proper", True), "RFL, run-outs dropped": ("rfl", "drop", True),
                  "RFL, run-outs as failures": ("rfl", "asfail", True),
                  "Basquin log-normal, censored": ("rfl", "proper", False),
                  "GBM (drop)": ("gbm", "drop", None), "GBM (asfail)": ("gbm", "asfail", None),
                  "GBM (flag)": ("gbm", "flag", None)}
        for mname, (kind, strat, fl) in models.items():
            lp_all = np.full(n, np.nan); lnQ_all = np.full(n, np.nan)
            for f in range(len(folds)):
                te = folds[f]; tr = np.setdiff1d(np.arange(n), te)
                if kind == "rfl":
                    cfg, ref = single_cfg(x[tr], N[tr], d[tr], C["real_steps"], seed, fl)
                    M = rfl.fit([as_study(x[tr], N[tr], d[tr], ref)], cfg, censor=strat, nmax=n)
                    P = ev.RFLPredictor(M)
                    pr = P.predict([as_study(x[te], N[te], d[te], ref)], None, ALPHA)
                else:
                    Xtr = np.log(x[tr])[:, None]; Xte = np.log(x[te])[:, None]
                    mdl = bl.GaussRegressor("GBM", strat, seed)
                    mdl.fit(Xtr, np.log(N[tr]), d[tr], np.arange(len(tr)) % 3)
                    pr = dict(lp=mdl.logpred(Xte, np.log(N[te]), d[te]), lnQ=mdl.quantile(Xte, ALPHA))
                lp_all[te] = pr["lp"]; lnQ_all[te] = pr["lnQ"]
            # conformal LPB coverage by cross-conformal (scores from other folds, exchangeable within one study)
            E = cf.scores(lnQ_all, np.log(N), d, lc0)
            cov = []
            for f in range(len(folds)):
                te = folds[f]; ca = np.setdiff1d(np.arange(n), te)
                eta = cf.split_eta(E[ca], ALPHA)
                L = cf.lpb(lnQ_all[te], eta, lc0)
                obs = np.where(d[te] > 0, np.minimum(np.log(N[te]), lc0), lc0)
                cov.append(obs >= L - 1e-12)
            cov = np.concatenate(cov)
            raw_cov = np.mean(np.where(d > 0, np.minimum(np.log(N), lc0), lc0) >= np.minimum(lnQ_all, lc0) - 1e-12)
            res.append(dict(dataset=ds["name"], rep=rep, n=n, runout_frac=float(1 - d.mean()), model=mname,
                            cv_nll=float(-np.mean(lp_all)), raw_cov=float(raw_cov), conformal_cov=float(cov.mean())))
            log(ds["name"], mname, f"NLL {-np.mean(lp_all):.3f}")
      # fatigue strength at the run-out limit (median and 10 %), full-data fits
      for strat in ("proper", "drop", "asfail"):
            cfg, ref = single_cfg(x, N, d, C["real_steps"], seed, True)
            M = rfl.fit([as_study(x, N, d, ref)], cfg, censor=strat, nmax=n)
            P = ev.RFLPredictor(M)
            st = as_study(x, N, d, ref)
            f50 = P.allowable([st], None, lc0, 0.5)[0]; f10 = P.allowable([st], None, lc0, 0.1)[0]
            fs_rows.append(dict(dataset=ds["name"], strategy=strat, fs50=f50, fs10=f10, unit=ds["unit"],
                                at_cycles=ds["runout_limit"], min_test_level=float(x.min()), max_test_level=float(x.max())))
    return pd.DataFrame(res), pd.DataFrame(fs_rows)


def load_hcf():
    d = _rd("hcfdata")
    d = d[d.step.astype(str) == "No"].copy()
    d["lot"] = d.specimen.astype(str).str.split("-").str[0]
    d["sa"] = d["range"] / 2.0
    d["d"] = (d.event == 1).astype(float)
    return d


def r2(C, seed=0):
    d = load_hcf()
    lots = sorted(d.lot.unique())
    nmax_lot = int(d.groupby("lot").size().max())

    def studies_of(sub, ref):
        out = []
        for lot, g in sub.groupby("lot"):
            out.append(dict(z=np.array([np.log(g.hertz.values.mean())]), sa=g.sa.values, R=g.ratio.values,
                            N=g.cycles.values.astype(float), d=g.d.values, ref=ref, lot=lot))
        return out

    rows = []
    rng = np.random.default_rng(seed)
    for held in lots:
        dt = d[d.lot != held]  # starting values and reference stress from the TRAINING lots only
        xeq = dt.sa.values * (2.0 / (1.0 - np.minimum(dt.ratio.values, 0.9))) ** 0.4  # Walker start w = 0.6
        ref, init = ls_init(xeq, dt.cycles.values.astype(float), dt.d.values)
        tr = studies_of(dt, ref); te = studies_of(d[d.lot == held], ref)
        for mname, kw in (("Hierarchical RFL + Walker (ours)", dict(covnet="none", re=2)),
                          ("Pooled Pascual-Meeker RFL + Walker", dict(covnet="none", re=0)),
                          ("Hierarchical, w/o Walker", dict(covnet="none", re=2, walker=False))):
            cfg = rfl.RFLConfig(steps=C["real_steps"], lr=2e-2, seed=seed, init_theta=init, n_gh_train=C["gh_train"],
                                n_gh_pred=C["gh_pred"], K=C["K"], init_re_sd=(0.3, 0.05), **kw)
            M = rfl.fit(tr, cfg, nmax=nmax_lot)
            P = ev.RFLPredictor(M)
            n = len(te[0]["N"])
            for k in (0, 2):
                if k and kw["re"] == 0:
                    continue
                vals = []
                for rep in range(10 if k else 1):
                    shot = np.zeros((1, n))
                    if k:
                        shot[0, rng.choice(n, k, replace=False)] = 1
                    pr = P.predict(te, shot if k else None, ALPHA)
                    ev_mask = shot[0] == 0
                    vals.append(-np.mean(pr["lp"][ev_mask]))
                rows.append(dict(held_out_lot=held, n_lot=n, model=mname, k=k, nll=float(np.mean(vals)),
                                 walker=M.walker))
            log("hcf", held, mname)
    return pd.DataFrame(rows)


def main(config, only=None):
    C = CONFIGS[config]
    outdir = os.path.join(os.path.dirname(__file__), "..", "results")
    t0 = time.time()
    if only == "r2":  # rerun R2 only, keep the stored R1 results
        old = pickle.load(open(os.path.join(outdir, "raw", f"real_{config}.pkl"), "rb"))
        r1df, fsdf = old["r1"], old["fs"]
    else:
        r1df, fsdf = r1(C)
    r2df = r2(C)
    os.makedirs(os.path.join(outdir, "raw"), exist_ok=True)
    with open(os.path.join(outdir, "raw", f"real_{config}.pkl"), "wb") as f:
        pickle.dump(dict(r1=r1df, fs=fsdf, r2=r2df), f)
    log("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    ap.add_argument("--only", default=None, choices=[None, "r2"])
    a = ap.parse_args()
    main(a.config, a.only)
