"""E9: the whole suite on FatigueData-AM2022 (real AM data). One command once the file is available:

    python experiments/run_am2022.py --data /path/to/FatigueData-AM2022.json --config fast

Splits are by ARTICLE (all datasets of one paper go to the same split), so no paper leaks across splits.
There is no ground truth on real data, so coverage is evaluated where it is observable: on test studies whose
run-out limit is >= c0, min(T, c0) is observed for every specimen.
Outputs: results/tables/am2022_*.csv and results/raw/am2022_<config>.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import am2022  # noqa: E402
from amf import baselines as bl  # noqa: E402
from amf import conformal as cf  # noqa: E402
from amf import evaluate as ev  # noqa: E402
from amf import metrics, rfl  # noqa: E402
from experiments.config import ALPHA, C0, CONFIGS, DELTA, N_FS  # noqa: E402

LC0 = np.log(C0)
ROOT = os.path.join(os.path.dirname(__file__), "..")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def split_by_article(studies, seed, frac=(0.5, 0.25, 0.25)):
    dois = sorted({s["doi"] or f"nodoi-{s['id']}" for s in studies})
    rng = np.random.default_rng(seed)
    rng.shuffle(dois)
    n = len(dois)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    grp = {d: (0 if i < a else 1 if i < b else 2) for i, d in enumerate(dois)}
    parts = ([], [], [])
    for s in studies:
        parts[grp[s["doi"] or f"nodoi-{s['id']}"]].append(s)
    return parts


def base(studies):
    X, lt, d, g = bl.specimen_table(studies)
    C = np.concatenate([np.full(len(s["N"]), s["C"]) for s in studies])
    # the independent unit is the article (assumption A1): several datasets of one article share an index
    keys = [s["doi"] or f"nodoi-{s['id']}" for s in studies]
    art_id = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    art = np.concatenate([np.full(len(s["N"]), art_id[k]) for s, k in zip(studies, keys)])
    return dict(lt=lt, d=d, sid=g, C=C, art=art)


def obs_cov(L, b, sel=None):
    """Observable coverage on studies with C >= c0: min(T, c0) >= L, averaged per study (optionally a subset)."""
    ok = b["C"] >= C0 * (1 - 1e-9)
    if sel is not None:
        ok = ok & sel
    if not ok.any():
        return np.nan
    tc = np.where(b["d"] > 0, np.minimum(b["lt"], LC0), LC0)
    cov = (tc >= np.asarray(L) - 1e-12)[ok]
    df = pd.DataFrame(dict(c=cov.astype(float), s=b["sid"][ok]))
    return float(df.groupby("s").c.mean().mean())


def draw_E(pr, b, shot, rng):
    """One calibration specimen per ARTICLE (not per dataset), so calibration units are independent (A1)."""
    ok = (shot == 0) & (b["C"] >= C0 * (1 - 1e-9))
    idx = np.flatnonzero(ok)
    pick = idx[cf.one_per_study(b["art"][idx], rng)]
    return cf.scores(pr["lnQ"][pick], b["lt"][pick], b["d"][pick], LC0)


def study_scores(pr, b, n_st):
    F = np.clip(pr["F"], 1e-12, 1 - 1e-12)
    z = stats.norm.ppf(F)
    r = np.where(b["d"] > 0, -z, -stats.norm.pdf(z) / (1 - F))
    return np.array([r[b["sid"] == s].mean() for s in range(n_st)])


def main(path, config, outdir):
    C = CONFIGS[config]
    t0 = time.time()
    studies, info = am2022.load(path)
    log("loaded", info["n_raw"], "datasets;", info["n_used"], "usable uniaxial RT smooth S-N datasets")
    qual = pd.DataFrame([dict(alloy=s["alloy"], am=s["am"], n=len(s["N"]), runout_frac=1 - s["d"].mean(), C=s["C"],
                              hip=s["hip"], machined=s["mach"]) for s in studies])
    os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)
    qual.groupby("alloy").agg(datasets=("n", "size"), specimens=("n", "sum"), runout_frac=("runout_frac", "mean"),
                              share_C_ge_1e7=("C", lambda c: float((c >= C0 * 0.999).mean()))).to_csv(
        os.path.join(ROOT, "results", "tables", "am2022_data_summary.csv"))
    rows = {k: [] for k in ("zero", "few", "pac", "fs", "flags", "rc")}
    for seed in C["seeds"]:
        train, cal, test = split_by_article(studies, seed)
        bc, bt = base(cal), base(test)
        nmax = max(len(s["N"]) for s in studies)
        common = dict(steps=C["steps"], batch_studies=C["batch"], n_gh_train=C["gh_train"], n_gh_pred=C["gh_pred"],
                      K=C["K"], seed=seed)
        models = {"H-NRFL (ours)": (rfl.RFLConfig(**common), "proper"),
                  "H-NRFL, run-outs dropped": (rfl.RFLConfig(**common), "drop"),
                  "H-NRFL, run-outs as failures": (rfl.RFLConfig(**common), "asfail"),
                  "Pascual-Meeker RFL, pooled": (rfl.RFLConfig(covnet="linear", re=0, **common), "proper"),
                  "w/o covariates (RE only)": (rfl.RFLConfig(covnet="none", **common), "proper")}
        preds = {}
        rng = np.random.default_rng(seed + 5)
        for name, (cfg, cens) in models.items():
            log("seed", seed, "fit", name)
            M = rfl.fit(train, cfg, censor=cens, nmax=nmax)
            P = ev.RFLPredictor(M, chunk=50)
            ks = C["ks"] if name in ("H-NRFL (ours)", "w/o covariates (RE only)") else [0]
            sh_c, sh_t = ev.shot_masks(cal, ks, seed + 11), ev.shot_masks(test, ks, seed + 12)
            for k in ks:
                pc = P.predict(cal, sh_c[k] if k else None, ALPHA)
                pt = P.predict(test, sh_t[k] if k else None, ALPHA)
                fsc, fst = ev.flat_shot(sh_c[k], cal), ev.flat_shot(sh_t[k], test)
                preds[(name, k)] = (pc, pt)
                E = draw_E(pc, bc, fsc, rng)
                eta = cf.split_eta(E, ALPHA)
                L = cf.lpb(pt["lnQ"], eta, LC0)
                evm = fst == 0
                rows["few"].append(dict(seed=seed, model=name, k=k, nll=float(-np.mean(pt["lp"][evm])),
                                        obs_cov=obs_cov(L, bt, evm),
                                        mean_lpb_log10=float(np.mean(L[evm]) / np.log(10))))
                if k == 0:
                    c = metrics.cindex(bt["lt"], bt["d"], pt["lnmed"])
                    _, pv, md = metrics.dcal(pt["F"], bt["d"])
                    rows["zero"].append(dict(seed=seed, model=name, nll=float(-np.mean(pt["lp"])), cindex=c,
                                             dcal_maxdev=md, dcal_p=pv, conf_obs_cov=obs_cov(L, bt)))
                    res = {"split": [], "pac": []}
                    for _ in range(C["pac_draws"]):
                        Ed = draw_E(pc, bc, fsc, rng)
                        res["split"].append(obs_cov(cf.lpb(pt["lnQ"], cf.split_eta(Ed, ALPHA), LC0), bt))
                        res["pac"].append(obs_cov(cf.lpb(pt["lnQ"], cf.pac_eta(Ed, ALPHA, DELTA), LC0), bt))
                    for lab, v in res.items():
                        v = np.array(v)
                        rows["pac"].append(dict(seed=seed, model=name, method=lab, mean_cov=float(np.nanmean(v)),
                                                p_cov_ge_90=float(np.nanmean(v >= 0.9)), m_cal=len(E)))
            # run-out audit: fatigue strength at N_FS per test study
            rows["fs"].append(pd.DataFrame(dict(seed=seed, model=name, study=[s["id"] for s in test],
                                                doi=[s["doi"] for s in test], alloy=[s["alloy"] for s in test],
                                                fs50=P.allowable(test, None, np.log(N_FS), 0.5),
                                                fs10=P.allowable(test, None, np.log(N_FS), 0.1))))
            if name == "H-NRFL (ours)":
                rows["rc"].append(dict(seed=seed, walker=M.walker, sd_beta0=float(np.sqrt(M.re_cov[0, 0])),
                                       sd_mu=float(np.sqrt(M.re_cov[1, 1]))))
                # forensic screen on test studies (study-level conformal p-values, BH q = 0.10)
                pc0, pt0 = preds[(name, 0)]
                Sc = study_scores(pc0, bc, len(cal)); St = study_scores(pt0, bt, len(test))
                p = cf.pvalues(Sc, St)
                rej = cf.bh(p, 0.10)
                for s, pv_, r_ in zip(test, p, rej):
                    rows["flags"].append(dict(seed=seed, study=s["id"], doi=s["doi"], alloy=s["alloy"], am=s["am"],
                                              hip=s["hip"], machined=s["mach"], n=len(s["N"]), p_value=float(pv_),
                                              flagged=bool(r_)))
        log("seed", seed, "done", f"{time.time()-t0:.0f}s")
    out = {k: (pd.concat(v, ignore_index=True) if v and isinstance(v[0], pd.DataFrame) else pd.DataFrame(v))
           for k, v in rows.items()}
    for k, df in out.items():
        df.to_csv(os.path.join(ROOT, "results", "tables", f"am2022_{config}_{k}.csv"), index=False)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"am2022_{config}.pkl"), "wb") as f:
        pickle.dump(dict(out=out, info=info), f)
    log("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="fast")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "raw"))
    a = ap.parse_args()
    main(a.data, a.config, a.out)
