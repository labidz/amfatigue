"""Stage 1 (synthetic): fit all models for one seed and save every prediction needed by E1-E8.

Usage: python experiments/run_synthetic.py --config fast --seed 0
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import baselines as bl  # noqa: E402
from amf import conformal as cf  # noqa: E402
from amf import evaluate as ev  # noqa: E402
from amf import rfl, synth  # noqa: E402
from experiments.config import ALPHA, C0, C0_FORENSIC, CONFIGS, DELTA, N_FS, ND  # noqa: E402

LC0, LND = np.log(C0), np.log(ND)
ETA_GRID = np.round(np.arange(-1.0, 1.0001, 0.1), 2)  # conformal offsets for which allowables are stored


def jax_to_np(tree):
    return {k: np.asarray(v) for k, v in tree.items()}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def base_arrays(studies):
    X, lt, d, g = bl.specimen_table(studies)
    lT = np.concatenate([np.log(st["T_true"]) for st in studies]) if "T_true" in studies[0] else None
    C = np.concatenate([np.full(len(st["N"]), st["C"]) for st in studies])
    return dict(lt=lt, d=d, sid=g, lT=lT, C=C)


def calibrate_eta(pred_cal, cal_base, shot_flat, lc0, rng, alpha=ALPHA, delta=DELTA):
    """One-per-study draw among non-shot specimens of calibration studies whose run-out limit is >= c0.

    Selecting studies on their (exogenous) run-out limit keeps T ^ c0 observed for every eligible specimen
    without conditioning on the outcome. Returns (eta_split, eta_pac, m).
    """
    ok = (shot_flat == 0) & (cal_base["C"] >= np.exp(lc0) * (1 - 1e-9))
    idx = np.flatnonzero(ok)
    pick = idx[cf.one_per_study(cal_base["sid"][idx], rng)]
    E = cf.scores(pred_cal["lnQ"][pick], cal_base["lt"][pick], cal_base["d"][pick], lc0)
    return cf.split_eta(E, alpha), cf.pac_eta(E, alpha, delta), len(E)


def e739_allowable(st, shot_idx, p=0.9, conf=0.95):
    """Per-study ASTM E739 regression on the k shot coupons (failures only) + one-sided tolerance bound."""
    from scipy import stats
    d = np.asarray(st["d"])[shot_idx]
    x = np.log(np.asarray(st["sa"])[shot_idx])[d > 0]
    y = np.log(np.asarray(st["N"])[shot_idx])[d > 0]
    n = len(x)
    if n < 3 or len(np.unique(x)) < 2:
        return np.nan
    b, a = np.polyfit(x, y, 1)
    if b >= 0:
        return np.nan
    resid = y - (a + b * x)
    s = np.sqrt((resid ** 2).sum() / (n - 2))
    if s <= 0:
        return np.nan
    xbar, Sxx = x.mean(), ((x - x.mean()) ** 2).sum()
    zp = stats.norm.ppf(p)
    sig = st["ref"] * np.exp(np.linspace(np.log(0.01), np.log(2.0), 200))
    xs = np.log(sig)
    h = 1 / n + (xs - xbar) ** 2 / Sxx
    K = np.sqrt(h) * stats.nct.ppf(conf, n - 2, zp / np.sqrt(h))
    yL = a + b * xs - K * s
    okk = np.flatnonzero(yL >= LND)
    return sig[okk.max()] if len(okk) else 0.0


def main(config, seed, outdir):
    C = CONFIGS[config]
    t0 = time.time()
    ntr, nca, nte = C["n_train"], C["n_cal"], C["n_test"]
    allst = synth.generate(ntr + nca + nte, seed=1000 + seed)
    train, cal, test = allst[:ntr], allst[ntr:ntr + nca], allst[ntr + nca:]
    anom = synth.generate(C["n_anom"], seed=5000 + seed, anomaly_frac=0.2)
    ks = C["ks"]
    shots = {"cal": ev.shot_masks(cal, ks, seed + 11), "test": ev.shot_masks(test, ks, seed + 12)}
    shots_flat = {s: {k: ev.flat_shot(m, cal if s == "cal" else test) for k, m in shots[s].items()} for s in shots}
    base = {"cal": base_arrays(cal), "test": base_arrays(test), "anom": base_arrays(anom)}
    out = dict(config=config, seed=seed, base=base, shots_flat=shots_flat, preds={}, allow={}, fs={}, eta={}, info={},
               allow_grid={}, allow_grid_rel={}, eta_grid=ETA_GRID, model_params={})

    # ---------------- ground truth per test study
    log("truth")
    truth = dict(
        fs50=np.array([synth.true_allowable(st, N_FS, 0.5) for st in test]),
        fs10=np.array([synth.true_allowable(st, N_FS, 0.9) for st in test]),
        allowB=np.array([synth.true_allowable(st, ND, 1 - ALPHA) for st in test]),
        alloy=np.array([st["alloy"] for st in test]), C=np.array([st["C"] for st in test]),
        anomalous=np.array([st["truth"]["anomalous"] for st in anom]),
    )
    out["truth"] = truth

    def common(**kw):
        return rfl.RFLConfig(steps=C["steps"], batch_studies=C["batch"], n_gh_train=C["gh_train"],
                             n_gh_pred=C["gh_pred"], K=C["K"], seed=seed, **kw)

    variants = {
        "H-NRFL (ours)": (common(), "proper"),
        "H-NRFL, run-outs dropped": (common(), "drop"),
        "H-NRFL, run-outs as failures": (common(), "asfail"),
        "w/o random effects": (common(re=0), "proper"),
        "w/o fatigue limit (Basquin)": (common(fatigue_limit=False), "proper"),
        "linear g (no MLP)": (common(covnet="linear"), "proper"),
        "w/o covariates (RE only)": (common(covnet="none"), "proper"),
        "w/o Walker (amplitude only)": (common(walker=False), "proper"),
        "Pascual-Meeker RFL, pooled (linear, no RE)": (common(covnet="linear", re=0), "proper"),
    }
    rng_cal = np.random.default_rng(seed + 99)
    for name, (cfg, cens) in variants.items():
        log("fit", name)
        M = rfl.fit(train, cfg, censor=cens)
        P = ev.RFLPredictor(M, chunk=50)
        out["info"][name] = dict(walker=M.walker, re_cov=M.re_cov, hist=M.hist)
        if name in ("H-NRFL (ours)", "w/o covariates (RE only)"):
            kk = ks
        elif cfg.re > 0:
            kk = [0, 3]
        else:
            kk = [0]
        for k in kk:
            for split, studies in (("cal", cal), ("test", test)):
                sh = shots[split][k] if k > 0 else None
                out["preds"][(name, k, split)] = P.predict(studies, sh, ALPHA)
            # conformal allowables (E5) + model-only allowables
            e_s, e_p, m_cal = calibrate_eta(out["preds"][(name, k, "cal")], base["cal"], shots_flat["cal"][k], LC0,
                                            rng_cal)
            out["eta"][(name, k)] = dict(split=e_s, pac=e_p, m=m_cal)
            sh = shots["test"][k] if k > 0 else None
            for lab, eta in (("model", 0.0), ("split", e_s), ("pac", e_p)):
                if not np.isfinite(eta) or LND + eta > LC0:
                    out["allow"][(name, k, lab)] = np.zeros(len(test))
                else:
                    out["allow"][(name, k, lab)] = P.allowable(test, sh, LND + eta, ALPHA)
            if name in ("H-NRFL (ours)", "w/o covariates (RE only)"):
                grid = P.allowable(test, sh, LND + ETA_GRID, ALPHA)  # (S, n_eta)
                out["allow_grid"][(name, k)] = grid
                out["allow_grid_rel"][(name, k)] = np.array(
                    [[np.exp(synth.true_logsf(st, g, LND)) for g in row] for st, row in zip(test, grid)])
        if name == "H-NRFL (ours)":
            out["model_params"][name] = dict(P=jax_to_np(M.P), cfg=vars(M.cfg), norm_stats=M.norm_stats)
        # fatigue strength at N_FS (E1), zero-shot and 5-shot
        for k in ([0, 5] if 5 in kk else [0]):
            sh = shots["test"][k] if k > 0 else None
            out["fs"][(name, k)] = dict(fs50=P.allowable(test, sh, np.log(N_FS), 0.5),
                                        fs10=P.allowable(test, sh, np.log(N_FS), 0.1))
        if name == "H-NRFL (ours)":
            out["preds"][(name, 0, "anom")] = P.predict(anom, None, ALPHA)
            # root-cause contrasts (E8): HIP effect on mu for machined vs as-built, per alloy
            out["rootcause"] = root_cause(M, test)
        log("  done", name, f"{time.time()-t0:.0f}s")

    # ---------------- baselines (pooled, zero-shot)
    Xtr, ltr, dtr, gtr = bl.specimen_table(train)
    bls = {}
    for mname in ("GBM", "RF", "MLP"):
        for strat in ("drop", "asfail", "flag"):
            bls[f"{mname} ({strat})"] = bl.GaussRegressor(mname, strat, seed)
    bls["ASTM E739 pooled (drop)"] = bl.GaussRegressor("E739", "drop", seed)
    bls["Deep AFT (censored log-normal MLP)"] = bl.CensoredLogNormalMLP(1, steps=C["aft_steps"], seed=seed)
    bls[f"Deep ensemble x{C['ens']} (censored)"] = bl.CensoredLogNormalMLP(C["ens"], steps=C["aft_steps"], seed=seed)
    bls["DeepSurv (Cox)"] = bl.DeepSurv(steps=C["cox_steps"], seed=seed)
    for name, mdl in bls.items():
        log("fit", name)
        mdl.fit(Xtr, ltr, dtr, gtr)
        P = ev.BaselinePredictor(mdl)
        for split, studies in (("cal", cal), ("test", test)):
            out["preds"][(name, 0, split)] = P.predict(studies, None, ALPHA)
        e_s, e_p, m_cal = calibrate_eta(out["preds"][(name, 0, "cal")], base["cal"], shots_flat["cal"][0], LC0,
                                        rng_cal)
        out["eta"][(name, 0)] = dict(split=e_s, pac=e_p, m=m_cal)
        for lab, eta in (("model", 0.0), ("split", e_s), ("pac", e_p)):
            if not np.isfinite(eta) or LND + eta > LC0:
                out["allow"][(name, 0, lab)] = np.zeros(len(test))
            else:
                out["allow"][(name, 0, lab)] = P.allowable(test, None, LND + eta, ALPHA)
        out["fs"][(name, 0)] = dict(fs50=P.allowable(test, None, np.log(N_FS), 0.5),
                                    fs10=P.allowable(test, None, np.log(N_FS), 0.1))
        if name in ("GBM (asfail)", "ASTM E739 pooled (drop)"):
            out["preds"][(name, 0, "anom")] = P.predict(anom, None, ALPHA)
            out["info"][name] = dict(scale=getattr(mdl, "scale", np.nan), n_fail=int((dtr > 0).sum()))
        log("  done", name, f"{time.time()-t0:.0f}s")

    # ---------------- classical per-study E739 with k coupons (E5)
    for k in ks:
        if k < 3:
            continue
        perm_masks = shots["test"][k]
        out["allow"][("Per-study E739 + tolerance bound", k, "classical")] = np.array(
            [e739_allowable(st, np.flatnonzero(perm_masks[i, :len(st["N"])] > 0)) for i, st in enumerate(test)])
    # true reliability of every allowable
    rel = {}
    for key, sig in out["allow"].items():
        rel[key] = np.array([np.exp(synth.true_logsf(st, s, LND)) if (np.isfinite(s) and s > 0) else np.nan
                             for st, s in zip(test, sig)])
    out["allow_rel"] = rel
    out["anom_meta"] = dict(C=np.array([st["C"] for st in anom]))
    out["runtime_s"] = time.time() - t0
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"syn_{config}_seed{seed}.pkl"), "wb") as f:
        pickle.dump(out, f)
    log("saved", f"{time.time()-t0:.0f}s")


def root_cause(M, studies):
    """Fitted effect of HIP on the fatigue-limit location mu, split by surface state and alloy."""
    import jax.numpy as jnp
    rows = []
    Zraw = np.stack([st["z"] for st in studies])
    names = synth.FEATURE_NAMES
    iH, iM = names.index("HIP"), names.index("machined")
    fm, fs = M.norm_stats
    for a_i, alloy in enumerate(synth.ALLOYS):
        sel = Zraw[:, a_i] == 1
        if sel.sum() == 0:
            continue
        for mach in (0, 1):
            mus = []
            for hip in (0, 1):
                Z = Zraw[sel].copy()
                Z[:, iH] = hip; Z[:, iM] = mach
                th = rfl.theta_raw(M.P, jnp.asarray((Z - fm) / fs), M.cfg)
                mus.append(np.asarray(rfl.map_theta(th)[3]))
            est = float(np.mean(mus[1] - mus[0]))
            true = synth.INC_HIP_M[alloy] if mach else synth.INC_HIP_AB
            rows.append(dict(alloy=alloy, surface="machined" if mach else "as-built", est=est, true=true))
        # machining effect (no HIP)
        mus = []
        for mach in (0, 1):
            Z = Zraw[sel].copy(); Z[:, iH] = 0; Z[:, iM] = mach
            th = rfl.theta_raw(M.P, jnp.asarray((Z - fm) / fs), M.cfg)
            mus.append(np.asarray(rfl.map_theta(th)[3]))
        rows.append(dict(alloy=alloy, surface="machining effect", est=float(np.mean(mus[1] - mus[0])),
                         true=synth.INC_MACH[alloy]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    a = ap.parse_args()
    main(a.config, a.seed, a.out)
