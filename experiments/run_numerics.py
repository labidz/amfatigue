"""Numerical sensitivity (reviewer items C1-C3) on seed-0 data.

Prediction side (saved model, no refit): Gauss-Hermite prediction grid 13x13 vs 25x25 and Gauss-Legendre
fatigue-limit nodes K = 12 vs 32 -> test NLL at k = 0 and k = 3, posterior effective sample size at k = 3,
and exact true coverage of the split-conformal lower bound.
Training side: refit with a 9x9 instead of 5x5 Gauss-Hermite training grid (same seed, 400 steps).
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
from amf import rfl  # noqa: E402
from experiments.analyze_synthetic import LC0, OURS, draw_eta, regen_test, true_cov, true_sf_table  # noqa: E402
from experiments.config import ALPHA, CONFIGS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def evaluate(M, run, test, tab, shots, label, rng):
    P = ev.RFLPredictor(M, chunk=20)
    cal = regen_cal()
    out = dict(setting=label)
    for k in (0, 3):
        pt = P.predict(test, shots["test"][k] if k else None, ALPHA)
        pc = P.predict(cal, shots["cal"][k] if k else None, ALPHA)
        evm = run["shots_flat"]["test"][k] == 0
        out[f"nll_k{k}"] = float(-np.mean(pt["lp"][evm]))
        etas = [cf.split_eta(draw_eta(pc, run["base"]["cal"], run["shots_flat"]["cal"][k], rng), ALPHA) for _ in range(30)]
        out[f"cov_k{k}"] = float(np.mean([true_cov(tab, cf.lpb(pt["lnQ"], e, LC0), run["base"]["test"]["sid"], evm)
                                         for e in etas]))
    D, mask, logw = M.posterior(test, shots["test"][3])
    w = np.exp(np.asarray(logw))
    out["posterior_ESS_k3_median"] = float(np.median(1.0 / (w ** 2).sum(1)))
    out["posterior_max_weight_k3_median"] = float(np.median(w.max(1)))
    return out


_CAL = {}


def regen_cal():
    if "cal" not in _CAL:
        from amf import synth
        C = CONFIGS["fast"]
        allst = synth.generate(C["n_train"] + C["n_cal"] + C["n_test"], seed=1000)
        _CAL["cal"] = allst[C["n_train"]:C["n_train"] + C["n_cal"]]
        _CAL["train"] = allst[:C["n_train"]]
    return _CAL["cal"]


def main():
    t0 = time.time()
    run = pickle.load(open(os.path.join(ROOT, "results", "raw", "syn_fast_seed0.pkl"), "rb"))
    test = regen_test("fast", 0); tab = true_sf_table(test, nq=1000)
    cal = regen_cal()
    shots = {"cal": ev.shot_masks(cal, [0, 1, 2, 3, 5, 8], 0 + 11), "test": ev.shot_masks(test, [0, 1, 2, 3, 5, 8], 0 + 12)}
    mp = run["model_params"][OURS]
    rows, rng = [], np.random.default_rng(5)
    for gh, K in ((13, 12), (25, 12), (13, 32), (25, 32)):
        cfgd = dict(mp["cfg"]); cfgd.update(n_gh_pred=gh, K=K)
        M = rfl.FittedRFL({k: jnp.asarray(v) for k, v in mp["P"].items()}, rfl.RFLConfig(**cfgd), mp["norm_stats"], [], 0)
        rows.append(evaluate(M, run, test, tab, shots, f"saved model; predict GH {gh}x{gh}, K={K}", rng))
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    for ght in (5, 9):
        cfgd = dict(mp["cfg"]); cfgd.update(n_gh_train=ght)
        M = rfl.fit(_CAL["train"], rfl.RFLConfig(**cfgd))
        r = evaluate(M, run, test, tab, shots, f"refit; train GH {ght}x{ght}", rng)
        r.update(walker=M.walker, sd_beta0=float(np.sqrt(M.re_cov[0, 0])), sd_mu=float(np.sqrt(M.re_cov[1, 1])))
        rows.append(r)
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "tables", "numerical_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
