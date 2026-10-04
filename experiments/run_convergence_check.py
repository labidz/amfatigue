"""Sensitivity check: does the fast training budget (400 steps) change the conclusions?

Refits H-NRFL on seed-0 data with 400 and 1500 steps and compares zero-shot and 3-shot test NLL,
the learned Walker exponent and the random-effect covariance.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import evaluate as ev  # noqa: E402
from amf import rfl, synth  # noqa: E402
from experiments.config import CONFIGS  # noqa: E402


def main():
    C = CONFIGS["fast"]
    allst = synth.generate(C["n_train"] + C["n_cal"] + C["n_test"], seed=1000)
    train, test = allst[:C["n_train"]], allst[C["n_train"] + C["n_cal"]:]
    shot3 = ev.shot_masks(test, [3], 12)[3]
    rows = []
    for steps in (400, 1500):
        t0 = time.time()
        cfg = rfl.RFLConfig(steps=steps, batch_studies=C["batch"], n_gh_train=C["gh_train"], n_gh_pred=C["gh_pred"],
                            K=C["K"], seed=0)
        M = rfl.fit(train, cfg)
        lp0, mask = M.logpred(test)
        lp3, _ = M.logpred(test, shot3)
        ev3 = np.asarray(mask) * (1 - shot3[:, :mask.shape[1]])
        rc = M.re_cov
        rows.append(dict(steps=steps, nll_k0=float(-(lp0 * np.asarray(mask)).sum() / np.asarray(mask).sum()),
                         nll_k3=float(-(lp3 * ev3).sum() / ev3.sum()), walker=M.walker,
                         sd_beta0=float(np.sqrt(rc[0, 0])), sd_mu=float(np.sqrt(rc[1, 1])),
                         final_loss=M.hist[-1], minutes=(time.time() - t0) / 60))
        print(rows[-1], flush=True)
    import pandas as pd
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "tables",
                                           "convergence_check.csv"), index=False)


if __name__ == "__main__":
    main()
