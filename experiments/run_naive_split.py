"""E4 supplement: 'naive point split' conformal, as done in most published ML-fatigue papers.

Specimens of the TRAINING studies are split at random into fit (75 %) and calibration (25 %) sets, so
calibration specimens come from studies the model has already seen. Coverage is then evaluated on
new, unseen studies. Same data and seeds as run_synthetic.py.
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
from experiments.config import ALPHA, C0, CONFIGS  # noqa: E402

LC0 = np.log(C0)


def main(config, seed, outdir):
    C = CONFIGS[config]
    t0 = time.time()
    ntr, nca, nte = C["n_train"], C["n_cal"], C["n_test"]
    allst = synth.generate(ntr + nca + nte, seed=1000 + seed)
    train, test = allst[:ntr], allst[ntr + nca:]
    rng = np.random.default_rng(seed + 321)
    fit_st, cal_st, cal_masks = [], [], []
    for st in train:
        n = len(st["N"])
        cal_idx = rng.random(n) < 0.25
        if cal_idx.all() or not cal_idx.any():
            cal_idx[:] = False; cal_idx[rng.integers(n)] = True
        cal_masks.append(cal_idx)
        for idx, dest in ((~cal_idx, fit_st), (cal_idx, cal_st)):
            s2 = dict(st)
            for k in ("sa", "R", "N", "d", "T_true"):
                s2[k] = np.asarray(st[k])[idx]
            dest.append(s2)
    out = {}
    cfg = rfl.RFLConfig(steps=C["steps"], batch_studies=C["batch"], n_gh_train=C["gh_train"], n_gh_pred=C["gh_pred"],
                        K=C["K"], seed=seed)
    M = rfl.fit(fit_st, cfg)
    models = {"H-NRFL (ours)": ev.RFLPredictor(M, chunk=50)}
    Xf, lf, df, gf = bl.specimen_table(fit_st)
    gbm = bl.GaussRegressor("GBM", "asfail", seed).fit(Xf, lf, df, gf)
    models["GBM (asfail)"] = ev.BaselinePredictor(gbm)
    ens = bl.CensoredLogNormalMLP(C["ens"], steps=C["aft_steps"], seed=seed).fit(Xf, lf, df, gf)
    models[f"Deep ensemble x{C['ens']} (censored)"] = ev.BaselinePredictor(ens)
    for name, P in models.items():
        # calibration specimens: held-out specimens of seen studies with C >= c0 (no grouping).
        # For the hierarchical model a "seen" study is predicted conditional on its own fit specimens
        # (they act as coupons), which is what a random point split gives such a model.
        if name == "H-NRFL (ours)":
            nmax = max(len(st["N"]) for st in train)
            shot = np.zeros((len(train), nmax))
            for i, cm in enumerate(cal_masks):
                shot[i, :len(cm)] = (~cm).astype(float)
            pfull = P.predict(train, shot, ALPHA)
            sel = np.concatenate(cal_masks)
            pc = {k: v[sel] for k, v in pfull.items()}
        else:
            pc = P.predict(cal_st, None, ALPHA)
        Xc, lc, dc, gc = bl.specimen_table(cal_st)
        Cc = np.concatenate([np.full(len(st["N"]), st["C"]) for st in cal_st])
        ok = Cc >= C0 * (1 - 1e-9)
        E = cf.scores(pc["lnQ"][ok], lc[ok], dc[ok], LC0)
        pt = P.predict(test, None, ALPHA)
        out[name] = dict(E_naive=E, lnQ_test=pt["lnQ"])
    out["lT_test"] = np.concatenate([np.log(st["T_true"]) for st in test])
    out["sid_test"] = np.concatenate([np.full(len(st["N"]), i) for i, st in enumerate(test)])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"naive_{config}_seed{seed}.pkl"), "wb") as f:
        pickle.dump(out, f)
    print(f"naive split seed {seed} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    a = ap.parse_args()
    main(a.config, a.seed, a.out)
