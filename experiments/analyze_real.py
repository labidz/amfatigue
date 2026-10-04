"""Stage 2 (real data): summary tables for R1 (single-study datasets) and R2 (multi-lot HCF)."""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main(config):
    raw = pickle.load(open(os.path.join(ROOT, "results", "raw", f"real_{config}.pkl"), "rb"))
    tab = os.path.join(ROOT, "results", "tables"); os.makedirs(tab, exist_ok=True)
    r1 = raw["r1"]
    g = r1.groupby(["dataset", "model"])
    s1 = g.agg(n=("n", "first"), runout_frac=("runout_frac", "first"), cv_nll=("cv_nll", "mean"),
               cv_nll_sd=("cv_nll", "std"), raw_q10_cov=("raw_cov", "mean"),
               conformal_cov=("conformal_cov", "mean")).reset_index()
    s1.to_csv(os.path.join(tab, f"real_{config}_r1_summary.csv"), index=False)
    # rank of the censored RFL among strategies, per dataset
    fs = raw["fs"].copy()
    prop = fs[fs.strategy == "proper"].set_index("dataset")
    fs["fs50_shift_pct"] = [100 * (r.fs50 / prop.loc[r.dataset, "fs50"] - 1) for r in fs.itertuples()]
    fs["fs10_shift_pct"] = [100 * (r.fs10 / prop.loc[r.dataset, "fs10"] - 1) for r in fs.itertuples()]
    fs.to_csv(os.path.join(tab, f"real_{config}_fatigue_strength.csv"), index=False)
    r2 = raw["r2"]
    s2 = r2.groupby(["model", "k"]).agg(mean_nll=("nll", "mean"), sd_nll=("nll", "std"), lots=("nll", "size"),
                                        walker=("walker", "mean")).reset_index()
    r2.to_csv(os.path.join(tab, f"real_{config}_r2_perlot.csv"), index=False)
    s2.to_csv(os.path.join(tab, f"real_{config}_r2_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s1.to_string(index=False)); print(fs.to_string(index=False)); print(s2.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    main(ap.parse_args().config)
