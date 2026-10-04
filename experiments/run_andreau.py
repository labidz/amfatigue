"""E10-lite: forensic screening against fractography on real AM data (Andreau et al., Int. J. Fatigue 2021).

Step tests (20 MPa steps of 1e6 cycles, R = 0.1, L-PBF 316L, machined + polished). For each specimen the
fatigue strength at 1e6 cycles is estimated with the Maxwell-Nicholas step-test interpolation
    sigma_f = sigma_prev + (sigma_end - sigma_prev) * n_last / N_step.
Reference population ('reported process, no anomaly') = specimens whose fractographic origin is natural
(microstructure, natural surface defect, non-observable). Conformal p-values for 'strength lower than the
reference population' are computed for every specimen (leave-one-out for reference specimens), and
Benjamini-Hochberg at q = 0.10 flags specimens. The check: do flags and p-values track the artificial
(deterministic) defects and their size, which the test itself never sees?
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amf import conformal as cf  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "raw", "andreau2021_table3.csv"), comment="#")
    step = 20.0
    nstep = np.where(df.note.fillna("").str.contains("1e5"), 1e5, 1e6)
    n_last = df.cycles_last_step_x1e5.values * 1e5
    sig_end = df.end_step_MPa.values.astype(float)
    sig_prev = sig_end - step
    df["sigma_f_max"] = sig_prev + step * np.clip(n_last / nstep, 0, 1)
    df["sigma_f_amp"] = df.sigma_f_max * (1 - 0.1) / 2
    df["one_step_only"] = df.number_of_steps == 1  # no surviving previous step: strength is extrapolated
    df["natural_origin"] = df.critical_defect != "Deterministic"
    score = -np.log(df.sigma_f_max.values)  # low strength -> large score
    ref = df.natural_origin.values
    p = np.empty(len(df))
    for i in range(len(df)):
        cal = ref.copy(); cal[i] = False  # leave-one-out for reference specimens
        p[i] = cf.pvalues(score[cal], [score[i]])[0]
    df["p_value"] = p
    df["flag_BH_q10"] = cf.bh(p, 0.10)
    sq = df.sqrt_area_um.values.astype(float)
    has = np.isfinite(sq)
    rho, rho_p = stats.spearmanr(sq[has], p[has])
    rho_s, rho_s_p = stats.spearmanr(sq[has], df.sigma_f_max.values[has])
    summ = df.groupby("critical_defect").agg(n=("sample", "size"), mean_sigma_f_max=("sigma_f_max", "mean"),
                                              median_p=("p_value", "median"), flagged=("flag_BH_q10", "sum"))
    det = df[~df.natural_origin]
    size_grp = det.assign(series=det["sample"].str.extract(r"^([A-Z]\d+)")[0]).groupby("series").agg(
        n=("sample", "size"), sqrt_area=("sqrt_area_um", "mean"), mean_sigma_f_max=("sigma_f_max", "mean"),
        median_p=("p_value", "median"), flagged=("flag_BH_q10", "sum"))
    tabdir = os.path.join(ROOT, "results", "tables"); os.makedirs(tabdir, exist_ok=True)
    df.to_csv(os.path.join(tabdir, "andreau_forensic_specimens.csv"), index=False)
    summ.to_csv(os.path.join(tabdir, "andreau_forensic_by_origin.csv"))
    size_grp.to_csv(os.path.join(tabdir, "andreau_forensic_by_series.csv"))
    with open(os.path.join(tabdir, "andreau_forensic_summary.txt"), "w") as f:
        f.write(f"Spearman(sqrt_area, p-value) = {rho:.3f} (p = {rho_p:.2g}, n = {has.sum()})\n")
        f.write(f"Spearman(sqrt_area, sigma_f) = {rho_s:.3f} (p = {rho_s_p:.2g})\n")
        f.write(f"Flagged at BH q=0.10: {int(df.flag_BH_q10.sum())} of {len(df)}; natural-origin flagged: "
                f"{int(df.flag_BH_q10[df.natural_origin].sum())} of {int(df.natural_origin.sum())}\n")
    print(summ.to_string()); print(size_grp.to_string())
    print(open(os.path.join(tabdir, "andreau_forensic_summary.txt")).read())


if __name__ == "__main__":
    main()
