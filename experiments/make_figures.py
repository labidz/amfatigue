"""Paper figures (PNG, 300 dpi) from the analysed tables. Palette: reference categorical order, fixed."""
from __future__ import annotations

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OURS = "H-NRFL (ours)"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "lines.linewidth": 2})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def agg_mean_sd(df, keys, col):
    g = df.groupby(keys)[col]
    return g.mean().rename("m").to_frame().join(g.std().rename("sd")).reset_index()


LBL = {OURS: "H-RFL (ours)", "GBM (asfail)": "GBM", "Pascual-Meeker RFL, pooled (linear, no RE)": "Pascual-Meeker, pooled",
       "w/o fatigue limit (Basquin)": "Basquin variant"}


def lbl(m):
    if m.startswith("Deep ensemble"):
        return "Deep ensemble ×3"
    return LBL.get(m, m)


def fig_runout_bias(A):
    fs = A["fs"]; fs = fs[fs.quantity == "fs10"]
    a = fs.groupby(["model", "k"]).bias_pct.agg(["mean", "std"])
    ens = [m for m in fs.model.unique() if m.startswith("Deep ensemble")][0]
    rows = [  # (label, model, k, strategy)
        ("H-RFL (ours)", OURS, 0, "censored"), ("H-RFL, linear covariate map", "linear g (no MLP)", 0, "censored"),
        ("H-RFL, 5 coupons", OURS, 5, "censored"),
        ("Pascual-Meeker, pooled", "Pascual-Meeker RFL, pooled (linear, no RE)", 0, "censored"),
        ("DeepSurv (Cox)", "DeepSurv (Cox)", 0, "censored"), ("Deep AFT", "Deep AFT (censored log-normal MLP)", 0, "censored"),
        ("Deep ensemble ×3", ens, 0, "censored"),
        ("H-RFL", "H-NRFL, run-outs dropped", 0, "dropped"), ("MLP regressor", "MLP (drop)", 0, "dropped"),
        ("Pooled linear regression", "ASTM E739 pooled (drop)", 0, "dropped"), ("GBM / random forest", "GBM (drop)", 0, "dropped"),
        ("H-RFL", "H-NRFL, run-outs as failures", 0, "as failures"), ("MLP regressor", "MLP (asfail)", 0, "as failures"),
        ("GBM / random forest", "GBM (asfail)", 0, "as failures"),
        ("MLP regressor", "MLP (flag)", 0, "flag feature"), ("GBM / random forest", "GBM (flag)", 0, "flag feature")]
    col = {"censored": S[0], "dropped": S[1], "as failures": S[2], "flag feature": S[3]}
    head = {"censored": "Run-outs censored", "dropped": "Run-outs dropped", "as failures": "Run-outs coded as failures",
            "flag feature": "Run-out flag as a feature"}
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y, ticks, labels, prev = 0, [], [], None
    for lab, m, k, s in rows:
        if s != prev:
            if prev is not None:
                y += 0.6
            ax.text(-101, y - 0.05, head[s], fontsize=7.6, fontweight="bold", color=col[s], va="center", ha="left")
            y += 0.85; prev = s
        mu, sd = a.loc[(m, k), "mean"], a.loc[(m, k), "std"]
        trees = mu < -90
        ax.barh(y, mu, 0.62, color=col[s], alpha=0.35 if trees else 1.0, hatch="///" if trees else None,
                edgecolor=col[s], lw=0.6)
        if not trees:
            ax.errorbar(mu, y, xerr=sd, color=INK2, lw=0.9, capsize=2)
        txt = "no stress certified (−99 %)" if trees else f"{mu:.1f} %".replace("-", "−")
        ax.text(-60 if trees else mu - sd - 2.0, y, txt, va="center", ha="center" if trees else "right",
                fontsize=7, color=INK)
        ticks.append(y); labels.append(lab); y += 1
    ax.set_yticks(ticks, labels, fontsize=7.6); ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(-101, 3)
    ax.set_xlabel("Bias of the stress at 90 % survival, 10$^7$ cycles, zero coupons unless stated (%)\n"
                  "negative = strength under-estimated (conservative, over-designed)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_runout_bias.png")


def fig_few_shot(A):
    few = A["few"]
    ours = few[few.model == OURS]; reo = few[few.model == "w/o covariates (RE only)"]
    cl = few[few.model.str.startswith("Per-study")]
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.8))
    ax = axs[0]
    for df, lab, c in ((ours, "H-RFL (ours)", S[0]), (reo, "H-RFL without covariates", S[1])):
        a = agg_mean_sd(df, ["k"], "nll")
        ax.errorbar(a.k, a.m, yerr=a.sd, color=c, marker="o", ms=4, label=lab, capsize=2)
    ax.set_xlabel("coupons from the new study (k)"); ax.set_ylabel("predictive NLL (nats per specimen)")
    ax.set_ylim(0.6, 2.2)
    ax.legend(fontsize=6.8, loc="upper right"); ax.set_title("(a) Updating with coupons", loc="left", fontsize=8.5)
    ax = axs[1]
    series = (("pac_mean_rel", "H-RFL, PAC offset", S[0]), ("local_pac_mean_rel", "H-RFL, PAC, local", S[6]),
              ("model_mean_rel", "H-RFL, no conformal step", S[3]))
    for col_, lab, c in series:
        a = agg_mean_sd(ours, ["k"], col_)
        ax.errorbar(a.k, a.m, yerr=a.sd, color=c, marker="o", ms=4, label=lab, capsize=2)
    if len(cl):
        a = agg_mean_sd(cl, ["k"], "classical_mean_rel")
        ax.errorbar(a.k, a.m, yerr=a.sd, color=S[1], marker="s", ms=4, label="per-study E739-form bound*", capsize=2)
    ax.axhline(0.9, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(0.78, 1.04); ax.set_xlabel("coupons (k)"); ax.set_ylabel("mean true reliability of the allowable")
    ax.legend(fontsize=6.2, loc="lower right", ncol=1)
    ax.set_title("(b) Reliability (target 0.90)", loc="left", fontsize=8.5)
    ax = axs[2]
    for col_, lab, c in (("pac_median_eff", "H-RFL, PAC offset", S[0]), ("local_pac_median_eff", "H-RFL, PAC, local", S[6])):
        a = agg_mean_sd(ours, ["k"], col_)
        ax.errorbar(a.k, a.m, yerr=a.sd, color=c, marker="o", ms=4, label=lab, capsize=2)
    if len(cl):
        a = agg_mean_sd(cl, ["k"], "classical_median_eff")
        ax.errorbar(a.k, a.m, yerr=a.sd, color=S[1], marker="s", ms=4, label="per-study E739-form bound*", capsize=2)
        c2 = agg_mean_sd(cl, ["k"], "computable")
        txt = " / ".join(f"{100 * v:.1f} %" for v in c2.m) + " of studies\nat k = " + \
            " / ".join(str(int(k)) for k in c2.k)
        ax.text(0.97, 0.04, "*computable for\n" + txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.2, color=INK2)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(-0.05, 1.12); ax.set_xlim(-0.5, 8.6); ax.set_xlabel("coupons (k)")
    ax.set_ylabel("allowable / oracle allowable (median)")
    ax.legend(fontsize=6.2, loc="lower left", bbox_to_anchor=(0.0, 0.30))
    ax.set_title("(c) Efficiency", loc="left", fontsize=8.5)
    for ax in axs:
        ax.set_xticks(sorted(ours.k.unique()))
    fig.tight_layout()
    save(fig, "fig4_coupons_allowables.png")


def fig_conformal(A, raw_pac_draws=None):
    sh = A["shift"]
    ens = [m for m in sh.model.unique() if m.startswith("Deep ensemble")][0]
    models = [OURS, "GBM (asfail)", ens]
    fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.9), gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.05]))
    ax = axs[0]
    for j, (cal, lab, c, mk) in enumerate((("Naive point split (seen studies)", "point split", S[1], "s"),
                                           ("Study split, one per study (exact)", "study-level", S[0], "o"))):
        v = sh[sh.calibration == cal].groupby("model").coverage.agg(["mean", "std"]).reindex(models)
        y = np.arange(len(models)) + (j - 0.5) * 0.3
        ax.errorbar(v["mean"], y, xerr=v["std"], fmt=mk, color=c, ms=5, capsize=2, lw=1, label=lab)
    ax.axvline(0.9, color=INK, lw=0.8, ls="--")
    ax.set_yticks(range(len(models)), [lbl(m) for m in models], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(0.76, 0.95); ax.set_ylim(2.6, -0.6)
    ax.set_xlabel("coverage on new studies (target 0.90)")
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    ax.set_title("(a) Calibration split", loc="left", fontsize=8.5)
    ax = axs[1]
    cond = A["cond_fresh"]
    v = cond.groupby("group").coverage.agg(["mean", "std"])
    order = [("Walker-equivalent stress Q1 (lowest)", "Q1 (low)", S[0]), ("Walker-equivalent stress Q2", "Q2", S[0]),
             ("Walker-equivalent stress Q3", "Q3", S[0]), ("Walker-equivalent stress Q4 (highest)", "Q4 (high)", S[0]),
             ("alloy Ti64", "Ti-6Al-4V", S[3]), ("alloy AlSi10Mg", "AlSi10Mg", S[3]), ("alloy 316L", "316L", S[3]),
             ("alloy IN718", "IN718", S[3]), ("surface as-built", "as-built", S[2]), ("surface machined", "machined", S[2])]
    order = [o for o in order if o[0] in v.index]
    ax.bar(range(len(order)), v.loc[[o[0] for o in order], "mean"], 0.62, yerr=v.loc[[o[0] for o in order], "std"],
           color=[o[2] for o in order], error_kw=dict(lw=1, capsize=2))
    ax.axhline(0.9, color=INK, lw=0.8, ls="--")
    ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=6.6, rotation=90)
    for x0, x1, gl in ((0, 3, "stress quartile"), (4, 7, "alloy"), (8, 9, "surface")):
        ax.text((x0 + x1) / 2, 0.993, gl, ha="center", va="top", fontsize=6.6, color=INK2)
    ax.set_ylim(0.80, 1.0); ax.set_ylabel("coverage, 1,000 fresh studies")
    ax.set_title("(b) Coverage by stress, alloy, surface", loc="left", fontsize=8.5)
    ax = axs[2]
    pac = A["pac"]
    mods = [(OURS, "H-RFL (ours)"), ("Pascual-Meeker RFL, pooled (linear, no RE)", "Pascual-Meeker"),
            ("GBM (asfail)", "GBM"), (ens, "Deep ensemble ×3")]
    for j, (me, lab, c) in enumerate((("Split conformal (1-α)", "split conformal", S[2]),
                                      ("PAC conformal (B-basis 90/95)", "PAC order statistic", S[0]))):
        vals = [pac[(pac.model == m) & (pac.method == me)].p_cov_ge_90.mean() for m, _ in mods]
        y = np.arange(len(mods)) + (j - 0.5) * 0.36
        ax.barh(y, vals, 0.34, color=c, label=lab)
        for yy, v_ in zip(y, vals):
            ax.text(v_ - 0.02, yy, f"{v_:.2f}", va="center", ha="right", fontsize=6.6, color="white", fontweight="bold")
    ax.axvline(0.95, color=INK, lw=0.8, ls="--")
    ax.set_yticks(range(len(mods)), [m[1] for m in mods], fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel("P(coverage ≥ 0.90) over 200 draws\n(same 200-study test sets; target 0.95)")
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.4, -0.33), ncol=2)
    ax.set_title("(c) Confidence over calibration draws", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig3_conformal_calibration.png")


def fig_forensic_rootcause(A, andreau, mfdr=None):
    fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))
    ax = axs[0]
    fo = A["for"].groupby(["model", "test"])[["power", "fdr"]].mean().reset_index()
    fo = fo[fo.test.str.contains("marginal") | fo.test.str.contains("Simes")]
    rows = []
    for m, lab in ((OURS, "H-RFL"), ("GBM (asfail)", "GBM"), ("ASTM E739 pooled (drop)", "Pooled linear")):
        for tst, tl in (("marginal", "conformal, fixed set" if m == OURS else "conformal"), ("Simes", "uncalibrated")):
            r = fo[(fo.model == m) & fo.test.str.contains(tst)]
            if len(r):
                rows.append((f"{lab}, {tl}", float(r.fdr.iloc[0]), float(r.power.iloc[0])))
    if mfdr is not None:
        rows.insert(0, ("H-RFL, conformal,\nfresh studies", float(mfdr.fdr.mean()), float(mfdr.power.mean())))
    y = np.arange(len(rows))
    ax.scatter([r[2] for r in rows], y, color=S[0], s=26, label="power", zorder=3)
    ax.scatter([r[1] for r in rows], y, color=S[1], marker="D", s=22, label="false discovery rate", zorder=3)
    for yy, r in zip(y, rows):
        ax.text(1.38, yy, f"{r[1]:.2f} / {r[2]:.2f}", fontsize=6.4, color=INK, ha="right", va="center")
    ax.text(1.38, -0.75, "FDR / power", fontsize=6.4, color=INK2, ha="right", va="center")
    ax.axvline(0.10, color=INK, lw=0.8, ls="--")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.8); ax.invert_yaxis(); ax.set_xlim(-0.03, 1.4)
    ax.set_ylim(len(rows) - 0.5, -1.1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("rate (screen at q = 0.10)")
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.35, -0.22), ncol=2)
    ax.set_title("(a) Screening (synthetic)", loc="left", fontsize=8.5)
    ax = axs[1]
    rc = A["rc_cf"]
    a = rc.groupby(["alloy", "effect"])[["est", "true"]].mean().reset_index()
    for eff, c, mk in (("HIP effect, machined", S[0], "o"), ("HIP effect, as-built", S[1], "s"),
                       ("machining effect (no HIP)", S[2], "^")):
        b = a[a.effect == eff]
        ax.scatter(b.true, b.est, color=c, marker=mk, s=26, edgecolor="white", linewidth=0.8, zorder=3,
                   label={"HIP effect, machined": "HIP, machined", "HIP effect, as-built": "HIP, as-built",
                          "machining effect (no HIP)": "machining, no HIP"}[eff])
    lim = [0.0, 0.62]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true effect on ln(strength at 10$^7$ cycles)"); ax.set_ylabel("estimated effect")
    ax.legend(fontsize=5.9, loc="upper left", handletextpad=0.1, borderaxespad=0.2)
    ax.set_title("(b) Process effects (synthetic)", loc="left", fontsize=8.5)
    ax = axs[2]
    d = andreau.copy()
    rng = np.random.default_rng(3)
    groups = (("natural", S[0], "o", "natural defect"), ("micro", S[6], "v", "no visible defect"),
              ("artificial", S[1], "s", "artificial defect"))
    cls = np.where(~d.natural_origin, "artificial", np.where(d.sqrt_area_um.isna(), "micro", "natural"))
    for g, c, mk, lab in groups:
        b = d[cls == g]
        x = b.sqrt_area_um.fillna(8.0).values * np.exp(rng.normal(0, 0.06, len(b)))
        yv = b.p_value.values + rng.normal(0, 0.008, len(b))
        ax.scatter(x, yv, color=c, marker=mk, s=20, label=lab, edgecolor="white", linewidth=0.6, zorder=3, alpha=0.9)
    ax.set_xscale("log"); ax.set_xlim(5, 1500); ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel("√area of crack-origin defect (µm)\n(no measurable origin plotted at 8 µm; jittered)")
    ax.set_ylabel("rank p-value of step-test strength"); ax.legend(fontsize=6.0, loc="upper right")
    ax.set_title("(c) Real L-PBF 316L (Andreau et al.)", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig5_forensic_rootcause.png")


def fig_real(r1):
    keep = [("RFL, censored (proper)", "RFL, run-outs censored"), ("RFL, run-outs dropped", "RFL, run-outs dropped"),
            ("RFL, run-outs as failures", "RFL, run-outs as failures"), ("GBM (asfail)", "GBM, run-outs as failures")]
    cols = [S[0], S[1], S[2], S[3]]
    dsets = list(dict.fromkeys(r1.dataset))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for j, ((m, lab), c) in enumerate(zip(keep, cols)):
        b = r1[r1.model == m].set_index("dataset").reindex(dsets)
        y = np.arange(len(dsets)) + (j - 1.5) * 0.16
        ax.errorbar(b.cv_nll, y, xerr=b.cv_nll_sd, fmt="o", color=c, ms=5, label=lab, capsize=2, lw=1)
    ax.set_yticks(range(len(dsets)), [s.replace("pylife demo", "pyLife example data") for s in dsets], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold cross-validated censored NLL (nats per specimen; lower is better), mean ± sd of 3 repeats")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    save(fig, "fig6_real_data.png")


def fig_workflow():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 840); ax.set_ylim(300, 20); ax.axis("off")
    W, H, y1, y2 = 244, 90, 40, 190
    cx = {1: 140, 2: 420, 3: 700}
    boxes = [(1, y1, "Literature S-N studies", ["failures and run-outs", "Type I censored", "process descriptors z"], False),
             (2, y1, "H-RFL model", ["censored random fatigue limit", "study random effects", "covariate map g(z)"], False),
             (3, y1, "New process: k coupons", ["quadrature-grid posterior", "no retraining", "k = 0: database only"], False),
             (2, y2, "Calibration studies", ["one specimen per study", "scores capped at c$_0$ = 10$^7$",
                                              "offset η and p-values"], False),
             (1, y2, "Screening test", ["study-level p-values", "BH flags at q = 0.10", "leads to root-cause analysis"], False),
             (3, y2, "Screening allowable", ["PAC offset; reliability checked", "stress at 2×10$^6$ cycles",
                                              "corrective action"], True)]
    for c, y0, name, lines, main in boxes:
        ax.add_patch(FancyBboxPatch((cx[c] - W / 2, y0), W, H, boxstyle="round,pad=0,rounding_size=8",
                                    fc=(S[0] if main else "white"), alpha=1, ec=S[0] if main else INK2,
                                    lw=1.6 if main else 1.0, zorder=1))
        if main:
            ax.add_patch(FancyBboxPatch((cx[c] - W / 2, y0), W, H, boxstyle="round,pad=0,rounding_size=8",
                                        fc="white", alpha=0.85, ec="none", zorder=1.5))
        ax.text(cx[c], y0 + 22, name, ha="center", va="center", fontsize=7.8, fontweight="bold", color=INK, zorder=2)
        for i, ln in enumerate(lines):
            ax.text(cx[c], y0 + 44 + 15 * i, ln, ha="center", va="center", fontsize=6.8, color=INK2, zorder=2)
    kw = dict(arrowstyle="-|>", color=INK2, lw=1.0, mutation_scale=9)
    ym1, ym2 = y1 + H / 2, y2 + H / 2
    for a, b in (((cx[1] + W / 2, ym1), (cx[2] - W / 2, ym1)), ((cx[2] + W / 2, ym1), (cx[3] - W / 2, ym1)),
                 ((cx[2], y1 + H), (cx[2], y2)), ((cx[3], y1 + H), (cx[3], y2)),
                 ((cx[2] + W / 2, ym2), (cx[3] - W / 2, ym2)), ((cx[2] - W / 2, ym2), (cx[1] + W / 2, ym2))):
        ax.annotate("", xy=b, xytext=a, arrowprops=kw)
    save(fig, "fig1_workflow.png")


def main(config):
    A = pickle.load(open(os.path.join(ROOT, "results", "tables", f"syn_{config}_all.pkl"), "rb"))
    andreau = pd.read_csv(os.path.join(ROOT, "results", "tables", "andreau_forensic_specimens.csv"))
    r1 = pd.read_csv(os.path.join(ROOT, "results", "tables", f"real_{config}_r1_summary.csv"))
    mf = os.path.join(ROOT, "results", "tables", "review_marginal_fdr.csv")
    mfdr = pd.read_csv(mf) if os.path.exists(mf) else None
    fig_workflow(); fig_runout_bias(A); fig_few_shot(A); fig_conformal(A); fig_forensic_rootcause(A, andreau, mfdr); fig_real(r1)
    print("figures written to", FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    main(ap.parse_args().config)
