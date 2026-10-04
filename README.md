# When is a data-driven fatigue allowable safe for a new AM process? Code and data

Code and data for the manuscript *"When is a data-driven fatigue allowable safe for a new additive manufacturing process? Run-out censoring, study-level calibration and the number of qualification coupons"* (Amin, Aktar and Rana).

Naming: the model called **H-RFL** in the paper is labelled `H-NRFL (ours)` in the code and result tables (the label predates the decision to recommend the linear covariate map).

## What is here

| Path | Contents |
|---|---|
| `amf/rfl.py` | H-NRFL: censored hierarchical random-fatigue-limit model in JAX — censored likelihood, study random effects (Gauss-Hermite), covariate map (MLP or linear; linear is the recommended default), Walker mean-stress exponent, few-shot posterior on the quadrature grid, quantiles and allowables |
| `amf/conformal.py` | Study-level conformal lower bounds under Type I censoring, PAC (Wilks) offsets for population-level 90/95 allowables (a B-basis analogue, not an MMPDS B-basis value), conformal p-values, Simes, Benjamini-Hochberg |
| `amf/baselines.py` | GBM / RF / MLP with three run-out strategies, ASTM E739, censored log-normal deep AFT + ensembles, DeepSurv |
| `amf/synth.py` | Semi-synthetic multi-study AM benchmark (SynAM) with exact ground-truth functionals |
| `amf/am2022.py` | Loader for FatigueData-AM2022 (JSON preferred; Excel fallback) |
| `experiments/` | One script per experiment, analysis and figures; `config.py` holds the fast and full configurations |
| `data/raw/` | SMRD and pylife datasets (public), `andreau2021_table3.csv` (transcribed twice; provenance in the file header) |
| `results/tables/` | All result tables (CSV); `results/raw/` raw predictions (pickles) |
| `figures/` | Paper figures (PNG, 300 dpi) |
| `tests/make_mock_am2022.py` | Builds a mock FatigueData-AM2022 JSON to test the loader |

## Run

```bash
pip install -r requirements.txt
python run_all.py --config fast                                   # about 4 h on 2 CPU cores (incl. stress test and numerics)
python run_all.py --config full                                   # paper configuration
python run_all.py --config fast --am2022 FatigueData-AM2022.json  # adds the real-AM suite (E9)
```

FatigueData-AM2022 (Zhang & Xu, *Sci. Data* 10:249, 2023, CC BY 4.0) is at
https://doi.org/10.6084/m9.figshare.22337629. Download the JSON (or Excel) file and pass it with `--am2022`.

## Configurations

| | fast (draft) | full (paper) |
|---|---|---|
| Studies train / cal / test | 300 / 200 / 200 | 600 / 450 / 450 |
| Seeds | 3 | 10 |
| Training steps | 400 | 2,000 |
| Quadrature (train GH / predict GH / fatigue limit GL) | 5×5 / 13×13 / 12 | 9×9 / 25×25 / 24 |
| Conformal repetitions / PAC draws | 50 / 200 | 200 / 1,000 |

`experiments/run_convergence_check.py` compares 400 with 1,500 training steps (test NLL 1.200 vs 1.197).
`experiments/run_numerics.py` checks the quadrature (`results/tables/numerical_sensitivity.csv`): the prediction grid and
fatigue-limit nodes change NLL by at most 0.001, but a 9×9 training grid lowers the zero-shot NLL from 1.200 to 1.144,
so the full configuration trains with 9×9.
`experiments/run_review_checks.py` adds the checks reported after the internal review: initial values, monotonicity, marginal FDR with fresh studies, and the confidence of the allowable over calibration draws (`results/tables/review_*.csv`).
`experiments/run_misspec.py` is the model-misspecification stress test (`results/tables/misspec_fast_seed{0,1}.csv`).

## Data provenance

- `data/raw/smrd/*.RData`: datasets from the SMRD R package (Meeker & Escobar), cloned from GitHub.
- `data/raw/fatigue-data-*.csv`: pylife demo data (Bosch Research), cloned from GitHub.
- `data/raw/andreau2021_table3.csv`: Table 3 of Andreau et al., *Int. J. Fatigue* (2021), open-access manuscript.

## Known limitations

- The real-AM multi-study experiment (E9) has not been run: the database host was unreachable from the
  environment used to produce the draft.
- Coverage guarantees are marginal over a new study's specimens and stresses; conditional coverage by stress
  level and alloy is reported separately. About 21–24 % of individual studies fall below 0.90 reliability.
- The network covariate map brought no gain over a linear map; the neural variant is not claimed.
- Forensic FDR (E7) varied from 0.03 to 0.27 across the 3 fast seeds; run the full 10 seeds before claiming control.
- The fast configuration trains with a 5×5 grid, which understates the database-only fit by about 0.06 nats.
