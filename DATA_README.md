# Research data for "When is a data-driven fatigue allowable safe for a new additive manufacturing process?"

Authors: Rohul Amin, Sumya Aktar, Md. Mehedi Rahman Rana (BAUST, Khulna; RUET, Rajshahi, Bangladesh).
Corresponding author: rohul@baust.edu.bd

Suggested Mendeley Data entry
- Title: Code and data for: When is a data-driven fatigue allowable safe for a new additive manufacturing process?
- Description: Python (JAX) code for the censored hierarchical random-fatigue-limit model (H-RFL), study-level
  conformal calibration and the semi-synthetic multi-study AM fatigue benchmark; all result tables and figures of
  the paper; the real censored fatigue datasets used (from the SMRD R package and the pyLife library); and a
  transcription of Table 3 of Andreau et al. (2021, Int. J. Fatigue 143, 105930).
- Licence: CC BY 4.0 for the authors' data, tables and figures; MIT for the authors' code (choose in Mendeley
  Data). Third-party data in data/raw/ keep their original licences (SMRD R package; pyLife, Apache-2.0).

Contents
- amf/, experiments/, run_all.py, requirements.txt: code; README.md explains how to reproduce every number.
- results/tables/: every table behind the paper (CSV); results/raw/: saved predictions of the fast runs (pickles).
- results/logs/: run logs. figures/: the paper's figures (PNG, 300 dpi).
- data/raw/: real datasets and the Andreau et al. (2021) transcription.
- SHA256SUMS.txt: checksums of every file.

Naming: the paper's H-RFL is labelled "H-NRFL (ours)" in code and tables.
