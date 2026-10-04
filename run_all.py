"""Run every experiment in order. Usage:

    python run_all.py --config fast                       # what was run for the draft
    python run_all.py --config full                       # paper configuration (hours; GPU not needed)
    python run_all.py --config fast --am2022 path/to/FatigueData-AM2022.json   # adds the real-AM suite (E9)
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fast")
    ap.add_argument("--am2022", default=None, help="FatigueData-AM2022 JSON (preferred) or Excel file")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    from experiments.config import CONFIGS
    for s in CONFIGS[a.config]["seeds"]:
        run("experiments/run_synthetic.py", "--config", a.config, "--seed", str(s))
        run("experiments/run_naive_split.py", "--config", a.config, "--seed", str(s))
    run("experiments/run_real.py", "--config", a.config)
    run("experiments/run_andreau.py")
    run("experiments/run_convergence_check.py")
    for s in CONFIGS[a.config]["seeds"][:2]:  # A2 stress test (model misspecification), 2 seeds
        run("experiments/run_misspec.py", "--config", a.config, "--seed", str(s))
    if a.config == "fast":  # checks that reuse the saved seed-0..2 fast runs
        run("experiments/run_numerics.py")  # quadrature sensitivity (~50 min on CPU)
        run("experiments/run_review_checks.py")  # init values, monotonicity, fresh-study FDR, allowable confidence
    if a.am2022:
        run("experiments/run_am2022.py", "--data", a.am2022, "--config", a.config)
    run("experiments/analyze_synthetic.py", "--config", a.config)
    run("experiments/analyze_real.py", "--config", a.config)
    run("experiments/make_figures.py", "--config", a.config)


if __name__ == "__main__":
    main()
