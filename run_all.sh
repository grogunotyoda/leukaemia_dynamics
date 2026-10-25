#!/usr/bin/env bash
# Regenerates every simulation output in data/ with the seeds used in the manuscript,
# then the figures. Takes about three hours on two cores.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data figures results

# NPM1-AML cohorts: calibrated range and two robustness ranges
#   arguments: kill range, number of patients, niche capacity range, seed
python3 npm1_cohort.py 0.5 5 4000 1e-11 1e-7 11
python3 npm1_cohort.py 1 10 1500 1e-11 1e-6 12
python3 npm1_cohort.py 0.3 3 1500 1e-12 1e-7 13
for f in data/npm1_0.5_5.0_1e-11_1e-07.json data/npm1_1.0_10.0_1e-11_1e-06.json data/npm1_0.3_3.0_1e-12_1e-07.json; do
  python3 npm1_analyse.py "$f"
done
python3 npm1_second_input.py data/npm1_0.5_5.0_1e-11_1e-07.json

# NPM1-AML model-based estimator: correct and misspecified reference sets, then evaluation
python3 npm1_estimator.py build 30000 11 0.5 5 1e-11 1e-7 data/prior_correct.npz
python3 npm1_estimator.py build 30000 12 1 10 1e-10 1e-6 data/prior_misspecified.npz
python3 npm1_estimator.py eval data/npm1_0.5_5.0_1e-11_1e-07.json data/prior_correct.npz data/prior_misspecified.npz \
  | tee data/npm1_estimator.log

# CML cohorts under exact kinetics
#   arguments: niche capacity range, dose ratio in the final year, number of patients, seed
python3 cml_cohort.py 1e-8 1e-4 1.0 1500 2
python3 cml_cohort.py 1e-9 1e-5 1.0 1500 2
python3 cml_cohort.py 1e-10 1e-6 1.0 1500 2
for f in data/cml_1e-08_0.0001_1.0_2.json data/cml_1e-09_1e-05_1.0_2.json data/cml_1e-10_1e-06_1.0_2.json; do
  python3 cml_analyse.py "$f" 0.5
done

# CML de-escalation with qPCR noise and detection limit
python3 cml_deescalation.py 1e-8 1e-4 400 5
python3 cml_deescalation.py 1e-10 1e-6 400 6
for f in data/cml_deesc_1e-08_0.0001_5.json data/cml_deesc_1e-10_1e-06_6.json; do
  python3 cml_deescalation_analyse.py "$f"
done

python3 make_figures.py
