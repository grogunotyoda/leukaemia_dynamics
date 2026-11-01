# Leukaemia dynamics: identifying drug-free dynamics from response data

Code and simulation outputs for the manuscript *Identifying the drug-free dynamics of residual
leukaemia from response data* by D. Koonthalakadu Baby and R. Geetha Nivas, submitted to the
Journal of the Royal Society Interface.

The model has two leukaemic compartments, cycling (A) and protected (Q), with treatment acting
as a kill on A and the measured marker y = A + Q censored at the assay limit. The scripts
simulate virtual cohorts for NPM1-mutated AML (pulsed chemotherapy) and CML (continuous TKI),
evaluate how well different response measurements predict relapse, and check the identification
result for models with more compartments. No patient data are used.

The reference sets (`data/*.npz`) are not tracked because of their size; `run_all.sh` rebuilds them.

## Requirements

Python 3.11 or later with the packages in `requirements.txt`:

    pip install -r requirements.txt

The published outputs were checked with Python 3.13, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2
and scikit-learn 1.9.1: `make_figures.py` reproduces `results/figure_numbers.json` exactly.

## Reproducing the results

The simulation outputs used in the manuscript are in `data/`. To redraw the figures from them:

    python3 make_figures.py

This writes `figures/*.pdf` and `results/figure_numbers.json`, which holds every number quoted
from the figures. Numbers quoted in the text and tables that are not in a figure are in the logs in
`data/`: the calibration statistics, fate shares and AUCs of the NPM1 cohorts in
`data/analyse_npm1_*.log`, the second input level in `data/npm1_second_input.log`, the CML results in
`data/analyse_cml_*.log`, and the estimator, sampling-time, incomplete-protection and general
identity results in the remaining logs. To regenerate all outputs from scratch with the seeds used in the manuscript
(about four hours on two cores):

    ./run_all.sh

## Scripts

| Script | Purpose | Output |
| --- | --- | --- |
| `common.py` | Model matrices, parameter sampling, AUC | |
| `npm1_cohort.py` | NPM1-AML cohort: four 28-day courses (7 drug days, 21 rest), three-year follow-up, MRD after course 2, per-course multiplier, drug-free regime | `data/npm1_*.json` |
| `npm1_analyse.py` | Calibration statistics, fate by MRD level, AUC of MRD, multiplier and drug-free growth rate | `data/analyse_npm1_*.log` |
| `npm1_second_input.py` | Information added by a second dose level or rest length (exact kinetics) | `data/npm1_second_input.log` |
| `npm1_estimator.py` | Model-based estimate of the drug-free growth rate from noisy, censored MRD samples; standard and within-course sampling; correct and misspecified priors | `data/prior_*.npz`, `data/npm1_estimator.log` |
| `npm1_sample_timing.py` | Timing of the within-course sample (day 7, 14 or 21) and reduced assay sensitivity at low counts; weekly samples through courses 1 and 2 | `data/timing_*.npz`, `data/npm1_sample_timing.log` |
| `cml_cohort.py` | CML cohort: five years of TKI, stopping at MR4, three-year follow-up; on-treatment eigenvalues at full and half dose | `data/cml_*.json` |
| `cml_analyse.py` | AUC of on-treatment quantities and the two-level estimate; numerical check of the identity | `data/analyse_cml_*.log` |
| `cml_deescalation.py` | CML half-dose de-escalation with qPCR noise and detection limit; censored fits | `data/cml_deesc_*.json` |
| `cml_deescalation_analyse.py` | AUC of depth, slope and the two-level estimate | `data/analyse_cml_deesc_*.log` |
| `cml_incomplete_protection.py` | Two-level and three-level estimates when the drug also kills protected cells | `data/cml_incomplete_protection.log` |
| `general_identity.py` | Numerical check of the identification result for n compartments and r treated compartments | `data/general_identity.log` |
| `make_figures.py` | Figures 2 to 7 (Figure 1 is a schematic drawn in the manuscript) | `figures/`, `results/` |

Output file names encode the parameter ranges and seed, for example
`npm1_<kill range>_<niche range>.json` and `cml_<niche range>_<dose ratio>_<seed>.json`.

## Notes

- Burden is relative to the diagnostic burden (= 1); rates are per day.
- Equations are integrated with LSODA (SciPy). The CML model is integrated in logarithmic
  coordinates.
- The CML scripts skip a patient if the solver exceeds a wall-clock limit (5 s, or 20 s for
  de-escalation) and prints the number skipped. On a much slower machine this could change
  which patients enter the cohort.

## Licence

The code is released under the MIT licence (`LICENSE`). The simulation outputs in `data/`,
`results/` and `figures/` are released under the Creative Commons Attribution 4.0 International
licence (CC BY 4.0).

## Citation

If you use this code, please cite the manuscript and this archive; citation metadata are in
`CITATION.cff`.

