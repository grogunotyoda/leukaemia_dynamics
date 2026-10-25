# Leukaemia dynamics: identifying drug-free dynamics from response data

Code and simulation outputs for the manuscript *Identifying the drug-free dynamics of residual
leukaemia from response data* (D. Koonthalakadu Baby, R. Geetha Nivas, C. Edwards).

The model has two leukaemic compartments, cycling (A) and protected (Q), with treatment acting
as a kill on A and the measured marker y = A + Q censored at the assay limit. The scripts
simulate virtual cohorts for NPM1-mutated AML (pulsed chemotherapy) and CML (continuous TKI),
and evaluate how well different response measurements predict relapse. No patient data are used.

## Requirements

Python 3.11 or later with the packages in `requirements.txt`:

    pip install -r requirements.txt

## Reproducing the results

The simulation outputs used in the manuscript are in `data/`. To redraw the figures from them:

    python3 make_figures.py

This writes `figures/*.pdf` and `results/figure_numbers.json`, which holds every number quoted
from the figures. To regenerate all outputs from scratch with the seeds used in the manuscript
(about three hours on two cores):

    ./run_all.sh

## Scripts

| Script | Purpose | Output |
| --- | --- | --- |
| `common.py` | Model matrices, parameter sampling, AUC | |
| `npm1_cohort.py` | NPM1-AML cohort: four 28-day courses (7 drug days, 21 rest), three-year follow-up, MRD after course 2, per-course multiplier, drug-free regime | `data/npm1_*.json` |
| `npm1_analyse.py` | Calibration statistics, fate by MRD level, AUC of MRD, multiplier and drug-free growth rate | printed |
| `npm1_second_input.py` | Information added by a second dose level or rest length (exact kinetics) | printed |
| `npm1_estimator.py` | Model-based estimate of the drug-free growth rate from noisy, censored MRD samples; standard and within-course sampling; correct and misspecified priors | `data/prior_*.npz`, `data/npm1_estimator.log` |
| `cml_cohort.py` | CML cohort: five years of TKI, stopping at MR4, three-year follow-up; on-treatment eigenvalues at full and half dose | `data/cml_*.json` |
| `cml_analyse.py` | AUC of on-treatment quantities and the two-level estimate; numerical check of the identity | printed |
| `cml_deescalation.py` | CML half-dose de-escalation with qPCR noise and detection limit; censored fits | `data/cml_deesc_*.json` |
| `cml_deescalation_analyse.py` | AUC of depth, slope and the two-level estimate | printed |
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
