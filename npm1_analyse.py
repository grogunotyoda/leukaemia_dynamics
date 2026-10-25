"""NPM1 cohort: calibration statistics, fate by MRD level after course 2, and the AUC for
three-year relapse of MRD, the per-course multiplier and the drug-free growth rate.

Usage: python3 npm1_analyse.py data/npm1_<...>.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from common import auc, drug_free_matrix, growth_rate
from npm1_cohort import LOD, relapsed

with open(sys.argv[1]) as f:
    R = json.load(f)
y2 = np.array([r["y_c2"] for r in R])
mu = np.array([r["mu"] for r in R])
reg = np.array([r["regime"] for r in R])
ext = np.array([r["extinct"] for r in R])
rel = np.array([relapsed(r) for r in R])
lam_off = np.array([growth_rate(drug_free_matrix(r["p"])) for r in R])
fate = np.where(ext, "cleared in treatment", reg)
FATES = ["cleared in treatment", "cure", "persist", "grow"]

print(f"n {len(R)} | MRD negative after course 2: {(y2 < LOD).mean():.3f} | 3-year relapse "
      f"MRD positive {rel[y2 >= LOD].mean():.3f}, MRD negative {rel[y2 < LOD].mean():.3f}")

bins = [(-np.inf, LOD, "undetectable (<1e-5)"), (LOD, 1e-4, "1e-5 to 1e-4"),
        (1e-4, 1e-3, "1e-4 to 1e-3"), (1e-3, np.inf, ">= 1e-3")]
print("\nMRD after course 2 | n | 3-year relapse | fate shares", FATES)
for lo, hi, name in bins:
    m = (y2 >= lo) & (y2 < hi) if np.isfinite(lo) else (y2 < hi)
    shares = [round((fate[m] == f).mean(), 2) for f in FATES]
    print(f"  {name:22s} {m.sum():5d}  {rel[m].mean():.2f}  {shares}")

print("\nAUC for 3-year relapse")
print(f"  MRD after course 2        {auc(np.log10(np.maximum(y2, 1e-14)), rel):.3f}")
print(f"  per-course multiplier mu  {auc(np.log(mu), rel):.3f}")
print(f"  drug-free growth rate     {auc(lam_off, rel):.3f}")

ne = ~ext
rho = spearmanr(np.log(mu[ne]), lam_off[ne]).correlation
print(f"\nSpearman correlation, log mu vs drug-free growth rate (not cleared): {rho:.3f}")

m = y2 < LOD
print("\nFate among patients MRD negative after course 2")
for f in FATES:
    print(f"  {f:20s} {(fate[m] == f).mean():.3f}")
print(f"  relapse within 3 years: {rel[m].mean():.3f}")
