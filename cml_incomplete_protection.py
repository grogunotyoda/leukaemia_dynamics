"""CML cohorts, exact kinetics: effect of a kill on protected cells (incomplete protection).

The drug kills protected cells at a fraction eps of the rate on cycling cells, M(k) = M_off - k diag(1, eps).
For the stop-eligible patients, the drug-free dominant eigenvalue is estimated from the spectra at
dose ratios 1 and 0.5 (two-level formula) or 1, 0.5 and 0.25 (Lagrange interpolation), and its
AUC for relapse after stopping is computed. Eligibility and outcome are those of the eps = 0 cohorts.

Usage: python3 cml_incomplete_protection.py
"""
import json

import numpy as np

from common import auc, drug_free_matrix

FILES = ["data/cml_1e-08_0.0001_1.0_2.json", "data/cml_1e-09_1e-05_1.0_2.json",
         "data/cml_1e-10_1e-06_1.0_2.json"]
RHO = 0.5


def charpoly(M):
    return np.real(np.poly(np.linalg.eigvals(M)))


def weight(a, b, c):
    """Lagrange weight at k = 0 for node a with the other nodes b and c (ratios of the dose)."""
    return (b * c) / ((a - b) * (a - c))


cohorts = []
for fn in FILES:
    with open(fn) as f:
        cohorts.append([r for r in json.load(f) if r["eligible"] and not r["extinct"]])

print("eps    AUC two levels   AUC three levels   median eps*kappa/|m|   (mean over niche ranges)")
for eps in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]:
    auc2, auc3, ratio = [], [], []
    for R in cohorts:
        rel = np.array([r["relapse"] for r in R])
        est2, est3 = [], []
        for r in R:
            p = r["p"]
            M0, k, D = drug_free_matrix(p), p["kappa"], np.diag([1.0, eps])
            P = {f: charpoly(M0 - f * k * D) for f in (1.0, 0.5, 0.25)}
            p2 = (P[0.5] - RHO * P[1.0]) / (1 - RHO)
            p3 = (weight(1.0, 0.5, 0.25) * P[1.0] + weight(0.5, 1.0, 0.25) * P[0.5]
                  + weight(0.25, 1.0, 0.5) * P[0.25])
            est2.append(max(np.roots(p2).real))
            est3.append(max(np.roots(p3).real))
            ratio.append(abs(eps * k / (p["s"] - p["beta"] - p["delta"])))
        auc2.append(auc(est2, rel))
        auc3.append(auc(est3, rel))
    print(f"{eps:<6} {np.mean(auc2):.3f}            {np.mean(auc3):.3f}              {np.median(ratio):.2f}")
