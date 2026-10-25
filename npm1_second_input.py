"""NPM1 cohort, exact kinetics: information added to the per-course multiplier mu by a second
dose level (kappa/2) or a second rest length (42 days). Cross-validated gradient boosting
predicts three-year relapse from each feature set.

Usage: python3 npm1_second_input.py data/npm1_<...>.json
"""
import json
import sys

import numpy as np
from scipy.linalg import expm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from common import drug_free_matrix, treated_matrix, growth_rate
from npm1_cohort import T_ON, T_OFF, relapsed

with open(sys.argv[1]) as f:
    R = [r for r in json.load(f) if not r["extinct"]]
rel = np.array([relapsed(r) for r in R])


def log_mu(p, k, t_off=T_OFF):
    M = expm(drug_free_matrix(p) * t_off) @ expm(treated_matrix(p, k) * T_ON)
    return np.log(max(abs(np.linalg.eigvals(M))))


full = np.array([[log_mu(r["p"], r["p"]["kappa"])] for r in R])
half = np.array([[log_mu(r["p"], 0.5 * r["p"]["kappa"])] for r in R])
long_rest = np.array([[log_mu(r["p"], r["p"]["kappa"], t_off=42.0)] for r in R])
features = {
    "mu, full dose": full,
    "mu, full and half dose": np.hstack([full, half]),
    "mu, 21- and 42-day rest": np.hstack([full, long_rest]),
    "mu, all three": np.hstack([full, half, long_rest]),
    "MRD after course 2": np.array([[np.log10(max(r["y_c2"], 1e-14))] for r in R]),
    "drug-free growth rate (exact)": np.array([[growth_rate(drug_free_matrix(r["p"]))] for r in R]),
}
print(f"patients not cleared: {len(R)}, relapse: {rel.mean():.3f}")
for name, X in features.items():
    pr = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), X, rel, cv=5,
                           method="predict_proba")[:, 1]
    print(f"  {name:32s} cross-validated AUC {roc_auc_score(rel, pr):.3f}")
