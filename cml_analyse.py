"""CML cohort, exact kinetics: AUC for relapse after stopping of on-treatment quantities and of
the two-level estimate, among stop-eligible patients. Also checks the identity numerically.

    det_off = (det M(rho k) - rho det M(k)) / (1 - rho),  likewise for the trace.

Usage: python3 cml_analyse.py data/cml_<...>.json <rho>
"""
import json
import sys

import numpy as np

from common import auc, drug_free_matrix, growth_rate

with open(sys.argv[1]) as f:
    R = [r for r in json.load(f) if r["eligible"] and not r["extinct"]]
rho = float(sys.argv[2])
rel = np.array([r["relapse"] for r in R])

level = np.log10(np.maximum([r["y_stop"] for r in R], 1e-14))
mr45 = np.array([r["mr45_years"] for r in R])
slow_full = np.array([max(r["lam_full"]) for r in R])
slow_red = np.array([max(r["lam_red"]) for r in R])
det_full = np.array([r["det_full"] for r in R])
det_red = np.array([r["det_red"] for r in R])
det_off = np.array([r["det_off"] for r in R])
tr_full = np.array([sum(r["lam_full"]) for r in R])
tr_red = np.array([sum(r["lam_red"]) for r in R])

det_hat = (det_red - rho * det_full) / (1 - rho)
tr_hat = (tr_red - rho * tr_full) / (1 - rho)
lam_hat = (tr_hat + np.sqrt(np.maximum(tr_hat ** 2 - 4 * det_hat, 0))) / 2
lam_true = np.array([growth_rate(drug_free_matrix(r["p"])) for r in R])

print(f"eligible, not cleared: {len(R)}, relapse: {rel.mean():.3f}")
print("identity check, max |det_off estimate - det_off|:", float(np.abs(det_hat - det_off).max()))
print("drug-free regime among eligible:",
      {f: round(np.mean([r["fate_off"] == f for r in R]), 3) for f in ["cure", "persist", "grow"]})
print("AUC for relapse after stopping (higher score = more relapse)")
print(f"  level at stopping              {auc(level, rel):.3f}")
print(f"  years in MR4.5 (negated)       {auc(-mr45, rel):.3f}")
print(f"  slow slope, full dose          {auc(slow_full, rel):.3f}")
print(f"  det M, full dose (negated)     {auc(-det_full, rel):.3f}")
print(f"  slow slope, half dose          {auc(slow_red, rel):.3f}")
print(f"  two-level det_off (negated)    {auc(-det_hat, rel):.3f}")
print(f"  two-level drug-free growth rate {auc(lam_hat, rel):.3f}"
      f" | max error vs exact {float(np.abs(lam_hat - lam_true).max()):.2e}")
fate = np.array([r["fate_off"] for r in R])
for f in ["cure", "persist", "grow"]:
    m = fate == f
    print(f"  relapse rate, drug-free regime {f:8s} n={m.sum():4d}: {rel[m].mean():.2f}")
