"""CML de-escalation: AUC for relapse of depth, time in MR4.5, the half-dose slope and the
two-level estimate (with the true or the population-median kill rate).

Usage: python3 cml_deescalation_analyse.py data/cml_deesc_<...>.json
"""
import json
import sys

import numpy as np

from cml_deescalation import lam_off_from
from common import auc

with open(sys.argv[1]) as f:
    R = json.load(f)
rel = np.array([r["rel"] for r in R])
k_med = float(np.median([r["kappa"] for r in R]))
slope = np.array([r["slope6"] if r["slope6"] is not None else np.nan for r in R])
# series undetectable throughout the probe are assigned the lowest observed slope
slope_imp = np.where(np.isfinite(slope), slope, np.nanmin(slope) - 1e-3)
print(f"eligible {len(R)}, relapse {rel.mean():.2f}, probe undetectable throughout in "
      f"{np.mean(~np.isfinite(slope)):.2f}")
rows = [("depth at start of de-escalation", np.array([r["depth"] for r in R])),
        ("years in MR4.5 (negated)", -np.array([r["mr45"] for r in R])),
        ("6-month half-dose slope", slope_imp)]
for label, kill in [("true kill", lambda r: r["kappa"]), ("population-median kill", lambda r: k_med)]:
    rows.append((f"two-level estimate, {label}",
                 np.array([lam_off_from(r["l1"], r["l2"], s, kill(r)) for r, s in zip(R, slope_imp)])))
rows.append(("exact drug-free growth rate", np.array([r["lam_off"] for r in R])))
for name, score in rows:
    print(f"  {name:44s} AUC {auc(score, rel):.3f} (n={int(np.isfinite(score).sum())})")
