"""Model-based estimate of the drug-free growth rate from MRD samples in courses 1 and 2.

A reference set is drawn from the prior and simulated through two courses (no follow-up,
no outcomes). For each cohort patient, noisy and censored samples weight the reference
patients by a censored (Tobit) likelihood; the estimate is the weighted mean drug-free
growth rate. Evaluated by the AUC for three-year relapse in the calibrated cohort.

Usage:
  python3 npm1_estimator.py build <n> <seed> <kappa_lo> <kappa_hi> <K_lo> <K_hi> <out.npz>
  python3 npm1_estimator.py eval <cohort.json> <reference.npz> [<reference2.npz> ...]
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from common import drug_free_matrix, growth_rate
from npm1_cohort import phase, sample_parameters, relapsed, T_ON, T_OFF

warnings.filterwarnings("ignore")
SIGMA = 0.2           # qPCR noise, log10 units
SAMPLE_DAYS = [7, 28, 35, 56]


def trajectory(p, fq):
    """Burden at the end of each phase of courses 1 and 2 (days 7, 28, 35, 56)."""
    q0 = min(fq, p["K"])
    x = np.array([1 - q0, q0])
    out = []
    for _ in range(2):
        for kill, T in [(p["kappa"], T_ON), (0.0, T_OFF)]:
            x = None if x is None else phase(x, p, kill, T)
            out.append(0.0 if x is None else x.sum())
    return np.array(out)


def reference_patient(args):
    seed, kappa_range, K_range = args
    rng = np.random.default_rng(seed)
    p, fq = sample_parameters(rng, kappa_range, K_range)
    y = trajectory(p, fq)
    return np.log10(np.maximum(y, 1e-30)), growth_rate(drug_free_matrix(p)), p["r"] >= p["alpha"]


def build(n, seed, kappa_range, K_range, out):
    with Pool(2) as pool:
        res = pool.map(reference_patient, [(seed * 1000003 + i, kappa_range, K_range) for i in range(n)],
                       chunksize=200)
    LY = np.array([r[0] for r in res], np.float32)
    LAM = np.array([r[1] for r in res])
    G = np.array([r[2] for r in res])
    np.savez_compressed(out, LY=LY, LAM=LAM, G=G)
    print(out, LY.shape, flush=True)


def evaluate(cohort_file, references):
    with open(cohort_file) as f:
        R = json.load(f)
    rng = np.random.default_rng(4)
    Y = np.log10(np.maximum(np.array([trajectory(r["p"], r["fq"]) for r in R]), 1e-30))
    rel = np.array([relapsed(r) for r in R])
    lam_true = np.array([growth_rate(drug_free_matrix(r["p"])) for r in R])
    print("truth patients", len(R), "relapse", round(rel.mean(), 3),
          "| ceiling AUC (true growth rate)", round(roc_auc_score(rel, lam_true), 3))
    for ref in references:
        L = np.load(ref)
        LY, LAM, G = L["LY"].astype(float), L["LAM"], L["G"]
        print(f"== library {ref.split('/')[-1]} (n={LY.shape[0]})")
        for lod, assay in [(1e-5, "blood"), (1e-6, "marrow")]:
            zl = np.log10(lod)
            Z = Y + SIGMA * rng.standard_normal(Y.shape)
            censored = Z < zl
            for cols, name in [([1, 3], "standard (days 28, 56)"),
                               ([0, 1, 2, 3], "within-course (days 7, 28, 35, 56)")]:
                est, p_grow, ess = np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R))
                for i in range(len(R)):
                    ll = np.zeros(LY.shape[0])
                    for j in cols:
                        if censored[i, j]:
                            ll += norm.logcdf((zl - LY[:, j]) / SIGMA)
                        else:
                            ll += norm.logpdf((Z[i, j] - LY[:, j]) / SIGMA)
                    w = np.exp(ll - ll.max())
                    w /= w.sum()
                    est[i], p_grow[i], ess[i] = (w * LAM).sum(), (w * G).sum(), 1 / (w ** 2).sum()
                print(f"  {assay:6s} {name:36s} AUC posterior growth rate {roc_auc_score(rel, est):.3f}"
                      f" | P(grow) {roc_auc_score(rel, p_grow):.3f}"
                      f" | median effective sample size {np.median(ess):.0f}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        n, seed = int(sys.argv[2]), int(sys.argv[3])
        build(n, seed, (float(sys.argv[4]), float(sys.argv[5])),
              (float(sys.argv[6]), float(sys.argv[7])), sys.argv[8])
    else:
        evaluate(sys.argv[2], sys.argv[3:])
