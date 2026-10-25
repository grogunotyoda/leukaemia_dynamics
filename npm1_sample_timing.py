"""NPM1 cohort: timing of the within-course sample and reduced assay sensitivity at low counts.

Courses 1 and 2 are simulated with weekly samples (days 7, 14, ..., 56) for the cohort and
for reference sets drawn from the prior. Each design uses a subset of the samples and the
model-based estimator of npm1_estimator.py (censored likelihood weights, weighted mean of the
drug-free growth rate). Designs: end-of-course samples only, plus one within-course sample
at day 7, 14 or 21 of each course, or plus all three. The assay limit for within-course
samples is 1e-5, 1e-4 or 1e-3; end-of-course samples keep 1e-5.

Usage:
  python3 npm1_sample_timing.py cohort <cohort.json> <out.npz>
  python3 npm1_sample_timing.py build <n> <seed> <kappa_lo> <kappa_hi> <K_lo> <K_hi> <out.npz>
  python3 npm1_sample_timing.py eval <cohort.npz> <reference.npz> [<reference2.npz> ...]
"""
import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from common import drug_free_matrix, growth_rate
from npm1_cohort import CELL, T_ON, T_OFF, rhs, sample_parameters, relapsed

warnings.filterwarnings("ignore")
SIGMA = 0.2
DAYS = [7, 14, 21, 28, 35, 42, 49, 56]
END_OF_COURSE = (28, 56)
DESIGNS = [("end of course only (days 28, 56)", [28, 56]),
           ("+ day 7 and 35 (end of drug days)", [7, 28, 35, 56]),
           ("+ day 14 and 42", [14, 28, 42, 56]),
           ("+ day 21 and 49", [21, 28, 49, 56]),
           ("+ days 7, 14, 21 and 35, 42, 49", DAYS)]
LIMITS = [1e-5, 1e-4, 1e-3]


def phase_sampled(x, p, kill, T, t_eval):
    """Integrate one phase, returning the burden at t_eval and the end state (None if cleared)."""
    cleared = lambda t, x, p, k: x[0] + x[1] - CELL
    cleared.terminal = True
    cleared.direction = -1
    sol = solve_ivp(rhs, (0, T), x, args=(p, kill), method="LSODA", rtol=1e-9, atol=1e-30,
                    events=cleared, t_eval=t_eval)
    y = np.zeros(len(t_eval))
    if sol.y.size:
        y[:sol.y.shape[1]] = sol.y.sum(0)
    if sol.t_events[0].size:
        return y, None
    return y, np.maximum(sol.y[:, -1], 0.0)


def trajectory(p, fq):
    """Burden at days 7, 14, ..., 56 of courses 1 and 2."""
    q0 = min(fq, p["K"])
    x = np.array([1 - q0, q0])
    out = []
    for _ in range(2):
        for kill, T, t_eval in [(p["kappa"], T_ON, [T_ON]), (0.0, T_OFF, [7.0, 14.0, 21.0])]:
            if x is None:
                out += [0.0] * len(t_eval)
                continue
            y, x = phase_sampled(x, p, kill, T, t_eval)
            out += list(y)
    return np.array(out)


def reference_patient(args):
    seed, kappa_range, K_range = args
    rng = np.random.default_rng(seed)
    p, fq = sample_parameters(rng, kappa_range, K_range)
    return np.log10(np.maximum(trajectory(p, fq), 1e-30)), growth_rate(drug_free_matrix(p))


def build(n, seed, kappa_range, K_range, out):
    with Pool(2) as pool:
        res = pool.map(reference_patient, [(seed * 1000003 + i, kappa_range, K_range) for i in range(n)],
                       chunksize=200)
    np.savez_compressed(out, LY=np.array([r[0] for r in res], np.float32), LAM=np.array([r[1] for r in res]))
    print(out, n, "reference patients", flush=True)


def cohort(cohort_file, out):
    with open(cohort_file) as f:
        R = json.load(f)
    with Pool(2) as pool:
        Y = pool.starmap(trajectory, [(r["p"], r["fq"]) for r in R], chunksize=50)
    np.savez_compressed(out, Y=np.log10(np.maximum(np.array(Y), 1e-30)),
                        rel=np.array([relapsed(r) for r in R]))
    print(out, len(R), "cohort patients", flush=True)


def evaluate(cohort_file, references):
    C = np.load(cohort_file)
    Y, rel = C["Y"], C["rel"]
    col = {d: i for i, d in enumerate(DAYS)}
    for ref in references:
        L = np.load(ref)
        LY, LAM = L["LY"].astype(float), L["LAM"]
        for lod in LIMITS:
            rng = np.random.default_rng(4)
            Z = Y + SIGMA * rng.standard_normal(Y.shape)
            zl = np.log10([1e-5 if d in END_OF_COURSE else lod for d in DAYS])
            censored = Z < zl
            print(f"== {ref.split('/')[-1]} | blood, within-course limit {lod:.0e}")
            for name, days in DESIGNS:
                est = np.zeros(len(Y))
                for i in range(len(Y)):
                    ll = np.zeros(LY.shape[0])
                    for j in (col[d] for d in days):
                        if censored[i, j]:
                            ll += norm.logcdf((zl[j] - LY[:, j]) / SIGMA)
                        else:
                            ll += norm.logpdf((Z[i, j] - LY[:, j]) / SIGMA)
                    w = np.exp(ll - ll.max())
                    est[i] = (w * LAM).sum() / w.sum()
                print(f"   {name:36s} AUC {roc_auc_score(rel, est):.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[2]), int(sys.argv[3]), (float(sys.argv[4]), float(sys.argv[5])),
              (float(sys.argv[6]), float(sys.argv[7])), sys.argv[8])
    elif sys.argv[1] == "cohort":
        cohort(sys.argv[2], sys.argv[3])
    else:
        evaluate(sys.argv[2], sys.argv[3:])
