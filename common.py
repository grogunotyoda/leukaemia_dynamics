"""Shared helpers: the two-compartment model matrices, parameter sampling and the AUC."""
import numpy as np


def log_uniform(rng, a, b):
    return 10 ** rng.uniform(np.log10(a), np.log10(b))


def drug_free_matrix(p):
    """Linearisation of the drug-free dynamics at small burden (state: cycling A, protected Q)."""
    return np.array([[p["r"] - p["alpha"], p["beta"]],
                     [p["alpha"], p["s"] - p["beta"] - p["delta"]]])


def treated_matrix(p, k):
    M = drug_free_matrix(p)
    M[0, 0] -= k
    return M


def growth_rate(M):
    """Largest real part of the eigenvalues of M."""
    return float(max(np.linalg.eigvals(M).real))


def auc(score, label):
    """Area under the ROC curve by the rank (Mann-Whitney) formula; non-finite scores are dropped."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(score)
    s1, s0 = score[ok & label], score[ok & ~label]
    if s1.size == 0 or s0.size == 0:
        return float("nan")
    ranks = np.concatenate([s1, s0]).argsort().argsort() + 1
    return float((ranks[:s1.size].sum() - s1.size * (s1.size + 1) / 2) / (s1.size * s0.size))
