"""Virtual NPM1-mutated AML cohort under pulsed treatment.

Model (burden relative to diagnosis = 1):
    A' = (r - alpha - kappa g(t)) A + beta Q
    Q' = alpha A + s Q (1 - Q/K) - (beta + delta) Q
with g(t) = 1 on drug days and 0 otherwise. Four 28-day courses (7 drug days, 21 rest days),
then three years without treatment. A burden below 1e-12 (one cell for 1e12 cells at
diagnosis) is treated as cleared. Relapse: burden reaches 1e-2 within three years.

Recorded per patient: parameters, blood MRD after course 2, per-course multiplier mu
(spectral radius of the one-course monodromy matrix), drug-free regime and relapse.

Usage: python3 npm1_cohort.py <kappa_lo> <kappa_hi> <n> <K_lo> <K_hi> [seed]
Output: data/npm1_<kappa_lo>_<kappa_hi>_<K_lo>_<K_hi>.json
"""
import json
import sys
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from common import log_uniform, drug_free_matrix, treated_matrix

CELL = 1e-12
T_ON, T_OFF, N_COURSES = 7.0, 21.0, 4
LOD = 1e-5            # blood qPCR limit
RELAPSE = 1e-2
FOLLOW_UP = 1095.0    # days


def rhs(t, x, p, kill):
    A, Q = x
    dA = (p["r"] - p["alpha"] - kill) * A + p["beta"] * Q
    dQ = p["alpha"] * A + p["s"] * Q * (1 - Q / p["K"]) - (p["beta"] + p["delta"]) * Q
    return [dA, dQ]


def phase(x, p, kill, T):
    """Integrate one phase; return the end state, or None if the burden falls below one cell."""
    cleared = lambda t, x, p, k: x[0] + x[1] - CELL
    cleared.terminal = True
    cleared.direction = -1
    sol = solve_ivp(rhs, (0, T), x, args=(p, kill), method="LSODA", rtol=1e-9, atol=1e-30,
                    events=cleared)
    if sol.t_events[0].size:
        return None
    return np.maximum(sol.y[:, -1], 0.0)


def course_multiplier(p):
    M_on, M_off = treated_matrix(p, p["kappa"]), drug_free_matrix(p)
    return float(max(abs(np.linalg.eigvals(expm(M_off * T_OFF) @ expm(M_on * T_ON)))))


def regime(p):
    if p["r"] >= p["alpha"]:
        return "grow"
    return "cure" if np.linalg.det(drug_free_matrix(p)) > 0 else "persist"


def sample_parameters(rng, kappa_range, K_range):
    lu = lambda a, b: log_uniform(rng, a, b)
    p = dict(r=rng.uniform(-0.2, 0.2), alpha=lu(1e-3, 0.2), beta=lu(1e-3, 0.2), s=lu(1e-4, 0.05),
             K=lu(*K_range), delta=lu(1e-4, 0.02), kappa=lu(*kappa_range))
    fq = lu(1e-3, 0.3)    # initial protected fraction
    return p, fq


def patient(rng, kappa_range, K_range=(1e-6, 1e-2)):
    p, fq = sample_parameters(rng, kappa_range, K_range)
    q0 = min(fq, p["K"])
    x = np.array([1 - q0, q0])
    y_c2 = None
    extinct = False
    for c in range(N_COURSES):
        for kill, T in [(p["kappa"], T_ON), (0.0, T_OFF)]:
            x = phase(x, p, kill, T)
            if x is None:
                extinct = True
                break
        if extinct:
            break
        if c == 1:
            y_c2 = x.sum()
    if extinct and y_c2 is None:
        y_c2 = 0.0
    out = dict(p=p, fq=fq, y_c2=y_c2, mu=course_multiplier(p), regime=regime(p), extinct=extinct)

    relapse, y_eot = False, None
    if not extinct:
        y_eot = x.sum()
        reached = lambda t, x, p, k: x[0] + x[1] - RELAPSE
        reached.terminal = True
        sol = solve_ivp(rhs, (0, FOLLOW_UP), x, args=(p, 0.0), method="LSODA", rtol=1e-9, atol=1e-30,
                        events=reached)
        relapse = bool(sol.t_events[0].size) or bool(y_eot >= RELAPSE)
    out.update(y_eot=y_eot, relapse3y=relapse)
    return out


def relapsed(r):
    """Relapse within three years, counting patients already above the threshold at end of treatment."""
    return r["relapse3y"] or (r["y_eot"] is not None and r["y_eot"] >= RELAPSE)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    lo, hi, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    K_lo, K_hi = float(sys.argv[4]), float(sys.argv[5])
    rng = np.random.default_rng(int(sys.argv[6]) if len(sys.argv) > 6 else 5)
    res = [patient(rng, (lo, hi), (K_lo, K_hi)) for _ in range(n)]
    pos = [r for r in res if r["y_c2"] >= LOD]
    neg = [r for r in res if r["y_c2"] < LOD]
    rp = np.mean([r["relapse3y"] for r in pos]) if pos else float("nan")
    rn = np.mean([r["relapse3y"] for r in neg]) if neg else float("nan")
    print(f"kappa [{lo}, {hi}], K [{K_lo}, {K_hi}]: MRD negative after course 2 {len(neg) / n:.2f}; "
          f"3-year relapse MRD positive {rp:.2f}, MRD negative {rn:.2f}", flush=True)
    with open(f"data/npm1_{lo}_{hi}_{K_lo}_{K_hi}.json", "w") as f:
        json.dump(res, f)
