"""Virtual CML cohort under continuous TKI treatment.

Same two-compartment model as npm1_cohort.py, with constant kill kappa on the cycling
compartment. Protocol: TKI for five years; eligible to stop if BCR-ABL1 <= 1e-4 (MR4) at every
three-monthly test in the final year; then three years without treatment. Molecular relapse:
loss of major molecular response (burden > 1e-3). Burden is relative to diagnosis (100% IS).
Patients whose burden exceeds twice the diagnostic burden on treatment are classed as failing.

The on-treatment eigenvalues at full and half dose are stored for the two-level estimate:
det M(k) = det M_off - k m and tr M(k) = tr M_off - k, with m = s - beta - delta.

Usage: python3 cml_cohort.py <K_lo> <K_hi> <rho> <n> <seed>
  rho scales the kill in the final year before stopping (1.0 in the manuscript).
Output: data/cml_<K_lo>_<K_hi>_<rho>_<seed>.json
"""
import json
import signal
import sys
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from common import log_uniform, treated_matrix, drug_free_matrix

warnings.filterwarnings("ignore")
CELL = 1e-12
MR4, MMR = 1e-4, 1e-3
FOLLOW_UP = 1095.0


class Progressed(Exception):
    """Burden exceeded twice the diagnostic burden on treatment."""


def rhs_log(t, u, p, k):
    """Model in log coordinates u = (ln A, ln Q), well scaled over many orders of magnitude."""
    a, q = u
    e_qa = np.exp(np.clip(q - a, -700, 700))
    e_aq = np.exp(np.clip(a - q, -700, 700))
    return [p["r"] - p["alpha"] - k + p["beta"] * e_qa,
            p["alpha"] * e_aq + p["s"] * (1 - np.exp(q) / p["K"]) - (p["beta"] + p["delta"])]


def run(x, p, k, T, t_eval):
    """Integrate with kill k over [0, T]; return burden at t_eval and the end state (None if cleared)."""
    u0 = np.log(np.maximum(x, 1e-300))
    cleared = lambda t, u, p, k: np.logaddexp(u[0], u[1]) - np.log(CELL)
    cleared.terminal = True
    cleared.direction = -1
    failed = lambda t, u, p, k: np.logaddexp(u[0], u[1]) - np.log(2.0)
    failed.terminal = True
    failed.direction = 1
    sol = solve_ivp(rhs_log, (0, T), u0, args=(p, k), t_eval=t_eval, method="LSODA",
                    rtol=1e-8, atol=1e-8, events=[cleared, failed])
    U = np.asarray(sol.y, float).reshape(2, -1)
    y = np.full(t_eval.size, 1.0)
    n = U.shape[1]
    y[:n] = np.exp(U[0]) + np.exp(U[1])
    if sol.t_events[1].size:
        raise Progressed()
    if sol.t_events[0].size:
        y[n:] = 0.0
        return y, None
    if not sol.success or n < t_eval.size:
        raise RuntimeError("solver failure")
    return y, np.exp(U[:, -1])


def time_to_relapse(x, p):
    """Days from stopping until the burden exceeds MMR, or None within the follow-up."""
    reached = lambda t, u, p, k: np.logaddexp(u[0], u[1]) - np.log(MMR)
    reached.terminal = True
    sol = solve_ivp(rhs_log, (0, FOLLOW_UP), np.log(np.maximum(x, 1e-300)), args=(p, 0.0),
                    method="LSODA", rtol=1e-8, atol=1e-8, events=reached)
    return float(sol.t_events[0][0]) if sol.t_events[0].size else None


def sample_parameters(rng, K_range):
    lu = lambda a, b: log_uniform(rng, a, b)
    p = dict(r=rng.uniform(0.0, 0.12), alpha=lu(1e-3, 0.15), beta=lu(1e-4, 0.02), s=lu(1e-5, 0.01),
             K=lu(*K_range), delta=lu(1e-5, 0.005), kappa=lu(0.02, 0.3))
    fq = lu(1e-3, 0.1)
    return p, fq


def patient(rng, K_range, rho):
    p, fq = sample_parameters(rng, K_range)
    q0 = min(fq, p["K"])
    x = np.array([1 - q0, q0])
    T1, T2 = 4 * 365.0, 365.0
    t1 = np.arange(91, T1 + 1, 91.0)
    failing = dict(p=p, eligible=False, extinct=False, relapse=False, y_stop=None, progressed=True)
    try:
        y1, x = run(x, p, p["kappa"], T1, t1)
    except Progressed:
        return failing
    if x is None:
        return dict(p=p, eligible=True, extinct=True, relapse=False, y_stop=0.0)
    t2 = np.arange(91, T2 + 1, 91.0)
    try:
        y2, x2 = run(x, p, rho * p["kappa"], T2, t2)
    except Progressed:
        return failing
    tests = np.concatenate([y1, y2])
    out = dict(p=p, eligible=bool((y2 <= MR4).all()), extinct=x2 is None, y_stop=float(y2[-1]),
               tests=tests.tolist(), mr45_years=float(np.sum(tests <= 10 ** -4.5) * 91 / 365))
    for k, key in [(p["kappa"], "full"), (0.5 * p["kappa"], "red")]:
        M = treated_matrix(p, k)
        out["lam_" + key] = sorted(np.linalg.eigvals(M).real.tolist())
        out["det_" + key] = float(np.linalg.det(M))
    out["det_off"] = float(np.linalg.det(drug_free_matrix(p)))
    out["grow_off"] = bool(p["r"] >= p["alpha"])
    out["fate_off"] = "grow" if out["grow_off"] else ("cure" if out["det_off"] > 0 else "persist")
    relapse = False
    if x2 is not None:
        out["t_relapse"] = time_to_relapse(x2, p)
        relapse = out["t_relapse"] is not None
    out["relapse"] = relapse
    return out


def _timeout(*_):
    raise RuntimeError("timeout")


if __name__ == "__main__":
    K_lo, K_hi, rho = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    n, seed = int(sys.argv[4]), int(sys.argv[5])
    rng = np.random.default_rng(seed)
    signal.signal(signal.SIGALRM, _timeout)
    R, skipped = [], 0
    while len(R) < n:
        signal.alarm(5)
        try:
            R.append(patient(rng, (K_lo, K_hi), rho))
        except RuntimeError:
            skipped += 1
        finally:
            signal.alarm(0)
    print("solver failures or timeouts skipped:", skipped, flush=True)
    E = [r for r in R if r["eligible"]]
    rel = np.mean([r["relapse"] for r in E]) if E else float("nan")
    tr = [r["t_relapse"] for r in E if r.get("t_relapse") is not None]
    print(f"K [{K_lo}, {K_hi}], rho {rho}: eligible {len(E)}/{n}; relapse after stopping {rel:.2f}; "
          f"median time to relapse {np.median(tr) if tr else float('nan'):.0f} days", flush=True)
    with open(f"data/cml_{K_lo}_{K_hi}_{rho}_{seed}.json", "w") as f:
        json.dump(R, f)
