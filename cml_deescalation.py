"""CML de-escalation with qPCR noise and a detection limit.

Protocol: four years of full-dose TKI (qPCR monthly in year 1, then three-monthly). Patients at
or below MR4 at every test in year 4 enter de-escalation: 12 months at half dose with monthly
qPCR, then stop, with three years of follow-up. Outcome: loss of MMR (> 1e-3) during
de-escalation or after stopping. Prediction uses the full-dose data and the first six months
of de-escalation. qPCR: noise SD 0.2 log10, detection limit 1e-5.

Stored per eligible patient:
  depth   last full-dose result (censored values set to the limit)
  mr45    years with results <= 10^-4.5
  slope6  censored (Tobit) slope over the first six half-dose months
  l1, l2  on-treatment rates from a censored biexponential fit to the full-dose decline
  lam_off exact drug-free growth rate
The drug-free growth rate is then estimated from (l1, l2, slope6) and a kill rate by
lam_off_from(), using the two-level identity.

Usage: python3 cml_deescalation.py <K_lo> <K_hi> <n> <seed>
Output: data/cml_deesc_<K_lo>_<K_hi>_<seed>.json
"""
import json
import signal
import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import cml_cohort as cml
from common import treated_matrix, drug_free_matrix, growth_rate

warnings.filterwarnings("ignore")
SIGMA, LOD, RHO = 0.2, 1e-5, 0.5
ZL = np.log10(LOD)


def tobit_nll(mean, z, censored):
    return -np.where(censored, norm.logcdf((ZL - mean) / SIGMA), norm.logpdf((z - mean) / SIGMA)).sum()


def slope_fit(t, z, censored):
    """Censored linear fit of log10 burden; returns the natural-log rate per day."""
    if (~censored).sum() == 0:
        return np.nan
    best = None
    for b0 in (-0.01, 0.0, 0.01):
        r = minimize(lambda th: tobit_nll(th[0] + th[1] * t, z, censored),
                     [np.mean(np.where(censored, ZL, z)), b0 / np.log(10)],
                     method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=2000))
        best = r if best is None or r.fun < best.fun else best
    return best.x[1] * np.log(10)


def biexp_fit(t, z, censored):
    """Censored fit of log10 y = log10(c1 e^{l1 t} + c2 e^{l2 t}) with l1 < l2; returns (l1, l2)."""
    def model(th):
        lc1, l1, lc2, dl = th
        l2 = l1 + np.exp(dl)
        return np.log10(10 ** lc1 * np.exp(l1 * t) + 10 ** lc2 * np.exp(l2 * t))
    best = None
    for l1, l2 in [(-0.05, -0.005), (-0.1, -0.002), (-0.03, -0.01), (-0.2, 0.0)]:
        th0 = [0.0, l1, -2.0, np.log(l2 - l1)]
        r = minimize(lambda th: tobit_nll(model(th), z, censored), th0, method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
        best = r if best is None or r.fun < best.fun else best
    l1 = best.x[1]
    return l1, l1 + np.exp(best.x[3])


def lam_off_from(l1, l2, lr, k):
    """Drug-free growth rate from the full-dose rates (l1, l2), the reduced-dose slow rate lr
    and the kill rate k, using tr and det affine in the kill."""
    tr_on, det_on = l1 + l2, l1 * l2
    m = (-lr ** 2 + (tr_on + (1 - RHO) * k) * lr - det_on) / ((1 - RHO) * k)
    tr_off, det_off = tr_on + k, det_on + k * m
    return (tr_off + np.sqrt(max(tr_off ** 2 - 4 * det_off, 0))) / 2


def patient(rng, K_range):
    p, fq = cml.sample_parameters(rng, K_range)
    q0 = min(fq, p["K"])
    x = np.array([1 - q0, q0])
    t1 = np.concatenate([np.arange(30, 361, 30.0), np.arange(450, 1461, 91.0)])
    try:
        y1, x = cml.run(x, p, p["kappa"], 1460.0, t1)
    except cml.Progressed:
        return None
    if x is None or not (y1[t1 > 1095] <= cml.MR4).all():
        return None                                   # cleared on treatment or not eligible
    t2 = np.arange(30, 361, 30.0)
    try:
        y2, x2 = cml.run(x, p, RHO * p["kappa"], 360.0, t2)
    except cml.Progressed:
        y2, x2 = np.ones(t2.size), None
    relapse = bool((y2 > cml.MMR).any())
    if not relapse and x2 is not None:
        relapse = cml.time_to_relapse(x2, p) is not None

    z1 = np.log10(np.maximum(y1, 1e-30)) + SIGMA * rng.standard_normal(t1.size)
    z2 = np.log10(np.maximum(y2, 1e-30)) + SIGMA * rng.standard_normal(t2.size)
    c1, c2 = z1 < ZL, z2 < ZL
    z1c, z2c = np.where(c1, ZL, z1), np.where(c2, ZL, z2)
    l1, l2 = biexp_fit(np.concatenate([[0.0], t1]), np.concatenate([[0.0], z1c]),
                       np.concatenate([[False], c1]))
    lr = slope_fit(t2[:6], z2c[:6], c2[:6])
    lam_on = sorted(np.linalg.eigvals(treated_matrix(p, p["kappa"])).real)
    return dict(rel=relapse, kappa=p["kappa"], depth=float(z1c[-1]),
                mr45=float(np.sum(z1c <= -4.5) * 91 / 365),
                slope6=float(lr) if np.isfinite(lr) else None, l1=float(l1), l2=float(l2),
                lam_off=growth_rate(drug_free_matrix(p)),
                l1_true=float(lam_on[0]), l2_true=float(lam_on[1]))


def _timeout(*_):
    raise RuntimeError("timeout")


if __name__ == "__main__":
    K_lo, K_hi, n, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(seed)
    signal.signal(signal.SIGALRM, _timeout)
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        signal.alarm(20)
        try:
            r = patient(rng, (K_lo, K_hi))
        except RuntimeError:
            r = None
        finally:
            signal.alarm(0)
        if r is not None:
            out.append(r)
    with open(f"data/cml_deesc_{K_lo}_{K_hi}_{seed}.json", "w") as f:
        json.dump(out, f)
    print("eligible patients", len(out), "from", tries, "drawn", flush=True)
