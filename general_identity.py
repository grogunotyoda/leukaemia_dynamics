"""Numerical check of the identification result for models with n compartments.

For random irreducible Metzler matrices M0 (n = 2..6) and a treatment acting on nu compartments
with relative sensitivities D, the characteristic polynomial of M0 - kD is a polynomial of degree
at most nu in k, so nu + 1 dose levels with known ratios recover the drug-free polynomial by
Lagrange interpolation at k = 0. For nu = 1 the check also covers the limit of the poles as
k -> infinity and the monotone decrease of the dominant pole.

Usage: python3 general_identity.py
"""
import numpy as np

rng = np.random.default_rng(1)
RATIOS = [1.0, 0.7, 0.45, 0.25]


def random_metzler(n):
    M = rng.uniform(0, 0.05, (n, n))
    np.fill_diagonal(M, rng.uniform(-0.15, 0.05, n))
    return M


def charpoly(M):
    return np.real(np.poly(np.linalg.eigvals(M)))


def dominant(M):
    return max(np.linalg.eigvals(M).real)


def interpolate_at_zero(ratios, polys):
    out = 0.0
    for j, rj in enumerate(ratios):
        w = np.prod([ri / (ri - rj) for i, ri in enumerate(ratios) if i != j])
        out = out + w * polys[j]
    return out


print("(i) nu + 1 dose levels recover the drug-free characteristic polynomial")
for n in range(2, 7):
    for nu in range(1, min(n, 3) + 1):
        err_poly = err_rate = 0.0
        for _ in range(500):
            M0 = random_metzler(n)
            d = np.zeros(n)
            d[rng.choice(n, nu, replace=False)] = rng.uniform(0.1, 1.0, nu)
            kappa = rng.uniform(0.05, 1.0)
            ratios = RATIOS[:nu + 1]
            p0 = interpolate_at_zero(ratios, [charpoly(M0 - f * kappa * np.diag(d)) for f in ratios])
            err_poly = max(err_poly, np.abs(p0 - charpoly(M0)).max())
            err_rate = max(err_rate, abs(max(np.roots(p0).real) - dominant(M0)))
        print(f"  n={n} nu={nu} ({nu + 1} levels): max coefficient error {err_poly:.1e}, "
              f"max error in drug-free growth rate {err_rate:.1e}")

print("\n(ii) nu = 1: poles as k -> infinity and monotone dominant pole")
for n in range(2, 7):
    err_end, monotone = 0.0, True
    for _ in range(500):
        M0 = random_metzler(n)
        D = np.zeros((n, n))
        D[0, 0] = 1.0
        poles = np.sort(np.linalg.eigvals(M0 - 1e6 * D).real)[1:]
        err_end = max(err_end, np.abs(poles - np.sort(np.linalg.eigvals(M0[1:, 1:]).real)).max())
        dom = [dominant(M0 - k * D) for k in np.linspace(0, 5, 60)]
        monotone &= bool(np.all(np.diff(dom) <= 1e-12)) and dom[-1] >= dominant(M0[1:, 1:]) - 1e-9
    print(f"  n={n}: max distance of n-1 poles from the protected-block eigenvalues at k=1e6 {err_end:.1e}; "
          f"dominant pole decreasing towards that of the protected block: {monotone}")
