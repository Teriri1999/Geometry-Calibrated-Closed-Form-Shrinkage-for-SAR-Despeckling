"""c*(gamma, L): the regularisation constant as an analytic function.

The regularisation constant c is usually treated as a free parameter, set by hand
per image.  It is not free.  The group matrix is d x K with d = ps^2, and its
dictionary is the SVD of the *noisy* group itself,
so the retained subspace absorbs noise in proportion to the aspect ratio

    gamma = d / K = ps^2 / K,

whose Marchenko-Pastur edge sits at sigma^2 (1 + sqrt(gamma))^2.  Measured directly
on pure noise (`mp_check.py`): S_max/(sigma sqrt(K)) = 0.959 (1+sqrt(gamma)) - 0.042,
R^2 = 0.9959.  The PSNR-optimal c follows the same shape,

    c*(gamma, L) = a(L) (1 + sqrt(gamma)) + b(L),

with R^2 = 0.93 / 0.95 / 0.97 at ENL 2 / 4 / 8 over gamma in [0.8, 14.4].  a rises
and b falls monotonically in log L, which is fitted here rather than hard-coded so
the constants stay tied to the sweep that produced them.

**Validity**: L >= 2.  At one look the law breaks down (R^2 = 0.11) -- c* runs past
the top of the search grid without turning over, which reflects PSNR rewarding
oversmoothing at 1 look rather than a threshold calibration.  `c_star` refuses to
extrapolate there; pass the measured optimum instead.
"""

import numpy as np

# (a, b) from analyze_hyper.py, fitted per look number over 10 (ps, K) pairs.
_FIT = {2: (0.403, 0.327), 4: (0.475, 0.012), 8: (0.519, -0.177)}


def _ab(L):
    """a(L), b(L), linear in log2 L through the three fitted look numbers."""
    x = np.log2(sorted(_FIT))
    a = np.polyfit(x, [_FIT[k][0] for k in sorted(_FIT)], 1)
    b = np.polyfit(x, [_FIT[k][1] for k in sorted(_FIT)], 1)
    return np.polyval(a, np.log2(L)), np.polyval(b, np.log2(L))


def c_star(ps, K, L):
    """Regularisation constant for a ps^2 x K group at look number L."""
    if L < 2:
        raise ValueError("c_star is not valid at L < 2; see the module docstring")
    gamma = ps * ps / K
    a, b = _ab(L)
    return float(a * (1 + np.sqrt(gamma)) + b)


if __name__ == "__main__":
    print(f"{'ps':>4}{'K':>4}{'gamma':>8}" + "".join(f"{f'L={L}':>9}" for L in [2, 4, 8, 16]))
    for ps in [4, 6, 8, 10, 12, 16]:
        for K in [10, 20]:
            row = "".join(f"{c_star(ps, K, L):9.2f}" for L in [2, 4, 8, 16])
            print(f"{ps:>4}{K:>4}{ps * ps / K:8.2f}{row}")
