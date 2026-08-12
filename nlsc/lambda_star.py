"""Analytic selection of the Yeo-Johnson parameter from speckle statistics alone.

The published criterion picks lambda by minimising the skewness/kurtosis of the
*whole transformed image*.  That is the wrong target: the image is signal plus
noise, and the signal's own distribution drags lambda away from the value that
Gaussianises the speckle.  Measured cost of this on Set12: at ENL=8 the criterion
returns lambda=2.46 (PSNR 19.29) where lambda=1.25 gives 24.48 -- 5.2 dB lost.

Under y = x * n, log y = log x + log n, so the additive noise term is log n with
n ~ Gamma(L, L).  Its law depends only on L:
    E[log n]  = psi(L) - log(L)
    Var[log n]= psi'(L)
    skew      = psi''(L)  / psi'(L)^{3/2}
    ex.kurt   = psi'''(L) / psi'(L)^2
None of that involves the scene.  So the lambda that best Gaussianises the noise
is a function of L alone and can be tabulated offline, independent of any image.
"""

import numpy as np
import torch

from .transforms import yeo_johnson


def _gaussianity_score(z):
    """|skewness| + |excess kurtosis| of a sample, both scale-free."""
    z = (z - z.mean()) / z.std().clamp_min(1e-12)
    return (z.pow(3).mean().abs() + (z.pow(4).mean() - 3.0).abs()).item()


def lambda_star(L, n_samples=400_000, grid=None, device="cpu", seed=0):
    """Optimal Yeo-Johnson lambda for pure Gamma(L, L) speckle in the log domain.

    Depends only on L -- no image is involved, which is the whole point.
    """
    if grid is None:
        grid = torch.linspace(0.0, 4.0, 401)
    g = torch.Generator(device=device).manual_seed(seed)
    n = torch.distributions.Gamma(
        torch.tensor(float(L), device=device), torch.tensor(float(L), device=device)
    ).sample((n_samples,))
    # resample deterministically for reproducibility across calls
    logn = torch.log(n.clamp_min(1e-12)).double()

    best, best_score = None, float("inf")
    for lam in grid:
        z = yeo_johnson(logn, lam.double().to(device))
        if not torch.isfinite(z).all():
            continue
        s = _gaussianity_score(z)
        if s < best_score:
            best, best_score = float(lam), s
    return best, best_score


def log_gamma_moments(L):
    """Analytic moments of log(n), n ~ Gamma(L, L); a check on the sampling above."""
    from scipy.special import polygamma

    var = polygamma(1, L)
    return {
        "var": var,
        "skew": polygamma(2, L) / var ** 1.5,
        "ex_kurt": polygamma(3, L) / var ** 2,
    }


def speckle_stats(L, lam, n_samples=400_000, device="cpu", seed=0):
    """Mean and std of YJ(log n, lam) for n ~ Gamma(L, L).

    Both depend only on (L, lam), so they can be tabulated offline:

    * ``std``  is the noise level in the transformed domain -- the quantity the
      baseline currently *estimates* from the image by variance differencing.
      Knowing it exactly removes an estimation error from w1.
    * ``mean`` is the log-domain bias.  E[log n] = psi(L) - log(L) != 0, so
      exponentiating back is systematically low (a factor of 1.78 at L=1).
      Subtracting it before the inverse transform is the principled alternative
      to rescaling the output by a mean ratio, which needs a reference to match to.
    """
    torch.manual_seed(seed)
    n = torch.distributions.Gamma(
        torch.tensor(float(L), device=device), torch.tensor(float(L), device=device)
    ).sample((n_samples,))
    z = yeo_johnson(torch.log(n.clamp_min(1e-12)).double(), torch.tensor(float(lam), device=device).double())
    return z.mean().item(), z.std().item()


def build_table(Ls=(1, 2, 3, 4, 6, 8, 12, 16), **kw):
    """Tabulate lambda*(L); interpolate in log L for intermediate looks."""
    return {L: lambda_star(L, **kw)[0] for L in Ls}


_SIGMA_CACHE = {}


def sigma_transformed(L, n_samples=400_000, seed=0):
    """Speckle standard deviation in the Log-Yeo-Johnson domain, from L alone.

    The noise level entering the threshold is otherwise obtained from the
    eigenvalue spectrum of the patch covariance, which is the only quantity in the
    method still estimated blindly rather than derived.  That estimator returns
    roughly twice this value on five of the six sensors we tested, but only 0.76
    times it on 0.1 m imagery whose fine texture fills the covariance spectrum, and
    the method then barely filters.  This function supplies the model-based
    alternative; see scripts/run_real.py and the --sigma_scale option for how it is used.
    """
    if L not in _SIGMA_CACHE:
        g = torch.Generator().manual_seed(seed)
        gam = torch.distributions.Gamma(torch.tensor(float(L)), torch.tensor(float(L)))
        ln = torch.log(gam.sample((n_samples,)).clamp_min(1e-12)).double()
        lam = torch.tensor(float(lambda_star(L)[0]), dtype=torch.float64)
        from .transforms import yeo_johnson
        _SIGMA_CACHE[L] = float(yeo_johnson(ln, lam).std())
    return _SIGMA_CACHE[L]
