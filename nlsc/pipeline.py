"""Parameter-free despeckling.

Everything that used to be a hand-set constant is now determined from speckle
statistics or from the observation itself:

* the Yeo-Johnson parameter comes from lambda_star(L) -- an offline function of
  the number of looks, never fitted on the image (the published criterion fitted
  it on signal+noise, which dragged it off by up to 5 dB at ENL=8);
* the threshold constant comes from c_star(ps, K, L), fixed by the aspect ratio
  of the group rather than tuned per image.

No clean reference is used anywhere, at any stage, and the estimator is
deterministic: nothing is chosen by comparing candidate reconstructions.
"""

import torch

from .baseline import Params, despeckle, estimate_noise
from .lambda_star import lambda_star
from .robust import winsorize
from .transforms import log_yeo_johnson, log_yeo_johnson_inv

_LAM_CACHE = {}


def _fwd_inv(noisy, L, mode, device, dtype):
    """Forward/inverse transform pair.

    mode="lyj"  : log then Yeo-Johnson with lambda*(L)   (the published pipeline)
    mode="log"  : log only -- multiplicative speckle becomes additive and
                  homoscedastic, but the residual is log-Gamma, not Gaussian
    mode="none" : operate directly on intensities (speckle stays multiplicative)
    """
    if mode == "none":
        return noisy, (lambda z: z), None
    if mode == "log":
        return torch.log(noisy.clamp_min(1e-3)), torch.exp, None
    lam = get_lambda(L, device=device)
    lt = torch.tensor(float(lam), device=device, dtype=dtype)
    return log_yeo_johnson(noisy, lt), (lambda z: log_yeo_johnson_inv(z, lt)), lam

def get_lambda(L, device="cpu"):
    """Cached lambda*(L); depends only on the number of looks."""
    if L not in _LAM_CACHE:
        _LAM_CACHE[L] = lambda_star(L, device=device)[0]
    return _LAM_CACHE[L]


def despeckle_once(noisy, L, match_smooth=0.0, par_kw=None, lam=None, init=None,
                   winsor_alpha=0.0, transform="lyj", n_sig=None):
    """One despeckling run at a given match_smooth, in the intensity domain.

    ``winsor_alpha`` clips the speckle draws that Gamma(L, L) says are extreme
    before the log transform; set to 0 to disable.  It roughly halves the isolated
    dark dots at ENL=1 at no cost in PSNR (they come from the heavy left tail:
    ~1% of 1-look pixels are observed below a hundredth of their true radiance,
    and log turns those into outliers no patch can survive).
    """
    device, dtype = noisy.device, noisy.dtype
    obs = noisy
    if winsor_alpha:
        noisy = winsorize(noisy, L, alpha=winsor_alpha)
    t, inv, _ = _fwd_inv(noisy, L, transform, device, dtype)
    par = Params(match_smooth=match_smooth, **(par_kw or {}))
    init_t = None if init is None else _fwd_inv(init.clamp_min(1e-3), L, transform, device, dtype)[0]
    # ``n_sig`` overrides the blind estimate with a supplied noise level; used where
    # the eigenvalue-spectrum estimator is known to fail (see lambda_star.sigma_transformed).
    sg = estimate_noise(t, par.ps) if n_sig is None else \
        torch.as_tensor(float(n_sig), device=device, dtype=dtype)
    est = despeckle(t, par, n_sig=sg, init=init_t,
                    intensity=obs if getattr(par, 'glrt_match', False) else None)
    out = inv(est)
    # Radiometric normalisation against the observation's own mean -- uses no
    # reference, and compensates the log-domain bias E[log n] = psi(L) - log(L).
    return out * (obs.mean() / out.mean())
