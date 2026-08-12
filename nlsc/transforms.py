"""Log-Yeo-Johnson transform, differentiable in lambda.

Written so that gradients flow to ``lam``.  The log is applied first and the
Yeo-Johnson map second, which is why that map has to handle negative inputs.
"""

import torch

# Below this |lam| distance to a pole (0 for the positive branch, 2 for the
# negative one) we switch to the analytic limit to avoid 0/0.
_POLE_EPS = 1e-4


def _yj_pos(x, lam):
    """Yeo-Johnson on the x >= 0 branch: ((x+1)^lam - 1)/lam, -> log(x+1) at lam=0.

    ``x`` is clamped inside the branch: torch.where evaluates both branches, so a
    NaN produced on the discarded side would still poison the gradient.
    """
    near = lam.abs() < _POLE_EPS
    log1p = torch.log1p(x.clamp_min(0))
    # Taylor of ((x+1)^lam - 1)/lam around lam=0 keeps the gradient w.r.t. lam finite.
    taylor = log1p * (1.0 + 0.5 * lam * log1p)
    safe_lam = torch.where(near, torch.ones_like(lam), lam)
    exact = torch.expm1((safe_lam * log1p).clamp(max=60.0)) / safe_lam
    return torch.where(near, taylor, exact)


def _yj_neg(x, lam):
    """Yeo-Johnson on the x < 0 branch: -((-x+1)^(2-lam) - 1)/(2-lam)."""
    p = 2.0 - lam
    near = p.abs() < _POLE_EPS
    log1p = torch.log1p(-x.clamp_max(0))
    taylor = -log1p * (1.0 + 0.5 * p * log1p)
    safe_p = torch.where(near, torch.ones_like(p), p)
    exact = -torch.expm1((safe_p * log1p).clamp(max=60.0)) / safe_p
    return torch.where(near, taylor, exact)


def yeo_johnson(x, lam):
    """Yeo-Johnson power transform. ``lam`` may be a 0-dim tensor requiring grad."""
    lam = torch.as_tensor(lam, dtype=x.dtype, device=x.device)
    return torch.where(x >= 0, _yj_pos(x, lam), _yj_neg(x, lam))


def yeo_johnson_inv(y, lam):
    """Inverse of :func:`yeo_johnson`."""
    lam = torch.as_tensor(lam, dtype=y.dtype, device=y.device)

    near0 = lam.abs() < _POLE_EPS
    safe_lam = torch.where(near0, torch.ones_like(lam), lam)
    # expm1/log1p keep precision for the small-|lam| regime.
    pos = torch.where(
        near0,
        torch.expm1(y),
        torch.expm1(torch.log1p(torch.clamp(safe_lam * y, min=-1.0 + 1e-6)) / safe_lam),
    )

    p = 2.0 - lam
    near2 = p.abs() < _POLE_EPS
    safe_p = torch.where(near2, torch.ones_like(p), p)
    neg = torch.where(
        near2,
        -torch.expm1(-y),
        -torch.expm1(torch.log1p(torch.clamp(-safe_p * y, min=-1.0 + 1e-6)) / safe_p),
    )
    return torch.where(y >= 0, pos, neg)


def log_yeo_johnson(img, lam, eps=1e-8):
    """Forward pipeline: multiplicative speckle -> additive -> approximately Gaussian."""
    return yeo_johnson(torch.log(img.clamp_min(eps)), lam)


def log_yeo_johnson_inv(t, lam):
    """Inverse pipeline, returning to intensity domain."""
    return torch.exp(yeo_johnson_inv(t, lam))


def fit_lambda(img, candidates=None, eps=1e-8):
    """Pick lambda by minimising |skewness| + |excess kurtosis| of the transformed image.

    This is the exhaustive search the paper describes; used to initialise the
    learnable lambda so the unrolled model starts from the published operating point.
    """
    if candidates is None:
        candidates = torch.linspace(-2.0, 8.0, 201)
    logimg = torch.log(img.clamp_min(eps)).flatten()
    best, best_score = None, float("inf")
    for lam in candidates:
        t = yeo_johnson(logimg, lam.to(logimg.device))
        std = t.std()
        if not torch.isfinite(std) or std < 1e-12:
            continue
        z = (t - t.mean()) / std
        score = (z.pow(3).mean().abs() + (z.pow(4).mean() - 3.0).abs()).item()
        if score < best_score:
            best, best_score = float(lam), score
    return best
