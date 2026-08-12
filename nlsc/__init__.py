"""Parameter-free SAR despeckling by non-local sparse coding.

The estimator groups similar patches, codes each group on its own left singular
basis, and shrinks the coefficients by a single soft threshold.  Because that
basis is orthonormal the weighted lasso has a closed-form solution, so no
iterative solver is involved, and every constant follows from the number of
looks and the shape of the group:

    lambda*(L)      the Yeo-Johnson parameter          (lambda_star)
    c*(gamma, L)    the noise-scale correction         (c_star)
    tau_ik          the resulting threshold            (baseline)

Typical use::

    import cv2, torch
    from nlsc import despeckle_image

    obs = torch.tensor(cv2.imread("scene.png", 0).astype("float32"), device="cuda")
    est = despeckle_image(obs, L=4)

``L`` may be omitted on real data, in which case it is estimated from the
observation itself; see ``nlsc.enl_estimate``.
"""

from .c_star import c_star
from .enl_estimate import estimate_enl, snap
from .lambda_star import lambda_star, sigma_transformed
from .pipeline import despeckle_once

__all__ = ["despeckle_image", "despeckle_once", "c_star", "lambda_star",
           "sigma_transformed", "estimate_enl", "snap"]

__version__ = "1.0.0"


# At one look c*(gamma, L) is not calibrated -- the optimum runs past the top of
# the search grid without turning over -- so the fixed constant below is used
# instead, which is the value the one-look results were produced with.
C_ONE_LOOK = 1.5


def despeckle_image(noisy, L=None, ps=10, nlsp=20, rank_select=None, c=None,
                    sigma_scale=None, smooth=1.0):
    """Despeckle one intensity image.

    Parameters
    ----------
    noisy : torch.Tensor
        Two-dimensional intensity image, strictly positive.
    L : int, optional
        Number of looks.  Estimated from ``noisy`` when omitted.
    ps, nlsp : int
        Patch size and stacking number; together they fix the aspect ratio
        ``gamma = ps**2 / nlsp`` from which the threshold constant follows.
    rank_select : float, optional
        Atoms whose singular value falls below ``rank_select * sigma * sqrt(K)``
        are discarded.  Defaults to ``1.5``, or to ``1.0`` at a single look,
        where the stricter cutoff discards genuine structure with the speckle.
    c : float, optional
        Threshold constant.  Defaults to ``c_star(ps, nlsp, L)`` for ``L >= 2``
        and to ``C_ONE_LOOK`` below that, where the law is not calibrated.
    sigma_scale : float, optional
        Replaces the blind noise estimate by ``sigma_scale * sigma_transformed(L)``.
        Needed only where the eigenvalue estimator fails, which in our study was
        the finest-resolution imagery; see the paper's discussion of miniSAR.
    smooth : float
        Gaussian bandwidth of the block-matching reference.  The optimum is flat
        -- 0 to 1 spans about 0.4 dB -- so it is fixed rather than searched.  The
        synthetic results use 1.0, the default here; the real-SAR results were
        produced with 0.0, which ``scripts/run_real.py`` passes explicitly.

    Returns
    -------
    torch.Tensor
        The despeckled intensity image, same shape and dtype as ``noisy``.
    """
    if L is None:
        L = snap(estimate_enl(noisy.detach().cpu().numpy()))
    if c is None:
        c = c_star(ps, nlsp, L) if L >= 2 else C_ONE_LOOK
    if rank_select is None:
        rank_select = 1.5 if L >= 2 else 1.0
    par_kw = dict(ps=ps, nlsp=nlsp, lambda2=c,
                  rank_select=rank_select, rank_decouple=True)
    n_sig = None if sigma_scale is None else sigma_scale * sigma_transformed(L)
    return despeckle_once(noisy.clamp_min(1e-3), L, match_smooth=smooth,
                          par_kw=par_kw, n_sig=n_sig)
