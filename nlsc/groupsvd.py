"""Batched thin SVD of grouped patch matrices, via the K x K Gram matrix.

``torch.linalg.svd`` on many small (d x K) matrices is dominated by cuSOLVER's
per-matrix loop -- profiled at 21 s for 7225 groups of 36x20 in float64, which is
the entire runtime of the baseline.  Since K <= d, eigendecomposing the K x K Gram
matrix instead is ~200x faster and gives identical singular values.

The catch is U = Yc V / S, which is meaningless for the near-zero singular values
(and numerically explodes in float32).  Those directions carry no signal -- the
group matrix is rank-deficient after DC removal, so S_K is exactly 0 -- and the
sparse-coding threshold sigma^2/(S+eps) would zero their coefficients anyway.  We
mask the corresponding columns of U to zero, which makes that implicit truncation
explicit and numerically safe.
"""

import torch


def group_svd(Yc, rtol=None):
    """Thin SVD of a batch of (g, d, K) matrices with K <= d.

    Returns ``U (g, d, K)`` with unreliable columns zeroed, and ``S (g, K)``
    in descending order.
    """
    if rtol is None:
        rtol = 1e-7

    # Forming the Gram matrix squares the condition number, so this step stays in
    # float64 even when the network runs in float32 -- measured in float32 the
    # recovered U blows up to ~1e7.  The matrices are K x K (tiny), so the cost is
    # negligible next to the rest of the stage.
    Yd = Yc.double()
    G = Yd.transpose(1, 2) @ Yd                       # (g, K, K), symmetric PSD
    evals, V = torch.linalg.eigh(G)                   # ascending
    evals = evals.flip(-1).clamp_min(0)               # descending
    V = V.flip(-1)
    S = evals.sqrt()                                  # (g, K)

    keep = S > rtol * S[:, :1].clamp_min(torch.finfo(S.dtype).tiny)
    U = Yd @ V / torch.where(keep, S, torch.ones_like(S))[:, None, :]
    U = U * keep[:, None, :].to(U.dtype)
    return U.to(Yc.dtype), S.to(Yc.dtype)
