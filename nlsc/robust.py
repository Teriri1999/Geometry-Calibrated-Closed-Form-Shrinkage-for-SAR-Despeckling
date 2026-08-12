"""Speckle-aware winsorisation of the observation before the log transform.

Gamma(L, L) has a heavy left tail: at L=1, P(n < 0.01) is about 1%, so roughly one
pixel in a hundred is observed at under a hundredth of its true radiance.  The log
transform maps such a pixel to an offset of -4.6, an extreme outlier that corrupts
every patch containing it; the sparse coding cannot recover from it, and the
inverse exp turns it into a black dot.  Measured on the ENL=1 image: the pixels
sitting 60 grey levels below their local median have a *noisy* median of 12 while
their true median is around 120.

The cutoff is not a tuned constant: for Gamma(L, L) the probability of falling
below the quantile q_alpha is exactly alpha, so thresholding at

    y < median_local(y) * q_alpha(L)

flags the alpha-fraction of genuine left-tail draws, with q_alpha from the speckle
law alone.  Flagged pixels are pulled up to that threshold (winsorised, not
deleted) so no structure is invented.

The same reasoning applies to the right tail, which is much lighter but produces
the bright specks seen at high ENL.
"""

import torch
import torch.nn.functional as F
from scipy.stats import gamma as _gamma


def local_median(x, ksize=5):
    """Median over a ksize x ksize window."""
    pad = ksize // 2
    xp = F.pad(x[None, None], (pad, pad, pad, pad), mode="reflect")
    patches = xp.unfold(2, ksize, 1).unfold(3, ksize, 1)
    return patches.reshape(*x.shape, -1).median(dim=-1).values


def repair_dark_outliers(x, y, L, alpha=0.005, ksize=3, min_dev=0.0,
                         target_frac=0.0025):
    """Post-hoc repair of isolated dark dots, leaving everything else untouched.

    A pixel is repaired only if it is both

      * far below its local median, by more than the estimate's own local scale
        would allow, and
      * *inconsistent with the speckle model*: the ratio y/x_hat it implies sits
        beyond the (1-alpha) quantile of Gamma(L, L), i.e. the observation would
        have to be a wilder speckle draw than alpha of all pixels to justify it.

    Only the dark side is repaired.  In SAR an isolated bright pixel is usually a
    genuine strong scatterer (a point target), and removing those would destroy
    real information; an isolated dark pixel has no such physical counterpart and
    comes from the heavy left tail of the speckle law.

    Repairs are done by local median, so no structure is invented, and typically
    touch well under 0.1% of pixels.
    """
    med = local_median(x, ksize)
    # The dark dots are NOT pixels whose implied speckle draw is impossible -- at
    # those pixels y/x_hat sits around 0.36, a perfectly ordinary draw.  What is
    # extreme is the observation itself: y there is ~27 where the truth is ~120,
    # a left-tail draw carrying almost no information about the scene.  The
    # estimate stays low because it is still anchored to that observation.
    #
    # So the test is on y against the local radiance level: a pixel observed below
    # the alpha-quantile of Gamma(L, L) times its neighbourhood's level is
    # uninformative and should be taken from the neighbourhood instead.  Requiring
    # the estimate to be low as well keeps the intersection small.
    darker = x < med - min_dev
    if target_frac is not None:
        # Fix the fraction of pixels repaired rather than the tail probability.
        # q_alpha depends on L, so a fixed alpha touches 0.24% of pixels at ENL=1
        # but 0.85% at ENL=8 (the law concentrates as L grows), which starts
        # eating genuine dark structure and cost 0.46 dB at ENL=8.
        r = (y / med.clamp_min(1e-6)).flatten()
        k = max(int(target_frac * r.numel()), 1)
        thr = r.kthvalue(k).values
        uninformative = (y / med.clamp_min(1e-6)) <= thr
    else:
        q_lo = float(_gamma.ppf(alpha, a=float(L), scale=1.0 / float(L)))
        uninformative = y < q_lo * med.clamp_min(1e-6)
    return torch.where(uninformative & darker, med, x)


def winsorize(y, L, alpha=0.01, ksize=5, both_tails=True):
    """Clip speckle draws that the Gamma(L, L) law says are extreme outliers.

    ``alpha`` is the tail probability treated as outlying; the corresponding
    quantiles come from the speckle distribution, so nothing here is fitted.
    """
    med = local_median(y, ksize).clamp_min(1e-6)
    q_lo = float(_gamma.ppf(alpha, a=float(L), scale=1.0 / float(L)))
    out = torch.maximum(y, med * q_lo)
    if both_tails:
        q_hi = float(_gamma.ppf(1.0 - alpha, a=float(L), scale=1.0 / float(L)))
        out = torch.minimum(out, med * q_hi)
    return out
