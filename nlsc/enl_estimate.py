"""Estimate the equivalent number of looks of a real SAR image.

The despeckler needs $L$: it enters the transform through lambda*(L) and the
threshold through c*(gamma, L).  On synthetic data $L$ is known; on real data it
has to be measured, and measuring it on the whole image is wrong -- scene
variance inflates the denominator of mean^2/var and drives the estimate far below
the true look number (whole-image values on these six sensors come out between
0.6 and 4.9, which for FARAD would imply less than one look).

ENL is only meaningful over a homogeneous area, so we compute mean^2/var in
sliding windows and read the estimate off the upper tail of that distribution:
windows that straddle an edge or a bright target have high variance and low
apparent ENL, while the most homogeneous windows approach the true value.  We
take a high quantile rather than the maximum, which would be set by whichever
single window happened to be flattest.
"""

import numpy as np
import torch
import torch.nn.functional as F


def local_enl_map(img, win=32, stride=8):
    """mean^2 / var over every ``win`` x ``win`` window, on a ``stride`` grid."""
    x = torch.as_tensor(np.asarray(img), dtype=torch.float64)[None, None]
    k = torch.ones(1, 1, win, win, dtype=torch.float64) / (win * win)
    m = F.conv2d(x, k, stride=stride)
    m2 = F.conv2d(x * x, k, stride=stride)
    var = (m2 - m * m).clamp_min(1e-12)
    return (m * m / var).squeeze()


def estimate_enl(img, win=32, stride=8, q=0.95, min_mean=1.0):
    """Look number of a real SAR image, read off the upper tail of the local ENL.

    ``min_mean`` discards near-black windows, whose ratio is dominated by the
    quantisation floor rather than by speckle.
    """
    x = torch.as_tensor(np.asarray(img), dtype=torch.float64)[None, None]
    k = torch.ones(1, 1, win, win, dtype=torch.float64) / (win * win)
    m = F.conv2d(x, k, stride=stride).squeeze()
    e = local_enl_map(img, win, stride)
    e = e[m > min_mean]
    if e.numel() == 0:
        return 1.0
    return float(torch.quantile(e.flatten(), q))


def snap(L, grid=(1, 2, 3, 4, 6, 8, 12, 16)):
    """Nearest tabulated look number, in log space since c* and lambda* vary in log L."""
    g = np.asarray(grid, dtype=float)
    return int(g[np.argmin(np.abs(np.log(g) - np.log(max(L, 1.0))))])
