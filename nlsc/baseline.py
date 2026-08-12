"""The non-local sparse estimator: grouping, group SVD, and shrinkage.

Each group of similar patches is coded on its own left singular basis.  Because
that basis is orthonormal the weighted lasso separates over coefficients, so the
estimate is a *single* soft-threshold step rather than an iterative solve.

Symbol convention:
    sigma_col : per-patch noise level                   -> the w1 role
    wsc       : per-atom threshold sigma_col^2 / S      -> the w2 role
"""

import torch
import torch.nn.functional as F

from .groupsvd import group_svd
from .patches import (
    block_matching,
    image_to_patches,
    patches_to_image,
    seed_positions,
)


class Params:
    ps = 6            # patch size
    step = 3          # stride between group seeds
    win = 20          # block-matching search half-window
    outer_loop = 12   # outer iterations
    inner_loop = 2    # re-run block matching every inner_loop iterations
    nlsp = 20         # patches per group, K
    lambda2 = 1.5     # the threshold constant c
    delta = 0.0       # iterative regularisation; unused
    match_smooth = 0.0  # initial Gaussian sigma for the block-matching reference
    # Scales sigma inside the singular-value debiasing only.  Default 0 disables
    # debiasing entirely: the threshold sigma^2/S already shrinks the coefficients,
    # and debiasing shrinks S as well, which raises that same threshold -- a double
    # shrinkage that zeroed 84% of the singular values and left ~3 atoms per group.
    # Measured on the 2-image protocol: removing it gains 1.55 dB at ENL=8, 0.21 at
    # ENL=1.  Set to 1.0 to recover the published behaviour.
    debias_scale = 0.0
    sigma_floor = 0.2   # eta in the paper's sigma_min = eta * sigma_0
    quadrant_balance = False  # isotropic non-local aggregation (anti-striping)
    # Marchenko-Pastur rank selection.  Off by default: it does suppress the wispy
    # background texture (sky high-frequency energy 4.84 -> 2.97) but costs 4 dB at
    # ENL=8, and the estimate is already SMOOTHER than the ground truth there
    # (4.84 vs 7.23), so the problem is the shape of the residual structure, not
    # its amount.  Kept for ablation.
    mp_rank = False     # discard atoms below the Marchenko-Pastur noise edge
    mp_scale = 1.0      # multiplies that edge; <1 keeps more atoms
    max_rank = 0        # hard cap on atoms kept per group (0 = no cap)
    # Rank selection decoupled from magnitude shrinkage.  The published debiasing
    # S <- sqrt(S^2 - K*sigma^2) does BOTH: it shrinks every singular value AND,
    # because the threshold is sigma^2/S, raises that threshold -- a double
    # shrinkage that zeroes 84% of the atoms and leaves ~3 per group.  That is what
    # removes the artefacts, but it costs 1.8 dB.  Here the same statistic decides
    # which atoms are noise (and drops them outright), while the survivors keep
    # their UNSHRUNK S, so their threshold stays small and detail is preserved.
    rank_select = 0.0   # keep atoms with S > rank_select * sigma * sqrt(K); 0 = off
    # Measure that cutoff against the noise estimate BEFORE lambda2 scales it, so the
    # rank cutoff and the soft threshold are independent knobs.  Default False keeps
    # the published (coupled) behaviour; see the note at the rank_select branch.
    rank_decouple = False
    # Ablation of the two factors of the threshold sigma^2/S.  Because D is
    # orthonormal the weighted lasso collapses to a single threshold
    # tau_ik = c*w2_i/(2*w1_k^2), so w1 and w2 are NOT two independent matrices --
    # they are two factors of tau.  A meaningful ablation freezes one at its group
    # mean rather than "deleting a matrix".
    w_mode = "both"     # both | w1_only (freeze w2) | w2_only (freeze w1) | none
    glrt_match = False  # first-pass block matching by the speckle GLRT distance
    eps = 1e-12
    group_chunk = 4096

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(Params, k):
                raise KeyError(f"unknown parameter {k!r}")
            setattr(self, k, v)


def _gaussian_blur(x, sigma):
    """Separable Gaussian blur, used only to stabilise block matching."""
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-ax.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    y = F.conv2d(x[None, None], g.view(1, 1, 1, -1), padding=(0, k // 2))
    return F.conv2d(y, g.view(1, 1, -1, 1), padding=(k // 2, 0)).squeeze()


def estimate_noise(img, ps, stride=3):
    """Noise level from the low end of the patch covariance spectrum."""
    P = image_to_patches(img, ps)[:, ::1]
    h, w = img.shape
    maxc = w - ps + 1
    idx = torch.arange(P.shape[1], device=img.device)
    keep = ((idx // maxc) % stride == 0) & ((idx % maxc) % stride == 0)
    P = P[:, keep].double()

    mu = P.mean(dim=1, keepdim=True)
    Pc = P - mu
    cov = Pc @ Pc.T / P.shape[1]
    ev = torch.linalg.eigvalsh(cov).sort().values

    # Walk down the spectrum until the mean of the retained eigenvalues splits
    # them evenly -- the point where only noise-like components remain.
    mean_val = ev.mean()
    for n in range(ev.numel(), 0, -1):
        m = ev[:n].mean()
        if int((ev[:n] > m).sum()) == int((ev[:n] < m).sum()):
            mean_val = m
            break
    return mean_val.clamp_min(0).sqrt().to(img.dtype)


def _aggregate(num, den, hw, ps, fallback):
    """Fold weighted patches back to an image, falling back where nothing landed.

    Only pixels whose *every* overlapping patch went unselected have zero weight;
    those keep the previous estimate rather than being divided by ~0.
    """
    h, w = hw
    n_img = F.fold(num[None], output_size=(h, w), kernel_size=ps).squeeze()
    d_img = F.fold(den[None], output_size=(h, w), kernel_size=ps).squeeze()
    empty = d_img <= 0
    out = n_img / d_img.clamp_min(1e-12)
    return torch.where(empty, fallback, out)


def _denoise_groups(Y, blk, sigma_col, par, sigma_raw=None):
    """One pass of grouped sparse coding. Returns aggregated (values, weights).

    Y         : (d, L) patches of the current estimate
    blk       : (G, K) patch indices per group
    sigma_col : (L,) per-patch noise level, already scaled by lambda2
    sigma_raw : (L,) the same estimate *without* the lambda2 scaling
    """
    d, L = Y.shape
    G, K = blk.shape
    num = torch.zeros(d, L, dtype=Y.dtype, device=Y.device)
    den = torch.zeros(d, L, dtype=Y.dtype, device=Y.device)

    for s in range(0, G, par.group_chunk):
        idx = blk[s : s + par.group_chunk]                  # (g, K)
        g = idx.shape[0]
        nlY = Y[:, idx.reshape(-1)].view(d, g, K).permute(1, 0, 2)   # (g, d, K)
        dc = nlY.mean(dim=2, keepdim=True)
        Yc = nlY - dc

        sig = sigma_col[idx]                                # (g, K)
        # Aggregation weight: reliable (low-noise) patches count for more.
        wcol = 1.0 / (sig + par.eps)

        D, S = group_svd(Yc)                                 # D: (g, d, r)
        # Debias the singular values by the energy the speckle contributes,
        # taking the seed patch's sigma as representative of the whole group.
        #
        # lambda2 plays two distinct roles in the original formulation: it scales
        # sigma_col, which then feeds BOTH this debiasing and the threshold below.
        # Debiasing is a statistical correction -- speckle adds K*sigma^2 to each
        # squared singular value, with the *true* sigma -- whereas lambda2 is a
        # regularisation strength.  Coupling them squares lambda2's effect: at
        # lambda2=1.5 the debiasing over-subtracts by 2.25x, zeroing 84% of the
        # singular values and leaving only ~3 atoms per group, which is what the
        # striping artefacts are.  debias_scale=1/lambda2 decouples the two.
        sig_deb = sig * par.debias_scale
        S_keep = S                                                    # unshrunk
        S = (S.pow(2) - K * sig_deb[:, :1].pow(2)).clamp_min(0).sqrt()   # (g, r)
        if par.rank_select:
            S = S_keep                       # thresholds use the unshrunk values

        # Threshold is per (atom, patch): sigma_col^2 / S_i
        sig_t = sig
        S_t = S
        if par.w_mode in ("w2_only", "none"):
            sig_t = sig.mean(dim=1, keepdim=True).expand_as(sig)   # freeze w1
        if par.w_mode in ("w1_only", "none"):
            S_t = S.mean(dim=1, keepdim=True).expand_as(S)         # freeze w2
        wsc = sig_t.pow(2)[:, None, :] / (S_t[:, :, None] + par.eps)  # (g, r, K)
        B = D.transpose(1, 2) @ Yc                                    # (g, r, K)
        C = torch.sign(B) * (B.abs() - wsc).clamp_min(0)

        if par.rank_select:
            # Which atoms are noise is a statistical question about sigma, not about
            # how hard we choose to regularise -- but sig has already been multiplied
            # by lambda2, so the cutoff moves with c and the two get confounded: the
            # c*(gamma, L) law lowers c for small gamma, which silently weakens
            # artefact suppression at the same time.  Same argument as debias_scale
            # above.  rank_decouple uses the unscaled estimate so the rank cutoff and
            # the threshold can be set independently.
            s_rank = sigma_raw[idx] if (par.rank_decouple and sigma_raw is not None) else sig
            keep = S_keep > par.rank_select * s_rank[:, :1] * (K ** 0.5)
            C = C * keep.to(C.dtype)[:, :, None]
        if par.max_rank:
            # Hard rank cap.  In flat-but-shaded areas the leading singular vector
            # captures the real gradient while the next few capture speckle; with
            # w2 = 1/S their thresholds are too small to remove them, and after
            # overlapping aggregation they merge into the organic blotches.  A cap
            # is far more controllable than the Marchenko-Pastur edge, which uses a
            # single global sigma and over-truncates structured groups.
            C[:, par.max_rank:, :] = 0
        if par.mp_rank:
            # Marchenko-Pastur rank selection.
            #
            # w2 = 1/S gives the LARGEST singular value the SMALLEST threshold, on
            # the assumption that a large S means an important feature.  In a flat
            # region that assumption fails: once the DC is removed the group is pure
            # speckle, and its leading singular value is merely the strongest noise
            # mode -- which w2 then preserves almost intact.  That is the wispy
            # background texture.
            #
            # Random matrix theory says a d x K matrix of iid noise with std sigma
            # has its largest singular value at sigma*(sqrt(d)+sqrt(K)).  Anything
            # below that is statistically indistinguishable from noise and should be
            # discarded outright rather than lightly thresholded.  Note the existing
            # debiasing implies a cutoff of only sigma*sqrt(K) (4.5 sigma here versus
            # 10.5), which is why it never removed these modes.
            mp = par.mp_scale * sig[:, :1] * (d ** 0.5 + K ** 0.5)    # (g, 1)
            C = C * (S > mp).to(C.dtype)[:, :, None]
        Yhat = D @ C + dc                                             # (g, d, K)

        flat = idx.reshape(-1)
        num.index_add_(1, flat, (Yhat * wcol[:, None, :]).permute(1, 0, 2).reshape(d, -1))
        den.index_add_(1, flat, wcol[:, None, :].expand(g, d, K).permute(1, 0, 2).reshape(d, -1))
    return num, den


@torch.no_grad()
def despeckle(nim, par=None, n_sig=None, verbose=False, match_ref=None, init=None,
              intensity=None):
    """Run the baseline on a single-channel image already in the transformed domain.

    ``nim`` is expected to be the Log-Yeo-Johnson transformed observation; the
    caller handles the forward/inverse transform.

    ``match_ref`` optionally supplies a separate image on which to compute the
    non-local block matching.  At ENL=1 the Euclidean distance between raw noisy
    patches is nearly uninformative, so which patches get grouped is largely
    chance; passing a pre-filtered (or, for diagnostics, oracle) reference
    decouples "who gets grouped" from "what gets denoised".
    """
    par = par or Params()
    device = nim.device
    h, w = nim.shape
    maxr, maxc = h - par.ps + 1, w - par.ps + 1

    if n_sig is None:
        n_sig = estimate_noise(nim, par.ps)
    sigma_glob = torch.as_tensor(n_sig, dtype=nim.dtype, device=device)

    seed_r, seed_c = seed_positions(maxr, maxc, par.step)
    NY = image_to_patches(nim, par.ps)
    # ``init`` seeds the iteration from an existing estimate (used by the coarse-to-
    # fine scheme).  NY stays the *observation's* patches either way, so the noise
    # bookkeeping and the data term are unaffected -- only the starting point moves.
    im_out = nim.clone() if init is None else init.clone()
    blk = None

    for ite in range(par.outer_loop):
        im_out = im_out + par.delta * (nim - im_out)
        Y = image_to_patches(im_out, par.ps)

        # Local noise level: what is left of the global variance after the
        # current estimate has explained part of it.
        resid = (NY - Y).pow(2).mean(dim=0)
        sigma_raw = (sigma_glob.pow(2) - resid).abs().sqrt()
        sigma_col = par.lambda2 * sigma_raw
        # Floor it at sigma_min = eta * sigma_0, which the threshold needs and an
        # epsilon guard does not provide.  Where
        # resid ~ sigma_glob^2 the estimate collapses toward zero and the
        # aggregation weight 1/sigma explodes: measured max 331 against a median
        # of 0.75, so a single patch outweighs the ~36 others covering that pixel
        # and paints an isolated dark dot.
        sigma_col = sigma_col.clamp_min(par.sigma_floor * sigma_glob)
        sigma_raw = sigma_raw.clamp_min(par.sigma_floor * sigma_glob)

        if ite % par.inner_loop == 0:
            metric = "l2"
            if intensity is not None and ite == 0:
                # First pass matches on the raw intensities with the speckle-derived
                # GLRT distance.  Later passes match on the current (already
                # denoised) estimate, where Euclidean is appropriate again because
                # the residual there is near-Gaussian.
                ref = image_to_patches(intensity, par.ps)
                metric = "glrt"
            elif match_ref is not None:
                ref = image_to_patches(match_ref, par.ps)
            elif par.match_smooth > 0:
                # Matching reliability scales with SNR, which is worst at the start.
                # Smooth the reference hard early and taper off; this only changes
                # which patches are grouped, never what is reconstructed.
                sm = par.match_smooth * (1.0 - ite / max(par.outer_loop - 1, 1))
                ref = image_to_patches(_gaussian_blur(im_out, sm), par.ps) if sm > 0.1 else Y
            else:
                ref = Y
            blk = block_matching(ref, maxr, maxc, seed_r, seed_c, par.win, par.nlsp,
                                 quadrant_balance=par.quadrant_balance, metric=metric)

        num, den = _denoise_groups(Y, blk, sigma_col, par, sigma_raw)
        # Patches that no group selected must contribute *nothing* (num=den=0):
        # patches overlap with stride 1, so each pixel still gets
        # ~ps^2 other contributions.  Writing the observation back into them with
        # weight 1.0 instead re-injects raw speckle: with step=3/K=20 that is 28% of
        # all patches, which is exactly the black-speckle and blockiness we saw.
        im_out = _aggregate(num, den, (h, w), par.ps, fallback=im_out)

        if verbose:
            print(f"  iter {ite + 1}/{par.outer_loop}  sigma_col mean={sigma_col.mean():.5f}")
    return im_out
