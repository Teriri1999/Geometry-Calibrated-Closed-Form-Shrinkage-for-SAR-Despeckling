"""Patch extraction, aggregation and non-local block matching.

Patch positions are indexed row-major, following torch's unfold convention.  The
numbering is internally consistent, so it is only a labelling choice.
"""

import torch
import torch.nn.functional as F


def image_to_patches(img, ps):
    """(H, W) -> (d, L) with d = ps*ps, L = (H-ps+1)*(W-ps+1), stride 1.

    The d axis is ordered row-major within the patch.
    """
    x = img[None, None]
    return F.unfold(x, kernel_size=ps).squeeze(0)


def patches_to_image(vals, weights, hw, ps):
    """Weighted aggregation of overlapping patches back to an image."""
    h, w = hw
    num = F.fold(vals[None], output_size=(h, w), kernel_size=ps).squeeze()
    den = F.fold(weights[None], output_size=(h, w), kernel_size=ps).squeeze()
    return num / den.clamp_min(1e-12)


def seed_positions(maxr, maxc, step):
    """Seed rows/cols: every ``step`` positions, then densely to the last one."""
    r = torch.arange(0, maxr, step)
    if r[-1] + 1 < maxr:
        r = torch.cat([r, torch.arange(r[-1] + 1, maxr)])
    c = torch.arange(0, maxc, step)
    if c[-1] + 1 < maxc:
        c = torch.cat([c, torch.arange(c[-1] + 1, maxc)])
    return r, c


def glrt_distance(neigh, seed, eps=1e-6):
    """Speckle likelihood-ratio distance between patches, in the INTENSITY domain.

    The Euclidean distance ||y1-y2||^2 assumes additive Gaussian noise, so on
    multiplicative speckle it penalises bright areas more than dark ones purely
    because of the noise scaling.  Under y = x*n with n ~ Gamma(L,L), testing
    x1 == x2 by generalised likelihood ratio gives

        d = sum_i log[ (y1i+y2i)/2 / sqrt(y1i*y2i) ] = sum_i log(AM/GM)

    which is scale invariant (y -> c*y leaves it unchanged), zero iff y1 == y2,
    and non-negative by AM >= GM.  This is the one place speckle statistics can
    still be exploited: block matching happens *before* the SVD projection, which
    is what Gaussianises the noise (skew -1.14 -> +0.09).
    """
    am = 0.5 * (neigh + seed)
    gm = (neigh.clamp_min(eps) * seed.clamp_min(eps)).sqrt()
    return torch.log(am.clamp_min(eps) / gm.clamp_min(eps)).mean(0)


def block_matching(P, maxr, maxc, seed_r, seed_c, win, nlsp, chunk=1024,
                   quadrant_balance=False, metric="l2"):
    """Find the ``nlsp`` most similar patches within a +-win window of each seed.

    Returns int64 (lenrc, nlsp) patch indices.  Column 0 is forced to be the seed
    itself, so that every group contains the patch it was built around.
    """
    device = P.device
    d = P.shape[0]
    # Grid of window offsets, shared by every seed; out-of-range entries are masked.
    off = torch.arange(-win, win + 1, device=device)
    dr, dc = torch.meshgrid(off, off, indexing="ij")
    dr, dc = dr.reshape(-1), dc.reshape(-1)

    sr, sc = torch.meshgrid(seed_r.to(device), seed_c.to(device), indexing="ij")
    sr, sc = sr.reshape(-1), sc.reshape(-1)
    lenrc = sr.numel()
    self_idx = sr * maxc + sc

    out = torch.empty(lenrc, nlsp, dtype=torch.long, device=device)
    for s in range(0, lenrc, chunk):
        e = min(s + chunk, lenrc)
        rr = sr[s:e, None] + dr[None]           # (n, W)
        cc = sc[s:e, None] + dc[None]
        valid = (rr >= 0) & (rr < maxr) & (cc >= 0) & (cc < maxc)
        cand = (rr.clamp(0, maxr - 1) * maxc + cc.clamp(0, maxc - 1))

        neigh = P[:, cand.reshape(-1)].view(d, e - s, -1)   # (d, n, W)
        seed = P[:, self_idx[s:e]].view(d, e - s, 1)
        if metric == "glrt":
            dist = glrt_distance(neigh, seed)
        else:
            dist = (neigh - seed).pow(2).mean(0)            # (n, W)
        dist = dist.masked_fill(~valid, float("inf"))

        if quadrant_balance:
            # Split the search window into four quadrants and take nlsp/4 from
            # each.  In flat, anisotropic areas (sky, water) horizontal
            # self-similarity dominates, so plain top-k draws its blocks from a
            # narrow band of rows spread widely across columns -- measured mean
            # row offset +3.8 with std 6.2 against std 10.0 for columns.  Every
            # pixel is then reconstructed from the same few rows, which is exactly
            # the horizontal striping.  Balancing quadrants forces the aggregation
            # to be isotropic without touching the similarity criterion itself.
            per = max(nlsp // 4, 1)
            picks = []
            for qr in (dr < 0, dr >= 0):
                for qc in (dc < 0, dc >= 0):
                    q = (qr & qc)[None].expand_as(dist)
                    dq = dist.masked_fill(~q, float("inf"))
                    picks.append(dq.topk(per, dim=1, largest=False).indices)
            order = torch.cat(picks, dim=1)[:, :nlsp]
            # pad if quadrants could not supply enough (image borders)
            if order.shape[1] < nlsp:
                extra = dist.topk(nlsp - order.shape[1], dim=1, largest=False).indices
                order = torch.cat([order, extra], dim=1)
        else:
            order = dist.topk(nlsp, dim=1, largest=False).indices
        sel = cand.gather(1, order)
        # Drop any duplicate of the seed, then pin the seed to column 0.
        sel = torch.where(sel == self_idx[s:e, None], sel[:, :1], sel)
        sel[:, 0] = self_idx[s:e]
        out[s:e] = sel
    return out
