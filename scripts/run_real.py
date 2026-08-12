"""Reproduce the real-SAR results (Tables V-VI of the paper).

Expects one directory per sensor holding the observations:

    <root>/<dataset>/origin/*.png      (Origin/ is also accepted)
    <root>/<dataset>/<outdir>/*.png    written here

The look number is not known for real data, so it is measured per image from the
upper tail of the local ENL distribution and snapped to the tabulated grid.
Nothing else is set by hand.

    python scripts/run_real.py --root /data --dataset Sentinel-1

The one exception reported in the paper is miniSAR, whose 0.1 m texture fills the
low end of the patch covariance spectrum so that the blind estimator assigns most
of the variance to signal.  There the model-based level is substituted at twice
sigma_model(L), the factor the estimator itself returns on the other five
configurations.  That exception is applied automatically by dataset name and is
announced when it fires; --sigma_scale overrides it either way.

Unlike the synthetic experiments, the real-SAR results use no smoothing of the
block-matching reference (b = 0 rather than 1).  The optimum is flat: 0 to 1 spans
about 0.4 dB.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlsc import despeckle_image, estimate_enl, snap  # noqa: E402
from nlsc.robust import repair_dark_outliers  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# The blind estimator needs the low end of the patch covariance spectrum to be
# free of scene content.  On the finest imagery it is not, and the estimate
# collapses; there the model-based level is used instead.
SIGMA_SCALE = {"miniSAR": 2.0}


def origin_dir(root):
    for name in ("origin", "Origin"):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"no origin directory under {root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="directory holding the sensor folders")
    p.add_argument("--dataset", required=True, help="sensor folder name, e.g. Sentinel-1")
    p.add_argument("--outdir", default="Ours")
    p.add_argument("--ps", type=int, default=10)
    p.add_argument("--nlsp", type=int, default=20)
    p.add_argument("--rank_select", type=float, default=1.5)
    # The real-SAR results were produced without smoothing the matching
    # reference, unlike the synthetic ones, which use 1.0.  The optimum is flat
    # either way: 0 to 1 spans about 0.4 dB.
    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--sigma_scale", type=float, default=None,
                   help="use sigma_scale * sigma_model(L) instead of the blind "
                        "estimate; defaults to the per-sensor value below")
    p.add_argument("--json", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sigma_scale = args.sigma_scale
    if sigma_scale is None and args.dataset in SIGMA_SCALE:
        sigma_scale = SIGMA_SCALE[args.dataset]
        print(f"[{args.dataset}] blind noise estimate replaced by "
              f"{sigma_scale} x sigma_model(L); see the module docstring", flush=True)

    root = os.path.join(args.root, args.dataset)
    src = origin_dir(root)
    dst = os.path.join(root, args.outdir)
    os.makedirs(dst, exist_ok=True)

    record = {}
    for f in sorted(x for x in os.listdir(src)
                    if os.path.splitext(x)[1].lower() in EXTS):
        im = cv2.imread(os.path.join(src, f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"[skip] {f}: unreadable", flush=True)
            continue
        L = snap(estimate_enl(im))
        x = torch.tensor(im.astype(np.float32), device=args.device).clamp_min(1e-3)

        t0 = time.time()
        est = despeckle_image(x, L=L, ps=args.ps, nlsp=args.nlsp,
                              rank_select=args.rank_select,
                              sigma_scale=sigma_scale, smooth=args.smooth)
        # The log transform turns the heavy left tail of the speckle into outliers
        # that no patch can survive; this repairs the isolated dark dots they leave.
        est = repair_dark_outliers(est, x, L, target_frac=0.0025)
        arr = est.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
        cv2.imwrite(os.path.join(dst, f), arr)
        record[f] = {"L": L, "sigma_scale": sigma_scale, "shape": list(im.shape),
                     "seconds": round(time.time() - t0, 1)}
        print(f"{args.dataset}/{f}: L={L} {im.shape} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"=== {args.dataset}: {len(record)} images -> {dst}", flush=True)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=2)


if __name__ == "__main__":
    main()
