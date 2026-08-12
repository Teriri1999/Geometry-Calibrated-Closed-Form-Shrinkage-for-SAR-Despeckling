"""Despeckle one image or a folder of images.

    python despeckle.py scene.png out.png --looks 4
    python despeckle.py input_dir/ output_dir/          # L estimated per image

With no --looks the number of looks is estimated from each observation and
snapped to the tabulated grid, which is what the real-data experiments do.
Everything else follows from it: the Yeo-Johnson parameter from lambda*(L) and
the threshold constant from c*(gamma, L).  Nothing is tuned per image and the
result is deterministic.
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

from nlsc import despeckle_image, estimate_enl, snap
from nlsc.robust import repair_dark_outliers

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def run_one(path, out_path, args, device):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print(f"[skip] {path}: unreadable", flush=True)
        return
    L = args.looks or snap(estimate_enl(im))
    x = torch.tensor(im.astype(np.float32), device=device).clamp_min(1e-3)
    t0 = time.time()
    est = despeckle_image(x, L=L, ps=args.ps, nlsp=args.nlsp, c=args.c,
                          rank_select=args.rank_select,
                          sigma_scale=args.sigma_scale, smooth=args.smooth)
    if not args.no_repair:
        # The log transform turns the heavy left tail of the speckle into outliers
        # no patch can survive; this repairs the isolated dark dots they leave.
        est = repair_dark_outliers(est, x, L, target_frac=0.0025)
    arr = est.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    cv2.imwrite(out_path, arr)
    print(f"{os.path.basename(path)}: L={L} {im.shape} "
          f"({time.time() - t0:.0f}s) -> {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="image file or directory")
    p.add_argument("output", help="output file or directory")
    p.add_argument("--looks", type=int, default=None,
                   help="number of looks; estimated from the observation if omitted")
    p.add_argument("--ps", type=int, default=10, help="patch size (default 10)")
    p.add_argument("--nlsp", type=int, default=20, help="stacking number (default 20)")
    p.add_argument("--c", type=float, default=None,
                   help="threshold constant; default c*(gamma, L)")
    p.add_argument("--rank_select", type=float, default=None,
                   help="rank cutoff; default 1.5, or 1.0 at one look")
    p.add_argument("--sigma_scale", type=float, default=None,
                   help="use sigma_scale * sigma_model(L) instead of the blind estimate")
    p.add_argument("--smooth", type=float, default=1.0,
                   help="Gaussian bandwidth of the block-matching reference "
                        "(1.0 as in the synthetic experiments; the real-SAR "
                        "results were produced with 0.0)")
    p.add_argument("--no_repair", action="store_true",
                   help="skip the dark-outlier repair applied by default")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(f for f in os.listdir(args.input)
                       if os.path.splitext(f)[1].lower() in EXTS)
        for f in files:
            run_one(os.path.join(args.input, f),
                    os.path.join(args.output, f), args, args.device)
        print(f"=== {len(files)} images -> {args.output}", flush=True)
    else:
        d = os.path.dirname(args.output)
        if d:
            os.makedirs(d, exist_ok=True)
        run_one(args.input, args.output, args, args.device)


if __name__ == "__main__":
    main()
