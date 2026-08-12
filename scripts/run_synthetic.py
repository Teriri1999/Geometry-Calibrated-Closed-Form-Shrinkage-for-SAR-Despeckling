"""Reproduce the synthetic-benchmark results (Tables II-IV of the paper).

Expects the layout the released noisy sets use:

    <root>/<dataset>/<base>.png                                ground truth
    <root>/<dataset>_noisy/Noisy/<base>_gamma_noise_ENL<L>.png  observation
    <root>/<dataset>_noisy/<outdir>/<base>_gamma_noise_ENL<L>.png  written here

PSNR and SSIM use data_range = gt.max() - gt.min() on uint8 greyscale, which is
the convention the comparison numbers were produced with.

Everything follows from the look number: lambda*(L) for the transform and
c*(gamma, L) for the threshold, except at one look, where c*(gamma, L) is not
calibrated and the fixed nlsc.C_ONE_LOOK is used instead.  The matching reference
is smoothed with b = 1 here, unlike the real-SAR runs.

    python scripts/run_synthetic.py --root /data --dataset Set12 --enl 1 2 4 8
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlsc import despeckle_image  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="directory holding the datasets")
    p.add_argument("--dataset", default="Set12", help="Set12 | McM | Kodak24")
    p.add_argument("--enl", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--outdir", default="Ours")
    p.add_argument("--ps", type=int, default=10)
    p.add_argument("--nlsp", type=int, default=20)
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", default=None, help="write per-look metrics here")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    gt_dir = os.path.join(args.root, args.dataset)
    noisy_root = os.path.join(args.root, f"{args.dataset}_noisy")
    src = os.path.join(noisy_root, "Noisy")
    dst = os.path.join(noisy_root, args.outdir)
    os.makedirs(dst, exist_ok=True)

    summary = {}
    for L in args.enl:
        files = sorted(f for f in os.listdir(src) if f.endswith(f"_gamma_noise_ENL{L}.png"))
        if args.limit:
            files = files[:args.limit]
        ps_, ss_ = [], []
        for f in files:
            gt = cv2.imread(os.path.join(gt_dir, f.split("_gamma")[0] + ".png"),
                            cv2.IMREAD_GRAYSCALE)
            ny = cv2.imread(os.path.join(src, f), cv2.IMREAD_GRAYSCALE)
            if gt is None or ny is None:
                print(f"[skip] {f}", flush=True)
                continue
            x = torch.tensor(ny.astype(np.float32), device=args.device).clamp_min(1e-3)
            t0 = time.time()
            est = despeckle_image(x, L=L, ps=args.ps, nlsp=args.nlsp, smooth=args.smooth)
            arr = est.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
            cv2.imwrite(os.path.join(dst, f), arr)
            dr = float(gt.max() - gt.min())
            ps_.append(sk_psnr(gt, arr, data_range=dr))
            ss_.append(sk_ssim(gt, arr, data_range=dr))
            print(f"  ENL={L} {f}: {ps_[-1]:.2f} / {100*ss_[-1]:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.cuda.empty_cache() if args.device == "cuda" else None
        if ps_:
            summary[L] = {"psnr": float(np.mean(ps_)), "ssim": float(np.mean(ss_)),
                          "n": len(ps_)}
            print(f"ENL={L}: PSNR {summary[L]['psnr']:.2f}  "
                  f"SSIM {100*summary[L]['ssim']:.2f}  ({len(ps_)} images)", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
