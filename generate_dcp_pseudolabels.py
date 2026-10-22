"""
Generates pseudo-label ("prior-dehazed") images for the real-hazy branch of
DNMGDT training, using ONLY the Dark Channel Prior (DCP) generator already
built into this repo's model.py (DCPDehazeGenerator).

Why: the original repo's SemiDatasets loader (load_dataset.py) expects FIVE
prior-dehazed images per real hazy image -- filenames with 'hazy' replaced by
'NLD', 'IDRLP', 'DCP', 'IDE', 'FGT'. The NLD/IDRLP/IDE prior methods and the
FGT (fused) image are not included in this repo and were not released.

Rather than editing the loader/loss code (risk of introducing new bugs) or
sourcing three separate external classical-dehazing implementations, this
script runs DCP (free, already in model.py) once per image and writes the
SAME dehazed output under all five expected filenames. This reproduces the
paper's own "single-prior guidance" ablation configuration (Table III) with
zero changes to load_dataset.py / train_DNMGDT.py -- IQGAW will compute
weights over four identical candidates, which is a harmless no-op, and
training proceeds exactly as the existing code expects.

Usage:
    python generate_dcp_pseudolabels.py --input_dir /path/to/real_hazy --output_dir /path/to/real_hazy_with_priors

Input images should have filenames containing the substring 'hazy'
(e.g. 0001_hazy.png), matching what SemiDatasets expects.
"""
import argparse
import os
import glob

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import DCPDehazeGenerator

PRIOR_SUFFIXES = ["NLD", "IDRLP", "DCP", "IDE", "FGT"]


def load_image_tensor(path, device):
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(img)  # [0,1]
    t = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(t)  # -> [-1,1]
    return t.unsqueeze(0).to(device), img.size  # (tensor, (W,H))


def tensor_to_pil(t):
    # t is in roughly [0,1] (DCP output range), clamp defensively
    arr = t.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="folder of real hazy images, filenames must contain 'hazy'")
    ap.add_argument("--output_dir", required=True, help="where to write hazy copies + NLD/IDRLP/DCP/IDE/FGT pseudo-labels")
    ap.add_argument("--win_size", type=int, default=7)
    ap.add_argument("--r", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[generate_dcp_pseudolabels] device={device}")

    net_dcp = DCPDehazeGenerator(win_size=args.win_size, r=args.r).to(device)
    net_dcp.eval()

    paths = sorted(
        p for p in glob.glob(os.path.join(args.input_dir, "*"))
        if "hazy" in os.path.basename(p) and p.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not paths:
        raise SystemExit(f"No images with 'hazy' in the filename found under {args.input_dir}")

    print(f"[generate_dcp_pseudolabels] found {len(paths)} real hazy images")

    for i, p in enumerate(paths):
        fname = os.path.basename(p)
        if "hazy" not in fname:
            continue

        x, orig_size = load_image_tensor(p, device)
        with torch.no_grad():
            j_dcp, t_dcp, a_map = net_dcp(x)
        dehazed_pil = tensor_to_pil(j_dcp).resize(orig_size, Image.BICUBIC)

        # copy the original hazy image into output_dir unchanged
        Image.open(p).convert("RGB").save(os.path.join(args.output_dir, fname))

        # write the same DCP output under all 5 expected prior filenames
        for suffix in PRIOR_SUFFIXES:
            out_name = fname.replace("hazy", suffix)
            dehazed_pil.save(os.path.join(args.output_dir, out_name))

        if (i + 1) % 50 == 0 or (i + 1) == len(paths):
            print(f"[generate_dcp_pseudolabels] {i + 1}/{len(paths)} done")

    print(f"[generate_dcp_pseudolabels] wrote hazy + {len(PRIOR_SUFFIXES)} pseudo-label copies per image to {args.output_dir}")


if __name__ == "__main__":
    main()
