"""
test_simulate_unet_output.py

Purpose
-------
Simulate the output of a 2D U-Net inference by generating probability maps (.npy)
from a ground-truth NIfTI mask.

Input (per case)
----------------
data/nii_pre_unet/segmentation-{index}.nii   (or .nii.gz)

Output (per case)
-----------------
data/maps_post_unet/post_unet_{index}.npy

Constraints guaranteed
----------------------
- Output is a numpy array of shape (Z, H, W)
- dtype float32
- values in [0, 1]

How it simulates a U-Net
------------------------
- Starts from a binary mask derived from the GT
- Applies light 2D blur per-slice to soften boundaries
- Adds controllable Gaussian noise
- Optionally injects small false positives / false negatives
- Clips to [0,1]

Run
---
python test_simulate_unet_output.py --index 9
python test_simulate_unet_output.py --index 9 --gt-prefix segmentation- --nii-ext .nii
python test_simulate_unet_output.py --all --start 0 --end 130
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import nibabel as nib
from skimage.filters import gaussian


def load_gt_nii(
    nii_dir: Path,
    index: int,
    gt_prefix: str = "segmentation-",
    nii_ext: str = ".nii",
) -> nib.Nifti1Image:
    path = nii_dir / f"{gt_prefix}{index}{nii_ext}"
    if not path.exists():
        # Try .nii.gz if user passed .nii
        alt = path.with_suffix(path.suffix + ".gz") if path.suffix == ".nii" else None
        if alt and alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"GT NIfTI not found: {path}")
    return nib.load(str(path))


def nii_to_binary_mask_xyz(
    nii: nib.Nifti1Image,
    tumor_label: Optional[int] = None,
) -> np.ndarray:
    """
    Returns a binary mask in NIfTI array order (X, Y, Z).

    If tumor_label is provided, uses (data == tumor_label).
    Else uses (data > 0).
    """
    data = np.asarray(nii.get_fdata())
    if data.ndim != 3:
        raise ValueError(f"Expected 3D GT NIfTI. Got shape={data.shape}")

    if tumor_label is None:
        mask = (data > 0)
    else:
        mask = (data == tumor_label)

    return mask.astype(np.uint8)


def mask_xyz_to_unet_proba_zhw(
    mask_xyz: np.ndarray,
    blur_sigma_xy: float = 1.2,
    noise_sigma: float = 0.08,
    fp_rate: float = 0.0015,
    fn_rate: float = 0.002,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a (Z, H, W) float32 probability volume in [0,1] from a binary mask (X,Y,Z).

    Strategy:
    - Transpose to (Z, X, Y) == (Z, H, W)
    - For each slice, blur to soften edges (probabilistic boundary)
    - Add Gaussian noise
    - Inject sparse FP/FN flips (optional)
    - Clip to [0,1]
    """
    if mask_xyz.ndim != 3:
        raise ValueError(f"mask_xyz must be 3D. Got {mask_xyz.shape}")

    rng = np.random.default_rng(seed)

    # Convert to (Z, H, W) where H=X and W=Y
    mask_zhw = np.transpose(mask_xyz, (2, 0, 1)).astype(np.float32)

    Z, H, W = mask_zhw.shape

    # Per-slice blur (2D) to mimic model uncertainty at boundaries
    proba = np.empty_like(mask_zhw, dtype=np.float32)
    for z in range(Z):
        # gaussian expects float; preserve_range keeps [0,1] scale
        proba[z] = gaussian(mask_zhw[z], sigma=blur_sigma_xy, preserve_range=True).astype(np.float32)

    # Add Gaussian noise (mimic imperfect predictions)
    if noise_sigma > 0:
        proba += rng.normal(loc=0.0, scale=noise_sigma, size=proba.shape).astype(np.float32)

    # Inject sparse false positives / false negatives
    # FP: random background voxels get a boost
    # FN: random tumor voxels get a drop
    if fp_rate > 0:
        bg = (mask_zhw < 0.5)
        fp = bg & (rng.random(proba.shape) < fp_rate)
        proba[fp] = np.maximum(proba[fp], 0.65).astype(np.float32)

    if fn_rate > 0:
        fg = (mask_zhw >= 0.5)
        fn = fg & (rng.random(proba.shape) < fn_rate)
        proba[fn] = np.minimum(proba[fn], 0.35).astype(np.float32)

    # Final clip + dtype
    proba = np.clip(proba, 0.0, 1.0).astype(np.float32)
    return proba


def save_proba_map(
    out_dir: Path,
    index: int,
    proba_zhw: np.ndarray,
    out_prefix: str = "post_unet_",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_prefix}{index}.npy"
    np.save(out_path, proba_zhw.astype(np.float32))
    return out_path


def process_one(
    index: int,
    nii_dir: Path,
    out_dir: Path,
    gt_prefix: str,
    nii_ext: str,
    tumor_label: Optional[int],
    blur_sigma_xy: float,
    noise_sigma: float,
    fp_rate: float,
    fn_rate: float,
    seed: int,
) -> Path:
    nii = load_gt_nii(nii_dir, index, gt_prefix=gt_prefix, nii_ext=nii_ext)
    mask_xyz = nii_to_binary_mask_xyz(nii, tumor_label=tumor_label)
    proba_zhw = mask_xyz_to_unet_proba_zhw(
        mask_xyz,
        blur_sigma_xy=blur_sigma_xy,
        noise_sigma=noise_sigma,
        fp_rate=fp_rate,
        fn_rate=fn_rate,
        seed=seed + index,  # different seed per case
    )

    # Sanity checks for your downstream constraints
    assert proba_zhw.ndim == 3
    assert proba_zhw.dtype == np.float32
    assert float(proba_zhw.min()) >= 0.0 and float(proba_zhw.max()) <= 1.0

    out_path = save_proba_map(out_dir, index, proba_zhw)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None, help="Case index to process.")
    p.add_argument("--all", action="store_true", help="Process a range of indices.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=130, help="Exclusive end index.")
    p.add_argument("--nii-dir", type=str, default="data/nii_pre_unet")
    p.add_argument("--out-dir", type=str, default="data/maps_post_unet")

    # Naming
    p.add_argument("--gt-prefix", type=str, default="segmentation-",
                   help="Prefix for GT files, e.g. 'segmentation-' -> segmentation-{index}.nii")
    p.add_argument("--nii-ext", type=str, default=".nii", help="Extension: .nii or .nii.gz")

    # Label handling
    p.add_argument("--tumor-label", type=int, default=None,
                   help="If GT is multi-class, set tumor label value (e.g. 2). If None, uses >0 as tumor.")

    # Simulation knobs
    p.add_argument("--blur-sigma-xy", type=float, default=1.2)
    p.add_argument("--noise-sigma", type=float, default=0.08)
    p.add_argument("--fp-rate", type=float, default=0.0015)
    p.add_argument("--fn-rate", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    nii_dir = Path(args.nii_dir)
    out_dir = Path(args.out_dir)

    if args.all:
        for idx in range(args.start, args.end):
            out_path = process_one(
                idx, nii_dir, out_dir,
                gt_prefix=args.gt_prefix,
                nii_ext=args.nii_ext,
                tumor_label=args.tumor_label,
                blur_sigma_xy=args.blur_sigma_xy,
                noise_sigma=args.noise_sigma,
                fp_rate=args.fp_rate,
                fn_rate=args.fn_rate,
                seed=args.seed,
            )
            print(f"[OK] index={idx} -> {out_path}")
    else:
        if args.index is None:
            raise SystemExit("Provide --index <int> or use --all --start --end")
        out_path = process_one(
            args.index, nii_dir, out_dir,
            gt_prefix=args.gt_prefix,
            nii_ext=args.nii_ext,
            tumor_label=args.tumor_label,
            blur_sigma_xy=args.blur_sigma_xy,
            noise_sigma=args.noise_sigma,
            fp_rate=args.fp_rate,
            fn_rate=args.fn_rate,
            seed=args.seed,
        )
        print(f"[OK] index={args.index} -> {out_path}")


if __name__ == "__main__":
    main()
