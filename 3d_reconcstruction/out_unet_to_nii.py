from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import nibabel as nib
from skimage.measure import label as cc_label
from skimage.filters import gaussian as sk_gaussian
from skimage.filters import apply_hysteresis_threshold


@dataclass
class SmoothingConfig:
    """
    Smoothing applied on probability maps (float in [0,1]) BEFORE thresholding.

    method:
      - "none"
      - "gaussian_3d"        : skimage.filters.gaussian on the 3D prob volume
      - "z_moving_average"   : fast moving-average along Z only
      - "z_gaussian"         : gaussian smoothing along Z only (uses scipy if available, else falls back)
    """
    method: str = "none"

    # gaussian_3d
    sigma_xyz: Union[float, Tuple[float, float, float]] = 1.0  # sigma for (X,Y,Z) array

    # z_moving_average
    z_window: int = 3  # must be odd

    # z_gaussian
    z_sigma: float = 1.0


@dataclass
class ThresholdConfig:
    """
    Thresholding to turn probabilities into a binary mask.
    method:
      - "global"     : pred = prob >= t
      - "hysteresis" : apply_hysteresis_threshold(prob, low, high)
    """
    method: str = "global"
    t: float = 0.5
    low: float = 0.3
    high: float = 0.6


@dataclass
class CCFilterConfig:
    """
    3D connected component filtering on the binary mask.

    - connectivity can be 6 / 18 / 26 OR 1 / 2 / 3 (skimage connectivity for 3D).
    - You may choose:
        * min_volume_cm3 : remove components smaller than this physical volume (cm^3)
        * keep_top_n     : keep only the N largest components (by voxel count)
      You can combine both; min_volume is applied first, then keep_top_n.
    """
    connectivity: int = 26
    min_volume_cm3: Optional[float] = 0.5
    keep_top_n: Optional[int] = None


class Out_unet_to_nii:
    """
    Pipeline: UNet 2D output (proba map) -> 3D proba volume -> smoothing -> threshold -> 3D CC filtering -> NIfTI save.

    Inputs:
      - Probability maps from UNet: data/maps_post_unet/post_unet_{index}(.npy/.npz/.pkl)
        Expected shape: (nb_slices, H, W) with float values in [0,1]
      - Reference input NIfTI: data/nii_pre_unet/volume-{index}.nii
      - Ground truth NIfTI: data/nii_pre_unet/seg_gt-{index}.nii

    Output:
      - Predicted binary NIfTI: data/nii_predicted/seg_predicted-{index}.nii
    """

    def __init__(
        self,
        maps_dir: str = "data/maps_post_unet",
        nii_pre_dir: str = "data/nii_pre_unet",
        nii_out_dir: str = "data/nii_predicted",
        maps_prefix: str = "post_unet_",
        ref_prefix: str = "volume-",
        gt_prefix: str = "seg_gt-",
        out_prefix: str = "seg_predicted-",
        nii_ext: str = ".nii",
    ):
        self.maps_dir = Path(maps_dir)
        self.nii_pre_dir = Path(nii_pre_dir)
        self.nii_out_dir = Path(nii_out_dir)

        self.maps_prefix = maps_prefix
        self.ref_prefix = ref_prefix
        self.gt_prefix = gt_prefix
        self.out_prefix = out_prefix
        self.nii_ext = nii_ext

        self.nii_out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Paths / I/O
    # -------------------------

    def _find_map_file(self, index: int) -> Path:
        """
        Find a file matching post_unet_{index}* in maps_dir.
        Supports .npy, .npz, .pkl by default.
        """
        pattern = f"{self.maps_prefix}{index}*"
        matches = sorted(self.maps_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No UNet output map found for index={index} with pattern {pattern} in {self.maps_dir}")
        # Prefer deterministic ordering: .npy > .npz > .pkl > others
        def rank(p: Path) -> int:
            suf = p.suffix.lower()
            return {".npy": 0, ".npz": 1, ".pkl": 2}.get(suf, 99)
        matches.sort(key=lambda p: (rank(p), p.name))
        return matches[0]

    def load_unet_proba_map(self, index: int) -> np.ndarray:
        """
        Load UNet output probability map for a given index.
        Returns: numpy array, expected shape (Z, H, W), float in [0,1].
        """
        path = self._find_map_file(index)
        suf = path.suffix.lower()

        if suf == ".npy":
            arr = np.load(path)
        elif suf == ".npz":
            data = np.load(path)
            # take first array by key
            if len(data.files) == 0:
                raise ValueError(f"Empty npz file: {path}")
            arr = data[data.files[0]]
        elif suf == ".pkl":
            with open(path, "rb") as f:
                arr = pickle.load(f)
        else:
            raise ValueError(f"Unsupported map file format: {path} (expected .npy/.npz/.pkl)")

        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"UNet output must be 3D (Z,H,W). Got shape={arr.shape} from {path}")

        # Ensure float32 and clip to [0,1]
        arr = arr.astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    def load_reference_nii(self, index: int) -> nib.Nifti1Image:
        path = self.nii_pre_dir / f"{self.ref_prefix}{index}{self.nii_ext}"
        if not path.exists():
            raise FileNotFoundError(f"Reference NIfTI not found: {path}")
        return nib.load(str(path))

    def get_ground_truth_path(self, index: int) -> Path:
        return self.nii_pre_dir / f"{self.gt_prefix}{index}{self.nii_ext}"

    def get_predicted_path(self, index: int) -> Path:
        return self.nii_out_dir / f"{self.out_prefix}{index}{self.nii_ext}"

    # -------------------------
    # Shape alignment
    # -------------------------

    @staticmethod
    def _proba_zhw_to_ref_xyz(proba_zhw: np.ndarray, ref_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert proba from (Z,H,W) to match ref NIfTI array shape.

        Typical case:
          proba_zhw.shape == (Z, X, Y) and ref_shape == (X, Y, Z)
          => transpose to (X, Y, Z) via (1,2,0)

        If already matches ref_shape, return as-is.
        If cannot match, raise.
        """
        if proba_zhw.shape == ref_shape:
            return proba_zhw

        Z, H, W = proba_zhw.shape
        X, Y, Zref = ref_shape

        # Most common: (Z, X, Y) -> (X, Y, Z)
        if (Z == Zref) and (H == X) and (W == Y):
            return np.transpose(proba_zhw, (1, 2, 0))

        # Another possible: (Z, Y, X) -> (X, Y, Z)
        if (Z == Zref) and (H == Y) and (W == X):
            return np.transpose(proba_zhw, (2, 1, 0))

        raise ValueError(
            f"Cannot align UNet proba shape={proba_zhw.shape} to reference shape={ref_shape}. "
            "Expected (Z,H,W) to match (Z,X,Y) or (Z,Y,X)."
        )

    # -------------------------
    # Smoothing methods
    # -------------------------

    @staticmethod
    def smooth_probabilities(prob_xyz: np.ndarray, cfg: SmoothingConfig) -> np.ndarray:
        """
        Apply selected smoothing on probability volume (X,Y,Z).
        Returns float32 in [0,1].
        """
        method = (cfg.method or "none").lower()
        p = prob_xyz.astype(np.float32)

        if method == "none":
            return np.clip(p, 0.0, 1.0)

        if method == "gaussian_3d":
            # skimage gaussian supports nD. preserve_range=True avoids internal rescaling.
            p2 = sk_gaussian(p, sigma=cfg.sigma_xyz, preserve_range=True)
            return np.clip(p2.astype(np.float32), 0.0, 1.0)

        if method == "z_moving_average":
            w = int(cfg.z_window)
            if w < 1 or w % 2 == 0:
                raise ValueError("z_window must be a positive odd integer.")
            # Fast moving average along Z using cumulative sum
            pad = w // 2
            p_pad = np.pad(p, ((0, 0), (0, 0), (pad, pad)), mode="edge")
            csum = np.cumsum(p_pad, axis=2, dtype=np.float64)
            # window sum: csum[z+w] - csum[z]
            out = (csum[:, :, w:] - csum[:, :, :-w]) / float(w)
            return np.clip(out.astype(np.float32), 0.0, 1.0)

        if method == "z_gaussian":
            # Prefer scipy if available (fast). Otherwise fall back to manual convolution.
            try:
                from scipy.ndimage import gaussian_filter1d
                out = gaussian_filter1d(p, sigma=float(cfg.z_sigma), axis=2, mode="nearest")
                return np.clip(out.astype(np.float32), 0.0, 1.0)
            except Exception:
                # Fallback: build kernel and convolve along Z (slower)
                sigma = float(cfg.z_sigma)
                if sigma <= 0:
                    return np.clip(p, 0.0, 1.0)
                radius = int(max(1, round(3 * sigma)))
                xs = np.arange(-radius, radius + 1, dtype=np.float32)
                k = np.exp(-(xs**2) / (2 * sigma**2))
                k = k / np.sum(k)
                pad = radius
                p_pad = np.pad(p, ((0, 0), (0, 0), (pad, pad)), mode="edge")
                out = np.zeros_like(p, dtype=np.float32)
                # Convolution along Z (vectorized over X,Y)
                for i, wgt in enumerate(k):
                    out += wgt * p_pad[:, :, i:i + p.shape[2]]
                return np.clip(out, 0.0, 1.0)

        raise ValueError(f"Unknown smoothing method: {cfg.method}")

    # -------------------------
    # Thresholding methods
    # -------------------------

    @staticmethod
    def threshold_probabilities(prob_xyz: np.ndarray, cfg: ThresholdConfig) -> np.ndarray:
        """
        Convert probabilities to binary mask (uint8) in (X,Y,Z).
        """
        method = (cfg.method or "global").lower()

        if method == "global":
            t = float(cfg.t)
            if not (0.0 <= t <= 1.0):
                raise ValueError("Global threshold t must be in [0,1].")
            m = (prob_xyz >= t)
            return m.astype(np.uint8)

        if method == "hysteresis":
            low = float(cfg.low)
            high = float(cfg.high)
            if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0 and low < high):
                raise ValueError("Hysteresis thresholds must satisfy 0<=low<high<=1.")
            m = apply_hysteresis_threshold(prob_xyz.astype(np.float32), low, high)
            return m.astype(np.uint8)

        raise ValueError(f"Unknown threshold method: {cfg.method}")

    # -------------------------
    # 3D Connected components filtering
    # -------------------------

    @staticmethod
    def _normalize_connectivity(connectivity: int) -> int:
        """
        Accepts 6/18/26 or 1/2/3. Returns 1/2/3 for skimage.measure.label connectivity in 3D.
        """
        if connectivity in (1, 2, 3):
            return connectivity
        mapping = {6: 1, 18: 2, 26: 3}
        if connectivity in mapping:
            return mapping[connectivity]
        raise ValueError("connectivity must be one of: 6, 18, 26, 1, 2, 3")

    @staticmethod
    def filter_connected_components(
        mask_xyz: np.ndarray,
        spacing_xyz: Tuple[float, float, float],
        cfg: CCFilterConfig
    ) -> np.ndarray:
        """
        Apply 3D connected components filtering on a binary mask (X,Y,Z).

        Steps:
          1) label components with desired connectivity
          2) optional remove small components by physical volume threshold (cm^3)
          3) optional keep only top-N largest components
        """
        if mask_xyz.ndim != 3:
            raise ValueError(f"mask must be 3D. got {mask_xyz.shape}")
        # Ensure binary
        m = (mask_xyz > 0).astype(np.uint8)
        if not np.any(m):
            return m

        conn = Out_unet_to_nii._normalize_connectivity(int(cfg.connectivity))
        labeled = cc_label(m, connectivity=conn)  # 0 background, 1..K components

        # Compute voxel counts per label
        counts = np.bincount(labeled.ravel())
        # counts[0] is background count

        # 1) Remove small components by volume threshold (cm^3)
        if cfg.min_volume_cm3 is not None:
            min_cm3 = float(cfg.min_volume_cm3)
            if min_cm3 < 0:
                raise ValueError("min_volume_cm3 must be >= 0")
            voxel_vol_mm3 = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
            min_mm3 = min_cm3 * 1000.0  # 1 cm^3 = 1000 mm^3
            min_vox = int(np.ceil(min_mm3 / voxel_vol_mm3))
            if min_vox > 0:
                remove_labels = np.where((counts < min_vox) & (np.arange(len(counts)) != 0))[0]
                if remove_labels.size > 0:
                    # Build mask to keep: remove those labels
                    remove_mask = np.isin(labeled, remove_labels)
                    labeled[remove_mask] = 0
                    # Re-binarize after removal
                    m = (labeled > 0).astype(np.uint8)
                    labeled = cc_label(m, connectivity=conn)
                    counts = np.bincount(labeled.ravel())

        # 2) Keep top-N largest components
        if cfg.keep_top_n is not None:
            n = int(cfg.keep_top_n)
            if n <= 0:
                raise ValueError("keep_top_n must be >= 1")
            if len(counts) <= 1:
                return (labeled > 0).astype(np.uint8)

            # Get labels sorted by size (exclude background)
            comp_labels = np.arange(1, len(counts))
            comp_sizes = counts[1:]
            order = np.argsort(comp_sizes)[::-1]
            keep_labels = comp_labels[order[:n]]
            m = np.isin(labeled, keep_labels).astype(np.uint8)

        return m

    @staticmethod
    def keep_top_n_components(mask_xyz: np.ndarray, n: int, connectivity: int = 26) -> np.ndarray:
        """
        Convenience method: keep only top N largest components.
        """
        cfg = CCFilterConfig(connectivity=connectivity, min_volume_cm3=None, keep_top_n=n)
        # spacing doesn't matter if no min_volume_cm3; pass dummy
        return Out_unet_to_nii.filter_connected_components(mask_xyz, (1.0, 1.0, 1.0), cfg)

    # -------------------------
    # NIfTI save + validations
    # -------------------------

    @staticmethod
    def _assert_binary(arr: np.ndarray, name: str = "array") -> None:
        vals = np.unique(arr)
        if vals.size == 0:
            raise ValueError(f"{name} is empty.")
        if not np.all(np.isin(vals, [0, 1])):
            raise ValueError(f"{name} must be binary (0/1). Found values: {vals[:10]}{'...' if vals.size > 10 else ''}")

    def save_predicted_nii(self, mask_xyz: np.ndarray, ref_nii: nib.Nifti1Image, out_path: Path) -> None:
        """
        Save binary mask as NIfTI with affine/header from reference NIfTI.
        """
        if mask_xyz.shape != ref_nii.shape:
            raise ValueError(f"Pred mask shape {mask_xyz.shape} != ref shape {ref_nii.shape}")

        mask_u8 = (mask_xyz > 0).astype(np.uint8)
        self._assert_binary(mask_u8, "predicted mask")

        header = ref_nii.header.copy()
        header.set_data_dtype(np.uint8)

        out_nii = nib.Nifti1Image(mask_u8, affine=ref_nii.affine, header=header)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_nii, str(out_path))

    # -------------------------
    # Main generation method
    # -------------------------

    def generate_predicted_nii(
        self,
        index: int,
        smoothing_cfg: Optional[SmoothingConfig] = None,
        threshold_cfg: Optional[ThresholdConfig] = None,
        cc_cfg: Optional[CCFilterConfig] = None,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Generate seg_predicted-{index}.nii.

        Returns a dict with paths and (optionally) intermediate arrays.
        """
        smoothing_cfg = smoothing_cfg or SmoothingConfig(method="none")
        threshold_cfg = threshold_cfg or ThresholdConfig(method="global", t=0.5)
        cc_cfg = cc_cfg or CCFilterConfig(connectivity=26, min_volume_cm3=0.5, keep_top_n=None)

        # 1) Load proba map from UNet output (Z,H,W)
        proba_zhw = self.load_unet_proba_map(index)

        # 2) Load reference nii to copy affine/header and match shape
        ref_nii = self.load_reference_nii(index)
        ref_shape = tuple(ref_nii.shape)  # (X,Y,Z) in nibabel

        # 3) Align proba to reference shape (X,Y,Z)
        proba_xyz = self._proba_zhw_to_ref_xyz(proba_zhw, ref_shape)

        # 4) Smoothing on probabilities (optional)
        proba_smooth = self.smooth_probabilities(proba_xyz, smoothing_cfg)

        # 5) Threshold to binary
        mask_bin = self.threshold_probabilities(proba_smooth, threshold_cfg)

        # 6) Connected components filtering (optional)
        spacing_xyz = tuple(float(z) for z in ref_nii.header.get_zooms()[:3])
        mask_clean = self.filter_connected_components(mask_bin, spacing_xyz, cc_cfg)

        # 7) Save NIfTI
        out_path = self.get_predicted_path(index)
        self.save_predicted_nii(mask_clean, ref_nii, out_path)

        result: Dict[str, Any] = {
            "index": index,
            "out_path": str(out_path),
            "ref_path": str(self.nii_pre_dir / f"{self.ref_prefix}{index}{self.nii_ext}"),
            "map_file": str(self._find_map_file(index)),
            "spacing_xyz": spacing_xyz,
            "smoothing": smoothing_cfg,
            "threshold": threshold_cfg,
            "cc_filter": cc_cfg,
        }

        if return_intermediates:
            result.update({
                "proba_zhw": proba_zhw,
                "proba_xyz": proba_xyz,
                "proba_smooth": proba_smooth,
                "mask_bin": mask_bin,
                "mask_clean": mask_clean,
            })

        return result

    def generate_for_indices(
        self,
        indices: List[int],
        smoothing_cfg: Optional[SmoothingConfig] = None,
        threshold_cfg: Optional[ThresholdConfig] = None,
        cc_cfg: Optional[CCFilterConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to process many cases.
        """
        outputs = []
        for idx in indices:
            outputs.append(self.generate_predicted_nii(
                idx,
                smoothing_cfg=smoothing_cfg,
                threshold_cfg=threshold_cfg,
                cc_cfg=cc_cfg,
                return_intermediates=False
            ))
        return outputs

    # -------------------------
    # Evaluation utility
    # -------------------------

    @staticmethod
    def eval_diff_volume_relative(path_ground_truth_nii: str, path_predicted_nii: str) -> Dict[str, float]:
        """
        Compute relative volume difference between GT and prediction.
        Checks:
          - same shape (same number of slices and same in-plane size)
          - same affine (np.allclose)
          - both binary masks (0/1)

        Returns dict:
          - gt_volume_mm3, pred_volume_mm3
          - diff_mm3 (pred - gt)
          - rel_diff (pred - gt) / gt   (nan if gt==0)
        """
        gt_nii = nib.load(path_ground_truth_nii)
        pr_nii = nib.load(path_predicted_nii)

        if gt_nii.shape != pr_nii.shape:
            raise ValueError(f"Shape mismatch: GT {gt_nii.shape} vs Pred {pr_nii.shape}")

        if not np.allclose(gt_nii.affine, pr_nii.affine, atol=1e-5):
            raise ValueError("Affine mismatch between GT and predicted NIfTI (not aligned).")

        gt = np.asarray(gt_nii.get_fdata())
        pr = np.asarray(pr_nii.get_fdata())

        # Accept {0,1} or {0,255} if you want? The user requested binary; enforce strict 0/1.
        # If your masks are 0/255, convert upstream or uncomment normalization.
        # gt = (gt > 0).astype(np.uint8)
        # pr = (pr > 0).astype(np.uint8)

        # Strict binary check 0/1
        Out_unet_to_nii._assert_binary(gt.astype(np.int64), "ground truth")
        Out_unet_to_nii._assert_binary(pr.astype(np.int64), "predicted")

        spacing = gt_nii.header.get_zooms()[:3]
        voxel_vol_mm3 = float(spacing[0] * spacing[1] * spacing[2])

        gt_vol_mm3 = float(np.sum(gt == 1) * voxel_vol_mm3)
        pr_vol_mm3 = float(np.sum(pr == 1) * voxel_vol_mm3)

        diff_mm3 = pr_vol_mm3 - gt_vol_mm3
        rel_diff = np.nan if gt_vol_mm3 == 0 else (diff_mm3 / gt_vol_mm3)

        return {
            "gt_volume_mm3": gt_vol_mm3,
            "pred_volume_mm3": pr_vol_mm3,
            "diff_mm3": diff_mm3,
            "rel_diff": rel_diff,
        }
