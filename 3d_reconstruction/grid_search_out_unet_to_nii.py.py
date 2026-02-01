"""
grid_search_out_unet_to_nii.py

Grid-search (sort-of) over a small set of post-processing hyperparameters:
- smoothing: gaussian_3d vs z_gaussian
- hysteresis thresholding: 2 (low, high) pairs
- connected components connectivity: 18 vs 26

Scoring:
- Uses absolute relative volume error |(V_pred - V_gt) / V_gt| averaged over cases.
  (Lower is better.)
- Skips cases where GT tumor volume is 0 (to avoid NaNs dominating). Those cases are logged.

Outputs:
- Prints the best parameter set (lowest mean score)
- Prints a ranked table of all parameter sets

Requirements:
- class Out_unet_to_nii must be importable.
- Data layout must match your project:
  - npy probas: data/maps_post_unet/post_unet_{index}.npy
  - ref nii:    data/nii_pre_unet/volume-{index}.nii
  - gt nii:     data/nii_pre_unet/segmentation-{index}.nii  (or seg_gt-{index}.nii; adjust GT_PREFIX)

Run examples:
  python grid_search_out_unet_to_nii.py --indices 9 10 11
  python grid_search_out_unet_to_nii.py --start 0 --end 20
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from out_unet_to_nii import Out_unet_to_nii, SmoothingConfig, ThresholdConfig, CCFilterConfig


GT_PREFIX = "segmentation-"   
NII_EXT = ".nii"              


def build_param_grid() -> List[Dict[str, Any]]:
    """
    Define the grid to search.
    Note: we only search the options you requested.
    """
    smoothing_options = [
        SmoothingConfig(method="gaussian_3d", sigma_xyz=(0.8, 0.8, 0.3)),
        SmoothingConfig(method="z_gaussian", z_sigma=1.0),
    ]

    # Two hysteresis pairs (low, high)
    hysteresis_pairs = [
        (0.25, 0.55),
        (0.35, 0.65),
    ]

    connectivities = [18, 26]

    grid: List[Dict[str, Any]] = []
    for sm in smoothing_options:
        for (low, high) in hysteresis_pairs:
            th = ThresholdConfig(method="hysteresis", low=low, high=high)
            for conn in connectivities:
                cc = CCFilterConfig(connectivity=conn, min_volume_cm3=0.5, keep_top_n=None)
                grid.append({"smoothing": sm, "threshold": th, "cc": cc})
    return grid


def score_param_set(
    pipe: Out_unet_to_nii,
    indices: List[int],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run generation + evaluation for one parameter set, return metrics.
    Score = mean(abs(rel_diff)) over valid cases (gt tumor volume > 0).
    """
    smoothing_cfg: SmoothingConfig = params["smoothing"]
    threshold_cfg: ThresholdConfig = params["threshold"]
    cc_cfg: CCFilterConfig = params["cc"]

    rel_diffs: List[float] = []
    per_case: List[Dict[str, Any]] = []
    skipped_zero_gt: List[int] = []

    for idx in indices:
        # Generate predicted nii
        pipe.generate_predicted_nii(
            index=idx,
            smoothing_cfg=smoothing_cfg,
            threshold_cfg=threshold_cfg,
            cc_cfg=cc_cfg,
            return_intermediates=False,
        )

        gt_path = pipe.nii_pre_dir / f"{GT_PREFIX}{idx}{NII_EXT}"
        pred_path = pipe.get_predicted_path(idx)

        res = Out_unet_to_nii.eval_diff_volume_relative(str(gt_path), str(pred_path))

        # Skip cases with 0 GT tumor volume (rel_diff is nan)
        if np.isnan(res["rel_diff"]):
            skipped_zero_gt.append(idx)
            continue

        rel_diffs.append(abs(float(res["rel_diff"])))
        per_case.append({"index": idx, **res})

    score = float(np.mean(rel_diffs)) if rel_diffs else float("inf")

    return {
        "score_mean_abs_rel_diff": score,
        "n_valid": len(rel_diffs),
        "n_total": len(indices),
        "skipped_zero_gt": skipped_zero_gt,
        "params": {
            "smoothing": asdict(smoothing_cfg),
            "threshold": asdict(threshold_cfg),
            "cc": asdict(cc_cfg),
        },
        "per_case": per_case,
    }


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--indices", nargs="+", type=int, help="Explicit list of indices, e.g. --indices 9 10 11")
    group.add_argument("--start", type=int, help="Start index (inclusive) for a range")
    ap.add_argument("--end", type=int, help="End index (exclusive) for a range (required if --start is used)")
    ap.add_argument("--maps-dir", type=str, default="data/maps_post_unet")
    ap.add_argument("--nii-pre-dir", type=str, default="data/nii_pre_unet")
    ap.add_argument("--nii-out-dir", type=str, default="data/nii_predicted")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.indices is not None:
        indices = args.indices
    else:
        if args.end is None:
            raise SystemExit("--end is required when using --start")
        indices = list(range(args.start, args.end))
    print(indices)

    pipe = Out_unet_to_nii(
        maps_dir=args.maps_dir,
        nii_pre_dir=args.nii_pre_dir,
        nii_out_dir=args.nii_out_dir,
    )

    grid = build_param_grid()
    results: List[Dict[str, Any]] = []

    for i, params in tqdm(enumerate(grid, 1)):
        res = score_param_set(pipe, indices, params)
        results.append(res)

        if args.verbose:
            print(f"[{i}/{len(grid)}] score={res['score_mean_abs_rel_diff']:.6f} "
                  f"valid={res['n_valid']}/{res['n_total']} params={res['params']}")

    # Rank results
    results_sorted = sorted(results, key=lambda r: (r["score_mean_abs_rel_diff"], -r["n_valid"]))

    best = results_sorted[0]
    print("\n==================== BEST PARAMS ====================")
    print(f"Mean |relative volume diff| : {best['score_mean_abs_rel_diff']:.6f}")
    print(f"Valid cases               : {best['n_valid']}/{best['n_total']}")
    if best["skipped_zero_gt"]:
        print(f"Skipped (GT tumor vol=0)  : {best['skipped_zero_gt']}")
    print("Params:")
    print(best["params"])

    print("\n==================== ALL RESULTS (ranked) ====================")
    for rank, r in enumerate(results_sorted, 1):
        sm = r["params"]["smoothing"]["method"]
        th = r["params"]["threshold"]
        cc = r["params"]["cc"]
        if th["method"] == "hysteresis":
            th_str = f"hyst(low={th['low']},high={th['high']})"
        else:
            th_str = f"global(t={th['t']})"
        print(
            f"{rank:02d}) score={r['score_mean_abs_rel_diff']:.6f} "
            f"valid={r['n_valid']}/{r['n_total']} "
            f"smooth={sm} thresh={th_str} conn={cc['connectivity']}"
        )


if __name__ == "__main__":
    main()
