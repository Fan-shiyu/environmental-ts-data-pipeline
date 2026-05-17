"""Validation: compare new outputs against existing files to detect regressions or gaps."""

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.stats import pearsonr


def compare_rasters(new_path: str, legacy_path: str) -> dict:
    """Open both rasters with rasterio. Reproject new onto legacy's grid using
    nearest-neighbor if shapes/CRS/transforms differ. Return a dict with:
    - shape_new, shape_legacy
    - n_valid_new, n_valid_legacy
    - mean_new, mean_legacy, std_new, std_legacy, min_new, min_legacy, max_new, max_legacy
    - n_overlapping (pixels where both have valid data)
    - correlation (Pearson on overlapping)
    - mean_abs_diff, median_abs_diff (on overlapping)
    - pct_diff_gt_005, pct_diff_gt_010
    """
    with rasterio.open(new_path) as src:
        new_data = src.read(1).astype(np.float32)
        new_nodata = src.nodata
        new_transform = src.transform
        new_crs = src.crs
        new_shape = src.shape

    with rasterio.open(legacy_path) as src:
        ref_data = src.read(1).astype(np.float32)
        ref_nodata = src.nodata
        ref_transform = src.transform
        ref_crs = src.crs
        ref_shape = src.shape

    print(f"\n  New raster  -- shape: {new_shape}, CRS: {new_crs}, nodata: {new_nodata}")
    print(f"  Ref raster  -- shape: {ref_shape}, CRS: {ref_crs}, nodata: {ref_nodata}")

    grids_match = (
        new_crs == ref_crs
        and new_shape == ref_shape
        and np.allclose(
            [new_transform.a, new_transform.e, new_transform.c, new_transform.f],
            [ref_transform.a, ref_transform.e, ref_transform.c, ref_transform.f],
            atol=1e-8,
        )
    )

    if not grids_match:
        print("\n  Grids differ -- reprojecting new raster onto ref grid (nearest-neighbor) ...")
        reprojected = np.full(ref_shape, np.nan, dtype=np.float32)
        with rasterio.open(new_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=reprojected,
                src_transform=new_transform,
                src_crs=new_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                src_nodata=new_nodata,
                dst_nodata=np.nan,
            )
        new_data = reprojected
        print(f"  After reprojection -- shape: {new_data.shape}")
    else:
        print("\n  Grids match -- no reprojection needed.")

    if new_nodata is not None:
        new_data = np.where(new_data == new_nodata, np.nan, new_data)
    if ref_nodata is not None:
        ref_data = np.where(ref_data == ref_nodata, np.nan, ref_data)

    new_data = np.where(np.abs(new_data) > 1e6, np.nan, new_data)
    ref_data = np.where(np.abs(ref_data) > 1e6, np.nan, ref_data)

    new_valid = ~np.isnan(new_data)
    ref_valid = ~np.isnan(ref_data)
    overlap = new_valid & ref_valid
    n_overlap = int(overlap.sum())

    stats = {
        "shape_new": new_shape,
        "shape_legacy": ref_shape,
        "n_valid_new": int(new_valid.sum()),
        "n_valid_legacy": int(ref_valid.sum()),
        "mean_new": float(np.nanmean(new_data)),
        "mean_legacy": float(np.nanmean(ref_data)),
        "std_new": float(np.nanstd(new_data)),
        "std_legacy": float(np.nanstd(ref_data)),
        "min_new": float(np.nanmin(new_data)),
        "min_legacy": float(np.nanmin(ref_data)),
        "max_new": float(np.nanmax(new_data)),
        "max_legacy": float(np.nanmax(ref_data)),
        "n_overlapping": n_overlap,
        "correlation": None,
        "mean_abs_diff": None,
        "median_abs_diff": None,
        "pct_diff_gt_005": None,
        "pct_diff_gt_010": None,
    }

    if n_overlap >= 10:
        new_ov = new_data[overlap]
        ref_ov = ref_data[overlap]
        abs_diff = np.abs(new_ov - ref_ov)
        corr, _ = pearsonr(new_ov, ref_ov)
        stats["correlation"] = float(corr)
        stats["mean_abs_diff"] = float(np.mean(abs_diff))
        stats["median_abs_diff"] = float(np.median(abs_diff))
        stats["pct_diff_gt_005"] = float(100.0 * np.mean(abs_diff > 0.05))
        stats["pct_diff_gt_010"] = float(100.0 * np.mean(abs_diff > 0.10))

    return stats


def smoke_test(path: str) -> dict:
    """Open a single raster and return basic stats (shape, CRS, valid pixel count, mean/min/max).

    Used when no legacy comparison file exists (e.g. new AoIs like Zambia_WL).
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        crs = src.crs
        shape = src.shape

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    data = np.where(np.abs(data) > 1e6, np.nan, data)

    valid = ~np.isnan(data)
    return {
        "shape": shape,
        "crs": str(crs),
        "n_valid": int(valid.sum()),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data)),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
    }


def print_smoke_test(stats: dict) -> None:
    """Print smoke-test stats for a raster with no legacy comparison file."""
    print("\n  --- Smoke test (no legacy file for comparison) ---")
    print(f"  Shape:        {stats['shape']}")
    print(f"  CRS:          {stats['crs']}")
    print(f"  Valid pixels: {stats['n_valid']:,}")
    print(f"  Mean NDVI:    {stats['mean']:.4f}")
    print(f"  Std NDVI:     {stats['std']:.4f}")
    print(f"  Min NDVI:     {stats['min']:.4f}")
    print(f"  Max NDVI:     {stats['max']:.4f}")


def print_comparison(stats: dict) -> None:
    """Print the stats dict in a readable formatted block. No pass/fail verdict."""
    print("\n  --- Per-raster statistics ---")
    print(f"  {'Metric':<35} {'New (corrected mask)':>22} {'Existing (legacy)':>20}")
    print(f"  {'-'*78}")
    print(f"  {'Valid pixels':<35} {stats['n_valid_new']:>22,} {stats['n_valid_legacy']:>20,}")
    print(f"  {'Mean NDVI':<35} {stats['mean_new']:>22.4f} {stats['mean_legacy']:>20.4f}")
    print(f"  {'Std NDVI':<35} {stats['std_new']:>22.4f} {stats['std_legacy']:>20.4f}")
    print(f"  {'Min NDVI':<35} {stats['min_new']:>22.4f} {stats['min_legacy']:>20.4f}")
    print(f"  {'Max NDVI':<35} {stats['max_new']:>22.4f} {stats['max_legacy']:>20.4f}")

    n = stats["n_overlapping"]
    print(f"\n  --- Overlap statistics ({n:,} pixels with valid data in both) ---")

    if n < 10:
        print("  WARNING: fewer than 10 overlapping valid pixels -- comparison unreliable.")
        return

    corr = stats["correlation"]
    mad = stats["mean_abs_diff"]
    med = stats["median_abs_diff"]
    p005 = stats["pct_diff_gt_005"]
    p010 = stats["pct_diff_gt_010"]

    print(f"  {'Pearson correlation':<45} {corr:.4f}")
    print(f"  {'Mean absolute difference':<45} {mad:.4f}")
    print(f"  {'Median absolute difference':<45} {med:.4f}")
    print(f"  {'% pixels differing by > 0.05':<45} {p005:.2f}%")
    print(f"  {'% pixels differing by > 0.10':<45} {p010:.2f}%")
