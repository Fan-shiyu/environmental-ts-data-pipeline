"""
Validation functions for Pass A preprocessing outputs.

Goal: confirm Python-computed Parquet values agree with the R app conclusions —
same trend direction, same order of magnitude. Not bit-exact floating point match.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask

from preprocess.core import _load_geometry, _masked_mean, _ndvi_files, _parse_ym


def validate_ndvi_monthly(
    parquet_path: str,
    config: dict,
    aoi: str,
    sensor: str,
    resolution: int,
    test_year: int = 2024,
    test_month: int = 3,
) -> dict:
    """
    Compare monthly NDVI mean in Parquet against a fresh computation from the TIF.

    The fresh computation uses the same rasterio + polygon mask method as core.py,
    so this validates that the Parquet was written correctly (not a method difference).

    Returns dict with:
      python_mean   : value from Parquet
      raster_mean   : value computed fresh from TIF
      abs_diff      : absolute difference
      passes        : bool (abs_diff < 0.01 is acceptable)
      notes         : explanation string
    """
    df = pd.read_parquet(parquet_path)
    row = df[(df["aoi"] == aoi) & (df["year"] == test_year) & (df["month"] == test_month)]

    if row.empty:
        return {
            "python_mean": float("nan"),
            "raster_mean": float("nan"),
            "abs_diff": float("nan"),
            "passes": False,
            "notes": f"No row found for {aoi} {test_year}-{test_month:02d} in {parquet_path}",
        }

    python_mean = float(row["mean_ndvi"].iloc[0])

    # Re-compute from TIF
    aoi_geom = _load_geometry(config["aois"][aoi]["path"])
    files = _ndvi_files(aoi, sensor, resolution, config)
    tif = next(
        (f for f in files if _parse_ym(f) == (test_year, test_month)),
        None,
    )
    if tif is None:
        return {
            "python_mean": python_mean,
            "raster_mean": float("nan"),
            "abs_diff": float("nan"),
            "passes": False,
            "notes": f"Source TIF not found for {test_year}-{test_month:02d}",
        }

    raster_mean, _ = _masked_mean(tif, aoi_geom)
    abs_diff = abs(python_mean - raster_mean)
    passes = abs_diff < 0.01

    return {
        "python_mean": round(python_mean, 6),
        "raster_mean": round(raster_mean, 6),
        "abs_diff": round(abs_diff, 8),
        "passes": passes,
        "notes": "OK" if passes else f"FAIL: diff {abs_diff:.6f} exceeds 0.01 threshold",
    }


def validate_trend_conclusions(
    trend_stats_path: str,
    expected_results: dict,
) -> dict:
    """
    Validate that trend conclusions (not exact p-values) match known app outputs.

    expected_results format:
    {
        ('Zambia_Mponda', 'modis', 1000): {
            'trend_direction': 'decreasing',
        }
    }

    Returns dict mapping each key to {'expected', 'actual', 'passes', 'notes'}.
    """
    df = pd.read_parquet(trend_stats_path)
    results = {}

    for key, expected in expected_results.items():
        aoi, sensor, resolution = key
        row = df[
            (df["aoi"] == aoi)
            & (df["sensor"] == sensor)
            & (df["resolution"] == resolution)
        ]

        if row.empty:
            results[key] = {
                "expected": expected,
                "actual": None,
                "passes": False,
                "notes": f"No row found for {aoi}/{sensor}/{resolution}m",
            }
            continue

        actual_direction = str(row["trend_direction"].iloc[0])
        expected_direction = expected.get("trend_direction")
        passes = actual_direction == expected_direction

        results[key] = {
            "expected": expected_direction,
            "actual": actual_direction,
            "passes": passes,
            "notes": "OK" if passes else f"FAIL: expected '{expected_direction}', got '{actual_direction}'",
        }

    return results


def validate_ba_monthly(
    parquet_path: str,
    config: dict,
    aoi: str,
    year: int = 2024,
    month: int = 8,
    expected_range: tuple[float, float] = (300.0, 500.0),
) -> dict:
    """
    Check burned area km² for a specific month is in a plausible range.

    Known target from app screenshot: Zambia_WL August 2024 ~ 300–500 km².
    Tolerance for geodesic vs fixed-pixel differences: within 5%.

    Returns dict with:
      burned_km2    : value from Parquet
      in_range      : bool (value within expected_range)
      passes        : bool
      notes         : explanation
    """
    df = pd.read_parquet(parquet_path)
    row = df[(df["aoi"] == aoi) & (df["year"] == year) & (df["month"] == month)]

    if row.empty:
        return {
            "burned_km2": float("nan"),
            "in_range": False,
            "passes": False,
            "notes": f"No row for {aoi} {year}-{month:02d} in {parquet_path}",
        }

    burned_km2 = float(row["burned_km2"].iloc[0])
    lo, hi = expected_range
    in_range = lo <= burned_km2 <= hi

    return {
        "burned_km2": round(burned_km2, 2),
        "in_range": in_range,
        "passes": in_range,
        "notes": "OK" if in_range else f"FAIL: {burned_km2:.1f} km² outside expected range [{lo}, {hi}]",
    }
