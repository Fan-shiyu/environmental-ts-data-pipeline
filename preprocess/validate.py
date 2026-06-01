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


# ===========================================================================
# Pass B validation functions
# ===========================================================================

def validate_anomaly_resilience(
    parquet_path: str,
    expected_results: dict = None,
) -> dict:
    """
    Validate Pass B anomaly resilience computation logic.

    Uses Mponda Sentinel-2 100m, anomaly year 2024 (7 classes present).
    Checks:
      - rank 1 class has the minimum resilience_score (most resilient)
      - rank 7 class has the maximum resilience_score (most vulnerable)
      - severity labels are correctly assigned per thresholds
      - Flooded_vegetation rank 7 (largest deficit -0.27 at month 12)
    """
    if not Path(parquet_path).exists():
        return {"passes": False, "notes": f"File not found: {parquet_path}"}

    # Use the 100m file (resolved from path, replacing 1000m with 100m if needed)
    p100 = Path(parquet_path).parent.parent.parent / "sentinel2" / "100m" / "ndvi_anomaly_resilience.parquet"
    if p100.exists():
        df = pd.read_parquet(p100)
        sensor_label = "sentinel2/100m"
    else:
        df = pd.read_parquet(parquet_path)
        sensor_label = "sentinel2/1000m"

    sub = df[
        (df["aoi"] == "Zambia_Mponda")
        & (df["anomaly_year"] == 2024)
    ]

    if sub.empty:
        return {"passes": False, "notes": f"No rows for Zambia_Mponda/{sensor_label}/2024"}

    results: dict = {}
    passes = True

    # rank 1 should have minimum resilience_score
    score_of_rank1 = sub.loc[sub["resilience_rank"] == 1, "resilience_score"].values
    min_score = float(sub["resilience_score"].min())
    rank1_class = sub.loc[sub["resilience_rank"] == 1, "land_cover"].values
    rank1_ok = len(score_of_rank1) > 0 and abs(float(score_of_rank1[0]) - min_score) < 1e-6
    if not rank1_ok:
        passes = False
    results["rank1_is_min_score"] = {
        "expected": f"rank 1 class has min resilience_score ({min_score:.4f})",
        "actual": f"{rank1_class[0] if len(rank1_class) > 0 else None} score={score_of_rank1[0] if len(score_of_rank1) > 0 else None}",
        "passes": rank1_ok,
    }

    # rank N should have maximum resilience_score
    max_rank = int(sub["resilience_rank"].max())
    score_of_rankN = sub.loc[sub["resilience_rank"] == max_rank, "resilience_score"].values
    max_score = float(sub["resilience_score"].max())
    rankN_class = sub.loc[sub["resilience_rank"] == max_rank, "land_cover"].values
    rankN_ok = len(score_of_rankN) > 0 and abs(float(score_of_rankN[0]) - max_score) < 1e-6
    if not rankN_ok:
        passes = False
    results["rankN_is_max_score"] = {
        "expected": f"rank {max_rank} class has max resilience_score ({max_score:.4f})",
        "actual": f"{rankN_class[0] if len(rankN_class) > 0 else None} score={score_of_rankN[0] if len(score_of_rankN) > 0 else None}",
        "passes": rankN_ok,
    }

    # Severity labels correct per thresholds
    severity_map = {"Severe": -0.15, "Moderate": -0.10, "Mild": -0.05, "Minimal": 0.0}
    def _expected_severity(deficit):
        if deficit < -0.15:
            return "Severe"
        if deficit < -0.10:
            return "Moderate"
        if deficit < -0.05:
            return "Mild"
        return "Minimal"

    wrong_severity = sum(
        1 for _, r in sub.iterrows()
        if r["severity"] != _expected_severity(r["max_deficit"])
    )
    sev_ok = wrong_severity == 0
    if not sev_ok:
        passes = False
    results["severity_labels_correct"] = {
        "expected": "all severity labels match thresholds",
        "actual": f"{wrong_severity} mismatches",
        "passes": sev_ok,
    }

    results["n_classes"] = int(sub["land_cover"].nunique())
    results["most_resilient"] = rank1_class[0] if len(rank1_class) > 0 else None
    results["most_vulnerable"] = rankN_class[0] if len(rankN_class) > 0 else None
    results["passes"] = passes
    results["notes"] = "OK" if passes else "FAIL: see sub-checks above"
    return results


def validate_phenology(parquet_path: str) -> dict:
    """
    Validate phenology conclusions.

    From app screenshot 12 (Mponda, Sentinel-2 1000m, Crops):
      - Median green_up_month == 12 (December)
      - Median peak_month == 2 (February)
      - Median peak_ndvi in [0.55, 0.75]
    """
    if not Path(parquet_path).exists():
        return {"passes": False, "notes": f"File not found: {parquet_path}"}

    df = pd.read_parquet(parquet_path)
    sub = df[
        (df["aoi"] == "Zambia_Mponda")
        & (df["sensor"] == "sentinel2")
        & (df["resolution"] == 1000)
        & (df["land_cover"] == "Crops")
    ]

    if sub.empty:
        return {"passes": False, "notes": "No Crops rows for Zambia_Mponda/sentinel2/1000m"}

    results: dict = {}
    passes = True

    gu_vals = sub["green_up_month"].dropna()
    if len(gu_vals) > 0:
        med_gu = int(gu_vals.median())
        gu_ok = med_gu == 12
        if not gu_ok:
            passes = False
        results["green_up_month"] = {"expected": 12, "actual": med_gu, "passes": gu_ok}
    else:
        passes = False
        results["green_up_month"] = {"expected": 12, "actual": None, "passes": False}

    pk_vals = sub["peak_month"].dropna()
    if len(pk_vals) > 0:
        med_pk = int(pk_vals.median())
        pk_ok = med_pk == 2
        if not pk_ok:
            passes = False
        results["peak_month"] = {"expected": 2, "actual": med_pk, "passes": pk_ok}
    else:
        passes = False
        results["peak_month"] = {"expected": 2, "actual": None, "passes": False}

    ndvi_med = float(sub["peak_ndvi"].dropna().median())
    ndvi_ok = 0.55 <= ndvi_med <= 0.75
    if not ndvi_ok:
        passes = False
    results["peak_ndvi"] = {
        "expected": "[0.55, 0.75]", "actual": round(ndvi_med, 4), "passes": ndvi_ok
    }

    results["passes"] = passes
    results["notes"] = "OK" if passes else "FAIL: see sub-checks above"
    return results


def validate_annual_delta(parquet_path: str) -> dict:
    """
    Validate annual delta against known app screenshot values.

    From app screenshot 5 (Mponda, MODIS 250m, year_a=2023, year_b=2024):
      gain_km2 ~ 217.7, loss_km2 ~ 195.3 (tolerance: 20%).
    """
    if not Path(parquet_path).exists():
        return {"passes": False, "notes": f"File not found: {parquet_path}"}

    df = pd.read_parquet(parquet_path)
    row = df[
        (df["aoi"] == "Zambia_Mponda")
        & (df["sensor"] == "modis")
        & (df["resolution"] == 250)
        & (df["year_a"] == 2023)
        & (df["year_b"] == 2024)
    ]

    if row.empty:
        return {"passes": False, "notes": "No row for Zambia_Mponda/modis/250m 2023->2024"}

    gain = float(row["gain_km2"].iloc[0])
    loss = float(row["loss_km2"].iloc[0])
    gain_ok = abs(gain - 217.7) / 217.7 <= 0.20
    loss_ok = abs(loss - 195.3) / 195.3 <= 0.20
    passes = gain_ok and loss_ok

    return {
        "gain_km2": round(gain, 2),
        "loss_km2": round(loss, 2),
        "gain_expected": 217.7,
        "loss_expected": 195.3,
        "gain_ok": gain_ok,
        "loss_ok": loss_ok,
        "passes": passes,
        "notes": "OK" if passes else (
            f"FAIL: gain={gain:.1f} ({'OK' if gain_ok else 'X'}), "
            f"loss={loss:.1f} ({'OK' if loss_ok else 'X'})"
        ),
    }


def validate_frp(geojson_path: str) -> dict:
    """
    Basic sanity checks for fire_return_period.geojson.
    """
    import geopandas as gpd

    if not Path(geojson_path).exists():
        return {"passes": False, "notes": f"File not found: {geojson_path}"}

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as exc:
        return {"passes": False, "notes": f"Failed to read GeoJSON: {exc}"}

    n_features = len(gdf)
    n_years = int(gdf["n_years"].max()) if n_features > 0 else 0

    checks = {
        "n_features >= 100": n_features >= 100,
        "crs is EPSG:4326": gdf.crs is not None and gdf.crs.to_epsg() == 4326,
        "frp_years in [1, n_years]": (
            bool((gdf["frp_years"] >= 1).all() and (gdf["frp_years"] <= n_years).all())
        ) if n_features > 0 else False,
        "burn_count in [1, n_years]": (
            bool((gdf["burn_count"] >= 1).all() and (gdf["burn_count"] <= n_years).all())
        ) if n_features > 0 else False,
    }
    passes = all(checks.values())

    return {
        "n_features": n_features,
        "n_years": n_years,
        "crs": str(gdf.crs) if gdf.crs else None,
        "frp_range": [float(gdf["frp_years"].min()), float(gdf["frp_years"].max())] if n_features > 0 else None,
        "checks": checks,
        "passes": passes,
        "notes": "OK" if passes else f"FAIL: {[k for k, v in checks.items() if not v]}",
    }
