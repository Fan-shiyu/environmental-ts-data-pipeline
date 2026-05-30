"""
Python reimplementation of statistical computations from the Shiny app:
  - app/src/utilities.R         (raster loading, global means)
  - app/src/generate_plots.R    (time series aggregation, ribbons)
  - app/src/scenario_analysis.R (per-class NDVI, baselines)

All functions read from the deployed TIFs in the Shiny app's www/data/ folder
(config keys legacy_data_root and legacy_burned_area_root) and produce
pandas DataFrames ready to write as Parquet.
"""

import datetime
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pymannkendall as mk
import rasterio
import rasterio.features
import yaml
from pyproj import Geod
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, shape as shapely_shape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LC_CLASSES = [
    "Bare_ground",
    "Built_Area",
    "Crops",
    "Flooded_vegetation",
    "Rangeland",
    "Trees",
    "Water",
]

# Maps (sensor, resolution_metres) → app resolution folder name
RESOLUTION_FOLDER_MAP: dict[tuple[str, int], str] = {
    ("sentinel2",    100): "100m_resolution",
    ("sentinel2",   1000): "Sentinel_1000m_resolution",
    ("modis",        250): "250m_resolution",
    ("modis",        500): "500m_resolution",
    ("modis",       1000): "MODIS_1000m_resolution",
    ("burned_area",  500): "500m_resolution",
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_geometry(path: str) -> list:
    """Load GeoJSON → EPSG:4326, return list of shapely geometries."""
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    return [mapping(geom) for geom in gdf.geometry]


def _masked_array(tif_path: Path, geometries: list) -> np.ndarray:
    """Open raster, apply polygon mask, return float32 array with NaN outside polygon."""
    with rasterio.open(tif_path) as src:
        out, _ = rio_mask(src, geometries, crop=False, nodata=np.nan, filled=True)
        data = out[0].astype(np.float32)
        # Replace rasterio nodata with NaN
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        return data


def _masked_mean(tif_path: Path, geometries: list) -> tuple[float, int]:
    """Return (nanmean, n_valid_pixels) after applying polygon mask."""
    data = _masked_array(tif_path, geometries)
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return float("nan"), 0
    return float(np.nanmean(valid)), int(len(valid))


def _parse_ym(path: Path) -> tuple[int, int]:
    """Parse YYYY-MM from filename like 2024-03_NDVI_Zambia_Mponda.tif."""
    m = re.match(r"(\d{4})-(\d{2})_", path.name)
    if not m:
        raise ValueError(f"Cannot parse year/month from filename: {path.name}")
    return int(m.group(1)), int(m.group(2))


def _ndvi_files(aoi: str, sensor: str, resolution: int, config: dict) -> list[Path]:
    """List all NDVI TIF files for given aoi/sensor/resolution, sorted by name."""
    folder_key = RESOLUTION_FOLDER_MAP[(sensor, resolution)]
    root = Path(config["legacy_data_root"])
    folder = root / aoi / folder_key
    if not folder.exists():
        return []
    return sorted(folder.glob("*.tif"))


def _ba_files(aoi: str, config: dict) -> list[Path]:
    """List all burned area TIF files for given AoI, sorted by name."""
    root = Path(config["legacy_burned_area_root"])
    folder = root / aoi / "500m_resolution"
    if not folder.exists():
        return []
    return sorted(folder.glob("*.tif"))


def _lulc_path(aoi: str, lc_class: str, config: dict) -> Path:
    """Return path to land cover GeoJSON for given AoI and class (always 2023)."""
    lulc_root = Path(config["legacy_data_root"]).parent / "LandUse"
    return lulc_root / aoi / "S2_10m_LULC_2023" / f"{aoi}_{lc_class}_2023.geojson"


def _geodesic_row_areas_km2(transform, nrows: int, ncols: int) -> np.ndarray:
    """
    Compute geodesic area in km² for each pixel row of a raster in EPSG:4326.

    Returns array of shape (nrows,) where element i is the area of one pixel
    in row i. Equivalent to terra::expanse(unit='km') per pixel.
    """
    geod = Geod(ellps="WGS84")
    # transform.e is negative (north-up), so pixel height is -transform.e
    dy = transform.e  # degrees per pixel in y (negative)
    dx = transform.a  # degrees per pixel in x (positive)

    areas = np.empty(nrows, dtype=np.float64)
    for row_idx in range(nrows):
        lat_top = transform.f + row_idx * dy
        lat_bot = lat_top + dy  # dy < 0 so lat_bot < lat_top
        lon_l = transform.c
        lon_r = lon_l + dx
        # Polygon for ONE pixel cell
        lons = [lon_l, lon_r, lon_r, lon_l]
        lats = [lat_top, lat_top, lat_bot, lat_bot]
        area_m2, _ = geod.polygon_area_perimeter(lons, lats)
        areas[row_idx] = abs(area_m2) / 1e6  # m² → km²
    return areas


# ---------------------------------------------------------------------------
# Table 1 — ndvi_monthly.parquet
# ---------------------------------------------------------------------------

def compute_ndvi_monthly(
    aoi: str,
    sensor: str,
    resolution: int,
    config: dict,
) -> pd.DataFrame:
    """
    AoI-wide monthly NDVI means across all available years.

    Equivalent to R: terra::global(ndvi_rast, fun='mean', na.rm=TRUE)
    Applied after AoI polygon mask (same mask used during backfill download).
    """
    aoi_geom = _load_geometry(config["aois"][aoi]["path"])
    files = _ndvi_files(aoi, sensor, resolution, config)
    if not files:
        return pd.DataFrame(
            columns=["aoi", "sensor", "resolution", "year", "month", "mean_ndvi", "n_valid_px"]
        )

    rows = []
    for f in files:
        year, month = _parse_ym(f)
        mean_val, n_px = _masked_mean(f, aoi_geom)
        rows.append({
            "aoi": aoi,
            "sensor": sensor,
            "resolution": resolution,
            "year": year,
            "month": month,
            "mean_ndvi": mean_val,
            "n_valid_px": n_px,
        })

    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype("int32")
    df["month"] = df["month"].astype("int32")
    df["resolution"] = df["resolution"].astype("int32")
    df["n_valid_px"] = df["n_valid_px"].astype("int32")
    return df.sort_values(["year", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 2 — ndvi_monthly_baselines.parquet
# ---------------------------------------------------------------------------

def compute_ndvi_monthly_baselines(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical statistics per calendar month computed from Table 1.

    ts_lower / ts_upper: min/max across all years for each calendar month.
    Equivalent to R: get_monthly_historic_range() — NOT percentiles.

    climatology: mean across all years.
    Equivalent to R: get_monthly_climatology()
    """
    if monthly_df.empty:
        return pd.DataFrame()

    meta_cols = ["aoi", "sensor", "resolution"]
    aoi, sensor, resolution = (
        monthly_df["aoi"].iloc[0],
        monthly_df["sensor"].iloc[0],
        monthly_df["resolution"].iloc[0],
    )

    grp = monthly_df.groupby("month")["mean_ndvi"]
    baselines = pd.DataFrame({
        "month": grp.mean().index,
        "ts_lower": grp.min().values,
        "ts_upper": grp.max().values,
        "climatology": grp.mean().values,
        "hist_std": grp.std(ddof=1).values,
        "n_years": grp.count().values,
    })
    baselines.insert(0, "resolution", resolution)
    baselines.insert(0, "sensor", sensor)
    baselines.insert(0, "aoi", aoi)
    baselines["month"] = baselines["month"].astype("int32")
    baselines["resolution"] = baselines["resolution"].astype("int32")
    baselines["n_years"] = baselines["n_years"].astype("int32")
    return baselines.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 3 — ndvi_annual.parquet + ndvi_trend_stats.parquet
# ---------------------------------------------------------------------------

def compute_ndvi_annual(
    monthly_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Annual aggregates and trend statistics from Table 1 output.

    Annual view:
      - mean of monthly means per year
      - is_complete = (n_months == 12)  — incomplete years excluded from stats
      - typical range = Q1/Q3 of complete years

    Trend stats:
      - Annual MK: trend::mk.test() equivalent, complete years only, ≥5 required
      - Seasonal MK: trend::smk.test() equivalent, full monthly series, ≥60 required
      - trend_direction uses p < 0.05 (status badge threshold)
      - mk_p and smk_p also stored so caller can apply p < 0.1 for trend line
    """
    if monthly_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    aoi = monthly_df["aoi"].iloc[0]
    sensor = monthly_df["sensor"].iloc[0]
    resolution = int(monthly_df["resolution"].iloc[0])

    # Annual aggregates
    annual = (
        monthly_df.groupby("year")
        .agg(mean_ndvi=("mean_ndvi", "mean"), n_months=("month", "count"))
        .reset_index()
    )
    annual["is_complete"] = annual["n_months"] == 12
    annual.insert(0, "resolution", resolution)
    annual.insert(0, "sensor", sensor)
    annual.insert(0, "aoi", aoi)
    annual["year"] = annual["year"].astype("int32")
    annual["n_months"] = annual["n_months"].astype("int32")
    annual["resolution"] = annual["resolution"].astype("int32")

    # Trend stats
    complete = annual[annual["is_complete"]].copy()
    n_complete = len(complete)

    trend_row: dict = {
        "aoi": aoi,
        "sensor": sensor,
        "resolution": resolution,
        "n_complete_years": n_complete,
        "mk_p": float("nan"),
        "sen_slope": float("nan"),
        "trend_direction": "insufficient_data",
        "q1_ndvi": float("nan"),
        "q3_ndvi": float("nan"),
        "min_ndvi": float("nan"),
        "max_ndvi": float("nan"),
        "long_term_avg": float("nan"),
        "smk_p": float("nan"),
        "smk_sen_slope": float("nan"),
    }

    if n_complete >= 5:
        vals = complete["mean_ndvi"].values
        result = mk.original_test(vals)
        trend_row["mk_p"] = float(result.p)
        trend_row["sen_slope"] = float(result.slope)
        if result.p < 0.05:
            trend_row["trend_direction"] = (
                "decreasing" if result.slope < 0 else "increasing"
            )
        else:
            trend_row["trend_direction"] = "no_trend"

        trend_row["q1_ndvi"] = float(np.percentile(vals, 25))
        trend_row["q3_ndvi"] = float(np.percentile(vals, 75))
        trend_row["min_ndvi"] = float(vals.min())
        trend_row["max_ndvi"] = float(vals.max())
        trend_row["long_term_avg"] = float(vals.mean())

    # Seasonal MK on full monthly series (must be sorted by year then month)
    monthly_sorted = monthly_df.sort_values(["year", "month"])
    monthly_series = monthly_sorted["mean_ndvi"].dropna().values
    if len(monthly_series) >= 60:
        smk_result = mk.seasonal_test(monthly_series, period=12)
        trend_row["smk_p"] = float(smk_result.p)
        trend_row["smk_sen_slope"] = float(smk_result.slope)

    trend_df = pd.DataFrame([trend_row])
    trend_df["resolution"] = trend_df["resolution"].astype("int32")
    trend_df["n_complete_years"] = trend_df["n_complete_years"].astype("int32")

    return annual.reset_index(drop=True), trend_df


# ---------------------------------------------------------------------------
# Table 4 — ndvi_monthly_by_class.parquet
# ---------------------------------------------------------------------------

def compute_ndvi_monthly_by_class(
    aoi: str,
    sensor: str,
    resolution: int,
    config: dict,
) -> pd.DataFrame:
    """
    Per-land-cover-class monthly NDVI means across all years.

    Masking order (MUST match R):
      1. Apply AoI polygon mask first
      2. Then apply class polygon mask
    Equivalent to R: terra::global(terra::mask(ndvi_rast, lc_polygon), fun='mean')

    Land cover year is always 2023 regardless of NDVI year.
    Class name comes from filename — GeoJSONs have no attribute properties.
    """
    aoi_geom = _load_geometry(config["aois"][aoi]["path"])
    files = _ndvi_files(aoi, sensor, resolution, config)
    if not files:
        return pd.DataFrame(
            columns=[
                "aoi", "sensor", "resolution", "year", "month",
                "land_cover", "mean_ndvi", "n_valid_px",
            ]
        )

    # Load class geometries once (2023 only)
    class_geoms: dict[str, list] = {}
    lulc_root = Path(config["legacy_data_root"]).parent / "LandUse"
    for lc in LC_CLASSES:
        lc_path = lulc_root / aoi / "S2_10m_LULC_2023" / f"{aoi}_{lc}_2023.geojson"
        if lc_path.exists():
            class_geoms[lc] = _load_geometry(str(lc_path))
        else:
            print(f"  [warn] LC GeoJSON not found, skipping: {lc_path}")

    if not class_geoms:
        print(f"  [warn] No LC GeoJSONs found for {aoi} — skipping by-class tables")
        return pd.DataFrame(
            columns=[
                "aoi", "sensor", "resolution", "year", "month",
                "land_cover", "mean_ndvi", "n_valid_px",
            ]
        )

    rows = []
    for f in files:
        year, month = _parse_ym(f)

        # Step 1: Apply AoI mask to get base array
        aoi_data = _masked_array(f, aoi_geom)

        for lc, lc_geom in class_geoms.items():
            # Step 2: Apply class mask on top of AoI-masked data
            # Re-open with rasterio to apply class mask cleanly
            with rasterio.open(f) as src:
                out, _ = rio_mask(src, lc_geom, crop=False, nodata=np.nan, filled=True)
                class_data = out[0].astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    class_data[class_data == nodata] = np.nan

            # Combine: pixel must be valid in BOTH aoi and class masks
            combined = class_data.copy()
            combined[~np.isfinite(aoi_data)] = np.nan

            valid = combined[np.isfinite(combined)]
            mean_val = float(np.nanmean(valid)) if len(valid) > 0 else float("nan")
            n_px = int(len(valid))

            rows.append({
                "aoi": aoi,
                "sensor": sensor,
                "resolution": resolution,
                "year": year,
                "month": month,
                "land_cover": lc,
                "mean_ndvi": mean_val,
                "n_valid_px": n_px,
            })

    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype("int32")
    df["month"] = df["month"].astype("int32")
    df["resolution"] = df["resolution"].astype("int32")
    df["n_valid_px"] = df["n_valid_px"].astype("int32")
    return df.sort_values(["year", "month", "land_cover"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 5 — ndvi_monthly_baselines_by_class.parquet
# ---------------------------------------------------------------------------

def compute_class_baselines(by_class_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-class per-month historical statistics from Table 4 output.

    95% CI: mean ± 1.96 * std / sqrt(n)
    Equivalent to R: get_summary_ndvi_df()

    NOTE: lower_ci is NOT clamped to 0 for NDVI (only burned area gets that treatment).
    """
    if by_class_df.empty:
        return pd.DataFrame()

    aoi = by_class_df["aoi"].iloc[0]
    sensor = by_class_df["sensor"].iloc[0]
    resolution = int(by_class_df["resolution"].iloc[0])

    def _ci(grp: pd.Series) -> pd.Series:
        n = grp.count()
        mean = grp.mean()
        std = grp.std(ddof=1)
        se = std / np.sqrt(n) if n > 1 else float("nan")
        return pd.Series({
            "lc_mean": mean,
            "lc_lower_ci": mean - 1.96 * se,
            "lc_upper_ci": mean + 1.96 * se,
            "hist_mean": mean,
            "hist_sd": std,
            "n_years": n,
        })

    # First aggregate to per-year means per (land_cover, month)
    yr_means = (
        by_class_df.groupby(["land_cover", "year", "month"])["mean_ndvi"]
        .mean()
        .reset_index()
        .rename(columns={"mean_ndvi": "yr_mean"})
    )

    baselines = (
        yr_means.groupby(["land_cover", "month"])["yr_mean"]
        .apply(_ci)
        .reset_index()
    )
    # _ci returns a Series; after apply the level structure needs flattening
    if "level_2" in baselines.columns:
        baselines = baselines.pivot_table(
            index=["land_cover", "month"], columns="level_2", values="yr_mean"
        ).reset_index()
        baselines.columns.name = None

    baselines.insert(0, "resolution", resolution)
    baselines.insert(0, "sensor", sensor)
    baselines.insert(0, "aoi", aoi)
    baselines["month"] = baselines["month"].astype("int32")
    baselines["resolution"] = baselines["resolution"].astype("int32")
    if "n_years" in baselines.columns:
        baselines["n_years"] = baselines["n_years"].astype("int32")
    return baselines.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 6 — ba_monthly.parquet
# ---------------------------------------------------------------------------

def compute_ba_monthly(aoi: str, config: dict) -> pd.DataFrame:
    """
    Monthly burned area in km² using geodesic pixel areas.

    Uses pyproj.Geod to compute latitude-dependent cell areas,
    equivalent to R: terra::expanse(burned_rast, unit='km').

    Historical baseline: 95% CI with lower_ci clamped to 0.
    Equivalent to R: get_summary_ba_df()
    """
    aoi_geom = _load_geometry(config["aois"][aoi]["path"])
    files = _ba_files(aoi, config)
    if not files:
        return pd.DataFrame(
            columns=[
                "aoi", "resolution", "year", "month", "burned_km2",
                "ba_mean", "ba_lower_ci", "ba_upper_ci", "n_years",
            ]
        )

    rows = []
    for f in files:
        year, month = _parse_ym(f)
        with rasterio.open(f) as src:
            out, _ = rio_mask(src, aoi_geom, crop=False, nodata=0, filled=True)
            data = out[0].astype(np.int32)
            transform = src.transform
            nrows, ncols = data.shape

        # Geodesic area per row (km²/pixel)
        row_areas = _geodesic_row_areas_km2(transform, nrows, ncols)
        pixel_areas = np.broadcast_to(row_areas[:, np.newaxis], (nrows, ncols))

        burned_mask = data > 0
        burned_km2 = float(np.sum(pixel_areas[burned_mask]))

        rows.append({
            "aoi": aoi,
            "resolution": 500,
            "year": year,
            "month": month,
            "burned_km2": burned_km2,
        })

    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype("int32")
    df["month"] = df["month"].astype("int32")
    df["resolution"] = df["resolution"].astype("int32")

    # Historical baseline per calendar month
    grp = df.groupby("month")["burned_km2"]
    yr_means = df.groupby(["month", "year"])["burned_km2"].mean()
    mo_stats = yr_means.groupby("month").agg(
        ba_mean="mean",
        ba_std="std",
        n_years="count",
    ).reset_index()
    mo_stats["ba_se"] = mo_stats["ba_std"] / np.sqrt(mo_stats["n_years"])
    mo_stats["ba_lower_ci"] = (mo_stats["ba_mean"] - 1.96 * mo_stats["ba_se"]).clip(lower=0)
    mo_stats["ba_upper_ci"] = mo_stats["ba_mean"] + 1.96 * mo_stats["ba_se"]

    df = df.merge(mo_stats[["month", "ba_mean", "ba_lower_ci", "ba_upper_ci", "n_years"]], on="month")
    df["n_years"] = df["n_years"].astype("int32")
    return df.sort_values(["year", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 7 — ba_daily.parquet
# ---------------------------------------------------------------------------

def compute_ba_daily(aoi: str, config: dict) -> pd.DataFrame:
    """
    Per-day burned area from BurnDate Julian day rasters.

    Uses FIXED pixel area of 0.25 km² for 500 m resolution,
    matching R: get_ba_daily_activity(pixel_area_km2 = (res_m/1000)^2).

    Julian day conversion:
      date = datetime(year, 1, 1) + timedelta(days=julian_day - 1)
    Equivalent to R: as.Date(vals - 1, origin = paste0(year, '-01-01'))
    """
    PIXEL_AREA_KM2 = 0.25  # (500m / 1000)^2

    aoi_geom = _load_geometry(config["aois"][aoi]["path"])
    files = _ba_files(aoi, config)
    if not files:
        return pd.DataFrame(
            columns=["aoi", "resolution", "year", "date", "burned_km2"]
        )

    rows = []
    for f in files:
        year, _ = _parse_ym(f)
        with rasterio.open(f) as src:
            out, _ = rio_mask(src, aoi_geom, crop=False, nodata=0, filled=True)
            data = out[0].astype(np.int32)

        vals = data.ravel()
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue

        # Convert Julian day → calendar date
        origin = datetime.date(year, 1, 1)
        for jday, count in zip(*np.unique(vals, return_counts=True)):
            cal_date = origin + datetime.timedelta(days=int(jday) - 1)
            rows.append({
                "aoi": aoi,
                "resolution": 500,
                "year": year,
                "date": cal_date,
                "burned_km2": int(count) * PIXEL_AREA_KM2,
            })

    if not rows:
        return pd.DataFrame(
            columns=["aoi", "resolution", "year", "date", "burned_km2"]
        )

    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype("int32")
    df["resolution"] = df["resolution"].astype("int32")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["year", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 8 — ndvi_anomaly_monthly.parquet
# ---------------------------------------------------------------------------

def compute_ndvi_anomaly_monthly(
    aoi: str,
    sensor: str,
    resolution: int,
    processed_root: str,
) -> pd.DataFrame:
    """
    Monthly NDVI anomaly per land cover class: mean_ndvi - hist_mean.

    Reads two existing Pass A outputs (no raster reads):
      - ndvi_monthly_by_class.parquet
      - ndvi_monthly_baselines_by_class.parquet

    Skips gracefully when either input is absent (Zambia_WL — no LULC data).
    """
    base = Path(processed_root) / aoi / sensor / f"{resolution}m"
    by_class_path = base / "ndvi_monthly_by_class.parquet"
    baselines_path = base / "ndvi_monthly_baselines_by_class.parquet"

    if not by_class_path.exists():
        print(
            f"  [warn] Skipping ndvi_anomaly_monthly for {aoi}/{sensor}/{resolution}m: "
            f"ndvi_monthly_by_class not found (no LULC data)"
        )
        return pd.DataFrame(
            columns=["aoi", "sensor", "resolution", "year", "month",
                     "land_cover", "anomaly_value", "mean_ndvi", "hist_mean"]
        )
    if not baselines_path.exists():
        print(
            f"  [warn] Skipping ndvi_anomaly_monthly for {aoi}/{sensor}/{resolution}m: "
            f"ndvi_monthly_baselines_by_class not found (no LULC data)"
        )
        return pd.DataFrame(
            columns=["aoi", "sensor", "resolution", "year", "month",
                     "land_cover", "anomaly_value", "mean_ndvi", "hist_mean"]
        )

    by_class = pd.read_parquet(by_class_path)
    baselines = pd.read_parquet(baselines_path)

    if by_class.empty or baselines.empty:
        return pd.DataFrame(
            columns=["aoi", "sensor", "resolution", "year", "month",
                     "land_cover", "anomaly_value", "mean_ndvi", "hist_mean"]
        )

    merged = by_class.merge(
        baselines[["aoi", "sensor", "resolution", "land_cover", "month", "hist_mean"]],
        on=["aoi", "sensor", "resolution", "land_cover", "month"],
        how="inner",
    )

    merged["anomaly_value"] = (merged["mean_ndvi"] - merged["hist_mean"]).astype("float64")
    merged["mean_ndvi"] = merged["mean_ndvi"].astype("float64")
    merged["hist_mean"] = merged["hist_mean"].astype("float64")

    out = merged[["aoi", "sensor", "resolution", "year", "month", "land_cover",
                  "anomaly_value", "mean_ndvi", "hist_mean"]].copy()
    out["year"] = out["year"].astype("int32")
    out["month"] = out["month"].astype("int32")
    out["resolution"] = out["resolution"].astype("int32")
    return out.sort_values(["year", "month", "land_cover"]).reset_index(drop=True)


# ===========================================================================
# Pass B — Derived Analytical Tables
# ===========================================================================

# ---------------------------------------------------------------------------
# B-Table 1 — ndvi_annual_by_class.parquet
# ---------------------------------------------------------------------------

def compute_ndvi_annual_by_class(
    aoi: str,
    sensor: str,
    resolution: int,
    processed_root: str,
) -> pd.DataFrame:
    """
    Annual NDVI statistics per land cover class.

    Reads ndvi_monthly_by_class.parquet (Pass A Table 4).
    Cross-year stats (hist_min, hist_max, hist_mean_all, hist_sd, cv) are
    computed across ALL years and denormalized onto each per-year row.
    Equivalent to R: .compute_productivity_stats()
    """
    base = Path(processed_root) / aoi / sensor / f"{resolution}m"
    by_class_path = base / "ndvi_monthly_by_class.parquet"

    _empty_cols = ["aoi", "sensor", "resolution", "year", "land_cover",
                   "annual_mean", "hist_min", "hist_max", "hist_mean_all", "hist_sd", "cv"]

    if not by_class_path.exists():
        print(
            f"  [warn] Skipping ndvi_annual_by_class for {aoi}/{sensor}/{resolution}m: "
            f"ndvi_monthly_by_class not found (no LULC data)"
        )
        return pd.DataFrame(columns=_empty_cols)

    by_class = pd.read_parquet(by_class_path)
    if by_class.empty:
        return pd.DataFrame(columns=_empty_cols)

    # Step 1: Annual mean per (year, land_cover) — mean of monthly means
    annual = (
        by_class.groupby(["aoi", "sensor", "resolution", "year", "land_cover"])["mean_ndvi"]
        .mean()
        .reset_index()
        .rename(columns={"mean_ndvi": "annual_mean"})
    )

    # Step 2: Cross-year stats per class — ddof=1 matches R's sd()
    grp = annual.groupby(["aoi", "sensor", "resolution", "land_cover"])["annual_mean"]
    stats = grp.agg(hist_min="min", hist_max="max", hist_mean_all="mean").reset_index()
    std_df = (
        grp.std(ddof=1)
        .reset_index()
        .rename(columns={"annual_mean": "hist_sd"})
    )
    stats = stats.merge(std_df, on=["aoi", "sensor", "resolution", "land_cover"])
    stats["cv"] = np.where(
        stats["hist_mean_all"] != 0,
        stats["hist_sd"] / stats["hist_mean_all"].abs(),
        np.nan,
    )

    # Step 3: Denormalize — join stats onto each per-year row
    out = annual.merge(
        stats[["aoi", "sensor", "resolution", "land_cover",
               "hist_min", "hist_max", "hist_mean_all", "hist_sd", "cv"]],
        on=["aoi", "sensor", "resolution", "land_cover"],
        how="left",
    )

    out["year"] = out["year"].astype("int32")
    out["resolution"] = out["resolution"].astype("int32")
    for col in ("annual_mean", "hist_min", "hist_max", "hist_mean_all", "hist_sd", "cv"):
        out[col] = out[col].astype("float64")
    return out.sort_values(["year", "land_cover"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# B-Table 2 — ndvi_anomaly_resilience.parquet
# ---------------------------------------------------------------------------

def compute_anomaly_resilience(
    aoi: str,
    sensor: str,
    resolution: int,
    processed_root: str,
) -> pd.DataFrame:
    """
    Per-class per-year annual resilience summary for complete years only.

    Reads ndvi_anomaly_monthly.parquet (Pass A Table 8) and
    ndvi_monthly_baselines_by_class.parquet (Pass A Table 5).
    Feeds Anomaly Resilience bar chart and summary table.
    """
    base = Path(processed_root) / aoi / sensor / f"{resolution}m"
    anomaly_path = base / "ndvi_anomaly_monthly.parquet"
    baselines_path = base / "ndvi_monthly_baselines_by_class.parquet"

    _empty_cols = [
        "aoi", "sensor", "resolution", "anomaly_year", "land_cover",
        "max_deficit", "deficit_month", "recovery_months",
        "resilience_score", "resilience_rank", "severity",
    ]

    for p, label in [
        (anomaly_path, "ndvi_anomaly_monthly"),
        (baselines_path, "ndvi_monthly_baselines_by_class"),
    ]:
        if not p.exists():
            print(
                f"  [warn] Skipping ndvi_anomaly_resilience for {aoi}/{sensor}/{resolution}m: "
                f"{label} not found (no LULC data)"
            )
            return pd.DataFrame(columns=_empty_cols)

    anomaly = pd.read_parquet(anomaly_path)
    baselines = pd.read_parquet(baselines_path)

    if anomaly.empty:
        return pd.DataFrame(columns=_empty_cols)

    # Join hist_sd onto anomaly rows for recovery-threshold check
    anomaly = anomaly.merge(
        baselines[["aoi", "sensor", "resolution", "land_cover", "month", "hist_sd"]],
        on=["aoi", "sensor", "resolution", "land_cover", "month"],
        how="left",
    )

    # Complete years: all 12 months × all classes present
    n_classes = anomaly["land_cover"].nunique()
    yr_counts = anomaly.groupby("year").apply(
        lambda g: g[["month", "land_cover"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    complete_years = yr_counts[yr_counts == 12 * n_classes].index.tolist()
    if not complete_years:
        return pd.DataFrame(columns=_empty_cols)

    anomaly = anomaly[anomaly["year"].isin(complete_years)].copy()

    # Pass 1: raw deficit + raw recovery per (year, land_cover)
    raw_rows = []
    for (year, lc), grp in anomaly.groupby(["year", "land_cover"]):
        grp = grp.sort_values("month")
        valid_av = grp["anomaly_value"].dropna()
        if valid_av.empty:
            continue  # all anomaly values NaN for this class/year — skip
        idx_min = valid_av.idxmin()
        max_deficit = float(grp.loc[idx_min, "anomaly_value"])
        deficit_month = int(grp.loc[idx_min, "month"])

        # Recovery: first month AFTER deficit_month where |mean_ndvi - hist_mean| <= hist_sd
        rm_raw = None
        for _, row in grp[grp["month"] > deficit_month].sort_values("month").iterrows():
            h_sd = row["hist_sd"]
            if pd.notna(h_sd) and abs(float(row["mean_ndvi"]) - float(row["hist_mean"])) <= float(h_sd):
                rm_raw = int(row["month"]) - deficit_month
                break

        severity = (
            "Severe" if max_deficit < -0.15 else
            "Moderate" if max_deficit < -0.10 else
            "Mild" if max_deficit < -0.05 else
            "Minimal"
        )
        raw_rows.append({
            "year": year, "land_cover": lc,
            "max_deficit": max_deficit, "deficit_month": deficit_month,
            "_rm": rm_raw, "severity": severity,
        })

    raw_df = pd.DataFrame(raw_rows)

    # Pass 2: max_rec per year → resilience_score
    # Note: pandas stores None as NaN in numeric columns, so use pd.notna() throughout
    max_rec_by_year = (
        raw_df.groupby("year")["_rm"]
        .apply(lambda s: max((v for v in s if pd.notna(v)), default=None))
        .apply(lambda v: int(v) * 2 if pd.notna(v) else 12)
        .to_dict()
    )

    raw_df["resilience_score"] = raw_df.apply(
        lambda r: abs(r["max_deficit"]) * (int(r["_rm"]) if pd.notna(r["_rm"]) else max_rec_by_year[r["year"]]),
        axis=1,
    )
    rank_floats = raw_df.groupby("year")["resilience_score"].rank(method="min", ascending=True)
    raw_df["resilience_rank"] = pd.array(
        [pd.NA if pd.isna(v) else int(v) for v in rank_floats],
        dtype="Int32",
    )
    raw_df["recovery_months"] = pd.array(
        [pd.NA if pd.isna(v) else int(v) for v in raw_df["_rm"]],
        dtype="Int32",
    )

    raw_df["aoi"] = aoi
    raw_df["sensor"] = sensor
    raw_df["resolution"] = resolution
    out = raw_df.rename(columns={"year": "anomaly_year"})[_empty_cols].copy()
    out["anomaly_year"] = out["anomaly_year"].astype("int32")
    out["resolution"] = out["resolution"].astype("int32")
    out["deficit_month"] = out["deficit_month"].astype("int32")
    out["max_deficit"] = out["max_deficit"].astype("float64")
    out["resilience_score"] = out["resilience_score"].astype("float64")
    # resilience_rank already Int32 (nullable) from rank computation
    return out.sort_values(["anomaly_year", "land_cover"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# B-Table 3 — ndvi_phenology.parquet
# ---------------------------------------------------------------------------

def compute_phenology(
    aoi: str,
    sensor: str,
    resolution: int,
    processed_root: str,
    profiles_path: str = "config/phenology_profiles.yaml",
) -> pd.DataFrame:
    """
    Per-class per-year phenological event detection for Crops and Rangeland.

    Green-up, peak, and senescence detected from monthly NDVI series.
    Profiles loaded from config/phenology_profiles.yaml.
    AoI name mapped to country key by prefix match ("Zambia_Mponda" -> "Zambia").
    Equivalent to R phenological detection in scenario_analysis.R.
    """
    base = Path(processed_root) / aoi / sensor / f"{resolution}m"
    by_class_path = base / "ndvi_monthly_by_class.parquet"

    _empty_cols = [
        "aoi", "sensor", "resolution", "year", "land_cover",
        "green_up_month", "green_up_conf", "peak_month", "peak_ndvi",
        "peak_conf", "senescence_month", "season_length",
    ]

    if not by_class_path.exists():
        print(
            f"  [warn] Skipping ndvi_phenology for {aoi}/{sensor}/{resolution}m: "
            f"ndvi_monthly_by_class not found (no LULC data)"
        )
        return pd.DataFrame(columns=_empty_cols)

    with open(profiles_path) as fh:
        all_profiles = yaml.safe_load(fh)["profiles"]

    country = next((k for k in all_profiles if aoi.startswith(k)), None)
    if country is None:
        print(f"  [warn] No phenology profile for {aoi} in {profiles_path}")
        return pd.DataFrame(columns=_empty_cols)

    profiles = all_profiles[country]
    pheno_classes = [lc for lc in ("Crops", "Rangeland") if lc in profiles]

    by_class = pd.read_parquet(by_class_path)
    if by_class.empty:
        return pd.DataFrame(columns=_empty_cols)

    pheno_df = by_class[by_class["land_cover"].isin(pheno_classes)].copy()
    if pheno_df.empty:
        return pd.DataFrame(columns=_empty_cols)

    # Historical average NDVI per (land_cover, month) across all years
    hist_avg = (
        pheno_df.groupby(["land_cover", "month"])["mean_ndvi"]
        .mean()
        .to_dict()
    )

    rows = []
    for (year, lc), grp in pheno_df.groupby(["year", "land_cover"]):
        prof = profiles[lc]
        ndvi = {int(r["month"]): float(r["mean_ndvi"]) for _, r in grp.iterrows()}

        # --- Green-up ---
        gu_win = range(prof["green_up_window"][0], prof["green_up_window"][1] + 1)
        baseline = ndvi.get(prof["green_up_baseline_month"], float("nan"))
        gu_vals = {m: ndvi[m] for m in gu_win if m in ndvi and not np.isnan(ndvi[m])}

        if np.isnan(baseline) or not gu_vals:
            green_up_month = pd.NA
            green_up_conf = "Not detected"
        else:
            gu_thresh = prof["green_up_threshold"]
            rise = max(gu_vals.values()) - baseline
            if rise >= prof["green_up_conf_lo_min"]:
                crossed = sorted(m for m, v in gu_vals.items() if v > baseline + gu_thresh)
                green_up_month = crossed[0] if crossed else max(gu_vals, key=gu_vals.get)
                green_up_conf = (
                    "High" if rise > 2 * gu_thresh else
                    "Medium" if rise > gu_thresh else "Low"
                )
            else:
                green_up_month = pd.NA
                green_up_conf = "Not detected"

        # --- Peak ---
        pk_win = range(prof["peak_window"][0], prof["peak_window"][1] + 1)
        pk_vals = {m: ndvi[m] for m in pk_win if m in ndvi and not np.isnan(ndvi[m])}

        if pk_vals:
            peak_month = max(pk_vals, key=pk_vals.get)
            peak_ndvi_val = pk_vals[peak_month]
            h_avg = hist_avg.get((lc, peak_month), float("nan"))
            if not np.isnan(h_avg):
                dev = peak_ndvi_val - h_avg
                pk_delta = prof["peak_conf_delta"]
                peak_conf = "High" if dev > pk_delta else ("Low" if dev < -pk_delta else "Medium")
            else:
                peak_conf = "Medium"
        else:
            peak_month = pd.NA
            peak_ndvi_val = float("nan")
            peak_conf = "Medium"

        # --- Senescence ---
        sen_win = range(prof["senescence_window"][0], prof["senescence_window"][1] + 1)
        senescence_month = pd.NA
        for m in sen_win:
            v = ndvi.get(m, float("nan"))
            if not np.isnan(v) and v < prof["senescence_threshold"]:
                senescence_month = m
                break

        # --- Season length ---
        if pd.notna(green_up_month) and pd.notna(senescence_month):
            season_length = (int(senescence_month) + 12 - int(green_up_month)) % 12
        else:
            season_length = pd.NA

        rows.append({
            "aoi": aoi, "sensor": sensor, "resolution": resolution,
            "year": year, "land_cover": lc,
            "green_up_month": green_up_month, "green_up_conf": green_up_conf,
            "peak_month": peak_month, "peak_ndvi": peak_ndvi_val,
            "peak_conf": peak_conf, "senescence_month": senescence_month,
            "season_length": season_length,
        })

    if not rows:
        return pd.DataFrame(columns=_empty_cols)

    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype("int32")
    df["resolution"] = df["resolution"].astype("int32")
    df["peak_ndvi"] = df["peak_ndvi"].astype("float64")

    def _to_nullable_int(series):
        return pd.array(
            [pd.NA if (v is pd.NA or v is None or (isinstance(v, float) and np.isnan(v))) else int(v)
             for v in series],
            dtype="Int32",
        )

    for col in ("green_up_month", "peak_month", "senescence_month", "season_length"):
        df[col] = _to_nullable_int(df[col])

    return df.sort_values(["year", "land_cover"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# B-Table 4 — ndvi_annual_delta.parquet
# ---------------------------------------------------------------------------

def compute_annual_delta(
    aoi: str,
    sensor: str,
    resolution: int,
    processed_root: str,
    config: dict,
) -> pd.DataFrame:
    """
    Gain/loss km² for ALL ordered (year_a, year_b) pairs where year_a != year_b.

    Caches per-pixel annual mean rasters in memory before iterating pairs to
    avoid repeated TIF reads (O(N) reads instead of O(N^2)).

    Uses plain subtraction (rast_b - rast_a), NOT the normalized formula used
    for monthly delta maps.
    """
    base = Path(processed_root) / aoi / sensor / f"{resolution}m"
    annual_path = base / "ndvi_annual.parquet"

    _empty_cols = ["aoi", "sensor", "resolution", "year_a", "year_b", "gain_km2", "loss_km2", "total_km2"]

    if not annual_path.exists():
        print(f"  [warn] ndvi_annual.parquet not found -- run preprocess_pass_a.py first")
        return pd.DataFrame(columns=_empty_cols)

    annual = pd.read_parquet(annual_path)
    complete_years = sorted(annual[annual["is_complete"]]["year"].tolist())
    n_complete = len(complete_years)

    if n_complete < 2:
        print(f"  [warn] Fewer than 2 complete years for {aoi}/{sensor}/{resolution}m -- skipping delta")
        return pd.DataFrame(columns=_empty_cols)

    files = _ndvi_files(aoi, sensor, resolution, config)
    if not files:
        return pd.DataFrame(columns=_empty_cols)

    files_by_year: dict[int, list] = {}
    for f in files:
        yr, _ = _parse_ym(f)
        if yr in complete_years:
            files_by_year.setdefault(yr, []).append(f)

    n_pairs = n_complete * (n_complete - 1)
    print(f"  [delta] computing {n_pairs} year pairs...")
    print(f"  [delta] Caching {n_complete} annual means...", end="", flush=True)

    aoi_geom = _load_geometry(config["aois"][aoi]["path"])
    annual_means: dict[int, np.ndarray] = {}
    transform_ref = None
    nrows_ref = ncols_ref = None

    for yr in complete_years:
        yr_files = sorted(files_by_year.get(yr, []))
        if not yr_files:
            continue
        arrays = [_masked_array(f, aoi_geom).astype(np.float64) for f in yr_files]
        annual_means[yr] = np.nanmean(np.stack(arrays, axis=0), axis=0)
        if transform_ref is None:
            with rasterio.open(yr_files[0]) as src:
                transform_ref = src.transform
                nrows_ref, ncols_ref = src.height, src.width

    print(" done")

    if transform_ref is None:
        return pd.DataFrame(columns=_empty_cols)

    row_areas = _geodesic_row_areas_km2(transform_ref, nrows_ref, ncols_ref)
    pixel_areas = np.broadcast_to(row_areas[:, np.newaxis], (nrows_ref, ncols_ref))

    rows = []
    for yr_a in complete_years:
        for yr_b in complete_years:
            if yr_a == yr_b or yr_a not in annual_means or yr_b not in annual_means:
                continue
            delta = annual_means[yr_b] - annual_means[yr_a]
            valid = np.isfinite(annual_means[yr_a]) & np.isfinite(annual_means[yr_b])
            rows.append({
                "aoi": aoi, "sensor": sensor, "resolution": resolution,
                "year_a": yr_a, "year_b": yr_b,
                "gain_km2": float(np.sum(pixel_areas[valid & (delta > 0)])),
                "loss_km2": float(np.sum(pixel_areas[valid & (delta < 0)])),
                "total_km2": float(np.sum(pixel_areas[valid])),
            })

    print(f"  [delta] Written: {len(rows)} rows")

    df = pd.DataFrame(rows)
    df["year_a"] = df["year_a"].astype("int32")
    df["year_b"] = df["year_b"].astype("int32")
    df["resolution"] = df["resolution"].astype("int32")
    return df.sort_values(["year_a", "year_b"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# B-Table 5 — fire_return_period.geojson
# ---------------------------------------------------------------------------

def compute_fire_return_period(
    aoi: str,
    config: dict,
    processed_root: str,
) -> gpd.GeoDataFrame:
    """
    Vectorized fire return period per pixel.

    FRP = n_years / burn_count.
    Rounded to 1 decimal BEFORE vectorizing so adjacent same-value pixels
    merge into fewer polygons, reducing GeoJSON size.
    Equivalent to R: build_ba_frp_leaflet()
    """
    aoi_geom = _load_geometry(config["aois"][aoi]["path"])
    files = _ba_files(aoi, config)

    if not files:
        return gpd.GeoDataFrame()

    files_by_year: dict[int, list] = {}
    for f in files:
        yr, _ = _parse_ym(f)
        files_by_year.setdefault(yr, []).append(f)

    years = sorted(files_by_year.keys())
    n_years = len(years)

    with rasterio.open(files[0]) as src:
        transform = src.transform
        raster_shape = (src.height, src.width)

    # Per year: binary burned mask (1 = burned any month)
    yearly = []
    for yr in years:
        monthly = []
        for f in sorted(files_by_year[yr]):
            with rasterio.open(f) as src:
                out, _ = rio_mask(src, aoi_geom, crop=False, nodata=0, filled=True)
                monthly.append(out[0].astype(np.int32))
        if monthly:
            yr_max = np.max(np.stack(monthly, axis=0), axis=0)
            yearly.append((yr_max > 0).astype(np.float32))
        else:
            yearly.append(np.zeros(raster_shape, dtype=np.float32))

    burn_count = np.sum(np.stack(yearly, axis=0), axis=0).astype(np.int32)

    # FRP: round BEFORE vectorizing to merge adjacent same-value pixels
    frp = np.where(burn_count > 0, n_years / burn_count.astype(np.float64), np.nan)
    frp = np.round(frp, 1).astype(np.float32)

    valid_mask = (~np.isnan(frp)).astype(np.uint8)
    shapes = list(rasterio.features.shapes(frp, mask=valid_mask, transform=transform))

    if not shapes:
        return gpd.GeoDataFrame()

    records = []
    for geom_dict, val in shapes:
        frp_val = round(float(val), 1)
        if frp_val <= 0 or np.isnan(frp_val):
            continue
        records.append({
            "geometry": shapely_shape(geom_dict),
            "frp_years": frp_val,
            "burn_count": int(round(n_years / frp_val)),
            "n_years": n_years,
        })

    if not records:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["frp_years"] = gdf["frp_years"].astype("float64")
    gdf["burn_count"] = gdf["burn_count"].astype("int32")
    gdf["n_years"] = gdf["n_years"].astype("int32")
    return gdf
