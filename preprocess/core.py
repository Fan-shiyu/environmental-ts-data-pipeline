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
from pyproj import Geod
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

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
