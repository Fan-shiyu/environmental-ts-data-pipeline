"""Shared utilities: path resolution, Parquet reading, auto-resolution, date scans.

No caching by design — Parquet files are small and reads are fast, and the
pipeline rewrites them, so reading fresh on each request avoids stale data.
"""

import csv
import datetime
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask

from pipeline.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = REPO_ROOT / "outputs" / "processed"
UPDATE_LOG_PATH = REPO_ROOT / "outputs" / "update_log.csv"

# Cache TTLs (seconds)
TTL_DATA = 3600       # ndvi/* + burned-area/* — stable until next monthly run
TTL_GEOMETRY = 86400  # geometry/* — changes at most annually

# Metric CRS used for distance-based geometry simplification (UTM zone 35S).
SIMPLIFY_CRS = "EPSG:32735"

AOIS = ["Zambia_Mponda", "Zambia_WL"]

# (sensor, resolution_metres) available combos, from config/sites.yaml
SENSOR_RESOLUTIONS = {
    "sentinel2": [100, 1000],
    "modis": [250, 500, 1000],
    "burned_area": [500],
}


# ---------------------------------------------------------------------------
# Path resolution + safe reads
# ---------------------------------------------------------------------------

def get_parquet_path(aoi: str, sensor: str, resolution: int, table: str) -> Path:
    """outputs/processed/{aoi}/{sensor}/{resolution}m/{table}.parquet (repo-relative)."""
    return PROCESSED_ROOT / aoi / sensor / f"{resolution}m" / f"{table}.parquet"


def read_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Read a Parquet file, or None if it doesn't exist.

    If the file was modified within the last 30s (pipeline may be writing),
    wait 2s and retry once.
    """
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age < 30:
            time.sleep(2)
        return pd.read_parquet(path)
    except Exception:
        # One retry on transient read error (e.g. mid-write)
        try:
            time.sleep(2)
            return pd.read_parquet(path)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# resolution=auto
# ---------------------------------------------------------------------------

def _span_years(start: str | None, end: str | None) -> float | None:
    """Year span between two YYYY-MM strings, or None if either is missing."""
    if not start or not end:
        return None
    try:
        sy, sm = int(start[:4]), int(start[5:7])
        ey, em = int(end[:4]), int(end[5:7])
        return (ey + em / 12) - (sy + sm / 12)
    except (ValueError, IndexError):
        return None


def resolve_resolution(
    sensor: str, resolution: int | str, start: str | None = None, end: str | None = None
) -> int:
    """Pick the best resolution when 'auto'/0; otherwise return the int as-is.

    - sentinel2  -> 100  (finest available)
    - burned_area-> 500  (only option)
    - modis      -> 1000 if date span > 5 years (long-term trends) else 250
    """
    if isinstance(resolution, str) and resolution.lower() != "auto":
        try:
            resolution = int(resolution)
        except ValueError:
            resolution = 0

    is_auto = (isinstance(resolution, str) and resolution.lower() == "auto") or resolution == 0
    if not is_auto:
        return int(resolution)

    if sensor == "sentinel2":
        return 100
    if sensor == "burned_area":
        return 500
    if sensor == "modis":
        span = _span_years(start, end)
        if span is not None and span > 5:
            return 1000
        return 250 if span is not None else 1000
    return int(resolution) if str(resolution).isdigit() else 1000


# ---------------------------------------------------------------------------
# Date helpers / scans
# ---------------------------------------------------------------------------

def _max_ym(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "year" not in df or "month" not in df:
        return None
    sub = df.dropna(subset=["year", "month"])
    if sub.empty:
        return None
    y, m = sub.sort_values(["year", "month"]).iloc[-1][["year", "month"]]
    return f"{int(y):04d}-{int(m):02d}"


def _min_ym(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "year" not in df or "month" not in df:
        return None
    sub = df.dropna(subset=["year", "month"])
    if sub.empty:
        return None
    y, m = sub.sort_values(["year", "month"]).iloc[0][["year", "month"]]
    return f"{int(y):04d}-{int(m):02d}"


def data_through(aoi: str = "Zambia_Mponda") -> str | None:
    """Latest YYYY-MM present in this AoI's NDVI (fallback burned area)."""
    for sensor, res in (("sentinel2", 100), ("modis", 1000)):
        df = read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_monthly"))
        ym = _max_ym(df)
        if ym:
            return ym
    df = read_parquet_safe(get_parquet_path(aoi, "burned_area", 500, "ba_monthly"))
    return _max_ym(df)


def last_pipeline_run() -> str | None:
    """Most recent run_timestamp from outputs/update_log.csv, or None."""
    if not UPDATE_LOG_PATH.exists():
        return None
    last = None
    try:
        with open(UPDATE_LOG_PATH, newline="") as f:
            for row in csv.DictReader(f):
                ts = row.get("run_timestamp")
                if ts:
                    last = ts
    except Exception:
        return None
    return last


def count_parquet_files() -> int:
    if not PROCESSED_ROOT.exists():
        return 0
    return sum(1 for _ in PROCESSED_ROOT.rglob("*.parquet"))


def next_scheduled_run() -> str:
    """Next monthly run: 5th of the month at 06:13 UTC (matches the cron)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    candidate = now.replace(day=5, hour=6, minute=13, second=0, microsecond=0)
    if candidate <= now:
        # roll to the 5th of next month
        year = now.year + (1 if now.month == 12 else 0)
        month = 1 if now.month == 12 else now.month + 1
        candidate = candidate.replace(year=year, month=month)
    return candidate.strftime("%Y-%m-%dT%H:%M:%SZ")


def ym_filter(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Filter a year/month DataFrame to rows within [start, end] (YYYY-MM)."""
    if df is None or df.empty:
        return df
    out = df
    if start:
        sy, sm = int(start[:4]), int(start[5:7])
        out = out[(out["year"] > sy) | ((out["year"] == sy) & (out["month"] >= sm))]
    if end:
        ey, em = int(end[:4]), int(end[5:7])
        out = out[(out["year"] < ey) | ((out["year"] == ey) & (out["month"] <= em))]
    return out


# ---------------------------------------------------------------------------
# JSON-safe record conversion (handles NaN, pd.NA, numpy types, dates)
# ---------------------------------------------------------------------------

def _clean_value(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
        return v.strftime("%Y-%m-%d")
    return v


def df_to_records(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts, optionally restricted to columns."""
    if df is None or df.empty:
        return []
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    return [{k: _clean_value(v) for k, v in row.items()}
            for row in df.to_dict(orient="records")]


# ---------------------------------------------------------------------------
# Geometry loading + optional simplification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-pixel grid helpers (shared by ndvi + burned_area grid endpoints)
# ---------------------------------------------------------------------------

# Raw monthly TIFs live under the pipeline output root (NOT outputs/processed/).
TIF_OUTPUT_ROOT = REPO_ROOT / "outputs"
TTL_GRID_ANNUAL = 7200   # 2 h
TTL_GRID_MONTHLY = 3600  # 1 h


def tif_folder(aoi: str, sensor: str, resolution: int) -> Path:
    """Folder holding the raw monthly GeoTIFFs: outputs/{aoi}/{sensor}/{res}m."""
    return TIF_OUTPUT_ROOT / aoi / sensor / f"{resolution}m"


def apply_aoi_mask(arr, aoi: str, transform, shape):
    """Set pixels outside the AoI polygon to NaN. Returns arr unchanged if the
    AoI GeoJSON is missing."""
    aoi_path = REPO_ROOT / "config" / "aoi" / f"AoI_{aoi}.geojson"
    if not aoi_path.exists():
        return arr
    gdf = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    shapes = [g.__geo_interface__ for g in gdf.geometry]
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=shape[0], width=shape[1], count=1,
                     dtype="float64", crs="EPSG:4326", transform=transform) as ds:
            ds.write(arr.astype("float64"), 1)
        with mf.open() as ds:
            masked, _ = rio_mask(ds, shapes, crop=False, nodata=np.nan, filled=True)
    return masked[0]


def build_grid_response(arr, transform, shape, aoi, sensor, resolution, crs,
                        extra_meta: dict | None = None) -> dict:
    """Compact 2D grid: 1D lats + 1D lons + 2D values matrix (NaN -> None).
    Coords rounded to 6 dp, values to 4 dp."""
    rows, cols = shape
    lons = [round(transform.c + (j + 0.5) * transform.a, 6) for j in range(cols)]
    lats = [round(transform.f + (i + 0.5) * transform.e, 6) for i in range(rows)]
    values = [
        [round(float(v), 4) if not np.isnan(v) else None for v in row]
        for row in arr
    ]
    metadata = {
        "aoi": aoi, "sensor": sensor, "resolution": resolution,
        "n_pixels": rows * cols,
        "n_valid_pixels": int(np.sum(~np.isnan(arr))),
        "crs": str(crs),
    }
    if extra_meta:
        metadata.update(extra_meta)
    return {
        "grid": {"lats": lats, "lons": lons, "values": values},
        "metadata": metadata,
        "status": "ok",
    }


def load_geojson_4326(path, simplified: bool = True, tolerance_m: int = 50):
    """Read a GeoJSON and return it in EPSG:4326.

    When simplified, run Douglas-Peucker at `tolerance_m` metres in a metric CRS
    BEFORE reprojecting (a degree tolerance is meaningless for these layers).
    Land cover files are already EPSG:32735 (metres); FRP is EPSG:4326 and is
    reprojected to metres for simplification, then back.
    """
    gdf = gpd.read_file(path)
    if simplified:
        metric = gdf.to_crs(SIMPLIFY_CRS) if (gdf.crs and gdf.crs.is_geographic) else gdf
        metric = metric.copy()
        metric["geometry"] = metric.geometry.simplify(tolerance_m, preserve_topology=True)
        gdf = metric
    return gdf.to_crs("EPSG:4326")
