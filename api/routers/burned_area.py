"""Burned area endpoints: monthly summary, daily activity. Resolution fixed at 500m."""

import numpy as np
import rasterio
from fastapi import APIRouter, Query

from api.cache import response_cache
from api.dependencies import (
    TTL_DATA,
    TTL_GRID_ANNUAL,
    TTL_GRID_MONTHLY,
    apply_aoi_mask,
    build_grid_response,
    df_to_records,
    get_parquet_path,
    read_parquet_safe,
    tif_folder,
    ym_filter,
    _max_ym,
)
from api.schemas import APIResponse

router = APIRouter(prefix="/burned-area", tags=["burned_area"])

_RES = 500


def _error(aoi, msg) -> APIResponse:
    return APIResponse(
        data=[],
        metadata={"aoi": aoi, "resolution": _RES, "n_records": 0},
        status="error",
        error=msg,
    )


@router.get("/summary", response_model=APIResponse)
def burned_area_summary(
    aoi: str,
    start: str | None = None,
    end: str | None = None,
    format: str = Query("table", pattern="^(table|agent)$"),
) -> APIResponse:
    cache_key = response_cache.make_key(
        "/burned-area/summary", {"aoi": aoi, "start": start, "end": end},
    )
    cached = response_cache.get(cache_key)
    if cached is None:
        df_raw = read_parquet_safe(get_parquet_path(aoi, "burned_area", _RES, "ba_monthly"))
        if df_raw is None or df_raw.empty:
            return _error(aoi, f"No burned area data for {aoi}")
        cached = {"df": df_raw}
        response_cache.set(cache_key, cached, ttl_seconds=TTL_DATA)

    df = ym_filter(cached["df"], start, end).sort_values(["year", "month"])
    dt = _max_ym(df)
    meta = {"aoi": aoi, "resolution": _RES, "data_through": dt, "n_records": len(df)}

    if format == "table":
        cols = ["year", "month", "burned_km2", "ba_mean", "ba_lower_ci", "ba_upper_ci", "n_years"]
        return APIResponse(data=df_to_records(df, cols), metadata=meta)

    # agent — flag months above the historical upper CI
    records = []
    for r in df_to_records(df):
        burned, hi = r.get("burned_km2"), r.get("ba_upper_ci")
        mean = r.get("ba_mean")
        r["vs_baseline"] = round(burned - mean, 3) if (burned is not None and mean is not None) else None
        r["status"] = "above_normal" if (burned is not None and hi is not None and burned > hi) else "normal"
        records.append(r)
    return APIResponse(data=records, metadata=meta)


@router.get("/daily", response_model=APIResponse)
def burned_area_daily(
    aoi: str,
    year: int,
    format: str = Query("table", pattern="^(table|agent)$"),
) -> APIResponse:
    cache_key = response_cache.make_key("/burned-area/daily", {"aoi": aoi})
    cached = response_cache.get(cache_key)
    if cached is None:
        df_raw = read_parquet_safe(get_parquet_path(aoi, "burned_area", _RES, "ba_daily"))
        if df_raw is None or df_raw.empty:
            return _error(aoi, f"No daily burned area data for {aoi}")
        cached = {"df": df_raw}
        response_cache.set(cache_key, cached, ttl_seconds=TTL_DATA)

    df = cached["df"][cached["df"]["year"] == year].sort_values("date")
    if df.empty:
        return _error(aoi, f"No daily burned area for {aoi} in {year}")

    meta = {"aoi": aoi, "resolution": _RES, "year": year, "n_records": len(df)}
    cols = ["year", "date", "burned_km2"]
    return APIResponse(data=df_to_records(df, cols), metadata=meta)


# ---------------------------------------------------------------------------
# Per-pixel grid endpoints (BurnDate; compact 2D grid). Resolution always 500m.
# ---------------------------------------------------------------------------

_PIXEL_AREA_KM2 = (_RES / 1000) ** 2  # 0.25 km² per 500m pixel


def _read_ba_tif(path):
    """Read a BurnDate GeoTIFF -> (float array with NaN nodata, transform, crs, shape)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.transform, src.crs, arr.shape


@router.get("/monthly-grid")
def burned_area_monthly_grid(aoi: str, year: int, month: int) -> dict:
    cache_key = response_cache.make_key(
        "/burned-area/monthly-grid", {"aoi": aoi, "year": year, "month": month},
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        return cached

    matches = list(tif_folder(aoi, "burned_area", _RES).glob(f"{year:04d}-{month:02d}_*{aoi}*.tif"))
    if not matches:
        return {"status": "error", "error": f"No burned area TIF for {aoi}/{year}-{month:02d}"}

    arr, transform, crs, shape = _read_ba_tif(matches[0])
    arr = apply_aoi_mask(arr, aoi, transform, shape)

    burned_pixels = int(np.sum((arr > 0) & (~np.isnan(arr))))
    burned_km2 = round(burned_pixels * _PIXEL_AREA_KM2, 2)

    result = build_grid_response(
        arr, transform, shape, aoi, "burned_area", _RES, crs,
        extra_meta={"year": year, "month": month,
                    "burned_pixels": burned_pixels, "burned_km2": burned_km2},
    )
    response_cache.set(cache_key, result, ttl_seconds=TTL_GRID_MONTHLY)
    return result


@router.get("/annual-grid")
def burned_area_annual_grid(aoi: str, year: int) -> dict:
    cache_key = response_cache.make_key("/burned-area/annual-grid", {"aoi": aoi, "year": year})
    cached = response_cache.get(cache_key)
    if cached is not None:
        return cached

    files = sorted(tif_folder(aoi, "burned_area", _RES).glob(f"{year:04d}-??_*{aoi}*.tif"))
    if not files:
        return {"status": "error", "error": f"No burned area TIFs for {aoi}/{year}"}

    arrays, transform, crs, shape = [], None, None, None
    for f in files:
        arr, t, c, s = _read_ba_tif(f)
        # binary burned flag per month, NaN-preserving
        burned = np.where(np.isnan(arr), np.nan, np.where(arr > 0, 1.0, 0.0))
        arrays.append(burned)
        if transform is None:
            transform, crs, shape = t, c, s

    stack = np.stack(arrays, axis=0)
    burn_count = np.nansum(stack, axis=0).astype(float)
    burn_count[np.all(np.isnan(stack), axis=0)] = np.nan  # never-valid pixels -> NaN
    burn_count = apply_aoi_mask(burn_count, aoi, transform, shape)

    burned_pixels = int(np.nansum(burn_count > 0))
    multi_burn_pixels = int(np.nansum(burn_count > 1))
    total_burned_km2 = round(burned_pixels * _PIXEL_AREA_KM2, 2)

    result = build_grid_response(
        burn_count, transform, shape, aoi, "burned_area", _RES, crs,
        extra_meta={"year": year, "months_available": len(files),
                    "burned_pixels": burned_pixels,
                    "total_burned_km2": total_burned_km2,
                    "multi_burn_pixels": multi_burn_pixels},
    )
    response_cache.set(cache_key, result, ttl_seconds=TTL_GRID_ANNUAL)
    return result
