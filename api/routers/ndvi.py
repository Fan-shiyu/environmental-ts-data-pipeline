"""NDVI endpoints: timeseries, by-landcover, anomaly, phenology."""

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
    resolve_resolution,
    tif_folder,
    ym_filter,
    _max_ym,
)
from api.schemas import APIResponse

router = APIRouter(prefix="/ndvi", tags=["ndvi"])


def _error(aoi, sensor, resolution, msg) -> APIResponse:
    return APIResponse(
        data=[],
        metadata={"aoi": aoi, "sensor": sensor, "resolution": resolution, "n_records": 0},
        status="error",
        error=msg,
    )


def _meta(aoi, sensor, resolution, n, data_through=None, extra=None) -> dict:
    m = {
        "aoi": aoi, "sensor": sensor, "resolution": resolution,
        "data_through": data_through, "n_records": n,
    }
    if extra:
        m.update(extra)
    return m


# ---------------------------------------------------------------------------
# /ndvi/timeseries
# ---------------------------------------------------------------------------

@router.get("/timeseries", response_model=APIResponse)
def ndvi_timeseries(
    aoi: str,
    sensor: str,
    resolution: str = "auto",
    start: str | None = None,
    end: str | None = None,
    format: str = Query("table", pattern="^(table|agent)$"),
) -> APIResponse:
    res = resolve_resolution(sensor, resolution, start, end)

    # Cache the raw tables (format-agnostic); apply filter/format after.
    cache_key = response_cache.make_key(
        "/ndvi/timeseries",
        {"aoi": aoi, "sensor": sensor, "resolution": res, "start": start, "end": end},
    )
    cached = response_cache.get(cache_key)
    if cached is None:
        monthly_raw = read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_monthly"))
        if monthly_raw is None or monthly_raw.empty:
            return _error(aoi, sensor, res, f"ndvi_monthly not available for {aoi}/{sensor}/{res}m")
        cached = {
            "monthly": monthly_raw,
            "baselines": read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_monthly_baselines")),
            "trend": read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_trend_stats")),
        }
        response_cache.set(cache_key, cached, ttl_seconds=TTL_DATA)

    monthly = ym_filter(cached["monthly"], start, end).sort_values(["year", "month"])
    baselines = cached["baselines"]
    trend = cached["trend"]

    # Join per-calendar-month baseline columns onto each monthly row.
    if baselines is not None and not baselines.empty:
        bcols = ["month", "ts_lower", "ts_upper", "climatology", "hist_std"]
        merged = monthly.merge(
            baselines[[c for c in bcols if c in baselines.columns]], on="month", how="left"
        )
    else:
        merged = monthly.copy()
        for c in ("ts_lower", "ts_upper", "climatology", "hist_std"):
            merged[c] = None

    dt = _max_ym(monthly)

    if format == "table":
        cols = ["year", "month", "mean_ndvi", "n_valid_px", "ts_lower", "ts_upper", "climatology"]
        return APIResponse(data=df_to_records(merged, cols),
                           metadata=_meta(aoi, sensor, res, len(merged), dt))

    # agent format — enrich with climatology comparison + trend
    trend_row = trend.iloc[0] if (trend is not None and not trend.empty) else None
    trend_direction = str(trend_row["trend_direction"]) if trend_row is not None else None
    mk_p = float(trend_row["mk_p"]) if (trend_row is not None and trend_row["mk_p"] == trend_row["mk_p"]) else None
    sen_slope = float(trend_row["sen_slope"]) if (trend_row is not None and trend_row["sen_slope"] == trend_row["sen_slope"]) else None
    trend_significant = (mk_p is not None and mk_p < 0.05)

    records = []
    for r in df_to_records(merged):
        mean = r.get("mean_ndvi")
        clim = r.get("climatology")
        lo, hi = r.get("ts_lower"), r.get("ts_upper")
        vs_clim = round(mean - clim, 4) if (mean is not None and clim is not None) else None
        if mean is None or lo is None or hi is None:
            status = "unknown"
        elif mean > hi:
            status = "above"
        elif mean < lo:
            status = "below"
        else:
            status = "normal"
        records.append({
            "year": r["year"], "month": r["month"], "mean_ndvi": mean,
            "vs_climatology": vs_clim, "status": status,
            "trend_direction": trend_direction,
            "trend_significant": trend_significant,
            "sen_slope_per_year": sen_slope,
        })
    return APIResponse(
        data=records,
        metadata=_meta(aoi, sensor, res, len(records), dt,
                       extra={"trend_direction": trend_direction,
                              "trend_significant": trend_significant}),
    )


# ---------------------------------------------------------------------------
# /ndvi/by-landcover
# ---------------------------------------------------------------------------

@router.get("/by-landcover", response_model=APIResponse)
def ndvi_by_landcover(
    aoi: str,
    sensor: str,
    year: int,
    resolution: str = "auto",
    format: str = Query("table", pattern="^(table|agent)$"),
) -> APIResponse:
    res = resolve_resolution(sensor, resolution, None, None)

    cache_key = response_cache.make_key(
        "/ndvi/by-landcover",
        {"aoi": aoi, "sensor": sensor, "resolution": res, "year": year},
    )
    cached = response_cache.get(cache_key)
    if cached is None:
        monthly_raw = read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_monthly_by_class"))
        if monthly_raw is None or monthly_raw.empty:
            return _error(aoi, sensor, res, f"No land cover data for {aoi}")
        cached = {
            "monthly": monthly_raw,
            "annual": read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_annual_by_class")),
        }
        response_cache.set(cache_key, cached, ttl_seconds=TTL_DATA)

    monthly = cached["monthly"][cached["monthly"]["year"] == year].sort_values(["land_cover", "month"])
    if monthly.empty:
        return _error(aoi, sensor, res, f"No by-landcover data for {aoi} in {year}")

    annual = cached["annual"]

    if format == "table":
        cols = ["year", "month", "land_cover", "mean_ndvi", "n_valid_px"]
        return APIResponse(data=df_to_records(monthly, cols),
                           metadata=_meta(aoi, sensor, res, len(monthly), extra={"year": year}))

    # agent — merge per-class historical range from annual_by_class
    if annual is not None and not annual.empty:
        ann_year = annual[annual["year"] == year]
        hist_cols = ["land_cover", "annual_mean", "hist_min", "hist_max", "hist_mean_all", "cv"]
        merged = monthly.merge(
            ann_year[[c for c in hist_cols if c in ann_year.columns]],
            on="land_cover", how="left",
        )
    else:
        merged = monthly
    return APIResponse(data=df_to_records(merged),
                       metadata=_meta(aoi, sensor, res, len(merged), extra={"year": year}))


# ---------------------------------------------------------------------------
# /ndvi/anomaly
# ---------------------------------------------------------------------------

@router.get("/anomaly", response_model=APIResponse)
def ndvi_anomaly(
    aoi: str,
    sensor: str,
    year: int,
    resolution: str = "auto",
    format: str = Query("table", pattern="^(table|agent)$"),
) -> APIResponse:
    res = resolve_resolution(sensor, resolution, None, None)

    cache_key = response_cache.make_key(
        "/ndvi/anomaly",
        {"aoi": aoi, "sensor": sensor, "resolution": res, "year": year},
    )
    cached = response_cache.get(cache_key)
    if cached is None:
        monthly_raw = read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_anomaly_monthly"))
        if monthly_raw is None or monthly_raw.empty:
            return _error(aoi, sensor, res, f"No anomaly data for {aoi}")
        cached = {
            "monthly": monthly_raw,
            "resilience": read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_anomaly_resilience")),
        }
        response_cache.set(cache_key, cached, ttl_seconds=TTL_DATA)

    monthly = cached["monthly"][cached["monthly"]["year"] == year].sort_values(["land_cover", "month"])
    if monthly.empty:
        return _error(aoi, sensor, res, f"No anomaly data for {aoi} in {year}")

    resilience = cached["resilience"]
    resilience_records = []
    if resilience is not None and not resilience.empty:
        ry = resilience[resilience["anomaly_year"] == year]
        resilience_records = df_to_records(ry.sort_values("resilience_rank")) if not ry.empty else []

    extra = {"year": year, "resilience": resilience_records}

    if format == "table":
        cols = ["year", "month", "land_cover", "anomaly_value", "mean_ndvi", "hist_mean"]
        return APIResponse(data=df_to_records(monthly, cols),
                           metadata=_meta(aoi, sensor, res, len(monthly), extra=extra))

    # agent — flag each month's direction
    records = []
    for r in df_to_records(monthly):
        av = r.get("anomaly_value")
        r["status"] = "deficit" if (av is not None and av < 0) else "surplus"
        records.append(r)
    return APIResponse(data=records,
                       metadata=_meta(aoi, sensor, res, len(records), extra=extra))


# ---------------------------------------------------------------------------
# /ndvi/phenology
# ---------------------------------------------------------------------------

@router.get("/phenology", response_model=APIResponse)
def ndvi_phenology(
    aoi: str,
    sensor: str,
    land_cover: str = Query(..., pattern="^(Crops|Rangeland)$"),
    resolution: str = "auto",
    format: str = Query("table", pattern="^(table|agent)$"),
) -> APIResponse:
    res = resolve_resolution(sensor, resolution, None, None)

    cache_key = response_cache.make_key(
        "/ndvi/phenology", {"aoi": aoi, "sensor": sensor, "resolution": res},
    )
    cached = response_cache.get(cache_key)
    if cached is None:
        pheno_raw = read_parquet_safe(get_parquet_path(aoi, sensor, res, "ndvi_phenology"))
        if pheno_raw is None or pheno_raw.empty:
            return _error(aoi, sensor, res, f"No phenology data for {aoi}")
        cached = {"pheno": pheno_raw}
        response_cache.set(cache_key, cached, ttl_seconds=TTL_DATA)

    pheno = cached["pheno"]
    sub = pheno[pheno["land_cover"] == land_cover].sort_values("year")
    if sub.empty:
        return _error(aoi, sensor, res, f"No phenology data for {aoi} / {land_cover}")

    return APIResponse(
        data=df_to_records(sub),
        metadata=_meta(aoi, sensor, res, len(sub), extra={"land_cover": land_cover}),
    )


# ---------------------------------------------------------------------------
# Per-pixel grid endpoints (compact 2D grid for the Shiny Delta Map hot path)
# ---------------------------------------------------------------------------

def _read_ndvi_tif(path):
    """Read a single NDVI GeoTIFF -> (float array with NaN nodata, transform, crs, shape)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.transform, src.crs, arr.shape


@router.get("/annual-grid")
def ndvi_annual_grid(aoi: str, sensor: str, year: int, resolution: str = "auto") -> dict:
    res = resolve_resolution(sensor, resolution, None, None)
    cache_key = response_cache.make_key(
        "/ndvi/annual-grid",
        {"aoi": aoi, "sensor": sensor, "resolution": res, "year": year},
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        return cached

    folder = tif_folder(aoi, sensor, res)
    files = sorted(folder.glob(f"{year:04d}-??_NDVI_{aoi}.tif"))
    if not files:
        return {"status": "error", "error": f"No TIF files for {aoi}/{sensor}/{res}m/{year}"}

    arrays, transform, crs, shape = [], None, None, None
    for f in files:
        arr, t, c, s = _read_ndvi_tif(f)
        arrays.append(arr)
        if transform is None:
            transform, crs, shape = t, c, s

    annual_mean = np.nanmean(np.stack(arrays, axis=0), axis=0)
    annual_mean = apply_aoi_mask(annual_mean, aoi, transform, shape)
    result = build_grid_response(
        annual_mean, transform, shape, aoi, sensor, res, crs,
        extra_meta={"year": year, "months_available": len(files)},
    )
    response_cache.set(cache_key, result, ttl_seconds=TTL_GRID_ANNUAL)
    return result


@router.get("/monthly-grid")
def ndvi_monthly_grid(aoi: str, sensor: str, year: int, month: int,
                      resolution: str = "auto") -> dict:
    res = resolve_resolution(sensor, resolution, None, None)
    cache_key = response_cache.make_key(
        "/ndvi/monthly-grid",
        {"aoi": aoi, "sensor": sensor, "resolution": res, "year": year, "month": month},
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        return cached

    tif = tif_folder(aoi, sensor, res) / f"{year:04d}-{month:02d}_NDVI_{aoi}.tif"
    if not tif.exists():
        return {"status": "error",
                "error": f"No TIF for {aoi}/{sensor}/{res}m/{year}-{month:02d}"}

    arr, transform, crs, shape = _read_ndvi_tif(tif)
    arr = apply_aoi_mask(arr, aoi, transform, shape)
    result = build_grid_response(
        arr, transform, shape, aoi, sensor, res, crs,
        extra_meta={"year": year, "month": month},
    )
    response_cache.set(cache_key, result, ttl_seconds=TTL_GRID_MONTHLY)
    return result
