"""Burned area endpoints: monthly summary, daily activity. Resolution fixed at 500m."""

from fastapi import APIRouter, Query

from api.cache import response_cache
from api.dependencies import (
    TTL_DATA,
    df_to_records,
    get_parquet_path,
    read_parquet_safe,
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
