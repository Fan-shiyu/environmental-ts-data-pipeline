"""Health and discovery endpoints."""

from fastapi import APIRouter

from api.dependencies import (
    AOIS,
    SENSOR_RESOLUTIONS,
    count_parquet_files,
    data_through,
    get_parquet_path,
    last_pipeline_run,
    next_scheduled_run,
    read_parquet_safe,
    _max_ym,
    _min_ym,
)

router = APIRouter(tags=["health"])

API_VERSION = "1.0.0"


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "data_through": data_through(),
        "last_pipeline_run": last_pipeline_run(),
        "next_scheduled_run": next_scheduled_run(),
        "parquet_files_present": count_parquet_files(),
        "api_version": API_VERSION,
    }


def _sensor_date_range(sensor: str) -> tuple[str | None, str | None]:
    """Min start / max end across all AoIs for a sensor, scanned from Parquet."""
    table = "ba_monthly" if sensor == "burned_area" else "ndvi_monthly"
    res = SENSOR_RESOLUTIONS[sensor][0]
    starts, ends = [], []
    for aoi in AOIS:
        df = read_parquet_safe(get_parquet_path(aoi, sensor, res, table))
        lo, hi = _min_ym(df), _max_ym(df)
        if lo:
            starts.append(lo)
        if hi:
            ends.append(hi)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _land_cover_classes() -> dict:
    """Distinct land cover classes per AoI (None when no by-class data)."""
    out: dict[str, list[str] | None] = {}
    for aoi in AOIS:
        df = read_parquet_safe(
            get_parquet_path(aoi, "sentinel2", 100, "ndvi_monthly_by_class")
        )
        if df is None or df.empty or "land_cover" not in df:
            out[aoi] = None
        else:
            out[aoi] = sorted(df["land_cover"].dropna().unique().tolist())
    return out


@router.get("/available-data")
def available_data() -> dict:
    sensors = {}
    for sensor, resolutions in SENSOR_RESOLUTIONS.items():
        start, end = _sensor_date_range(sensor)
        sensors[sensor] = {"start": start, "end": end, "resolutions": resolutions}

    return {
        "aois": AOIS,
        "sensors": sensors,
        "land_cover_classes": _land_cover_classes(),
        "last_pipeline_run": last_pipeline_run(),
    }
