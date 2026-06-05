"""Geometry endpoints: AoI polygon, land cover polygons, fire return period.

These return GeoJSON (FeatureCollection), not the APIResponse envelope, so the
Shiny app / agent can render them directly on a map. Responses are cached
(24 h) since geometry changes at most annually. The `simplified` parameter
(landcover + FRP only) Douglas-Peucker-simplifies at 50 m in metric space to
shrink the large raw pixel-boundary payloads.
"""

import json

import geopandas as gpd
from fastapi import APIRouter, HTTPException
from pyproj import Geod

from api.cache import response_cache
from api.dependencies import (
    REPO_ROOT,
    TTL_DATA,
    TTL_GEOMETRY,
    get_parquet_path,
    load_geojson_4326,
    read_parquet_safe,
)

router = APIRouter(prefix="/geometry", tags=["geometry"])

_GEOD = Geod(ellps="WGS84")
LANDCOVER_CLASSES = [
    "Bare_ground", "Built_Area", "Crops",
    "Flooded_vegetation", "Rangeland", "Trees", "Water",
]


def _geod_area_km2(geom) -> float:
    """Geodesic area of a shapely geometry (lon/lat) in km²."""
    area_m2, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(area_m2) / 1e6


def _aoi_geojson_path(aoi: str):
    return REPO_ROOT / "config" / "aoi" / f"AoI_{aoi}.geojson"


def _aoi_total_area_km2(aoi: str) -> float | None:
    path = _aoi_geojson_path(aoi)
    if not path.exists():
        return None
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    return float(sum(_geod_area_km2(g) for g in gdf.geometry))


@router.get("/aoi")
def geometry_aoi(aoi: str) -> dict:
    cache_key = response_cache.make_key("/geometry/aoi", {"aoi": aoi})
    cached = response_cache.get(cache_key)
    if cached is not None:
        return cached

    path = _aoi_geojson_path(aoi)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"AoI geometry not found for {aoi}")

    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf["area_km2"] = [round(_geod_area_km2(g), 4) for g in gdf.geometry]
    fc = json.loads(gdf.to_json())
    fc["area_km2"] = round(float(gdf["area_km2"].sum()), 4)
    fc["aoi"] = aoi

    response_cache.set(cache_key, fc, ttl_seconds=TTL_GEOMETRY)
    return fc


@router.get("/landcover")
def geometry_landcover(
    aoi: str,
    land_cover_class: str | None = None,
    simplified: bool = True,
) -> dict:
    cache_key = response_cache.make_key(
        "/geometry/landcover",
        {"aoi": aoi, "land_cover_class": land_cover_class, "simplified": simplified},
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        return cached

    lc_dir = REPO_ROOT / "config" / "landcover" / aoi
    if not lc_dir.exists():
        raise HTTPException(status_code=404, detail=f"No land cover geometry for {aoi}")

    classes = [land_cover_class] if land_cover_class else LANDCOVER_CLASSES
    aoi_area = _aoi_total_area_km2(aoi)

    features: list[dict] = []
    for cls in classes:
        path = lc_dir / f"{aoi}_{cls}_2023.geojson"
        if not path.exists():
            if land_cover_class:  # explicit class requested but missing
                raise HTTPException(status_code=404, detail=f"No '{cls}' geometry for {aoi}")
            continue
        gdf = load_geojson_4326(str(path), simplified=simplified)
        for geom in gdf.geometry:
            area_km2 = _geod_area_km2(geom)
            features.append({
                "type": "Feature",
                "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
                "properties": {
                    "land_cover": cls,
                    "area_ha": round(area_km2 * 100, 4),
                    "pct_of_study_area": round(area_km2 / aoi_area * 100, 4) if aoi_area else None,
                },
            })

    if not features:
        raise HTTPException(status_code=404, detail=f"No land cover geometry for {aoi}")

    fc = {"type": "FeatureCollection", "aoi": aoi, "simplified": simplified, "features": features}
    response_cache.set(cache_key, fc, ttl_seconds=TTL_GEOMETRY)
    return fc


@router.get("/fire-return-period")
def geometry_fire_return_period(
    aoi: str,
    resolution: int = 500,
    simplified: bool = False,  # FRP polygons are fragmented 500m cells; 50m
                               # simplification gives no size benefit, so default off.
) -> dict:
    cache_key = response_cache.make_key(
        "/geometry/fire-return-period",
        {"aoi": aoi, "resolution": resolution, "simplified": simplified},
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        return cached

    path = (REPO_ROOT / "outputs" / "processed" / aoi / "burned_area"
            / f"{resolution}m" / "fire_return_period.geojson")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No fire return period data for {aoi}")

    if simplified:
        # FRP is EPSG:4326 -> helper reprojects to metres, simplifies 50 m, back to 4326.
        gdf = load_geojson_4326(str(path), simplified=True)
        fc = json.loads(gdf.to_json())
    else:
        with open(path) as f:
            fc = json.load(f)

    # Additive: year-range metadata derived from ba_monthly (reuses the cache).
    fc["metadata"] = _ba_year_metadata(aoi)

    response_cache.set(cache_key, fc, ttl_seconds=TTL_GEOMETRY)
    return fc


def _ba_year_metadata(aoi: str) -> dict:
    """Year range for an AoI from ba_monthly.parquet. Cached raw DataFrame."""
    ba_key = response_cache.make_key("ba_monthly_raw", {"aoi": aoi})
    ba_df = response_cache.get(ba_key)
    if ba_df is None:
        ba_df = read_parquet_safe(get_parquet_path(aoi, "burned_area", 500, "ba_monthly"))
        if ba_df is not None and not ba_df.empty:
            response_cache.set(ba_key, ba_df, ttl_seconds=TTL_DATA)

    if ba_df is None or ba_df.empty:
        return {"year_start": None, "year_end": None, "n_years": None, "aoi": aoi}

    year_start = int(ba_df["year"].min())
    year_end = int(ba_df["year"].max())
    return {
        "year_start": year_start,
        "year_end": year_end,
        "n_years": year_end - year_start + 1,
        "aoi": aoi,
    }
