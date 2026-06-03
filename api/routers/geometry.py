"""Geometry endpoints: AoI polygon, land cover polygons, fire return period.

These return GeoJSON (FeatureCollection), not the APIResponse envelope, so the
Shiny app / agent can render them directly on a map.
"""

import json

import geopandas as gpd
from fastapi import APIRouter, HTTPException
from pyproj import Geod

from api.dependencies import REPO_ROOT

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
    path = _aoi_geojson_path(aoi)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"AoI geometry not found for {aoi}")

    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf["area_km2"] = [round(_geod_area_km2(g), 4) for g in gdf.geometry]
    fc = json.loads(gdf.to_json())
    fc["area_km2"] = round(float(gdf["area_km2"].sum()), 4)
    fc["aoi"] = aoi
    return fc


@router.get("/landcover")
def geometry_landcover(aoi: str, land_cover_class: str | None = None) -> dict:
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
        gdf = gpd.read_file(path).to_crs("EPSG:4326")
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

    return {"type": "FeatureCollection", "aoi": aoi, "features": features}


@router.get("/fire-return-period")
def geometry_fire_return_period(aoi: str, resolution: int = 500) -> dict:
    path = (REPO_ROOT / "outputs" / "processed" / aoi / "burned_area"
            / f"{resolution}m" / "fire_return_period.geojson")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No fire return period data for {aoi}")
    with open(path) as f:
        return json.load(f)
