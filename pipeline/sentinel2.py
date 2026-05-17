"""Sentinel-2 processing: collection loading, SCL cloud masking, and NDVI calculation."""

import calendar
import sys

import ee
import geopandas as gpd


def load_aoi(geojson_path: str) -> ee.Geometry:
    """Read a GeoJSON file, reproject to EPSG:4326 if needed, return ee.Geometry."""
    from pathlib import Path

    path = Path(geojson_path)
    if not path.exists():
        print(f"ERROR: AoI file not found at:\n  {path}")
        sys.exit(1)

    gdf = gpd.read_file(path)
    print(f"  AoI loaded. CRS: {gdf.crs}, features: {len(gdf)}")

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        print("  Reprojecting AoI to EPSG:4326 ...")
        gdf = gdf.to_crs(epsg=4326)

    aoi_geom = gdf.geometry.union_all()
    aoi_ee = ee.Geometry(aoi_geom.__geo_interface__)
    print("  AoI converted to ee.Geometry.\n")
    return aoi_ee


def mask_scl_clouds(image: ee.Image) -> ee.Image:
    """Apply corrected SCL mask: classes 1, 2, 3, 7, 8, 9, 10, 11 masked.

    See reference/README.md for the bug this fixes: the legacy preprocGEE.js
    had three .and(scl.neq(7)) lines that were meant to mask classes 2, 3, 7
    but all three only masked class 7.
    """
    scl = image.select("SCL")
    mask = (
        scl.neq(1)            # 1 = Saturated / Defective
           .And(scl.neq(2))   # 2 = Dark area pixels
           .And(scl.neq(3))   # 3 = Cloud shadows
           .And(scl.neq(7))   # 7 = Unclassified
           .And(scl.neq(8))   # 8 = Cloud medium probability
           .And(scl.neq(9))   # 9 = Cloud high probability
           .And(scl.neq(10))  # 10 = Thin cirrus
           .And(scl.neq(11))  # 11 = Snow / ice
    )
    return image.updateMask(mask)


def add_ndvi(image: ee.Image) -> ee.Image:
    """Compute NDVI = (B8 - B4) / (B8 + B4), add as 'NDVI' band."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def monthly_composite(aoi: ee.Geometry, year: int, month: int) -> ee.Image:
    """Build the monthly NDVI composite for the given AoI and month.

    Internal flow: filter S2_SR_HARMONIZED by bounds and date range,
    filter cloudy_pixel_percentage < 100, mask SCL, compute NDVI,
    take median, clip to AoI.
    """
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    date_start = f"{year}-{month:02d}-01"
    date_end = f"{next_year}-{next_month:02d}-01"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 100))
    )

    print(f"  Date range: {date_start} to {date_end}")
    print(f"  Images in collection: {collection.size().getInfo()}")

    composite = (
        collection
        .map(mask_scl_clouds)
        .map(add_ndvi)
        .select("NDVI")
        .median()
        .clip(aoi)
    )
    print("  NDVI composite built (median reducer, corrected SCL mask).\n")
    return composite
