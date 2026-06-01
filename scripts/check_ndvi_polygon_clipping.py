"""
Investigation: do the NDVI pipelines apply polygon clipping or only bounding-box clipping?

Checks Sentinel-2 and MODIS NDVI outputs for Zambia_WL (no legacy) and Zambia_Mponda
(legacy exists for comparison).  Generates missing outputs on the fly via the pipeline.

Run with:
    python scripts/check_ndvi_polygon_clipping.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import shape

from pipeline.auth import init_gee
from pipeline.config import load_config

# -- constants -----------------------------------------------------------------

YEAR, MONTH = 2024, 3
LABEL = f"{YEAR}-{MONTH:02d}"

AOI_WL_GEOJSON  = (
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series"
    r"\app\www\data\AoI\AoI_Zambia_WL.geojson"
)
AOI_MPN_GEOJSON = (
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series"
    r"\app\www\data\AoI\AoI_Zambia_Mponda_By_Life_Connected.geojson"
)

LEGACY_ROOT = (
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series"
    r"\app\www\data\NDVI"
)

CHECKS = [
    # (sensor, resolution, new_path, legacy_path_or_None)
    (
        "sentinel2", 100,
        "outputs/Zambia_WL/sentinel2/100m/2024-03_NDVI_Zambia_WL.tif",
        None,
    ),
    (
        "sentinel2", 1000,
        "outputs/Zambia_WL/sentinel2/1000m/2024-03_NDVI_Zambia_WL.tif",
        fr"{LEGACY_ROOT}\Zambia_Mponda\250m_resolution\2024-03_NDVI_Zambia_Mponda.tif",
    ),
    (
        "modis", 250,
        "outputs/Zambia_WL/modis/250m/2024-03_NDVI_Zambia_WL.tif",
        fr"{LEGACY_ROOT}\Zambia_Mponda\250m_resolution\2024-03_NDVI_Zambia_Mponda.tif",
    ),
    (
        "modis", 500,
        "outputs/Zambia_WL/modis/500m/2024-03_NDVI_Zambia_WL.tif",
        fr"{LEGACY_ROOT}\Zambia_Mponda\500m_resolution\2024-03_NDVI_Zambia_Mponda.tif",
    ),
    (
        "modis", 1000,
        "outputs/Zambia_WL/modis/1000m/2024-03_NDVI_Zambia_WL.tif",
        fr"{LEGACY_ROOT}\Zambia_Mponda\MODIS_1000m_resolution\2024-03_NDVI_Zambia_Mponda.tif",
    ),
]

# -- helpers -------------------------------------------------------------------

def _load_polygon(geojson_path: str):
    with open(geojson_path) as f:
        gj = json.load(f)
    features = gj.get("features", [])
    if features:
        return shape(features[0]["geometry"])
    if gj.get("type") == "Feature":
        return shape(gj["geometry"])
    return shape(gj)


def _outside_mask(polygon, raster_path: str):
    """Return bool array: True = pixel center is outside the polygon."""
    with rasterio.open(raster_path) as src:
        shp = src.shape
        tr  = src.transform
    return rasterio.features.geometry_mask(
        [polygon.__geo_interface__],
        out_shape=shp,
        transform=tr,
        invert=False,
        all_touched=False,
    )


def _check_clip(path: str, polygon, label: str) -> str:
    """Return 'polygon clip' / 'bbox clip only' / 'no clip' based on outside-polygon pixels."""
    with rasterio.open(path) as src:
        data    = src.read(1).astype(np.float32)
        nodata  = src.nodata
        shape_  = src.shape

    if nodata is not None:
        data = np.where(data == float(nodata), np.nan, data)
    # also treat very large fill values as nodata
    data = np.where(np.abs(data) > 1e6, np.nan, data)

    outside = _outside_mask(polygon, path)
    n_total   = data.size
    n_out     = int(outside.sum())
    n_out_nan = int(np.isnan(data[outside]).sum())
    n_out_val = n_out - n_out_nan  # outside + has finite value

    pct_out     = 100.0 * n_out     / n_total if n_total > 0 else 0.0
    pct_out_val = 100.0 * n_out_val / n_out   if n_out   > 0 else 0.0

    print(f"  [{label}]  shape={shape_}")
    print(f"    Total pixels:           {n_total:>8,}")
    print(f"    Outside polygon:        {n_out:>8,}  ({pct_out:.1f}%)")
    print(f"      Of which NaN/nodata:  {n_out_nan:>8,}  ({100-pct_out_val:.1f}% of outside)")
    print(f"      Of which has values:  {n_out_val:>8,}  ({pct_out_val:.1f}% of outside)")

    if n_out == 0:
        verdict = "no outside-polygon pixels (raster fits polygon exactly)"
    elif n_out_val == 0:
        verdict = "polygon clip applied (all outside pixels are NaN/nodata)"
    elif n_out_nan == 0:
        verdict = "bbox clip only (all outside pixels have values)"
    else:
        verdict = f"partial polygon clip ({n_out_nan} NaN, {n_out_val} with values outside polygon)"

    print(f"    -> {verdict}")
    return verdict


def _ensure_output(sensor: str, resolution: int, new_path: str) -> bool:
    """Generate the new output via backfill.py if it doesn't exist. Returns True if available."""
    if Path(new_path).exists():
        print(f"  Exists: {new_path}")
        return True
    print(f"  Not found, generating via backfill.py: {new_path}")
    cmd = [
        sys.executable, "scripts/backfill.py",
        "--aoi", "Zambia_WL",
        "--sensor", sensor,
        "--resolution", str(resolution),
        "--year", str(YEAR),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: backfill failed.\n{result.stdout}\n{result.stderr}")
        return False
    if not Path(new_path).exists():
        print(f"  ERROR: still missing after backfill.")
        return False
    print(f"  Generated: {new_path}")
    return True


# -- main ----------------------------------------------------------------------

print("=" * 60)
print("NDVI Polygon Clipping Check")
print("=" * 60)

config = load_config()

poly_wl  = _load_polygon(AOI_WL_GEOJSON)
poly_mpn = _load_polygon(AOI_MPN_GEOJSON)
print(f"\nZambia_WL  polygon bounds: {poly_wl.bounds}")
print(f"Zambia_Mponda polygon bounds: {poly_mpn.bounds}")

# Pre-generate any missing outputs
print("\n" + "=" * 60)
print("Generating missing output files (via backfill)...")
print("=" * 60)

# Need GEE for backfill
init_gee(config["project"])

for sensor, resolution, new_path, _ in CHECKS:
    _ensure_output(sensor, resolution, new_path)

# Per-sensor checks
print()
summary_rows = []

for sensor, resolution, new_path, legacy_path in CHECKS:
    print("=" * 60)
    print(f"{sensor.upper()}  {resolution}m")
    print("=" * 60)

    # New output
    if Path(new_path).exists():
        v_new = _check_clip(new_path, poly_wl, f"New  (Zambia_WL {resolution}m)")
    else:
        print(f"  SKIPPED: {new_path} not found")
        v_new = "SKIPPED"

    # Legacy
    if legacy_path and Path(legacy_path).exists():
        print()
        v_leg = _check_clip(legacy_path, poly_mpn, f"Legacy (Zambia_Mponda {resolution}m)")
    elif legacy_path:
        print(f"  Legacy not found: {legacy_path}")
        v_leg = "file missing"
    else:
        print("  Legacy: no legacy file for Zambia_WL")
        v_leg = "no legacy for WL"

    summary_rows.append((sensor, str(resolution) + "m", v_new, v_leg))
    print()

# Summary table
print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"\n  {'Sensor':<12} {'Res':<8} {'New polygon mask?':<40} {'Legacy polygon mask?'}")
print(f"  {'-'*100}")
for sensor, res, v_new, v_leg in summary_rows:
    new_yn  = "YES" if "polygon clip applied" in v_new or "fits polygon" in v_new else "NO " if "SKIPPED" not in v_new else "???"
    leg_yn  = "YES" if "polygon clip applied" in v_leg or "fits polygon" in v_leg else "NO " if v_leg not in ("file missing", "no legacy for WL", "SKIPPED") else v_leg
    print(f"  {sensor:<12} {res:<8} {new_yn}  {v_new:<37}  {leg_yn}  {v_leg}")

needs_fix = [
    f"{sensor} {res}"
    for sensor, res, v_new, v_leg in summary_rows
    if "NO" in ("YES" if "polygon clip applied" in v_new or "fits polygon" in v_new else "NO")
    and v_new != "SKIPPED"
]

print(f"\nVerdict: pipelines needing polygon-clip fix: {needs_fix if needs_fix else 'none'}")
