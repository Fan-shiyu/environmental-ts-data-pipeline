"""
Investigation: verify AoI polygon clipping in the burned-area pipeline.

Four checks on August 2024 Zambia_WL data:
  1. Raster footprint comparison (shape, transform, CRS, bbox)
  2. Polygon-vs-bounding-box clip verification
  3. Are "only-in-new" pixels inside the AoI polygon?
  4. Same check on overlap and legacy-only pixels

Run with:
    python scripts/check_burned_area_aoi.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.export import download_image
from pipeline.sentinel2 import load_aoi

# -- constants ------------------------------------------------------------------

YEAR  = 2024
MONTH = 8
AOI   = "Zambia_WL"
LABEL = f"{YEAR}-{MONTH:02d}"
SCALE = 500

LEGACY_PATH = (
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series"
    r"\app\www\data\BurnedArea\Zambia_WL\500m_resolution"
    r"\2024-08_BurnedArea_Zambia_WL.tif"
)
NEW_PATH = f"outputs/{AOI}/burned_area/500m/{LABEL}_BurnedArea_{AOI}.tif"
AOI_GEOJSON = (
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series"
    r"\app\www\data\AoI\AoI_Zambia_WL.geojson"
)

# -- helpers --------------------------------------------------------------------

def _load_burned(path: str, ref_transform=None, ref_crs=None, ref_shape=None):
    """Load a BurnDate raster. If ref grid given, reproject onto it.
    Returns (data int32, transform, crs, shape, nodata).
    Negative values and nodata zeroed (0=unburned, >0=burn DOY).
    """
    with rasterio.open(path) as src:
        data       = src.read(1).astype(np.int32)
        nodata_val = src.nodata
        src_tr     = src.transform
        src_crs    = src.crs
        src_shape  = src.shape

    if ref_transform is not None:
        grids_match = (
            src_crs == ref_crs
            and src_shape == ref_shape
            and abs(src_tr.a - ref_transform.a) < 1e-8
            and abs(src_tr.e - ref_transform.e) < 1e-8
            and abs(src_tr.c - ref_transform.c) < 1e-8
            and abs(src_tr.f - ref_transform.f) < 1e-8
        )
        if not grids_match:
            print(f"    Reprojecting {Path(path).name} onto reference grid ...")
            reprojected = np.zeros(ref_shape, dtype=np.int32)
            with rasterio.open(path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=reprojected,
                    src_transform=src_tr,
                    src_crs=src_crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest,
                    src_nodata=nodata_val,
                    dst_nodata=0,
                )
            data = reprojected
            nodata_val = 0
            src_tr, src_crs, src_shape = ref_transform, ref_crs, ref_shape

    if nodata_val is not None:
        data = np.where(data == int(nodata_val), 0, data)
    data = np.where(data < 0, 0, data)
    return data, src_tr, src_crs, src_shape


def _load_polygon(geojson_path: str):
    """Load first geometry from a GeoJSON file as a Shapely geometry."""
    with open(geojson_path) as f:
        gj = json.load(f)
    features = gj.get("features", [])
    if not features:
        if gj.get("type") == "Feature":
            return shape(gj["geometry"])
        return shape(gj)
    return shape(features[0]["geometry"])


def _polygon_mask(polygon, transform, shape_hw):
    """Return boolean array (H, W): True = OUTSIDE polygon (rasterio convention)."""
    mask = rasterio.features.geometry_mask(
        [polygon.__geo_interface__],
        out_shape=shape_hw,
        transform=transform,
        invert=False,   # True where outside (default)
        all_touched=True,
    )
    return mask  # True = outside polygon


def _pixel_latlon(row, col, transform):
    """Convert (row, col) to (lat, lon) using raster transform."""
    lon, lat = rasterio.transform.xy(transform, row, col)
    return lat, lon


def _inside_outside_report(label: str, pixel_mask: np.ndarray,
                             outside_mask: np.ndarray, transform,
                             print_samples: bool = False):
    """Given a bool pixel_mask (True = pixel of interest) and outside_mask
    (True = outside polygon), report how many are inside/outside polygon."""
    n_total   = int(pixel_mask.sum())
    n_outside = int((pixel_mask & outside_mask).sum())
    n_inside  = n_total - n_outside

    pct_inside  = 100.0 * n_inside  / n_total if n_total > 0 else 0.0
    pct_outside = 100.0 * n_outside / n_total if n_total > 0 else 0.0

    print(f"    {label}")
    print(f"      Total pixels:          {n_total:>8,}")
    print(f"      Inside polygon:        {n_inside:>8,}  ({pct_inside:.1f}%)")
    print(f"      Outside polygon:       {n_outside:>8,}  ({pct_outside:.1f}%)")

    if print_samples and n_outside > 0:
        rows, cols = np.where(pixel_mask & outside_mask)
        rng = np.random.default_rng(42)
        indices = rng.choice(len(rows), size=min(5, len(rows)), replace=False)
        print(f"      Sample outside-polygon pixels (row, col, lat, lon):")
        for i in indices:
            r, c = int(rows[i]), int(cols[i])
            lat, lon = _pixel_latlon(r, c, transform)
            print(f"        row={r:4d}, col={c:4d}, lat={lat:.4f}, lon={lon:.4f}")

    return n_inside, n_outside


# -- main ----------------------------------------------------------------------─

print("=" * 60)
print(f"AoI Verification  {LABEL}  AoI={AOI}")
print("=" * 60)

config = load_config()

# Ensure new output exists
if not Path(NEW_PATH).exists():
    print(f"\nNew output not found at {NEW_PATH} -- regenerating ...")
    init_gee(config["project"])
    aoi_geom = load_aoi(config["aois"][AOI]["path"])
    from pipeline.burned_area import monthly_image as ba_monthly_image
    try:
        composite = ba_monthly_image(aoi_geom, YEAR, MONTH)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    try:
        download_image(composite, aoi_geom, NEW_PATH, scale=SCALE)
    except RuntimeError as exc:
        print(f"\nERROR downloading: {exc}")
        sys.exit(1)
else:
    print(f"\nNew output exists: {NEW_PATH}")

# -- Check 1: footprint comparison --------------------------------------------─

print("\n" + "=" * 60)
print("Check 1: Raster footprint comparison")
print("=" * 60)

with rasterio.open(LEGACY_PATH) as src:
    leg_shape = src.shape
    leg_tr    = src.transform
    leg_crs   = src.crs
    leg_bbox  = src.bounds

with rasterio.open(NEW_PATH) as src:
    new_shape = src.shape
    new_tr    = src.transform
    new_crs   = src.crs
    new_bbox  = src.bounds

def _fmt_tr(t):
    return f"({t.a:.8f}, {t.b:.8f}, {t.c:.8f}, {t.d:.8f}, {t.e:.8f}, {t.f:.8f})"

print(f"\n  {'Property':<20} {'Legacy':>30}  {'New':>30}")
print(f"  {'-'*85}")
print(f"  {'Shape (rows,cols)':<20} {str(leg_shape):>30}  {str(new_shape):>30}")
print(f"  {'CRS':<20} {str(leg_crs):>30}  {str(new_crs):>30}")
print(f"  {'Transform a (dX)':<20} {leg_tr.a:>30.8f}  {new_tr.a:>30.8f}")
print(f"  {'Transform e (dY)':<20} {leg_tr.e:>30.8f}  {new_tr.e:>30.8f}")
print(f"  {'Transform c (W)':<20} {leg_tr.c:>30.8f}  {new_tr.c:>30.8f}")
print(f"  {'Transform f (N)':<20} {leg_tr.f:>30.8f}  {new_tr.f:>30.8f}")
print(f"  {'BBox left':<20} {leg_bbox.left:>30.8f}  {new_bbox.left:>30.8f}")
print(f"  {'BBox right':<20} {leg_bbox.right:>30.8f}  {new_bbox.right:>30.8f}")
print(f"  {'BBox top':<20} {leg_bbox.top:>30.8f}  {new_bbox.top:>30.8f}")
print(f"  {'BBox bottom':<20} {leg_bbox.bottom:>30.8f}  {new_bbox.bottom:>30.8f}")

shape_ok = leg_shape == new_shape
crs_ok   = leg_crs == new_crs
tr_ok    = all(abs(getattr(leg_tr, k) - getattr(new_tr, k)) < 1e-8
               for k in ("a", "b", "c", "d", "e", "f"))

if shape_ok and crs_ok and tr_ok:
    verdict1 = "footprints match"
else:
    diffs = []
    if not shape_ok:  diffs.append("shape")
    if not crs_ok:    diffs.append("CRS")
    if not tr_ok:     diffs.append("transform")
    verdict1 = f"footprints differ in {', '.join(diffs)}"

print(f"\n  Verdict 1: {verdict1}")

# -- Check 2: polygon-vs-bounding-box clip ------------------------------------─

print("\n" + "=" * 60)
print("Check 2: Polygon-vs-bounding-box clip verification")
print("=" * 60)

polygon = _load_polygon(AOI_GEOJSON)
print(f"\n  AoI polygon loaded. Bounds: {polygon.bounds}")

for label, path, tr, shp in [
    ("New output", NEW_PATH, new_tr, new_shape),
    ("Legacy    ", LEGACY_PATH, leg_tr, leg_shape),
]:
    print(f"\n  [{label}]")
    with rasterio.open(path) as src:
        raw = src.read(1).astype(np.int32)
        nd  = src.nodata

    n_total = raw.size

    # Outside polygon mask
    outside = _polygon_mask(polygon, tr, shp)
    n_outside_px = int(outside.sum())
    n_inside_px  = n_total - n_outside_px

    # Nodata handling
    if nd is not None:
        is_nodata = (raw == int(nd)) | (raw < -1000)
    else:
        is_nodata = raw < -1000

    # Breakdown for outside-polygon pixels
    outside_nodata = outside & is_nodata
    outside_hasval = outside & ~is_nodata

    # Inside-polygon pixels with nodata vs values
    inside_nodata  = ~outside & is_nodata
    inside_hasval  = ~outside & ~is_nodata

    print(f"    Total raster pixels:         {n_total:>8,}")
    print(f"    Pixels inside polygon:       {n_inside_px:>8,}")
    print(f"    Pixels outside polygon:      {n_outside_px:>8,}")
    print(f"      Of which: nodata (correct):  {int(outside_nodata.sum()):>8,}")
    print(f"      Of which: have values (bad): {int(outside_hasval.sum()):>8,}")
    print(f"    Inside polygon - with values:{int(inside_hasval.sum()):>8,}")
    print(f"    Inside polygon - nodata:     {int(inside_nodata.sum()):>8,}")

    n_bad = int(outside_hasval.sum())
    if n_bad == 0:
        vd = "polygon clip applied"
    elif n_outside_px > 0 and n_bad > 0:
        pct_bad = 100.0 * n_bad / n_outside_px
        if pct_bad > 50:
            vd = "only bounding box clip applied"
        else:
            vd = f"partial polygon clip ({n_bad} outside-polygon pixels with values)"
    else:
        vd = "no clip applied"
    print(f"    -> {vd}")

verdict2_new = None
if int(outside_hasval.sum()) == 0:
    verdict2_new = "polygon clip applied"
else:
    verdict2_new = "partial or missing polygon clip"

# -- Check 3: are "only-in-new" pixels inside polygon? ------------------------─

print("\n" + "=" * 60)
print("Check 3: Are 'only-in-new' burned pixels inside the AoI polygon?")
print("=" * 60)

# Load both on common grid (use legacy as ref since footprints should match)
legacy_data, ref_tr, ref_crs, ref_shp = _load_burned(LEGACY_PATH)
new_data,    _,      _,        _       = _load_burned(NEW_PATH, ref_tr, ref_crs, ref_shp)

burned_legacy = legacy_data > 0
burned_new    = new_data    > 0

only_in_new   = burned_new  & ~burned_legacy
overlap       = burned_new  &  burned_legacy
only_in_legacy = burned_legacy & ~burned_new

outside_mask = _polygon_mask(polygon, ref_tr, ref_shp)

print(f"\n  Burned pixel counts:")
print(f"    Legacy:        {int(burned_legacy.sum()):>8,}")
print(f"    New:           {int(burned_new.sum()):>8,}")
print(f"    Only-in-new:   {int(only_in_new.sum()):>8,}")
print(f"    Overlap:       {int(overlap.sum()):>8,}")
print(f"    Only-in-legacy:{int(only_in_legacy.sum()):>8,}")

print()
n_onlynew_in, n_onlynew_out = _inside_outside_report(
    "Only-in-new pixels", only_in_new, outside_mask, ref_tr, print_samples=True
)

if n_onlynew_out == 0:
    verdict3 = "all 'only-in-new' inside polygon -- AoI is not the cause of the gap"
elif n_onlynew_out / max(1, int(only_in_new.sum())) < 0.05:
    verdict3 = f"nearly all inside polygon ({n_onlynew_out} marginal edge pixels outside -- at all_touched boundary)"
else:
    verdict3 = f"SIGNIFICANT share outside polygon ({n_onlynew_out} pixels) -- AoI handling may be a contributor"

print(f"\n  Verdict 3: {verdict3}")

# -- Check 4: same check on overlap and legacy-only ----------------------------

print("\n" + "=" * 60)
print("Check 4: Polygon containment for overlap and legacy-only pixels")
print("=" * 60)

print()
n_ov_in, n_ov_out = _inside_outside_report(
    "Overlap pixels (burned in both)", overlap, outside_mask, ref_tr
)
print()
n_leg_in, n_leg_out = _inside_outside_report(
    "Only-in-legacy pixels", only_in_legacy, outside_mask, ref_tr
)

if n_ov_out == 0 and n_leg_out == 0:
    verdict4 = "all overlap and legacy-only pixels inside polygon"
elif n_ov_out > 0 or n_leg_out > 0:
    verdict4 = f"some pixels outside polygon (overlap: {n_ov_out}, legacy-only: {n_leg_out})"
else:
    verdict4 = "no legacy or overlap pixels present to check"

print(f"\n  Verdict 4: {verdict4}")

# -- Structured conclusion ------------------------------------------------------

print("\n" + "=" * 60)
print("=== AOI VERIFICATION CONCLUSION ===")
print("=" * 60)
print(f"Check 1 (footprints):       {verdict1}")

# Determine Check 2 verdict using last label processed (new output)
with rasterio.open(NEW_PATH) as src:
    raw_new  = src.read(1).astype(np.int32)
    nd_new   = src.nodata
outside_new = _polygon_mask(polygon, new_tr, new_shape)
if nd_new is not None:
    is_nodata_new = (raw_new == int(nd_new)) | (raw_new < -1000)
else:
    is_nodata_new = raw_new < -1000
outside_hasval_new = outside_new & ~is_nodata_new
c2_new = "polygon clip applied" if int(outside_hasval_new.sum()) == 0 else f"partial clip ({int(outside_hasval_new.sum())} outside-polygon pixels with values)"

with rasterio.open(LEGACY_PATH) as src:
    raw_leg = src.read(1).astype(np.int32)
    nd_leg  = src.nodata
outside_leg = _polygon_mask(polygon, leg_tr, leg_shape)
if nd_leg is not None:
    is_nodata_leg = (raw_leg == int(nd_leg)) | (raw_leg < -1000)
else:
    is_nodata_leg = raw_leg < -1000
outside_hasval_leg = outside_leg & ~is_nodata_leg
c2_leg = "polygon clip applied" if int(outside_hasval_leg.sum()) == 0 else f"partial clip ({int(outside_hasval_leg.sum())} outside-polygon pixels with values)"

print(f"Check 2 (polygon clip):     new={c2_new}  |  legacy={c2_leg}")
print(f"Check 3 (only-in-new):      {verdict3}")
print(f"Check 4 (overlap & legacy): {verdict4}")

all_correct = (
    ("match" in verdict1 or "differ" not in verdict1)
    and "polygon clip applied" in c2_new
    and "not the cause" in verdict3
)

print()
if all_correct:
    print("Overall: AOI handling is CORRECT")
    print()
    print("AoI is fully ruled out as a cause of the 33% burned-pixel gap.")
else:
    issues = []
    if "differ" in verdict1:
        issues.append(f"footprint mismatch: {verdict1}")
    if "polygon clip applied" not in c2_new:
        issues.append(f"new output clip: {c2_new}")
    if "polygon clip applied" not in c2_leg:
        issues.append(f"legacy clip: {c2_leg}")
    if "not the cause" not in verdict3 and "nearly all" not in verdict3:
        issues.append(f"only-in-new location: {verdict3}")

    if issues:
        print("Overall: AOI handling is PARTIALLY CORRECT or INCORRECT")
        print()
        for issue in issues:
            print(f"  Issue: {issue}")
    else:
        print("Overall: AOI handling is CORRECT (minor edge effects only)")
        print()
        print("AoI is fully ruled out as a cause of the 33% burned-pixel gap.")

print(f"\nDone.  {LABEL}  AoI={AOI}")
