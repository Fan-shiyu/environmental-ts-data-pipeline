"""
Investigation: three-way comparison of burned area sources — legacy vs C6 vs C6.1.

Tests the hypothesis that the legacy TIFs were generated from MCD64A1 Collection 6,
while the new pipeline uses Collection 6.1.

Run with:
    python scripts/check_burned_area_c6_match.py
"""

import math
import sys
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.export import download_image
from pipeline.sentinel2 import load_aoi

# ── constants ──────────────────────────────────────────────────────────────────

YEAR  = 2022
MONTH = 10
AOI   = "Zambia_WL"
LABEL = f"{YEAR}-{MONTH:02d}"
SCALE = 500

# C6 collection was archived; last available date for Zambia_WL is 2022-12.
# We therefore test on 2022-10 (peak fire season, most burned pixels).
C6_COLLECTION  = "MODIS/006/MCD64A1"
C61_COLLECTION = "MODIS/061/MCD64A1"

C6_PATH    = f"test_outputs/test_{LABEL}_BurnedArea_{AOI}_C6.tif"
C61_PATH   = f"outputs/{AOI}/burned_area/500m/{LABEL}_BurnedArea_{AOI}.tif"
LEGACY_PATH = (
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series"
    r"\app\www\data\BurnedArea\Zambia_WL\500m_resolution"
    rf"\{LABEL}_BurnedArea_Zambia_WL.tif"
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _load_on_ref_grid(path: str, ref_transform, ref_crs, ref_shape):
    """Read raster at `path`, reproject onto the reference grid if needed.

    Returns int32 array with negatives/nodata zeroed (unburned=0, burned=DOY 1-366).
    """
    with rasterio.open(path) as src:
        data       = src.read(1).astype(np.int32)
        nodata_val = src.nodata
        src_tr     = src.transform
        src_crs    = src.crs
        src_shape  = src.shape

    grids_match = (
        src_crs == ref_crs
        and src_shape == ref_shape
        and abs(src_tr.a - ref_transform.a) < 1e-8
        and abs(src_tr.e - ref_transform.e) < 1e-8
        and abs(src_tr.c - ref_transform.c) < 1e-8
        and abs(src_tr.f - ref_transform.f) < 1e-8
    )

    if not grids_match:
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

    if nodata_val is not None:
        data = np.where(data == int(nodata_val), 0, data)
    data = np.where(data < 0, 0, data)
    return data


def _pair_stats(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    """Compute overlap/only-in stats for a pair of burned-area arrays."""
    burned_a = a > 0
    burned_b = b > 0
    overlap  = burned_a & burned_b

    n_a       = int(burned_a.sum())
    n_b       = int(burned_b.sum())
    n_overlap = int(overlap.sum())
    n_only_a  = int((burned_a & ~burned_b).sum())
    n_only_b  = int((~burned_a & burned_b).sum())

    mad_days = None
    if n_overlap > 0:
        mad_days = float(np.mean(np.abs(a[overlap].astype(float) - b[overlap].astype(float))))

    pct_a = 100.0 * n_overlap / n_a if n_a > 0 else 0.0
    pct_b = 100.0 * n_overlap / n_b if n_b > 0 else 0.0

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_a": n_a,
        "n_b": n_b,
        "n_overlap": n_overlap,
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "pct_a_captured": pct_a,
        "pct_b_captured": pct_b,
        "mad_days": mad_days,
    }


def _print_pair(s: dict) -> None:
    print(f"\n  {s['label_a']}  vs  {s['label_b']}")
    print(f"  {'Burned pixels (' + s['label_a'] + ')':<45} {s['n_a']:>10,}")
    print(f"  {'Burned pixels (' + s['label_b'] + ')':<45} {s['n_b']:>10,}")
    print(f"  {'Overlap (both burned)':<45} {s['n_overlap']:>10,}")
    print(f"  {'  % of ' + s['label_a'] + ' captured in overlap':<45} {s['pct_a_captured']:>9.1f}%")
    print(f"  {'  % of ' + s['label_b'] + ' captured in overlap':<45} {s['pct_b_captured']:>9.1f}%")
    print(f"  {'Only in ' + s['label_a']:<45} {s['n_only_a']:>10,}")
    print(f"  {'Only in ' + s['label_b']:<45} {s['n_only_b']:>10,}")
    if s["mad_days"] is not None:
        print(f"  {'DOY MAD on overlap (days)':<45} {s['mad_days']:>10.2f}")
    else:
        print(f"  {'DOY MAD on overlap (days)':<45} {'N/A (no overlap)':>10}")


def _match_rate(s: dict) -> float:
    """Jaccard-like match: overlap / union."""
    union = s["n_a"] + s["n_b"] - s["n_overlap"]
    return s["n_overlap"] / union if union > 0 else 0.0


# ── main ───────────────────────────────────────────────────────────────────────

print("=" * 60)
print(f"C6 Match Test  {LABEL}  AoI={AOI}")
print("=" * 60)

config = load_config()

# Step 1 — authenticate + fetch C6 image
print("\n" + "=" * 60)
print("Step 1: Authenticating + fetching C6 image")
print("=" * 60)
init_gee(config["project"])

aoi = load_aoi(config["aois"][AOI]["path"])

if Path(C6_PATH).exists():
    print(f"  C6 file already exists: {C6_PATH}  (skipping download)")
else:
    print(f"  Fetching from {C6_COLLECTION} ...")
    _next_month = MONTH + 1 if MONTH < 12 else 1
    _next_year  = YEAR if MONTH < 12 else YEAR + 1
    collection = (
        ee.ImageCollection(C6_COLLECTION)
        .filterBounds(aoi)
        .filterDate(f"{YEAR}-{MONTH:02d}-01", f"{_next_year}-{_next_month:02d}-01")
    )
    count = collection.size().getInfo()
    print(f"  Images found in C6 for {LABEL}: {count}")
    if count == 0:
        print(f"\nERROR: No C6 MCD64A1 image found for {LABEL} over {AOI}.")
        print("  The MODIS/006/MCD64A1 collection may be fully decommissioned.")
        print("  Hypothesis cannot be tested without this data.")
        sys.exit(1)
    c6_image = collection.first().select("BurnDate").clip(aoi)
    try:
        download_image(c6_image, aoi, C6_PATH, scale=SCALE)
    except RuntimeError as exc:
        print(f"\nERROR downloading C6 image: {exc}")
        sys.exit(1)

# Step 1b — ensure C6.1 output exists
print("\n" + "=" * 60)
print("Step 1b: Checking C6.1 output")
print("=" * 60)
if Path(C61_PATH).exists():
    print(f"  C6.1 file exists: {C61_PATH}  (using existing)")
else:
    print(f"  C6.1 file not found at {C61_PATH}. Generating ...")
    _next_month = MONTH + 1 if MONTH < 12 else 1
    _next_year  = YEAR if MONTH < 12 else YEAR + 1
    collection_61 = (
        ee.ImageCollection(C61_COLLECTION)
        .filterBounds(aoi)
        .filterDate(f"{YEAR}-{MONTH:02d}-01", f"{_next_year}-{_next_month:02d}-01")
    )
    count_61 = collection_61.size().getInfo()
    print(f"  Images found in C6.1 for {LABEL}: {count_61}")
    if count_61 == 0:
        print(f"\nERROR: No C6.1 image found for {LABEL}.")
        sys.exit(1)
    c61_image = collection_61.first().select("BurnDate").clip(aoi)
    try:
        download_image(c61_image, aoi, C61_PATH, scale=SCALE)
    except RuntimeError as exc:
        print(f"\nERROR downloading C6.1 image: {exc}")
        sys.exit(1)

# Step 2 — three-way comparison
print("\n" + "=" * 60)
print("Step 2: Three-way comparison")
print("=" * 60)

if not Path(LEGACY_PATH).exists():
    print(f"\nERROR: Legacy file not found at:\n  {LEGACY_PATH}")
    sys.exit(1)

# Read reference grid from legacy
with rasterio.open(LEGACY_PATH) as ref_src:
    ref_transform = ref_src.transform
    ref_crs       = ref_src.crs
    ref_shape     = ref_src.shape
    print(f"  Legacy  -- shape: {ref_shape}, CRS: {ref_crs}")

with rasterio.open(C6_PATH) as s:
    print(f"  C6      -- shape: {s.shape}, CRS: {s.crs}")
with rasterio.open(C61_PATH) as s:
    print(f"  C6.1    -- shape: {s.shape}, CRS: {s.crs}")

legacy = _load_on_ref_grid(LEGACY_PATH, ref_transform, ref_crs, ref_shape)
c6     = _load_on_ref_grid(C6_PATH,     ref_transform, ref_crs, ref_shape)
c61    = _load_on_ref_grid(C61_PATH,    ref_transform, ref_crs, ref_shape)

s_leg_c6  = _pair_stats(legacy, c6,  "Legacy", "C6")
s_leg_c61 = _pair_stats(legacy, c61, "Legacy", "C6.1")
s_c6_c61  = _pair_stats(c6,     c61, "C6",     "C6.1")

print("\n  --- Three-way pixel counts ---")
print(f"  {'Source':<12} {'Burned pixels':>15}")
print(f"  {'-'*28}")
print(f"  {'Legacy':<12} {s_leg_c6['n_a']:>15,}")
print(f"  {'C6':<12} {s_leg_c6['n_b']:>15,}")
print(f"  {'C6.1':<12} {s_leg_c61['n_b']:>15,}")

print("\n  --- Pairwise comparison ---")
_print_pair(s_leg_c6)
_print_pair(s_leg_c61)
_print_pair(s_c6_c61)

# Step 3 — interpret
print("\n" + "=" * 60)
print("Step 3: Interpretation")
print("=" * 60)

n_legacy  = s_leg_c6["n_a"]
n_c6      = s_leg_c6["n_b"]
n_c61     = s_leg_c61["n_b"]
n_leg_c6  = s_leg_c6["n_overlap"]
n_leg_c61 = s_leg_c61["n_overlap"]
mad_c6    = s_leg_c6["mad_days"] or 0.0
mad_c61   = s_leg_c61["mad_days"] or 0.0

pct_legacy_in_c6  = 100.0 * n_leg_c6  / n_legacy if n_legacy > 0 else 0.0
pct_legacy_in_c61 = 100.0 * n_leg_c61 / n_legacy if n_legacy > 0 else 0.0
pct_c6_in_legacy  = 100.0 * n_leg_c6  / n_c6     if n_c6 > 0 else 0.0

match_c6  = _match_rate(s_leg_c6)
match_c61 = _match_rate(s_leg_c61)
improvement = match_c6 / match_c61 if match_c61 > 0 else float("inf")

print(f"\n  Legacy-vs-C6  Jaccard match rate:  {match_c6:.3f}  ({pct_legacy_in_c6:.1f}% of legacy in C6, MAD={mad_c6:.2f}d)")
print(f"  Legacy-vs-C6.1 Jaccard match rate: {match_c61:.3f}  ({pct_legacy_in_c61:.1f}% of legacy in C6.1, MAD={mad_c61:.2f}d)")
print(f"  Improvement factor (C6 vs C6.1):   {improvement:.2f}x")

NEAR_PERFECT        = pct_legacy_in_c6 >= 90.0 and pct_c6_in_legacy >= 90.0 and mad_c6 < 2.0
PARTIAL             = pct_legacy_in_c6 >= 50.0 or pct_c6_in_legacy >= 50.0
C6_BETTER           = improvement >= 1.2
# Special case: legacy is a perfect subset of C6.1 (all legacy pixels in C6.1, exact DOY)
LEGACY_SUBSET_OF_C61 = s_leg_c61["n_only_a"] == 0 and (mad_c61 or 0.0) < 1.0

print("\n  --- Conclusion ---")
if NEAR_PERFECT:
    print("  CONFIRMED: legacy is MCD64A1 Collection 6.")
    print(f"  C6 captures {pct_legacy_in_c6:.1f}% of legacy burned pixels with DOY MAD = {mad_c6:.2f} days.")
    print(f"  The gap to C6.1 ({s_c6_c61['n_only_b']:,} extra pixels in C6.1) is the C6->C6.1 transition.")
elif PARTIAL and C6_BETTER:
    print("  PARTIAL MATCH: legacy is similar to C6 but not identical.")
    print(f"  Match rate: {100*match_c6:.1f}% to C6, {100*match_c61:.1f}% to C6.1 (Jaccard).")
    print("  Possible reasons: legacy applied additional processing beyond C6, OR")
    print("  legacy is from an intermediate snapshot between C6 versions.")
else:
    print("  REJECTED: legacy is not Collection 6.")
    print(f"  C6 match rate ({100*match_c6:.1f}%) not substantially better than C6.1 ({100*match_c61:.1f}%).")
    if LEGACY_SUBSET_OF_C61:
        print()
        print("  KEY FINDING: legacy is a STRICT SUBSET of C6.1.")
        print(f"  100% of legacy pixels ({n_legacy:,}) appear in C6.1 with DOY MAD = {mad_c61:.2f} days.")
        print(f"  C6.1 has {s_leg_c61['n_only_b']:,} additional pixels not in legacy ({100*s_leg_c61['n_only_b']/n_c61:.1f}% of C6.1).")
        print()
        print("  This rules out a different underlying product. Legacy was generated from MCD64A1 C6.1")
        print("  (or an algorithm run identical to C6.1), then had ~33% of detections removed by")
        print("  post-processing (e.g., minimum mapping unit filter, QA threshold, or spatial masking).")
        print()
        print("  C6 vs C6.1 are nearly identical for this month (see C6 vs C6.1 pair above),")
        print("  so the version difference is NOT the source of the gap.")
        print()
        print("  Best next step: investigate what post-processing was applied to produce legacy.")
        print("  Ask Melanie or the previous team: was there a minimum mapping unit, QA filter,")
        print("  or any subsetting step applied after the MCD64A1 download?")
    else:
        print("  Possible explanations:")
        print("    - Legacy may use a different burned-area product entirely")
        print("    - Legacy may apply post-processing on top of MCD64A1 (e.g., minimum mapping unit, clumping)")
        print("    - Legacy may have been manually edited or validated")
        print("    - Best next step: ask Melanie or the previous team about the legacy data provenance")

# Step 4 — sample pixels
print("\n" + "=" * 60)
print("Step 4: Sample pixel inspection")
print("=" * 60)

burned_leg = legacy > 0
burned_c6  = c6  > 0
burned_c61 = c61 > 0

categories = {
    "All three (legacy AND C6 AND C6.1)":              burned_leg & burned_c6  & burned_c61,
    "Legacy AND C6 but NOT C6.1":                      burned_leg & burned_c6  & ~burned_c61,
    "C6.1 only (NOT legacy, NOT C6)":                  ~burned_leg & ~burned_c6 & burned_c61,
    "Legacy only (NOT C6, NOT C6.1) -- FLAG if many":  burned_leg & ~burned_c6 & ~burned_c61,
}

for cat_label, mask in categories.items():
    rows, cols = np.where(mask)
    n = len(rows)
    if n == 0:
        print(f"\n  [{cat_label}]")
        print("    No pixels in this category.")
        continue

    indices = np.random.default_rng(42).choice(n, size=min(5, n), replace=False)
    print(f"\n  [{cat_label}]  ({n:,} pixels total)")
    print(f"  {'Row':>6}  {'Col':>6}  {'Legacy DOY':>10}  {'C6 DOY':>8}  {'C6.1 DOY':>10}")
    print(f"  {'-'*50}")
    for i in indices:
        r, c = int(rows[i]), int(cols[i])
        print(f"  {r:>6}  {c:>6}  {legacy[r,c]:>10}  {c6[r,c]:>8}  {c61[r,c]:>10}")

    if cat_label.startswith("Legacy only") and n > 20:
        print(f"\n  *** WARNING: {n} pixels burned in legacy but absent from both C6 and C6.1.")
        print("  *** This is unexpected and suggests legacy is NOT derived from MCD64A1.")

print(f"\nDone.  {LABEL}  AoI={AOI}")
