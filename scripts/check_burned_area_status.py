"""
One-off investigation: check the status of the MODIS MCD64A1 burned area product
on Google Earth Engine, and inspect any existing burned area files in the Shiny
app data folder.

Run with:
    python scripts/check_burned_area_status.py
"""

import datetime
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GEE authentication
# ---------------------------------------------------------------------------

print("=" * 65)
print("Authenticating to Google Earth Engine")
print("=" * 65)

try:
    import ee
    ee.Initialize(project="sensingclues-ndvi")
    print("  GEE initialised OK.\n")
except Exception as exc:
    print(f"\nERROR: GEE authentication failed.\n  {exc}")
    print("Fix: run  earthengine authenticate  and retry.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helper: query a date range and return (count, earliest, latest)
# ---------------------------------------------------------------------------

COLLECTION = "MODIS/061/MCD64A1"


def query_range(label, start, end):
    """Query MCD64A1 for a date range and print a one-line summary."""
    col = ee.ImageCollection(COLLECTION).filterDate(start, end)
    count = col.size().getInfo()

    if count == 0:
        print(f"  {label:<35}  images: 0   (no data)")
        return None, None, count

    dates = col.aggregate_array("system:time_start").getInfo()
    dates_dt = [datetime.datetime.utcfromtimestamp(ms / 1000) for ms in dates]
    earliest = min(dates_dt).strftime("%Y-%m-%d")
    latest   = max(dates_dt).strftime("%Y-%m-%d")
    print(f"  {label:<35}  images: {count:>3}   earliest: {earliest}   latest: {latest}")
    return earliest, latest, count


# ---------------------------------------------------------------------------
# Section 1 — GEE collection status
# ---------------------------------------------------------------------------

print("=" * 65)
print(f"Collection: {COLLECTION}")
print("=" * 65)

today = datetime.date.today().isoformat()

ranges = [
    ("2024-01-01 to 2024-12-31",   "2024-01-01", "2025-01-01"),
    ("2025-01-01 to 2025-08-31",   "2025-01-01", "2025-09-01"),
    ("2025-09-01 to 2025-12-31",   "2025-09-01", "2026-01-01"),
    (f"2026-01-01 to {today}",     "2026-01-01", today),
]

all_latest = []
for label, start, end in ranges:
    _, latest, count = query_range(label, start, end)
    if latest:
        all_latest.append(latest)

# ---------------------------------------------------------------------------
# Section 2 — Most recent image details
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Most recent image details")
print("=" * 65)

# Query the last 3 years to find the most recent image globally
recent_col = (
    ee.ImageCollection(COLLECTION)
    .filterDate("2023-01-01", today)
    .sort("system:time_start", False)
    .limit(1)
)
recent_list = recent_col.toList(1)
recent_count = recent_col.size().getInfo()

if recent_count == 0:
    print("  No images found in the past 3 years.")
    most_recent_date = None
else:
    img = ee.Image(recent_list.get(0))
    img_id = img.id().getInfo()
    ts_ms  = img.get("system:time_start").getInfo()
    ts_dt  = datetime.datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
    props  = img.propertyNames().getInfo()

    print(f"  Image ID:            {img_id}")
    print(f"  system:time_start:   {ts_dt}")
    print(f"  Property keys:       {sorted(props)}")
    most_recent_date = ts_dt

# ---------------------------------------------------------------------------
# Section 3 — Summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Summary")
print("=" * 65)

if most_recent_date:
    print(f"  MCD64A1 status: data available through {most_recent_date}")
else:
    print("  MCD64A1 status: no recent data found")

# ---------------------------------------------------------------------------
# Section 4 — Existing burned area files in the Shiny app repo
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Existing burned area files (Shiny app repo)")
print("=" * 65)

import re
import numpy as np
import rasterio

BURNED_ROOT = Path(
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series\app\www\data\BurnedArea"
)

# Collect all TIFs recursively, group by immediate subfolder
all_tifs = sorted(BURNED_ROOT.rglob("*.tif")) if BURNED_ROOT.exists() else []

# Group by parent directory
from collections import defaultdict
by_dir = defaultdict(list)
for t in all_tifs:
    by_dir[t.parent].append(t)

if not all_tifs:
    print(f"  No TIF files found under:\n    {BURNED_ROOT}")
    print("  Conclusion: no legacy burned area files exist.")
else:
    for d, tifs in sorted(by_dir.items()):
        tifs = sorted(tifs)
        print(f"\n  Folder: {d}")
        print(f"  Files:  {len(tifs)} TIFs")

        if not tifs:
            print("  (folder is empty)")
            continue

        # Infer naming pattern from first file
        sample_name = tifs[0].name
        # Try to extract date token (YYYY-MM or YYYYMMDD pattern)
        date_tokens = re.findall(r"\d{4}-\d{2}|\d{8}", sample_name)
        pattern = re.sub(r"\d{4}-\d{2}|\d{8}", "<DATE>", sample_name) if date_tokens else sample_name
        print(f"  Naming pattern: {pattern}")

        # Earliest / latest by filename
        dates_found = []
        for f in tifs:
            tokens = re.findall(r"\d{4}-\d{2}|\d{8}", f.name)
            if tokens:
                dates_found.append(tokens[0])
        if dates_found:
            print(f"  Earliest file date: {min(dates_found)}")
            print(f"  Latest file date:   {max(dates_found)}")

        # Inspect the latest file
        latest_tif = tifs[-1]
        print(f"\n  Inspecting latest file: {latest_tif.name}")
        try:
            with rasterio.open(latest_tif) as src:
                data    = src.read(1).astype(float)
                nodata  = src.nodata
                crs     = src.crs
                shape   = src.shape
                dtype   = src.dtypes[0]

            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)

            valid    = ~np.isnan(data)
            nonzero  = np.sum((data != 0) & valid)

            print(f"    Shape:           {shape}")
            print(f"    Dtype:           {dtype}")
            print(f"    CRS:             {crs}")
            print(f"    Valid pixels:    {valid.sum():,}")
            print(f"    Non-zero pixels: {int(nonzero):,}  (burned area indicator)")
            print(f"    Min:             {np.nanmin(data):.4f}")
            print(f"    Max:             {np.nanmax(data):.4f}")
            print(f"    Mean (valid):    {np.nanmean(data):.4f}")
        except Exception as exc:
            print(f"    ERROR reading file: {exc}")

print()
print("Investigation complete.")
