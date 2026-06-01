"""
One-off investigation: test two alternative explanations for why the new
burned area pipeline finds ~928 more burned pixels than the legacy (Zambia_WL,
August 2024).

Hypotheses tested:
  1. Reprocessing timestamp — GEE asset produced after legacy file was written
  2. FirstDay/LastDay reliable-window filter — legacy kept only BurnDate within
     the [FirstDay, LastDay] reliable change-detection window

Previous investigation (check_burned_area_qa.py) ruled out:
  - QA threshold
  - AoI boundary clipping
  - Reprojection / CRS mismatch
  - Minimum mapping unit (cluster size filter)

Run with:
    python scripts/check_burned_area_reliability.py
"""

import datetime
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.sentinel2 import load_aoi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AOI_KEY     = "Zambia_WL"
YEAR        = 2024
MONTH       = 8
SCALE       = 500
CRS         = "EPSG:4326"
LEGACY_PATH = Path(
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series"
    r"\app\www\data\BurnedArea\Zambia_WL\500m_resolution"
    r"\2024-08_BurnedArea_Zambia_WL.tif"
)

# ---------------------------------------------------------------------------
# Step 1 — Fetch MCD64A1 with ALL bands + image metadata
# ---------------------------------------------------------------------------

print("=" * 65)
print("Step 1 — Authenticate, load AoI, fetch MCD64A1 all-band image")
print("=" * 65)

config = load_config()
init_gee(config["project"])
aoi = load_aoi(config["aois"][AOI_KEY]["path"])

import ee

collection = (
    ee.ImageCollection("MODIS/061/MCD64A1")
    .filterBounds(aoi)
    .filterDate("2024-08-01", "2024-09-01")
)

img = collection.first()

print("\nImage metadata properties:")
props = img.toDictionary().getInfo()
time_keys = [k for k in props if any(t in k.lower() for t in
             ["time", "date", "version", "produc", "creat", "index", "size"])]
for k in sorted(time_keys):
    v = props[k]
    if isinstance(v, (int, float)) and v > 1e12:
        ts = datetime.datetime.utcfromtimestamp(v / 1000).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  {k}: {v}  ({ts})")
    else:
        print(f"  {k}: {v}")

# Also print all properties for completeness
print("\nAll image properties:")
for k in sorted(props):
    if k not in time_keys:
        v = props[k]
        if isinstance(v, (int, float)) and abs(v) > 1e12:
            ts = datetime.datetime.utcfromtimestamp(v / 1000).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"  {k}: {v}  ({ts})")
        else:
            print(f"  {k}: {v}")

# Download 4-band image
print("\nDownloading 4-band image (BurnDate, QA, FirstDay, LastDay) ...")
four_band = img.select(["BurnDate", "QA", "FirstDay", "LastDay"]).clip(aoi)
url = four_band.getDownloadURL({
    "scale":  SCALE,
    "crs":    CRS,
    "region": aoi,
    "format": "GEO_TIFF",
})
resp = requests.get(url, timeout=300)
resp.raise_for_status()
raw = resp.content

tmp_dir = Path("test_outputs")
tmp_dir.mkdir(exist_ok=True)
tmp_path = tmp_dir / "check_reliability_4band.tif"

if zipfile.is_zipfile(io.BytesIO(raw)):
    print("  ZIP response — extracting ...")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        tif_names = sorted(n for n in zf.namelist() if n.lower().endswith(".tif"))
        print(f"  TIFs in ZIP: {tif_names}")
        # Read each band by filename
        band_data = {}
        for name in tif_names:
            data_bytes = zf.read(name)
            with rasterio.open(io.BytesIO(data_bytes)) as src:
                band_data[Path(name).stem.split(".")[-1]] = src.read(1).astype(np.int32)
                if "transform" not in dir(locals()):
                    ref_transform = src.transform
                    ref_crs = src.crs
                    ref_shape = src.shape
                    ref_nodata = src.nodata
        burndate = band_data.get("BurnDate", list(band_data.values())[0])
        qa       = band_data.get("QA",        list(band_data.values())[1])
        firstday = band_data.get("FirstDay",   list(band_data.values())[2])
        lastday  = band_data.get("LastDay",    list(band_data.values())[3])
else:
    tmp_path.write_bytes(raw)
    with rasterio.open(tmp_path) as src:
        print(f"  Multi-band GeoTIFF: {src.count} bands, dtypes={src.dtypes}")
        burndate = src.read(1).astype(np.int32)
        qa       = src.read(2).astype(np.int32)
        firstday = src.read(3).astype(np.int32)
        lastday  = src.read(4).astype(np.int32)
        ref_transform = src.transform
        ref_crs   = src.crs
        ref_shape = src.shape
        ref_nodata = src.nodata

print(f"  Downloaded. Shape={ref_shape}, CRS={ref_crs}, nodata={ref_nodata}")
print(f"  BurnDate range: {burndate.min()} to {burndate.max()}")
print(f"  FirstDay range: {firstday.min()} to {firstday.max()}")
print(f"  LastDay range:  {lastday.min()} to {lastday.max()}")

# Normalise fill values
fill = int(ref_nodata) if ref_nodata is not None else -32768
for arr in [burndate, qa, firstday, lastday]:
    arr[arr == fill] = 0
    arr[arr < 0] = 0

# ---------------------------------------------------------------------------
# Step 2 — Reprocessing timestamp hypothesis
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Step 2 — Reprocessing timestamp hypothesis")
print("=" * 65)

asset_id = props.get("system:index", "")
print(f"\n  Asset ID (system:index): {asset_id!r}")

# Parse production timestamp from asset ID
# Format example: MCD64A1.A2024214.h20v09.061.2024291153442
# The last 13-digit numeric segment is YYYYDDDHHmmss where DDD = Julian day
gee_production_dt = None
parts = asset_id.replace(".", "_").split("_")
for part in reversed(parts):
    if len(part) == 13 and part.isdigit():
        try:
            yr  = int(part[:4])
            doy = int(part[4:7])
            dt  = datetime.datetime(yr, 1, 1) + datetime.timedelta(days=doy - 1)
            gee_production_dt = dt
            print(f"  Production timestamp parsed from asset ID segment '{part}'")
            print(f"    Year={yr}, DOY={doy} => {dt.strftime('%Y-%m-%d')}")
        except Exception:
            pass
        break

# Fallback 1: google:max_source_file_timestamp (production/processing date)
if gee_production_dt is None:
    ts = props.get("google:max_source_file_timestamp")
    if ts:
        gee_production_dt = datetime.datetime.utcfromtimestamp(ts / 1000)
        print(f"  Production timestamp from 'google:max_source_file_timestamp': "
              f"{gee_production_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

# Fallback 2: system:time_start (acquisition date, not production date)
if gee_production_dt is None:
    ts = props.get("system:time_start")
    if ts:
        gee_production_dt = datetime.datetime.utcfromtimestamp(ts / 1000)
        print(f"  (Using system:time_start as last resort: {gee_production_dt.strftime('%Y-%m-%d')})")

legacy_mtime = os.path.getmtime(str(LEGACY_PATH))
legacy_dt    = datetime.datetime.utcfromtimestamp(legacy_mtime)
print(f"\n  Legacy file last-modified: {legacy_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

if gee_production_dt:
    delta_days  = (gee_production_dt - legacy_dt).days
    delta_months = abs(delta_days) / 30.44
    direction   = "newer" if delta_days > 0 else "older"
    print(f"  GEE asset produced:        {gee_production_dt.strftime('%Y-%m-%d')}")
    print(f"  GEE asset is {direction} than legacy file by ~{delta_months:.1f} months ({abs(delta_days)} days)")
    if delta_days > 90:
        print("  => SUPPORTS reprocessing hypothesis (GEE asset materially newer).")
    elif delta_days < -90:
        print("  => WEAKENS reprocessing hypothesis (GEE asset is OLDER than legacy).")
    else:
        print("  => INCONCLUSIVE (within 3 months — no clear signal).")

# ---------------------------------------------------------------------------
# Open legacy file
# ---------------------------------------------------------------------------

with rasterio.open(LEGACY_PATH) as src:
    legacy_data  = src.read(1).astype(np.int32)
    legacy_nodata = src.nodata
if legacy_nodata is not None:
    legacy_data[legacy_data == int(legacy_nodata)] = 0
legacy_data[legacy_data < 0] = 0

# Burned masks
burned_new    = burndate > 0
burned_legacy = legacy_data > 0
overlap       = burned_new & burned_legacy
only_in_new   = burned_new & ~burned_legacy

print(f"\n  Burned in new:    {burned_new.sum():,}")
print(f"  Burned in legacy: {burned_legacy.sum():,}")
print(f"  Overlap:          {overlap.sum():,}")
print(f"  Only in new:      {only_in_new.sum():,}")

# ---------------------------------------------------------------------------
# Step 3 — FirstDay / LastDay reliable-window hypothesis
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Step 3 — FirstDay/LastDay reliable-window hypothesis")
print("=" * 65)

# For burned pixels, check FirstDay <= BurnDate <= LastDay
rows_b, cols_b = np.where(burned_new)

in_window_overlap  = 0
in_window_only_new = 0
out_window_overlap  = 0
out_window_only_new = 0

for r, c in zip(rows_b, cols_b):
    bd = int(burndate[r, c])
    fd = int(firstday[r, c])
    ld = int(lastday[r, c])
    in_win = (fd > 0 and ld > 0 and fd <= bd <= ld)
    in_leg = bool(overlap[r, c])
    if in_win:
        if in_leg:
            in_window_overlap += 1
        else:
            in_window_only_new += 1
    else:
        if in_leg:
            out_window_overlap += 1
        else:
            out_window_only_new += 1

in_window_total  = in_window_overlap  + in_window_only_new
out_window_total = out_window_overlap + out_window_only_new

print(f"\n  {'In-window?':<25}  {'Total burned':>14}  {'In legacy':>12}  {'Only in new':>12}")
print(f"  {'-'*65}")
print(f"  {'Yes (FD <= BD <= LD)':<25}  {in_window_total:>14,}  {in_window_overlap:>12,}  {in_window_only_new:>12,}")
print(f"  {'No':<25}  {out_window_total:>14,}  {out_window_overlap:>12,}  {out_window_only_new:>12,}")

# Compute fractions
def pct(a, b):
    return f"{100*a/b:.1f}%" if b > 0 else "N/A"

print()
print("  Interpretation:")
print(f"  Of burned pixels IN the window:")
print(f"    {pct(in_window_overlap, in_window_total)} also in legacy, {pct(in_window_only_new, in_window_total)} only in new")
print(f"  Of burned pixels OUTSIDE the window:")
print(f"    {pct(out_window_overlap, out_window_total)} also in legacy, {pct(out_window_only_new, out_window_total)} only in new")

# Check how many of the 928 "only in new" are out-of-window
n_only = int(only_in_new.sum())
print()
if out_window_only_new > 0:
    print(f"  Of {n_only} 'only in new' pixels:")
    print(f"    In-window:     {in_window_only_new:,}  ({pct(in_window_only_new, n_only)})")
    print(f"    Out-of-window: {out_window_only_new:,}  ({pct(out_window_only_new, n_only)})")

    if out_window_only_new / n_only > 0.7:
        print("\n  => STRONG WINDOW HYPOTHESIS: majority of extra pixels are out-of-window.")
        print("     Legacy likely filtered to BurnDate within [FirstDay, LastDay].")
    elif out_window_only_new / n_only > 0.3:
        print("\n  => PARTIAL WINDOW EFFECT: some extra pixels are out-of-window, but not all.")
    else:
        print("\n  => WEAK WINDOW HYPOTHESIS: most extra pixels ARE in-window.")
        print("     Reliable-window filter does not explain the gap.")

# Also check overlap pixels: what fraction are in-window?
n_overlap = int(overlap.sum())
print()
print(f"  Of {n_overlap} overlap pixels (in both): {pct(in_window_overlap, n_overlap)} are in-window")

# ---------------------------------------------------------------------------
# Step 4 — Representative pixel samples
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Step 4 — Representative pixel samples")
print("=" * 65)

def print_sample(label, mask, n=8):
    r_arr, c_arr = np.where(mask)
    if len(r_arr) == 0:
        print(f"\n  {label}: (none)")
        return
    idx = np.linspace(0, len(r_arr) - 1, min(n, len(r_arr)), dtype=int)
    print(f"\n  {label} ({min(n, len(r_arr))} of {len(r_arr):,}):")
    print(f"  {'row':>5}  {'col':>5}  {'BurnDate':>10}  {'QA':>4}  {'FirstDay':>10}  {'LastDay':>9}  {'in_window':>10}")
    for i in idx:
        r, c = int(r_arr[i]), int(c_arr[i])
        bd, q = int(burndate[r, c]), int(qa[r, c])
        fd, ld = int(firstday[r, c]), int(lastday[r, c])
        in_win = "YES" if (fd > 0 and ld > 0 and fd <= bd <= ld) else "NO"
        print(f"  {r:>5}  {c:>5}  {bd:>10}  {q:>4}  {fd:>10}  {ld:>9}  {in_win:>10}")

print_sample("'Only in new' pixels (burned in new, not in legacy)", only_in_new)
print_sample("Overlap pixels (burned in both new and legacy)", overlap)

# ---------------------------------------------------------------------------
# Step 5 — Conclusion
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Step 5 — Conclusion and recommendation")
print("=" * 65)

print()
print("  Summary of findings:")
if gee_production_dt and delta_days > 90:
    print(f"  A. Reprocessing: GEE asset produced ~{delta_months:.1f} months AFTER legacy file.")
    print("     Supports the reprocessing hypothesis.")
elif gee_production_dt and delta_days < -90:
    print(f"  A. Reprocessing: GEE asset is ~{delta_months:.1f} months OLDER than legacy file.")
    print("     Weakens the reprocessing hypothesis.")
else:
    print("  A. Reprocessing: no clear temporal gap between GEE asset and legacy file.")

if out_window_only_new / max(n_only, 1) > 0.7:
    print(f"\n  B. Reliable-window filter: {pct(out_window_only_new, n_only)} of extra pixels")
    print(f"     are OUTSIDE the [FirstDay, LastDay] window. This is the dominant signal.")
    print("\n  RECOMMENDATION: A — Add reliable-window filter to match legacy.")
    print("  In pipeline/burned_area.py, mask pixels where BurnDate < FirstDay")
    print("  or BurnDate > LastDay. This is a defensible quality filter and")
    print("  would bring the new output into alignment with the legacy.")
elif in_window_only_new / max(n_only, 1) > 0.7:
    print(f"\n  B. Reliable-window filter: {pct(in_window_only_new, n_only)} of extra pixels")
    print(f"     are INSIDE the window — the filter does NOT explain the gap.")
    if gee_production_dt and delta_days > 90:
        print("\n  RECOMMENDATION: B — Keep current behavior, document as data refresh.")
        print("  The GEE asset is newer than the legacy file, and the extra pixels")
        print("  pass the reliable-window check. The gap reflects updated product data.")
    else:
        print("\n  RECOMMENDATION: C — Inconclusive. Neither hypothesis explains the gap cleanly.")
        print("  Suggest: compare GEE Collection 6 vs 6.1 for this tile/month.")
else:
    print(f"\n  B. Reliable-window filter: mixed signal.")
    print(f"     {pct(out_window_only_new, n_only)} of extra pixels are out-of-window.")
    print("\n  RECOMMENDATION: C — Inconclusive. Both hypotheses have partial support.")
    print("  Suggest: check GEE asset provenance (Collection 6 vs 6.1) and whether")
    print("  applying a window filter reduces the gap to near-zero.")

print()
print("Investigation complete.")
