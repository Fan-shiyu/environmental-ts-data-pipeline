"""
Validation script: compare a freshly generated GEE NDVI composite against
the corresponding legacy TIF in the environmental-time-series Shiny app repo.

Run with:
    python scripts/test_mponda_validation.py --year 2024 --month 3
    python scripts/test_mponda_validation.py --year 2020 --month 9
"""

import argparse
import sys
from pathlib import Path

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.export import download_image
from pipeline.sentinel2 import load_aoi, monthly_composite
from pipeline.validate import compare_rasters, print_comparison

parser = argparse.ArgumentParser(description="Validate GEE NDVI composite against legacy TIF.")
parser.add_argument("--year",  type=int, required=True, help="Year to validate (e.g. 2024)")
parser.add_argument("--month", type=int, required=True, help="Month to validate (1-12)")
args = parser.parse_args()

YEAR  = args.year
MONTH = args.month
LABEL = f"{YEAR}-{MONTH:02d}"

print("=" * 60)
print(f"Validating NDVI composite: {LABEL}")
print("=" * 60)

config = load_config()

print("\n" + "=" * 60)
print("Step 1: Authenticating to Google Earth Engine")
print("=" * 60)
init_gee(config["project"])

print("=" * 60)
print("Step 2: Loading AoI")
print("=" * 60)
aoi = load_aoi(config["aois"]["Zambia"]["path"])

print("=" * 60)
print(f"Step 3: Building Sentinel-2 NDVI composite ({LABEL})")
print("=" * 60)
composite = monthly_composite(aoi, YEAR, MONTH)

print("=" * 60)
print("Step 4: Downloading NDVI composite")
print("=" * 60)
output_path = f"test_outputs/test_{LABEL}_NDVI_Zambia_100m_v2.tif"
try:
    download_image(composite, aoi, output_path)
except RuntimeError as exc:
    print(f"\nWARNING: {exc}")
    sys.exit(1)

print("=" * 60)
print("Step 5: Comparing rasters")
print("=" * 60)
legacy_path = Path(config["legacy_data_root"]) / "Zambia" / "100m_resolution" / f"{LABEL}_NDVI_Zambia.tif"
if not legacy_path.exists():
    print(f"ERROR: Legacy TIF not found at:\n  {legacy_path}")
    sys.exit(1)

stats = compare_rasters(output_path, str(legacy_path))
print_comparison(stats)

print(f"\nDone. Interpret numbers above for {LABEL}.")
