"""
Validation script: compare a freshly generated GEE NDVI composite against
the corresponding legacy TIF in the environmental-time-series Shiny app repo.

Run with:
    python scripts/test_mponda_validation.py --year 2024 --month 3
    python scripts/test_mponda_validation.py --year 2024 --month 3 --aoi Zambia --resolution 100
    python scripts/test_mponda_validation.py --year 2024 --month 3 --aoi Zambia --resolution 1000
    python scripts/test_mponda_validation.py --year 2024 --month 3 --aoi Zambia_WL --resolution 100
"""

import argparse
import sys
from pathlib import Path

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.export import download_image
from pipeline.sentinel2 import load_aoi, monthly_composite
from pipeline.validate import compare_rasters, print_comparison, print_smoke_test, smoke_test

parser = argparse.ArgumentParser(description="Validate GEE NDVI composite against legacy TIF.")
parser.add_argument("--year",       type=int, required=True,          help="Year to validate")
parser.add_argument("--month",      type=int, required=True,          help="Month to validate (1-12)")
parser.add_argument("--aoi",        default="Zambia",                 help="AoI key from config (default: Zambia)")
parser.add_argument("--resolution", type=int, default=100,            help="Resolution in metres (default: 100)")
args = parser.parse_args()

YEAR       = args.year
MONTH      = args.month
AOI        = args.aoi
RESOLUTION = args.resolution
LABEL      = f"{YEAR}-{MONTH:02d}"

print("=" * 60)
print(f"Validating NDVI composite: {LABEL}  AoI={AOI}  resolution={RESOLUTION}m")
print("=" * 60)

config = load_config()

if AOI not in config["aois"]:
    print(f"ERROR: AoI '{AOI}' not in config. Available: {list(config['aois'].keys())}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 1: Authenticating to Google Earth Engine")
print("=" * 60)
init_gee(config["project"])

print("=" * 60)
print("Step 2: Loading AoI")
print("=" * 60)
aoi = load_aoi(config["aois"][AOI]["path"])

print("=" * 60)
print(f"Step 3: Building Sentinel-2 NDVI composite ({LABEL})")
print("=" * 60)
composite = monthly_composite(aoi, YEAR, MONTH)

print("=" * 60)
print("Step 4: Downloading NDVI composite")
print("=" * 60)
output_path = f"test_outputs/test_{LABEL}_NDVI_{AOI}_{RESOLUTION}m_v2.tif"
try:
    download_image(composite, aoi, output_path, scale=RESOLUTION)
except RuntimeError as exc:
    print(f"\nWARNING: {exc}")
    sys.exit(1)

print("=" * 60)
print("Step 5: Comparing rasters")
print("=" * 60)

# Resolve the legacy file path based on AoI + resolution.
# Zambia_WL has no legacy file — fall back to smoke test.
legacy_root = Path(config["legacy_data_root"])

if AOI == "Zambia":
    legacy_path = legacy_root / "Zambia" / f"{RESOLUTION}m_resolution" / f"{LABEL}_NDVI_Zambia.tif"
else:
    legacy_path = None  # No legacy file for WL or other new AoIs

if legacy_path is not None and legacy_path.exists():
    stats = compare_rasters(output_path, str(legacy_path))
    print_comparison(stats)
else:
    if legacy_path is not None:
        print(f"  Legacy file not found at: {legacy_path}")
        print("  Switching to smoke-test mode.")
    else:
        print(f"  No legacy file defined for AoI '{AOI}' — smoke-test mode.")
    stats = smoke_test(output_path)
    print_smoke_test(stats)

print(f"\nDone. {LABEL}  AoI={AOI}  resolution={RESOLUTION}m")
