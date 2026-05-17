"""
Validation script: compare a freshly generated GEE composite against the
corresponding legacy TIF in the environmental-time-series Shiny app repo.

Run with:
    python scripts/test_mponda_validation.py --year 2024 --month 3
    python scripts/test_mponda_validation.py --year 2024 --month 3 --aoi Zambia --resolution 100
    python scripts/test_mponda_validation.py --year 2024 --month 3 --aoi Zambia --resolution 1000
    python scripts/test_mponda_validation.py --year 2024 --month 3 --aoi Zambia_WL --resolution 100
    python scripts/test_mponda_validation.py --sensor modis --resolution 250 --aoi Zambia --year 2024 --month 3
    python scripts/test_mponda_validation.py --sensor modis --resolution 500 --aoi Zambia --year 2024 --month 3
    python scripts/test_mponda_validation.py --sensor modis --resolution 1000 --aoi Zambia --year 2024 --month 3
    python scripts/test_mponda_validation.py --sensor burned_area --resolution 500 --aoi Zambia --year 2024 --month 8
    python scripts/test_mponda_validation.py --sensor burned_area --resolution 500 --aoi Zambia_WL --year 2024 --month 8
"""

import argparse
import sys
from pathlib import Path

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.export import download_image
from pipeline.sentinel2 import load_aoi
from pipeline.sentinel2 import monthly_composite as s2_monthly_composite
from pipeline.validate import (
    compare_burned_area,
    compare_rasters,
    print_burned_area_comparison,
    print_comparison,
    print_smoke_test,
    smoke_test,
)

parser = argparse.ArgumentParser(description="Validate GEE composite against legacy TIF.")
parser.add_argument("--year",       type=int, required=True,           help="Year to validate")
parser.add_argument("--month",      type=int, required=True,           help="Month to validate (1-12)")
parser.add_argument("--aoi",        default="Zambia",                  help="AoI key from config (default: Zambia)")
parser.add_argument("--resolution", type=int, default=100,             help="Resolution in metres (default: 100)")
parser.add_argument("--sensor",     default="sentinel2",               help="Sensor: sentinel2, modis, burned_area (default: sentinel2)")
args = parser.parse_args()

YEAR       = args.year
MONTH      = args.month
AOI        = args.aoi
RESOLUTION = args.resolution
SENSOR     = args.sensor
LABEL      = f"{YEAR}-{MONTH:02d}"

print("=" * 60)
print(f"Validating composite: {LABEL}  AoI={AOI}  sensor={SENSOR}  resolution={RESOLUTION}m")
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
print(f"Step 3: Building {SENSOR} composite ({LABEL})")
print("=" * 60)

if SENSOR == "sentinel2":
    composite = s2_monthly_composite(aoi, YEAR, MONTH)
elif SENSOR == "modis":
    from pipeline.modis import monthly_composite as modis_monthly_composite
    composite = modis_monthly_composite(aoi, YEAR, MONTH, RESOLUTION)
elif SENSOR == "burned_area":
    from pipeline.burned_area import monthly_image as ba_monthly_image
    try:
        composite = ba_monthly_image(aoi, YEAR, MONTH)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
else:
    print(f"ERROR: Unknown sensor '{SENSOR}'. Supported: sentinel2, modis, burned_area")
    sys.exit(1)

print("=" * 60)
print("Step 4: Downloading composite")
print("=" * 60)

if SENSOR == "burned_area":
    output_path = f"test_outputs/test_{LABEL}_BurnedArea_{AOI}_{RESOLUTION}m_burned_area_v2.tif"
else:
    output_path = f"test_outputs/test_{LABEL}_NDVI_{AOI}_{RESOLUTION}m_{SENSOR}_v2.tif"

try:
    download_image(composite, aoi, output_path, scale=RESOLUTION)
except RuntimeError as exc:
    print(f"\nWARNING: {exc}")
    sys.exit(1)

print("=" * 60)
print("Step 5: Comparing rasters")
print("=" * 60)

legacy_root = Path(config["legacy_data_root"])
legacy_burned_area_root = Path(config["legacy_burned_area_root"])
legacy_path = None

if SENSOR == "sentinel2":
    if AOI == "Zambia":
        legacy_path = legacy_root / "Zambia" / f"{RESOLUTION}m_resolution" / f"{LABEL}_NDVI_Zambia.tif"

elif SENSOR == "modis":
    modis_mponda_aois = {"Zambia", "Zambia_Mponda"}
    if AOI in modis_mponda_aois:
        if RESOLUTION == 250:
            legacy_path = legacy_root / "Zambia_Mponda" / "250m_resolution" / f"{LABEL}_NDVI_Zambia_Mponda.tif"
        elif RESOLUTION == 500:
            legacy_path = legacy_root / "Zambia_Mponda" / "500m_resolution" / f"{LABEL}_NDVI_Zambia_Mponda.tif"
        elif RESOLUTION == 1000:
            legacy_path = legacy_root / "Zambia_Mponda" / "MODIS_1000m_resolution" / f"{LABEL}_NDVI_Zambia_Mponda.tif"

elif SENSOR == "burned_area":
    if AOI in {"Zambia", "Zambia_Mponda"}:
        legacy_path = (
            legacy_burned_area_root / "Zambia_Mponda" / "500m_resolution"
            / f"{LABEL}_BurnedArea_Zambia_Mponda.tif"
        )
    elif AOI == "Zambia_WL":
        legacy_path = (
            legacy_burned_area_root / "Zambia_WL" / "500m_resolution"
            / f"{LABEL}_BurnedArea_Zambia_WL.tif"
        )

if legacy_path is not None and legacy_path.exists():
    if SENSOR == "burned_area":
        stats = compare_burned_area(output_path, str(legacy_path))
        print_burned_area_comparison(stats)
    else:
        stats = compare_rasters(output_path, str(legacy_path))
        print_comparison(stats)
else:
    if legacy_path is not None:
        print(f"  Legacy file not found at: {legacy_path}")
        print("  Switching to smoke-test mode.")
    else:
        print(f"  No legacy file defined for AoI '{AOI}' + sensor '{SENSOR}' + {RESOLUTION}m — smoke-test mode.")
    stats = smoke_test(output_path)
    print_smoke_test(stats)

print(f"\nDone. {LABEL}  AoI={AOI}  sensor={SENSOR}  resolution={RESOLUTION}m")
