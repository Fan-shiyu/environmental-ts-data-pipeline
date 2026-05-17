"""One-off historical backfill: run the pipeline over all 12 months of a given year."""

import argparse
import sys
from pathlib import Path

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.export import download_image
from pipeline.sentinel2 import load_aoi, monthly_composite

parser = argparse.ArgumentParser(description="Backfill monthly NDVI composites for a full year.")
parser.add_argument("--aoi",        required=True, help="AoI key from config (e.g. Zambia)")
parser.add_argument("--sensor",     required=True, help="Sensor key from config (e.g. sentinel2)")
parser.add_argument("--resolution", required=True, type=int, help="Resolution in metres (e.g. 100)")
parser.add_argument("--year",       required=True, type=int, help="Year to backfill (e.g. 2024)")
args = parser.parse_args()

config = load_config()

if args.aoi not in config["aois"]:
    print(f"ERROR: AoI '{args.aoi}' not found in config. Available: {list(config['aois'].keys())}")
    sys.exit(1)

if args.sensor != "sentinel2":
    raise NotImplementedError(
        f"Sensor '{args.sensor}' is not yet supported. Only 'sentinel2' is implemented."
    )

if args.resolution != 100:
    raise NotImplementedError(
        f"Resolution {args.resolution}m is not yet supported. Only 100m is implemented."
    )

init_gee(config["project"])
aoi = load_aoi(config["aois"][args.aoi]["path"])

output_root = Path(config["output_root"]) / args.aoi / args.sensor / f"{args.resolution}m"
output_root.mkdir(parents=True, exist_ok=True)

n_downloaded = 0
n_skipped = 0
n_failed = 0

for month in range(1, 13):
    label = f"{args.year}-{month:02d}"
    output_path = output_root / f"{label}_NDVI_{args.aoi}.tif"

    if output_path.exists():
        print(f"[{label}] skipped (exists)")
        n_skipped += 1
        continue

    try:
        composite = monthly_composite(aoi, args.year, month)
        download_image(composite, aoi, str(output_path), scale=args.resolution)
        print(f"[{label}] downloaded")
        n_downloaded += 1
    except Exception as exc:
        print(f"[{label}] FAILED: {exc}")
        n_failed += 1

print(f"\nSummary: {n_downloaded} downloaded, {n_skipped} skipped, {n_failed} failed")
