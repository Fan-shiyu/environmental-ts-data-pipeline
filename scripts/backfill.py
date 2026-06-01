"""One-off historical backfill: run the pipeline over all 12 months of a given year."""

import argparse
import sys
from pathlib import Path

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.export import download_image
from pipeline.sentinel2 import load_aoi, load_polygon, monthly_composite as s2_monthly_composite

parser = argparse.ArgumentParser(description="Backfill monthly NDVI/BurnedArea composites for a full year.")
parser.add_argument("--aoi",        required=True, help="AoI key from config (e.g. Zambia)")
parser.add_argument("--sensor",     required=True, help="Sensor key: sentinel2, modis, burned_area")
parser.add_argument("--resolution", required=True, type=int, help="Resolution in metres (e.g. 100, 500)")
parser.add_argument("--year",       required=True, type=int, help="Year to backfill (e.g. 2024)")
args = parser.parse_args()

config = load_config()

if args.aoi not in config["aois"]:
    print(f"ERROR: AoI '{args.aoi}' not found in config. Available: {list(config['aois'].keys())}")
    sys.exit(1)

MODIS_RESOLUTIONS = {250, 500, 1000}
S2_RESOLUTIONS = {100, 1000}

if args.sensor == "sentinel2":
    if args.resolution not in S2_RESOLUTIONS:
        raise ValueError(
            f"Sentinel-2 resolution {args.resolution}m not supported. Supported: {sorted(S2_RESOLUTIONS)}"
        )
elif args.sensor == "modis":
    if args.resolution not in MODIS_RESOLUTIONS:
        raise ValueError(
            f"MODIS resolution {args.resolution}m not supported. Supported: {sorted(MODIS_RESOLUTIONS)}"
        )
    from pipeline.modis import monthly_composite as modis_monthly_composite
elif args.sensor == "burned_area":
    if args.resolution != 500:
        raise ValueError(
            f"Burned area resolution {args.resolution}m not supported. Only 500m is supported."
        )
    from pipeline.burned_area import monthly_image as ba_monthly_image
else:
    raise NotImplementedError(
        f"Sensor '{args.sensor}' is not supported. Supported: sentinel2, modis, burned_area"
    )

init_gee(config["project"])
aoi     = load_aoi(config["aois"][args.aoi]["path"])
polygon = load_polygon(config["aois"][args.aoi]["path"])

output_root = Path(config["output_root"]) / args.aoi / args.sensor / f"{args.resolution}m"
output_root.mkdir(parents=True, exist_ok=True)

n_downloaded = 0
n_skipped = 0
n_failed = 0
n_not_published = 0

for month in range(1, 13):
    label = f"{args.year}-{month:02d}"

    if args.sensor == "burned_area":
        filename = f"{label}_BurnedArea_{args.aoi}.tif"
    else:
        filename = f"{label}_NDVI_{args.aoi}.tif"

    output_path = output_root / filename

    if output_path.exists():
        print(f"[{label}] skipped (exists)")
        n_skipped += 1
        continue

    try:
        if args.sensor == "sentinel2":
            composite = s2_monthly_composite(aoi, args.year, month)
        elif args.sensor == "modis":
            composite = modis_monthly_composite(aoi, args.year, month, args.resolution)
        else:
            composite = ba_monthly_image(aoi, args.year, month)

        download_image(composite, aoi, str(output_path), scale=args.resolution, mask_polygon=polygon)
        print(f"[{label}] downloaded")
        n_downloaded += 1
    except ValueError as exc:
        # Burned area: month not yet published
        print(f"[{label}] not yet published")
        n_not_published += 1
    except Exception as exc:
        print(f"[{label}] FAILED: {exc}")
        n_failed += 1

summary = f"Summary: {n_downloaded} downloaded, {n_skipped} skipped, {n_failed} failed"
if n_not_published:
    summary += f", {n_not_published} not yet published"
print(f"\n{summary}")
