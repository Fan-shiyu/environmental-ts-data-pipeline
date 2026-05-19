"""
Deploy pipeline outputs to the Shiny app's www/data/ folder.

The pipeline writes to outputs/{aoi}/{sensor}/{res}m/.
The app reads from www/data/NDVI/{aoi}/{res_key}m_resolution/ (NDVI)
                    www/data/BurnedArea/{aoi}/500m_resolution/ (burned area).

Run with:
    python scripts/deploy.py --dry-run
    python scripts/deploy.py --stage sentinel2
    python scripts/deploy.py --stage all
"""

import argparse
import shutil
import sys
from pathlib import Path

import rasterio

from pipeline.config import load_config

APP_DATA_ROOT = Path(
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series\app\www\data"
)

AOIS = ["Zambia_Mponda", "Zambia_WL"]

# Maps (sensor, resolution_metres) -> app resolution folder key
_RES_KEY = {
    ("sentinel2",    100): "100",
    ("sentinel2",   1000): "Sentinel_1000",
    ("modis",        250): "250",
    ("modis",        500): "500",
    ("modis",       1000): "MODIS_1000",
    ("burned_area",  500): "500",
}


def get_deploy_mapping(config: dict) -> list[dict]:
    """Return list of mapping dicts for all sensor/resolution/aoi combos."""
    output_root = Path(config["output_root"])
    mappings = []

    for aoi in AOIS:
        for sensor in ("sentinel2", "modis", "burned_area"):
            if sensor == "sentinel2":
                resolutions = config["sensors"]["sentinel2"]["resolutions"]
            elif sensor == "modis":
                resolutions = list(config["sensors"]["modis"]["resolutions"].keys())
            else:
                resolutions = config["sensors"]["burned_area"]["resolutions"]

            for res in resolutions:
                res_key = _RES_KEY.get((sensor, res))
                if res_key is None:
                    continue

                source = output_root / aoi / sensor / f"{res}m"

                if sensor == "burned_area":
                    dest = APP_DATA_ROOT / "BurnedArea" / aoi / f"{res_key}m_resolution"
                else:
                    dest = APP_DATA_ROOT / "NDVI" / aoi / f"{res_key}m_resolution"

                mappings.append({
                    "aoi": aoi,
                    "sensor": sensor,
                    "resolution": res,
                    "source": source,
                    "dest": dest,
                })

    return mappings


def _validate_tif(path: Path) -> bool:
    """Return True if file is non-empty and openable with rasterio."""
    if path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path):
            pass
        return True
    except Exception:
        return False


def deploy_stage(stage: str, aoi: str, config: dict, dry_run: bool = False) -> dict:
    """Copy .tif files from outputs/ into the app's www/data/ folder.

    Returns dict with counts: copied, skipped, failed, failed_paths.
    """
    result = {"copied": 0, "skipped": 0, "failed": 0, "failed_paths": []}
    mappings = [
        m for m in get_deploy_mapping(config)
        if m["aoi"] == aoi and (stage == "all" or m["sensor"] == stage)
    ]

    for m in mappings:
        source_dir: Path = m["source"]
        dest_dir: Path   = m["dest"]

        if not source_dir.exists():
            print(f"  [skip] source missing: {source_dir}")
            continue

        tif_files = sorted(source_dir.glob("*.tif"))
        if not tif_files:
            print(f"  [skip] no .tif files in: {source_dir}")
            continue

        print(f"\n  {m['sensor']} {m['resolution']}m  {source_dir.name} -> {dest_dir}")

        for src in tif_files:
            dst = dest_dir / src.name

            if dst.exists():
                result["skipped"] += 1
                if dry_run:
                    print(f"    would skip (exists): {src.name}")
                continue

            if not _validate_tif(src):
                print(f"    ERROR: invalid source file: {src.name}")
                result["failed"] += 1
                result["failed_paths"].append(str(src))
                continue

            if dry_run:
                print(f"    would copy: {src.name}")
                result["copied"] += 1
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                if dst.stat().st_size != src.stat().st_size:
                    print(f"    ERROR: size mismatch after copy: {src.name}")
                    dst.unlink(missing_ok=True)
                    result["failed"] += 1
                    result["failed_paths"].append(str(src))
                else:
                    result["copied"] += 1
            except Exception as exc:
                print(f"    ERROR copying {src.name}: {exc}")
                result["failed"] += 1
                result["failed_paths"].append(str(src))

    return result


def deploy_all(config: dict, dry_run: bool = False) -> None:
    """Deploy all stages for all AoIs. Prints a summary table."""
    if not APP_DATA_ROOT.exists():
        print(f"ERROR: app data root not found: {APP_DATA_ROOT}")
        sys.exit(1)

    totals = {"copied": 0, "skipped": 0, "failed": 0, "failed_paths": []}

    print(f"{'Stage':<15} {'AoI':<18} {'Copied':>7} {'Skipped':>8} {'Failed':>7}")
    print("-" * 60)

    for stage in ("sentinel2", "modis", "burned_area"):
        for aoi in AOIS:
            r = deploy_stage(stage, aoi, config, dry_run=dry_run)
            mode = "(dry)" if dry_run else ""
            print(
                f"{stage:<15} {aoi:<18} {r['copied']:>7} {r['skipped']:>8}"
                f" {r['failed']:>7}  {mode}"
            )
            for k in ("copied", "skipped", "failed"):
                totals[k] += r[k]
            totals["failed_paths"].extend(r["failed_paths"])

    print("-" * 60)
    print(
        f"{'TOTAL':<15} {'':<18} {totals['copied']:>7} {totals['skipped']:>8}"
        f" {totals['failed']:>7}"
    )

    if totals["failed_paths"]:
        print("\nFailed files:")
        for p in totals["failed_paths"]:
            print(f"  {p}")


# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Deploy pipeline outputs to Shiny app data folder.")
parser.add_argument("--stage",   default="all", help="sentinel2 | modis | burned_area | all")
parser.add_argument("--aoi",     default="all", help="Zambia_Mponda | Zambia_WL | all")
parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without doing it")
args = parser.parse_args()

config = load_config()

if not APP_DATA_ROOT.exists():
    print(f"ERROR: app data root not found: {APP_DATA_ROOT}")
    sys.exit(1)

mode_label = "DRY RUN -- " if args.dry_run else ""
print(f"{mode_label}Deploying stage={args.stage}  aoi={args.aoi}")
print(f"Source root: {config['output_root']}")
print(f"Dest root:   {APP_DATA_ROOT}")

if args.aoi == "all" and args.stage == "all":
    deploy_all(config, dry_run=args.dry_run)
else:
    aois_to_run = AOIS if args.aoi == "all" else [args.aoi]
    stages_to_run = ["sentinel2", "modis", "burned_area"] if args.stage == "all" else [args.stage]

    for stage in stages_to_run:
        for aoi in aois_to_run:
            r = deploy_stage(stage, aoi, config, dry_run=args.dry_run)
            mode = "(dry)" if args.dry_run else ""
            print(
                f"\n{stage} / {aoi}: copied={r['copied']} skipped={r['skipped']}"
                f" failed={r['failed']} {mode}"
            )
            for p in r["failed_paths"]:
                print(f"  FAILED: {p}")
