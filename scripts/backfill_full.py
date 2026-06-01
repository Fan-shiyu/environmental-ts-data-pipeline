"""
Full historical backfill: generate all monthly composites from sensor start dates
to the latest available GEE data.

Three stages (run sequentially, one at a time recommended):
  sentinel2    -- 2019-01 to present, 100m + 1000m, Zambia_Mponda + Zambia_WL
  modis        -- 2000-02 to present, 250m + 500m + 1000m, both AoIs
  burned_area  -- 2000-11 to present, 500m, both AoIs

Run with:
    python scripts/backfill_full.py --stage sentinel2 --dry-run
    python scripts/backfill_full.py --stage sentinel2
    python scripts/backfill_full.py --stage modis
    python scripts/backfill_full.py --stage burned_area
    python scripts/backfill_full.py --stage all
"""

import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path

import ee
import numpy as np
import rasterio

from pipeline.auth import init_gee
from pipeline.config import load_config
from pipeline.export import download_image
from pipeline.sentinel2 import load_aoi, load_polygon

AOIS = ["Zambia_Mponda", "Zambia_WL"]
LOG_PATH = Path("outputs/backfill_log.csv")
LOG_FIELDS = ["aoi", "sensor", "resolution", "year", "month",
              "status", "filepath", "elapsed_seconds", "timestamp"]


# -- helpers -------------------------------------------------------------------

def _months_between(start_year: int, start_month: int,
                    end_year: int, end_month: int) -> list[tuple[int, int]]:
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def safe_end_date() -> tuple[int, int]:
    """Return (year, month) of the last complete month.

    Called on 2026-05-19 → returns (2026, 4), meaning April 2026 is the
    last fully-elapsed month. The current month is never downloaded because
    it is still accumulating images.
    """
    today = datetime.date.today()
    if today.month == 1:
        return (today.year - 1, 12)
    return (today.year, today.month - 1)


def _latest_month_for_collection(collection_id: str, aoi: ee.Geometry) -> tuple[int, int]:
    """Query GEE for the most recent image month in a collection."""
    img = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .sort("system:time_start", False)
        .first()
    )
    ts_ms = img.get("system:time_start").getInfo()
    dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
    return (dt.year, dt.month)


def _load_log() -> set[tuple]:
    """Return set of (aoi, sensor, resolution, year, month) already logged as success."""
    done = set()
    if not LOG_PATH.exists():
        return done
    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "success":
                done.add((
                    row["aoi"], row["sensor"], int(row["resolution"]),
                    int(row["year"]), int(row["month"])
                ))
    return done


def _append_log(aoi: str, sensor: str, resolution: int, year: int, month: int,
                status: str, filepath: str, elapsed: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({
            "aoi": aoi, "sensor": sensor, "resolution": resolution,
            "year": year, "month": month, "status": status,
            "filepath": filepath, "elapsed_seconds": f"{elapsed:.1f}",
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        })


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    return "#" * filled + "." * (width - filled)


def _fmt_seconds(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _print_progress(stage_label: str, aoi: str, res: int, done: int, total: int,
                    current_label: str, elapsed: float, n_ok: int, n_fail: int) -> None:
    pct = int(100 * done / total) if total else 0
    bar = _bar(done, total)
    eta = (elapsed / done * (total - done)) if done > 0 else 0.0
    line = (
        f"\r[{stage_label}] {aoi} {res}m  {done}/{total}  {bar}  {pct}%\n"
        f"  Current: {current_label} | Elapsed: {_fmt_seconds(elapsed)}"
        f" | ETA: {_fmt_seconds(eta)} | OK: {n_ok} | Failed: {n_fail}"
    )
    # Move cursor up 1 line to overwrite both lines on next call
    print(f"\033[F\033[F{line}", end="", flush=True)


def _try_download(composite, aoi_ee, output_path: str, scale: int, polygon,
                  max_attempts: int = 3) -> tuple[bool, str]:
    """Attempt download up to max_attempts times. Returns (success, error_msg)."""
    waits = [0, 10, 30]
    for attempt in range(max_attempts):
        if attempt > 0:
            wait = waits[attempt]
            print(f"\n    retry {attempt}/{max_attempts - 1}: waiting {wait}s ...", flush=True)
            time.sleep(wait)
        try:
            download_image(composite, aoi_ee, output_path, scale=scale, mask_polygon=polygon)
            return True, ""
        except Exception as exc:
            err = str(exc).splitlines()[0]
            if attempt < max_attempts - 1:
                print(f"\n    attempt {attempt + 1} failed: {err}", flush=True)
    return False, err


# -- latest end date per sensor ------------------------------------------------

def _get_end_date(stage: str, config: dict, aoi_ee: ee.Geometry) -> tuple[int, int]:
    safe = safe_end_date()
    if stage in ("sentinel2", "modis"):
        return safe
    elif stage == "burned_area":
        latest = _latest_month_for_collection(
            config["sensors"]["burned_area"]["collection"], aoi_ee
        )
        return min(safe, latest)
    else:
        raise ValueError(f"Unknown stage: {stage}")


# -- stage runner --------------------------------------------------------------

def run_stage(stage: str, config: dict, dry_run: bool = False) -> None:
    output_root = Path(config["output_root"])
    done_log    = _load_log()

    if stage == "sentinel2":
        start_str   = config["sensors"]["sentinel2"]["start_date"]
        resolutions = config["sensors"]["sentinel2"]["resolutions"]
    elif stage == "modis":
        start_str   = config["sensors"]["modis"]["start_date"]
        resolutions = list(config["sensors"]["modis"]["resolutions"].keys())
    elif stage == "burned_area":
        start_str   = config["sensors"]["burned_area"]["start_date"]
        resolutions = config["sensors"]["burned_area"]["resolutions"]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    start_dt = datetime.date.fromisoformat(start_str)
    start_year, start_month = start_dt.year, start_dt.month

    for aoi_name in AOIS:
        aoi_ee  = load_aoi(config["aois"][aoi_name]["path"])
        polygon = load_polygon(config["aois"][aoi_name]["path"])

        end_year, end_month = _get_end_date(stage, config, aoi_ee)
        all_months = _months_between(start_year, start_month, end_year, end_month)

        for res in resolutions:
            out_dir = output_root / aoi_name / stage / f"{res}m"
            out_dir.mkdir(parents=True, exist_ok=True)

            total   = len(all_months)
            n_done  = 0
            n_ok    = 0
            n_skip  = 0
            n_fail  = 0
            retried = []   # (year, month, recovered_attempt)

            stage_start = time.time()

            # Print two blank lines so _print_progress can overwrite them
            print(f"\n")

            for year, month in all_months:
                label = f"{year}-{month:02d}"

                if stage == "burned_area":
                    filename = f"{label}_BurnedArea_{aoi_name}.tif"
                else:
                    filename = f"{label}_NDVI_{aoi_name}.tif"

                output_path = out_dir / filename
                log_key = (aoi_name, stage, res, year, month)

                # Progress update (before processing)
                _print_progress(
                    stage_label=f"Stage: {stage}", aoi=aoi_name, res=res,
                    done=n_done, total=total, current_label=label,
                    elapsed=time.time() - stage_start, n_ok=n_ok, n_fail=n_fail,
                )

                # Skip if already logged as success
                if log_key in done_log:
                    print(f"\n  [{label}] skipped (log)", flush=True)
                    n_skip += 1
                    n_done += 1
                    continue

                # Skip if file already exists on disk
                if output_path.exists():
                    print(f"\n  [{label}] skipped (exists)", flush=True)
                    _append_log(aoi_name, stage, res, year, month, "success",
                                str(output_path), 0.0)
                    n_skip += 1
                    n_done += 1
                    continue

                if dry_run:
                    print(f"\n  [{label}] would download: {output_path}", flush=True)
                    n_ok += 1
                    n_done += 1
                    continue

                t0 = time.time()

                try:
                    if stage == "sentinel2":
                        from pipeline.sentinel2 import monthly_composite as s2_composite
                        composite = s2_composite(aoi_ee, year, month)
                    elif stage == "modis":
                        from pipeline.modis import monthly_composite as modis_composite
                        composite = modis_composite(aoi_ee, year, month, res)
                    else:
                        from pipeline.burned_area import monthly_image as ba_image
                        composite = ba_image(aoi_ee, year, month)
                except ValueError as exc:
                    # Burned area: month not published yet
                    elapsed = time.time() - t0
                    print(f"\n  [{label}] not published: {exc}", flush=True)
                    _append_log(aoi_name, stage, res, year, month, "not_published", "", elapsed)
                    n_done += 1
                    continue
                except Exception as exc:
                    elapsed = time.time() - t0
                    print(f"\n  [{label}] composite failed: {exc}", flush=True)
                    _append_log(aoi_name, stage, res, year, month, "failed", "", elapsed)
                    n_fail += 1
                    n_done += 1
                    continue

                ok, err = _try_download(composite, aoi_ee, str(output_path), res, polygon)
                elapsed = time.time() - t0

                if ok:
                    _append_log(aoi_name, stage, res, year, month, "success",
                                str(output_path), elapsed)
                    n_ok += 1
                else:
                    print(f"\n  [{label}] FAILED after 3 attempts: {err}", flush=True)
                    _append_log(aoi_name, stage, res, year, month, "failed", "", elapsed)
                    n_fail += 1
                    retried.append((year, month, False))

                n_done += 1

            # Final progress line
            _print_progress(
                stage_label=f"Stage: {stage}", aoi=aoi_name, res=res,
                done=n_done, total=total, current_label="done",
                elapsed=time.time() - stage_start, n_ok=n_ok, n_fail=n_fail,
            )
            print()  # newline after final progress

            print(f"\n  Summary {aoi_name}/{stage}/{res}m: "
                  f"{n_ok} ok, {n_skip} skipped, {n_fail} failed")

            if retried:
                print(f"\n  {n_fail} failed months logged for manual retry.")

    print(f"\nStage '{stage}' complete.")


# -- verification --------------------------------------------------------------

def verify_stage(stage: str, config: dict) -> dict:
    """Verify all expected output files for a completed stage."""
    output_root = Path(config["output_root"])

    if stage == "sentinel2":
        start_str   = config["sensors"]["sentinel2"]["start_date"]
        resolutions = config["sensors"]["sentinel2"]["resolutions"]
        expected_dtype   = "float32"
        data_type        = "NDVI"
        val_min, val_max = -0.2, 0.95
    elif stage == "modis":
        start_str   = config["sensors"]["modis"]["start_date"]
        resolutions = list(config["sensors"]["modis"]["resolutions"].keys())
        expected_dtype   = "float32"
        data_type        = "NDVI"
        val_min, val_max = -0.2, 0.95
    elif stage == "burned_area":
        start_str   = config["sensors"]["burned_area"]["start_date"]
        resolutions = config["sensors"]["burned_area"]["resolutions"]
        expected_dtype   = None  # int variants differ
        data_type        = "BurnedArea"
        val_min, val_max = 0, 366
    else:
        raise ValueError(f"Unknown stage: {stage}")

    start_dt = datetime.date.fromisoformat(start_str)
    start_year, start_month = start_dt.year, start_dt.month

    problems = []
    n_checked = 0
    n_empty = 0
    n_bad_crs = 0
    n_out_of_range = 0

    legacy_root = (
        Path(config["legacy_burned_area_root"]) if stage == "burned_area"
        else Path(config["legacy_data_root"])
    )

    # Spot-check months: pick 5 random-ish indices per sub-stage
    import random
    rng = random.Random(42)

    for aoi_name in AOIS:
        aoi_ee  = load_aoi(config["aois"][aoi_name]["path"])
        end_year, end_month = _get_end_date(stage, config, aoi_ee)
        all_months = _months_between(start_year, start_month, end_year, end_month)

        for res in resolutions:
            out_dir = output_root / aoi_name / stage / f"{res}m"
            spot_indices = rng.sample(range(len(all_months)), min(5, len(all_months)))

            for i, (year, month) in enumerate(all_months):
                label = f"{year}-{month:02d}"
                if stage == "burned_area":
                    filename = f"{label}_BurnedArea_{aoi_name}.tif"
                else:
                    filename = f"{label}_NDVI_{aoi_name}.tif"

                path = out_dir / filename
                n_checked += 1

                if not path.exists():
                    problems.append(f"MISSING: {path}")
                    continue

                # Skip size check for burned area: no-fire months compress to ~558B
                # (all-zero int raster after polygon mask), which is valid data.
                if stage != "burned_area" and path.stat().st_size < 1024:
                    n_empty += 1
                    problems.append(f"TINY ({path.stat().st_size}B): {path.name}")
                    continue

                try:
                    with rasterio.open(path) as src:
                        crs_epsg = src.crs.to_epsg() if src.crs else None
                        data     = src.read(1).astype(np.float32)
                        nodata   = src.nodata

                    if nodata is not None:
                        data = np.where(data == float(nodata), np.nan, data)
                    if stage == "burned_area":
                        valid = data[data > 0]
                    else:
                        valid = data[np.isfinite(data)]

                    if crs_epsg != 4326:
                        n_bad_crs += 1
                        problems.append(f"BAD CRS (EPSG:{crs_epsg}): {path.name}")

                    if len(valid) == 0:
                        n_empty += 1
                        problems.append(f"NO VALID PIXELS: {path.name}")
                        continue

                    mean_val = float(np.mean(valid))
                    if not (val_min <= mean_val <= val_max):
                        n_out_of_range += 1
                        problems.append(
                            f"OUT-OF-RANGE mean={mean_val:.3f} [{val_min},{val_max}]: {path.name}"
                        )

                    # Spot check: compare against legacy
                    if i in spot_indices:
                        leg_path = None
                        if stage == "burned_area" and aoi_name in {"Zambia_Mponda", "Zambia_WL"}:
                            leg_path = legacy_root / aoi_name / "500m_resolution" / filename
                        elif stage in ("sentinel2", "modis") and aoi_name == "Zambia_Mponda":
                            if stage == "sentinel2":
                                if res == 100:
                                    leg_path = legacy_root / "Zambia_Mponda" / "100m_resolution" / filename
                                elif res == 1000:
                                    leg_path = legacy_root / "Zambia_Mponda" / "Sentinel_1000m_resolution" / filename
                            elif stage == "modis":
                                if res == 250:
                                    leg_path = legacy_root / "Zambia_Mponda" / "250m_resolution" / filename
                                elif res == 500:
                                    leg_path = legacy_root / "Zambia_Mponda" / "500m_resolution" / filename
                                elif res == 1000:
                                    leg_path = legacy_root / "Zambia_Mponda" / "MODIS_1000m_resolution" / filename

                        if leg_path is not None and leg_path.exists():
                            with rasterio.open(leg_path) as lsrc:
                                leg_data = lsrc.read(1).astype(np.float32)
                                leg_nd   = lsrc.nodata
                            if leg_nd is not None:
                                leg_data = np.where(leg_data == float(leg_nd), np.nan, leg_data)
                            leg_valid = leg_data[np.isfinite(leg_data)]
                            if len(leg_valid) > 0:
                                leg_mean = float(np.mean(leg_valid))
                                diff = abs(mean_val - leg_mean)
                                if diff > 0.15:
                                    problems.append(
                                        f"SPOT CHECK mean diff={diff:.3f} vs legacy: {path.name}"
                                    )

                except Exception as exc:
                    problems.append(f"RASTERIO ERROR ({exc}): {path.name}")

    total_problems = len(problems)
    print(f"\nVerification: {n_checked} files checked")
    print(f"  Empty/tiny: {n_empty}, Bad CRS: {n_bad_crs}, Out-of-range: {n_out_of_range}")

    if problems:
        print(f"\n  {total_problems} problem(s):")
        for p in problems:
            print(f"    - {p}")
    else:
        print(f"  STAGE '{stage.upper()}' VERIFIED")

    return {
        "n_checked": n_checked,
        "n_problems": total_problems,
        "problems": problems,
        "n_empty": n_empty,
        "n_bad_crs": n_bad_crs,
        "n_out_of_range": n_out_of_range,
    }


# -- main ----------------------------------------------------------------------

STAGES = ["sentinel2", "modis", "burned_area"]

parser = argparse.ArgumentParser(
    description="Full historical backfill: generate all monthly composites."
)
parser.add_argument(
    "--stage", required=True,
    choices=STAGES + ["all"],
    help="Stage to run: sentinel2 | modis | burned_area | all"
)
parser.add_argument("--dry-run", action="store_true",
                    help="Print what would be generated without downloading")
args = parser.parse_args()

config = load_config()

stages_to_run = STAGES if args.stage == "all" else [args.stage]

# GEE_SERVICE_ACCOUNT_KEY set in CI -> service account; unset locally -> interactive.
init_gee(config["project"], os.environ.get("GEE_SERVICE_ACCOUNT_KEY"))

n_stages = len(stages_to_run)

for i, stage in enumerate(stages_to_run, 1):
    print(f"\n{'=' * 60}")
    print(f"Stage {i}/{n_stages}: {stage}  {'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)

    run_stage(stage, config, dry_run=args.dry_run)

    if not args.dry_run:
        result = verify_stage(stage, config)

        if result["n_problems"] > 0 and stage != stages_to_run[-1]:
            ans = input(f"\nContinue to next stage? [y/n]: ").strip().lower()
            if ans != "y":
                print("Stopped by user.")
                sys.exit(1)

print("\nAll done.")
