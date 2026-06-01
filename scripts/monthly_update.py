"""
Monthly Update -- fetch new rasters, run Pass A/B, deploy.
Triggered by GitHub Actions on the 5th of each month.

Steps:
  [1/7] Check latest available data (GEE)
  [2/7] Check existing data, compute gaps
  [3/7] Fetch new rasters from GEE
  [4/7] Run Pass A (monthly Parquet tables)
  [5/7] Check for new complete year; run Pass B if detected
  [6/7] Deploy to app
  [7/7] Write log entry to outputs/update_log.csv

Usage:
    python scripts/monthly_update.py [--dry-run] [--force-pass-b]
                                     [--skip-fetch] [--skip-deploy]
                                     [--aoi AOI_NAME]
"""

import argparse
import csv
import datetime
import subprocess
import sys
import time
from pathlib import Path

from pipeline.config import load_config

UPDATE_LOG_PATH = Path("outputs/update_log.csv")
LOG_FIELDS = [
    "run_timestamp", "new_months_fetched", "pass_a_ran", "pass_b_ran",
    "new_year_detected", "deploy_ran", "status", "error_message",
]
AOIS = ["Zambia_Mponda", "Zambia_WL"]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def safe_end_date() -> tuple[int, int]:
    """Return (year, month) of the last fully-elapsed calendar month."""
    today = datetime.date.today()
    if today.month == 1:
        return (today.year - 1, 12)
    return (today.year, today.month - 1)


def _months_between(
    sy: int, sm: int, ey: int, em: int
) -> list[tuple[int, int]]:
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


# ---------------------------------------------------------------------------
# File-scanning helpers
# ---------------------------------------------------------------------------

def get_latest_fetched_month(
    aoi: str, sensor: str, resolution: int, outputs_root: str
) -> tuple[int, int] | None:
    """Return (year, month) of the most recent TIF for this combo, or None."""
    folder = Path(outputs_root) / aoi / sensor / f"{resolution}m"
    if not folder.exists():
        return None
    best: tuple[int, int] | None = None
    for tif in folder.glob("*.tif"):
        prefix = tif.name[:7]  # "YYYY-MM"
        if len(prefix) == 7 and prefix[4] == "-":
            try:
                y, m = int(prefix[:4]), int(prefix[5:7])
                if best is None or (y, m) > best:
                    best = (y, m)
            except ValueError:
                continue
    return best


def _build_combos(config: dict, aoi_filter: str | None) -> list[tuple[str, str, int]]:
    aois = [aoi_filter] if aoi_filter else AOIS
    combos = []
    for aoi in aois:
        for res in config["sensors"]["sentinel2"]["resolutions"]:
            combos.append((aoi, "sentinel2", int(res)))
        for res in config["sensors"]["modis"]["resolutions"]:
            combos.append((aoi, "modis", int(res)))
        for res in config["sensors"]["burned_area"]["resolutions"]:
            combos.append((aoi, "burned_area", int(res)))
    return combos


def get_complete_years_from_files(
    outputs_root: str, config: dict, aoi_filter: str | None = None
) -> set[int]:
    """
    Return set of years where ALL combos have a December TIF file.
    A year is complete when December data exists for every (aoi, sensor, res).
    """
    combos = _build_combos(config, aoi_filter)
    root = Path(outputs_root)
    complete: set[int] = set()
    for year in range(2000, datetime.date.today().year):
        if all(
            any((root / aoi / sensor / f"{res}m").glob(f"{year}-12_*.tif"))
            for aoi, sensor, res in combos
        ):
            complete.add(year)
    return complete


def detect_new_complete_year(
    outputs_root: str,
    update_log_path: Path,
    config: dict,
    aoi_filter: str | None = None,
) -> int | None:
    """
    Return the newest complete year if it is newer than the last year Pass B ran on.
    Uses output files as truth; update_log.csv only to know what Pass B last processed.
    """
    complete_now = get_complete_years_from_files(outputs_root, config, aoi_filter)
    if not complete_now:
        return None

    last_pass_b_year = 0
    if update_log_path.exists():
        with open(update_log_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("pass_b_ran") == "True":
                    try:
                        yr = int(row.get("new_year_detected") or 0)
                        if yr > last_pass_b_year:
                            last_pass_b_year = yr
                    except (ValueError, TypeError):
                        pass

    max_complete = max(complete_now)
    return max_complete if max_complete > last_pass_b_year else None


# ---------------------------------------------------------------------------
# GEE fetch (single month, single combo)
# ---------------------------------------------------------------------------

def fetch_month(
    aoi: str,
    sensor: str,
    resolution: int,
    year: int,
    month: int,
    config: dict,
    aoi_ee,
    polygon,
) -> tuple[bool, float]:
    """Fetch one month for one combo. Returns (success, elapsed_seconds)."""
    from pipeline.export import download_image

    outputs_root = Path(config["output_root"])
    filename = (
        f"{year}-{month:02d}_BurnedArea_{aoi}.tif"
        if sensor == "burned_area"
        else f"{year}-{month:02d}_NDVI_{aoi}.tif"
    )
    output_path = outputs_root / aoi / sensor / f"{resolution}m" / filename

    if output_path.exists():
        return True, 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        if sensor == "sentinel2":
            from pipeline.sentinel2 import monthly_composite as _s2
            composite = _s2(aoi_ee, year, month)
        elif sensor == "modis":
            from pipeline.modis import monthly_composite as _mo
            composite = _mo(aoi_ee, year, month, resolution)
        else:
            from pipeline.burned_area import monthly_image as _ba
            composite = _ba(aoi_ee, year, month)

        download_image(composite, aoi_ee, str(output_path), scale=resolution, mask_polygon=polygon)
        elapsed = time.time() - t0
        print(f"    {aoi} {sensor} {resolution}m {year}-{month:02d}... done ({elapsed:.1f}s)")
        return True, elapsed

    except ValueError as exc:
        # Burned area not yet published — non-fatal
        elapsed = time.time() - t0
        print(f"    {aoi} {sensor} {resolution}m {year}-{month:02d}... not published: {str(exc).splitlines()[0]}")
        return True, elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"    {aoi} {sensor} {resolution}m {year}-{month:02d}... FAILED: {str(exc).splitlines()[0]}")
        return False, elapsed


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_subprocess(cmd: list[str], label: str, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        print(f"  [dry] would run: {' '.join(cmd)}")
        return True, ""
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return False, f"{label} failed (exit code {result.returncode})"
    return True, ""


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

def write_log(
    new_months: int,
    pass_a: bool,
    pass_b: bool,
    new_year_int: int,
    deploy: bool,
    status: str,
    error: str,
) -> None:
    UPDATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not UPDATE_LOG_PATH.exists()
    with open(UPDATE_LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({
            "run_timestamp":      datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "new_months_fetched": new_months,
            "pass_a_ran":         pass_a,
            "pass_b_ran":         pass_b,
            "new_year_detected":  new_year_int,
            "deploy_ran":         deploy,
            "status":             status,
            "error_message":      error,
        })
    print(f"  Logged to {UPDATE_LOG_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly Update -- fetch new rasters, run Pass A/B, deploy."
    )
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print plan without GEE calls, subprocesses, or writes")
    parser.add_argument("--force-pass-b", action="store_true",
                        help="Run Pass B regardless of new-year detection")
    parser.add_argument("--skip-fetch",   action="store_true",
                        help="Skip GEE fetch; go straight to Pass A")
    parser.add_argument("--skip-deploy",  action="store_true",
                        help="Skip deploy step")
    parser.add_argument("--aoi",          default=None,
                        help="Process only this AoI (e.g. Zambia_Mponda)")
    args = parser.parse_args()

    config = load_config()
    outputs_root = config["output_root"]
    aoi_filter = args.aoi
    aois = [aoi_filter] if aoi_filter else AOIS
    combos = _build_combos(config, aoi_filter)

    start_time = time.time()
    state = {
        "new_months": 0,
        "pass_a_ran": False,
        "pass_b_ran": False,
        "new_year": None,
        "deploy_ran": False,
        "error": "",
    }

    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print("=" * 40, flush=True)
    print(f"Monthly Update -- {now_str}", flush=True)
    print("=" * 40, flush=True)
    if args.dry_run:
        print("DRY RUN -- no GEE calls, no subprocesses, no file writes\n")

    # GEE init — always, matching backfill_full.py pattern (skip in dry-run)
    if not args.dry_run:
        from pipeline.auth import init_gee
        init_gee(config["project"])

    try:
        # ------------------------------------------------------------------
        # [1/7] Latest available data
        # ------------------------------------------------------------------
        print("\n[1/7] Checking latest available data...", flush=True)
        s2_end = safe_end_date()
        mo_end = safe_end_date()

        if args.dry_run:
            # Approximate BA availability (typically ~3 months behind)
            y, m = s2_end
            m -= 3
            while m < 1:
                m += 12
                y -= 1
            ba_end = (y, m)
        else:
            from pipeline.sentinel2 import load_aoi as _load_aoi
            from pipeline.burned_area import latest_available_month as _ba_latest
            _aoi0_ee = _load_aoi(config["aois"][aois[0]]["path"])
            ba_end = _ba_latest(_aoi0_ee)

        print(f"  Sentinel-2:  latest available = {s2_end[0]}-{s2_end[1]:02d}")
        print(f"  MODIS:       latest available = {mo_end[0]}-{mo_end[1]:02d}")
        print(f"  Burned area: latest available = {ba_end[0]}-{ba_end[1]:02d}")

        # ------------------------------------------------------------------
        # [2/7] Check existing data
        # ------------------------------------------------------------------
        print("\n[2/7] Checking existing data...", flush=True)
        gaps: dict[tuple, list[tuple[int, int]]] = {}
        total_gap_months = 0

        for aoi, sensor, res in combos:
            sensor_end = ba_end if sensor == "burned_area" else (s2_end if sensor == "sentinel2" else mo_end)
            latest = get_latest_fetched_month(aoi, sensor, res, outputs_root)

            if latest is None:
                print(f"  {aoi}/{sensor}/{res}m: no existing files (needs full backfill first)")
                continue

            print(f"  Last fetched: {aoi}/{sensor}/{res}m = {latest[0]}-{latest[1]:02d}")

            next_y, next_m = latest[0], latest[1] + 1
            if next_m > 12:
                next_m, next_y = 1, next_y + 1

            to_fetch = _months_between(next_y, next_m, sensor_end[0], sensor_end[1])
            if to_fetch:
                gaps[(aoi, sensor, res)] = to_fetch
                total_gap_months += len(to_fetch)

        if total_gap_months == 0:
            print("  All combos up to date.")
        else:
            print(f"  New months to fetch: {total_gap_months} across {len(gaps)} combos")

        # ------------------------------------------------------------------
        # [3/7] Fetch new rasters
        # ------------------------------------------------------------------
        print("\n[3/7] Fetching new rasters from GEE...", flush=True)
        n_fetched = 0
        n_failed = 0

        if args.skip_fetch:
            print("  Skipped (--skip-fetch)")
        elif not gaps:
            print("  Nothing to fetch.")
        elif args.dry_run:
            for (aoi, sensor, res), months in gaps.items():
                for y, m in months:
                    print(f"  [dry] {aoi} {sensor} {res}m {y}-{m:02d}")
            n_fetched = total_gap_months
        else:
            from pipeline.sentinel2 import load_aoi as _load_aoi, load_polygon as _load_polygon
            aoi_ees = {a: _load_aoi(config["aois"][a]["path"]) for a in aois}
            polygons = {a: _load_polygon(config["aois"][a]["path"]) for a in aois}

            fetched_files: list[dict] = []
            for (aoi, sensor, res), months in gaps.items():
                for y, m in months:
                    ok, _ = fetch_month(aoi, sensor, res, y, m, config, aoi_ees[aoi], polygons[aoi])
                    if ok:
                        n_fetched += 1
                        filename = (
                            f"{y}-{m:02d}_BurnedArea_{aoi}.tif"
                            if sensor == "burned_area"
                            else f"{y}-{m:02d}_NDVI_{aoi}.tif"
                        )
                        fp = Path(outputs_root) / aoi / sensor / f"{res}m" / filename
                        if fp.exists():
                            fetched_files.append({
                                "path": str(fp),
                                "aoi": aoi, "sensor": sensor, "resolution": res,
                                "year": y, "month": m,
                                "data_type": "burned_area" if sensor == "burned_area" else "ndvi",
                            })
                    else:
                        n_failed += 1

            print(f"  [{n_fetched} completed, {n_failed} failed]")
            if n_failed > 0:
                if not state["error"]:
                    state["error"] = f"{n_failed} GEE fetch(es) failed"

            # ------------------------------------------------------------------
            # [3b/7] Verify downloaded rasters (before wasting compute on bad data)
            # ------------------------------------------------------------------
            if fetched_files:
                print("\n[3b/7] Verifying downloaded rasters...", flush=True)
                from preprocess.verify_outputs import verify_rasters
                raster_result = verify_rasters(new_files=fetched_files, config=config)
                print(raster_result.summary, flush=True)
                if not raster_result.passed:
                    failed_paths = {path for path, _, _ in raster_result.failures}
                    n_excluded = len([f for f in fetched_files if f["path"] in failed_paths])
                    fetched_files = [f for f in fetched_files if f["path"] not in failed_paths]
                    print(f"  {n_excluded} file(s) failed verification -- "
                          f"excluded from preprocessing", flush=True)

        # Build list of new (aoi, sensor, res, year, month) tuples for Parquet check
        new_months_for_verify: list[tuple] = [
            (f["aoi"], f["sensor"], f["resolution"], f["year"], f["month"])
            for f in (fetched_files if not args.skip_fetch and not args.dry_run else [])
        ]

        state["new_months"] = n_fetched

        # ------------------------------------------------------------------
        # [4/7] Pass A
        # ------------------------------------------------------------------
        print("\n[4/7] Running Pass A (monthly Parquet tables)...", flush=True)
        should_run_a = n_fetched > 0 or args.skip_fetch

        if should_run_a:
            cmd_a = [sys.executable, "-m", "scripts.preprocess_pass_a", "--force", "--no-validate"]
            if aoi_filter:
                cmd_a += ["--aoi", aoi_filter]
            ok_a, err_a = run_subprocess(cmd_a, "Pass A", args.dry_run)
            state["pass_a_ran"] = ok_a
            if not ok_a:
                state["error"] = err_a
                print(f"  Pass A FAILED: {err_a}")
                print("  Skipping Pass B and Deploy")
        else:
            print("  No new data -- skipping Pass A")

        # ------------------------------------------------------------------
        # [5/7] New complete year + Pass B
        # ------------------------------------------------------------------
        print("\n[5/7] Checking for new complete year...", flush=True)
        new_year = detect_new_complete_year(outputs_root, UPDATE_LOG_PATH, config, aoi_filter)

        if args.force_pass_b and new_year is None:
            complete = get_complete_years_from_files(outputs_root, config, aoi_filter)
            new_year = max(complete, default=None)
            if new_year:
                print(f"  --force-pass-b active: treating {new_year} as new complete year")

        state["new_year"] = new_year

        if new_year:
            print(f"  New complete year detected: {new_year}")
            pass_a_ok = state["pass_a_ran"] or args.dry_run
            if pass_a_ok:
                cmd_b = [sys.executable, "-m", "scripts.preprocess_pass_b", "--no-validate"]
                if aoi_filter:
                    cmd_b += ["--aoi", aoi_filter]
                ok_b, err_b = run_subprocess(cmd_b, "Pass B", args.dry_run)
                state["pass_b_ran"] = ok_b
                if not ok_b:
                    if not state["error"]:
                        state["error"] = err_b
                    print(f"  Pass B FAILED: {err_b} -- will still attempt deploy")
            else:
                print("  Pass B skipped -- Pass A did not succeed")
        else:
            latest_complete = max(
                get_complete_years_from_files(outputs_root, config, aoi_filter), default=None
            )
            print(f"  No new complete year detected (latest complete: {latest_complete})")
            print("  Skipping Pass B")

        # ------------------------------------------------------------------
        # [5b/7] Verify preprocessed Parquet outputs (before deploy)
        # ------------------------------------------------------------------
        verify_passed = True
        if state["pass_a_ran"] or args.dry_run:
            print("\n[5b/7] Verifying preprocessed outputs...", flush=True)
            from preprocess.verify_outputs import verify_parquet_outputs
            parquet_result = verify_parquet_outputs(
                outputs_root=outputs_root,
                new_months=new_months_for_verify,
                config=config,
            )
            print(parquet_result.summary, flush=True)
            if not parquet_result.passed:
                verify_passed = False
                print("  Parquet verification FAILED -- aborting deploy", flush=True)
                if not state["error"]:
                    state["error"] = (
                        f"Parquet verification failed: "
                        f"{len(parquet_result.failures)} check(s) failed"
                    )

        # ------------------------------------------------------------------
        # [6/7] Deploy
        # ------------------------------------------------------------------
        print("\n[6/7] Deploying to app...", flush=True)
        pass_a_ok_for_deploy = (state["pass_a_ran"] or args.dry_run) and verify_passed
        if args.skip_deploy:
            print("  Skipped (--skip-deploy)")
        elif not pass_a_ok_for_deploy:
            print("  Skipped -- Pass A did not run successfully")
        else:
            cmd_d = [sys.executable, "-m", "scripts.deploy", "--stage", "all"]
            ok_d, err_d = run_subprocess(cmd_d, "Deploy", args.dry_run)
            state["deploy_ran"] = ok_d
            if not ok_d:
                if not state["error"]:
                    state["error"] = err_d
                print(f"  Deploy FAILED: {err_d}")

    except Exception as exc:
        state["error"] = f"Unexpected error: {exc}"
        print(f"\nFATAL: {exc}")

    finally:
        # ------------------------------------------------------------------
        # [7/7] Write log
        # ------------------------------------------------------------------
        print("\n[7/7] Writing log entry...", flush=True)
        # Only record the year when Pass B actually ran; 0 otherwise.
        # detect_new_complete_year() uses this to avoid re-running Pass B
        # for a year it already processed — so only set when Pass B succeeded.
        new_year_int = state["new_year"] if (state["pass_b_ran"] and state["new_year"]) else 0
        status = "error" if state["error"] else "success"

        if args.dry_run:
            print(
                f"  [dry] would log: new_months={state['new_months']} "
                f"pass_a={state['pass_a_ran']} pass_b={state['pass_b_ran']} "
                f"new_year={new_year_int} deploy={state['deploy_ran']} "
                f"status={status}"
            )
        else:
            write_log(
                state["new_months"],
                state["pass_a_ran"],
                state["pass_b_ran"],
                new_year_int,
                state["deploy_ran"],
                status,
                state["error"],
            )

        elapsed = time.time() - start_time
        if state["error"]:
            print(f"\nError: {state['error']}")
        print(f"\n{'=' * 40}", flush=True)
        print(f"Update complete. Duration: {elapsed:.0f}s", flush=True)
        print("=" * 40, flush=True)
        sys.exit(1 if state["error"] else 0)


if __name__ == "__main__":
    main()
