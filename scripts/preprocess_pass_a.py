"""
Preprocessing Pipeline Pass A -- Core Time Series Tables

Computes 9 Parquet summary tables from deployed GeoTIFF rasters:
  1. ndvi_monthly.parquet
  2. ndvi_monthly_baselines.parquet
  3. ndvi_annual.parquet
  4. ndvi_trend_stats.parquet
  5. ndvi_monthly_by_class.parquet
  6. ndvi_monthly_baselines_by_class.parquet
  7. ndvi_anomaly_monthly.parquet
  8. ba_monthly.parquet          (burned_area only)
  9. ba_daily.parquet            (burned_area only)

Outputs: outputs/processed/{aoi}/{sensor}/{resolution}m/{table}.parquet

Usage:
    python scripts/preprocess_pass_a.py --dry-run
    python scripts/preprocess_pass_a.py --aoi Zambia_Mponda
    python scripts/preprocess_pass_a.py
    python scripts/preprocess_pass_a.py --force
    python scripts/preprocess_pass_a.py --force-table ndvi_anomaly_monthly
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from pipeline.config import load_config
from preprocess.core import (
    LC_CLASSES,
    RESOLUTION_FOLDER_MAP,
    compute_ba_daily,
    compute_ba_monthly,
    compute_class_baselines,
    compute_ndvi_annual,
    compute_ndvi_anomaly_monthly,
    compute_ndvi_monthly,
    compute_ndvi_monthly_baselines,
    compute_ndvi_monthly_by_class,
)
from preprocess.validate import (
    validate_ba_monthly,
    validate_ndvi_monthly,
    validate_trend_conclusions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir(aoi: str, sensor: str, resolution: int, config: dict) -> Path:
    root = Path(config["output_root"]) / "processed" / aoi / sensor / f"{resolution}m"
    return root


def _combos(config: dict, aoi_filter: str | None) -> list[tuple[str, str, int]]:
    """Return list of (aoi, sensor, resolution) tuples from config."""
    aois = list(config["aois"].keys())
    if aoi_filter:
        if aoi_filter not in aois:
            print(f"ERROR: AoI '{aoi_filter}' not in config. Valid: {aois}")
            sys.exit(1)
        aois = [aoi_filter]

    combos = []
    for aoi in aois:
        for res in config["sensors"]["sentinel2"]["resolutions"]:
            combos.append((aoi, "sentinel2", int(res)))
        for res in config["sensors"]["modis"]["resolutions"]:
            combos.append((aoi, "modis", int(res)))
        for res in config["sensors"]["burned_area"]["resolutions"]:
            combos.append((aoi, "burned_area", int(res)))
    return combos


def _source_count(aoi: str, sensor: str, resolution: int, config: dict) -> int:
    """Count TIF files available for this combo."""
    from preprocess.core import _ndvi_files, _ba_files
    if sensor == "burned_area":
        return len(_ba_files(aoi, config))
    return len(_ndvi_files(aoi, sensor, resolution, config))


def _write(df, path: Path, label: str, dry_run: bool) -> None:
    if dry_run:
        print(f"    [dry] {label}: {len(df)} rows")
        return
    if df.empty:
        print(f"    [skip] {label}: 0 rows (no data)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"    {label}: {len(df)} rows -> {path.relative_to(Path('.'))}")


def _should_skip(path: Path, table_stem: str, force: bool, dry_run: bool, force_table: str | None) -> bool:
    """Return True if this table should be skipped.

    With --force-table, only the named table runs; all others are skipped.
    """
    if force_table is not None:
        return force_table != table_stem
    return not force and not dry_run and path.exists()


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(
    config: dict,
    aoi_filter: str | None,
    dry_run: bool,
    force: bool,
    force_table: str | None = None,
) -> None:
    combos = _combos(config, aoi_filter)
    processed_root = str(Path(config["output_root"]) / "processed")

    all_skipped = 0
    all_written = 0

    for aoi, sensor, resolution in combos:
        out_dir = _output_dir(aoi, sensor, resolution, config)
        n_src = _source_count(aoi, sensor, resolution, config)
        print(f"\n[{aoi} / {sensor} / {resolution}m]  source TIFs: {n_src}")

        if sensor in ("sentinel2", "modis"):
            # --- Tables 1-7 ---
            tables = {
                "ndvi_monthly.parquet":                    None,
                "ndvi_monthly_baselines.parquet":          None,
                "ndvi_annual.parquet":                     None,
                "ndvi_trend_stats.parquet":                None,
                "ndvi_monthly_by_class.parquet":           None,
                "ndvi_monthly_baselines_by_class.parquet": None,
                "ndvi_anomaly_monthly.parquet":            None,
            }

            # Skip-all only when no force_table override is active
            skip = (
                not force
                and not dry_run
                and force_table is None
                and all((out_dir / t).exists() for t in tables)
            )
            if skip:
                print(f"  [skip] all tables exist (use --force to recompute)")
                all_skipped += len(tables)
                continue

            # Table 1
            t1_path = out_dir / "ndvi_monthly.parquet"
            if _should_skip(t1_path, "ndvi_monthly", force, dry_run, force_table):
                if force_table is None:
                    print(f"    [skip] ndvi_monthly.parquet (exists)")
                monthly_df = pd.read_parquet(t1_path) if t1_path.exists() else pd.DataFrame()
            else:
                print(f"  Computing ndvi_monthly ...")
                monthly_df = compute_ndvi_monthly(aoi, sensor, resolution, config)
                _write(monthly_df, t1_path, "ndvi_monthly.parquet", dry_run)
                all_written += 1

            if monthly_df.empty:
                print(f"  [warn] No data -- skipping derived tables")
                continue

            # Table 2
            t2_path = out_dir / "ndvi_monthly_baselines.parquet"
            if _should_skip(t2_path, "ndvi_monthly_baselines", force, dry_run, force_table):
                if force_table is None:
                    print(f"    [skip] ndvi_monthly_baselines.parquet (exists)")
            else:
                baselines_df = compute_ndvi_monthly_baselines(monthly_df)
                _write(baselines_df, t2_path, "ndvi_monthly_baselines.parquet", dry_run)
                all_written += 1

            # Table 3
            t3a_path = out_dir / "ndvi_annual.parquet"
            t3b_path = out_dir / "ndvi_trend_stats.parquet"
            skip_3a = _should_skip(t3a_path, "ndvi_annual", force, dry_run, force_table)
            skip_3b = _should_skip(t3b_path, "ndvi_trend_stats", force, dry_run, force_table)
            if skip_3a and skip_3b:
                if force_table is None:
                    print(f"    [skip] ndvi_annual.parquet + ndvi_trend_stats.parquet (exist)")
            else:
                annual_df, trend_df = compute_ndvi_annual(monthly_df)
                if not skip_3a:
                    _write(annual_df, t3a_path, "ndvi_annual.parquet", dry_run)
                    all_written += 1
                if not skip_3b:
                    _write(trend_df, t3b_path, "ndvi_trend_stats.parquet", dry_run)
                    all_written += 1

            # Table 4
            t4_path = out_dir / "ndvi_monthly_by_class.parquet"
            if _should_skip(t4_path, "ndvi_monthly_by_class", force, dry_run, force_table):
                if force_table is None:
                    print(f"    [skip] ndvi_monthly_by_class.parquet (exists)")
                by_class_df = pd.read_parquet(t4_path) if t4_path.exists() else pd.DataFrame()
            else:
                print(f"  Computing ndvi_monthly_by_class (slow -- {n_src} files x {len(LC_CLASSES)} classes) ...")
                by_class_df = compute_ndvi_monthly_by_class(aoi, sensor, resolution, config)
                _write(by_class_df, t4_path, "ndvi_monthly_by_class.parquet", dry_run)
                all_written += 1

            # Table 5
            t5_path = out_dir / "ndvi_monthly_baselines_by_class.parquet"
            if _should_skip(t5_path, "ndvi_monthly_baselines_by_class", force, dry_run, force_table):
                if force_table is None:
                    print(f"    [skip] ndvi_monthly_baselines_by_class.parquet (exists)")
            else:
                if not by_class_df.empty:
                    class_baselines_df = compute_class_baselines(by_class_df)
                    _write(class_baselines_df, t5_path, "ndvi_monthly_baselines_by_class.parquet", dry_run)
                    all_written += 1
                else:
                    print(f"    [skip] ndvi_monthly_baselines_by_class.parquet (no by-class data)")

            # Table 7 — anomaly (pure Parquet join, no raster reads)
            t7_path = out_dir / "ndvi_anomaly_monthly.parquet"
            if _should_skip(t7_path, "ndvi_anomaly_monthly", force, dry_run, force_table):
                if force_table is None:
                    print(f"    [skip] ndvi_anomaly_monthly.parquet (exists)")
            else:
                anomaly_df = compute_ndvi_anomaly_monthly(aoi, sensor, resolution, processed_root)
                _write(anomaly_df, t7_path, "ndvi_anomaly_monthly.parquet", dry_run)
                if not anomaly_df.empty:
                    all_written += 1

        elif sensor == "burned_area":
            # ndvi_anomaly_monthly does not apply to burned_area
            if force_table == "ndvi_anomaly_monthly":
                continue

            # --- Tables 8-9 (BA) ---
            t8_path = out_dir / "ba_monthly.parquet"
            t9_path = out_dir / "ba_daily.parquet"

            skip = (
                not force
                and not dry_run
                and t8_path.exists()
                and t9_path.exists()
            )
            if skip:
                print(f"  [skip] ba_monthly + ba_daily (exist, use --force to recompute)")
                all_skipped += 2
                continue

            if not force and not dry_run and t8_path.exists():
                print(f"    [skip] ba_monthly.parquet (exists)")
            else:
                print(f"  Computing ba_monthly ...")
                ba_monthly_df = compute_ba_monthly(aoi, config)
                _write(ba_monthly_df, t8_path, "ba_monthly.parquet", dry_run)
                all_written += 1

            if not force and not dry_run and t9_path.exists():
                print(f"    [skip] ba_daily.parquet (exists)")
            else:
                print(f"  Computing ba_daily ...")
                ba_daily_df = compute_ba_daily(aoi, config)
                _write(ba_daily_df, t9_path, "ba_daily.parquet", dry_run)
                all_written += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Done: {all_written} tables written, {all_skipped} skipped.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(config: dict) -> None:
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    all_pass = True

    # 1. NDVI monthly mean within 0.01 of fresh TIF read
    ndvi_parquet = Path("outputs/processed/Zambia_Mponda/sentinel2/100m/ndvi_monthly.parquet")
    if ndvi_parquet.exists():
        result = validate_ndvi_monthly(
            str(ndvi_parquet), config, "Zambia_Mponda", "sentinel2", 100,
            test_year=2024, test_month=3,
        )
        status = "PASS" if result["passes"] else "FAIL"
        print(f"\n[{status}] NDVI monthly mean (Zambia_Mponda/S2/100m, 2024-03)")
        print(f"  Parquet : {result['python_mean']}")
        print(f"  TIF     : {result['raster_mean']}")
        print(f"  Diff    : {result['abs_diff']}")
        print(f"  Notes   : {result['notes']}")
        if not result["passes"]:
            all_pass = False
    else:
        print(f"\n[SKIP] NDVI monthly validation -- {ndvi_parquet} not found")

    # 2. Trend direction for MODIS 1000m
    trend_parquet = Path("outputs/processed/Zambia_Mponda/modis/1000m/ndvi_trend_stats.parquet")
    if trend_parquet.exists():
        expected = {("Zambia_Mponda", "modis", 1000): {"trend_direction": "decreasing"}}
        results = validate_trend_conclusions(str(trend_parquet), expected)
        for key, r in results.items():
            status = "PASS" if r["passes"] else "FAIL"
            print(f"\n[{status}] Trend direction ({key[0]}/{key[1]}/{key[2]}m)")
            print(f"  Expected : {r['expected']}")
            print(f"  Actual   : {r['actual']}")
            print(f"  Notes    : {r['notes']}")
            if not r["passes"]:
                all_pass = False
    else:
        print(f"\n[SKIP] Trend validation -- {trend_parquet} not found")

    # 3. Burned area Zambia_WL August 2024
    ba_parquet = Path("outputs/processed/Zambia_WL/burned_area/500m/ba_monthly.parquet")
    if ba_parquet.exists():
        result = validate_ba_monthly(
            str(ba_parquet), config, "Zambia_WL",
            year=2024, month=8, expected_range=(300.0, 500.0),
        )
        status = "PASS" if result["passes"] else "FAIL"
        print(f"\n[{status}] BA monthly (Zambia_WL, 2024-08)")
        print(f"  burned_km2 : {result['burned_km2']}")
        print(f"  in_range   : {result['in_range']}  [300-500 km2]")
        print(f"  Notes      : {result['notes']}")
        if not result["passes"]:
            all_pass = False
    else:
        print(f"\n[SKIP] BA validation -- {ba_parquet} not found")

    print(f"\n{'All validations PASSED.' if all_pass else 'Some validations FAILED -- review above.'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Preprocessing Pipeline Pass A -- compute core time series Parquet tables."
)
parser.add_argument("--aoi",         default=None,  help="Process only this AoI (e.g. Zambia_Mponda)")
parser.add_argument("--dry-run",     action="store_true", help="Show what would be computed without writing")
parser.add_argument("--force",       action="store_true", help="Recompute even if output Parquet already exists")
parser.add_argument("--no-validate", action="store_true", help="Skip validation step")
parser.add_argument(
    "--force-table",
    default=None,
    metavar="TABLE_NAME",
    help="Run only this table, overwriting if it exists (e.g. ndvi_anomaly_monthly)",
)
args = parser.parse_args()

config = load_config()

if args.dry_run:
    print("DRY RUN -- no files will be written\n")

run(config, aoi_filter=args.aoi, dry_run=args.dry_run, force=args.force, force_table=args.force_table)

if not args.dry_run and not args.no_validate:
    run_validation(config)
