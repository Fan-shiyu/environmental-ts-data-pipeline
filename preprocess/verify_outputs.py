"""
Verification module for the monthly update pipeline.

Two main functions called by scripts/monthly_update.py:
  verify_rasters()        -- lightweight checks on newly downloaded GeoTIFFs
  verify_parquet_outputs() -- schema, value-range, and regression checks on Parquets

Both return VerificationResult. Can also be run standalone:
    python preprocess/verify_outputs.py --check all
    python preprocess/verify_outputs.py --check parquets
    python preprocess/verify_outputs.py --check rasters
    python preprocess/verify_outputs.py --force-snapshot
"""

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from pipeline.config import load_config


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    passed: bool
    failures: list[tuple] = field(default_factory=list)   # (path, check, description)
    warnings: list[tuple] = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        lines = [self.summary]
        if self.failures:
            lines.append(f"FAILURES ({len(self.failures)}):")
            for path, check, desc in self.failures:
                lines.append(f"  FAIL [{check}] {path}: {desc}")
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for path, check, desc in self.warnings:
                lines.append(f"  WARN [{check}] {path}: {desc}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expected schemas
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "ndvi_monthly": [
        "aoi", "sensor", "resolution", "year", "month", "mean_ndvi", "n_valid_px",
    ],
    "ndvi_monthly_by_class": [
        "aoi", "sensor", "resolution", "year", "month", "land_cover",
        "mean_ndvi", "n_valid_px",
    ],
    "ba_monthly": [
        "aoi", "resolution", "year", "month", "burned_km2",
        "ba_mean", "ba_lower_ci", "ba_upper_ci", "n_years",
    ],
    "ba_daily": ["aoi", "resolution", "year", "date", "burned_km2"],
    "ndvi_anomaly_monthly": [
        "aoi", "sensor", "resolution", "year", "month", "land_cover",
        "anomaly_value", "mean_ndvi", "hist_mean",
    ],
}

_FLOAT_COLUMNS: dict[str, list[str]] = {
    "ndvi_monthly":          ["mean_ndvi"],
    "ndvi_monthly_by_class": ["mean_ndvi"],
    "ba_monthly":            ["burned_km2", "ba_mean", "ba_lower_ci", "ba_upper_ci"],
    "ba_daily":              ["burned_km2"],
    "ndvi_anomaly_monthly":  ["anomaly_value", "mean_ndvi", "hist_mean"],
}

_VALUE_RANGES: dict[str, tuple] = {
    "mean_ndvi":     (-1.0,  1.0),
    "anomaly_value": (-2.0,  2.0),
    "n_valid_px":    (0,     None),
    "burned_km2":    (0,     None),
}


# ---------------------------------------------------------------------------
# Function 1: verify_rasters
# ---------------------------------------------------------------------------

def verify_rasters(new_files: list[dict], config: dict) -> VerificationResult:
    """
    Lightweight quality checks on newly downloaded GeoTIFF rasters.
    Called immediately after GEE fetch, before any preprocessing.

    Each dict in new_files:
        path, aoi, sensor, resolution, year, month,
        data_type ('ndvi' or 'burned_area')
    """
    if not new_files:
        return VerificationResult(
            passed=True,
            summary="Raster verification: no new files to check.",
        )

    failures: list[tuple] = []
    warnings: list[tuple] = []
    outputs_root = Path(config.get("output_root", "outputs"))

    # Pre-populate shape expectations from existing files for each (aoi, resolution)
    seen_shapes: dict[tuple, tuple] = {}
    for finfo in new_files:
        key = (finfo["aoi"], finfo["resolution"])
        if key in seen_shapes:
            continue
        folder = outputs_root / finfo["aoi"] / finfo["sensor"] / f"{finfo['resolution']}m"
        if folder.exists():
            for tif in folder.glob("*.tif"):
                if str(tif) == finfo["path"]:
                    continue
                try:
                    with rasterio.open(tif) as src:
                        seen_shapes[key] = src.shape
                        break
                except Exception:
                    continue

    for finfo in new_files:
        path = finfo["path"]
        p = Path(path)
        is_ba = finfo["data_type"] == "burned_area"

        # Check 1: exists and size > 1KB
        # Burned area: no-fire months compress to ~558B (all-zero int raster) — skip size check.
        # This mirrors the same exception in scripts/deploy.py _validate_tif().
        if not p.exists():
            failures.append((path, "exists", "file not found"))
            continue
        size = p.stat().st_size
        if not is_ba and size <= 1024:
            failures.append((path, "size", f"file too small ({size} bytes)"))
            continue

        # Check 2: opens with rasterio
        try:
            with rasterio.open(path) as src:
                crs   = src.crs
                n_bands = src.count
                shape = src.shape
                nodata = src.nodata
                data  = src.read(1)
        except Exception as exc:
            failures.append((path, "readable", f"rasterio error: {exc}"))
            continue

        # Check 3: CRS is EPSG:4326
        if crs is None or crs.to_epsg() != 4326:
            failures.append((path, "crs", f"expected EPSG:4326, got {crs}"))

        # Check 4: exactly 1 band
        if n_bands != 1:
            failures.append((path, "bands", f"expected 1 band, got {n_bands}"))

        # Check 5: valid pixel coverage (NDVI only — BA months are often all-zero)
        if not is_ba:
            data_f = data.astype(np.float32)
            if nodata is not None:
                data_f = np.where(data_f == float(nodata), np.nan, data_f)
            n_total = data_f.size
            n_valid = int(np.sum(np.isfinite(data_f)))
            pct_valid = 100.0 * n_valid / n_total if n_total > 0 else 0.0
            if n_valid == 0:
                failures.append((path, "valid_pixels", "0 valid pixels (all NaN/nodata)"))
            elif pct_valid < 10.0:
                warnings.append((path, "valid_pixels",
                                 f"only {pct_valid:.1f}% valid pixels (<10% threshold)"))

        # Check 6: value range
        data_f2 = data.astype(np.float32)
        if nodata is not None:
            data_f2 = np.where(data_f2 == float(nodata), np.nan, data_f2)
        valid_vals = data_f2[np.isfinite(data_f2)]
        if len(valid_vals) > 0:
            vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
            if not is_ba:
                if vmin < -1.0 or vmax > 1.0:
                    failures.append((path, "value_range",
                                     f"NDVI out of [-1, 1]: min={vmin:.3f}, max={vmax:.3f}"))
            else:
                # For burned area (int data): BurnDate in [0, 366]; ignore NaN-converted
                ba_int = data.astype(np.int32)
                ba_valid = ba_int[ba_int > 0]
                if len(ba_valid) > 0 and (ba_valid.max() > 366):
                    failures.append((path, "value_range",
                                     f"BurnDate > 366: max={ba_valid.max()}"))

        # Check 7: dimension consistency with other files for same (aoi, resolution)
        key = (finfo["aoi"], finfo["resolution"])
        if key in seen_shapes:
            ref_h, ref_w = seen_shapes[key]
            h, w = shape
            if abs(h - ref_h) > 1 or abs(w - ref_w) > 1:
                failures.append((path, "dimensions",
                                 f"shape {shape} differs from expected {seen_shapes[key]}"
                                 " by more than 1 pixel"))
        else:
            seen_shapes[key] = shape

    n_checked = len(new_files)
    n_fail = len(failures)
    n_warn = len(warnings)
    passed = n_fail == 0
    summary = (
        f"Raster verification: {n_checked} files checked, "
        f"{n_fail} failures, {n_warn} warnings. "
        f"{'PASSED' if passed else 'FAILED'}"
    )
    return VerificationResult(passed=passed, failures=failures, warnings=warnings,
                              summary=summary)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _snapshot_path(outputs_root: str) -> Path:
    return Path(outputs_root) / "processed" / "verification_snapshot.json"


def create_verification_snapshot(outputs_root: str, n_samples: int = 5) -> None:
    """
    Sample n_samples random rows from each Parquet table and store as JSON.
    Called automatically on first run; can be forced with --force-snapshot.
    """
    rng = random.Random(42)
    processed_root = Path(outputs_root) / "processed"
    snapshot: dict = {
        "created_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "tables": {},
    }

    for parquet_file in sorted(processed_root.rglob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_file)
            if df.empty:
                continue
            n = min(n_samples, len(df))
            indices = rng.sample(range(len(df)), n)
            rows = []
            for i in indices:
                row: dict = {}
                for col in df.columns:
                    val = df.iloc[i][col]
                    if pd.isna(val):
                        row[col] = None
                    elif isinstance(val, np.integer):
                        row[col] = int(val)
                    elif isinstance(val, np.floating):
                        row[col] = float(val)
                    elif hasattr(val, "isoformat"):
                        row[col] = val.isoformat()
                    else:
                        row[col] = str(val)
                rows.append(row)
            # Key by path relative to outputs_root parent so it stays portable
            rel = str(parquet_file.relative_to(Path(outputs_root).parent))
            snapshot["tables"][rel] = rows
        except Exception as exc:
            print(f"  [snapshot] skipping {parquet_file.name}: {exc}")

    snap_path = _snapshot_path(outputs_root)
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"  Snapshot written: {snap_path} ({len(snapshot['tables'])} tables, {n_samples} rows each)")


def load_verification_snapshot(outputs_root: str) -> dict | None:
    p = _snapshot_path(outputs_root)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Function 2: verify_parquet_outputs
# ---------------------------------------------------------------------------

def verify_parquet_outputs(
    outputs_root: str,
    new_months: list[tuple],
    config: dict,
) -> VerificationResult:
    """
    Quality check on preprocessed Parquet files before deployment.
    Called after Pass A (and Pass B if it ran), before deploy.

    new_months: list of (aoi, sensor, resolution, year, month) for this run.
                Pass empty list when --skip-fetch (no new months to spot-check).
    """
    failures: list[tuple] = []
    warnings: list[tuple] = []
    processed_root = Path(outputs_root) / "processed"

    if not processed_root.exists():
        return VerificationResult(
            passed=False,
            failures=[("processed/", "exists",
                       "outputs/processed/ not found -- run Pass A first")],
            summary="Parquet verification FAILED: no processed outputs found.",
        )

    # Collect all Parquet files for each tracked table
    found_files: dict[str, list[Path]] = {
        stem: list(processed_root.rglob(f"{stem}.parquet"))
        for stem in _REQUIRED_COLUMNS
    }

    # ------------------------------------------------------------------
    # Check 1: Schema validation
    # ------------------------------------------------------------------
    for table_stem, paths in found_files.items():
        required = _REQUIRED_COLUMNS[table_stem]
        float_cols = _FLOAT_COLUMNS.get(table_stem, [])
        for path in paths:
            label = str(path.relative_to(processed_root))
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                failures.append((label, "readable", f"cannot open: {exc}"))
                continue

            missing = [c for c in required if c not in df.columns]
            if missing:
                failures.append((label, "schema_columns", f"missing columns: {missing}"))

            for col in float_cols:
                if col in df.columns and not pd.api.types.is_float_dtype(df[col]):
                    warnings.append((label, "schema_dtype",
                                     f"column '{col}' expected float, got {df[col].dtype}"))

    # ------------------------------------------------------------------
    # Check 2: New month rows present
    # ------------------------------------------------------------------
    for (aoi, sensor, res, year, month) in new_months:
        is_ba = sensor == "burned_area"
        table_stem = "ba_monthly" if is_ba else "ndvi_monthly"
        glob_pattern = (
            f"{aoi}/burned_area/{res}m/ba_monthly.parquet"
            if is_ba else
            f"{aoi}/{sensor}/{res}m/ndvi_monthly.parquet"
        )
        paths = list(processed_root.glob(glob_pattern))
        label = f"{aoi}/{sensor}/{res}m/{table_stem}.parquet"

        if not paths:
            failures.append((label, "new_month_present",
                             f"file not found for {year}-{month:02d}"))
            continue

        try:
            df = pd.read_parquet(paths[0])
            if is_ba:
                row = df[(df["aoi"] == aoi) & (df["year"] == year) & (df["month"] == month)]
            else:
                row = df[
                    (df["aoi"] == aoi) & (df["sensor"] == sensor)
                    & (df["resolution"] == res)
                    & (df["year"] == year) & (df["month"] == month)
                ]
            if row.empty:
                failures.append((label, "new_month_present",
                                 f"no row for {year}-{month:02d}"))
        except Exception as exc:
            warnings.append((label, "new_month_present", f"could not read: {exc}"))
            continue

        # Zambia_Mponda: ndvi_monthly_by_class must have rows for this month
        if not is_ba and aoi == "Zambia_Mponda":
            bc_paths = list(processed_root.glob(
                f"{aoi}/{sensor}/{res}m/ndvi_monthly_by_class.parquet"
            ))
            if bc_paths:
                try:
                    bc_df = pd.read_parquet(bc_paths[0])
                    bc_row = bc_df[
                        (bc_df["aoi"] == aoi) & (bc_df["sensor"] == sensor)
                        & (bc_df["resolution"] == res)
                        & (bc_df["year"] == year) & (bc_df["month"] == month)
                    ]
                    if bc_row.empty:
                        warnings.append((
                            f"{aoi}/{sensor}/{res}m/ndvi_monthly_by_class.parquet",
                            "new_month_by_class",
                            f"no by-class rows for {year}-{month:02d}",
                        ))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Check 3: Value ranges sensible
    # ------------------------------------------------------------------
    for table_stem, paths in found_files.items():
        for path in paths:
            label = str(path.relative_to(processed_root))
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue

            for col, (lo, hi) in _VALUE_RANGES.items():
                if col not in df.columns:
                    continue
                series = df[col].dropna()
                if series.empty:
                    continue
                if lo is not None and (series < lo).any():
                    failures.append((label, "value_range",
                                     f"'{col}' has values below {lo}: min={series.min():.4f}"))
                if hi is not None and (series > hi).any():
                    failures.append((label, "value_range",
                                     f"'{col}' has values above {hi}: max={series.max():.4f}"))

            # Flag entirely NaN columns for key metrics
            for col in ("mean_ndvi", "burned_km2", "anomaly_value"):
                if col in df.columns and df[col].isna().all():
                    warnings.append((label, "all_nan", f"column '{col}' is entirely NaN"))

    # ------------------------------------------------------------------
    # Check 4: Regression against snapshot
    # ------------------------------------------------------------------
    snapshot = load_verification_snapshot(outputs_root)
    if snapshot is None:
        print("  [verify] No snapshot found — creating baseline snapshot...")
        create_verification_snapshot(outputs_root)
        print("  [verify] Snapshot created. Regression check skipped on first run.")
    else:
        repo_root = Path(outputs_root).parent
        for snap_rel, snap_rows in snapshot["tables"].items():
            full_path = repo_root / snap_rel
            if not full_path.exists():
                continue
            try:
                df = pd.read_parquet(full_path)
            except Exception:
                continue

            for snap_row in snap_rows:
                # Build key-column filter — covers all tables including ba_daily, delta, resilience
                _KEY_CANDIDATES = (
                    "aoi", "sensor", "resolution",
                    "year", "year_a", "year_b", "anomaly_year",
                    "month", "land_cover", "date",
                )
                key_cols = [
                    c for c in _KEY_CANDIDATES
                    if c in snap_row and snap_row[c] is not None and c in df.columns
                ]
                if not key_cols:
                    continue

                mask = pd.Series([True] * len(df), index=df.index)
                for kc in key_cols:
                    try:
                        col_dtype = df[kc].dtype
                        if col_dtype.kind == "M":  # datetime column (e.g. ba_daily.date)
                            mask &= df[kc] == pd.Timestamp(snap_row[kc])
                        elif col_dtype == object:
                            mask &= df[kc] == str(snap_row[kc])
                        else:
                            mask &= df[kc] == int(snap_row[kc])
                    except (TypeError, ValueError):
                        pass

                matched = df[mask]
                if matched.empty:
                    continue

                # Compare float values within 0.001 tolerance
                for col, snap_val in snap_row.items():
                    if snap_val is None or col not in df.columns:
                        continue
                    if not pd.api.types.is_float_dtype(df[col]):
                        continue
                    try:
                        actual = float(matched[col].iloc[0])
                        expected = float(snap_val)
                        if abs(actual - expected) > 0.001:
                            try:
                                rel_label = str(full_path.relative_to(processed_root))
                            except ValueError:
                                rel_label = str(full_path)
                            failures.append((rel_label, "regression",
                                             f"'{col}' changed: was {expected:.6f}, "
                                             f"now {actual:.6f}"))
                    except (TypeError, ValueError):
                        continue

    # ------------------------------------------------------------------
    # Check 5: Pass B outputs (validate if they exist)
    # ------------------------------------------------------------------
    for aoi in ("Zambia_Mponda", "Zambia_WL"):
        frp_path = processed_root / aoi / "burned_area" / "500m" / "fire_return_period.geojson"
        if not frp_path.exists():
            continue
        try:
            import geopandas as gpd
            gdf = gpd.read_file(str(frp_path))
            if len(gdf) < 100:
                warnings.append((
                    f"{aoi}/burned_area/500m/fire_return_period.geojson",
                    "frp_features",
                    f"only {len(gdf)} features (expected >= 100)",
                ))
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                failures.append((
                    f"{aoi}/burned_area/500m/fire_return_period.geojson",
                    "frp_crs",
                    f"expected EPSG:4326, got {gdf.crs}",
                ))
        except Exception as exc:
            failures.append((
                f"{aoi}/burned_area/500m/fire_return_period.geojson",
                "frp_readable",
                f"cannot read: {exc}",
            ))

    n_fail = len(failures)
    n_warn = len(warnings)
    passed = n_fail == 0
    summary = (
        f"Parquet verification: "
        f"{n_fail} failures, {n_warn} warnings. "
        f"{'PASSED' if passed else 'FAILED'}"
    )
    return VerificationResult(passed=passed, failures=failures, warnings=warnings,
                              summary=summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify pipeline outputs (rasters and/or Parquet files)."
    )
    parser.add_argument(
        "--check", choices=["rasters", "parquets", "all"], default="all",
        help="What to check (default: all)",
    )
    parser.add_argument("--aoi", default=None, help="Limit raster check to one AoI")
    parser.add_argument(
        "--force-snapshot", action="store_true",
        help="Recreate the verification snapshot from current Parquet values",
    )
    args = parser.parse_args()

    cfg = load_config()
    outputs_root = cfg["output_root"]

    if args.force_snapshot:
        print("Recreating verification snapshot...")
        create_verification_snapshot(outputs_root)

    if args.check in ("rasters", "all"):
        print("\n=== Checking rasters ===")
        root = Path(outputs_root)
        all_tifs: list[dict] = []
        for tif in sorted(root.rglob("*.tif")):
            parts = tif.relative_to(root).parts
            if len(parts) < 3:
                continue
            aoi_name, sensor = parts[0], parts[1]
            if args.aoi and aoi_name != args.aoi:
                continue
            res_str = parts[2]
            try:
                res = int(res_str.rstrip("m"))
            except ValueError:
                continue
            prefix = tif.name[:7]
            if len(prefix) < 7 or prefix[4] != "-":
                continue
            try:
                year, month = int(prefix[:4]), int(prefix[5:7])
            except ValueError:
                continue
            all_tifs.append({
                "path": str(tif),
                "aoi": aoi_name,
                "sensor": sensor,
                "resolution": res,
                "year": year,
                "month": month,
                "data_type": "burned_area" if "BurnedArea" in tif.name else "ndvi",
            })

        if not all_tifs:
            print("  No TIF files found.")
        else:
            result = verify_rasters(all_tifs, cfg)
            print(result)

    if args.check in ("parquets", "all"):
        print("\n=== Checking Parquet outputs ===")
        result = verify_parquet_outputs(outputs_root, [], cfg)
        print(result)
