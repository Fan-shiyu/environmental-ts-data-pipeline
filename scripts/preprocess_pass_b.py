"""
Preprocessing Pipeline Pass B -- Derived Analytical Tables

Reads Pass A outputs from outputs/processed/ to compute higher-level
analytical tables. Always a full recompute -- no skip-if-exists logic,
because adding one year of data shifts all anomaly/resilience/phenology
values for prior years too.

Tables:
  B1. ndvi_annual_by_class.parquet      (LULC required; S2 + MODIS)
  B2. ndvi_anomaly_resilience.parquet   (LULC required; S2 + MODIS)
  B3. ndvi_phenology.parquet            (LULC required; Crops + Rangeland only)
  B4. ndvi_annual_delta.parquet         (all AoIs; S2 + MODIS)
  B5. fire_return_period.geojson        (burned_area; one file per AoI)

Usage:
    python -m scripts.preprocess_pass_b
    python -m scripts.preprocess_pass_b --aoi Zambia_Mponda
    python -m scripts.preprocess_pass_b --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from pipeline.config import load_config
from preprocess.core import (
    compute_anomaly_resilience,
    compute_annual_delta,
    compute_fire_return_period,
    compute_ndvi_annual_by_class,
    compute_phenology,
)
from preprocess.validate import (
    validate_anomaly_resilience,
    validate_annual_delta,
    validate_frp,
    validate_phenology,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir(aoi: str, sensor: str, resolution: int, config: dict) -> Path:
    return Path(config["output_root"]) / "processed" / aoi / sensor / f"{resolution}m"


def _combos(config: dict, aoi_filter: str | None) -> list[tuple[str, str, int]]:
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


def _write(df: pd.DataFrame, path: Path, label: str) -> bool:
    if df.empty:
        print(f"  [skip] {label}: 0 rows (no data)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  {label}: {len(df)} rows -> {path.relative_to(Path('.'))}")
    return True


def _check_pass_a(out_dir: Path, sensor: str) -> bool:
    name = "ba_monthly.parquet" if sensor == "burned_area" else "ndvi_monthly.parquet"
    p = out_dir / name
    if not p.exists():
        print(f"  ERROR: {name} not found. Run preprocess_pass_a.py first.")
        return False
    return True


def _has_lulc(processed_root: str, aoi: str, sensor: str, resolution: int) -> bool:
    return (
        Path(processed_root) / aoi / sensor / f"{resolution}m" / "ndvi_monthly_by_class.parquet"
    ).exists()


# ---------------------------------------------------------------------------
# Dry-run estimates (reads existing Parquet metadata, no computation)
# ---------------------------------------------------------------------------

def _dry_run_ndvi_combo(aoi: str, sensor: str, resolution: int, out_dir: Path, processed_root: str) -> None:
    annual_path = out_dir / "ndvi_annual.parquet"
    has_lulc = _has_lulc(processed_root, aoi, sensor, resolution)

    n_years = n_complete = 0
    if annual_path.exists():
        tmp = pd.read_parquet(annual_path, columns=["year", "is_complete"])
        n_years = tmp["year"].nunique()
        n_complete = int(tmp["is_complete"].sum())

    if has_lulc:
        print(f"  [dry] ndvi_annual_by_class.parquet:    ~{n_years * 7} rows ({n_years} yrs x 7 classes)")
        print(f"  [dry] ndvi_anomaly_resilience.parquet: ~{n_complete * 7} rows ({n_complete} complete yrs x 7 classes)")
        print(f"  [dry] ndvi_phenology.parquet:          ~{n_years * 2} rows ({n_years} yrs x 2 classes)")
    else:
        print(f"  [dry] ndvi_annual_by_class, ndvi_anomaly_resilience, ndvi_phenology: skip (no LULC for {aoi})")

    n_pairs = n_complete * (n_complete - 1)
    print(f"  [dry] ndvi_annual_delta.parquet:       ~{n_pairs} rows ({n_complete} complete yrs, {n_pairs} pairs)")


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(config: dict, aoi_filter: str | None, dry_run: bool) -> None:
    combos = _combos(config, aoi_filter)
    processed_root = str(Path(config["output_root"]) / "processed")

    written = 0
    seen_frp_aois: set[str] = set()

    for aoi, sensor, resolution in combos:
        out_dir = _output_dir(aoi, sensor, resolution, config)
        print(f"\n[{aoi} / {sensor} / {resolution}m]")

        if sensor in ("sentinel2", "modis"):
            if not dry_run and not _check_pass_a(out_dir, sensor):
                continue

            if dry_run:
                _dry_run_ndvi_combo(aoi, sensor, resolution, out_dir, processed_root)
                continue

            # B1: ndvi_annual_by_class
            df1 = compute_ndvi_annual_by_class(aoi, sensor, resolution, processed_root)
            if _write(df1, out_dir / "ndvi_annual_by_class.parquet", "ndvi_annual_by_class.parquet"):
                written += 1

            # B2: ndvi_anomaly_resilience
            df2 = compute_anomaly_resilience(aoi, sensor, resolution, processed_root)
            if _write(df2, out_dir / "ndvi_anomaly_resilience.parquet", "ndvi_anomaly_resilience.parquet"):
                written += 1

            # B3: ndvi_phenology
            df3 = compute_phenology(aoi, sensor, resolution, processed_root)
            if _write(df3, out_dir / "ndvi_phenology.parquet", "ndvi_phenology.parquet"):
                written += 1

            # B4: ndvi_annual_delta (all AoIs, reads raw TIFs)
            df4 = compute_annual_delta(aoi, sensor, resolution, processed_root, config)
            if _write(df4, out_dir / "ndvi_annual_delta.parquet", "ndvi_annual_delta.parquet"):
                written += 1

        elif sensor == "burned_area":
            if aoi in seen_frp_aois:
                continue

            frp_path = (
                Path(processed_root) / aoi / "burned_area" / "500m" / "fire_return_period.geojson"
            )

            if dry_run:
                print(f"  [dry] fire_return_period.geojson: (vectorized FRP polygons per pixel)")
                seen_frp_aois.add(aoi)
                continue

            if not _check_pass_a(out_dir, sensor):
                seen_frp_aois.add(aoi)
                continue

            gdf = compute_fire_return_period(aoi, config, processed_root)
            if not gdf.empty:
                frp_path.parent.mkdir(parents=True, exist_ok=True)
                gdf.to_file(str(frp_path), driver="GeoJSON")
                print(f"  fire_return_period.geojson: {len(gdf)} features -> {frp_path.relative_to(Path('.'))}")
                written += 1
            else:
                print(f"  [skip] fire_return_period.geojson: 0 features (no data)")

            seen_frp_aois.add(aoi)

    suffix = "[DRY RUN] " if dry_run else ""
    print(f"\n{suffix}Done: {written} tables/files written.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation() -> None:
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)
    all_pass = True

    # 1. Anomaly resilience — Mponda S2 1000m, 2024
    res_path = Path("outputs/processed/Zambia_Mponda/sentinel2/1000m/ndvi_anomaly_resilience.parquet")
    if res_path.exists():
        r = validate_anomaly_resilience(str(res_path))
        status = "PASS" if r["passes"] else "FAIL"
        print(f"\n[{status}] Anomaly resilience (Mponda/S2/1000m, 2024)")
        for k, v in r.items():
            if k not in ("passes", "notes") and isinstance(v, dict):
                mark = "+" if v.get("passes") else "X"
                print(f"  {k}: expected={v.get('expected')}  actual={v.get('actual')}  {mark}")
        print(f"  Notes: {r['notes']}")
        if not r["passes"]:
            all_pass = False
    else:
        print(f"\n[SKIP] Anomaly resilience -- {res_path} not found")

    # 2. Phenology — Mponda S2 1000m, Crops
    pheno_path = Path("outputs/processed/Zambia_Mponda/sentinel2/1000m/ndvi_phenology.parquet")
    if pheno_path.exists():
        r = validate_phenology(str(pheno_path))
        status = "PASS" if r["passes"] else "FAIL"
        print(f"\n[{status}] Phenology (Mponda/S2/1000m, Crops)")
        for k, v in r.items():
            if k not in ("passes", "notes") and isinstance(v, dict):
                mark = "+" if v.get("passes") else "X"
                print(f"  {k}: expected={v.get('expected')}  actual={v.get('actual')}  {mark}")
        print(f"  Notes: {r['notes']}")
        if not r["passes"]:
            all_pass = False
    else:
        print(f"\n[SKIP] Phenology -- {pheno_path} not found")

    # 3. Annual delta — Mponda MODIS 250m, 2023->2024
    delta_path = Path("outputs/processed/Zambia_Mponda/modis/250m/ndvi_annual_delta.parquet")
    if delta_path.exists():
        r = validate_annual_delta(str(delta_path))
        status = "PASS" if r["passes"] else "FAIL"
        print(f"\n[{status}] Annual delta (Mponda/MODIS/250m, 2023->2024)")
        print(f"  gain_km2 : {r.get('gain_km2')}  (expected ~{r.get('gain_expected')})")
        print(f"  loss_km2 : {r.get('loss_km2')}  (expected ~{r.get('loss_expected')})")
        print(f"  Notes    : {r['notes']}")
        if not r["passes"]:
            all_pass = False
    else:
        print(f"\n[SKIP] Annual delta -- {delta_path} not found")

    # 4. FRP — Zambia_WL
    frp_path = Path("outputs/processed/Zambia_WL/burned_area/500m/fire_return_period.geojson")
    if frp_path.exists():
        r = validate_frp(str(frp_path))
        status = "PASS" if r["passes"] else "FAIL"
        print(f"\n[{status}] FRP GeoJSON (Zambia_WL)")
        print(f"  n_features : {r.get('n_features')}")
        print(f"  n_years    : {r.get('n_years')}")
        print(f"  frp_range  : {r.get('frp_range')}")
        print(f"  crs        : {r.get('crs')}")
        for k, v in (r.get("checks") or {}).items():
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
        print(f"  Notes      : {r['notes']}")
        if not r["passes"]:
            all_pass = False
    else:
        print(f"\n[SKIP] FRP -- {frp_path} not found")

    print(f"\n{'All validations PASSED.' if all_pass else 'Some validations FAILED -- review above.'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Preprocessing Pipeline Pass B -- compute derived analytical tables."
)
parser.add_argument("--aoi",         default=None,        help="Process only this AoI (e.g. Zambia_Mponda)")
parser.add_argument("--dry-run",     action="store_true", help="Show expected row counts without computing")
parser.add_argument("--no-validate", action="store_true", help="Skip validation step")
args = parser.parse_args()

config = load_config()

if args.dry_run:
    print("DRY RUN -- no files will be written\n")

run(config, aoi_filter=args.aoi, dry_run=args.dry_run)

if not args.dry_run and not args.no_validate:
    run_validation()
