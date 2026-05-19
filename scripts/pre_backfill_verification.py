"""
Pre-backfill verification: confirm polygon-clipping fix is correct and all
regression baselines hold before launching the full historical backfill.

Three checks:
  1. NDVI mean shifts at Zambia_WL AoI boundaries (pre vs post polygon fix)
  2. Pass 1/2 regression baselines across all sensor/resolution combos
  3. Different-month spot check (June 2022) outside previously tested months

Run with:
    python scripts/pre_backfill_verification.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import json
from shapely.geometry import shape

# -- helpers -------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> str:
    """Run a subprocess with PYTHONPATH=. and return combined stdout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    print(f"  Running: {' '.join(cmd[2:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0 and "ERROR" in result.stdout:
        print(f"  WARNING: {label} exited with code {result.returncode}")
    return result.stdout + result.stderr


def _validation_cmd(aoi: str, sensor: str, resolution: int, year: int, month: int) -> list[str]:
    return [
        sys.executable, "scripts/test_mponda_validation.py",
        "--aoi", aoi,
        "--sensor", sensor,
        "--resolution", str(resolution),
        "--year", str(year),
        "--month", str(month),
    ]


def _parse_corr_mad(output: str) -> tuple[float | None, float | None]:
    """Extract Pearson correlation and mean absolute difference from validation output."""
    corr = mad = None
    m = re.search(r"Pearson correlation\s+([\d.]+)", output)
    if m:
        corr = float(m.group(1))
    m = re.search(r"Mean absolute difference\s+([\d.]+)", output)
    if m:
        mad = float(m.group(1))
    return corr, mad


def _load_polygon(geojson_path: str):
    with open(geojson_path) as f:
        gj = json.load(f)
    features = gj.get("features", [])
    if features:
        return shape(features[0]["geometry"])
    return shape(gj.get("geometry", gj))


def _is_pre_fix(path: str, polygon) -> bool:
    """Return True if the raster has finite values outside the polygon (pre-fix)."""
    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nodata = src.nodata
        tr     = src.transform
        shp    = src.shape

    if nodata is not None:
        data = np.where(data == float(nodata), np.nan, data)
    data = np.where(np.abs(data) > 1e6, np.nan, data)

    outside = rasterio.features.geometry_mask(
        [polygon.__geo_interface__],
        out_shape=shp,
        transform=tr,
        invert=False,
        all_touched=False,
    )
    n_outside_with_values = int(np.isfinite(data[outside]).sum())
    return n_outside_with_values > 0


def _mean_ndvi(path: str) -> float:
    """Return nanmean of all finite pixels."""
    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        data = np.where(data == float(nodata), np.nan, data)
    data = np.where(np.abs(data) > 1e6, np.nan, data)
    return float(np.nanmean(data))


def _backfill_cmd(aoi: str, sensor: str, resolution: int, year: int) -> list[str]:
    return [
        sys.executable, "scripts/backfill.py",
        "--aoi", aoi,
        "--sensor", sensor,
        "--resolution", str(resolution),
        "--year", str(year),
    ]


# -- constants -----------------------------------------------------------------

AOI_WL_GEOJSON = (
    r"C:\Users\20244650\Documents\GitHub\environmental-time-series"
    r"\app\www\data\AoI\AoI_Zambia_WL.geojson"
)

# Reference means from Pass 2/3 (pre-fix, bbox clip)
PASS_REF = {
    ("sentinel2", 100):  0.503,
    ("sentinel2", 1000): 0.503,
    ("modis",     250):  0.79,
    ("modis",     500):  0.79,
    ("modis",     1000): 0.79,
}

WL_SENSORS = [
    ("sentinel2", 100,  "outputs/Zambia_WL/sentinel2/100m/2024-03_NDVI_Zambia_WL.tif"),
    ("sentinel2", 1000, "outputs/Zambia_WL/sentinel2/1000m/2024-03_NDVI_Zambia_WL.tif"),
    ("modis",     250,  "outputs/Zambia_WL/modis/250m/2024-03_NDVI_Zambia_WL.tif"),
    ("modis",     500,  "outputs/Zambia_WL/modis/500m/2024-03_NDVI_Zambia_WL.tif"),
    ("modis",     1000, "outputs/Zambia_WL/modis/1000m/2024-03_NDVI_Zambia_WL.tif"),
]

# Check 2: (aoi, sensor, res, year, month, expected_corr, expected_mad_or_None)
REGRESSION_CASES = [
    ("Zambia", "sentinel2", 100,  2024, 3, 0.9717, 0.0412),
    ("Zambia", "sentinel2", 100,  2020, 9, 0.9478, 0.0098),
    ("Zambia", "sentinel2", 1000, 2024, 3, 0.9548, None),
    ("Zambia", "modis",     250,  2024, 3, 0.977,  0.0217),
    ("Zambia", "modis",     500,  2024, 3, 0.979,  0.0205),
    ("Zambia", "modis",     1000, 2024, 3, 0.963,  0.0184),
]

CORR_TOL = 0.005
MAD_TOL  = 0.005

# ==============================================================================
print("=" * 60)
print("PRE-BACKFILL VERIFICATION")
print("=" * 60)

polygon_wl = _load_polygon(AOI_WL_GEOJSON)

# ==============================================================================
print("\n" + "=" * 60)
print("Check 1: NDVI boundary shifts at Zambia_WL")
print("=" * 60)

check1_rows = []

for sensor, res, path in WL_SENSORS:
    label = f"{sensor} {res}m"
    old_mean = PASS_REF[(sensor, res)]

    if Path(path).exists() and _is_pre_fix(path, polygon_wl):
        print(f"\n  [{label}] Pre-fix file detected. Regenerating ...")
        Path(path).unlink()
        out = _run(_backfill_cmd("Zambia_WL", sensor, res, 2024), label)
        if not Path(path).exists():
            print(f"  ERROR: regeneration failed for {path}")
            check1_rows.append((sensor, res, old_mean, None, None))
            continue
        print(f"  Regenerated OK.")
    elif not Path(path).exists():
        print(f"\n  [{label}] File missing. Generating ...")
        out = _run(_backfill_cmd("Zambia_WL", sensor, res, 2024), label)
        if not Path(path).exists():
            print(f"  ERROR: generation failed for {path}")
            check1_rows.append((sensor, res, old_mean, None, None))
            continue
    else:
        print(f"\n  [{label}] Post-fix file already exists.")

    new_mean = _mean_ndvi(path)
    shift    = new_mean - old_mean
    check1_rows.append((sensor, res, old_mean, new_mean, shift))
    print(f"  Old mean: {old_mean:.3f}  New mean: {new_mean:.4f}  Shift: {shift:+.4f}")

print("\n  --- Check 1 table ---")
print(f"  {'Sensor':<12} {'Res':<8} {'Old mean':>10} {'New mean':>10} {'Shift':>8}")
print(f"  {'-'*55}")
for sensor, res, old, new, shift in check1_rows:
    old_s   = f"{old:.3f}"
    new_s   = f"{new:.4f}" if new is not None else "FAILED"
    shift_s = f"{shift:+.4f}" if shift is not None else "  N/A"
    print(f"  {sensor:<12} {str(res)+'m':<8} {old_s:>10} {new_s:>10} {shift_s:>8}")

# Verdict logic
valid_rows    = [(s, r, o, n, sh) for s, r, o, n, sh in check1_rows if sh is not None]
s2_shifts     = [sh for s, r, o, n, sh in valid_rows if s == "sentinel2"]
modis_shifts  = [sh for s, r, o, n, sh in valid_rows if s == "modis"]

anomalous = False
reasons   = []
for s, r, o, n, sh in valid_rows:
    if abs(sh) > 0.2:
        anomalous = True
        reasons.append(f"{s} {r}m shift {sh:+.4f} exceeds 0.2 threshold")
if s2_shifts and max(s2_shifts) - min(s2_shifts) > 0.05:
    # S2 shifts across resolutions going in opposite directions
    if min(s2_shifts) < 0 < max(s2_shifts):
        anomalous = True
        reasons.append(f"Sentinel-2 shifts go in opposite directions across resolutions")
if modis_shifts and min(modis_shifts) < 0 < max(modis_shifts):
    anomalous = True
    reasons.append(f"MODIS shifts go in opposite directions across resolutions")

if anomalous:
    verdict1 = "shifts look anomalous -- investigate: " + "; ".join(reasons)
else:
    verdict1 = "shifts are plausible"
print(f"\n  Verdict 1: {verdict1}")

# ==============================================================================
print("\n" + "=" * 60)
print("Check 2: Pass 1/2 regression baselines")
print("=" * 60)

check2_rows   = []
n_flagged     = 0

for aoi, sensor, res, year, month, exp_corr, exp_mad in REGRESSION_CASES:
    label = f"{aoi}/{sensor}/{res}m/{year}-{month:02d}"
    print(f"\n  [{label}]")
    cmd = _validation_cmd(aoi, sensor, res, year, month)
    out = _run(cmd, label)
    corr, mad = _parse_corr_mad(out)

    corr_ok = mad_ok = True
    corr_flag = mad_flag = ""
    if corr is None:
        corr_ok = False
        corr_flag = "PARSE_FAIL"
    elif abs(corr - exp_corr) > CORR_TOL:
        corr_ok = False
        corr_flag = f"FLAGGED (exp {exp_corr:.4f}, got {corr:.4f}, diff {corr-exp_corr:+.4f})"
        n_flagged += 1

    if exp_mad is not None:
        if mad is None:
            mad_ok = False
            mad_flag = "PARSE_FAIL"
        elif abs(mad - exp_mad) > MAD_TOL:
            mad_ok = False
            mad_flag = f"FLAGGED (exp {exp_mad:.4f}, got {mad:.4f}, diff {mad-exp_mad:+.4f})"
            n_flagged += 1

    corr_str = f"{corr:.4f}" if corr is not None else "N/A"
    mad_str  = f"{mad:.4f}"  if mad  is not None else "N/A"
    corr_disp = corr_str if corr_ok else f"{corr_str} {corr_flag}"
    mad_disp  = mad_str  if mad_ok  else f"{mad_str} {mad_flag}"
    print(f"    Corr: {corr_disp}   MAD: {mad_disp}")
    check2_rows.append((label, exp_corr, exp_mad, corr, mad, corr_ok, mad_ok))

print("\n  --- Check 2 table ---")
print(f"  {'Case':<30} {'Exp corr':>9} {'Got corr':>9} {'Exp MAD':>8} {'Got MAD':>8} {'OK?':>5}")
print(f"  {'-'*75}")
for label, ec, em, gc, gm, co, mo in check2_rows:
    gc_s = f"{gc:.4f}" if gc is not None else "N/A"
    gm_s = f"{gm:.4f}" if gm is not None else "N/A"
    em_s = f"{em:.4f}" if em is not None else " n/a"
    ok   = "OK" if co and mo else "FLAG"
    print(f"  {label:<30} {ec:>9.4f} {gc_s:>9} {em_s:>8} {gm_s:>8} {ok:>5}")

if n_flagged == 0:
    verdict2 = "all regressions hold"
else:
    verdict2 = f"{n_flagged} row(s) shifted unexpectedly"
print(f"\n  Verdict 2: {verdict2}")

# ==============================================================================
print("\n" + "=" * 60)
print("Check 3: Different-month spot check (Zambia, sentinel2, 100m, Jun 2022)")
print("=" * 60)

cmd3 = _validation_cmd("Zambia", "sentinel2", 100, 2022, 6)
out3 = _run(cmd3, "sentinel2/100m/2022-06")
corr3, mad3 = _parse_corr_mad(out3)

print(f"\n  Correlation: {corr3:.4f}" if corr3 else "\n  Correlation: N/A (parse failed)")
print(f"  MAD:         {mad3:.4f}"  if mad3  else "  MAD:         N/A")

# Print key lines from the output for context
for line in out3.splitlines():
    if any(kw in line for kw in ["correlation", "absolute diff", "Valid pixels", "Overlap"]):
        print(f"  {line.strip()}")

if corr3 is None:
    verdict3 = "results look unexpected (parse failed)"
elif corr3 >= 0.9:
    verdict3 = f"validates cleanly (corr={corr3:.4f} >= 0.9)"
else:
    verdict3 = f"results look unexpected (corr={corr3:.4f} < 0.9 threshold)"
print(f"\n  Verdict 3: {verdict3}")

# ==============================================================================
print("\n" + "=" * 60)
print("=== PRE-BACKFILL VERIFICATION ===")
print("=" * 60)
print(f"Check 1 (NDVI boundary shifts): {verdict1}")
print(f"Check 2 (Pass 1/2 regression):  {verdict2}")
print(f"Check 3 (different month):      {verdict3}")

all_ok = (
    "plausible" in verdict1
    and "all regressions" in verdict2
    and "cleanly" in verdict3
)

print()
if all_ok:
    print("Overall: READY for full historical backfill")
else:
    print("Overall: NOT READY -- issues to address")
    issues = []
    if "plausible" not in verdict1:
        issues.append(f"Check 1: {verdict1}")
    if "all regressions" not in verdict2:
        issues.append(f"Check 2: {verdict2}")
    if "cleanly" not in verdict3:
        issues.append(f"Check 3: {verdict3}")
    for issue in issues:
        print(f"  - {issue}")
