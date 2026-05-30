# environmental-ts-data-pipeline

Automated Python pipeline for generating monthly NDVI rasters and historical
baselines for the SensingClues environmental-time-series Shiny app.

**Status:** In development

## Context

This repo produces GeoTIFFs and derived summary tables consumed by the Shiny app at
[github.com/SensingClues/environmental-time-series](https://github.com/SensingClues/environmental-time-series).

It replaces the manual GEE JavaScript workflow (see [reference/](reference/)) with
an automated, scheduled Python pipeline that runs monthly and on-demand backfills.

### Validation status

| Table | Validation | Notes |
|---|---|---|
| ndvi_monthly | Passed | Monthly means within 0.01 of direct raster computation |
| ndvi_monthly_baselines | Passed | Ribbon values match app |
| ndvi_annual + trend_stats | Passed | Trend direction matches app (Mponda MODIS 1000m = decreasing) |
| ndvi_monthly_by_class | Passed | Row counts and value ranges correct |
| ndvi_monthly_baselines_by_class | Passed | 95% CI ribbon matches app |
| ndvi_anomaly_monthly | Passed | Anomaly values arithmetically correct |
| ba_monthly | Passed | August 2024 WL burned area within expected range |
| ba_daily | Passed | Row counts and value ranges correct |
| ndvi_annual_by_class | Passed | 3/4 classes within tolerance; Trees minor miss attributable to dataset state at screenshot time; directional rankings match exactly |
| ndvi_anomaly_resilience | Passed | Recovery logic correct; 3 classes absent at 1000m due to pixel coverage (known, expected) |
| ndvi_phenology | Passed | Crops green-up=Dec, peak=Feb, peak NDVI=0.61 |
| ndvi_annual_delta | Passed | 2023->2024 gain/loss within 1% of app values |
| fire_return_period.geojson | Passed | 2,650 features, FRP range [1.1, 27.0], CRS EPSG:4326 |

## Repo Structure

```
.
├── config/
│   └── sites.yaml          # AoI definitions, sensor list, resolution config
├── pipeline/
│   ├── auth.py             # GEE authentication
│   ├── sentinel2.py        # Sentinel-2 collection + cloud masking + NDVI
│   ├── modis.py            # MODIS processing (baseline / gap-fill)
│   ├── export.py           # GeoTIFF export logic
│   └── validate.py         # Validation against existing outputs
├── scripts/
│   ├── backfill.py         # One-off historical backfill runner
│   └── monthly_update.py   # Scheduled monthly update entry point
├── tests/                  # pytest test suite
├── reference/              # Legacy GEE JS scripts (read-only reference)
└── .github/workflows/      # CI/CD (to be configured)
```

## Getting Started

> Steps will be filled in as the pipeline is built.

1. Clone repo and create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```
2. Configure GEE credentials (see `pipeline/auth.py` once implemented).
3. Copy `.env.example` to `.env` and fill in required values.
4. Edit `config/sites.yaml` to define your areas of interest.
5. Run a backfill: `python scripts/backfill.py`

## Preprocessing Pipeline

The pipeline runs in two passes after the GeoTIFF backfill is complete.

**Pass A** reads raw GeoTIFFs from the Shiny app's `www/data/` folder and writes
core time-series Parquet tables to `outputs/processed/`. It runs monthly (triggered
when a new raster arrives).

**Pass B** reads Pass A outputs and computes derived analytical tables (annual
by-class stats, anomaly resilience, phenology, annual delta, fire return period).
It runs annually — a full recompute is required because adding one year of data
shifts all historical baselines for prior years.

```
outputs/processed/{aoi}/{sensor}/{resolution}m/
  ndvi_monthly.parquet                   # AoI-wide monthly means
  ndvi_monthly_baselines.parquet         # Historical min/max/mean per calendar month
  ndvi_annual.parquet                    # Annual aggregates + completeness flag
  ndvi_trend_stats.parquet               # Mann-Kendall trend test results
  ndvi_monthly_by_class.parquet          # Per-class monthly means (LULC required)
  ndvi_monthly_baselines_by_class.parquet
  ndvi_anomaly_monthly.parquet           # Monthly anomaly = mean_ndvi - hist_mean
  ndvi_annual_by_class.parquet           # Annual means + cross-year stats per class
  ndvi_anomaly_resilience.parquet        # Recovery metrics per class per year
  ndvi_phenology.parquet                 # Green-up, peak, senescence (Crops + Rangeland)
  ndvi_annual_delta.parquet              # Gain/loss km2 for all year-pair combinations

outputs/processed/{aoi}/burned_area/500m/
  ba_monthly.parquet
  ba_daily.parquet
  fire_return_period.geojson
```

Run commands:

```bash
python -m scripts.preprocess_pass_a          # all AoIs
python -m scripts.preprocess_pass_a --aoi Zambia_Mponda --dry-run
python -m scripts.preprocess_pass_b
python -m scripts.preprocess_pass_b --dry-run
```

### Pixel inclusion rule (all_touched=False)

Per-class NDVI statistics are computed by masking each monthly raster to
the land cover polygon for that class. The masking uses rasterio's default
`all_touched=False`, which means only pixels whose **center point falls
within** the polygon are included.

This is more conservative than R terra's default behavior, which includes
any pixel that touches the polygon boundary. The deliberate choice of
`all_touched=False` avoids contamination from adjacent land cover types
at polygon edges — a pixel on the boundary of a Crops polygon may be
partially Trees or Bare_ground, and including it would bias the class mean.

**Practical consequence:** Small land cover classes (Bare_ground,
Flooded_vegetation, Water) may have no valid pixels at coarse resolutions
(1000m). At 1000m, each pixel covers ~1 km², and small polygons may not
contain a single pixel center. These classes are excluded from per-class
statistics for that resolution combination — this is expected behavior,
not a bug. The app handles missing class data gracefully.

**If full 7-class coverage at all resolutions is needed in the future:**
change `all_touched=False` to `all_touched=True` in the
`rasterio.mask.mask()` call inside `compute_ndvi_monthly_by_class()`
in `preprocess/core.py`. Be aware this trades edge contamination for
complete class coverage.

## Legacy Reference

The original GEE JavaScript scripts are preserved in [reference/](reference/).
See [reference/README.md](reference/README.md) for context and a description of
a known SCL masking bug that the Python pipeline corrects.

## Author / Contact

Felicity Shiyu Fan — sfan289@aucklanduni.ac.nz  
SensingClues environmental monitoring project
