# environmental-ts-data-pipeline

Automated Python pipeline for generating monthly NDVI rasters and historical
baselines for the SensingClues environmental-time-series Shiny app.

**Status:** In development

## Context

This repo produces GeoTIFFs and derived summary tables consumed by the Shiny app at
[github.com/SensingClues/environmental-time-series](https://github.com/SensingClues/environmental-time-series).

It replaces the manual GEE JavaScript workflow (see [reference/](reference/)) with
an automated, scheduled Python pipeline that runs monthly and on-demand backfills.

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

## Legacy Reference

The original GEE JavaScript scripts are preserved in [reference/](reference/).
See [reference/README.md](reference/README.md) for context and a description of
a known SCL masking bug that the Python pipeline corrects.

## Author / Contact

Felicity Shiyu Fan — sfan289@aucklanduni.ac.nz  
SensingClues environmental monitoring project
