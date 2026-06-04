"""FastAPI app for the environmental-ts-data-pipeline data service.

Run from repo root:
    uvicorn api.main:app --reload --port 8000

Endpoints under /api/v1; auto-generated docs at /docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.cache import response_cache
from api.routers import burned_area, geometry, health, ndvi

app = FastAPI(
    title="SensingClues Environmental Data API",
    version="1.0.0",
    description="Read-only service serving pre-computed NDVI / burned-area "
                "Parquet tables and GeoJSON geometries to the Shiny app and AI agent.",
)

# Allow the Shiny app (and local dev) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PREFIX = "/api/v1"
app.include_router(health.router, prefix=_PREFIX)
app.include_router(ndvi.router, prefix=_PREFIX)
app.include_router(burned_area.router, prefix=_PREFIX)
app.include_router(geometry.router, prefix=_PREFIX)


@app.post("/cache/clear", tags=["cache"])
def clear_cache():
    """Clear all cache entries. Called after the pipeline deploys new data."""
    response_cache.clear()
    return {"status": "ok", "message": "Cache cleared"}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
