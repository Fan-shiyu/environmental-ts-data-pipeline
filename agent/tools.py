"""Tool definitions (wrapping the Part-1 data API) + in-process dispatcher.

Tools are stored in Anthropic-ish form (name/description/input_schema); the
LLM client converts them to OpenAI tool format before the call. call_tool()
invokes the Part-1 router functions directly in-process (no HTTP), always
passing format="agent" since those functions use FastAPI Query() defaults.
"""

import json
import statistics

from api.dependencies import REPO_ROOT
from api.routers import burned_area, health, ndvi

TOOLS = [
    {
        "name": "get_available_data",
        "description": """
            Get a summary of all available data: which study areas,
            sensors, date ranges, resolutions, and land cover classes
            exist in the system. Call this first when the user asks
            what data is available, or when you need to check whether
            a specific combination exists before calling other tools.
        """,
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_ndvi_timeseries",
        "description": """
            Get monthly NDVI values for a study area and sensor,
            with historical baseline statistics for comparison.
            Use this when the user asks about vegetation trends,
            NDVI values, greenness, vegetation health over time,
            or whether this year is better or worse than usual.
            Returns one data point per month plus the historical
            typical range and long-term average.
            For land-cover-specific questions, use get_ndvi_by_landcover.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda", "Zambia_WL"]},
                "sensor": {"type": "string", "enum": ["sentinel2", "modis"]},
                "resolution": {"type": "string", "default": "auto",
                               "description": "Use 'auto' unless user specifies"},
                "start": {"type": "string", "description": "YYYY-MM format"},
                "end": {"type": "string", "description": "YYYY-MM format"},
            },
            "required": ["aoi", "sensor"],
        },
    },
    {
        "name": "get_ndvi_by_landcover",
        "description": """
            Get NDVI broken down by land cover class (Trees, Crops,
            Rangeland, etc.) for a specific year. Use this when the
            user asks which land cover type is most productive, how
            crops compare to trees, or anything about specific
            vegetation classes. Only available for Zambia_Mponda.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda"]},
                "sensor": {"type": "string", "enum": ["sentinel2", "modis"]},
                "resolution": {"type": "string", "default": "auto"},
                "year": {"type": "integer"},
            },
            "required": ["aoi", "sensor", "year"],
        },
    },
    {
        "name": "get_ndvi_anomaly",
        "description": """
            Get vegetation anomaly analysis for a specific year:
            which months had below-normal NDVI, how severe the
            deficit was, and how quickly each land cover class
            recovered. Use this when the user asks about drought,
            stress events, resilience, or unusual vegetation conditions
            in a specific year. Only available for Zambia_Mponda.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda"]},
                "sensor": {"type": "string", "enum": ["sentinel2", "modis"]},
                "resolution": {"type": "string", "default": "auto"},
                "year": {"type": "integer"},
            },
            "required": ["aoi", "sensor", "year"],
        },
    },
    {
        "name": "get_phenology",
        "description": """
            Get crop or rangeland phenological events (green-up, peak,
            senescence) per year. Use this when the user asks about
            planting seasons, harvest timing, when crops peak, or
            whether the growing season is shifting earlier or later.
            Only available for Zambia_Mponda, Crops and Rangeland classes.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda"]},
                "sensor": {"type": "string", "enum": ["sentinel2", "modis"]},
                "resolution": {"type": "string", "default": "auto"},
                "land_cover": {"type": "string", "enum": ["Crops", "Rangeland"]},
            },
            "required": ["aoi", "sensor", "land_cover"],
        },
    },
    {
        "name": "get_burned_area_summary",
        "description": """
            Get monthly burned area in km² compared to historical
            baseline. Use this when the user asks about fires, burning,
            fire season extent, or whether this year had more or less
            fire than usual. Note: data has ~3 month publication lag.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda", "Zambia_WL"]},
                "start": {"type": "string", "description": "YYYY-MM"},
                "end": {"type": "string", "description": "YYYY-MM"},
            },
            "required": ["aoi"],
        },
    },
    {
        "name": "get_burned_area_daily",
        "description": """
            Get daily burned area for a specific year showing the
            exact timing of fire events. Use this when the user asks
            about when fires peaked, fire season timing, or comparing
            fire patterns across years.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda", "Zambia_WL"]},
                "year": {"type": "integer"},
            },
            "required": ["aoi", "year"],
        },
    },
    {
        "name": "get_fire_return_period",
        "description": """
            Get the fire return period map showing how frequently
            each part of the study area burns (in years). Use this
            when the user asks about fire-prone areas, which zones
            burn most often, or fire management planning.
            Returns spatial data as a summary (not the full GeoJSON).
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda", "Zambia_WL"]},
            },
            "required": ["aoi"],
        },
    },
]


def _as_dict(obj):
    """APIResponse -> dict; pass dicts through."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _fire_return_period_summary(aoi: str) -> dict:
    """Summarise the FRP GeoJSON instead of returning the full (large) file."""
    path = (REPO_ROOT / "outputs" / "processed" / aoi / "burned_area"
            / "500m" / "fire_return_period.geojson")
    if not path.exists():
        return {"error": f"No fire return period data for {aoi}"}
    with open(path) as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    frp_vals = [
        ft["properties"].get("frp_years")
        for ft in feats
        if ft.get("properties", {}).get("frp_years") is not None
    ]
    n_years = max((ft.get("properties", {}).get("n_years", 0) for ft in feats), default=0)
    if not frp_vals:
        return {"aoi": aoi, "n_features": len(feats), "note": "no FRP values found"}
    return {
        "aoi": aoi,
        "n_features": len(feats),
        "n_years_in_dataset": int(n_years),
        "frp_years_min": round(min(frp_vals), 1),
        "frp_years_max": round(max(frp_vals), 1),
        "frp_years_mean": round(statistics.mean(frp_vals), 2),
        "interpretation": "frp_years = average years between burns; "
                          "lower means that area burns more frequently.",
    }


def call_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call to the Part-1 API (in-process). Returns a dict."""
    try:
        if name == "get_available_data":
            return health.available_data()

        if name == "get_ndvi_timeseries":
            return _as_dict(ndvi.ndvi_timeseries(
                aoi=args["aoi"], sensor=args["sensor"],
                resolution=args.get("resolution", "auto"),
                start=args.get("start"), end=args.get("end"),
                format="agent",
            ))

        if name == "get_ndvi_by_landcover":
            return _as_dict(ndvi.ndvi_by_landcover(
                aoi=args["aoi"], sensor=args["sensor"], year=int(args["year"]),
                resolution=args.get("resolution", "auto"), format="agent",
            ))

        if name == "get_ndvi_anomaly":
            return _as_dict(ndvi.ndvi_anomaly(
                aoi=args["aoi"], sensor=args["sensor"], year=int(args["year"]),
                resolution=args.get("resolution", "auto"), format="agent",
            ))

        if name == "get_phenology":
            return _as_dict(ndvi.ndvi_phenology(
                aoi=args["aoi"], sensor=args["sensor"], land_cover=args["land_cover"],
                resolution=args.get("resolution", "auto"), format="agent",
            ))

        if name == "get_burned_area_summary":
            return _as_dict(burned_area.burned_area_summary(
                aoi=args["aoi"], start=args.get("start"), end=args.get("end"),
                format="agent",
            ))

        if name == "get_burned_area_daily":
            return _as_dict(burned_area.burned_area_daily(
                aoi=args["aoi"], year=int(args["year"]), format="agent",
            ))

        if name == "get_fire_return_period":
            return _fire_return_period_summary(args["aoi"])

        return {"error": f"Unknown tool: {name}"}

    except Exception as exc:  # surface errors to the LLM rather than 500
        return {"error": f"{type(exc).__name__}: {exc}"}
