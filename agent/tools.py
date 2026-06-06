"""Tool definitions (wrapping the Part-1 data API) + in-process dispatcher.

Tools are stored in Anthropic-ish form (name/description/input_schema); the
LLM client converts them to OpenAI tool format before the call. call_tool()
invokes the Part-1 router functions directly in-process (no HTTP), always
passing format="agent" since those functions use FastAPI Query() defaults.
"""

import json
import statistics

from pyproj import Geod
from shapely.geometry import shape as shapely_shape

from api.dependencies import REPO_ROOT
from api.routers import burned_area, geometry, health, ndvi

_GEOD = Geod(ellps="WGS84")

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
            Get quick overall fire-return-period statistics for the
            study area: min, max, and mean years between burns, plus
            the number of features. A single high-level snapshot.
            For 'which parts burn most often', how much AREA burns
            frequently, fire-prone zones, or the distribution of fire
            frequencies, use get_fire_return_summary instead.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda", "Zambia_WL"]},
            },
            "required": ["aoi"],
        },
    },
    {
        "name": "get_landcover_spatial_summary",
        "description": """
            Get area statistics for each land cover class: how much of
            the study area each class covers in hectares and percentage.
            Use this when the user asks about land cover distribution,
            where specific vegetation types are located, how much of the
            area is crops/trees/rangeland, or the spatial composition
            of the study area. Only available for Zambia_Mponda.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda"]},
            },
            "required": ["aoi"],
        },
    },
    {
        "name": "get_fire_return_summary",
        "description": """
            THE tool for spatial fire-frequency questions. Returns the
            AREA-WEIGHTED distribution of fire return periods (what % of
            the study area burns every 1-3 / 3-7 / 7-15 / 15+ years) plus
            mean and median FRP, over 25+ years of data.
            Use whenever the user asks which parts or zones burn most
            often, where the fire-prone areas are, how much area burns
            frequently, fire frequency patterns, or fire management
            planning. Prefer this over get_fire_return_period for any
            'where' / 'which part' / 'how much area' fire question.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda", "Zambia_WL"]},
            },
            "required": ["aoi"],
        },
    },
    {
        "name": "get_ndvi_spatial_change",
        "description": """
            Get a spatial summary of NDVI change between two time periods:
            total area gaining/losing vegetation in km² and percentage.

            Two modes:
            - Annual: compare annual mean NDVI between year_a and year_b
              (omit month parameter)
            - Monthly: compare a specific month across two years
              (provide month parameter, 1-12)

            Use when the user asks where vegetation changed, which areas
            improved or declined, or how a specific month compares
            across years.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "aoi": {"type": "string", "enum": ["Zambia_Mponda", "Zambia_WL"]},
                "sensor": {"type": "string", "enum": ["sentinel2", "modis"]},
                "resolution": {"type": "string", "default": "auto"},
                "year_a": {"type": "integer", "description": "baseline year"},
                "year_b": {"type": "integer", "description": "comparison year"},
                "month": {"type": "integer", "description": "1-12. Omit for annual comparison."},
            },
            "required": ["aoi", "sensor", "year_a", "year_b"],
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


def _landcover_spatial_summary(aoi: str) -> dict:
    """Area stats per land cover class from the geometry/landcover endpoint."""
    fc = geometry.geometry_landcover(aoi=aoi, land_cover_class=None, simplified=True)
    out: dict = {}
    for ft in fc.get("features", []):
        p = ft.get("properties", {})
        cls = p.get("land_cover")
        if cls is None:
            continue
        out[cls] = {
            "area_ha": round(p.get("area_ha"), 1) if p.get("area_ha") is not None else None,
            "pct_of_study_area": p.get("pct_of_study_area"),
        }
    return out


def _fire_return_summary(aoi: str) -> dict:
    """Area-weighted distribution of fire return periods across the AoI."""
    fc = geometry.geometry_fire_return_period(aoi=aoi)
    meta = fc.get("metadata", {})
    buckets = {
        "burns_almost_every_year_1_to_3": 0.0,
        "frequent_3_to_7_years": 0.0,
        "moderate_7_to_15_years": 0.0,
        "rare_over_15_years": 0.0,
    }
    total_area = 0.0
    weighted_vals: list[tuple[float, float]] = []  # (frp_years, area)
    for ft in fc.get("features", []):
        frp = ft.get("properties", {}).get("frp_years")
        if frp is None:
            continue
        try:
            area_m2, _ = _GEOD.geometry_area_perimeter(shapely_shape(ft["geometry"]))
        except Exception:
            continue
        area = abs(area_m2)
        total_area += area
        weighted_vals.append((frp, area))
        if frp < 3:
            buckets["burns_almost_every_year_1_to_3"] += area
        elif frp < 7:
            buckets["frequent_3_to_7_years"] += area
        elif frp < 15:
            buckets["moderate_7_to_15_years"] += area
        else:
            buckets["rare_over_15_years"] += area

    if total_area == 0:
        return {"error": f"No fire return period data for {aoi}"}

    # Area-weighted mean; median by cumulative area over sorted FRP.
    mean_frp = sum(v * a for v, a in weighted_vals) / total_area
    weighted_vals.sort(key=lambda x: x[0])
    cum, median_frp = 0.0, weighted_vals[-1][0]
    for v, a in weighted_vals:
        cum += a
        if cum >= total_area / 2:
            median_frp = v
            break

    return {
        "n_years_data": meta.get("n_years"),
        "year_start": meta.get("year_start"),
        "year_end": meta.get("year_end"),
        "mean_frp_years": round(mean_frp, 2),
        "median_frp_years": round(median_frp, 1),
        "distribution": {
            k: {"pct_area": round(100 * v / total_area, 1)} for k, v in buckets.items()
        },
    }


def _ndvi_spatial_change(aoi, sensor, year_a, year_b, resolution="auto", month=None) -> dict:
    """Per-pixel NDVI delta summary between two years (annual or a given month)."""
    if month is not None:
        ga = ndvi.ndvi_monthly_grid(aoi=aoi, sensor=sensor, year=int(year_a),
                                    month=int(month), resolution=resolution)
        gb = ndvi.ndvi_monthly_grid(aoi=aoi, sensor=sensor, year=int(year_b),
                                    month=int(month), resolution=resolution)
        period = f"{year_a} to {year_b} (month {int(month)})"
    else:
        ga = ndvi.ndvi_annual_grid(aoi=aoi, sensor=sensor, year=int(year_a), resolution=resolution)
        gb = ndvi.ndvi_annual_grid(aoi=aoi, sensor=sensor, year=int(year_b), resolution=resolution)
        period = f"{year_a} to {year_b}"

    for g, yr in ((ga, year_a), (gb, year_b)):
        if g.get("status") != "ok":
            return {"error": g.get("error", f"No grid data for {yr}")}

    res_m = ga["metadata"]["resolution"]
    px_area = (res_m / 1000) ** 2  # nominal km² per pixel
    va, vb = ga["grid"]["values"], gb["grid"]["values"]

    deltas = []
    for row_a, row_b in zip(va, vb):
        for a, b in zip(row_a, row_b):
            if a is not None and b is not None:
                deltas.append(b - a)

    if not deltas:
        return {"error": f"No overlapping valid pixels for {period}"}

    thr = 0.02
    n_total = len(deltas)
    gain = [d for d in deltas if d > thr]
    loss = [d for d in deltas if d < -thr]
    return {
        "period": period,
        "sensor": sensor,
        "resolution": res_m,
        "gain_km2": round(len(gain) * px_area, 2),
        "loss_km2": round(len(loss) * px_area, 2),
        "total_area_km2": round(n_total * px_area, 2),
        "pct_gaining": round(100 * len(gain) / n_total, 1),
        "pct_losing": round(100 * len(loss) / n_total, 1),
        "mean_ndvi_change": round(sum(deltas) / n_total, 4),
        "max_gain": round(max(deltas), 4),
        "max_loss": round(min(deltas), 4),
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

        if name == "get_landcover_spatial_summary":
            return _landcover_spatial_summary(args["aoi"])

        if name == "get_fire_return_summary":
            return _fire_return_summary(args["aoi"])

        if name == "get_ndvi_spatial_change":
            return _ndvi_spatial_change(
                aoi=args["aoi"], sensor=args["sensor"],
                year_a=args["year_a"], year_b=args["year_b"],
                resolution=args.get("resolution", "auto"),
                month=args.get("month"),
            )

        return {"error": f"Unknown tool: {name}"}

    except Exception as exc:  # surface errors to the LLM rather than 500
        return {"error": f"{type(exc).__name__}: {exc}"}
