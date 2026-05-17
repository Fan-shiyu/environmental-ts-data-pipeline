"""MODIS processing: quality masking, NDVI scaling, and monthly composites."""

import ee

from pipeline.config import load_config


def get_collection_id(resolution: int) -> str:
    """Return GEE collection ID for the given resolution (250, 500, or 1000m)."""
    config = load_config()
    resolutions = config["sensors"]["modis"]["resolutions"]
    if resolution not in resolutions:
        supported = sorted(resolutions.keys())
        raise ValueError(
            f"MODIS resolution {resolution}m not supported. Supported: {supported}"
        )
    return resolutions[resolution]["collection"]


def mask_quality(image: ee.Image) -> ee.Image:
    """Apply MODIS SummaryQA mask. Keep 0 (good) and 1 (marginal); mask 2 and 3."""
    qa = image.select("SummaryQA")
    return image.updateMask(qa.lte(1))


def scale_ndvi(image: ee.Image) -> ee.Image:
    """Convert native MODIS NDVI (int16 x10000) to float (-1.0 to 1.0)."""
    return image.select("NDVI").multiply(0.0001).rename("NDVI")


def monthly_composite(aoi: ee.Geometry, year: int, month: int, resolution: int) -> ee.Image:
    """Build monthly NDVI composite for the given AoI, month, and resolution.

    Flow: load MODIS collection for resolution, filter by bounds and date range,
    apply quality mask, scale NDVI to float, take median, clip to AoI.
    """
    collection_id = get_collection_id(resolution)

    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    date_start = f"{year}-{month:02d}-01"
    date_end = f"{next_year}-{next_month:02d}-01"

    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
    )

    print(f"  Collection: {collection_id}")
    print(f"  Date range: {date_start} to {date_end}")
    print(f"  Images in collection: {collection.size().getInfo()}")

    composite = (
        collection
        .map(mask_quality)
        .map(scale_ndvi)
        .select("NDVI")
        .median()
        .clip(aoi)
    )
    print("  NDVI composite built (median reducer, SummaryQA mask).\n")
    return composite
