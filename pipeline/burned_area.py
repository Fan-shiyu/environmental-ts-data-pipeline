"""MODIS burned area: MCD64A1 BurnDate monthly images."""

import datetime

import ee

from pipeline.config import load_config


def monthly_image(aoi: ee.Geometry, year: int, month: int) -> ee.Image:
    """Fetch MCD64A1 BurnDate for the given month, clip to AoI.

    Raises ValueError if no image is available (e.g. not yet published due to
    the ~2-3 month publication lag). Call latest_available_month() to find what
    is currently available.
    """
    config = load_config()
    collection_id = config["sensors"]["burned_area"]["collection"]

    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    date_start = f"{year}-{month:02d}-01"
    date_end = f"{next_year}-{next_month:02d}-01"

    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
    )

    count = collection.size().getInfo()
    print(f"  Collection: {collection_id}")
    print(f"  Date range: {date_start} to {date_end}")
    print(f"  Images in collection: {count}")

    if count == 0:
        raise ValueError(
            f"No MCD64A1 image found for {year}-{month:02d}. "
            "The product has a ~2-3 month publication lag. "
            "Use latest_available_month() to find the most recent published month."
        )

    image = collection.first().select("BurnDate").clip(aoi)
    print("  BurnDate image selected.\n")
    return image


def latest_available_month(aoi: ee.Geometry) -> tuple[int, int]:
    """Return (year, month) of the most recent MCD64A1 image covering the AoI."""
    config = load_config()
    collection_id = config["sensors"]["burned_area"]["collection"]

    img = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .sort("system:time_start", False)
        .first()
    )

    ts_ms = img.get("system:time_start").getInfo()
    dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
    return (dt.year, dt.month)
