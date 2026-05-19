"""Export logic: write NDVI composites to GeoTIFF via GEE Drive export or direct download."""

import io
import os
import zipfile
from pathlib import Path

import ee
import numpy as np
import rasterio
import rasterio.features
import requests


def _apply_polygon_mask(path: str, polygon) -> None:
    """Zero/NaN pixels outside the AoI polygon in a GeoTIFF, in-place.

    - float rasters (NDVI): outside pixels -> NaN, nodata metadata updated to NaN
    - int rasters (burned area): outside pixels -> 0
    """
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        data    = src.read(1)

    outside = rasterio.features.geometry_mask(
        [polygon.__geo_interface__],
        out_shape=data.shape,
        transform=profile["transform"],
        invert=False,
        all_touched=False,
    )

    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
        data[outside] = np.nan
        profile["nodata"] = float("nan")
        profile["dtype"] = "float32"
    else:
        data = data.copy()
        data[outside] = 0

    tmp = path + ".mask_tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    os.replace(tmp, path)


def download_image(
    image: ee.Image,
    aoi: ee.Geometry,
    output_path: str,
    scale: int = 100,
    crs: str = 'EPSG:4326',
    mask_polygon=None,
) -> None:
    """Use image.getDownloadURL to fetch the GeoTIFF directly.

    Creates parent directory if missing. If the response is a ZIP,
    extracts the TIF inside. On image-too-large errors, raises with
    a clear message pointing to the Drive fallback.

    mask_polygon: optional Shapely geometry. When provided, pixels outside
    the polygon are zeroed (int) or set to NaN (float) after download.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    download_params = {
        "scale": scale,
        "crs": crs,
        "region": aoi,
        "format": "GEO_TIFF",
    }

    try:
        url = image.getDownloadURL(download_params)
        print("  Download URL obtained. Fetching ...")
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        raw = response.content
    except Exception as exc:
        raise RuntimeError(
            f"getDownloadURL failed: {exc}\n\n"
            "If the image is too large, use ee.batch.Export.image.toDrive instead:\n"
            f"  task = ee.batch.Export.image.toDrive(\n"
            f"      image=image, description='{out.stem}',\n"
            f"      folder='GEE_exports', fileNamePrefix='{out.stem}',\n"
            f"      scale={scale}, crs='{crs}', region=aoi, maxPixels=1e9)\n"
            f"  task.start()\n"
        ) from exc

    if zipfile.is_zipfile(io.BytesIO(raw)):
        print("  Response is a ZIP archive -- extracting GeoTIFF ...")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                raise RuntimeError("ZIP from GEE contains no .tif files.")
            raw = zf.read(tif_names[0])
            print(f"  Extracted: {tif_names[0]}")

    out.write_bytes(raw)
    print(f"  Saved to: {out}")

    if mask_polygon is not None:
        _apply_polygon_mask(str(out), mask_polygon)
        print("  Polygon mask applied.\n")
    else:
        print()
