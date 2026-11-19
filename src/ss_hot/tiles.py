"""OAM chip name -> ground resolution -> model input scale (no CRS, no ONNX layout).

fAIr names every chip `OAM-<x>-<y>-<z>.<ext>` after its Web-Mercator XYZ tile, so the zoom `z` and the tile row `y`
(hence the latitude) give the ground resolution.
"""

import math
import re

from .params import MAX_SCALE, SCALE_TOLERANCE

OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)
WEB_MERCATOR_WIDTH_M = 40075016.68558  # width of the zoom-0 tile at the equator


def parse_oam_tile(name: str) -> tuple[int, int, int]:
    """`(x, y, z)` from an OAM tile file name; anything else raises `ValueError`."""
    match = OAM_TILE_RE.match(name)
    if match is None:
        raise ValueError(f"tile name {name!r} does not match OAM-<x>-<y>-<z>.<ext>")
    x, y, z = (int(match.group(i)) for i in (1, 2, 3))
    if z > 30 or x >= 2**z or y >= 2**z:
        raise ValueError(f"tile name {name!r}: tile ({x}, {y}) is outside the zoom-{z} grid")
    return x, y, z


def tile_gsd_m(name: str, width_px: int) -> float:
    """Ground size of one tile pixel in metres, at the latitude of the tile centre."""
    _, y, z = parse_oam_tile(name)
    lat = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / 2**z)))
    return WEB_MERCATOR_WIDTH_M * math.cos(lat) / 2**z / width_px


def chip_scale(name: str, width_px: int, model_gsd_m: float) -> float:
    """Factor that makes one model pixel cover `model_gsd_m` metres of ground (1.0 within `SCALE_TOLERANCE`).

    A tile coarser than the model resolution is upscaled; one that would need more than `MAX_SCALE` raises.

    >>> chip_scale("OAM-14246392-8787682-24.tif", 256, 0.0092)  # a training tile: used as it is
    1.0
    >>> round(chip_scale("OAM-3561598-2196920-22.tif", 256, 0.0092), 2)  # zoom 22, Timor-Leste: upscaled
    4.01
    """
    scale = tile_gsd_m(name, width_px) / model_gsd_m
    if abs(scale - 1.0) <= SCALE_TOLERANCE:
        return 1.0
    if scale > MAX_SCALE:
        raise ValueError(
            f"{name}: scale {scale:.1f} exceeds {MAX_SCALE}; the zoom is too coarse for this model "
            f"(tile pixel is {scale * model_gsd_m:.4f} m, model wants {model_gsd_m} m)"
        )
    return scale
