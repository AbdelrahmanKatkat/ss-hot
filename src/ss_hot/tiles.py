"""OAM chip name -> ground resolution -> model input scale (no CRS, no ONNX layout).

fAIr names every chip `OAM-<x>-<y>-<z>.<ext>` after its Web-Mercator XYZ tile, so the zoom `z` and the tile row `y`
(hence the latitude) give the ground resolution even for a plain PNG/JPEG without georeferencing. A dataset whose
names do not describe the chip's real footprint passes its ground resolution explicitly (`source_gsd_m`).
"""

import math
import re

from .params import MAX_SCALE

OAM_TILE_RE = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)
WEB_MERCATOR_WIDTH_M = 40075016.68558  # width of the zoom-0 tile at the equator


def parse_oam_tile(name: str) -> tuple[int, int, int]:
    """`(x, y, z)` from an OAM chip file name; anything else raises `ValueError`."""
    match = OAM_TILE_RE.match(name)
    if match is None:
        raise ValueError(f"chip name {name!r} does not match OAM-<x>-<y>-<z>.<ext>")
    x, y, z = (int(match.group(i)) for i in (1, 2, 3))
    if z > 30 or x >= 2**z or y >= 2**z:
        raise ValueError(f"chip name {name!r}: tile ({x}, {y}) is outside the zoom-{z} grid")
    return x, y, z


def tile_gsd_m(name: str, width_px: int) -> float:
    """Ground size of one chip pixel in metres, at the latitude of the tile centre."""
    _, y, z = parse_oam_tile(name)
    lat = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / 2**z)))
    return WEB_MERCATOR_WIDTH_M * math.cos(lat) / 2**z / width_px


def chip_scale(
    name: str,
    width_px: int,
    model_gsd_m: float,
    *,
    height_px: int | None = None,
    max_input_px: int | None = None,
    source_gsd_m: float | None = None,
) -> float:
    """Factor that makes one model pixel cover `model_gsd_m` metres of ground.

    The chip's own resolution comes from its name, or from `source_gsd_m` when given. A chip that would need more
    than `MAX_SCALE` raises. With `max_input_px` the scale is then capped so the longest side of the scaled chip stays
    within it (a coarse zoom is upscaled less, trading resolution for memory). Downscaling is not limited.

    >>> round(chip_scale("OAM-524288-524288-20.tif", 256, 0.03), 2)
    4.98
    >>> round(chip_scale("OAM-1-1-22.tif", 512, 0.03, source_gsd_m=0.009), 2)
    0.3
    """
    gsd = source_gsd_m if source_gsd_m is not None else tile_gsd_m(name, width_px)
    if gsd <= 0:
        raise ValueError(f"{name}: ground resolution must be positive, got {gsd}")
    scale = gsd / model_gsd_m
    if scale > MAX_SCALE:
        raise ValueError(
            f"{name}: scale {scale:.1f} exceeds {MAX_SCALE}; the zoom is too coarse for this model "
            f"(chip pixel is {gsd:.3f} m, model wants {model_gsd_m} m)"
        )
    if max_input_px is not None:
        scale = min(scale, max_input_px / max(width_px, height_px or width_px))
    return scale
