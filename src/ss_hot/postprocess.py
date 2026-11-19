"""Postprocess: the model output (seagrass probability per chip) -> polygons -> GeoJSON FeatureCollection.

`postprocess_outputs` is the ready-to-use step a pack's `postprocess` calls on what `predict_session` returned: it decodes
each probability (stage 3, `inference.*` decoding keys) and georeferences the polygons (stage 4). Inference itself
(`serve.py`) does neither.
"""

from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Geod

from .decode import decode_polygons
from .georef import polygons_to_chip_pixels, ring_to_epsg4326
from .params import parse_postprocess_params
from .preprocess import PreprocessMeta

CLASS_NAMES = {1: "seagrass"}
_GEOD = Geod(ellps="WGS84")


def _signed_area(ring: list[list[float]]) -> float:
    """Shoelace area of a closed `[[x, y], ...]` ring: positive when counter-clockwise (x right, y up)."""
    xy = np.asarray(ring, dtype=float)
    return float(np.dot(xy[:-1, 0], xy[1:, 1]) - np.dot(xy[1:, 0], xy[:-1, 1])) / 2


def orient_rings(rings: list[list[list[float]]]) -> list[list[list[float]]]:
    """RFC 7946 winding: exterior counter-clockwise, holes clockwise (the pixel -> map flip reverses the order)."""
    oriented = []
    for index, ring in enumerate(rings):
        counter_clockwise = _signed_area(ring) > 0
        oriented.append(ring if counter_clockwise == (index == 0) else ring[::-1])
    return oriented


def geodesic_area_m2(rings: list[list[list[float]]]) -> float:
    """Area on the WGS84 ellipsoid of an EPSG:4326 polygon (exterior minus holes), in square metres."""
    exterior, *holes = rings
    return abs(_GEOD.polygon_area_perimeter(*zip(*exterior))[0]) - sum(
        abs(_GEOD.polygon_area_perimeter(*zip(*hole))[0]) for hole in holes
    )


def polygons_to_feature_collection(
    polygons: list[dict[str, Any]],
    meta: PreprocessMeta,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """`polygons` are in source-chip pixels; output is EPSG:4326 polygons with their holes.

    A chip with a CRS gets `area_m2` (geodesic) on every feature; a chip without one stays in chip pixels.
    """
    features: list[dict[str, Any]] = []
    for polygon in polygons:
        rings = orient_rings([ring_to_epsg4326(ring, meta.transform, meta.crs) for ring in polygon["rings"]])
        props: dict[str, Any] = {
            "confidence": round(float(polygon["confidence"]), 4),
            "class_id": int(polygon["class_id"]),
            "class_name": CLASS_NAMES.get(int(polygon["class_id"]), "seagrass"),
        }
        if meta.crs is not None:
            props["area_m2"] = round(geodesic_area_m2(rings), 3)
        if source_name:
            props["source"] = source_name
        features.append(
            {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": rings}}
        )
    return {"type": "FeatureCollection", "features": features}


def postprocess_outputs(
    outputs: list[tuple[Path, np.ndarray, PreprocessMeta]], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """`predict_session` output -> one EPSG:4326 FeatureCollection with the features of every chip."""
    cfg = parse_postprocess_params(params)
    features: list[dict[str, Any]] = []
    for path, probability, meta in outputs:
        polygons = polygons_to_chip_pixels(decode_polygons(probability, cfg), meta)
        features.extend(
            polygons_to_feature_collection(polygons, meta, source_name=Path(path).name)["features"]
        )
    return {"type": "FeatureCollection", "features": features}
