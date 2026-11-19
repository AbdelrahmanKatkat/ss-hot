import numpy as np
import pytest
from rasterio.transform import from_origin

from ss_hot.decode import decode_polygons
from ss_hot.georef import polygons_to_chip_pixels, ring_to_chip_pixels
from ss_hot.params import PostprocessParams
from ss_hot.postprocess import (
    _signed_area,
    geodesic_area_m2,
    orient_rings,
    polygons_to_feature_collection,
    postprocess_outputs,
)
from ss_hot.preprocess import PreprocessMeta


def _meta(crs="EPSG:3857") -> PreprocessMeta:
    """A 64 px chip with origin (1 000 000, 2 000 000) and 1 m pixels, scaled 2x for the model."""
    return PreprocessMeta(
        width=64, height=64, resized_width=128, resized_height=128, rows=1, cols=1,
        transform=from_origin(1_000_000.0, 2_000_000.0, 1.0, 1.0), crs=crs,
    )  # fmt: skip


def _blob(value: float = 0.8) -> np.ndarray:
    probability = np.zeros((128, 128), dtype=np.float32)
    probability[20:80, 20:80] = value
    return probability


def test_model_pixels_are_divided_by_the_scale_to_give_chip_pixels():
    ring = np.array([[0.0, 0.0], [128.0, 0.0], [128.0, 64.0], [0.0, 0.0]])
    assert ring_to_chip_pixels(ring, _meta()).tolist() == [[0, 0], [64, 0], [64, 32], [0, 0]]
    (polygon,) = decode_polygons(_blob(), PostprocessParams())
    (chip,) = polygons_to_chip_pixels([polygon], _meta())
    assert chip["confidence"] == polygon["confidence"] and chip["rings"][0].max() == 40.0  # 80 / 2


def test_features_are_epsg4326_polygons_with_confidence_class_area_and_source():
    meta = _meta()
    polygons = polygons_to_chip_pixels(decode_polygons(_blob(), PostprocessParams()), meta)
    (feature,) = polygons_to_feature_collection(polygons, meta, source_name="OAM-1-2-3.tif")["features"]
    props = feature["properties"]
    assert {k: props[k] for k in ("confidence", "class_id", "class_name", "source")} == {
        "confidence": 0.8, "class_id": 1, "class_name": "seagrass", "source": "OAM-1-2-3.tif",
    }  # fmt: skip
    # 30 x 30 chip pixels of 1 Web-Mercator metre at latitude 17.7: about 900 * cos(17.7)^2 = 818 m2
    assert props["area_m2"] == pytest.approx(818, rel=0.02)
    ring = np.array(feature["geometry"]["coordinates"][0])
    assert np.array_equal(ring[0], ring[-1]) and _signed_area(ring.tolist()) > 0  # exterior counter-clockwise
    assert (np.abs(ring[:, 0] - 8.98) < 0.01).all() and (np.abs(ring[:, 1] - 17.68) < 0.05).all()


def test_holes_are_kept_clockwise_and_a_chip_without_crs_has_no_area():
    probability = _blob()
    probability[40:60, 40:60] = 0.0  # a 10 x 10 chip-pixel hole (400 model pixels, above fill_holes_px)
    meta = _meta()
    (feature,) = polygons_to_feature_collection(
        polygons_to_chip_pixels(decode_polygons(probability), meta), meta
    )["features"]
    exterior, hole = feature["geometry"]["coordinates"]
    assert _signed_area(exterior) > 0 > _signed_area(hole)
    plain = _meta(crs=None)
    (feature,) = polygons_to_feature_collection(
        polygons_to_chip_pixels(decode_polygons(_blob()), plain), plain
    )["features"]
    assert "area_m2" not in feature["properties"]


def test_orientation_and_geodesic_area_of_a_known_square():
    square = [[0.0, 0.0], [0.0, 0.001], [0.001, 0.001], [0.001, 0.0], [0.0, 0.0]]  # clockwise
    (oriented,) = orient_rings([square])
    assert _signed_area(oriented) > 0
    assert geodesic_area_m2([oriented]) == pytest.approx(111.3 * 110.6, rel=0.01)  # about 0.1 x 0.1 km


def test_empty_predictions_give_an_empty_collection():
    outputs = [("OAM-1-1-22.tif", np.zeros((128, 128), dtype=np.float32), _meta())]
    assert postprocess_outputs(outputs, None) == {"type": "FeatureCollection", "features": []}
    assert postprocess_outputs([], {}) == {"type": "FeatureCollection", "features": []}
