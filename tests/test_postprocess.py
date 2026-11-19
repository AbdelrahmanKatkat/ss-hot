import numpy as np
import pytest
from rasterio.transform import from_origin

from ss_hot.decode import DecodeParams, decode_polygons
from ss_hot.georef import polygons_to_chip_pixels, ring_to_chip_pixels
from ss_hot.params import MODEL_GSD_M, PostprocessParams
from ss_hot.postprocess import (
    _signed_area,
    decode_params,
    geodesic_area_m2,
    orient_rings,
    polygons_to_feature_collection,
    postprocess_outputs,
)
from ss_hot.preprocess import PreprocessMeta


def _meta(crs="EPSG:3857") -> PreprocessMeta:
    """A 64 px tile with origin (1 000 000, 2 000 000) and 1 m pixels, scaled 2x for the model."""
    return PreprocessMeta(
        width=64, height=64, resized_width=128, resized_height=128, rows=1, cols=1,
        transform=from_origin(1_000_000.0, 2_000_000.0, 1.0, 1.0), crs=crs,
    )  # fmt: skip


def _blob(value: float = 0.8) -> np.ndarray:
    probability = np.zeros((128, 128), dtype=np.float32)
    probability[20:80, 20:80] = value
    return probability


def test_ground_sizes_become_model_pixels():
    cfg = decode_params(PostprocessParams(), MODEL_GSD_M)
    assert cfg.min_area_px == pytest.approx(0.25 / 0.0092**2) and cfg.fill_holes_px == cfg.min_area_px
    assert (cfg.open_px, cfg.close_px) == (3, 7) and cfg.simplify_px == pytest.approx(0.06 / 0.0092)
    assert decode_params(PostprocessParams(threshold=0.3), 0.03).open_px == 1


def test_model_pixels_are_divided_by_the_scale_to_give_tile_pixels():
    ring = np.array([[0.0, 0.0], [128.0, 0.0], [128.0, 64.0], [0.0, 0.0]])
    assert ring_to_chip_pixels(ring, _meta()).tolist() == [[0, 0], [64, 0], [64, 32], [0, 0]]
    (polygon,) = decode_polygons(_blob())
    (chip,) = polygons_to_chip_pixels([polygon], _meta())
    assert chip["confidence"] == polygon["confidence"] and chip["rings"][0].max() == 40.0  # 80 / 2


def test_features_are_epsg4326_polygons_with_confidence_class_area_and_source():
    meta = _meta()
    polygons = polygons_to_chip_pixels(decode_polygons(_blob()), meta)
    (feature,) = polygons_to_feature_collection(polygons, meta, source_name="OAM-1-2-3.tif")["features"]
    props = feature["properties"]
    assert {k: props[k] for k in ("confidence", "class_id", "class_name", "source")} == {
        "confidence": 0.8, "class_id": 1, "class_name": "seagrass", "source": "OAM-1-2-3.tif",
    }  # fmt: skip
    # 30 x 30 tile pixels of 1 Web-Mercator metre at latitude 17.7: about 900 * cos(17.7)^2 = 818 m2
    assert props["area_m2"] == pytest.approx(818, rel=0.02)
    ring = np.array(feature["geometry"]["coordinates"][0])
    assert np.array_equal(ring[0], ring[-1]) and _signed_area(ring.tolist()) > 0  # exterior counter-clockwise
    assert (np.abs(ring[:, 0] - 8.98) < 0.01).all() and (np.abs(ring[:, 1] - 17.68) < 0.05).all()


def test_holes_are_kept_clockwise_and_a_tile_without_crs_has_no_area():
    probability = _blob()
    probability[40:60, 40:60] = 0.0
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


def test_postprocess_sizes_are_in_metres():
    outputs = [("OAM-1-1-22.tif", _blob(), _meta())]  # 60 x 60 model pixels of 0.92 cm: 0.30 m2
    assert len(postprocess_outputs(outputs, None)["features"]) == 1
    assert postprocess_outputs(outputs, {"min_area_m2": 0.5})["features"] == []
    holed = _blob()
    holed[40:60, 40:60] = 0.0  # 400 model pixels: 0.034 m2, under the 0.25 m2 default, so it is filled
    (feature,) = postprocess_outputs([("t.tif", holed, _meta())], {"open_m": 0, "close_m": 0})["features"]
    assert len(feature["geometry"]["coordinates"]) == 1
    (feature,) = postprocess_outputs([("t.tif", holed, _meta())], {"fill_holes_m2": 0, "close_m": 0})[
        "features"
    ]
    assert len(feature["geometry"]["coordinates"]) == 2


def test_orientation_and_geodesic_area_of_a_known_square():
    square = [[0.0, 0.0], [0.0, 0.001], [0.001, 0.001], [0.001, 0.0], [0.0, 0.0]]  # clockwise
    (oriented,) = orient_rings([square])
    assert _signed_area(oriented) > 0
    assert geodesic_area_m2([oriented]) == pytest.approx(111.3 * 110.6, rel=0.01)  # about 0.1 x 0.1 km


def test_empty_predictions_give_an_empty_collection():
    outputs = [("OAM-1-1-22.tif", np.zeros((128, 128), dtype=np.float32), _meta())]
    assert postprocess_outputs(outputs, None) == {"type": "FeatureCollection", "features": []}
    assert postprocess_outputs([], {}) == {"type": "FeatureCollection", "features": []}
    assert decode_polygons(np.zeros((8, 8), np.float32), DecodeParams()) == []
