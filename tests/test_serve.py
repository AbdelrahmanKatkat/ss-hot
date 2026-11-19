"""Inference with a fake ONNX session: `predict_session` gives probabilities; `postprocess_outputs` gives GeoJSON."""

import subprocess
import sys

import numpy as np
import pytest

from conftest import CHIP_NAME, FakeSession
from ss_hot import postprocess_outputs, predict_session
from ss_hot.params import WINDOW_PX
from ss_hot.serve import list_images


def test_every_chip_is_cut_into_windows_and_stitched_back(write_chip, tmp_path):
    folder = tmp_path / "chips"
    write_chip(name=CHIP_NAME, folder=folder)
    write_chip(name="OAM-1048576-1048576-21.tif", folder=folder)
    session = FakeSession(0.9)
    outputs = predict_session(session, folder, {})
    assert [path.name for path, _, _ in outputs] == sorted(p.name for p in folder.iterdir())
    for _, probability, meta in outputs:
        assert probability.shape == (meta.resized_height, meta.resized_width)
    assert {shape[1:] for shape in session.shapes} == {(3, WINDOW_PX, WINDOW_PX)}
    assert sorted(shape[0] for shape in session.shapes) == [
        1,
        4,
    ]  # z22: 318 px, one window; z21: 637 px, 2 x 2


def test_predict_then_postprocess_is_one_feature_collection(write_chip, tmp_path):
    folder = tmp_path / "chips"
    write_chip(name=CHIP_NAME, folder=folder)
    collection = postprocess_outputs(predict_session(FakeSession(0.9), folder, None), None)
    assert collection["type"] == "FeatureCollection" and len(collection["features"]) == 1
    feature = collection["features"][0]
    assert feature["properties"]["class_name"] == "seagrass" and feature["properties"]["confidence"] == 0.9
    empty = postprocess_outputs(predict_session(FakeSession(), folder, None), None)
    assert empty == {"type": "FeatureCollection", "features": []}


def test_an_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path)


def test_the_serving_path_does_not_import_torch():
    code = (
        "import sys, ss_hot, ss_hot.evaluate; "
        "assert not any(m.split('.')[0] in ('torch', 'timm') for m in sys.modules), 'heavy import'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_scores_through_a_session(mask_dataset):
    from ss_hot.evaluate import best_threshold, score_session

    images, masks, _ = mask_dataset
    results = score_session(
        FakeSession(), images_dir=images, labels=masks, params={}, source_gsd_m=0.015, thresholds=(0.3, 0.5)
    )
    assert set(results) == {0.3, 0.5}
    metrics = results[0.5]
    assert metrics["tp"] == 0 and metrics["fn"] == 256 * 128 and metrics["iou"] == 0.0
    assert best_threshold(results)[0] in (0.3, 0.5)
    assert np.isfinite(metrics["precision"])
