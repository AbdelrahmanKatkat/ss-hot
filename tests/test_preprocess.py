import numpy as np
import pytest

from conftest import CHIP_NAME, COARSE_NAME
from ss_hot.params import MODEL_GSD_M, WINDOW_PX, PreprocessParams
from ss_hot.preprocess import (
    preprocess_chip,
    preprocess_rgb,
    read_chip,
    scale_rgb,
    stitch_windows,
    to_windows,
)
from ss_hot.tiles import chip_scale, tile_gsd_m


def test_a_training_tile_is_used_as_it_is_and_a_coarser_one_is_upscaled():
    assert tile_gsd_m(CHIP_NAME, 256) == pytest.approx(0.00923, abs=1e-5)
    assert chip_scale(CHIP_NAME, 256, MODEL_GSD_M) == 1.0
    assert chip_scale(COARSE_NAME, 256, MODEL_GSD_M) == pytest.approx(4.057, abs=1e-3)


def test_too_coarse_a_zoom_is_refused():
    with pytest.raises(ValueError, match="too coarse"):
        chip_scale("OAM-1048576-1048576-21.tif", 256, MODEL_GSD_M)  # zoom 21 needs x 8.1


def test_scaling_keeps_range_layout_and_rounds_the_size():
    rgb = np.random.default_rng(0).integers(0, 256, size=(100, 80, 3), dtype=np.uint8)
    image = scale_rgb(rgb, 0.5)
    assert image.shape == (3, 50, 40) and image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert np.array_equal(scale_rgb(rgb, 1.0), rgb.transpose(2, 0, 1) / np.float32(255))
    with pytest.raises(ValueError):
        scale_rgb(rgb[..., :2], 1.0)


def test_windows_cover_the_tile_and_stitching_returns_the_original_pixels():
    image = np.random.default_rng(1).random((3, 300, 530), dtype=np.float32)
    windows, rows, cols = to_windows(image, WINDOW_PX)
    assert windows.shape == (6, 3, WINDOW_PX, WINDOW_PX) and (rows, cols) == (2, 3)
    assert np.array_equal(
        windows[1, :, :256, :256], image[:, :256, 256:512]
    )  # second window of the first row
    rgb = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    windows, meta = preprocess_rgb(rgb, 1.0)
    assert (meta.rows, meta.cols, meta.resized_height, meta.resized_width) == (2, 3, 300, 530)
    assert np.array_equal(stitch_windows(windows[:, 0], meta), scale_rgb(rgb, 1.0)[0])


def test_a_training_tile_is_one_window_and_keeps_its_georeferencing(write_chip):
    windows, meta = preprocess_chip(write_chip(), PreprocessParams())
    assert windows.shape == (1, 3, WINDOW_PX, WINDOW_PX) and meta.scale_x == 1.0
    assert str(meta.crs) == "EPSG:3857" and meta.transform.a == 1.0  # the file's own transform


def test_a_chip_without_georeferencing_has_no_crs(write_chip, tmp_path):
    _, transform, crs = read_chip(write_chip(folder=tmp_path / "plain", crs=None))
    assert crs is None and transform.a == 1.0
