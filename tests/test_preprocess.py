import numpy as np
import pytest

from conftest import CHIP_NAME
from ss_hot.params import WINDOW_PX, PreprocessParams
from ss_hot.preprocess import preprocess_chip, preprocess_rgb, scale_rgb, stitch_windows, to_windows
from ss_hot.tiles import chip_scale, parse_oam_tile, tile_gsd_m


def test_ground_resolution_and_scale_from_the_chip_name():
    assert tile_gsd_m(CHIP_NAME, 256) == pytest.approx(0.0373, abs=1e-4)
    assert chip_scale(CHIP_NAME, 256, 0.03) == pytest.approx(1.244, abs=1e-3)
    assert chip_scale(CHIP_NAME, 256, 0.03, max_input_px=300) == pytest.approx(300 / 256)


def test_an_explicit_ground_resolution_overrides_a_misleading_name():
    name = "OAM-7123225-4393839-22.tif"  # x/y are on the zoom-23 grid although the name says 22
    with pytest.raises(ValueError):
        parse_oam_tile(name)
    assert chip_scale(name, 512, 0.03, source_gsd_m=0.0092) == pytest.approx(0.3067, abs=1e-4)


def test_too_coarse_a_zoom_is_refused():
    with pytest.raises(ValueError, match="too coarse"):
        chip_scale("OAM-262144-262144-19.tif", 256, 0.03)  # zoom 19 needs x 9.8


def test_scaling_keeps_range_layout_and_rounds_the_size():
    rgb = np.random.default_rng(0).integers(0, 256, size=(100, 80, 3), dtype=np.uint8)
    image = scale_rgb(rgb, 0.5)
    assert image.shape == (3, 50, 40) and image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert np.array_equal(scale_rgb(rgb, 1.0), rgb.transpose(2, 0, 1) / np.float32(255))
    with pytest.raises(ValueError):
        scale_rgb(rgb[..., :2], 1.0)


def test_windows_cover_the_chip_and_stitching_returns_the_original_pixels():
    image = np.random.default_rng(1).random((3, 700, 530), dtype=np.float32)
    windows, rows, cols = to_windows(image, 512)
    assert windows.shape == (4, 3, 512, 512) and (rows, cols) == (2, 2)
    assert np.array_equal(windows[1, :, :512, :18], image[:, :512, 512:530])  # second window of the first row
    rgb = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    windows, meta = preprocess_rgb(rgb, 1.0)
    assert (meta.rows, meta.cols, meta.resized_height, meta.resized_width) == (2, 2, 700, 530)
    assert np.array_equal(stitch_windows(windows[:, 0], meta), windows_to_image(windows, meta)[0])


def windows_to_image(windows: np.ndarray, meta) -> np.ndarray:
    """Reference stitching, written the slow way."""
    full = np.zeros((3, meta.rows * WINDOW_PX, meta.cols * WINDOW_PX), dtype=np.float32)
    for index, window in enumerate(windows):
        row, col = divmod(index, meta.cols)
        full[:, row * WINDOW_PX : (row + 1) * WINDOW_PX, col * WINDOW_PX : (col + 1) * WINDOW_PX] = window
    return full[:, : meta.resized_height, : meta.resized_width]


def test_preprocess_chip_reads_georeferencing_and_scales(write_chip):
    windows, meta = preprocess_chip(write_chip(), PreprocessParams())
    assert windows.shape == (1, 3, WINDOW_PX, WINDOW_PX)
    assert (meta.width, meta.resized_width) == (256, round(256 * chip_scale(CHIP_NAME, 256, 0.03)))
    assert str(meta.crs) == "EPSG:3857" and meta.transform.a == 1.0
    _, plain = preprocess_chip(write_chip(name="OAM-2097152-2097153-22.tif", crs=None))
    assert plain.crs is None
