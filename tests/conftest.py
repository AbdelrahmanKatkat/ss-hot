import json

import numpy as np
import pytest
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin

ORIGIN_X, ORIGIN_Y = (
    1_000_000.0,
    2_000_000.0,
)  # top-left corner of every synthetic georeferenced tile, EPSG:3857

# A zoom-24 training tile in Timor-Leste: 0.923 cm pixels, within 2% of the model resolution, so it is not resampled.
CHIP_NAME = "OAM-14246392-8787682-24.tif"
# A zoom-22 tile near the equator: 3.73 cm pixels, upscaled x 4.06 (256 -> 1039 px) to the model resolution.
COARSE_NAME = "OAM-2097152-2097152-22.tif"


class FakeSession:
    """Stands in for onnxruntime: `(N, 3, w, w)` windows in, `(N, w, w)` probabilities out (zero, or a planted value)."""

    class _In:
        name = "image"

    def __init__(self, value: float | None = None):
        self.value = value
        self.shapes: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feeds):
        windows = feeds["image"]
        assert windows.ndim == 4 and windows.shape[1] == 3 and windows.dtype == np.float32
        assert 0.0 <= windows.min() <= windows.max() <= 1.0
        self.shapes.append(windows.shape)
        probability = np.zeros((windows.shape[0], *windows.shape[2:]), dtype=np.float32)
        if self.value is not None:
            probability[:, 100:200, 100:200] = self.value
        return [probability]


@pytest.fixture
def write_chip(tmp_path):
    """Write a synthetic RGB tile: a georeferenced GeoTIFF (EPSG:3857, 1 m pixels) or, with `crs=None`, a plain one."""

    def _write(
        name: str = CHIP_NAME, size: int = 256, folder=None, crs: str | None = "EPSG:3857", seed: int = 0
    ):
        directory = folder or tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        data = np.random.default_rng(seed).integers(0, 256, size=(3, size, size), dtype=np.uint8)
        profile = dict(driver="GTiff", height=size, width=size, count=3, dtype="uint8")
        if crs is not None:
            profile.update(crs=crs, transform=from_origin(ORIGIN_X, ORIGIN_Y, 1.0, 1.0))
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
        return path

    return _write


def lon_lat(col: float, row: float) -> list[float]:
    """Longitude and latitude of pixel `(col, row)` of a synthetic georeferenced tile."""
    return list(
        Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
            ORIGIN_X + col, ORIGIN_Y - row
        )
    )


@pytest.fixture
def mask_dataset(tmp_path, write_chip):
    """A 2 x 2 grid of plain 256 px zoom-24 tiles with 0/1 masks: the left column all seagrass, the right one not."""
    images, masks = tmp_path / "images", tmp_path / "masks"
    masks.mkdir()
    names = []
    for dx in range(2):
        for dy in range(2):
            name = f"OAM-{2**23 + dx}-{2**23 + dy}-24.tif"
            write_chip(name=name, folder=images, crs=None, seed=dx * 2 + dy)
            Image.fromarray(np.full((256, 256), 1 if dx == 0 else 0, dtype=np.uint8)).save(
                masks / f"{name[:-4]}.png"
            )
            names.append(name)
    return images, masks, sorted(names)


@pytest.fixture
def geojson_dataset(tmp_path, write_chip):
    """One georeferenced tile and a GeoJSON with a seagrass square over its top-left 128 x 128 px."""
    chips, labels = tmp_path / "chips", tmp_path / "labels"
    labels.mkdir()
    write_chip(name=CHIP_NAME, folder=chips)
    square = [lon_lat(0, 0), lon_lat(128, 0), lon_lat(128, 128), lon_lat(0, 128), lon_lat(0, 0)]
    feature = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [square]}}
    (labels / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    return chips, labels
