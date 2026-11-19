import json

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

ORIGIN_X, ORIGIN_Y = (
    1_000_000.0,
    2_000_000.0,
)  # top-left corner of the synthetic georeferenced chips, EPSG:3857, 1 m pixels

# A zoom-24 training chip in Timor-Leste: 0.923 cm pixels, within 2% of the model resolution, so it is not resampled.
CHIP_NAME = "OAM-14246392-8787682-24.tif"
# A zoom-22 chip near the equator: 3.73 cm pixels, upscaled x 4.06 (256 -> 1039 px) to the model resolution.
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
    """Write a synthetic RGB chip: a georeferenced GeoTIFF (EPSG:3857, 1 m pixels, top-left at `origin` pixels from
    the shared origin) or, with `crs=None`, a plain one."""

    def _write(
        name: str = CHIP_NAME,
        size: int = 256,
        folder=None,
        crs: str | None = "EPSG:3857",
        seed: int = 0,
        origin: tuple[int, int] = (0, 0),
    ):
        directory = folder or tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        data = np.random.default_rng(seed).integers(0, 256, size=(3, size, size), dtype=np.uint8)
        profile = dict(driver="GTiff", height=size, width=size, count=3, dtype="uint8")
        if crs is not None:
            x, y = ORIGIN_X + origin[0], ORIGIN_Y - origin[1]
            profile.update(crs=crs, transform=from_origin(x, y, 1.0, 1.0))
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
        return path

    return _write


def lon_lat(col: float, row: float) -> list[float]:
    """Longitude and latitude of pixel `(col, row)` counted from the shared origin of the synthetic chips."""
    return list(
        Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
            ORIGIN_X + col, ORIGIN_Y - row
        )
    )


def write_labels(path, boxes: list[tuple[float, float, float, float]]) -> None:
    """A labels GeoJSON (EPSG:4326) with one seagrass rectangle per `(col0, row0, col1, row1)` box, in pixels."""
    features = []
    for c0, r0, c1, r1 in boxes:
        ring = [lon_lat(c0, r0), lon_lat(c1, r0), lon_lat(c1, r1), lon_lat(c0, r1), lon_lat(c0, r0)]
        features.append(
            {
                "type": "Feature",
                "properties": {"seamark:type": "seagrass"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


@pytest.fixture
def fair_dataset(tmp_path, write_chip):
    """The fAIr layout: a 2 x 2 grid of georeferenced 256 px zoom-24 chips and one `labels.geojson` whose single
    polygon covers the left column (seagrass) and not the right one."""
    chips, labels = tmp_path / "chips", tmp_path / "labels" / "labels.geojson"
    names = []
    for dx in range(2):
        for dy in range(2):
            name = f"OAM-{2**23 + dx}-{2**23 + dy}-24.tif"
            write_chip(name=name, folder=chips, seed=dx * 2 + dy, origin=(256 * dx, 256 * dy))
            names.append(name)
    write_labels(labels, [(0, 0, 256, 512)])
    return chips, labels, sorted(names)
