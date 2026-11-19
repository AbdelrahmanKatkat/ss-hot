"""Dataset helpers for ss-hot (numpy, pyproj and rasterio only, no torch; safe to import at serving time).

The fAIr layout: georeferenced RGB GeoTIFF chips `OAM-<x>-<y>-<z>.tif` and one `labels.geojson` of seagrass polygons
(`LabelSource`). Each chip's labels become a `uint8` mask of `BACKGROUND` (0) and `SEAGRASS` (1), drawn at the size of
the chip scaled exactly as at serving time (`scale_pair`); a chip at the training zoom is not resampled. One chip is
one training sample (`fit_tile`).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize

from .params import PreprocessParams
from .preprocess import read_chip, scale_rgb
from .tiles import OAM_TILE_RE, chip_scale

CHIP_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
BACKGROUND, SEAGRASS = 0, 1


def list_chip_paths(root: str | Path, names: Sequence[str] | None = None) -> list[Path]:
    """RGB chips in a folder, sorted; with `names`, only those (a missing name raises)."""
    folder = Path(root)
    paths = sorted(path for pattern in CHIP_PATTERNS for path in folder.glob(pattern) if path.is_file())
    if names is not None:
        wanted = set(names)
        paths = [path for path in paths if path.name in wanted]
        missing = wanted - {path.name for path in paths}
        if missing:
            raise ValueError(f"{len(missing)} named chips not found in {folder}, e.g. {sorted(missing)[:3]}")
    return paths


def spatial_split(
    chip_names: Sequence[str], val_ratio: float, seed: int, *, block_size: int = 4
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coordinates: whole `(x // K, y // K)` blocks go to one side.

    Neighbouring chips share meadows cut at chip borders, so a random chip split leaks. Returns sorted
    `(train, validation)` name lists; every name must be `OAM-x-y-z.ext`.
    """
    blocks: dict[tuple[int, int], list[str]] = {}
    for name in chip_names:
        match = OAM_TILE_RE.match(name)
        if match is None:
            raise ValueError(f"chip name {name!r} does not match OAM-<x>-<y>-<z>.<ext>")
        blocks.setdefault((int(match.group(1)) // block_size, int(match.group(2)) // block_size), []).append(
            name
        )
    order = sorted(blocks)
    np.random.default_rng(seed).shuffle(order)
    target = max(1, int(len(chip_names) * val_ratio)) if chip_names else 0
    train: list[str] = []
    val: list[str] = []
    for key in order:
        (val if len(val) < target else train).extend(blocks[key])
    return sorted(train), sorted(val)


# --- labels: one GeoJSON of seagrass polygons ---


def _find_labels_geojson(labels: str | Path) -> Path:
    """The labels file: the path itself if it is a `.geojson` / `.json` file, else the one such file under a folder."""
    root = Path(labels)
    if root.is_file() and root.suffix.lower() in (".geojson", ".json"):
        return root
    matches = sorted(root.rglob("*.geojson")) or sorted(root.rglob("*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .geojson (or .json) in {root}, found {len(matches)}")
    return matches[0]


def read_label_polygons(labels_geojson: str | Path) -> tuple[list[list[np.ndarray]], str]:
    """The polygons of a GeoJSON (a multipolygon gives one per part) as ring lists, plus the file's CRS name."""
    collection = json.loads(Path(labels_geojson).read_text(encoding="utf-8"))
    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        raise ValueError("labels file must be a GeoJSON FeatureCollection")
    polygons: list[list[np.ndarray]] = []
    for feature in collection.get("features") or []:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry, dict):
            continue
        parts = {"Polygon": [geometry.get("coordinates")], "MultiPolygon": geometry.get("coordinates")}.get(
            geometry.get("type"), []
        )
        for rings in parts or []:
            if rings:
                polygons.append([np.asarray(ring, dtype=float)[:, :2] for ring in rings])
    crs = (
        collection.get("crs", {}).get("properties", {}).get("name")
        if isinstance(collection.get("crs"), dict)
        else None
    )
    return polygons, crs or "EPSG:4326"


@dataclass
class LabelSource:
    """The seagrass polygons of one labels GeoJSON (the fAIr layout: `chips/` plus `labels.geojson`).

    The polygons are reprojected once per chip CRS and kept with their bounding boxes, so drawing one chip's mask only
    touches the polygons near it.
    """

    polygons: list[list[np.ndarray]]
    crs: str = "EPSG:4326"
    _projected: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_path(cls, labels: str | Path) -> "LabelSource":
        """A `.geojson` file, or a folder holding exactly one."""
        polygons, crs = read_label_polygons(_find_labels_geojson(labels))
        return cls(polygons=polygons, crs=crs)

    def _in_crs(self, crs: Any) -> tuple[list[list[np.ndarray]], np.ndarray]:
        key = str(crs)
        if key not in self._projected:
            polygons = self.polygons
            if key != str(self.crs):
                to_chip = Transformer.from_crs(self.crs, crs, always_xy=True).transform
                polygons = [[np.column_stack(to_chip(r[:, 0], r[:, 1])) for r in rings] for rings in polygons]
            bounds = np.array([[*rings[0].min(axis=0), *rings[0].max(axis=0)] for rings in polygons])
            self._projected[key] = (polygons, bounds.reshape(-1, 4))
        return self._projected[key]

    def mask(self, chip_path: str | Path, height: int, width: int) -> np.ndarray:
        """The seagrass mask of a georeferenced chip, drawn at `(height, width)` (the scaled chip)."""
        import rasterio

        with rasterio.open(chip_path) as src:
            if src.crs is None:
                raise ValueError(f"{chip_path} has no CRS; fAIr chips are georeferenced GeoTIFFs")
            t, crs, (west, south, east, north) = src.transform, src.crs, src.bounds
            sx, sy = src.width / width, src.height / height
        polygons, bounds = self._in_crs(crs)
        near = (
            (bounds[:, 0] <= east)
            & (bounds[:, 2] >= west)
            & (bounds[:, 1] <= north)
            & (bounds[:, 3] >= south)
        )
        shapes = [
            ({"type": "Polygon", "coordinates": [ring.tolist() for ring in polygons[i]]}, SEAGRASS)
            for i in np.flatnonzero(near)
        ]
        if not shapes:
            return np.zeros((height, width), dtype=np.uint8)
        transform = Affine(
            t.a * sx, t.b * sy, t.c, t.d * sx, t.e * sy, t.f
        )  # the chip's transform, scaled pixels
        return rasterize(
            shapes, out_shape=(height, width), transform=transform, fill=BACKGROUND, dtype="uint8"
        )


def scale_pair(
    chip_path: str | Path, labels: LabelSource, pre: PreprocessParams
) -> tuple[np.ndarray, np.ndarray]:
    """A chip scaled exactly as at serving time and its mask at that size: `(3, H', W')` float32, `(H', W')` uint8."""
    path = Path(chip_path)
    rgb, _, _ = read_chip(path)
    image = scale_rgb(rgb, chip_scale(path.name, rgb.shape[1], pre.model_gsd_m))
    return image, labels.mask(path, image.shape[1], image.shape[2])


def fit_tile(
    image: np.ndarray, mask: np.ndarray, window: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """A pair as exactly `window` x `window`: a crop (at random with `rng`, else the top-left) of a larger chip, or a
    reflection of a smaller one. The mask is reflected with the image, so the padding carries true labels.

    A 256 px chip at the training zoom is returned as it is.
    """
    _, h, w = image.shape
    top = int(rng.integers(0, h - window + 1)) if rng is not None and h > window else 0
    left = int(rng.integers(0, w - window + 1)) if rng is not None and w > window else 0
    image = image[:, top : top + window, left : left + window]
    mask = mask[top : top + window, left : left + window]
    pad_h, pad_w = window - image.shape[1], window - image.shape[2]
    if pad_h or pad_w:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="symmetric")
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def load_label_source(labels: str | Path | LabelSource) -> LabelSource:
    return labels if isinstance(labels, LabelSource) else LabelSource.from_path(labels)
