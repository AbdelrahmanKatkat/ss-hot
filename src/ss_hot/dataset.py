"""Dataset helpers for ss-hot (numpy, Pillow and rasterio only, no torch; safe to import at serving time).

Labels come in one of two forms, and both end as a `uint8` mask of `BACKGROUND`, `SEAGRASS` and `IGNORE` per chip:

- a **mask folder**: one `<chip stem>.png` / `.tif` per chip, values 0 (background), 1 (seagrass) and 255 (ignore);
  this is the layout of the prepared seagrass dataset (`images/` + `masks/` per split);
- a **GeoJSON** of seagrass polygons over georeferenced chips (the fAIr layout: `chips/` + one labels file).

Every chip is scaled to the model's ground resolution exactly as at serving time (`scale_pair`). Chips that become
small after scaling are joined with their grid neighbours into one mosaic (`plan_blocks`, `assemble_block`), so a
training window holds real surroundings instead of padding; positions without a chip are `IGNORE`.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .params import WINDOW_PX, PreprocessParams
from .preprocess import read_chip, scale_rgb
from .tiles import OAM_TILE_RE, chip_scale

CHIP_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
MASK_SUFFIXES = (".png", ".tif", ".tiff")
BACKGROUND, SEAGRASS, IGNORE = 0, 1, 255


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


# --- labels: mask folder ---


def read_mask(path: str | Path) -> np.ndarray:
    """A label mask as `uint8` `(H, W)`; only 0, 1 and 255 are allowed (a 0/255 seagrass mask must be converted)."""
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    unexpected = set(np.unique(mask).tolist()) - {BACKGROUND, SEAGRASS, IGNORE}
    if unexpected:
        raise ValueError(f"{path}: mask values must be 0, 1 or 255, found {sorted(unexpected)[:5]}")
    return mask


def mask_path_for(masks_dir: str | Path, chip_name: str) -> Path | None:
    """The mask of a chip in a mask folder (same stem, `.png` or `.tif`), or `None`."""
    stem = Path(chip_name).stem
    for suffix in MASK_SUFFIXES:
        candidate = Path(masks_dir) / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def scale_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a label mask to `(height, width)`: seagrass where at least half of the area is seagrass.

    The seagrass and ignore fractions are resized as areas (antialiased when shrinking), so a downscaled mask keeps the
    majority label of each new pixel instead of a nearest-neighbour sample.
    """
    if mask.shape == (height, width):
        return mask.copy()

    def fraction(binary: np.ndarray) -> np.ndarray:
        image = Image.fromarray(binary.astype(np.float32), mode="F")
        return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))

    out = np.where(fraction(mask == SEAGRASS) >= 0.5, SEAGRASS, BACKGROUND).astype(np.uint8)
    out[fraction(mask == IGNORE) >= 0.5] = IGNORE
    return out


# --- labels: GeoJSON over georeferenced chips ---


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


def rasterize_chip_polygons(
    chip_path: str | Path, polygons: list[list[np.ndarray]], labels_crs: str, height: int, width: int
) -> np.ndarray:
    """Seagrass polygons over one georeferenced chip, drawn at `(height, width)` (the scaled chip) as a mask."""
    import rasterio
    from affine import Affine
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom

    with rasterio.open(chip_path) as src:
        if src.crs is None:
            raise ValueError(f"{chip_path} has no CRS; GeoJSON labels need a georeferenced chip")
        sx, sy, t = src.width / width, src.height / height, src.transform
        transform = Affine(
            t.a * sx, t.b * sy, t.c, t.d * sx, t.e * sy, t.f
        )  # the chip's transform, scaled pixels
        chip_crs = src.crs
    shapes = []
    for rings in polygons:
        geometry = {"type": "Polygon", "coordinates": [ring.tolist() for ring in rings]}
        if str(chip_crs) != str(labels_crs):
            geometry = transform_geom(labels_crs, chip_crs, geometry)
        shapes.append((geometry, SEAGRASS))
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)
    return rasterize(shapes, out_shape=(height, width), transform=transform, fill=BACKGROUND, dtype="uint8")


# --- one labelled chip, scaled ---


@dataclass(frozen=True)
class LabelSource:
    """Where the labels of a chips folder are: a mask folder, or the polygons of one GeoJSON."""

    masks_dir: Path | None = None
    polygons: tuple = ()
    crs: str = "EPSG:4326"

    @classmethod
    def from_path(cls, labels: str | Path) -> "LabelSource":
        """A folder with mask images is a mask folder; a `.geojson` (or a folder holding one) is a GeoJSON."""
        root = Path(labels)
        if root.is_dir() and any(next(root.glob(f"*{suffix}"), None) for suffix in MASK_SUFFIXES):
            return cls(masks_dir=root)
        polygons, crs = read_label_polygons(_find_labels_geojson(root))
        return cls(polygons=tuple(polygons), crs=crs)

    def mask(self, chip_path: Path, height: int, width: int) -> np.ndarray:
        """The chip's label mask at `(height, width)`; a chip without a mask file is all `IGNORE`."""
        if self.masks_dir is not None:
            path = mask_path_for(self.masks_dir, chip_path.name)
            if path is None:
                return np.full((height, width), IGNORE, dtype=np.uint8)
            return scale_mask(read_mask(path), height, width)
        return rasterize_chip_polygons(chip_path, list(self.polygons), self.crs, height, width)


def scale_pair(
    chip_path: str | Path,
    labels: LabelSource,
    pre: PreprocessParams,
    *,
    source_gsd_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A chip scaled exactly as at serving time and its label mask at that size: `(3, H', W')` float32, `(H', W')` uint8."""
    path = Path(chip_path)
    rgb, _, _ = read_chip(path)
    scale = chip_scale(
        path.name, rgb.shape[1], pre.model_gsd_m, height_px=rgb.shape[0], max_input_px=pre.max_input_px,
        source_gsd_m=source_gsd_m,
    )  # fmt: skip
    image = scale_rgb(rgb, scale)
    return image, labels.mask(path, image.shape[1], image.shape[2])


# --- mosaics of neighbouring chips ---


def chips_per_side(scaled_side_px: int, window: int = WINDOW_PX) -> int:
    """How many scaled chips fit along one side of a window (at least one)."""
    return max(1, window // max(1, scaled_side_px))


def plan_blocks(names: Sequence[str], per_side: int) -> list[list[tuple[str, int, int]]]:
    """Group OAM chip names into `per_side` x `per_side` blocks of the tile grid: `[[(name, col, row), ...], ...]`.

    `col` and `row` are the chip's place in its block. With `per_side` 1 every chip is its own block. Names that are
    not `OAM-x-y-z` can only be single blocks.
    """
    if per_side <= 1:
        return [[(name, 0, 0)] for name in sorted(names)]
    blocks: dict[tuple[int, int, int], list[tuple[str, int, int]]] = {}
    for name in sorted(names):
        match = OAM_TILE_RE.match(name)
        if match is None:
            raise ValueError(
                f"chip name {name!r} does not match OAM-<x>-<y>-<z>.<ext>; mosaics need tile names"
            )
        x, y, z = (int(match.group(i)) for i in (1, 2, 3))
        blocks.setdefault((z, x // per_side, y // per_side), []).append((name, x % per_side, y % per_side))
    return [blocks[key] for key in sorted(blocks)]


def assemble_block(
    block: list[tuple[str, int, int]],
    images_dir: str | Path,
    labels: LabelSource,
    pre: PreprocessParams,
    *,
    source_gsd_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The scaled chips of one block placed side by side: `(3, H, W)` image and `(H, W)` mask; gaps are `IGNORE`."""
    pairs = [
        (col, row, *scale_pair(Path(images_dir) / name, labels, pre, source_gsd_m=source_gsd_m))
        for name, col, row in block
    ]
    side_h = max(image.shape[1] for _, _, image, _ in pairs)
    side_w = max(image.shape[2] for _, _, image, _ in pairs)
    rows = 1 + max(row for _, row, _, _ in pairs)
    cols = 1 + max(col for col, _, _, _ in pairs)
    image_out = np.zeros((3, rows * side_h, cols * side_w), dtype=np.float32)
    mask_out = np.full((rows * side_h, cols * side_w), IGNORE, dtype=np.uint8)
    for col, row, image, mask in pairs:
        top, left = row * side_h, col * side_w
        image_out[:, top : top + image.shape[1], left : left + image.shape[2]] = image
        mask_out[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return image_out, mask_out


def fit_window(
    image: np.ndarray, mask: np.ndarray, window: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Crop (at random with `rng`, else the top-left) or pad a pair to exactly `window` x `window`.

    Padding is a reflection of the image and `IGNORE` in the mask, so it is neither rewarded nor punished.
    """
    _, h, w = image.shape
    top = int(rng.integers(0, h - window + 1)) if rng is not None and h > window else 0
    left = int(rng.integers(0, w - window + 1)) if rng is not None and w > window else 0
    image, mask = (
        image[:, top : top + window, left : left + window],
        mask[top : top + window, left : left + window],
    )
    pad_h, pad_w = window - image.shape[1], window - image.shape[2]
    if pad_h or pad_w:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=IGNORE)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def grid_windows(image: np.ndarray, mask: np.ndarray, window: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every `window` x `window` piece of a pair, row by row, the last ones padded (for validation and scoring)."""
    _, h, w = image.shape
    pieces = []
    for top in range(0, max(h, 1), window):
        for left in range(0, max(w, 1), window):
            pieces.append(
                fit_window(
                    image[:, top : top + window, left : left + window],
                    mask[top : top + window, left : left + window],
                    window,
                )
            )
    return pieces


def scaled_side(
    images_dir: str | Path, names: Sequence[str], pre: PreprocessParams, source_gsd_m: float | None
) -> int:
    """Side of the first chip after scaling (all chips of one dataset are assumed to share zoom and size)."""
    path = Path(images_dir) / sorted(names)[0]
    with Image.open(path) as image:
        width, height = image.size
    scale = chip_scale(
        path.name,
        width,
        pre.model_gsd_m,
        height_px=height,
        max_input_px=pre.max_input_px,
        source_gsd_m=source_gsd_m,
    )
    return max(round(width * scale), round(height * scale))


def load_label_source(labels: str | Path | LabelSource) -> LabelSource:
    return labels if isinstance(labels, LabelSource) else LabelSource.from_path(labels)


def dataset_blocks(
    images_dir: str | Path,
    pre: PreprocessParams,
    *,
    names: Sequence[str] | None = None,
    source_gsd_m: float | None = None,
    window: int = WINDOW_PX,
) -> list[list[tuple[str, int, int]]]:
    """The mosaic blocks of a chips folder: neighbouring chips grouped so that one block fills about one window."""
    chip_names = [path.name for path in list_chip_paths(images_dir, names)]
    if not chip_names:
        raise ValueError(f"No chips found in {images_dir}")
    side = scaled_side(images_dir, chip_names, pre, source_gsd_m)
    return plan_blocks(chip_names, chips_per_side(side, window))
