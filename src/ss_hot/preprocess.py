"""Chip -> model input: one resize to the model's ground resolution, then fixed-size windows.

1. `scale_rgb`: RGB uint8 `(H, W, 3)` -> float32 `(3, H', W')` in [0, 1], one resize (`H' = round(H * scale)`).
2. `to_windows`: pad the scaled chip (reflected image) to whole `WINDOW_PX` windows -> `(N, 3, WINDOW_PX, WINDOW_PX)`.

The ONNX graph normalises (ImageNet statistics) itself, so this module only scales, pads and cuts. `stitch_windows`
puts the per-window probabilities back together and crops the padding, giving one `(H', W')` probability per chip;
`PreprocessMeta` then maps it back to chip pixels and to the map.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .params import WINDOW_PX, PreprocessParams
from .tiles import chip_scale


@dataclass(frozen=True)
class PreprocessMeta:
    """Everything needed to map a prediction back to chip pixels and then to the map."""

    width: int  # source chip, pixels
    height: int
    resized_width: int  # scaled chip (model pixels), before padding
    resized_height: int
    rows: int  # windows down and across the padded chip
    cols: int
    transform: Any = None  # the chip's affine transform (identity-like for a chip without georeferencing)
    crs: Any = None

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.height


def read_chip(image_path: str | Path) -> tuple[np.ndarray, Any, Any]:
    """Bands 1-3 as uint8 `(H, W, 3)` RGB plus the chip's affine transform and CRS.

    A PNG/JPEG (or a TIFF without georeferencing) has no CRS: the transform is the identity and the CRS is `None`.
    """
    import rasterio

    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError(f"{image_path}: expected >= 3 bands (RGB), found {src.count}")
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        return np.clip(rgb, 0, 255).astype(np.uint8), src.transform, src.crs


def scale_rgb(rgb_hwc: np.ndarray, scale: float) -> np.ndarray:
    """RGB uint8 `(H, W, 3)` -> float32 `(3, H', W')` in [0, 1], one bilinear resize (antialiased when shrinking)."""
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB array, got shape {rgb_hwc.shape}")
    h, w, _ = rgb_hwc.shape
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    chw = rgb_hwc.astype(np.float32).transpose(2, 0, 1) / 255.0
    if (new_h, new_w) == (h, w):
        return chw
    resized = np.empty((3, new_h, new_w), dtype=np.float32)
    for channel in range(3):
        resized[channel] = np.asarray(
            Image.fromarray(chw[channel], mode="F").resize((new_w, new_h), Image.Resampling.BILINEAR)
        )
    return np.clip(resized, 0.0, 1.0)


def window_grid(height: int, width: int, window: int = WINDOW_PX) -> tuple[int, int]:
    """Rows and columns of `window`-sized windows that cover a `height` x `width` image."""
    return max(1, -(-height // window)), max(1, -(-width // window))


def to_windows(image: np.ndarray, window: int = WINDOW_PX) -> tuple[np.ndarray, int, int]:
    """`(3, H, W)` -> `(rows * cols, 3, window, window)`: padded at the bottom and right with a reflection of the image."""
    _, h, w = image.shape
    rows, cols = window_grid(h, w, window)
    padded = np.pad(image, ((0, 0), (0, rows * window - h), (0, cols * window - w)), mode="symmetric")
    windows = (
        padded.reshape(3, rows, window, cols, window).transpose(1, 3, 0, 2, 4).reshape(-1, 3, window, window)
    )
    return np.ascontiguousarray(windows, dtype=np.float32), rows, cols


def stitch_windows(probabilities: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    """`(rows * cols, window, window)` window outputs -> the `(H', W')` probability of the scaled chip (padding cut)."""
    windows = np.asarray(probabilities, dtype=np.float32)
    window = windows.shape[-1]
    full = windows.reshape(meta.rows, meta.cols, window, window).transpose(0, 2, 1, 3)
    full = full.reshape(meta.rows * window, meta.cols * window)
    return full[: meta.resized_height, : meta.resized_width]


def preprocess_rgb(
    rgb_hwc: np.ndarray, scale: float, *, transform: Any = None, crs: Any = None, window: int = WINDOW_PX
) -> tuple[np.ndarray, PreprocessMeta]:
    """RGB uint8 chip -> `(N, 3, window, window)` float32 windows in [0, 1] plus `PreprocessMeta`."""
    image = scale_rgb(rgb_hwc, scale)
    windows, rows, cols = to_windows(image, window)
    meta = PreprocessMeta(
        width=rgb_hwc.shape[1], height=rgb_hwc.shape[0], resized_width=image.shape[2],
        resized_height=image.shape[1], rows=rows, cols=cols, transform=transform, crs=crs,
    )  # fmt: skip
    return windows, meta


def preprocess_chip(
    image_path: str | Path, cfg: PreprocessParams | None = None, *, source_gsd_m: float | None = None
) -> tuple[np.ndarray, PreprocessMeta]:
    """Read a chip named `OAM-<x>-<y>-<z>.<ext>`, derive the scale from its zoom (or `source_gsd_m`), preprocess it."""
    cfg = cfg or PreprocessParams()
    rgb, transform, crs = read_chip(image_path)
    scale = chip_scale(
        Path(image_path).name, rgb.shape[1], cfg.model_gsd_m, height_px=rgb.shape[0],
        max_input_px=cfg.max_input_px, source_gsd_m=source_gsd_m,
    )  # fmt: skip
    return preprocess_rgb(rgb, scale, transform=transform, crs=crs)
