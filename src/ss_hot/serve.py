"""Inference: directory of OAM chips -> scale -> windows -> ONNX -> seagrass probability. Nothing is decoded here.

1. `preprocess_chip`: `model_gsd_m`, `max_input_px` (the scale), then fixed-size windows.
2. the ONNX graph (`image (N, 3, 512, 512)` -> `probability (N, 512, 512)`): no keys.

`predict_session` returns each chip's stitched probability and metadata. The rest is the postprocess
(`postprocess.py`, `postprocess_outputs`): decode with the `PostprocessParams` keys, then georeference to EPSG:4326.
A fAIr pack's `predict` calls both.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .params import parse_preprocess_params
from .preprocess import PreprocessMeta, preprocess_chip, stitch_windows

log = logging.getLogger(__name__)


def list_images(input_dir: str | Path) -> list[Path]:
    """Chip files of a directory (png / tif / jpg)."""
    root = Path(input_dir)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    files = sorted(path for pattern in patterns for path in root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input images found in {root}")
    return files


def run_windows(session: Any, windows: np.ndarray) -> np.ndarray:
    """All windows of one chip through the session in one call: `(N, 3, w, w)` -> `(N, w, w)` probability."""
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: windows})[0]


def predict_chip(
    session: Any, image_path: str | Path, params: dict[str, Any] | None
) -> tuple[np.ndarray, PreprocessMeta]:
    """One chip, stages 1 and 2: its `(H', W')` seagrass probability in model pixels and the metadata to map it back."""
    windows, meta = preprocess_chip(image_path, parse_preprocess_params(params))
    log.debug(
        "%s: scale %.3f -> %dx%d px, %d window(s)",
        Path(image_path).name, meta.scale_x, meta.resized_width, meta.resized_height, len(windows),
    )  # fmt: skip
    return stitch_windows(run_windows(session, windows), meta), meta


def predict_session(
    session: Any, input_images: str | Path, params: dict[str, Any] | None
) -> list[tuple[Path, np.ndarray, PreprocessMeta]]:
    """Inference for every chip of a directory: `[(chip path, seagrass probability, metadata), ...]`.

    Nothing is decoded and nothing is georeferenced here; `postprocess_outputs` does that.
    """
    started = time.time()
    outputs = [(path, *predict_chip(session, path, params)) for path in list_images(input_images)]
    log.info("predict_session: %d chips in %.1fs", len(outputs), time.time() - started)
    return outputs
