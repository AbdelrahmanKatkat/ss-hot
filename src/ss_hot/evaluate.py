"""Pixel precision / recall / F1 / IoU of the seagrass class (numpy; torch only for `score_model`).

Scoring happens in model pixels on the same windows as validation: chips scaled as at serving time, joined into
mosaic blocks and cut into windows (`dataset.py`). `IGNORE` pixels count for nothing. One prediction per window gives
the counts at every threshold at once, so a threshold sweep costs one pass.
"""

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import (
    IGNORE,
    SEAGRASS,
    LabelSource,
    assemble_block,
    dataset_blocks,
    grid_windows,
    load_label_source,
)
from .params import WINDOW_PX, PreprocessParams, parse_preprocess_params

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def pixel_counts(probability: np.ndarray, truth: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """`(len(thresholds), 3)` counts of true positive, false positive and false negative pixels at each threshold."""
    valid = truth != IGNORE
    is_seagrass = (truth == SEAGRASS) & valid
    is_background = ~is_seagrass & valid
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for row, threshold in enumerate(thresholds):
        predicted = probability >= threshold
        counts[row] = (
            (predicted & is_seagrass).sum(),
            (predicted & is_background).sum(),
            (~predicted & is_seagrass).sum(),
        )
    return counts


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def score_windows(
    images_dir: str | Path,
    labels: str | Path | LabelSource,
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    pre: PreprocessParams,
    names: Sequence[str] | None = None,
    source_gsd_m: float | None = None,
    thresholds: Sequence[float] = (0.5,),
    window: int = WINDOW_PX,
) -> dict[float, dict[str, float]]:
    """Metrics at each threshold; `predict` maps `(K, 3, w, w)` windows to `(K, w, w)` probabilities."""
    source = load_label_source(labels)
    totals = np.zeros((len(thresholds), 3), dtype=np.int64)
    started = time.time()
    blocks = dataset_blocks(images_dir, pre, names=names, source_gsd_m=source_gsd_m, window=window)
    for block in blocks:
        pieces = grid_windows(
            *assemble_block(block, images_dir, source, pre, source_gsd_m=source_gsd_m), window
        )
        probabilities = predict(np.stack([image for image, _ in pieces]))
        for probability, (_, mask) in zip(probabilities, pieces):
            totals += pixel_counts(probability, mask, thresholds)
    results = {float(t): metrics_from_counts(*row.tolist()) for t, row in zip(thresholds, totals)}
    for threshold, metrics in results.items():
        log.info(
            "evaluation on %d chips, threshold %.2f: precision=%.3f recall=%.3f f1=%.3f iou=%.3f in %.0fs",
            sum(len(block) for block in blocks), threshold, metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["iou"], time.time() - started,
        )  # fmt: skip
    return results


def score_session(
    session: Any,
    *,
    images_dir: str | Path,
    labels: str | Path | LabelSource,
    params: dict[str, Any] | None = None,
    names: Sequence[str] | None = None,
    source_gsd_m: float | None = None,
    thresholds: Sequence[float] = (0.5,),
) -> dict[float, dict[str, float]]:
    """Metrics through an ONNX session: the deployed path, with no torch involved."""
    input_name = session.get_inputs()[0].name
    return score_windows(
        images_dir, labels, lambda windows: session.run(None, {input_name: windows})[0],
        pre=parse_preprocess_params(params), names=names, source_gsd_m=source_gsd_m, thresholds=thresholds,
    )  # fmt: skip


def score_model(
    model: Any,
    *,
    images_dir: str | Path,
    labels: str | Path | LabelSource,
    params: dict[str, Any] | None = None,
    names: Sequence[str] | None = None,
    source_gsd_m: float | None = None,
    thresholds: Sequence[float] = (0.5,),
    batch_size: int = 8,
) -> dict[float, dict[str, float]]:
    """Metrics of a torch `SeagrassUPerNet` (needs the `train` extra), wherever the model lives (CPU or GPU)."""
    import torch

    model.eval()
    device = next(model.parameters()).device

    def predict(windows: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
                outputs.append(torch.sigmoid(model(batch))[:, 0].float().cpu().numpy())
        return np.concatenate(outputs)

    return score_windows(
        images_dir, labels, predict, pre=parse_preprocess_params(params), names=names,
        source_gsd_m=source_gsd_m, thresholds=thresholds,
    )  # fmt: skip


def best_threshold(
    results: dict[float, dict[str, float]], metric: str = "f1"
) -> tuple[float, dict[str, float]]:
    """The threshold with the highest `metric` in a sweep, and its metrics."""
    threshold = max(results, key=lambda key: results[key][metric])
    return threshold, results[threshold]
