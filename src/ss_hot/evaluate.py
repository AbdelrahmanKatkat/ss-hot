"""Pixel precision / recall / F1 / IoU of the seagrass class (numpy; torch only for `score_model`).

Scoring follows the serving path: each chip is scaled as at serving time, cut into windows, predicted, stitched back
and compared with its labels at the model resolution (`dataset.py`, `preprocess.py`). One prediction per chip gives the
counts at every threshold at once, so a threshold sweep costs one pass; run it on validation chips and serve with the
best threshold (`best_threshold`).
"""

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import SEAGRASS, LabelSource, list_chip_paths, load_label_source, scale_pair
from .params import WINDOW_PX, PreprocessParams, parse_preprocess_params
from .preprocess import stitch_grid, to_windows

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def pixel_counts(probability: np.ndarray, truth: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """`(len(thresholds), 3)` counts of true positive, false positive and false negative pixels at each threshold."""
    is_seagrass = truth == SEAGRASS
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for row, threshold in enumerate(thresholds):
        predicted = probability >= threshold
        counts[row] = (
            (predicted & is_seagrass).sum(),
            (predicted & ~is_seagrass).sum(),
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


def predict_chip_probability(
    image: np.ndarray, predict: Callable[[np.ndarray], np.ndarray], window: int = WINDOW_PX
) -> np.ndarray:
    """A scaled `(3, H, W)` chip -> its `(H, W)` probability, window by window and stitched, as at serving."""
    windows, _, _ = to_windows(image, window)
    return stitch_grid(predict(windows), *image.shape[1:])


def score_chips(
    images_dir: str | Path,
    labels: str | Path | LabelSource,
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    pre: PreprocessParams,
    names: Sequence[str] | None = None,
    thresholds: Sequence[float] = (0.5,),
) -> dict[float, dict[str, float]]:
    """Metrics at each threshold; `predict` maps `(K, 3, w, w)` windows to `(K, w, w)` probabilities."""
    source = load_label_source(labels)
    paths = list_chip_paths(images_dir, names)
    totals = np.zeros((len(thresholds), 3), dtype=np.int64)
    started = time.time()
    for path in paths:
        image, mask = scale_pair(path, source, pre)
        totals += pixel_counts(predict_chip_probability(image, predict), mask, thresholds)
    results = {float(t): metrics_from_counts(*row.tolist()) for t, row in zip(thresholds, totals)}
    for threshold, metrics in results.items():
        log.info(
            "evaluation on %d chips, threshold %.2f: precision=%.3f recall=%.3f f1=%.3f iou=%.3f in %.0fs",
            len(paths), threshold, metrics["precision"], metrics["recall"], metrics["f1"], metrics["iou"],
            time.time() - started,
        )  # fmt: skip
    return results


def score_session(
    session: Any,
    *,
    images_dir: str | Path,
    labels: str | Path | LabelSource,
    params: dict[str, Any] | None = None,
    names: Sequence[str] | None = None,
    thresholds: Sequence[float] = (0.5,),
) -> dict[float, dict[str, float]]:
    """Metrics through an ONNX session: the deployed path, with no torch involved."""
    input_name = session.get_inputs()[0].name
    return score_chips(
        images_dir, labels, lambda windows: session.run(None, {input_name: windows})[0],
        pre=parse_preprocess_params(params), names=names, thresholds=thresholds,
    )  # fmt: skip


def score_model(
    model: Any,
    *,
    images_dir: str | Path,
    labels: str | Path | LabelSource,
    params: dict[str, Any] | None = None,
    names: Sequence[str] | None = None,
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

    return score_chips(
        images_dir, labels, predict, pre=parse_preprocess_params(params), names=names, thresholds=thresholds
    )


def best_threshold(
    results: dict[float, dict[str, float]], metric: str = "f1"
) -> tuple[float, dict[str, float]]:
    """The threshold with the highest `metric` in a sweep, and its metrics."""
    threshold = max(results, key=lambda key: results[key][metric])
    return threshold, results[threshold]
