"""Entry point to `ss_hot.evaluate`: pixel precision / recall / F1 / IoU of an exported ONNX at several thresholds (CPU).

    python scripts/evaluate.py --onnx data/runs/upernet/seagrass_upernet.onnx \
        --images <split>/chips --labels <split>/labels/labels.geojson --out data/runs/upernet/val_metrics.json

Run it on the validation chips to choose the serving threshold (the best F1 of the sweep), then on test.

Needs `onnxruntime` (the `infer` extra); no torch.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import onnxruntime as ort

from ss_hot.evaluate import DEFAULT_THRESHOLDS, best_threshold, score_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path, help="labels GeoJSON (or a folder holding one)")
    parser.add_argument("--params", default="{}", help="JSON of PreprocessParams keys")
    parser.add_argument("--out", type=Path, help="write the metrics as JSON")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout
    )
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    results = score_session(
        session, images_dir=args.images, labels=args.labels, params=json.loads(args.params),
        thresholds=DEFAULT_THRESHOLDS,
    )  # fmt: skip
    threshold, metrics = best_threshold(results)
    logging.getLogger("evaluate").info(
        "best f1 %.3f (iou %.3f) at threshold %.2f", metrics["f1"], metrics["iou"], threshold
    )
    if args.out:
        args.out.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
