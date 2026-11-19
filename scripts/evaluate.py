"""Entry point to `ss_hot.evaluate`: pixel precision / recall / F1 / IoU of an exported ONNX at several thresholds (CPU).

    python scripts/evaluate.py --onnx data/runs/upernet/seagrass_upernet.onnx \
        --images F:/HOTOSM/seagrass/data/test/images --labels F:/HOTOSM/seagrass/data/test/masks \
        --source-gsd-m 0.0092 --out data/runs/upernet/test_metrics.json

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
    parser.add_argument("--labels", required=True, type=Path, help="mask folder or labels GeoJSON")
    parser.add_argument("--source-gsd-m", type=float)
    parser.add_argument("--params", default="{}", help="JSON of PreprocessParams keys")
    parser.add_argument("--out", type=Path, help="write the metrics as JSON")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout
    )
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    results = score_session(
        session, images_dir=args.images, labels=args.labels, params=json.loads(args.params),
        source_gsd_m=args.source_gsd_m, thresholds=DEFAULT_THRESHOLDS,
    )  # fmt: skip
    threshold, metrics = best_threshold(results)
    logging.getLogger("evaluate").info(
        "best f1 %.3f (iou %.3f) at threshold %.2f", metrics["f1"], metrics["iou"], threshold
    )
    if args.out:
        args.out.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
