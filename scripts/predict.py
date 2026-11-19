"""Entry point to `ss_hot.predict_session` then `postprocess_outputs`: run an exported ONNX on a folder of chips (CPU).

    python scripts/predict.py --onnx data/runs/upernet/seagrass_upernet.onnx --chips <folder of OAM chips> \
        --out predictions.geojson --params '{"threshold": 0.5}'

Chips are named `OAM-<x>-<y>-<z>.<ext>`. Needs `onnxruntime` (the `infer` extra); no torch.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import onnxruntime as ort

from ss_hot import postprocess_outputs, predict_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--chips", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--params", default="{}", help='JSON, for example {"threshold": 0.5, "min_area_px": 278}'
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout
    )
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    params = json.loads(args.params)
    collection = postprocess_outputs(predict_session(session, args.chips, params), params)
    args.out.write_text(json.dumps(collection), encoding="utf-8")
    logging.getLogger("predict").info("wrote %d features to %s", len(collection["features"]), args.out)


if __name__ == "__main__":
    main()
