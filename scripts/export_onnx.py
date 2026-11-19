"""Entry point to `ss_hot.export`: turn a saved `<name>.pt` into the serving ONNX (`image` -> `probability`).

    python scripts/export_onnx.py --weights data/runs/upernet/seagrass_upernet.pt \
        --out data/runs/upernet/seagrass_upernet.onnx --check

`--check` runs the ONNX with onnxruntime on random windows and compares it with torch (needs the `infer` extra as
well). Needs the `train` extra; nothing is downloaded.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from ss_hot.export import INPUT_NAME, build_model, export_onnx_bytes
from ss_hot.params import WINDOW_PX

PARITY_TOLERANCE = 1e-4

log = logging.getLogger("export_onnx")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="a SeagrassUPerNet state dict (<name>.pt)",
    )
    parser.add_argument("--out", required=True, type=Path, help="the .onnx file to write")
    parser.add_argument(
        "--check", action="store_true", help="compare onnxruntime with torch on random windows"
    )
    return parser


def _parity(model, data: bytes) -> float:
    """Largest absolute probability difference between torch and onnxruntime on three random windows."""
    import onnxruntime as ort
    import torch

    windows = np.random.default_rng(0).random((3, 3, WINDOW_PX, WINDOW_PX), dtype=np.float32)
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    (onnx_probability,) = session.run(None, {INPUT_NAME: windows})
    with torch.no_grad():
        torch_probability = torch.sigmoid(model(torch.from_numpy(windows)))[:, 0].numpy()
    return float(np.abs(onnx_probability - torch_probability).max())


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )  # fmt: skip
    if args.out.suffix != ".onnx":
        raise SystemExit(f"--out must end in .onnx, got {args.out}")
    model = build_model(weights=args.weights)
    data = export_onnx_bytes(model)
    if args.check:
        difference = _parity(model, data)
        if difference >= PARITY_TOLERANCE:
            raise SystemExit(
                f"ONNX differs from torch by {difference:.2e} (limit {PARITY_TOLERANCE:.0e}); not written"
            )
        log.info("parity: largest difference %.2e", difference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
