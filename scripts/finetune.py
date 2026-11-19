"""Entry point to `ss_hot.train`: fine-tune the seagrass UPerNet and write `<name>.pt`, `<name>.onnx` and `history.json`.

    python scripts/finetune.py --train-images F:/HOTOSM/seagrass/data/train/images \
        --train-labels F:/HOTOSM/seagrass/data/train/masks --val-images .../validation/images \
        --val-labels .../validation/masks --source-gsd-m 0.0092 --out data/runs/upernet --params '{"epochs": 30}'

Labels are a mask folder (`<stem>.png`, 0 / 1 / 255) or one GeoJSON over georeferenced chips. `--source-gsd-m` is
needed only when chip names do not describe their real footprint. Training options go in `--params` (JSON, the
`TrainParams` keys); anything left out keeps its default. Needs the `train` extra; with a local CUDA torch use the
venv python (or `uv run --no-sync`), not a plain `uv run`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from ss_hot.export import export_onnx_bytes
from ss_hot.params import parse_preprocess_params, parse_train_params
from ss_hot.train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-labels", type=Path)
    parser.add_argument("--source-gsd-m", type=float, help="metres per chip pixel, if chip names are wrong")
    parser.add_argument(
        "--weights", type=Path, help="a <name>.pt to continue; default: ImageNet encoder weights"
    )
    parser.add_argument("--out", required=True, type=Path, help="output folder; must be empty or new")
    parser.add_argument("--name", default="seagrass_upernet", help="output file stem")
    parser.add_argument("--params", default="{}", help="JSON of TrainParams and PreprocessParams keys")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; choose a new --out")
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.out / "train.log", encoding="utf-8")],
    )  # fmt: skip
    params = json.loads(args.params)
    result = train(
        train_images_dir=args.train_images, train_labels=args.train_labels,
        val_images_dir=args.val_images, val_labels=args.val_labels,
        cfg=parse_train_params(params), pre=parse_preprocess_params(params),
        checkpoint=args.weights, pretrained=args.weights is None, source_gsd_m=args.source_gsd_m,
        num_workers=args.num_workers, epoch_checkpoint_dir=args.out / "epochs",
    )  # fmt: skip
    torch.save(result.model.state_dict(), args.out / f"{args.name}.pt")
    (args.out / f"{args.name}.onnx").write_bytes(export_onnx_bytes(result.model))
    history = {**result.history, "best_epoch": result.best_epoch + 1, "best_value": result.best_value}
    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    logging.getLogger("finetune").info(
        "wrote %s.pt, %s.onnx and history.json in %s", args.name, args.name, args.out
    )


if __name__ == "__main__":
    main()
