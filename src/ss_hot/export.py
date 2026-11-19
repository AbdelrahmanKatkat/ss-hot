"""Building the seagrass UPerNet and its ONNX export (needs the `train` extra; torch is imported inside the functions).

The exported graph is the contract the serving side reads:

    input  `image`        (N, 3, 512, 512) float32 in [0, 1], RGB windows cut by `preprocess.to_windows`
    output `probability`  (N, 512, 512) float32 in [0, 1], the seagrass probability of every pixel

The batch axis is dynamic; the window size is fixed (`params.WINDOW_PX`) because the pyramid pooling exports only for
a static spatial size. ImageNet normalisation, the network and the sigmoid are inside the graph; the threshold and
the polygon steps (`decode.py`) stay outside so they remain tunable.
"""

import logging
from pathlib import Path
from typing import BinaryIO

from .params import WINDOW_PX

log = logging.getLogger(__name__)

OPSET = 17  # LayerNormalization is an ONNX operator from opset 17 on
INPUT_NAME = "image"
OUTPUT_NAME = "probability"


def build_model(
    *, pretrained: bool = False, weights: str | Path | BinaryIO | None = None, drop_path_rate: float = 0.0
):
    """A `SeagrassUPerNet` in eval mode: ImageNet encoder weights (`pretrained`, downloaded by timm from the Hugging
    Face Hub on first use), a saved `SeagrassUPerNet` state dict (`weights`, a path or binary stream), or random."""
    import torch

    from .model import SeagrassUPerNet

    if pretrained and weights is not None:
        raise ValueError(
            "give either pretrained (to start a new model) or weights (to continue one), not both"
        )
    model = SeagrassUPerNet(pretrained=pretrained, drop_path_rate=drop_path_rate)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def probability_network(model):
    """Wrap `model` (switched to eval mode) as `image (N, 3, H, W) -> probability (N, H, W)`, the exported contract."""
    import torch

    class ProbabilityNetwork(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            return torch.sigmoid(self.model(image))[:, 0]

    return ProbabilityNetwork(model.eval()).eval()


def export_onnx_bytes(model, *, opset: int = OPSET, window: int = WINDOW_PX) -> bytes:
    """Export `probability_network(model)` with a dynamic batch; validate the file and return its bytes."""
    import tempfile

    import onnx
    import torch

    network = probability_network(model).cpu()
    example = torch.rand(2, 3, window, window)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "ss_hot.onnx"
        torch.onnx.export(
            network, (example,), str(path), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}}, opset_version=opset, dynamo=False,
        )  # fmt: skip
        onnx.checker.check_model(onnx.load(str(path)))
        data = path.read_bytes()
    log.info("exported ONNX (opset %d, %s -> %s): %.1f MB", opset, INPUT_NAME, OUTPUT_NAME, len(data) / 1e6)
    return data
