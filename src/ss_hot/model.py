"""UPerNet on a ConvNeXt-Tiny encoder (needs the `train` extra; import explicitly, not from `ss_hot`).

Encoder: timm `convnext_tiny.fb_in22k_ft_in1k` (ConvNeXt v1, ImageNet-22k pre-trained and fine-tuned on ImageNet-1k,
Apache-2.0), used as a feature extractor of its four stages (strides 4, 8, 16, 32). Decoder: UPerNet (Xiao et al.,
2018): a pyramid pooling module on the deepest stage, a top-down feature pyramid, and the fusion of all levels at
stride 4. The model takes RGB in [0, 1] and returns one seagrass logit per pixel at the input size; the ImageNet
normalisation happens inside, so training and serving cannot disagree about it.

The pooling scales are (1, 2, 4, 8) instead of the usual (1, 2, 3, 6): on a 256 px window the deepest stage is 8 x 8,
and only scales that divide it export to ONNX as plain average pooling.
"""

import torch
import torch.nn.functional as functional
from torch import nn

from .params import IMAGE_MEAN, IMAGE_STD

ENCODER_NAME = "convnext_tiny.fb_in22k_ft_in1k"
DECODER_CHANNELS = 256
POOL_SCALES = (1, 2, 4, 8)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPooling(nn.Module):
    """Average-pool the deepest features to each scale, project, upsample back, and fuse with the input."""

    def __init__(self, in_channels: int, channels: int, scales: tuple[int, ...] = POOL_SCALES) -> None:
        super().__init__()
        # The 1 x 1 bin has no BatchNorm: one value per channel and image cannot be normalised in a batch of one.
        self.stages = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Sequential(nn.Conv2d(in_channels, channels, 1), nn.ReLU(inplace=True))
                if scale == 1
                else ConvBNReLU(in_channels, channels),
            )
            for scale in scales
        )
        self.bottleneck = ConvBNReLU(in_channels + len(scales) * channels, channels, kernel_size=3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        size = features.shape[-2:]
        pooled = [
            functional.interpolate(stage(features), size=size, mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat([features, *pooled], dim=1))


class UPerHead(nn.Module):
    """UPerNet decoder: PPM on the deepest level, a top-down pyramid, all levels fused at the shallowest stride."""

    def __init__(self, in_channels: list[int], channels: int = DECODER_CHANNELS) -> None:
        super().__init__()
        self.ppm = PyramidPooling(in_channels[-1], channels)
        self.lateral = nn.ModuleList(ConvBNReLU(c, channels) for c in in_channels[:-1])
        self.smooth = nn.ModuleList(ConvBNReLU(channels, channels, kernel_size=3) for _ in in_channels[:-1])
        self.fuse = ConvBNReLU(len(in_channels) * channels, channels, kernel_size=3)
        self.out_channels = channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        levels = [lateral(feature) for lateral, feature in zip(self.lateral, features[:-1])]
        levels.append(self.ppm(features[-1]))
        for index in range(len(levels) - 1, 0, -1):  # top-down: add each coarser level to the finer one
            levels[index - 1] = levels[index - 1] + functional.interpolate(
                levels[index], size=levels[index - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        outputs = [smooth(level) for smooth, level in zip(self.smooth, levels[:-1])] + [levels[-1]]
        size = outputs[0].shape[-2:]
        outputs = [outputs[0]] + [
            functional.interpolate(level, size=size, mode="bilinear", align_corners=False)
            for level in outputs[1:]
        ]
        return self.fuse(torch.cat(outputs, dim=1))


class SeagrassUPerNet(nn.Module):
    """RGB in [0, 1], `(B, 3, H, W)` -> seagrass logits `(B, 1, H, W)`; `H` and `W` multiples of 32."""

    def __init__(
        self, *, pretrained: bool = False, drop_path_rate: float = 0.0, encoder_name: str = ENCODER_NAME
    ):
        super().__init__()
        import timm

        self.register_buffer("mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1), persistent=False)
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
        )  # fmt: skip
        self.decoder = UPerHead(list(self.encoder.feature_info.channels()))
        self.dropout = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder((image - self.mean) / self.std)
        logits = self.head(self.dropout(self.decoder(features)))
        return functional.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def encoder_stage_modules(model: SeagrassUPerNet) -> dict[str, nn.Module]:
    """The encoder's four ConvNeXt stages by `params.ENCODER_STAGES` name (`stage0` .. `stage3`)."""
    children = dict(model.encoder.named_children())
    stages = {}
    for index in range(4):
        module = children.get(f"stages_{index}") or children.get(f"stages.{index}")
        if module is None:
            raise KeyError(f"encoder has no stage {index}; children are {sorted(children)}")
        stages[f"stage{index}"] = module
    return stages
