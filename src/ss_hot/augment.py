"""Training-time changes to a tile and its mask (numpy and Pillow only).

Used only on training tiles; validation, test and serving never change a tile. `AugmentParams` (`params.py`) sets
which steps run and how strongly; every step is off until asked for. The image and the mask change together, so a
label stays on its pixels. Two steps need a second training tile (`partner`, same size); without one they are
skipped. `augment_chip_and_mask` applies, in order:

1. orientation: a random quarter turn and mirror (`dihedral`) or nothing (`none`);
2. CutMix: a random box of the partner tile (image and mask) pasted in (Yun et al. 2019);
3. scale jitter: zoom in by a random factor from 1 to 1 + `scale_jitter`, then crop back to the same size (zoom-in
   only, so no padding is needed);
4. histogram matching: each channel matched to the partner's histogram, blended with the original by a random
   weight (randomised histogram matching, Yaras et al. 2024; image only);
5. photometric: brightness, contrast and gamma (image only);
6. low resolution: a resample down by 1 to `lowres_max` and back up, like a coarser tile upscaled at serving
   (image only);
7. degrade: a Gaussian blur or a JPEG round trip (image only).

There is no hue or saturation change: seagrass differs from sand by being darker and slightly bluer, and no source
gives a safe range (docs/augmentation.md).
"""

import io

import numpy as np
from PIL import Image, ImageFilter

from .params import AugmentParams


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize of a `(3, H, W)` float image."""
    return np.stack(
        [
            np.asarray(Image.fromarray(channel, mode="F").resize((width, height), Image.Resampling.BILINEAR))
            for channel in image
        ]
    )


def _cutmix(
    image: np.ndarray, mask: np.ndarray, partner: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Paste a random box of the partner (side x sqrt(1 - lambda), lambda uniform) into the image and the mask."""
    height, width = mask.shape
    ratio = float(np.sqrt(1.0 - rng.uniform(0.0, 1.0)))
    box_h, box_w = max(1, round(height * ratio)), max(1, round(width * ratio))
    top, left = int(rng.integers(0, height - box_h + 1)), int(rng.integers(0, width - box_w + 1))
    image, mask = image.copy(), mask.copy()
    image[:, top : top + box_h, left : left + box_w] = partner[0][:, top : top + box_h, left : left + box_w]
    mask[top : top + box_h, left : left + box_w] = partner[1][top : top + box_h, left : left + box_w]
    return image, mask


def _zoom_in(
    image: np.ndarray, mask: np.ndarray, factor: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Enlarge the pair by `factor` >= 1 (image bilinear, mask nearest) and crop a random piece of the original size."""
    height, width = mask.shape
    new_h, new_w = max(height, round(height * factor)), max(width, round(width * factor))
    resized = _resize(image, new_w, new_h)
    resized_mask = np.asarray(Image.fromarray(mask).resize((new_w, new_h), Image.Resampling.NEAREST))
    top, left = int(rng.integers(0, new_h - height + 1)), int(rng.integers(0, new_w - width + 1))
    return resized[:, top : top + height, left : left + width], resized_mask[
        top : top + height, left : left + width
    ]


def match_histograms(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Each channel of `image` mapped so that its value distribution equals the one of `reference` (both `(3, H, W)`)."""
    matched = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[0]):
        values, inverse, counts = np.unique(image[channel].ravel(), return_inverse=True, return_counts=True)
        ref_values, ref_counts = np.unique(reference[channel].ravel(), return_counts=True)
        quantiles = np.cumsum(counts) / counts.sum()
        ref_quantiles = np.cumsum(ref_counts) / ref_counts.sum()
        matched[channel] = np.interp(quantiles, ref_quantiles, ref_values)[inverse].reshape(image.shape[1:])
    return matched


def _photometric(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Random brightness, contrast (each x 1 +- `strength`) and gamma (exp(+- `strength`)) on an image in [0, 1]."""
    brightness, contrast = rng.uniform(1 - strength, 1 + strength, size=2)
    gamma = float(np.exp(rng.uniform(-strength, strength)))
    bright = image * brightness
    image = np.clip((bright - bright.mean()) * contrast + bright.mean(), 0.0, 1.0)
    return np.power(image, gamma)


def _lowres(image: np.ndarray, max_factor: float, rng: np.random.Generator) -> np.ndarray:
    """Resample down by a random factor in [1, `max_factor`] and back up, bilinear (a coarser tile, upscaled)."""
    factor = float(rng.uniform(1.0, max_factor))
    _, h, w = image.shape
    return _resize(_resize(image, max(1, round(w / factor)), max(1, round(h / factor))), w, h)


def _degrade(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A Gaussian blur (radius 0.5 to 1.5 px) or a JPEG round trip (quality 40 to 90), chosen at random."""
    picture = Image.fromarray((image.transpose(1, 2, 0) * 255).round().astype(np.uint8))
    if rng.integers(0, 2):
        picture = picture.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
    else:
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=int(rng.integers(40, 91)))
        picture = Image.open(buffer).convert("RGB")
    return np.asarray(picture, dtype=np.float32).transpose(2, 0, 1) / 255.0


def augment_chip_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams | None = None,
    partner: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Change a `(3, H, W)` image in [0, 1] and its `(H, W)` mask together; the size never changes.

    `partner` is another training tile `(image, mask)` of the same size, for CutMix and histogram matching.
    """
    params = params or AugmentParams()
    if params.orientation == "dihedral":
        turns = int(rng.integers(0, 4))
        image, mask = np.rot90(image, turns, axes=(1, 2)), np.rot90(mask, turns, axes=(0, 1))
        if rng.integers(0, 2):
            image, mask = image[:, :, ::-1], mask[:, ::-1]
    image, mask = np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask)
    if partner is not None and params.cutmix_prob > 0 and rng.random() < params.cutmix_prob:
        image, mask = _cutmix(image, mask, partner, rng)
    if params.scale_jitter > 0:
        image, mask = _zoom_in(image, mask, float(rng.uniform(1.0, 1.0 + params.scale_jitter)), rng)
    if partner is not None and params.histogram_prob > 0 and rng.random() < params.histogram_prob:
        weight = float(rng.uniform(0.0, 1.0))
        image = (1.0 - weight) * image + weight * match_histograms(image, partner[0])
    if params.photometric > 0:
        image = _photometric(image, params.photometric, rng)
    if params.lowres_prob > 0 and rng.random() < params.lowres_prob:
        image = _lowres(np.ascontiguousarray(image, dtype=np.float32), params.lowres_max, rng)
    if params.degrade_prob > 0 and rng.random() < params.degrade_prob:
        image = _degrade(np.clip(image, 0.0, 1.0), rng)
    return np.ascontiguousarray(np.clip(image, 0.0, 1.0), dtype=np.float32), np.ascontiguousarray(mask)
