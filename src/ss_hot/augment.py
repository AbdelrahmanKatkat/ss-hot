"""Training-time changes to a chip and its mask (numpy and Pillow only).

Used only on training windows; validation, test and serving never change a chip. `AugmentParams` (`params.py`) sets
which steps run and how strongly; every step is off until asked for. The image and the mask change together, so a
label stays on its pixels. `augment_chip_and_mask` applies, in order:

1. orientation: a random quarter turn and mirror (`dihedral`), a mirror only (`mirror`), or nothing (`none`);
2. scale jitter: resize by a random factor near 1, then crop or pad back to the same size (padding is `IGNORE`);
3. photometric: brightness, contrast and gamma (image only);
4. stretch: contrast and saturation each x exp(+- `stretch`), around the chip's mean (image only);
5. water tint: blend toward a random blue-green water colour, red fading most (turbidity, depth; image only);
6. low resolution: a resample down by 1 to `lowres_max` and back up (coarser serving imagery; image only);
7. degrade: a Gaussian blur or a JPEG round trip (image only).

There is no hue rotation: seagrass differs from sand by being darker and slightly bluer, and a hue change can flip
that cue (docs/augmentation.md).
"""

import io

import numpy as np
from PIL import Image, ImageFilter

from .dataset import IGNORE
from .params import AugmentParams

# Water colours (RGB in [0, 1]) the tint step blends toward: clear turquoise to murky green.
WATER_LOW = np.array([0.15, 0.30, 0.30], dtype=np.float32)
WATER_HIGH = np.array([0.45, 0.65, 0.60], dtype=np.float32)
# Relative blend weight per channel: water absorbs red first (Akkaynak & Treibitz 2018).
TINT_WEIGHTS = np.array([1.0, 0.7, 0.5], dtype=np.float32)
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _resize_and_fit(
    image: np.ndarray, mask: np.ndarray, factor: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resize the pair by `factor` (image bilinear, mask nearest) and crop or pad back to the original size."""
    height, width = mask.shape
    new_h, new_w = max(2, round(height * factor)), max(2, round(width * factor))
    resized = np.stack(
        [
            np.asarray(Image.fromarray(channel, mode="F").resize((new_w, new_h), Image.BILINEAR))
            for channel in image
        ]
    )
    resized_mask = np.asarray(Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST))
    if factor >= 1.0:
        top, left = int(rng.integers(0, new_h - height + 1)), int(rng.integers(0, new_w - width + 1))
        return (
            resized[:, top : top + height, left : left + width],
            resized_mask[top : top + height, left : left + width],
        )
    pad_h, pad_w = height - new_h, width - new_w
    top, left = int(rng.integers(0, pad_h + 1)), int(rng.integers(0, pad_w + 1))
    spec = ((top, pad_h - top), (left, pad_w - left))
    return (
        np.pad(resized, ((0, 0), *spec), mode="symmetric"),
        np.pad(resized_mask, spec, mode="constant", constant_values=IGNORE),
    )


def _photometric(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Random brightness, contrast (each x 1 +- `strength`) and gamma (exp(+- `strength`)) on an image in [0, 1]."""
    brightness, contrast = rng.uniform(1 - strength, 1 + strength, size=2)
    gamma = float(np.exp(rng.uniform(-strength, strength)))
    bright = image * brightness
    image = np.clip((bright - bright.mean()) * contrast + bright.mean(), 0.0, 1.0)
    return np.power(image, gamma)


def _stretch(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Contrast (around the mean) and saturation (around the grey of each pixel) each x exp(+- `strength`)."""
    contrast, saturation = np.exp(rng.uniform(-strength, strength, size=2))
    image = (image - image.mean()) * contrast + image.mean()
    grey = np.einsum("c,chw->hw", LUMA, image)[None]
    return np.clip(grey + (image - grey) * saturation, 0.0, 1.0)


def _water_tint(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Blend toward a random water colour with a weight up to `strength`, red most and blue least."""
    colour = rng.uniform(WATER_LOW, WATER_HIGH).astype(np.float32)[:, None, None]
    weight = (float(rng.uniform(0.0, strength)) * TINT_WEIGHTS)[:, None, None]
    return (1.0 - weight) * image + weight * colour


def _lowres(image: np.ndarray, max_factor: float, rng: np.random.Generator) -> np.ndarray:
    """Resample down by a random factor in [1, `max_factor`] and back up, bilinear (a coarser image, upscaled)."""
    factor = float(rng.uniform(1.0, max_factor))
    _, h, w = image.shape
    small = (max(1, round(w / factor)), max(1, round(h / factor)))
    return np.stack(
        [
            np.asarray(
                Image.fromarray(channel, mode="F")
                .resize(small, Image.BILINEAR)
                .resize((w, h), Image.BILINEAR)
            )
            for channel in image
        ]
    )


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
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, params: AugmentParams | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Change a `(3, H, W)` image in [0, 1] and its `(H, W)` mask together; the size never changes."""
    params = params or AugmentParams()
    if params.orientation != "none":
        turns = int(rng.integers(0, 4)) if params.orientation == "dihedral" else 0
        image, mask = np.rot90(image, turns, axes=(1, 2)), np.rot90(mask, turns, axes=(0, 1))
        if rng.integers(0, 2):
            image, mask = image[:, :, ::-1], mask[:, ::-1]
    if params.scale_jitter > 0:
        factor = float(rng.uniform(1 - params.scale_jitter, 1 + params.scale_jitter))
        image, mask = _resize_and_fit(np.ascontiguousarray(image), np.ascontiguousarray(mask), factor, rng)
    if params.photometric > 0:
        image = _photometric(image, params.photometric, rng)
    if params.stretch > 0:
        image = _stretch(image, params.stretch, rng)
    if params.water_tint > 0:
        image = _water_tint(image, params.water_tint, rng)
    if params.lowres_prob > 0 and rng.random() < params.lowres_prob:
        image = _lowres(np.ascontiguousarray(image, dtype=np.float32), params.lowres_max, rng)
    if params.degrade_prob > 0 and rng.random() < params.degrade_prob:
        image = _degrade(np.clip(image, 0.0, 1.0), rng)
    return np.ascontiguousarray(np.clip(image, 0.0, 1.0), dtype=np.float32), np.ascontiguousarray(mask)
