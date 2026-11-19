import numpy as np
import pytest

from ss_hot.augment import augment_chip_and_mask
from ss_hot.dataset import IGNORE
from ss_hot.params import AugmentParams


def _pair(size: int = 64):
    rng = np.random.default_rng(0)
    mask = (rng.random((size, size)) > 0.5).astype(np.uint8)
    image = np.stack(
        [
            mask.astype(np.float32),
            rng.random((size, size), dtype=np.float32),
            np.zeros((size, size), np.float32),
        ]
    )
    return image, mask


def test_defaults_change_nothing():
    image, mask = _pair()
    out_image, out_mask = augment_chip_and_mask(image, mask, np.random.default_rng(1))
    assert np.array_equal(out_image, image) and np.array_equal(out_mask, mask)


@pytest.mark.parametrize("orientation", ["dihedral", "mirror"])
def test_the_mask_moves_with_the_image(orientation):
    image, mask = _pair()  # channel 0 is the mask itself
    for seed in range(8):
        out_image, out_mask = augment_chip_and_mask(
            image, mask, np.random.default_rng(seed), AugmentParams(orientation=orientation)
        )
        assert np.array_equal(out_image[0], out_mask.astype(np.float32))


def test_scale_jitter_pads_the_mask_with_ignore_and_keeps_the_size():
    image, mask = _pair()
    params = AugmentParams(scale_jitter=0.5)
    seen_ignore = False
    for seed in range(10):
        out_image, out_mask = augment_chip_and_mask(image, mask, np.random.default_rng(seed), params)
        assert out_image.shape == image.shape and out_mask.shape == mask.shape
        assert set(np.unique(out_mask).tolist()) <= {0, 1, IGNORE}
        seen_ignore |= bool((out_mask == IGNORE).any())
    assert seen_ignore


def test_colour_steps_touch_only_the_image_and_stay_in_range():
    image, mask = _pair()
    params = AugmentParams(photometric=0.3, stretch=0.7, water_tint=0.5, lowres_prob=1.0, degrade_prob=1.0)
    for seed in range(6):
        out_image, out_mask = augment_chip_and_mask(image, mask, np.random.default_rng(seed), params)
        assert np.array_equal(out_mask, mask) and out_image.dtype == np.float32
        assert 0.0 <= out_image.min() and out_image.max() <= 1.0
        assert not np.array_equal(out_image, image)


def test_the_same_seed_gives_the_same_result():
    image, mask = _pair()
    params = AugmentParams(
        orientation="dihedral",
        photometric=0.2,
        stretch=0.5,
        scale_jitter=0.2,
        water_tint=0.3,
        degrade_prob=0.5,
        lowres_prob=0.5,
    )
    first = augment_chip_and_mask(image, mask, np.random.default_rng(42), params)
    second = augment_chip_and_mask(image, mask, np.random.default_rng(42), params)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_low_resolution_removes_fine_detail_and_tint_fades_red_most():
    image = np.random.default_rng(3).random((3, 64, 64), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    blurred, _ = augment_chip_and_mask(
        image, mask, np.random.default_rng(0), AugmentParams(lowres_prob=1.0, lowres_max=4.0)
    )
    assert np.abs(np.diff(blurred, axis=2)).mean() < np.abs(np.diff(image, axis=2)).mean()
    white = np.ones((3, 8, 8), dtype=np.float32)
    tinted, _ = augment_chip_and_mask(
        white, np.zeros((8, 8), np.uint8), np.random.default_rng(5), AugmentParams(water_tint=0.9)
    )
    assert tinted[0].mean() < tinted[2].mean()  # red loses the most
