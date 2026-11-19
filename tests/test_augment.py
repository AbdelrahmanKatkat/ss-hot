import numpy as np

from ss_hot.augment import augment_chip_and_mask, match_histograms
from ss_hot.params import AugmentParams


def _pair(size: int = 64, seed: int = 0):
    """Channel 0 of the image is the mask itself, so a test can see that both moved together."""
    rng = np.random.default_rng(seed)
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
    out_image, out_mask = augment_chip_and_mask(image, mask, np.random.default_rng(1), partner=_pair(seed=1))
    assert np.array_equal(out_image, image) and np.array_equal(out_mask, mask)


def test_the_mask_moves_with_the_image():
    image, mask = _pair()
    params = AugmentParams(orientation="dihedral", cutmix_prob=1.0)
    for seed in range(8):
        out_image, out_mask = augment_chip_and_mask(
            image, mask, np.random.default_rng(seed), params, _pair(seed=9)
        )
        assert np.array_equal(out_image[0], out_mask.astype(np.float32))


def test_scale_jitter_zooms_in_and_keeps_the_size_and_labels():
    image = np.zeros((3, 64, 64), np.float32)
    image[0, :, :32] = 1.0
    mask = (image[0] > 0).astype(np.uint8)
    for seed in range(6):
        out_image, out_mask = augment_chip_and_mask(
            image, mask, np.random.default_rng(seed), AugmentParams(scale_jitter=1.0)
        )
        assert out_image.shape == image.shape and set(np.unique(out_mask).tolist()) <= {0, 1}
        assert np.abs((out_image[0] > 0.5) != out_mask).mean() < 0.05  # the edge may move by a pixel


def test_cutmix_pastes_a_box_of_the_partner():
    image, mask = np.zeros((3, 64, 64), np.float32), np.zeros((64, 64), np.uint8)
    partner = (np.ones((3, 64, 64), np.float32), np.ones((64, 64), np.uint8))
    pasted = [
        augment_chip_and_mask(
            image, mask, np.random.default_rng(seed), AugmentParams(cutmix_prob=1.0), partner
        )
        for seed in range(10)
    ]
    assert all(np.array_equal(out_image[0], out_mask) for out_image, out_mask in pasted)
    assert any(0 < out_mask.mean() < 1 for _, out_mask in pasted)
    assert np.array_equal(
        augment_chip_and_mask(image, mask, np.random.default_rng(0), AugmentParams(cutmix_prob=1.0))[1], mask
    )


def test_histogram_matching_takes_the_reference_distribution():
    rng = np.random.default_rng(2)
    image = rng.random((3, 32, 32), dtype=np.float32)
    reference = (0.2 + 0.3 * rng.random((3, 32, 32))).astype(np.float32)
    matched = match_histograms(image, reference)
    assert np.allclose(np.sort(matched.ravel()), np.sort(reference.ravel()), atol=1e-6)
    assert np.array_equal(np.argsort(matched[0].ravel()), np.argsort(image[0].ravel()))  # the order is kept


def test_colour_steps_touch_only_the_image_and_stay_in_range():
    image, mask = _pair()
    params = AugmentParams(photometric=0.3, histogram_prob=1.0, lowres_prob=1.0, degrade_prob=1.0)
    for seed in range(6):
        out_image, out_mask = augment_chip_and_mask(
            image, mask, np.random.default_rng(seed), params, _pair(seed=5)
        )
        assert np.array_equal(out_mask, mask) and out_image.dtype == np.float32
        assert 0.0 <= out_image.min() and out_image.max() <= 1.0
        assert not np.array_equal(out_image, image)


def test_the_same_seed_gives_the_same_result():
    image, mask = _pair()
    params = AugmentParams(
        orientation="dihedral", photometric=0.2, scale_jitter=0.2, histogram_prob=0.5, cutmix_prob=0.5,
        lowres_prob=0.5, degrade_prob=0.5,
    )  # fmt: skip
    first = augment_chip_and_mask(image, mask, np.random.default_rng(42), params, _pair(seed=3))
    second = augment_chip_and_mask(image, mask, np.random.default_rng(42), params, _pair(seed=3))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_low_resolution_removes_fine_detail():
    image = np.random.default_rng(3).random((3, 64, 64), dtype=np.float32)
    blurred, _ = augment_chip_and_mask(
        image,
        np.zeros((64, 64), np.uint8),
        np.random.default_rng(0),
        AugmentParams(lowres_prob=1.0, lowres_max=4.0),
    )
    assert np.abs(np.diff(blurred, axis=2)).mean() < np.abs(np.diff(image, axis=2)).mean()
