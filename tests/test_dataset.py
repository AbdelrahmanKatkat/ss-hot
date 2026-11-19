import numpy as np
import pytest

from conftest import COARSE_NAME, write_labels
from ss_hot.dataset import BACKGROUND, SEAGRASS, LabelSource, fit_tile, scale_pair, spatial_split
from ss_hot.params import MODEL_GSD_M, PreprocessParams
from ss_hot.tiles import chip_scale


def test_labels_are_drawn_on_each_chip_from_one_geojson(fair_dataset):
    chips, labels, names = fair_dataset
    source = LabelSource.from_path(labels.parent)  # a folder holding one GeoJSON works too
    assert len(source.polygons) == 1
    left, right = chips / names[0], chips / names[-1]
    assert (source.mask(left, 256, 256) == SEAGRASS).all() and (
        source.mask(right, 256, 256) == BACKGROUND
    ).all()


def test_masks_are_drawn_at_the_scaled_size(write_chip, tmp_path):
    chip = write_chip(folder=tmp_path / "chips")
    labels = tmp_path / "labels.geojson"
    write_labels(labels, [(0, 0, 128, 128)])
    source = LabelSource.from_path(labels)
    mask = source.mask(chip, 256, 256)
    assert mask[:128, :128].min() == SEAGRASS and mask[130:, :].max() == 0 and mask[:, 130:].max() == 0
    half = source.mask(chip, 64, 64)  # drawn at the scaled size, so edges stay sharp
    assert half[:32, :32].min() == SEAGRASS and half[33:, 33:].max() == 0


def test_a_chip_without_georeferencing_is_refused(write_chip, tmp_path):
    labels = tmp_path / "labels.geojson"
    write_labels(labels, [(0, 0, 10, 10)])
    with pytest.raises(ValueError, match="no CRS"):
        LabelSource.from_path(labels).mask(write_chip(folder=tmp_path / "plain", crs=None), 8, 8)


def test_scale_pair_keeps_a_training_chip_and_upscales_a_coarse_one(fair_dataset, write_chip, tmp_path):
    chips, labels, names = fair_dataset
    source = LabelSource.from_path(labels)
    image, mask = scale_pair(chips / names[0], source, PreprocessParams())
    assert image.shape == (3, 256, 256) and (mask == SEAGRASS).all()
    coarse = write_chip(name=COARSE_NAME, folder=tmp_path / "coarse")  # over the labelled left column
    image, mask = scale_pair(coarse, source, PreprocessParams())
    side = round(256 * chip_scale(COARSE_NAME, 256, MODEL_GSD_M))
    assert image.shape == (3, side, side) and mask.shape == (side, side) and (mask == SEAGRASS).all()


def test_fit_tile_crops_a_larger_chip_and_reflects_a_smaller_one_with_its_labels():
    image = np.random.default_rng(0).random((3, 300, 700), dtype=np.float32)
    mask = (np.arange(700)[None, :] < 350).repeat(300, axis=0).astype(np.uint8)
    cropped_image, cropped_mask = fit_tile(image, mask, 256, np.random.default_rng(0))
    assert cropped_image.shape == (3, 256, 256) and cropped_mask.shape == (256, 256)
    small_image, small_mask = image[:, :200, :200], mask[:200, :200].copy()
    small_mask[:, 150:] = BACKGROUND
    padded_image, padded_mask = fit_tile(small_image, small_mask, 256)
    assert np.array_equal(padded_mask[:, 200:], small_mask[:, :143:-1])  # mirrored with the image
    assert np.array_equal(padded_image[0, :200, 200:], small_image[0, :, :143:-1])
    assert np.array_equal(fit_tile(image[:, :256, :256], mask[:256, :256], 256)[1], mask[:256, :256])


def test_spatial_split_keeps_blocks_together():
    names = [f"OAM-{x}-{y}-22.tif" for x in range(8) for y in range(8)]
    train, val = spatial_split(names, 0.25, seed=1, block_size=4)
    assert len(val) == 16 and not set(train) & set(val)
    blocks = {(int(n.split("-")[1]) // 4, int(n.split("-")[2]) // 4) for n in val}
    assert len(blocks) == 1
