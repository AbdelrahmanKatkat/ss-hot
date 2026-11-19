import json

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer

from conftest import CHIP_NAME, COARSE_NAME
from ss_hot.dataset import (
    BACKGROUND,
    SEAGRASS,
    LabelSource,
    fit_tile,
    read_mask,
    scale_mask,
    scale_pair,
    spatial_split,
)
from ss_hot.params import MODEL_GSD_M, PreprocessParams
from ss_hot.tiles import chip_scale, tile_transform


def test_masks_only_take_background_and_seagrass(tmp_path):
    path = tmp_path / "m.png"
    Image.fromarray(np.array([[0, 1, 1]], dtype=np.uint8)).save(path)
    assert read_mask(path).tolist() == [[0, 1, 1]]
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="0 or 1"):
        read_mask(path)


def test_a_downscaled_mask_keeps_the_majority_label():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:, :3] = SEAGRASS  # the left column of the result is all seagrass, the right one half
    small = scale_mask(mask, 2, 2)
    assert small[:, 0].tolist() == [SEAGRASS, SEAGRASS] and set(small[:, 1].tolist()) <= {
        BACKGROUND,
        SEAGRASS,
    }
    assert scale_mask(mask, 4, 4).tolist() == mask.tolist()


def test_label_source_reads_a_mask_folder_and_a_geojson(mask_dataset, geojson_dataset):
    images, masks, names = mask_dataset
    assert LabelSource.from_path(masks).masks_dir == masks
    chips, labels = geojson_dataset
    source = LabelSource.from_path(labels)
    assert source.masks_dir is None and len(source.polygons) == 1
    mask = source.mask(next(chips.iterdir()), 256, 256)
    assert mask[:128, :128].min() == SEAGRASS and mask[130:, 130:].max() == 0
    half = source.mask(next(chips.iterdir()), 64, 64)  # drawn at the scaled size, so edges stay sharp
    assert half[:32, :32].min() == SEAGRASS and half[33:, 33:].max() == 0


def test_geojson_labels_on_a_plain_tile_are_placed_by_its_name(write_chip, tmp_path):
    chip = write_chip(name=CHIP_NAME, folder=tmp_path / "chips", crs=None)
    t = tile_transform(CHIP_NAME, 256, 256)
    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    corners = [(0, 0), (128, 0), (128, 128), (0, 128), (0, 0)]
    square = [list(to_lonlat(t.c + col * t.a, t.f + row * t.e)) for col, row in corners]
    labels = tmp_path / "labels.geojson"
    feature = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [square]}}
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    mask = LabelSource.from_path(labels).mask(chip, 256, 256)
    assert mask[2:126, 2:126].min() == SEAGRASS and mask[130:, :].max() == 0 and mask[:, 130:].max() == 0


def test_a_tile_without_a_mask_raises(mask_dataset):
    images, masks, names = mask_dataset
    (masks / f"{names[0][:-4]}.png").unlink()
    with pytest.raises(FileNotFoundError):
        LabelSource(masks_dir=masks).mask(images / names[0], 8, 8)


def test_scale_pair_keeps_a_training_tile_and_upscales_a_coarse_one(mask_dataset, write_chip, tmp_path):
    images, masks, names = mask_dataset
    source = LabelSource(masks_dir=masks)
    image, mask = scale_pair(images / names[0], source, PreprocessParams())
    assert image.shape == (3, 256, 256) and (mask == SEAGRASS).all()
    coarse = write_chip(name=COARSE_NAME, folder=tmp_path / "coarse", crs=None)
    Image.fromarray(np.ones((256, 256), dtype=np.uint8)).save(masks / f"{COARSE_NAME[:-4]}.png")
    image, mask = scale_pair(coarse, source, PreprocessParams())
    side = round(256 * chip_scale(COARSE_NAME, 256, MODEL_GSD_M))
    assert image.shape == (3, side, side) and mask.shape == (side, side) and (mask == SEAGRASS).all()


def test_fit_tile_crops_a_larger_tile_and_reflects_a_smaller_one_with_its_labels():
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
