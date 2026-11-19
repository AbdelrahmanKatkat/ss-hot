import numpy as np
import pytest
from PIL import Image

from ss_hot.dataset import (
    IGNORE,
    SEAGRASS,
    LabelSource,
    assemble_block,
    chips_per_side,
    dataset_blocks,
    fit_window,
    grid_windows,
    plan_blocks,
    read_mask,
    scale_mask,
    scale_pair,
    spatial_split,
)
from ss_hot.params import PreprocessParams


def test_masks_only_take_background_seagrass_and_ignore(tmp_path):
    path = tmp_path / "m.png"
    Image.fromarray(np.array([[0, 1, 255]], dtype=np.uint8)).save(path)
    assert read_mask(path).tolist() == [[0, 1, 255]]
    Image.fromarray(np.array([[0, 128]], dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="0, 1 or 255"):
        read_mask(path)


def test_a_downscaled_mask_keeps_the_majority_label():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:, :3] = SEAGRASS  # three quarters seagrass
    mask[0, 0] = IGNORE
    small = scale_mask(mask, 2, 2)
    assert small.tolist() == [[SEAGRASS, BACKGROUND_OR(small[0, 1])], [SEAGRASS, small[1, 1]]]
    assert small[:, 0].tolist() == [SEAGRASS, SEAGRASS]
    full_ignore = np.full((4, 4), IGNORE, dtype=np.uint8)
    assert (scale_mask(full_ignore, 2, 2) == IGNORE).all()


def BACKGROUND_OR(value):  # the right column is half seagrass: either label is a valid majority
    assert value in (0, SEAGRASS)
    return value


def test_label_source_reads_a_mask_folder_and_a_geojson(mask_dataset, geojson_dataset):
    images, masks, names = mask_dataset
    assert LabelSource.from_path(masks).masks_dir == masks
    chips, labels = geojson_dataset
    source = LabelSource.from_path(labels)
    assert source.masks_dir is None and len(source.polygons) == 1
    mask = source.mask(next(chips.iterdir()), 256, 256)
    assert mask[:128, :128].mean() == SEAGRASS and mask[130:, 130:].max() == 0
    half = source.mask(next(chips.iterdir()), 64, 64)  # drawn at the scaled size, so edges stay sharp
    assert half[:32, :32].min() == SEAGRASS and half[33:, 33:].max() == 0


def test_a_chip_without_a_mask_is_ignored(mask_dataset, tmp_path):
    images, masks, names = mask_dataset
    (masks / f"{names[0][:-4]}.png").unlink()
    assert (LabelSource(masks_dir=masks).mask(images / names[0], 8, 8) == IGNORE).all()


def test_scale_pair_uses_the_serving_scale_and_an_explicit_resolution(mask_dataset):
    images, masks, names = mask_dataset
    source = LabelSource(masks_dir=masks)
    image, mask = scale_pair(images / names[0], source, PreprocessParams())
    assert image.shape == (3, 318, 318) and mask.shape == (318, 318)  # 256 px * 3.73 cm / 3 cm
    image, mask = scale_pair(images / names[0], source, PreprocessParams(), source_gsd_m=0.015)
    assert image.shape == (3, 128, 128) and (mask == SEAGRASS).all()


def test_neighbouring_chips_form_blocks_and_gaps_are_ignored(mask_dataset):
    images, masks, names = mask_dataset
    assert chips_per_side(128) == 4 and chips_per_side(600) == 1
    blocks = plan_blocks(names, 2)
    assert len(blocks) == 1 and sorted((col, row) for _, col, row in blocks[0]) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert len(plan_blocks(names, 1)) == 4
    pre = PreprocessParams()
    block = [entry for entry in blocks[0] if entry[1:] != (1, 1)]  # drop the bottom-right chip
    image, mask = assemble_block(block, images, LabelSource(masks_dir=masks), pre, source_gsd_m=0.015)
    assert image.shape == (3, 256, 256)
    assert (
        (mask[:, :128] == SEAGRASS).all()
        and (mask[:128, 128:] == 0).all()
        and (mask[128:, 128:] == IGNORE).all()
    )
    assert len(dataset_blocks(images, pre, source_gsd_m=0.015)) == 1  # 128 px chips: four per 512 px side


def test_windows_are_padded_with_ignore_and_cropped_at_random():
    image = np.ones((3, 300, 700), dtype=np.float32)
    mask = np.ones((300, 700), dtype=np.uint8)
    fitted_image, fitted_mask = fit_window(image, mask, 512, np.random.default_rng(0))
    assert fitted_image.shape == (3, 512, 512) and (fitted_mask[300:] == IGNORE).all()
    assert (fitted_mask[:300] == 1).all()
    pieces = grid_windows(image, mask, 512)
    assert len(pieces) == 2 and (pieces[1][1][:, 188:] == IGNORE).all()


def test_spatial_split_keeps_blocks_together():
    names = [f"OAM-{x}-{y}-22.tif" for x in range(8) for y in range(8)]
    train, val = spatial_split(names, 0.25, seed=1, block_size=4)
    assert len(val) == 16 and not set(train) & set(val)
    blocks = {(int(n.split("-")[1]) // 4, int(n.split("-")[2]) // 4) for n in val}
    assert len(blocks) == 1
