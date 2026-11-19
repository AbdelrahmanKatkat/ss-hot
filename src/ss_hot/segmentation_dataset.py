"""Chips + labels as torch datasets of fixed-size windows (needs the `train` extra; import explicitly).

Both datasets scale every chip as at serving time and join neighbouring chips into mosaic blocks
(`dataset.dataset_blocks`). `TrainWindows` gives one random window per block per epoch, augmented; `EvalWindows`
gives every window of a block, unchanged, stacked (use `batch_size=1`).
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .augment import augment_chip_and_mask
from .dataset import LabelSource, assemble_block, dataset_blocks, fit_window, grid_windows, load_label_source
from .params import WINDOW_PX, AugmentParams, PreprocessParams


class _BlockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels: str | Path | LabelSource,
        pre: PreprocessParams,
        *,
        names: Sequence[str] | None = None,
        source_gsd_m: float | None = None,
        window: int = WINDOW_PX,
    ) -> None:
        self.images_dir, self.pre, self.window, self.source_gsd_m = (
            Path(images_dir),
            pre,
            window,
            source_gsd_m,
        )
        self.labels = load_label_source(labels)
        self.blocks = dataset_blocks(images_dir, pre, names=names, source_gsd_m=source_gsd_m, window=window)

    def __len__(self) -> int:
        return len(self.blocks)

    def _block(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return assemble_block(
            self.blocks[idx], self.images_dir, self.labels, self.pre, source_gsd_m=self.source_gsd_m
        )

    @property
    def chip_count(self) -> int:
        return sum(len(block) for block in self.blocks)


class TrainWindows(_BlockDataset):
    """One `window` x `window` training sample per block: a random crop (or padding) of the mosaic, augmented."""

    def __init__(self, *args, augment: AugmentParams | None = None, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the random crops and augmentation from epoch to epoch while staying reproducible."""
        self.epoch = epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        image, mask = fit_window(*self._block(idx), self.window, rng)
        if self.augment is not None:
            image, mask = augment_chip_and_mask(image, mask, rng, self.augment)
        return torch.from_numpy(image), torch.from_numpy(mask)


class EvalWindows(_BlockDataset):
    """Every `window` x `window` piece of one block, stacked: `(K, 3, w, w)` images and `(K, w, w)` masks."""

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pieces = grid_windows(*self._block(idx), self.window)
        images = np.stack([image for image, _ in pieces])
        masks = np.stack([mask for _, mask in pieces])
        return torch.from_numpy(images), torch.from_numpy(masks)
