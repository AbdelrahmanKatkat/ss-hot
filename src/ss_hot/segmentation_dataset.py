"""Tiles + labels as torch datasets (needs the `train` extra; import explicitly).

Both datasets scale every tile as at serving time (`dataset.scale_pair`; a tile at the training zoom is not
resampled). `TrainTiles` gives one `window` x `window` sample per tile and epoch, augmented, with a second random tile
as the partner for CutMix and histogram matching. `EvalTiles` gives each tile cut into windows exactly as at serving
(`preprocess.to_windows`) and its full mask, so predictions are stitched and scored like served ones (use
`batch_size=1`).
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .augment import augment_chip_and_mask
from .dataset import LabelSource, fit_tile, list_chip_paths, load_label_source, scale_pair
from .params import WINDOW_PX, AugmentParams, PreprocessParams
from .preprocess import to_windows


class _TileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels: str | Path | LabelSource,
        pre: PreprocessParams,
        *,
        names: Sequence[str] | None = None,
        window: int = WINDOW_PX,
    ) -> None:
        self.paths = list_chip_paths(images_dir, names)
        if not self.paths:
            raise ValueError(f"No tiles found in {images_dir}")
        self.labels = load_label_source(labels)
        self.pre, self.window = pre, window

    def __len__(self) -> int:
        return len(self.paths)

    def _pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return scale_pair(self.paths[idx], self.labels, self.pre)


class TrainTiles(_TileDataset):
    """One `window` x `window` training sample per tile (a random crop of a larger scaled tile), augmented."""

    def __init__(self, *args, augment: AugmentParams | None = None, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the random crops and augmentation from epoch to epoch while staying reproducible."""
        self.epoch = epoch

    def _needs_partner(self) -> bool:
        return self.augment is not None and (self.augment.cutmix_prob > 0 or self.augment.histogram_prob > 0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        image, mask = fit_tile(*self._pair(idx), self.window, rng)
        if self.augment is not None:
            partner = None
            if self._needs_partner() and len(self) > 1:
                other = int(rng.integers(0, len(self) - 1))
                partner = fit_tile(*self._pair(other + (other >= idx)), self.window, rng)
            image, mask = augment_chip_and_mask(image, mask, rng, self.augment, partner)
        return torch.from_numpy(image), torch.from_numpy(mask)


class EvalTiles(_TileDataset):
    """One tile as served: `(K, 3, w, w)` windows and its `(H, W)` mask at the model resolution."""

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._pair(idx)
        windows, _, _ = to_windows(image, self.window)
        return torch.from_numpy(windows), torch.from_numpy(mask)
