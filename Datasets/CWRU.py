"""CWRU dataset adapter built on top of the shared dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from .Dataset_utils.DatasetsBase import DiskDatasetConfig, TimeSeriesDataset
from .Dataset_utils.sequence_aug import add_gaussian_noise
from .Dataset_utils.get_files.CWRU_get_files import load_cwru_numpy


@dataclass(slots=True)
class CWRUConfig:
    """Configuration for constructing a :class:`CWRUDataset`."""

    root: Path
    augment: bool = False
    noise_std: float = 0.01


class CWRUDataset(TimeSeriesDataset):
    """Dataset wrapper around the NumPy export of the CWRU benchmark."""

    def __init__(
        self,
        config: CWRUConfig,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        extra_augmentations: Optional[Sequence[callable]] = None,
    ) -> None:
        root = config.root
        signals, targets = load_cwru_numpy(root)

        augmentations = self._build_augmentations(config, extra_augmentations)

        if root.is_file():
            # When a single ``.mat`` file is provided ``TimeSeriesDataset`` cannot
            # locate the cached ``signals.npy``/``targets.npy`` pair.  We handle
            # this case manually while retaining the same attributes that the
            # base class would normally populate.
            self.config = DiskDatasetConfig(root=root.parent)
            self.transform = transform
            self.target_transform = target_transform
            self.augmentations = augmentations
        else:
            disk_config = DiskDatasetConfig(root=root)
            super().__init__(
                disk_config,
                transform=transform,
                target_transform=target_transform,
                augmentations=augmentations,
            )

        # Replace the data loaded by the base class with the more efficient
        # tensors created from ``load_cwru_numpy`` to avoid double disk access.
        self.signals = torch.from_numpy(signals).float()
        self.targets = torch.from_numpy(targets).float()

        if self.targets.ndim == 1:
            self.targets = self.targets.unsqueeze(-1)

    @staticmethod
    def _build_augmentations(
        config: CWRUConfig, extra: Optional[Sequence[callable]]
    ) -> Sequence[callable]:
        augmentations: list[callable] = []
        if config.augment:
            augmentations.append(add_gaussian_noise(config.noise_std))
        if extra:
            augmentations.extend(extra)
        return augmentations
