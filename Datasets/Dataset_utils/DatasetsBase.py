"""Base dataset definitions shared across TFN1 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

Transform = Optional[Callable[[torch.Tensor], torch.Tensor]]


@dataclass(slots=True)
class DiskDatasetConfig:
    """Configuration describing how a dataset is organised on disk."""

    root: Path
    signal_file: str = "signals.npy"
    target_file: str = "targets.npy"


class TimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Generic time-series dataset backed by NumPy files.

    Parameters
    ----------
    config:
        Description of the file layout on disk.
    transform:
        Optional callable applied to each signal.
    target_transform:
        Optional callable applied to each target.
    augmentations:
        Sequence of callables that modify the signal in-place or return a new
        tensor. The augmentations are applied in the order provided.
    """

    def __init__(
        self,
        config: DiskDatasetConfig,
        transform: Transform = None,
        target_transform: Transform = None,
        augmentations: Optional[Sequence[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self.target_transform = target_transform
        self.augmentations = augmentations or []

        signal_path = config.root / config.signal_file
        target_path = config.root / config.target_file

        if not signal_path.exists() or not target_path.exists():
            raise FileNotFoundError(
                "Dataset files were not found. Expected NumPy files named "
                f"'{config.signal_file}' and '{config.target_file}' in {config.root}."
            )

        self.signals = torch.from_numpy(np.load(signal_path)).float()
        self.targets = torch.from_numpy(np.load(target_path)).float()

        if self.signals.ndim != 2:
            raise ValueError("Signals array must have shape (num_samples, sequence_length).")

        if self.targets.ndim == 1:
            self.targets = self.targets.unsqueeze(-1)

        if len(self.signals) != len(self.targets):
            raise ValueError("Signals and targets must contain the same number of samples.")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        signal = self.signals[idx]
        target = self.targets[idx]

        for augmentation in self.augmentations:
            signal = augmentation(signal)

        if self.transform:
            signal = self.transform(signal)
        if self.target_transform:
            target = self.target_transform(target)

        return signal, target
