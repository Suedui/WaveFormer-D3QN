"""Dataset utilities for WaveFormer-D3QN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    """Configuration describing how the dataset is organised on disk."""

    root: Path
    signal_file: str = "signals.npy"
    target_file: str = "targets.npy"


class TimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A thin wrapper around NumPy arrays stored on disk.

    The dataset expects two NumPy files located at ``root / signal_file`` and
    ``root / target_file``. Each row in ``signals`` represents a sample in the
    time domain. ``targets`` should either be a 1-D vector (for regression) or
    a 2-D array with one-hot encoded labels.
    """

    def __init__(
        self,
        config: DatasetConfig,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self.target_transform = target_transform

        if not config.root.exists():
            raise FileNotFoundError(
                "Dataset root does not exist. Please place your dataset at "
                f"'{config.root}' before running training."
            )

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

        if self.transform:
            signal = self.transform(signal)
        if self.target_transform:
            target = self.target_transform(target)

        return signal, target

