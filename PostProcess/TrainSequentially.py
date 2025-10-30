"""Utility for training multiple checkpoints sequentially."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import torch

from utils.train_utils import save_checkpoint


class Trainable(Protocol):
    def train_epoch(self) -> float:
        ...

    def state_dict(self) -> dict:
        ...


def train_multiple(
    trainer: Trainable,
    checkpoints: Iterable[Path],
    epochs_per_checkpoint: int,
) -> list[float]:
    """Iteratively train ``trainer`` and persist checkpoints."""

    losses: list[float] = []
    for checkpoint_path in checkpoints:
        epoch_losses = [trainer.train_epoch() for _ in range(epochs_per_checkpoint)]
        losses.extend(epoch_losses)
        save_checkpoint(trainer, checkpoint_path)
    return losses
