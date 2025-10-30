"""Simple time-series augmentation helpers."""

from __future__ import annotations

from typing import Callable

import torch


def add_gaussian_noise(std: float = 0.01) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an augmentation that adds zero-mean Gaussian noise."""

    def _augment(signal: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(signal) * std
        return signal + noise

    return _augment


def time_flip(signal: torch.Tensor) -> torch.Tensor:
    """Reverse the temporal order of the signal."""

    return signal.flip(0)


def compose(*funcs: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compose multiple augmentations into a single callable."""

    def _augment(signal: torch.Tensor) -> torch.Tensor:
        for func in funcs:
            signal = func(signal)
        return signal

    return _augment
