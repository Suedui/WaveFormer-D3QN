"""Shared post-processing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def rolling_window(values: Iterable[float], window: int) -> np.ndarray:
    """Compute a rolling mean with the specified ``window`` size."""

    array = np.array(list(values), dtype=np.float32)
    if window <= 0:
        raise ValueError("window must be positive")
    if window > array.size:
        raise ValueError("window is larger than the number of values")

    shape = (array.size - window + 1, window)
    strides = (array.strides[0], array.strides[0])
    sliding = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return sliding.mean(axis=1)


def save_curve(curve: np.ndarray, path: Path) -> None:
    """Persist a curve to disk as ``.npy``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, curve)
