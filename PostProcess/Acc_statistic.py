"""Utilities for aggregating accuracy statistics after training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def summarise_accuracies(records: Iterable[float]) -> dict[str, float]:
    """Return mean and standard deviation of the provided accuracy values."""

    values = np.array(list(records), dtype=np.float32)
    if values.size == 0:
        raise ValueError("At least one accuracy value is required.")

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_statistics(stats: dict[str, float], path: Path) -> None:
    """Persist the statistics dictionary as a NumPy npz archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)
