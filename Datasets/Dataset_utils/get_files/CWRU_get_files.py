"""Helpers specific to the CWRU bearing dataset layout."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .generalfunction import ensure_exists


def load_cwru_numpy(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``signals.npy`` and ``targets.npy`` from the provided ``root``."""

    root = ensure_exists(root)
    signal_path = ensure_exists(root / "signals.npy")
    target_path = ensure_exists(root / "targets.npy")

    signals = np.load(signal_path)
    targets = np.load(target_path)
    return signals, targets
