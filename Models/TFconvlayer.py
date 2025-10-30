"""Wavelet transform utilities for the TFN1 architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

try:  # pragma: no cover - optional dependency
    import pywt
except Exception as exc:  # pragma: no cover - informative error message
    pywt = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - executed when pywt is present
    _IMPORT_ERROR = None


@dataclass(slots=True)
class WaveletTransformConfig:
    """Configuration for the wavelet transform."""

    level: int = 1
    mode: str = "symmetric"


class WaveletTransform:
    """Apply discrete wavelet transforms to 1-D sequences."""

    def __init__(
        self, kernels: Iterable[str], config: WaveletTransformConfig | None = None
    ) -> None:
        if pywt is None:  # pragma: no cover - executed only when pywt missing
            raise ImportError(
                "pywt is required for WaveletTransform but could not be imported"
            ) from _IMPORT_ERROR

        self.kernels: List[str] = list(kernels)
        if not self.kernels:
            raise ValueError("At least one wavelet kernel must be provided.")

        self.config = config or WaveletTransformConfig()

    def apply(self, signal: np.ndarray, kernel: str) -> np.ndarray:
        """Apply the wavelet transform with the specified kernel."""

        if kernel not in self.kernels:
            raise ValueError(f"Kernel '{kernel}' was not registered with the transform.")

        coeffs = pywt.wavedec(
            signal,
            kernel,
            mode=self.config.mode,
            level=self.config.level,
        )
        features = np.concatenate([c.ravel() for c in coeffs])
        return features.astype(np.float32)
