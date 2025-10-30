"""Wavelet transform utilities for the WaveFormer architecture."""

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


@dataclass
class WaveletTransformConfig:
    """Configuration for the wavelet transform.

    Attributes
    ----------
    level:
        The decomposition level to use when computing the discrete wavelet
        transform. Larger values capture coarser features.
    mode:
        The signal extension mode used by :mod:`pywt` when handling the
        boundaries of the input sequence.
    """

    level: int = 1
    mode: str = "symmetric"


class WaveletTransform:
    """Apply discrete wavelet transforms to 1-D sequences.

    Parameters
    ----------
    kernels:
        An iterable of wavelet kernel names that are supported by
        :func:`pywt.wavedec`.
    config:
        Optional :class:`WaveletTransformConfig` that governs the
        decomposition level and padding behaviour.
    """

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
        """Apply the wavelet transform with the specified kernel.

        Parameters
        ----------
        signal:
            A one-dimensional numpy array representing the raw input sequence.
        kernel:
            The name of the wavelet kernel to use. The value must be included in
            the kernel list provided during initialisation.

        Returns
        -------
        np.ndarray
            The concatenated wavelet coefficients across all decomposition
            levels. The resulting feature vector is suitable as input to the
            Transformer encoder.
        """

        if kernel not in self.kernels:
            raise ValueError(f"Kernel '{kernel}' was not registered with the transform.")

        coeffs = pywt.wavedec(
            signal,
            kernel,
            mode=self.config.mode,
            level=self.config.level,
        )
        # Flatten the coefficient tree into a feature vector.
        features = np.concatenate([c.ravel() for c in coeffs])
        return features.astype(np.float32)

