"""Wavelet transform utilities for the WaveFormer-D3QN architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

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
    """Configuration for the continuous wavelet transform."""

    scale_bands: Sequence[Tuple[float, float]] = (
        (1.0, 32.0),
        (1.0, 64.0),
        (1.0, 128.0),
    )
    num_scales: int = 64
    sampling_period: float = 1.0
    normalise: bool = True


class WaveletTransform:
    """Apply continuous wavelet transforms to 1-D sequences."""

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

        if not self.config.scale_bands:
            raise ValueError("At least one scale band must be configured for the transform.")

    def available_scale_bands(self) -> range:
        """Return the indices of the configured scale bands."""

        return range(len(self.config.scale_bands))

    def _build_scales(self, band_index: int) -> np.ndarray:
        try:
            scale_min, scale_max = self.config.scale_bands[band_index]
        except IndexError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Scale band index {band_index} is out of range") from exc

        if scale_min <= 0 or scale_max <= 0:
            raise ValueError("Scale bounds must be positive numbers.")
        if scale_min >= scale_max:
            raise ValueError("Scale band lower bound must be smaller than upper bound.")

        return np.linspace(
            float(scale_min),
            float(scale_max),
            self.config.num_scales,
            dtype=np.float32,
        )

    def apply(
        self, signal: np.ndarray, kernel: str, band_index: int
    ) -> np.ndarray:
        """Apply the continuous wavelet transform with the specified kernel."""

        if kernel not in self.kernels:
            raise ValueError(f"Kernel '{kernel}' was not registered with the transform.")

        scales = self._build_scales(band_index)
        coefficients, _ = pywt.cwt(
            signal,
            scales,
            kernel,
            sampling_period=self.config.sampling_period,
        )
        scalogram = np.abs(coefficients).astype(np.float32)

        if self.config.normalise:
            mean = scalogram.mean()
            std = scalogram.std()
            scalogram = (scalogram - mean) / (std + 1e-6)

        # ``pywt.cwt`` returns an array of shape ``(num_scales, time_steps)``.
        # We transpose it so that the transformer sees ``time_steps`` tokens with
        # ``num_scales`` features, matching the sequence-first layout.
        return scalogram.T
