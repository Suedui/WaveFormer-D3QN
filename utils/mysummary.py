"""Convenience wrapper around :func:`torchinfo.summary` if available."""

from __future__ import annotations

from typing import Any, Iterable

try:  # pragma: no cover - optional dependency
    from torchinfo import summary as torchinfo_summary
except Exception:  # pragma: no cover - optional dependency
    torchinfo_summary = None


def model_summary(model: Any, input_size: Iterable[int]) -> str:
    """Return a string summary of ``model`` for the given ``input_size``."""

    if torchinfo_summary is None:
        return "torchinfo is not installed; install it to view model summaries."

    return str(torchinfo_summary(model, input_size=input_size))
