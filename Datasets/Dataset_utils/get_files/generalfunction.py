"""Utilities for discovering dataset files on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def list_files(root: Path, suffixes: Iterable[str]) -> list[Path]:
    """Recursively list files within ``root`` that match ``suffixes``."""

    suffix_set = {suffix.lower() for suffix in suffixes}
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffix_set
    ]


def ensure_exists(path: Path) -> Path:
    """Return ``path`` if it exists, otherwise raise a helpful error."""

    if not path.exists():
        raise FileNotFoundError(f"Expected path '{path}' to exist.")
    return path
