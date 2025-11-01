"""Helpers specific to the CWRU bearing dataset layout."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import warnings

import numpy as np
from scipy.io import loadmat

from .generalfunction import ensure_exists, list_files

MAT_SUFFIXES = (".mat",)


def load_cwru_numpy(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load or derive ``signals.npy`` and ``targets.npy`` from ``root``.

    The original project expected pre-converted NumPy files. Many publicly
    available copies of the CWRU dataset, however, only contain the original
    MATLAB ``.mat`` files.  To improve usability we now transparently convert
    those files into the cached NumPy representation on first use.

    When ``root`` points directly to a single ``.mat`` file we bypass the cache
    generation and construct a one-sample dataset on the fly.  This is handy for
    quick smoke tests without downloading the full dataset structure.
    """

    root = ensure_exists(root)

    if root.is_file():
        return _load_single_mat_file(root)

    signal_path = root / "signals.npy"
    target_path = root / "targets.npy"

    if signal_path.exists() and target_path.exists():
        return np.load(signal_path), np.load(target_path)

    mat_files = sorted(list_files(root, MAT_SUFFIXES))
    if not mat_files:
        raise FileNotFoundError(
            "Expected 'signals.npy' and 'targets.npy' or a collection of .mat "
            f"files under {root}"
        )

    signals: list[np.ndarray] = []
    labels: list[int] = []
    label_map: dict[str, int] = {}

    for mat_file in mat_files:
        signal = _load_signal_from_mat(mat_file)
        label_name = _derive_label(mat_file, root)
        label_id = label_map.setdefault(label_name, len(label_map))

        signals.append(signal)
        labels.append(label_id)

    signals_array = _stack_signals(signals, mat_files)
    targets_array = np.asarray(labels, dtype=np.int64)

    np.save(signal_path, signals_array)
    np.save(target_path, targets_array)

    return signals_array, targets_array


def _load_signal_from_mat(path: Path) -> np.ndarray:
    """Extract the first 1D array from ``path`` assumed to contain the signal."""

    mat = loadmat(path)
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            array = np.asarray(value).squeeze()
            if array.ndim == 1:
                return array.astype(np.float32)

    raise ValueError(f"Unable to locate a 1D signal array inside '{path}'.")


def _label_from_root(root: Path) -> str:
    """Create a deterministic label prefix based on ``root``."""

    tail_components = list(root.parts)[-3:]
    tokens = [_normalise(component) for component in tail_components]
    tokens = [token for token in tokens if token]
    if not tokens:
        return "dataset"
    return "_".join(tokens)


def _derive_label(path: Path, root: Path) -> str:
    """Generate a stable class label from the dataset directory hierarchy."""

    parts = list(path.relative_to(root).parts)
    if len(parts) <= 1:
        # ``root`` points directly to a directory containing ``.mat`` files.
        # Use the directory context instead of the file name to avoid
        # assigning a unique class to every sample in that folder.
        return _label_from_root(root)

    dataset_group = _normalise(parts[0])
    fault_type = None
    position = None
    severity = None

    for part in parts[1:-1]:
        normalised = _normalise(part)
        if normalised in {"ball", "innerrace", "outerrace"}:
            fault_type = normalised
        elif normalised in {"centered", "opposite", "orthogonal"}:
            position = normalised
        elif part.isdigit():
            severity = part

    stem = path.stem
    main_token = stem.split("_")[0]
    location_token = None
    if "@" in main_token:
        main_token, location_token = main_token.split("@", maxsplit=1)

    if severity is None:
        digits = "".join(ch for ch in main_token if ch.isdigit())
        severity = digits or main_token.lower()

    label_parts = [dataset_group]
    if fault_type:
        label_parts.append(fault_type)
    if position:
        label_parts.append(position)
    label_parts.append(_normalise(severity))
    if location_token:
        label_parts.append(f"at{_normalise(location_token)}")

    return "_".join(part for part in label_parts if part)


def _normalise(token: str) -> str:
    """Convert ``token`` into a filesystem-agnostic identifier."""

    return "".join(ch.lower() if ch.isalnum() else "_" for ch in token).strip("_")


def _stack_signals(signals: Iterable[np.ndarray], mat_files: list[Path]) -> np.ndarray:
    """Validate that all signals share the same length and stack them."""

    signals_list = [np.asarray(signal, dtype=np.float32).ravel() for signal in signals]
    lengths = [signal.shape[0] for signal in signals_list]
    min_length = min(lengths)
    if len(set(lengths)) != 1:
        warnings.warn(
            "Signals extracted from the provided .mat files do not all share the "
            "same length. They will be truncated to the shortest sequence to "
            "enable batching.",
            RuntimeWarning,
            stacklevel=2,
        )
        signals_list = [signal[:min_length] for signal in signals_list]

    return np.stack(signals_list, axis=0)


def _load_single_mat_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Create a one-sample dataset from ``path`` when it points to a ``.mat`` file."""

    if path.suffix.lower() not in MAT_SUFFIXES:
        raise ValueError(
            "Single-file loading only supports MATLAB '.mat' files; "
            f"received '{path.suffix}'."
        )

    signal = _load_signal_from_mat(path)
    signals_array = signal[np.newaxis, :]
    # ``targets`` represent class indices; with a single sample we simply
    # assign it to class ``0`` so that downstream classification heads remain
    # well-defined.
    targets_array = np.zeros((1,), dtype=np.int64)
    return signals_array, targets_array
