"""Logging utilities for WaveFormer-D3QN."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path) -> None:
    """Configure a root logger that writes to both stdout and a log file."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Logging initialised. Writing to %s", log_path)
