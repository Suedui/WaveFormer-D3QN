"""Training utilities shared by the CLI entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from Models.TFN import TFNModel


def build_state_features(batch: torch.Tensor) -> torch.Tensor:
    """Extract coarse statistics used by the D3QN agent for kernel selection."""

    mean = batch.mean().item()
    std = batch.std().item()
    seq_len = float(batch.size(1))
    return torch.tensor([mean, std, seq_len], dtype=torch.float32)


def compute_reward(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Simple negative MSE reward used for guiding the D3QN agent."""

    mse = nn.functional.mse_loss(predictions, targets)
    return float(-mse.item())


def train_epoch(
    model: TFNModel,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float | None = None,
) -> float:
    """Train ``model`` for a single epoch."""

    model.train()
    total_loss = 0.0

    for signals, targets in dataloader:
        signals = signals.to(device)
        targets = targets.to(device)

        state = build_state_features(signals)
        kernel_idx = model.select_kernel(state)

        predictions = model(signals, kernel_idx)
        loss = nn.functional.mse_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        reward = compute_reward(predictions.detach(), targets)
        next_state = build_state_features(signals)
        model.agent.store_transition(state, kernel_idx, reward, next_state, False)
        model.agent.update()

        total_loss += float(loss.item()) * len(signals)

    return total_loss / len(dataloader.dataset)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Persist the model weights to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, map_location: str | None = None) -> None:
    """Load weights from ``path`` into ``model``."""

    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
