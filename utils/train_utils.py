"""Training utilities shared by the CLI entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from Models.WaveFormer import WaveFormerModel


def build_state_features(batch: torch.Tensor) -> torch.Tensor:
    """Extract coarse statistics used by the D3QN agent for kernel selection."""

    mean = batch.mean().item()
    std = batch.std().item()
    seq_len = float(batch.size(1))
    return torch.tensor([mean, std, seq_len], dtype=torch.float32)


def _prepare_class_labels(targets: torch.Tensor) -> torch.Tensor:
    """Convert dataset targets to class indices expected by classification losses."""

    if targets.ndim > 1 and targets.size(-1) > 1:
        return targets.argmax(dim=-1).long()
    return targets.view(-1).long()


def compute_reward(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> float:
    """Negative cross-entropy reward for guiding the D3QN agent."""

    labels = _prepare_class_labels(targets)
    loss = nn.functional.cross_entropy(
        predictions,
        labels,
        label_smoothing=label_smoothing,
    )
    return float(-loss.item())


def train_epoch(
    model: WaveFormerModel,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float | None = None,
    label_smoothing: float = 0.0,
    augment_noise_std: float = 0.0,
) -> Tuple[float, float]:
    """Train ``model`` for a single epoch and return loss/accuracy."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for signals, targets in dataloader:
        signals = signals.to(device)
        if augment_noise_std > 0.0:
            noise = torch.randn_like(signals) * augment_noise_std
            signals = signals + noise

        targets = targets.to(device)

        labels = _prepare_class_labels(targets)

        state = build_state_features(signals)
        kernel_idx = model.select_kernel(state)

        predictions = model(signals, kernel_idx)
        loss = nn.functional.cross_entropy(
            predictions,
            labels,
            label_smoothing=label_smoothing,
        )

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        reward = compute_reward(
            predictions.detach(),
            targets,
            label_smoothing=label_smoothing,
        )
        next_state = build_state_features(signals)
        model.agent.store_transition(state, kernel_idx, reward, next_state, False)
        model.agent.update()

        total_loss += float(loss.item()) * len(signals)
        predictions_labels = predictions.argmax(dim=1)
        total_correct += (predictions_labels == labels).sum().item()
        total_samples += labels.numel()

    epoch_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_samples if total_samples else 0.0
    return epoch_loss, accuracy


@torch.no_grad()
def evaluate(
    model: WaveFormerModel,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate ``model`` and return average loss and accuracy."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for signals, targets in dataloader:
        signals = signals.to(device)
        targets = targets.to(device)

        labels = _prepare_class_labels(targets)
        state = build_state_features(signals)
        kernel_idx = model.select_kernel(state)

        predictions = model(signals, kernel_idx)
        loss = nn.functional.cross_entropy(predictions, labels)

        total_loss += float(loss.item()) * len(signals)
        predicted_labels = predictions.argmax(dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.numel()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_samples if total_samples else 0.0
    return avg_loss, accuracy


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Persist the model weights to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, map_location: str | None = None) -> None:
    """Load weights from ``path`` into ``model``."""

    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
