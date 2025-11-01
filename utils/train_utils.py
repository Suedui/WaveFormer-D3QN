"""Training utilities shared by the CLI entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from Models.WaveFormer import WaveFormerModel


def build_state_features(batch: torch.Tensor) -> torch.Tensor:
    """Extract coarse statistics used by the D3QN agent for kernel selection."""

    mean = batch.mean().item()
    std = batch.std(unbiased=False).item()
    rms = torch.sqrt((batch ** 2).mean()).item()
    peak = batch.abs().max().item()
    return torch.tensor([mean, std, rms, peak], dtype=torch.float32)


def _prepare_class_labels(targets: torch.Tensor) -> torch.Tensor:
    """Convert dataset targets to class indices expected by classification losses."""

    if targets.ndim > 1 and targets.size(-1) > 1:
        return targets.argmax(dim=-1).long()
    return targets.view(-1).long()


def compute_reward(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Reward proportional to the classification accuracy of the batch."""

    labels = _prepare_class_labels(targets)
    predicted = predictions.argmax(dim=1)
    accuracy = (predicted == labels).float().mean().item()
    return float(accuracy)


def train_epoch(
    model: WaveFormerModel,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float | None = None,
) -> Tuple[float, float]:
    """Train ``model`` for a single epoch and return loss/accuracy."""

    model.train()
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

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        reward = compute_reward(predictions.detach(), targets)
        next_state = build_state_features(signals)
        model.agent.store_transition(state, kernel_idx, reward, next_state, True)
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
        kernel_idx = model.select_kernel(state, explore=False)

        predictions = model(signals, kernel_idx)
        loss = nn.functional.cross_entropy(predictions, labels)

        total_loss += float(loss.item()) * len(signals)
        predicted_labels = predictions.argmax(dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.numel()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_samples if total_samples else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_with_metrics(
    model: WaveFormerModel,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Evaluate ``model`` and compute accuracy, recall and F1-score."""

    model.eval()
    total_correct = 0
    total_samples = 0
    true_positive = torch.zeros(num_classes, dtype=torch.float32)
    false_positive = torch.zeros(num_classes, dtype=torch.float32)
    false_negative = torch.zeros(num_classes, dtype=torch.float32)

    for signals, targets in dataloader:
        signals = signals.to(device)
        targets = targets.to(device)

        labels = _prepare_class_labels(targets)
        state = build_state_features(signals)
        kernel_idx = model.select_kernel(state, explore=False)

        predictions = model(signals, kernel_idx)
        predicted_labels = predictions.argmax(dim=1)

        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.numel()

        for cls in range(num_classes):
            cls_tensor = torch.tensor(cls, device=labels.device)
            tp = ((predicted_labels == cls_tensor) & (labels == cls_tensor)).sum().float()
            fp = ((predicted_labels == cls_tensor) & (labels != cls_tensor)).sum().float()
            fn = ((predicted_labels != cls_tensor) & (labels == cls_tensor)).sum().float()
            true_positive[cls] += tp.cpu()
            false_positive[cls] += fp.cpu()
            false_negative[cls] += fn.cpu()

    accuracy = total_correct / total_samples if total_samples else 0.0

    recall_per_class = true_positive / (true_positive + false_negative + 1e-6)
    precision_per_class = true_positive / (true_positive + false_positive + 1e-6)
    f1_per_class = 2 * precision_per_class * recall_per_class / (
        precision_per_class + recall_per_class + 1e-6
    )

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy),
        "recall": float(recall_per_class.mean().item()),
        "f1": float(f1_per_class.mean().item()),
    }
    return metrics


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Persist the model weights to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, map_location: str | None = None) -> None:
    """Load weights from ``path`` into ``model``."""

    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
