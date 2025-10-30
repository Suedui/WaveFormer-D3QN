"""Training script for the WaveFormer-D3QN architecture."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from waveformer.d3qn import AgentConfig
from waveformer.dataset import DatasetConfig, TimeSeriesDataset
from waveformer.model import WaveFormerConfig, WaveFormerD3QN
from waveformer.wavelet_transform import WaveletTransform, WaveletTransformConfig


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
    model: WaveFormerD3QN,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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
        optimizer.step()

        reward = compute_reward(predictions.detach(), targets)
        next_state = build_state_features(signals)
        model.agent.store_transition(state, kernel_idx, reward, next_state, False)
        model.agent.update()

        total_loss += float(loss.item()) * len(signals)

    return total_loss / len(dataloader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/your-dataset"),
        help="Path to the directory containing signals.npy and targets.npy.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset_config = DatasetConfig(root=args.dataset_root)
    dataset = TimeSeriesDataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Wavelet transform expands the input dimensionality; compute once from data.
    sample_signal = dataset[0][0].numpy()
    base_config = WaveFormerConfig(input_dim=len(sample_signal))
    transform = WaveletTransform(
        base_config.wavelet_kernels,
        WaveletTransformConfig(level=base_config.wavelet_level),
    )
    wavelet_dim = len(transform.apply(sample_signal, base_config.wavelet_kernels[0]))

    model_config = WaveFormerConfig(
        input_dim=wavelet_dim,
        output_dim=dataset[0][1].numel(),
        max_seq_len=1,
        wavelet_kernels=base_config.wavelet_kernels,
        wavelet_level=base_config.wavelet_level,
    )
    model = WaveFormerD3QN(model_config, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch:03d} | Loss: {loss:.6f}")


if __name__ == "__main__":
    main()

