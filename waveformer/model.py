"""Model definition for the WaveFormer with D3QN-guided wavelet selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from torch import nn

from .d3qn import AgentConfig, WaveletSelectionAgent
from .wavelet_transform import WaveletTransform, WaveletTransformConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5_000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10_000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


@dataclass
class WaveFormerConfig:
    """Configuration object for :class:`WaveFormerD3QN`."""

    input_dim: int
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 1
    max_seq_len: int = 512
    wavelet_kernels: Iterable[str] = (
        "db2",
        "sym4",
        "coif1",
    )
    wavelet_level: int = 2


class WaveFormerBackbone(nn.Module):
    """Transformer encoder that consumes wavelet features."""

    def __init__(self, config: WaveFormerConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.model_dim)
        self.pos_encoder = PositionalEncoding(config.model_dim, config.max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.model_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.head(pooled)


class WaveFormerD3QN(nn.Module):
    """WaveFormer model with an embedded D3QN-controlled wavelet transform."""

    def __init__(
        self,
        model_config: WaveFormerConfig,
        agent_config: AgentConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device("cpu")

        self.wavelet_transform = WaveletTransform(
            model_config.wavelet_kernels,
            WaveletTransformConfig(level=model_config.wavelet_level),
        )

        kernels = list(model_config.wavelet_kernels)
        self.kernels: List[str] = kernels

        if agent_config is None:
            agent_config = AgentConfig(
                state_dim=3,
                action_dim=len(kernels),
            )
        self.agent = WaveletSelectionAgent(agent_config, device=self.device)
        self.backbone = WaveFormerBackbone(model_config)

    def forward(self, batch: torch.Tensor, kernel_index: int) -> torch.Tensor:
        """Forward pass through the backbone given a fixed kernel selection."""

        kernel = self.kernels[kernel_index]
        features = [
            torch.from_numpy(self.wavelet_transform.apply(sample.cpu().numpy(), kernel))
            for sample in batch
        ]
        feature_tensor = torch.stack(features).unsqueeze(1).to(batch.device)
        return self.backbone(feature_tensor)

    def select_kernel(self, state: torch.Tensor) -> int:
        """Delegate kernel selection to the D3QN agent."""

        return self.agent.select_action(state)

