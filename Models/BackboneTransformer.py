"""Transformer backbone used by the WaveFormer model."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5_000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10_000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerBackbone(nn.Module):
    """Transformer encoder that consumes wavelet features."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self._warned_truncation = False
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            if not self._warned_truncation:
                logging.getLogger(__name__).warning(
                    "Input sequence length %d exceeds max_seq_len %d; "
                    "downsampling before transformer encoding.",
                    seq_len,
                    self.max_seq_len,
                )
                self._warned_truncation = True
            # ``F.interpolate`` expects channels-first layout for 1-D signals.
            x = x.transpose(1, 2)
            if seq_len > 1:
                x = F.interpolate(
                    x,
                    size=self.max_seq_len,
                    mode="linear",
                    align_corners=False,
                )
            else:
                x = F.interpolate(
                    x,
                    size=self.max_seq_len,
                    mode="nearest",
                )
            x = x.transpose(1, 2)

        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.head(pooled)
