"""High-level WaveFormer model that marries wavelet features with a transformer backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F

from .BackboneTransformer import TransformerBackbone
from .TFconvlayer import WaveletTransform, WaveletTransformConfig
from .WaveletRLConv import AgentConfig, WaveletSelectionAgent


@dataclass(slots=True)
class WaveFormerConfig:
    """Configuration object for :class:`WaveFormerModel`."""

    input_dim: int
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 1
    max_seq_len: int = 512
    wavelet_kernels: Iterable[str] = ("db2", "sym4", "coif1")
    wavelet_level: int = 2


class WaveFormerModel(nn.Module):
    """Wavelet-aware transformer that relies on a D3QN agent for kernel selection."""

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
        self.backbone = TransformerBackbone(
            input_dim=model_config.input_dim,
            model_dim=model_config.model_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            max_seq_len=model_config.max_seq_len,
            output_dim=model_config.output_dim,
        )

    def forward(self, batch: torch.Tensor, kernel_index: int) -> torch.Tensor:
        """Forward pass through the backbone given a fixed kernel selection."""

        kernel = self.kernels[kernel_index]
        target_len = self.backbone.input_proj.in_features
        processed_features: list[torch.Tensor] = []
        for sample in batch:
            feature = torch.from_numpy(
                self.wavelet_transform.apply(sample.cpu().numpy(), kernel)
            ).float()
            numel = feature.numel()
            if numel < target_len:
                feature = F.pad(feature, (0, target_len - numel))
            elif numel > target_len:
                feature = feature[:target_len]
            processed_features.append(feature)

        feature_tensor = torch.stack(processed_features).unsqueeze(1).to(batch.device)
        return self.backbone(feature_tensor)

    def select_kernel(self, state: torch.Tensor) -> int:
        """Delegate kernel selection to the D3QN agent."""

        return self.agent.select_action(state)
