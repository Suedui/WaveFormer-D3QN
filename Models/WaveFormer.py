"""High-level WaveFormer model that marries wavelet features with a transformer backbone."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn

from .BackboneTransformer import TransformerBackbone
from .TFconvlayer import WaveletTransform, WaveletTransformConfig
from .WaveletRLConv import AgentConfig, WaveletSelectionAgent


@dataclass(slots=True)
class WaveFormerConfig:
    """Configuration object for :class:`WaveFormerModel`."""

    input_dim: int
    model_dim: int = 512
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 1
    max_seq_len: int = 512
    wavelet_kernels: Iterable[str] = ("morl", "mexh", "gaus8")
    wavelet_scale_bands: Sequence[Tuple[float, float]] = (
        (1.0, 32.0),
        (1.0, 64.0),
        (1.0, 128.0),
    )
    wavelet_num_scales: int = 64
    sampling_period: float = 1.0


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

        transform_config = WaveletTransformConfig(
            scale_bands=model_config.wavelet_scale_bands,
            num_scales=model_config.wavelet_num_scales,
            sampling_period=model_config.sampling_period,
        )
        self.wavelet_transform = WaveletTransform(
            model_config.wavelet_kernels,
            transform_config,
        )

        kernels = list(model_config.wavelet_kernels)
        self.kernels: List[str] = kernels
        band_indices = list(self.wavelet_transform.available_scale_bands())
        self.actions: List[Tuple[str, int]] = [
            (kernel, band_idx) for kernel in kernels for band_idx in band_indices
        ]

        if agent_config is None:
            agent_config = AgentConfig(
                state_dim=4,
                action_dim=len(self.actions),
            )
        else:
            agent_config = replace(agent_config, action_dim=len(self.actions))
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

    def forward(self, batch: torch.Tensor, action_index: int) -> torch.Tensor:
        """Forward pass through the backbone given a fixed action selection."""

        kernel, band_idx = self.actions[action_index]
        processed_features: list[torch.Tensor] = []
        for sample in batch:
            scalogram = self.wavelet_transform.apply(
                sample.cpu().numpy(), kernel, band_idx
            )
            feature = torch.from_numpy(scalogram).to(batch.device)
            processed_features.append(feature)

        feature_tensor = torch.stack(processed_features).to(batch.device)
        return self.backbone(feature_tensor)

    def select_kernel(self, state: torch.Tensor, *, explore: bool = True) -> int:
        """Delegate wavelet action selection to the D3QN agent.

        Parameters
        ----------
        state:
            Summary statistics describing the current batch of signals.
        explore:
            If ``True`` the epsilon-greedy policy is used. When ``False`` the
            greedy action is selected without updating the agent state.
        """

        if explore:
            return self.agent.select_action(state)
        return self.agent.select_greedy_action(state)
