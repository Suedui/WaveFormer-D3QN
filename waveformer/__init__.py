"""WaveFormer-D3QN package.

This package exposes the key building blocks for constructing a Transformer
model whose wavelet feature extractor is optimised via a D3QN agent.
"""

from .d3qn import WaveletSelectionAgent
from .model import WaveFormerD3QN

__all__ = [
    "WaveFormerD3QN",
    "WaveletSelectionAgent",
]

