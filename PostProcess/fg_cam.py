"""Placeholder implementation of feature-gradient CAM visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch


def generate_cam(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    signal: torch.Tensor,
    output_index: int = 0,
    save_path: Path | None = None,
) -> torch.Tensor:
    """Generate a simple gradient-based class activation map."""

    signal = signal.clone().detach().requires_grad_(True)
    output = model_forward(signal.unsqueeze(0))
    target = output[0, output_index]
    target.backward()

    cam = signal.grad.abs()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 3))
        plt.plot(cam.cpu().numpy())
        plt.title("Gradient CAM")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return cam.detach()
