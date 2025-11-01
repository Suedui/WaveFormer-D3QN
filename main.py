"""Command line interface for training the WaveFormer-D3QN model on the CWRU dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Datasets import CWRUConfig, CWRUDataset
from Models import WaveFormerConfig, WaveFormerModel
from Models.TFconvlayer import WaveletTransform, WaveletTransformConfig
from utils.logger import configure_logging
from utils.train_utils import train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset_root = Path(__file__).resolve().parent / "Datasets" / "CWRU"
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
        help=(
            "Path to the directory containing signals.npy and targets.npy. "
            "Defaults to the bundled CWRU dataset shipped with the project."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("checkpoint/logs"),
        help="Directory used for training logs.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoint/waveformer.ckpt"),
        help="Path for saving the final checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    log_dir = args.log_dir.expanduser()
    checkpoint_path = args.checkpoint_path.expanduser()
    device = torch.device(args.device)

    configure_logging(log_dir)

    dataset_config = CWRUConfig(root=dataset_root)
    dataset = CWRUDataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    sample_signal = dataset[0][0].numpy()
    base_config = WaveFormerConfig(input_dim=len(sample_signal))
    transform = WaveletTransform(
        base_config.wavelet_kernels,
        WaveletTransformConfig(level=base_config.wavelet_level),
    )
    wavelet_dim = max(
        len(transform.apply(sample_signal, kernel))
        for kernel in base_config.wavelet_kernels
    )

    model_config = WaveFormerConfig(
        input_dim=wavelet_dim,
        output_dim=dataset[0][1].numel(),
        max_seq_len=1,
        wavelet_kernels=base_config.wavelet_kernels,
        wavelet_level=base_config.wavelet_level,
    )
    model = WaveFormerModel(model_config, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, device)
        logging.info("Epoch %03d | Loss: %.6f", epoch, loss)

    logging.info("Training complete. Saving checkpoint to %s", checkpoint_path.resolve())
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
