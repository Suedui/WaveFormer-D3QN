"""Command line interface for training the TFN1 model on the CWRU dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Datasets import CWRUConfig, CWRUDataset
from Models import TFNConfig, TFNModel
from Models.TFconvlayer import WaveletTransform, WaveletTransformConfig
from utils.logger import configure_logging
from utils.train_utils import train_epoch


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
        default=Path("checkpoint/tfn1.ckpt"),
        help="Path for saving the final checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    configure_logging(args.log_dir)

    dataset_config = CWRUConfig(root=args.dataset_root)
    dataset = CWRUDataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    sample_signal = dataset[0][0].numpy()
    base_config = TFNConfig(input_dim=len(sample_signal))
    transform = WaveletTransform(
        base_config.wavelet_kernels,
        WaveletTransformConfig(level=base_config.wavelet_level),
    )
    wavelet_dim = len(transform.apply(sample_signal, base_config.wavelet_kernels[0]))

    model_config = TFNConfig(
        input_dim=wavelet_dim,
        output_dim=dataset[0][1].numel(),
        max_seq_len=1,
        wavelet_kernels=base_config.wavelet_kernels,
        wavelet_level=base_config.wavelet_level,
    )
    model = TFNModel(model_config, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, device)
        logging.info("Epoch %03d | Loss: %.6f", epoch, loss)

    logging.info("Training complete. Saving checkpoint to %s", args.checkpoint_path)
    torch.save(model.state_dict(), args.checkpoint_path)


if __name__ == "__main__":
    main()
