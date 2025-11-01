"""Command line interface for training the WaveFormer-D3QN model on the CWRU dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from Datasets import CWRUConfig, CWRUDataset
from Models import WaveFormerConfig, WaveFormerModel
from Models.TFconvlayer import WaveletTransform, WaveletTransformConfig
from utils.logger import configure_logging
from utils.train_utils import evaluate, train_epoch


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
    parser.add_argument("--epochs", type=int, default=100)
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
    num_samples = len(dataset)
    if num_samples > 1:
        val_size = max(1, int(0.2 * num_samples))
        train_size = num_samples - val_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )
    else:
        train_dataset = dataset
        val_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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

    num_classes = int(dataset.targets.max().item()) + 1
    model_config = WaveFormerConfig(
        input_dim=wavelet_dim,
        output_dim=num_classes,
        max_seq_len=1,
        wavelet_kernels=base_config.wavelet_kernels,
        wavelet_level=base_config.wavelet_level,
    )
    model = WaveFormerModel(model_config, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_train_acc = 0.0
    best_train_epoch = 0
    best_val_acc = 0.0
    best_val_epoch = 0
    final_train_acc = 0.0
    final_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if train_acc >= best_train_acc:
            best_train_acc = train_acc
            best_train_epoch = epoch
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch

        final_train_acc = train_acc
        final_val_acc = val_acc

        logging.info(
            "Epoch %03d | Train Loss: %.6f | Train Acc: %.4f | Val Loss: %.6f | Val Acc: %.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

    logging.info(
        "Max train acc: %.4f (epoch %03d) | Max val acc: %.4f (epoch %03d)",
        best_train_acc,
        best_train_epoch,
        best_val_acc,
        best_val_epoch,
    )
    logging.info(
        "Final train acc: %.4f | Final val acc: %.4f",
        final_train_acc,
        final_val_acc,
    )

    logging.info("Training complete. Saving checkpoint to %s", checkpoint_path.resolve())
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
