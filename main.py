"""Command line interface for training the WaveFormer-D3QN model on the CWRU dataset."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, _LRScheduler
from torch.utils.data import DataLoader, Subset, random_split

from utils.train_utils import _prepare_class_labels, evaluate, train_epoch

from Datasets import CWRUConfig, CWRUDataset
from Models import WaveFormerConfig, WaveFormerModel
from Models.TFconvlayer import WaveletTransform, WaveletTransformConfig
from utils.logger import configure_logging


def _build_labels_tensor(dataset: CWRUDataset) -> torch.Tensor:
    labels = _prepare_class_labels(dataset.targets)
    return labels.cpu()


def _stratified_split(
    dataset: CWRUDataset,
    val_ratio: float,
    generator: torch.Generator,
) -> Tuple[Subset[CWRUDataset], Subset[CWRUDataset]]:
    labels = _build_labels_tensor(dataset)
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        class_indices[int(label)].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for indices in class_indices.values():
        cls_tensor = torch.tensor(indices, dtype=torch.long)
        if cls_tensor.numel() <= 1:
            train_indices.extend(cls_tensor.tolist())
            continue

        perm = torch.randperm(cls_tensor.numel(), generator=generator)
        shuffled = cls_tensor[perm].tolist()
        cls_size = len(shuffled)
        val_count = max(1, int(round(val_ratio * cls_size)))
        if cls_size - val_count <= 0:
            val_count = cls_size - 1
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    if not train_indices or not val_indices:
        total = len(dataset)
        if total <= 1:
            return Subset(dataset, list(range(total))), Subset(dataset, list(range(total)))
        val_size = max(1, int(round(val_ratio * total)))
        if val_size >= total:
            val_size = total - 1
        return random_split(dataset, [total - val_size, val_size], generator=generator)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


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
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay applied by the optimizer for L2 regularization.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--no-stratified-split",
        action="store_true",
        help="Disable stratified splitting and fall back to random_split.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable Gaussian noise augmentation for the CWRU dataset.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Standard deviation of the Gaussian noise augmentation.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm applied during training (non-positive to disable).",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "step", "plateau"],
        default="none",
        help="Learning rate scheduler strategy.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=30,
        help="Step size for StepLR when --lr-scheduler=step.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.5,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement before reducing LR when using ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Number of epochs to wait for validation improvement before stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation accuracy improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate applied inside the WaveFormer transformer backbone.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of transformer layers in the WaveFormer backbone.",
    )
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

    val_ratio = max(0.0, min(args.val_ratio, 0.9))
    if args.val_ratio != val_ratio:
        logging.warning(
            "Validation ratio %.3f adjusted to %.3f to keep it within a safe range (0, 0.9).",
            args.val_ratio,
            val_ratio,
        )
    if val_ratio == 0.0:
        logging.warning("Validation ratio of 0.0 detected; defaulting to 0.2 for splitting.")
        val_ratio = 0.2

    dataset_config = CWRUConfig(
        root=dataset_root,
        augment=args.augment,
        noise_std=args.noise_std,
    )
    dataset = CWRUDataset(dataset_config)
    num_samples = len(dataset)
    if num_samples > 1:
        generator = torch.Generator().manual_seed(42)
        val_size = max(1, int(round(val_ratio * num_samples)))
        if val_size >= num_samples:
            val_size = num_samples - 1
        train_size = num_samples - val_size
        if args.no_stratified_split:
            logging.info("Using random_split with validation ratio %.3f.", val_ratio)
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=generator,
            )
        else:
            logging.info("Using stratified split with validation ratio %.3f.", val_ratio)
            train_dataset, val_dataset = _stratified_split(dataset, val_ratio, generator)
    else:
        train_dataset = dataset
        val_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info(
        "Dataset split: %d training samples | %d validation samples",
        len(train_dataset),
        len(val_dataset),
    )

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

    labels = _build_labels_tensor(dataset)
    num_classes = int(labels.max().item()) + 1
    model_config = WaveFormerConfig(
        input_dim=wavelet_dim,
        output_dim=num_classes,
        max_seq_len=1,
        wavelet_kernels=base_config.wavelet_kernels,
        wavelet_level=base_config.wavelet_level,
        dropout=args.dropout,
        num_layers=args.num_layers,
    )
    model = WaveFormerModel(model_config, device=device).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler: _LRScheduler | ReduceLROnPlateau | None
    if args.lr_scheduler == "step":
        scheduler = StepLR(
            optimizer,
            step_size=max(1, args.lr_step_size),
            gamma=args.lr_gamma,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_gamma,
            patience=max(1, args.lr_patience),
            verbose=True,
        )
    else:
        scheduler = None

    best_train_acc = 0.0
    best_train_epoch = 0
    best_val_acc = 0.0
    best_val_epoch = 0
    final_train_acc = 0.0
    final_val_acc = 0.0
    best_model_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    patience = max(0, args.early_stopping_patience)
    min_delta = args.early_stopping_min_delta
    gradient_clip = None if args.gradient_clip <= 0 else args.gradient_clip

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gradient_clip=gradient_clip,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        if train_acc >= best_train_acc:
            best_train_acc = train_acc
            best_train_epoch = epoch
        if val_acc >= best_val_acc + min_delta:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        final_train_acc = train_acc
        final_val_acc = val_acc

        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            (
                "Epoch %03d | Train Loss: %.6f | Train Acc: %.4f | "
                "Val Loss: %.6f | Val Acc: %.4f | LR: %.6f"
            ),
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if patience and patience_counter >= patience:
            logging.info(
                "Early stopping triggered after %d epochs without validation improvement.",
                patience_counter,
            )
            break

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

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Restored model weights from epoch %03d for checkpointing.", best_val_epoch)

    logging.info("Training complete. Saving checkpoint to %s", checkpoint_path.resolve())
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
